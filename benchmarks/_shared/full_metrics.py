"""The shared metrics of the end-to-end `full_*` benchmarks (FULL_METRICS_CONTRACT.md), scoring-only.

Three verdict metrics, each with its own verdict, never collapsed:
  - mAP: same-label retrieval of unit profiles (`unit_profiles`: the mean of the L2-normalised cells, L2-normalised;
    cosine); a query is a test-side unit, its candidates the train-side units of the same fold part; per-query AP
    (`average_precision`: equal similarities share one precision, sklearn's step rule); macro over the labels;
  - macro F1 and balanced accuracy of the cell-level kNN probe (`scores`, sklearn's definitions).
The kNN / logistic-regression probe on cells and on unit profiles is reported for all four combinations. The
cell-level probes use <= CELL_CAP cells per unit (`cap_cells`, seed CELL_SEED, from the arm's own delivered and
excluded cells; a sampled excluded cell is a miss). A unit without delivered cells counts as misses (`evaluate`):
CELL_CAP cells without a prediction at the cell level when it has no excluded cells either, one at the unit level,
AP 0 in mAP (and never a candidate).

A benchmark builds a `Layout` from its units' labels and its splits (each split a list of folds, each fold a list of
(train units, test units) parts whose test predictions pool into the fold's confusion matrix); a split's value is the
mean over its folds. `Layout` draws the bootstrap from the layout alone (BOOTSTRAP resamples of each fold's test
units, seed BOOTSTRAP_SEED, a drawn unit bringing all its cells), so every arm scored on the same layout gets the
same draws. `evaluate` scores one arm (points and draws), `save` / `load` keep a reading for a later comparison,
`verdict` compares an arm with the control on the same draws, and `record` / `markdown` give RESULT.json
`full_metrics` and the RESULT.md table.

The probe core (`scores`, `chance`, `knn_predict`, `logreg_predict` and KNN_K, KNN_CHUNK, LOGREG) is the
representation_RxRx1 probe, moved here unchanged; representation_RxRx1/runners/secondary.py imports it.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

KNN_K = 10
KNN_CHUNK = 1024
LOGREG = {"solver": "lbfgs", "max_iter": 2000, "class_weight": "balanced", "random_state": 0}
BOOTSTRAP, BOOTSTRAP_SEED = 1000, 0
CELL_CAP, CELL_SEED = 100, 0          # cells per unit in the cell-level probes (full_HCMA's MAX_CELLS, SAMPLE_SEED)
METRICS = ("mAP", "macro_f1", "balanced_accuracy")
PROBE_METRICS = ("balanced_accuracy", "macro_f1")
UNITS = ("cell", "unit")
CLASSIFIERS = ("knn", "logreg")
VERDICT_PROBE = ("cell", "knn")
SPLITS = ("main", "guard")
GAIN, LOSS, NO_RESOLVABLE, SHORTCUT = "GAIN", "LOSS", "NO_RESOLVABLE_DIFFERENCE", "SOURCE_SHORTCUT_SUSPECTED"
VERDICT_RULE = (f"delta = arm minus control on the main split, SE = the s.d. of the paired bootstrap delta (the same "
                f"draws); {GAIN} if delta > 2 SE and the guard-split delta >= 0, {SHORTCUT} if delta > 2 SE and the "
                f"guard delta < 0, {LOSS} if delta < -2 SE, else {NO_RESOLVABLE} (also when SE is 0)")


# ---- the probe core (moved unchanged from representation_RxRx1/runners/secondary.py) -------------------------------

def l2(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to unit norm (float64; a zero row stays zero): representation_RxRx1.runners.score.l2."""
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def scores(confusion: np.ndarray) -> dict[str, np.ndarray]:
    """Balanced accuracy and macro F1 of confusion matrices (..., C, C; rows true, columns predicted), as sklearn
    defines them: balanced accuracy = the mean recall over the classes with a true sample; macro F1 = the mean
    2 TP / (2 TP + FP + FN) over the classes with a true or a predicted sample."""
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)
    true, pred = confusion.sum(axis=-1), confusion.sum(axis=-2)
    present, labelled = true > 0, (true > 0) | (pred > 0)
    recall = np.where(present, tp / np.maximum(true, 1), 0.0).sum(axis=-1) / present.sum(axis=-1)
    f1 = np.where(labelled, 2 * tp / np.maximum(true + pred, 1), 0.0).sum(axis=-1) / labelled.sum(axis=-1)
    return {"balanced_accuracy": recall, "macro_f1": f1}


def chance(train_labels: np.ndarray, test_labels: np.ndarray, n_classes: int) -> dict[str, float]:
    """The expected scores of a stratified random predictor (each prediction an independent draw from the train
    class frequencies p_c): balanced accuracy = the mean p_c over the test classes (exact); macro F1 = the F1 of the
    expected confusion matrix, 2 n_c p_c / (n_c + N p_c) per class (n_c test samples of N), averaged over the classes
    with a test sample or p_c > 0."""
    p = np.bincount(train_labels, minlength=n_classes) / len(train_labels)
    n = np.bincount(test_labels, minlength=n_classes).astype(np.float64)
    present, labelled = n > 0, (n > 0) | (p > 0)
    f1 = 2 * n * p / np.maximum(n + len(test_labels) * p, 1e-300)
    return {"balanced_accuracy": float(p[present].mean()), "macro_f1": float(f1[labelled].mean())}


def knn_predict(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray, n_classes: int,
                k: int = KNN_K) -> np.ndarray:
    """Cosine kNN: the k most similar train rows (equal similarities by train row order), a majority vote, a tied
    vote to the tied class of the nearest neighbour."""
    train, test = l2(train), l2(test)
    k = min(k, len(train))
    out = np.empty(len(test), dtype=np.int64)
    for start in range(0, len(test), KNN_CHUNK):
        sim = test[start:start + KNN_CHUNK] @ train.T
        rows = np.arange(len(sim))[:, None]
        kth = -np.partition(-sim, k - 1, axis=1)[:, k - 1:k]               # the k-th largest similarity per row
        above, equal = sim > kth, sim == kth
        chosen = above | (equal & (np.cumsum(equal, axis=1) <= k - above.sum(axis=1, keepdims=True)))
        keep = np.nonzero(chosen)[1].reshape(len(sim), k)                  # exactly k per row, by train row
        keep = np.take_along_axis(keep, np.lexsort((keep, -sim[rows, keep]), axis=1), axis=1)
        neighbours = train_labels[keep]                                   # by similarity, then by train row
        votes = (neighbours[:, :, None] == np.arange(n_classes)).sum(axis=1)
        tied = votes == votes.max(axis=1, keepdims=True)
        first = tied[rows, neighbours].argmax(axis=1)                     # the nearest neighbour of a tied class
        out[start:start + len(sim)] = neighbours[np.arange(len(sim)), first]
    return out


def logreg_predict(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, dict]:
    """StandardScaler fitted on train, then LogisticRegression(LOGREG); (predictions, {n_iter, converged})."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train = np.asarray(train, dtype=np.float64)
    scaler = StandardScaler().fit(train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)                    # recorded as converged: False
        model = LogisticRegression(**LOGREG).fit(scaler.transform(train), train_labels)
    n_iter = int(np.max(model.n_iter_))
    return (model.predict(scaler.transform(np.asarray(test, dtype=np.float64))),
            {"n_iter": n_iter, "converged": n_iter < LOGREG["max_iter"]})


# ---- units, retrieval, folds ---------------------------------------------------------------------------------------

def unit_profiles(cells: np.ndarray, cell_unit: np.ndarray, n_units: int, chunk: int = 65536) -> np.ndarray:
    """Per unit the mean of its L2-normalised cells, L2-normalised (cells with unit -1 left out; a unit without
    cells: a zero row). `cells` may be a memory map: it is read in chunks."""
    cell_unit = np.asarray(cell_unit, dtype=np.int64)
    sums = np.zeros((n_units, np.shape(cells)[1]), dtype=np.float64)
    for start in range(0, len(cell_unit), chunk):
        units = cell_unit[start:start + chunk]
        mine = units >= 0
        if mine.any():
            np.add.at(sums, units[mine], l2(np.asarray(cells[start:start + chunk])[mine]))
    counts = np.bincount(cell_unit[cell_unit >= 0], minlength=n_units).astype(np.float64)
    return l2(sums / np.maximum(counts, 1)[:, None])


def cap_cells(cell_unit: np.ndarray, cell_ids=None, cap: int = CELL_CAP, seed: int = CELL_SEED) -> np.ndarray:
    """Row indices of <= `cap` cells per unit (cells with unit -1 left out): a unit's cells sorted by cell id (None:
    by row), permuted by numpy.random.default_rng(seed).permutation (a fresh generator per unit), the first `cap` of
    that permutation in sorted order (full_HCMA's SAMPLE_RULE); a unit with fewer cells keeps all of them."""
    cell_unit = np.asarray(cell_unit, dtype=np.int64)
    keys = np.arange(len(cell_unit)) if cell_ids is None else np.asarray(cell_ids).astype(str)
    order = np.lexsort((keys, cell_unit))
    order = order[cell_unit[order] >= 0]
    _, starts = np.unique(cell_unit[order], return_index=True)
    picked = []
    for start, end in zip(starts, [*starts[1:], len(order)]):
        rows = order[start:end]
        if len(rows) > cap:
            rows = rows[np.sort(np.random.default_rng(seed).permutation(len(rows))[:cap])]
        picked.append(rows)
    return np.concatenate(picked) if picked else np.zeros(0, dtype=np.int64)


def average_precision(similarity: np.ndarray, positives: np.ndarray) -> float:
    """AP of one query over its candidates: the mean over the positives of (positives >= its similarity) /
    (candidates >= its similarity); equal similarities share one precision (sklearn's average_precision_score)."""
    every, hit = np.sort(similarity), np.sort(similarity[positives])
    rank = len(every) - np.searchsorted(every, hit, side="left")
    hits = len(hit) - np.searchsorted(hit, hit, side="left")
    return float(np.mean(hits / rank))


def grouped_folds(labels: np.ndarray, groups: np.ndarray, seed: int = 0) -> tuple[list[list], int]:
    """The fewest grouped folds (StratifiedGroupKFold, shuffle, random_state `seed`, n = 2, 3, ...; then leave one
    group out) in which every class of a fold's test side is also on its train side; no group on both sides.
    ([fold = [(train units, test units)]], n folds). A class with fewer than 2 groups cannot meet it: ValueError."""
    import warnings

    from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

    labels, groups = np.asarray(labels), np.asarray(groups)
    n_groups = len(np.unique(groups))
    per_class = {c: len(np.unique(groups[labels == c])) for c in np.unique(labels)}
    if min(per_class.values()) < 2:
        raise ValueError("a class has fewer than 2 groups: no grouped fold keeps it on both sides")

    def valid(folds) -> bool:
        return all(len(test) and set(labels[test]) <= set(labels[train]) for train, test in folds)

    for n in range(2, n_groups + 1):
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)           # fewer members of a class than folds
            try:
                folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
            except ValueError:                                     # n above every class's size: no stratified split
                break
        if valid(folds):
            return [[(train, test)] for train, test in folds], n
    folds = list(LeaveOneGroupOut().split(np.zeros(len(labels)), labels, groups))
    if valid(folds):
        return [[(train, test)] for train, test in folds], len(folds)
    raise ValueError("no grouped fold keeps every class on both sides")


class Layout:
    """The fixed scoring layout, from labels and splits only (identical for every arm): the units' class codes,
    per split its folds, and per fold its test units (the parts' tests, in order) and bootstrap draws (BOOTSTRAP x
    test units, the times each test unit is drawn; one generator, seed BOOTSTRAP_SEED, over the splits and folds in
    order). `splits[name]`: a list of folds, a fold a list of (train unit indices, test unit indices) parts; a unit is
    on the test side of a fold at most once."""

    def __init__(self, unit_ids, unit_labels, splits: dict[str, list[list]], describe: dict | None = None):
        self.unit_ids = [str(u) for u in unit_ids]
        self.classes = np.unique(np.asarray(unit_labels).astype(str))
        self.codes = np.searchsorted(self.classes, np.asarray(unit_labels).astype(str))
        self.n_classes = len(self.classes)
        self.splits = {name: [[(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)) for tr, te in fold]
                              for fold in folds] for name, folds in splits.items()}
        self.describe = describe or {}
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        self.tests, self.draws = {}, {}
        for name, folds in self.splits.items():
            self.tests[name], self.draws[name] = [], []
            for fold in folds:
                test = np.concatenate([te for _, te in fold])
                if len(np.unique(test)) != len(test):
                    raise ValueError(f"split {name}: a unit is tested twice in one fold")
                picks = rng.integers(0, len(test), size=(BOOTSTRAP, len(test)))
                draws = np.zeros((BOOTSTRAP, len(test)))
                np.add.at(draws, (np.arange(BOOTSTRAP)[:, None], picks), 1)
                self.tests[name].append(test)
                self.draws[name].append(draws)
        digest = hashlib.sha256(json.dumps({"units": self.unit_ids, "labels": self.classes[self.codes].tolist(),
                                            "splits": {n: [[(tr.tolist(), te.tolist()) for tr, te in f] for f in fs]
                                                       for n, fs in self.splits.items()},
                                            "bootstrap": [BOOTSTRAP, BOOTSTRAP_SEED]}).encode())
        self.fingerprint = digest.hexdigest()

    def summary(self) -> dict:
        return {"n_units": len(self.unit_ids), "n_classes": int(self.n_classes), "fingerprint": self.fingerprint,
                "splits": {name: {"n_folds": len(folds), "n_test_units": [int(len(t)) for t in self.tests[name]],
                                  "n_test_classes": [int(len(np.unique(self.codes[t]))) for t in self.tests[name]],
                                  **(self.describe.get(name) or {})}
                           for name, folds in self.splits.items()}}


# ---- scoring one arm -----------------------------------------------------------------------------------------------

def scores_with_misses(confusion: np.ndarray, misses: np.ndarray) -> dict[str, np.ndarray]:
    """`scores` with `misses` (..., C): test samples of each class without a prediction (a unit without delivered
    cells). They are false negatives of their class and false positives of none, i.e. sklearn's balanced accuracy
    and macro F1 with the class labels only (a sample's 'no prediction' is not a class); misses = 0 gives `scores`."""
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)
    true, pred = confusion.sum(axis=-1) + misses, confusion.sum(axis=-2)
    present, labelled = true > 0, (true > 0) | (pred > 0)
    recall = np.where(present, tp / np.maximum(true, 1), 0.0).sum(axis=-1) / present.sum(axis=-1)
    f1 = np.where(labelled, 2 * tp / np.maximum(true + pred, 1), 0.0).sum(axis=-1) / labelled.sum(axis=-1)
    return {"balanced_accuracy": recall, "macro_f1": f1}


def _fold_map(layout: Layout, profiles: np.ndarray, has: np.ndarray, fold: list) -> tuple[np.ndarray, np.ndarray]:
    """Per test unit of the fold: its AP over the part's train units with a profile (a query without a profile, or
    without a positive candidate: AP 0; a unit without a profile is never a candidate), and the chance AP (positives /
    candidates over all the part's train units: the layout's, the same for every arm)."""
    ap, base = [], []
    for train, test in fold:
        candidates = train[has[train]]
        sim = profiles[test] @ profiles[candidates].T
        for row, unit in enumerate(test):
            every = layout.codes[train] == layout.codes[unit]
            if not every.any():
                raise ValueError("a test unit's label has no train-side unit")
            base.append(every.mean())
            positives = layout.codes[candidates] == layout.codes[unit]
            ap.append(average_precision(sim[row], positives) if has[unit] and positives.any() else 0.0)
    return np.asarray(ap), np.asarray(base)


def _macro_over_labels(values: np.ndarray, codes: np.ndarray, weights: np.ndarray, n_classes: int) -> np.ndarray:
    """Per row of `weights` (draws x units): the mean over the labels drawn of the weighted mean of `values`."""
    onehot = (codes[:, None] == np.arange(n_classes)).astype(np.float64)
    sums, counts = weights @ (onehot * values[:, None]), weights @ onehot
    present = counts > 0
    return np.where(present, sums / np.maximum(counts, 1e-300), 0.0).sum(axis=1) / present.sum(axis=1)


def _split_map(layout: Layout, name: str, profiles: np.ndarray, has: np.ndarray) -> tuple[float, np.ndarray, float]:
    """A split's mAP (the mean over its folds), its draws and its chance."""
    values, draws, chances = [], [], []
    for f, fold in enumerate(layout.splits[name]):
        ap, base = _fold_map(layout, profiles, has, fold)
        codes, ones = layout.codes[layout.tests[name][f]], np.ones((1, len(ap)))
        values.append(float(_macro_over_labels(ap, codes, ones, layout.n_classes)[0]))
        draws.append(_macro_over_labels(ap, codes, layout.draws[name][f], layout.n_classes))
        chances.append(float(_macro_over_labels(base, codes, ones, layout.n_classes)[0]))
    return float(np.mean(values)), np.mean(draws, axis=0), float(np.mean(chances))


def map_reading(layout: Layout, profiles: np.ndarray, has: np.ndarray) -> dict:
    """A reported (not verdict) mAP of other unit profiles: per split its value and 95% interval."""
    out = {}
    for name in layout.splits:
        if layout.splits[name]:
            point, draws, _ = _split_map(layout, name, l2(profiles), np.asarray(has, dtype=bool))
            out[name] = {"value": point, "ci95": _interval(draws)}
    return out


def evaluate(layout: Layout, cells: np.ndarray, cell_unit: np.ndarray, cell_ids=None,
             profiles: np.ndarray | None = None, classifiers=CLASSIFIERS, cap: int = CELL_CAP,
             excluded_unit=None, excluded_ids=None) -> dict:
    """One arm's reading: per split, "points" / "draws" for "<split>/mAP" and "<split>/<unit>/<classifier>/<metric>",
    chance under "<split>/chance/..." and the logreg fit records. `cells` (n x d; a memory map is read in chunks)
    are the arm's delivered cell vectors, `cell_unit` each cell's unit index (-1: not a unit's cell), `cell_ids` the
    cells' ids (the cap's sort key; None: row order); `profiles` (units x d): the unit profiles (None: the contract's
    `unit_profiles` of all the unit's cells). The cell-level probes use <= `cap` cells per unit (`cap_cells`).
    The cells the agent excluded (`excluded_unit`, `excluded_ids`: their units and ids, no vectors) enter the cap's
    sample with the delivered cells (by cell id); a sampled excluded cell is a test cell without a prediction (a
    miss) and never a train cell. A unit with neither delivered nor excluded cells counts as `cap` such misses; a
    unit without delivered cells has no profile: in the unit-level probes one miss and no train unit, in mAP an AP
    of 0 and never a candidate."""
    started = time.time()
    cell_unit = np.asarray(cell_unit, dtype=np.int64)
    if len(cell_unit) != len(cells):
        raise ValueError(f"{len(cells)} cells, {len(cell_unit)} unit indices")
    n_units, n = len(layout.unit_ids), layout.n_classes
    has = np.bincount(cell_unit[cell_unit >= 0], minlength=n_units) > 0
    profiles = unit_profiles(cells, cell_unit, n_units) if profiles is None else l2(profiles)
    if len(profiles) != n_units:
        raise ValueError(f"{len(profiles)} unit profiles for {n_units} units")
    excluded_unit = np.zeros(0, dtype=np.int64) if excluded_unit is None else np.asarray(excluded_unit, dtype=np.int64)
    if len(excluded_unit) and (cell_ids is None or excluded_ids is None or len(excluded_ids) != len(excluded_unit)):
        raise ValueError("excluded cells need their ids and the delivered cells' ids (the cap's sort key)")
    ids = None if cell_ids is None else np.concatenate([np.asarray(cell_ids).astype(str),
                                                        np.asarray(excluded_ids if len(excluded_unit) else [],
                                                                   dtype=str)])
    rows = cap_cells(np.concatenate([cell_unit, excluded_unit]), ids, cap)
    picked_excluded = rows[rows >= len(cell_unit)] - len(cell_unit)
    rows = np.sort(rows[rows < len(cell_unit)])
    sampled_excluded = np.bincount(excluded_unit[picked_excluded], minlength=n_units)
    with_any = has | (np.bincount(excluded_unit[excluded_unit >= 0], minlength=n_units) > 0)
    cell_misses = np.where(with_any, sampled_excluded, cap)       # per unit: its test cells without a prediction
    unit_misses = (~has).astype(np.int64)
    sample, sample_unit = np.asarray(cells[rows]), cell_unit[rows]
    order = np.argsort(sample_unit, kind="stable")
    bounds = np.searchsorted(sample_unit[order], np.arange(n_units + 1))
    members = [order[bounds[u]:bounds[u + 1]] for u in range(n_units)]          # each unit's sampled rows

    def rows_of(units: np.ndarray) -> np.ndarray:
        return np.concatenate([members[u] for u in units]) if len(units) else np.zeros(0, dtype=np.int64)

    points, draws, fits = {}, {}, {}
    for name, folds in layout.splits.items():
        if not folds:
            continue
        points[f"{name}/mAP"], draws[f"{name}/mAP"], points[f"{name}/chance/mAP"] = _split_map(
            layout, name, profiles, has)
        per_fold, chances = {}, {}
        for f, fold in enumerate(folds):
            test_units, weights = layout.tests[name][f], layout.draws[name][f]
            position = {int(u): i for i, u in enumerate(test_units)}
            for unit in UNITS:
                train_y, test_y = [], []
                for classifier in classifiers:
                    counts = np.zeros((len(test_units), n * n + n))           # confusion, then misses per class
                    for train, test in fold:
                        if unit == "cell":
                            tr, te = rows_of(train), rows_of(test)
                            x_tr, y_tr, x_te = sample[tr], layout.codes[sample_unit[tr]], sample[te]
                            y_te = layout.codes[sample_unit[te]]
                            at = np.array([position[int(u)] for u in sample_unit[te]], dtype=np.int64)
                            missing = cell_misses
                        else:
                            kept, shown = train[has[train]], test[has[test]]
                            x_tr, y_tr, x_te = profiles[kept], layout.codes[kept], profiles[shown]
                            y_te = layout.codes[shown]
                            at = np.array([position[int(u)] for u in shown], dtype=np.int64)
                            missing = unit_misses
                        absent = test[missing[test] > 0]
                        at_absent = np.array([position[int(u)] for u in absent], dtype=np.int64)
                        np.add.at(counts, (at_absent, n * n + layout.codes[absent]), missing[absent])
                        if classifier == classifiers[0]:
                            train_y.append(y_tr)
                            test_y += [y_te, np.repeat(layout.codes[absent], missing[absent])]
                        if not len(y_te):
                            continue
                        if not len(y_tr):                              # nothing to learn from: every test sample a miss
                            np.add.at(counts, (at, n * n + y_te), 1)
                            continue
                        if len(np.unique(y_tr)) < 2:
                            predicted = np.full(len(y_te), y_tr[0])
                        elif classifier == "knn":
                            predicted = knn_predict(x_tr, y_tr, x_te, n)
                        else:
                            predicted, fit = logreg_predict(x_tr, y_tr, x_te)
                            fits.setdefault(f"{name}/{unit}/logreg", []).append(fit)
                        np.add.at(counts, (at, y_te * n + predicted), 1)

                    def split_scores(c: np.ndarray) -> dict:
                        return scores_with_misses(c[..., :n * n].reshape(*c.shape[:-1], n, n), c[..., n * n:])

                    point, drawn = split_scores(counts.sum(axis=0)), split_scores(weights @ counts)
                    for metric in PROBE_METRICS:
                        key = f"{unit}/{classifier}/{metric}"
                        per_fold.setdefault(key, ([], []))
                        per_fold[key][0].append(float(point[metric]))
                        per_fold[key][1].append(drawn[metric])
                tr_y, te_y = np.concatenate(train_y), np.concatenate(test_y).astype(np.int64)
                if len(tr_y) and len(te_y):
                    for metric, value in chance(tr_y, te_y, n).items():
                        chances.setdefault(f"{unit}/{metric}", []).append(value)
        for key, (values, sampled) in per_fold.items():
            points[f"{name}/{key}"] = float(np.mean(values))
            draws[f"{name}/{key}"] = np.mean(sampled, axis=0)
        for key, values in chances.items():
            points[f"{name}/chance/{key}"] = float(np.mean(values))
    return {"fingerprint": layout.fingerprint, "points": points, "draws": draws, "fits": fits,
            "n_cells": int((cell_unit >= 0).sum()), "n_cells_probed": int(len(rows)), "cell_cap": int(cap),
            "n_excluded_cells": int((excluded_unit >= 0).sum()), "n_excluded_sampled": int(len(picked_excluded)),
            "n_units_without_cells": int((~has).sum()), "seconds": round(time.time() - started, 1)}


def save(path: Path, reading: dict) -> None:
    """A reading as <path> (.npz: the draws) and <path>.json (the rest)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(handle, **{key.replace("/", "|"): value for key, value in reading["draws"].items()})
    rest = {k: v for k, v in reading.items() if k != "draws"}
    path.with_name(path.name + ".json").write_text(json.dumps(rest, indent=1), encoding="utf-8")


def load(path: Path) -> dict:
    path = Path(path)
    reading = json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))
    with np.load(path) as data:
        reading["draws"] = {key.replace("|", "/"): data[key] for key in data.files}
    return reading


# ---- the comparison with the control -------------------------------------------------------------------------------

def _se(diff: np.ndarray) -> float:
    return 0.0 if np.ptp(diff) <= 1e-12 else float(np.std(diff, ddof=1))


def _metric_key(metric: str) -> str:
    return "mAP" if metric == "mAP" else "/".join((*VERDICT_PROBE, metric))


def verdict(arm: dict, control: dict, metric: str) -> dict:
    """One verdict metric of `arm` vs `control` (readings on the same layout): delta and SE on the main split, the
    guard split's delta, the verdict (VERDICT_RULE; a layout without a guard split: the verdict from the main split
    alone, with a note)."""
    if arm["fingerprint"] != control["fingerprint"]:
        raise ValueError("the arm and the control were scored on different layouts (units, labels or splits)")
    main, guard = (f"{split}/{_metric_key(metric)}" for split in SPLITS)
    delta = arm["points"][main] - control["points"][main]
    diff = np.asarray(arm["draws"][main]) - np.asarray(control["draws"][main])
    se = _se(diff)
    guard_delta = guard_se = None
    if guard in arm["points"] and guard in control["points"]:
        guard_delta = float(arm["points"][guard] - control["points"][guard])
        guard_se = _se(np.asarray(arm["draws"][guard]) - np.asarray(control["draws"][guard]))
    notes = []
    if se > 0 and delta > 2 * se:
        decision = GAIN if guard_delta is None or guard_delta >= 0 else SHORTCUT
        if guard_delta is None:
            notes.append("no guard split: the verdict from the main split alone")
    elif se > 0 and delta < -2 * se:
        decision = LOSS
    else:
        decision = NO_RESOLVABLE
    if se == 0:
        notes.append("se_zero")
    return {"delta": float(delta), "se": se, "ci95": _interval(diff), "guard_delta": guard_delta,
            "guard_se": guard_se, "verdict": decision, **({"note": "; ".join(notes)} if notes else {})}


def _interval(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def record(layout: Layout, reading: dict, control: dict | None = None, *, settings: dict | None = None) -> dict:
    """RESULT.json `full_metrics`: per verdict metric each split's value and 95% interval, the chance and (with a
    control reading) the delta, its SE and the verdict; the four probe combinations as reported readings."""
    points, draws = reading["points"], reading["draws"]

    def value(key: str) -> dict:
        return {"value": points[key], "ci95": _interval(draws[key])}

    out = {"contract": "benchmarks/_shared/FULL_METRICS_CONTRACT.md", "verdict_rule": VERDICT_RULE,
           "settings": {"knn": {"metric": "cosine", "k": KNN_K, "vote": "majority; a tie to the tied class of the "
                                "nearest neighbour; equal similarities by train row order"},
                        "logreg": {**LOGREG, "scaler": "StandardScaler fitted on train"},
                        "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED,
                                      "unit": "each fold's test units with replacement; a drawn unit brings all its "
                                              "cells; the same draws for every arm",
                                      "interval": "95% percentile"},
                        "split_value": "the mean over the split's folds",
                        "cell_cap": f"<= {CELL_CAP} cells per unit in the cell-level probes (cap_cells, seed "
                                    f"{CELL_SEED}, from the arm's own delivered cells)",
                        "units_without_cells": "misses: cell level {cap} test cells without a prediction, unit level "
                                               "one; mAP: AP 0 as a query, never a candidate".format(cap=CELL_CAP),
                        "excluded_cells": "the agent's excluded cells are sampled with the delivered ones (by cell "
                                          "id); a sampled excluded cell is a test cell without a prediction",
                        "macro_f1": "sklearn: the mean over the classes with a true or a predicted sample",
                        "verdict_probe": "cell-level kNN", **(settings or {})},
           "layout": layout.summary(), "n_cells": reading["n_cells"],
           "n_units_without_cells": reading["n_units_without_cells"], "cell_cap": reading["cell_cap"],
           "n_cells_probed": reading["n_cells_probed"], "n_excluded_cells": reading.get("n_excluded_cells", 0),
           "n_excluded_sampled": reading.get("n_excluded_sampled", 0), "seconds": reading["seconds"],
           "control": None if control is None else "compared", "metrics": {}, "probes": {}, "verdicts": None}
    for metric in METRICS:
        key = _metric_key(metric)
        chance_key = "mAP" if metric == "mAP" else f"{VERDICT_PROBE[0]}/{metric}"
        entry = {"source": "unit profiles, cosine retrieval" if metric == "mAP" else "cell-level kNN probe",
                 **{split: value(f"{split}/{key}") for split in SPLITS if f"{split}/{key}" in points},
                 "chance": {split: points[f"{split}/chance/{chance_key}"] for split in SPLITS
                            if f"{split}/chance/{chance_key}" in points},
                 "vs_control": verdict(reading, control, metric) if control is not None else None}
        out["metrics"][metric] = entry
    for unit in UNITS:
        out["probes"][unit] = {}
        for classifier in CLASSIFIERS:
            entry = {split: {m: value(f"{split}/{unit}/{classifier}/{m}") for m in PROBE_METRICS}
                     for split in SPLITS if f"{split}/{unit}/{classifier}/{PROBE_METRICS[0]}" in points}
            if not entry:
                continue
            if classifier == "logreg":
                entry["fit"] = {split: reading["fits"].get(f"{split}/{unit}/logreg") for split in SPLITS}
            out["probes"][unit][classifier] = entry
        out["probes"][unit]["chance"] = {split: {m: points[f"{split}/chance/{unit}/{m}"] for m in PROBE_METRICS}
                                         for split in SPLITS if f"{split}/chance/{unit}/{PROBE_METRICS[0]}" in points}
    if control is not None:
        out["verdicts"] = {metric: out["metrics"][metric]["vs_control"]["verdict"] for metric in METRICS}
    return out


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def markdown(full: dict) -> str:
    """The RESULT.md table of a `record`: per verdict metric each split's value [95% interval], chance, and the
    comparison with the control; then the four probe combinations."""
    def cell(entry: dict | None) -> str:
        return "n/a" if not entry else f"{_f(entry['value'])} [{_f(entry['ci95'][0])}, {_f(entry['ci95'][1])}]"

    lines = ["| metric | main | guard | chance (main / guard) | delta vs control | SE | guard delta | verdict |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for metric, entry in full["metrics"].items():
        vs = entry.get("vs_control") or {}
        lines.append(f"| {metric} ({entry['source']}) | {cell(entry.get('main'))} | {cell(entry.get('guard'))} | "
                     f"{_f(entry['chance'].get('main'))} / {_f(entry['chance'].get('guard'))} | "
                     f"{_f(vs.get('delta'), '+.3f')} | {_f(vs.get('se'))} | {_f(vs.get('guard_delta'), '+.3f')} | "
                     f"{vs.get('verdict') or ('control' if full.get('control') is None else 'n/a')} |")
    lines += ["", "| probe (reported) | main BA | main F1 | guard BA | guard F1 |", "| --- | --- | --- | --- | --- |"]
    for unit, classifiers in full["probes"].items():
        for classifier, entry in classifiers.items():
            if classifier == "chance":
                row = [f"{_f(entry.get(s, {}).get(m))}" for s in SPLITS for m in PROBE_METRICS]
            else:
                row = [cell(entry.get(s, {}).get(m)) for s in SPLITS for m in PROBE_METRICS]
            lines.append(f"| {unit} {classifier} | " + " | ".join(row) + " |")
    splits = full["layout"]["splits"]
    lines += ["", "Splits: " + "; ".join(f"{name}: {s.get('rule', '')} ({s['n_folds']} fold{'s' * (s['n_folds'] != 1)}, test units "
                                         f"{s['n_test_units']})" for name, s in splits.items())
              + f". {VERDICT_RULE}. BA = balanced accuracy, F1 = macro F1; value [95% interval over "
              f"{BOOTSTRAP} resamples of the test units]."]
    return "\n".join(lines)
