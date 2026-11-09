"""Secondary evaluation metrics (contract amendment A9): macro F1 and balanced accuracy of a siRNA probe across
experiments, scoring-only, reported next to well mAP, never the reading.

score.py runs the probe (`Probe.run`) on the vectors `score.post_process` returns for each arm and table, with the
siRNA labels it read itself, keeps each run in its per-arm score cache, and `score.final` assembles the record from
those runs (`record`) after every existing score and comparison (`evaluate` does both in one call); this module reads
no file and no label of its own. The arms are
the A9 list: frozen, dinov3 (dinov3_s<k>), shape, react, llm, openphenom and cellprofiler (random is no longer scored;
shape_top, analysis only, is not among them). Both tables (native, matched) are scored the same way.

The probe: train on the seen experiments (the cells of the train, val and unused splits), test on the held-out
experiments; the classes are the siRNAs. Units: well profiles (the mean of the L2-normalised cells, L2-normalised:
`score.well_map`'s profile) and cells (the post-processed vectors as they are). Classifiers: kNN (cosine, k = KNN_K,
majority vote; a tied vote goes to the tied class of the nearest neighbour; equal similarities by train row order) and
logistic regression (StandardScaler fitted on train, lbfgs, max_iter 2000, class_weight balanced, random_state 0).
Balanced accuracy and macro F1 (sklearn's definitions, `scores`), each with a 95% percentile interval from BOOTSTRAP
resamples of the held-out wells (seed BOOTSTRAP_SEED; a drawn well brings all its cells; the same draws for every arm,
unit and table); paired differences for the well mAP's comparisons (score.compare's names) with the same draws; the
chance level of a stratified random predictor (`chance`). The negative controls (score.NEGATIVE_CONTROL, "EMPTY") are
not in the probe: neither a class, nor a train or a test sample (score.py's rule for the well mAP).
The scores need per class only its true positives and its true and predicted counts (`class_scores`), so each held-out
well contributes three count vectors (`Probe.per_well`, wells x classes), never a classes x classes matrix (at the
full set's ~1,138 classes and ~23,000 held-out wells that would be wells x 1.3 M counts).
Units: `Probe(units=...)` (score.py `probe_units`: both units for the v3 preps; the well unit only for the full-set
v4 preps, whose cell unit -- kNN and logistic regression over every held-out cell -- would take days per run at
their size). A skipped unit is named, with its reason, in the layout (`skipped_units`) and has no runs, rows or
columns.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

CAVEAT = ("Secondary, not the pre-registered reading (well mAP, native, seen / heldout); the probe reads the siRNA "
          "labels at scoring only; small within-model differences on these metrics are not established to track "
          "biology (ViTally Consistent, Kenyon-Dean et al., ICML 2025).")
ARM_NAMES = ("frozen", "shape", "react", "llm", "openphenom", "cellprofiler")    # + dinov3, dinov3_s<k>
METRICS = ("balanced_accuracy", "macro_f1")
UNITS = ("well", "cell")
CLASSIFIERS = ("knn", "logreg")
KNN_K = 10
KNN_CHUNK = 1024
LOGREG = {"solver": "lbfgs", "max_iter": 2000, "class_weight": "balanced", "random_state": 0}
BOOTSTRAP, BOOTSTRAP_SEED = 1000, 0


def a9_arms(names) -> list[str]:
    """The A9 arms among `names`, in their order."""
    from benchmarks.representation_RxRx1.runners.control import is_dinov3

    return [a for a in names if a in ARM_NAMES or is_dinov3(a)]


def scores(confusion: np.ndarray) -> dict[str, np.ndarray]:
    """Balanced accuracy and macro F1 of confusion matrices (..., C, C; rows true, columns predicted), as sklearn
    defines them: balanced accuracy = the mean recall over the classes with a true sample; macro F1 = the mean
    2 TP / (2 TP + FP + FN) over the classes with a true or a predicted sample (`class_scores` of its diagonal and
    its row and column sums)."""
    return class_scores(np.diagonal(confusion, axis1=-2, axis2=-1), confusion.sum(axis=-1), confusion.sum(axis=-2))


def class_scores(tp: np.ndarray, true: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """`scores` from the per-class counts (..., C): true positives, true samples and predicted samples."""
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
    from benchmarks.representation_RxRx1.runners.score import l2

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


class Probe:
    """The probe's fixed layout, from the labels and the split only (the same for every arm and table): train / test
    rows per unit, the class codes, the held-out wells and the bootstrap draws (`draws`, BOOTSTRAP x held-out wells,
    the number of times each well is drawn)."""

    def __init__(self, cells, sirna: np.ndarray, units=UNITS, skipped: dict[str, str] | None = None):
        from benchmarks.representation_RxRx1.runners.score import GROUPS, NEGATIVE_CONTROL, scored_cells

        self.units = tuple(u for u in UNITS if u in units)                  # UNITS order; `skipped`: unit -> why
        self.skipped = {u: (skipped or {}).get(u, "not requested") for u in UNITS if u not in self.units}

        split, well, sirna = cells.split.to_numpy(), cells.well.to_numpy(), np.asarray(sirna)
        keep = scored_cells(sirna)                                           # the negative controls: out
        self.classes = np.unique(sirna[keep])
        self.codes = np.where(keep, np.searchsorted(self.classes, sirna), -1)
        self.well = well
        self.cell_train = np.isin(split, GROUPS["seen"]) & keep
        self.cell_test = np.isin(split, GROUPS["heldout"]) & keep
        wells, first = np.unique(well, return_index=True)                   # score._mean_by's well order
        self.well_codes = self.codes[first]
        self.well_train = np.isin(split[first], GROUPS["seen"]) & keep[first]
        self.well_test = np.isin(split[first], GROUPS["heldout"]) & keep[first]
        test_wells = wells[self.well_test]
        self.cell_test_well = np.searchsorted(test_wells, well[self.cell_test])    # each test cell's test well
        n_wells, n = len(test_wells), len(self.classes)
        picks = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n_wells, size=(BOOTSTRAP, n_wells))
        self.draws = np.zeros((BOOTSTRAP, n_wells))
        np.add.at(self.draws, (np.arange(BOOTSTRAP)[:, None], picks), 1)
        self.layout = {
            "train": "the seen experiments (train, val and unused splits)", "test": "the held-out experiments",
            "n_classes": int(n), "n_test_wells": int(n_wells),
            "well": {"n_train": int(self.well_train.sum()), "n_test": int(self.well_test.sum()),
                     "chance": chance(self.well_codes[self.well_train], self.well_codes[self.well_test], n)},
            "cell": {"n_train": int(self.cell_train.sum()), "n_test": int(self.cell_test.sum()),
                     "chance": chance(self.codes[self.cell_train], self.codes[self.cell_test], n)},
            "negative_controls": {"sirna": NEGATIVE_CONTROL, "cells": int((~keep).sum()),
                                  "wells": int((~keep[first]).sum()), "rule": "not a class, not trained or tested"}}
        for unit in self.skipped:                                           # a skipped unit: its reason only
            self.layout.pop(unit)
        if self.skipped:
            self.layout["skipped_units"] = dict(self.skipped)

    def per_well(self, unit: str, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per held-out well, its per-class counts (each wells x C): true positives, true samples, predicted samples
        (`class_scores`' inputs) of its profile's prediction, or of its cells'."""
        n = len(self.classes)
        if unit == "well":
            true, wells = self.well_codes[self.well_test], np.arange(int(self.well_test.sum()))
        else:
            true, wells = self.codes[self.cell_test], self.cell_test_well
        tp, n_true, n_pred = (np.zeros((int(self.well_test.sum()), n)) for _ in range(3))
        hit = true == predicted
        np.add.at(tp, (wells[hit], true[hit]), 1)
        np.add.at(n_true, (wells, true), 1)
        np.add.at(n_pred, (wells, predicted), 1)
        return tp, n_true, n_pred

    def run(self, vectors: np.ndarray) -> dict:
        """Per (unit, classifier): the point scores, each score's bootstrap draws and the logreg fit record."""
        from benchmarks.representation_RxRx1.runners.score import _mean_by, l2

        n = len(self.classes)
        _, profiles, _ = _mean_by(self.well, l2(vectors))
        data = {"well": lambda: (profiles[self.well_train], self.well_codes[self.well_train], profiles[self.well_test]),
                "cell": lambda: (vectors[self.cell_train], self.codes[self.cell_train], vectors[self.cell_test])}
        out = {}
        for unit in self.units:
            train, labels, test = data[unit]()
            for name in CLASSIFIERS:
                started, fit = time.time(), None
                if name == "knn":
                    predicted = knn_predict(train, labels, test, n)
                else:
                    predicted, fit = logreg_predict(train, labels, test)
                counts = self.per_well(unit, predicted)
                point = class_scores(*(c.sum(axis=0) for c in counts))
                out[(unit, name)] = {"point": {m: float(v) for m, v in point.items()},
                                     "draws": class_scores(*(self.draws @ c for c in counts)), "fit": fit,
                                     "seconds": round(time.time() - started, 1)}
        return out


def _interval(draws: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def _sides(name: str) -> tuple[str, str]:
    """A comparison "<current>_vs_<baseline>" (score.compare's names) as (current, baseline)."""
    current, _, baseline = name.partition("_vs_")
    return current, baseline


def a9_comparisons(groups: dict[str, dict | None], arms: list[str]) -> list[str]:
    """The well mAP comparison names (score.compare, every group) whose two sides are A9 arms or the default mean /
    best default seed, in their order."""
    names: list[str] = []
    allowed = set(arms) | {"default_mean", "best_default_seed"}
    for entry in groups.values():
        for name in (entry or {}).get("comparisons") or {}:
            if name not in names and all(side in allowed for side in _sides(name)):
                names.append(name)
    return names


def probe_table(runs: dict[str, dict], comparisons: list[str], family: list[str], best: str | None,
                units=UNITS) -> dict:
    """One table's record: per unit (of `units`) and classifier, every arm's scores with their 95% intervals (logreg:
    its fit) and every comparison's paired delta with the same draws (default_mean = the per-draw mean over the
    default seeds; best_default_seed = `best`)."""
    out: dict[str, Any] = {}
    for unit in units:
        out[unit] = {}
        for name in CLASSIFIERS:
            arms = {arm: {m: {"value": r[(unit, name)]["point"][m], "ci95": _interval(r[(unit, name)]["draws"][m])}
                          for m in METRICS} for arm, r in runs.items()}
            if name == "logreg":
                for arm, r in runs.items():
                    arms[arm]["fit"] = r[(unit, name)]["fit"]

            def side(arm: str, metric: str):
                if arm == "default_mean":
                    return (float(np.mean([runs[a][(unit, name)]["point"][metric] for a in family])),
                            np.mean([runs[a][(unit, name)]["draws"][metric] for a in family], axis=0))
                entry = runs[best if arm == "best_default_seed" else arm][(unit, name)]
                return entry["point"][metric], entry["draws"][metric]

            bands = {}
            for comparison in comparisons:
                current, baseline = _sides(comparison)
                bands[comparison] = {}
                for metric in METRICS:
                    (cur, cur_draws), (base, base_draws) = side(current, metric), side(baseline, metric)
                    low, high = _interval(cur_draws - base_draws)
                    bands[comparison][metric] = {"delta": float(cur - base), "ci95": [low, high],
                                                 "ci_excludes_zero": bool(low > 0 or high < 0)}
            out[unit][name] = {"arms": arms, "comparisons": bands}
    return out


def evaluate(table_vectors: Callable[[str], dict[str, np.ndarray]], keys: dict[str, str], cells, sirna: np.ndarray,
             family: list[str], groups: dict[str, dict]) -> dict:
    """RESULT.json `secondary`. `table_vectors(table)`: the A9 arms' vectors as `score.post_process` returns them for
    that table; `keys`: each arm's state key (an arm with an earlier arm's key reuses its probe, as score scores one
    state once); `groups[table]`: score.compare's groups of that table (the comparison names, the best seed)."""
    from benchmarks.representation_RxRx1.runners.score import TABLES

    started = time.time()
    probe = Probe(cells, sirna)
    runs, seconds = {}, {}
    for table in TABLES:
        tick = time.time()
        vectors = table_vectors(table)
        by_key: dict[str, dict] = {}
        runs[table] = {}
        for arm in a9_arms(vectors):
            if keys[arm] not in by_key:
                by_key[keys[arm]] = probe.run(np.asarray(vectors[arm]))
            runs[table][arm] = by_key[keys[arm]]
        del vectors
        seconds[table] = time.time() - tick
    return record(probe, runs, family, groups, started, seconds)


def record(probe: Probe, runs: dict[str, dict[str, dict]], family: list[str], groups: dict[str, dict],
           started: float, seconds: dict[str, float] | None = None) -> dict:
    """RESULT.json `secondary` from the probe runs already made: `runs[table][arm]` = `Probe.run` of that arm's vectors
    in that table (the A9 arms, in their order; score.py keeps them in its per-arm score cache); `seconds[table]`: the
    time those runs took here (none when they were read from the cache), the assembly's own time is added; `started`:
    the start of the step (`seconds.total`)."""
    from benchmarks.representation_RxRx1.runners.score import TABLES

    out: dict[str, Any] = {
        "rule": "contract amendment A9: macro F1 and balanced accuracy of a siRNA probe across experiments, "
                "scoring-only; labels read only by score.final; both tables, the A9 arms (shape_top excluded)",
        "caveat": CAVEAT,
        "settings": {
            "layout": probe.layout,
            "units": {u: {"well": "mean of the L2-normalised cells, L2-normalised (the well mAP's profile)",
                          "cell": "the post-processed vector"}[u] for u in probe.units},
            "knn": {"metric": "cosine", "k": KNN_K,
                    "vote": "majority; a tie goes to the tied class of the nearest neighbour; equal similarities by "
                            "train row order"},
            "logreg": {**LOGREG, "scaler": "StandardScaler fitted on train"},
            "metrics": "balanced accuracy and macro F1 (sklearn definitions)",
            "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED,
                          "unit": "the held-out wells, with replacement; a drawn well brings all its cells",
                          "interval": "95% percentile", "paired": "the same draws for every arm, unit and table"},
            "chance": "a stratified random predictor in expectation (the train class frequencies): balanced accuracy "
                      "exact; macro F1 = the F1 of the expected confusion matrix",
            "comparisons": "the well mAP comparisons between A9 arms; default_mean = the per-draw mean of the "
                           "default seeds; best_default_seed = the well mAP's best default seed on the heldout group "
                           "of the table"},
        "arms": [], "seconds": {}}
    if probe.skipped:
        out["settings"]["skipped_units"] = dict(probe.skipped)
    for table in TABLES:
        tick = time.time()
        arms = list(runs[table])
        comparisons = a9_comparisons(groups[table], arms)
        best = (groups[table].get("heldout") or {}).get("best_default_seed")
        out["arms"] = arms
        out[table] = {"comparisons": comparisons, **probe_table(runs[table], comparisons, family, best, probe.units)}
        out["seconds"][table] = round((seconds or {}).get(table, 0.0) + time.time() - tick, 1)
    out["seconds"]["total"] = round(time.time() - started, 1)
    return out


# ---- RESULT.md ----------------------------------------------------------------------------------------------------

def _f(value, spec=".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def markdown(record: dict) -> str:
    """RESULT.md's "Secondary metrics" section: per table the probe's scores, chance and paired deltas; the caveat."""
    from benchmarks.representation_RxRx1.runners.score import TABLES

    layout = record["settings"]["layout"]
    units = [u for u in UNITS if u in record["settings"]["units"]]
    columns = [(u, c, m) for u in units for c in CLASSIFIERS for m in METRICS]
    cells = (f", cells {layout['cell']['n_train']} / {layout['cell']['n_test']}" if "cell" in units
             else "; the cell unit is skipped")
    head = " | ".join(f"{u} {c} {'BA' if m == 'balanced_accuracy' else 'F1'}" for u, c, m in columns)
    rule = "| --- |" + " --- |" * len(columns)
    text = ("\n\n## Secondary metrics (contract amendment A9)\n\n" + record["caveat"] + "\n\n"
            f"siRNA probe across experiments: train on the seen experiments, test on the held-out experiments "
            f"({layout['n_classes']} siRNAs; wells {layout['well']['n_train']} / {layout['well']['n_test']}{cells}); "
            f"kNN cosine k = {KNN_K}, logistic regression "
            f"(StandardScaler, lbfgs, class_weight balanced). BA = balanced accuracy, F1 = macro F1; value [95% "
            f"interval over {BOOTSTRAP} resamples of the {layout['n_test_wells']} held-out wells]; chance = a "
            "stratified random predictor in expectation; deltas are paired over the same resamples (* = the "
            "interval excludes 0).")
    for unit, why in (layout.get("skipped_units") or {}).items():
        text += f" The {unit} unit is skipped for this prep: {why}."
    negative = layout.get("negative_controls") or {}
    if negative.get("wells"):
        text += (f" The {negative['wells']} negative-control wells ({negative['sirna']}, {negative['cells']} cells) "
                 "are not in the probe.")
    for table in TABLES:
        entry = record[table]

        def cell(value: dict) -> str:
            return f"{_f(value['value'])} [{_f(value['ci95'][0])}, {_f(value['ci95'][1])}]"

        def band(value: dict) -> str:
            return (f"{_f(value['delta'], '+.3f')} [{_f(value['ci95'][0], '+.3f')}, {_f(value['ci95'][1], '+.3f')}]"
                    + (" *" if value["ci_excludes_zero"] else ""))

        rows = [f"### {table} table: siRNA probe", "", f"| arm | {head} |", rule]
        rows += [f"| {arm} | " + " | ".join(cell(entry[u][c]["arms"][arm][m]) for u, c, m in columns) + " |"
                 for arm in record["arms"]]
        rows.append("| chance | " + " | ".join(_f(layout[u]["chance"][m]) for u, _, m in columns) + " |")
        rows += ["", f"| comparison | {head} |", rule]
        rows += [f"| {name} | " + " | ".join(band(entry[u][c]["comparisons"][name][m]) for u, c, m in columns) + " |"
                 for name in entry["comparisons"]]
        text += "\n\n" + "\n".join(rows)
    return text
