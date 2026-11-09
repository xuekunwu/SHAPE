#!/usr/bin/env python3
"""Post-hoc discriminative readouts of ONE scored full_LIVECell arm run (reported, not verdicts).

    python extra_readouts.py --prep PREP --run RUN --out OUT.json [--md] [--workers N]

Needs the repo on PYTHONPATH (benchmarks.*). Reads the delivered run (verified against DELIVERY.json exactly as
harness/score.py does; a changed artifact, another preparation or a changed private file is refused) and the
scorer-side private labels and GT of the preparation; never writes into either. Writes OUT.json and, with --md, a
markdown with the same stem. The harness (score.py, compare.py, data.py, cli.py, spec.py) is only imported.

Layout: the verdict layout of score.full_layout on the heldout units, main split only (single direction): fit on
heldout_fit (stages 2-3), test on heldout_test (stages 4-5). Cell-level probes use <= full_metrics.CELL_CAP cells per
unit (cap_cells, from the delivered and the excluded cells exactly as full_metrics.evaluate); a sampled excluded cell
is a miss, a unit without delivered cells is a miss (one at the unit level, CELL_CAP at the cell level when it has no
excluded cells either). A miss is a false negative of its class and never a false positive.

Readouts (the verdict probes plus):
  1. classification: per class precision / recall / F1, the confusion matrix and the largest off-diagonal confusions
     of the cell-level kNN (the verdict probe), the cell-level logistic regression and the unit-level kNN and logistic
     regression (delivered corrected profiles);
  2. one-vs-rest AUROC and AUPRC per class and macro, from the logistic-regression probabilities and from the kNN vote
     fractions (a miss ranks below every prediction; positives only), with the chance level (AUROC 0.5, AUPRC = the
     class prevalence among the scored samples);
  3. cell morphology: ridge regression from the uncorrected cell embedding to the matched GT instance's log area,
     aspect ratio (major / minor axis) and solidity; R^2, Pearson r, MSE on heldout_test cells, overall and per type;
  4. stage leakage: within each cell type, leave-one-well-index-out prediction of the culture-time stage from the unit
     profiles over ALL units, balanced accuracy macro over types, uncorrected vs delivered (lower is better).

  5. batch integration (scib-style, shape/tools/batch_corrector/quant.py exactly as the batch search calls it): batch
     silhouette, iLISI, graph connectivity, cLISI, label silhouette, NMI and ARI of the unit profiles, batch = the
     culture-time stage, label = cell type, uncorrected vs delivered, over all units and over the heldout units;
     point estimates, the paired delta and the admission evidence of batch_SEARCH.json.

Bootstrap: 1000 resamples of the heldout_test UNITS (seed 0, the verdict layout's own draws, the same for every arm);
a drawn unit brings all its sampled cells (cell level) or its one profile (unit level). The fitted models are not
refitted per resample. The morphology intervals use the same unit draws; the stage-leakage intervals resample the units
of each type (seed 0, one generator over the types in sorted order).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks._shared import full_metrics as fm
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, within
from benchmarks.full_CPJUMP1.harness.score import excluded_image
from benchmarks.full_LIVECell.harness import score as S
from benchmarks.full_LIVECell.harness.data import load_prepared

NEEDED = ("cells.json", "excluded_cells.json", "embedding.npy", "corrected.npy", "profiles.npy", "profiles.csv",
          "segmentation.json")
TOP_CONFUSIONS = 10
RANK_CHUNK = 250                                  # bootstrap rows per block of the weighted rank metrics
RIDGE_ALPHAS = tuple(float(a) for a in 10.0 ** np.arange(-2, 6))
RIDGE_FOLDS = 5
TARGETS = ("log_area", "aspect_ratio", "solidity")
N_STAGES = 6
ROUND = 6
PROBES = (("cell", "knn"), ("cell", "logreg"), ("unit", "knn"), ("unit", "logreg"))
DEFINITIONS = {
    "classification": "per class precision = TP / predicted, recall = TP / (test samples of the class incl. misses), "
                      "F1 = 2 TP / (test + predicted); a zero denominator gives 0 (sklearn zero_division=0)",
    "auroc_auprc": "one-vs-rest per class from the probe's class scores (logreg: predict_proba; kNN: the vote "
                   "fraction of the 10 neighbours), ties share one threshold (sklearn's rule); misses are positives "
                   "of their class ranked below every prediction and never negatives; chance AUROC 0.5, chance "
                   "AUPRC = P / (P + N) of the scored set; macro = the mean over the classes with a positive and a "
                   "negative",
    "morphology": "cells of the heldout units; target GT instance = the one of the highest IoU (then the lowest GT "
                  "index) with the cell's delivered mask instance, matched when IoU >= 0.5 (score.crop_scores); "
                  "unmatched cells are left out of the fit and the test; ridge on the standardised uncorrected "
                  f"embedding, alpha per target from {RIDGE_ALPHAS} by {RIDGE_FOLDS}-fold GroupKFold over heldout_fit "
                  "units (heldout_fit cells only); R^2 = 1 - SSE / SST about the evaluated group's own mean (overall: "
                  "all test cells, so it includes between-type variance; per type: the type's own mean)",
    "stage_leakage": "per cell type, leave-one-well-index-out over ALL units (train, val, heldout), predict the "
                     "culture-time stage (6 classes) from the L2-normalised unit profile (logistic regression of "
                     "full_metrics.LOGREG and cosine kNN k=10); balanced accuracy of the pooled out-of-fold "
                     "predictions per type, macro over types; chance 1/6; lower is better after correction",
}


# ---- verification and loading ------------------------------------------------------------------------------------

def load_run(prep: Path, run: Path) -> dict:
    """The verified inputs: the delivery is complete, every delivered artifact and the preparation's private files
    match their hashes (as score.score); the metadata, units and verdict layout of the scorer."""
    prep, run = Path(prep), Path(run)
    delivery = read_json(run / "DELIVERY.json")
    if delivery.get("status") != "complete" or delivery.get("smoke"):
        raise ValueError("run has not delivered a complete pipeline")
    for path, sha in delivery["artifacts"].items():
        if digest(within(run, path)) != sha:
            raise ValueError(f"delivered artifact changed: {path}")
    missing = [n for n in NEEDED if n not in delivery["artifacts"]]
    if missing:
        raise ValueError(f"the delivery lacks {missing}")
    images = load_prepared(prep)
    if delivery["prepare_sha256"] != digest(prep / "PREPARE.json"):
        raise ValueError("delivery belongs to another preparation")
    prepared = read_json(prep / "PREPARE.json")
    if digest(prep / "private" / "labels.json") != prepared["labels_sha256"]:
        raise ValueError("private evaluation labels changed")
    if digest(prep / S.GT_INDEX) != prepared.get("gt_index_sha256"):
        raise ValueError("private GT index changed")
    labels = pd.DataFrame(read_json(prep / "private" / "labels.json"))
    meta = S.unit_meta(images, labels, run / "profiles.csv")
    layout, units = S.full_layout(meta)
    return {"prep": prep, "run": run, "delivery": delivery, "images": images, "prepared": prepared, "meta": meta,
            "layout": layout, "units": units, "design": S.design_counts(meta)}


def load_cells(inp: dict) -> dict:
    """The delivered cells of the heldout units (their unit index in `units`, -1 otherwise) and the excluded cells."""
    run, units = inp["run"], inp["units"]
    cells = read_json(run / "cells.json")
    embedding = np.load(run / "embedding.npy", mmap_mode="r")
    if len(cells) != len(embedding):
        raise ValueError("cell embedding rows disagree with cells.json")
    position = {w: i for i, w in enumerate(units.well)}
    well_of = {r["image_id"]: r["well"] for r in inp["images"]}
    excluded = [str(c) for c in read_json(run / "excluded_cells.json")]
    if any(excluded_image(c) not in well_of for c in excluded):
        raise ValueError("an excluded cell names an image outside the preparation")
    return {"cells": cells, "embedding": embedding, "cell_ids": [c["cell_id"] for c in cells],
            "cell_unit": np.array([position.get(c["well"], -1) for c in cells], dtype=np.int64),
            "excluded": excluded,
            "excluded_unit": np.array([position.get(well_of[excluded_image(c)], -1) for c in excluded],
                                      dtype=np.int64)}


# ---- classifiers with class scores ---------------------------------------------------------------------------------

def knn_votes(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray, n_classes: int,
              k: int = fm.KNN_K) -> tuple[np.ndarray, np.ndarray]:
    """full_metrics.knn_predict's neighbours (cosine, equal similarities by train row order): (vote fractions
    (n_test, C), predictions with the tie rule of knn_predict: a tied vote to the tied class of the nearest neighbour)."""
    train, test = fm.l2(train), fm.l2(test)
    k = min(k, len(train))
    fractions = np.empty((len(test), n_classes))
    predicted = np.empty(len(test), dtype=np.int64)
    for start in range(0, len(test), fm.KNN_CHUNK):
        sim = test[start:start + fm.KNN_CHUNK] @ train.T
        rows = np.arange(len(sim))[:, None]
        kth = -np.partition(-sim, k - 1, axis=1)[:, k - 1:k]
        above, equal = sim > kth, sim == kth
        chosen = above | (equal & (np.cumsum(equal, axis=1) <= k - above.sum(axis=1, keepdims=True)))
        keep = np.nonzero(chosen)[1].reshape(len(sim), k)
        keep = np.take_along_axis(keep, np.lexsort((keep, -sim[rows, keep]), axis=1), axis=1)
        neighbours = train_labels[keep]
        votes = (neighbours[:, :, None] == np.arange(n_classes)).sum(axis=1)
        tied = votes == votes.max(axis=1, keepdims=True)
        first = tied[rows, neighbours].argmax(axis=1)
        fractions[start:start + len(sim)] = votes / k
        predicted[start:start + len(sim)] = neighbours[np.arange(len(sim)), first]
    return fractions, predicted


def logreg_proba(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba (n_test, C) of full_metrics.logreg_predict's model (StandardScaler + LogisticRegression(LOGREG));
    a class absent from the train labels gets probability 0."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train = np.asarray(train, dtype=np.float64)
    scaler = StandardScaler().fit(train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(**fm.LOGREG).fit(scaler.transform(train), train_labels)
    out = np.zeros((len(test), n_classes))
    out[:, model.classes_] = model.predict_proba(scaler.transform(np.asarray(test, dtype=np.float64)))
    return out


def class_scores(classifier: str, x_tr, y_tr, x_te, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """(class scores (n_test, C), predictions) of one probe; fewer than two train classes: the single class."""
    if len(np.unique(y_tr)) < 2:
        scores = np.zeros((len(x_te), n_classes))
        scores[:, y_tr[0]] = 1.0
        return scores, np.full(len(x_te), y_tr[0], dtype=np.int64)
    if classifier == "knn":
        return knn_votes(x_tr, y_tr, x_te, n_classes)
    proba = logreg_proba(x_tr, y_tr, x_te, n_classes)
    return proba, proba.argmax(axis=1).astype(np.int64)


# ---- weighted rank metrics ---------------------------------------------------------------------------------------

def rank_metrics(score: np.ndarray, pos: np.ndarray, keep: np.ndarray, base: np.ndarray, unit_weights: np.ndarray,
                 sample_unit: np.ndarray) -> dict:
    """One-vs-rest AUROC and AUPRC of weighted samples, one value per row of `unit_weights` (R x U; a sample's weight
    = base * the weight of its unit `sample_unit`). Samples with `keep` False are dropped; tied scores share one
    threshold (sklearn's roc_auc_score / average_precision_score with sample_weight). {auroc, auprc, pos, neg}: (R,);
    NaN where a resample has no positive or no negative."""
    idx = np.flatnonzero(keep)
    if not len(idx):
        return {name: np.full(len(unit_weights), np.nan) for name in ("auroc", "auprc", "pos", "neg")}
    order = idx[np.argsort(-score[idx], kind="stable")]
    s = score[order]
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    p, scale, owner = pos[order], base[order].astype(np.float64), sample_unit[order]
    n_rows = len(unit_weights)
    out = {name: np.full(n_rows, np.nan) for name in ("auroc", "auprc", "pos", "neg")}
    for start in range(0, n_rows, RANK_CHUNK):
        w = unit_weights[start:start + RANK_CHUNK][:, owner] * scale
        tp = np.cumsum(w * p, axis=1)[:, ends]
        fp = np.cumsum(w * ~p, axis=1)[:, ends]
        total_p, total_n = tp[:, -1], fp[:, -1]
        ok = (total_p > 0) & (total_n > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            tpr, fpr = tp / total_p[:, None], fp / total_n[:, None]
            recall = tpr
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        zero = np.zeros((len(tp), 1))
        d_fpr, d_rec = np.diff(np.hstack([zero, fpr]), axis=1), np.diff(np.hstack([zero, recall]), axis=1)
        mean_tpr = (np.hstack([zero, tpr])[:, 1:] + np.hstack([zero, tpr])[:, :-1]) / 2
        block = slice(start, start + len(tp))
        out["auroc"][block] = np.where(ok, (d_fpr * mean_tpr).sum(axis=1), np.nan)
        out["auprc"][block] = np.where(ok, (d_rec * precision).sum(axis=1), np.nan)
        out["pos"][block], out["neg"][block] = total_p, total_n
    return out


# ---- reporting helpers ---------------------------------------------------------------------------------------------

def _interval(draws: np.ndarray) -> list[float] | None:
    draws = np.asarray(draws, dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])] if len(draws) else None


def vc(value, draws) -> dict:
    """{value, ci95} of a point value and its bootstrap draws (None where undefined)."""
    value = None if value is None or not np.isfinite(value) else float(value)
    return {"value": value, "ci95": None if value is None else _interval(draws)}


def clean(obj):
    """JSON-ready: numpy scalars and arrays to python, floats rounded (ROUND), NaN to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), ROUND)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def prf(conf: np.ndarray, miss: np.ndarray) -> dict[str, np.ndarray]:
    """Per class precision, recall, F1 and test support of confusion matrices (..., C, C; rows true) with `miss`
    (..., C) test samples without a prediction (false negatives of their class)."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    true, pred = conf.sum(axis=-1) + miss, conf.sum(axis=-2)
    return {"precision": np.where(pred > 0, tp / np.maximum(pred, 1), 0.0),
            "recall": np.where(true > 0, tp / np.maximum(true, 1), 0.0),
            "f1": np.where(true + pred > 0, 2 * tp / np.maximum(true + pred, 1), 0.0), "support": true}


# ---- readouts 1 and 2: classification ----------------------------------------------------------------------------

def probe_sets(inp: dict, data: dict) -> dict:
    """The main split's train / test matrices of the cell-level (sampled cells, uncorrected embedding) and unit-level
    (delivered corrected profiles) probes, as full_metrics.evaluate builds them: per level x_tr, y_tr, x_te, y_te,
    `te_pos` (a test sample's column in the test units) and `miss` (per test unit: test samples without a
    prediction)."""
    layout, units = inp["layout"], inp["units"]
    (train, test), = layout.splits["main"][0]
    n_units, cap = len(layout.unit_ids), fm.CELL_CAP
    cell_unit, excluded_unit = data["cell_unit"], data["excluded_unit"]
    has = np.bincount(cell_unit[cell_unit >= 0], minlength=n_units) > 0
    ids = np.concatenate([np.asarray(data["cell_ids"]).astype(str), np.asarray(data["excluded"], dtype=str)])
    rows = fm.cap_cells(np.concatenate([cell_unit, excluded_unit]), ids, cap)
    sampled_excluded = np.bincount(excluded_unit[rows[rows >= len(cell_unit)] - len(cell_unit)], minlength=n_units)
    rows = np.sort(rows[rows < len(cell_unit)])
    with_any = has | (np.bincount(excluded_unit[excluded_unit >= 0], minlength=n_units) > 0)
    cell_misses = np.where(with_any, sampled_excluded, cap)
    sample, sample_unit = np.asarray(data["embedding"][rows]), cell_unit[rows]
    order = np.argsort(sample_unit, kind="stable")
    bounds = np.searchsorted(sample_unit[order], np.arange(n_units + 1))
    members = [order[bounds[u]:bounds[u + 1]] for u in range(n_units)]

    def rows_of(selected):
        return np.concatenate([members[u] for u in selected]) if len(selected) else np.zeros(0, dtype=np.int64)

    place = np.full(n_units, -1, dtype=np.int64)
    place[test] = np.arange(len(test))
    codes = layout.codes
    tr, te = rows_of(train), rows_of(test)
    sets = {"cell": {"x_tr": sample[tr], "y_tr": codes[sample_unit[tr]], "x_te": sample[te],
                     "y_te": codes[sample_unit[te]], "te_pos": place[sample_unit[te]], "miss": cell_misses[test]}}
    profiles = fm.l2(np.load(inp["run"] / "corrected.npy")[units["index"].to_numpy()])
    kept, shown = train[has[train]], test[has[test]]
    sets["unit"] = {"x_tr": profiles[kept], "y_tr": codes[kept], "x_te": profiles[shown], "y_te": codes[shown],
                    "te_pos": place[shown], "miss": (~has[test]).astype(np.int64)}
    for name, s in sets.items():
        if not len(s["y_tr"]) or not len(s["y_te"]):
            raise ValueError(f"{name}-level probe has no train or no test samples")
    return {"sets": sets, "test": test, "train": train, "n_cells_sampled": int(len(rows)),
            "n_excluded_sampled": int((sampled_excluded[test]).sum()), "has": has}


def classification(inp: dict, data: dict) -> dict:
    """Readouts 1 and 2 for the four probes of the main split."""
    layout = inp["layout"]
    n = layout.n_classes
    names = [str(c) for c in layout.classes]
    built = probe_sets(inp, data)
    test, sets = built["test"], built["sets"]
    n_test = len(test)
    draws_main = layout.draws["main"][0]
    unit_weights = np.vstack([np.ones((1, n_test)), draws_main])          # row 0 = the point estimate
    test_class = layout.codes[test]
    out = {"classes": names, "n_test_units": int(n_test), "n_fit_units": int(len(built["train"])),
           "n_cells_sampled": built["n_cells_sampled"], "n_excluded_sampled_test": built["n_excluded_sampled"],
           "n_units_without_cells_test": int((~built["has"][test]).sum()), "probes": {}}
    for level, classifier in PROBES:
        s = sets[level]
        scores, predicted = class_scores(classifier, s["x_tr"], s["y_tr"], s["x_te"], n)
        conf = np.zeros((n_test, n, n))
        np.add.at(conf, (s["te_pos"], s["y_te"], predicted), 1)
        miss = np.zeros((n_test, n))
        miss[np.arange(n_test), test_class] = s["miss"]
        counts = np.concatenate([conf.reshape(n_test, n * n), miss], axis=1)
        pooled = unit_weights @ counts                                    # row 0 = point, then the draws
        c_all, m_all = pooled[:, :n * n].reshape(-1, n, n), pooled[:, n * n:]
        per = prf(c_all, m_all)
        macro = fm.scores_with_misses(c_all, m_all)
        # rank metrics: the scored samples plus one pseudo-sample per test unit with misses (score -inf, weight = count)
        absent = np.flatnonzero(s["miss"] > 0)
        score_all = np.vstack([scores, np.full((len(absent), n), -np.inf)])
        y_all = np.concatenate([s["y_te"], test_class[absent]])
        owner = np.concatenate([s["te_pos"], absent])
        base = np.concatenate([np.ones(len(s["y_te"])), s["miss"][absent].astype(np.float64)])
        is_miss = np.concatenate([np.zeros(len(s["y_te"]), bool), np.ones(len(absent), bool)])
        auroc, auprc, prevalence = [], [], []
        for c in range(n):
            r = rank_metrics(score_all[:, c], y_all == c, ~is_miss | (y_all == c), base, unit_weights, owner)
            auroc.append(r["auroc"])
            auprc.append(r["auprc"])
            prevalence.append(r["pos"][0] / (r["pos"][0] + r["neg"][0]) if r["pos"][0] + r["neg"][0] > 0 else np.nan)
        auroc, auprc = np.array(auroc).T, np.array(auprc).T              # (R+1, C)
        with np.errstate(invalid="ignore"):
            macro_auroc, macro_auprc = np.nanmean(auroc, axis=1), np.nanmean(auprc, axis=1)
            chance_auprc = float(np.nanmean(prevalence))
        point_conf, point_miss = c_all[0], m_all[0]
        off = [(int(point_conf[i, j]), i, j) for i in range(n) for j in range(n) if i != j and point_conf[i, j] > 0]
        off.sort(key=lambda t: (-t[0], t[1], t[2]))
        true_total = np.maximum(c_all.sum(axis=-1) + m_all, 1)
        out["probes"][f"{level}_{classifier}"] = {
            "n_test_samples": int(len(s["y_te"]) + s["miss"].sum()), "n_misses": int(s["miss"].sum()),
            "macro_f1": vc(macro["macro_f1"][0], macro["macro_f1"][1:]),
            "balanced_accuracy": vc(macro["balanced_accuracy"][0], macro["balanced_accuracy"][1:]),
            "macro_auroc": vc(macro_auroc[0], macro_auroc[1:]), "macro_auprc": vc(macro_auprc[0], macro_auprc[1:]),
            "chance": {"auroc": 0.5, "auprc_macro_prevalence": chance_auprc},
            "per_class": {names[c]: {
                "support": int(per["support"][0, c]),
                **{m: vc(per[m][0, c], per[m][1:, c]) for m in ("precision", "recall", "f1")},
                "auroc": vc(auroc[0, c], auroc[1:, c]), "auprc": vc(auprc[0, c], auprc[1:, c]),
                "auprc_chance": None if not np.isfinite(prevalence[c]) else float(prevalence[c])} for c in range(n)},
            "confusion": {"rows_true_columns_predicted": point_conf.astype(int), "misses_per_true_class": point_miss.astype(int)},
            "top_confusions": [{"true": names[i], "predicted": names[j], "count": count,
                                "rate_of_true_class": vc(point_conf[i, j] / true_total[0, i],
                                                         c_all[1:, i, j] / true_total[1:, i])}
                               for count, i, j in off[:TOP_CONFUSIONS]]}
    return out


# ---- readout 3: cell morphology ----------------------------------------------------------------------------------

_CTX: dict = {}


def _init_worker(prep, run, artifacts, gt_index, mask_rows):
    _CTX.update(prep=Path(prep), run=Path(run), artifacts=artifacts, gt_index=gt_index, mask_rows=mask_rows)


def local_mask_row(run: Path, row: dict) -> dict:
    """The segmentation.json row with its mask path inside `run` (a copied run keeps the original absolute paths); the
    binding to the delivery (delivered_mask) still applies."""
    path = Path(row["mask"])
    try:
        path.resolve().relative_to(Path(run).resolve())
        return row
    except ValueError:
        if len(path.parts) < 3 or path.parts[-3] != "segment":
            raise ValueError(f"segmentation mask outside the run: {row['image_id']}") from None
        return {**row, "mask": str(Path(run).resolve() / "segment" / path.parts[-2] / path.parts[-1])}


def best_targets(o: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per delivered pred instance (index into o['pred_labels']): the GT index of the highest IoU (then the lowest GT
    index), and that IoU (-1, 0 for an instance without overlap): score.crop_scores's matching."""
    target = np.full(len(o["pred_labels"]), -1, dtype=np.int64)
    iou = np.zeros(len(o["pred_labels"]))
    order = np.lexsort((o["pair_gt"], -o["pair_iou"], o["pair_pred"]))
    first = np.unique(o["pair_pred"][order], return_index=True)[1]
    target[o["pair_pred"][order][first]] = o["pair_gt"][order][first]
    iou[o["pair_pred"][order][first]] = o["pair_iou"][order][first]
    return target, iou


def image_targets(task: tuple) -> dict:
    """One image: per cell (its mask_label, in the task's order) matched flag and the matched GT instance's log area,
    aspect ratio and solidity; plus the image's GT instance count and the distinct matched instances."""
    from skimage.measure import regionprops_table

    image_id, labels = task
    ctx = _CTX
    gt = S.gt_labels(ctx["prep"], ctx["gt_index"][image_id])
    mask = S.delivered_mask(ctx["run"], local_mask_row(ctx["run"], ctx["mask_rows"][image_id]), ctx["artifacts"])
    o = S.overlap(gt, mask)
    target, iou = best_targets(o)
    values = np.full((len(labels), len(TARGETS)), np.nan)
    at = np.searchsorted(o["pred_labels"], labels)
    if len(labels) and (len(o["pred_labels"]) == 0 or (
            o["pred_labels"][np.minimum(at, len(o["pred_labels"]) - 1)] != labels).any()):
        raise ValueError(f"{image_id}: a cell's mask_label is not in its delivered mask")
    matched = iou[at] >= S.MATCH_IOU if len(labels) else np.zeros(0, dtype=bool)
    wanted = np.unique(o["gt_labels"][target[at[matched]]])
    if len(wanted):
        table = regionprops_table(np.where(np.isin(gt, wanted), gt, 0).astype(np.int64),
                                  properties=("label", "area", "major_axis_length", "minor_axis_length", "solidity"))
        with np.errstate(invalid="ignore", divide="ignore"):
            aspect = np.where(table["minor_axis_length"] > 0, table["major_axis_length"] / table["minor_axis_length"],
                              np.nan)
            props = np.stack([np.log(table["area"]), aspect, table["solidity"]], axis=1)
        where = np.searchsorted(table["label"], o["gt_labels"][target[at[matched]]])
        values[matched] = props[where]
    return {"image_id": image_id, "matched": matched, "values": values, "n_gt": int(len(o["gt_labels"])),
            "n_gt_matched": int(len(np.unique(target[at[matched]])))}


def cell_targets(inp: dict, data: dict, workers: int) -> dict:
    """Per heldout cell (rows into cells.json): the GT readings of its matched instance (NaN where unmatched)."""
    cells, cell_unit = data["cells"], data["cell_unit"]
    rows = np.flatnonzero(cell_unit >= 0)
    by_image: dict[str, list[int]] = {}
    for r in rows:
        by_image.setdefault(cells[r]["image_id"], []).append(int(r))
    mask_rows = {r["image_id"]: r for r in read_json(inp["run"] / "segmentation.json")}
    index = {r["image_id"]: r for r in read_json(inp["prep"] / S.GT_INDEX)}
    for image in by_image:
        if image not in mask_rows or image not in index:
            raise ValueError(f"cell image {image} is outside the preparation or the segmentation")
    for r in rows:
        label = cells[r].get("mask_label")
        if type(label) is not int or label <= 0:
            raise ValueError(f"cell {cells[r]['cell_id']}: mask_label must be a positive integer")
    # every heldout image is read (also one without delivered cells: its GT instances count in the coverage)
    heldout_images = sorted({r["image_id"] for r in inp["images"] if r["usage"] == S.HELDOUT})
    tasks = [(image, np.array([cells[r]["mask_label"] for r in by_image.get(image, [])], dtype=np.int64))
             for image in heldout_images]
    init = (inp["prep"], inp["run"], inp["delivery"]["artifacts"], index, mask_rows)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=init) as pool:
            results = list(pool.map(image_targets, tasks, chunksize=4))
    else:
        _init_worker(*init)
        results = [image_targets(t) for t in tasks]
    matched = np.zeros(len(cells), dtype=bool)
    values = np.full((len(cells), len(TARGETS)), np.nan)
    for (image, _), res in zip(tasks, results):
        at = by_image.get(image, [])
        matched[at] = res["matched"]
        values[at] = res["values"]
    n_gt = {r["image_id"]: r["n_gt"] for r in results}
    n_gt_matched = {r["image_id"]: r["n_gt_matched"] for r in results}
    return {"rows": rows, "matched": matched, "values": values, "n_gt": n_gt, "n_gt_matched": n_gt_matched,
            "images": {c["image_id"] for c in cells}}


def ridge_fit(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas=RIDGE_ALPHAS, folds: int = RIDGE_FOLDS) -> dict:
    """Ridge with an intercept on the standardised features (the fit set's mean and s.d.); alpha per target by
    GroupKFold(`folds`, unshuffled) over `groups` on the fit set only (the smallest alpha wins a tie)."""
    from sklearn.model_selection import GroupKFold

    x = np.asarray(x, dtype=np.float64)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = (x - mu) / sd
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("ridge needs at least two heldout_fit units with matched cells")
    sse = np.zeros((len(alphas), y.shape[1]))
    for tr, va in GroupKFold(n_splits=min(folds, n_groups)).split(z, groups=groups):
        m, ym = z[tr].mean(axis=0), y[tr].mean(axis=0)
        zc = z[tr] - m
        w, v = np.linalg.eigh(zc.T @ zc)
        vb = v.T @ (zc.T @ (y[tr] - ym))
        u = (z[va] - m) @ v
        for a, alpha in enumerate(alphas):
            pred = u @ (vb / (w + alpha)[:, None]) + ym
            sse[a] += ((y[va] - pred) ** 2).sum(axis=0)
    chosen = np.array([alphas[i] for i in sse.argmin(axis=0)])
    ym = y.mean(axis=0)
    w, v = np.linalg.eigh(z.T @ z)
    vb = v.T @ (z.T @ (y - ym))
    coef = np.stack([v @ (vb[:, t] / (w + chosen[t])) for t in range(y.shape[1])], axis=1)
    return {"mu": mu, "sd": sd, "coef": coef, "intercept": ym, "alpha": chosen}


def ridge_predict(model: dict, x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float64) - model["mu"]) / model["sd"]) @ model["coef"] + model["intercept"]


def group_stats(y: np.ndarray, p: np.ndarray, unit: np.ndarray, n_units: int) -> np.ndarray:
    """Per unit and target the sufficient statistics (n, sum y, sum p, sum y^2, sum p^2, sum y p, sum (y-p)^2):
    (n_units, T, 7); additive over units, so a bootstrap draw is a weighted sum."""
    out = np.zeros((n_units, y.shape[1], 7))
    for t in range(y.shape[1]):
        for k, v in enumerate((np.ones(len(y)), y[:, t], p[:, t], y[:, t] ** 2, p[:, t] ** 2, y[:, t] * p[:, t],
                               (y[:, t] - p[:, t]) ** 2)):
            out[:, t, k] = np.bincount(unit, weights=v, minlength=n_units)
    return out


def regression_scores(stats: np.ndarray) -> dict[str, np.ndarray]:
    """R^2, Pearson r and MSE from summed `group_stats` (..., T, 7); R^2 about the group's own mean; NaN where
    undefined (no cells, a constant target or prediction)."""
    n, sy, sp, syy, spp, syp, sse = (stats[..., k] for k in range(7))
    with np.errstate(invalid="ignore", divide="ignore"):
        sst = syy - sy ** 2 / n
        cov, vy, vp = syp - sy * sp / n, sst, spp - sp ** 2 / n
        r2 = np.where(sst > 1e-12, 1 - sse / sst, np.nan)
        r = np.where((vy > 1e-12) & (vp > 1e-12), cov / np.sqrt(vy * vp), np.nan)
        mse = np.where(n > 0, sse / n, np.nan)
    return {"r2": r2, "pearson_r": r, "mse": mse}


def morphology(inp: dict, data: dict, workers: int) -> dict:
    """Readout 3."""
    units, layout = inp["units"], inp["layout"]
    (train, test), = layout.splits["main"][0]
    place = np.full(len(units), -1, dtype=np.int64)
    place[test] = np.arange(len(test))
    found = cell_targets(inp, data, workers)
    rows, cell_unit = found["rows"], data["cell_unit"]
    pool = units.pool.to_numpy()[cell_unit[rows]]
    finite = np.isfinite(found["values"][rows]).all(axis=1)
    usable = found["matched"][rows] & finite
    kind = units.cell_type.to_numpy()[cell_unit[rows]]
    counts = {}
    for name, mask in (("heldout_fit", pool == S.FIT), ("heldout_test", pool == S.TEST)):
        counts[name] = {"n_delivered": int(mask.sum()), "n_matched": int((mask & found["matched"][rows]).sum()),
                        "n_unmatched_left_out": int((mask & ~found["matched"][rows]).sum()),
                        "n_matched_with_undefined_target_left_out": int((mask & found["matched"][rows] & ~finite).sum()),
                        "n_used": int((mask & usable).sum())}
    well_pool = dict(zip(units.well, units.pool))
    image_pool = {r["image_id"]: well_pool[r["well"]] for r in inp["images"] if r["usage"] == S.HELDOUT}
    for name, side in (("heldout_fit", S.FIT), ("heldout_test", S.TEST)):
        ids = [i for i, p in image_pool.items() if p == side]
        gt_total = sum(found["n_gt"][i] for i in ids)
        counts[name].update(n_gt_instances=int(gt_total),
                            gt_instances_matched=int(sum(found["n_gt_matched"][i] for i in ids)))
    unit_of = cell_unit[rows]
    fit = usable & (pool == S.FIT)
    use_test = usable & (pool == S.TEST)
    embedding = data["embedding"]
    if not fit.any() or not use_test.any():
        raise ValueError("no matched cells on the heldout_fit or the heldout_test side")
    model = ridge_fit(np.asarray(embedding[rows[fit]]), found["values"][rows][fit], unit_of[fit])
    pred = ridge_predict(model, np.asarray(embedding[rows[use_test]]))
    y_test, t_unit = found["values"][rows][use_test], unit_of[use_test]
    stats = group_stats(y_test, pred, place[t_unit], len(test))                  # (n_test_units, T, 7)
    weights = np.vstack([np.ones((1, len(test))), layout.draws["main"][0]])
    test_kind = units.cell_type.to_numpy()[test]

    def report(select: np.ndarray) -> dict:
        s = np.einsum("ru,utk->rtk", weights[:, select], stats[select])
        sc = regression_scores(s)
        return {"n_cells": int(s[0, 0, 0]),
                **{t: {m: vc(sc[m][0, i], sc[m][1:, i]) for m in ("r2", "pearson_r", "mse")}
                   for i, t in enumerate(TARGETS)}}

    out = {"definition": DEFINITIONS["morphology"], "targets": list(TARGETS), "counts": counts,
           "alpha_per_target": {t: float(a) for t, a in zip(TARGETS, model["alpha"])},
           "n_fit_cells": int(fit.sum()), "overall": report(np.ones(len(test), dtype=bool)), "per_type": {}}
    for t in sorted(set(test_kind)):
        out["per_type"][t] = report(test_kind == t)
    per_type_counts = {}
    for t in sorted(set(kind)):
        for name, mask in (("heldout_fit", pool == S.FIT), ("heldout_test", pool == S.TEST)):
            sel = mask & (kind == t)
            per_type_counts.setdefault(t, {})[name] = {"n_delivered": int(sel.sum()),
                                                       "n_unmatched_left_out": int((sel & ~found["matched"][rows]).sum())}
    out["counts_per_type"] = per_type_counts
    return out


# ---- readout 4: stage leakage -----------------------------------------------------------------------------------

def stage_leakage(meta: pd.DataFrame, variants: dict[str, np.ndarray]) -> dict:
    """Readout 4. `variants`: name -> unit profiles (rows = meta rows). Per type, leave one well index out over all
    units; the balanced accuracy of the pooled out-of-fold predictions, macro over types, the paired bootstrap over
    the units of each type (the same draws for every variant)."""
    kind, stage, well = meta.cell_type.to_numpy(), meta.stage.to_numpy(), meta.well_index.to_numpy()
    types = sorted(set(kind))
    rng = np.random.default_rng(fm.BOOTSTRAP_SEED)
    draws = {}
    for t in types:
        n_t = int((kind == t).sum())
        picks = rng.integers(0, n_t, size=(fm.BOOTSTRAP, n_t))
        w = np.zeros((fm.BOOTSTRAP, n_t))
        np.add.at(w, (np.arange(fm.BOOTSTRAP)[:, None], picks), 1)
        draws[t] = np.vstack([np.ones((1, n_t)), w])
    out = {"definition": DEFINITIONS["stage_leakage"], "chance": 1 / N_STAGES, "n_units": int(len(meta)),
           "variants": {}, "per_type_units": {t: int((kind == t).sum()) for t in types}}
    macro = {}
    for name, profiles in variants.items():
        x = fm.l2(profiles)
        res = {}
        for classifier in fm.CLASSIFIERS:
            per_type, rows = {}, []
            for t in types:
                idx = np.flatnonzero(kind == t)
                predicted = np.full(len(idx), -1, dtype=np.int64)
                for g in sorted(set(well[idx])):
                    te, tr = idx[well[idx] == g], idx[well[idx] != g]
                    at = np.flatnonzero(well[idx] == g)
                    if len(np.unique(stage[tr])) < 2:
                        predicted[at] = stage[tr][0]
                    elif classifier == "knn":
                        predicted[at] = fm.knn_predict(x[tr], stage[tr], x[te], N_STAGES)
                    else:
                        predicted[at] = fm.logreg_predict(x[tr], stage[tr], x[te])[0]
                conf = np.zeros((len(idx), N_STAGES, N_STAGES))
                conf[np.arange(len(idx)), stage[idx], predicted] = 1
                ba = fm.scores(np.einsum("ru,uij->rij", draws[t], conf))["balanced_accuracy"]
                per_type[t] = ba
                rows.append(ba)
            mean = np.mean(rows, axis=0)
            macro[(name, classifier)] = mean
            res[classifier] = {"macro_balanced_accuracy": vc(mean[0], mean[1:]),
                               "per_type": {t: vc(per_type[t][0], per_type[t][1:]) for t in types}}
        out["variants"][name] = res
    if {"uncorrected", "delivered"} <= set(variants):
        out["delta_delivered_minus_uncorrected"] = {
            c: vc(macro[("delivered", c)][0] - macro[("uncorrected", c)][0],
                  macro[("delivered", c)][1:] - macro[("uncorrected", c)][1:]) for c in fm.CLASSIFIERS}
    return out


# ---- readout 5: batch integration -----------------------------------------------------------------------------------

BATCH_INTEGRATION = (("batch_ASW", "silhouette_batch"), ("iLISI", "lisi_batch"), ("graph_conn", "graph_conn"),
                     ("cLISI", "lisi_label"), ("label_ASW", "asw"), ("NMI", "nmi_ari"), ("ARI", "nmi_ari"))
DEFINITIONS["batch_integration"] = (
    "quant.py's metrics on the unit profiles (profiles.npy uncorrected, corrected.npy delivered), batch = the "
    "culture-time stage (6 groups), label = cell type (8): batch_ASW = silhouette_batch, iLISI = lisi_batch, "
    "graph_conn, cLISI = lisi_label, label_ASW = asw, NMI / ARI = nmi_ari (Leiden, resolution 1), k = 25 where it "
    "applies; all scaled so that higher is better; point estimates on all units and on the heldout units; a metric "
    "whose precondition fails (quant.MetricUnavailable) or whose package is missing is null with its reason")


def _quant():
    from shape.tools.batch_corrector import quant
    return quant


def _call(quant, fn, *args, **kwargs):
    """quant._safe, and a missing package (scib_metrics) is reported as null with its reason too."""
    try:
        return quant._safe(fn, *args, **kwargs)
    except ImportError as error:
        return None, f"{fn.__name__}: ImportError: {error}"


def integration_metrics(quant, feats: np.ndarray, labels: pd.Series, batch: pd.Series) -> tuple[dict, dict]:
    """({name: value or None}, {name: reason}) of one representation, calling quant as composite_all does."""
    feats = np.asarray(feats, dtype=np.float64)
    values, reasons = {}, {}
    plan = {"batch_ASW": (quant.silhouette_batch, (feats, labels, batch), {}),
            "iLISI": (quant.lisi_batch, (feats, batch), {"n_neighbors": 25}),
            "graph_conn": (quant.graph_conn, (feats, labels), {"n_neighbors": 25}),
            "cLISI": (quant.lisi_label, (feats, labels), {"n_neighbors": 25}),
            "label_ASW": (quant.asw, (feats, labels), {})}
    for name, (fn, args, kwargs) in plan.items():
        value, reason = _call(quant, fn, *args, **kwargs)
        values[name], reasons[name] = (None if value is None else float(value)), reason
    value, reason = _call(quant, quant.nmi_ari, feats, labels, n_neighbors=25)
    for name, key in (("NMI", "nmi"), ("ARI", "ari")):
        values[name], reasons[name] = (None if value is None else float(value[key])), reason
    return values, {k: r for k, r in reasons.items() if r is not None}


def batch_integration(meta: pd.DataFrame, variants: dict[str, np.ndarray], record: dict | None, quant=None) -> dict:
    """Readout 5. `meta` rows align with the profiles of `variants`; `record` = batch_SEARCH.json."""
    quant = _quant() if quant is None else quant
    masks = {"all_units": np.ones(len(meta), dtype=bool), "heldout_units": (meta.usage == S.HELDOUT).to_numpy()}
    kind, stage = meta.cell_type.to_numpy(), meta.plate.to_numpy()
    out = {"definition": DEFINITIONS["batch_integration"], "metrics": [name for name, _ in BATCH_INTEGRATION],
           "row_sets": {}}
    for name, mask in masks.items():
        labels, batch = pd.Series(kind[mask]), pd.Series(stage[mask])
        entry = {"n_units": int(mask.sum()), "n_batch_groups": int(batch.nunique()), "n_labels": int(labels.nunique()),
                 "variants": {}}
        for variant, profiles in variants.items():
            values, reasons = integration_metrics(quant, np.asarray(profiles)[mask], labels, batch)
            entry["variants"][variant] = {"values": values, "unavailable": reasons}
        if {"uncorrected", "delivered"} <= set(variants):
            a, b = (entry["variants"][v]["values"] for v in ("delivered", "uncorrected"))
            entry["delta_delivered_minus_uncorrected"] = {
                m: None if a[m] is None or b[m] is None else a[m] - b[m] for m in a}
        out["row_sets"][name] = entry
    evidence = (record or {}).get("evidence") or {}
    gate = evidence.get("gate_all_rows")
    out["admission_evidence"] = {
        "decision": ((record or {}).get("admission") or {}).get("decision"),
        "gate_all_rows": None if not gate else {k: gate.get(k) for k in (
            "batch_predictability", "chance_accuracy", "effect_margin", "detected", "reason", "n_samples",
            "n_batches")},
        **({} if gate else {"unavailable": "batch_SEARCH.json holds no evidence.gate_all_rows"})}
    return out


# ---- assembling, markdown, CLI -------------------------------------------------------------------------------------

def extra_readouts(prep: Path, run: Path, workers: int | None = None) -> dict:
    """All four readouts of one verified arm run."""
    started = time.time()
    inp = load_run(prep, run)
    data = load_cells(inp)
    layout = inp["layout"]
    result = {"status": "complete", "arm": inp["delivery"]["arm"], "provenance": inp["delivery"]["provenance"],
              "manifest_sha256": inp["prepared"]["manifest_sha256"], "design": inp["design"],
              "layout": layout.summary(), "rule": S.MAIN_RULE,
              "settings": {"split": "main only (single direction)", "cell_cap": fm.CELL_CAP,
                           "knn": {"metric": "cosine", "k": fm.KNN_K}, "logreg": fm.LOGREG,
                           "bootstrap": {"resamples": fm.BOOTSTRAP, "seed": fm.BOOTSTRAP_SEED,
                                         "resampled": "heldout_test units (the verdict layout's draws); a drawn unit "
                                                      "brings all its sampled cells; fitted models are not refitted"},
                           "ridge": {"alphas": list(RIDGE_ALPHAS), "folds": RIDGE_FOLDS},
                           "definitions": DEFINITIONS, "unit_rule": S.UNIT_RULE},
              "n_cells_heldout_delivered": int((data["cell_unit"] >= 0).sum()),
              "n_cells_delivered": len(data["cells"])}
    workers = max(1, os.cpu_count() or 1) if workers is None else max(1, int(workers))
    result["classification"] = classification(inp, data)
    result["morphology"] = morphology(inp, data, workers)
    meta, run = inp["meta"], inp["run"]
    result["stage_leakage"] = stage_leakage(meta, {"uncorrected": np.load(run / "profiles.npy"),
                                                   "delivered": np.load(run / "corrected.npy")})
    result["batch_integration"] = batch_integration(
        meta, {"uncorrected": np.load(run / "profiles.npy"), "delivered": np.load(run / "corrected.npy")},
        read_json(run / "batch_SEARCH.json"))
    if any("ImportError" in r for e in result["batch_integration"]["row_sets"].values()
           for v in e["variants"].values() for r in v["unavailable"].values()):
        print("extra_readouts: WARNING: scib_metrics is not importable; the batch-integration metrics are null",
              file=sys.stderr)
    print(f"extra_readouts: {time.time() - started:.0f} s", file=sys.stderr)
    return clean(result)


def _c(entry: dict | None, spec: str = ".3f") -> str:
    if not entry or entry.get("value") is None:
        return "n/a"
    ci = entry.get("ci95")
    return format(entry["value"], spec) + ("" if not ci else f" [{format(ci[0], spec)}, {format(ci[1], spec)}]")


def markdown(result: dict) -> str:
    cls, morph, leak = result["classification"], result["morphology"], result["stage_leakage"]
    design = result["design"]
    lines = [f"# full_LIVECell extra readouts ({result['arm']})", "",
             "Reported, not verdicts. Main split only: fit on "
             f"{design['n_heldout_fit_units']} heldout_fit units (stages 2-3), test on "
             f"{design['n_heldout_test_units']} heldout_test units (stages 4-5). Value [95% interval over "
             f"{result['settings']['bootstrap']['resamples']} resamples of the test units]. "
             f"{cls['n_units_without_cells_test']} test units without delivered cells; "
             f"{cls['n_excluded_sampled_test']} excluded cells sampled in the test cells (misses).", "",
             "## 1-2. Classification probes", "",
             "| probe | macro F1 | balanced accuracy | macro AUROC | macro AUPRC | chance AUROC / AUPRC |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, p in cls["probes"].items():
        lines.append(f"| {name} | {_c(p['macro_f1'])} | {_c(p['balanced_accuracy'])} | {_c(p['macro_auroc'])} | "
                     f"{_c(p['macro_auprc'])} | 0.5 / {p['chance']['auprc_macro_prevalence']:.3f} |")
    for name, p in cls["probes"].items():
        lines += ["", f"### {name}" + (" (the verdict probe)" if name == "cell_knn" else ""), "",
                  "| class | test n | precision | recall | F1 | AUROC | AUPRC | AUPRC chance |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c, e in p["per_class"].items():
            chance = "n/a" if e["auprc_chance"] is None else f"{e['auprc_chance']:.3f}"
            lines.append(f"| {c} | {e['support']} | {_c(e['precision'])} | {_c(e['recall'])} | {_c(e['f1'])} | "
                         f"{_c(e['auroc'])} | {_c(e['auprc'])} | {chance} |")
        classes = cls["classes"]
        conf, miss = p["confusion"]["rows_true_columns_predicted"], p["confusion"]["misses_per_true_class"]
        lines += ["", "Confusion (rows true, columns predicted; last column = misses):", "",
                  "| true \\ pred | " + " | ".join(classes) + " | miss |", "| --- |" + " --- |" * (len(classes) + 1)]
        for i, c in enumerate(classes):
            lines.append(f"| {c} | " + " | ".join(str(v) for v in conf[i]) + f" | {miss[i]} |")
        lines += ["", "Largest confusions: " + ("; ".join(
            f"{t['true']} -> {t['predicted']} {t['count']} ({_c(t['rate_of_true_class'])} of {t['true']})"
            for t in p["top_confusions"]) or "none") + "."]
    lines += ["", "## 3. Cell morphology (ridge from the uncorrected cell embedding)", "",
              "Ridge alpha per target " + ", ".join(f"{t} {a:g}" for t, a in morph["alpha_per_target"].items())
              + f"; fitted on {morph['n_fit_cells']} heldout_fit cells. Cells with no matched GT instance (IoU < 0.5) "
              "are left out:", "",
              "| pool | delivered cells | matched | left out (unmatched) | left out (undefined target) | used | "
              "GT instances | GT matched |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for pool, c in morph["counts"].items():
        lines.append(f"| {pool} | {c['n_delivered']} | {c['n_matched']} | {c['n_unmatched_left_out']} | "
                     f"{c['n_matched_with_undefined_target_left_out']} | {c['n_used']} | {c['n_gt_instances']} | "
                     f"{c['gt_instances_matched']} |")
    lines += ["", "| group | test cells | target | R^2 | Pearson r | MSE |", "| --- | --- | --- | --- | --- | --- |"]
    for group, e in [("overall", morph["overall"]), *morph["per_type"].items()]:
        for t in morph["targets"]:
            lines.append(f"| {group} | {e['n_cells']} | {t} | {_c(e[t]['r2'])} | {_c(e[t]['pearson_r'])} | "
                         f"{_c(e[t]['mse'], '.4f')} |")
    lines += ["", "R^2 is about each group's own mean (overall: includes the between-type variance).", "",
              "## 4. Stage leakage (lower is better; chance " + f"{leak['chance']:.3f})", "",
              "| profiles | classifier | macro balanced accuracy |", "| --- | --- | --- |"]
    for name, res in leak["variants"].items():
        for classifier, e in res.items():
            lines.append(f"| {name} | {classifier} | {_c(e['macro_balanced_accuracy'])} |")
    if "delta_delivered_minus_uncorrected" in leak:
        lines += ["", "Delivered minus uncorrected (negative = less stage information after correction): "
                  + "; ".join(f"{c} {_c(e, '+.3f')}" for c, e in leak["delta_delivered_minus_uncorrected"].items())
                  + "."]
    lines += ["", "| type | units | " + " | ".join(f"{v} {c}" for v in leak["variants"] for c in fm.CLASSIFIERS) + " |",
              "| --- | --- |" + " --- |" * (len(leak["variants"]) * len(fm.CLASSIFIERS))]
    for t, n in leak["per_type_units"].items():
        lines.append(f"| {t} | {n} | " + " | ".join(
            _c(leak["variants"][v][c]["per_type"][t]) for v in leak["variants"] for c in fm.CLASSIFIERS) + " |")
    lines += ["", "## 5. Batch integration (scib-style; higher is better; batch = stage, label = cell type)", ""]
    names = result["batch_integration"]["metrics"]
    lines += ["| rows | profiles | " + " | ".join(names) + " |", "| --- | --- |" + " --- |" * len(names)]
    notes = []
    for rows, e in result["batch_integration"]["row_sets"].items():
        for variant, v in e["variants"].items():
            lines.append(f"| {rows} ({e['n_units']}) | {variant} | " + " | ".join(
                "n/a" if v["values"][m] is None else f"{v['values'][m]:.3f}" for m in names) + " |")
            notes += [f"{rows} / {variant}: {m} unavailable ({r})" for m, r in v["unavailable"].items()]
        if "delta_delivered_minus_uncorrected" in e:
            lines.append(f"| {rows} | delivered - uncorrected | " + " | ".join(
                "n/a" if e["delta_delivered_minus_uncorrected"][m] is None
                else f"{e['delta_delivered_minus_uncorrected'][m]:+.3f}" for m in names) + " |")
    gate = result["batch_integration"]["admission_evidence"]
    lines += ["", "Admission evidence (batch_SEARCH.json): decision " + str(gate["decision"]) + "; "
              + (", ".join(f"{k} {v}" for k, v in gate["gate_all_rows"].items()) if gate["gate_all_rows"]
                 else gate["unavailable"]) + "."]
    lines += [f"- {n}" for n in notes]
    lines += ["", "Definitions:"] + [f"- {k}: {v}" for k, v in result["settings"]["definitions"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prep", required=True, type=Path, help="the prepared LIVECell tree (private labels and GT)")
    ap.add_argument("--run", required=True, type=Path, help="one delivered arm run directory (never written)")
    ap.add_argument("--out", required=True, type=Path, help="output JSON")
    ap.add_argument("--md", action="store_true", help="also write a markdown with the same stem")
    ap.add_argument("--workers", type=int, default=None, help="processes for the GT matching (default: all cores)")
    args = ap.parse_args(argv)
    try:
        result = extra_readouts(args.prep, args.run, args.workers)
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if args.md:
        args.out.with_suffix(".md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
