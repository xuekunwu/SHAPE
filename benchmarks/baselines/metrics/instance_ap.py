"""Instance-segmentation metrics for the baselines harness -- OUR OWN copy of SHAPE's scoring,
so a baseline agent's delivered masks are graded identically to SHAPE without the agent ever
seeing this code. AP = TP/(TP+FP+FN) per IoU threshold, matched by optimal assignment.

A delivered mask and the GT label are both integer instance-label images (0 = background).
"""
from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]


def matched_ious(pred: np.ndarray, gt: np.ndarray):
    """Optimal instance assignment -> the IoU of every matched (pred, gt) pair, plus counts.
    Fully vectorized (bincount contingency table + Hungarian assignment)."""
    from scipy.optimize import linear_sum_assignment
    pred = np.asarray(pred).astype(np.int64).ravel()
    gt = np.asarray(gt).astype(np.int64).ravel()
    pl = np.unique(pred[pred > 0])
    gl = np.unique(gt[gt > 0])
    if pl.size == 0 or gl.size == 0:
        return np.array([]), int(pl.size), int(gl.size)
    pmap = np.zeros(pl.max() + 1, dtype=np.int64); pmap[pl] = np.arange(1, pl.size + 1)
    gmap = np.zeros(gl.max() + 1, dtype=np.int64); gmap[gl] = np.arange(1, gl.size + 1)
    pr, gr = pmap[pred], gmap[gt]
    parea = np.bincount(pr, minlength=pl.size + 1)[1:]
    garea = np.bincount(gr, minlength=gl.size + 1)[1:]
    m = (pr > 0) & (gr > 0)
    flat = (pr[m] - 1) * gl.size + (gr[m] - 1)
    inter = np.bincount(flat, minlength=pl.size * gl.size).reshape(pl.size, gl.size)
    iou = inter / (parea[:, None] + garea[None, :] - inter + 1e-9)
    r, c = linear_sum_assignment(-iou)
    return np.array([iou[i, j] for i, j in zip(r, c) if iou[i, j] > 0]), int(pl.size), int(gl.size)


def image_metrics(pred: np.ndarray, gt: np.ndarray, thresholds=IOU_THRESHOLDS) -> dict:
    """Per-image AP/precision/recall/F1 at each IoU threshold + AP50/AP75/mAP summaries."""
    ious, n_pred, n_gt = matched_ious(pred, gt)
    per_t = {}
    aps = []
    for t in thresholds:
        tp = int((ious >= t).sum())
        fp, fn = n_pred - tp, n_gt - tp
        ap = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_t[f"{t:.2f}"] = {"true_positive": tp, "false_positive": fp, "false_negative": fn,
                             "average_precision": ap, "precision": prec, "recall": rec, "f1": f1}
        aps.append(ap)
    return {
        "predicted_cell_count": n_pred, "ground_truth_cell_count": n_gt,
        "ap50": per_t["0.50"]["average_precision"],
        "ap75": per_t["0.75"]["average_precision"],
        "map_50_95": float(np.mean(aps)),
        "precision_at_50": per_t["0.50"]["precision"],
        "recall_at_50": per_t["0.50"]["recall"],
        "f1_at_50": per_t["0.50"]["f1"],
        "mean_matched_iou_at_50": float(ious[ious >= 0.50].mean()) if (ious >= 0.50).any() else 0.0,
        "thresholds": per_t,
    }


def macro_summary(per_image: dict) -> dict:
    """Aggregate per-image metrics into macro means (the headline numbers SHAPE reports)."""
    keys = ["ap50", "ap75", "map_50_95", "precision_at_50", "recall_at_50", "f1_at_50"]
    vals = {k: float(np.mean([m[k] for m in per_image.values()])) for k in keys if per_image}
    vals["n_images"] = len(per_image)
    return vals
