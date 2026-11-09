"""Instance-segmentation AP at arbitrary IoU thresholds.

Verbatim port of the matching used by the SHAPE benchmark harness
(benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_shape_neurips_cellseg.py at 30f799d9):
pairwise IoU over non-background labels, then at each threshold a maximum-cardinality matching
with maximum total IoU (Hungarian on validity * bonus + IoU). AP(t) = TP / (TP + FP + FN).
Standard library + numpy + scipy only.
"""
import hashlib
import os

import numpy as np

_STEP = float(os.environ.get("AP_STEP", "0.05"))  # 0.05 -> 20 thresholds, 0.01 -> 100 thresholds
FULL_THRESHOLDS = tuple(round(_STEP * k, 2) for k in range(1, int(round(1.0 / _STEP)) + 1))  # _STEP ... 1.00
STANDARD_THRESHOLDS = tuple(round(0.50 + 0.05 * k, 2) for k in range(10))  # 0.50 ... 0.95


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_label(path):
    """Instance label map from mask.tif / .npy / Cellpose *_seg.npy (its 'masks' entry)."""
    path = str(path)
    if path.lower().endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
        if arr.shape == () and arr.dtype == object and isinstance(arr.item(), dict):
            d = arr.item()
            arr = next(d[k] for k in ("masks", "mask", "labels") if k in d)
        arr = np.asarray(arr).squeeze()
    else:
        import tifffile
        arr = np.asarray(tifffile.imread(path)).squeeze()
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.integer) or np.any(arr < 0):
        raise ValueError("expected a non-negative 2-D integer label map: %s" % path)
    return arr


def iou_matrix(ground_truth, prediction):
    from scipy import sparse
    ground_truth = np.asarray(ground_truth).squeeze()
    prediction = np.asarray(prediction).squeeze()
    if ground_truth.shape != prediction.shape:
        raise ValueError("Ground-truth shape %s != prediction shape %s" % (ground_truth.shape, prediction.shape))
    gt_labels, gt_inverse = np.unique(ground_truth, return_inverse=True)
    pred_labels, pred_inverse = np.unique(prediction, return_inverse=True)
    overlap = sparse.coo_matrix(
        (np.ones(ground_truth.size, dtype=np.int64), (gt_inverse.ravel(), pred_inverse.ravel())),
        shape=(len(gt_labels), len(pred_labels))).toarray()
    gt_keep = gt_labels != 0
    pred_keep = pred_labels != 0
    intersections = overlap[np.ix_(gt_keep, pred_keep)].astype(np.float64)
    gt_areas = overlap[gt_keep, :].sum(axis=1, keepdims=True)
    pred_areas = overlap[:, pred_keep].sum(axis=0, keepdims=True)
    unions = gt_areas + pred_areas - intersections
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(unions > 0, intersections / unions, 0.0)
    return iou


def match(iou, threshold):
    from scipy.optimize import linear_sum_assignment
    if not iou.size:
        return 0, []
    valid = iou >= threshold
    cardinality_bonus = min(iou.shape) + 1.0
    rows, cols = linear_sum_assignment(-(valid.astype(np.float64) * cardinality_bonus + iou))
    accepted = valid[rows, cols]
    return int(accepted.sum()), iou[rows[accepted], cols[accepted]].tolist()


def evaluate(ground_truth, prediction, thresholds=FULL_THRESHOLDS):
    """Per-threshold TP/FP/FN/AP (+ precision, recall, f1, mean matched IoU) for one image."""
    iou = iou_matrix(ground_truth, prediction)
    n_gt, n_pred = iou.shape
    out = {}
    for t in thresholds:
        tp, matched = match(iou, t)
        fp, fn = n_pred - tp, n_gt - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        out["%.2f" % t] = {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "average_precision": tp / max(1, tp + fp + fn),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "mean_matched_iou": float(np.mean(matched)) if matched else 0.0,
        }
    return {"ground_truth_cell_count": int(n_gt), "predicted_cell_count": int(n_pred), "thresholds": out}
