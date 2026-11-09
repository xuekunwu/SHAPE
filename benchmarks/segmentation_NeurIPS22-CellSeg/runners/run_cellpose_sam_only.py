#!/usr/bin/env python3
"""Pure Cellpose-SAM baseline on the full NeurIPS22-CellSeg test set.

NO SHAPE / VLRR / agentic layer: load each image, run Cellpose-SAM ONCE with default
settings, score the masks against the ground-truth instance labels, and write per-image
AP50 / AP75 / mAP (+ a few extras) to a CSV. This is the clean baseline the agentic
pipeline (run_shape_neurips_cellseg.py) is compared against, run at full scale (~1000
images) rather than on a sample.

Data layout (defaults):
    <data-root>/images/<name>.{png,bmp,tif,tiff,jpg}
    <data-root>/labels/<name>_label.tiff

Output:
    <data-root>/cellpose_sam_metrics.csv   (one row per image; written incrementally)

Example:
    python benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_cellpose_sam_only.py \
        --data-root /scratch/users/your-user/shape_benchmark/NeurIPS22-CellSeg
"""

from __future__ import annotations

import os
import argparse
import csv
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional; fall back to a plain iterator
    tqdm = None

IOU_THRESHOLDS = tuple(round(v, 2) for v in np.arange(0.50, 1.00, 0.05))
_IMAGE_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
_CSV_FIELDS = [
    "image_name", "ground_truth_cell_count", "predicted_cell_count",
    "ap50", "ap75", "map_50_95",
    "precision_at_50", "recall_at_50", "f1_at_50",
    "mean_matched_iou_at_50", "cell_count_error_fraction",
    "runtime_seconds", "status",
]


# --------------------------------------------------------------------------- #
# Image / label loading
# --------------------------------------------------------------------------- #
def _read_image_array(path: str) -> np.ndarray:
    """Read a TIFF/PNG/BMP image as a raw array (no rescaling)."""
    try:
        from skimage.io import imread
        return np.asarray(imread(path))
    except Exception:
        from PIL import Image
        with Image.open(path) as image:
            return np.asarray(image)


def load_label_mask(path) -> np.ndarray:
    """Load a ground-truth instance-label map (.tif/.tiff/.png) as a 2-D int array."""
    arr = np.asarray(_read_image_array(str(path))).squeeze()
    if arr.ndim == 3:                       # label image stored as RGB -> first channel
        arr = arr[..., 0]
    return arr


def load_image_for_cellpose(path) -> np.ndarray:
    """Load an image as a single-channel float array for Cellpose-SAM.

    Cellpose-SAM normalizes intensity internally, so we just hand it the signal.
    Multi-channel images (fluorescence stored as RGB, etc.) collapse to the
    highest-variance channel -- this keeps the signal when it lives in only one
    channel (a plain mean would dilute it), and is a no-op for true grayscale.
    """
    arr = np.asarray(_read_image_array(str(path))).squeeze()
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[2] >= 4:               # drop alpha / extra channels
            arr = arr[..., :3]
        stds = [float(arr[..., c].std()) for c in range(arr.shape[2])]
        return arr[..., int(np.argmax(stds))].astype(np.float32)
    raise ValueError(f"Unsupported image shape {arr.shape} for {path}")


# --------------------------------------------------------------------------- #
# Instance-segmentation metrics (same definitions as the SHAPE benchmark)
# --------------------------------------------------------------------------- #
def _instance_iou_matrix(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    from scipy import sparse

    ground_truth = np.asarray(ground_truth).squeeze()
    prediction = np.asarray(prediction).squeeze()
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Ground-truth shape {ground_truth.shape} != prediction shape {prediction.shape}")
    gt_labels, gt_inverse = np.unique(ground_truth, return_inverse=True)
    pred_labels, pred_inverse = np.unique(prediction, return_inverse=True)
    overlap = sparse.coo_matrix(
        (np.ones(ground_truth.size, dtype=np.int64),
         (gt_inverse.ravel(), pred_inverse.ravel())),
        shape=(len(gt_labels), len(pred_labels)),
    ).toarray()
    gt_keep = gt_labels != 0
    pred_keep = pred_labels != 0
    intersections = overlap[np.ix_(gt_keep, pred_keep)].astype(np.float64)
    gt_areas = overlap[gt_keep, :].sum(axis=1, keepdims=True)
    pred_areas = overlap[:, pred_keep].sum(axis=0, keepdims=True)
    unions = gt_areas + pred_areas - intersections
    return np.divide(intersections, unions,
                     out=np.zeros_like(intersections), where=unions > 0)


def _match_instances(iou: np.ndarray, threshold: float) -> tuple[int, list[float]]:
    from scipy.optimize import linear_sum_assignment

    if not iou.size:
        return 0, []
    valid = iou >= threshold
    cardinality_bonus = min(iou.shape) + 1.0
    rows, cols = linear_sum_assignment(-(valid.astype(np.float64) * cardinality_bonus + iou))
    accepted = valid[rows, cols]
    return int(accepted.sum()), iou[rows[accepted], cols[accepted]].tolist()


def evaluate_instance_segmentation(prediction: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Per-image instance metrics across IoU 0.50-0.95 (AP = TP/(TP+FP+FN))."""
    iou = _instance_iou_matrix(ground_truth, prediction)
    n_gt, n_pred = iou.shape
    by_threshold = {}
    for threshold in IOU_THRESHOLDS:
        tp, matched = _match_instances(iou, threshold)
        fp, fn = n_pred - tp, n_gt - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        by_threshold[f"{threshold:.2f}"] = {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "average_precision": tp / max(1, tp + fp + fn),
            "precision": precision, "recall": recall, "f1": f1,
            "matched_ious": matched,
        }
    at_50 = by_threshold["0.50"]
    matched = at_50["matched_ious"]
    return {
        "ground_truth_cell_count": n_gt,
        "predicted_cell_count": n_pred,
        "ap50": at_50["average_precision"],
        "ap75": by_threshold["0.75"]["average_precision"],
        "map_50_95": float(np.mean([v["average_precision"] for v in by_threshold.values()])),
        "precision_at_50": at_50["precision"],
        "recall_at_50": at_50["recall"],
        "f1_at_50": at_50["f1"],
        "mean_matched_iou_at_50": float(np.mean(matched)) if matched else 0.0,
        "cell_count_error_fraction": abs(n_pred - n_gt) / max(1, n_gt),
        "_thresholds": by_threshold,
    }


# --------------------------------------------------------------------------- #
# Cellpose-SAM
# --------------------------------------------------------------------------- #
def load_cellpose_sam():
    """Load the Cellpose-SAM model (GPU if available, CPU fallback)."""
    from cellpose import models
    try:
        import torch
        use_gpu = bool(torch.cuda.is_available())
    except Exception:
        use_gpu = False
    try:
        model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
        print(f"[cellpose-sam] model loaded (gpu={use_gpu})")
        return model
    except Exception as exc:
        print(f"[cellpose-sam] GPU load failed ({exc}); retrying on CPU")
        return models.CellposeModel(gpu=False, pretrained_model="cpsam")


def segment(model, image: np.ndarray, batch_size: int, flow_threshold: float,
            cellprob_threshold: float) -> np.ndarray:
    """One default Cellpose-SAM pass; returns the integer instance-label mask."""
    masks, _flows, _styles = model.eval(
        [image], channels=[0, 0], batch_size=batch_size,
        flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
    return np.asarray(masks[0])


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _aggregate(rows: list[dict]) -> None:
    ev = [r for r in rows if r.get("status") == "ok" and r.get("ap50") is not None]
    if not ev:
        print("No evaluable images.")
        return
    def macro(key):
        return float(np.mean([float(r[key]) for r in ev]))
    print(f"\n=== Cellpose-SAM over {len(ev)} evaluable images ===")
    print(f"  MACRO (per-image mean):  AP50={macro('ap50'):.4f}  "
          f"AP75={macro('ap75'):.4f}  mAP={macro('map_50_95'):.4f}")
    print(f"  recall@50={macro('recall_at_50'):.4f}  precision@50={macro('precision_at_50'):.4f}")


def run(image_dir: Path, label_dir: Path, out_csv: Path, *,
        label_suffix: str, label_ext: str, limit: int | None,
        batch_size: int, flow_threshold: float, cellprob_threshold: float) -> None:
    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
                    and not p.stem.endswith(label_suffix))
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError(f"No images ({sorted(_IMAGE_EXTS)}) found in {image_dir}")
    print(f"[run] {len(images)} images from {image_dir}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = load_cellpose_sam()
    rows: list[dict] = []
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        iterator = tqdm(images, desc="Cellpose-SAM", unit="img") if tqdm else images
        for image_path in iterator:
            name = image_path.stem
            gt_path = label_dir / f"{name}{label_suffix}{label_ext}"
            row = {f: "" for f in _CSV_FIELDS}
            row["image_name"] = name
            t0 = time.perf_counter()
            try:
                if not gt_path.is_file():
                    row["status"] = "no_ground_truth"
                else:
                    image = load_image_for_cellpose(image_path)
                    mask = segment(model, image, batch_size, flow_threshold, cellprob_threshold)
                    gt = load_label_mask(gt_path)
                    metrics = evaluate_instance_segmentation(mask, gt)
                    metrics.pop("_thresholds", None)
                    row.update({k: metrics[k] for k in metrics if k in _CSV_FIELDS})
                    row["status"] = "ok"
            except Exception as exc:
                row["status"] = f"error: {exc}"
            row["runtime_seconds"] = round(time.perf_counter() - t0, 3)
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if tqdm is not None:
                iterator.set_postfix(
                    ap50=(round(float(row["ap50"]), 3) if row["ap50"] != "" else None),
                    status=str(row["status"])[:14])
            elif not str(row["status"]).startswith("ok"):
                print(f"  {name}: {row['status']}")

    print(f"\nPer-image metrics saved to {out_csv}")
    _aggregate(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure Cellpose-SAM baseline on NeurIPS22-CellSeg -> per-image CSV.")
    parser.add_argument(
        "--data-root", type=Path,
        default=(Path(os.environ["SHAPE_BENCH_DATA_ROOT"]) / "NeurIPS22-CellSeg"
                 if os.environ.get("SHAPE_BENCH_DATA_ROOT") else None),
        required=not os.environ.get("SHAPE_BENCH_DATA_ROOT"),
        help="Dataset root containing images/ and labels/ subfolders."
             "Defaults to $SHAPE_BENCH_DATA_ROOT/NeurIPS22-CellSeg.")
    parser.add_argument("--image-dir", type=Path, help="Override images dir.")
    parser.add_argument("--label-dir", type=Path, help="Override labels dir.")
    parser.add_argument("--label-suffix", default="_label")
    parser.add_argument("--label-ext", default=".tiff")
    parser.add_argument("--out-csv", type=Path,
                        help="Output CSV (default <data-root>/cellpose_sam_metrics.csv).")
    parser.add_argument("--limit", type=int, help="Process only the first N images (debug).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    args = parser.parse_args()

    image_dir = args.image_dir or (args.data_root / "images")
    label_dir = args.label_dir or (args.data_root / "labels")
    out_csv = args.out_csv or (args.data_root / "cellpose_sam_metrics.csv")
    run(image_dir, label_dir, out_csv,
        label_suffix=args.label_suffix, label_ext=args.label_ext, limit=args.limit,
        batch_size=args.batch_size, flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold)


if __name__ == "__main__":
    main()
