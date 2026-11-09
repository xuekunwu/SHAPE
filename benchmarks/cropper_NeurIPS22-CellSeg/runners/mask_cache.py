"""The cached Cellpose-SAM predicted mask of one image, shared by the drivers.

Re-running a cropper or VLM stage over images whose masks are already cached spends no
GPU: `run_concurrent_benchmark.py` pre-segments through this helper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))  # repo root, for `shape.*`
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segmentation_NeurIPS22-CellSeg" / "runners"))

import run_cellpose_sam_only as cpsam  # noqa: E402 -- the Cellpose-SAM call, reused


def _load_or_segment_mask(image_path: Path, masks_dir: Path, model_holder: dict,
                          batch_size: int, flow_threshold: float,
                          cellprob_threshold: float) -> tuple[np.ndarray, bool]:
    """The predicted instance-label mask for ONE image. Loads the cached
    `<masks-dir>/<stem>_pred.tiff` (uint16 -- the SAME format `tool.py` reads a GT
    `_label.tiff`) if it already exists (Cellpose is SKIPPED for that image), else runs
    Cellpose-SAM once and caches the result. `model_holder` is a 1-entry mutable dict
    (`{"model": ...}`) that LAZILY loads the Cellpose-SAM model at most once across the
    whole run, only if at least one image is a cache miss. Returns `(mask, cellpose_ran)`."""
    pred_path = masks_dir / f"{image_path.stem}_pred.tiff"
    if pred_path.exists():
        return cpsam.load_label_mask(pred_path), False
    if model_holder.get("model") is None:
        model_holder["model"] = cpsam.load_cellpose_sam()
    image = cpsam.load_image_for_cellpose(image_path)
    mask = cpsam.segment(model_holder["model"], image, batch_size, flow_threshold,
                         cellprob_threshold)
    masks_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(pred_path), np.asarray(mask).astype(np.uint16))
    return mask, True
