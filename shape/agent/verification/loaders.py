"""Impedance-matching adapters: SHAPE tool outputs -> VLRR verifier inputs.

Each managed tool returns a result dict with file-path artifacts (masks, crop
metadata JSON, h5ad). The pure verifiers in ``shape.agent.verification.verifiers``
expect raw arrays / precomputed metric dicts. These adapters bridge the gap:
they load the artifacts a tool produced and reshape them into exactly the
``(tool_output, context)`` pair a verifier's ``verify()`` expects.

Heavy/optional dependencies (anndata, scanpy, scikit-learn, scipy, skimage) are
imported lazily inside the functions that need them, so importing this module --
and the verification package as a whole -- stays cheap and never drags in the
science stack until an adapter actually runs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

from shape.utils.logger import logger


# ---- helpers ----------------------------------------------------------------

def _load_gray(path: str) -> Optional[np.ndarray]:
    """Load an image as a 2D float array via the project's ImageProcessor."""
    try:
        from shape.utils.image_processor import ImageProcessor
        data = ImageProcessor.load_image(path)
        return np.asarray(data.to_uint8(0), dtype=np.float64)
    except Exception as e:
        logger.warning(f"adapters._load_gray failed for {path}: {e}")
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return None if img is None else np.asarray(img, dtype=np.float64)
        except Exception:
            return None


def _codes(values) -> np.ndarray:
    """Turn a sequence of labels into integer codes (stable by first appearance)."""
    seen: dict = {}
    out = []
    for v in values:
        key = str(v)
        if key not in seen:
            seen[key] = len(seen)
        out.append(seen[key])
    return np.asarray(out, dtype=int)


# ---- image loading (v1) -----------------------------------------------------

def adapt_image_loading(image_paths: list[str], context: Optional[dict] = None):
    """([paths]) -> (list[np.ndarray], context). Loads each image to a 2D array."""
    imgs = [im for im in (_load_gray(p) for p in image_paths) if im is not None]
    return imgs, dict(context or {})


# ---- segmentation (v3) ------------------------------------------------------

def _uncovered_foreground(image: np.ndarray, masks: np.ndarray) -> Optional[dict]:
    """Estimate uncovered textured foreground (a missed-cell cue) and its size.

    Conservative, illumination-invariant under-detection cue for phase contrast:
    cells carry high local gradient energy, background is flat. We threshold the
    gradient map (above the masked regions' own texture floor), then keep only
    UNCOVERED blobs at least half the median cell area -- scattered edge pixels and
    halos do not count. Returns a dict with:
      - ``missed_foreground_fraction``: missed_area / (masked_area + missed_area);
      - ``missed_blob_size_ratio``: median uncovered-blob area / median cell area,
        which tells the diagnosis WHICH knob to turn (large vs small missed cells).
    Returns None if it cannot be computed. A backstop: the vision judge reading the
    overlay is the primary missed-cell detector.
    """
    try:
        from scipy import ndimage
    except Exception:
        return None
    img = np.asarray(image, dtype=np.float64)
    masks = np.asarray(masks)
    if img.ndim != 2 or img.shape != masks.shape or img.size == 0:
        return None
    mask_union = masks > 0
    if not mask_union.any() or mask_union.all():
        return None
    gy, gx = np.gradient(img)
    grad = np.hypot(gx, gy)
    fg_texture = grad[mask_union]      # texture inside real cells
    if fg_texture.size == 0:
        return None
    # foreground = clearly more textured than the cell-interior median floor
    thr = float(np.median(fg_texture)) * 0.5
    uncovered = (grad > thr) & ~mask_union
    # vectorized per-label areas (one pass) instead of a full-image scan per mask
    _labels, _counts = np.unique(masks, return_counts=True)
    _areas = _counts[_labels != 0]
    detected = int(_areas.size)
    med_area = float(np.median(_areas)) if detected else 50.0
    zero = {"missed_foreground_fraction": 0.0, "missed_blob_size_ratio": 0.0,
            "count_miss_fraction": 0.0}
    if not uncovered.any():
        return zero
    labeled, n = ndimage.label(uncovered)
    if n == 0:
        return zero
    sizes = ndimage.sum(np.ones_like(labeled, dtype=np.float64), labeled,
                        index=range(1, n + 1))
    kept = sizes[sizes >= 0.5 * med_area]
    if kept.size == 0:
        return zero
    missed_area = float(kept.sum())
    total = float(mask_union.sum()) + missed_area
    # COUNT-based density cue: how many cell-sized regions are uncovered relative to
    # the detected count. In a crowded field where MANY cells are dropped individually,
    # their total AREA can stay below the area gate (each missed cell is small relative
    # to the whole), but their COUNT does not -- so this catches under-detection the
    # area fraction misses.
    count_miss_fraction = (kept.size / (detected + kept.size)
                           if (detected + kept.size) > 0 else 0.0)
    return {
        "missed_foreground_fraction": missed_area / total if total > 0 else 0.0,
        "missed_blob_size_ratio": float(np.median(kept)) / med_area if med_area > 0 else 0.0,
        "count_miss_fraction": float(count_miss_fraction),
    }


# adapt_segmentation was folded into verifiers/segmentation.py (Phase 2b).
# _uncovered_foreground (above) and _load_gray stay here as the shared helpers it
# imports; test_under_detection imports _uncovered_foreground from here too.


# ---- preprocessing (v2) -----------------------------------------------------
# The preprocessing stage and its verifier were removed from the agentic loop;
# `adapt_preprocessing` went with them. Only `absolute_quality_metrics` survives,
# in verifiers/preprocessing.py, read by the crop verifier.


# ---- crop (v4) --------------------------------------------------------------
# adapt_crop + _read_crop_metadata were folded into verifiers/crop.py (Phase 2b).


# ---- representation (v5) ----------------------------------------------------

def _load_adata(adata_path: str):
    import anndata as ad
    if not adata_path or not os.path.exists(adata_path):
        raise ValueError(f"adata not found: {adata_path}")
    return ad.read_h5ad(adata_path)


# adapt_representation was folded into verifiers/representation.py (Phase 2b);
# _load_adata (above) and _codes stay here as shared loaders it imports.


# ---- morphological state (v6) ----------------------------------------------
# adapt_morphological (+ _silhouette_safe / _enrichment_stats /
# _within_cluster_variance / _replicate_consistency) were folded into
# verifiers/morphological_state.py (Phase 2b). _load_adata / _codes / _load_gray /
# _uncovered_foreground remain here as the shared low-level loaders the folded
# verifiers import.
