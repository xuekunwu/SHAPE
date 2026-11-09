"""Shared segmentation QC helpers used by the segmenter tools.

Kept separate from the tools so the three segmenters (cell / nuclei / organoid)
share one implementation instead of triplicating it.
"""

from __future__ import annotations

import inspect
from typing import Optional

import numpy as np

from shape.utils.logger import logger


def _get_flow_error_fn():
    """Locate Cellpose's flow_error across versions.

    Cellpose v4 (CellposeSAM) exposes it as ``cellpose.dynamics.flow_error``;
    v3 and earlier kept it in ``cellpose.metrics``. Returns None if neither is
    importable.
    """
    try:
        from cellpose.dynamics import flow_error as fn  # cellpose >= 4
        return fn
    except Exception:
        pass
    try:
        from cellpose.metrics import flow_error as fn  # cellpose <= 3
        return fn
    except Exception:
        return None


def _extract_dP(flows):
    """Pull the dP flow field out of Cellpose's `flows` return value.

    `model.eval([img], ...)` returns `flows` as a list over images, where each
    per-image entry is itself `[flow_RGB, dP, cellprob, ...]`. Older/edge cases
    may hand back a single per-image list directly. Be defensive about both.
    """
    if flows is None:
        return None
    first = flows[0] if len(flows) else None
    # per-image case: first is the [RGB, dP, cellprob, ...] list
    if isinstance(first, (list, tuple)) and len(first) > 1:
        return first[1]
    # already a per-image flow list: flows == [RGB, dP, cellprob, ...]
    if len(flows) > 1:
        return flows[1]
    return None


def compute_flow_error(mask: np.ndarray, flows, device=None) -> Optional[float]:
    """Mean Cellpose flow error for a single segmented image.

    Compares the network-predicted flow field (dP) with the flow recomputed from
    the final masks via ``cellpose.metrics.flow_error``, on ``device`` (the device the
    model ran on: a model that fell back to the CPU must not have its flows recomputed on
    a GPU it could not open). Returns the mean over labels; 0.0 when there are no labels.
    When it cannot be measured (missing dP, version mismatch, a failure) it returns None
    and logs: a missing metric is reported as missing, never as a perfect 0.0.
    """
    try:
        if mask is None or not np.any(mask):
            return 0.0
        dP = _extract_dP(flows)
        if dP is None:
            logger.warning("compute_flow_error: no dP flow field available; flow_error missing")
            return None
        flow_error_fn = _get_flow_error_fn()
        if flow_error_fn is None:
            logger.warning("compute_flow_error: cellpose flow_error unavailable; flow_error missing")
            return None
        kwargs = {}
        if device is not None and "device" in inspect.signature(flow_error_fn).parameters:
            kwargs["device"] = device
        flow_errors, _ = flow_error_fn(mask, dP, **kwargs)
        flow_errors = np.asarray(flow_errors, dtype=float)
        if flow_errors.size == 0:
            return 0.0
        return float(np.nanmean(flow_errors))
    except Exception as e:  # pragma: no cover - defensive, depends on cellpose version
        logger.warning(f"compute_flow_error failed ({e}); flow_error missing")
        return None


def normalize_for_overlay(image: np.ndarray) -> np.ndarray:
    """Convert arbitrary grayscale intensity ranges to uint8 for Cellpose plots."""
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    # BLOCKER FIX (defect 3b, adversarial defect hunt): `lo`/`hi` are correctly computed
    # over the finite subset above, but this line used to apply them to the original `arr`,
    # which can still hold NaN/Inf. `(NaN - lo) * ... = NaN`, `np.clip` does NOT remove NaN
    # (NaN comparisons are always False, so it passes through unchanged), and
    # `.astype(np.uint8)` on NaN is UNDEFINED C behaviour (RuntimeWarning + a garbage byte).
    # Replace non-finite entries with the same defined bound `clip` would already apply to a
    # finite extremum -- NaN/-Inf -> the low end, +Inf -> the high end -- before the cast.
    safe = np.where(np.isnan(arr), lo, arr)
    safe = np.where(np.isneginf(safe), lo, safe)
    safe = np.where(np.isposinf(safe), hi, safe)
    return np.clip((safe - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def to_display_uint8(image: np.ndarray) -> np.ndarray:
    """Canonical image -> uint8 used for display, overlays, and the SSL crop input.

    uint8 passes through; everything wider (uint16, float) is range-normalized by
    the 1-99 percentile. A fixed uint16 scale (`/65535`) was WRONG for microscopy:
    fluorescence/16-bit images use only a small part of the range (cells at a few
    thousand counts), so dividing by 65535 crushed them to near-black -- which then
    made the segmenter's quality check misfire and Cellpose see nothing. Percentile
    normalization makes the same raw image look the same, and visible, everywhere.
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    return normalize_for_overlay(arr)


def evaluate_segmentation_visual(mask: np.ndarray, overlay: np.ndarray) -> dict:
    """Measure whether mask and overlay contain inspectable segmentation output."""
    mask = np.asarray(mask)
    overlay = np.asarray(overlay)
    labels = np.unique(mask)
    n_cells = int(len(labels) - (1 if 0 in labels else 0))
    foreground_coverage = float(np.mean(mask > 0)) if mask.size else 0.0
    overlay_max = float(np.max(overlay)) if overlay.size else 0.0
    overlay_std = float(np.std(overlay)) if overlay.size else 0.0
    overlay_dynamic_range = (
        float(np.max(overlay) - np.min(overlay)) if overlay.size else 0.0
    )
    normalized_scale = overlay_max <= 1.0
    black_level = 0.02 if normalized_scale else 5.0
    flat_range = 0.01 if normalized_scale else 2.0
    overlay_is_black = bool(
        overlay.size == 0
        or overlay_max <= black_level
        or overlay_dynamic_range <= flat_range
    )
    coverage_implausible = bool(
        n_cells > 0 and (foreground_coverage <= 0.0001 or foreground_coverage >= 0.98)
    )
    return {
        "cell_count": n_cells,
        "foreground_coverage": foreground_coverage,
        "overlay_max": overlay_max,
        "overlay_std": overlay_std,
        "overlay_dynamic_range": overlay_dynamic_range,
        "overlay_is_black": overlay_is_black,
        "coverage_implausible": coverage_implausible,
        "passed": bool(n_cells > 0 and not overlay_is_black and not coverage_implausible),
    }
