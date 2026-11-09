"""Verifier 4: Crop (pure Python, partially tunable).

A crop is good when the TARGET cell is fully contained and centered (occupies the
main position). Neighbour cells are TOLERATED: in a dense field a tight bbox crop
always admits neighbours, and we do NOT mask them out or shrink the box to chase a
dominance target -- that destroys real phase-contrast context and biases SSL.
`target_dominance` is reported for visibility but is NOT a validity gate: shrinking
crop_size cannot lift it (the box is already bbox-sized, so the crop_size cap is a
no-op), so gating on it only produced false failures. The real levers are crop
geometry: pad a truncated cell, or recenter an off-center one.

Two verifiers: the linear `CropVerifier` (fixed geometry rules, one retune per round) and
`CropAgenticVerifier`, the cropper's `base.AgenticVerifier` (the agentic VRR the crop stage runs for
Single_Cell_Cropper_Tool; CARD.md "Agentic VRR").
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Literal, Optional, Union
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, model_validator

from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool, _TARGET_RETENTION_FLOOR, pixel_args
# the tool's own min_area / margin rule by mask file name (`execute` with both omitted): one function, the tool's
from shape.tools.single_cell_cropper.tool import default_min_area_margin as tool_default_min_area_margin
from shape.utils.logger import logger
from shape.utils.schemas import VLRRResult, WaveMove, WavePlan, WaveSelection

from ..base import INPUT, PARAM_BOUNDS, AgenticVerifier, BaseVerifier, apply_bounded, numeric
from .preprocessing import absolute_quality_metrics


# Per-crop validity thresholds (the single source of truth for "is this crop usable",
# shared by crop_qc(), CropVerifier, and the bare-crop path). Centering is gated;
# dominance is intentionally NOT (neighbours are tolerated, never masked). The
# pixel-quality gates only fire when their metric is present (the mask path omits them,
# so its verdict stays containment + centering exactly as before).
# _CONTAINMENT_MIN / _MIN_DYNAMIC_RANGE / _MAX_SATURATION are absolute quality standards
# (a blank/blown-out/truncated crop is bad regardless of the batch), so they are NOT
# data-derived from the batch. They are agent-tunable knobs (see is_crop_valid's override
# params below) documented as per-input reasoned cuts in CARD.md; these constants are the
# deterministic FALLBACK PRIOR crop_qc() uses when no reasoning overrides them.
# `_CONTAINMENT_MIN` is the cropper's retention floor, `tool.py::_TARGET_RETENTION_FLOOR` (0.90, the single
# source; human decision 2026-10-01 -- it was 0.95 here, so a crop the tool and the agentic guard keep at
# 0.90-0.95 read as truncated): a crop is truncated when it keeps less than the floor of its target.
_CONTAINMENT_MIN = _TARGET_RETENTION_FLOOR   # target must keep >= the floor inside the crop (mask path only)
_CENTROID_OFFSET_MAX = 0.30      # centering: offset as a fraction of half the short side
_MIN_DYNAMIC_RANGE = 0.05        # below this the crop is blank/flat (no cell signal)
_MAX_SATURATION = 0.5            # above this the crop is blown out
_MIN_CROP_PIXELS = 32            # smaller side below this is a degenerate crop

# Validity reasons grouped by facet, so a caller can choose WHICH facets exclude a crop
# (the metrics are always computed/reported; gating is the experiment knob). "center" is
# split out because centering estimated from a bare crop can be noisy for some stains, so
# a caller may want pixel quality without centering.
CROP_QC_GROUPS = {
    "center": ("off_center",),
    "pixel": ("truncated", "non_finite", "blank_low_contrast", "saturated", "too_small"),
}


def is_crop_valid(crop: dict, gate_groups=None,
                  centroid_offset_max: float = _CENTROID_OFFSET_MAX,
                  containment_min: float = _CONTAINMENT_MIN,
                  min_dynamic_range: float = _MIN_DYNAMIC_RANGE,
                  max_saturation: float = _MAX_SATURATION) -> tuple[bool, list[str]]:
    """Single per-crop validity gate, used by crop_qc() and the benchmark exclusion.

    Each criterion fires ONLY when its metric is present, so a cropper-derived crop
    (containment + centering, no pixel-quality fields) is judged exactly as before, while
    a bare crop additionally gets the pixel-quality facets. Dominance is never a gate.

    `gate_groups` (a subset of CROP_QC_GROUPS keys) chooses which facets invalidate a crop;
    None = all facets (the verifier's default). Returns (ok, reasons) where `reasons` are
    the GATING reasons that fired (filtered to the selected groups).

    `centroid_offset_max` is the off-center cut. It defaults to the fixed `_CENTROID_OFFSET_MAX`,
    but centering here is an ESTIMATE (Otsu largest-CC centroid), noisy for reticular/peripheral
    stains, so a caller may pass a DATA-DERIVED bound (see
    initial_params.recommend_off_center_bound) that removes only extreme outliers.

    `containment_min` / `min_dynamic_range` / `max_saturation` default to the fixed
    `_CONTAINMENT_MIN` / `_MIN_DYNAMIC_RANGE` / `_MAX_SATURATION` (the fallback prior); a
    caller may instead pass an AGENT-REASONED cutoff tuned to the input's modality/contrast
    context (see CARD.md's per-input knob documentation for this tool)."""
    fired: list[str] = []
    mc = crop.get("mask_containment")
    # Task 038 v12 (human-locked, the brief's own SS2.5 containment hazard): a
    # `mask_bbox`-mode crop's `mask_containment` MAY legitimately drop below
    # `containment_min` once a directional `bbox_inset_px` has been applied -- that is
    # the INTENDED effect of the v12 directional-tighten lever (CARD.md SS1/SS5's own
    # updated rationale, `verifiers/crop.py::CropVerifier`'s own `_TARGET_RETENTION_
    # FLOOR` already guards the hard floor separately), not a truncation defect -- so
    # `mask_bbox`-mode crops are EXEMPT from this gate (the "cleaner" option per
    # the brief's own §2.5, chosen over tuning `containment_min` down to a number that
    # would "happen" to sit below the retention floor for these crops). A PLAIN,
    # uninsetted mask_bbox crop -- the square around the target's bbox since the human
    # rule of 2026-09-30 -- has `mask_containment` exactly 1.0 whenever the bbox fits the
    # image's shorter side (`tool.py`'s own comment on that mode; a longer bbox is truncated
    # by the largest square that fits and kept only at the tool's 0.90 retention floor),
    # so this exemption is a no-op until an inset is actually applied. A crop AT `containment_min` keeps it:
    # not truncated (strict `<`, like the floor everywhere -- "keeps at least").
    if mc is not None and mc < containment_min and crop.get("margin_mode") != "mask_bbox":
        fired.append("truncated")
    off = crop.get("centroid_offset")
    if off is not None and off >= centroid_offset_max:
        fired.append("off_center")
    fin = crop.get("finite_fraction")
    if fin is not None and fin < 1.0:
        fired.append("non_finite")
    dr = crop.get("dynamic_range")
    if dr is not None and dr < min_dynamic_range:
        fired.append("blank_low_contrast")
    sat = crop.get("saturation_fraction")
    if sat is not None and sat > max_saturation:
        fired.append("saturated")
    msp = crop.get("min_side_px")
    if msp is not None and msp < _MIN_CROP_PIXELS:
        fired.append("too_small")
    if gate_groups is None:
        gating = fired
    else:
        allowed = set().union(*(CROP_QC_GROUPS.get(g, ()) for g in gate_groups))
        gating = [r for r in fired if r in allowed]
    return (not gating), gating


# --- Task 042 (B): lenient, KEEP-BIASED "big problem" reject gate -----------------
# Deterministic, NOT a vision judge (041's crop_triage over-rejected 87% of good crops
# -- BRIEF). Drops a crop ONLY on a "big problem" the tight-margin lever (A, tool.py's
# `margin_mode="tight"`) cannot fix: severe blur, blank, saturated, the target being a
# swamped minority of its own crop's foreground, or the target's own mask fragmented
# into multiple pieces.
#
# Every default below is calibrated LENIENT (reject rarely), grounded in the REAL "good"
# crop population from 035's own eval set (archives/035_cropper-gtfree-quant/results/
# per_crop_scores.csv -- 43 REAL, non-synthetic base crops, only the defect groups in
# that set are synthetic degradations): on that data `sharpness` for good crops ranges
# 0.0036-0.38 (median 0.026, p5=0.0054) -- so 035's OWN "0.5" discrimination midpoint
# would wrongly reject ~72% of good crops if reused as a reject threshold here (see the
# coding log). `_SHARPNESS_MIN` is instead set just below that empirical p5, so <5% of a
# representative good population trips it. `_SINGLE_OBJECT_MIN=0.5` is an exact cut (the
# score is 1/n_significant for an integer piece-count n, so 0.5 fires only at n>=2 --
# rejects 2.3% of the same good set). `_TARGET_DOMINANCE_MIN=0.5` is a reasoned (not
# data-fit) cut: "swamped" is read literally as the target being a MINORITY of its own
# crop's foreground. Every threshold is human-tunable (CARD.md) -- tighten ONLY for an
# input where a default is demonstrably wrong for that input's modality.
_SHARPNESS_MIN = 0.005            # blur: below this the crop is genuinely, severely blurry
_SINGLE_OBJECT_MIN = 0.5          # fragmentation: target mask itself splits into >=2 significant pieces
_TARGET_DOMINANCE_MIN = 0.5       # target-swamped: OTHER cells outnumber the target's own pixels


def is_crop_keep(crop: np.ndarray, mask: np.ndarray | None = None, *,
                  target_dominance: float | None = None,
                  sharpness_min: float = _SHARPNESS_MIN,
                  single_object_min: float = _SINGLE_OBJECT_MIN,
                  target_dominance_min: float = _TARGET_DOMINANCE_MIN,
                  min_dynamic_range: float = _MIN_DYNAMIC_RANGE,
                  max_saturation: float = _MAX_SATURATION) -> tuple[bool, list[str]]:
    """Task 042 (B): lenient, KEEP-BIASED reject gate -- the companion to (A)'s
    tightest per-cell margin (`tool.py::effective_margin(margin_mode="tight")`). Drops
    a crop ONLY for a "big problem" the margin lever cannot fix; every threshold is a
    clearly-named, human-tunable param (see the module comment above for how the
    defaults were calibrated). Returns (keep, reasons); `reasons` is a subset of
    {"blur", "blank", "saturated", "target_swamped", "fragmented"} (empty when kept).

    Reuses, does NOT reimplement, the existing deterministic gates:
      - blur -> `quant.sharpness` (035's GT-free blur estimate).
      - fragmented -> `quant.single_object`; when `mask` (the target's OWN per-cell
        mask, the same file the cropper already persists per crop) is given, it counts
        THAT mask's connected components directly (authoritative -- a per-cell mask by
        definition holds no neighbour-cell pixels, so this measures the TARGET's own
        mask being split, never a tolerated neighbour -- see this module's docstring).
      - blank / saturated -> `is_crop_valid`'s existing pixel gates (013), the single
        source of truth for those two facets (reused via its `pixel` gate group).
      - target_swamped -> the caller-supplied `target_dominance` when given -- the
        MASK-based value `tool.py::_compute_crop_qc` already computes from the
        full-field labeled mask (e.g. `crop_metadata["qc"]["target_dominance"]`), the
        metric BRIEF caveat 1 cites (dense-field crops can be genuine doublets even at
        the tightest margin: "margin=1, still 79% doublets"). quant.py's OWN
        `target_dominance` is a DIFFERENT, weaker pixel-Otsu estimate that quant.py's
        own docstring already excludes from scoring ("crop.py itself documents
        dominance as... NOT a validity gate"), and on the real 035 calibration data it
        does NOT separate doublet crops from good ones (doublet p5=0.87 > good
        p5=0.78 -- backwards; see the coding log) -- so it is used here ONLY as a
        last-resort fallback when no mask-based value is supplied (e.g. a bare crop
        with no full-field context)."""
    from shape.tools.single_cell_cropper import quant

    crop = np.asarray(crop)
    img2d = crop if crop.ndim == 2 else crop[..., 0]
    if mask is None:
        mask2d = None
    else:
        mask_arr = np.asarray(mask)
        mask2d = mask_arr if mask_arr.ndim == 2 else mask_arr[..., 0]
    reasons: list[str] = []

    if quant.sharpness(img2d) < sharpness_min:
        reasons.append("blur")
    # single_object is EXACTLY 1/n_significant for an integer piece-count n (1, 0.5,
    # 0.333, ...) -- "<=" (not "<") so the default 0.5 threshold actually fires at the
    # n=2 boundary (a genuine second significant piece), not only for n>=3.
    if quant.single_object(img2d, mask2d) <= single_object_min:
        reasons.append("fragmented")

    dominance = (quant.target_dominance(img2d) if target_dominance is None
                 else float(target_dominance))
    if dominance < target_dominance_min:
        reasons.append("target_swamped")

    if np.issubdtype(img2d.dtype, np.integer):
        bit_range = float(np.iinfo(img2d.dtype).max)
    else:
        finite = img2d[np.isfinite(img2d)]
        bit_range = float(max(finite.max(), 1.0)) if finite.size else 1.0
    pix = absolute_quality_metrics(img2d, bit_range=bit_range)
    _, pixel_reasons = is_crop_valid(
        {"dynamic_range": pix["dynamic_range"], "saturation_fraction": pix["saturation_fraction"]},
        gate_groups=["pixel"], min_dynamic_range=min_dynamic_range, max_saturation=max_saturation,
    )
    if "blank_low_contrast" in pixel_reasons:
        reasons.append("blank")
    if "saturated" in pixel_reasons:
        reasons.append("saturated")

    return (not reasons), reasons


def _centering_dominance_from_pixels(raw: np.ndarray) -> tuple[float, float]:
    """Estimate (centroid_offset, target_dominance) for a bare crop from pixels alone:
    Otsu foreground -> largest connected component = target cell. centroid_offset uses
    the SAME normalization as the cropper's mask-based _compute_crop_qc (offset /
    (0.5*min(H,W))). Returns (0.0, 0.0) when there is no usable foreground (a flat/blank
    crop, which the dynamic-range gate excludes anyway)."""
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    from shape.utils.segmentation_qc import to_display_uint8

    img = np.asarray(to_display_uint8(raw), dtype=np.float64)
    if img.ndim != 2:
        return 0.0, 0.0
    h, w = img.shape
    if not np.isfinite(img).any() or float(np.ptp(img)) < 1e-6:
        return 0.0, 0.0                      # flat: no foreground to locate
    try:
        foreground = img > threshold_otsu(img)
    except Exception:
        return 0.0, 0.0
    total_fg = int(foreground.sum())
    if total_fg == 0:
        return 0.0, 0.0
    regions = regionprops(label(foreground))
    if not regions:
        return 0.0, 0.0
    target = max(regions, key=lambda r: r.area)
    cy, cx = target.centroid
    offset = float(np.hypot(cy - (h - 1) / 2.0, cx - (w - 1) / 2.0) / (0.5 * min(h, w)))
    return offset, float(target.area / total_fg)


def bare_crop_qc(path: str) -> dict:
    """Per-crop QC for a bare (mask-less) crop image, producing the same keys crop_qc /
    is_crop_valid consume. Pixel quality is read from RAW pixels (the SSL percentile
    stretch would amplify a blank crop's noise and hide it); centering/dominance are
    estimated from the displayed image. mask_containment is omitted (unmeasurable without
    a mask), so the truncation gate does not fire here."""
    try:
        from skimage import io as skio
        raw = np.asarray(skio.imread(path))
    except Exception as exc:
        # Unreadable -> non_finite gate excludes it.
        return {"path": path, "error": str(exc), "finite_fraction": 0.0,
                "dynamic_range": 0.0, "saturation_fraction": 0.0,
                "min_side_px": 0, "centroid_offset": 0.0, "target_dominance": 0.0}
    if raw.ndim == 3:
        raw = raw[..., 0]                    # single-channel dataset: first channel
    if raw.ndim != 2 or raw.size == 0:
        return {"path": path, "finite_fraction": 0.0, "dynamic_range": 0.0,
                "saturation_fraction": 0.0, "min_side_px": 0,
                "centroid_offset": 0.0, "target_dominance": 0.0}
    if np.issubdtype(raw.dtype, np.integer):
        bit_range = float(np.iinfo(raw.dtype).max)        # 255 (uint8) / 65535 (uint16)
    else:
        finite = raw[np.isfinite(raw)]
        bit_range = float(max(finite.max(), 1.0)) if finite.size else 1.0
    m = absolute_quality_metrics(raw, bit_range=bit_range)
    h, w = raw.shape
    offset, dominance = _centering_dominance_from_pixels(raw)
    return {
        "path": path,
        "finite_fraction": m["finite_fraction"],
        "dynamic_range": m["dynamic_range"],
        "saturation_fraction": m["saturation_fraction"],
        "min_side_px": int(min(h, w)),
        "centroid_offset": offset,
        "target_dominance": dominance,       # reported, not gated
    }


def adapt_bare_crops(crop_paths, workers: int = 8) -> tuple[list[dict], dict]:
    """Pre-made bare crop image paths -> (list[crop-qc dict], context), the same shape
    adapt_crop returns, so the SAME CropVerifier / crop_qc / is_crop_valid run on crops
    that have no segmentation mask (e.g. the HPA benchmark). Threaded: bare_crop_qc is
    disk-I/O + Otsu bound (skimage.io.imread + threshold_otsu + connected components), so
    threads help and a GPU does not -- order-preserving, mirrors the HPA benchmark's own
    agentic-sample crop QC (diagnostic_planner._parallel_qc)."""
    crop_paths = list(crop_paths)
    if len(crop_paths) <= 1:
        crops = [bare_crop_qc(p) for p in crop_paths]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(workers, len(crop_paths))) as ex:
            crops = list(ex.map(bare_crop_qc, crop_paths))
    return crops, {"upstream_seg": {}}


def _read_crop_metadata(result_or_path) -> tuple[str, dict]:
    if isinstance(result_or_path, dict):
        path = result_or_path.get("cell_crops_metadata_path")
    else:
        path = result_or_path
    if not path or not os.path.exists(path):
        raise ValueError(f"adapt_crop: metadata file not found ({path})")
    with open(path, "r") as f:
        return path, json.load(f)


def adapt_crop(result_or_metadata_path, seg_metrics=None):
    """Cropper output -> (list[crop-qc dict], context).

    Folded out of the shared adapters bridge (Phase 2b). Reads the per-crop `qc`
    blocks the cropper wrote into its metadata JSON and flattens them into the list
    the crop verifier expects. `seg_metrics` (the upstream segmentation metrics)
    populates context.upstream_seg so contamination can be routed to segmentation vs
    the crop box.

    Task 038 C1/C4 (FIX round): each crop dict ALSO carries the raw `qc` block verbatim
    as `crop["qc"]`, IN ADDITION TO the flattened fields above (kept for backward compat,
    unchanged) -- so `crop_triage._quant_evidence` can read the cropper's own mask-based
    target_dominance/border_touch instead of re-deriving a weaker pixel-Otsu estimate.
    `context` also now carries `metadata_path` (the resolved metadata file's own path), so
    the crop-stage Recommend can write its kept-crops manifest (`llm_recommend.
    write_kept_manifest`, task 038 M-B) next to the right metadata file.

    Task 038 follow-up (part B, result-driven per-cell tighten): each crop dict ALSO
    carries `cell_id` (the tool's own stable per-cell identifier, `tool.py::
    _generate_cell_crops`'s own `cell_id`) -- `CropVerifier`'s new per-cell geometry
    trigger/retune correlates a crop back to the `cell_overrides` key that would tighten
    it. `context` ALSO now carries `cell_overrides` -- the tool's OWN
    `parameters.cell_overrides` echo (whatever was ALREADY applied to produce THIS
    result), read back so `adjudicate` can MERGE a newly-flagged cell into it rather than
    silently dropping a previous iteration's tighten (the tool's per-cell override,
    unlike padding/margin_frac, is not itself a numeric param `apply_bounded` can
    increment -- it is a whole dict, categorically REPLACED each retune, so the caller
    must carry its own history forward).

    Task 038 v11 (FIX, human-locked): each crop dict ALSO carries `mask_path` (`tool.py`
    has written this into `m` since the v9 restructure, but this adapter never threaded
    it through until now) -- WITHOUT it, `crop_triage._target_mask_pixels` always
    returns None end-to-end, so the montage's own colorful grounding overlay
    (`_crop_overlay_array`) silently falls back to plain grayscale for every REAL crop,
    even though the direct-unit-tested overlay function itself works fine with a mask.
    A real v11 QC inspection of the saved montage PNGs (task 038 v10's own persistence
    feature) is what caught this: they came out grayscale.

    Task 038 v12 (human-locked): each crop dict ALSO carries `margin_mode` (`tool.py`'s
    own RESOLVED per-cell mode for THIS crop, `crop_metadata["margin_mode"]`) --
    `margin_mode` lets `CropVerifier.adjudicate` tell round-1 (not yet mask_bbox) from
    round-2+ (already mask_bbox, may need a further directional tighten) apart, and
    lets `is_crop_valid` exempt a mask_bbox crop from its own containment gate (§2.5's
    hazard, above). `contamination_side`/`target_retention`/`contamination_inset_px`
    (the v12 lever's own QC fields, `tool.py::_compute_crop_qc`) need NO new plumbing
    here -- they already arrive inside the existing, verbatim `crop["qc"]` block below.
    (Task 038 v12 R2, human-directed: `crop_size` was threaded here ONLY for the
    now-REMOVED fixed-fraction step projection this round replaced with a
    neighbour-aware, data-computed depth -- `tool.py` now computes the exact inset
    depth itself from the mask, so this adapter needs no crop-dimension plumbing for
    that purpose any more; removed as an orphan of that removal.)

    Task 038 v13 (human-directed): each crop dict ALSO carries `baseline_crop_path`
    (`tool.py`'s own render of the plain, zero-inset mask_bbox box for a genuinely
    tightened cell, `crop_metadata["baseline_crop_path"]`) -- the second, REAL image
    `crop_triage.ab_recommend_crop`'s blinded A/B channel compares the tightened
    candidate against, with no history threading needed. None for a crop not on
    `mask_bbox`, or on `mask_bbox` with no inset applied yet (nothing to compare).
    """
    metadata_path, meta = _read_crop_metadata(result_or_metadata_path)
    crops = []
    for m in meta.get("cell_metadata", []):
        qc = m.get("qc") if isinstance(m, dict) else None
        if qc:
            contamination = float(qc.get("neighbor_contamination", 0.0))
            crops.append({
                "cell_id": m.get("cell_id"),
                "mask_containment": float(qc.get("mask_containment", 1.0)),
                "occupancy": float(qc.get("occupancy", 0.3)),
                "centroid_offset": float(qc.get("centroid_offset", 0.0)),
                "border_touch": float(qc.get("border_touch", 0.0)),
                "neighbor_contamination": contamination,
                "target_dominance": float(qc.get("target_dominance", 1.0 - contamination)),
                "crop_path": m.get("crop_path"),   # image path, for the optional vision triage judge
                "mask_path": m.get("mask_path"),   # task 038 v11 fix: per-cell mask path, for
                                                    # `crop_triage._target_mask_pixels` -- WITHOUT
                                                    # this the colorful grounding overlay silently
                                                    # never fires end-to-end (`tool.py` already
                                                    # writes `mask_path` into `m`, task 038 v9; this
                                                    # adapter just never threaded it through -- a
                                                    # real gap a v11 QC inspection of the saved
                                                    # montages caught: they came out grayscale).
                "margin_mode": m.get("margin_mode"),   # task 038 v12: round-1-vs-round-2+ dispatch
                                                        # + the is_crop_valid containment exemption
                "bbox_inset_px": dict(m.get("bbox_inset_px") or {}),   # task 038 FIX round:
                                      # THIS cell's own applied directional inset (`tool.py`'s
                                      # own `crop_metadata["bbox_inset_px"]`, {} unless a
                                      # genuine cut was applied) -- lets `crop_qc()` tell a
                                      # DELIBERATELY tightened crop (truncation/off-centering
                                      # is the intended effect, floor-guarded) from an
                                      # accidental defect on any OTHER geometry.
                "baseline_crop_path": m.get("baseline_crop_path"),   # task 038 v13: the plain
                                      # mask_bbox render (`tool.py`'s own `buf_minr`/etc box,
                                      # BEFORE this cell's inset), for the blinded A/B channel --
                                      # None unless this cell has a genuine applied inset.
                "baseline_mask_path": m.get("baseline_mask_path"),   # task 038 v13: the
                                      # baseline's own correctly-sized target mask (a mismatched
                                      # mask size would silently fall back to no overlay).
                "qc": qc,                          # raw mask-based QC block (task 038 C1)
            })
    if not crops:
        raise ValueError("adapt_crop: no per-crop QC found in metadata")
    cell_overrides = dict(meta.get("parameters", {}).get("cell_overrides") or {})
    context = {"upstream_seg": dict(seg_metrics or {}), "metadata_path": metadata_path,
              "cell_overrides": cell_overrides}
    return crops, context


def crop_qc(crops: list[dict]) -> dict[str, float]:
    """Each crop dict: mask_containment, centroid_offset, target_dominance
    (and legacy occupancy / border_touch / neighbor_contamination), per-crop floats.

    A crop is valid when the target is contained and centered. Dominance / neighbour
    contamination are reported but NOT part of validity (neighbours are tolerated)."""
    def col(k, default):
        return np.array([c.get(k, default) for c in crops], dtype=float)

    arr = {
        "mask_containment": col("mask_containment", 1.0),
        "centroid_offset": col("centroid_offset", 0.0),
        "border_touch": col("border_touch", 0.0),
        # per-crop default (1 - contamination); element-wise so a mix of crops with and
        # without target_dominance does not collapse into an array-valued default.
        "target_dominance": np.array(
            [float(c.get("target_dominance",
                         1.0 - float(c.get("neighbor_contamination", 0.0))))
             for c in crops], dtype=float),
    }
    # validity via the single-source gate (containment + centering on the mask path;
    # pixel-quality facets too where present, e.g. bare crops). No overrides here -> the
    # fixed fallback-prior constants (see comment above _CONTAINMENT_MIN); an agentic caller
    # may pass reasoned overrides directly to is_crop_valid (see CARD.md). `vision_reject`
    # (set by the optional crop_triage judge, task 019/041) additionally invalidates a crop --
    # absent unless a vision engine ran, so this is a strict no-op on the deterministic path.
    valid = np.array([is_crop_valid(c)[0] and not c.get("vision_reject", False)
                      for c in crops], dtype=bool)
    mean_centered = float(1.0 - arr["centroid_offset"].mean())

    # task 038 FIX round (coordinator-directed): a DELIBERATELY tightened crop
    # (`margin_mode=="mask_bbox"` AND a non-empty APPLIED `bbox_inset_px` -- a genuine
    # directional cut was actually committed, not just plain mask_bbox) truncates the
    # target (down to `CropVerifier._TARGET_RETENTION_FLOOR`, 0.90) and off-centers it
    # (a directional inset on the neighbour side only) BY DESIGN -- the v13 auto_tighten
    # lever's own intended effect, already floor-guarded at the tool level (never
    # truncates further than the retention floor allows). A real headless run showed
    # `localize()`'s `truncation_fraction`/`off_center_fraction` triggers firing on
    # EXACTLY this intended effect (a cleanly-tightened image reported WARNING/FAIL:
    # "target cell touches/exceeds the crop border" / "off-center"), reporting the
    # tool's own successful work as a defect. A tightened crop's own truncation/off-
    # centering therefore does NOT count toward these two fractions below -- an
    # UNTIGHTENED crop (plain `fraction`/`tight` mode, or plain zero-inset `mask_bbox`)
    # keeps the EXISTING defect semantics, completely unchanged.
    is_tightened = np.array([
        c.get("margin_mode") == "mask_bbox"
        and any((c.get("bbox_inset_px") or {}).get(k) for k in ("top", "bottom", "left", "right"))
        for c in crops], dtype=bool)

    # task 038 follow-up (item 3 part 1): a MASK-ONLY (no pixel I/O -- cheap even at
    # 300+ crops per VLRR iteration) aggregate crop_quality_score PROXY, reusing
    # `quant.py`'s own `_WEIGHTS` proportions directly (task 038 v9 restructure:
    # `_WEIGHTS` now holds EXACTLY these 3 axes -- not_truncated/target_dominance/
    # centered, the ones this tool's own geometry levers actually adjust -- so this
    # proxy and `quant.py`'s own weighted score are now the SAME 3-axis formula, not a
    # 4-of-5 subset; `sharpness`/`single_object` were already excluded here for needing
    # pixel I/O this stage's flattened crops list does not carry, and are now ALSO
    # excluded from `_WEIGHTS` itself for a second, independent reason -- see that
    # module's own comment). Drives the CONTINUOUS headroom trigger in
    # `CropVerifier.localize()` below, replacing the old binary local-defect-only
    # thresholds tight+recenter geometry is built to almost never trip. NOT the same
    # number as the benchmark's own `crop_quality_score` (`run_shape_crop.py`, which
    # DOES include real pixel sharpness in its OWN reported dict, just not its weighted
    # sum either) -- that stays the headline metric reported in trace.jsonl; this proxy
    # only decides IN-LOOP whether there is headroom worth trying to retune.
    from shape.tools.single_cell_cropper.quant import _WEIGHTS as _quant_weights
    proxy_parts = {
        "not_truncated": float((1.0 - arr["border_touch"]).mean()),
        "target_dominance": float(arr["target_dominance"].mean()),
        "centered": mean_centered,
    }
    proxy_weights = {k: _quant_weights[k] for k in proxy_parts}
    mean_crop_quality_score_proxy = (
        sum(proxy_parts[k] * proxy_weights[k] for k in proxy_parts) / sum(proxy_weights.values()))

    return {
        "valid_crop_rate": float(valid.mean()),
        "mean_containment": float(arr["mask_containment"].mean()),
        "mean_centroid_offset": float(arr["centroid_offset"].mean()),
        "mean_dominance": float(arr["target_dominance"].mean()),   # informational
        "mean_centered": mean_centered,
        "mean_crop_quality_score_proxy": float(mean_crop_quality_score_proxy),
        # same fixed constant the gate above just used (closes the partial-wiring gap: this
        # metric and is_crop_valid's truncation reason now read the same source). A
        # DELIBERATELY tightened crop (`is_tightened`, above) never counts toward either
        # fraction's own numerator -- the denominator (population size) is unaffected, so
        # this reads as "what fraction of crops are DEFECTIVELY truncated/off-center",
        # not "what fraction happen to measure below the raw threshold regardless of why".
        "truncation_fraction": float(((arr["mask_containment"] < _CONTAINMENT_MIN) & ~is_tightened).mean()),
        "off_center_fraction": float(((arr["centroid_offset"] >= _CENTROID_OFFSET_MAX) & ~is_tightened).mean()),
    }


# Task 038 Round B, S1 channel (coordinator-directed, evidence-driven -- NOT a
# preference): the human flagged 5 kept crops (cell_00299/0037, cell_00994/0001,
# /0011, /0068, /0323) as obviously-should-be-discarded. All 5 share ONE pattern:
# the mask claims a cell, but the visible cell material sits OUTSIDE the target
# mask -- the mask itself is wrong, so NO mask-derived metric (contamination
# 0.10-0.26, retention 0.90-1.0 for these 5 -- all comfortably "fine" by that lens)
# can ever catch it. Three cheap data-only detectors were tried and all three
# failed (mean contrast: modality-confounded; boundary adherence: density-
# confounded; intensity histogram: missed cell_0323 outright at frac<10=0.09) --
# this is the VLM's job (`crop_triage.s1_recommend_crop`). This function is only
# the DELIBERATELY LOOSE data screen deciding which crops are even worth asking
# about (a false candidate costs one montage slot; a missed one ships a garbage
# crop straight through) -- it is NEVER itself an accept/reject authority.
#
# Rule (measured, not assumed -- see CODING_LOG.md for the full derivation against
# the real delivered v15 crop files): the 90th percentile of pixel brightness
# WITHIN the target mask falls below `_S1_TARGET_P90_MAX` (160/255). `target_p90`
# is a ROBUST refinement of "target-region max intensity low" -- the literal
# per-crop MAX inside the mask is NOT reliable on its own (`cell_00994/0011`'s own
# target region contains a single pixel at 255, the crop's own brightest pixel,
# purely from noise/an edge artifact, even though the BULK of its target region is
# genuinely dim) -- the 90th percentile is robust to that single outlier while
# still catching a target region that is substantially, not just momentarily,
# dim. Measured against the real delivered v15 crops: all 5 named crops have
# target_p90 in [54.9, 138.0], comfortably below 160 (22+ point margin even at the
# WORST case); a random sample of the broader kept population puts ~17% below
# this same bar -- a deliberately loose, NOT curve-fit, net (a round number with
# margin, not tuned to exactly clear these 5 -- the threshold was chosen BEFORE
# checking how tightly it could be shrunk around them).
_S1_TARGET_P90_MAX = 160.0


def _s1_target_p90(crop: dict) -> float | None:
    """The 90th percentile of pixel brightness WITHIN the target mask -- arm (a)'s
    (the DIM arm's) own SUSPICION SCORE (LOWER = more suspicious) AND the shared
    per-image `ref_p90` stat (`_gate_image_stats`) is built from this SAME value.
    Extracted as its own function (task 038 Round D, coordinator-directed) so a
    CALLER can also SORT candidates by it, not only filter on it -- a real v17 LIVE
    run showed the OLD arbitrary (crop-iteration) candidate order truncated by
    `s1_candidate_cap` before ever reaching `cell_00994/0068` (target_p90=54.9, the
    SINGLE MOST suspicious of the human's own 5 named crops) -- `CropVerifier.
    adjudicate` now sorts by this value ASCENDING before capping, so the cap keeps
    the MOST suspicious candidates, not whichever happened to be reached first.
    Reuses `crop_triage._crop_pixels` (max-projected across channels, task 038
    Round C) and `_target_mask_pixels` -- NOT reimplemented, this module already
    imports OTHER `crop_triage` functions the same way (`ab_recommend_crop`). None
    on ANY missing/unreadable pixel/mask data."""
    from ..crop_triage import _crop_pixels, _target_mask_pixels

    pixels = _crop_pixels(crop)
    mask = _target_mask_pixels(crop)
    if pixels is None or mask is None:
        return None
    mask_bool = np.asarray(mask) > 0
    if mask_bool.shape != pixels.shape or not mask_bool.any():
        return None
    return float(np.percentile(pixels[mask_bool], 90))


# --- Task 058 M4 (human-directed, 2026-07-17): the THREE-ARM OR gate -------------
# REPLACES the single p90-only screen above (`_s1_target_region_signal_poor`, now
# removed): the VLM (the mask check) is now TRUSTED (the domain expert re-confirmed its
# rejects are defensible), but the single DIM-only screen sends only dim crops to
# it -- production Cellpose mask errors are typically BRIGHT (debris false-
# positives, doublets), which sail through un-judged and get KEPT (measured: 134/
# 345 kept crops per band have p90>=160, never sent to the VLM). A crop is now an
# mask-check candidate if ANY of:
#   (a) DIM, per-image ADAPTIVE (REPLACES the hardcoded 160 floor): `target_p90 <
#       _GATE_DIM_FRAC * ref_p90`, `ref_p90` = the 75th percentile of THIS image's
#       own crops' `target_p90` (038's arm B rule, a scale-free strict subset of the
#       old 160 rule) -- falls back to the fixed `_S1_TARGET_P90_MAX` only when
#       fewer than `_GATE_REF_MIN_CROPS` crops make that percentile unstable.
#   (b) DOUBLET / merged -- SHAPE, not brightness: `solidity < _GATE_SOLIDITY_MIN`
#       (solidity = mask_area / convex_hull_area). Two touching cells sharing one
#       mask label form a concave "waist" (low solidity); a single round/oval cell
#       stays near-convex. An intensity-BIMODALITY signal was tried and REJECTED
#       (good compact cells scored HIGHER bimodality than doublets) -- do not use it.
#   (c) BRIGHT-DEBRIS false-positive: `mask_area < _GATE_DEBRIS_AREA_FRAC *
#       (this image's MEDIAN crop mask_area)` (a spurious fragment is far smaller
#       than a real cell) OR in-mask intensity std `< _GATE_FLAT_TEXTURE_MAX` (a
#       flat bright blob has no cellular texture).
# Every threshold below is a NAMED, human-set CANDIDATE PER-INPUT KNOB (PROJECT.md
# thesis), never tuned to maximise a result. The OR union is still subject to the
# existing `s1_candidate_cap` (name unchanged -- "keep that gate").
_GATE_DIM_FRAC = 0.5           # arm (a): dim iff target_p90 < this * ref_p90
_GATE_REF_MIN_CROPS = 5        # below this many crops/image, the p75 `ref_p90` is
                                # unstable -- fall back to the fixed _S1_TARGET_P90_MAX
_GATE_SOLIDITY_MIN = 0.90      # arm (b): doublet/merged iff mask_area/convex_hull_area < this
_GATE_DEBRIS_AREA_FRAC = 0.25  # arm (c)/area: too-small iff mask_area < this * image-median mask_area
# arm (c)/texture: too-flat iff in-mask intensity std < this -- picked from the REAL
# m2_clean_eval2 crops (1388 crops across 6 images x {shape, maskcrop}, in-mask std
# measured directly off the delivered crop/mask TIFF+PNG pairs): p1=4.5, p5=11.1,
# median=69.9. Set just below the empirical p5 -- mirrors this module's OWN
# `_SHARPNESS_MIN` precedent (a round number just under the good population's own
# p5, so <5% of a representative population trips it on texture alone), a
# deliberately loose, NOT curve-fit, net.
_GATE_FLAT_TEXTURE_MAX = 10.0


def _target_mask_bool(crop: dict) -> np.ndarray | None:
    """The target's own mask as a boolean array, the shared read the doublet/debris
    arms below build on (`_target_mask_pixels(crop) > 0`). None on any missing/
    unreadable/empty mask (`crop_triage._target_mask_pixels`'s own contract)."""
    from ..crop_triage import _target_mask_pixels

    mask = _target_mask_pixels(crop)
    if mask is None:
        return None
    mask_bool = np.asarray(mask) > 0
    return mask_bool if mask_bool.any() else None


def _target_mask_area(crop: dict) -> float | None:
    """Pixel count of the target's own mask in THIS crop -- arm (c)'s area-vs-
    image-median signal AND the shared per-image `median_area` stat
    (`_gate_image_stats`) is built from this SAME value. None when the mask is
    missing/unreadable/empty."""
    mask_bool = _target_mask_bool(crop)
    return float(mask_bool.sum()) if mask_bool is not None else None


def _target_solidity(crop: dict) -> float | None:
    """mask_area / convex_hull_area of the target's own mask -- arm (b)'s
    doublet/merged SHAPE signature (a concave "waist" where two touching cells
    share one mask label lowers this well below 1.0; a single round/oval cell stays
    near-convex). Uses `skimage.measure.regionprops`'s own `solidity` property
    (exactly this ratio), not reimplemented. The LARGEST connected component is the
    target (mirrors `_centering_dominance_from_pixels`'s own convention above) -- a
    genuine merged doublet is already ONE component by construction (`tool.py`'s
    own docstring: a doublet is one connected label, `single_object` scores it
    1.0), so this is a SHAPE check on that one component, not a fragmentation
    check. None when the mask is missing/unreadable/empty."""
    mask_bool = _target_mask_bool(crop)
    if mask_bool is None:
        return None
    from skimage.measure import label, regionprops

    regions = regionprops(label(mask_bool))
    if not regions:
        return None
    target = max(regions, key=lambda r: r.area)
    return float(target.solidity)


def _target_mask_std(crop: dict) -> float | None:
    """Standard deviation of pixel intensity WITHIN the target mask -- arm (c)'s
    flat-texture signature (a spurious bright fragment/debris blob is featureless;
    a real cell's interior carries organelle/membrane texture). Reuses
    `_crop_pixels`/`_target_mask_pixels` (the SAME primitives `_s1_target_p90`
    already uses), not reimplemented. None on any missing/unreadable/shape-
    mismatched/empty data."""
    from ..crop_triage import _crop_pixels, _target_mask_pixels

    pixels = _crop_pixels(crop)
    mask = _target_mask_pixels(crop)
    if pixels is None or mask is None:
        return None
    mask_bool = np.asarray(mask) > 0
    if mask_bool.shape != pixels.shape or not mask_bool.any():
        return None
    return float(np.std(pixels[mask_bool]))


def _gate_image_stats(crops: list[dict]) -> tuple[float, float]:
    """The two PER-IMAGE stats the three-arm gate needs, computed ONCE over ALL of
    this image's crops (`adjudicate` sees the whole image's own crop population
    every round): `ref_p90` = the 75th percentile of the image's own crop
    `target_p90` values (038's arm B rule, the SAME per-image-adaptive precedent
    `_S1_TARGET_P90_MAX`'s own module comment already used) -- falls back to the
    fixed `_S1_TARGET_P90_MAX` when fewer than `_GATE_REF_MIN_CROPS` crops have a
    usable `target_p90` (a p75 over <5 points is unstable); `median_area` = the
    median target-mask AREA over the image's own crops (0.0, a safe no-op for arm
    (c)'s area check, when no crop has a usable mask area)."""
    p90s = [v for v in (_s1_target_p90(c) for c in crops) if v is not None]
    ref_p90 = (float(np.percentile(p90s, 75)) if len(p90s) >= _GATE_REF_MIN_CROPS
              else _S1_TARGET_P90_MAX)
    areas = [v for v in (_target_mask_area(c) for c in crops) if v is not None]
    median_area = float(np.median(areas)) if areas else 0.0
    return ref_p90, median_area


def _gate_arms(crop: dict, ref_p90: float, median_area: float) -> list[str]:
    """Which of the three arms fire for `crop`, given this image's own `ref_p90`/
    `median_area` (`_gate_image_stats`). A subset of `["dim", "doublet", "debris"]`
    (that order), `[]` when none fire (the crop is not a mask-check candidate). Any
    single arm's own missing/unreadable data simply does not fire that arm --
    conservative, mirrors this module's own "any failure -> no rejection" contract
    for a SCREEN (fails closed onto "never ask" for data it cannot read, not open)."""
    arms: list[str] = []
    p90 = _s1_target_p90(crop)
    if p90 is not None and p90 < _GATE_DIM_FRAC * ref_p90:
        arms.append("dim")
    solidity = _target_solidity(crop)
    if solidity is not None and solidity < _GATE_SOLIDITY_MIN:
        arms.append("doublet")
    area = _target_mask_area(crop)
    too_small = (area is not None and median_area > 0
                and area < _GATE_DEBRIS_AREA_FRAC * median_area)
    std = _target_mask_std(crop)
    too_flat = std is not None and std < _GATE_FLAT_TEXTURE_MAX
    if too_small or too_flat:
        arms.append("debris")
    return arms


def _gate_candidate(crop: dict, ref_p90: float, median_area: float) -> bool:
    """The THREE-ARM OR gate itself (task 058 M4, REPLACES the single
    `_s1_target_region_signal_poor` p90-only screen) -- True iff ANY of
    `_gate_arms` fires. DELIBERATELY LOOSE, never itself an accept/reject
    authority (a false candidate costs one montage slot; a missed one ships a
    garbage crop straight through) -- see the module comment above `_GATE_DIM_
    FRAC` for the full three-arm rule."""
    return bool(_gate_arms(crop, ref_p90, median_area))


def _gate_sort_key(crop: dict) -> float:
    """Suspicion-sort key for the mask-check candidate cap (ascending = most suspicious
    first -- the SAME ordering `_s1_target_p90` ascending always used, task 038
    Round D's own v17-measured fix). `target_p90` when readable; `+inf` (sorts
    LAST, never displacing a genuinely brightness-suspicious candidate) for a crop
    flagged ONLY by the doublet/debris arm with unreadable pixels."""
    p90 = _s1_target_p90(crop)
    return p90 if p90 is not None else float("inf")


def _build_crop_decision_trace(crops: list[dict],
                               ref_p90: float, median_area: float,
                               mask_check_montage_by_item: dict | None = None) -> list[dict]:
    """Task 058 M3 (human directive: "the process must be shown clearly", a CLEAR per-crop decision trace),
    task 058 M4 (`data_screen` now reports the THREE-ARM gate, not just p90) -- ONE
    legible record per crop: `{cell_id, geometry, data_screen, mask_check, final}`. A
    human reading the trace can see, per crop: what geometry did (margin_mode/
    inset_px/neighbor_contamination/target_retention, read straight off `qc` -- the
    SAME fields `tool.py::_compute_crop_qc` already computes, no new tool.py field)
    -> whether the data screen flagged it for the mask check, and WHICH arm(s) fired
    (`_gate_arms`: `target_p90`/`ref_p90`/`solidity`/`mask_area`/`median_area`/
    `texture_std`/`arms_fired`/`verdict`) -> what the mask check said (`sent_to_vlm`/
    `verdict`/`confidence`, None when the crop was never a candidate or the cap
    dropped it) -> kept or rejected and WHY (`final.kept`, `final.reason` -- the
    crop's OWN CURRENT `vision_problem` when rejected by a vision channel, else the
    metric gate's own reasons, else "kept").

    Task 058 M8 (QC fix): `sent_to_vlm` is derived from `crop["mask_check_sent"]`, the
    ACTUAL-dispatch signal `mask_check_recommend_crop` now sets IN PLACE (see its own
    docstring) on every crop placed into a montage batch actually handed to
    `engine.generate` -- set True regardless of whether a verdict comes back for it,
    since `MaskCheckVerdict` explicitly permits a SUBSET response and `_apply_mask_check_
    verdicts` only sets `mask_check_verdict` for positions the response names. (Task 058
    M7's own fix, `mask_check_verdict is not None`, was ITSELF still wrong: a crop that
    WAS sent but whose verdict the model omitted -- or a malformed/empty verdict
    list -- wrongly read `sent_to_vlm=False`. A crop skipped at render time
    (`_mask_check_overlay_array` returns None -- an unreadable overlay) or dropped by the
    cap before the mask check ran never gets `mask_check_sent` set at all, so it still correctly
    reads `sent_to_vlm=False`.) A crop can now legitimately be `sent_to_vlm=True,
    verdict=None` (sent but omitted by a subset response) -- visibly different from
    a crop the screen never flagged (`flagged=False`) or one dropped before it was
    ever sent (`flagged=True, sent_to_vlm=False`).

    Called AFTER every channel this round (geometry / the mask check / (B) REJECT) has
    already run, so `final` reflects each crop's own ACTUAL current state, not a
    snapshot from mid-round -- see the caller's own comment for why.

    COMPATIBILITY (verified, not assumed -- a real cross-benchmark test run caught
    this): this sidecar is ALSO read by `benchmarks/crop_NeurIPS22-CellSeg/
    run_shape_crop.py::_read_triage_trace` (a DIFFERENT benchmark, outside this
    task's authorized files), whose own established, tested contract expects EVERY
    entry to carry the OLDER flat shape (`item_id`/`reject`/`verdict`/`confidence`/
    `problem`/`reasoning`/`montage_path` -- the same shape `_apply_montage_verdicts`/
    `_apply_s1_verdicts` always wrote). Each entry below carries BOTH shapes -- the
    flat keys DERIVED from the SAME `kept`/`reason` this entry's own `final` reports
    (so they can never disagree) -- extending the sidecar strictly ADDITIVELY, never
    replacing/breaking that other benchmark's own reader."""
    from ..crop_triage import crop_kept   # is_crop_valid is already local to this module
    montage_by_item = mask_check_montage_by_item or {}
    trace = []
    for c in crops:
        qc = c.get("qc") or {}
        target_p90 = _s1_target_p90(c)
        solidity = _target_solidity(c)
        mask_area = _target_mask_area(c)
        texture_std = _target_mask_std(c)
        arms_fired = _gate_arms(c, ref_p90, median_area)
        # Task 058 M8 (QC fix): the ACTUAL-dispatch signal, not verdict presence --
        # see this function's own docstring.
        sent_to_vlm = c.get("mask_check_sent") is True
        kept = crop_kept(c)
        if kept:
            reason = "kept"
        elif c.get("vision_reject"):
            reason = c.get("vision_problem") or "vision_reject"
        else:
            _, gate_reasons = is_crop_valid(c)
            reason = ", ".join(gate_reasons) if gate_reasons else "rejected"
        trace.append({
            # legacy-flat COMPATIBILITY keys (see the docstring's own COMPATIBILITY
            # section) -- derived from the SAME kept/reason `final` reports below.
            "item_id": c.get("crop_path"),
            "reject": not kept,
            "verdict": "keep" if kept else "reject",
            "confidence": c.get("mask_check_confidence"),
            "problem": "none" if kept else reason,
            "reasoning": f"[crop decision trace] {reason}",
            "montage_path": montage_by_item.get(c.get("crop_path")),
            # task 058 M3's own new per-crop record ("the process must be shown clearly")
            "cell_id": c.get("cell_id"),
            "geometry": {
                "margin_mode": c.get("margin_mode"),
                "inset_px": qc.get("contamination_inset_px"),
                "neighbor_contamination": qc.get("neighbor_contamination"),
                "target_retention": qc.get("target_retention"),
            },
            "data_screen": {
                "target_p90": target_p90,
                "ref_p90": ref_p90,
                "solidity": solidity,
                "mask_area": mask_area,
                "median_area": median_area,
                "texture_std": texture_std,
                "arms_fired": arms_fired,
                "verdict": "flagged" if arms_fired else "not_flagged",
            },
            "mask_check": {
                "sent_to_vlm": sent_to_vlm,
                "verdict": c.get("mask_check_verdict"),
                "confidence": c.get("mask_check_confidence"),
                "montage_path": montage_by_item.get(c.get("crop_path")),
            },
            "final": {"kept": kept, "reason": reason},
        })
    return trace


class CropVerifier(BaseVerifier):
    name = "crop"
    tool_dir = "single_cell_cropper"   # CARD.md: the tuning manual this stage's recommend follows

    # Opt-in to the vision-LLM crop-TRIAGE Recommend (task 019/041). `llm_engine` (a
    # multimodal engine) is injected by the integration layer when enable_llm_verifier is
    # on -- the SAME hook segmentation/representation use (verifier.py:resolve_managed_stage).
    uses_llm = True
    llm_engine = None
    # Optional per-run cost cap on how many metric-valid crops the vision judge scores (None
    # = judge all metric-valid crops); a caller that wires llm_engine may set this.
    triage_sample_cap = None
    # Task 058 M3 (human-authorized controller change, 2026-07-16): the S3/A-B blinded
    # geometry-comparison gate (below, `ab_recommend_crop`) is now OFF by default -- task
    # 038's own M2 measurement found it HARMFUL (vetoes real tightening, degraded the
    # deterministic metric 0.2565->0.1983) and expensive (7.3M tokens/15 images, one
    # image at 1.9M -- 213 A/B montages vs 84 S1). Guarded behind this flag rather than
    # deleted -- RECOVERABLE, not gone: a caller wanting the old behavior sets
    # `verifier.enable_ab_channel = True` explicitly. When False (the default), the
    # round-2+ VoI screen below still identifies `ab_candidates` (harmless bookkeeping),
    # but `ab_recommend_crop` is never called and no `ab_verdict` is ever set -- every
    # VoI-screened cell then falls through the SAME "no verdict -> commit nothing,
    # SILENT" path a HEADLESS run (no `llm_engine` at all) already used, an existing,
    # already-tested contract (CARD.md: "headless -> pure data, accept every
    # constraint-satisfying cut") -- so disabling this channel needed no change to that
    # landing loop at all.
    enable_ab_channel = False
    # Task 058 M3 round 2 (coordinator-directed fix -- verified, not assumed: grepped
    # `adjudicate` and confirmed BOTH `mask_check_recommend_crop` AND `vision_recommend_
    # crop` were firing on every crop batch, TWO separate VLM reject channels, not
    # one -- `vision_recommend_crop` is the (B) REJECT channel (cut_off/empty_debris,
    # task 038's ORIGINAL restructure, PREDATES S1/the mask check), a candidate population
    # genuinely disjoint from the mask check's own (`border_touch`/`qc.foreground_fraction`
    # vs `target_p90` -- different data, different defect signature; NOT literally
    # redundant with the mask check's own mask-correctness question) -- but the CODING_PROMPT's
    # own architecture diagram draws the mask check as the SOLE reject step, and the coordinator
    # confirmed that is the intended default: OFF by default, mirroring
    # `enable_ab_channel` exactly -- RECOVERABLE (a caller wanting cut_off/empty_debris
    # coverage back sets `verifier.enable_reject_channel = True`), not deleted, since
    # its own candidate population is real and NOT covered by the mask check's own verdict
    # space (a badly truncated-at-the-edge OR literally-zero-foreground crop with a
    # BRIGHT target region would never even reach the mask check's own data screen). Gates
    # ONLY `vision_recommend_crop`'s OWN cut_off/empty_debris candidate screening
    # (`screen_reject_candidates`, below) -- its OTHER role, unconditionally writing
    # the kept-crops manifest sidecar so the mask check's OWN rejections (set on `crops`
    # in-memory BEFORE this call) actually reach the persisted, downstream-trusted
    # kept population, is NEVER gated (verified: `vision_recommend_crop` is still
    # called every round `llm_engine` is wired, `screen_reject_candidates` only
    # controls whether it ALSO looks for cut_off/empty_debris candidates of its own).
    enable_reject_channel = False
    # Task 038 montage round: grid shape for the BATCHED reject-candidate montage
    # (`crop_triage._judge_reject_montages`) -- human wants 2x2/3x3/4x4 tested; a caller
    # may override on the instance/subclass. S1 (below) reuses this SAME grid.
    triage_montage_grid = (3, 3)
    # Task 038 Round B, S1's OWN VoI cap (coordinator-directed: "cap the candidate
    # count per image ... state the cap") -- a SEPARATE knob from `triage_sample_cap`
    # (that one guards the REJECT channel's cost, a different candidate volume/false-
    # positive rate) rather than silently sharing it. 27 = exactly 3 full 3x3 montages
    # (`triage_montage_grid`'s own default) -- a reasoned, round per-image budget, not
    # a data-fit number; `cell_00994` alone has 323 kept crops, so an unbounded S1
    # screen (loose by design, ~17% of the population flagged on the real v15 data
    # under the ORIGINAL p90-only rule -- see the module comment above `_GATE_DIM_
    # FRAC` for the current, wider three-arm gate, task 058 M4) could otherwise cost
    # dozens of montage calls on ONE image. Task 038 Round D (v17-measured):
    # `adjudicate` now applies this cap to candidates SORTED by suspicion (`_gate_
    # sort_key`, `_s1_target_p90` ascending) first, not crop-iteration order -- a real v17 run showed the
    # unsorted cap discarding `cell_00994/0068` (target_p90=54.9, the single MOST
    # suspicious of the human's own 5 named crops) before it was ever judged; the
    # cap number itself is UNCHANGED this round (the coordinator's own explicit
    # instruction -- the right cap depends on the montage-size/efficacy probe's own
    # result, a separate experiment).
    s1_candidate_cap = 27

    # config-declared gates (soft priors), not inline literals in localize()
    _TRUNCATION_MAX = 0.05
    # `_OFF_CENTER_MAX` (task 038 restructure): the SAME per-crop `centroid_offset` floor
    # drives BOTH the localize() off_center note-only trigger above AND the data-driven
    # geometry `recenter` trigger in adjudicate() below -- one named constant, two
    # consumers, deliberately kept in sync (CARD.md SS3/SS5).
    _OFF_CENTER_MAX = 0.20
    # `_MASK_BBOX_CONTAMINATION_MIN` (task 038 v10, human-locked, CARD.md SS1/SS3/SS5):
    # the PURE-DATA `mask_bbox` geometry trigger -- a cell whose `neighbor_contamination`
    # exceeds this fires `margin_mode="mask_bbox"` directly, no VLM confirmation (REPLACES
    # the v9 `n_complete_cells>=2`-gated, VLM-confirmed doublet trigger: a real v10 QC
    # audit found that trigger missed most real contamination -- only 12.5% of crops got
    # isolated despite 79% carrying `neighbor_contamination>0.2`, because most
    # contamination is a PARTIAL neighbour clipped at the crop edge, `n_complete_cells=1`,
    # invisible to a "genuinely WHOLE second cell" trigger). Safe as pure data: mask_bbox
    # (the v10 rectangle tight to the target's own bbox; since the human rule of
    # 2026-09-30 the square around it, cut on its neighbour sides by `auto_tighten`)
    # excludes a neighbour regardless of whether it is whole or partial, and keeps the
    # target (at least the 0.90 retention floor), so no VLM judgment is needed to apply it.
    _MASK_BBOX_CONTAMINATION_MIN = 0.2
    # Task 038 v12 (human directive -- DELIBERATELY REVERSES the "geometry is 100%
    # pure-data, no VLM" lock this SAME class's own comments used to state as settled;
    # CARD.md SS1/SS5 record the updated rationale): a mask_bbox'd cell that is STILL
    # contaminated keeps getting a DIRECTIONAL `bbox_inset_px` grown, round after round
    # (`adjudicate`'s own round-2+ pass, below), instead of converging after the ONE-SHOT
    # mask_bbox retune. `_TIGHTEN_CLEAN_MAX` is NOT a separate constant -- the brief's
    # own "propose 0.2, same as the trigger" -- `adjudicate` reuses
    # `_MASK_BBOX_CONTAMINATION_MIN` directly for the "is this cell clean enough to
    # freeze" check too (one constant, two roles, deliberately kept in sync rather than
    # duplicated).
    #
    # `_TARGET_RETENTION_FLOOR`: the SAME hard floor `tool.py`'s own
    # `_TARGET_RETENTION_FLOOR` uses as its defensive backstop -- imported from `tool.py`,
    # the single source of the value (2026-10-01; it was a duplicated copy, the agent layer
    # may import the tool). Task 038 v12 R2 (human-directed): kept here for DOCUMENTATION/consistency
    # with `tool.py`'s own value -- `adjudicate`'s round-2+ pass no longer independently
    # re-projects or re-checks this floor itself; `tool.py::_compute_crop_qc`'s own
    # `contamination_inset_px` is ALREADY floor-capped by construction (the neighbour-
    # aware data cut caps its own depth so `target_retention` never drops below this
    # SAME value), so a `contamination_inset_px` of 0 already means "no cutting
    # headroom left" -- the verifier reads that directly rather than re-deriving it.
    # Task 038 v13 (human decision): raised 0.85 -> 0.90, a MORE conservative floor
    # for the "safe candidate-generator" scope (see `tool.py`'s own copy for the full
    # rationale) -- ONE of TWO tuning knobs for the v13 control layer, the other being
    # `_VOI_RETENTION_MARGIN` immediately below.
    _TARGET_RETENTION_FLOOR = _TARGET_RETENTION_FLOOR
    # `_VOI_RETENTION_MARGIN` (task 038 v13, human-directed, the SECOND tuning knob):
    # the VoI (Value-of-Information) A/B trigger is expressed RELATIVE to the floor,
    # not as a separate absolute number -- a cut whose `target_retention` has landed
    # within this margin ABOVE the floor is "close enough to the edge" to be genuinely
    # uncertain, so it gets a blinded A/B check; a cut comfortably above this band is
    # accepted on data alone. `_TARGET_RETENTION_FLOOR + _VOI_RETENTION_MARGIN` = 0.95
    # at the current floor (0.90) -- this band TRACKS the floor (if the floor moves,
    # the VoI trigger moves with it) rather than collapsing to a fixed, narrow window
    # that could drift out of sync with the floor over future tuning rounds.
    _VOI_RETENTION_MARGIN = 0.05

    def quantify(self, tool_output, context) -> dict[str, float]:
        # stashed for the optional vision-triage Recommend below (adjudicate has no direct
        # access to tool_output, only to metrics/context -- mirrors overlay_path's role in
        # the segmentation verifier's context)
        context["crops"] = tool_output
        return crop_qc(tool_output)

    def localize(self, metrics, context):
        # Task 038 follow-up (human directive): the CONTINUOUS `geometry_headroom`
        # trigger (task 038 follow-up, item 3) is REMOVED -- it fired a `margin_frac up`
        # (widening) retune to fix low `target_dominance`, which is BACKWARDS (widening
        # ADDS background, lowering dominance -- a real run showed it widen
        # margin_frac 0.0->0.05, worsen the score 0.8366->0.8246, then revert).
        #
        # Task 038 restructure: geometry is now a per-crop, purely DATA-DRIVEN decision
        # made in `adjudicate` below (reads each crop's own `qc` block directly, no
        # judge involved at all) -- `localize` has nothing further to compute here; only
        # the two binary local-defect (note-only) triggers remain.
        triggered = []
        if metrics["truncation_fraction"] > self._TRUNCATION_MAX:
            triggered.append("truncation")
        if metrics["off_center_fraction"] > self._OFF_CENTER_MAX:
            triggered.append("off_center")
        return "this_step", triggered, []

    def _pareto_scalar(self, crops: list[dict], vetoed_this_round: set) -> float:
        """Task 038 v13 (human-directed): `scalar_score` MADE PARETO-CONSISTENT --
        REPLACES `mean_crop_quality_score_proxy` (the `_WEIGHTS`-composite), which is
        DISQUALIFIED as an acceptance authority (a methodology critique: it was
        reweighted toward the conclusion, AND even un-tuned, mask geometry is not the
        task objective -- using the SAME score to both SELECT a tightened crop and
        PROVE it better is the 048 self-proving trap this replaces).

        Exact formula (deliberately INTERPRETABLE, NOT a re-weighted multi-axis blend,
        per the brief): the population MEAN, over every crop, of a per-crop objective
        that is
          - a SENTINEL-BAD constant (-1.0) if this crop is CURRENTLY BEING VETOED this
            round (`cell_id in vetoed_this_round` -- the A/B judge just preferred its
            baseline/abstained, about to be reverted on the NEXT tool call) OR its
            OWN measured `mask_containment` (== `target_retention`) is below the hard
            floor (a defensive check -- the tool guarantees this never actually
            happens, `_TARGET_RETENTION_FLOOR`'s own docstring);
          - `1.0 - neighbor_contamination` (== `target_dominance`) otherwise -- LOWER
            for a naive/baseline/just-reverted crop, HIGHER for a crop with an
            ACCEPTED, genuinely contamination-reducing cut applied.

        WHY this is Pareto-consistent (closes the self-proving gap without touching
        `step_verifier.py` -- layering): `_best` (the SHARED keep-best, `step_verifier.
        py::StepVerifier._best`) picks `min(history, key=-scalar_score)` for this
        tool's own UNJUDGED trials (the cropper's geometry trials never set
        `judge_severity`) -- i.e. the trial with the HIGHEST scalar wins. A round
        where a cut is ABOUT TO BE VETOED scores that cell SENTINEL-BAD (-1.0) in
        THIS SAME round (the veto decision is already known here, even though the
        revert itself only takes effect on the NEXT tool call) -- so that round can
        NEVER outrank a LATER round where the SAME cell sits back at its own (higher,
        un-vetoed) baseline dominance. A vetoed/floor-violating state therefore always
        ranks WORSE than the accepted state, exactly the brief's own requirement."""
        if not crops:
            return 0.0
        _SENTINEL_BAD = -1.0
        vals = []
        for c in crops:
            cid = c.get("cell_id")
            retention = c.get("mask_containment")
            floor_violated = retention is not None and float(retention) < self._TARGET_RETENTION_FLOOR
            if (cid in vetoed_this_round) or floor_violated:
                vals.append(_SENTINEL_BAD)
            else:
                # `target_dominance` when explicitly present, else its own complement
                # -- the SAME fallback expression `crop_qc()` already uses to build its
                # own `target_dominance` array, so a fixture that only sets ONE of the
                # two fields (either is valid, this codebase's established convention)
                # still gets a meaningful, non-tied objective either way.
                vals.append(float(c.get("target_dominance",
                                        1.0 - float(c.get("neighbor_contamination", 0.0)))))
        return float(np.mean(vals))

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        # Task 038 v13 (human-directed): `scalar_score` is set at the VERY END of this
        # method (`self._pareto_scalar`, above) -- it needs to know THIS round's own
        # A/B veto decisions, which are only known after the round-2+ pass below runs.
        # `mean_crop_quality_score_proxy` (still computed in `metrics`, informational
        # only) is DISQUALIFIED as the acceptance authority (see `_pareto_scalar`'s own
        # docstring) -- it is placeholder-0.0 here, overwritten unconditionally below,
        # on EVERY exit path (including after (B) REJECT reassigns `det` entirely).
        # `rate` (valid_crop_rate, below) still drives the PASS/WARNING/FAIL decision
        # -- UNCHANGED by this v13 round.
        rate = metrics["valid_crop_rate"]
        if not triggered:
            det = VLRRResult("PASS", 0.0, {}, action="accept")
        else:
            # Task 038 follow-up (human directive, finishing the "geometry is one-shot
            # deterministic, no widening retune" cleanup): `truncation` is now NOTE-ONLY,
            # mirroring `off_center` below -- the ORIGINAL `padding up` proposal here
            # predates the tight-mode default (task 038 quality-geometry round) and has
            # been INERT ever since: `effective_margin`'s "tight" branch never reads its
            # `margin` argument (half_side comes from the bbox/centroid geometry, not a
            # padding floor) -- `padding` is now literally 0-cost to retune in tight
            # mode, so proposing it achieves nothing, exactly the kind of dead lever this
            # directive is cleaning up. `expect`/`why` are still reported (useful QC
            # notes for a human/the trace), just no `param_delta` any more.
            expect, why = {}, ""
            if "truncation" in triggered:
                expect["mean_containment"] = "up"
                why = ("Target cells are truncated. padding is a live lever ONLY for a "
                      "cell still on the naive fraction-mode starting margin; a cell "
                      "already moved to tight+recenter geometry (the per-crop data-driven "
                      "override below) ignores it -- no scalar param proposed here "
                      "either way, geometry is a per-cell, data-driven decision (part A "
                      "below), not this note.")
            elif "off_center" in triggered:
                expect["mean_centroid_offset"] = "down"
                why = "Recenter crops on the mask centroid (no scalar param)."

            issue_text = {
                "truncation": "target cell touches/exceeds the crop border (incomplete cell)",
                "off_center": "target cell is off-center in the crop",
            }
            feedback = "; ".join(issue_text.get(t, t) for t in triggered)
            notes = f"crop QC: {feedback}. {why}".strip() if feedback else why

            decision = "FAIL" if rate <= 0.80 else "WARNING"
            det = VLRRResult(decision, 0.0, {}, action="accept",
                             param_delta=None, expected_effect=expect, notes=notes)

        crops = context.get("crops") or []

        # (A) GEOMETRY: task 038 v12/v13 -- a GENUINELY ITERATIVE loop. ROUND 1
        # (unchanged since v10, pure data, no judge, decided the SAME way regardless
        # of whether `llm_engine` is wired) reads each NOT-yet-mask_bbox crop's own
        # `neighbor_contamination`/`centroid_offset` directly:
        # `neighbor_contamination > self._MASK_BBOX_CONTAMINATION_MIN` (0.2) ->
        # `{"margin_mode": "mask_bbox"}` (mask_bbox is inherently centered, SUBSUMES
        # `recenter`, so a contamination-triggered cell SKIPS the off-center check);
        # otherwise `centroid_offset > self._OFF_CENTER_MAX` -> `{"recenter": True}`.
        #
        # ROUND 2+ (task 038 architectural FIX round, coordinator-directed -- REPLACES
        # v13 FIX #2's own per-round incremental-growth design; CARD.md SS1/SS2/SS5
        # record the full rationale): the round-1 `auto_tighten: True` commit (below)
        # now makes `tool.py::_auto_tighten_mask_bbox` ITERATE the directional inset
        # lever INSIDE the tool, to convergence, in that SAME call -- carrying v13 FIX
        # #2's own plateau/regression semantics with it (duplicated into `tool.py`,
        # CARD.md is the single source of truth). Real evidence this fixes: a v13 FIX
        # #2 headless re-run showed CORRECT convergence but ZERO efficiency gain
        # (`wasted_tool_call_rate` unchanged 0.455->0.455, byte-identical) -- because
        # the per-round VLRR loop applies ONE cut per cell per ROUND while
        # re-rendering the WHOLE image every round, so the round count is driven by
        # the SLOWEST-converging cell; freezing an individual plateaued cell cannot
        # reduce that. Moving the iteration INSIDE one `execute()` call fixes the
        # structural cause directly. The verifier's OWN round-2+ job shrinks to
        # exactly what v13 always intended it to be -- the VoI screen + the blinded
        # A/B gate + Pareto/constraint acceptance, operating on the FINAL,
        # already-tightened result, NOT a per-round incremental-growth proposal any
        # more: a cell ALREADY on `margin_mode="mask_bbox"` this iteration that has
        # not yet been SETTLED (`tighten_frozen`, below) is classified by its OWN
        # current, already-measured (already fully auto-tightened) state:
        #   - "risky" -- genuinely UNCERTAIN from data alone -- when EITHER its
        #     `target_retention` has landed within `_VOI_RETENTION_MARGIN` of the
        #     floor (the internal tighten loop's own cut ATE INTO the target) OR it is
        #     STUCK at the floor (`contamination_inset_px<=0`) while STILL
        #     contaminated (the internal loop ran out of headroom before reaching
        #     clean) -- ONLY these VoI-screened cells go to the blinded A/B gate below
        #     (§(A cont'd)); a cell risky but with NO `baseline_crop_path` yet
        #     (`tool.py` only renders one when SOME inset was actually applied) falls
        #     through to "nothing further to propose" instead (never A/B'd).
        #   - otherwise ("clean", or still residually contaminated but not VoI-risky,
        #     e.g. plateaued by data alone inside the tool) -- FREEZE, converged, no
        #     VLM, SILENTLY (no override change at all -- `auto_tighten` already did
        #     everything a data-only pass can for this cell in round 1; no VLM call
        #     was ever at risk here, so there is nothing to remember; re-checking this
        #     cheaply again next round, if the image is still iterating for OTHER
        #     cells' sake, is harmless).
        # `tighten_frozen` is set (a genuine, one-time override change) ONLY AFTER an
        # ACTUAL A/B decision leaves nothing further to propose (below) -- THAT is the
        # one place a real VLM call is at risk of being wastefully repeated on an
        # unchanged state, so THAT is the one place worth remembering (matches this
        # task's own established "an explicit freeze costs one real tool call, only
        # worth it when it prevents something more expensive" finding from FIX #2).
        carried_overrides = dict(context.get("cell_overrides") or {})
        merged_overrides = dict(carried_overrides)
        changed = False
        # FIX round (coordinator-directed): track WHICH kind of geometry change was
        # actually committed this round, so the `expected_effect` prediction below
        # matches what the cut really does (contamination-driven cuts move
        # `mean_dominance`, NOT `mean_centroid_offset` -- a `recenter` moves the
        # latter). Set ONLY at the sites that commit a real, checkable geometry
        # change (mask_bbox trigger / recenter / bbox_inset_px growth) -- NOT at an
        # A/B-gate revert-to-baseline or a bookkeeping-only freeze, neither of which
        # predicts a metric improvement.
        contamination_driven_changed = False
        recenter_driven_changed = False
        voi_margin = self._TARGET_RETENTION_FLOOR + self._VOI_RETENTION_MARGIN
        ab_candidates: list[dict] = []
        for c in crops:
            cid = c.get("cell_id")
            if not cid:
                continue
            contamination = c.get("neighbor_contamination")

            if c.get("margin_mode") != "mask_bbox":
                # ROUND 1: contamination-driven mask_bbox trigger, else off-center
                # recenter -- KEY-MERGE onto the carried entry (QC fix, task 038 v7
                # restructure round) -- ONLY ever ADD a key, never rebuild from scratch
                # (monotonic accumulation, never a regression).
                mask_bbox_trigger = (contamination is not None
                                     and float(contamination) > self._MASK_BBOX_CONTAMINATION_MIN)
                if mask_bbox_trigger:
                    prior = merged_overrides.get(cid) or {}
                    new_override = dict(prior)
                    new_override["margin_mode"] = "mask_bbox"
                    # task 038 architectural FIX round (coordinator-directed):
                    # `auto_tighten: True` -- makes `tool.py` iterate the directional
                    # inset lever internally, to convergence, in THIS SAME retune
                    # (round 1 and the tightening are now ONE tool call, not one
                    # `mask_bbox` call followed by N separate per-round inset calls).
                    new_override["auto_tighten"] = True
                    if new_override != prior:
                        changed = True
                        contamination_driven_changed = True
                    merged_overrides[cid] = new_override
                    continue
                centroid_offset = c.get("centroid_offset")
                off_center = centroid_offset is not None and float(centroid_offset) > self._OFF_CENTER_MAX
                if not off_center:
                    continue
                prior = merged_overrides.get(cid) or {}
                new_override = dict(prior)
                new_override["recenter"] = True
                if new_override != prior:
                    changed = True
                    recenter_driven_changed = True
                merged_overrides[cid] = new_override
                continue

            # ROUND 2+ (task 038 architectural FIX round): a settled cell (already
            # accepted-with-residual, reverted+vetoed, or cleanly converged) is never
            # revisited.
            prior = merged_overrides.get(cid) or {}
            if prior.get("tighten_frozen"):
                continue

            qc = c.get("qc") or {}
            retention = c.get("mask_containment")
            inset_px = int(qc.get("contamination_inset_px") or 0)
            is_clean = contamination is None or float(contamination) <= self._MASK_BBOX_CONTAMINATION_MIN
            is_low_retention = retention is not None and float(retention) <= voi_margin
            is_stuck = (not is_clean) and inset_px <= 0
            is_risky = (is_low_retention or is_stuck) and bool(c.get("baseline_crop_path"))

            if is_risky:
                ab_candidates.append(c)
                continue
            # Clean, OR still residually contaminated but genuinely nothing further a
            # data-only pass could do about it (the tool's own internal `auto_tighten`
            # loop already plateaued/hit the floor/converged for this cell in round 1)
            # -- either way, nothing to propose here. SILENT (no override change): no
            # VLM call was ever at risk here, so there is nothing to remember/prevent
            # -- re-evaluating this cheap check again next round, if the image is
            # still iterating for OTHER cells' sake, costs nothing.
            continue

        # (A cont'd) THE BLINDED A/B GATE, VoI-GATED (task 038 v13, human-directed --
        # RETIRES v12 R2's own absolute-QC tighten gate; CARD.md SS1/SS2/SS5 record
        # the full rationale) -- task 058 M3 (human-authorized): OFF BY DEFAULT
        # (`enable_ab_channel`, above), 038's own M2 measurement found it harmful and
        # expensive. `ab_candidates` above are ONLY the genuinely UNCERTAIN cuts
        # (§(A)'s own VoI screen) -- a comfortable, clean cut never reaches here at
        # all. Opt-in, only when BOTH `llm_engine` is wired AND `enable_ab_channel` is
        # explicitly turned on -- BOTH the tightened candidate AND its own plain
        # mask_bbox baseline (`tool.py`'s own render, no history threading) are sent,
        # side by side, L/R randomized. Sets `crop["ab_verdict"]` IN PLACE on each
        # judged candidate ("candidate"/"baseline"/"abstain", already de-randomized);
        # a HEADLESS run, a DISABLED channel, or a risky cell somehow
        # judged-but-not-set never sets it, so every candidate below falls through to
        # the DATA-only accept path (CARD.md: "headless -> pure data, accept every
        # constraint-satisfying cut") -- the SAME path this channel's own veto/revert
        # loop already used for a headless run, so disabling the channel needed no
        # change to that loop at all.
        if (ab_candidates and getattr(self, "llm_engine", None) is not None
                and getattr(self, "enable_ab_channel", False)):
            from ..crop_triage import ab_recommend_crop
            ab_recommend_crop(
                self.llm_engine, ab_candidates,
                sample_cap=getattr(self, "triage_sample_cap", None),
                metadata_path=context.get("metadata_path"),
                montage_grid=getattr(self, "triage_montage_grid", (3, 3)))

        vetoed_this_round: set = set()
        for c in ab_candidates:
            cid = c.get("cell_id")
            verdict = c.get("ab_verdict")   # None when headless/unjudged -- data-only accept
            prior = merged_overrides.get(cid) or {}

            if verdict in ("baseline", "abstain"):
                # No clear winner (or the judge preferred the plain baseline) -- REVERT
                # to the plain mask_bbox baseline and FREEZE (never re-propose the same
                # cut, per the "never discard, KEEP the safer state" mandate -- the
                # crop is KEPT, just at its own baseline geometry, not discarded from
                # the output). Task 038 architectural FIX round: the revert is now
                # `auto_tighten: False` (the tool's own internal loop computed the
                # accumulated inset; there is no externally-tracked `bbox_inset_px` to
                # clear any more -- `tool.py` renders the plain, zero-inset box
                # whenever `auto_tighten` is falsy, exactly the "back to baseline"
                # semantics this needs).
                vetoed_this_round.add(cid)
                new_override = dict(prior)
                new_override["auto_tighten"] = False
                new_override["tighten_frozen"] = True
                if new_override != prior:
                    changed = True
                merged_overrides[cid] = new_override
                continue

            # verdict == "candidate" (a REAL, judged accept): the tool's own
            # `auto_tighten` result IS the accepted geometry -- freeze so this cell is
            # never re-judged again (the ONE place a real VLM call is at risk of being
            # wastefully repeated on an unchanged state, so THAT is the one place worth
            # remembering). verdict is None (task 038 FIX round: headless, or a risky
            # cell VoI-screened but never actually judged because no engine is wired)
            # commits NOTHING -- SILENT. There is no "next cut" for `auto_tighten` to
            # propose (it already iterated to convergence/floor/plateau internally in
            # round 1) and no VLM call to protect in a headless run, so committing a
            # freeze-only marker here buys nothing and costs one real tool call for
            # nothing (a real headless run showed this: trial1 kept returning
            # `retune_here` for exactly this reason, one extra, pointless tool call --
            # the SAME "silent unless it prevents something more expensive" reasoning
            # already applied to clean/stuck cells, task 038 FIX round #2).
            if verdict == "candidate" and not prior.get("tighten_frozen"):
                new_override = dict(prior)
                new_override["tighten_frozen"] = True
                changed = True
                merged_overrides[cid] = new_override

        # (MASK CHECK) "IS THE MASK EVEN POINTING AT THE RIGHT CELL": task 058 M3
        # (human-authorized controller change, 2026-07-16) -- REPLACES the S1 channel
        # (`s1_recommend_crop`, kept in `crop_triage.py` for other callers/tests, no
        # longer called from here). The mask check's own richer verdict space (`cell_present`/
        # `target_region_empty`/`wrong_object`/`merged`/`abstain`) subsumes S1's, and
        # is the task 038 M2 pilot's own VALIDATED production-regime finding: on
        # imperfect (Cellpose) masks the raw-image VLM catches merged/split/false-
        # positive mask errors (73%/51% precision in the hard/medium AP50 bands) that
        # mask-derived rules (this SAME data screen included) are structurally blind
        # to (a merged cell reads `neighbor_contamination=0` -- "perfect single
        # cell" -- from the mask alone). Task 058 M4 (human-directed, 2026-07-17):
        # the DATA screen itself is now the THREE-ARM OR gate (`_gate_candidate`,
        # `s1_candidate_cap` -- name unchanged, "keep that gate") -- REPLACES the
        # old single p90-only screen (the VLM is now TRUSTED, so the bottleneck
        # became the gate sending it too little: a bright doublet/debris crop used
        # to sail through un-judged). A confirmed `target_region_empty`/
        # `wrong_object`/`merged` sets `crop["vision_reject"]` IN PLACE, the SAME
        # flag (B) REJECT below already sets, so it needs no separate "kept"
        # concept. Opt-in (same `llm_engine` hook every other channel uses); MUST
        # run BEFORE (B) below so its own `vision_reject` flags are already set by
        # the time (B)'s own `vision_recommend_crop` call writes the shared
        # kept-manifest.
        # Task 038 Round D (coordinator-directed, v17-measured; UNCHANGED by M4):
        # SORTED by suspicion (`_gate_sort_key` -- `_s1_target_p90` ASCENDING, lower
        # = more suspicious, `+inf` for a candidate flagged only by the doublet/
        # debris arm with unreadable pixels) BEFORE capping, not arbitrary crop-
        # iteration order -- a real v17 run measured the unsorted cap discarding the
        # SINGLE MOST suspicious of the human's own 5 named crops before it was ever
        # judged. The cap is applied to the SORTED list as a plain slice (not
        # `mask_check_recommend_crop`'s own evenly-spaced `sample_cap` sampling -- a
        # stride sample across a sorted list would still drop some of the MOST
        # suspicious candidates in favour of less-suspicious ones later in the
        # list, defeating the point) -- the `s1_candidate_cap` LOWEST-sort-key
        # candidates are kept, full stop.
        mask_check_summary: dict = {"verdicts": []}
        ref_p90, median_area = _gate_image_stats(crops)
        mask_check_candidates = sorted(
            (c for c in crops if _gate_candidate(c, ref_p90, median_area)), key=_gate_sort_key)
        mask_check_cap = getattr(self, "s1_candidate_cap", None)
        if mask_check_cap is not None:
            mask_check_candidates = mask_check_candidates[:mask_check_cap]
        if mask_check_candidates and getattr(self, "llm_engine", None) is not None:
            from ..crop_triage import mask_check_recommend_crop
            # persist mask-check verdicts across VLRR iterations (this verifier instance is
            # reused by StepVerifier.resolve_stage) so a geometry-invariant verdict
            # (cell_present/wrong_object/target_region_empty) is judged once, never
            # re-sent to the VLM in a later round.
            if getattr(self, "_mask_check_verdict_cache", None) is None:
                self._mask_check_verdict_cache = {}
            mask_check_summary = mask_check_recommend_crop(
                self.llm_engine, mask_check_candidates,
                metadata_path=context.get("metadata_path"),
                montage_grid=getattr(self, "triage_montage_grid", (3, 3)),
                verdict_cache=self._mask_check_verdict_cache)

        # Task 058 M3 round 2 (coordinator-directed fix): the mask check's OWN confirmed
        # rejects now TIGHTEN `det` by one severity notch, the SAME contract (B)
        # REJECT always had (`crop_triage.tighten_for_confirmed_rejects`, extracted
        # from `vision_recommend_crop`'s own tail so both channels share ONE
        # implementation). This is a genuine GAP-FIX, not just parity: before this
        # fix, the mask check's (and S1's, before it) own rejects NEVER tightened `det` at
        # all -- the escalation lived ENTIRELY inside `vision_recommend_crop`'s own
        # early-return, gated on ITS OWN local `rejected` count, blind to what any
        # OTHER channel had already excluded. Now that the mask check is the PRODUCTION
        # DEFAULT reject channel (below, (B) REJECT is off by default), leaving this
        # ungated would have meant NO channel ever tightened severity by default at
        # all -- verified against the coordinator's own explicit "do not let
        # removing it loosen any gate" requirement.
        from ..crop_triage import tighten_for_confirmed_rejects
        det = tighten_for_confirmed_rejects(det, crops, mask_check_summary.get("rejected", 0),
                                            mask_check_summary.get("judged", 0), tag="mask-check")

        # (B) REJECT: the mask data SCREENS which crops are even worth asking the judge
        # about (cut_off UNION empty_debris, CARD.md SS2/SS3), and the judge is the
        # AUTHORITY on a screened candidate -- confirms the flagged defect -> reject;
        # disagrees -> keep; no second-guessing pass after it answers (see
        # `crop_triage.vision_recommend_crop`). Grounded on a mask overlay marking the
        # target (`crop_triage._crop_overlay_array`) -- the fix that lets the judge
        # reliably tell target from neighbour instead of hallucinating.
        #
        # Task 058 M3 round 2 (coordinator-directed fix, verified not assumed --
        # `enable_reject_channel`'s own class-attr comment above has the full
        # investigation): this channel's OWN cut_off/empty_debris candidate
        # screening is now OFF BY DEFAULT (`screen_reject_candidates=False` unless
        # `enable_reject_channel=True`) -- the mask check is the SOLE default VLM reject
        # channel (one montage pass per round, not two). `vision_recommend_crop`
        # ITSELF is still called every round `llm_engine` is wired, REGARDLESS of
        # this flag -- it is NOT gated behind `enable_ab_channel`-style opt-in as a
        # WHOLE call, because its OTHER role (`_rewrite_kept_manifest`, run
        # unconditionally inside it) is how the mask check's OWN rejections (set on `crops`
        # in-memory just above) actually reach the persisted kept-crops manifest
        # sidecar every downstream reader trusts -- gating the WHOLE call off would
        # have silently made the mask check's own rejections have NO effect on the delivered
        # population at all. Fully INDEPENDENT of the A/B gate above (a SEPARATE VLM
        # channel, its own candidate set, its own verdict schema, recoverable via
        # its own flag).
        if getattr(self, "llm_engine", None) is not None:
            from ..crop_triage import vision_recommend_crop
            # persist the triage verdict cache across VLRR iterations (this verifier
            # instance is reused by StepVerifier.resolve_stage) so a geometry-invariant
            # reject is judged once, never re-sent to the VLM in a later round.
            if getattr(self, "_triage_verdict_cache", None) is None:
                self._triage_verdict_cache = {}
            det = vision_recommend_crop(self.llm_engine, crops, det,
                                        sample_cap=getattr(self, "triage_sample_cap", None),
                                        metadata_path=context.get("metadata_path"),
                                        montage_grid=getattr(self, "triage_montage_grid", (3, 3)),
                                        screen_reject_candidates=getattr(
                                            self, "enable_reject_channel", False),
                                        verdict_cache=self._triage_verdict_cache)

            # Task 058 M3 (human directive -- "the process must be shown clearly", a CLEAR per-crop
            # decision trace, first-class not an afterthought): ONE legible record
            # per crop -- {cell_id, geometry, data_screen, mask_check, final} (PLUS
            # legacy-flat compatibility keys, see `_build_crop_decision_trace`'s own
            # COMPATIBILITY section) -- built AFTER every channel above (geometry /
            # the mask check / (B) REJECT) has already run this round, so `final.kept`/
            # `final.reason` reflect each crop's own ACTUAL current state (not a
            # snapshot from mid-round). Overwrites the SAME `*_triage_trace.json`
            # sidecar `vision_recommend_crop` just wrote (extending it -- CARD.md's
            # own sidecar-naming idiom, no new file format) with this richer,
            # per-crop-complete version. Nested INSIDE the SAME `llm_engine is not
            # None` guard `vision_recommend_crop` itself uses -- a headless run must
            # write NO trace at all (`benchmarks/crop_NeurIPS22-CellSeg/
            # run_shape_crop.py`'s own established, tested "no engine -> crop_
            # verdicts == []" contract, verified against a real cross-benchmark test
            # run, not assumed).
            if context.get("metadata_path"):
                from .. import llm_recommend
                mask_check_montage_by_item = {v["item_id"]: v.get("montage_path")
                                         for v in mask_check_summary.get("verdicts", [])}
                trace = _build_crop_decision_trace(crops, ref_p90, median_area,
                                                   mask_check_montage_by_item)
                llm_recommend.write_triage_trace(trace, context["metadata_path"])

        # `param_delta` is only proposed when something GENUINELY NEW changed this
        # iteration (the single, combined geometry pass above) -- an unchanged
        # accumulated state converges to `action="accept"` (StepVerifier's own
        # "satisfied" path) instead of relying on the oscillation-stop safety net.
        # `context["cell_overrides"]` (from `adapt_crop`) already carries whatever was
        # applied to produce THIS result, so a PREVIOUSLY-overridden cell's own state is
        # preserved when this iteration's data no longer flags it at all (simply left
        # as-is, never visited above) -- `apply_bounded`'s categorical passthrough
        # REPLACES the whole dict on every retune, so silently dropping a past key would
        # UNDO it on the next tool call.
        if changed:
            det.param_delta = {"cell_overrides": merged_overrides}
            # FIX round (coordinator-directed, root cause: this used to unconditionally
            # claim "mean_centroid_offset down" even when the committed change was a
            # contamination-driven cut (mask_bbox / bbox_inset_px growth), which moves
            # `mean_dominance` (= 1 - neighbor_contamination), NOT centroid_offset. The
            # shared reconcile stop-gate (`step_verifier.py`) then checked the WRONG
            # metric on the NEXT round, found it hadn't moved, and conceded
            # `hypothesis_invalidated` after just one retune -- before any
            # `bbox_inset_px` cut was ever applied. REPLACES (not merges onto)
            # `det.expected_effect`: the top-of-function note-only prediction (`expect`,
            # set when "truncation"/"off_center" is triggered) is deliberately NOT
            # authoritative here (its own comment: "no scalar param proposed here
            # either way") and must not leak into a checkable claim once a geometry
            # change actually commits.
            #
            # FIX round #2 (coordinator-directed): `contamination_driven_changed` and
            # `recenter_driven_changed` used to BOTH predict independently (an image
            # with SOME contamination cells AND SOME recenter cells in the SAME round
            # predicted BOTH keys) -- a real headless run showed this conceding 2 of 3
            # images: the tighten OFF-CENTERS the contamination cells BY DESIGN
            # (directional inset on the neighbour side only) and swamps the recenter
            # cells' own improvement at the POPULATION level, so `mean_centroid_offset`
            # does NOT go down even though a genuine recenter also committed -- the
            # shared reconcile gate concedes `hypothesis_invalidated` on the one
            # predicted key that fails, even though `mean_dominance` (the OTHER
            # predicted key) genuinely did move as expected. `elif`, not two
            # independent `if`s: predict ONLY `mean_dominance` whenever ANY
            # contamination-driven cut committed this round, regardless of whether a
            # recenter ALSO committed -- `mean_centroid_offset: down` is predicted ONLY
            # when a recenter committed AND no contamination-driven cut did (reconcile
            # concedes if ANY predicted key fails, so only ever predict what is
            # reliably achievable population-wide).
            new_expect: dict[str, str] = {}
            if contamination_driven_changed:
                new_expect["mean_dominance"] = "up"
            elif recenter_driven_changed:
                new_expect["mean_centroid_offset"] = "down"
            det.expected_effect = new_expect
            det.action = "retune_here"
            geo_note = (f"[geometry] data-driven per-crop geometry -- "
                       f"{len(merged_overrides)} cell(s) now overridden.")
            det.notes = f"{det.notes} {geo_note}".strip() if det.notes else geo_note

        # Task 038 v13: the Pareto-consistent scalar, computed LAST (needs this
        # round's own A/B veto decisions, `vetoed_this_round`) and applied
        # UNCONDITIONALLY, overwriting whatever placeholder `det.scalar_score` carried
        # from construction above (including after (B) REJECT reassigned `det`
        # entirely) -- see `_pareto_scalar`'s own docstring for the exact formula.
        det.scalar_score = self._pareto_scalar(crops, vetoed_this_round)

        return det


# ---- the agentic VRR (`CropAgenticVerifier`; CARD.md "Agentic VRR") ---------------------------------------
# The loop itself is `base.AgenticVerifier`; below is only the cropper's own part: trial 0 (the image and
# the mask, measured per cell), the per-call readouts (`adapt_crop`, the mask check), the cell rules compiled into
# `cell_overrides`, the configuration key, the paired per-cell noise, the guard, the admission's per-crop
# counts (the linear loop's round-1 geometry rule), the prompt text and images, and the delivery of the
# chosen call into `tool_cache`.

THREADS_PER_CALL = 2         # threads one tool call uses: the concurrency cap = cores // THREADS_PER_CALL
NOISE_BAND = 0.005           # the smallest readout difference counted (the tool itself is deterministic)
RETENTION_FLOOR = _TARGET_RETENTION_FLOOR   # every delivered crop keeps at least this share of its target (tool.py's)
PANEL_CELLS = 9              # cells in a call's crop panel (the same cells for every call)
PROMPT_IMAGE_PX = 1024       # longest side of the input overlay and of a crop panel in a round prompt
FINAL_IMAGE_PX = 512         # longest side of each crop panel in the final-choice prompt
FINAL_PANELS_MAX = 16        # crop panels attached to the final-choice prompt at most
NN_GAP_CAP_PX = 32           # nn_gap_px is measured up to this distance
EDGE_TOLERANCE_PX = 1        # a crop box this close to the image edge is flush (`edge_limited`)
MASK_CHECK_CAP = CropVerifier.s1_candidate_cap   # crops the mask check judges on the delivered call at most
INPUT_STATUS = "INPUT"       # trial 0's record status
MARGIN_MODES = ("fraction", "tight", "mask_bbox")
# the per-cell keys a cell rule may set (tool.py `cell_overrides`): name -> (kind, bounds)
CELL_KEYS = {
    "margin_mode": ("choice", MARGIN_MODES),
    "recenter": ("bool", None),
    "auto_tighten": ("bool", None),
    "min_containment": ("number", (RETENTION_FLOOR, 1.0)),
    "reset": ("bool", None),
}
INPUT_CELL_READOUTS = ("area", "eq_diameter", "bbox_h", "bbox_w", "aspect_ratio", "solidity",
                       "bbox_other_fraction", "nn_gap_px", "touches_image_border")
CALL_CELL_READOUTS = ("neighbor_contamination", "target_dominance", "target_retention", "centroid_offset",
                      "centered", "occupancy", "border_touch", "n_complete_cells", "contamination_inset_px",
                      "crop_h", "crop_w", "crop_aspect", "margin_mode", "edge_limited")
_CATEGORICAL = {"margin_mode", "touches_image_border", "edge_limited"}
# a call-level readout -> the per-cell readout it averages (the paired noise, the vs-parent facts)
_CELL_KEY = {"mean_neighbor_contamination": "neighbor_contamination", "mean_target_dominance": "target_dominance",
             "mean_target_retention": "target_retention", "mean_centroid_offset": "centroid_offset",
             "mean_centered": "centered", "mean_occupancy": "occupancy"}
# a key readout the calls table's vs-parent facts leave out -> the kept one it is the exact complement of, per crop
# (tool.py: neighbor_contamination = other / (fg + eps), target_dominance = (fg - other) / (fg + eps), eps 1e-9)
_FACT_COMPLEMENT = {"mean_neighbor_contamination": "mean_target_dominance"}
# the linear CropVerifier's round-1 geometry policy as cell rules (only the deterministic fallback uses it)
_POLICY_RULES = [
    {"when": [{"readout": "neighbor_contamination", "op": ">", "value": "0.2"}],
     "apply": [{"key": "margin_mode", "value": '"mask_bbox"'}, {"key": "auto_tighten", "value": "true"}]},
    {"when": [{"readout": "neighbor_contamination", "op": "<=", "value": "0.2"},
              {"readout": "centroid_offset", "op": ">", "value": "0.2"}],
     "apply": [{"key": "recenter", "value": "true"}]},
]


class CellCondition(BaseModel):
    readout: str = Field(description="a per-cell readout name")
    op: Literal[">", ">=", "<", "<=", "=="]
    value: str = Field(description='a JSON literal, e.g. 0.2, 12, true, "mask_bbox"')


class CellSetting(BaseModel):
    key: Literal["margin_mode", "recenter", "auto_tighten", "min_containment", "reset"]
    value: Optional[Union[bool, float, str]] = Field(
        description='a native JSON value, e.g. "mask_bbox", true, or 0.95; legacy JSON-literal strings are accepted')

    @model_validator(mode="after")
    def validate_value_type(self):
        kind, bounds = CELL_KEYS[self.key]
        value = self.value
        if isinstance(value, str):
            text = value.strip()
            try:
                value = json.loads(text)
            except ValueError:
                value = text.strip('"').strip() if kind == "choice" else text
        if kind == "bool" and not isinstance(value, bool):
            raise ValueError(f"{self.key} must be a JSON boolean true or false")
        if kind == "choice" and value not in bounds:
            raise ValueError(f"{self.key} must be one of {list(bounds)}")
        if kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{self.key} must be a finite JSON number")
            if not bounds[0] <= float(value) <= bounds[1]:
                raise ValueError(f"{self.key} must be within [{bounds[0]}, {bounds[1]}]")
        return self


class CellRule(BaseModel):
    when: list[CellCondition] = Field(
        description="conditions that must all hold for a cell to match; none = every cell")
    apply: list[CellSetting] = Field(description="the per-cell keys set for every matching cell")


class CropMove(WaveMove):
    """One cropper call: the shared move (a parent and the image-level levers it changes) and cell rules."""
    cell_rules: list[CellRule] = Field(
        default_factory=list, description="rules applied in order on top of the parent's per-cell settings")


class CropPlan(WavePlan):
    moves: list[CropMove] = Field(description="the calls of this round; none = satisfied, stop")


@dataclass(frozen=True)
class CropTask:
    """What a caller knows about the crop stage (``upstream_ctx["crop_task"]``); every field optional."""
    cores: Optional[int] = None          # cores for the tool calls (None: the cores this process may use)
    deadline: Optional[float] = None     # epoch seconds by which the stage must end (e.g. the job's limit)
    brief: str = ""                      # a one-line task description shown to the agent
    enable_mask_check: bool = True       # false for masks the caller trusts (the mask check would reject clean objects)


def _load_label_mask(path: str):
    """The label mask the tool crops from, loaded the way ``execute`` loads it (``.npy``, then tifffile
    for ``.tif``/``.tiff``, else an 8-bit grayscale read; squeezed to 2D, uint16; a binary mask is
    labelled into connected components)."""
    import cv2
    import tifffile
    from skimage.measure import label

    lower = str(path).lower()
    if lower.endswith(".npy"):
        mask = np.load(path)
    elif lower.endswith((".tif", ".tiff")):
        try:
            mask = tifffile.imread(path)
        except Exception:  # noqa: BLE001 -- the tool falls back to cv2 the same way
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask = mask.astype(np.uint16)
    return mask if len(np.unique(mask)) > 2 else label(mask)


def _image_array(path: str):
    """The source image as (H, W, C), loaded the way the tool loads it."""
    from shape.utils.image_processor import ImageProcessor

    data = np.asarray(ImageProcessor.load_image(path).data)
    return data[:, :, np.newaxis] if data.ndim == 2 else data


def _compact(value):
    """A readout value of the calls table: rounded to 4 decimals, an integral one as an int; None stays None."""
    if value is None:
        return None
    value = round(float(value), 4)
    return int(value) if value.is_integer() else value


def _quantiles(values, qs=(10, 50, 90)) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not arr.size:
        return {}
    return {f"p{q}": round(float(np.percentile(arr, q)), 4) for q in qs}


def _safe_name(text: str, limit: int = 50) -> str:
    return "".join(c for c in str(text) if c.isalnum() or c in ("_", "-"))[:limit] or "image"


def _edge_limited(box, centroid, image_shape) -> bool:
    """Whether the image edge forces a crop's centroid offset: no square of the crop's side inside the image
    places the target's centroid closer to the crop centre. Per axis, the centroid is within
    EDGE_TOLERANCE_PX of the crop centre, or the box is flush (within EDGE_TOLERANCE_PX) against the image
    edge on the side the centroid lies toward -- the side the square would have to move to; edge-limited =
    every axis is one of the two and at least one is flush. `box` = (minr, minc, maxr, maxc), `centroid` =
    (row, col), the conventions of `tool.py::_compute_crop_qc`'s `centroid_offset`."""
    r0, c0, r1, c1 = box
    flush = False
    for lo, hi, size, pos in ((r0, r1, image_shape[0], centroid[0]), (c0, c1, image_shape[1], centroid[1])):
        d = float(pos) - (lo + hi) / 2.0
        if abs(d) <= EDGE_TOLERANCE_PX:
            continue
        if (d < 0 and lo > EDGE_TOLERANCE_PX) or (d > 0 and hi < size - EDGE_TOLERANCE_PX):
            return False
        flush = True
    return flush


def _largest_square_retention(labeled: np.ndarray, region) -> float:
    """The largest share of an object's mask the tool can keep in a crop when its bbox is longer than the
    image's shorter side: the largest square that fits the image, placed the way the tool places a square
    that cannot hold the bbox (`compute_square_crop_containing_bbox`), centred on the bbox midpoint (float,
    as `mask_bbox` centres it; integer, as `fraction` / `tight` do) or on the centroid (`recenter`) -- the
    best of the three (a smaller square at the same centre lies inside the largest one, up to a pixel of
    rounding)."""
    from shape.tools.single_cell_cropper.tool import compute_square_crop_containing_bbox

    height, width = labeled.shape[:2]
    minr, minc, maxr, maxc = region.bbox
    side = min(height, width)
    best = 0.0
    for center in (((minr + maxr) / 2.0, (minc + maxc) / 2.0), ((minr + maxr) // 2, (minc + maxc) // 2),
                   tuple(region.centroid)):
        r0, c0, r1, c1 = compute_square_crop_containing_bbox(minr, minc, maxr, maxc, *center, side / 2.0,
                                                             height, width)
        best = max(best, float(np.count_nonzero(labeled[r0:r1, c0:c1] == region.label)) / float(region.area))
    return best


class _CellBoxes(Single_Cell_Cropper_Tool):
    """The tool's own per-cell geometry (`cell_crop_box`) on one input's label mask, for the configuration key --
    computed like `_generate_cell_crops` computes it: the label areas counted once (`areas`, passed as
    `full_field_areas`) and the target's full-field mask on one shared array (the tool's `_show_target`); and
    `auto_tighten`'s result kept per cell and plain square (it depends on nothing else, so a configuration that
    changes other cells or levers does not run it again)."""

    def __init__(self, labeled: np.ndarray):
        super().__init__()
        self.areas = np.bincount(labeled.ravel())
        self._target = np.zeros(labeled.shape[:2], dtype=np.uint8)
        self._tightened: dict[tuple, tuple] = {}

    def _auto_tighten_mask_bbox(self, labeled_mask, region, region_label, cell_mask, *square_and_min_area,
                                **kwargs):
        key = (int(region_label), *square_and_min_area)
        if key not in self._tightened:
            self._show_target(self._target, region, 255)
            try:
                self._tightened[key] = super()._auto_tighten_mask_bbox(labeled_mask, region, region_label,
                                                                       self._target, *square_and_min_area, **kwargs)
            finally:
                self._show_target(self._target, region, 0)
        return self._tightened[key]


def _cell_measurements(labeled: np.ndarray, min_area: int, group: str, image_name: str) -> tuple[dict, int, int]:
    """Per-cell input measurements of every object at min_area, keyed by the tool's own cell_id (the
    regionprops order ``_generate_cell_crops`` enumerates), plus the object count and the count of
    objects no square crop can contain at RETENTION_FLOOR (per cell: `uncroppable` -- the bbox is longer
    than the image's shorter side and even the tool's best placement of the largest square that fits keeps
    less than RETENTION_FLOOR of the object, `_largest_square_retention`; the guard exempts them)."""
    from scipy import ndimage
    from skimage.measure import regionprops

    height, width = labeled.shape[:2]
    regions = regionprops(labeled)
    cells: dict[str, dict] = {}
    n_uncroppable = 0
    for idx, region in enumerate(regions):
        if region.area < min_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        h, w = maxr - minr, maxc - minc
        uncroppable = bool(max(h, w) > min(height, width)
                           and _largest_square_retention(labeled, region) < RETENTION_FLOOR)
        n_uncroppable += uncroppable
        sub = labeled[minr:maxr, minc:maxc]
        fg = int(np.count_nonzero(sub))
        other = int(np.count_nonzero((sub != 0) & (sub != region.label)))
        r0, r1 = max(0, minr - NN_GAP_CAP_PX - 1), min(height, maxr + NN_GAP_CAP_PX + 1)
        c0, c1 = max(0, minc - NN_GAP_CAP_PX - 1), min(width, maxc + NN_GAP_CAP_PX + 1)
        window = labeled[r0:r1, c0:c1]
        others = (window != 0) & (window != region.label)
        if others.any():
            dist = ndimage.distance_transform_edt(~others)
            gap = min(float(NN_GAP_CAP_PX), max(0.0, float(dist[window == region.label].min()) - 1.0))
        else:
            gap = float(NN_GAP_CAP_PX)
        cells[f"{group}_{image_name}_cell_{idx:04d}"] = {
            "area": int(region.area),
            "eq_diameter": round(2.0 * math.sqrt(float(region.area) / math.pi), 3),
            "bbox_h": int(h), "bbox_w": int(w),
            "aspect_ratio": round(max(h, w) / max(1, min(h, w)), 4),
            "solidity": round(float(region.solidity), 4),
            "bbox_other_fraction": round(other / fg, 4) if fg else 0.0,
            "nn_gap_px": round(gap, 3),
            "touches_image_border": bool(minr == 0 or minc == 0 or maxr == height or maxc == width),
            "uncroppable": uncroppable,
        }
    return cells, len(regions), n_uncroppable


def input_record(base_kwargs: Mapping[str, Any]) -> dict:
    """Trial 0 of the agentic crop loop: the input as a record (not a tool call) -- the image (size,
    channels, dtype, per-channel range), the mask (objects, their sizes, shapes and spacing, per cell and
    as quantiles; per cell whether no square crop can hold it at RETENTION_FLOOR, `uncroppable`), the fixed
    min_area (the caller's -- the delivering segmentation's min_size --, else the tool's default by mask file
    name, the cell segmenter's min_size), the resolved defaults the CARD leaves to the mask (margin, and
    margin_frac when the caller fixes it) and the panel cells shown for every call."""
    kw = dict(base_kwargs or {})
    image_path, mask_path = kw.get("original_image"), kw.get("nuclei_mask")
    group = kw.get("group") or "default"
    image_name = kw.get("source_image_id") or (Path(image_path).stem if image_path else "image")
    rule_area, rule_margin = tool_default_min_area_margin(mask_path)
    min_area = int(kw["min_area"]) if kw.get("min_area") is not None else rule_area
    defaults = {"margin": int(round(float(kw["margin"]))) if kw.get("margin") is not None else rule_margin}
    if kw.get("margin_frac") is not None:
        defaults["margin_frac"] = float(kw["margin_frac"])
    record: dict[str, Any] = {
        "status": INPUT_STATUS, "original_image": image_path, "nuclei_mask": mask_path,
        "source_image_id": image_name, "group": group, "image_key": _safe_name(image_name),
        "min_area": min_area, "defaults": defaults, "error": None, "cells": {}, "panel_cells": [],
        "n_objects": 0, "n_croppable": 0, "n_uncroppable": 0}
    try:
        labeled = _load_label_mask(mask_path) if mask_path else None
        if labeled is None:
            raise ValueError(f"cannot load the mask {mask_path!r}")
        record["image_shape"] = [int(s) for s in labeled.shape[:2]]
        try:
            image = _image_array(image_path) if image_path else None
        except Exception as exc:  # noqa: BLE001 -- the tool has its own fallback readers; the mask still counts
            image = None
            record["image_read_error"] = f"{type(exc).__name__}: {exc}"
        if image is not None:
            if image.shape[:2] != labeled.shape[:2]:
                raise ValueError(f"image {tuple(image.shape[:2])} and mask {tuple(labeled.shape[:2])} differ in size")
            step = max(1, int(math.sqrt(image.shape[0] * image.shape[1] / 1e6)))
            sample = np.asarray(image[::step, ::step], dtype=np.float64)
            record.update(channels=int(image.shape[2]), dtype=str(image.dtype),
                          channel_p1=[round(float(np.percentile(sample[..., c], 1)), 2) for c in range(sample.shape[2])],
                          channel_p99=[round(float(np.percentile(sample[..., c], 99)), 2) for c in range(sample.shape[2])])
        cells, n_objects, uncroppable = _cell_measurements(labeled, min_area, group, image_name)
        record.update(cells=cells, n_objects=n_objects, n_croppable=len(cells), n_uncroppable=uncroppable,
                      foreground_fraction=round(float(np.count_nonzero(labeled)) / labeled.size, 4))
        ranked = sorted(cells, key=lambda c: (cells[c]["bbox_other_fraction"], c))
        if ranked:
            picks = np.linspace(0, len(ranked) - 1, num=min(PANEL_CELLS, len(ranked)))
            record["panel_cells"] = list(dict.fromkeys(ranked[int(round(p))] for p in picks))
    except Exception as exc:  # noqa: BLE001 -- an unreadable input ends the stage (no objects)
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("crop agentic: input not measured (%s)", record["error"])
    return record


def _png(array: np.ndarray, max_px: int) -> bytes | None:
    from PIL import Image

    if array is None or array.size == 0:
        return None
    img = Image.fromarray(np.asarray(array, dtype=np.uint8))
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _input_overlay(image_path: str, mask_path: str, max_px: int) -> bytes | None:
    """The input image (per-channel display range, max over channels) with the mask's object boundaries
    in green, longest side at most `max_px`."""
    from skimage.segmentation import find_boundaries

    from shape.tools.single_cell_cropper.tool import normalize_source_image

    try:
        labeled = _load_label_mask(mask_path)
        gray = normalize_source_image(_image_array(image_path)).max(axis=2)
        rgb = np.stack([gray] * 3, axis=-1)
        rgb[find_boundaries(labeled, mode="inner")] = (0, 255, 0)
        return _png(rgb, max_px)
    except Exception as exc:  # noqa: BLE001 -- a prompt without the overlay still carries the facts
        logger.warning("crop agentic: no input overlay (%s)", exc)
        return None


class CropAgenticVerifier(AgenticVerifier):
    """Agentic VRR for Single_Cell_Cropper_Tool (``base.AgenticVerifier``, StepVerifier wave mode; CARD.md
    "Agentic VRR"). Trial 0 is the input (``input_record``: the image and the mask, measured per cell), not
    a tool call. Round 1 is the tool at its defaults (``admission_call``): a measured problem of that call
    (it failed, its guard, or its crops the linear loop's round-1 geometry rule would re-run,
    ``admission_problems``) starts the tuning; without one the agent decides whether the tuning starts. A
    call's params are the tool's own kwargs: the CARD's image-level levers, the
    ``cell_overrides`` compiled from the agent's cell rules, the fixed ``min_area`` and the call's own
    ``output_subdir`` (the crop file names are deterministic per cell, so calls must not share a
    directory). Two configurations are the same call when they crop every cell identically: the same box for
    every croppable cell, computed before the call by the tool's own per-cell geometry (``config_key``). A
    readout difference between two calls is beyond the noise when it exceeds the larger
    of NOISE_BAND and 2SE of the per-cell paired difference (the tool is deterministic)."""

    name = "crop"
    tool_dir = "single_cell_cropper"
    lever_section = "## Levers"
    threads_per_call = THREADS_PER_CALL
    score_key = "mean_target_dominance"
    key_readouts = ("mean_target_dominance", "mean_neighbor_contamination", "mean_target_retention",
                    "mean_centered", "mean_occupancy")
    objectives = ("mean_target_dominance", "mean_target_retention", "mean_centered")
    admission_call = True                # round 1 = the tool at its defaults; the admission is on that call
    plan_schema = CropPlan
    tool_phrase = "single-cell cropping tool"
    unit = "image"
    value_examples = '0.1, 8, true, null, "mask_bbox"'
    input_record = staticmethod(input_record)
    # set by the cropping cohort (`solver._cropping_cohort`, through `verifier_options`):
    cohort_levers: dict | None = None    # the cohort mode: the group's consensus levers, delivered unasked
    start_params: dict | None = None     # a routed image: its group's consensus as the tool defaults

    def __init__(self, *, log: Callable[..., None] | None = None, **kwargs):
        super().__init__(log=log, **kwargs)
        self._cells: dict[int, dict[str, dict]] = {}       # trial -> cell_id -> per-crop readouts
        self._crops: dict[int, dict[str, dict]] = {}       # trial -> cell_id -> adapt_crop crop dict
        self._unusable: dict[int, str] = {}                # trial -> why a finished call has no crops
        self._index_of: dict[int, int] = {}                # id(trial VLRRResult) -> trial index
        self._keys: dict[str, str] = {}                    # (mask, configuration) json -> config_key
        self._planner: dict | None = None                  # the input's per-cell geometry (`_box_planner`)
        self._mask_check_cache: dict = {}
        self.enable_mask_check = True                      # `CropTask.enable_mask_check`, read at the input trial
        self._triage_cache: dict = {}
        self._input_png: bytes | None = None

    # ---- Verify -------------------------------------------------------------------------
    def quantify(self, tool_output, context) -> dict[str, float]:
        """Trial 0: the input facts. A call: its crops' readouts (all crops; the mask check runs only on the
        delivered call, `deliver`). The objects without a crop are identified per cell (by cell_id, against the input
        record): `n_missing_crops` counts those a square crop can hold at RETENTION_FLOOR (the guard's
        count), `n_exempt_uncroppable` those it cannot (`uncroppable`, exempt from the guard)."""
        result = tool_output if isinstance(tool_output, Mapping) else {}
        index = int(context.get("trial_index") or 0)
        if index == 0 or result.get("status") == INPUT_STATUS:
            return numeric({k: result.get(k) for k in ("n_objects", "n_croppable", "n_uncroppable",
                                                        "foreground_fraction")})
        if result.get("error") or result.get("execution_status") == "error":
            self._unusable[index] = str(result.get("error") or "the tool reported an error")[-300:]
            return {}
        try:
            crops, ctx = adapt_crop(dict(result))
        except ValueError as exc:
            self._unusable[index] = f"no crops: {exc}"
            return {}
        metadata_path = ctx.get("metadata_path")
        with open(metadata_path, "r") as fh:
            meta = json.load(fh)
        by_id = {m.get("cell_id"): m for m in meta.get("cell_metadata") or [] if isinstance(m, dict)}
        image_shape = self.input.get("image_shape")
        cells: dict[str, dict] = {}
        for c in crops:
            qc, m = c.get("qc") or {}, by_id.get(c.get("cell_id")) or {}
            minr, minc, maxr, maxc = (m.get("crop_bbox") or [0, 0, 0, 0])[:4]
            h, w = int(maxr - minr), int(maxc - minc)
            cells[c["cell_id"]] = {
                **{k: float(qc[k]) for k in ("neighbor_contamination", "target_dominance", "target_retention",
                                             "centroid_offset", "occupancy", "border_touch") if k in qc},
                "centered": 1.0 - float(qc.get("centroid_offset", 0.0)),
                "n_complete_cells": int(qc.get("n_complete_cells") or 0),
                "contamination_inset_px": int(qc.get("contamination_inset_px") or 0),
                "crop_h": h, "crop_w": w, "crop_aspect": round(max(h, w) / max(1, min(h, w)), 4),
                "margin_mode": c.get("margin_mode"),
                "edge_limited": bool(image_shape and m.get("centroid")
                                     and _edge_limited((minr, minc, maxr, maxc), m["centroid"], image_shape)),
                "auto_tighten_iterations": int(m.get("auto_tighten_iterations") or 0),
            }
        self._cells[index] = cells
        self._crops[index] = {c["cell_id"]: c for c in crops}
        stats = result.get("processing_statistics") or meta.get("statistics") or {}
        metrics = self._readouts(cells)
        metrics["n_lost_to_geometry"] = float(int(stats.get("filtered_by_border") or 0)
                                              + int(stats.get("invalid_crop_data") or 0))
        objects = self.input.get("cells") or {}
        uncroppable = {cid for cid, cell in objects.items() if cell.get("uncroppable")}
        missing, exempt = sorted(set(objects) - uncroppable - set(cells)), sorted(uncroppable - set(cells))
        metrics["n_missing_crops"], metrics["n_exempt_uncroppable"] = float(len(missing)), float(len(exempt))
        report = {"seconds": result.get("seconds"), "metadata_path": metadata_path,
                  "missing_crop_cells": missing, "exempt_uncroppable_cells": exempt}
        self._reports[index] = report
        return metrics

    @staticmethod
    def _readouts(cells: Mapping[str, dict]) -> dict[str, float]:
        """Population readouts of one call, over all its crops."""
        def col(key):
            return np.asarray([v[key] for v in cells.values() if key in v], dtype=float)

        out: dict[str, float] = {"n_crops": float(len(cells))}
        for key in ("neighbor_contamination", "target_retention", "centroid_offset"):
            values = col(key)
            if values.size:
                out[f"mean_{key}"] = float(values.mean())
                for q in (10, 50, 90):
                    out[f"p{q}_{key}"] = float(np.percentile(values, q))
        for key in ("target_dominance", "centered", "occupancy"):
            values = col(key)
            if values.size:
                out[f"mean_{key}"] = float(values.mean())
        if "mean_target_retention" in out:
            out["min_target_retention"] = float(col("target_retention").min())
        sides = [(v["crop_h"], v["crop_w"]) for v in cells.values()]
        if sides:
            out["frac_non_square"] = float(np.mean([h != w for h, w in sides]))
            out["p90_crop_aspect"] = float(np.percentile(col("crop_aspect"), 90))
            out["median_crop_side_px"] = float(np.median([max(h, w) for h, w in sides]))
            for mode in MARGIN_MODES:
                out[f"frac_mode_{mode}"] = float(np.mean([v["margin_mode"] == mode for v in cells.values()]))
            out["mean_auto_tighten_iterations"] = float(col("auto_tighten_iterations").mean())
            out["n_edge_limited"] = float(sum(1 for v in cells.values() if v.get("edge_limited")))
        return out

    def _mask_check(self, crops: list[dict], metadata_path: str, report: dict) -> dict[str, float]:
        """The mask check (the linear CropVerifier's mask-error channel, unchanged) on the delivered call's crops:
        the three-arm data screen picks the candidates, the vision model judges up to MASK_CHECK_CAP of them
        (the verifier's verdict cache, shared with the linear loop's code path, holds the verdicts that do not
        depend on the geometry; one check per image never reuses them), confirmed errors are excluded
        through the kept-crops manifest, and the decision trace is written next to the call's metadata. Returns
        its counts. Runs only with an engine."""
        if not self.enable_mask_check or self.llm_engine is None or not crops:
            return {}
        from .. import llm_recommend
        from ..crop_triage import mask_check_recommend_crop, vision_recommend_crop

        ref_p90, median_area = _gate_image_stats(crops)
        candidates = sorted((c for c in crops if _gate_candidate(c, ref_p90, median_area)), key=_gate_sort_key)
        summary = mask_check_recommend_crop(self.llm_engine, candidates[:MASK_CHECK_CAP], metadata_path=metadata_path,
                                       montage_grid=CropVerifier.triage_montage_grid,
                                       verdict_cache=self._mask_check_cache)
        from shape.utils.schemas import QuestionFailed
        for montage in summary.get("montages", []):
            if montage.get("error"):
                self.reasoner_failures += 1
                self.question_failed(QuestionFailed(montage["error"], label="mask_check", cause=montage["error"]))
        vision_recommend_crop(self.llm_engine, crops, VLRRResult("PASS", 0.0, {}), metadata_path=metadata_path,
                              montage_grid=CropVerifier.triage_montage_grid, screen_reject_candidates=False,
                              verdict_cache=self._triage_cache)
        montages = {v["item_id"]: v.get("montage_path") for v in summary.get("verdicts", [])}
        llm_recommend.write_triage_trace(_build_crop_decision_trace(crops, ref_p90, median_area, montages),
                                         metadata_path)
        report["mask_check_rejected_cells"] = sorted(c["cell_id"] for c in crops if c.get("vision_reject"))
        report["mask_check_montages"] = summary.get("montages", [])
        report["mask_check_verdicts"] = [{key: v.get(key) for key in ("crop_id", "verdict", "confidence", "reject",
                                                                      "montage_path")}
                                         for v in summary.get("verdicts", [])]
        return {"mask_check_candidates": float(len(candidates)), "mask_check_judged": float(summary.get("judged", 0)),
                "mask_check_rejected": float(sum(1 for c in crops if c.get("vision_reject"))),
                "mask_check_reused": float(summary.get("reused", 0))}

    def localize(self, metrics, context):
        return "this_step", [], []

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        index = int(context.get("trial_index") or 0)
        if index:
            vres = VLRRResult("PASS" if metrics else "FAIL", float(metrics.get(self.score_key, float("nan"))),
                              dict(metrics), action="retune_here")
        else:
            self.begin(context.get("crop_task"))
            self.enable_mask_check = getattr(context.get("crop_task"), "enable_mask_check", True)
            n = int(metrics.get("n_croppable") or 0)
            if not n:
                self.set_admission({"decision": "ABSTAIN", "reasons": ["no object to crop"]}, source="input")
            vres = VLRRResult("PASS" if n else "FAIL", float("nan"), dict(metrics),
                              action="retune_here" if n else "conceded", fault=None if n else "ceiling",
                              notes="the input image and mask" if n else "no object to crop")
        self._index_of[id(vres)] = index
        return vres

    def _noise(self, key: str) -> Optional[float]:
        """The stated noise of a readout with a per-cell pairing (StepVerifier's ledger); a difference
        between two calls uses `noise`."""
        return NOISE_BAND if key in _CELL_KEY else None

    def _paired(self, a: int, b: int, cell_key: str) -> tuple[float, float, int] | None:
        """(mean of b - a, 2SE, cells) over the cells both calls cropped; None without a common cell."""
        ca, cb = self._cells.get(a) or {}, self._cells.get(b) or {}
        diffs = [cb[c][cell_key] - ca[c][cell_key] for c in ca.keys() & cb.keys()
                 if cell_key in ca[c] and cell_key in cb[c]]
        if not diffs:
            return None
        arr = np.asarray(diffs, dtype=float)
        se = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
        return float(arr.mean()), 2.0 * se, int(arr.size)

    def noise(self, a: int, b: int, key: str) -> float:
        """The noise of a difference in `key` between calls a and b: the larger of NOISE_BAND and 2SE of
        the per-cell paired difference."""
        paired = self._paired(a, b, _CELL_KEY.get(key, key))
        return max(NOISE_BAND, paired[1]) if paired else NOISE_BAND

    def guard(self, inp: VLRRResult, vres: VLRRResult) -> list[str]:
        """The conditions a final choice fails (empty = admissible): a crop for every object at min_area that a
        square crop can hold at RETENTION_FLOOR (the others, `uncroppable` on the input record, are exempt --
        the human decision of 2026-09-30: no call can crop them, so they cannot make every call inadmissible),
        every crop square (height = width; the human rule of 2026-09-30 -- the analyzer resizes every crop to
        224 x 224 without keeping the aspect ratio) and every crop keeping at least RETENTION_FLOOR of its
        target. Reads only the call's own readouts: `n_missing_crops` (counted per cell, so a croppable object's
        loss never hides behind an exempt one) and the non-square count, `frac_non_square` x `n_crops`."""
        m, fails = vres.metrics or {}, []
        if not m.get("n_crops"):
            fails.append("no crops")
        non_square = int(round(float(m.get("frac_non_square") or 0.0) * float(m.get("n_crops") or 0.0)))
        if non_square:
            fails.append(f"{non_square} crop(s) not square")
        low = m.get("min_target_retention")
        if low is not None and low < RETENTION_FLOOR - 1e-9:
            fails.append(f"a crop keeps {low:.3f} of its target, below {RETENTION_FLOOR}")
        missing = int(m.get("n_missing_crops") or 0)
        if missing:
            fails.append(f"{missing} object(s) at min_area got no crop")
        return fails

    @staticmethod
    def _retune_rule() -> list[tuple[str, float]]:
        """The per-crop readouts the admission counts, with their thresholds: the linear `CropVerifier`'s
        round-1 geometry rule (CARD.md section 5) re-runs a crop in mask_bbox when its neighbor_contamination is
        above `_MASK_BBOX_CONTAMINATION_MIN`, else recentres it when its centroid_offset is above
        `_OFF_CENTER_MAX` (every crop at the defaults is in fraction mode, where the rule applies)."""
        return [("neighbor_contamination", CropVerifier._MASK_BBOX_CONTAMINATION_MIN),
                ("centroid_offset", CropVerifier._OFF_CENTER_MAX)]

    def admission_problems(self, inp: VLRRResult, vres: VLRRResult) -> list[str]:
        """The call's crops (all of them, as its readouts) above each threshold of the linear loop's round-1
        rule (`_retune_rule`), counted per readout -- a crop above both counts in both -- from the `adapt_crop`
        values that rule reads; the guard's conditions are the guard's. A crop whose centroid_offset the image
        edge forces (`edge_limited`: no square of its side inside the image places the centroid closer to its
        centre) is not counted as off-centre (the human decision of 2026-09-30: no placement fixes it); the
        text names how many more such crops are above the threshold."""
        index = self._index_of.get(id(vres))
        crops = list((self._crops.get(index) or {}).values())
        cells = self._cells.get(index) or {}
        problems = []
        for key, limit in self._retune_rule():
            above = [c for c in crops if c[key] > limit]
            limited = (sum(1 for c in above if (cells.get(c.get("cell_id")) or {}).get("edge_limited"))
                       if key == "centroid_offset" else 0)
            if len(above) > limited:
                problems.append(f"{len(above) - limited} of {len(crops)} crops with {key} above {limit}"
                                + (f"; {limited} more are edge-limited" if limited else ""))
        return problems

    # ---- the loop state in the cropper's terms ----------------------------------------------
    def call_error(self, index: int, state) -> Optional[str]:
        """A failed call, or a finished call that produced no crops (a tool error result, zero crops)."""
        if index in self._unusable:
            return self._unusable[index]
        return super().call_error(index, state)

    def defaults(self) -> dict:
        """The CARD defaults, with what the CARD leaves to the mask resolved on trial 0 (margin: the
        caller's value, else the tool's rule by mask file name; margin_frac when the caller fixes it); an image the
        cohort routes to its own VRR takes its group's consensus in place of the CARD defaults (`start_params`)."""
        out = super().defaults()
        out.update({k: v for k, v in (self.input.get("defaults") or {}).items() if k in out})
        out.update({k: v for k, v in (self.start_params or {}).items() if k in out})
        return out

    def coerce(self, lever: str, value: Any) -> Any:
        """The CARD's Type / Range; a null margin is the tool's rule, resolved for this mask."""
        value = super().coerce(lever, value)
        return self.defaults()["margin"] if lever == "margin" and value is None else value

    def trial_config(self, params: Mapping[str, Any]) -> dict:
        config = {lever: params.get(lever) for lever in self.levers}
        config["cell_overrides"] = copy.deepcopy(params.get("cell_overrides") or {})
        return config

    def search_dimensions(self) -> list[str]:
        return [*super().search_dimensions(), "cell_rules"]

    def search_aliases(self) -> dict[str, str]:
        """The per-cell keys (`CELL_KEYS`) are parts of cell_rules; margin_mode and recenter alone name the
        image-level levers, so only their "per-cell" form maps to cell_rules."""
        aliases = {f"per-cell {key}": "cell_rules" for key in CELL_KEYS}
        aliases.update({key: "cell_rules" for key in CELL_KEYS if key not in self.levers})
        aliases["cell_overrides"] = "cell_rules"
        return aliases

    def search_values(self, params: Mapping[str, Any], call: Optional[str]) -> dict:
        values = super().search_values(params, call)
        overrides = params.get("cell_overrides") or {}
        cells = self.input.get("cells") or {}
        modes = {overrides.get(cid, {}).get("margin_mode", params.get("margin_mode")) for cid in cells}
        if not modes:
            modes = {params.get("margin_mode")}
        inactive = set()
        if "fraction" not in modes:
            inactive.update(("margin", "margin_frac"))
        if modes == {"mask_bbox"}:
            inactive.update(("recenter", "crop_size"))
        if modes == {"fraction"}:
            inactive.add("tight_buffer_px")
        if cells:
            for key in ("margin_mode", "recenter"):
                if all(key in overrides.get(cid, {}) for cid in cells):
                    inactive.add(key)
        for key in inactive:
            values.pop(key, None)
        values["cell_rules"] = copy.deepcopy(overrides)
        return values

    def search_context(self, params: Mapping[str, Any], call: str) -> Optional[dict]:
        return {"image": self.input.get("image_key"), "mask": self.input.get("nuclei_mask"),
                "min_area": self.input.get("min_area")}

    def search_notes(self) -> str:
        return ("cell_rules is one structured focus dimension: it includes the complete per-cell overrides, not just\n"
                "the written rule. Changing its population is a compound intervention. The per-cell keys (auto_tighten,\n"
                "min_containment, reset, and a cell's own margin_mode or recenter) are not dimensions: they belong to\n"
                "cell_rules, so name cell_rules for them in focus_levers and deferred; margin_mode and recenter there\n"
                "mean the image-level levers. Geometry-equivalent calls remain deduplicated. Mask checks at delivery are\n"
                "not independent validation.\n")

    def trial_params(self, config: Mapping[str, Any]) -> dict:
        """The tool kwargs of a configuration: its levers and cell_overrides, the fixed min_area and the
        call's own directory inside tool_cache (named by its config_key)."""
        return {**{lever: config.get(lever) for lever in self.levers},
                "cell_overrides": copy.deepcopy(config.get("cell_overrides") or {}),
                # fixed, not a lever: a tunable min_area would let the loop drop hard objects to raise its score
                "min_area": int(self.input.get("min_area") or 0),
                "output_subdir": f"crop_calls/{self.input.get('image_key') or 'image'}/{self.config_key(config)[:12]}"}

    def config_key(self, params: Mapping[str, Any]) -> str:
        """The crops a configuration cuts: the box of every croppable cell (`cell_boxes`, by the tool's own
        geometry, before the call). Two configurations with the same key crop every cell identically -- the
        same call, whatever levers and per-cell keys they write (e.g. auto_tighten on a cell it does not cut,
        or tight and plain mask_bbox, the same square); a one-pixel difference in one cell is another call."""
        text = json.dumps([self.input.get("nuclei_mask"), {k: params.get(k) for k in (*self.levers, "cell_overrides")}],
                          sort_keys=True, default=str)       # per mask: one verifier can serve several inputs
        if text not in self._keys:
            boxes = sorted(self.cell_boxes(params).items())
            self._keys[text] = hashlib.sha1(json.dumps(boxes).encode()).hexdigest()
        return self._keys[text]

    def _box_planner(self) -> Optional[dict]:
        """The input's per-cell geometry for `cell_boxes`: the tool (`_CellBoxes`), the label mask and the objects
        at min_area by cell_id with their regionprops index (the tool's enumeration), built once per input;
        None without a croppable object."""
        mask_path = self.input.get("nuclei_mask")
        if not mask_path or not self.input.get("cells"):
            return None
        if self._planner is None or self._planner["mask"] != mask_path:
            from skimage.measure import regionprops

            labeled = _load_label_mask(mask_path)
            prefix = f"{self.input.get('group') or 'default'}_{self.input.get('source_image_id')}_cell_"
            regions = {f"{prefix}{idx:04d}": (idx, region) for idx, region in enumerate(regionprops(labeled))}
            self._planner = {"mask": mask_path, "tool": _CellBoxes(labeled), "labeled": labeled,
                             "regions": {cid: regions[cid] for cid in self.input["cells"] if cid in regions}}
        return self._planner

    def cell_boxes(self, params: Mapping[str, Any]) -> dict[str, Optional[list[int]]]:
        """The crop box the tool gives every croppable cell under `params` (None: the cell gets no crop), before
        the call: the tool's own per-cell geometry (`Single_Cell_Cropper_Tool.cell_crop_box` -- every margin mode,
        recenter, the crop_size cap, min_containment, auto_tighten's cuts and the containment rules) on the
        input's label mask, with the arguments read the way the tool reads them (`pixel_args`; a null margin is
        the tool's rule for this mask). {} without a croppable object."""
        planner = self._box_planner()
        if planner is None:
            return {}
        defaults = self.defaults()
        lever = {k: params.get(k, defaults.get(k)) for k in self.levers}
        margin = defaults.get("margin") if lever.get("margin") is None else lever["margin"]
        min_area, margin, crop_size, buffer = pixel_args(self.input.get("min_area") or 0, margin,
                                                         lever.get("crop_size"), lever.get("tight_buffer_px"))
        labeled, overrides, out = planner["labeled"], params.get("cell_overrides") or {}, {}
        for cid, (idx, region) in planner["regions"].items():
            plan = planner["tool"].cell_crop_box(
                region, labeled, None, labeled.shape[:2], overrides.get(cid) or {}, idx=idx, min_area=min_area,
                margin=margin, crop_size=crop_size, margin_frac=lever.get("margin_frac"),
                margin_mode=lever.get("margin_mode"), tight_buffer_px=buffer, recenter=lever.get("recenter"),
                full_field_areas=planner["tool"].areas)
            out[cid] = None if plan["box"] is None else [int(v) for v in plan["box"]]
        return out

    def _cell_value(self, parent: str, cid: str, readout: str):
        if readout in INPUT_CELL_READOUTS:
            return (self.input.get("cells") or {}).get(cid, {}).get(readout)
        return (self._cells.get(self._labels.get(parent, -1)) or {}).get(cid, {}).get(readout)

    def _condition(self, cond: Mapping[str, Any], parent: str) -> Callable[[str], bool]:
        readout, op = cond.get("readout"), cond.get("op")
        known = INPUT_CELL_READOUTS + (CALL_CELL_READOUTS if parent != INPUT else ())
        if readout not in known:
            where = "the input" if parent == INPUT else f"the input or {parent}"
            raise ValueError(f"{readout!r} is not a per-cell readout of {where} (known: {list(known)})")
        compare = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "==": np.equal}.get(op)
        if compare is None:
            raise ValueError(f"{op!r} is not one of >, >=, <, <=, ==")
        value = self.literal(readout, str(cond.get("value", "")).strip(), "choice")
        if readout in _CATEGORICAL:
            if op != "==":
                raise ValueError(f"{readout} is categorical: only == applies")
            return lambda cid: self._cell_value(parent, cid, readout) == value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{readout} {op} {cond.get('value')!r}: the value is not a number")

        def holds(cid):
            v = self._cell_value(parent, cid, readout)
            return v is not None and bool(compare(float(v), float(value)))
        return holds

    def _settings(self, settings, k: int) -> dict:
        out = {}
        for s in settings or []:
            key = s.get("key")
            if key not in CELL_KEYS:
                raise ValueError(f"cell rule {k}: {key!r} is not a per-cell key (known: {list(CELL_KEYS)})")
            if key in out:
                raise ValueError(f"cell rule {k}: {key} is set twice")
            kind, bounds = CELL_KEYS[key]
            out[key] = self.parse_value(key, s.get("value"), kind, bounds)
        if not out:
            raise ValueError(f"cell rule {k} sets no per-cell key")
        if "reset" in out and (out["reset"] is not True or len(out) > 1):
            raise ValueError(f"cell rule {k}: reset is true and alone in its rule")
        return out

    def extend_move(self, move: Mapping[str, Any], params: dict, parent: str, attempts) -> dict:
        """The move's cell rules, each matching at least one cell (a rule without a condition matches every
        cell: vacuous truth), compiled on top of a copy of the parent's cell_overrides into `params`;
        auto_tighten only for a cell that resolves to mask_bbox (after the move's lever changes). A cell a rule
        of this move moves out of mask_bbox drops the auto_tighten it inherited from the parent (the second
        user-approved relaxation, 2026-10-01: logged as `dropped_auto_tighten` and stated in the move's
        param_delta, `auto_tighten_dropped`); an auto_tighten a rule of this move sets on a cell outside
        mask_bbox, or an inherited one on a cell the lever changes alone move out, rejects the move. Returns
        the rules as written, with the number of cells each matched, for the move's param_delta; raises
        ValueError."""
        overrides, written = copy.deepcopy(params.get("cell_overrides") or {}), []
        moved, tightened = set(), set()      # cells whose margin_mode / auto_tighten a rule of this move sets
        for k, rule in enumerate(move.get("cell_rules") or [], 1):
            conditions = [self._condition(c, parent) for c in rule.get("when") or []]
            settings = self._settings(rule.get("apply"), k)
            cells = [cid for cid in sorted(self.input.get("cells") or {}) if all(cond(cid) for cond in conditions)]
            if not cells:
                raise ValueError(f"cell rule {k} matches no cell")
            for cid in cells:
                if settings.get("reset"):
                    overrides.pop(cid, None)
                    continue
                overrides[cid] = {**(overrides.get(cid) or {}), **settings}
                if "margin_mode" in settings:
                    moved.add(cid)
                if "auto_tighten" in settings:
                    tightened.add(cid)
            written.append((" and ".join(f"{c.get('readout')} {c.get('op')} {c.get('value')}"
                                         for c in rule.get("when") or []) or "every cell")
                           + " -> " + ", ".join(f"{s.get('key')}={s.get('value')}" for s in rule["apply"])
                           + f" ({len(cells)} cells)")
        dropped = []
        for cid, ov in overrides.items():
            mode = ov.get("margin_mode", params.get("margin_mode"))
            if not ov.get("auto_tighten") or mode == "mask_bbox":
                continue
            if cid in moved and cid not in tightened:            # inherited, and a rule moved the cell out
                del ov["auto_tighten"]
                dropped.append(cid)
                continue
            raise ValueError(f"auto_tighten acts only in mask_bbox mode ({cid} resolves to {mode})")
        params["cell_overrides"] = overrides
        extra = {"cell_rules": written} if written else {}
        if dropped:
            extra["auto_tighten_dropped"] = (f"{len(dropped)} cell(s) a rule moved out of mask_bbox: the auto_tighten "
                                             f"they had in {parent} dropped (it acts only in mask_bbox)")
            self.log("dropped_auto_tighten", text=f"{extra['auto_tighten_dropped']}: {dropped}")
        return extra

    def move_name(self, move: Mapping[str, Any]) -> str:
        rules = move.get("cell_rules") or []
        return super().move_name(move) + (f" + {len(rules)} cell rule(s)" if rules else "")

    # ---- progress: the paired per-cell noise, over the admissible calls -------------------------
    # The Pareto front and the plateau compare admissible calls only: each crops every object at min_area,
    # so their means are over the same cells. An inadmissible call (e.g. one whose crop_size cap dropped
    # most objects: its few crops can score 1.0 on every objective) is never on the front, dominates no
    # call and is no progress.
    def _admissible_calls(self, attempts) -> set[str]:
        """The finished calls the guard admits (the guard reads only a call's own readouts)."""
        return {c for c, (_, _, v) in attempts.items() if not self.guard(None, v)}

    def _improves(self, c: str, d: str, attempts) -> bool:
        """`c` is better than `d` beyond the noise on at least one objective."""
        ic, id_ = self._labels[c], self._labels[d]
        for key in self.objectives:
            x, y = self._score(attempts[c][2], key), self._score(attempts[d][2], key)
            if x is not None and y is not None and x > y + self.noise(id_, ic, key):
                return True
        return False

    def pareto(self, attempts) -> set[str]:
        """The admissible calls no other admissible call dominates on the objectives beyond the paired
        noise."""
        pool = self._admissible_calls(attempts)

        def dominates(a, b):
            ia, ib = self._labels[a], self._labels[b]
            pairs = [(self._score(attempts[a][2], k), self._score(attempts[b][2], k), self.noise(ia, ib, k))
                     for k in self.objectives]
            pairs = [(x, y, n) for x, y, n in pairs if x is not None and y is not None]
            return bool(pairs) and all(x >= y - n for x, y, n in pairs) and any(x > y + n for x, y, n in pairs)

        return {c for c in pool if not any(dominates(o, c) for o in pool if o != c)}

    def _stall(self, inp, attempts) -> int:
        """Rounds in a row without progress: the finished rounds (a round of failed calls included) after
        the last round with an admissible call that is better than every admissible call of the earlier
        rounds beyond the noise on at least one objective."""
        pool, stall, earlier = self._admissible_calls(attempts), 0, []
        for indices in self.rounds:
            calls = [self.label(i) for i in indices if self.label(i) in pool]
            if calls and any(all(self._improves(c, d, attempts) for d in earlier) for c in calls):
                stall = 0
            else:
                stall += 1
            earlier += calls
        return stall

    def _prediction(self, parent: VLRRResult, vres: VLRRResult) -> dict[str, str]:
        """Each stated expectation next to the observed change from the parent, beyond or within the paired
        noise (a measurement; whether a prediction failed is the agent's judgement): metric -> "expected ->
        observed"."""
        ip, iv = self._index_of.get(id(parent), 0), self._index_of.get(id(vres), 0)
        out = {}
        for metric, direction in (vres.expected_effect or {}).items():
            a, b = self._score(parent, metric), self._score(vres, metric)
            if ip == 0:
                observed = "the parent is the input (no crops)"
            elif a is None or b is None:
                observed = "not computable"
            else:
                d = b - a
                if metric in _CELL_KEY:
                    seen = "within noise" if abs(d) <= self.noise(ip, iv, metric) else ("up" if d > 0 else "down")
                else:
                    seen = "moved" if d else "unchanged"
                observed = f"{d:+.4f} ({seen})"
            out[metric] = f"{direction} -> {observed}"
        return out

    def _facts(self, before: int, after: int) -> dict:
        """Key readout differences `after - before` over the cells both calls cropped, without a key readout
        that is the exact complement of a kept one (`_FACT_COMPLEMENT`)."""
        out = {}
        for key in self.key_readouts:
            if _FACT_COMPLEMENT.get(key) in self.key_readouts:
                continue
            paired = self._paired(before, after, _CELL_KEY[key])
            if paired is None:
                continue
            mean, se2, n = paired
            ca, cb = self._cells[before], self._cells[after]
            diffs = [cb[c][_CELL_KEY[key]] - ca[c][_CELL_KEY[key]] for c in ca.keys() & cb.keys()]
            out[key] = (f"{mean:+.4f} +/- {se2:.4f} ({'beyond' if abs(mean) > max(NOISE_BAND, se2) else 'within'} "
                        f"noise; {sum(d > 1e-9 for d in diffs)} cells up, {sum(d < -1e-9 for d in diffs)} down of {n})")
        return out

    def _defaults_call(self, attempts) -> str | None:
        key = self.config_key(self.defaults())
        return next((c for c, (_, p, _) in attempts.items() if self.config_key(p) == key), None)

    # ---- the deterministic fallback, the choice without an admissible call ----------------------
    def _rule_matches(self, rule, parent: str) -> bool:
        try:
            conditions = [self._condition(c, parent) for c in rule["when"]]
        except ValueError:
            return False
        return any(all(cond(cid) for cond in conditions) for cid in (self.input.get("cells") or {}))

    def _step(self, lever: str, value: Any, direction: str) -> Any:
        """One step of `lever` from `value` in its CARD Fallback direction (`next` = the next margin mode);
        None when there is none."""
        if direction == "next":
            later = MARGIN_MODES[MARGIN_MODES.index(value) + 1:] if value in MARGIN_MODES else ()
            return later[0] if later else None
        if lever == "recenter":
            return True if direction == "up" and value is False else None
        if direction not in ("up", "down") or value is None or lever not in PARAM_BOUNDS:
            return None
        stepped = float(f"{apply_bounded({lever: float(value)}, {lever: direction})[lever]:.6g}")
        try:
            new = self.coerce(lever, stepped)
        except ValueError:
            return None
        return None if new == value else new

    def fallback_moves(self, inp, attempts, add) -> None:
        """After the defaults: from the defaults call, the linear CropVerifier's round-1 geometry policy as
        cell rules; single-lever steps from the incumbent (CARD order, Fallback direction)."""
        defaults = self._defaults_call(attempts)
        if defaults is not None:
            matching = [r for r in _POLICY_RULES if self._rule_matches(r, defaults)]
            if matching:
                add(defaults, [], "the linear CropVerifier's round-1 geometry policy", cell_rules=matching)
        incumbent = self._incumbent(inp, attempts)
        params = self._params(attempts, incumbent)
        for lever in self.levers:
            new = self._step(lever, params.get(lever), self.lever_fields[lever].get("Fallback", "none"))
            if new is not None:
                add(incumbent, [{"lever": lever, "value": json.dumps(new)}], f"{lever} one step from {incumbent}")

    def least_bad(self, inp, attempts) -> Optional[str]:
        """No admissible call: the fewest objects without a crop that a square crop can hold
        (`n_missing_crops`), then the highest lowest retention, then the highest score."""
        def rank(c):
            m = attempts[c][2].metrics or {}
            return (-int(m.get("n_missing_crops") or 0), m.get("min_target_retention") or 0.0,
                    self._score(attempts[c][2]) or -1.0, -self._labels[c])
        return max(attempts, key=rank) if attempts else None

    # ---- the cohort mode: the group's consensus applied, no agent round -------------------------------
    def reference_moves(self, inp, attempts) -> list[dict]:
        """None normally; in the cohort mode (`cohort_levers` set) the group's consensus levers as ONE reference
        call on top of the call at the defaults (round 1) -- the crop analogue of the segmentation cohort's
        corrections-plus-consensus reference, but the cropper has no per-image correction, so the consensus is the
        only change. The shared loop runs it before the (empty) agent round; `choose` then delivers it unasked."""
        if self.cohort_levers is None or not self.rounds:
            return super().reference_moves(inp, attempts)
        call = self.label(self.rounds[0][0])
        if call not in attempts:
            return []
        params = attempts[call][1]
        consensus = {k: v for k, v in self.cohort_levers.items() if k in self.levers and v != params.get(k)}
        if not consensus:
            return []
        changes = [{"lever": k, "value": json.dumps(v, default=str)} for k, v in consensus.items()]
        return [{"parent": call, "changes": changes,
                 "diagnosis": f"the group's calibrated parameters {json.dumps(consensus, default=str)}"}]

    def reference_default(self, inp, attempts) -> Optional[str]:
        """The default delivery: the system's reference call (the base rule). In the cohort mode only, without one,
        the call at the defaults (round 1, the admission's call) when it finished and the guard admits it, so a
        cohort image whose admission SKIPped still delivers the call at its (consensus) defaults; None without
        either. (Gated on the cohort mode: a normal run keeps the base rule, no default-call fallback.)"""
        default = super().reference_default(inp, attempts)
        if default is not None or self.cohort_levers is None or not self.rounds:
            return default
        call = self.label(self.rounds[0][0])
        return call if call in attempts and not self.guard(inp, attempts[call][2]) else None

    def plan_round(self, inp, attempts, k_max: int, remaining: int) -> list[dict]:
        """The agent's round; in the cohort mode (`cohort_levers` set) none: the shared loop has run the call at
        the defaults, its admission and, after a RUN, the one reference call at the group's consensus levers, so the
        search stops here ("cohort_applied") and `choose` delivers the reference default unasked."""
        if self.cohort_levers is None:
            return super().plan_round(inp, attempts, k_max, remaining)
        self.stop_reason = "cohort_applied"
        return []

    def choose(self, inp, attempts, admissible: list[str]) -> tuple[Optional[str], str]:
        """The base's final question; in the cohort mode the reference default (the consensus reference call when it
        ran and the guard admits it, else the call at the defaults), unasked."""
        default = self.reference_default(inp, attempts)
        if self.cohort_levers is None or default not in admissible:
            return super().choose(inp, attempts, admissible)
        return default, "cohort_applied"

    def select_on_input(self, state, details: dict) -> Optional[WaveSelection]:
        if not self.input.get("n_croppable"):
            return WaveSelection(None, "conceded", "no_objects", "ceiling", details)
        return super().select_on_input(state, details)

    def extra_details(self, inp, attempts) -> dict:
        """Every call's configuration, readouts and metadata path; the input record without its cells."""
        return {"calls": {c: {"parent": p, "configuration": self._summary(params),
                              "metadata_path": (self._reports.get(self._labels[c]) or {}).get("metadata_path"),
                              "readouts": {k: round(v, 5) for k, v in (v.metrics or {}).items()}}
                          for c, (p, params, v) in attempts.items()},
                "input": {k: v for k, v in self.input.items() if k not in ("cells", "status")}}

    def deliver(self, state, index: int, details: dict) -> None:
        """Runs the mask check on the delivered call only (`_mask_check`, on every delivery path: it writes the
        call's kept-crops manifest and decision trace; its counts, its montage questions (`montages`: prompt, image
        ref, reply, error) and its per-crop `verdicts` go into `details["mask_check"]`, a measurement on what is
        delivered that no rule or prompt reads), then copies the call's metadata and its kept-crops
        manifest and decision trace (fresh name, fresh mtime) from the call's own directory into
        `<query_cache_dir>/tool_cache`, where downstream readers (the analyzer) take the newest metadata file per
        image, and points the chosen result at the copy; the crop paths inside stay absolute, into the call's
        directory."""
        report = self._reports.get(index) or {}
        if report.get("metadata_path"):
            counts = self._mask_check(list((self._crops.get(index) or {}).values()), report["metadata_path"], report)
            details["reasoner_failures"] = self.reasoner_failures
            if counts:
                details["mask_check"] = {"call": self.label(index), **counts,
                                         "rejected_cells": report["mask_check_rejected_cells"],
                                         "montages": report["mask_check_montages"],
                                         "verdicts": report["mask_check_verdicts"]}
        params, _, result = state.trials[index]
        source = (result or {}).get("cell_crops_metadata_path")
        if not source or not os.path.exists(source):
            return
        target_dir = Path(source).parent
        for _ in Path(str(params.get("output_subdir") or "")).parts:
            target_dir = target_dir.parent
        new_id, name = uuid4().hex[:8], Path(source).name
        target = target_dir / f"cell_crops_metadata_{new_id}.json"
        shutil.copyfile(source, target)
        for prefix in ("cell_crops_kept_manifest_", "cell_crops_triage_trace_"):
            sidecar = Path(source).parent / name.replace("cell_crops_metadata_", prefix)
            if sidecar.exists():
                shutil.copyfile(sidecar, target_dir / f"{prefix}{new_id}.json")
        result["call_metadata_path"] = source
        result["cell_crops_metadata_path"] = str(target)
        details["delivered_metadata_path"] = str(target)

    # ---- prompts: the cropper's text and images ------------------------------------------------
    def _summary(self, params: Mapping[str, Any]) -> dict:
        """A configuration's non-default image-level values and its per-cell settings counted by value."""
        defaults = self.defaults()
        counts: dict[str, int] = {}
        for ov in (params.get("cell_overrides") or {}).values():
            for k, v in sorted(ov.items()):
                counts[f"{k}={json.dumps(v)}"] = counts.get(f"{k}={json.dumps(v)}", 0) + 1
        return {"image_level": {k: params.get(k) for k in self.levers if params.get(k) != defaults.get(k)},
                "per_cell": counts or "none"}

    def _input_facts(self) -> dict:
        by_id = self.input.get("cells") or {}
        cells = by_id.values()
        keys = ("image_shape", "channels", "dtype", "channel_p1", "channel_p99", "n_objects", "n_croppable",
                "n_uncroppable", "foreground_fraction", "min_area", "error")
        facts = {k: self.input.get(k) for k in keys if self.input.get(k) is not None}
        facts["uncroppable_cells"] = sorted(cid for cid, c in by_id.items() if c.get("uncroppable"))
        facts["per_cell_quantiles"] = {k: _quantiles([c[k] for c in cells]) for k in INPUT_CELL_READOUTS
                                       if k != "touches_image_border"}
        facts["touching_image_border"] = sum(1 for c in cells if c.get("touches_image_border"))
        return facts

    def task_text(self) -> str:
        return self.task_brief or (
            f"crop every object of the given segmentation mask of image {self.input.get('source_image_id')}: one "
            f"crop per object, {self.input.get('n_croppable')} objects at min_area {self.input.get('min_area')}.")

    def final_rule_text(self) -> str:
        return (f"Final output: at the end you choose one finished call; its crops, minus those the mask-error\n"
                f"check rejected, are delivered. It must have a crop for every object at min_area that a square crop\n"
                f"can hold at target_retention {RETENTION_FLOOR} (the input's uncroppable_cells, for which no square\n"
                f"crop keeps that share, are exempt), every crop square (height = width) and every crop's\n"
                f"target_retention at least {RETENTION_FLOOR}. A choice that does not qualify is answered with the\n"
                f"conditions it fails and asked once more; if it still does not qualify, the qualifying call with the\n"
                f"highest {self.score_key} is delivered.\n"
                f"If no call qualifies, you are not asked: the call with the fewest of those objects without a crop\n"
                f"(n_missing_crops), then the highest min_target_retention, then the highest {self.score_key} is\n"
                f"delivered.")

    def plateau_text(self) -> str:
        return (f"{self.patience} rounds in a row have not produced an admissible call better than every\n"
                f"earlier admissible call beyond the noise on at least one of {list(self.objectives)}")

    def evidence_text(self) -> str:
        return (f"Noise: a difference between two calls counts as beyond the noise when it exceeds the larger of the\n"
                f"stated band {NOISE_BAND} and 2SE of the per-cell paired difference. The tool is deterministic: the\n"
                "same configuration gives the same crops.")

    def defaults_note(self) -> str:
        return ("(margin: the caller's value, else the tool's rule by mask file name, resolved for this mask; a move\n"
                "from \"input\" has no per-cell settings)")

    def move_text(self) -> str:
        return (
            "A move names a parent (\"input\" = the tool at its defaults, or a finished call id), the image-level\n"
            "levers it changes relative to the parent's configuration (one or several; from \"input\", no change =\n"
            "the tool at its defaults) and, optionally, cell rules. A cell rule selects the cells for which all its\n"
            "conditions hold (no condition: every cell) and sets per-cell keys for them; rules apply in order on\n"
            "top of the parent's per-cell settings, a later rule overwriting an earlier one's keys, reset removes a\n"
            "cell's per-cell settings, and a cell a rule moves out of mask_bbox loses the auto_tighten it had in the\n"
            "parent.\n"
            f"Values are JSON literals (e.g. {self.value_examples}). A configuration that crops every cell like a\n"
            "call already run or failed is rejected. Give each move a diagnosis naming the evidence behind it and\n"
            "the readouts you expect to move (up, down or preserve). Return your hypothesis list with every plan\n"
            "(id, statement, status open | supported | refuted | abandoned, evidence); it is shown back to you next\n"
            "round."
        )

    def readouts_text(self) -> str:
        return (
            "Readouts of a call (GT-free, measured on the given mask over all its crops, before the mask-error\n"
            "check; not ground truth): n_crops; n_lost_to_geometry (objects at min_area without a crop), of them\n"
            f"n_missing_crops (objects a square crop can hold at target_retention {RETENTION_FLOOR}) and\n"
            "n_exempt_uncroppable (the input's uncroppable_cells: no square crop inside the image keeps that share);\n"
            "n_edge_limited (crops whose centroid_offset the image edge forces: no square of their side inside the\n"
            f"image places the centroid closer to the crop centre -- the box within {EDGE_TOLERANCE_PX} px of the image\n"
            "edge on the side the centroid lies toward);\n"
            "mean / p10 / p50 / p90 of neighbor_contamination (share of the mask foreground inside a crop that\n"
            "belongs to other objects), target_retention (share of the target's mask pixels inside its crop;\n"
            "min_target_retention is the lowest) and centroid_offset (distance of the target's centroid from the\n"
            "crop centre over half the crop's shorter side); mean_target_dominance = 1 - mean\n"
            "neighbor_contamination; mean_centered = 1 - mean centroid_offset; mean_occupancy (target pixels /\n"
            "crop pixels); frac_non_square, p90_crop_aspect (longer / shorter side), median_crop_side_px;\n"
            "frac_mode_<mode> (crops per resolved margin mode); mean_auto_tighten_iterations. The mask-error check\n"
            "(a vision model judges whether a screened crop's mask outlines one real cell; confirmed errors are left\n"
            "out of the delivered crops) runs once, on the delivered call only, after the choice: no call's\n"
            "readouts carry it.\n"
            "Per-cell readouts for rule conditions -- of the input (always): area, eq_diameter, bbox_h, bbox_w\n"
            "(px), aspect_ratio (longer / shorter bbox side), solidity (area / convex hull area),\n"
            "bbox_other_fraction (share of the mask foreground inside the object's own bbox that belongs to other\n"
            f"objects), nn_gap_px (pixel gap to the nearest other object, measured up to {NN_GAP_CAP_PX}),\n"
            "touches_image_border (true / false); of the parent call's crop of the cell (when the parent is a\n"
            "call): neighbor_contamination, target_dominance, target_retention, centroid_offset, centered,\n"
            "occupancy, border_touch (share of the target's pixels on the crop border), n_complete_cells (objects\n"
            "at least 90% inside the crop), contamination_inset_px (the tool's measured cut depth on the most\n"
            "contaminated side, capped by the retention floor), crop_h, crop_w, crop_aspect, margin_mode,\n"
            "edge_limited (true / false: the image edge forces the crop's centroid_offset).")

    def input_text(self, inp: VLRRResult) -> str:
        panel = self.input.get("panel_cells") or []
        return (f"Input: {json.dumps(self._input_facts(), default=str)}\n"
                f"Crop panels show the same {len(panel)} cells for every call (position i = the i-th of {panel};\n"
                "picked from the input over bbox_other_fraction), each with the target's mask contour in green.")

    def calls_legend(self) -> str:
        return ("change = relative to the parent; configuration = the non-default image-level values and the\n"
                "per-cell settings counted by value; readouts = the call's readouts (values in the order the first\n"
                "line below names); vs_parent / vs_defaults_call (the latter only when the parent is not the defaults\n"
                "call) = key readout differences over the cells both calls cropped, mean +/- 2SE, beyond or within\n"
                "the noise, and how many cells moved; prediction = each stated expectation with the observed change\n"
                "from the parent; guard = whether it is an admissible final choice; pareto = an admissible call that\n"
                "no other admissible call dominates on the objectives beyond the noise (admissible calls crop the same\n"
                "cells); seconds = the call's run time")

    @staticmethod
    def _readout_names(attempts) -> list[str]:
        """The calls table's readout names: every finished call's, in first-seen order over the calls (in run
        order)."""
        return list(dict.fromkeys(k for c in attempts for k in (attempts[c][2].metrics or {})))

    def _table(self, inp, attempts) -> str:
        """The rows after two header lines: the readout names (a row's `readouts` are its values in this order,
        null = not measured), the prediction format and the key readouts the vs-parent facts leave out as the
        exact complement of a kept one."""
        if not attempts:
            return super()._table(inp, attempts)
        left_out = "; ".join(f"{k} (per crop, {_CELL_KEY[k]} = 1 - {_CELL_KEY[kept]}: its difference is {kept}'s "
                             f"negated, cells up and down swapped)" for k, kept in _FACT_COMPLEMENT.items()
                             if kept in self.key_readouts and k in self.key_readouts)
        return (f"Readouts, in this order (null = not measured for that call): "
                f"{json.dumps(self._readout_names(attempts))}\n"
                f'prediction = {{metric: "expected -> observed"}}; vs_parent / vs_defaults_call leave out {left_out}\n'
                f"{super()._table(inp, attempts)}")

    def _rows(self, inp, attempts) -> list[dict]:
        front, rows = self.pareto(attempts), []
        defaults = self._defaults_call(attempts)
        names = self._readout_names(attempts)
        for call in attempts:                    # in run order
            parent, params, vres = attempts[call]
            index = self._labels[call]
            parent_vres = inp if parent == INPUT else attempts[parent][2]
            fails = self.guard(inp, vres)
            report = self._reports.get(index) or {}
            row = {"call": call, "parent": parent, "change": vres.param_delta or "none (the tool at its defaults)",
                   "configuration": self._summary(params), "diagnosis": vres.notes,
                   "readouts": [_compact((vres.metrics or {}).get(k)) for k in names],
                   "vs_parent": (self._facts(self._labels[parent], index) if parent != INPUT
                                 else "the parent is the input (no crops)"),
                   "prediction": self._prediction(parent_vres, vres),
                   "guard": "admissible" if not fails else "; ".join(fails), "pareto": call in front,
                   "seconds": report.get("seconds")}
            if defaults and defaults not in (call, parent):        # from the defaults call, vs_parent says it
                row["vs_defaults_call"] = {"call": defaults, **self._facts(self._labels[defaults], index)}
            rows.append(row)
        return rows

    def untried_list(self, attempts) -> str:
        keys = [k for k in CELL_KEYS if not any(
            k in ov for _, p, _ in attempts.values() for ov in (p.get("cell_overrides") or {}).values())]
        return f"{super().untried_list(attempts)}; per-cell keys no rule has set yet: {keys}"

    def _input_image(self) -> bytes | None:
        if self._input_png is None and self.input.get("original_image") and self.input.get("nuclei_mask"):
            self._input_png = _input_overlay(self.input["original_image"], self.input["nuclei_mask"], PROMPT_IMAGE_PX)
        return self._input_png

    def _panel(self, index: int, max_px: int) -> bytes | None:
        """The crop panel of a call: its crops of the panel cells (target contour in green), in panel order."""
        from ..crop_triage import _build_mask_check_montage

        crops = self._crops.get(index) or {}
        chosen = [crops[c] for c in self.input.get("panel_cells") or [] if c in crops]
        if not chosen:
            return None
        side = int(math.ceil(math.sqrt(len(chosen))))
        try:
            data, _ = _build_mask_check_montage(chosen, (side, side))
        except Exception as exc:  # noqa: BLE001 -- a prompt without the panel still carries the facts
            logger.warning("crop agentic: no crop panel for trial %s (%s)", index, exc)
            return None
        if data is None:
            return None
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _call_images(self, calls, max_px: int) -> list:
        """The input overlay and the crop panel of each of `calls`, as label / image pairs."""
        out: list = []
        image = self._input_image()
        if image:
            out += ["INPUT IMAGE (the mask's object boundaries in green):", image]
        for call in calls:
            panel = self._panel(self._labels[call], max_px)
            if panel:
                out += [f"CROP PANEL OF {call}:", panel]
        return out

    def round_attachments(self, inp, attempts) -> list:
        """The input overlay and the crop panel of every call of the previous round (every question to the
        reasoner is stateless, so each prompt carries its own images)."""
        return self._call_images([self.label(i) for i in (self.rounds[-1] if self.rounds else [])
                                  if self.label(i) in attempts], PROMPT_IMAGE_PX)

    def admission_attachments(self, inp, attempts, call: str) -> list:
        """The input overlay and the crop panel of the call at the defaults, as a round shows a call."""
        return self._call_images([call], PROMPT_IMAGE_PX)

    def choice_text(self) -> str:
        return "Choose the delivered crops of this image: one finished call."

    def admissible_text(self) -> str:
        return (f"a crop for every object at min_area that a square crop can hold at target_retention "
                f"{RETENTION_FLOOR} (the uncroppable_cells are exempt), every crop square (height = width), every "
                f"crop's target_retention at least {RETENTION_FLOOR}")

    def admission_checks_text(self) -> str:
        """The conditions the call at the defaults met, as the guard's are stated: the guard's and, per crop,
        each readout of `_retune_rule` at most its threshold (an edge-limited crop's centroid_offset not
        counted)."""
        checks = ", ".join(f"{key} at most {limit}" for key, limit in self._retune_rule())
        return (f"the guard: {self.admissible_text()}; per crop: {checks} (a crop whose centroid_offset the image "
                f"edge forces, edge_limited, not counted)")

    def _final_panels(self, attempts, admissible: list[str]) -> tuple[list[str], str]:
        """The admissible calls whose crop panels the final prompt carries (at most FINAL_PANELS_MAX), and
        which they are: those on the Pareto front; when none is (dominance within the noise can cycle), the
        admissible calls with the highest score."""
        front = self.pareto(attempts)
        calls = [c for c in admissible if c in front]
        if calls:
            return calls[:FINAL_PANELS_MAX], "the admissible calls on the Pareto front"

        def rank(call):
            score = self._score(attempts[call][2])
            return (-score if score is not None else math.inf, self._labels[call])
        return (sorted(admissible, key=rank)[:FINAL_PANELS_MAX],
                f"no admissible call is on the Pareto front: the admissible calls with the highest {self.score_key}")

    def final_evidence_text(self, inp, attempts, admissible: list[str]) -> str:
        calls, which = self._final_panels(attempts, admissible)
        return f"Crop panels attached ({which}): {calls}\n{self.readouts_text()}"

    def final_attachments(self, inp, attempts, admissible: list[str]) -> list:
        out: list = []
        image = self._input_image()
        if image:
            out += ["INPUT IMAGE (the mask's object boundaries in green):", image]
        for call in self._final_panels(attempts, admissible)[0]:
            panel = self._panel(self._labels[call], FINAL_IMAGE_PX)
            if panel:
                out += [f"CROP PANEL OF {call}:", panel]
        return out
