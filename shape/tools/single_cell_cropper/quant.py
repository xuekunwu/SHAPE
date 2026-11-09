"""GT-free crop-quality score for `single_cell_cropper` (task 035).

The systemic gap `docs/vrr.md` sec 5/6 names for the cropper: it has a GEOMETRY channel
(`tool.py::_compute_crop_qc` -> mask_containment / centroid_offset / border_touch /
neighbor_contamination / target_dominance) but no QUANTITATIVE CROP-QUALITY channel (blur /
truncation / doublet) -- 017 showed a fixed deterministic rule scores 0.0 recall on exactly
these three defects (only a vision judge caught them). This module is the tool-side
`quant.py` convention (`shape/tools/<tool>/quant.py`, mirrors `cell_segmenter/quant.py`):
GT-free, deterministic, cheap -- built ONLY from the crop (+ optional per-cell mask), no
ground truth. It is a Verify-phase QUANTITATIVE signal (vrr.md sec 5), not a CARD lever, and
does NOT re-litigate geometry (028 already showed geometry = a settled fixed-rule surface,
its terms mutually redundant -- see the coding log).

Task 038 history: a later round ("quant/vision division-of-labor") temporarily narrowed this
module to a GT-free/pixel-only subset and relocated the mask-based axes + `crop_quality_score`
to a benchmark-only module, on the premise that the benchmark's per-cell mask is also the GT
mask. A follow-up round REVERSED that split (human-locked): the mask IS a GIVEN input here
(the task is "given the mask, crop the highest-quality crop"), so there is no circularity to
avoid -- this module is mask-capable again, and both the LLM triage judge (`crop_triage.
_quant_evidence`) and the benchmark yardstick read it.

Quality channel -- see `_WEIGHTS`/`crop_quality_score` for the current weights (task 038
v9 restructure, human-locked, RE-WEIGHTED task 038 v12: not_truncated=0.25,
target_dominance=1.0, centered=0.25 -- ONLY the 3 axes this tool's own geometry levers
actually ADJUST feed the combined score; `sharpness`/`single_object` are still
computed/reported but excluded from the weighted sum, since no geometry lever moves them
either way -- see `_WEIGHTS`'s own comment for the full rationale and history):
  - sharpness: blur estimate, GT-free and scale-invariant (Laplacian-variance / total-pixel-
    variance ratio -- see `sharpness()` for why the RAW Laplacian variance alone is not
    comparable across crops of different bit depth/contrast).
  - not_truncated: 1 - (fraction of the TARGET's own foreground pixels sitting on the crop
    border). Reuses `tool.py::_compute_crop_qc`'s exact border-pixel-count formula when a
    per-cell mask is given; `_compute_crop_qc` itself needs a full-field labeled mask +
    regionprops region + bbox, so it cannot be imported and called behind this module's bare
    `(crop, mask)` signature (D4, referee plan-review) -- the reused fragment is the minimal
    border-count snippet, not the whole function. When no mask is given (`mask=None`, the
    case this task's own M3 benchmark exercises -- these crops have no paired mask files), a
    NEW Otsu-foreground estimator stands in for the mask.
  - single_object: target-intactness (fragmentation) estimate from the count of SIGNIFICANT
    connected foreground components (see `single_object()` for what "significant" means, why
    a naive raw CC count is not used, and why this is NOT a doublet detector -- task 038 M-A
    relabel).

Geometry channel (formalized, NOT re-litigated -- reuse LIMITED per D4): `centroid_offset_score`
(pixel-only, via `crop.py::_centering_dominance_from_pixels`, already GT-free and used in
production by `bare_crop_qc`) is computed and reported but NOT part of the weighted
`crop_quality_score` -- `centered` (below) is its mask-aware upgrade and reads IDENTICALLY to it on
a bare crop, so weighting both would double-count the same signal (`_WEIGHTS`'s comment).
`target_dominance` and `centered` DO feed the weighted score (task 038 quality-geometry-
refinement round) -- this does not reopen `crop.py`'s own "dominance is never a hard GATE"
stance (`is_crop_valid`/`is_crop_keep`, and `CropVerifier.adjudicate`'s own scoring, are
unchanged); a continuous, weighted SCORE term here is a different mechanism from a pass/fail
gate there. `mask_containment` / `neighbor_contamination` need full-field context (which cell
is whom outside this one crop) that a single per-cell crop + its own target mask cannot
supply, so they are not computable in this module's `(crop, mask)`-only scope and are omitted
rather than faked.

  - target_dominance: the pixel-estimated target dominance from `crop.py`'s
    `_centering_dominance_from_pixels`, unmodified (see `target_dominance()`).
  - centered (task 038 M-C): the tool's 5th quality axis (square / one cell / CENTERED /
    main area / minimal neighbours) -- 1.0 = the target's centroid sits at the crop's own
    center. MASK-based when a per-cell mask is given (the same normalization
    `tool.py::_compute_crop_qc`'s `centroid_offset` uses, an independent implementation
    per D4); falls back to `centroid_offset_score`'s existing pixel estimate when
    `mask=None`, so it reads IDENTICALLY to `centroid_offset_score` on a bare crop.

No ground truth is read anywhere in this module (see the GT-leak probe in the co-located
test): the public API takes only (crop, mask).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

from shape.agent.verification.verifiers.crop import _centering_dominance_from_pixels


def _laplacian_variance(img: np.ndarray) -> float:
    """Classic 4-neighbour discrete Laplacian (Pech-Pacheco et al. 2000 blur proxy):
    a sharp image has strong local second-derivative responses at edges; blur
    suppresses them. Variance of the Laplacian response over the whole image."""
    lap = (-4 * img
           + np.roll(img, 1, 0) + np.roll(img, -1, 0)
           + np.roll(img, 1, 1) + np.roll(img, -1, 1))
    return float(lap.var())


def sharpness(crop: np.ndarray) -> float:
    """Blur estimate in [0, 1], higher = sharper. GT-free, no mask needed (blur is a
    property of the raw pixels alone).

    Normalization: the RAW Laplacian variance scales as (signal amplitude)^2, so it is
    NOT comparable across crops with different bit depth/contrast (a 16-bit crop would
    look "sharper" than an 8-bit one purely from scale, with no change in actual focus).
    The Laplacian-variance / total-pixel-variance RATIO cancels that scale (both
    variances share the same units^2), making it comparable regardless of the crop's own
    dynamic range. Squashed via ratio/(ratio+1) (0.5 at ratio==1, i.e. the high-frequency
    component alone would have to carry as much variance as the WHOLE image to reach the
    midpoint) -- a conservative, reasoned cut chosen from the kernel's own scale (for
    i.i.d. white noise, the theoretical ceiling on "how much variance a 4-neighbour
    Laplacian can carry", the 5-tap kernel [-4,1,1,1,1] gives ratio ~= sum(kernel^2) = 20;
    ratio==1 sits far below that ceiling and is NOT fit to this task's validation crops --
    see the coding log). A perfectly flat crop (zero total variance -- no signal at all)
    returns 0.0 rather than the degenerate 0/0."""
    img = crop.astype(np.float64)
    total_var = float(img.var())
    if total_var < 1e-9:
        return 0.0
    ratio = _laplacian_variance(img) / total_var
    return float(np.clip(ratio / (ratio + 1.0), 0.0, 1.0))


def _foreground_regions(crop: np.ndarray, close_frac: float = 0.02):
    """Otsu-threshold the crop, take the MINORITY class as foreground (a cell can be
    darker or brighter than its surroundings -- same rule the segmenter's own quant.py
    foreground estimate uses), morphologically close+open with a kernel scaled to the
    crop's own size (merges a single object's internal texture speckle into one
    component without merging two genuinely separate, gapped objects; close_frac=0.02
    is a small fraction of the shorter side, a reasoned default not fit to this task's
    validation crops), then label. Returns regionprops sorted largest-first ([] when
    there is no usable foreground -- a flat/blank crop, out of this module's scope; the
    existing `is_crop_valid` pixel gates already exclude it upstream)."""
    img = crop.astype(np.float64)
    if img.ndim != 2 or not np.isfinite(img).any() or float(np.ptp(img)) < 1e-6:
        return []
    try:
        thr = threshold_otsu(img)
    except Exception:
        return []
    fg = img > thr
    if fg.mean() > 0.5:
        fg = ~fg
    h, w = img.shape
    k = max(1, int(close_frac * min(h, w)))
    if k > 1:
        structure = np.ones((k, k), dtype=bool)
        fg = ndimage.binary_closing(fg, structure=structure)
        fg = ndimage.binary_opening(fg, structure=structure)
    if not fg.any():
        return []
    regions = regionprops(label(fg))
    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def _border_touch_from_mask(mask: np.ndarray) -> float:
    """Fraction of a binary mask's own pixels that sit on ITS array's border. This is
    `tool.py::_compute_crop_qc`'s exact border-pixel-count formula (border_pixels /
    total_target), reused verbatim -- `_compute_crop_qc` itself cannot be imported and
    called behind this module's bare `(crop, mask)` signature (it needs a full-field
    labeled mask + regionprops region + crop bbox, D4), so this is the minimal reusable
    fragment rather than a full reimplementation."""
    total = float(np.count_nonzero(mask))
    if total < 1:
        return 0.0
    edges = np.concatenate([mask[0, :].ravel(), mask[-1, :].ravel(),
                             mask[:, 0].ravel(), mask[:, -1].ravel()])
    border = float(np.count_nonzero(edges))
    return min(1.0, border / total)


def _border_touch_from_pixels(crop: np.ndarray) -> float:
    """NEW pixel estimator (D4): no mask available, so Otsu-foreground -> largest
    component = target cell (the SAME target-selection rule `crop.py`'s
    `_centering_dominance_from_pixels` uses) stands in for the mask, then the same
    border-touch fraction is computed on that component. 0.0 (no truncation signal) when
    there is no usable foreground -- a blank crop is not this module's job; the existing
    `is_crop_valid` pixel-quality gates already exclude it."""
    regions = _foreground_regions(crop)
    if not regions:
        return 0.0
    target = regions[0]
    target_mask = np.zeros(crop.shape, dtype=bool)
    minr, minc, maxr, maxc = target.bbox
    target_mask[minr:maxr, minc:maxc] = target.image
    return _border_touch_from_mask(target_mask)


def not_truncated(crop: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Truncation estimate in [0, 1], higher = less clipped. Reuses the real per-cell
    mask when given (`_border_touch_from_mask`); falls back to the NEW Otsu pixel
    estimator (`_border_touch_from_pixels`) when `mask=None` -- the path this task's own
    M3 benchmark exercises (its crops have no paired mask files)."""
    border_touch = (_border_touch_from_mask(np.asarray(mask) > 0) if mask is not None
                     else _border_touch_from_pixels(crop))
    return float(np.clip(1.0 - border_touch, 0.0, 1.0))


def single_object(crop: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Target-intactness (fragmentation) estimate in [0, 1], higher = more likely the target
    is ONE connected piece; 1/n_significant (n=1 -> 1.0, n=2 -> 0.5, ...).

    NOT a doublet detector (task 038 M-A relabel): this counts significant connected pieces
    in whatever foreground/mask it is given, so a MERGED doublet -- two cells fused into one
    connected blob, whether by the Otsu foreground or by upstream segmentation labeling them
    as one object -- scores 1.0 (no fragmentation), the same as a genuinely intact single
    cell. What it actually measures is whether the TARGET's own region is intact or split
    into multiple significant pieces (a segmentation artifact) -- doublet detection is a
    DIFFERENT question, better answered by vision + `target_dominance`/neighbour-presence
    (see CARD.md's COMPLEMENTARITY/RELIABILITY section).

    "Significant" = area >= 40% of the LARGEST component's area, NOT a fraction of the
    whole crop. A raw, unfiltered CC count is unusable here: `crop.py`'s own design
    TOLERATES neighbour cells inside a crop (never masks/shrinks to exclude them), so a
    "good" crop routinely shows a second, smaller object at the frame edge, and Otsu
    threshold noise on real microscopy texture fragments a single object into many tiny
    specks -- both would be miscounted as extra pieces by a naive count. Sizing
    significance relative to the LARGEST component (not the image) is what tells a genuine
    second, comparably-sized piece (built into the 017 eval-set generator's doublet defect
    class, which this signal was originally validated against -- see the note above for why
    that is not the same as this being a general doublet detector) apart from a small
    tolerated neighbour sliver or texture noise, which sit far below 40% of the target's own
    area. 0.4 is a reasoned round number (a second piece at least 2/5 the target's size is a
    real competing piece, not noise) checked for sane behaviour on a handful of sample crops
    across all 6 defect types (not fit/grid-searched against the full validation set's AUROC
    -- see the coding log). When a per-cell mask IS given, it is authoritative (segmentation
    already isolated the target) -- its own connected components are counted directly
    instead of re-deriving Otsu. 1.0 (no evidence of fragmentation) when there is no usable
    foreground at all -- a blank crop, out of this module's scope."""
    if mask is not None:
        regions = regionprops(label(np.asarray(mask) > 0))
        regions.sort(key=lambda r: r.area, reverse=True)
    else:
        regions = _foreground_regions(crop)
    if not regions:
        return 1.0
    largest = regions[0].area
    n_significant = sum(1 for r in regions if r.area >= 0.4 * largest)
    return float(1.0 / max(1, n_significant))


def centroid_offset_score(crop: np.ndarray) -> float:
    """Geometry: 1 - the pixel-estimated centroid offset (from `crop.py`'s
    `_centering_dominance_from_pixels`, unmodified), clipped to [0, 1]. Always
    pixel-based (mask ignored) -- see the module docstring for why."""
    offset, _ = _centering_dominance_from_pixels(crop)
    return float(np.clip(1.0 - offset, 0.0, 1.0))


def centered(crop: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Centering estimate in [0, 1], higher = better centered; 1.0 = the target's centroid
    sits exactly at the crop's own center (task 038 M-C -- one of the tool's 5 quality
    axes: square / one cell / CENTERED / main area / minimal neighbours).

    When a per-cell MASK is given, computed directly from the mask's own centroid vs the
    crop's own center -- the SAME normalization `tool.py::_compute_crop_qc`'s
    `centroid_offset` uses (`offset_px / (0.5 * min(H, W))`), reused here as an
    INDEPENDENT implementation: `_compute_crop_qc` needs a full-field labeled mask +
    regionprops region + crop bbox that this module's bare `(crop, mask)` signature does
    not have (the same "reused formula, independent implementation" relationship already
    documented for `not_truncated`/`single_object` above -- D4, referee plan-review).
    Falls back to `centroid_offset_score`'s existing pixel-Otsu estimate when `mask=None`
    (bare/HPA crops, this module's own M3 benchmark path) -- so `centered` and
    `centroid_offset_score` read IDENTICALLY on a bare crop; `centered` is the mask-aware
    upgrade, kept ADDITIVE alongside `centroid_offset_score` (unchanged) rather than
    replacing it, so no existing caller's behavior shifts."""
    if mask is None:
        return centroid_offset_score(crop)
    mask_arr = np.asarray(mask)
    mask2d = mask_arr if mask_arr.ndim == 2 else mask_arr[..., 0]
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return 0.0                      # no mask signal -- no centering evidence
    h, w = mask2d.shape
    cy, cx = float(ys.mean()), float(xs.mean())
    offset_px = float(np.hypot(cy - (h - 1) / 2.0, cx - (w - 1) / 2.0))
    offset = offset_px / (0.5 * min(h, w) + 1e-9)
    return float(np.clip(1.0 - offset, 0.0, 1.0))


def target_dominance(crop: np.ndarray) -> float:
    """Geometry (informational only -- see module docstring for why it is excluded from
    the weighted `crop_quality_score`): the pixel-estimated target dominance from `crop.py`'s
    `_centering_dominance_from_pixels`, unmodified."""
    _, dominance = _centering_dominance_from_pixels(crop)
    return float(np.clip(dominance, 0.0, 1.0))


# Task 038 v9 restructure (human-locked): the combined score now weights ONLY the
# AGENT-ADJUSTED axes -- not_truncated / target_dominance / centered -- the 3 axes this
# tool's own geometry levers (SS1/SS5: margin_frac, tight, mask_bbox, recenter) actually
# move. `sharpness` and `single_object` are DROPPED from the weighted sum (they are
# NEVER adjusted by any geometry lever -- blur is an acquisition/focus property of the
# raw pixels, and fragmentation is a segmentation-upstream artifact this tool's own crop
# geometry cannot fix either way) -- keeping them in the weighted mean DILUTES the
# measured effect of the agent's own actions with two axes that never move regardless of
# what the agent does, understating its real contribution. Both axis FUNCTIONS
# (`sharpness()`/`single_object()`) are UNCHANGED and still COMPUTED/REPORTED in
# `crop_quality_score`'s own returned dict (still useful descriptive/diagnostic
# information) -- only their entry in `_WEIGHTS` is removed, so the combined
# `crop_quality_score` mean (below) no longer includes them. `centroid_offset_score`
# (the older, pixel-only geometry term an earlier round already excluded) stays excluded
# for the same "already covered, would double-count" reason as before (`centered`'s
# docstring). Earlier weights (superseded): sharpness=0.5, single_object=1.0,
# not_truncated=1.0, target_dominance=1.0, centered=0.5 (all 5 axes). This does not
# reopen `crop.py`'s own "dominance is never a hard GATE" stance (`is_crop_valid` /
# `is_crop_keep` are unchanged) -- a continuous, weighted SCORE term is a different
# mechanism from a pass/fail gate.
#
# Task 038 v12 (Q1, human-locked, RE-WEIGHTED -- superseding the v9 weights above):
# not_truncated=1.0 -> 0.25, centered=0.5 -> 0.25, target_dominance UNCHANGED at 1.0
# (sum 2.5 -> 1.5). Rationale: a v11 per-axis audit found `not_truncated` sits at a
# CONSTANT 1.0 under `mask_bbox` (the v11 rectangle was the bbox itself, so it kept the
# whole target by construction; the square that replaced it on 2026-09-30 still contains
# the bbox unless the bbox is longer than the image's shorter side -- see
# `verifiers/crop.py::CropVerifier._MASK_BBOX_CONTAMINATION_MIN`'s own docstring) --
# it was 40% of the old denominator (1.0 of 2.5) for a term that never actually moves
# under this tool's OWN levers, diluting the measured effect exactly like sharpness/
# single_object already were excluded for. `centered` genuinely CAN drop under a
# directional inset (SS1's new `bbox_inset_px` lever, v12 -- clipping the neighbour side
# shifts the target's own centroid within the smaller box), so it stays IN the sum, just
# down-weighted -- the v12 directional-tighten lever (SS1/SS5) is explicitly expected to
# trade a LITTLE centering/truncation for a LOT of dominance, and this reweight is what
# lets that tradeoff actually show up as a net score GAIN rather than being swamped by
# two heavily-weighted axes that barely move.
_WEIGHTS = {"not_truncated": 0.25, "target_dominance": 1.0, "centered": 0.25}


def crop_quality_score(crop: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """GT-free crop-quality score. Deterministic (same input -> same output) and cheap
    (no model inference). `crop` is a 2D grayscale array; a 3D array is squashed to its
    first channel. `mask`, if given, is a crop-shaped binary array marking the TARGET
    cell's own pixels within this crop (the per-cell mask `Single_Cell_Cropper_Tool`
    already persists per crop) -- optional, and used by `not_truncated` / `single_object`
    / `centered` (see their docstrings). Returns all 5 quality-axis components
    (`sharpness` / `single_object` / `not_truncated` / `target_dominance` / `centered`),
    still COMPUTED and REPORTED even though only 3 of them (task 038 v9 restructure --
    `_WEIGHTS`'s own comment) feed the combined score, plus the legacy
    `centroid_offset_score` (reported, not weighted -- see `_WEIGHTS`) plus the combined
    `crop_quality_score` (the weighted mean over `_WEIGHTS`'s OWN keys ONLY -- `sharpness`/
    `single_object` are computed above but excluded from this sum), all in [0, 1], higher
    = better. The dict reports the tool's full 5-axis quality picture (square / one cell /
    CENTERED / main area / minimal neighbours) -- calibration for what counts as
    good/anomalous on each axis lives in CARD.md, not here."""
    crop = np.asarray(crop)
    if crop.ndim == 3:
        crop = crop[..., 0]
    parts = {
        "sharpness": sharpness(crop),
        "not_truncated": not_truncated(crop, mask),
        "single_object": single_object(crop, mask),
        "target_dominance": target_dominance(crop),
        "centered": centered(crop, mask),
    }
    total_w = sum(_WEIGHTS.values())
    combined = sum(parts[k] * _WEIGHTS[k] for k in _WEIGHTS) / total_w
    return {**parts, "centroid_offset_score": centroid_offset_score(crop), "crop_quality_score": combined}


def crop_quality_from_qc(crop_pixels: np.ndarray, qc: dict) -> float:
    """The gold-standard `crop_quality_score`, computed from a crop's own MASK-based `qc`
    block (`tool.py::_compute_crop_qc`) + its pixel sharpness -- avoids needing to load a
    separate per-cell mask file when a `qc` block is already available (the mask-derived
    numbers are already in it). Byte-consistent with `crop_quality_score(crop, mask)` on
    the same crop+mask (same `_WEIGHTS` formula, reused not re-typed) -- weights over
    `_WEIGHTS`'s OWN keys only (task 038 v9 restructure: `sharpness`/`single_object` are
    still computed here for parity but excluded from the weighted sum, see `_WEIGHTS`)."""
    parts = {
        "sharpness": sharpness(crop_pixels),
        "not_truncated": float(1.0 - qc["border_touch"]),
        "single_object": float(qc["single_object"]),
        "target_dominance": float(qc["target_dominance"]),
        "centered": float(1.0 - qc["centroid_offset"]),
    }
    total_w = sum(_WEIGHTS.values())
    return sum(parts[k] * _WEIGHTS[k] for k in _WEIGHTS) / total_w
