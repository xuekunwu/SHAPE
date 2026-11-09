"""GT-free segmentation-quality score for `cell_segmenter` (task 032 M4, task 033 recall fix).

The systemic gap vrr.md sec 6 names as priority #1: cell_segmenter's existing
metrics (`shape.agent.verification.verifiers.segmentation.mask_qc`) *describe* the
mask (cell_count, size distribution) but do not *grade* it against the image. This
module is the tool-side `quant.py` convention (shape/tools/<tool>/quant.py) --
GT-free, deterministic, cheap -- built ONLY from the (image, mask) pair, no ground
truth. It is a Verify-phase QUANTITATIVE signal (vrr.md sec 5), not a CARD lever.

Four components, each in [0, 1], higher = better, combined into one `quant_score`:
  - boundary_gradient_adherence: a mask's boundary should sit on a real image edge
    (a wrong/merged/split boundary cuts across a flat region instead).
  - within_mask_homogeneity: pixel intensity inside one mask should be relatively
    uniform (a merged mask spans two objects -> high internal variance).
  - over_under_seg_score: reuses two of the segmentation verifier's own indicators
    -- tiny_mask_fraction (over-segmentation) and how far the mask's own foreground
    coverage sits from a cheap, GT-free Otsu estimate of the image's OWN foreground
    fraction (badly off in either direction indicates over- or under-segmentation).
  - foreground_recall_coverage (task 033): the RECALL term the first three lack --
    what fraction of the image's OWN estimated foreground pixels (Otsu, independent
    of the mask) does SOME mask instance actually claim. Confident under-detection
    (e.g. a strict cellprob threshold that drops whole faint/ambiguous cells) leaves
    those cells' pixels unclaimed by any mask, so this term falls whenever real
    foreground goes uncovered -- however clean the SURVIVING masks look (it does not
    share the survivorship bias that can inflate the other three under aggressive
    filtering; see task 033 CODING_LOG). One-sided by construction: covering MORE
    than the Otsu estimate cannot push it past 1.0 -- over-segmentation stays
    `over_under_seg_score` (`tiny_mask_fraction`)'s job, not this term's.

No ground truth is read anywhere in this module (see the GT-leak probe in the
co-located test): the public API takes only (image, masks).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _sobel_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    gx = ndimage.sobel(image, axis=1)
    gy = ndimage.sobel(image, axis=0)
    return np.hypot(gx, gy)


def _mask_boundary(masks: np.ndarray) -> np.ndarray:
    """Pixels where the label differs from an adjacent pixel (4-connected) -- every
    mask's outer boundary AND any internal boundary between two touching masks."""
    boundary = np.zeros(masks.shape, dtype=bool)
    boundary[:-1, :] |= masks[:-1, :] != masks[1:, :]
    boundary[1:, :] |= masks[:-1, :] != masks[1:, :]
    boundary[:, :-1] |= masks[:, :-1] != masks[:, 1:]
    boundary[:, 1:] |= masks[:, :-1] != masks[:, 1:]
    return boundary


def boundary_gradient_adherence(image: np.ndarray, masks: np.ndarray) -> float:
    """How much more a mask's boundary pixels sit on strong image gradients than a
    random pixel does. Ratio > 1 => boundaries track real edges (good); ratio ~ 1 =>
    boundaries are no better than chance (a merged/split/misplaced boundary).
    Squashed to (0, 1) via ratio/(ratio+1) (0.5 at ratio==1), 0.5 when there is no
    boundary to judge (an empty mask) or the image is flat (gradient is all zero)."""
    boundary = _mask_boundary(masks)
    grad = _sobel_gradient_magnitude(image)
    if not boundary.any() or grad.max() <= 0:
        return 0.5
    grad_norm = grad / (grad.max() + 1e-9)
    ratio = float(grad_norm[boundary].mean()) / (float(grad_norm.mean()) + 1e-9)
    return float(np.clip(ratio / (ratio + 1.0), 0.0, 1.0))


def within_mask_homogeneity(image: np.ndarray, masks: np.ndarray) -> float:
    """1 minus the mean per-mask intensity spread (std, normalized by the image's
    own dynamic range) -- a correctly-bounded single-object mask should be fairly
    uniform inside; a merged/mis-segmented mask spans more than one object and its
    internal variance is higher. 0.0 when there are no masks at all."""
    ids = np.unique(masks)
    ids = ids[ids != 0]
    if ids.size == 0:
        return 0.0
    image = image.astype(np.float64)
    img_range = float(image.max() - image.min()) + 1e-9
    cvs = []
    for i in ids:
        vals = image[masks == i]
        if vals.size >= 2:
            cvs.append(float(vals.std()) / img_range)
    if not cvs:
        return 0.0
    mean_cv = float(np.mean(cvs))
    return float(np.clip(1.0 - mean_cv / 0.25, 0.0, 1.0))


def _tiny_mask_fraction(masks: np.ndarray) -> float:
    ids, counts = np.unique(masks, return_counts=True)
    areas = counts[ids != 0].astype(float)
    if areas.size == 0:
        return 0.0
    med = np.median(areas)
    return float((areas < 0.3 * med).mean())


def _image_foreground_mask(image: np.ndarray) -> np.ndarray:
    """A cheap, GT-free foreground/background split of the RAW image alone (Otsu
    threshold), independent of the mask -- lets the mask be cross-checked against it
    without ground truth. Falls back to a percentile split if Otsu degenerates (e.g.
    a near-constant image). Phase-contrast cells can be darker OR brighter than
    background, and fluorescence puncta sit brighter on a dark background -- take
    the minority class as "foreground" either way, so this holds across modalities."""
    img = image.astype(np.float64)
    try:
        from skimage.filters import threshold_otsu
        thr = threshold_otsu(img)
    except Exception:
        thr = float(np.percentile(img, 60))
    above = img > thr
    return above if bool(above.mean() <= 0.5) else ~above


def _image_foreground_estimate(image: np.ndarray) -> float:
    """The foreground mask's area fraction (see `_image_foreground_mask`)."""
    return float(_image_foreground_mask(image).mean())


def foreground_recall_coverage(image: np.ndarray, masks: np.ndarray) -> float:
    """GT-free RECALL term: of the pixels the RAW IMAGE's own Otsu threshold flags as
    foreground (independent of the mask), what fraction does SOME mask instance
    actually claim. Confident under-detection (a strict cellprob threshold, or an
    oversized min_size cutoff) drops whole faint/ambiguous cells -- their pixels stay
    unclaimed by any mask -- so this term falls specifically and only when real
    foreground goes uncovered, however clean the surviving masks look. One-sided by
    construction (clipped to [0, 1]): covering MORE than the Otsu estimate cannot
    push it past 1.0, so it never rewards over-segmentation (that stays
    `over_under_seg_score`'s `tiny_mask_fraction` job) -- it is not redundant with the
    existing symmetric coverage-vs-Otsu gap, which compares two scalar AREAS and can
    be satisfied by the wrong pixels. 1.0 when the image has no estimated foreground
    to miss."""
    fg = _image_foreground_mask(image)
    fg_total = int(fg.sum())
    if fg_total == 0:
        return 1.0
    covered = int(((masks > 0) & fg).sum())
    return float(np.clip(covered / fg_total, 0.0, 1.0))


def over_under_seg_score(image: np.ndarray, masks: np.ndarray) -> float:
    """Combines the tiny-mask (over-segmentation) indicator with a foreground-
    coverage plausibility check against the image's own Otsu estimate (both
    directions -- masks covering far more OR far less than the image's own
    estimated foreground indicate over- or under-segmentation, respectively)."""
    tiny_score = float(np.clip(1.0 - _tiny_mask_fraction(masks) / 0.5, 0.0, 1.0))
    mask_coverage = float((masks > 0).mean())
    gap = abs(mask_coverage - _image_foreground_estimate(image))
    coverage_score = float(np.clip(1.0 - gap / 0.5, 0.0, 1.0))
    return 0.5 * tiny_score + 0.5 * coverage_score


# task 033: `foreground_recall_coverage` carries weight 2.0 -- equal to the COMBINED
# weight of `boundary_gradient_adherence` + `within_mask_homogeneity` (1.0 + 1.0), the
# two terms whose survivorship bias under aggressive/confident filtering was the
# diagnosed 032 mechanism (task 033 CODING_LOG sec 3). This is sized so a severe
# recall failure (recall -> 0) can, on its own, cancel B and H BOTH simultaneously
# maxing out -- the 2-vs-1 outvote 032 measured cannot recur through B+H alone.
# `over_under_seg_score` is unchanged at 1.0 (a tie-breaker, not itself "fixed").
# Frozen on this reasoning BEFORE any cluster/NeurIPS22-CellSeg measurement --
# not tuned to an eval set (see CODING_LOG).
_WEIGHTS = {"boundary_gradient_adherence": 1.0, "within_mask_homogeneity": 1.0,
           "over_under_seg_score": 1.0, "foreground_recall_coverage": 2.0}


def quant_score(image: np.ndarray, masks: np.ndarray) -> dict:
    """GT-free segmentation-quality score. Deterministic (same input -> same output)
    and cheap (no model inference). Returns the 4 components plus the combined
    `quant_score` (their weighted mean), all in [0, 1], higher = better."""
    parts = {
        "boundary_gradient_adherence": boundary_gradient_adherence(image, masks),
        "within_mask_homogeneity": within_mask_homogeneity(image, masks),
        "over_under_seg_score": over_under_seg_score(image, masks),
        "foreground_recall_coverage": foreground_recall_coverage(image, masks),
    }
    total_w = sum(_WEIGHTS.values())
    combined = sum(parts[k] * _WEIGHTS[k] for k in parts) / total_w
    return {**parts, "quant_score": combined}
