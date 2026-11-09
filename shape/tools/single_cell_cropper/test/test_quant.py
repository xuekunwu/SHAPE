"""Task 035: `shape/tools/single_cell_cropper/quant.py`'s GT-free crop-quality score.
Co-located with the module under test (committed -- AGENTS.md sec 6 P5).

Covers: crop_quality_score orders a clean single-cell crop above a blurred / truncated /
doublet version (built the way the segmenter test builds its own synthetic over/under-seg
masks); the mask-given path also reuses the border-touch/CC-count formulas correctly;
determinism (same input -> same output); [0, 1] bound; a GT-leak probe (no ground-truth
parameter exists anywhere in the module's public API); `crop_quality_from_qc` (the
qc-block shortcut `run_shape_crop.py` uses) is byte-consistent with calling
`crop_quality_score(crop, mask)` directly.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from shape.tools.single_cell_cropper import quant


def _disk_crop(size=96, cy=48, cx=48, r=25, seed=0):
    """A bright disk on a dark, mildly noisy background -- one clean object."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    crop = np.full((size, size), 40.0)
    crop[np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < r] = 200.0
    crop += rng.normal(0, 3, size=crop.shape)
    return crop


def _clean_crop(seed=0):
    return _disk_crop(seed=seed)


def _blurred_crop(seed=1):
    """Same kind of object, heavily Gaussian-blurred -- no more sharp edge."""
    return gaussian_filter(_disk_crop(seed=seed), sigma=8)


def _truncated_crop(seed=2):
    """The SAME disk, but positioned at the crop's top-left corner instead of the
    centre -- part of the object is clipped by the crop boundary, the way a real
    off-center cropper window clips a cell (a real crop is sized around the target, so
    the object stays a substantial fraction of the crop area even when clipped -- unlike
    a tiny far-off object, which would make the crop.py percentile-stretch centroid
    estimate degrade to background noise, see the coding log)."""
    return _disk_crop(cy=5, cx=5, r=20, seed=seed)


def _doublet_crop(size=96, seed=3):
    """Two separate, comparably-sized disks -- two objects in one crop."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    crop = np.full((size, size), 40.0)
    for cy, cx in ((48, 20), (48, 76)):
        crop[np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < 20] = 200.0
    crop += rng.normal(0, 3, size=crop.shape)
    return crop


def test_sharpness_axis_orders_clean_above_blurred_but_is_not_weighted():
    """`sharpness` (the raw axis function) still orders clean above blurred -- blur is a
    real, unchanged property of the pixels. Task 038 v9 restructure: `crop_quality_score`
    no longer WEIGHTS sharpness at all (no geometry lever this tool has ever adjusts
    it) -- it is still computed/reported for diagnostics (see `_WEIGHTS`'s own comment),
    just excluded from the combined score."""
    clean = quant.crop_quality_score(_clean_crop())
    blurred = quant.crop_quality_score(_blurred_crop())
    assert clean["sharpness"] > blurred["sharpness"]
    assert "sharpness" not in quant._WEIGHTS


def test_crop_quality_score_orders_clean_above_truncated():
    clean = quant.crop_quality_score(_clean_crop())
    trunc = quant.crop_quality_score(_truncated_crop())
    assert clean["crop_quality_score"] > trunc["crop_quality_score"]
    assert clean["not_truncated"] > trunc["not_truncated"]


def test_crop_quality_score_orders_clean_above_doublet():
    clean = quant.crop_quality_score(_clean_crop())
    doublet = quant.crop_quality_score(_doublet_crop())
    assert clean["crop_quality_score"] > doublet["crop_quality_score"]
    assert clean["single_object"] > doublet["single_object"]


def test_not_truncated_uses_mask_when_given():
    """The mask path reuses `tool.py`'s border-touch formula directly -- a target mask
    fully inside the crop should score higher than one touching the border, regardless
    of the (here, blank) crop pixels."""
    crop = np.zeros((50, 50))
    inside_mask = np.zeros((50, 50), dtype=bool)
    inside_mask[20:30, 20:30] = True
    border_mask = np.zeros((50, 50), dtype=bool)
    border_mask[0:10, 20:30] = True
    inside = quant.not_truncated(crop, mask=inside_mask)
    border = quant.not_truncated(crop, mask=border_mask)
    assert inside > border
    assert inside == 1.0


def test_single_object_uses_mask_when_given():
    """Two disjoint components in the given mask -> a lower single_object score than a
    mask with just one, regardless of the crop pixels."""
    crop = np.zeros((50, 50))
    one_mask = np.zeros((50, 50), dtype=bool)
    one_mask[10:20, 10:20] = True
    two_mask = one_mask.copy()
    two_mask[30:40, 30:40] = True
    assert quant.single_object(crop, mask=one_mask) == 1.0
    assert quant.single_object(crop, mask=two_mask) == 0.5


def test_centered_uses_mask_when_given():
    """Task 038 M-C: a mask centered in the crop scores 1.0; the SAME-size mask shifted
    off-center scores lower, regardless of the (here, blank) crop pixels."""
    crop = np.zeros((50, 50))
    centered_mask = np.zeros((50, 50), dtype=bool)
    centered_mask[20:30, 20:30] = True             # centroid at (24.5, 24.5) ~ crop center (24.5, 24.5)
    offcenter_mask = np.zeros((50, 50), dtype=bool)
    offcenter_mask[0:10, 0:10] = True              # centroid at (4.5, 4.5), far from center
    centered_score = quant.centered(crop, mask=centered_mask)
    offcenter_score = quant.centered(crop, mask=offcenter_mask)
    assert centered_score > offcenter_score
    assert centered_score == pytest.approx(1.0, abs=1e-6)


def test_centered_falls_back_to_centroid_offset_score_without_mask():
    """No mask -> `centered` reads IDENTICALLY to the existing pixel-based
    `centroid_offset_score` (additive fallback, no new behavior for bare/HPA crops)."""
    crop = _clean_crop()
    assert quant.centered(crop) == quant.centroid_offset_score(crop)


def test_centered_no_mask_signal_returns_zero():
    """An empty (all-False) mask has no centroid to measure -- 0.0, not a crash/NaN."""
    crop = np.zeros((50, 50))
    empty_mask = np.zeros((50, 50), dtype=bool)
    assert quant.centered(crop, mask=empty_mask) == 0.0


def test_weights_match_task_038_v12_reweight():
    """Locks in the exact, human-specified weights (task 038 v9 restructure introduced
    the 3-axis set; task 038 v12, Q1, RE-WEIGHTED them): ONLY the 3 AGENT-ADJUSTED axes --
    not_truncated=0.25, target_dominance=1.0, centered=0.25 (sums to 1.5, was 2.5).
    `sharpness`/`single_object` (no geometry lever this tool has ever adjusts either) and
    `centroid_offset_score` (superseded by `centered`) are deliberately absent from
    `_WEIGHTS`."""
    assert quant._WEIGHTS == {"not_truncated": 0.25, "target_dominance": 1.0, "centered": 0.25}
    assert sum(quant._WEIGHTS.values()) == pytest.approx(1.5)
    assert "sharpness" not in quant._WEIGHTS
    assert "single_object" not in quant._WEIGHTS


def test_crop_quality_score_weights_only_the_3_agent_adjusted_axes():
    """Task 038 v9 restructure: `crop_quality_score` is the weighted mean over ONLY
    `_WEIGHTS`'s 3 keys (not_truncated / target_dominance / centered -- the axes this
    tool's own geometry levers actually move). `sharpness` / `single_object` are still
    COMPUTED and present in the returned dict (diagnostic, unaffected functions), and
    `centroid_offset_score` remains reported too, but NONE of the three feed the weighted
    sum."""
    crop = _clean_crop()
    mask = np.zeros(crop.shape, dtype=bool)
    mask[30:65, 30:65] = True

    out = quant.crop_quality_score(crop, mask=mask)
    for key in ("sharpness", "single_object", "not_truncated", "target_dominance",
                "centered", "centroid_offset_score", "crop_quality_score"):
        assert key in out

    # combined score reproduces the exact weighted-mean formula, byte for byte -- over
    # ONLY the 3 weighted axes.
    parts = {
        "not_truncated": quant.not_truncated(crop, mask),
        "target_dominance": quant.target_dominance(crop),
        "centered": quant.centered(crop, mask),
    }
    weights = {"not_truncated": 0.25, "target_dominance": 1.0, "centered": 0.25}
    expected_combined = sum(parts[k] * weights[k] for k in parts) / sum(weights.values())
    assert out["crop_quality_score"] == pytest.approx(expected_combined)

    # sharpness/single_object/centroid_offset_score are excluded from _WEIGHTS entirely
    # -- they genuinely carry no weight in the combined score.
    assert "sharpness" not in quant._WEIGHTS
    assert "single_object" not in quant._WEIGHTS
    assert "centroid_offset_score" not in quant._WEIGHTS


def test_crop_quality_score_is_deterministic():
    crop = _clean_crop()
    first = quant.crop_quality_score(crop)
    second = quant.crop_quality_score(crop)
    assert first == second


def test_crop_quality_score_bounded_0_to_1():
    for crop in (_clean_crop(), _blurred_crop(), _truncated_crop(), _doublet_crop()):
        out = quant.crop_quality_score(crop)
        for k, v in out.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_crop_quality_from_qc_matches_crop_quality_score():
    """`crop_quality_from_qc` (the qc-block shortcut `run_shape_crop.py` uses to avoid
    loading a separate mask file) must agree with calling `crop_quality_score(crop, mask)`
    directly on the SAME crop+mask -- both apply the SAME `_WEIGHTS` formula to the SAME
    5 axis values."""
    rng = np.random.RandomState(0)
    crop = rng.randint(0, 255, size=(40, 40)).astype(np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255

    direct = quant.crop_quality_score(crop, mask)
    qc = {
        "border_touch": 1.0 - direct["not_truncated"],
        "single_object": direct["single_object"],
        "target_dominance": direct["target_dominance"],
        "centroid_offset": 1.0 - direct["centered"],
    }
    via_qc = quant.crop_quality_from_qc(crop, qc)
    assert via_qc == pytest.approx(direct["crop_quality_score"])


def test_no_gt_leak_public_api_takes_no_ground_truth_argument():
    """A GT-leak probe: no public function in this module takes a ground-truth /
    label / keep-reject parameter -- only (crop, mask) [+ crop_quality_from_qc's `qc`
    dict, which is the tool's OWN mask-derived metadata, not a ground-truth label]."""
    gt_terms = ("gt", "ground_truth", "groundtruth", "label", "annotation", "truth",
                "keep", "reject")
    public_funcs = [f for name, f in vars(quant).items()
                    if inspect.isfunction(f) and not name.startswith("_")
                    and f.__module__ == quant.__name__]
    assert public_funcs, "expected at least crop_quality_score to be public"
    for f in public_funcs:
        params = [p.lower() for p in inspect.signature(f).parameters]
        for term in gt_terms:
            assert not any(term in p for p in params), (
                f"{f.__name__} takes a GT-looking parameter: {params}")
