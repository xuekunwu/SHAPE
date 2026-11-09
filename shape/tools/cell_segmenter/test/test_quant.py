"""Task 032 (M4): `shape/tools/cell_segmenter/quant.py`'s GT-free quality score.
Co-located with the module under test (committed -- AGENTS.md sec 6 P5).

Covers: quant orders over-seg vs a clean segmentation; deterministic (same input ->
same output); a GT-leak probe (no ground-truth parameter exists anywhere in the
module's public API).
"""

from __future__ import annotations

import inspect

import numpy as np

from shape.tools.cell_segmenter import quant


def _synthetic_image_and_clean_masks(seed: int = 0):
    """5 bright disks on a dark, mildly noisy background -- clean masks match the
    disks exactly (real edges, homogeneous interior, plausible coverage)."""
    rng = np.random.RandomState(seed)
    h, w = 120, 120
    image = np.full((h, w), 20.0)
    centers = [(20, 20), (20, 80), (80, 20), (80, 80), (60, 60)]
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx in centers:
        image[np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < 12] = 200.0
    image = image + rng.normal(0, 3, size=image.shape)

    masks = np.zeros((h, w), dtype=np.int32)
    for i, (cy, cx) in enumerate(centers):
        masks[np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < 12] = i + 1
    return image, masks, centers


def _shatter_into_over_segmentation(masks, centers, image):
    """Same underlying disks, but shattered into many small fragments whose
    boundaries do NOT track the disks' real edges, plus scattered background
    specks -- a synthetic over-segmentation."""
    rng = np.random.RandomState(1)
    h, w = masks.shape
    over = np.zeros_like(masks)
    label = 1
    for cy, cx in centers:
        for dy in range(-12, 12, 3):
            for dx in range(-12, 12, 3):
                y0, x0 = cy + dy, cx + dx
                if 0 <= y0 < h - 2 and 0 <= x0 < w - 2:
                    if np.sqrt(dy ** 2 + dx ** 2) < 12:
                        over[y0:y0 + 2, x0:x0 + 2] = label
                        label += 1
    for _ in range(30):
        y0, x0 = rng.randint(0, h - 2), rng.randint(0, w - 2)
        over[y0:y0 + 2, x0:x0 + 2] = label
        label += 1
    return over


def test_quant_score_orders_clean_above_over_segmented():
    image, clean_masks, centers = _synthetic_image_and_clean_masks()
    over_masks = _shatter_into_over_segmentation(clean_masks, centers, image)

    clean = quant.quant_score(image, clean_masks)
    over = quant.quant_score(image, over_masks)

    assert clean["quant_score"] > over["quant_score"]
    assert clean["boundary_gradient_adherence"] > over["boundary_gradient_adherence"]


def test_quant_score_orders_clean_above_merged_under_segmentation():
    image, clean_masks, _ = _synthetic_image_and_clean_masks()
    merged = np.zeros_like(clean_masks)
    merged[10:95, 10:95] = 1   # one giant mask spanning all 5 disks + background gaps

    clean = quant.quant_score(image, clean_masks)
    under = quant.quant_score(image, merged)

    assert clean["quant_score"] > under["quant_score"]
    assert clean["within_mask_homogeneity"] > under["within_mask_homogeneity"]


def test_quant_score_is_deterministic():
    image, masks, _ = _synthetic_image_and_clean_masks()
    first = quant.quant_score(image, masks)
    second = quant.quant_score(image, masks)
    assert first == second


def test_quant_score_bounded_0_to_1():
    image, masks, _ = _synthetic_image_and_clean_masks()
    out = quant.quant_score(image, masks)
    for k, v in out.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_no_gt_leak_public_api_takes_no_ground_truth_argument():
    """A GT-leak probe: every public function in this module takes ONLY (image,
    masks) -- no ground-truth mask/label/annotation parameter exists anywhere."""
    gt_terms = ("gt", "ground_truth", "groundtruth", "label", "annotation", "truth")
    public_funcs = [f for name, f in vars(quant).items()
                    if inspect.isfunction(f) and not name.startswith("_")
                    and f.__module__ == quant.__name__]
    assert public_funcs, "expected at least quant_score to be public"
    for f in public_funcs:
        params = [p.lower() for p in inspect.signature(f).parameters]
        for term in gt_terms:
            assert not any(term in p for p in params), (
                f"{f.__name__} takes a GT-looking parameter: {params}")
