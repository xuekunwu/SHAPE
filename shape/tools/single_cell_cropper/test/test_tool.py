"""Task 038 M-A: `Single_Cell_Cropper_Tool._compute_crop_qc`'s `single_object` upgrade.

Co-located with `tool.py` (committed, mirrors `test_quant.py`'s co-location convention).
`_compute_crop_qc` needs a full-field `region`/`labeled_mask` + a crop-box (see its own
signature) for its OTHER qc fields, but `single_object` depends ONLY on `cell_mask_crop` (the
target's own per-cell mask crop) -- these tests build the minimal-but-valid inputs needed to
exercise that one field directly, independent of `crop_triage._quant_evidence`'s consumption
of it (covered separately in `test_crop_triage.py`).

Task 038 follow-up (Part A, human-approved OPTION (a)): a second section below covers
`_generate_cell_crops`'s `recenter=True` sizing/placement fix -- half_side is now sized
from the TIGHT bbox, not the multi-axis distance-to-CENTROID (which over-sized the crop
for any cell whose centroid is off-center within its own bbox); `compute_square_crop_
containing_bbox` is now the PRIMARY placement mechanism, not only an edge-clamp fallback.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from skimage.measure import label, regionprops

from shape.tools.single_cell_cropper.tool import (
    Single_Cell_Cropper_Tool,
    _clamp_directional_inset,
    compute_crop_box,
)


def _qc_for_mask(cell_mask_crop: np.ndarray) -> dict:
    """`_compute_crop_qc` needs a `region`/`labeled_mask` too; for a `single_object`-only
    test the simplest valid choice is to derive them from the SAME mask (its own
    `regionprops`/`label`) -- the other qc fields these tests don't assert on."""
    labeled_mask = label(cell_mask_crop > 0)
    region = regionprops(labeled_mask)[0]
    h, w = cell_mask_crop.shape
    return Single_Cell_Cropper_Tool._compute_crop_qc(
        region, region.label, labeled_mask, cell_mask_crop, 0, 0, h, w)


def test_single_object_one_piece_is_1():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    assert _qc_for_mask(mask)["single_object"] == 1.0


def test_single_object_two_significant_pieces_is_half():
    """Two comparably-sized (both >= 40% of the largest) pieces -> the target-intactness/
    fragmentation cut fires at exactly 1/2 -- mirrors `quant.py::single_object`'s own n=2
    boundary."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:18, 2:18] = 255      # piece A, 16x16
    mask[22:38, 22:38] = 255    # piece B, 16x16 (same size -> both significant)
    assert _qc_for_mask(mask)["single_object"] == 0.5


def test_single_object_ignores_a_small_insignificant_speck():
    """A tiny second component (< 40% of the largest) does not count as fragmentation."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:34, 2:34] = 255      # large piece, 32x32 = 1024 px
    mask[36:38, 36:38] = 255    # speck, 2x2 = 4 px (< 0.4 * 1024)
    assert _qc_for_mask(mask)["single_object"] == 1.0


# --------------------------------------------------------------------------------- #
# Task 038 follow-up (VLM-hallucination-fix round, v6 the cluster finding): `n_complete_
# cells`/`foreground_fraction` -- the DATA gates `crop_triage.py`'s hard veto reads to
# CORROBORATE (never itself drive) a VLM's reject/tighten proposal. Unlike the
# `single_object` tests above (single-label mask, no full-field context needed),
# `n_complete_cells` needs a MULTI-label full-field `labeled_mask` bigger than the crop
# box, so a neighbour can genuinely be a "whole cell inside the crop" vs. a "sliver
# clipped by the crop edge" -- these tests build that directly.
# --------------------------------------------------------------------------------- #

def _qc_multi_label(full_mask: np.ndarray, target_label: int, crop_box: tuple,
                    min_area: int = 0) -> dict:
    """`full_mask`: a full-field label image (>=2 distinct labels). `crop_box` =
    (new_minr, new_minc, new_maxr, new_maxc). Builds the target's own `region`/
    `cell_mask_crop` from `target_label` and calls `_compute_crop_qc` exactly as
    `_generate_cell_crops` does at its own call site."""
    new_minr, new_minc, new_maxr, new_maxc = crop_box
    region = next(r for r in regionprops(full_mask) if r.label == target_label)
    cell_mask_crop = ((full_mask[new_minr:new_maxr, new_minc:new_maxc] == target_label)
                      .astype(np.uint8) * 255)
    return Single_Cell_Cropper_Tool._compute_crop_qc(
        region, target_label, full_mask, cell_mask_crop,
        new_minr, new_minc, new_maxr, new_maxc, min_area)


def test_n_complete_cells_counts_a_whole_neighbour_fully_inside_the_crop():
    """A crop containing the target PLUS one whole, fully-contained neighbour cell (the
    genuine, mask-corroborated doublet case) -> n_complete_cells == 2."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[10:30, 10:30] = 1   # target: 20x20=400px, fully inside the 40x40 crop below
    full_mask[32:38, 32:38] = 2   # whole neighbour: 6x6=36px, also fully inside the crop
    qc = _qc_multi_label(full_mask, target_label=1, crop_box=(0, 0, 40, 40))
    assert qc["n_complete_cells"] == 2


def test_n_complete_cells_ignores_a_sliver_neighbour_clipped_by_the_crop_edge():
    """A neighbour only PARTIALLY inside the crop box (a sliver -- most of its own
    full-field mask sits outside the crop) does NOT count as a complete cell -- only the
    fully-contained target itself does -- n_complete_cells == 1."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[10:30, 10:30] = 1   # target: fully inside the 40x40 crop below
    full_mask[35:55, 35:55] = 2   # neighbour: 20x20=400px total, only 5x5=25px inside crop
    qc = _qc_multi_label(full_mask, target_label=1, crop_box=(0, 0, 40, 40))
    assert qc["n_complete_cells"] == 1


def test_n_complete_cells_excludes_a_whole_but_undersized_neighbour_below_min_area():
    """A whole (fully-contained) neighbour cell is only a corroborated complete cell if
    its own full-field area also clears `min_area` -- a tiny fully-contained speck is
    noise, not a genuine second cell."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[10:30, 10:30] = 1   # target: area 400
    full_mask[32:35, 32:35] = 2   # tiny whole neighbour: area 9, below min_area=50
    qc = _qc_multi_label(full_mask, target_label=1, crop_box=(0, 0, 40, 40), min_area=50)
    assert qc["n_complete_cells"] == 1


def test_foreground_fraction_is_all_cell_mask_pixels_over_crop_area():
    """`foreground_fraction` = ALL cell-mask foreground pixels in the crop (not just the
    target's own) / crop area -- near 0 only when the crop is genuinely empty/debris."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[10:30, 10:30] = 1   # target: 20x20=400px, inside a 40x40=1600px crop
    qc = _qc_multi_label(full_mask, target_label=1, crop_box=(0, 0, 40, 40))
    assert qc["foreground_fraction"] == pytest.approx(400 / 1600)


def test_foreground_fraction_near_zero_on_a_genuinely_empty_crop():
    """A crop box that barely clips the target's own edge (almost no real cell mask
    inside it) -- the `empty_debris` data gate's own intended trigger case."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[10:30, 10:30] = 1   # target sits entirely OUTSIDE this crop box
    qc = _qc_multi_label(full_mask, target_label=1, crop_box=(40, 40, 60, 60))
    assert qc["foreground_fraction"] == 0.0


# --------------------------------------------------------------------------------- #
# Task 038 follow-up (Part A, human-approved OPTION (a)): `_generate_cell_crops`'s
# `recenter=True` sizing/placement fix. Both tests recompute the OLD max_dist-based
# formula's own crop box INLINE (via the SAME, unchanged `compute_crop_box` primitive)
# rather than depending on git history, so they stay meaningful indefinitely.
# --------------------------------------------------------------------------------- #

def _write_pair(data_dir: Path, name: str, mask: np.ndarray) -> tuple[Path, Path]:
    images_dir, labels_dir = data_dir / "images", data_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image = np.full(mask.shape, 40.0)
    image[mask > 0] = 200.0
    image = image.astype(np.uint8)
    image_path, label_path = images_dir / f"{name}.tif", labels_dir / f"{name}_label.tiff"
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(label_path, mask.astype(np.uint16))
    return image_path, label_path


def _old_max_dist_crop_box(bbox_mask: np.ndarray, eff_margin: int,
                           img_height: int, img_width: int) -> tuple:
    """The OLD (pre-Part-A) recenter=True formula, recomputed independently: half_side
    from the multi-axis distance-to-CENTROID (`max_dist`), placed via `compute_crop_box`
    (unchanged) -- what `_generate_cell_crops` produced before this fix."""
    props = regionprops(bbox_mask)[0]
    minr, minc, maxr, maxc = props.bbox
    cr, cc = props.centroid
    max_dist = max(cr - minr, maxr - cr, cc - minc, maxc - cc, 0.0)
    old_half_side = int(np.ceil(max_dist + eff_margin))
    return compute_crop_box(cr, cc, old_half_side, img_height, img_width)


def test_generate_cell_crops_symmetric_cell_byte_identical_sizing(tmp_path):
    """(i) Zero regression: a cell that IS symmetric about its own bbox (region.centroid
    coincides with the bbox's own discrete pixel-index midpoint -- a filled square) gets
    the SAME crop_bbox under the NEW tight-bbox sizing as the OLD max_dist-based formula
    would have produced -- `max_dist` already reduces to the tight-bbox value in this
    case, so tightening removes zero size here."""
    mask = np.zeros((200, 200), dtype=np.uint16)
    mask[40:60, 40:60] = 1     # 20x20 symmetric square, comfortably interior
    image_path, label_path = _write_pair(tmp_path / "data", "sym", mask)

    expected_box = _old_max_dist_crop_box(mask, eff_margin=2, img_height=200, img_width=200)

    tool = Single_Cell_Cropper_Tool()
    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=50, margin=2, margin_mode="tight", recenter=True, margin_frac=0.0,
        query_cache_dir=str(tmp_path / "out"),
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    assert len(meta["cell_metadata"]) == 1
    assert tuple(meta["cell_metadata"][0]["crop_bbox"]) == expected_box


def test_generate_cell_crops_asymmetric_cell_tighter_box_and_higher_dominance(tmp_path):
    """(ii) An L-shaped target (bbox 60x60, centroid pulled toward the bottom-right, area
    2700) plus a small neighbour placed exactly in the region the OLD max_dist-based
    formula's oversized crop reached (but the NEW tight-bbox crop does not): the
    neighbour's own bbox is fully inside the OLD-formula box (would have diluted
    target_dominance) but entirely OUTSIDE the NEW crop -- NEW's own crop is smaller
    (tighter) AND its target_dominance is measurably higher (1.0, vs the OLD box's own
    0.9677 -- see the coding log's measured before/after)."""
    mask = np.zeros((200, 200), dtype=np.int64)
    mask[40:100, 40:70] = 1
    mask[70:100, 40:100] = 1
    mask[45:60, 27:33] = 2     # neighbour: inside OLD's crop box, outside NEW's
    image_path, label_path = _write_pair(tmp_path / "data", "asym", mask)

    target_mask = (mask == 1).astype(np.uint16)
    old_box = _old_max_dist_crop_box(target_mask, eff_margin=2, img_height=200, img_width=200)
    old_minr, old_minc, old_maxr, old_maxc = old_box
    old_area = (old_maxr - old_minr) * (old_maxc - old_minc)

    neighbour_mask = (mask == 2).astype(np.uint16)
    n_minr, n_minc, n_maxr, n_maxc = regionprops(neighbour_mask)[0].bbox
    # sanity: the neighbour IS fully inside the OLD box (so it WOULD have diluted OLD's
    # own target_dominance) -- otherwise this fixture would not exercise the fix at all.
    assert old_minr <= n_minr and n_maxr <= old_maxr
    assert old_minc <= n_minc and n_maxc <= old_maxc

    tool = Single_Cell_Cropper_Tool()
    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=50, margin=2, margin_mode="tight", recenter=True, margin_frac=0.0,
        query_cache_dir=str(tmp_path / "out"),
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    target = max(meta["cell_metadata"], key=lambda m: m["area"])
    assert target["area"] == 2700   # sanity: this IS the asymmetric L-shape

    new_minr, new_minc, new_maxr, new_maxc = target["crop_bbox"]
    new_area = (new_maxr - new_minr) * (new_maxc - new_minc)
    assert new_area < old_area   # NEW's crop is genuinely tighter (smaller)

    # the neighbour is entirely OUTSIDE the NEW crop -- excluded, not just shrunk
    assert not (new_minr <= n_minr and n_maxr <= new_maxr
               and new_minc <= n_minc and n_maxc <= new_maxc)

    assert target["qc"]["target_dominance"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------------- #
# `margin_mode="mask_bbox"` -- task 038 v9 restructure (human-locked) made it a
# RECTANGULAR crop tight to the target's own mask bounding box; the human rule of
# 2026-09-30 (every crop the tool writes is square: the analyzer resizes every crop to
# 224 x 224 without keeping the aspect ratio) makes it the SQUARE around that bbox --
# its longer side + 2 x `tight_buffer_px`, centered on the bbox midpoint, RAW pixels (no
# mask-out, no padding). Along the bbox's longer side nothing beyond the bbox (+buffer)
# is included; along its shorter side the square extends past the bbox. `cell_overrides`
# is the live lever, but these tests exercise the geometry mechanism directly,
# independent of the judge.
# --------------------------------------------------------------------------------- #

def test_mask_bbox_excludes_a_neighbour_beyond_its_bbox_that_the_fraction_square_admits(tmp_path):
    """A neighbour just beyond the target's bbox along the bbox's LONGER side sits inside
    the looser default `fraction` square (which admits it) but is fully EXCLUDED by the
    `mask_bbox` square -- not merely shrunk, genuinely absent -- and the crop is square:
    the bbox's longer side + 2 x `tight_buffer_px`, exactly the bbox + buffer along that
    side."""
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[20:50, 20:40] = 1   # target: bbox 30 rows x 20 cols (non-square)
    mask[53:60, 25:35] = 2   # neighbour: 3 px beyond the target's bbox along its longer (row) side
    image_path, label_path = _write_pair(tmp_path / "data", "mb", mask)

    tool = Single_Cell_Cropper_Tool()
    cell_id = "default_mb_cell_0000"
    kwargs = dict(original_image=str(image_path), nuclei_mask=str(label_path),
                  min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False)
    plain = tool.execute(**kwargs, query_cache_dir=str(tmp_path / "out_fraction"))
    fraction = [m for m in json.loads(Path(plain["cell_crops_metadata_path"]).read_text())["cell_metadata"]
                if m["cell_id"] == cell_id][0]
    assert fraction["qc"]["neighbor_contamination"] > 0     # the fraction square admits the neighbour

    result = tool.execute(**kwargs, query_cache_dir=str(tmp_path / "out"),
                          cell_overrides={cell_id: {"margin_mode": "mask_bbox"}})
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    target = [m for m in meta["cell_metadata"] if m["area"] == 600][0]
    assert target["cell_id"] == cell_id

    new_minr, new_minc, new_maxr, new_maxc = target["crop_bbox"]
    assert (new_maxr - new_minr) == (new_maxc - new_minc) == 30 + 2 * 2   # SQUARE: longer side + 2 x buffer
    assert (new_minr, new_maxr) == (20 - 2, 50 + 2)        # exactly the bbox + buffer along the longer side
    # the neighbour (rows 53:60, cols 25:35) is entirely OUTSIDE this crop
    assert new_maxr <= 53
    assert target["qc"]["n_complete_cells"] == 1        # neighbour no longer "complete" in-crop
    assert target["qc"]["target_dominance"] == pytest.approx(1.0, abs=1e-6)
    assert target["qc"]["mask_containment"] == 1.0


def test_mask_bbox_is_inherently_centered_subsumes_recenter(tmp_path):
    """`mask_bbox` needs no separate `recenter` -- the bbox itself IS the crop, so the
    target's centroid sits close to the crop's own center by construction."""
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[20:50, 20:40] = 1
    image_path, label_path = _write_pair(tmp_path / "data", "ctr", mask)

    tool = Single_Cell_Cropper_Tool()
    cell_id = "default_ctr_cell_0000"
    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        query_cache_dir=str(tmp_path / "out"),
        cell_overrides={cell_id: {"margin_mode": "mask_bbox"}},
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    target = meta["cell_metadata"][0]
    assert target["qc"]["centroid_offset"] < 0.1


def test_mask_bbox_crop_of_an_elongated_cell_is_square_and_contains_its_bbox(tmp_path):
    """Human rule (2026-09-30, REPLACES the task 038 v9 rectangle this test used to check
    for -- a non-square `mask_bbox` crop for a non-square object): an elongated cell
    (bbox 60 x 15, aspect 4) gets a SQUARE `mask_bbox` crop -- side = the bbox's longer
    side + 2 x `tight_buffer_px`, centered on the bbox midpoint -- that fully contains its
    bbox, and the crop FILE written to disk is square too."""
    from PIL import Image

    mask = np.zeros((100, 100), dtype=np.int64)
    mask[20:80, 40:55] = 1   # 60 rows x 15 cols, aspect 4
    image_path, label_path = _write_pair(tmp_path / "data", "sq", mask)

    tool = Single_Cell_Cropper_Tool()
    cell_id = "default_sq_cell_0000"
    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        query_cache_dir=str(tmp_path / "out"),
        cell_overrides={cell_id: {"margin_mode": "mask_bbox"}},
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    assert len(meta["cell_metadata"]) == 1   # kept
    target = meta["cell_metadata"][0]
    h, w, _ = target["crop_size"]
    assert h == w == 60 + 2 * 2
    new_minr, new_minc, new_maxr, new_maxc = target["crop_bbox"]
    assert new_minr <= 20 and new_maxr >= 80 and new_minc <= 40 and new_maxc >= 55   # contains the bbox
    assert (new_minr, new_maxr) == (18, 82)                           # the bbox + buffer along its long side
    assert abs((new_minc + new_maxc) / 2.0 - 47.5) <= 0.5               # centered on the bbox midpoint
    assert target["qc"]["mask_containment"] == 1.0
    with Image.open(target["crop_path"]) as written:
        assert written.size == (64, 64)


def test_border_cell_keeps_full_containment_with_a_shifted_square(tmp_path):
    """Human rule (2026-09-30, REPLACES the task 038 architectural FIX round's non-square
    border clip this test used to keep): a `margin_mode="fraction"` crop of a target
    touching the image's top edge is the SAME-SIZE square an interior twin of the cell
    gets, shifted down inside the image -- the target is off-center instead of the crop
    being clipped to a rectangle -- and the target stays a COMPLETE cell
    (`mask_containment == 1.0`); nothing is filtered."""
    mask = np.zeros((60, 60), dtype=np.int64)
    mask[0:10, 25:35] = 1     # target touching the top edge (a border clip would cut the row axis)
    mask[30:40, 25:35] = 2    # its interior twin (same size, far from every border)
    image_path, label_path = _write_pair(tmp_path / "data", "border_sq", mask)

    tool = Single_Cell_Cropper_Tool()
    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        query_cache_dir=str(tmp_path / "out"),
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    by_id = {m["cell_id"]: m for m in meta["cell_metadata"]}
    border, twin = by_id["default_border_sq_cell_0000"], by_id["default_border_sq_cell_0001"]
    assert tuple(border["crop_size"]) == tuple(twin["crop_size"]) == (14, 14, 1)   # square, same size
    # half side 5 + margin round(0.2 x 11.28) = 2 around the bbox midpoint (5, 30) would reach row -2:
    # the square shifts down to row 0 instead of being clipped to 12 x 14
    assert border["crop_bbox"] == [0, 23, 14, 37]
    assert border["qc"]["mask_containment"] == 1.0     # the target is a COMPLETE cell
    assert border["qc"]["centroid_offset"] > twin["qc"]["centroid_offset"]   # off-center, not clipped
    # the stats stay coherent: nothing attributes this cell to any filter bucket.
    assert result["filtered_by_area"] == 0
    assert result["filtered_by_border"] == 0
    assert result["invalid_crop_data"] == 0
    assert "filtered_nonsquare" not in result   # the old square-only filter's stat stays retired


def test_mask_path_is_threaded_into_json_metadata(tmp_path):
    """Task 038 v9 restructure: `mask_path` is now threaded into the JSON-persisted
    `cell_metadata` (previously only on the separate, non-JSON `CellCrop` object) so
    `crop_triage.py`'s overlay-drawing judge evidence can find the target's own per-cell
    mask file at judge time."""
    mask = np.zeros((60, 60), dtype=np.int64)
    mask[10:30, 10:30] = 1
    image_path, label_path = _write_pair(tmp_path / "data", "mp", mask)

    tool = Single_Cell_Cropper_Tool()
    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, query_cache_dir=str(tmp_path / "out"),
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    mask_path = meta["cell_metadata"][0]["mask_path"]
    assert mask_path and Path(mask_path).exists()


def test_saved_per_cell_mask_is_a_solid_blob_not_a_2x_subsampled_lattice(tmp_path):
    """Task 038 CRITICAL fix (human directive, found by auditing the real v13d run's
    saved masks): `_generate_cell_crops`'s own per-cell mask save used to force a
    uint16 buffer through PIL's 8-bit `mode='L'` -- `cell_mask = np.zeros_like(mask)`
    INHERITS the full-field label mask's own dtype, which is uint16 for real
    NeurIPS22-CellSeg data (the FULL-image label file this tool actually loads, not a
    hand-built synthetic array). PIL's `'L'` mode reads 1 byte/pixel; forcing it onto a
    2-bytes/pixel uint16 buffer silently re-interprets every OTHER byte as a separate
    pixel, producing a 2x-subsampled lattice (the right silhouette, but half the real
    pixels and NO connectivity between them) -- measured in the real run: 100% of
    sampled masks across every image, both arms (mean horizontal-foreground-neighbour
    fraction 0.00).

    This is the THIRD time this exact bug CLASS has shipped (a dtype assumption a
    synthetic uint8 test fixture can never catch) -- the existing overlay/montage tests
    all pass because they hand-build uint8 masks directly; the REAL pipeline feeds
    uint16. This fixture is uint16 EXPLICITLY, and explicitly VERIFIED to be so below
    (via `_write_pair`, which already writes the on-disk label file as uint16 -- a
    synthetic-uint8-only fixture must never be able to pass this test again), and drives
    the REAL `_generate_cell_crops` path end to end (not a hand-called helper), then
    reads the SAVED PNG back off disk and checks its own pixel structure directly --
    the exact gap that let this ship undetected through every OTHER passing test (all
    of which only ever checked the IN-MEMORY qc/geometry, never the saved mask file's
    own bytes)."""
    from skimage.io import imread

    mask = np.zeros((60, 60), dtype=np.int64)
    mask[10:40, 10:40] = 1   # a real, sizeable target -- 30x30, plenty of interior pixels
    image_path, label_path = _write_pair(tmp_path / "data", "u16mask", mask)

    # Sanity: the fixture genuinely exercises a uint16-sourced mask, the SAME dtype
    # `_generate_cell_crops`'s own `mask` parameter receives for real data -- not just
    # assumed from `_write_pair`'s own docstring.
    assert tifffile.imread(label_path).dtype == np.uint16

    tool = Single_Cell_Cropper_Tool()
    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, query_cache_dir=str(tmp_path / "out"),
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    target = meta["cell_metadata"][0]
    saved_mask = np.asarray(imread(target["mask_path"]))

    fg = saved_mask > 0
    assert fg.any()   # sanity: the saved mask is not blank

    # SOLIDITY: every foreground pixel must have at least one foreground 4-neighbour --
    # the lattice bug leaves EVERY foreground pixel isolated (zero foreground
    # 4-neighbours at all, since only every-other pixel in each row/column survives
    # the byte-reinterpretation).
    has_fg_neighbour = np.zeros_like(fg)
    has_fg_neighbour[:-1, :] |= fg[1:, :]
    has_fg_neighbour[1:, :] |= fg[:-1, :]
    has_fg_neighbour[:, :-1] |= fg[:, 1:]
    has_fg_neighbour[:, 1:] |= fg[:, :-1]
    assert np.all(has_fg_neighbour[fg])

    # PIXEL COUNT: must exactly match ground truth -- computed INDEPENDENTLY from the
    # original numpy mask + the metadata's own `crop_bbox` (the lattice bug roughly
    # halves the count, since it drops every other pixel along each scan line).
    new_minr, new_minc, new_maxr, new_maxc = target["crop_bbox"]
    expected_count = int(np.count_nonzero(mask[new_minr:new_maxr, new_minc:new_maxc] == 1))
    assert int(np.count_nonzero(fg)) == expected_count


def test_saved_baseline_mask_is_also_a_solid_blob_not_a_lattice(tmp_path):
    """The SAME uint16-buffer-read-as-8-bit corruption, for the OTHER per-cell mask
    save site this tool has -- `baseline_mask_path` (task 038 v13's own blinded A/B
    render), a SEPARATE `Image.fromarray` call on a slice of the SAME `cell_mask`
    array. Reuses the L-shaped fixture (a real inset genuinely applies, so
    `baseline_mask_path` is actually rendered) with a uint16-sourced mask, same
    solidity + exact-count checks as the primary mask above."""
    from skimage.io import imread

    mask = _l_shaped_target_and_corner_neighbour()
    image_path, label_path = _write_pair(tmp_path / "data", "u16baseline", mask)
    assert tifffile.imread(label_path).dtype == np.uint16

    tool = Single_Cell_Cropper_Tool()
    cell_id = "default_u16baseline_cell_0000"
    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        query_cache_dir=str(tmp_path / "out"),
        cell_overrides={cell_id: {"margin_mode": "mask_bbox", "bbox_inset_px": {"right": 8}}},
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    target = [m for m in meta["cell_metadata"] if m["cell_id"] == cell_id][0]
    assert target["baseline_mask_path"]   # sanity: a genuine inset applied, baseline rendered

    saved_mask = np.asarray(imread(target["baseline_mask_path"]))
    fg = saved_mask > 0
    assert fg.any()

    has_fg_neighbour = np.zeros_like(fg)
    has_fg_neighbour[:-1, :] |= fg[1:, :]
    has_fg_neighbour[1:, :] |= fg[:-1, :]
    has_fg_neighbour[:, :-1] |= fg[:, 1:]
    has_fg_neighbour[:, 1:] |= fg[:, :-1]
    assert np.all(has_fg_neighbour[fg])


# --------------------------------------------------------------------------------- #
# Task 038 v12 (human-locked -- DELIBERATELY REVERSES the "geometry is 100% pure-data,
# no VLM" lock, CARD.md SS1/SS5's own updated rationale): the per-cell DIRECTIONAL
# `bbox_inset_px` lever -- applied AFTER `margin_mode="mask_bbox"`'s own square, clips the
# box on the neighbour side(s) ONLY (not symmetric), each cut paired with an equal cut on
# the perpendicular axis, split evenly between its two sides, so the crop stays square and
# the target centred on that axis (human rules, 2026-09-30), so the VLRR loop can keep
# tightening a still-contaminated mask_bbox crop. These tests
# build an IRREGULAR target shape (an "L") so a neighbour can sit genuinely INSIDE the
# target's own bbox rectangle -- the exact v11 human-audit finding this lever exists to
# fix (mask_bbox alone cannot exclude it; only a directional inset can).
# --------------------------------------------------------------------------------- #

def _l_shaped_target_and_corner_neighbour() -> np.ndarray:
    """Target (label 1) = an 'L' shape whose own bbox is rows[20:50] cols[20:40], but
    which does NOT occupy its own top-right corner (rows[20:45] cols[28:40]) -- that
    empty corner is where the neighbour (label 2) sits, INSIDE the target's own bbox
    rectangle, `mask_bbox` alone (no inset) cannot exclude it. The neighbour spans most
    of the target's own vertical extent (roughly straddling its centroid row), so it is
    unambiguously `contamination_side="right"` (verified directly, see
    `test_contamination_side_picks_the_correct_edge_right` below, sharing this exact
    construction)."""
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[20:50, 20:28] = 1    # target: narrow column, full height
    mask[45:50, 28:40] = 1    # target: a foot extending right at the bottom (widens the bbox)
    mask[20:45, 30:38] = 2    # neighbour: inside the target's own bbox, to the right
    return mask


def _vertical_bar_and_right_neighbour() -> np.ndarray:
    """Target (label 1) = a vertical bar, bbox rows[20:50] cols[30:40]: its mask_bbox square
    (rows and cols 18:52) extends past the bbox on both sides of its shorter (column) axis,
    and the neighbour (label 2) sits there, right of the bbox and inside the square."""
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[20:50, 30:40] = 1
    mask[25:45, 44:50] = 2
    return mask


def test_bbox_inset_px_excludes_neighbour_directionally_and_raises_dominance(tmp_path):
    """VALIDATE (task 038 v12, sec. 2.1/2.2; square-preserving since the human rules of
    2026-09-30): a `bbox_inset_px={"right": N}` override cuts N px off the right edge and
    pairs it with an N px cut on the perpendicular axis, split evenly between top and
    bottom so the target stays centred on that axis -- directional, NOT symmetric (the
    left edge is UNCHANGED) -- so the crop stays square; `target_dominance` RISES relative
    to the plain mask_bbox square and `target_retention` drops but stays >= the hard floor
    (task 038 v13: 0.90)."""
    mask = _vertical_bar_and_right_neighbour()
    image_path, label_path = _write_pair(tmp_path / "data", "inset", mask)
    tool = Single_Cell_Cropper_Tool()
    cell_id = "default_inset_cell_0000"

    baseline = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        query_cache_dir=str(tmp_path / "out_baseline"),
        cell_overrides={cell_id: {"margin_mode": "mask_bbox"}},
    )
    base_meta = json.loads(Path(baseline["cell_crops_metadata_path"]).read_text())
    base_target = base_meta["cell_metadata"][0]
    assert base_target["qc"]["contamination_side"] == "right"
    base_minr, base_minc, base_maxr, base_maxc = base_target["crop_bbox"]
    assert (base_minr, base_minc, base_maxr, base_maxc) == (18, 18, 52, 52)   # the plain mask_bbox square

    insetted = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        query_cache_dir=str(tmp_path / "out_inset"),
        cell_overrides={cell_id: {"margin_mode": "mask_bbox", "bbox_inset_px": {"right": 6}}},
    )
    ins_meta = json.loads(Path(insetted["cell_crops_metadata_path"]).read_text())
    ins_target = ins_meta["cell_metadata"][0]
    ins_minr, ins_minc, ins_maxr, ins_maxc = ins_target["crop_bbox"]

    # DIRECTIONAL: the RIGHT edge moved by 6, its pair split 3 / 3 on the rows -- the left edge UNCHANGED.
    assert ins_minr == base_minr + 3
    assert ins_minc == base_minc
    assert ins_maxr == base_maxr - 3
    assert ins_maxc == base_maxc - 6
    assert ins_maxr - ins_minr == ins_maxc - ins_minc                    # still square

    assert ins_target["qc"]["target_dominance"] > base_target["qc"]["target_dominance"]
    assert ins_target["qc"]["target_retention"] < 1.0
    assert ins_target["qc"]["target_retention"] >= 0.90
    assert ins_target["bbox_inset_px"] == {"right": 6, "top": 3, "bottom": 3}   # requested + its even pair


def test_bbox_inset_px_retention_floor_scales_back_never_drops(tmp_path):
    """VALIDATE (task 038 v12, §2.5's own "never discard" mandate -- "rather than just throwing it away"): a
    requested inset LARGE enough to breach the retention floor (task 038 v13: 0.90) does
    NOT drop the cell -- it is SCALED BACK (never the originally-requested amount) until
    measured retention clears the floor again, and the crop is KEPT in the output
    either way."""
    mask = _l_shaped_target_and_corner_neighbour()
    image_path, label_path = _write_pair(tmp_path / "data", "huge_inset", mask)
    tool = Single_Cell_Cropper_Tool()
    cell_id = "default_huge_inset_cell_0000"

    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        query_cache_dir=str(tmp_path / "out"),
        # 20px off an 24px-wide box would breach the floor if applied verbatim.
        cell_overrides={cell_id: {"margin_mode": "mask_bbox", "bbox_inset_px": {"right": 20}}},
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    assert len(meta["cell_metadata"]) == 2   # BOTH cells present -- KEPT, not dropped
    target = [m for m in meta["cell_metadata"] if m["cell_id"] == cell_id][0]
    assert target["qc"]["target_retention"] >= 0.90
    applied = target["bbox_inset_px"]["right"]
    assert 0 < applied < 20   # scaled BACK from the requested 20 -- never the full amount,
                              # never zero (there WAS genuine headroom to tighten some)
    h, w, _ = target["crop_size"]
    assert h == w             # the scaled-back cut stays square (its paired cut scaled with it)


def test_baseline_crop_path_rendered_for_a_genuinely_tightened_cell(tmp_path):
    """Task 038 v13, human-directed: a cell with a NONZERO applied `bbox_inset_px` gets
    a SEPARATE, real `baseline_crop_path` image on disk -- the plain, zero-inset
    mask_bbox box -- so `crop_triage.ab_recommend_crop`'s blinded A/B channel has BOTH
    real images to compare, no history threading needed."""
    mask = _l_shaped_target_and_corner_neighbour()
    image_path, label_path = _write_pair(tmp_path / "data", "baseline_render", mask)
    tool = Single_Cell_Cropper_Tool()
    cell_id = "default_baseline_render_cell_0000"

    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        query_cache_dir=str(tmp_path / "out"),
        cell_overrides={cell_id: {"margin_mode": "mask_bbox", "bbox_inset_px": {"right": 8}}},
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    target = [m for m in meta["cell_metadata"] if m["cell_id"] == cell_id][0]
    assert target["baseline_crop_path"]
    assert Path(target["baseline_crop_path"]).exists()
    assert target["baseline_crop_path"] != target["crop_path"]   # a genuinely SEPARATE image
    # its own matching mask -- correctly sized for `_overlay_target_on_crop`'s own
    # shape check, so the A/B channel's baseline panel actually gets the colourful
    # overlay too (a mismatched mask would silently fall back to no overlay).
    assert target["baseline_mask_path"]
    assert Path(target["baseline_mask_path"]).exists()
    import numpy as np
    from skimage.io import imread
    baseline_img = np.asarray(imread(target["baseline_crop_path"]))
    baseline_mask = np.asarray(imread(target["baseline_mask_path"]))
    assert baseline_mask.shape[:2] == baseline_img.shape[:2]
    assert baseline_img.shape[0] == baseline_img.shape[1]   # the baseline render is the mask_bbox square


def test_baseline_crop_path_absent_for_a_plain_mask_bbox_cell_with_no_inset(tmp_path):
    """A `mask_bbox` cell with NO requested inset has nothing to compare against (baseline
    == candidate) -- `baseline_crop_path` stays None, no wasted render."""
    mask = _l_shaped_target_and_corner_neighbour()
    image_path, label_path = _write_pair(tmp_path / "data", "no_inset", mask)
    tool = Single_Cell_Cropper_Tool()
    cell_id = "default_no_inset_cell_0000"

    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        query_cache_dir=str(tmp_path / "out"),
        cell_overrides={cell_id: {"margin_mode": "mask_bbox"}},
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    target = [m for m in meta["cell_metadata"] if m["cell_id"] == cell_id][0]
    assert target["baseline_crop_path"] is None


# --------------------------------------------------------------------------------- #
# Task 038 architectural FIX round (coordinator-directed): `auto_tighten` -- the
# per-cell directional `bbox_inset_px` lever now ITERATES INSIDE one `execute()` call
# (`_auto_tighten_mask_bbox`), instead of the VLRR loop applying one cut per real tool
# call and re-rendering the WHOLE image every round. Real evidence this round fixes:
# `cell_00860` (real cluster run) took 6 VLRR trials, the LAST a literal no-op;
# `cell_00994` (259 cells) hit `max_iters` while STILL improving, wasting 89% of the
# run's own time on repeated whole-image re-renders. Carries FIX #2's own plateau/
# regression semantics (`verifiers/crop.py::CropVerifier._tighten_delta`) into this
# internal loop -- the first two tests below exercise it end-to-end on REAL masks
# (reusing `test_contamination_inset_px_stops_at_the_floor_with_residual_contamination_
# on_a_two_side_mask`'s and `test_contamination_inset_px_fully_excludes_a_small_
# neighbour_without_touching_target`'s own fixtures from the `_qc_multi_label` section
# below -- SAME numbers, independently cross-checked here through the full `execute()`
# pipeline rather than assumed to carry over); the plateau/regression/floor-backstop
# tests use a SCRIPTED `_compute_crop_qc` stub (mirrors `test_crop_triage.py`'s own
# scripted-`tool_runner` pattern for the verifier-level version of this same fix) since
# a real mask geometry does not naturally plateau or regress within a small, hand-built
# fixture -- the coordinator's own test spec asks for a "crafted" scenario.
# --------------------------------------------------------------------------------- #

def test_auto_tighten_converges_via_no_headroom_on_a_two_side_mask(tmp_path):
    """End-to-end (`execute()`, real mask): a target with contamination on TWO
    different sides (a right and a bottom neighbour). Square-preserving since the human
    rules of 2026-09-30: the first projected step (right 16, which fully excludes the
    right neighbour) is paired with a 16 px cut on the perpendicular axis, split evenly
    8 / 8 between top and bottom (the target stays centred on that axis); that square
    keeps exactly 0.90 of the target, the bottom neighbour is still partly inside it, and
    `contamination_inset_px` caps to 0 (no headroom left), so the internal loop stops
    there: ONE iteration, a 24 x 24 square with genuine residual contamination -- the same
    end state `test_contamination_inset_px_stops_at_the_floor_with_residual_contamination_
    on_a_two_side_mask` reaches on the SAME fixture, with the paired rows.
    `tight_buffer_px=15` (vs. the tool's own default 2) so the PLAIN, zero-inset
    mask_bbox square (`buf_minr`/etc) already reaches both neighbours at all -- with the
    tiny default buffer, a neighbour 5px away from the target's own bbox edge would be
    excluded by construction before `auto_tighten` ever runs, which would test the
    buffer, not the lever this round adds."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[25:35, 25:35] = 1     # target: 10x10, centroid (29.5, 29.5)
    full_mask[20:40, 40:46] = 2     # neighbour A: right side
    full_mask[40:45, 20:40] = 3     # neighbour B: bottom side
    image_path, label_path = _write_pair(tmp_path / "data", "auto_two_side", full_mask)
    tool = Single_Cell_Cropper_Tool()
    cell_id = "default_auto_two_side_cell_0000"

    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        tight_buffer_px=15,
        query_cache_dir=str(tmp_path / "out"),
        cell_overrides={cell_id: {"margin_mode": "mask_bbox", "auto_tighten": True}},
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    target = [m for m in meta["cell_metadata"] if m["cell_id"] == cell_id][0]
    assert target["bbox_inset_px"] == {"right": 16, "top": 8, "bottom": 8}
    assert target["auto_tighten_iterations"] == 1
    assert target["crop_bbox"] == [18, 10, 42, 34]                # the 40 px square cut to 24 x 24
    assert target["qc"]["target_retention"] == pytest.approx(0.90)
    assert target["qc"]["contamination_side"] == "bottom"        # genuine residual
    assert target["qc"]["neighbor_contamination"] > 0.2           # still contaminated
    # the retention floor is NEVER breached -- the defining safety guarantee.
    assert target["qc"]["mask_containment"] >= 0.90


def test_auto_tighten_converges_to_clean_on_a_single_small_neighbour(tmp_path):
    """End-to-end (`execute()`, real mask): a neighbour small/far enough that fully
    excluding it never touches the target at all -- `auto_tighten` converges to
    GENUINELY CLEAN (`contamination_side is None`, `neighbor_contamination==0.0`) in
    ONE internal iteration, target retention fully UNCHANGED at 1.0 (matches
    `test_contamination_inset_px_fully_excludes_a_small_neighbour_without_touching_
    target`'s own real numbers). The right cut of 10 is paired with 10 px on the rows,
    split evenly 5 / 5, and the crop stays a 30 x 30 square. `tight_buffer_px=15` for the
    SAME reason as the two-side test above -- so the plain square already reaches the
    neighbour."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[25:35, 25:35] = 1   # target: 10x10, centroid (29.5, 29.5)
    full_mask[20:40, 40:46] = 2   # neighbour: clean right side
    image_path, label_path = _write_pair(tmp_path / "data", "auto_clean", full_mask)
    tool = Single_Cell_Cropper_Tool()
    cell_id = "default_auto_clean_cell_0000"

    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, margin_mode="fraction", margin_frac=0.20, recenter=False,
        tight_buffer_px=15,
        query_cache_dir=str(tmp_path / "out"),
        cell_overrides={cell_id: {"margin_mode": "mask_bbox", "auto_tighten": True}},
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    target = [m for m in meta["cell_metadata"] if m["cell_id"] == cell_id][0]
    assert target["bbox_inset_px"] == {"right": 10, "top": 5, "bottom": 5}
    assert target["crop_bbox"] == [15, 10, 45, 40]                # 30 x 30
    assert target["auto_tighten_iterations"] == 1
    assert target["qc"]["neighbor_contamination"] == 0.0
    assert target["qc"]["contamination_side"] is None
    assert target["qc"]["target_retention"] == 1.0   # target itself never touched


class _ScriptedQC:
    """Test-only stand-in for `Single_Cell_Cropper_Tool._compute_crop_qc` (task 038
    architectural FIX round) -- returns a pre-scripted sequence of qc dicts, one per
    call, clamped to the last entry if over-called (a bug would call this MORE times
    than expected, not fewer, so clamping surfaces that as a wrong final result rather
    than a crash). Set via `monkeypatch.setattr(tool, "_compute_crop_qc", ...)`: an
    INSTANCE attribute shadows the class's own `@staticmethod` for that one instance,
    so `self._compute_crop_qc(...)` inside `_auto_tighten_mask_bbox` resolves to this
    stub -- mirrors `test_crop_triage.py`'s own scripted-`tool_runner` pattern for the
    verifier-level version of this same fix, applied here at the tool level since a
    real mask does not naturally plateau/regress within a small hand-built fixture."""

    def __init__(self, sequence: list[dict]):
        self.sequence = sequence
        self.calls = 0

    def __call__(self, *args, **kwargs):
        qc = self.sequence[min(self.calls, len(self.sequence) - 1)]
        self.calls += 1
        return qc


def _stub_auto_tighten(monkeypatch, sequence: list[dict]):
    """A `Single_Cell_Cropper_Tool` with `_compute_crop_qc` replaced by `_ScriptedQC`,
    plus placeholder `region`/`labeled_mask`/`cell_mask`/box args -- none of the OTHER
    args are ever actually read (the stub ignores them), only `_clamp_directional_inset`
    (NOT stubbed -- real arithmetic) needs a real, big-enough square box; it pairs every
    cut on the perpendicular axis, split evenly, floor (top/left) / ceil (bottom/right)."""
    tool = Single_Cell_Cropper_Tool()
    stub = _ScriptedQC(sequence)
    monkeypatch.setattr(tool, "_compute_crop_qc", stub)
    cell_mask = np.zeros((100, 100), dtype=np.uint8)
    return tool, stub, cell_mask


def test_auto_tighten_mask_bbox_plateau_stops_keeping_the_marginal_cut(monkeypatch):
    """A cut that improves `neighbor_contamination` by `_TIGHTEN_PLATEAU_EPS` (0.02) or
    less is COMMITTED (not worse than before, so it is kept -- never reverted) but the
    loop STOPS chasing further cuts. Sequence: 0.6 -> 0.30 (delta 0.30, commits, keeps
    going) -> 0.29 (delta 0.01 <= 0.02, commits, THEN stops -- the stub is never called
    a 4th time, proving the loop genuinely stopped, not just that the assertions
    happen to match). Each right cut is paired on the rows, split evenly (16 -> 8 / 8,
    then 5 -> 2 / 3), so the box stays square."""
    sequence = [
        {"neighbor_contamination": 0.6, "contamination_side": "right",
         "contamination_inset_px": 16, "mask_containment": 1.0},
        {"neighbor_contamination": 0.30, "contamination_side": "right",
         "contamination_inset_px": 5, "mask_containment": 1.0},
        {"neighbor_contamination": 0.29, "contamination_side": "right",
         "contamination_inset_px": 5, "mask_containment": 0.95},
    ]
    tool, stub, cell_mask = _stub_auto_tighten(monkeypatch, sequence)
    minr, maxr, minc, maxc, applied_inset, n_iter = tool._auto_tighten_mask_bbox(
        None, None, 1, cell_mask, 0, 100, 0, 100, min_area=0)
    assert n_iter == 2
    assert applied_inset == {"right": 21, "top": 10, "bottom": 11}   # 16 + 5, the plateaued cut IS kept
    assert (maxr - minr) == (maxc - minc) == 100 - 21                 # square
    assert stub.calls == 3   # exactly the scripted sequence -- no 4th (over-)call


def test_auto_tighten_mask_bbox_regression_is_never_committed(monkeypatch):
    """A cut that makes `neighbor_contamination` WORSE is never committed -- there is
    nothing to "revert" here (no file has been written yet, this is in-memory box
    exploration), so the loop simply stops at the prior, already-committed state.
    Sequence: 0.6 -> 0.25 (delta 0.35, commits) -> 0.35 (WORSE than 0.25, delta -0.10,
    NEVER committed)."""
    sequence = [
        {"neighbor_contamination": 0.6, "contamination_side": "right",
         "contamination_inset_px": 10, "mask_containment": 1.0},
        {"neighbor_contamination": 0.25, "contamination_side": "right",
         "contamination_inset_px": 8, "mask_containment": 0.95},
        {"neighbor_contamination": 0.35, "contamination_side": "right",
         "contamination_inset_px": 4, "mask_containment": 0.92},
    ]
    tool, stub, cell_mask = _stub_auto_tighten(monkeypatch, sequence)
    minr, maxr, minc, maxc, applied_inset, n_iter = tool._auto_tighten_mask_bbox(
        None, None, 1, cell_mask, 0, 100, 0, 100, min_area=0)
    assert n_iter == 1                        # only the FIRST cut ever committed
    assert applied_inset == {"right": 10, "top": 5, "bottom": 5}   # the regressing +8 was NEVER applied
    assert (maxr - minr) == (maxc - minc) == 90
    assert stub.calls == 3


def test_auto_tighten_mask_bbox_never_commits_below_the_retention_floor(monkeypatch):
    """A candidate cut that would drop `mask_containment` below `_TARGET_RETENTION_
    FLOOR` (0.90) is NEVER committed, regardless of how much it would have improved
    contamination -- the SAME never-truncate-past-floor guarantee the plain (non-auto-
    tighten) `bbox_inset_px` path already has, now also verified for this NEW internal
    multi-step accumulation path directly (the docstring's own "genuinely new code
    path" caveat). Task 038 FIX round #4: the floor-breach handler is now a binary
    search for the largest SAFE partial depth (see `test_auto_tighten_mask_bbox_floor_
    breach_binary_searches_a_partial_depth_instead_of_abandoning`, below, for that
    mechanism's own dedicated proof) -- this `_ScriptedQC` stub returns the SAME
    `mask_containment=0.85` (< floor) for EVERY call regardless of the box passed
    (including the binary search's own `mid=0` trial, which represents "no additional
    depth at all"), so even the search's own most conservative trial never clears the
    floor -- correctly falls through to `best_depth<=0` -> no headroom, same observable
    outcome (`n_iter==0`) as the old `break`-on-breach behavior for THIS specific
    (deliberately floor-stuck-even-at-zero) stubbed scenario."""
    sequence = [
        {"neighbor_contamination": 0.6, "contamination_side": "right",
         "contamination_inset_px": 10, "mask_containment": 1.0},
        {"neighbor_contamination": 0.1, "contamination_side": "right",
         "contamination_inset_px": 5, "mask_containment": 0.85},   # < floor -- never commit
    ]
    tool, stub, cell_mask = _stub_auto_tighten(monkeypatch, sequence)
    minr, maxr, minc, maxc, applied_inset, n_iter = tool._auto_tighten_mask_bbox(
        None, None, 1, cell_mask, 0, 100, 0, 100, min_area=0)
    assert n_iter == 0            # nothing ever committed
    assert applied_inset == {}
    assert (minr, maxr, minc, maxc) == (0, 100, 0, 100)   # unchanged from the plain box


def test_auto_tighten_mask_bbox_floor_breach_binary_searches_a_partial_depth_instead_of_abandoning(monkeypatch):
    """Task 038 FIX round #4 (coordinator-directed, root cause CONFIRMED structurally
    via direct code comparison -- see the coding log): the OLD `break`-on-floor-breach
    behavior abandoned the ENTIRE latest increment whenever `contamination_inset_px`'s
    own single-step projection overshot the floor -- NEVER exploring any depth between
    0 and the full requested step, unlike the plain single-shot mask_bbox path's own
    0.75/0.5/0.25/0.0 scale-back ladder (`_generate_cell_crops`). A real headless run
    measured this costing real cut depth (`neighbor_contamination_reduction` regressed
    -0.0141, mean inset depth measurably shallower). This stub's own `mask_containment`
    is a DIRECT, deterministic function of the CURRENT candidate box's own width
    (simulating a real, roughly-linear depth-vs-containment relationship, unlike the
    call-count-based `_ScriptedQC` used elsewhere) -- always proposes a big,
    floor-breaching depth (30px), so the binary search MUST find and commit a smaller,
    safe depth instead of reverting to zero progress. Every trial is the paired, square
    box of its depth (its paired rows split evenly: 5 / 5 at depth 10)."""
    def scripted_qc(region, region_label, labeled_mask, cell_mask_crop,
                    minr, minc, maxr, maxc, min_area=0, full_field_areas=None):
        width = maxc - minc
        assert maxr - minr == width             # every box the loop evaluates is square
        return {
            "mask_containment": width / 100.0,             # 100px wide == 1.0 containment
            "neighbor_contamination": max(0.0, 0.5 - (100 - width) * 0.01),
            "contamination_side": "right",
            "contamination_inset_px": 30,   # always proposes the SAME big, breaching cut
        }

    tool = Single_Cell_Cropper_Tool()
    monkeypatch.setattr(tool, "_compute_crop_qc", scripted_qc)
    cell_mask = np.zeros((100, 100), dtype=np.uint8)
    minr, maxr, minc, maxc, applied_inset, n_iter = tool._auto_tighten_mask_bbox(
        None, None, 1, cell_mask, 0, 100, 0, 100, min_area=0)

    # a 30px full cut would leave width=70 -> containment=0.70, well below the 0.90
    # floor -- the binary search must find the LARGEST depth in [0, 30] that keeps
    # width/100 >= 0.90, i.e. depth<=10 (width>=90) -- commits depth=10, NOT depth=0
    # (the old "abandon entirely" outcome).
    assert applied_inset == {"right": 10, "top": 5, "bottom": 5}
    assert n_iter == 1                          # genuine progress recovered, not abandoned
    assert (maxc - minc) == 90                  # width == 90 -> containment exactly 0.90
    assert (maxr - minr) == 90                  # and the box stays square
    # confirm the SAME scenario, replayed with the OLD "break, abandon" behavior,
    # commits NOTHING -- the dedicated before/after contrast (not just an isolated
    # assertion): a `break` on the FIRST floor-breaching candidate (depth=30, width=70)
    # would revert to the initial, empty `applied_inset`, matching `n_iter==0`.


# --------------------------------------------------------------------------------- #
# Task 038 v12 (human-locked, §2.2): `contamination_side` -- which crop edge carries
# the MOST non-target foreground, the side the directional lever should inset next.
# Computed from the non-target-fg pixel MASS relative to the TARGET's own centroid (two
# independent binary splits: rows above/below the centroid row, columns left/right of
# the centroid col) -- these 4 tests place a neighbour cleanly on ONE side (spanning the
# ORTHOGONAL axis symmetrically around the target's own centroid, so that axis's own
# split cancels out and only the intended side wins unambiguously).
# --------------------------------------------------------------------------------- #

def _contamination_side_for(neighbour_slice: tuple) -> str | None:
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[25:35, 25:35] = 1   # target: 10x10, centroid (29.5, 29.5)
    full_mask[neighbour_slice] = 2
    return _qc_multi_label(full_mask, target_label=1, crop_box=(10, 10, 50, 50))["contamination_side"]


def test_contamination_side_picks_the_correct_edge_top():
    assert _contamination_side_for((slice(15, 20), slice(20, 40))) == "top"


def test_contamination_side_picks_the_correct_edge_bottom():
    assert _contamination_side_for((slice(40, 45), slice(20, 40))) == "bottom"


def test_contamination_side_picks_the_correct_edge_left():
    assert _contamination_side_for((slice(20, 40), slice(12, 18))) == "left"


def test_contamination_side_picks_the_correct_edge_right():
    assert _contamination_side_for((slice(20, 40), slice(40, 46))) == "right"


def test_contamination_side_is_none_when_no_non_target_foreground():
    """A crop with NOTHING but the target itself -> nothing to tighten -- None, not an
    arbitrary side."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[25:35, 25:35] = 1
    qc = _qc_multi_label(full_mask, target_label=1, crop_box=(10, 10, 50, 50))
    assert qc["contamination_side"] is None


def test_target_retention_equals_mask_containment():
    """`target_retention` is DELIBERATELY the SAME number as `mask_containment` (task
    038 v12's own docstring) -- reported under a second, purpose-built name for the
    directional-tighten loop's own hard floor, not re-derived independently."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[10:30, 10:30] = 1
    qc = _qc_multi_label(full_mask, target_label=1, crop_box=(5, 5, 35, 35))
    assert qc["target_retention"] == qc["mask_containment"]


# --------------------------------------------------------------------------------- #
# Task 038 v12 R2 (human-directed -- REPLACES v12's own original fixed-fraction step;
# human rejected that design: "the logic of this step does not work; then what is the point of the
# VLM?" -- the mask
# already knows exactly where the neighbour pixels are, so a blind fraction was
# arbitrary): `contamination_inset_px` -- the pixel DEPTH, read straight off the mask,
# to inset `contamination_side` by so the non-target foreground reaching furthest
# toward the target on that side is excluded, CAPPED so `target_retention` never drops
# below `_TARGET_RETENTION_FLOOR` (task 038 v13, human decision: raised 0.85 -> 0.90,
# a more conservative floor for the "safe candidate-generator" scope).
# --------------------------------------------------------------------------------- #

def test_contamination_inset_px_fully_excludes_a_small_neighbour_without_touching_target(tmp_path):
    """A neighbour small/far enough that FULLY excluding it never touches the target at
    all -- `contamination_inset_px` is the FULL exclusion depth (not floor-capped), and
    applying it (a second `_compute_crop_qc` call on the resulting smaller box, mirroring
    what `tool.py::_generate_cell_crops` does next) leaves `contamination_side is None`
    (genuinely clean) and `target_retention` UNCHANGED at 1.0 (no target pixels were
    ever in the removed band)."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[25:35, 25:35] = 1   # target: 10x10, centroid (29.5, 29.5)
    full_mask[20:40, 40:46] = 2   # neighbour: clean right side, symmetric around centroid row
    qc = _qc_multi_label(full_mask, target_label=1, crop_box=(10, 10, 50, 50))
    assert qc["contamination_side"] == "right"
    assert qc["contamination_inset_px"] == 10

    qc_after = _qc_multi_label(full_mask, target_label=1,
                               crop_box=(10, 10, 50, 50 - qc["contamination_inset_px"]))
    assert qc_after["contamination_side"] is None   # neighbour genuinely gone
    assert qc_after["target_retention"] == 1.0       # target itself untouched


def test_contamination_inset_px_caps_at_the_floor_when_full_exclusion_would_breach_it():
    """A neighbour sitting INSIDE the target's own bbox (the L-shaped fixture, the exact
    v11 human-audit finding this lever exists to fix) -- fully excluding it would clip
    into the target beyond the 0.90 floor, so `contamination_inset_px` is a PARTIAL,
    floor-capped depth: applying it leaves `target_retention` at EXACTLY the floor
    (never below), and `contamination_side` is STILL "right" afterward (proving the
    neighbour was only PARTIALLY excluded, not fully -- the cap genuinely bit). A
    further `_compute_crop_qc` call on THAT result reports `contamination_inset_px == 0`
    (no headroom left at all), matching the verifier's own "floor already reached,
    freeze" read."""
    mask = _l_shaped_target_and_corner_neighbour()
    qc = _qc_multi_label(mask, target_label=1, crop_box=(20, 20, 50, 40))
    assert qc["contamination_side"] == "right"
    assert 0 < qc["contamination_inset_px"] < 20   # a genuine, but PARTIAL, cut

    qc_after = _qc_multi_label(mask, target_label=1,
                               crop_box=(20, 20, 50, 40 - qc["contamination_inset_px"]))
    assert qc_after["target_retention"] == pytest.approx(0.90)   # exactly the floor
    assert qc_after["contamination_side"] == "right"   # neighbour NOT fully excluded
    assert qc_after["contamination_inset_px"] == 0      # no further headroom -- freeze


def test_contamination_inset_px_stops_at_the_floor_with_residual_contamination_on_a_two_side_mask():
    """A mask with contamination on TWO DIFFERENT sides -- round 1 FULLY excludes the
    (right) neighbour without touching the target at all (retention stays 1.0); round 2
    (on the resulting smaller box) finds a SECOND neighbour on the (bottom) side, but
    fully excluding IT would breach the floor, so `contamination_inset_px` caps to 0 --
    the cell FREEZES at the floor with GENUINE residual contamination still present
    (`neighbor_contamination` well above the 0.2 clean threshold). Task 038 v13, human-
    directed: under the raised 0.90 floor this is now a LEGITIMATE, expected outcome (a
    tighter floor leaves less headroom for a second cut) -- and it is EXACTLY the
    "floor-hitting, still contaminated" state `verifiers/crop.py::CropVerifier`'s own
    VoI gate is designed to flag as uncertain/risky (see
    `test_crop_triage.py::test_geometry_voi_gate_floor_stuck_still_contaminated_is_risky`,
    which hardcodes this exact scenario's own numbers, verified here directly against
    `_compute_crop_qc`, not assumed)."""
    full_mask = np.zeros((60, 60), dtype=np.int64)
    full_mask[25:35, 25:35] = 1     # target: 10x10, centroid (29.5, 29.5)
    full_mask[20:40, 40:46] = 2     # neighbour A: right side, symmetric rows around centroid row
    full_mask[40:45, 20:40] = 3     # neighbour B: bottom side, symmetric cols around centroid col
    box = (10, 10, 50, 50)

    qc1 = _qc_multi_label(full_mask, target_label=1, crop_box=box)
    assert qc1["contamination_side"] == "right"
    assert qc1["contamination_inset_px"] == 16
    assert qc1["target_retention"] == 1.0   # full exclusion, target untouched
    box = (box[0], box[1], box[2], box[3] - qc1["contamination_inset_px"])

    qc2 = _qc_multi_label(full_mask, target_label=1, crop_box=box)
    assert qc2["contamination_side"] == "bottom"        # a second, different-side neighbour
    assert qc2["contamination_inset_px"] == 0            # floor-capped to nothing
    assert qc2["target_retention"] == pytest.approx(0.90)   # exactly the (new) floor
    assert qc2["neighbor_contamination"] > 0.2               # genuinely, not marginally, dirty


# --------------------------------------------------------------------------------- #
# Task 038 (v9 CRITICAL fix): `normalize_source_image` -- the SOURCE image is
# normalized to display uint8 ONCE, up front, before any crop is cut. Fixes a real
# the cluster finding (4/10 v9 images produced UNUSABLE crops): NeurIPS22-CellSeg is
# MIXED-MODALITY, and every crop-SAVE path downstream did a NAIVE
# `np.clip(cell_crop, 0, 255).astype(np.uint8)` -- a 16-bit fluorescence image (values
# into the thousands) clipped to ALL WHITE; a float [0,1] image cast to ALL BLACK.
# `_write_pair` (above) always builds a `.astype(np.uint8)` image, which cannot
# exercise this bug at all -- these tests write the RAW array directly.
# --------------------------------------------------------------------------------- #

def _write_pair_raw(data_dir: Path, name: str, image: np.ndarray, mask: np.ndarray) -> tuple[Path, Path]:
    """Like `_write_pair`, but takes the image array AS-IS (no forced uint8 cast)."""
    images_dir, labels_dir = data_dir / "images", data_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_path, label_path = images_dir / f"{name}.tif", labels_dir / f"{name}_label.tiff"
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(label_path, mask.astype(np.uint16))
    return image_path, label_path


def _saved_crop_pixels(tmp_path, image: np.ndarray, mask: np.ndarray, name: str) -> np.ndarray:
    """Runs the REAL tool end to end and returns the saved crop's own pixel array --
    the exact artifact the v9 the cluster bug corrupted (the mask-based QC is computed
    from `mask`/the label mask, never these pixels, so it is deliberately NOT checked
    here -- see the module docstring)."""
    from skimage.io import imread
    image_path, label_path = _write_pair_raw(tmp_path / "data", name, image, mask)
    tool = Single_Cell_Cropper_Tool()
    result = tool.execute(
        original_image=str(image_path), nuclei_mask=str(label_path),
        min_area=10, margin=1, query_cache_dir=str(tmp_path / "out"),
    )
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    return imread(meta["cell_metadata"][0]["crop_path"])


def test_16bit_source_image_crop_is_not_blown_to_pure_white(tmp_path):
    """The v9 the cluster bug's FIRST failure mode: a 16-bit fluorescence image (values
    into the thousands, up to ~40000) -- the OLD `np.clip(crop, 0, 255)` sent every
    value > 255 to 255, so the WHOLE crop saved pure white (std=0, e.g.
    cell_00232/00673/00707). The FIX (`normalize_source_image`, a percentile stretch)
    must produce a crop with genuine contrast."""
    rng = np.random.RandomState(0)
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[20:60, 20:60] = 1
    image = np.full((100, 100), 500, dtype=np.uint16)
    image[20:60, 20:60] = 40000
    image = (image + rng.randint(0, 200, size=image.shape)).astype(np.uint16)

    crop = _saved_crop_pixels(tmp_path, image, mask, "hi16")
    assert crop.dtype == np.uint8
    assert not np.all(crop == 255)
    assert crop.std() > 0


def test_float01_source_image_crop_is_not_blown_to_pure_black(tmp_path):
    """The v9 the cluster bug's SECOND failure mode: a float [0, 1] image -- the OLD
    `astype(np.uint8)` truncated every sub-1.0 value to 0, so the WHOLE crop saved pure
    black (std=0, e.g. cell_00860's 68 all-black crops). The FIX must produce a crop
    with genuine contrast."""
    rng = np.random.RandomState(1)
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[20:60, 20:60] = 1
    image = np.full((100, 100), 0.1, dtype=np.float32)
    image[20:60, 20:60] = 0.9
    image = image + rng.uniform(0, 0.02, size=image.shape).astype(np.float32)

    crop = _saved_crop_pixels(tmp_path, image, mask, "flo")
    assert crop.dtype == np.uint8
    assert not np.all(crop == 0)
    assert crop.std() > 0


def test_uint8_source_image_crop_still_has_contrast(tmp_path):
    """A normal, already-0-255 uint8 image (one of the 6 currently-good v9 images) is
    UNAFFECTED by the fix -- `to_display_uint8` passes an already-uint8 array through
    BYTE-IDENTICAL, so it must not visibly degrade."""
    rng = np.random.RandomState(2)
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[20:60, 20:60] = 1
    image = np.full((100, 100), 40, dtype=np.uint8)
    image[20:60, 20:60] = 200
    image = np.clip(image.astype(np.int32) + rng.randint(0, 10, size=image.shape), 0, 255).astype(np.uint8)

    crop = _saved_crop_pixels(tmp_path, image, mask, "u8")
    assert crop.dtype == np.uint8
    assert crop.std() > 0
    assert crop.min() < crop.max()


def test_saved_crop_preserves_all_channels_when_signal_is_not_in_channel_0(tmp_path):
    """Task 038 QC N3 (an independent QC of the geometry release candidate,
    PASS_WITH_NOTES): ALL of this file's `execute()` tests use a 2D single-channel
    `_write_pair`/`_write_pair_raw` fixture, but real NeurIPS22-CellSeg images are
    3-channel WITH THE SIGNAL IN A DIFFERENT CHANNEL PER IMAGE -- a real, measured
    finding from an earlier round: `cell_00860`'s own real delivered signal sits in
    channel 2, `cell_00994`'s splits across channels 0/1, both confirmed directly on
    the actual delivered crop files (task 038 Round C, `crop_triage._crop_pixels`'s
    own channel-0 hardcode bug). This TOOL's own channel PRESERVATION (does the SAVED
    crop file genuinely carry every channel the source image had, not just channel 0)
    was verified by QC via a manual PROBE only -- completely unprotected by any
    committed test until now. A single-channel fixture (every OTHER `execute()` test
    in this file) structurally CANNOT exercise this at all -- there is no channel
    dimension to lose in the first place.
    [[synthetic_fixtures_hide_real_data_bugs]] -- the 5th instance of this pattern in
    this task (the uint16 mask lattice, `_crop_pixels`'s own channel-0 hardcode, the
    overlay contour thickness, the S1 screen's own real-crop-shape fixture, and now
    this).

    Builds a 3-channel fixture whose channel 0 is EMPTY (all zero, mirroring
    `cell_00860`'s own real layout) and whose signal lives in channels 1 AND 2 --
    drives the REAL `execute()` (`_saved_crop_pixels`, reused, not reimplemented) and
    asserts the SAVED crop file still has all 3 channels, with channel 0 genuinely
    empty and channels 1/2 genuinely carrying the real signal through -- proving
    `is_multi_channel`'s own TIFF-preserving save path is not silently collapsing to
    a single channel anywhere between the source read and the saved file."""
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[20:60, 20:60] = 1
    ch0 = np.zeros((100, 100), dtype=np.uint8)                # EMPTY -- mirrors cell_00860
    ch1 = np.full((100, 100), 30, dtype=np.uint8)
    ch1[20:60, 20:60] = 180                                   # signal here too
    ch2 = np.full((100, 100), 20, dtype=np.uint8)
    ch2[20:60, 20:60] = 220                                   # the dominant real signal
    image = np.stack([ch0, ch1, ch2], axis=-1)

    crop = _saved_crop_pixels(tmp_path, image, mask, "multi_ch0_empty")

    assert crop.ndim == 3
    assert crop.shape[2] == 3                 # all 3 channels preserved, none dropped
    assert crop[..., 0].max() == 0            # channel 0 genuinely stays empty
    assert crop[..., 1].max() > 100           # channel 1's own signal survived
    assert crop[..., 2].max() > 100           # channel 2's own signal survived


def test_normalize_source_image_is_noop_for_uint8():
    """Direct unit check on the seam itself: an already-uint8 array passes through
    BYTE-IDENTICAL (no re-stretch at all) -- `to_display_uint8`'s own no-op contract
    for uint8, which is what makes the currently-good images safe."""
    from shape.tools.single_cell_cropper.tool import normalize_source_image
    rng = np.random.RandomState(3)
    img = rng.randint(0, 256, size=(30, 30), dtype=np.uint8)
    out = normalize_source_image(img)
    assert out.dtype == np.uint8
    assert np.array_equal(out, img)


def test_normalize_source_image_multichannel_normalizes_each_channel_independently():
    """A multi-channel image with WILDLY different per-channel intensity ranges (e.g. a
    bright bright-field-like channel next to a dim, low-range fluorescence-like
    channel) gets EACH channel independently stretched -- a single global percentile
    computed across both channels combined would over/under-stretch one relative to
    the other, and the dim channel would come out crushed/near-blank."""
    from shape.tools.single_cell_cropper.tool import normalize_source_image
    rng = np.random.RandomState(4)
    bright = (rng.rand(20, 20) * 200 + 30000).astype(np.uint16)     # bright-field-like
    dim = (rng.rand(20, 20) * 50 + 100).astype(np.uint16)           # dim fluorescence-like
    img = np.stack([bright, dim], axis=-1)

    out = normalize_source_image(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # BOTH channels independently gain real contrast/dynamic range -- neither is
    # crushed by the other's own (very different) intensity scale.
    assert out[:, :, 0].std() > 10
    assert out[:, :, 1].std() > 10


def test_output_subdir_keeps_each_calls_files_apart(tmp_path):
    """`output_subdir` puts a call's crops, masks and metadata in its own directory inside tool_cache
    (the file names are deterministic per cell, so two calls in one directory overwrite each other);
    without it the files go into tool_cache itself; a path outside tool_cache is an error result."""
    mask = np.zeros((80, 80), dtype=np.uint16)
    mask[10:30, 10:30] = 1
    mask[40:70, 35:60] = 2
    image_path, label_path = _write_pair(tmp_path / "data", "img", mask)
    tool = Single_Cell_Cropper_Tool()

    def run(**kwargs):
        return tool.execute(original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=1,
                            query_cache_dir=str(tmp_path / "q"), **kwargs)

    plain = run()
    assert Path(plain["cell_crops_metadata_path"]).parent == tmp_path / "q" / "tool_cache"
    wide, tight = run(output_subdir="calls/a"), run(output_subdir="calls/b", margin_mode="tight")
    for result, sub in ((wide, "calls/a"), (tight, "calls/b")):
        meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
        assert Path(result["cell_crops_metadata_path"]).parent == tmp_path / "q" / "tool_cache" / sub
        assert all(Path(m["crop_path"]).parent == tmp_path / "q" / "tool_cache" / sub for m in meta["cell_metadata"])
    sizes = [[tuple(m["crop_size"]) for m in json.loads(Path(r["cell_crops_metadata_path"]).read_text())["cell_metadata"]]
             for r in (wide, tight)]
    assert sizes[0] != sizes[1]                                   # both calls' crops survive side by side
    bad = run(output_subdir="../outside")
    assert bad["execution_status"] == "error" and "output_subdir" in bad["error"]


def test_execute_reports_its_own_run_time(tmp_path):
    """A successful call carries `seconds`, its own wall time (the agentic loop shows each call's)."""
    import time

    mask = np.zeros((80, 80), dtype=np.uint16)
    mask[10:30, 10:30] = 1
    image_path, label_path = _write_pair(tmp_path / "data", "img", mask)
    tool = Single_Cell_Cropper_Tool()
    before = time.time()
    result = tool.execute(original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=1,
                          query_cache_dir=str(tmp_path / "q"))
    elapsed = time.time() - before
    assert result["cell_count"] == 1 and isinstance(result["seconds"], float)
    assert 0.0 <= result["seconds"] <= round(elapsed, 1) + 0.1


# --------------------------------------------------------------------------------- #
# Human rule (2026-09-30): every crop the tool writes is SQUARE (crop height == crop
# width) -- the analyzer resizes every crop to 224 x 224 without keeping the aspect
# ratio, so a rectangle stretches its cell -- cut from real image pixels (no padding, no
# masking), in every margin_mode, with or without recenter, cell_overrides, crop_size,
# min_containment, bbox_inset_px and auto_tighten, with no exemption at the image border.
# --------------------------------------------------------------------------------- #

def _dense_border_mask() -> np.ndarray:
    """A non-square 90 x 120 field: elongated cells (aspect 3+) touching the top and the
    left border, a touching pair, an L-shaped cell on the bottom border with a neighbour
    inside its bbox, a cell in the bottom-right corner and a diagonal elongated cell."""
    mask = np.zeros((90, 120), dtype=np.int64)
    mask[0:8, 10:42] = 1        # horizontal, 8 x 32 (aspect 4), touching the top border
    mask[16:66, 0:12] = 2       # vertical, 50 x 12 (aspect ~4.2), touching the left border
    mask[30:50, 30:44] = 3      # a touching pair
    mask[30:50, 44:58] = 4
    mask[60:90, 70:78] = 5      # an L on the bottom border ...
    mask[84:90, 78:96] = 5
    mask[62:82, 80:90] = 6      # ... with its neighbour inside the L's bbox
    mask[74:90, 104:120] = 7    # in the bottom-right corner
    for k in range(24):         # a diagonal elongated cell
        mask[10 + k, 70 + k:76 + k] = 8
    return mask


def _small_image_mask() -> np.ndarray:
    """A 24 x 30 image smaller than most crops: a cell spanning the image's full width
    (bbox 10 x 30 -- no square containing it fits the 24 px height), a small cell near the
    bottom-left and a cell in the bottom-right corner."""
    mask = np.zeros((24, 30), dtype=np.int64)
    mask[2:12, 0:30] = 1
    mask[15:22, 3:9] = 2
    mask[14:24, 20:30] = 3
    return mask


_TIGHTEN = {"none": {}, "auto_tighten": {"auto_tighten": True},
            "bbox_inset_px": {"bbox_inset_px": {"right": 6, "top": 3}}}


@pytest.mark.parametrize("fixture", [_dense_border_mask, _small_image_mask], ids=["dense_border", "small_image"])
@pytest.mark.parametrize("margin_mode", ["fraction", "tight", "mask_bbox"])
def test_every_written_crop_is_square_for_every_geometry_combination(tmp_path, capsys, fixture, margin_mode):
    """Property-style (human rule 2026-09-30): over fixtures with border cells, elongated
    cells (aspect 3+), touching neighbours and a small image, every combination of
    margin_mode x recenter x crop_size x min_containment x auto_tighten (and a single-shot
    bbox_inset_px) writes only square crops -- the metadata box, `crop_size`, the crop file
    on disk and the mask_bbox baseline render -- inside the image; the containment rules
    still decide keep or drop (a kept crop keeps at least its `min_containment`, a
    mask_bbox crop at least the 0.90 retention floor) and every object is accounted for.
    Every box is square by construction: the safety net never fires."""
    from PIL import Image
    from skimage.measure import regionprops

    mask = fixture()
    height, width = mask.shape
    image_path, label_path = _write_pair(tmp_path / "data", "prop", mask)
    regions = regionprops(mask)
    cell_ids = [f"default_prop_cell_{i:04d}" for i in range(len(regions))]
    tool = Single_Cell_Cropper_Tool()
    runs = 0
    for recenter in (False, True):
        for crop_size in (None, 24):
            for min_containment in (1.0, 0.9):
                for tighten, keys in _TIGHTEN.items():
                    overrides = {cid: {"min_containment": min_containment, **keys} for cid in cell_ids}
                    out = tmp_path / f"out_{recenter}_{crop_size}_{min_containment}_{tighten}"
                    result = tool.execute(original_image=str(image_path), nuclei_mask=str(label_path),
                                          min_area=10, margin=1, margin_mode=margin_mode, recenter=recenter,
                                          crop_size=crop_size, cell_overrides=overrides, query_cache_dir=str(out))
                    label = f"{margin_mode} recenter={recenter} crop_size={crop_size} min_containment={min_containment} {tighten}"
                    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
                    stats = result.get("processing_statistics") or result.get("statistics") or {}
                    for m in meta["cell_metadata"]:
                        r0, c0, r1, c1 = m["crop_bbox"]
                        assert r1 - r0 == c1 - c0 > 0, (label, m["cell_id"], m["crop_bbox"])
                        assert 0 <= r0 and 0 <= c0 and r1 <= height and c1 <= width, (label, m["crop_bbox"])
                        assert m["crop_size"][0] == m["crop_size"][1] == r1 - r0, (label, m["crop_size"])
                        with Image.open(m["crop_path"]) as written:
                            assert written.size == (r1 - r0, r1 - r0), (label, written.size)
                        if m.get("baseline_crop_path"):
                            with Image.open(m["baseline_crop_path"]) as baseline:
                                assert baseline.size[0] == baseline.size[1], (label, baseline.size)
                        if m["margin_mode"] == "mask_bbox":
                            assert m["qc"]["target_retention"] >= 0.90 - 1e-9, (label, m["qc"]["target_retention"])
                        else:
                            assert m["qc"]["mask_containment"] >= min_containment - 1e-9, (label, m["qc"])
                    assert (len(meta["cell_metadata"]) + stats["filtered_by_border"] + stats["invalid_crop_data"]
                            + stats["filtered_by_area"]) == len(regions), (label, stats)
                    assert "was not square" not in capsys.readouterr().out, label   # the safety net never fired
                    runs += 1
    assert runs == 24


# ---- `_clamp_directional_inset`: the square-preserving directional cut, unit by unit ----------------------

def _clamped(inset, box=(0, 40, 0, 40), min_side=8):
    """`_clamp_directional_inset` on a 40 x 40 square."""
    return _clamp_directional_inset(*box, inset, min_side=min_side)


def _assert_box_matches_cuts(result, box=(0, 40, 0, 40)):
    r0, r1, c0, c1, applied = result
    cut = {k: applied.get(k, 0) for k in ("top", "bottom", "left", "right")}
    assert (r0, r1, c0, c1) == (box[0] + cut["top"], box[1] - cut["bottom"], box[2] + cut["left"],
                                box[3] - cut["right"])                       # the reported cuts ARE the box
    assert r1 - r0 == c1 - c0                                               # square
    assert all(v > 0 for v in applied.values())                             # only the sides actually cut


def test_clamp_directional_inset_splits_the_paired_cut_evenly_floor_ceil():
    """The paired cut is split evenly between the perpendicular axis's two sides (the target stays centred
    on it): an odd difference floor to top/left, ceil to bottom/right; the reported `applied` cuts are
    exactly the box's."""
    even = _clamped({"right": 6})
    assert even == (3, 37, 0, 34, {"right": 6, "top": 3, "bottom": 3})
    odd_rows = _clamped({"right": 5})
    assert odd_rows == (2, 37, 0, 35, {"right": 5, "top": 2, "bottom": 3})
    odd_cols = _clamped({"top": 5})
    assert odd_cols == (5, 40, 2, 37, {"top": 5, "left": 2, "right": 3})
    for result in (even, odd_rows, odd_cols):
        _assert_box_matches_cuts(result)


def test_clamp_directional_inset_keeps_the_requested_cuts_and_splits_only_the_equalizing_difference():
    """The requested (primary) cuts are applied as they are, on both axes; only the difference that
    equalizes the two axes is added -- to the axis with the smaller cut, split evenly between its sides."""
    one_sided_pair = _clamped({"right": 8, "top": 3})       # rows 3 < cols 8: the rows get 5 more, 2 / 3
    assert one_sided_pair == (5, 37, 0, 32, {"right": 8, "top": 5, "bottom": 3})
    both_sides = _clamped({"left": 10, "right": 2})         # cols 12 on both sides of one axis: rows 6 / 6
    assert both_sides == (6, 34, 10, 38, {"left": 10, "right": 2, "top": 6, "bottom": 6})
    cols_smaller = _clamped({"top": 9, "left": 1})          # cols 1 < rows 9: the columns get 8 more, 4 / 4
    assert cols_smaller == (9, 40, 5, 36, {"top": 9, "left": 5, "right": 4})
    for result in (one_sided_pair, both_sides, cols_smaller):
        _assert_box_matches_cuts(result)


def test_clamp_directional_inset_clamps_a_cut_beyond_the_headroom_to_min_side():
    """A cut beyond the headroom (side - min_side) is clamped: on one side to the headroom; on both sides of
    an axis proportionally, the integer split's remainder to the bottom/right side; the pairing then cuts the
    other axis by the same total, so the box ends at exactly `min_side` (8 px), square."""
    one_side = _clamped({"right": 50})          # headroom 40 - 8 = 32
    assert one_side == (16, 24, 0, 8, {"right": 32, "top": 16, "bottom": 16})
    both_sides = _clamped({"left": 30, "right": 20})   # 50 > 32: left 30 * 32 // 50 = 19, right the remaining 13
    assert both_sides == (16, 24, 19, 27, {"left": 19, "right": 13, "top": 16, "bottom": 16})
    both_axes = _clamped({"top": 40, "left": 25, "right": 25})   # each axis clamped to 32: no pairing left
    assert both_axes == (32, 40, 16, 24, {"top": 32, "left": 16, "right": 16})
    for result in (one_side, both_sides, both_axes):
        r0, r1, c0, c1, _ = result
        assert r1 - r0 == c1 - c0 == 8
        _assert_box_matches_cuts(result)
    # another min_side: 40 - 20 = 20 of headroom
    assert _clamped({"right": 30}, min_side=20) == (10, 30, 0, 20, {"right": 20, "top": 10, "bottom": 10})


def test_clamp_directional_inset_leaves_a_box_at_or_below_min_side_and_a_square_request_as_it_is():
    """No headroom (a box at or below `min_side`) -> no cut at all; a request that already cuts both axes
    by the same total needs no pairing; negative or missing values count as 0."""
    assert _clamped({"right": 3}, box=(0, 8, 0, 8)) == (0, 8, 0, 8, {})
    assert _clamped({"top": 2, "left": 2}, box=(0, 6, 0, 6)) == (0, 6, 0, 6, {})
    assert _clamped({"right": 4, "top": 4}) == (4, 40, 0, 36, {"right": 4, "top": 4})
    assert _clamped({"right": -3, "top": None}) == (0, 40, 0, 40, {})


# ---- placement: a border or corner crop has the side its interior twin has --------------------------------

def test_border_and_corner_crops_have_the_side_of_their_interior_twin(tmp_path, capsys):
    """Every crop's side is min(2 x half_side, the image's shorter side), wherever the cell sits: a cell on
    the top border, one in the bottom-right corner and their interior twin (all 10 x 10) get the same
    square -- 14 x 14 at the bbox midpoint (half side 5 + margin round(0.2 x 11.3) = 2), 16 x 16 with
    recenter (half side ceil(5.5 + 2)) -- the border and corner squares shifted inside the image; the small
    image's corner cell (label 3 of `_small_image_mask`) gets (10, 16, 24, 30), 14 x 14 -- a per-axis border
    clip would cut the corner crops to 12 x 12 (14 x 14 with recenter) -- and the safety net never fires."""
    mask = np.zeros((60, 60), dtype=np.int64)
    mask[0:10, 25:35] = 1       # top border
    mask[30:40, 25:35] = 2      # the interior twin
    mask[50:60, 50:60] = 3      # bottom-right corner
    image_path, label_path = _write_pair(tmp_path / "data", "twins", mask)
    tool = Single_Cell_Cropper_Tool()
    expected = {False: {"0000": [0, 23, 14, 37], "0001": [28, 23, 42, 37], "0002": [46, 46, 60, 60]},
                True: {"0000": [0, 21, 16, 37], "0001": [26, 21, 42, 37], "0002": [44, 44, 60, 60]}}
    for recenter, boxes_expected in expected.items():
        result = tool.execute(original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=1,
                              recenter=recenter, query_cache_dir=str(tmp_path / f"out_{recenter}"))
        boxes = {m["cell_id"][-4:]: m["crop_bbox"] for m in
                 json.loads(Path(result["cell_crops_metadata_path"]).read_text())["cell_metadata"]}
        assert boxes == boxes_expected, recenter
        assert len({b[2] - b[0] for b in boxes.values()} | {b[3] - b[1] for b in boxes.values()}) == 1

    small_image, small_label = _write_pair(tmp_path / "data", "small", _small_image_mask())
    result = tool.execute(original_image=str(small_image), nuclei_mask=str(small_label), min_area=10, margin=1,
                          query_cache_dir=str(tmp_path / "out_small"))
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    corner = next(m for m in meta["cell_metadata"] if m["cell_id"].endswith("0002"))
    assert corner["original_bbox"] == [14, 20, 24, 30] and corner["crop_bbox"] == [10, 16, 24, 30]
    assert "was not square" not in capsys.readouterr().out


def test_a_square_larger_than_the_images_shorter_side_keeps_a_bbox_that_fits(tmp_path, capsys):
    """The needed square (38 px: half side 10 + margin round(0.4 x 22.6) = 9; 40 px with recenter) is larger
    than the 30 px image height but the 20 x 20 bbox fits: the crop is the largest square that fits (30 x 30),
    placed to hold the bbox as close to the centre as the image allows (the bbox midpoint, or the centroid
    with recenter), and the cell is kept whole -- without the safety net."""
    mask = np.zeros((30, 100), dtype=np.int64)
    mask[5:25, 40:60] = 1
    image_path, label_path = _write_pair(tmp_path / "data", "low", mask)
    tool = Single_Cell_Cropper_Tool()
    for recenter, box in ((False, [0, 35, 30, 65]), (True, [0, 34, 30, 64])):
        result = tool.execute(original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=1,
                              margin_frac=0.4, recenter=recenter, query_cache_dir=str(tmp_path / f"out_{recenter}"))
        assert result["filtered_by_border"] == 0 and result["cell_count"] == 1
        target = json.loads(Path(result["cell_crops_metadata_path"]).read_text())["cell_metadata"][0]
        assert target["crop_bbox"] == box, recenter
        assert target["qc"]["mask_containment"] == 1.0
    assert "was not square" not in capsys.readouterr().out


@pytest.mark.parametrize("name", ["vertical_bar", "right_and_bottom", "right_and_left"])
def test_auto_tighten_stays_square_at_every_step_keeps_the_floor_and_lowers_contamination(tmp_path, monkeypatch,
                                                                                            name):
    """`auto_tighten` on a contaminated mask_bbox cell (human rules 2026-09-30): every box
    its internal loop evaluates -- the plain square, each candidate cut, each binary-search
    trial -- and every crop's final box is square (a spy on `_compute_crop_qc`, which
    measures each of them), the target keeps at least the 0.90 retention floor and the
    contamination ends lower than the plain mask_bbox square's. Three fixtures: a vertical
    bar with a neighbour beside its bbox (the first paired cut breaches the floor, the
    binary search commits a smaller one), a right + bottom neighbour (one paired cut down
    to the floor, residual contamination) and a right + left neighbour (two committed
    steps, the second on the other axis)."""
    if name == "vertical_bar":
        mask, buffer = _vertical_bar_and_right_neighbour(), 2
    else:
        mask, buffer = np.zeros((60, 60), dtype=np.int64), 15
        mask[25:35, 25:35] = 1           # target: 10 x 10
        mask[20:40, 40:46] = 2           # right neighbour
        if name == "right_and_bottom":
            mask[40:45, 20:40] = 3       # bottom neighbour
        else:
            mask[22:38, 14:19] = 3       # left neighbour
    image_path, label_path = _write_pair(tmp_path / "data", name, mask)
    cell_ids = [f"default_{name}_cell_{i:04d}" for i in range(int(mask.max()))]
    tool = Single_Cell_Cropper_Tool()
    boxes = []
    real_qc = Single_Cell_Cropper_Tool._compute_crop_qc

    def spy(region, region_label, labeled_mask, cell_mask_crop, minr, minc, maxr, maxc, min_area=0,
            full_field_areas=None):
        boxes.append((maxr - minr, maxc - minc))
        return real_qc(region, region_label, labeled_mask, cell_mask_crop, minr, minc, maxr, maxc, min_area,
                       full_field_areas=full_field_areas)

    monkeypatch.setattr(tool, "_compute_crop_qc", spy)

    def run(auto):
        result = tool.execute(original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=1,
                              tight_buffer_px=buffer, query_cache_dir=str(tmp_path / f"out_{auto}"),
                              cell_overrides={c: {"margin_mode": "mask_bbox", "auto_tighten": auto} for c in cell_ids})
        meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
        return {m["cell_id"]: m for m in meta["cell_metadata"]}

    plain = run(False)[cell_ids[0]]
    boxes.clear()
    tightened = run(True)
    target = tightened[cell_ids[0]]
    assert len(boxes) > len(tightened)                         # the loop evaluated boxes beyond the final ones
    assert all(h == w for h, w in boxes), boxes                # square at every step
    for m in tightened.values():
        r0, c0, r1, c1 = m["crop_bbox"]
        assert r1 - r0 == c1 - c0
        assert m["qc"]["target_retention"] >= 0.90 - 1e-9
    assert target["auto_tighten_iterations"] >= 1 and target["bbox_inset_px"]
    assert target["qc"]["neighbor_contamination"] < plain["qc"]["neighbor_contamination"]
    if name == "vertical_bar":
        assert target["bbox_inset_px"] == {"right": 7, "top": 3, "bottom": 4}   # depth 8 breached, 7 kept 0.90
        assert target["qc"]["target_retention"] == pytest.approx(0.90)
    if name == "right_and_left":
        assert target["auto_tighten_iterations"] == 2
        assert target["bbox_inset_px"] == {"right": 10, "left": 9, "top": 9, "bottom": 10}
        assert target["crop_bbox"] == [19, 19, 40, 40]
        assert target["qc"]["neighbor_contamination"] == 0.0 and target["qc"]["target_retention"] == 1.0


def test_auto_tighten_leaves_the_cell_uncut_when_every_evenly_paired_cut_breaches_the_floor(tmp_path):
    """The cost of the even split (human decision 2026-09-30, the target stays centred): on the L whose
    neighbour sits inside its bbox, every right cut deep enough to reach the neighbour is paired with an
    even top + bottom cut through the target, below the 0.90 floor, so `auto_tighten` commits nothing: the
    crop stays the plain mask_bbox square, whole target, contamination unchanged."""
    image_path, label_path = _write_pair(tmp_path / "data", "lcut", _l_shaped_target_and_corner_neighbour())
    cell_id = "default_lcut_cell_0000"
    result = Single_Cell_Cropper_Tool().execute(
        original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=1,
        query_cache_dir=str(tmp_path / "out"),
        cell_overrides={cell_id: {"margin_mode": "mask_bbox", "auto_tighten": True}})
    target = [m for m in json.loads(Path(result["cell_crops_metadata_path"]).read_text())["cell_metadata"]
              if m["cell_id"] == cell_id][0]
    assert target["auto_tighten_iterations"] == 0 and target["bbox_inset_px"] == {}
    assert target["crop_bbox"] == [18, 13, 52, 47]                 # the plain mask_bbox square
    assert target["qc"]["target_retention"] == 1.0
    assert target["qc"]["neighbor_contamination"] == pytest.approx(0.4)


def test_safety_net_squares_a_non_square_box_around_the_bbox_and_reports_the_cuts_it_makes(tmp_path, monkeypatch):
    """The safety net in `_generate_cell_crops` (human rule 2026-09-30): a non-square box
    -- injected here through the mask_bbox tightening step, since every real path is
    square by construction -- is never written as a rectangle. It is replaced by the
    square placed like every crop around the TARGET's bbox (side = the box's longer side),
    the containment check judges that square by the bbox (kept: it holds the whole
    target), and `bbox_inset_px` is re-derived from the square -- the cuts it makes into
    the plain mask_bbox square -- so the metadata describes the crop written."""
    from PIL import Image

    mask = np.zeros((100, 100), dtype=np.int64)
    mask[40:60, 40:50] = 1      # bbox 20 x 10: the plain mask_bbox square is [38, 33, 62, 57] (24 x 24)
    image_path, label_path = _write_pair(tmp_path / "data", "net", mask)
    tool = Single_Cell_Cropper_Tool()
    monkeypatch.setattr(tool, "_auto_tighten_mask_bbox",
                        lambda *args, **kwargs: (40, 62, 38, 52, {"right": 1}, 1))   # a 22 x 14 rectangle
    result = tool.execute(original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=1,
                          query_cache_dir=str(tmp_path / "out"),
                          cell_overrides={"default_net_cell_0000": {"margin_mode": "mask_bbox", "auto_tighten": True}})
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    assert len(meta["cell_metadata"]) == 1 and result["filtered_by_border"] == 0
    target = meta["cell_metadata"][0]
    # 22 x 22 around the bbox, centred on the injected box's midpoint (51, 45)
    assert target["crop_bbox"] == [40, 34, 62, 56]
    assert target["qc"]["mask_containment"] == 1.0
    # the cuts the square makes into the plain mask_bbox square [38, 33, 62, 57]
    assert target["bbox_inset_px"] == {"top": 2, "left": 1, "right": 1}
    with Image.open(target["crop_path"]) as written:
        assert written.size == (22, 22)


def test_safety_net_judges_by_the_bbox_not_by_the_boxs_longer_side(tmp_path, monkeypatch):
    """A non-square box longer than the image's shorter side does not drop a cell whose
    bbox fits: the safety net places the largest square that fits around the target's
    bbox, and the containment check keeps it (a fraction-mode cell; the rectangle is
    injected as the placement's result, the safety net's own placement runs normally)."""
    from shape.tools.single_cell_cropper import tool as tool_module

    real_place = tool_module.compute_square_crop_containing_bbox
    calls = []

    def first_placement_is_a_rectangle(*args, **kwargs):
        calls.append(args)
        r0, c0, r1, c1 = real_place(*args, **kwargs)
        return (r0, 10, r1, 90) if len(calls) == 1 else (r0, c0, r1, c1)   # 30 x 80 in a 40 x 100 image

    monkeypatch.setattr(tool_module, "compute_square_crop_containing_bbox", first_placement_is_a_rectangle)
    mask = np.zeros((40, 100), dtype=np.int64)
    mask[10:30, 40:60] = 1      # bbox 20 x 20: it fits a 40 px square
    image_path, label_path = _write_pair(tmp_path / "data", "wide", mask)
    result = Single_Cell_Cropper_Tool().execute(original_image=str(image_path), nuclei_mask=str(label_path),
                                                min_area=10, margin=1, query_cache_dir=str(tmp_path / "out"))
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    assert len(calls) == 2 and result["filtered_by_border"] == 0 and len(meta["cell_metadata"]) == 1
    target = meta["cell_metadata"][0]
    assert target["crop_bbox"] == [0, 30, 40, 70]           # the largest square that fits, holding the bbox
    assert target["qc"]["mask_containment"] == 1.0


def test_safety_net_leaves_keep_or_drop_to_the_containment_rules(tmp_path, monkeypatch):
    """When no square can hold the target's bbox (longer than the image's shorter side),
    the safety net still writes no rectangle: its square truncates the target and the
    containment rules drop the cell (mask_bbox: below the 0.90 retention floor), counted
    in `filtered_by_border`; the other cell is unaffected."""
    mask = np.zeros((40, 100), dtype=np.int64)
    mask[10:30, 20:80] = 1      # bbox 20 x 60: no square inside the 40 px tall image holds it
    mask[10:20, 5:15] = 2       # an ordinary cell
    image_path, label_path = _write_pair(tmp_path / "data", "drop", mask)
    tool = Single_Cell_Cropper_Tool()
    monkeypatch.setattr(tool, "_auto_tighten_mask_bbox",
                        lambda *args, **kwargs: (0, 30, 20, 80, {}, 0))      # a 30 x 60 rectangle
    result = tool.execute(original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=1,
                          query_cache_dir=str(tmp_path / "out"),
                          cell_overrides={"default_drop_cell_0000": {"margin_mode": "mask_bbox", "auto_tighten": True}})
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    assert [m["cell_id"] for m in meta["cell_metadata"]] == ["default_drop_cell_0001"]
    assert result["filtered_by_border"] == 1 and result["cell_count"] == 1
    r0, c0, r1, c1 = meta["cell_metadata"][0]["crop_bbox"]
    assert r1 - r0 == c1 - c0


# ---- odd bbox sides, the retention floor of `min_containment`, the target's own label (2026-10-01) ----------

def _odd_sides_mask() -> np.ndarray:
    """Two cells with an odd longer bbox side (41): 41 x 20 and 41 x 41."""
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[10:51, 10:30] = 1
    mask[55:96, 50:91] = 2
    return mask


@pytest.mark.parametrize("mode, recenter", [("fraction", False), ("tight", False), ("fraction", True),
                                            ("tight", True), ("mask_bbox", False)])
def test_an_odd_longer_side_is_held_at_margin_0_in_every_mode(tmp_path, mode, recenter):
    """The square holds an odd longer bbox side with no margin at all (margin 0, margin_frac 0,
    tight_buffer_px 0) in every mode: `fraction` / `tight` size the half side as ceil(longer side / 2) +
    margin -- a 41 px side gets a 42 px square (before, `longer // 2` gave 40 px and the cell was dropped,
    the pilot's `tight` call with tight_buffer_px 0 that lost 40 objects); `recenter` already rounded up;
    `mask_bbox` is the exact 41 px square."""
    image_path, label_path = _write_pair(tmp_path / "data", "odd", _odd_sides_mask())
    result = Single_Cell_Cropper_Tool().execute(
        original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=0, margin_frac=0.0,
        margin_mode=mode, tight_buffer_px=0, recenter=recenter, query_cache_dir=str(tmp_path / "out"))
    assert result["filtered_by_border"] == 0 and result["cell_count"] == 2
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    for m in meta["cell_metadata"]:
        (r0, c0, r1, c1), (minr, minc, maxr, maxc) = m["crop_bbox"], m["original_bbox"]
        assert r1 - r0 == c1 - c0 == (41 if mode == "mask_bbox" else 42), (m["cell_id"], m["crop_bbox"])
        assert r0 <= minr and c0 <= minc and r1 >= maxr and c1 >= maxc            # holds the bbox
        assert m["qc"]["mask_containment"] == 1.0


def test_min_containment_below_the_retention_floor_acts_as_the_floor(tmp_path):
    """`min_containment` acts within [`_TARGET_RETENTION_FLOOR`, 1] (the single 0.90 floor): a cell asking
    for 0.5 gets the crop it gets at 0.90 -- here the square shrinks by sqrt(0.9) and still holds the whole
    40 x 40 cell, where sqrt(0.5) would have cut it to a 30 px square keeping 0.56 of it."""
    from shape.tools.single_cell_cropper.tool import _TARGET_RETENTION_FLOOR

    assert _TARGET_RETENTION_FLOOR == 0.90
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[30:70, 30:70] = 1
    image_path, label_path = _write_pair(tmp_path / "data", "floor", mask)
    boxes = {}
    for value in (0.5, 0.9):
        result = Single_Cell_Cropper_Tool().execute(
            original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=1, margin_frac=0.0,
            cell_overrides={"default_floor_cell_0000": {"min_containment": value}},
            query_cache_dir=str(tmp_path / f"out_{value}"))
        target = json.loads(Path(result["cell_crops_metadata_path"]).read_text())["cell_metadata"][0]
        assert target["qc"]["target_retention"] >= _TARGET_RETENTION_FLOOR
        boxes[value] = target["crop_bbox"]
    assert boxes[0.5] == boxes[0.9] == [30, 30, 70, 70]


def _ring_and_l_mask() -> np.ndarray:
    """A ring (label 1) whose centroid pixel lies on the small cell inside its hole (label 2), and an L
    (label 3) whose centroid pixel lies on background, far apart."""
    mask = np.zeros((120, 120), dtype=np.int64)
    yy, xx = np.mgrid[0:120, 0:120]
    distance = np.hypot(yy - 40, xx - 40)
    mask[(distance >= 10) & (distance < 20)] = 1
    mask[distance < 5] = 2
    mask[70:110, 70:78] = 3
    mask[102:110, 78:110] = 3
    return mask


def test_every_readout_is_about_the_cells_own_label_when_its_centroid_lies_outside_it(tmp_path):
    """The target is the region's own label, not the label under its centroid pixel (2026-10-01). The ring's
    centroid pixel is on the cell inside its hole and the L's on background: before, the ring's readouts
    described the inner cell (containment 0.07, the ring counted as contamination) and the L's the
    background (contamination 1.0), its saved mask was the background, and the two containment backstops
    measured the wrong label -- the L cut to 0.47 of its pixels by a requested inset, and a crop_size cap
    keeping 0.69 of it, were kept, the ring under the same cap dropped. Now: the readouts are the ring's
    and the L's own, and both backstops measure the cell itself."""
    from skimage.io import imread

    mask = _ring_and_l_mask()
    regions = {r.label: r for r in regionprops(mask)}
    for lbl, other in ((1, 2), (3, 0)):                                  # the fixture's premise
        cy, cx = regions[lbl].centroid
        assert mask[int(cy), int(cx)] == other
    image_path, label_path = _write_pair(tmp_path / "data", "rl", mask)
    tool = Single_Cell_Cropper_Tool()
    ring, inner, l_cell = "default_rl_cell_0000", "default_rl_cell_0001", "default_rl_cell_0002"

    def run(name, **kwargs):
        result = tool.execute(original_image=str(image_path), nuclei_mask=str(label_path), min_area=10, margin=1,
                              query_cache_dir=str(tmp_path / name), **kwargs)
        return result, {m["cell_id"]: m for m in json.loads(Path(result["cell_crops_metadata_path"]).read_text())
                        ["cell_metadata"]}

    _, cells = run("plain")
    ring_qc, l_qc = cells[ring]["qc"], cells[l_cell]["qc"]
    assert ring_qc["mask_containment"] == 1.0 and ring_qc["n_complete_cells"] == 2
    assert ring_qc["neighbor_contamination"] == pytest.approx(69 / (940 + 69))       # the inner cell
    assert ring_qc["target_dominance"] == pytest.approx(940 / (940 + 69))
    assert l_qc["mask_containment"] == 1.0 and l_qc["neighbor_contamination"] == 0.0
    assert l_qc["target_dominance"] == pytest.approx(1.0) and l_qc["single_object"] == 1.0
    for cid, area in ((ring, 940), (l_cell, 576)):                     # the saved mask is the cell's own
        assert int(np.count_nonzero(np.asarray(imread(cells[cid]["mask_path"])))) == area
    # auto_tighten measures the ring itself: its inner cell is not worth a cut (contamination 0.07)
    _, tightened = run("auto", margin_mode="mask_bbox", cell_overrides={ring: {"auto_tighten": True}})
    assert tightened[ring]["bbox_inset_px"] == {} and tightened[ring]["qc"]["target_retention"] == 1.0
    # the mask_bbox backstop measures the L: a 30 px inset that keeps 0.47 of it is scaled back to none
    _, inset = run("inset", cell_overrides={l_cell: {"margin_mode": "mask_bbox", "bbox_inset_px": {"right": 30}}})
    assert inset[l_cell]["bbox_inset_px"] == {} and inset[l_cell]["qc"]["target_retention"] == 1.0
    # the min_containment backstop measures each cell's own label: a 36 px cap keeps 0.91 of the ring (kept)
    # and 0.69 of the L (dropped); the old lookup dropped the ring (0.07 -- the inner cell) and kept the L
    capped, kept = run("cap", crop_size=36, cell_overrides={c: {"min_containment": 0.9} for c in (ring, l_cell)})
    assert capped["filtered_by_border"] == 1 and set(kept) == {ring, inner}
    assert kept[ring]["qc"]["target_retention"] == pytest.approx(854 / 940)


# ---- the cropper inherits the segmenter's object definition (min_area, 2026-10-01) ---------------------------

def _execute_default(execute_def):
    args = execute_def.args.args
    return dict(zip([a.arg for a in args[len(args) - len(execute_def.args.defaults):]], execute_def.args.defaults))


def test_the_tools_default_min_area_is_the_cell_segmenters_default_min_size():
    """One object definition: `SEGMENTER_DEFAULT_MIN_SIZE` equals `Cell_Segmenter_Tool.execute`'s `min_size`
    default, read with `ast` (the cropper never imports the segmenter: cellpose, torch), and the one default
    rule (`default_min_area_margin`, which the agentic loop imports) uses it for every mask name but organoid."""
    import ast

    from shape.tools.single_cell_cropper.tool import SEGMENTER_DEFAULT_MIN_SIZE, default_min_area_margin

    source = Path(__file__).resolve().parents[2] / "cell_segmenter" / "tool.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    segmenter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Cell_Segmenter_Tool")
    execute = next(n for n in segmenter.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
    assert ast.literal_eval(_execute_default(execute)["min_size"]) == SEGMENTER_DEFAULT_MIN_SIZE == 15
    assert default_min_area_margin("a_cell_mask.png") == (15, 1)
    assert default_min_area_margin("a_nuclei_mask.png") == default_min_area_margin("a_pred.tiff") == (15, 25)
    assert default_min_area_margin("an_organoid_mask.png") == (1000, 10)        # out of scope, unchanged
    from shape.agent.verification.verifiers import crop as verifier_module
    assert verifier_module.tool_default_min_area_margin is default_min_area_margin


@pytest.mark.parametrize("name, kept", [("img_cell_mask", True), ("img_nuclei_mask", True), ("img_pred", True),
                                        ("img_organoid_mask", False)])
def test_a_20_to_30_px_object_gets_a_crop_at_the_tools_default(tmp_path, name, kept):
    """Without a caller's min_area a 25 px object (5 x 5) gets a crop on a cell, nuclei or unnamed mask (the
    cell segmenter's default min_size, 15; before, 50 dropped it) and a 9 px one does not; an organoid mask keeps
    its 1000 px default."""
    mask = np.zeros((60, 60), dtype=np.int64)
    mask[10:15, 10:15] = 1               # 25 px
    mask[40:43, 40:43] = 2               # 9 px: below 15
    mask[20:60, 30:60] = 3               # 1200 px
    images_dir = tmp_path / "data"
    images_dir.mkdir()
    image = np.where(mask > 0, 200, 40).astype(np.uint8)
    tifffile.imwrite(images_dir / "img.tif", image)
    tifffile.imwrite(images_dir / f"{name}.tiff", mask.astype(np.uint16))
    result = Single_Cell_Cropper_Tool().execute(original_image=str(images_dir / "img.tif"),
                                                nuclei_mask=str(images_dir / f"{name}.tiff"),
                                                query_cache_dir=str(tmp_path / "out"))
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    areas = sorted(m["area"] for m in meta["cell_metadata"])
    assert meta["parameters"]["min_area"] == (1000 if name == "img_organoid_mask" else 15)
    assert areas == ([25, 1200] if kept else [1200])


# ---- full-image work once per call, not once per cell (2026-10-01): identical outputs ------------------------

class _FullImageReference(Single_Cell_Cropper_Tool):
    """The per-cell full-image computations of the tool before 2026-10-01, as a reference path: a fresh
    full-size target mask per cell (the whole array cleared, the cell's pixels set from its coordinates) and the
    label areas counted again in every QC call."""

    @staticmethod
    def _show_target(cell_mask, region, value):
        cell_mask[...] = 0
        if value:
            rows, cols = region.coords.T
            cell_mask[rows, cols] = value

    def _compute_crop_qc(self, *args, full_field_areas=None, **kwargs):
        return Single_Cell_Cropper_Tool._compute_crop_qc(*args, **kwargs)


def _dense_field_mask() -> np.ndarray:
    """A dense 200 x 200 field: about 90 random rectangles (later ones overwrite earlier ones, so most touch and
    some become concave), odd and even sides, objects on the border, a ring around a cell, an L, and specks below
    min_area."""
    rng = np.random.RandomState(7)
    mask = np.zeros((200, 200), dtype=np.int64)
    label_id = 1
    for _ in range(90):
        h, w = rng.randint(6, 31), rng.randint(6, 31)
        r, c = rng.randint(-5, 200 - h + 5), rng.randint(-5, 200 - w + 5)
        mask[max(r, 0):r + h, max(c, 0):c + w] = label_id
        label_id += 1
    yy, xx = np.mgrid[0:200, 0:200]
    distance = np.hypot(yy - 100, xx - 100)
    mask[(distance >= 8) & (distance < 16)] = label_id
    mask[distance < 4] = label_id + 1
    mask[20:60, 160:166] = label_id + 2
    mask[54:60, 166:190] = label_id + 2
    mask[190:192, 10:12] = label_id + 3
    return mask


def _call_outputs(tool, image_path, label_path, out: Path, **kwargs) -> dict:
    """Everything a call delivers, run-independent: every file under tool_cache by name with its bytes (the
    metadata's random id and this run's directory masked; the zip by member), and the returned dict without
    the run time."""
    import re
    import zipfile

    def plain(text: str) -> str:
        for root in (str(out), str(out).replace("\\", "\\\\"), out.as_posix()):
            text = text.replace(root, "<out>")
        return re.sub(r"cell_crops_metadata_[0-9a-f]{8}", "cell_crops_metadata_<id>", text)

    result = tool.execute(original_image=str(image_path), nuclei_mask=str(label_path), min_area=20, margin=1,
                          query_cache_dir=str(out), **kwargs)
    files = {}
    for path in sorted((out / "tool_cache").rglob("*")):
        if not path.is_file():
            continue
        name = plain(path.relative_to(out).as_posix())
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                files[name] = {plain(m): (plain(archive.read(m).decode()) if m.endswith(".json") else archive.read(m))
                               for m in sorted(archive.namelist())}
        else:
            files[name] = plain(path.read_text()) if path.suffix == ".json" else path.read_bytes()
    returned = {k: v for k, v in result.items() if k != "seconds"}
    returned = json.loads(plain(json.dumps(returned, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o))))
    return {"files": files, "returned": returned}


def test_the_tools_once_per_call_full_image_work_gives_identical_outputs(tmp_path, monkeypatch):
    """The tool counts the label areas once per call and keeps one shared target mask instead of a full-image
    pass per cell; every output is identical to the per-cell full-image reference path: the crop, mask and
    baseline files byte for byte, the metadata (the crop boxes, every per-crop QC readout, the summary stats),
    the zip's members, the file names and the returned dict (the run time aside). On a dense field, for every
    margin mode, recenter, crop_size, min_containment, a single-shot bbox_inset_px and auto_tighten, image-level
    and per-cell. Every QC call gets the areas counted once."""
    mask = _dense_field_mask()
    regions = regionprops(mask)
    cells = [f"default_dense_cell_{i:04d}" for i in range(len(regions))]
    kinds = [{"margin_mode": "tight", "recenter": True}, {"margin_mode": "mask_bbox", "auto_tighten": True},
             {"margin_mode": "mask_bbox", "bbox_inset_px": {"right": 6}}, {"min_containment": 0.95},
             {"recenter": True}, {}]
    configs = {"defaults": {},
               "tight": {"margin_mode": "tight", "tight_buffer_px": 0, "recenter": True},
               "auto_tighten": {"margin_mode": "mask_bbox", "cell_overrides": {c: {"auto_tighten": True} for c in cells}},
               "crop_size": {"crop_size": 24, "cell_overrides": {c: {"min_containment": 0.9} for c in cells[::2]}},
               "mixed": {"margin_frac": 0.3, "cell_overrides": {c: kinds[i % len(kinds)] for i, c in enumerate(cells)}}}
    image_path, label_path = _write_pair(tmp_path / "data", "dense", mask)
    areas_passed = []
    real_qc = Single_Cell_Cropper_Tool._compute_crop_qc

    def counting_qc(*args, full_field_areas=None, **kwargs):
        areas_passed.append(full_field_areas is not None)
        return real_qc(*args, full_field_areas=full_field_areas, **kwargs)

    covered = {"tightened": 0, "dropped": 0, "baselines": 0}
    for name, kwargs in configs.items():
        reference = _call_outputs(_FullImageReference(), image_path, label_path, tmp_path / f"ref_{name}", **kwargs)
        fast_tool = Single_Cell_Cropper_Tool()
        monkeypatch.setattr(fast_tool, "_compute_crop_qc", counting_qc)
        fast = _call_outputs(fast_tool, image_path, label_path, tmp_path / f"fast_{name}", **kwargs)
        assert fast["files"].keys() == reference["files"].keys(), name
        for file_name, content in reference["files"].items():
            assert fast["files"][file_name] == content, (name, file_name)
        assert fast["returned"] == reference["returned"], name
        metadata = json.loads(next(v for k, v in fast["files"].items() if k.endswith(".json") and "metadata" in k))
        covered["tightened"] += sum(1 for m in metadata["cell_metadata"] if m["auto_tighten_iterations"])
        covered["dropped"] += metadata["statistics"]["filtered_by_border"]
        covered["baselines"] += sum(1 for m in metadata["cell_metadata"] if m["baseline_crop_path"])
    assert areas_passed and all(areas_passed)
    assert all(covered.values()), covered                              # every path was exercised
