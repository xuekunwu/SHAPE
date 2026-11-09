"""Guards for the task-087 stable per-cell identity contract inside the analyzer:
fail-closed validation, the ordered identity-sequence digest, DataLoader propagation of
identity WITH each sample, and H5AD round-trip persistence of the reviewed key.

Deliberately does NOT exercise `Cell_State_Analyzer_Tool.execute()` end to end
(that needs a GPU + a network DINOv3 download); it targets the helper functions and the
Dataset directly, mirroring `test_embedding_reuse.py`'s convention.
"""

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

ad = pytest.importorskip("anndata")

from shape.tools.cell_state_analyzer.tool import (  # noqa: E402
    CellCropDataset,
    _identity_sequence_digest,
    _validate_identity_metadata,
)


# --------------------------------------------------------------- _validate_identity_metadata

def test_no_identity_anywhere_returns_none():
    """Legacy behavior: no row carries identity -> callers keep synthetic row names."""
    meta = [{"group": "g0"}, {"group": "g1"}]
    assert _validate_identity_metadata(meta) is None


def test_legacy_cropper_cell_id_alone_is_not_stable_identity():
    """single_cell_cropper's OWN per-crop metadata already carries a 'cell_id' key for
    every existing non-HPA caller of this analyzer (its own override-lookup key, format
    '{group}_{image_name}_cell_{idx:04d}'), but never 'crop_id'. The trigger is a present
    `crop_id` on any row -- 'cell_id' alone (never accompanied by 'crop_id' anywhere in the
    batch) must never trigger the new contract, or every pre-existing pipeline would start
    reshaping/aborting the moment this contract lands."""
    meta = [{"cell_id": "g0_img1_cell_0000", "group": "g0"},
            {"cell_id": "g0_img1_cell_0001", "group": "g0"}]
    assert _validate_identity_metadata(meta) is None


def test_full_identity_returns_ordered_ids():
    meta = [{"cell_id": "hpa/a/x/1.png", "crop_id": "hpa/a/x/1.png"},
            {"cell_id": "hpa/a/x/2.png", "crop_id": "hpa/a/x/2.png"}]
    assert _validate_identity_metadata(meta) == ["hpa/a/x/1.png", "hpa/a/x/2.png"]


def test_partial_identity_aborts():
    meta = [{"cell_id": "hpa/a/x/1.png", "crop_id": "hpa/a/x/1.png"},
            {"group": "g1"}]                      # missing identity entirely
    with pytest.raises(ValueError, match="partial identity"):
        _validate_identity_metadata(meta)


def test_missing_crop_id_on_one_row_aborts():
    meta = [{"cell_id": "hpa/a/x/1.png", "crop_id": "hpa/a/x/1.png"},
            {"cell_id": "hpa/a/x/2.png"}]         # crop_id missing on this row
    with pytest.raises(ValueError, match="partial identity"):
        _validate_identity_metadata(meta)


def test_duplicate_cell_id_aborts():
    meta = [{"cell_id": "hpa/a/x/1.png", "crop_id": "hpa/a/x/1.png"},
            {"cell_id": "hpa/a/x/1.png", "crop_id": "hpa/a/x/1.png"}]
    with pytest.raises(ValueError, match="unique"):
        _validate_identity_metadata(meta)


def test_non_dict_metadata_entries_are_treated_as_no_identity():
    assert _validate_identity_metadata([None, {}]) is None


def test_mismatched_crop_id_aborts():
    """The contract defines the join key as cell_id == crop_id (BRIEF: 'cell_id = crop_id
    = hpa/<organelle>/<cell_line>/<crop_filename>'). A row where they disagree is a
    provenance error and must hard-abort BEFORE any write -- never silently resolved by
    preferring one field over the other (diff-review round 1 finding)."""
    meta = [{"cell_id": "hpa/nucleus/A431/a.png", "crop_id": "DIFFERENT"}]
    with pytest.raises(ValueError, match="cell_id must equal crop_id"):
        _validate_identity_metadata(meta)


def test_mismatched_crop_id_in_a_mixed_batch_aborts():
    """One good (cell_id == crop_id) row plus one mismatched row must still hard-abort --
    a single bad row is enough to reject the whole batch, not just the offending row."""
    meta = [{"cell_id": "hpa/nucleus/A431/a.png", "crop_id": "hpa/nucleus/A431/a.png"},
            {"cell_id": "hpa/nucleus/A431/b.png", "crop_id": "DIFFERENT"}]
    with pytest.raises(ValueError, match="cell_id must equal crop_id"):
        _validate_identity_metadata(meta)


def test_matching_crop_id_batch_still_passes():
    """Sanity check: the new equality gate does not reject the ordinary, contract-
    compliant case (cell_id == crop_id on every row)."""
    meta = [{"cell_id": "hpa/nucleus/A431/a.png", "crop_id": "hpa/nucleus/A431/a.png"},
            {"cell_id": "hpa/nucleus/A431/b.png", "crop_id": "hpa/nucleus/A431/b.png"}]
    assert _validate_identity_metadata(meta) == [
        "hpa/nucleus/A431/a.png", "hpa/nucleus/A431/b.png"]


# ------------------------------------------- diff-review round 3: crop_id-only trigger gap

def test_crop_id_only_all_rows_aborts():
    """`crop_id` is the NEW contract key single_cell_cropper never sets; a batch where
    EVERY row has a `crop_id` but no `cell_id` at all is a genuinely incomplete identity
    request and must hard-abort -- NOT silently fall back to legacy row names (the trigger
    must be crop_id presence alone, not 'both fields present together')."""
    meta = [{"crop_id": "hpa/nucleus/A431/a.png"}, {"crop_id": "hpa/nucleus/A431/b.png"}]
    with pytest.raises(ValueError, match="partial identity"):
        _validate_identity_metadata(meta)


def test_crop_id_only_mixed_with_a_legacy_row_aborts():
    """One crop_id-only row plus one row with no identity metadata at all: the crop_id
    alone is enough to trigger the contract for the WHOLE batch, and neither row supplies
    a valid pair, so this must abort rather than silently return None."""
    meta = [{"crop_id": "hpa/nucleus/A431/a.png"}, {"group": "legacy"}]
    with pytest.raises(ValueError, match="partial identity"):
        _validate_identity_metadata(meta)


def test_whitespace_only_crop_id_alone_is_treated_as_absent_legacy():
    """A blank/whitespace-only crop_id is not a real identity signal -- if that is the ONLY
    thing present anywhere in the batch, the batch is correctly read as 'nothing requested'
    (the safe, inert legacy default), not as a malformed request to abort on."""
    meta = [{"crop_id": "   "}, {"group": "g"}]
    assert _validate_identity_metadata(meta) is None


def test_whitespace_only_cell_id_with_a_real_crop_id_aborts():
    """Once a real crop_id anywhere triggers the contract, a whitespace-only cell_id does
    NOT count as a present pairing partner -- this row is incomplete and must abort."""
    meta = [{"cell_id": "   ", "crop_id": "hpa/nucleus/A431/a.png"}]
    with pytest.raises(ValueError, match="partial identity"):
        _validate_identity_metadata(meta)


def test_non_string_cell_id_with_a_real_crop_id_aborts():
    """A non-string cell_id (a caller bug, e.g. an accidental int) must not be silently
    treated as a usable id once the batch is triggered by a real crop_id elsewhere."""
    meta = [{"cell_id": 123, "crop_id": "hpa/nucleus/A431/a.png"}]
    with pytest.raises(ValueError, match="partial identity"):
        _validate_identity_metadata(meta)


def test_non_string_crop_id_alone_is_treated_as_no_identity_requested():
    """A non-string crop_id (not a real string at all) does not even count as the
    new-contract discriminator being present -- with nothing else in the batch, this reads
    as 'no identity requested', the safe inert default."""
    meta = [{"crop_id": 123, "group": "g"}]
    assert _validate_identity_metadata(meta) is None


def test_empty_string_crop_id_alone_is_treated_as_no_identity_requested():
    meta = [{"crop_id": "", "group": "g"}]
    assert _validate_identity_metadata(meta) is None


# --------------------------------------------------------------- _identity_sequence_digest

def test_identity_sequence_digest_is_order_sensitive():
    """The load-bearing contrast with embedding_fingerprint (crop-order-BLIND): the same
    SET of ids in a different order must produce a DIFFERENT digest, so a reordered input
    sequence cannot reuse a row-associated cache."""
    ids = ["hpa/a/x/1.png", "hpa/a/x/2.png", "hpa/a/x/3.png"]
    assert _identity_sequence_digest(ids) == _identity_sequence_digest(list(ids))
    assert _identity_sequence_digest(ids) != _identity_sequence_digest(list(reversed(ids)))


def test_identity_sequence_digest_differs_for_a_different_id_set():
    a = ["hpa/a/x/1.png", "hpa/a/x/2.png"]
    b = ["hpa/a/x/1.png", "hpa/a/x/3.png"]
    assert _identity_sequence_digest(a) != _identity_sequence_digest(b)


# --------------------------------------------------------------- DataLoader propagation

def _write_tiny_png(path):
    Image.fromarray((np.random.default_rng(0).random((8, 8)) * 255).astype(np.uint8)).save(path)


def test_dataset_carries_identity_with_each_sample(tmp_path):
    """Identity travels WITH the sample through the DataLoader (contract SS4): pulling the
    dataset out of order (here, a plain DataLoader with shuffle=False over 3 items) must
    still hand back the correct identity for each image, never index-zipped afterwards."""
    paths = [str(tmp_path / f"crop_{i}.png") for i in range(3)]
    for p in paths:
        _write_tiny_png(p)
    cell_ids = [f"hpa/a/x/crop_{i}.png" for i in range(3)]
    mask_ids = ["", "", "mask_2"]
    rel_paths = [f"a/x/crop_{i}.png" for i in range(3)]

    dataset = CellCropDataset(
        paths, groups=["g0", "g0", "g1"], image_names=["img0", "img0", "img1"],
        transform=None, cell_ids=cell_ids, mask_ids=mask_ids, rel_paths=rel_paths)

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    seen_cell_ids, seen_mask_ids, seen_rel_paths = [], [], []
    for _v1, _v2, _names, _groups, identity in loader:
        seen_cell_ids.extend(identity["cell_id"])
        seen_mask_ids.extend(identity["mask_id"])
        seen_rel_paths.extend(identity["rel_path"])

    assert seen_cell_ids == cell_ids
    assert seen_mask_ids == mask_ids
    assert seen_rel_paths == rel_paths


def test_dataset_defaults_identity_to_empty_string_when_not_supplied(tmp_path):
    """Legacy construction (no cell_ids/mask_ids/rel_paths): identity is present but empty,
    never None -- default_collate cannot batch a mix of None values, and "" is the
    established "not requested" sentinel `_validate_identity_metadata` also uses."""
    p = tmp_path / "crop_0.png"
    _write_tiny_png(p)
    dataset = CellCropDataset([str(p)], groups=["g0"], image_names=["img0"],
                                            transform=None)
    _v1, _v2, _name, _group, identity = dataset[0]
    assert identity == {"cell_id": "", "mask_id": "", "rel_path": ""}


# ------------------------------------ diff-review round 4: mask_id=None collation crash

def test_identity_value_none_safe_at_the_element_level():
    """`_identity_value` must guard the PER-ELEMENT value, not just whole-list presence:
    a supplied (truthy, non-empty) list whose elements are `None` -- exactly HPA's
    `mask_ids`, since no real mask manifest exists -- must still normalize to `""`."""
    from shape.tools.cell_state_analyzer.tool import _identity_value

    assert _identity_value(None, 0) == ""                     # whole list absent
    assert _identity_value([None, None], 0) == ""              # HPA-shaped: present list, None element
    assert _identity_value([None, None], 1) == ""
    assert _identity_value(["real", None], 0) == "real"
    assert _identity_value(["real", None], 1) == ""


def test_hpa_shaped_mask_id_none_collates_without_error(tmp_path):
    """Regression for the run-breaking collation crash (diff-review round 4): HPA has NO
    masks, so every record's `mask_id` is `None` (`discover_organelles`), giving the
    Dataset a present-but-all-None `mask_ids` list -- exactly the case a naive
    `self.mask_ids[idx] if self.mask_ids else ""` (whole-list truthiness) misses, because a
    non-empty list of `None`s is itself truthy. Uncaught, a raw `None` reaches
    `torch.utils.data.dataloader.default_collate`, which raises `TypeError` on `NoneType` --
    this would have crashed the real 13-unit HPA regeneration at the first batch >1. Covers
    both the single-item `__getitem__` path and the batched default-collate path."""
    paths = [str(tmp_path / f"crop_{i}.png") for i in range(2)]
    for p in paths:
        _write_tiny_png(p)
    cell_ids = ["hpa/nucleus/A431/crop_0.png", "hpa/nucleus/A431/crop_1.png"]
    rel_paths = ["nucleus/A431/crop_0.png", "nucleus/A431/crop_1.png"]

    dataset = CellCropDataset(
        paths, groups=["A431", "A431"], image_names=["img0", "img0"], transform=None,
        cell_ids=cell_ids, mask_ids=[None, None], rel_paths=rel_paths)

    # single-item path
    _v1, _v2, _name, _group, identity = dataset[0]
    assert identity["cell_id"] == cell_ids[0]
    assert identity["mask_id"] == ""
    assert identity["rel_path"] == rel_paths[0]

    # batched default-collate path (this is what actually crashed before the fix)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batches = list(loader)
    assert len(batches) == 1
    _v1, _v2, _names, _groups, batched_identity = batches[0]
    assert batched_identity["cell_id"] == cell_ids
    assert batched_identity["mask_id"] == ["", ""]
    assert batched_identity["rel_path"] == rel_paths


# --------------------------------------------------------------- H5AD round-trip persistence

def test_h5ad_round_trip_preserves_obs_names_and_identity_columns(tmp_path):
    """The exact write/read path `execute()` uses (`ad.AnnData(...); adata.write(path)`,
    later `ad.read_h5ad(path)`): obs_names and the identity obs columns must survive."""
    cell_ids = ["hpa/nucleus/A431/img1_0.png", "hpa/nucleus/A431/img1_1.png"]
    adata = ad.AnnData(X=np.random.default_rng(0).normal(size=(2, 4)).astype("float32"))
    adata.obs_names = cell_ids
    adata.obs["cell_id"] = cell_ids
    adata.obs["crop_id"] = cell_ids
    adata.obs["crop_rel_path"] = ["nucleus/A431/img1_0.png", "nucleus/A431/img1_1.png"]
    adata.uns["identity_sequence_digest"] = _identity_sequence_digest(cell_ids)

    path = tmp_path / "cell_state_analyzed.h5ad"
    adata.write(path)
    reloaded = ad.read_h5ad(path)

    assert list(reloaded.obs_names) == cell_ids
    assert list(reloaded.obs["cell_id"].astype(str)) == cell_ids
    assert list(reloaded.obs["crop_id"].astype(str)) == cell_ids
    assert reloaded.uns["identity_sequence_digest"] == _identity_sequence_digest(cell_ids)
