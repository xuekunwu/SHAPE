"""Amendment A1: the packed-array pixel source (`execute(crop_array=...)`, `ArrayCrops`).

The same crops as TIFF files (read by the real skimage reader) and as rows of one .npy (in another order, with an
unread row) give bit-identical tensors: the pixels and their normalization, the channel spec, the per_plate
reference and affine, the training views (anchor and partner) under fixed seeds, the trained call's losses,
weights and embedding, the frozen embedding and the embed_only embedding, also under nn_other_image (frozen and
current), same_well, compute_alignment and val_fraction. A call without crop_array never builds the array source and
keeps the file fingerprint (golden values of HEAD fe87139e). The array fingerprint is the content of the rows read
plus the rows and the metadata, never the path, so a copy elsewhere (another node) reuses the embedding. The rows a
call needs are read once into memory with plain file reads (never memory-mapped); a transient read error is tried
again, a persistent one is the call's error result; a pickle holds the rows, not the pixels. CPU, stub backbone.
"""
import errno
import pickle
import shutil

import anndata as ad
import numpy as np
import pytest
import tifffile
import torch
from skimage import io

import shape.tools.cell_state_analyzer.tool as tool_mod
from shape.tools.cell_state_analyzer import quant


@pytest.fixture
def sources(tmp_path, crop_factory):
    """(paths, metadata, array path): `make_crops()` as TIFF files and as one packed array whose row 0 is an unread
    crop and whose other rows hold the crops in reverse order; every metadata row carries its `array_row`."""
    images, meta = crop_factory()
    crops = list(images.values())
    (tmp_path / "tif").mkdir()
    paths = []
    for i, (name, image) in enumerate(images.items()):
        tifffile.imwrite(tmp_path / "tif" / name, image)
        paths.append(str(tmp_path / "tif" / name))
        meta[i]["array_row"] = len(crops) - i
    unread = np.full_like(crops[0], 7)
    array = tmp_path / "packed" / "images.npy"
    array.parent.mkdir()
    np.save(array, np.stack([unread] + crops[::-1]))
    return paths, meta, str(array)


def _array(sources):
    paths, meta, array = sources
    return tool_mod.array_crops(array, meta)


def test_an_array_crop_reads_as_its_file_and_has_a_stable_key(sources):
    paths, meta, array = sources
    crops = _array(sources)
    assert len(crops) == len(paths) and crops[0] == f"images.npy#{len(paths)}" and list(crops)[-1] == "images.npy#1"
    for i, path in enumerate(paths):
        raw = tool_mod._crop_pixels(crops, i)
        assert raw.dtype == io.imread(path).dtype and np.array_equal(raw, io.imread(path))
        assert torch.equal(tool_mod._normalized_channel_tensor(raw, crops[i]),
                           tool_mod._normalized_channel_tensor(io.imread(path), path))
    assert tool_mod.infer_channel_spec(crops) == tool_mod.infer_channel_spec(paths) == (6, ["CHW"])
    assert tool_mod._crop_name(crops, 0) == crops[0]
    assert tool_mod._crop_name(paths, 0) == "crop_P1_W0_s0_c0"
    sub = tool_mod._crop_subset(crops, [3, 1])
    assert isinstance(sub, tool_mod.ArrayCrops) and list(sub) == [crops[3], crops[1]] and sub.loaded is crops.loaded


def test_a_uint16_array_normalizes_as_its_files(tmp_path):
    rng = np.random.default_rng(3)
    stack = (rng.random((3, 6, 20, 20)) * 4000).astype(np.uint16)
    np.save(tmp_path / "u16.npy", stack)
    crops = tool_mod.ArrayCrops(tmp_path / "u16.npy", [2, 0])
    for i, row in enumerate((2, 0)):
        tifffile.imwrite(tmp_path / f"c{row}.tif", stack[row])
        path = str(tmp_path / f"c{row}.tif")
        assert torch.equal(tool_mod._normalized_channel_tensor(tool_mod._crop_pixels(crops, i), crops[i]),
                           tool_mod._normalized_channel_tensor(io.imread(path), path))


def test_the_per_plate_reference_and_affine_are_bit_identical(sources):
    paths, meta, _ = sources
    crops = _array(sources)
    plates = [m["plate"] for m in meta]
    assert tool_mod.plate_reference(crops, plates) == tool_mod.plate_reference(paths, plates)
    p1 = [i for i, plate in enumerate(plates) if plate == "P1"]
    reference = tool_mod.plate_reference(tool_mod._crop_subset(crops, p1), [plates[i] for i in p1])
    assert reference == tool_mod.plate_reference([paths[i] for i in p1], [plates[i] for i in p1])
    for got, want in zip(tool_mod.norm_affine(crops, plates, reference),     # P2: its own crops' statistics
                         tool_mod.norm_affine(paths, plates, reference)):
        assert np.array_equal(got, want)


@pytest.mark.parametrize("augmentation", ["strong", "light"])
def test_the_training_views_are_bit_identical_under_fixed_seeds(cpu_tool, sources, augmentation):
    paths, meta, _ = sources
    crops = _array(sources)
    plates = [m["plate"] for m in meta]
    transform = cpu_tool._get_augmentation_transform(augmentation=augmentation)

    def views(source):
        affine = tool_mod.norm_affine(source, plates, tool_mod.plate_reference(source, plates))
        dataset = tool_mod.CellCropDataset(source, transform=transform, affine=affine, plate_ids=plates,
                                           units=tool_mod.CropUnits(quant.identities(meta), seed=0),
                                           view_pairing="same_site")
        torch.manual_seed(0)
        return [dataset[i] for i in range(len(dataset))]

    for (a1, a2, _, _, a_id), (f1, f2, _, _, f_id) in zip(views(crops), views(paths)):
        assert torch.equal(a1, f1) and torch.equal(a2, f2) and a_id == f_id


def _call(tool, tmp_path, subdir, source, meta, **kwargs):
    kwargs.setdefault("max_epochs", 1)
    if isinstance(source, str):
        kwargs["crop_array"] = source
    else:
        kwargs["cell_crops"] = list(source)
    return tool.execute(cell_metadata=[dict(m) for m in meta], query_cache_dir=str(tmp_path), output_subdir=subdir,
                        **kwargs)


def test_execute_on_the_array_equals_execute_on_the_files(cpu_tool, sources, tmp_path):
    paths, meta, array = sources
    levers = {"split_key": "split", "intensity_norm": "per_plate", "view_pairing": "same_site",
              "learning_rate": 1e-3}
    on_files = _call(cpu_tool, tmp_path, "files", paths, meta, **levers)
    on_array = _call(cpu_tool, tmp_path, "array", array, meta, **levers)
    assert on_array["epochs_trained"] == on_files["epochs_trained"] == 1
    assert on_array["loss_history"] == on_files["loss_history"]
    states = [torch.load(r["resume_best"], map_location="cpu", weights_only=True) for r in (on_array, on_files)]
    assert states[0]["intensity_reference"] == states[1]["intensity_reference"]
    assert all(torch.equal(states[0]["model"][k], states[1]["model"][k]) for k in states[1]["model"])
    x_array, x_files = (ad.read_h5ad(r["adata_path"]) for r in (on_array, on_files))
    assert np.array_equal(x_array.X, x_files.X)
    assert list(x_array.obs["image_name"]) == [f"images.npy#{m['array_row']}" for m in meta]    # no image_name given

    frozen = [_call(cpu_tool, tmp_path, f"frozen_{name}", source, meta, adaptation_depth="zero_shot")
              for name, source in (("array", array), ("files", paths))]
    assert np.array_equal(*(ad.read_h5ad(r["adata_path"]).X for r in frozen))

    embedded = [_call(cpu_tool, tmp_path, f"embed_{name}", source, meta, embed_only=True,
                      init_checkpoint=result["resume_best"])
                for name, source, result in (("array", array, on_array), ("files", paths, on_files))]
    assert embedded[0]["intensity_norm_used"] == "per_plate"
    assert np.array_equal(*(ad.read_h5ad(r["adata_path"]).X for r in embedded))


@pytest.mark.parametrize("levers", [
    {"view_pairing": "nn_other_image", "nn_space": "frozen", "nn_k": 2},
    {"view_pairing": "nn_other_image", "nn_space": "current", "nn_k": 2, "nn_refresh_epochs": 0.5, "batch_size": 4},
    {"view_pairing": "same_well"},
    {"compute_alignment": True},
    {"val_fraction": 0.5},
], ids=["nn_frozen", "nn_current", "same_well", "alignment", "val_fraction"])
def test_execute_on_the_array_equals_execute_on_the_files_under(cpu_tool, sources, tmp_path, levers):
    paths, meta, array = sources
    levers = {"learning_rate": 1e-3, **({} if "val_fraction" in levers else {"split_key": "split"}), **levers}
    on_files = _call(cpu_tool, tmp_path, "files", paths, meta, **levers)
    on_array = _call(cpu_tool, tmp_path, "array", array, meta, **levers)
    assert "error" not in on_files and on_array["epochs_trained"] == on_files["epochs_trained"] == 1
    assert on_array["loss_history"] == on_files["loss_history"] and on_array["split"] == on_files["split"]
    assert on_array["val_history"] == on_files["val_history"] and on_array["alignment"] == on_files["alignment"]
    assert (on_array["alignment"] is not None) == bool(levers.get("compute_alignment"))
    states = [torch.load(r["resume_best"], map_location="cpu", weights_only=True) for r in (on_array, on_files)]
    assert all(torch.equal(states[0]["model"][k], states[1]["model"][k]) for k in states[1]["model"])
    assert np.array_equal(*(ad.read_h5ad(r["adata_path"]).X for r in (on_array, on_files)))


# the file fingerprints of HEAD fe87139e (before amendment A1), computed with its tool.py: the default path keeps them
GOLDEN_FILE_FINGERPRINT = "2ea15f428f53428fa72a46b189e5117c9b26cffb0c5c8a3140f82630f8f13624"
# A12 (2026-10-11): the default augmentation is recorded as "strong" (was "shape"), so the execute fingerprint changed
GOLDEN_EXECUTE_FINGERPRINT = "7129abe21839946117221d5766d74b5e487a3765518af07d10189ee1bdd321e7"


def test_the_file_fingerprint_keeps_its_golden_value(crop_factory, tmp_path, monkeypatch):
    images, _ = crop_factory()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "golden").mkdir()
    for i, image in enumerate(list(images.values())[:3]):
        (tmp_path / "golden" / f"c{i}.bin").write_bytes(image.tobytes())
    crops = ["golden/c1.bin", "golden/c0.bin", "golden/missing.tif", "golden/c2.bin"]     # one path without a file
    params = {"seed": 0, "input_channels": 6, "learning_rate": 3e-5}
    assert tool_mod.embedding_fingerprint(params, crops) == GOLDEN_FILE_FINGERPRINT


def test_a_default_file_call_keeps_its_golden_fingerprint(run_call):
    out = run_call(adaptation_depth="zero_shot")       # the `crops` fixture: file paths read by a patched imread
    assert ad.read_h5ad(out["adata_path"]).uns["embedding_fingerprint"] == GOLDEN_EXECUTE_FINGERPRINT


# ---- the in-memory read --------------------------------------------------------------------------------------

def test_the_rows_are_read_once_into_memory_never_memory_mapped(sources, monkeypatch):
    paths, meta, array = sources
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("np.load reads the crop array"))
    monkeypatch.setattr(np, "memmap", lambda *args, **kwargs: pytest.fail("the crop array is memory-mapped"))
    crops = tool_mod.array_crops(array, meta)
    monkeypatch.undo()
    assert crops.loaded["rows"] == tuple(range(1, len(meta) + 1))                        # row 0 is never read
    assert crops.loaded["pixels"].nbytes == len(meta) * 6 * 24 * 24
    np.save(array, np.zeros((len(meta) + 1, 6, 24, 24), np.uint8))                     # the file changes ...
    for i, path in enumerate(paths):                                                     # ... the crops do not
        assert np.array_equal(crops.raw(i), io.imread(path))


def test_a_transient_read_error_is_tried_again(sources, monkeypatch):
    paths, meta, array = sources
    sleeps, real, tries = [], tool_mod._read_array_rows, []
    monkeypatch.setattr(tool_mod.time, "sleep", sleeps.append)

    def flaky(path, rows):
        tries.append(path)
        if len(tries) < 3:
            raise OSError((errno.EIO, 108)[len(tries) - 1], "Input/output error")
        return real(path, rows)

    monkeypatch.setattr(tool_mod, "_read_array_rows", flaky)
    crops = tool_mod.array_crops(array, meta)
    assert len(tries) == 3 and sleeps == [5, 10]
    assert all(np.array_equal(crops.raw(i), io.imread(path)) for i, path in enumerate(paths))


def test_a_persistent_read_error_fails_the_call_with_an_error_result(cpu_tool, sources, tmp_path, monkeypatch):
    paths, meta, array = sources
    sleeps, real = [], tool_mod._read_array_rows
    monkeypatch.setattr(tool_mod.time, "sleep", sleeps.append)
    monkeypatch.setattr(tool_mod, "_read_array_rows",
                        lambda path, rows: (_ for _ in ()).throw(OSError(errno.EIO, "Input/output error")))
    out = _call(cpu_tool, tmp_path, "eio", array, meta)
    assert "unreadable" in out["error"] and "Input/output error" in out["error"] and "seconds" in out
    assert sleeps == list(tool_mod.READ_RETRY_DELAYS)
    monkeypatch.setattr(tool_mod, "_read_array_rows", real)
    sleeps.clear()
    out = _call(cpu_tool, tmp_path, "missing", str(tmp_path / "no_such.npy"), meta)
    assert "FileNotFoundError" in out["error"] and sleeps == []                           # not transient: no retry


def test_a_pickle_holds_the_rows_not_the_pixels(sources):
    paths, meta, array = sources
    crops = tool_mod.array_crops(array, meta)
    full, one = pickle.dumps(crops), pickle.dumps(crops.subset([3]))
    assert len(one) < 2000 and len(full) < crops.loaded["pixels"].nbytes / 4
    back = pickle.loads(one)
    assert list(back) == [crops[3]] and back.loaded["rows"] == (crops.rows[3],)        # only its own row is read
    assert np.array_equal(back.raw(0), crops.raw(3)) and back.metadata == [meta[3]]
    assert back.content_record() == crops.subset([3]).content_record()


def test_a_call_without_crop_array_never_builds_the_array_source(cpu_tool, sources, tmp_path, monkeypatch):
    import inspect

    paths, meta, _ = sources
    assert inspect.signature(cpu_tool.execute).parameters["crop_array"].default is None
    monkeypatch.setattr(tool_mod, "array_crops", lambda *args: pytest.fail("array source built"))
    monkeypatch.setattr(tool_mod.ArrayCrops, "__init__", lambda *args: pytest.fail("array source built"))
    out = _call(cpu_tool, tmp_path, "files", paths, meta, adaptation_depth="zero_shot")
    adata = ad.read_h5ad(out["adata_path"])
    assert list(adata.obs["image_name"])[0] == "crop_P1_W0_s0_c0"


def test_the_fingerprint_is_the_arrays_content_not_its_path(cpu_tool, sources, tmp_path):
    paths, meta, array = sources
    elsewhere = tmp_path / "other_node" / "copy_of_images.npy"
    elsewhere.parent.mkdir()
    shutil.copyfile(array, elsewhere)
    params = {"seed": 0}

    def fingerprint(path, rows_meta=meta):
        return tool_mod.embedding_fingerprint(params, tool_mod.array_crops(path, rows_meta))

    reference = fingerprint(array)
    assert fingerprint(elsewhere) == reference
    assert reference != tool_mod.embedding_fingerprint(params, paths)
    assert fingerprint(array, meta[1:]) != reference                             # another row list
    assert fingerprint(array, [{**m, "split": "val"} for m in meta]) != reference  # other metadata

    stack = np.load(elsewhere)
    stack[0] += 1                                                                # the unread row
    np.save(elsewhere, stack)
    assert fingerprint(elsewhere) == reference
    stack[1] += 1                                                                # a row the call reads
    np.save(elsewhere, stack)
    assert fingerprint(elsewhere) != reference
    shutil.copyfile(array, elsewhere)

    first = _call(cpu_tool, tmp_path, "frozen", array, meta, adaptation_depth="zero_shot")
    again = _call(cpu_tool, tmp_path, "frozen", str(elsewhere), meta, adaptation_depth="zero_shot",
                  reuse_embedding=True)
    assert again["mode"] == "reused-embedding"
    assert again["embedding_fingerprint"] == ad.read_h5ad(first["adata_path"]).uns["embedding_fingerprint"]


def test_crop_array_input_errors(cpu_tool, sources, tmp_path):
    paths, meta, array = sources
    with pytest.raises(ValueError, match="not both"):
        cpu_tool.execute(cell_crops=paths, cell_metadata=meta, crop_array=array, query_cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="cell_metadata"):
        cpu_tool.execute(crop_array=array, query_cache_dir=str(tmp_path))
    for bad in (None, 1.0, True, "3"):
        rows = [dict(m) for m in meta]
        rows[2]["array_row"] = bad
        with pytest.raises(ValueError, match="integer array_row"):
            _call(cpu_tool, tmp_path, "bad", array, rows)
    rows = [dict(m) for m in meta]
    rows[0]["array_row"] = len(meta) + 1
    with pytest.raises(ValueError, match="outside"):
        _call(cpu_tool, tmp_path, "bad", array, rows)
    for name, stack in (("float.npy", np.zeros((2, 6, 8, 8), np.float32)), ("flat.npy", np.zeros((2, 8, 8), np.uint8)),
                        ("hwc.npy", np.zeros((2, 24, 24, 6), np.uint8)), ("small.npy", np.zeros((2, 6, 4, 4), np.uint8)),
                        ("fortran.npy", np.asfortranarray(np.zeros((2, 6, 8, 8), np.uint8)))):
        np.save(tmp_path / name, stack)
        with pytest.raises(ValueError, match=r"\(N, C, H, W\)"):
            tool_mod.ArrayCrops(tmp_path / name, [0])
