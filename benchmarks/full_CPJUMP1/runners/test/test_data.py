import copy
import numpy as np
import pytest
import tifffile

from benchmarks.full_CPJUMP1.harness.data import (
    CHANNELS, assemble_view, development, technical, validate_manifest, well_profiles, within)
from benchmarks.full_CPJUMP1.harness.cli import smoke_images
from benchmarks.full_CPJUMP1.harness.pipeline import select_development


def fixed_manifest():
    wells, sites, files = [], [], []
    for plate in ("P1", "P2"):
        designs = [(f"G{i}", f"g{i}_{j}", "calibration" if i < 20 else "evaluation", "trt")
                   for i in range(130) for j in range(2 if i < 112 else 1)]
        designs += [("NO_SITE", f"N{i}", "control", "negcon") for i in range(8)]
        for i, (gene, guide, role, pert) in enumerate(designs):
            usage = "evaluation_only" if role == "evaluation" else ("development_val" if i % 5 == 0 else "development_train")
            w = dict(plate=plate, well=f"W{i}", gene=gene, guide_sequence=guide, role=role,
                     pert_type=pert, usage=usage, sample=guide)
            wells.append(w)
            for site in (1, 9):
                ident = f"{plate}/W{i}/{site}"
                sites.append({**w, "site": site, "id": ident})
                files.extend({"site_id": ident, "channel": ch, "relative_path": f"{ident}/{ch}.tiff"} for ch in range(1, 6))
    return dict(plates=["P1", "P2"], wells=wells, sites=sites, files=files)


def test_manifest_fixed_design_and_rejects_channel_duplicate():
    m = fixed_manifest()
    validate_manifest(m)
    bad = copy.deepcopy(m)
    bad["files"][1]["channel"] = 1
    with pytest.raises(ValueError, match="channels"):
        validate_manifest(bad)


def test_manifest_rejects_evaluation_leak():
    m = fixed_manifest()
    m["sites"][100]["usage"] = "development_train"
    with pytest.raises(ValueError):
        validate_manifest(m)


@pytest.mark.parametrize("relative", ["../private", "/etc/passwd", "C:/data", "a/../../x", "a\\x"])
def test_path_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        within(tmp_path, relative)


def test_five_channel_roundtrip_and_real_shape_loader(tmp_path):
    from shape.utils.image_data import ImageData
    arrays = [np.arange(48 * 48, dtype=np.uint16).reshape(48, 48) + i * 9000 for i in range(5)]
    paths = []
    for i, x in enumerate(arrays):
        path = tmp_path / f"ch{i + 1}.tiff"
        tifffile.imwrite(path, x)
        paths.append(path)
    output, segment = tmp_path / "full.ome.tif", tmp_path / "seg.ome.tif"
    facts = assemble_view(paths, output, segment)
    assert facts["shape"] == [5, 48, 48]
    full = ImageData.from_path(str(output))
    seg = ImageData.from_path(str(segment))
    assert full.data.shape == (48, 48, 5)
    assert seg.data.shape == (48, 48, 2)
    np.testing.assert_array_equal(full.data, np.stack(arrays, axis=-1))
    np.testing.assert_array_equal(seg.data[..., 0], arrays[4])
    np.testing.assert_array_equal(seg.data[..., 1], arrays[1])
    with tifffile.TiffFile(output) as f:
        assert f.series[0].axes == "CYX"
        assert all(name in f.ome_metadata for name in CHANNELS)


def test_rejects_lossy_or_mismatched_source(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.tif"
        tifffile.imwrite(p, np.zeros((32, 32), dtype=np.uint8 if i == 4 else np.uint16))
        paths.append(p)
    with pytest.raises(ValueError, match="uint16"):
        assemble_view(paths, tmp_path / "out.ome.tif", tmp_path / "seg.ome.tif")


def cell(i, site, usage):
    return {"cell_id": f"{site}_{i}", "image_id": site, "crop_path": f"/{site}/{i}.tif", "well": site,
            "plate": "P1", "usage": usage, "area": 100, "gene": "SECRET", "guide": "SECRET"}


def test_training_filters_before_ssl_and_strips_biology():
    cells = [cell(i, site, usage) for site, usage in (("a", "development_train"),
             ("b", "development_val"), ("c", "evaluation_only")) for i in range(5)]
    paths, meta = select_development(cells, 2)
    assert len(paths) == 4
    assert not any("/c/" in p for p in paths)
    assert all(not {"gene", "guide", "calibration_group", "usage"} & m.keys() for m in meta)
    assert {m["split"] for m in meta} == {"train", "val"}
    assert select_development(list(reversed(cells)), 2) == (paths, meta)
    assert "split" not in technical(cells[0])


def test_training_split_fails_closed():
    with pytest.raises(ValueError, match="unknown usage"):
        development([cell(1, "a", "test")])
    with pytest.raises(ValueError, match="both train and val"):
        development([cell(1, "a", "development_train")])
    with pytest.raises(ValueError, match="well leakage"):
        development([cell(1, "a", "development_train"), cell(2, "a", "development_val")])


def test_well_profiles_weight_sites_not_cells():
    images = [{"image_id": site, "well": "w", "plate": "p", "usage": "evaluation_only",
               "negative_control": False, "calibration_group": ""} for site in ("s1", "s2")]
    cells = [{"cell_id": str(i), "image_id": "s1" if i < 9 else "s2"} for i in range(10)]
    x = np.array([[0., 2.]] * 9 + [[10., 4.]])
    profiles, meta = well_profiles(x, cells, images)
    np.testing.assert_array_equal(profiles, [[5., 3.]])
    assert len(meta) == 1
    with pytest.raises(ValueError, match="every selected site"):
        well_profiles(x[:-1], cells[:-1], images)


def test_smoke_keeps_whole_wells_and_three_splits():
    rows = []
    for s in fixed_manifest()["sites"]:
        rows.append({**s, "image_id": s["id"], "well": f"{s['plate']}_{s['well']}", "negative_control": s["pert_type"] == "negcon"})
    chosen = smoke_images(rows)
    assert len(chosen) == 20
    assert {r["usage"] for r in chosen} == {"development_train", "development_val", "evaluation_only"}
    assert all(sum(r["well"] == w for r in chosen) == 2 for w in {r["well"] for r in chosen})
    for plate in ("P1", "P2"):
        plate_rows = [r for r in chosen if r["plate"] == plate]
        assert {r["usage"] for r in plate_rows} == {
            "development_train", "development_val", "evaluation_only"
        }
        assert len({r["well"] for r in plate_rows if r["usage"] == "development_val"}) == 2
        assert len({r["well"] for r in plate_rows if r["negative_control"]}) == 1


def test_diagnostics_json_keeps_structure_and_null_missing_readouts(tmp_path):
    from benchmarks.full_CPJUMP1.harness.data import read_json, write_json
    p = tmp_path / "diagnostics.json"
    write_json(p, {"values": np.array([1., np.nan]), "missing": float("inf"), "count": np.int64(4)})
    assert read_json(p) == {"values": [1., None], "missing": None, "count": 4}
