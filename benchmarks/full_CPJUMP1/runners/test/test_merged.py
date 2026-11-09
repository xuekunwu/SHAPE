import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks.full_CPJUMP1.harness import data
from benchmarks.full_CPJUMP1.harness.merged import import_view, load_index, source_path

COLUMNS = ["output", "shape", "dtype", *[f"ch{c}" for c in range(1, 6)]]


def write_fields(root, rows):
    with (root / "fields.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def exported(tmp_path):
    raw_root, merged = tmp_path / "download", tmp_path / "download_5ch"
    raw_root.mkdir()
    merged.mkdir()
    well = dict(plate="P1", well="A01", gene="hidden_gene", guide_sequence="hidden_guide",
                role="calibration", pert_type="trt", usage="development_train", sample="sample1")
    manifest = {"plates": ["P1"], "wells": [well], "sites": [], "files": [], "limits": {}}
    rows = []
    for site in (2, 9):
        ident = f"P1/A01/{site}"
        manifest["sites"].append({**well, "id": ident, "site": site})
        row = {"shape": "(5, 1080, 1080)", "dtype": "uint16"}
        channels = []
        for channel in range(1, 6):
            rel = f"P1/A01/f{site}-ch{channel}.tiff"
            path = raw_root / "raw" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((1080, 1080), site * 100 + channel * 1000, np.uint16)
            pixels[0, 0] += 1
            tifffile.imwrite(path, pixels)
            channels.append(pixels)
            row[f"ch{channel}"] = str(path)
            manifest["files"].append({"site_id": ident, "relative_path": rel, "channel": channel})
        target = merged / "raw" / "P1" / "A01" / f"f{site}-5ch.ome.tif"
        target.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(target, np.stack(channels), ome=True, photometric="minisblack", metadata={"axes": "CYX"})
        row["output"] = str(target)
        rows.append(row)
    data.write_json(raw_root / "manifest.json", manifest)
    write_fields(merged, rows)
    (merged / "COMPLETE").write_text("fields=2\naxes=CYX\nchannels=1,2,3,4,5\npixels=verified\n")
    return raw_root, merged, manifest, rows


def test_index_uses_channel_paths_not_csv_order(exported):
    root, merged, manifest, rows = exported
    write_fields(merged, list(reversed(rows)))
    index, provenance = load_index(merged, root, manifest)
    assert index["P1/A01/2"] == Path(rows[0]["output"])
    assert index["P1/A01/9"] == Path(rows[1]["output"])
    assert provenance["mode"] == "import_merged"
    assert provenance["fields_sha256"] == data.digest(merged / "fields.csv")


@pytest.mark.parametrize("problem", ["missing", "duplicate", "same_output", "mixed_site", "swap_columns", "bad_shape", "bad_dtype"])
def test_index_rejects_invalid_identity_or_contract(exported, problem):
    root, merged, manifest, rows = exported
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(dict(rows[0]))
    elif problem == "same_output":
        rows[1]["output"] = rows[0]["output"]
    elif problem == "mixed_site":
        rows[0]["ch5"] = rows[1]["ch5"]
    elif problem == "swap_columns":
        rows[0]["ch1"], rows[0]["ch5"] = rows[0]["ch5"], rows[0]["ch1"]
    elif problem == "bad_shape":
        rows[0]["shape"] = "(1080, 1080, 5)"
    else:
        rows[0]["dtype"] = "uint8"
    write_fields(merged, rows)
    with pytest.raises(ValueError):
        load_index(merged, root, manifest)


def test_index_rejects_false_completion_and_escaped_source(exported, tmp_path):
    root, merged, manifest, rows = exported
    outsider = tmp_path / "outside.tif"
    outsider.write_bytes(b"not in merged root")
    with pytest.raises(ValueError, match="escapes"):
        source_path(merged, str(outsider))
    (merged / "COMPLETE").write_text("fields=2\naxes=YXC\nchannels=1,2,3,4,5\npixels=verified\n")
    with pytest.raises(ValueError, match="COMPLETE"):
        load_index(merged, root, manifest)


def test_import_keeps_exact_bytes_and_independent_source(exported, tmp_path):
    _, _, _, rows = exported
    row = rows[0]
    source = Path(row["output"])
    output, segment = tmp_path / "cache/full.ome.tif", tmp_path / "cache/seg.ome.tif"
    before = data.digest(source)
    facts = import_view(source, [Path(row[f"ch{c}"]) for c in range(1, 6)], output, segment)
    assert data.digest(output) == before == facts["sha256"]
    assert not output.samefile(source)
    original = tifffile.imread(source)
    np.testing.assert_array_equal(tifffile.imread(segment), original[[4, 1]])
    with tifffile.TiffFile(segment) as image:
        assert image.series[0].axes == "CYX"
        assert 'Name="DNA"' in image.ome_metadata and 'Name="AGP"' in image.ome_metadata
    output.write_bytes(b"mutating the cache must not modify the source")
    assert data.digest(source) == before


@pytest.mark.parametrize("problem", ["pixels", "channel_order", "dtype", "axes"])
def test_import_checks_pixels_not_just_complete_marker(exported, tmp_path, problem):
    _, _, _, rows = exported
    row = rows[0]
    source = Path(row["output"])
    array = tifffile.imread(source)
    axes = "CYX"
    if problem == "pixels":
        array[2, 40, 50] += 1
    elif problem == "channel_order":
        array = array[::-1]
    elif problem == "dtype":
        array = array.astype(np.uint8)
    else:
        axes = "ZYX"
    tifffile.imwrite(source, array, ome=True, photometric="minisblack", metadata={"axes": axes})
    with pytest.raises(ValueError):
        import_view(source, [Path(row[f"ch{c}"]) for c in range(1, 6)], tmp_path / "out.tif", tmp_path / "seg.tif")
    assert not (tmp_path / "out.tif").exists()


def test_prepare_reuses_stacks_and_keeps_biology_private(exported, tmp_path, monkeypatch):
    root, merged, manifest, _ = exported
    # Dataset-size validation is covered with the fixed 1000-view design in test_data.
    monkeypatch.setattr(data, "MANIFEST_SHA256", data.digest(root / "manifest.json"))
    monkeypatch.setattr(data, "validate_manifest", lambda value: None)
    monkeypatch.setattr(data, "assemble_view", lambda *args: pytest.fail("must not re-merge raw channels"))
    out = tmp_path / "prepared"
    data.prepare(root, out, merged_dir=merged)
    rows = data.load_prepared(out, verify_pixels=True)
    assert len(rows) == 2
    assert data.read_json(out / "PREPARE.json")["assembly"]["mode"] == "import_merged"
    assert len(data.read_json(out / "private/raw_hashes.json")) == 10
    assert "hidden_gene" not in (out / "public/images.json").read_text()
    assert "hidden_guide" not in (out / "public/images.json").read_text()
    assert all("merged_source" not in r for r in rows)
    private = data.read_json(out / "private/labels.json")
    assert private[0]["gene"] == "hidden_gene" and private[0]["source_site"] == "P1/A01/2"
    with pytest.raises(FileExistsError):
        data.prepare(root, out, merged_dir=merged)


def test_bad_index_creates_no_prepared_marker(exported, tmp_path, monkeypatch):
    root, merged, _, rows = exported
    monkeypatch.setattr(data, "MANIFEST_SHA256", data.digest(root / "manifest.json"))
    monkeypatch.setattr(data, "validate_manifest", lambda value: None)
    write_fields(merged, rows[:1])
    out = tmp_path / "prepared"
    with pytest.raises(ValueError, match="cover"):
        data.prepare(root, out, merged_dir=merged)
    assert not out.exists()


@pytest.mark.parametrize("problem", [None, "pixels", "labels", "raw_hashes", "incomplete"])
def test_assets_recovery_checks_and_preserves_prepared_data(exported, tmp_path, monkeypatch, capsys, problem):
    from benchmarks.full_CPJUMP1.harness import cli
    from benchmarks.full_HCMA.harness import assets

    root, merged, _, _ = exported
    monkeypatch.setattr(data, "MANIFEST_SHA256", data.digest(root / "manifest.json"))
    monkeypatch.setattr(data, "validate_manifest", lambda value: None)
    prep = tmp_path / "prepared"
    data.prepare(root, prep, merged_dir=merged)
    monkeypatch.setattr(cli, "prepare", lambda *a, **kw: pytest.fail("must not prepare images again"))
    if problem == "pixels":
        next((prep / "public/images").glob("*.tif")).write_bytes(b"changed")
    elif problem in ("labels", "raw_hashes"):
        (prep / "private" / f"{problem}.json").write_text("[]")
    elif problem == "incomplete":
        record = data.read_json(prep / "PREPARE.json")
        record["status"] = "incomplete"
        data.write_json(prep / "PREPARE.json", record)
    before = {p: data.digest(p) for p in prep.rglob("*") if p.is_file()}
    calls = []
    monkeypatch.setattr(assets, "build", lambda *a, **kw: calls.append((a, kw)))
    args = ["assets", "--prep", str(prep), "--root", str(tmp_path / "models"),
            "--dinov3-repo", str(tmp_path / "dinov3"), "--weights", str(tmp_path / "weights")]
    if problem is None:
        cli.main(args)
        assert len(calls) == 1
        assert "PREPARED_REUSED views=2" in capsys.readouterr().out
    else:
        with pytest.raises(ValueError):
            cli.main(args)
        assert not calls
        assert "PREPARED_REUSED" not in capsys.readouterr().out
    assert before == {p: data.digest(p) for p in prep.rglob("*") if p.is_file()}
