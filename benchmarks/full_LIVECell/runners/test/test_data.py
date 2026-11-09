import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest
import tifffile

from benchmarks.full_CPJUMP1.harness.data import development
from benchmarks.full_LIVECell.harness import data
from benchmarks.full_LIVECell.harness.data import (
    PUBLIC_FIELDS, TYPES, decode_rle, digest, load_prepared, prepare, rasterize, read_json, select,
    smoke_images, smoke_rows, validate_manifest, write_json)

SHAPE = (40, 56)
REAL = Path(os.environ.get("LIVECELL_ANNOTATIONS", "C:/Users/Wu/AppData/Local/Temp/claude/d--1-Data-Code-Worktree/"
                           "bf97dc5d-94e9-432c-94b6-9226459b6fc4/livecell_meta"))


def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def square(j: int) -> list[float]:
    x0, y0 = 2 + 10 * j, 2
    return [x0, y0, x0 + 6, y0, x0 + 6, y0 + 6, x0, y0 + 6]


def pixels(name: str, shape=SHAPE) -> np.ndarray:
    return np.random.default_rng(int(h(name)[:8], 16)).integers(0, 256, shape, dtype=np.uint8)


def catalog():
    """Synthetic LIVECell layout: per type 2-3 wells, 6-8 time points, both locations, crops 1-4."""
    images = []         # (kind, file_name, split, n_instances)
    for t, (kind, name) in enumerate(TYPES.items()):
        wells = ["B2", "B10", "B3"] if name == "A172" else ["B2", "B3"]
        for k in range(6 + t % 3):
            time = f"{(4 * k) // 24:02d}d{(4 * k) % 24:02d}h00m"
            for wi, well in enumerate(wells):
                split = "test" if well == "B3" else ("train" if k % 2 == 0 or wi == 1 else "val")
                for location in (1, 2):
                    crops = (1, 2, 3) if (name, well, k, location) == ("BV2", "B2", 1, 2) else (1, 2, 3, 4)
                    for crop in crops:
                        file_name = f"{name}_Phase_{well}_{location}_{time}_{crop}.tif"
                        n = 1 + int(h(file_name)[:4], 16) % 3
                        images.append((kind, file_name, split, n))
                        if name == "Huh7" and well == "B2" and k == 0:
                            images.append((kind, file_name, "val", n + 1))     # train/val duplicate
                        if name == "SKOV3" and well == "B3" and k == 0:
                            images.append((kind, file_name, "test", n + 2))    # within-file duplicate
    return images


def build_raw(raw: Path, *, bad_shape: str | None = None) -> None:
    files, members, next_id = {}, {}, [1000]
    by_file = {}
    for kind, file_name, split, n in catalog():
        by_file.setdefault((kind, split), []).append((file_name, n))
        members[f"{data.IMAGE_DIRS[split]}/{file_name}"] = file_name
    for kind in TYPES:
        for split in data.SPLITS:
            coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "cell"}]}
            for file_name, n in by_file.get((kind, split), []):
                image_id = next_id[0] = next_id[0] + 1
                coco["images"].append({"id": image_id, "file_name": file_name, "height": SHAPE[0], "width": SHAPE[1]})
                for j in reversed(range(n)):    # annotation ids out of file order
                    next_id[0] += 1
                    coco["annotations"].append({"id": next_id[0], "image_id": image_id, "segmentation": [square(j)],
                                                "iscrowd": 0, "category_id": 1})
            rel = data.annotation_path(kind, split)
            (raw / rel).parent.mkdir(parents=True, exist_ok=True)
            (raw / rel).write_text(json.dumps(coco), encoding="utf-8")
            files[rel] = {"bytes": (raw / rel).stat().st_size, "sha256": digest(raw / rel)}
    with zipfile.ZipFile(raw / "images.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for member, file_name in sorted(members.items()):
            buffer = io.BytesIO()
            tifffile.imwrite(buffer, pixels(file_name, (30, 30) if file_name == bad_shape else SHAPE))
            z.writestr(member, buffer.getvalue())
    files["images.zip"] = {"bytes": (raw / "images.zip").stat().st_size, "sha256": digest(raw / "images.zip")}
    write_json(raw / "SOURCE.json", {"files": files})


def n_units():
    return len({tuple(f.split("_")[i] for i in (0, 2, 4)) for _, f, _, _ in catalog()})


def patch(mp, values):
    for key, value in values.items():
        mp.setattr(data, key, value)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("livecell")
    raw = root / "raw"
    build_raw(raw)
    values = {"N_UNITS": n_units(), "IMAGE_SHAPE": SHAPE}
    with pytest.MonkeyPatch.context() as mp:
        patch(mp, values)
        select(raw, root / "manifest.json")
        values.update(MANIFEST_SHA256=digest(root / "manifest.json"), IMAGES_ZIP_BYTES=(raw / "images.zip").stat().st_size,
                      IMAGES_ZIP_SHA256=digest(raw / "images.zip"))
        patch(mp, values)
        prepare(raw, root / "prep")
    return {"root": root, "raw": raw, "manifest": root / "manifest.json", "prep": root / "prep", "values": values}


@pytest.fixture
def lc(built, monkeypatch):
    patch(monkeypatch, built["values"])
    return built


def tree_hashes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def labels_of(lc):
    return {r["image_id"]: r for r in read_json(lc["prep"] / "private" / "labels.json")}


def test_select_is_deterministic_and_follows_the_rules(lc, tmp_path):
    select(lc["raw"], tmp_path / "manifest.json")
    assert digest(tmp_path / "manifest.json") == digest(lc["manifest"])
    m = read_json(lc["manifest"])
    images = m["images"]
    assert len(images) == 2 * n_units()
    available = {}
    for _, f, _, _ in catalog():
        available.setdefault(f.rsplit("_", 1)[0], set()).add(f)
    for i in images:
        site = i["source_file"].rsplit("_", 1)[0]
        assert i["source_file"] == min(available[site], key=lambda f: h(f"20261013|{f}"))
    assert {i["crop"] for i in images if i["source_file"].startswith("BV2_Phase_B2_2_00d04h00m")} <= {1, 2, 3}
    for kind in TYPES.values():
        rows = [i for i in images if i["type"] == kind]
        hours = sorted({i["hours"] for i in rows})
        for i in rows:
            stage = 6 * hours.index(i["hours"]) // len(hours)
            assert i["stage"] == stage
            assert (i["usage"] == "heldout") == (stage >= 2)
            assert i["usage"] in ("train", "val", "heldout") and "half" not in i
        dev_wells = {i["well"] for i in rows if i["stage"] < 2}
        val = min(dev_wells, key=lambda w: h(f"20261013|val|{kind}|{w}"))
        assert {i["well"] for i in rows if i["usage"] == "val"} == {val}
        assert {i["well"] for i in rows if i["usage"] == "train"} == dev_wells - {val}
    a172 = sorted({(i["well"], i["well_index"]) for i in images if i["type"] == "A172"})
    assert a172 == [("B10", 2), ("B2", 0), ("B3", 1)]


def test_duplicates_resolve_train_val_test_and_record_counts(lc):
    images = {i["source_file"]: i for i in read_json(lc["manifest"])["images"]}
    huh7 = [i for f, i in images.items() if f.startswith("Huh7_Phase_B2_") and "_00d00h00m_" in f]
    assert len(huh7) == 2
    for i in huh7:
        first, second = i["annotation_sources"]
        assert (first["split"], second["split"]) == ("train", "val")
        assert second["n_instances"] == first["n_instances"] + 1
        assert i["official_split"] == "train" and i["annotation_source"].endswith("huh7/train.json")
        assert i["image_path"].startswith("images/livecell_train_val_images/")
    skov3 = [i for f, i in images.items() if f.startswith("SKOV3_Phase_B3_") and "_00d00h00m_" in f]
    for i in skov3:
        first, second = i["annotation_sources"]
        assert first["coco_image_id"] < second["coco_image_id"] and first["split"] == second["split"] == "test"
    labels = labels_of(lc)
    gt = {r["image_id"]: r for r in read_json(lc["prep"] / "private" / "gt_index.json")}
    for image, row in labels.items():
        if row["source_file"] in {i["source_file"] for i in huh7 + skov3}:
            assert gt[image]["n_instances"] == row["annotation_sources"][0]["n_instances"]


def test_validate_manifest_rejects_broken_designs(lc):
    m = read_json(lc["manifest"])
    for mutate in (
            lambda x: x["images"][0].update(usage="train", stage=3),
            lambda x: x["images"][0].update(usage="development_train"),
            lambda x: x["images"][0].update(half="early"),
            lambda x: x["images"][1].update(source_file=x["images"][0]["source_file"]),
            lambda x: x["images"].pop(),
            lambda x: [i.update(usage="train") for i in x["images"] if i["usage"] == "val"],
            lambda x: x["images"][0].update(stage=1)):
        bad = copy.deepcopy(m)
        mutate(bad)
        with pytest.raises(ValueError):
            validate_manifest(bad)


def test_select_fails_closed(lc, tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(lc["raw"], raw)
    rel = data.annotation_path("bv2", "train")
    (raw / rel).write_bytes((raw / rel).read_bytes() + b" ")
    with pytest.raises(ValueError, match="SOURCE.json"):
        select(raw, tmp_path / "m.json")
    shutil.copy(lc["raw"] / rel, raw / rel)
    coco = read_json(raw / rel)
    coco["images"] = [i for i in coco["images"] if "B2_2_00d00h00m" not in i["file_name"]]
    write_json(raw / rel, coco)
    source = read_json(raw / "SOURCE.json")
    source["files"][rel] = {"bytes": (raw / rel).stat().st_size, "sha256": digest(raw / rel)}
    write_json(raw / "SOURCE.json", source)
    with pytest.raises(ValueError, match="both locations"):
        select(raw, tmp_path / "m.json")
    with pytest.raises(FileExistsError):
        select(lc["raw"], lc["manifest"])


def test_prepare_is_deterministic_and_zip_equals_extracted_tree(lc, tmp_path):
    prepare(lc["raw"], tmp_path / "again", lc["manifest"])
    assert tree_hashes(tmp_path / "again") == tree_hashes(lc["prep"])
    raw = tmp_path / "raw"
    shutil.copytree(lc["raw"], raw)
    with zipfile.ZipFile(raw / "images.zip") as z:
        z.extractall(raw)
    (raw / "images.zip").unlink()
    prepare(raw, tmp_path / "tree", lc["manifest"])
    a, b = tree_hashes(tmp_path / "tree"), tree_hashes(lc["prep"])
    differ = {k for k in a if a[k] != b[k]}
    assert a.keys() == b.keys() and differ == {"PREPARE.json", "private/raw_hashes.json"}
    assert read_json(tmp_path / "tree" / "PREPARE.json")["images_source"] == {
        "mode": "extracted_tree", "zip_pin": "not_applicable_extracted_tree"}
    record = read_json(lc["prep"] / "PREPARE.json")
    assert record["images_source"]["zip_pin"] == "checked"
    assert record["versions"] == {"tifffile": tifffile.__version__, "numpy": np.__version__}


def test_public_table_is_anonymous(lc):
    rows = load_prepared(lc["prep"], verify_pixels=True)
    labels = labels_of(lc)
    assert all(tuple(r) == PUBLIC_FIELDS for r in rows)
    assert all("half" not in l and "half" not in PUBLIC_FIELDS for l in labels.values())      # no fit/test field
    assert [r["image_id"] for r in rows] == [f"img_{k:04d}" for k in range(len(rows))]
    secrets = set(TYPES.values()) | {"B2", "B3", "B10", "train", "val", "test"}
    secrets |= {l["source_file"] for l in labels.values()} | {l["source_file"].split("_")[4] for l in labels.values()}
    for r in rows:
        assert r["file"] == f"images/{r['image_id']}.ome.tif" and r["segment_file"] == f"segmentation/{r['image_id']}.ome.tif"
        for key in ("plate", "well", "site", "calibration_group", "file", "segment_file"):
            assert not any(s in r[key] for s in secrets), (key, r[key])
        assert r["negative_control"] is False
        assert r["plate"] == f"stage_{labels[r['image_id']]['stage']}"
        assert r["shape"] == [1, *SHAPE] and r["dtype"] == "uint8"
    order = {t: k for k, t in enumerate(sorted(TYPES.values()))}

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return abs(np.corrcoef(ra, rb)[0, 1])

    index = np.arange(len(rows))
    kinds = [order[labels[r["image_id"]]["cell_type"]] for r in rows]
    hours = [labels[r["image_id"]]["hours"] for r in rows]
    assert spearman(index, kinds) < 0.3 and spearman(index, hours) < 0.3
    unit_number = [int(r["well"].split("_")[1]) for r in rows]
    assert spearman(unit_number, kinds) < 0.3 and spearman(unit_number, hours) < 0.3
    wells = sorted({(l["cell_type"], l["well"]) for l in labels.values()})
    real = [wells.index((labels[r["image_id"]]["cell_type"], labels[r["image_id"]]["well"])) for r in rows]
    assert spearman(index, real) < 0.3 and spearman(unit_number, real) < 0.3
    units = {}
    for r in rows:
        units.setdefault(r["well"], []).append(r)
    assert set(map(len, units.values())) == {2}
    assert all(len({(labels[x["image_id"]]["cell_type"], labels[x["image_id"]]["hours"]) for x in u}) == 1 for u in units.values())
    groups = {}
    for r in rows:
        assert (r["calibration_group"] != "") == (r["usage"] in ("train", "val"))
        if r["calibration_group"]:
            groups.setdefault(r["calibration_group"], set()).add(labels[r["image_id"]]["cell_type"])
    assert sorted(groups) == [f"reference_{k:03d}" for k in range(8)] and all(len(v) == 1 for v in groups.values())
    assert [next(iter(groups[g])) for g in sorted(groups)] != sorted(TYPES.values())
    with tifffile.TiffFile(lc["prep"] / "public" / rows[0]["file"]) as f:
        assert not any(s in f.ome_metadata for s in secrets)


def test_views_round_trip_and_load_in_shape(lc):
    from shape.utils.image_data import ImageData
    rows = load_prepared(lc["prep"])
    labels = labels_of(lc)
    record = read_json(lc["prep"] / "PREPARE.json")
    assert (record["dataset"], record["channels"], record["image_shape"], record["dtype"]) == ("LIVECell", ["Phase"], list(SHAPE), "uint8")
    for r in rows[:3]:
        source = pixels(labels[r["image_id"]]["source_file"])
        full = tifffile.imread(lc["prep"] / "public" / r["file"])
        seg = tifffile.imread(lc["prep"] / "public" / r["segment_file"])
        assert full.shape == SHAPE and full.dtype == np.uint8     # tifffile squeezes C = 1 on read
        np.testing.assert_array_equal(full, source)
        np.testing.assert_array_equal(seg, source)
        with tifffile.TiffFile(lc["prep"] / "public" / r["file"]) as f:
            assert 'SizeC="1"' in f.ome_metadata and 'Name="Phase"' in f.ome_metadata
            assert f.series[0].get_shape(False)[2:5] == (1, *SHAPE)
        assert ImageData.from_path(str(lc["prep"] / "public" / r["file"])).data.shape == (*SHAPE, 1)
        assert ImageData.from_path(str(lc["prep"] / "public" / r["segment_file"])).data.shape == (*SHAPE, 1)


def test_rasterize_known_polygons_with_overlap():
    first = {"id": 1, "segmentation": [[2, 2, 6, 2, 6, 6, 2, 6]]}            # pixels rows/cols 2..5
    later = {"id": 5, "segmentation": [[4, 4, 9, 4, 9, 8, 4, 8]]}            # rows 4..7, cols 4..8
    labels, overlap = rasterize([later, first], (12, 12))
    assert labels.dtype == np.uint32
    assert (labels == 1).sum() == 16 and labels[2:6, 2:6].min() == 1
    assert overlap == 4 and (labels == 2).sum() == 20 - 4
    assert labels[4, 4] == 1 and labels[7, 8] == 2 and labels[0, 0] == 0
    mask = np.zeros((5, 4), dtype=bool)
    mask[1:4, 1:3] = True
    flat = mask.flatten(order="F")
    counts, value, run = [], False, 0
    for v in flat:
        if v != value:
            counts.append(run)
            value, run = v, 0
        run += 1
    counts.append(run)
    np.testing.assert_array_equal(decode_rle({"size": [5, 4], "counts": counts}), mask)
    np.testing.assert_array_equal(decode_rle({"size": [5, 4], "counts": rle_string(counts)}), mask)
    rle_labels, _ = rasterize([{"id": 3, "segmentation": {"size": [5, 4], "counts": rle_string(counts)}}], (5, 4))
    np.testing.assert_array_equal(rle_labels > 0, mask)
    with pytest.raises(ValueError, match="polygon"):
        rasterize([{"id": 1, "segmentation": [[1, 1, 2, 2]]}], (5, 5))


def rle_string(counts):
    """pycocotools rleToString, as the reference for the decoder."""
    out = []
    for i, x in enumerate(counts):
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if c & 0x10 else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def test_prepared_gt_matches_annotations(lc):
    labels = labels_of(lc)
    for row in read_json(lc["prep"] / "private" / "gt_index.json")[:5]:
        with np.load(lc["prep"] / "private" / row["file"]) as z:
            gt, n = z["labels"], int(z["n_instances"])
            assert gt.dtype == np.uint32 and gt.shape == SHAPE and int(z["overlap_pixels"]) == 0
            assert list(z["annotation_ids"]) == sorted(z["annotation_ids"])
        assert n == labels[row["image_id"]]["annotation_sources"][0]["n_instances"] == row["n_instances"]
        # Squares were written in reverse id order: the lowest id is square n-1.
        for k in range(n):
            j = n - 1 - k
            assert (gt[2:8, 2 + 10 * j:8 + 10 * j] == k + 1).all()
        assert set(np.unique(gt)) == set(range(n + 1))


def test_runtime_needs_no_private_files_and_smoke_composition(lc, tmp_path):
    prep = tmp_path / "prep"
    shutil.copytree(lc["prep"], prep)
    shutil.rmtree(prep / "private")
    rows = load_prepared(prep, verify_pixels=True)
    smoke = smoke_rows(prep, rows)
    labels = labels_of(lc)
    assert smoke == smoke_images(load_prepared(lc["prep"]), list(labels.values()))
    units = {}
    for r in smoke:
        units.setdefault(r["well"], []).append(r)
    assert len(smoke) == 24 and len(units) == 12 and set(map(len, units.values())) == {2}
    types = {labels[r["image_id"]]["cell_type"] for r in smoke}
    assert len(types) == 4
    for t in types:
        kinds = {u: rs[0]["usage"] for u, rs in units.items() if labels[rs[0]["image_id"]]["cell_type"] == t}
        assert sorted(kinds.values()) == ["heldout", "train", "val"]
    stage_of = {r["well"]: (r["plate"], r["usage"]) for r in smoke}
    per_stage = {}
    for unit, (stage, usage) in stage_of.items():
        per_stage.setdefault(stage, []).append(usage)
    assert {s: sorted(u) for s, u in per_stage.items()} == {
        "stage_0": ["train"] * 4, "stage_1": ["val"] * 4,
        "stage_2": ["heldout"] * 2, "stage_4": ["heldout"] * 2}      # >= 2 units per batch
    references = {}
    for r in smoke:
        if r["calibration_group"]:
            references.setdefault(r["calibration_group"], set()).add(r["plate"])
    assert len(references) == 4 and all(stages == {"stage_0", "stage_1"} for stages in references.values())
    paths, meta = development([{"cell_id": r["image_id"], "image_id": r["image_id"], "crop_path": r["file"],
                                 "well": r["well"], "plate": r["plate"], "usage": r["usage"], "area": 1}
                                for r in smoke], data.USAGES)
    assert {m["split"] for m in meta} == {"train", "val"} and len(paths) == 16      # heldout never enters


def test_prepare_fails_closed(lc, tmp_path, monkeypatch):
    with pytest.raises(FileExistsError):
        prepare(lc["raw"], lc["prep"], lc["manifest"])
    bad = tmp_path / "manifest.json"
    bad.write_bytes(lc["manifest"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="pinned"):
        prepare(lc["raw"], tmp_path / "a", bad)
    raw = tmp_path / "raw"
    shutil.copytree(lc["raw"], raw)
    rel = data.annotation_path("mcf7", "val")
    (raw / rel).write_bytes((raw / rel).read_bytes().replace(b'"iscrowd": 0', b'"iscrowd": 1', 1))
    with pytest.raises(ValueError, match="annotation file differs"):
        prepare(raw, tmp_path / "b", lc["manifest"])
    shutil.copy(lc["raw"] / rel, raw / rel)
    source = read_json(raw / "SOURCE.json")
    source["files"]["images.zip"]["sha256"] = "0" * 64
    write_json(raw / "SOURCE.json", source)
    with pytest.raises(ValueError, match="images.zip"):
        prepare(raw, tmp_path / "c", lc["manifest"])
    write_json(raw / "SOURCE.json", read_json(lc["raw"] / "SOURCE.json"))
    monkeypatch.setattr(data, "IMAGES_ZIP_SHA256", "1" * 64)     # SOURCE.json agrees, the published pin does not
    with pytest.raises(ValueError, match="pinned SHA256"):
        prepare(raw, tmp_path / "c", lc["manifest"])
    m = read_json(lc["manifest"])
    victim = m["images"][0]
    shutil.rmtree(raw)
    build_raw(raw, bad_shape=victim["source_file"])
    monkeypatch.setattr(data, "IMAGES_ZIP_BYTES", (raw / "images.zip").stat().st_size)
    monkeypatch.setattr(data, "IMAGES_ZIP_SHA256", digest(raw / "images.zip"))
    with pytest.raises(ValueError, match="dimensions"):
        prepare(raw, tmp_path / "d", lc["manifest"])
    with zipfile.ZipFile(raw / "images.zip") as z:
        z.extractall(raw)
    (raw / "images.zip").unlink()
    (raw / victim["image_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="download incomplete"):
        prepare(raw, tmp_path / "e", lc["manifest"])
    assert not any((tmp_path / x).exists() for x in "abce")
    assert not (tmp_path / "d" / "PREPARE.json").exists()


def test_load_prepared_rejects_tampering(lc, tmp_path):
    prep = tmp_path / "prep"
    shutil.copytree(lc["prep"], prep)
    rows = load_prepared(prep, verify_pixels=True)
    target = prep / "public" / rows[0]["file"]
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF
    target.write_bytes(bytes(payload))
    load_prepared(prep)
    with pytest.raises(ValueError, match="pixels changed"):
        load_prepared(prep, verify_pixels=True)
    (prep / "public" / "smoke.json").write_text('{"units": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="smoke"):
        smoke_rows(prep, rows)
    table = read_json(prep / "public" / "images.json")
    table[0]["usage"] = "train" if table[0]["usage"] != "train" else "val"
    write_json(prep / "public" / "images.json", table)
    with pytest.raises(ValueError, match="metadata changed"):
        load_prepared(prep)
    record = read_json(prep / "PREPARE.json")
    record.update(manifest_sha256="0" * 64, images_sha256=digest(prep / "public" / "images.json"))
    write_json(prep / "PREPARE.json", record)
    with pytest.raises(ValueError, match="wrong subset"):
        load_prepared(prep)


@pytest.mark.skipif(not REAL.is_dir(), reason="real LIVECell annotation files not available")
def test_real_annotations_select_the_pinned_manifest(tmp_path):
    raw = tmp_path / "raw"
    data.stage_flat_annotations(REAL, raw)
    m = select(raw, tmp_path / "manifest.json")
    assert digest(tmp_path / "manifest.json") == data.MANIFEST_SHA256
    units = {}
    for i in m["images"]:
        units.setdefault(i["unit"], []).append(i)
    assert len(units) == 655 and len(m["images"]) == 1310 and set(map(len, units.values())) == {2}
    per_type = {}
    for sites in units.values():
        per_type[sites[0]["type"]] = per_type.get(sites[0]["type"], 0) + 1
    assert per_type == {"A172": 76, "BT474": 84, "BV2": 76, "Huh7": 75, "MCF7": 92, "SHSY5Y": 88, "SkBr3": 88, "SKOV3": 76}
    usage_units, usage_images = {}, {}
    for sites in units.values():
        usage_units[sites[0]["usage"]] = usage_units.get(sites[0]["usage"], 0) + 1
    for i in m["images"]:
        usage_images[i["usage"]] = usage_images.get(i["usage"], 0) + 1
    assert usage_units == {"train": 174, "val": 61, "heldout": 420}
    assert usage_images == {"train": 348, "val": 122, "heldout": 840}
    pools = {"fit": {}, "test": {}}        # the scorer's pools of the heldout units, from the stage alone
    for sites in units.values():
        if sites[0]["usage"] == "heldout":
            pool = pools["fit" if sites[0]["stage"] in (2, 3) else "test"]
            pool[sites[0]["type"]] = pool.get(sites[0]["type"], 0) + 1
    assert sum(pools["fit"].values()) == 212 and sum(pools["test"].values()) == 208
    assert pools["fit"] == {"A172": 24, "BT474": 28, "BV2": 24, "Huh7": 24, "MCF7": 32, "SHSY5Y": 28, "SKOV3": 24, "SkBr3": 28}
    assert pools["test"] == {"A172": 24, "BT474": 28, "BV2": 24, "Huh7": 24, "MCF7": 28, "SHSY5Y": 28, "SKOV3": 24, "SkBr3": 28}
    # the smoke prepare writes: its stages, from the public unit ids and stages of the pinned selection
    public = data.public_units(m["images"])
    rows = [{"image_id": i["source_file"], "well": public[i["unit"]], "plate": f"stage_{i['stage']}",
             "usage": i["usage"]} for i in m["images"]]
    smoke = smoke_images(rows, [{"image_id": i["source_file"], "cell_type": i["type"]} for i in m["images"]])
    stages = {}
    for r in smoke:
        stages.setdefault(r["plate"], set()).add(r["well"])
    assert {s: len(u) for s, u in stages.items()} == {"stage_0": 4, "stage_1": 4, "stage_2": 2, "stage_4": 2}
