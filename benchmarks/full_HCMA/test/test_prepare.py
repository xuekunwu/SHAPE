"""prepare (anonymization, lossless metadata stripping, private labels and folds) and assets
(pinned DINOv3 hub dir, HF-layout weights resolvable offline, cpsam) on a synthetic mini-HCMA."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from benchmarks.cell_state_HCMA.hcma_labels import normalize_cell_state
from benchmarks.full_HCMA.harness import assets, prepare, subset
from benchmarks.full_HCMA.harness.cli import main

SECRET = b"ATCC__CRL-1573__HEK-293"          # an original-name fragment that must not survive


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def _jpeg(rng, mode="RGB", **options) -> bytes:
    shape = (24, 32, 3) if mode == "RGB" else (24, 32)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, size=shape, dtype=np.uint8), mode).save(buffer, "JPEG", quality=90,
                                                                                **options)
    return buffer.getvalue()


def _with_metadata(clean: bytes) -> bytes:
    """Insert EXIF (APP1), Photoshop (APP13) and COM segments after APP0, and junk after EOI."""
    assert clean[2:4] == b"\xff\xe0"
    app0_end = 4 + int.from_bytes(clean[4:6], "big")
    extra = (_segment(0xE1, b"Exif\x00\x00" + SECRET * 3) + _segment(0xED, b"Photoshop 3.0\x00" + SECRET)
             + _segment(0xFE, b"comment: " + SECRET))
    return clean[:app0_end] + extra + clean[app0_end:] + b"TRAILER" + SECRET


ROWS = [  # parent, source, cell_type, organ, modality, magnification, ext
    ("P1", "ATCC", "epithelial", "Kidney", "Bright-field", "", ".jpg"),
    ("P1", "ATCC", "epithelial", "Kidney", "Bright-field", "", ".jpg"),
    ("P2", "ATCC", "Epithelial Cells", "Lung", "Bright-field", "", ".JPG"),
    ("P3", "ScienCell", "fibroblast", "Skin", "Phase contrast", "100x", ".jpg"),
    ("P4", "ScienCell", "Fibroblast-like cells", "Skin", "Phase contrast", "200x", ".png"),
    ("P5", "ScienCell", "fibroblast", "Heart", "Phase contrast", "100x", ".tif"),
    ("P6", "LIVECell", "glioblastoma cell", "Nervous System", "Phase contrast", "", ".jpg"),
    ("P7", "Inhouse_Cell_Seg", "", "Blood", "Phase contrast", "", ".jpg"),
    ("P8", "ATCC", "endothelial", "Vascular", "Relief contrast", "", ".jpg"),
    ("P9", "ATCC", "endothelial", "Vascular", "Relief contrast", "", ".jpg"),
    ("P10", "ATCC", "naïve keratinocyte", "Skin µ", "Bright-field", "", ".jpg"),
    ("P11", "ScienCell", "epithelial", "Liver", "Bright-field", "", ".jpg"),     # epithelial spans 2 sources
    ("P12", "ATCC", "epithelial", "Liver", "Bright-field", "", ".jpg"),
    ("P12", "ATCC", "epithelial", "Liver", "Bright-field", "", ".jpg"),
]                     # StratifiedGroupKFold(5) needs one class with >= 5 images


KINDS = {".jpg": "rgb_jpg", ".JPG": "rgb_jpg", ".png": "rgb_png", ".tif": "rgb_tif"}
MIXED = {1: "gray_jpg", 4: "rgba_png", 5: "gray16_tif"}     # row -> image kind (the rest: KINDS)


def _write_image(path: Path, kind: str, rng) -> None:
    import tifffile

    if kind in ("rgb_jpg", "gray_jpg"):
        path.write_bytes(_with_metadata(_jpeg(rng, "L" if kind == "gray_jpg" else "RGB")))
    elif kind.endswith("_png"):
        channels = {"rgb_png": 3, "rgba_png": 4, "rgba_varying_png": 4, "la_png": 2}[kind]
        array = rng.integers(0, 255, size=(20, 30, channels), dtype=np.uint8)
        if kind == "rgba_png":
            array[..., 3] = 255                                   # opaque alpha
        info = PngInfo()
        info.add_text("Source", SECRET.decode())
        Image.fromarray(array, {2: "LA", 3: "RGB", 4: "RGBA"}[channels]).save(path, pnginfo=info)
    elif kind == "rgb_tif":
        tifffile.imwrite(path, rng.integers(0, 255, size=(18, 22, 3), dtype=np.uint8), photometric="rgb",
                         description=SECRET.decode(), software="VendorScope 2.1")
    else:                                                         # gray16_tif
        tifffile.imwrite(path, rng.integers(0, 4000, size=(18, 22), dtype=np.uint16),
                         description=SECRET.decode(), software="VendorScope 2.1")


def _make_hcma(tmp_path: Path, kinds: dict[int, str] | None = None) -> Path:
    rng = np.random.default_rng(1)
    root = tmp_path / "Human_Cell_Morphology_Atlas_HCMA"
    (root / "images").mkdir(parents=True)
    records = []
    for i, (parent, source, cell_type, organ, modality, mag, ext) in enumerate(ROWS):
        name = f"{source}__CRL-{1000 + i}__cell line {i}__panel01{ext}"
        _write_image(root / "images" / name, (kinds or {}).get(i, KINDS[ext]), rng)
        records.append({"image_id": f"HCMA-{source}-CRL-{1000 + i}-{i:04d}", "image_filename": name,
                        "relative_path": f"images/{name}", "parent_image_id": f"HCMA-{source}-{parent}",
                        "source": source, "source_catalog_id": f"CRL-{1000 + i}", "cell_name": f"cell line {i}",
                        "cell_type": cell_type, "organ_category": organ, "image_modality": modality,
                        "magnification": mag, "disease": "none"})
    pd.DataFrame(records).to_csv(root / "HCMA_metadata.csv", index=False, encoding="latin-1")
    return root


def _pixels(path: Path) -> np.ndarray:
    import tifffile

    return tifffile.imread(path) if path.suffix.lower() == ".tif" else np.asarray(Image.open(path))


@pytest.fixture()
def hcma(tmp_path) -> Path:
    """Every image 3-channel RGB (JPEG, PNG, TIFF): no channel conversion."""
    return _make_hcma(tmp_path)


def test_strip_jpeg_removes_metadata_and_keeps_the_scan_bytes():
    for clean in (_jpeg(np.random.default_rng(0)), _jpeg(np.random.default_rng(0), progressive=True)):
        stripped, info = prepare.strip_jpeg(_with_metadata(clean))
        assert stripped == clean                  # PIL's own JPEG: only SOI, APP0 and image segments
        assert info["removed"] == {"APP1": 1, "APP13": 1, "COM": 1} and info["trailing_bytes_dropped"] > 0
    adobe = _segment(0xEE, b"Adobe\x00\x64\x00\x00\x00\x00\x01")
    kept, info = prepare.strip_jpeg(clean[:2] + adobe + clean[2:])
    assert adobe in kept and info["removed"] == {}


def test_prepare_anonymizes_and_writes_private_labels(hcma, tmp_path):
    root = tmp_path / "full_HCMA"
    prep, private = prepare.prepare(hcma, root, seed=0)
    images = pd.read_csv(prep / "images.csv", dtype=str, keep_default_na=False)
    assert list(images.columns) == list(prepare.IMAGES_COLUMNS)
    assert list(images.image_id) == [f"img_{i:04d}" for i in range(1, len(ROWS) + 1)]
    assert set(images.parent_id) == {f"par_{i:04d}" for i in range(1, 13)}
    assert sorted(p.name for p in (prep / "images").iterdir()) == sorted(Path(f).name for f in images.file)
    assert set(images.file.map(lambda f: Path(f).suffix)) == {".jpg", ".png", ".tif"}

    meta = pd.read_csv(hcma / "HCMA_metadata.csv", encoding="latin-1", dtype=str, keep_default_na=False)
    needles = [SECRET, b"VendorScope", b"CRL-1", b"cell line", b"HCMA-"]
    needles += [v.encode() for v in meta.image_filename] + [v.encode() for v in meta.parent_image_id]
    for path in prep.rglob("*"):
        assert not path.is_symlink()
        if path.is_file():
            data = path.read_bytes()
            assert not [n for n in needles if n in data], path

    labels = pd.read_csv(private / "labels.csv", dtype=str, keep_default_na=False)
    assert list(labels.columns[:len(prepare.LABEL_COLUMNS)]) == list(prepare.LABEL_COLUMNS)
    assert {"relative_path", "source_catalog_id", "cell_name", "disease", "magnification"} <= set(labels.columns)
    merged = labels.merge(meta, left_on="orig_image_id", right_on="image_id", suffixes=("", "_raw"))
    assert len(merged) == len(ROWS)
    assert (merged.cell_state == merged.cell_type_raw.map(normalize_cell_state)).all()
    assert "naïve keratinocyte" in set(labels.cell_type)          # latin-1 read back intact
    for row in merged.itertuples():                       # the pixels match the source, nothing converted
        visible = images.set_index("image_id").loc[row.image_id]
        src, dst = hcma / row.relative_path, prep / visible.file
        assert np.array_equal(_pixels(src), _pixels(dst)) and dst.suffix == src.suffix.lower()
        if dst.suffix == ".jpg":
            assert dst.read_bytes() == prepare.strip_jpeg(src.read_bytes())[0]
        assert visible.source == row.source
    one = images.set_index("image_id").loc[labels.set_index("orig_image_id").loc[meta.image_id[1], "image_id"]]
    assert one.channels == "3" and one.height == "24" and one.width == "32"
    assert set(images.channels) == {"3"}

    folds = pd.read_csv(private / "folds.csv")
    assert sorted(folds.parent_id) == sorted(set(labels.parent_id)) and set(folds.fold) <= set(range(5))
    scorable = pd.read_csv(private / "scorable.csv")
    assert list(scorable.columns) == ["image_id", "cell_state", "parent_id", "fold"]
    # epithelial (P1, P2, P11, P12), fibroblast (P3, P4, P5), endothelial (P8, P9) span >= 2 parents
    assert set(scorable.cell_state) == {"epithelial", "fibroblast", "endothelial"} and len(scorable) == 11
    assert scorable.fold.notna().all()

    summary = json.loads((prep / "PREPARE.json").read_text())
    assert summary["n_images"] == len(ROWS) and summary["images_with_value"]["magnification"] == 3
    assert summary["channel_conversion"] == {"to_channels": None, "n_converted": 0, "from_channels": {}}
    text = json.dumps(summary)
    assert "epithelial" not in text and str(hcma) not in text
    private_summary = json.loads((private / "PREPARE_PRIVATE.json").read_text())
    for rel, digest in private_summary["prep_sha256"].items():
        assert prepare.sha256(prep / rel) == digest
    for name, digest in private_summary["private_sha256"].items():
        assert prepare.sha256(private / name) == digest
    assert private_summary["n_scorable_images"] == 11 and private_summary["n_scorable_classes"] == 3
    assert private_summary["stripped_segments"]["COM"] == 12        # every JPEG
    assert all(v["channels"] == 3 and v["delivered"] == v["format"] for v in private_summary["original_images"].values())

    for top in (prep, private):                                 # chmod -R a-w (directories keep x)
        for path in [top, *top.rglob("*")]:
            if path.is_file() or os.name == "posix":
                assert not path.stat().st_mode & 0o222, path
    with pytest.raises(PermissionError):
        open(prep / "images.csv", "a").close()

    again = tmp_path / "again"
    prepare.prepare(hcma, again, seed=0)
    assert (again / "private" / "labels.csv").read_bytes() == (private / "labels.csv").read_bytes()
    with pytest.raises(SystemExit, match="exists"):
        prepare.prepare(hcma, root, seed=0)


def test_prepare_fails_clearly(hcma, tmp_path):
    meta = pd.read_csv(hcma / "HCMA_metadata.csv", encoding="latin-1")
    meta.drop(columns=["magnification"]).to_csv(hcma / "HCMA_metadata.csv", index=False, encoding="latin-1")
    with pytest.raises(SystemExit, match="magnification"):
        prepare.prepare(hcma, tmp_path / "a")
    bad = hcma / "images" / "x.bmp"
    Image.new("RGB", (4, 4)).save(bad)
    meta.loc[0, "relative_path"] = "images/x.bmp"
    meta.to_csv(hcma / "HCMA_metadata.csv", index=False, encoding="latin-1")
    with pytest.raises(SystemExit, match="unsupported image extension"):
        prepare.prepare(hcma, tmp_path / "b")


def test_mixed_channel_counts_are_brought_to_three(tmp_path):
    hcma = _make_hcma(tmp_path, MIXED)
    prep, private = prepare.prepare(hcma, tmp_path / "full_HCMA", seed=0)
    images = pd.read_csv(prep / "images.csv", dtype=str, keep_default_na=False).set_index("image_id")
    labels = pd.read_csv(private / "labels.csv", dtype=str, keep_default_na=False).set_index("orig_image_id")
    meta = pd.read_csv(hcma / "HCMA_metadata.csv", encoding="latin-1", dtype=str, keep_default_na=False)
    original = json.loads((private / "PREPARE_PRIVATE.json").read_text())["original_images"]
    assert set(images.channels) == {"3"}
    assert json.loads((prep / "PREPARE.json").read_text())["channel_conversion"] == {
        "to_channels": 3, "n_converted": 3, "from_channels": {"1": 2, "4": 1}}
    for i, row in meta.iterrows():
        image_id = labels.loc[row.image_id, "image_id"]
        src, dst = hcma / row.relative_path, prep / images.loc[image_id, "file"]
        source, delivered, kind = _pixels(src), _pixels(dst), MIXED.get(i)
        assert delivered.dtype == source.dtype and delivered.shape == (*source.shape[:2], 3)
        if kind in ("gray_jpg", "gray16_tif"):                    # one channel replicated
            assert all(np.array_equal(delivered[..., c], source) for c in range(3))
        elif kind == "rgba_png":                                  # opaque alpha dropped
            assert np.array_equal(delivered, source[..., :3])
        else:                                                     # already RGB: untouched
            assert np.array_equal(delivered, source)
            if src.suffix.lower() == ".jpg":
                assert dst.read_bytes() == prepare.strip_jpeg(src.read_bytes())[0]
        expected = {"gray_jpg": ".png", "rgba_png": ".png", "gray16_tif": ".tif"}.get(kind, src.suffix.lower())
        assert dst.suffix == expected and images.loc[image_id, "file"] == f"images/{image_id}{expected}"
        assert original[image_id] == {"channels": {"gray_jpg": 1, "gray16_tif": 1, "rgba_png": 4}.get(kind, 3),
                                      "format": src.suffix.lower(), "delivered": expected}
        assert SECRET not in dst.read_bytes()
    gray16 = images.loc[labels.loc[meta.image_id[5], "image_id"]]
    assert _pixels(prep / gray16.file).dtype == np.uint16 and gray16.height == "18" and gray16.width == "22"


@pytest.mark.parametrize("kind, message", [("la_png", "mixed channel counts {2: 1, 3: 13}"),
                                           ("rgba_varying_png", "non-constant 4th channel")])
def test_channels_that_cannot_become_three_fail(tmp_path, kind, message):
    with pytest.raises(SystemExit, match=re.escape(message)):
        prepare.prepare(_make_hcma(tmp_path, {4: kind}), tmp_path / "full_HCMA", seed=0)


def test_pairs_subset(hcma, tmp_path, capsys):
    root = tmp_path / "full_HCMA"
    prep, private = prepare.prepare(hcma, root, seed=0)
    private_mode = private.stat().st_mode
    out = subset.subset(root, "pairs", per_class=2, seed=0)
    assert out == root / "prep_pairs" and private.stat().st_mode == private_mode
    labels = pd.read_csv(private / "labels.csv", dtype=str, keep_default_na=False).set_index("image_id")
    full = pd.read_csv(prep / "images.csv", dtype=str, keep_default_na=False).set_index("image_id")
    images = pd.read_csv(out / "images.csv", dtype=str, keep_default_na=False)
    assert list(images.columns) == list(prepare.IMAGES_COLUMNS) and list(images.image_id) == sorted(images.image_id)
    assert images.set_index("image_id").equals(full.loc[images.image_id])     # the full prep's rows, same ids
    picked = labels.loc[images.image_id]
    # epithelial (4 micrographs), fibroblast (3), endothelial (2); glioblastoma / keratinocyte have one
    assert sorted(picked.cell_state.value_counts().items()) == [("endothelial", 2), ("epithelial", 2), ("fibroblast", 2)]
    assert (picked.groupby("cell_state").parent_id.nunique() == 2).all()
    # epithelial has micrographs of 2 sources (P11 is ScienCell): its pair is cross-source
    n_sources = full.loc[images.image_id].assign(cell_state=picked.cell_state.to_numpy()).groupby("cell_state").source.nunique()
    assert n_sources.to_dict() == {"endothelial": 1, "epithelial": 2, "fibroblast": 1}
    for file in images.file:                                                   # real copies, read-only
        assert not (out / file).is_symlink() and (out / file).read_bytes() == (prep / file).read_bytes()
        assert not (out / file).stat().st_mode & 0o222
    summary = json.loads((out / "PREPARE.json").read_text())
    assert summary == {"n_images": 6, "n_parents": 6, "n_types": 3, "per_class": 2, "n_types_cross_source": 1,
                       "design": "2 images per cell type, each from a different micrograph",
                       "sources": images.source.value_counts().to_dict()}
    record = json.loads((private / "SUBSET_pairs.json").read_text())
    assert [r["image_id"] for r in record["selection"]] and set(r["image_id"] for r in record["selection"]) == set(images.image_id)
    assert all(labels.loc[r["image_id"], "cell_state"] == r["cell_state"] for r in record["selection"])
    assert record["rule"] == subset.SELECTION_RULE and record["n_types_cross_source"] == 1
    assert {t["cell_state"]: t["cross_source"] for t in record["types"]} == {
        "endothelial": False, "epithelial": True, "fibroblast": False}
    for path in (private / "SUBSET_pairs.json", out / "images.csv", out / "PREPARE.json"):   # read-only
        assert not path.stat().st_mode & 0o222, path
        with pytest.raises(PermissionError):
            open(path, "a").close()
    if os.name == "posix":
        assert not out.stat().st_mode & 0o222 and not (out / "images").stat().st_mode & 0o222
    assert record["prep_sha256"] == {p.relative_to(out).as_posix(): prepare.sha256(p)
                                     for p in out.rglob("*") if p.is_file()}
    assert subset.prep_problems(root, "pairs") == [] and subset.prep_problems(root, "full") == []
    assert subset.subset(root, "pairs") == out                                  # exists and verifies: kept
    main(["verify-prep", "--root", str(root), "--prep-name", "pairs"])
    assert "PREP pairs verifies" in capsys.readouterr().out
    # the launcher's NO_CLASS_MAP gate (`harness class-map`) on this prep
    states = sorted(set(picked.cell_state))
    class_map = tmp_path / "morph_classes.csv"
    with pytest.raises(SystemExit):                                              # no map at all
        main(["class-map", "--root", str(root), "--prep-name", "pairs", "--map", str(class_map)])
    pd.DataFrame({"cell_state": states[:-1], "morph_class": "class_a", "note": ""}).to_csv(class_map, index=False)
    with pytest.raises(SystemExit):                                              # a cell_state without a row
        main(["class-map", "--root", str(root), "--prep-name", "pairs", "--map", str(class_map)])
    assert "no morph_class row" in capsys.readouterr().out
    pd.DataFrame({"cell_state": states, "morph_class": "class_a", "note": ""}).to_csv(class_map, index=False)
    main(["class-map", "--root", str(root), "--prep-name", "pairs", "--map", str(class_map)])
    assert "CLASS_MAP ok for prep pairs" in capsys.readouterr().out
    for other in ({"per_class": 3}, {"seed": 1}):                              # a subset is made once
        with pytest.raises(SystemExit, match="other settings"):
            subset.subset(root, "pairs", **other)

    again = tmp_path / "again"                                                  # deterministic
    prepare.prepare(hcma, again, seed=0)
    subset.subset(again, "pairs", per_class=2, seed=0)
    assert (again / "prep_pairs" / "images.csv").read_bytes() == (out / "images.csv").read_bytes()

    tampered = tmp_path / "tampered"                                            # only from a verified full prep
    prepare.prepare(hcma, tampered, seed=0)
    first = tampered / "prep" / "images" / sorted(p.name for p in (tampered / "prep" / "images").iterdir())[0]
    first.chmod(0o644)
    first.write_bytes(b"changed")
    with pytest.raises(SystemExit, match="full prep does not verify"):
        subset.subset(tampered, "pairs")
    assert not (tampered / "prep_pairs").exists() and not (tampered / "private" / "SUBSET_pairs.json").exists()

    victim = out / images.file[0]                                               # a changed file is caught
    victim.chmod(0o644)
    victim.write_bytes(b"changed")
    assert subset.prep_problems(root, "pairs")
    with pytest.raises(SystemExit, match="does not verify"):
        subset.subset(root, "pairs")
    with pytest.raises(SystemExit):
        main(["verify-prep", "--root", str(root), "--prep-name", "pairs"])


def test_pairs_prefer_parents_of_different_sources():
    rows = [("a", "p1", "S1"), ("a", "p2", "S1"), ("a", "p3", "S1"), ("a", "p4", "S2"),
            ("b", "q1", "S1"), ("b", "q2", "S1"), ("c", "r1", "S1"), ("unknown", "u1", "S2"), ("unknown", "u2", "S1")]
    labels = pd.DataFrame([{"image_id": f"img_{parent}", "cell_state": state, "parent_id": parent, "source": source}
                           for state, parent, source in rows])
    for seed in range(20):
        chosen, types = subset.select(labels, 2, seed)
        by_type = chosen.groupby("cell_state")
        assert set(by_type.groups) == {"a", "b"}                                 # c: one parent; unknown: never
        assert by_type.source.nunique().to_dict() == {"a": 2, "b": 1}
        assert [t["cross_source"] for t in types] == [True, False]
        assert subset.select(labels, 2, seed)[0].equals(chosen)                  # seeded, deterministic
    chosen, _ = subset.select(labels, 3, 0)                                      # 2 sources, 3 parents
    assert sorted(chosen.loc[chosen.cell_state == "a", "source"]).count("S2") == 1


# ---------------------------------------------------------------- assets
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture()
def dinov3(tmp_path) -> tuple[Path, str]:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = tmp_path / "dinov3_pinned"
    (repo / "dinov3").mkdir(parents=True)
    (repo / "hubconf.py").write_text("def dinov3_vitb16(pretrained=False):\n    return 'model'\n")
    (repo / "dinov3" / "__init__.py").write_text("")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "pin")
    (repo / "untracked_notes.txt").write_text("local file")
    return repo, _git(repo, "rev-parse", "HEAD").strip()


def test_hf_weights_resolve_offline(tmp_path):
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"tiny fake weights")
    record = assets.place_hf_weights(weights, tmp_path / "hf", hashlib.sha256(b"tiny fake weights").hexdigest())
    assert record["revision"] == record["sha256"][:40]
    assert (tmp_path / "hf" / "hub" / "models--5xuekun--dinov3_vits16" / "refs" / "main").read_text() == record["revision"]
    resolved = assets.check_hf_offline(tmp_path / "hf", Path(record["path"]))
    assert Path(resolved).read_bytes() == b"tiny fake weights"
    with pytest.raises(SystemExit, match="sha256"):
        assets.place_hf_weights(weights, tmp_path / "hf2", "0" * 64)


def test_assets_build(tmp_path, dinov3):
    repo, commit = dinov3
    weights, cpsam = tmp_path / "w.pth", tmp_path / "cpsam_src"
    weights.write_bytes(b"weights")
    cpsam.write_bytes(b"cpsam weights")
    root = tmp_path / "full_HCMA"
    failed = root / "assets.partial"
    failed.mkdir(parents=True)
    (failed / "failure_evidence.txt").write_text("earlier failed attempt")
    manifest = json.loads(assets.build(root, dinov3_repo=repo, weights=weights, cpsam=cpsam, commit=commit,
                                       weights_sha256=None).read_text())
    hub = root / "assets" / "torch" / "hub" / assets.HUB_DIR_NAME
    assert (hub / "hubconf.py").is_file() and not (hub / ".git").exists()
    assert not (hub / "untracked_notes.txt").exists()
    assert manifest["dinov3_hub"]["commit"] == commit
    assert manifest["dinov3_hub"]["tree_sha256"] == assets.tree_digest(hub)[0]
    assert Path(manifest["dinov3_weights"]["path"]).read_bytes() == b"weights"
    assert manifest["cpsam"]["sha256"] == hashlib.sha256(b"cpsam weights").hexdigest()
    assert (root / "assets" / "cellpose" / "cpsam").read_bytes() == b"cpsam weights"
    assert manifest["env"]["HF_HUB_OFFLINE"] == "1" and manifest["env"]["TORCH_HOME"] == str(root / "assets" / "torch")
    assert manifest["env"]["SHAPE_DINOV3_REPO"] == str(hub)
    assert (failed / "failure_evidence.txt").read_text() == "earlier failed attempt"
    assert not list(root.glob("assets.partial.*"))
    with pytest.raises(SystemExit, match="exists"):
        assets.build(root, dinov3_repo=repo, weights=weights, cpsam=cpsam, commit=commit, weights_sha256=None)


def test_assets_refuse_a_wrong_or_dirty_checkout(tmp_path, dinov3):
    repo, commit = dinov3
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"weights")
    with pytest.raises(SystemExit, match="expected"):
        assets.build(tmp_path / "a", dinov3_repo=repo, weights=weights, cpsam=weights, commit="0" * 40,
                     weights_sha256=None)
    (repo / "hubconf.py").write_text("# edited\n")
    with pytest.raises(SystemExit, match="tracked changes"):
        assets.build(tmp_path / "b", dinov3_repo=repo, weights=weights, cpsam=weights, commit=commit,
                     weights_sha256=None)
