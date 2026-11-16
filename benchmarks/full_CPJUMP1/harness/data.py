"""Pinned raw-data assembly. Biological identities never enter the public table."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import time

import numpy as np
import pandas as pd
import tifffile

MANIFEST_SHA256 = "7cf7a28b12f703a152da4f2445e3a72aa62ed9078012e227e398ba4f92cadd31"  # v1 1000-view selection
# Every AGREED selection that prepare + load_prepared accept: the v1 subset and the two non-overlapping
# field-replicate selections (runners/generate_replica_manifests.py). PREPARE.json records the actual one.
KNOWN_MANIFEST_SHA256 = {
    MANIFEST_SHA256,
    "44be6b84bb5ecf51108a110436189a6470d5abc1775d4c5ad22620ab2409f9ce",  # replica seed20261009
    "9e85a049395cceb2e86c0e0d7e7c2fb6eb0215b8d17eb4627a0f183f98a184f1",  # replica seed20261010
}
CHANNELS = ("Mito", "AGP", "RNA", "ER", "DNA")
# (train, val, held-out) usage values of CPJUMP1; another dataset names its own in its Dataset spec (`usages`).
CPJUMP1_USAGES = ("development_train", "development_val", "evaluation_only")
USAGES = set(CPJUMP1_USAGES)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def json_value(value):
    """Keep structured diagnostics; an unavailable/non-finite readout is JSON null."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump())
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_value(value), indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    for attempt in range(5):     # Windows refuses to replace a file another process briefly holds open
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def within(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"unsafe relative path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes root: {relative!r}")
    return path


def validate_manifest(m: dict) -> None:
    sites, files, wells = m["sites"], m["files"], m["wells"]
    if (len(sites), len(files), len(wells)) != (1000, 5000, 500):
        raise ValueError("expected the fixed 1000-view / 5000-channel / 500-well subset")
    if len({s["id"] for s in sites}) != len(sites):
        raise ValueError("duplicate site identity")
    if len({f["relative_path"] for f in files}) != len(files):
        raise ValueError("duplicate channel path")
    channels = defaultdict(list)
    for f in files:
        within(Path.cwd(), f["relative_path"])
        channels[f["site_id"]].append(int(f["channel"]))
    if set(channels) != {s["id"] for s in sites} or any(sorted(c) != [1, 2, 3, 4, 5] for c in channels.values()):
        raise ValueError("each site must have exactly channels 1..5")
    by_well = {(w["plate"], w["well"]): w for w in wells}
    if len(by_well) != len(wells) or set(m["plates"]) != {w["plate"] for w in wells}:
        raise ValueError("duplicate well or inconsistent plate identity")
    counts = Counter()
    for site in sites:
        key = (site["plate"], site["well"])
        well = by_well[key]
        counts[key] += 1
        for field in ("usage", "gene", "role", "pert_type", "sample"):
            if site[field] != well[field]:
                raise ValueError(f"site/well {field} mismatch: {site['id']}")
        if site["usage"] not in USAGES:
            raise ValueError("unknown data split")
        if site["role"] == "evaluation" and site["usage"] != "evaluation_only":
            raise ValueError("evaluation gene leaks into development")
    if set(counts) != set(by_well) or set(counts.values()) != {2}:
        raise ValueError("expected two sites per well")
    treatment = [w for w in wells if w["pert_type"] == "trt"]
    dev = {w["gene"] for w in treatment if w["role"] == "calibration"}
    held = {w["gene"] for w in treatment if w["role"] == "evaluation"}
    if len(dev) != 20 or len(held) != 110 or dev & held:
        raise ValueError("expected disjoint 20 calibration / 110 evaluation genes")


def assemble_view(paths: list[Path], output: Path, segment: Path) -> dict:
    arrays = [tifffile.imread(p) for p in paths]
    if len(arrays) != 5 or any(a.ndim != 2 or a.dtype != np.uint16 for a in arrays):
        raise ValueError("expected five 2D uint16 channels")
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("channel dimensions differ")
    stack = np.stack(arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    segment.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output, stack, ome=True, photometric="minisblack",
                     metadata={"axes": "CYX", "Channel": {"Name": list(CHANNELS)}})
    # The segmenter otherwise truncates the five-channel input before reaching DNA.
    tifffile.imwrite(segment, stack[[4, 1]], ome=True, photometric="minisblack",
                     metadata={"axes": "CYX", "Channel": {"Name": ["DNA", "AGP"]}})
    if not np.array_equal(tifffile.imread(output), stack):
        raise ValueError("stack round-trip differs from source pixels")
    if not np.array_equal(tifffile.imread(segment), stack[[4, 1]]):
        raise ValueError("segmentation view round-trip differs")
    return {"shape": list(stack.shape), "dtype": "uint16", "sha256": digest(output),
            "segment_sha256": digest(segment)}


def prepare(data: Path, out: Path, *, merged_dir: Path | None = None) -> None:
    if out.exists():
        raise FileExistsError("prepared directory already exists; use a fresh output")
    manifest = data / "manifest.json"
    if digest(manifest) not in KNOWN_MANIFEST_SHA256:
        raise ValueError("manifest SHA256 is not an agreed CRISPR subset")
    m = read_json(manifest)
    validate_manifest(m)
    files = defaultdict(list)
    for item in m["files"]:
        files[item["site_id"]].append(item)
    # Check completeness before writing any assembled views.
    for f in m["files"]:
        if not within(data / "raw", f["relative_path"]).is_file():
            raise FileNotFoundError(f"download incomplete: {f['relative_path']}")
    merged, assembly = {}, {"mode": "assemble_raw"}
    if merged_dir is not None:
        from .merged import import_view, load_index
        merged, assembly = load_index(merged_dir, data, m)
    out.mkdir(parents=True, exist_ok=False)
    by_well = {(w["plate"], w["well"]): w for w in m["wells"]}
    plates = {p: f"plate_{i:02d}" for i, p in enumerate(sorted(m["plates"]))}
    well_ids = {k: f"well_{i:04d}" for i, k in enumerate(sorted(by_well))}
    guides = sorted({w["guide_sequence"] for w in m["wells"] if w["role"] == "calibration"})
    groups = {g: f"reference_{i:03d}" for i, g in enumerate(guides)}
    public, private, raw_hashes = [], [], []
    for i, site in enumerate(sorted(m["sites"], key=lambda s: s["id"])):
        image = f"img_{i:04d}"
        w = by_well[(site["plate"], site["well"])]
        source = sorted(files[site["id"]], key=lambda f: int(f["channel"]))
        paths = [within(data / "raw", f["relative_path"]) for f in source]
        image_file, segment_file = f"images/{image}.ome.tif", f"segmentation/{image}.ome.tif"
        if merged_dir is None:
            facts = assemble_view(paths, out / "public" / image_file, out / "public" / segment_file)
        else:
            facts = import_view(merged[site["id"]], paths, out / "public" / image_file, out / "public" / segment_file)
        if facts["shape"] != [5, 1080, 1080]:
            raise ValueError(f"unexpected raw dimensions: {site['id']} {facts['shape']}")
        for f, p in zip(source, paths):
            raw_hashes.append({"relative_path": f["relative_path"], "sha256": digest(p)})
        public.append({"image_id": image, "file": image_file, "segment_file": segment_file,
                       "plate": plates[site["plate"]], "well": well_ids[(site["plate"], site["well"])],
                       "site": image, "usage": site["usage"],
                       "calibration_group": groups.get(w["guide_sequence"], ""),
                       "negative_control": w["pert_type"] == "negcon", **facts})
        private.append({"image_id": image, "source_site": site["id"], "gene": w["gene"],
                        "guide": w["guide_sequence"], "role": w["role"], "pert_type": w["pert_type"]})
        if merged_dir is not None:
            private[-1]["merged_source"] = str(merged[site["id"]])
        if (i + 1) % 25 == 0:
            print(f"PREPARE {i + 1}/1000", flush=True)
    if merged_dir is not None and (digest(merged_dir / "fields.csv") != assembly["fields_sha256"] or
                                   digest(merged_dir / "COMPLETE") != assembly["complete_sha256"]):
        raise ValueError("merged index changed during preparation")
    write_json(out / "public" / "images.json", public)
    write_json(out / "private" / "labels.json", private)
    write_json(out / "private" / "raw_hashes.json", raw_hashes)
    write_json(out / "PREPARE.json", {"status": "complete", "manifest_sha256": digest(manifest),
               "images_sha256": digest(out / "public" / "images.json"),
               "labels_sha256": digest(out / "private" / "labels.json"),
               "raw_hashes_sha256": digest(out / "private" / "raw_hashes.json"),
               "channels": CHANNELS, "segmentation_channels": ["DNA", "AGP"], "limits": m["limits"],
               "assembly": assembly})


def load_prepared(prep: Path, *, verify_pixels: bool = False) -> list[dict]:
    record = read_json(prep / "PREPARE.json")
    if record["status"] != "complete" or record["manifest_sha256"] not in KNOWN_MANIFEST_SHA256:
        raise ValueError("preparation incomplete or wrong subset")
    table = prep / "public" / "images.json"
    if digest(table) != record["images_sha256"]:
        raise ValueError("prepared metadata changed")
    rows = read_json(table)
    if verify_pixels:
        for row in rows:
            for key, sha in (("file", "sha256"), ("segment_file", "segment_sha256")):
                if digest(within(prep / "public", row[key])) != row[sha]:
                    raise ValueError(f"prepared pixels changed: {row['image_id']} {key}")
    return rows


def development(cells: list[dict], usages: tuple = CPJUMP1_USAGES) -> tuple[list[str], list[dict]]:
    """The train and val cells; `usages` = the dataset's (train, val, held-out) values. Held-out cells never enter."""
    train, _, held = usages
    paths, meta = [], []
    for c in cells:
        if c["usage"] not in usages:
            raise ValueError(f"unknown usage: {c['usage']}")
        if c["usage"] == held:
            continue
        paths.append(c["crop_path"])
        meta.append(technical(c, split="train" if c["usage"] == train else "val"))
    if {m["split"] for m in meta} != {"train", "val"}:
        raise ValueError("development crops require both train and val wells")
    if {m["well"] for m in meta if m["split"] == "train"} & {m["well"] for m in meta if m["split"] == "val"}:
        raise ValueError("well leakage between train and val")
    return paths, meta


def technical(c: dict, **extra) -> dict:
    return {"cell_id": c["cell_id"], "crop_id": c["cell_id"], "image_name": c["image_id"],
            "plate": c["plate"], "well": c["well"], "site": c["image_id"], "batch": c["plate"],
            "technical_group": c["well"], "group": "default", "area": c["area"], **extra}


def well_profiles(matrix: np.ndarray, cells: list[dict], images: list[dict],
                  *, allowed_sites: tuple[int, ...] = (2,), allow_empty_sites: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Equal site weighting, not weighting a well by its number of segmented cells. `allowed_sites`: the
    well's site count must be one of these; CPJUMP1 images two sites per well (the default), a dataset with
    no multi-site pooling (ModalityProbe) passes (1,)."""
    x = np.asarray(matrix)
    if x.ndim != 2 or len(x) != len(cells) or not np.isfinite(x).all():
        raise ValueError("invalid cell embedding")
    table = pd.DataFrame(cells)
    if table.cell_id.duplicated().any():
        raise ValueError("duplicate cell identity")
    present = set(table.image_id)
    if not present <= {r["image_id"] for r in images} or (not allow_empty_sites and present != {r["image_id"] for r in images}):
        raise ValueError("every selected site must have retained cells; incomplete runs cannot be scored")
    site_rows, site_x = [], []
    for image in images:
        site_rows.append(image)
        selected = x[table.image_id.to_numpy() == image["image_id"]]
        site_x.append(np.median(selected, axis=0) if len(selected) else np.full(x.shape[1], np.nan))
    site_x = np.stack(site_x)
    sm = pd.DataFrame(site_rows)
    meta, profiles = [], []
    for well, group in sm.groupby("well", sort=True):
        if len(group) not in allowed_sites:
            raise ValueError(f"well {well} needs a site count in {allowed_sites}, found {len(group)}")
        available = np.isfinite(site_x[group.index]).all(axis=1)
        if not available.any():
            raise ValueError(f"well {well} has no cell embeddings in any view; cannot fabricate a profile")
        profiles.append(site_x[group.index][available].mean(axis=0))
        row = group.iloc[0]
        meta.append({k: row[k] for k in ("well", "plate", "usage", "calibration_group", "negative_control")})
        if allow_empty_sites:
            meta[-1].update(scheduled_sites=len(group), nonempty_sites=int(available.sum()),
                            no_detection_sites=[str(group.iloc[i]["image_id"]) for i in range(len(group)) if not available[i]])
        for key in ("experiment_date", "plate_id", "partition"):
            if key in group.columns:
                if group[key].nunique(dropna=False) != 1:
                    raise ValueError(f"well {well} has inconsistent {key}")
                meta[-1][key] = row[key]
    return np.asarray(profiles, dtype=np.float32), pd.DataFrame(meta)
