"""Pinned LIVECell selection and preparation. Cell type, real well, time and split never enter the public table."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
from pathlib import Path
import re
import shutil
import uuid
import zipfile

import numpy as np
import tifffile

from benchmarks.full_CPJUMP1.harness.data import (  # noqa: F401  (re-exported: two sites per unit fit as-is)
    digest, json_value, read_json, technical, well_profiles, within, write_json)

# The pin is the SHA256 of the manifest's LF bytes (`write_manifest`), the same on every platform. The first pin
# (4ede5c78...) came from a Windows run whose text mode wrote CRLF; it did not reproduce on the Linux VM.
MANIFEST_SHA256 = "1de21fe920d526368918629744e4aa1489509f6eca2403502856e1090171c513"
SEED = 20261013
USAGES = ("train", "val", "heldout")       # the (train, val, held-out) values of an image row's `usage`
TRAIN, VAL, HELDOUT = USAGES
DATASET = "LIVECell"
CHANNELS = ("Phase",)
IMAGE_SHAPE = (520, 704)
N_UNITS = 655
N_STAGES = 6
IMAGES_ZIP = "images.zip"
IMAGES_ZIP_BYTES = 1241715651
# SHA256 of the published images.zip (computed on the VM from the S3 object, 2026-10-13).
IMAGES_ZIP_SHA256 = "548a2c2853c422b1ccc6519c1fae470ac5945243cbbd50d08ed7fe81196383a6"
TYPES = {"a172": "A172", "bt474": "BT474", "bv2": "BV2", "huh7": "Huh7", "mcf7": "MCF7",
         "shsy5y": "SHSY5Y", "skbr3": "SkBr3", "skov3": "SKOV3"}
SPLITS = ("train", "val", "test")       # duplicate resolution order
# Measured from the images.zip central directory: the folders are flat, no per-type subfolder.
IMAGE_DIRS = {"train": "images/livecell_train_val_images", "val": "images/livecell_train_val_images",
              "test": "images/livecell_test_images"}
NAME = re.compile(r"^(?P<type>[A-Za-z0-9]+)_Phase_(?P<well>[A-Z]\d+)_(?P<location>[12])_"
                  r"(?P<time>(?P<dd>\d\d)d(?P<hh>\d\d)h(?P<mm>\d\d)m)_(?P<crop>[1-4])\.tif$")
MANIFEST_IMAGE_FIELDS = frozenset((      # no fit/test field: the scorer derives heldout_fit / heldout_test from `stage`
    "source_file", "image_path", "unit", "type", "well", "well_index", "location", "crop", "time", "hours", "stage",
    "usage", "official_split", "annotation_source", "annotation_sources"))
PUBLIC_FIELDS = ("image_id", "file", "segment_file", "plate", "well", "site", "usage", "calibration_group",
                 "negative_control", "shape", "dtype", "sha256", "segment_sha256")


def annotation_path(kind: str, split: str) -> str:
    return f"annotations/LIVECell_single_cells/{kind}/{split}.json"


ANNOTATION_FILES = tuple(annotation_path(k, s) for k in TYPES for s in SPLITS)


def rank(*parts) -> str:
    """Seeded order key; a sha256 rank is a permutation independent of library RNG streams."""
    return hashlib.sha256("|".join(str(p) for p in (SEED, *parts)).encode()).hexdigest()


def well_order(well: str) -> tuple[str, int]:
    return well[0], int(well[1:])


def stage_flat_annotations(flat: Path, raw: Path) -> None:
    """Copy <type>_<split>.json files into the S3 layout under raw/ and record them in SOURCE.json."""
    if (raw / "SOURCE.json").exists():
        raise FileExistsError("raw already has SOURCE.json")
    files = {}
    for kind in TYPES:
        for split in SPLITS:
            source, rel = flat / f"{kind}_{split}.json", annotation_path(kind, split)
            if not source.is_file():
                raise FileNotFoundError(f"missing annotation file: {source}")
            target = raw / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files[rel] = {"bytes": target.stat().st_size, "sha256": digest(target)}
    write_json(raw / "SOURCE.json", {"files": files})


def verified_annotations(raw: Path) -> list[dict]:
    files = read_json(raw / "SOURCE.json")["files"]
    out = []
    for rel in ANNOTATION_FILES:
        path = within(raw, rel)
        if rel not in files or not path.is_file():
            raise FileNotFoundError(f"download incomplete: {rel}")
        size, sha = path.stat().st_size, digest(path)
        if (size, sha) != (files[rel]["bytes"], files[rel]["sha256"]):
            raise ValueError(f"annotation file differs from SOURCE.json: {rel}")
        out.append({"relative_path": rel, "bytes": size, "sha256": sha})
    return out


def select(raw: Path, out_manifest: Path) -> dict:
    """Deterministic selection: every unit, both locations, one seeded crop each."""
    if out_manifest.exists():
        raise FileExistsError("manifest already exists; use a fresh path")
    annotations = verified_annotations(raw)
    copies = defaultdict(list)          # file_name -> annotation copies
    for item in annotations:
        rel = item["relative_path"]
        kind, split = rel.split("/")[-2], rel.split("/")[-1][:-5]
        coco = read_json(within(raw, rel))
        counts = Counter(a["image_id"] for a in coco["annotations"])
        for image in coco["images"]:
            name = image["file_name"]
            parsed = NAME.match(name)
            if parsed is None or parsed["type"] != TYPES[kind]:
                raise ValueError(f"unexpected image name in {rel}: {name!r}")
            if (image["height"], image["width"]) != IMAGE_SHAPE:
                raise ValueError(f"unexpected image dimensions: {name}")
            copies[name].append({"file": rel, "split": split, "coco_image_id": int(image["id"]),
                                 "n_instances": counts[image["id"]]})
    units = defaultdict(lambda: defaultdict(list))
    for name in copies:
        copies[name].sort(key=lambda c: (SPLITS.index(c["split"]), c["coco_image_id"]))
        p = NAME.match(name)
        units[(p["type"], p["well"], p["time"])][int(p["location"])].append(name)
    times = defaultdict(set)
    for kind, _, time in units:
        times[kind].add(time)
    stage = {}
    for kind, values in times.items():
        ordered = sorted(values, key=hours_of)
        stage.update({(kind, t): N_STAGES * r // len(ordered) for r, t in enumerate(ordered)})
    dev_wells = defaultdict(set)
    for kind, well, time in units:
        if stage[(kind, time)] < 2:
            dev_wells[kind].add(well)
    val_well = {kind: min(wells, key=lambda w: rank("val", kind, w)) for kind, wells in dev_wells.items()}
    well_index = {}
    for kind in times:
        ordered = sorted({w for k, w, _ in units if k == kind}, key=well_order)
        well_index.update({(kind, w): i for i, w in enumerate(ordered)})
    images = []
    for (kind, well, time), sites in sorted(units.items(), key=lambda u: (u[0][0], well_order(u[0][1]), hours_of(u[0][2]))):
        if sorted(sites) != [1, 2]:
            raise ValueError(f"unit without both locations: {kind} {well} {time}")
        s = stage[(kind, time)]
        usage = (VAL if well == val_well[kind] else TRAIN) if s < 2 else HELDOUT
        for location in (1, 2):
            name = min(sites[location], key=lambda n: rank(n))
            chosen = copies[name][0]
            images.append({"source_file": name, "image_path": f"{IMAGE_DIRS[chosen['split']]}/{name}",
                           "unit": f"{kind}_{well}_{time}", "type": kind, "well": well,
                           "well_index": well_index[(kind, well)], "location": location,
                           "crop": int(NAME.match(name)["crop"]), "time": time, "hours": hours_of(time),
                           "stage": s, "usage": usage, "official_split": chosen["split"],
                           "annotation_source": chosen["file"], "annotation_sources": copies[name]})
    manifest = {"dataset": DATASET, "seed": SEED, "image_shape": list(IMAGE_SHAPE),
                "annotations": annotations, "images": images}
    validate_manifest(manifest)
    write_manifest(out_manifest, manifest)
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    """The manifest as LF-terminated UTF-8 bytes on every platform (it is pinned by the SHA256 of these bytes)."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_value(manifest), indent=2, allow_nan=False, default=str) + "\n"
    path.write_bytes(text.encode("utf-8"))


def hours_of(time: str) -> float:
    p = re.fullmatch(r"(\d\d)d(\d\d)h(\d\d)m", time)
    return round(int(p[1]) * 24 + int(p[2]) + int(p[3]) / 60, 4)


def validate_manifest(m: dict) -> None:
    if m["dataset"] != DATASET or m["seed"] != SEED or m["image_shape"] != list(IMAGE_SHAPE):
        raise ValueError("manifest is not the seeded LIVECell selection")
    if [a["relative_path"] for a in m["annotations"]] != list(ANNOTATION_FILES):
        raise ValueError("manifest must list the 24 annotation files")
    images = m["images"]
    if len(images) != 2 * N_UNITS:
        raise ValueError(f"expected {2 * N_UNITS} images, found {len(images)}")
    if len({i["source_file"] for i in images}) != len(images) or len({i["image_path"] for i in images}) != len(images):
        raise ValueError("duplicate site")
    units = defaultdict(list)
    for i in images:
        if set(i) != MANIFEST_IMAGE_FIELDS:
            raise ValueError(f"unexpected manifest image fields: {sorted(set(i) ^ MANIFEST_IMAGE_FIELDS)}")
        within(Path.cwd(), i["image_path"])
        p = NAME.match(i["source_file"])
        if p is None or (p["type"], p["well"], int(p["location"]), int(p["crop"]), p["time"]) != (
                i["type"], i["well"], i["location"], i["crop"], i["time"]):
            raise ValueError(f"image fields disagree with the file name: {i['source_file']}")
        first = i["annotation_sources"][0]
        if (i["annotation_source"], i["official_split"]) != (first["file"], first["split"]) or \
                i["image_path"] != f"{IMAGE_DIRS[first['split']]}/{i['source_file']}":
            raise ValueError(f"annotation source is not the first copy: {i['source_file']}")
        if i["usage"] not in USAGES or (i["usage"] != HELDOUT) != (i["stage"] < 2):
            raise ValueError("train and val units must be stages 0-1 only")
        units[i["unit"]].append(i)
    if len(units) != N_UNITS:
        raise ValueError(f"expected {N_UNITS} units, found {len(units)}")
    keys = ("type", "well", "well_index", "time", "hours", "stage", "usage")
    for unit, sites in units.items():
        if sorted(s["location"] for s in sites) != [1, 2] or any(
                tuple(s[k] for k in keys) != tuple(sites[0][k] for k in keys) for s in sites):
            raise ValueError(f"unit needs two consistent sites: {unit}")
    if {i["type"] for i in images} != set(TYPES.values()):
        raise ValueError("expected the eight LIVECell types")
    for kind in TYPES.values():
        rows = [i for i in images if i["type"] == kind]
        ordered = sorted({i["time"] for i in rows}, key=hours_of)
        if any(i["stage"] != N_STAGES * ordered.index(i["time"]) // len(ordered) for i in rows):
            raise ValueError(f"stage rule violated: {kind}")
        val = {i["well"] for i in rows if i["usage"] == VAL}
        train = {i["well"] for i in rows if i["usage"] == TRAIN}
        if len(val) != 1 or not train or val & train:
            raise ValueError(f"each type needs one val well and train wells: {kind}")


def decode_rle(segmentation: dict) -> np.ndarray:
    """COCO RLE (list or compressed string counts), column-major, starting with background."""
    h, w = segmentation["size"]
    counts = segmentation["counts"]
    if isinstance(counts, str):
        values, p = [], 0
        while p < len(counts):
            x, k, more = 0, 0, True
            while more:
                c = ord(counts[p]) - 48
                x |= (c & 0x1F) << (5 * k)
                more = bool(c & 0x20)
                p, k = p + 1, k + 1
                if not more and c & 0x10:
                    x |= -1 << (5 * k)
            if len(values) > 2:
                x += values[-2]
            values.append(x)
        counts = values
    if sum(counts) != h * w or min(counts, default=0) < 0:
        raise ValueError("malformed RLE")
    flat = np.repeat(np.arange(len(counts)) % 2, counts).astype(bool)
    return flat.reshape((h, w), order="F")


def instance_mask(segmentation, shape: tuple[int, int]) -> np.ndarray:
    from skimage.draw import polygon
    if isinstance(segmentation, dict):
        if tuple(segmentation["size"]) != tuple(shape):
            raise ValueError("RLE size differs from the image")
        return decode_rle(segmentation)
    mask = np.zeros(shape, dtype=bool)
    for poly in segmentation:
        if len(poly) < 6 or len(poly) % 2:
            raise ValueError("malformed polygon")
        xy = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        # COCO coordinates are pixel edges; skimage fills pixels whose centre lies inside.
        rr, cc = polygon(xy[:, 1] - 0.5, xy[:, 0] - 0.5, shape)
        mask[rr, cc] = True
    return mask


def rasterize(annotations: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Instance k+1 is the k-th annotation by id; a later instance never overwrites an earlier one."""
    labels = np.zeros(shape, dtype=np.uint32)
    overlap = 0
    for k, a in enumerate(sorted(annotations, key=lambda a: a["id"])):
        mask = instance_mask(a["segmentation"], shape)
        free = mask & (labels == 0)
        overlap += int(mask.sum() - free.sum())
        labels[free] = k + 1
    return labels, overlap


def write_npz(path: Path, **arrays) -> None:
    """np.savez stamps the wall clock into the zip; a fixed date keeps prepared hashes reproducible."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with z.open(info, "w") as handle:
                np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def write_view(pixels: np.ndarray, image_id: str, output: Path, segment: Path) -> dict:
    """CYX (C=1) stack and YX segmenter view, native dtype; OME UUIDs derived from the id, not the clock."""
    stack = pixels[None]
    output.parent.mkdir(parents=True, exist_ok=True)
    segment.parent.mkdir(parents=True, exist_ok=True)
    for path, array, meta in ((output, stack, {"axes": "CYX", "Channel": {"Name": list(CHANNELS)}}),
                              (segment, pixels, {"axes": "YX"})):
        ident = uuid.UUID(bytes=hashlib.sha256(f"{SEED}|{path.parent.name}|{image_id}".encode()).digest()[:16])
        tifffile.imwrite(path, array, ome=True, photometric="minisblack", metadata={**meta, "UUID": f"urn:uuid:{ident}"})
        back = tifffile.imread(path)    # tifffile squeezes the C = 1 axis on read
        if back.dtype != pixels.dtype or back.shape != pixels.shape or not np.array_equal(back, pixels):
            raise ValueError(f"round-trip differs from source pixels: {path.name}")
    return {"shape": list(stack.shape), "dtype": str(pixels.dtype), "sha256": digest(output),
            "segment_sha256": digest(segment)}


class ImageSource:
    """images.zip read in place when present (pinned by IMAGES_ZIP_SHA256), else the extracted tree under raw/
    (no pin applies to its members; their SHA256s are recorded in raw_hashes.json)."""

    def __init__(self, raw: Path, source: dict):
        self.zip = None
        archive = raw / IMAGES_ZIP
        if archive.is_file():
            entry = source["files"].get(IMAGES_ZIP)
            size = archive.stat().st_size
            sha = digest(archive)
            if entry is None or size != IMAGES_ZIP_BYTES or entry["bytes"] != size or sha != entry["sha256"]:
                raise ValueError("images.zip differs from SOURCE.json or the published size")
            if sha != IMAGES_ZIP_SHA256:
                raise ValueError("images.zip differs from the pinned SHA256 of the published archive")
            self.zip = zipfile.ZipFile(archive)
            self.names = set(self.zip.namelist())
            self.record = {"mode": IMAGES_ZIP, "bytes": size, "sha256": sha, "zip_pin": "checked"}
        elif (raw / "images").is_dir():
            self.raw = raw
            self.record = {"mode": "extracted_tree", "zip_pin": "not_applicable_extracted_tree"}
        else:
            raise FileNotFoundError("neither images.zip nor an extracted images/ tree")

    def exists(self, rel: str) -> bool:
        return rel in self.names if self.zip else within(self.raw, rel).is_file()

    def read(self, rel: str) -> bytes:
        return self.zip.read(rel) if self.zip else within(self.raw, rel).read_bytes()


# The smoke's units per type: (usage, stage); the heldout stage is SMOKE_HELDOUT_STAGES[type rank // 2].
SMOKE_TRAIN_VAL = ((TRAIN, 0), (VAL, 1))
SMOKE_HELDOUT_STAGES = (2, 4)


def smoke_images(rows: list[dict], labels: list[dict]) -> list[dict]:
    """12 whole units of 4 types: per type its train unit at stage 0 and its val unit at stage 1 (the lowest unit
    id of each), and one heldout unit, at stage 2 for the first two types and at stage 4 for the other two. The types are the
    first 4 in seeded order that have all three. Every smoke stage then holds at least two units (the batch-effect detector needs two rows per batch) and every reference occurs in two
    stages, so the batch admission rule can return RUN."""
    kind = {l["image_id"]: l["cell_type"] for l in labels}

    def unit(t, usage, stage):
        found = [r["well"] for r in rows
                 if kind[r["image_id"]] == t and r["usage"] == usage and r["plate"] == f"stage_{stage}"]
        return min(found) if found else None

    chosen = []
    for t in sorted(set(kind.values()), key=lambda t: rank("smoke", t)):
        if len(chosen) == 4:
            break
        heldout = SMOKE_HELDOUT_STAGES[len(chosen) // 2]
        units = [unit(t, usage, stage) for usage, stage in (*SMOKE_TRAIN_VAL, (HELDOUT, heldout))]
        if None not in units:
            chosen.append(units)
    out = [r for r in rows if r["well"] in {u for units in chosen for u in units}]
    per_stage = Counter(stage for stage, _ in {(r["plate"], r["well"]) for r in out})
    if len(out) != 24 or len(per_stage) < 2 or min(per_stage.values()) < 2:
        raise ValueError(f"smoke set must be 24 images with at least two units per stage: {dict(per_stage)}")
    return out


def smoke_rows(prep: Path, rows: list[dict]) -> list[dict]:
    """Runtime smoke subset from public/smoke.json; needs no private file."""
    record = read_json(prep / "PREPARE.json")
    path = prep / "public" / "smoke.json"
    if digest(path) != record["smoke_sha256"]:
        raise ValueError("smoke list changed")
    units = set(read_json(path)["units"])
    out = [r for r in rows if r["well"] in units]
    if len(units) != 12 or len(out) != 24:
        raise ValueError("smoke list does not match the prepared images")
    return out


def public_units(images: list[dict]) -> dict:
    """{manifest unit: its public id unit_XXXX}, by a seeded permutation of the units."""
    names = sorted({i["unit"] for i in images}, key=lambda u: rank("unit", u))
    return {u: f"unit_{k:04d}" for k, u in enumerate(names)}


def prepare(raw: Path, out: Path, manifest: Path | None = None) -> None:
    """Write the prepared tree; the manifest defaults to <raw>/../manifest.json."""
    if out.exists():
        raise FileExistsError("prepared directory already exists; use a fresh output")
    manifest = manifest or raw.parent / "manifest.json"
    if digest(manifest) != MANIFEST_SHA256:
        raise ValueError("manifest SHA256 differs from the pinned LIVECell selection")
    m = read_json(manifest)
    validate_manifest(m)
    source = read_json(raw / "SOURCE.json")
    for item in m["annotations"]:
        entry = source["files"].get(item["relative_path"])
        if entry is None or entry["sha256"] != item["sha256"] or digest(within(raw, item["relative_path"])) != item["sha256"]:
            raise ValueError(f"annotation file differs from the manifest: {item['relative_path']}")
    images = ImageSource(raw, source)
    for i in m["images"]:       # completeness before any output
        if not images.exists(i["image_path"]):
            raise FileNotFoundError(f"download incomplete: {i['image_path']}")
    polygons = {}
    wanted = defaultdict(dict)  # annotation file -> coco id -> (source_file, expected count)
    for i in m["images"]:
        for c in i["annotation_sources"]:
            wanted[c["file"]][c["coco_image_id"]] = (i["source_file"], c["n_instances"])
    for rel, ids in wanted.items():
        coco = read_json(within(raw, rel))
        names = {im["id"]: im["file_name"] for im in coco["images"] if im["id"] in ids}
        found = defaultdict(list)
        for a in coco["annotations"]:
            if a["image_id"] in ids:
                found[a["image_id"]].append(a)
        for coco_id, (name, count) in ids.items():
            if names.get(coco_id) != name or len(found[coco_id]) != count:
                raise ValueError(f"annotation copy differs from the manifest: {rel} {name}")
        for i in m["images"]:
            if i["annotation_source"] == rel:
                polygons[i["source_file"]] = found[i["annotation_sources"][0]["coco_image_id"]]
    out.mkdir(parents=True, exist_ok=False)
    order = sorted(m["images"], key=lambda i: rank("image", i["source_file"]))
    image_ids = {i["source_file"]: f"img_{k:04d}" for k, i in enumerate(order)}
    unit_ids = public_units(m["images"])
    references = {t: f"reference_{k:03d}" for k, t in enumerate(sorted(TYPES.values(), key=lambda t: rank("reference", t)))}
    public, private, gt_index, hashes, dtype = [], [], [], [], None
    for k, i in enumerate(order):
        image = image_ids[i["source_file"]]
        payload = images.read(i["image_path"])
        pixels = tifffile.imread(io.BytesIO(payload))
        if pixels.ndim != 2 or tuple(pixels.shape) != IMAGE_SHAPE:
            raise ValueError(f"unexpected raw dimensions: {i['source_file']} {pixels.shape}")
        if dtype is None:
            dtype = str(pixels.dtype)
        elif str(pixels.dtype) != dtype:
            raise ValueError(f"source dtype differs across images: {i['source_file']} {pixels.dtype}")
        image_file, segment_file = f"images/{image}.ome.tif", f"segmentation/{image}.ome.tif"
        facts = write_view(pixels, image, out / "public" / image_file, out / "public" / segment_file)
        labels, overlap = rasterize(polygons[i["source_file"]], IMAGE_SHAPE)
        n = len(polygons[i["source_file"]])
        gt_file = f"gt/{image}.npz"
        write_npz(out / "private" / gt_file, labels=labels, n_instances=np.int64(n), overlap_pixels=np.int64(overlap),
                  annotation_ids=np.asarray(sorted(a["id"] for a in polygons[i["source_file"]]), dtype=np.int64))
        hashes.append({"image_id": image, "image_path": i["image_path"], "sha256": hashlib.sha256(payload).hexdigest()})
        gt_index.append({"image_id": image, "file": gt_file, "sha256": digest(out / "private" / gt_file),
                         "n_instances": n, "overlap_pixels": overlap})
        public.append({"image_id": image, "file": image_file, "segment_file": segment_file,
                       "plate": f"stage_{i['stage']}", "well": unit_ids[i["unit"]], "site": image,
                       "usage": i["usage"],
                       "calibration_group": references[i["type"]] if i["usage"] != HELDOUT else "",
                       "negative_control": False, **facts})
        private.append({"image_id": image, "cell_type": i["type"], "well": i["well"], "well_index": i["well_index"],
                        "location": i["location"], "crop": i["crop"], "hours": i["hours"], "stage": i["stage"],
                        "official_split": i["official_split"], "source_file": i["source_file"],
                        "annotation_source": i["annotation_source"], "annotation_sources": i["annotation_sources"]})
        if (k + 1) % 100 == 0:
            print(f"PREPARE {k + 1}/{len(order)}", flush=True)
    public.sort(key=lambda r: r["image_id"])
    private.sort(key=lambda r: r["image_id"])
    gt_index.sort(key=lambda r: r["image_id"])
    hashes.sort(key=lambda r: r["image_id"])
    smoke = sorted({r["well"] for r in smoke_images(public, private)})
    write_json(out / "public" / "images.json", public)
    write_json(out / "public" / "smoke.json", {"units": smoke})
    write_json(out / "private" / "labels.json", private)
    write_json(out / "private" / "gt_index.json", gt_index)
    write_json(out / "private" / "raw_hashes.json", {"annotations": m["annotations"], "images_source": images.record,
                                                     "images": hashes})
    write_json(out / "PREPARE.json", {
        "status": "complete", "dataset": DATASET, "manifest_sha256": MANIFEST_SHA256,
        "images_sha256": digest(out / "public" / "images.json"),
        "smoke_sha256": digest(out / "public" / "smoke.json"),
        "labels_sha256": digest(out / "private" / "labels.json"),
        "gt_index_sha256": digest(out / "private" / "gt_index.json"),
        "raw_hashes_sha256": digest(out / "private" / "raw_hashes.json"),
        "channels": list(CHANNELS), "segmentation_channels": list(CHANNELS), "image_shape": list(IMAGE_SHAPE),
        "dtype": dtype, "n_images": len(public), "n_units": len(unit_ids), "images_source": images.record,
        "versions": {"tifffile": tifffile.__version__, "numpy": np.__version__}})


def load_prepared(prep: Path, *, verify_pixels: bool = False) -> list[dict]:
    record = read_json(prep / "PREPARE.json")
    if record["status"] != "complete" or record["dataset"] != DATASET or record["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("preparation incomplete or wrong subset")
    table = prep / "public" / "images.json"
    if digest(table) != record["images_sha256"]:
        raise ValueError("prepared metadata changed")
    rows = read_json(table)
    if len(rows) != 2 * N_UNITS or any(tuple(r) != PUBLIC_FIELDS for r in rows):
        raise ValueError("prepared table is not the complete public LIVECell table")
    if verify_pixels:
        for row in rows:
            for key, sha in (("file", "sha256"), ("segment_file", "segment_sha256")):
                if digest(within(prep / "public", row[key])) != row[sha]:
                    raise ValueError(f"prepared pixels changed: {row['image_id']} {key}")
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m benchmarks.full_LIVECell.harness.data")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stage-annotations", help="copy flat <type>_<split>.json files into the S3 layout")
    s.add_argument("--flat", type=Path, required=True)
    s.add_argument("--raw", type=Path, required=True)
    s = sub.add_parser("select")
    s.add_argument("--raw", type=Path, required=True)
    s.add_argument("--manifest", type=Path, required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--raw", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--manifest", type=Path)
    args = p.parse_args(argv)
    if args.command == "stage-annotations":
        stage_flat_annotations(args.flat.resolve(), args.raw.resolve())
    elif args.command == "select":
        m = select(args.raw.resolve(), args.manifest.resolve())
        sha = digest(args.manifest)
        counts = Counter(i["type"] for i in m["images"])
        print(f"MANIFEST sha256={sha} pinned={sha == MANIFEST_SHA256} images={len(m['images'])} "
              f"units_per_type={ {k: v // 2 for k, v in sorted(counts.items())} }", flush=True)
    else:
        prepare(args.raw.resolve(), args.out.resolve(), args.manifest.resolve() if args.manifest else None)
        print("PREPARE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
