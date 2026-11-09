"""Prepare ALL images of `250429_96well_CO_DEFINE candidates` for a SHAPE run in the layout the shared harness reads:
public phase-contrast images (segment and crop decisions), public two-channel (PhaseContrast, GFP) CYX stacks (the SSL
input; crop boxes chosen on the phase image cut them at identical coordinates), private GFP files and tables.
Identifiers are seeded pseudonyms; the mapping is private.

Split (user decision 2026-10-25): train / validation only, 8:2, no evaluation set. The unit is the plate-well: every time
point of a plate-well is in one split (no leakage between train and val); within each plate the wells are shuffled with a
seeded RNG and the first round(0.8 * n) are train. The final embedding covers every image (train and val).

usage: python prepare_full.py SOURCE_DIR OUT_DIR       (Slurm job: ~10 GB of reads and writes)
"""
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile

SEED = 20261025
TRAIN_FRACTION = 0.8
SHAPE, DTYPE = [904, 1224], "uint16"
PROTOCOL = "define-candidates-plate-well-train-val-80-20-v1"
CHANNELS = ("PhaseContrast", "GFP")
DAYS = {"baseline": 0, "d2": 2, "d4": 4, "d8": 8, "d12": 12, "d16": 16}
FOLDER_RE = re.compile(r"^(\d{6})_(\d{6})_(Plate \d)_(baseline|d\d+)$")
PHASE_RE = re.compile(r"^([A-H])(\d{1,2})_(\d\d)_1_1_Phase Contrast_001\.tif$")
GFP = "{well}_{seq}_2_1_GFP_001.tif"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(path):
    a = tifffile.imread(path)
    if a.ndim != 2 or list(a.shape) != SHAPE or str(a.dtype) != DTYPE:
        raise ValueError(f"{path}: shape {a.shape} dtype {a.dtype}, expected {SHAPE} {DTYPE}")
    return a


def pseudonyms(keys, prefix, width, salt):
    order = sorted(keys)
    random.Random(f"{SEED}|{salt}").shuffle(order)
    return {k: f"{prefix}{i:0{width}d}" for i, k in enumerate(order)}


def main(source, out):
    source, out = Path(source), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    rows = []
    for folder in sorted(d for d in os.listdir(source) if (source / d).is_dir()):
        m = FOLDER_RE.match(folder)
        if not m:
            raise ValueError(f"unexpected folder name: {folder}")
        plate, tp = m.group(3), m.group(4)
        names = set(os.listdir(source / folder))
        for name in sorted(names):
            p = PHASE_RE.match(name)
            if not p:
                continue
            well, seq = f"{p.group(1)}{p.group(2)}", p.group(3)
            gfp = GFP.format(well=well, seq=seq)
            if gfp not in names:
                raise FileNotFoundError(f"GFP channel missing for {folder}/{name}")
            rows.append(dict(folder=folder, plate=plate, timepoint=tp, day=DAYS[tp], well=well, seq=seq,
                             phase=source / folder / name, gfp=source / folder / gfp))
    keys = [(r["plate"], r["well"], r["day"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (plate, well, day) image")
    print(f"INVENTORY {len(rows)} images, {len({(r['plate'], r['well']) for r in rows})} plate-wells", flush=True)

    # split by plate-well, within plate, seeded
    wells_by_plate = defaultdict(set)
    for r in rows:
        wells_by_plate[r["plate"]].add(r["well"])
    usage_of = {}
    for plate, wells in sorted(wells_by_plate.items()):
        order = sorted(wells)
        random.Random(f"{SEED}|split|{plate}").shuffle(order)
        n_train = round(TRAIN_FRACTION * len(order))
        for i, w in enumerate(order):
            usage_of[(plate, w)] = "development_train" if i < n_train else "development_val"
    plate_ids = pseudonyms(sorted(wells_by_plate), "p", 3, "plates")
    unit_ids = pseudonyms(sorted(usage_of), "w", 3, "wells")
    image_ids = pseudonyms(keys, "i", 4, "images")

    out.mkdir(parents=True)
    public, private = [], {}
    for n, r in enumerate(sorted(rows, key=lambda r: image_ids[(r["plate"], r["well"], r["day"])])):
        ident = image_ids[(r["plate"], r["well"], r["day"])]
        phase, gfp = check(r["phase"]), check(r["gfp"])
        phase_sha, gfp_sha = sha256(r["phase"]), sha256(r["gfp"])
        pub = out / "public" / "images" / f"{ident}.tif"
        pub.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(r["phase"], pub)
        if sha256(pub) != phase_sha:
            raise ValueError("phase copy changed pixels")
        priv = out / "private" / "gfp" / f"{ident}.tif"
        priv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(r["gfp"], priv)
        if sha256(priv) != gfp_sha:
            raise ValueError("gfp copy changed pixels")
        raw = out / "public" / "raw" / f"{ident}.tif"
        raw.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(raw, np.stack([phase, gfp]).astype(np.uint16), metadata={"axes": "CYX", "Channel": {"Name": list(CHANNELS)}})
        stack = tifffile.imread(raw)
        if not (np.array_equal(stack[0], phase) and np.array_equal(stack[1], gfp)):   # S: stack channels equal the source channels
            raise ValueError(f"two-channel stack differs from its sources: {ident}")
        usage = usage_of[(r["plate"], r["well"])]
        public.append(dict(image_id=ident, file=f"images/{ident}.tif", segment_file=f"images/{ident}.tif", sha256=phase_sha,
                           segment_sha256=phase_sha, plate=plate_ids[r["plate"]], plate_id=plate_ids[r["plate"]],
                           well=unit_ids[(r["plate"], r["well"])], site=ident, acquisition_id="a000", experiment_date="a000",
                           usage=usage, partition="seen", shape=[2, *SHAPE], dtype=DTYPE, calibration_group="",
                           negative_control=False, raw_file=f"raw/{ident}.tif", raw_sha256=sha256(raw)))
        private[ident] = dict(folder=r["folder"], plate=r["plate"], well=r["well"], day=r["day"], timepoint=r["timepoint"],
                              usage=usage, phase_source=str(r["phase"]), gfp_source=str(r["gfp"]), phase_sha256=phase_sha,
                              gfp_sha256=gfp_sha, gfp_file=f"gfp/{ident}.tif")
        if (n + 1) % 100 == 0:
            print(f"PREPARE {n + 1}/{len(rows)}", flush=True)

    # checks of the split contract
    by_unit = defaultdict(set)
    for p in public:
        by_unit[p["well"]].add(p["usage"])
    if any(len(v) != 1 for v in by_unit.values()):
        raise ValueError("a plate-well crosses splits")
    counts = Counter(p["usage"] for p in public)
    unit_counts = Counter(next(iter(v)) for v in by_unit.values())
    (out / "public" / "images.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    (out / "private" / "images.json").write_text(json.dumps(private, indent=2), encoding="utf-8")
    (out / "private" / "splits.json").write_text(json.dumps(dict(
        protocol=PROTOCOL, seed=SEED, unit="plate-well", train_fraction=TRAIN_FRACTION, images=dict(counts), plate_wells=dict(unit_counts),
        per_plate={pl: dict(Counter(usage_of[(pl, w)] for w in ws)) for pl, ws in sorted(wells_by_plate.items())},
        note="no evaluation set; the final embedding covers every image"), indent=2), encoding="utf-8")
    record = dict(status="complete", dataset="define_candidates_96well_CO", protocol=PROTOCOL, seed=SEED, images=len(public),
                  plate_wells=len(by_unit), images_by_usage=dict(counts), plate_wells_by_usage=dict(unit_counts),
                  variant="ssl_two_channel", ssl_channels=list(CHANNELS),
                  stage_channels=dict(segment=["PhaseContrast"], crop=["PhaseContrast"], embed=list(CHANNELS)),
                  source=str(source), image_shape=SHAPE, dtype=DTYPE,
                  images_sha256=sha256(out / "public" / "images.json"), private_images_sha256=sha256(out / "private" / "images.json"),
                  splits_sha256=sha256(out / "private" / "splits.json"),
                  identifier_policy="image, well and plate identifiers are seeded pseudonyms; the mapping is private")
    (out / "PREPARE.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("PREPARE_DONE", json.dumps({k: record[k] for k in ("images", "plate_wells", "images_by_usage", "plate_wells_by_usage")}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
