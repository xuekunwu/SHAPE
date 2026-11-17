"""Pinned source audit, batch-balanced well splits and complete-well smoke selection."""
from __future__ import annotations


from collections import Counter, defaultdict
import csv
import hashlib
import importlib
import io
import json
import itertools
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SPEC = json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))
USAGES = ("development_train", "development_val", "evaluation_only")
PROTOCOL = "well-batch-balanced-40-10-50-v2"
TARGET = "ctnt_positive_percent"


from .artifacts import digest
from .artifacts import read_json
from .artifacts import write_json
from .artifacts import within
from .artifacts import fresh_output




def source_rows(root: Path):
    raw = (root / SPEC["metadata_path"]).read_bytes()
    audit = importlib.import_module("case studies._shared.metadata").audit(raw, SPEC)
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))), audit


def apportion(groups: dict, totals: tuple[int, int, int], seed: int) -> dict:
    """Deterministic controlled rounding: exact totals, floor/ceil quota per stratum.

    Dynamic programming minimizes squared quota error. Metadata-only hash ordering
    breaks ties; neither target values nor the number of views enter allocation.
    """
    size = sum(groups.values())
    if size != sum(totals) or size <= 0:
        raise ValueError("invalid allocation totals")
    order = sorted(groups, key=lambda k: hashlib.sha256(f"{seed}|{k}".encode()).hexdigest())
    states = {(0, 0, 0): (0., {})}
    for key in order:
        quota = [groups[key]*v/size for v in totals]
        options = [v for v in itertools.product(*[sorted({math.floor(q), math.ceil(q)}) for q in quota])
                   if sum(v) == groups[key]]
        following = {}
        for used, (cost, chosen) in states.items():
            for counts in options:
                new = tuple(a+b for a,b in zip(used, counts))
                if any(a>b for a,b in zip(new, totals)):
                    continue
                value = cost + sum((v-q)**2 for v,q in zip(counts, quota))
                if new not in following or value < following[new][0] - 1e-12:
                    following[new] = (value, {**chosen, key: counts})
        states = following
    if totals not in states:
        raise ValueError("cannot balance strata with exact split totals")
    return states[totals][1]


def well_split(rows: list[dict], seed: int) -> dict[tuple[str, str], str]:
    """40/10/50 wells, balanced first by date and then plate within date."""
    wells = {}
    plate_dates = defaultdict(set)
    for row in rows:
        key = row["plate_id"], row["well_id"]
        date = row["experiment_date"].strip()
        if not date or not all(key):
            raise ValueError("missing split/batch identity")
        if key in wells and wells[key] != date:
            raise ValueError("conflicting experiment_date within well")
        wells[key] = date
        plate_dates[key[0]].add(date)
    if any(len(v) != 1 for v in plate_dates.values()):
        raise ValueError("plate_id must identify one experiment_date")
    n = len(wells)
    totals = (round(.4*n), round(.1*n), n-round(.4*n)-round(.1*n))
    if min(totals) < 1:
        raise ValueError("need wells in all three splits")
    dates = apportion(dict(Counter(wells.values())), totals, seed)
    splits = {}
    for date, counts in dates.items():
        groups = defaultdict(list)
        for key, d in wells.items():
            if d == date:
                groups[key[0]].append(key)
        plates = apportion({p: len(v) for p,v in groups.items()}, counts, seed)
        for plate, allocation in plates.items():
            ordered = sorted(groups[plate], key=lambda k: hashlib.sha256(f"{seed}|well|{k}".encode()).hexdigest())
            offset = 0
            for usage, count in zip(USAGES, allocation):
                splits.update({key: usage for key in ordered[offset:offset+count]})
                offset += count
    return splits


def prepare(root: Path, out: Path, seed: int = 20261015) -> dict:
    """Decode/hash every source and write anonymous, lossless uint16 TIFFs. Never modify the source."""
    import numpy as np
    import tifffile

    fresh_output(out, root)
    rows, audit = source_rows(root)
    splits = well_split(rows, seed)
    dates = {d: f"d{i:03d}" for i,d in enumerate(sorted({r["experiment_date"] for r in rows}))}
    plates = {p: f"p{i:03d}" for i, p in enumerate(sorted({r["plate_id"] for r in rows}))}
    wells = {w: f"w{i:04d}" for i, w in enumerate(sorted({(r["plate_id"], r["well_id"]) for r in rows}))}
    out.mkdir(parents=True)
    public, private, hashes = [], {}, []
    for index, row in enumerate(rows):
        source = within(root, row["relative_path"])
        before = digest(source)
        image = tifffile.imread(source)
        if image.shape != (2048, 2048) or image.dtype != np.dtype("uint16"):
            raise ValueError(f"unexpected image shape/dtype: {row['image_id']} {image.shape} {image.dtype}")
        if before != digest(source):
            raise ValueError("source changed during preparation")
        ident, well = f"i{index:04d}", wells[row["plate_id"], row["well_id"]]
        relative = f"images/{ident}.tif"
        target = out / "public" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(target, image, metadata={"axes": "YX"}, photometric="minisblack")
        if not np.array_equal(tifffile.imread(target), image):
            raise ValueError("prepared image round-trip changed pixels")
        sha = digest(target)
        public.append({"image_id": ident, "file": relative, "segment_file": relative,
                       "plate": plates[row["plate_id"]], "well": well, "site": row["site_id"],
                       "plate_id": plates[row["plate_id"]], "experiment_date": dates[row["experiment_date"]],
                       "usage": splits[row["plate_id"], row["well_id"]],
                       "partition": "held_out" if splits[row["plate_id"], row["well_id"]] == USAGES[2] else "seen",
                       "calibration_group": "", "negative_control": False,
                       "shape": [1, 2048, 2048], "dtype": "uint16", "sha256": sha, "segment_sha256": sha})
        private[well] = {"donor_id": row["donor_id"], "plate_id": row["plate_id"], "well_id": row["well_id"],
                         "flow_sample_id": row["flow_sample_id"], "usage": splits[row["plate_id"], row["well_id"]],
                         "experiment_date": row["experiment_date"],
                         TARGET: float(row[TARGET])}
        hashes.append({"image_id": ident, "relative_path": row["relative_path"], "sha256": before,
                       "bytes": source.stat().st_size})
        if (index + 1) % 50 == 0:
            print(f"PREPARE {index+1}/{len(rows)}", flush=True)
    write_json(out / "public" / "images.json", public)
    write_json(out / "private" / "labels.json", private)
    write_json(out / "private" / "development_labels.json",
               {w: r for w, r in private.items() if r["usage"] != "evaluation_only"})
    write_json(out / "private" / "raw_hashes.json", hashes)
    write_json(out / "private" / "splits.json", {"seed": seed, "unit": "well", "protocol": PROTOCOL,
               "wells": {wells[k]: v for k,v in splits.items()},
               "balance_fields": ["experiment_date", "plate_id"], "target_ratios": [.4, .1, .5],
               "counts": dict(Counter(splits.values()))})
    record = {"status": "complete", "dataset": SPEC["study"], "revision": SPEC["revision"],
              "metadata_sha256": SPEC["metadata_sha256"], "images": len(public), "wells": len(private),
              "image_shape": [2048, 2048], "channels": ["Brightfield"], "dtype": "uint16", "seed": seed,
              "protocol": PROTOCOL, "target": TARGET, "target_unit": "percent", "split_unit": "well",
              "batch_fields": ["experiment_date", "plate_id"],
              "split_balance": {field: {v: dict(Counter(r["usage"] for r in private.values() if r[field] == v))
                                for v in sorted({r[field] for r in private.values()})}
                                for field in ("experiment_date", "plate_id")},
              "split_wells": dict(Counter(r["usage"] for r in private.values())),
              "source_audit": audit, "image_pixels_verified": True,
              **{key: digest(out / folder / name) for key, folder, name in (
                  ("images_sha256", "public", "images.json"), ("labels_sha256", "private", "labels.json"),
                  ("raw_hashes_sha256", "private", "raw_hashes.json"), ("splits_sha256", "private", "splits.json"),
                  ("development_labels_sha256", "private", "development_labels.json"))}}
    write_json(out / "PREPARE.json", record)
    load_prepared(out)
    return record


def validate_rows(rows: list[dict]):
    if len(rows) != SPEC["expected_images"] or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("incomplete or duplicate image manifest")
    wells = defaultdict(list)
    for r in rows:
        if r["usage"] not in USAGES or r["shape"] != [1, 2048, 2048] or r["dtype"] != "uint16":
            raise ValueError("invalid prepared image contract")
        wells[r["well"]].append(r)
    if len(wells) != SPEC["expected_wells"] or Counter(map(len, wells.values())) != Counter({4:75, 25:12}):
        raise ValueError("incomplete well/view coverage")
    for group in wells.values():
        if any(len({r[k] for r in group}) != 1 for k in ("plate", "plate_id", "experiment_date", "usage", "partition")):
            raise ValueError("one well crosses a split or plate")
        if len({r["site"] for r in group}) != len(group):
            raise ValueError("duplicate site in a well")


def load_prepared(prep: Path, *, verify_pixels: bool = False) -> list[dict]:
    record = read_json(prep / "PREPARE.json")
    if record.get("protocol") != PROTOCOL or record.get("target") != TARGET:
        raise ValueError("preparation uses an obsolete split/target protocol; prepare a fresh directory")
    if (record.get("status"), record.get("dataset"), record.get("metadata_sha256"), record.get("revision")) != (
            "complete", SPEC["study"], SPEC["metadata_sha256"], SPEC["revision"]):
        raise ValueError("preparation is incomplete or belongs to another dataset revision")
    if digest(prep / "public" / "images.json") != record["images_sha256"]:
        raise ValueError("prepared manifest changed")
    rows = read_json(prep / "public" / "images.json")
    validate_rows(rows)
    if verify_pixels:
        for r in rows:
            if r["file"] != r["segment_file"] or r["sha256"] != r["segment_sha256"]:
                raise ValueError("brightfield raw and segmentation input must be identical")
            if digest(within(prep / "public", r["file"])) != r["sha256"]:
                raise ValueError(f"prepared pixels changed: {r['image_id']}")
    return rows


def verify_private(prep: Path, rows: list[dict]) -> dict:
    record = read_json(prep / "PREPARE.json")
    for name in ("labels", "raw_hashes", "splits"):
        if digest(prep / "private" / f"{name}.json") != record[f"{name}_sha256"]:
            raise ValueError(f"private preparation changed: {name}")
    labels = read_json(prep / "private" / "labels.json")
    splits = read_json(prep / "private" / "splits.json")
    if set(labels) != {r["well"] for r in rows}:
        raise ValueError("label and image wells differ")
    if splits.get("unit") != "well" or splits.get("protocol") != PROTOCOL:
        raise ValueError("obsolete split protocol")
    original = [{**label, "ctnt_positive_percent": 0} for label in labels.values()]
    expected = well_split(original, splits["seed"])
    plate_codes, date_codes = defaultdict(set), defaultdict(set)
    for r in rows:
        label = labels[r["well"]]
        if (r["usage"] != label["usage"] or r["usage"] != splits["wells"][r["well"]]
                or r["usage"] != expected[label["plate_id"], label["well_id"]]):
            raise ValueError("split mismatch")
        if not 0 <= label[TARGET] <= 100:
            raise ValueError("invalid percent")
        if r["plate"] != r["plate_id"] or r["partition"] != ("held_out" if r["usage"] == USAGES[2] else "seen"):
            raise ValueError("batch or seen/held-out partition mismatch")
        plate_codes[label["plate_id"]].add(r["plate_id"])
        date_codes[label["experiment_date"]].add(r["experiment_date"])
    for mapping in (plate_codes, date_codes):
        if any(len(v) != 1 for v in mapping.values()) or len(set.union(*mapping.values())) != len(mapping):
            raise ValueError("batch encoding is not one-to-one")
    return labels


def smoke_rows(rows: list[dict]) -> list[dict]:
    """Select about 5% of images as four complete wells spanning all three splits."""
    from itertools import combinations, product

    groups = {usage: defaultdict(list) for usage in USAGES}
    for row in rows:
        groups[row["usage"]][row["well"]].append(row)
    if any(len(groups[usage]) < n for usage, n in zip(USAGES, (2, 1, 1))):
        raise ValueError("not enough complete wells for smoke")
    target = round(len(rows) * .05)
    best = None
    chosen = None
    for train, val, held in product(combinations(sorted(groups[USAGES[0]]), 2),
                                    sorted(groups[USAGES[1]]), sorted(groups[USAGES[2]])):
        wells = (*train, val, held)
        counts = [sum(len(groups[usage][well]) for well in selection)
                  for usage, selection in zip(USAGES, (train, (val,), (held,)))]
        score = (abs(sum(counts) - target),
                 abs(counts[0] / sum(counts) - .4) + abs(counts[1] / sum(counts) - .1)
                 + abs(counts[2] / sum(counts) - .5),
                 wells)
        if best is None or score < best:
            best, chosen = score, set(wells)
    return [row for row in rows if row["well"] in chosen]
