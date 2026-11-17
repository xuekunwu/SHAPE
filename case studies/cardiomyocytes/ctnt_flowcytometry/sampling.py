"""Frozen metadata-only 20% sample of each existing split; no well is reassigned."""

from collections import Counter, defaultdict
import hashlib
from pathlib import Path

from . import preparation as data

FRACTION = .2
SEED = 20261016


def select(rows: list[dict], seed: int = SEED) -> list[dict]:
    wells = {}
    for row in rows:
        key = row["well"]
        identity = tuple(row[k] for k in ("usage", "experiment_date", "plate_id"))
        if key in wells and wells[key] != identity:
            raise ValueError("inconsistent well metadata")
        wells[key] = identity
    selected = set()
    for usage in data.USAGES:
        group = {w: v for w,v in wells.items() if v[0] == usage}
        count = round(len(group)*FRACTION)
        if count < (2 if usage == data.USAGES[0] else 1):
            raise ValueError("pilot too small for regression train/validation/evaluation")
        dates = data.apportion(dict(Counter(v[1] for v in group.values())),
                               (count, len(group)-count, 0), seed)
        for date, quota in dates.items():
            plates = defaultdict(list)
            for well, (_, d, plate) in group.items():
                if d == date:
                    plates[plate].append(well)
            quotas = data.apportion({p:len(v) for p,v in plates.items()}, quota, seed)
            for plate, allocation in quotas.items():
                order = sorted(plates[plate], key=lambda w: hashlib.sha256(
                    f"{seed}|pilot|{usage}|{w}".encode()).hexdigest())
                selected.update(order[:allocation[0]])
    return sorted((r for r in rows if r["well"] in selected), key=lambda r:r["image_id"])


def manifest(prep: Path, seed: int = SEED) -> dict:
    rows = select(data.load_prepared(prep), seed)
    wells = {r["well"]:r for r in rows}
    return {"protocol": "ctnt-pilot-20pct-within-fixed-splits-v1", "parent_protocol": data.PROTOCOL,
            "prepare_sha256": data.digest(prep/"PREPARE.json"), "seed": seed, "fraction": FRACTION,
            "unit": "well", "sampling": "metadata-only date/plate-stratified random sample within each fixed split",
            "images": [r["image_id"] for r in rows], "wells": sorted(wells),
            "split_wells": dict(Counter(r["usage"] for r in wells.values())),
            "split_images": dict(Counter(r["usage"] for r in rows)),
            "batch_balance": {field:{value:dict(Counter(r["usage"] for r in wells.values() if r[field]==value))
                for value in sorted({r[field] for r in wells.values()})}
                for field in ("experiment_date", "plate_id")},
            "arms": ["control", "shape", "codex"], "model": "gpt-5.6-luna", "reasoning_effort": "high"}


def verify(prep: Path, path: Path) -> dict:
    record = data.read_json(path)
    if record != manifest(prep, record["seed"]):
        raise ValueError("pilot manifest differs from its frozen preparation/sampling protocol")
    return record
