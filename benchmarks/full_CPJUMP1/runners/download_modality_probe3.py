#!/usr/bin/env python
"""Download ModalityProbe3: same 6 groups, same 12 locked plates as probe2 (MODALITY_PROBE2_PLAN.md), but
restructured so SHAPE's own SSL validation (sibling_site_map, shape/tools/cell_state_analyzer/quant.py) works
natively -- no gate/guard override needed. 25 shared well positions per group (half of probe2's 50), 2 REAL
sites per (plate, well) instead of 1: same plate, same well, same imaging session, just two different
fields of view -- exactly the "2 sites per well" structure original CPJUMP1 already has (confirmed working
via seed1/seed2's SHAPE arms, gate=RUN, adopted_proxy=sibling_site_map).

25 wells x 2 plates x 2 sites = 100 images/group (600 total, same overall scale as probe1/probe2).
Cross-plate identity twins (probe2's design) are preserved: both plates still share the same well
positions, so gene/compound retrieval still works, now aggregated over 2 real sites per (plate, well)
via sites_per_well=2 -- CPJUMP1's own default, no dataset-specific override needed either.

Blank-well screening (DNA_max >= 5000, probe1/probe2's established floor) now checks ALL 4 candidate images
per well (2 sites x 2 plates) before committing.

    cd /path/to/SHAPE
    .venv/bin/python benchmarks/full_CPJUMP1/runners/download_modality_probe3.py --n-wells 25 --workers 16
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile

OUT_ROOT = Path("data/CPJUMP1_modality_probe3")
LOAD_DATA_URL = ("https://cellpainting-gallery.s3.amazonaws.com/cpg0000-jump-pilot/source_4/"
                 "workspace/load_data_csv/2020_11_04_CPJUMP1/{plate}/load_data.csv")
CHANNELS = [(1, "Mito"), (2, "AGP"), (3, "RNA"), (4, "ER"), (5, "DNA")]
DNA_MAX_FLOOR = 5000
N_SITES_PER_WELL = 2

# label, cell_type, perturbation, time_h, plate_a, plate_b -- same 12 plates as MODALITY_PROBE2_PLAN.md
GROUPS = [
    ("A549_compound", "A549", "compound", "48", "BR00117052", "BR00117016"),
    ("A549_crispr", "A549", "crispr", "144", "BR00117000", "BR00117003"),
    ("A549_orf", "A549", "orf", "96", "BR00118050", "BR00117006"),
    ("U2OS_compound", "U2OS", "compound", "48", "BR00117013", "BR00117012"),
    ("U2OS_crispr", "U2OS", "crispr", "144", "BR00116996", "BR00116998"),
    ("U2OS_orf", "U2OS", "orf", "96", "BR00118039", "BR00118040"),
]


def fetch_load_data(plate: str) -> list[dict]:
    print(f"[catalog] fetching load_data.csv for {plate}", file=sys.stderr)
    with urllib.request.urlopen(LOAD_DATA_URL.format(plate=plate), timeout=180) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def site_rows(rows: list[dict]) -> dict[tuple, dict]:
    by_site: dict[tuple, dict] = {}
    for row in rows:
        key = (row["Metadata_Well"], int(row["Metadata_Site"]))
        urls = by_site.setdefault(key, {})
        for _, name in CHANNELS:
            col = f"URL_Orig{name}"
            if col in row and row[col]:
                urls[name] = row[col]
    return by_site


def best_sites_per_well(by_site: dict[tuple, dict], plate: str, n: int) -> dict[str, list[int]]:
    """well -> its n lowest-hash complete-channel sites (deterministic pick among that well's sites).
    A well with fewer than n complete sites is dropped entirely (can't give a real sibling pair)."""
    complete = [k for k, v in by_site.items() if len(v) == len(CHANNELS)]
    by_well: dict[str, list[int]] = {}
    for well, site in complete:
        by_well.setdefault(well, []).append(site)
    order_key = lambda well, site: hashlib.sha256(f"{plate}|{well}|{site}".encode()).hexdigest()
    return {well: sorted(sites, key=lambda s: order_key(well, s))[:n]
            for well, sites in by_well.items() if len(sites) >= n}


def s3_to_https(s3_url: str) -> str:
    return "https://cellpainting-gallery.s3.amazonaws.com/" + s3_url.split("cellpainting-gallery/", 1)[1]


def fetch_file(url: str, dest: Path, retries: int = 5) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as out:
                clen = r.headers.get("Content-Length")
                shutil.copyfileobj(r, out, length=8 * 1024 * 1024)
            n = tmp.stat().st_size
            if clen and n != int(clen):
                raise IOError(f"size {n} != Content-Length {clen} (truncated)")
            tmp.replace(dest)
            return "downloaded"
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"failed after {retries} tries: {url} ({last})")


def dna_signal_ok(path: Path) -> bool:
    return float(tifffile.imread(path).max()) >= DNA_MAX_FLOOR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-wells", type=int, default=25)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    a = ap.parse_args()

    manifest = {"name": "CPJUMP1-modality-probe3-v1", "n_wells_per_plate": a.n_wells,
               "sites_per_well": N_SITES_PER_WELL, "groups": {}}
    for label, cell, pert, time_h, plate_a, plate_b in GROUPS:
        by_site_a = site_rows(fetch_load_data(plate_a))
        by_site_b = site_rows(fetch_load_data(plate_b))
        sites_a = best_sites_per_well(by_site_a, plate_a, N_SITES_PER_WELL)
        sites_b = best_sites_per_well(by_site_b, plate_b, N_SITES_PER_WELL)
        shared_wells = sorted(set(sites_a) & set(sites_b),
                              key=lambda w: hashlib.sha256(f"{label}|{w}".encode()).hexdigest())
        print(f"[{label}] {plate_a}/{plate_b}: {len(shared_wells)} well positions with >= "
             f"{N_SITES_PER_WELL} complete sites on both plates", flush=True)

        kept: list[str] = []
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for well in shared_wells:
                if len(kept) >= a.n_wells:
                    break
                probe_jobs = {}
                for plate, by_site, sites_of in ((plate_a, by_site_a, sites_a), (plate_b, by_site_b, sites_b)):
                    for site in sites_of[well]:
                        url = s3_to_https(by_site[(well, site)]["DNA"])
                        dest = a.out / "raw" / label / f"_probe_{plate}_{well}_s{site:02d}_DNA.tiff"
                        probe_jobs[(plate, site)] = pool.submit(fetch_file, url, dest)
                ok = True
                for plate, by_site, sites_of in ((plate_a, by_site_a, sites_a), (plate_b, by_site_b, sites_b)):
                    for site in sites_of[well]:
                        dest = a.out / "raw" / label / f"_probe_{plate}_{well}_s{site:02d}_DNA.tiff"
                        probe_jobs[(plate, site)].result()
                        if not dna_signal_ok(dest):
                            ok = False
                if ok:
                    kept.append(well)
                else:
                    print(f"  [{label}] {well}: a site is blank on at least one plate, skipped", flush=True)
        if len(kept) < a.n_wells:
            raise SystemExit(f"{label}: only {len(kept)}/{a.n_wells} wells passed the DNA-signal screen "
                             f"out of {len(shared_wells)} candidates -- widen the candidate pool or lower the floor")
        print(f"[{label}] {len(kept)} wells passed DNA screen", flush=True)

        jobs, views = [], []
        for plate, by_site, sites_of in ((plate_a, by_site_a, sites_a), (plate_b, by_site_b, sites_b)):
            for well in kept:
                for site in sites_of[well]:
                    urls = by_site[(well, site)]
                    files = {}
                    for idx, name in CHANNELS:
                        https_url = s3_to_https(urls[name])
                        rel = f"{label}/{plate}_{well}_s{site:02d}_ch{idx}_{name}.tiff"
                        dest = a.out / "raw" / rel
                        jobs.append((https_url, dest))
                        files[name] = rel
                    views.append({"plate": plate, "well": well, "site": site, "files": files})
        manifest["groups"][label] = {"cell_type": cell, "perturbation": pert, "time_h": time_h,
                                     "plates": [plate_a, plate_b], "wells": kept, "views": views}

        done = 0
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(fetch_file, url, dest): (url, dest) for url, dest in jobs}
            for fut in as_completed(futures):
                url, dest = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAILED {dest.name}: {exc}", file=sys.stderr)
                    raise
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"  [{label}] {done}/{len(jobs)} files", flush=True)

        for f in (a.out / "raw" / label).glob("_probe_*_DNA.tiff"):
            f.unlink()

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(len(v["files"]) for g in manifest["groups"].values() for v in g["views"])
    print(f"\ndone: {total} files across {len(GROUPS)} groups -> {a.out}/raw/, manifest -> {a.out}/manifest.json")


if __name__ == "__main__":
    main()
