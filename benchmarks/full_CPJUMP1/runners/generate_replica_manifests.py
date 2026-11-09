#!/usr/bin/env python3
"""Generate replication-validation manifests for the frozen CPJUMP1 1000-view selection.

The frozen manifest (manifest.json) selected 1000 multichannel views = 500 wells x 2
sites. This script produces two NEW manifests that keep the SAME 500 wells (same
perturbations / genes / guides) but pick DIFFERENT microscope fields (sites) per well.
The non-overlap axis is therefore the SITE (field), giving independent field replicates
of the same biology -- the right design for validating that cohort / SSL-ES effects
replicate.

Guarantees:
  * manifest A (seed 20261009), manifest B (seed 20261010) and the current manifest are
    pairwise DISJOINT at the site_id level.
  * Each new manifest: 1000 sites / 500 wells / 2 sites per well, 5000 files
    (5 channels per site).
  * Field choices are deterministic: for each (plate, well) the available fields minus
    (current fields) are shuffled by a sha256(seed|plate|well) permutation; manifest A
    takes the first 2, manifest B the next 2 -- so A and B never collide and neither
    collides with the current manifest.

Field catalog: fetched from the public cellpainting-gallery load_data CSVs
  cpg0000-jump-pilot/source_4/workspace/load_data_csv/2020_11_04_CPJUMP1/<PLATE>/load_data.csv
(no credentials). Positions (position_x/y) also come from these CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

RUNNERS = Path(__file__).resolve().parent
CURRENT_MANIFEST = RUNNERS / "manifest.json"

LOAD_DATA_URL = (
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0000-jump-pilot/source_4/"
    "workspace/load_data_csv/2020_11_04_CPJUMP1/{plate}/load_data.csv"
)

# s3 image prefix (everything before the per-plate measurement folder)
S3_PREFIX = (
    "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/"
    "2020_11_04_CPJUMP1/images"
)
HTTPS_PREFIX = (
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0000-jump-pilot/source_4/"
    "images/2020_11_04_CPJUMP1/images"
)

SITES_PER_WELL = 2
# channel index -> (channel_name). Matches the frozen manifest and the load_data URLs.
CHANNELS = [
    (1, "Mito"),
    (2, "AGP"),
    (3, "RNA"),
    (4, "ER"),
    (5, "DNA"),
]

SEEDS = {
    "20261009": {"index": 0, "letter": "A"},  # takes fields at shared-perm offset 0
    "20261010": {"index": 1, "letter": "B"},  # takes fields at shared-perm offset 1
}

# Single permutation seed shared by BOTH replica manifests. Both order each well's free
# fields identically with this seed; A and B then take disjoint offset slices, so their
# site selections never collide. (Keying the permutation per-seed would let A and B pick
# overlapping fields -- they must share the ordering.)
PERM_SEED = "20261009+20261010"

_RC_RE = re.compile(r"/(r\d{2}c\d{2})f\d{2}p01-ch")
_MEAS_RE = re.compile(r"/([A-Z0-9]+)__([0-9T:_\-]+-Measurement1)/Images/")


def well_to_rc(well: str) -> str:
    """C01 -> r03c01 (row letter -> 1-based row number, trailing digits -> column)."""
    row = ord(well[0].upper()) - ord("A") + 1
    col = int(well[1:])
    return f"r{row:02d}c{col:02d}"


def fetch_field_catalog(plates):
    """Return (available_fields, positions).

    available_fields[(plate, well)] = sorted list of ints
    positions[(plate, well, site)] = (position_x_float_or_None, position_y_float_or_None)
    """
    available = defaultdict(set)
    positions = {}
    for plate in plates:
        url = LOAD_DATA_URL.format(plate=plate)
        print(f"[catalog] fetching {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=180) as resp:
            text = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            well = row["Metadata_Well"]
            site = int(row["Metadata_Site"])
            available[(plate, well)].add(site)

            def _f(key):
                v = row.get(key)
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            positions[(plate, well, site)] = (
                _f("Metadata_PositionX"),
                _f("Metadata_PositionY"),
            )
    available = {k: sorted(v) for k, v in available.items()}
    return available, positions


def shared_perm(fields, perm_seed, plate, well):
    """Deterministic permutation of `fields` keyed by sha256(perm_seed|plate|well|field).

    Both replica manifests share ONE permutation (same `perm_seed`) so that disjoint
    offset slices never collide. Manifest A takes the first SITES_PER_WELL fields of this
    ordering, manifest B the next SITES_PER_WELL -- guaranteeing A and B are disjoint and
    (because the candidate pool already excludes the v1 fields) both disjoint from v1.
    The permutation is stable and independent of insertion order.
    """
    def key(field):
        return hashlib.sha256(
            f"{perm_seed}|{plate}|{well}|{field}".encode("utf-8")
        ).hexdigest()

    return sorted(fields, key=key)


def build_manifest(current, available, positions, seed, letter):
    """Build a new manifest dict mirroring `current` with replacement fields."""
    # per-plate measurement folder from current files
    meas = {}
    for f in current["files"]:
        m = _MEAS_RE.search(f["s3_uri"])
        plate, folder = m.group(1), m.group(2)
        meas.setdefault(plate, folder)

    # current field set per (plate, well) -- to exclude
    cur_fields = defaultdict(set)
    for s in current["sites"]:
        cur_fields[(s["plate"], s["well"])].add(s["site"])

    # one representative site per well to copy biology metadata (gene/pert/role/...)
    well_meta = {}
    for s in current["sites"]:
        well_meta.setdefault((s["plate"], s["well"]), s)

    # iterate wells in current-manifest order (dedup, preserve order)
    seen = []
    seen_set = set()
    for s in current["sites"]:
        key = (s["plate"], s["well"])
        if key not in seen_set:
            seen_set.add(key)
            seen.append(key)

    new_sites = []
    new_files = []
    offset = SEEDS[seed]["index"] * SITES_PER_WELL  # A: 0, B: 2

    for (plate, well) in seen:
        avail = set(available[(plate, well)])
        excl = cur_fields[(plate, well)]
        candidates = sorted(avail - excl)
        perm = shared_perm(candidates, PERM_SEED, plate, well)
        chosen = perm[offset:offset + SITES_PER_WELL]
        if len(chosen) < SITES_PER_WELL:
            raise RuntimeError(
                f"not enough free fields for {plate}/{well}: avail={sorted(avail)} "
                f"excl={sorted(excl)} need offset {offset}+{SITES_PER_WELL}"
            )
        meta = well_meta[(plate, well)]
        rc = well_to_rc(well)
        folder = meas[plate]
        for field in sorted(chosen):
            site_id = f"{plate}/{well}/{field}"
            px, py = positions.get((plate, well, field), (None, None))
            site = {
                "id": site_id,
                "plate": plate,
                "technical_batch": meta.get("technical_batch", plate),
                "well": well,
                "site": field,
                "sample": meta["sample"],
                "gene": meta["gene"],
                "pert_type": meta["pert_type"],
                "role": meta["role"],
                "usage": meta["usage"],
                "position_x": px,
                "position_y": py,
            }
            new_sites.append(site)
            for ch, ch_name in CHANNELS:
                fname = f"{rc}f{field:02d}p01-ch{ch}sk1fk1fl1.tiff"
                s3 = f"{S3_PREFIX}/{plate}__{folder}/Images/{fname}"
                https = f"{HTTPS_PREFIX}/{plate}__{folder}/Images/{fname}"
                new_files.append({
                    "site_id": site_id,
                    "channel": ch,
                    "channel_name": ch_name,
                    "s3_uri": s3,
                    "https_url": https,
                    "relative_path": f"{plate}/{well}/{fname}",
                })

    # assemble new manifest mirroring current structure
    out = dict(current)  # shallow copy; replace the parts that change
    base_name = re.sub(r"-v1$", "", current["name"])
    out["name"] = f"{base_name}-seed{seed}"
    out["seed"] = seed
    out["supersedes"] = (
        "Independent field-replicate of "
        f"{current['name']} (seed {current['seed']}); historical selections remain "
        "historical only."
    )
    out["selection"] = (
        "Field-replicate of the v1 500-well selection: SAME 500 wells "
        "(same perturbations / genes / guides) as "
        f"{current['name']}, but DIFFERENT microscope fields (sites) per well. "
        "For each well, 2 fields are drawn by a sha256(seed|plate|well) permutation "
        "from the available fields minus the v1 fields; manifest A (seed 20261009) and "
        "manifest B (seed 20261010) take disjoint slices, so this manifest, its sibling "
        "and v1 are pairwise non-overlapping at the site level. Wells, genes, condition "
        "and limits are unchanged from v1."
    )
    out["sites"] = new_sites
    out["files"] = new_files
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(CURRENT_MANIFEST))
    ap.add_argument("--outdir", default=str(RUNNERS))
    args = ap.parse_args()

    current = json.loads(Path(args.manifest).read_text())
    available, positions = fetch_field_catalog(current["plates"])

    outdir = Path(args.outdir)
    written = []
    for seed in SEEDS:
        letter = SEEDS[seed]["letter"]
        man = build_manifest(current, available, positions, seed, letter)
        out_path = outdir / f"manifest_seed{seed}.json"
        out_path.write_text(json.dumps(man, indent=1))
        written.append(out_path)
        print(f"[write] {out_path}  sites={len(man['sites'])} files={len(man['files'])}",
              file=sys.stderr)
    return written


if __name__ == "__main__":
    main()
