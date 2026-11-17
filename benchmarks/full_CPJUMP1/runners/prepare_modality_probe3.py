#!/usr/bin/env python
"""Build the ModalityProbe3 prep dir from the downloaded raw tiffs (download_modality_probe3.py's
data/CPJUMP1_modality_probe3/{raw/,manifest.json}): 6 groups of 100 (2 plates x 25 shared well positions x 2
REAL sites per (plate, well)), assembled via data.py's assemble_view() (unchanged, same as probe1/probe2).

The well-naming convention ("well": f"{plate}_{well}") is the SAME one probe1/probe2 already use, and it is
also the SAME one original CPJUMP1 uses -- unlike probe2 (1 site per (plate, well), so this string was
already globally unique), here it is shared by exactly the 2 real sites of a (plate, well), which is exactly
what shape/tools/cell_state_analyzer/quant.py's sibling_site_map (and tier_of's "replicate" tier check)
needs: same well string, 2 distinct site values. No code change to that shared SHAPE infrastructure is
needed -- this dataset just finally has the structure it was designed for.

Cross-plate identity twins (pair_plate/pair_well, probe2's design) are preserved at the (plate, well) level,
aggregated over its 2 real sites via sites_per_well=2 (CPJUMP1's own default -- MODALITY_PROBE3 does not need
probe1/probe2's sites_per_well=1 override).

usage split: 70/15/15 deterministic per (plate, well) (sha256 terciles, same rule as probe1/probe2) -- both
sites of a (plate, well) always land on the same side, since the split key doesn't include site.

    cd /path/to/SHAPE
    python benchmarks/full_CPJUMP1/runners/prepare_modality_probe3.py \
        --raw data/CPJUMP1_modality_probe3 --out data/CPJUMP1_modality_probe3/prepared
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, ".")
from benchmarks.full_CPJUMP1.harness.data import CHANNELS, assemble_view, digest, write_json

CHANNEL_ORDER = [name for _, name in [(1, "Mito"), (2, "AGP"), (3, "RNA"), (4, "ER"), (5, "DNA")]]


def usage_of(plate: str, well: str) -> str:
    """15/10/75 train/val/held-out (user decision 2026-11-04: stress-test SHAPE's SSL under far less
    training data and a much bigger held-out set than the original 70/15/15; verified first that every
    plate keeps >= 1 val well at this ratio -- a straight 15/5/80 split left 2 plates with zero)."""
    h = int(hashlib.sha256(f"{plate}|{well}".encode()).hexdigest(), 16)
    frac = (h % 1_000_000) / 1_000_000
    if frac < 0.15:
        return "development_train"
    if frac < 0.25:
        return "development_val"
    return "evaluation_only"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/CPJUMP1_modality_probe3"))
    ap.add_argument("--out", type=Path, default=Path("data/CPJUMP1_modality_probe3/prepared"))
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(f"{a.out} already exists; use a fresh output")

    manifest = json.loads((a.raw / "manifest.json").read_text())
    a.out.mkdir(parents=True)

    public, private = [], []
    i = 0
    for label, group in manifest["groups"].items():
        plate_a, plate_b = group["plates"]
        pair_plate = {plate_a: plate_b, plate_b: plate_a}
        for view in group["views"]:
            plate, well, site = view["plate"], view["well"], view["site"]
            paths = [a.raw / "raw" / view["files"][name] for name in CHANNEL_ORDER]
            if any(not p.is_file() for p in paths):
                raise FileNotFoundError(f"missing channel file(s) for {label} {plate} {well} s{site}")
            image = f"img_{i:04d}"; i += 1
            image_file, segment_file = f"images/{image}.ome.tif", f"segmentation/{image}.ome.tif"
            facts = assemble_view(paths, a.out / "public" / image_file, a.out / "public" / segment_file)
            if facts["shape"] != [5, 1080, 1080]:
                raise ValueError(f"unexpected raw dimensions: {label} {plate} {well} s{site} {facts['shape']}")
            public.append({"image_id": image, "file": image_file, "segment_file": segment_file,
                           "plate": plate, "well": f"{plate}_{well}", "site": image,
                           "usage": usage_of(plate, well), "calibration_group": "", "negative_control": False,
                           **facts})
            private.append({"image_id": image, "cell_type": group["cell_type"],
                            "perturbation": group["perturbation"], "time_h": group["time_h"],
                            "plate": plate, "well": well, "site": site, "combo": label,
                            "pair_plate": pair_plate[plate], "pair_well": f"{pair_plate[plate]}_{well}"})
        print(f"[prepare] {label}: {len(group['views'])} views assembled", flush=True)

    write_json(a.out / "public" / "images.json", public)
    write_json(a.out / "private" / "labels.json", private)
    write_json(a.out / "PREPARE.json", {
        "status": "complete", "dataset": "ModalityProbe3",
        "images_sha256": digest(a.out / "public" / "images.json"),
        "labels_sha256": digest(a.out / "private" / "labels.json"),
        "channels": CHANNELS, "segmentation_channels": ["DNA", "AGP"],
        "groups": {k: {"cell_type": v["cell_type"], "perturbation": v["perturbation"], "time_h": v["time_h"],
                      "plates": v["plates"], "n_views": len(v["views"])} for k, v in manifest["groups"].items()},
    })
    usages = {}
    for r in public:
        usages[r["usage"]] = usages.get(r["usage"], 0) + 1
    print(f"\ndone: {len(public)} images -> {a.out}")
    print(f"usage split: {usages}")

    # structural self-check: every well string must be shared by exactly 2 images (the 2 real sites),
    # and distinct across plates/combos -- otherwise sibling_site_map would be back to square one.
    from collections import Counter
    counts = Counter(r["well"] for r in public)
    bad = {w: n for w, n in counts.items() if n != 2}
    if bad:
        raise ValueError(f"expected exactly 2 images per well string, got: {bad}")
    print("structural check passed: every well string shared by exactly 2 images (2 real sites)")


if __name__ == "__main__":
    main()
