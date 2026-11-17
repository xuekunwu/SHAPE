#!/usr/bin/env python
"""Add real per-well treatment identity to the ModalityProbe3 private labels -- identical join logic to
fetch_modality_probe2_identities.py (same JUMP-Target-1 platemap/metadata join, same cross-plate twin
consistency check via pair_plate/pair_well), just pointed at probe3's prep dir. Probe3's extra "2 real sites
per well" structure doesn't affect identity lookup at all: identity is a property of (plate, well), not site,
so both sites of a well always get the same identity -- this script doesn't need to check that separately,
only the pre-existing cross-plate twin check (same well position, other plate) still applies.

    cd /path/to/SHAPE
    python benchmarks/full_CPJUMP1/runners/fetch_modality_probe3_identities.py
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, ".")
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json

PREP = Path("data/CPJUMP1_modality_probe3/prepared")
PLATEMAP_URL = ("https://cellpainting-gallery.s3.amazonaws.com/cpg0000-jump-pilot/source_4/workspace/"
                "metadata/platemaps/2020_11_04_CPJUMP1/platemap/JUMP-Target-1_{kind}_platemap.txt")
METADATA_URL = ("https://raw.githubusercontent.com/jump-cellpainting/JUMP-Target/master/"
                "JUMP-Target-1_{kind}_metadata.tsv")
KIND_OF_PERTURBATION = {"crispr": "crispr", "orf": "orf", "compound": "compound"}


def fetch_tsv(url: str) -> list[dict]:
    text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def identity_by_well(kind: str) -> dict[str, dict]:
    platemap = fetch_tsv(PLATEMAP_URL.format(kind=kind))
    meta = {r["broad_sample"]: r for r in fetch_tsv(METADATA_URL.format(kind=kind))}
    out = {}
    for row in platemap:
        bs = row["broad_sample"]
        m = meta.get(bs, {})
        if kind in ("crispr", "orf"):
            out[row["well_position"]] = {"broad_sample": bs, "gene": m.get("gene") or ""}
        else:
            target = (m.get("target_list") or "").split("|")[0]
            out[row["well_position"]] = {"broad_sample": bs, "compound": m.get("pert_iname") or "",
                                         "compound_target": target}
    return out


def main() -> None:
    private = read_json(PREP / "private" / "labels.json")
    by_kind = {k: identity_by_well(k) for k in ("crispr", "orf", "compound")}

    counts = {}
    for row in private:
        kind = KIND_OF_PERTURBATION[row["perturbation"]]
        entry = by_kind[kind].get(row["well"], {})
        if kind in ("crispr", "orf"):
            row["gene"] = entry.get("gene", "")
            row["guide"] = entry.get("broad_sample", "")
        else:
            row["compound"] = entry.get("compound", "")
            row["compound_target"] = entry.get("compound_target", "")
            row["broad_sample"] = entry.get("broad_sample", "")
        counts.setdefault(row["combo"], [0, 0])[0] += 1
        if entry.get("gene") or entry.get("compound"):
            counts[row["combo"]][1] += 1

    mismatches = []
    by_plate_well = {}
    for r in private:
        by_plate_well.setdefault((r["plate"], r["well"]), []).append(r)
    for row in private:
        siblings = [r for r in by_plate_well.get((row["plate"], row["well"]), []) if r["image_id"] != row["image_id"]]
        for sib in siblings:
            own = row.get("gene") or row.get("compound")
            theirs = sib.get("gene") or sib.get("compound")
            if own != theirs:
                mismatches.append((row["image_id"], sib["image_id"], f"same-well sibling identity mismatch: {own!r} vs {theirs!r}"))

    pair_mismatches = []
    pair_lookup = {}
    for r in private:
        pair_lookup.setdefault((r["pair_plate"], r["well"]), []).append(r)
    for row in private:
        for pair in pair_lookup.get((row["plate"], row["well"]), []):
            if pair["pair_plate"] != row["plate"]:
                continue
            own = row.get("gene") or row.get("compound")
            theirs = pair.get("gene") or pair.get("compound")
            if own != theirs:
                pair_mismatches.append((row["image_id"], pair["image_id"], f"cross-plate twin identity mismatch: {own!r} vs {theirs!r}"))

    write_json(PREP / "private" / "labels.json", private)
    record = read_json(PREP / "PREPARE.json")
    record["labels_sha256"] = digest(PREP / "private" / "labels.json")
    write_json(PREP / "PREPARE.json", record)

    print("done -- identity coverage per combo (n rows, n with a real identity):")
    for combo, (n, hit) in sorted(counts.items()):
        print(f"  {combo}: {hit}/{n}")

    if mismatches:
        print(f"\nWARNING: {len(mismatches)} same-well (2-site) identity mismatches:")
        for a, b, reason in mismatches[:20]:
            print(f"  {a} / {b}: {reason}")
    else:
        print("\nsame-well sibling check: both sites of every well share the same identity, as expected.")

    if pair_mismatches:
        print(f"\nWARNING: {len(pair_mismatches)} cross-plate twin identity mismatches:")
        for a, b, reason in pair_mismatches[:20]:
            print(f"  {a} / {b}: {reason}")
    else:
        print("\ncross-plate twin check: every well's identity matches its pair_well's on the other plate.")


if __name__ == "__main__":
    main()
