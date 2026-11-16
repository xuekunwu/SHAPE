"""Prepared-data loader for the multi-modality probe (ModalityProbe Dataset, pipeline.py): 600 images, 6
groups of 100 (A549/U2OS cell types x compound/crispr/orf perturbations), NOT the pinned CRISPR-only
manifest (data/CPJUMP1/manifest.json, data.py's KNOWN_MANIFEST_SHA256). This is a SEPARATE loader (the
Dataset.loader() extension point, pipeline.py's Dataset.data field) precisely because the CRISPR loader's
load_prepared() hard-gates on manifest_sha256 in KNOWN_MANIFEST_SHA256 -- a genuinely new, non-pinned
selection can never satisfy that by design (it would defeat the point of the allowlist). This loader's own
PREPARE.json has no manifest_sha256 at all; it only checks its own images_sha256 self-consistency, the same
integrity property data.py's load_prepared enforces, without the CRISPR-subset-specific allowlist.

User decision 2026-11-01 (real JUMP CPJUMP1 structure, experiment-metadata.tsv): no fixed numeric time is
shared across all 3 perturbation modalities (compound reads at 24h, CRISPR needs >=96h, ORF has its own
Day0..Week4 schedule) -- each combo uses ITS OWN earliest/Day0 readout as the closest analogue of "fixed
time". Simplifications accepted for this exploratory probe, not hidden:
  - No per-well treatment identity (compound ID / CRISPR guide / ORF construct) was fetched -- only the
    plate-level (cell_type, perturbation) combo is known. `negative_control` is False for every row (no
    genuine negcon/treatment split); scoring instead classifies the 6-way (cell_type, perturbation) label
    (modality_probe_score.py), not gene/compound identity.
  - `calibration_group` is "" for every row (no calibration reference spans two or more plates -- each
    combo is its own single plate, unlike CPJUMP1's 2-plate same-biology replicate design) -- this makes
    the batch stage's own admission rule correctly ABSTAIN (pipeline.py's existing rule, unmodified): never
    try to batch-correct across what are actually 6 different biological conditions, not 6 technical
    replicates of the same one.
"""
from __future__ import annotations

from pathlib import Path

from .data import digest, read_json, within


def load_prepared(prep: Path, *, verify_pixels: bool = False) -> list[dict]:
    record = read_json(prep / "PREPARE.json")
    if record.get("status") != "complete":
        raise ValueError("preparation incomplete")
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
