#!/usr/bin/env python3
"""Validate a local NeurIPS22-CellSeg dataset root before running the benchmark.

No download/convert step: this benchmark expects the data already laid out as

    <data-root>/images/<name>.{png,bmp,tif,tiff,jpg,jpeg}
    <data-root>/labels/<name>_label.tiff

(the repo's own copy lives at data/NeurIPS22-CellSeg/, the convention cellseg_config.py and every
runner default to via $SHAPE_BENCH_DATA_ROOT). This script checks that layout is intact -- every
image has exactly one matching label, no orphans either way -- and that the difficulty-band reference
CSV (NeurIPS22-CellSeg-cell_image_metrics.csv, the Cellpose-SAM-alone per-image AP50/AP75/mAP
run_cellseg_benchmark.py reads to split hard/medium/easy) is present, either in the data root (it
ships with the Hugging Face copy) or in this benchmark's results/, telling you the exact command to
(re)generate it with run_cellpose_sam_only.py if it is not.

    python benchmarks/segmentation_NeurIPS22-CellSeg/runners/prepare_dataset.py [--data-root PATH]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

IMAGE_EXTS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")
LABEL_SUFFIX, LABEL_EXT = "_label", ".tiff"
METRICS_CSV = Path(__file__).resolve().parents[1] / "results" / "NeurIPS22-CellSeg-cell_image_metrics.csv"


def default_data_root() -> Path | None:
    root = os.environ.get("SHAPE_BENCH_DATA_ROOT")
    return Path(root) / "NeurIPS22-CellSeg" if root else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=default_data_root(),
                    help="Defaults to $SHAPE_BENCH_DATA_ROOT/NeurIPS22-CellSeg, else the repo's own "
                         "data/NeurIPS22-CellSeg.")
    args = ap.parse_args()
    data_root = args.data_root or Path(__file__).resolve().parents[3] / "data" / "NeurIPS22-CellSeg"
    image_dir, label_dir = data_root / "images", data_root / "labels"

    problems = []
    if not image_dir.is_dir():
        problems.append(f"no images/ under {data_root}")
    if not label_dir.is_dir():
        problems.append(f"no labels/ under {data_root}")
    if problems:
        raise SystemExit("dataset not found:\n  " + "\n  ".join(problems))

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    labels = {p.name for p in label_dir.iterdir() if p.suffix.lower() == LABEL_EXT.lower()}
    missing = [img.name for img in images if f"{img.stem}{LABEL_SUFFIX}{LABEL_EXT}" not in labels]
    expected = {f"{img.stem}{LABEL_SUFFIX}{LABEL_EXT}" for img in images}
    orphan_labels = sorted(labels - expected)

    print(f"data root: {data_root}")
    print(f"images: {len(images)}  labels: {len(labels)}")
    if missing:
        raise SystemExit(f"{len(missing)} image(s) have no matching label, e.g. {missing[:5]}")
    if orphan_labels:
        print(f"WARNING: {len(orphan_labels)} label(s) have no matching image, e.g. {orphan_labels[:5]}")
    print(f"OK: every image has exactly one matching {LABEL_SUFFIX}{LABEL_EXT} label")

    if (data_root / METRICS_CSV.name).is_file():
        print(f"difficulty-band reference: {data_root / METRICS_CSV.name} (present)")
    elif METRICS_CSV.is_file():
        print(f"difficulty-band reference: {METRICS_CSV} (present)")
    else:
        print(f"difficulty-band reference MISSING: {METRICS_CSV}\n"
              f"  regenerate with:\n"
              f"    .venv/bin/python benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_cellpose_sam_only.py "
              f"--data-root {data_root} --out-csv {METRICS_CSV}")


if __name__ == "__main__":
    main()
