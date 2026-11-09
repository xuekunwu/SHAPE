"""Prepare stage (CPU, once per cell type): the task-116 v3 preps as Cell_State_Analyzer_Tool inputs.

Reads the FULL prep (images.npy, uint8 (N, 6, 64, 64); cells.csv: row, cell_id, site_id, well_unit, sirna, plate,
experiment, split in train | val | heldout) and its run prep (cells.csv with source_row, the cell's FULL row, and
split in train | val). Writes under <root> = <output root>/<cell type> (Lustre) -- no crop file:

  prep/cells.csv                     one row per FULL cell, COLUMNS -- no label; `array_row` = the cell's row of the
                                     FULL images.npy
  prep/PREPARE.json                  the sources (paths, sha256), counts per split, the tool input, the split rule and
                                     `crop_array` (the FULL images.npy: absolute path, sha256, size, dtype, shape)
  private/labels.csv                 cell_id, sirna: read by score.py only

The pixels stay in the task-116 images.npy: the tool reads them as its packed-array input (`crop_array`, contract
amendment A1), one `array_row` per metadata row (`tool_input`); `load` refuses a changed images.npy: its size every
time, its sha256 against PREPARE.json once per file state (`verify_crop_array`: the stamp
prep/CROP_ARRAY_VERIFIED.json holds the path, the sha256 and the file's size, mtime, ctime and inode after a full hash
that matched; a load whose file state equals the stamp's skips the hash, any other state hashes the whole file again).

split: train = the run prep's train cells (every cell of its training wells), val = the val wells (every cell of them
is in the run prep), unused = the FULL train-well cells outside the run prep, heldout = the held-out experiments.
The tool sees the run prep only (`tool_input`: train + val). Each part is written into "<dir>.partial" and renamed
when complete; an existing prep is never overwritten.

`build_assets` stages the pinned DINOv3 code and weights for offline loading (full_HCMA's helpers); `check_assets`
returns the tool environment that points the analyzer at them.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from benchmarks.representation_RxRx1.runners.control import fs_retry

# the tool-facing table of every FULL cell (prep/cells.csv); `array_row` is the cell's row of the FULL images.npy
COLUMNS = ("cell_id", "crop_id", "image_name", "group", "plate", "experiment", "well", "site", "split", "array_row")
TOOL_KEYS = ("cell_id", "crop_id", "image_name", "group", "plate", "experiment", "well", "site", "split")
TOOL_SPLITS = ("train", "val")                  # the run prep: the only crops the tool sees during a run
SPLITS = ("train", "val", "unused", "heldout")
SOURCE_COLUMNS = ("row", "cell_id", "site_id", "well_unit", "sirna", "plate", "experiment", "split")
CROP_SHAPE = (6, 64, 64)
CROP_FORMAT = "the rows of the FULL images.npy (uint8, N x 6 x 64 x 64, CHW), the analyzer's crop_array; no crop file"
REPO = Path(__file__).resolve().parents[3]


def outside_repo(path: Path) -> Path:
    """`path`, after checking that it is not inside the repository: crops, arrays and runs go to scratch."""
    resolved = Path(path).resolve()
    if resolved == REPO or REPO in resolved.parents:
        raise SystemExit(f"NOT_READY: {path} is inside the repository {REPO}; outputs go to scratch")
    return Path(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_source(prep: Path, *extra: str):
    import pandas as pd

    cells = pd.read_csv(prep / "cells.csv", dtype=str, keep_default_na=False)
    missing = [c for c in (*SOURCE_COLUMNS, *extra) if c not in cells]
    if missing:
        raise SystemExit(f"NOT_A_116_PREP: {prep}/cells.csv lacks {missing}")
    return cells


def splits_of(full, run) -> np.ndarray:
    """The split (SPLITS) of every FULL cell, after checking the run prep against the FULL prep."""
    split = full.split.to_numpy().astype(object)
    unknown = sorted(set(split) - {"train", "val", "heldout"}) + sorted(set(run.split) - {"train", "val"})
    if unknown:
        raise SystemExit(f"NOT_A_116_PREP: unknown split values {unknown}")
    source, run_split = run.source_row.astype(int).to_numpy(), run.split.to_numpy()
    if (full.cell_id.to_numpy()[source] != run.cell_id.to_numpy()).any() or (split[source] != run_split).any():
        raise SystemExit("PREP_MISMATCH: the run prep's cells differ from their FULL rows (cell_id or split)")
    if not np.array_equal(np.sort(source[run_split == "val"]), np.flatnonzero(split == "val")):
        raise SystemExit("PREP_MISMATCH: the run prep does not hold exactly the val cells of the FULL prep")
    trained = np.zeros(len(full), bool)
    trained[source[run_split == "train"]] = True
    out = split.copy()
    out[(split == "train") & ~trained] = "unused"
    wells = full.well_unit.to_numpy()
    if set(wells[out == "train"]) & set(wells[out == "unused"]):
        raise SystemExit("PREP_MISMATCH: a training well of the run prep holds only part of its cells")
    return out


def _counts(table) -> dict:
    return {"cells": int(len(table)), "sites": int(table.site.nunique()), "wells": int(table.well.nunique()),
            "plates": int(table.plate.nunique()), "experiments": int(table.experiment.nunique())}


def prepare(full: Path, run_prep: Path, root: Path) -> dict:
    import pandas as pd

    targets = (outside_repo(root) / "prep", root / "private")
    for target in targets:
        if target.exists():
            raise SystemExit(f"{target} exists; earlier evidence is preserved")
    cells, run = _read_source(full), _read_source(run_prep, "source_row")
    images = np.load(full / "images.npy", mmap_mode="r")
    if images.dtype != np.uint8 or images.shape[1:] != CROP_SHAPE or len(images) != len(cells):
        raise SystemExit(f"NOT_A_116_PREP: images.npy is {images.dtype} {images.shape}, expected uint8 "
                         f"({len(cells)}, {', '.join(map(str, CROP_SHAPE))})")
    if not np.array_equal(cells.row.astype(int).to_numpy(), np.arange(len(cells))) or cells.cell_id.duplicated().any():
        raise SystemExit("NOT_A_116_PREP: cells.csv rows are not 0..N-1 or its cell ids repeat")
    if not all(w.startswith(f"{p}/") for w, p in zip(cells.well_unit, cells.plate)):
        raise SystemExit("NOT_A_116_PREP: well_unit is not plate-qualified (<plate>/<well>)")
    meta = pd.DataFrame({"cell_id": cells.cell_id, "crop_id": cells.cell_id, "image_name": cells.site_id,
                         "group": "default", "plate": cells.plate, "experiment": cells.experiment,
                         "well": cells.well_unit, "site": cells.site_id, "split": splits_of(cells, run),
                         "array_row": np.arange(len(cells))})
    started = time.time()
    partial = {t: t.with_name(t.name + ".partial") for t in targets}
    for directory in partial.values():
        if directory.exists():
            fs_retry(shutil.rmtree, directory)
        fs_retry(directory.mkdir, parents=True, exist_ok=True)
    prep = partial[root / "prep"]
    fs_retry(meta.to_csv, prep / "cells.csv", index=False)
    source = (full / "images.npy").absolute()
    crop_array = {"path": str(source), "sha256": sha256_file(source), "bytes": source.stat().st_size,
                  "dtype": str(images.dtype), "shape": list(images.shape)}
    labels = partial[root / "private"] / "labels.csv"
    fs_retry(pd.DataFrame({"cell_id": cells.cell_id, "sirna": cells.sirna}).to_csv, labels, index=False)
    sources = {name: json.loads((path / "PREPARE.json").read_text(encoding="utf-8"))
               if (path / "PREPARE.json").exists() else {} for name, path in (("full", full), ("run", run_prep))}
    tool = meta[meta.split.isin(TOOL_SPLITS)]
    summary = {
        "sources": {
            "full": {"path": str(full), "cells_sha256": sha256_file(full / "cells.csv"),
                     "images_sha256": crop_array["sha256"],
                     **{k: sources["full"].get(k) for k in ("experiments", "val_fraction", "holdout_experiments",
                                                             "split_design")}},
            "run": {"path": str(run_prep), "cells_sha256": sha256_file(run_prep / "cells.csv"),
                    "subsample": sources["run"].get("subsample")}},
        "n_cells": int(len(meta)),
        "splits": {s: _counts(meta[meta.split == s]) for s in SPLITS},
        "tool_input": {"splits": list(TOOL_SPLITS), **_counts(tool),
                       "train_cells": int((tool.split == "train").sum()), "val_cells": int((tool.split == "val").sum()),
                       "min_val_wells_per_plate": int(tool[tool.split == "val"].groupby("plate").well.nunique().min())
                       if (tool.split == "val").any() else 0},
        "crop_format": CROP_FORMAT,
        "crop_array": crop_array,
        "cells_sha256": sha256_file(prep / "cells.csv"), "labels_sha256": sha256_file(labels),
        "split_rule": "train = the run prep's train cells; val = the val wells (all in the run prep); unused = the "
                      "FULL train-well cells outside the run prep; heldout = the held-out experiments; the tool "
                      "sees train + val",
        "seconds": round(time.time() - started, 1),
    }
    fs_retry((prep / "PREPARE.json").write_text, json.dumps(summary, indent=1, default=str), encoding="utf-8")
    for target in targets:
        fs_retry(partial[target].rename, target)
    print(json.dumps({k: summary[k] for k in ("n_cells", "splits", "tool_input")}))
    return summary


def load(root: Path):
    """(PREPARE.json, cells.csv as strings) of the prep under `root`; refuses a changed table or one with other
    columns (no label ever enters it)."""
    import pandas as pd

    prep = root / "prep"
    if not (prep / "PREPARE.json").exists():
        raise SystemExit(f"NO_PREP: {prep}/PREPARE.json is missing (run the prepare stage)")
    summary = json.loads((prep / "PREPARE.json").read_text(encoding="utf-8"))
    if "crop_array" not in summary:
        raise SystemExit(f"NOT_CURRENT: {prep} is a prep of per-cell crop files; move {prep} and "
                         f"{root / 'private'} aside and run the prepare stage again")
    if sha256_file(prep / "cells.csv") != summary["cells_sha256"]:
        raise SystemExit(f"INTEGRITY_FAILED: {prep}/cells.csv changed since the prepare stage")
    cells = pd.read_csv(prep / "cells.csv", dtype=str, keep_default_na=False)
    if tuple(cells.columns) != COLUMNS:
        raise SystemExit(f"INTEGRITY_FAILED: {prep}/cells.csv has the columns {list(cells.columns)}, not {list(COLUMNS)}")
    verify_crop_array(prep, summary["crop_array"])
    return summary, cells


VERIFIED = "CROP_ARRAY_VERIFIED.json"     # prep/: the file state of images.npy at its last full sha256 match


def _file_state(path: Path) -> dict:
    """What a write to the file changes: its size, mtime and ctime (ns) and inode."""
    st = Path(path).stat()
    return {"bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns), "ctime_ns": int(st.st_ctime_ns),
            "inode": int(st.st_ino)}


def verify_crop_array(prep: Path, source: dict) -> None:
    """SystemExit SOURCE_CHANGED unless the images.npy of PREPARE.json `crop_array` (`source`) exists with its size
    and sha256. The full sha256 (a minute for a 35 GB array) runs once per file state: after a match the stamp
    prep/VERIFIED records the path, the sha256 and `_file_state`; a later call whose state equals the stamp's skips
    the hash, any other state (a rewrite, a copy, a touch, another path or sha256) hashes again. The stamp is written
    only when the state did not change during the hash; a stamp that cannot be written costs the next call a hash."""
    path = Path(source["path"])
    if not path.exists() or path.stat().st_size != source["bytes"]:
        raise SystemExit(f"SOURCE_CHANGED: {path} is missing or not the images.npy of the prepare stage")
    stamp = prep / VERIFIED
    before = {"path": str(path), "sha256": source["sha256"], **_file_state(path)}
    try:
        if json.loads(stamp.read_text(encoding="utf-8")) == before:
            return
    except (OSError, ValueError):                # no stamp, or an unreadable one: hash
        pass
    if sha256_file(path) != source["sha256"]:
        raise SystemExit(f"SOURCE_CHANGED: {path} is missing or not the images.npy of the prepare stage")
    if {"path": str(path), "sha256": source["sha256"], **_file_state(path)} == before:
        tmp = stamp.with_name(f"{stamp.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(before, indent=1), encoding="utf-8")
            tmp.replace(stamp)
        except OSError:
            pass


def tool_input(summary: dict, cells, splits=TOOL_SPLITS, keys=TOOL_KEYS) -> tuple[str, list[dict]]:
    """The analyzer's inputs for the cells of `splits`: the packed array (`crop_array`, the FULL images.npy of
    PREPARE.json) and one metadata row per crop (`keys` and its integer `array_row`)."""
    rows = cells[cells.split.isin(splits)]
    meta = rows[list(keys)].to_dict("records")
    for m, row in zip(meta, rows.array_row):
        m["array_row"] = int(row)
    return summary["crop_array"]["path"], meta


def build_assets(output_root: Path, *, dinov3_repo: Path, weights: Path) -> Path:
    """<output_root>/assets: the pinned DINOv3 checkout as torch.hub's cache of facebookresearch/dinov3 and the pinned
    weights in the Hugging Face cache layout (full_HCMA's helpers, without its Cellpose weights), ASSETS.json with
    every path and sha256. Refuses to overwrite; returns ASSETS.json. Every filesystem step retries a transient
    failure (`fs_retry`); ASSETS.json is written before the rename, so assets/ never exists without it."""
    from benchmarks.full_HCMA.harness import assets

    root = Path(output_root).absolute()
    final = root / "assets"
    if final.exists():
        raise SystemExit(f"NOT_READY: {final} exists; earlier evidence is preserved")
    partial = root / "assets.partial"
    if partial.exists():
        fs_retry(shutil.rmtree, partial)
    hub = partial / "torch" / "hub"
    dinov3 = fs_retry(assets.copy_dinov3, Path(dinov3_repo), assets.DINOV3_COMMIT, hub / assets.HUB_DIR_NAME)
    # torch.hub records trusted repos here; present up front, so loading never needs to write it
    fs_retry((hub / "trusted_list").write_text, "facebookresearch_dinov3\n", encoding="utf-8")
    hf = fs_retry(assets.place_hf_weights, Path(weights), partial / "hf", assets.WEIGHTS_SHA256)
    assets.check_hf_offline(partial / "hf", Path(hf["path"]))
    for record in (dinov3, hf):                  # paths as they are after the rename
        record["path"] = record["path"].replace(str(partial), str(final), 1)
    fs_retry((partial / "ASSETS.json").write_text,
             json.dumps({"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "env": assets.tool_env(root), "dinov3_hub": dinov3, "dinov3_weights": hf}, indent=1),
             encoding="utf-8")
    fs_retry(partial.rename, final)
    path = final / "ASSETS.json"
    print(f"ASSETS {path}: DINOv3 {assets.DINOV3_COMMIT[:8]} ({dinov3['n_files']} files), weights {hf['sha256'][:12]}")
    return path


def check_assets(output_root: Path) -> dict[str, str]:
    """The tool environment (TORCH_HOME, HF_HOME, HF_HUB_OFFLINE=1, ...) of the pinned assets, after checking that the
    copies the analyzer loads still match ASSETS.json. Set it before the analyzer is imported."""
    from benchmarks.full_HCMA.harness import assets

    path = Path(output_root) / "assets" / "ASSETS.json"
    if not path.exists():
        raise SystemExit(f"NOT_READY: {path} is missing (run the prepare stage with --dinov3-repo and --weights)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    hub, weights = manifest["dinov3_hub"], manifest["dinov3_weights"]
    if (hub["commit"] != assets.DINOV3_COMMIT or assets.tree_digest(Path(hub["path"]))[0] != hub["tree_sha256"]
            or sha256_file(Path(weights["path"])) != weights["sha256"]):
        raise SystemExit(f"NOT_READY: the pinned DINOv3 assets under {path.parent} changed since they were built")
    return assets.tool_env(Path(output_root))
