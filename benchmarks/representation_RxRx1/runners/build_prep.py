"""Build RxRx1 preps (CPU): the FULL prep of one cell type and its run prep, in the task-116 v3 format.

  python -m benchmarks.representation_RxRx1.runners.build_prep prepare   --metadata M --images Z --output FULL [...]
  python -m benchmarks.representation_RxRx1.runners.build_prep subsample --prep FULL --output RUN
                                                                  (--train-well-fraction T | --train-cells K)
  python -m benchmarks.representation_RxRx1.runners.build_prep resplit   --source FULL --output FULL2 --val-fraction V
                                                                  --holdout-experiments N
  python -m benchmarks.representation_RxRx1.runners.build_prep resplit   --source FULL --output FULL2 --split wells
                                                                  --seen-fraction S --val-of-seen F

A port of the task-116 harness's prep builder (SSL repository, `agentic_rxrx1/harness/data.py` and the prepare /
resplit / subsample commands of `harness/cli.py`), so that the preps this benchmark reads
(config.py PREPS, PREP_SETS) are built from this repository. Self-contained: numpy, pandas, scipy, scikit-image and
pillow; nothing of SHAPE.

Selection, nucleus detection, per-site stretch and crop geometry are the harness's (the defaults: HUVEC, train split,
first 4 experiments, positive-control wells, <= 24 nucleus-centred 64 x 64 crops per site, 6 channels). The seeds and
the sha256 rank strings are the harness's, so the same options give the same prep. The split reads no label:
  heldout  every cell of `--holdout-experiments` N experiments (seeded sha256 order);
  val      per plate, the first max(2, round(val_fraction * n_wells)) wells of a seeded sha256 order of its wells;
  train    the rest.
`resplit --split wells` (the full_v5 design, WELL_SPLIT_RULE) instead splits every plate's wells, of every
experiment, into seen (train + val) and held-out wells; no experiment is held out. Units are whole wells, never cells.

Outputs (PREP dir), the format prepare.py reads:
  images.npy   uint8 (N, 6, 64, 64)
  cells.csv    PRIVATE (it holds the siRNA): row, cell_id, site_id, well_unit, sirna, plate, experiment, split
               (a run prep also source_row, the cell's row of the FULL prep, after split)
  PREPARE.json summary (split_design, val_fraction, holdout_experiments, cells_per_split, wells_per_split, ...)

`prepare --workers N` crops N contiguous shards of the sites in parallel (spawned processes) and streams them into
images.npy; the output is the same for every N. `--well-types all` takes every well_type of metadata.csv.
`resplit` gives an already prepared full prep a new split (images.npy hard-linked, a copy where linking fails; never
re-read); `prepare` has the experiment-level split only (a full_v5 prep is a `resplit --split wells` of a full_v4 one).
`subsample` derives the run prep from a full prep: ALL cells of a seeded `train_well_fraction` of every plate's TRAIN
wells (the training wells) plus ALL cells of the VAL wells; the other train wells and the held-out experiments are
not in it (`--train-well-fraction 1.0`: every train well, no train cell left out; the full_v5 run prep).
`--train-cells K` instead resolves the fraction that gives about K training cells (TRAIN_CELLS_RULE).
Every command writes "<output>.partial" and renames it to the output when complete; an existing output is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import time
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

CHANNELS = ("hoechst", "concanavalin_a", "phalloidin", "syto14", "mitotracker", "wga")
SPLITS = ("train", "val", "heldout")
SPLIT_MODES = ("experiments", "wells")   # resplit --split
MIN_VAL_WELLS = 2
CELL_COLUMNS = ("cell_id", "site_id", "well_unit", "sirna", "plate", "experiment")
COPY_CHUNK_BYTES = 1 << 24


def _rank(*parts) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- the crops

class SiteReader:
    """The 6 channel PNGs of a site, from the RxRx1 zip (rxrx1/images/...) or an unpacked images directory."""

    def __init__(self, images: str | Path):
        self.path = Path(images)
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.zip is None:
            for candidate in (self.path / "rxrx1" / "images", self.path / "images", self.path):
                if candidate.is_dir() and any(candidate.iterdir()):
                    self.root = candidate
                    break
            else:
                raise FileNotFoundError(f"no RxRx1 images under {self.path}")

    def channel(self, experiment, plate, well, site, channel) -> np.ndarray:
        from PIL import Image

        name = f"{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png"
        content = (self.zip.read(f"rxrx1/images/{name}") if self.zip is not None
                   else (self.root / name).read_bytes())
        with Image.open(io.BytesIO(content)) as image:
            array = np.array(image)
        if array.dtype != np.uint8:
            raise ValueError(f"expected 8-bit PNG, got {array.dtype} for {name}")
        return array

    def site(self, row) -> np.ndarray:
        return np.stack([self.channel(row.experiment, int(row.plate), row.well, int(row.site), c)
                         for c in range(1, len(CHANNELS) + 1)])


def stretch_site(image: np.ndarray, low: float = 0.1, high: float = 99.9) -> np.ndarray:
    out = np.empty_like(image)
    for c in range(image.shape[0]):
        data = image[c].astype(np.float32)
        lo, hi = np.percentile(data, (low, high))
        scaled = (data - lo) / (hi - lo) if hi > lo else np.zeros_like(data)
        out[c] = np.round(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


def nucleus_centroids(hoechst, *, min_area, max_area, half):
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_opening, disk

    smooth = ndimage.gaussian_filter(hoechst.astype(np.float32), 2.0)
    if smooth.max() <= smooth.min():
        return []
    mask = binary_opening(smooth > threshold_otsu(smooth), disk(2))
    height, width = hoechst.shape
    out = []
    for region in regionprops(label(mask)):
        if not min_area <= region.area <= max_area:
            continue
        r, c = (int(round(v)) for v in region.centroid)
        if half <= r <= height - half and half <= c <= width - half:
            out.append((int(region.label), r, c))
    return out


# ---------------------------------------------------------------- the split

def holdout_experiments(experiments: list[str], n: int, seed: int = 0) -> list[str]:
    """n experiments chosen by a seeded sha256 order (no bias toward early / late batches)."""
    return sorted(sorted(experiments, key=lambda e: _rank(seed, "holdout", e))[:n])


def assign_splits(well_unit: list[str], plate: list[str], experiment: list[str], *, val_fraction: float,
                  holdout_experiments: list[str], seed: int = 0) -> list[str]:
    """The split of every cell: `heldout` for every cell of `holdout_experiments`; among the other wells, per plate,
    the first max(MIN_VAL_WELLS, round(val_fraction * n_wells)) of a seeded sha256 order of the plate's wells are
    `val`, the rest `train`. No label is read."""
    if not 0 <= val_fraction < 1:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    held = set(holdout_experiments)
    wells_of_plate: dict[str, set[str]] = {}
    for unit, p, e in zip(well_unit, plate, experiment):
        if e not in held:
            wells_of_plate.setdefault(p, set()).add(unit)
    split_of: dict[str, str] = {}
    for p, members in wells_of_plate.items():
        ordered = sorted(members, key=lambda u: _rank(seed, "val", p, u))
        n_val = max(MIN_VAL_WELLS, round(val_fraction * len(ordered)))
        if len(ordered) - n_val < 1:
            raise ValueError(f"plate {p} has too few wells ({len(ordered)}) for {n_val} val wells and a train well")
        for position, unit in enumerate(ordered):
            split_of[unit] = "val" if position < n_val else "train"
    return ["heldout" if e in held else split_of[u] for u, e in zip(well_unit, experiment)]


def split_design(val_fraction: float, holdout: list[str]) -> str:
    """PREPARE.json `split_design`."""
    return (f"wells stratified by plate: val {val_fraction:g} per plate (>= {MIN_VAL_WELLS}); held-out experiments "
            f"{', '.join(holdout) if holdout else 'none'}; train = the rest")


WELL_SPLIT_RULE = ("per plate, its n wells (the plate's well_units in cells.csv: every experiment, every well type) "
                   "in a seeded sha256 order; n_seen = floor(seen_fraction * n + 1/2) and n_val = max(2, "
                   "floor(val_of_seen * n_seen + 1/2)), both exact on the decimal value of the option (round half "
                   "up); the first n_val wells of the order are val, the next n_seen - n_val train (at least 1), the "
                   "other n - n_seen heldout; no experiment is held out")


def _round_half_up(fraction: float, n: int) -> int:
    """floor(fraction * n + 1/2), exact on the decimal value of `fraction` (no binary floating-point product)."""
    return math.floor(Fraction(str(fraction)) * n + Fraction(1, 2))


def assign_well_splits(well_unit: list[str], plate: list[str], *, seen_fraction: float, val_of_seen: float,
                       seed: int = 0) -> list[str]:
    """The split of every cell by WELL_SPLIT_RULE: per plate, `val`, `train` or `heldout` by the position of its well
    in a seeded sha256 order of the plate's wells. No label and no experiment is read."""
    if not 0 < seen_fraction < 1 or not 0 < val_of_seen < 1:
        raise ValueError(f"seen_fraction and val_of_seen must be in (0, 1), got {seen_fraction}, {val_of_seen}")
    wells_of_plate: dict[str, set[str]] = {}
    for unit, p in zip(well_unit, plate):
        wells_of_plate.setdefault(p, set()).add(unit)
    split_of: dict[str, str] = {}
    for p, members in wells_of_plate.items():
        ordered = sorted(members, key=lambda u: _rank(seed, "seen", p, u))
        n_seen = _round_half_up(seen_fraction, len(ordered))
        n_val = max(MIN_VAL_WELLS, _round_half_up(val_of_seen, n_seen))
        if n_seen - n_val < 1:
            raise ValueError(f"plate {p} has too few wells ({len(ordered)}: {n_seen} seen) for {n_val} val wells and "
                             "a train well")
        for position, unit in enumerate(ordered):
            split_of[unit] = "val" if position < n_val else "train" if position < n_seen else "heldout"
    return [split_of[u] for u in well_unit]


def _wells_per_plate(cells) -> dict:
    """Per split, the (min, max) of its well count over the plates."""
    counts = cells.groupby(["plate", "split"]).well_unit.nunique().unstack(fill_value=0)
    return {s: [int(counts[s].min()), int(counts[s].max())] if s in counts else [0, 0] for s in SPLITS}


def _split_counts(cells) -> dict:
    return {"cells_per_split": {s: int((cells.split == s).sum()) for s in SPLITS},
            "wells_per_split": {s: int(cells.well_unit[cells.split == s].nunique()) for s in SPLITS}}


def require_split(prep: Path) -> dict:
    """PREPARE.json of a prep; SystemExit NO_SPLIT_PREP unless cells.csv has a `split` column with values in SPLITS
    and PREPARE.json has `val_fraction` (a well-level split, `split_mode` wells: `seen_fraction`)."""
    import pandas as pd

    try:
        summary = json.loads((prep / "PREPARE.json").read_text(encoding="utf-8"))
        split = pd.read_csv(prep / "cells.csv", usecols=["split"]).split.astype(str)
    except (OSError, ValueError) as error:
        raise SystemExit(f"NO_SPLIT_PREP: no readable PREPARE.json and cells.csv with a split column at {prep} "
                         f"({type(error).__name__}: {error})") from None
    unknown = sorted(set(split) - set(SPLITS))
    designed = "seen_fraction" in summary if summary.get("split_mode") == "wells" else "val_fraction" in summary
    if unknown or not designed:
        raise SystemExit(f"NO_SPLIT_PREP: {prep} has split values {unknown or 'ok'} and split_design "
                         f"{summary.get('split_design')!r} (needs the values {SPLITS} and PREPARE.json val_fraction: "
                         "resplit or prepare the full prep and subsample it under new names)")
    return summary


def _link_or_copy(source: Path, target: Path) -> str:
    """`target` as a hard link to `source`, a copy where linking fails (e.g. another filesystem)."""
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copyfile(source, target)
        return "copied"


def _partial(output: Path) -> Path:
    """The directory a command writes into, "<output>.partial", renamed to `output` once it is complete (a failed run
    leaves only it); one left by an earlier failed run is removed first."""
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
        print(f"removed {partial}, left by an earlier run that did not finish", flush=True)
    partial.mkdir(parents=True)
    return partial


# ---------------------------------------------------------------- resplit

def resplit(source: Path, output: Path, val_fraction: float | None, n_holdout: int | None, seed: int = 0, *,
            split: str = "experiments", seen_fraction: float | None = None, val_of_seen: float | None = None) -> dict:
    """A full prep with a new split from an already prepared one, without re-reading the raw images: images.npy
    hard-linked (copied where that fails), cells.csv the source's with the `split` of assign_splits (`n_holdout` of
    the source's experiments held out by holdout_experiments; the column is added to a source without one),
    PREPARE.json the source's keys with the split keys of `prepare` and the `resplit` block (the held-out design's
    test_experiments is dropped). The same arguments give the split `prepare` gives.

    `split="wells"` (with `seen_fraction` and `val_of_seen`, not `val_fraction` / `n_holdout`): the split of
    assign_well_splits (WELL_SPLIT_RULE, no experiment held out); PREPARE.json drops the source's experiment-level
    keys (val_fraction, test_experiments) and records split_mode, the fractions, the rule, the per-plate well counts
    and in the `resplit` block also the split mode and whether images.npy was linked or copied."""
    import pandas as pd

    if split not in SPLIT_MODES:
        raise ValueError(f"split must be one of {SPLIT_MODES}, got {split!r}")
    wells = split == "wells"
    given = (seen_fraction, val_of_seen) if wells else (val_fraction, n_holdout)
    other = (val_fraction, n_holdout) if wells else (seen_fraction, val_of_seen)
    if any(v is None for v in given) or any(v is not None for v in other):
        raise SystemExit("resplit: --split wells takes --seen-fraction and --val-of-seen; --split experiments takes "
                         "--val-fraction and --holdout-experiments")
    if output.exists():
        raise SystemExit(f"{output} exists; earlier evidence is preserved")
    summary = json.loads((source / "PREPARE.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(source / "cells.csv")
    if "source_row" in cells:
        raise SystemExit(f"{source} is a run prep (cells.csv has source_row): resplit the full prep it came from")
    if wells:
        held = []
        cells["split"] = assign_well_splits(cells.well_unit.tolist(), cells.plate.tolist(),
                                            seen_fraction=seen_fraction, val_of_seen=val_of_seen, seed=seed)
    else:
        experiments = summary.get("experiments") or sorted(cells.experiment.unique())
        held = holdout_experiments(experiments, n_holdout, seed) if n_holdout else []
        if len(held) >= len(experiments):
            raise SystemExit(f"--holdout-experiments {n_holdout} leaves no training experiment of {len(experiments)}")
        cells["split"] = assign_splits(cells.well_unit.tolist(), cells.plate.tolist(), cells.experiment.tolist(),
                                       val_fraction=val_fraction, holdout_experiments=held, seed=seed)
    partial = _partial(output)
    files = {"images.npy": _link_or_copy(source / "images.npy", partial / "images.npy")}
    cells.to_csv(partial / "cells.csv", index=False)
    summary.pop("test_experiments", None)
    if wells:
        summary.pop("val_fraction", None)
        summary.update(**_split_counts(cells), split_mode="wells", seen_fraction=seen_fraction,
                       val_of_seen=val_of_seen, min_val_wells=MIN_VAL_WELLS, holdout_experiments=held,
                       split_design=(f"wells per plate: seen {seen_fraction:g} of every plate's wells (val "
                                     f"{val_of_seen:g} of them, >= {MIN_VAL_WELLS}; train the rest of them), heldout "
                                     "the other wells; no held-out experiment"),
                       split_rule=WELL_SPLIT_RULE, wells_per_plate=_wells_per_plate(cells),
                       resplit={"source": str(source), "seed": seed, "split": "wells", "images": files["images.npy"]})
    else:
        summary.update(**_split_counts(cells), val_fraction=val_fraction, holdout_experiments=held,
                       split_design=split_design(val_fraction, held), resplit={"source": str(source), "seed": seed})
    (partial / "PREPARE.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    partial.rename(output)
    print(json.dumps({"files": files, **{k: summary[k] for k in ("cells_per_split", "wells_per_split",
                                                                  "holdout_experiments", "resplit")}}))
    return summary


# ---------------------------------------------------------------- subsample

TRAIN_CELLS_RULE = ("the smallest fraction f of the form k/n (n = a plate's train-well count, k = 1..n) "
                    "whose training wells, per plate the first max(1, round(f * n)) of the seeded order, hold "
                    ">= train_cells cells in total; 1.0 (every train well) if no f does")


def _train_well_orders(cells, seed: int) -> dict:
    """Per plate (sorted), its train wells in the seeded sha256 order and their cell counts."""
    orders = {}
    train = cells[cells.split == "train"]
    sizes = train.groupby("well_unit").size()
    for plate, wells in train.groupby("plate", sort=True)["well_unit"]:
        ordered = sorted(set(wells), key=lambda u: _rank(seed, "train", plate, u))
        orders[plate] = (ordered, [int(sizes[u]) for u in ordered])
    return orders


def resolve_train_well_fraction(orders: dict, train_cells: int) -> float:
    """--train-cells: the train_well_fraction of TRAIN_CELLS_RULE (the cell count is nondecreasing in f, so the first
    candidate reaching train_cells is the smallest)."""
    candidates = sorted({k / len(ordered) for ordered, _ in orders.values() for k in range(1, len(ordered) + 1)})
    cumulative = {plate: np.cumsum(counts) for plate, (_, counts) in orders.items()}
    for fraction in candidates:
        total = sum(int(cumulative[plate][max(1, round(fraction * len(ordered))) - 1])
                    for plate, (ordered, _) in orders.items())
        if total >= train_cells:
            return fraction
    return 1.0


def subsample(source: Path, output: Path, train_well_fraction: float | None, seed: int = 0,
              train_cells: int | None = None) -> dict:
    """The run prep: per plate, the first max(1, round(train_well_fraction * n)) of a seeded sha256 order of its n
    TRAIN wells are the training wells; the run prep holds every cell of the training wells and of the VAL wells. The
    other train wells and the held-out experiments are left out. With `train_cells` K instead of a fraction, the
    fraction is resolved by TRAIN_CELLS_RULE (about K training cells; the same fraction on every plate) and the
    `subsample` block records train_cells and the rule next to the resolved train_well_fraction (given as
    --train-well-fraction it gives the same run prep). Rows are renumbered (cells.csv keeps `split` and adds
    `source_row`). PREPARE.json: the source's keys with the per-split counts of the run prep and the `subsample`
    block."""
    import pandas as pd

    if train_cells is not None:
        if train_cells < 1:
            raise ValueError(f"train_cells must be >= 1, got {train_cells}")
    elif not 0 < train_well_fraction <= 1:
        raise ValueError(f"train_well_fraction must be in (0, 1], got {train_well_fraction}")
    if output.exists():
        raise SystemExit(f"{output} exists; earlier evidence is preserved")
    summary = require_split(source)
    cells = pd.read_csv(source / "cells.csv")
    orders = _train_well_orders(cells, seed)
    resolved = {}
    if train_cells is not None:
        train_well_fraction = resolve_train_well_fraction(orders, train_cells)
        resolved = {"train_cells": train_cells, "train_cells_rule": TRAIN_CELLS_RULE}
        print(f"subsample: --train-cells {train_cells} -> train_well_fraction {train_well_fraction:.6g}", flush=True)
    training: set[str] = set()
    for ordered, _ in orders.values():
        training.update(ordered[:max(1, round(train_well_fraction * len(ordered)))])
    keep = np.sort(cells.row[(cells.split == "val") | cells.well_unit.isin(training)].to_numpy())
    sub = cells.set_index("row").loc[keep].reset_index()
    sub.insert(sub.columns.get_loc("split") + 1, "source_row", sub["row"])
    sub["row"] = np.arange(len(sub))
    partial = _partial(output)
    np.save(partial / "images.npy", np.ascontiguousarray(np.load(source / "images.npy", mmap_mode="r")[keep]))
    sub.to_csv(partial / "cells.csv", index=False)
    if "wells_per_plate" in summary:             # a well-level split (resplit --split wells): the run prep's counts
        summary["wells_per_plate"] = _wells_per_plate(sub)
    summary.update(n_cells=int(len(sub)), **_split_counts(sub),
                   subsample={"source": str(source), "train_well_fraction": train_well_fraction, "seed": seed,
                              "stratified_by": "plate", "n_source_cells": int(len(cells)),
                              "n_train_wells": len(training), "n_train_cells": int((sub.split == "train").sum()),
                              **resolved})
    (partial / "PREPARE.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    partial.rename(output)
    print(json.dumps({k: summary[k] for k in ("n_cells", "cells_per_split", "wells_per_split", "subsample")}))
    return summary


# ---------------------------------------------------------------- prepare

def _crop_shard(task: dict) -> dict:
    """Segment and crop the sites of one shard in their order; the crops' bytes are appended to task["raw"] (uint8,
    C order, one (6, side, side) crop after the other: never all in RAM). Returns the shard's cell columns, its
    (site_id, n_cells) in site order and its seconds. Top-level and argument-only, so a spawned worker can run it
    with its own SiteReader (zip handle)."""
    sites, half, seed, label = task["sites"], task["half"], task["seed"], task["label"]
    reader = SiteReader(task["images"])
    columns = {name: [] for name in CELL_COLUMNS}
    per_site, n_cells = [], 0
    started = time.time()
    with open(task["raw"], "wb") as raw_file:
        for position, row in enumerate(sites.itertuples(index=False), start=1):
            raw = reader.site(row)
            nuclei = nucleus_centroids(raw[0], min_area=task["min_area"], max_area=task["max_area"], half=half)
            image = stretch_site(raw)
            nuclei = sorted(nuclei, key=lambda n: _rank(seed, row.site_id, n[0]))[:task["max_cells_per_site"]]
            plate = f"{row.experiment}_{row.plate}"
            for label_id, r, c in nuclei:
                raw_file.write(np.ascontiguousarray(image[:, r - half:r + half, c - half:c + half]).tobytes())
                for name, value in zip(CELL_COLUMNS, (f"{row.site_id}_n{label_id:04d}", row.site_id,
                                                      # SHAPE's qualified_well: "<plate>/<well>"
                                                      f"{plate}/{row.well_id}", str(row.sirna),
                                                      plate, row.experiment)):
                    columns[name].append(value)
            n_cells += len(nuclei)
            per_site.append((row.site_id, len(nuclei)))
            if position % 100 == 0:
                print(f"{label}: {position}/{len(sites)} sites, {n_cells} cells, "
                      f"{time.time() - started:.0f}s", flush=True)
    seconds = time.time() - started
    print(f"{label}: done, {len(sites)} sites, {n_cells} cells, {seconds:.0f}s", flush=True)
    return {"columns": columns, "per_site": per_site, "n_cells": n_cells, "seconds": seconds}


def _well_types(meta, well_types: str) -> list[str]:
    """--well-types: comma-separated values of metadata.csv `well_type`, or "all" for every value in the file
    (RxRx1: treatment, positive_control, negative_control)."""
    if well_types == "all":
        return sorted(meta.well_type.dropna().astype(str).unique())
    return well_types.split(",")


def prepare(args) -> dict:
    """The full prep. --workers N splits the sites (sorted by site_id) into N contiguous shards, each segmented and
    cropped by its own spawned process into a raw shard file; the parent then streams the shards in order into
    images.npy (header written by open_memmap, the same bytes np.save writes) and builds cells.csv / PREPARE.json from
    the shards' cells in order. Every site's crops depend on that site alone, so any N gives the same images.npy,
    cells.csv and PREPARE.json (but `config`, `workers`, `seconds`, `shard_seconds`); N = 1 runs in-process."""
    import pandas as pd

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output {out} exists and is not empty")
    workers = getattr(args, "workers", 1)
    if workers < 1:
        raise SystemExit(f"--workers must be >= 1, got {workers}")
    meta = pd.read_csv(args.metadata)
    well_types = _well_types(meta, args.well_types)
    rows = meta[(meta.cell_type == args.cell_type) & meta.dataset.isin(args.datasets.split(","))
                & meta.well_type.isin(well_types)]
    experiments = sorted(rows.experiment.unique())[:args.n_experiments]
    held = holdout_experiments(experiments, args.holdout_experiments, args.seed) if args.holdout_experiments else []
    if len(held) >= len(experiments):
        raise SystemExit(f"--holdout-experiments {args.holdout_experiments} leaves no training experiment of "
                         f"{len(experiments)}")
    sites = rows[rows.experiment.isin(experiments)].sort_values("site_id")
    half = args.crop_size // 2
    side = 2 * half
    workers = max(1, min(workers, len(sites)))
    partial = _partial(out)
    shards_dir = partial / "shards"
    shards_dir.mkdir()
    bounds = np.linspace(0, len(sites), workers + 1).round().astype(int)
    tasks = [{"sites": sites.iloc[bounds[i]:bounds[i + 1]], "images": str(args.images), "half": half,
              "seed": args.seed, "min_area": args.min_area, "max_area": args.max_area,
              "max_cells_per_site": args.max_cells_per_site, "raw": str(shards_dir / f"shard_{i:03d}.u8"),
              "label": "prepare" if workers == 1 else f"prepare shard {i + 1}/{workers}"}
             for i in range(workers)]
    started = time.time()
    if workers == 1:
        results = [_crop_shard(tasks[0])]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        print(f"prepare: {len(sites)} sites in {workers} shards of {np.diff(bounds).min()}-{np.diff(bounds).max()} "
              "sites", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_crop_shard, tasks))
    n_cells = sum(r["n_cells"] for r in results)
    if n_cells == 0:
        raise SystemExit(f"prepare: no cells in {len(sites)} sites")
    # images.npy: the header of np.save for (n_cells, 6, side, side) uint8, then the shards' bytes in order
    images = np.lib.format.open_memmap(partial / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(n_cells, len(CHANNELS), side, side))
    offset = images.offset
    del images
    with open(partial / "images.npy", "r+b") as target:
        target.seek(offset)
        for task in tasks:
            with open(task["raw"], "rb") as source:
                shutil.copyfileobj(source, target, COPY_CHUNK_BYTES)
            Path(task["raw"]).unlink()
        written = target.tell() - offset
    if written != n_cells * len(CHANNELS) * side * side:
        raise RuntimeError(f"prepare: the shards hold {written} bytes, expected {n_cells} crops")
    shards_dir.rmdir()
    cells = pd.DataFrame({name: [v for r in results for v in r["columns"][name]] for name in CELL_COLUMNS})
    cells.insert(0, "row", np.arange(len(cells)))
    cells["split"] = assign_splits(cells.well_unit.tolist(), cells.plate.tolist(), cells.experiment.tolist(),
                                   val_fraction=args.val_fraction, holdout_experiments=held, seed=args.seed)
    cells.to_csv(partial / "cells.csv", index=False)
    per_site = Counter()
    for result in results:
        for site_id, count in result["per_site"]:
            per_site[site_id] = count
    summary = {
        "config": vars(args), "experiments": experiments, "channels": list(CHANNELS),
        "well_types": well_types,
        "n_sites": int(len(sites)), "n_cells": int(len(cells)),
        "n_sirna": int(cells.sirna.nunique()),
        **_split_counts(cells),
        "sites_without_cells": sum(1 for v in per_site.values() if v == 0),
        "metadata_sha256": hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest(),
        "val_fraction": args.val_fraction, "holdout_experiments": held,
        "split_design": split_design(args.val_fraction, held),
        "workers": workers, "seconds": round(time.time() - started, 1),
        "shard_seconds": [round(r["seconds"], 1) for r in results],
    }
    (partial / "PREPARE.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    if out.exists():
        out.rmdir()           # empty (checked above): rename needs the name free
    partial.rename(out)
    print(json.dumps({k: summary[k] for k in ("n_sites", "n_cells", "n_sirna", "cells_per_split", "wells_per_split",
                                              "holdout_experiments", "workers", "seconds")}))
    return summary


# ---------------------------------------------------------------- CLI

def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="build_prep", description="RxRx1 preps in the task-116 v3 format")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare", help="the full prep: crops of the raw images + the well split")
    pr.add_argument("--metadata", required=True)
    pr.add_argument("--images", required=True, help="rxrx1-images.zip or an unpacked images directory")
    pr.add_argument("--output", required=True)
    pr.add_argument("--cell-type", default="HUVEC")
    pr.add_argument("--datasets", default="train")
    pr.add_argument("--n-experiments", type=int, default=4)
    pr.add_argument("--well-types", default="positive_control",
                    help='comma-separated metadata.csv well_type values, or "all" (every value in the file)')
    pr.add_argument("--max-cells-per-site", type=int, default=24)
    pr.add_argument("--crop-size", type=int, default=64)
    pr.add_argument("--min-area", type=int, default=60)
    pr.add_argument("--max-area", type=int, default=2500)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--val-fraction", type=float, default=0.1,
                    help="share of every plate's wells that are val (at least 2 per plate)")
    pr.add_argument("--holdout-experiments", type=int, default=2,
                    help="number of experiments held out entirely (seeded sha256 order)")
    pr.add_argument("--workers", type=int, default=1,
                    help="processes cropping contiguous shards of the sites (the output is the same for any value)")
    rs = sub.add_parser("resplit", help="a full prep with a new split from an already prepared one (no image re-read)")
    rs.add_argument("--source", required=True)
    rs.add_argument("--output", required=True)
    rs.add_argument("--split", choices=SPLIT_MODES, default="experiments",
                    help="experiments (default): held-out experiments + val wells per plate; wells: per plate seen "
                         "and held-out wells of every experiment (WELL_SPLIT_RULE)")
    rs.add_argument("--val-fraction", type=float, default=None,
                    help="--split experiments (required): share of every plate's wells that are val (at least 2 per "
                         "plate)")
    rs.add_argument("--holdout-experiments", type=int, default=None,
                    help="--split experiments (required): number of experiments held out entirely (seeded sha256 "
                         "order)")
    rs.add_argument("--seen-fraction", type=float, default=None,
                    help="--split wells (required): share of every plate's wells that are seen (train + val)")
    rs.add_argument("--val-of-seen", type=float, default=None,
                    help="--split wells (required): share of the seen wells that are val (at least 2 per plate)")
    rs.add_argument("--seed", type=int, default=0)
    su = sub.add_parser("subsample", help="the run prep: all cells of a share of the train wells + all val cells")
    su.add_argument("--prep", required=True)
    su.add_argument("--output", required=True)
    size = su.add_mutually_exclusive_group()
    size.add_argument("--train-well-fraction", type=float, default=0.1,
                      help="share of every plate's train wells whose cells are the training cells (1.0: every train "
                           "well)")
    size.add_argument("--train-cells", type=int, default=None,
                      help="instead of --train-well-fraction: about this many training cells (TRAIN_CELLS_RULE)")
    su.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    if a.cmd == "prepare":
        prepare(a)
    elif a.cmd == "resplit":
        resplit(Path(a.source), Path(a.output), a.val_fraction, a.holdout_experiments, a.seed, split=a.split,
                seen_fraction=a.seen_fraction, val_of_seen=a.val_of_seen)
    elif a.train_cells is not None:
        subsample(Path(a.prep), Path(a.output), None, a.seed, train_cells=a.train_cells)
    else:
        subsample(Path(a.prep), Path(a.output), a.train_well_fraction, a.seed)


if __name__ == "__main__":
    main()
