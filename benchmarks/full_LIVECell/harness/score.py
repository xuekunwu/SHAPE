"""Post-delivery scoring of full_LIVECell; the private labels and GT are read only here.

The verdict metrics are the shared full_* metrics (benchmarks/_shared/FULL_METRICS_CONTRACT.md, `full_metrics`):
label = cell type over the heldout units (stages 2-5), unit = the anonymous unit id (profiles.csv `well`). The scorer
alone derives two pools from `stage` (no field carries them): heldout_fit (stages 2-3) and heldout_test (stages 4-5).
Main split = one fold, cross-time and single-shot: fit on heldout_fit, test on heldout_test, never the reverse; guard
split = cross-time and cross-well: per well index k one part, the heldout_test units of well index k against the
heldout_fit units of the other well indices. The cell-level probe reads the uncorrected cell embedding (embedding.npy,
cells.json), the unit level the delivered corrected unit profiles (corrected.npy), as in full_CPJUMP1. Reported, not
verdicts: the cross-time retrieval secondaries, and the stage GT metrics (segmentation vs private/gt, crops via
crop_bbox and mask_label) with a paired bootstrap over images against the control run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks._shared import full_metrics as fm
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, within, write_json
from benchmarks.full_CPJUMP1.harness.score import cosine, excluded_image

from .data import HELDOUT, N_UNITS, TRAIN, VAL, load_prepared

N_IMAGES, N_TYPES = 2 * N_UNITS, 8
FIT, TEST = "heldout_fit", "heldout_test"           # the scorer's two heldout pools, derived from `stage` alone
POOL_STAGES = {FIT: (2, 3), TEST: (4, 5)}
MAIN_RULE = "single-shot cross-time: fit on heldout_fit (stages 2-3), test on heldout_test (stages 4-5)"
GUARD_RULE = ("single-shot cross-time and cross-well: per well index k one part, the heldout_test units of well index "
              "k against the heldout_fit units of the other well indices; empty parts dropped")
DRAWS = "full_metrics_draws.npz"
GT_KEY = "labels"                                   # private/gt/<image_id>.npz: the uint32 instance label image
GT_INDEX = "private/gt_index.json"                  # rows {image_id, file (under private/), sha256, ...}
GT_TABLE = "stage_gt_images.json"                   # evaluation/: the per-image GT readings (the control's, reread)
IOU_THRESHOLDS = tuple(round(v, 2) for v in np.arange(0.50, 1.00, 0.05))   # the NeurIPS22-CellSeg thresholds
MATCH_IOU, FULL_COVER = 0.5, 0.95
BOOTSTRAP, BOOTSTRAP_SEED = 1000, 0
UNIT_RULE = ("cell-level probe: the uncorrected cell embedding (embedding.npy rows of cells.json; batch correction "
             "runs on unit profiles only); unit level (mAP, unit-level probes): the delivered corrected unit profile "
             "(corrected.npy, L2-normalised; the pipeline's unit profile is the mean over the two sites of the "
             "per-site median cell, not the contract's mean of L2-normalised cells)")
# Per family: metric -> (per-image numerator, per-image denominator); a group's value = sum(num) / sum(den).
SEGMENTATION = {"AP50": ("ap50", "one"), "mean_AP_50_95": ("mean_ap", "one"), "F1_50": ("f1_50", "one"),
                "count_ratio": ("n_pred", "n_gt")}
CROP = {"retention": ("retention_sum", "n_matched"), "contamination": ("contamination_sum", "n_matched"),
        "full_cover_rate": ("n_full_cover", "n_matched"), "unmatched_crop_rate": ("n_unmatched", "n_delivered"),
        "gt_recall": ("n_gt_recalled", "n_gt")}
STAGE_GT_RULE = {
    "segmentation": "per image vs private/gt: AP = TP / (TP + FP + FN) at an IoU threshold (maximum-cardinality "
                    "matching, the NeurIPS22-CellSeg definition; an image without GT and predictions has AP 0), "
                    "AP50 at IoU >= 0.5, mean AP over IoU 0.50:0.05:0.95, F1 = 2 TP / (2 TP + FP + FN) at 0.5; a "
                    "group's AP50 / mean AP / F1 = the mean over its images; count_ratio = the group's predicted "
                    "instances / its GT instances; GT instances = the labels present in the GT label image (an "
                    "annotation fully covered by earlier ones has no pixels: never a false negative, counted in "
                    "n_gt_instances_without_pixels)",
    "crop": "per delivered cell (cells.json; excluded cells deliver nothing): target = the GT instance of the highest "
            "IoU with the delivered mask instance `mask_label`, matched if IoU >= 0.5; crop_bbox = [min_row, min_col, "
            "max_row, max_col], half-open (image[min_row:max_row, min_col:max_col]); retention = |target in box| / "
            "|target|, contamination = |other GT instances in box| / |box|, full cover = retention >= 0.95, each "
            "over the matched cells; unmatched_crop_rate over the delivered cells; gt_recall = the share of GT "
            "instances that are a matched cell's target; group values pool the group's cells",
    "bootstrap": f"{BOOTSTRAP} resamples of each group's images with replacement, seed {BOOTSTRAP_SEED} (one generator "
                 "over the groups in order), the same draws for every arm; delta = arm minus control, SE = the s.d. "
                 "of the paired bootstrap delta, 95% percentile interval; reported, not verdicts"}


# ---- units, splits, retrieval --------------------------------------------------------------------------------------

def unit_meta(images: list[dict], labels: pd.DataFrame, profiles_csv: Path) -> pd.DataFrame:
    """profiles.csv rows (in order) joined with their unit's private identity: cell_type, real_well, well_index,
    stage, usage, plate. Every fixed-design invariant the scorer relies on is checked here."""
    if len(images) != N_IMAGES or len({r["image_id"] for r in images}) != N_IMAGES:
        raise ValueError(f"evaluation needs all {N_IMAGES} prepared images")
    public = pd.DataFrame(images)[["image_id", "well", "plate", "usage"]]
    if len(labels) != N_IMAGES:
        raise ValueError("private labels do not cover the images")
    joined = public.merge(labels.rename(columns={"well": "real_well"}), on="image_id", validate="one_to_one")
    if len(joined) != N_IMAGES:
        raise ValueError("private labels do not cover the images")
    keys = ["well", "cell_type", "real_well", "well_index", "stage", "usage", "plate"]
    identities = joined[keys].drop_duplicates()
    if identities.well.duplicated().any():
        raise ValueError("site labels disagree within unit")
    if (identities.groupby(["cell_type", "well_index"]).real_well.nunique() != 1).any():
        raise ValueError("a well index names two wells")
    if (identities.plate != "stage_" + identities.stage.astype(str)).any():
        raise ValueError("the public batch key disagrees with the private stage")
    if (identities.usage.isin([TRAIN, VAL]) != (identities.stage < 2)).any() or (
            (identities.usage == HELDOUT) != (identities.stage >= 2)).any():
        raise ValueError("heldout units are not exactly stages 2-5")
    csv = pd.read_csv(profiles_csv, keep_default_na=False)
    if not {"well", "plate", "usage"} <= set(csv.columns):
        raise ValueError("profiles.csv lacks well / plate / usage")
    meta = csv[["well"]].merge(identities, on="well", how="left", validate="one_to_one", indicator=True)
    if len(meta) != N_UNITS or (meta.pop("_merge") != "both").any() or set(meta.well) != set(identities.well):
        raise ValueError(f"evaluation needs all {N_UNITS} units")
    if (csv.plate.to_numpy() != meta.plate.to_numpy()).any() or (csv.usage.to_numpy() != meta.usage.to_numpy()).any():
        raise ValueError("profiles.csv disagrees with the preparation")
    meta["well_index"] = meta.well_index.astype(int)
    meta["stage"] = meta.stage.astype(int)
    return meta


def heldout_units(meta: pd.DataFrame) -> pd.DataFrame:
    """The heldout units sorted by unit id; column `index` = the row in profiles.csv / *.npy, column `pool` =
    heldout_fit (stages 2-3) or heldout_test (stages 4-5), derived from `stage` alone."""
    units = meta[meta.usage == HELDOUT].sort_values("well").reset_index()
    if units.cell_type.nunique() != N_TYPES:
        raise ValueError(f"the heldout units do not hold all {N_TYPES} cell types")
    units["pool"] = np.where(units.stage.isin(POOL_STAGES[FIT]), FIT, TEST)
    if not {FIT, TEST} <= set(units.pool):
        raise ValueError("the heldout units do not fill both pools (stages 2-3 and stages 4-5)")
    return units


def layout_parts(units: pd.DataFrame) -> tuple[list, list]:
    """(main, guard) folds over `units` (heldout_units rows). main = one fold of one part, heldout_fit -> heldout_test.
    guard = one fold whose parts are the well indices k (test: the heldout_test units of well index k, train: the
    heldout_fit units of the other well indices; empty parts dropped)."""
    fit, test = (units.pool == FIT).to_numpy(), (units.pool == TEST).to_numpy()
    index = units.well_index.to_numpy()
    parts = [(np.flatnonzero(fit & (index != k)), np.flatnonzero(test & (index == k))) for k in sorted(set(index))]
    main, guard = [(np.flatnonzero(fit), np.flatnonzero(test))], [part for part in parts if len(part[1])]
    kind = units.cell_type.to_numpy()
    for name, fold in (("main", main), ("guard", guard)):
        for number, (train, held) in enumerate(fold):
            if set(kind[held]) - set(kind[train]):
                raise ValueError(f"{name} part {number}: test cell types {sorted(set(kind[held]) - set(kind[train]))} "
                                 "do not occur on the train side")
    return [main], [guard]


def full_layout(meta: pd.DataFrame) -> tuple[fm.Layout, pd.DataFrame]:
    """The full_metrics layout over the heldout units (sorted by unit id): main = one fold, heldout_fit ->
    heldout_test; guard = one fold with one part per well index k (`layout_parts`)."""
    units = heldout_units(meta)
    main, guard = layout_parts(units)
    describe = {"main": {"rule": MAIN_RULE, "n_types": int(units.cell_type.nunique()),
                         "n_parts": [len(fold) for fold in main]},
                "guard": {"rule": GUARD_RULE, "n_types": int(units.cell_type.nunique()),
                          "n_parts": [len(fold) for fold in guard]}}
    return fm.Layout(units.well.tolist(), units.cell_type.tolist(), {"main": main, "guard": guard}, describe), units


def retrieval(units: pd.DataFrame, matrix: np.ndarray, *, cross_well: bool) -> tuple[dict, pd.DataFrame]:
    """Raw-cosine retrieval over the heldout units (`matrix` rows = `units` rows): the queries are the heldout_test
    units, the candidates the heldout_fit units (cross_well: the query's own well index left out), positives the same
    type; pessimistic ties (fm.average_precision; reciprocal rank = 1 / (1 + the negatives >= the best positive));
    macro over types."""
    sim = cosine(matrix)
    kind, pool, index = units.cell_type.to_numpy(), units.pool.to_numpy(), units.well_index.to_numpy()
    records = []
    for i in np.flatnonzero(pool == TEST):
        candidates = pool == FIT
        if cross_well:
            candidates &= ~((kind == kind[i]) & (index == index[i]))
        ids = np.flatnonzero(candidates)
        positives = kind[ids] == kind[i]
        if not positives.any():
            continue
        values = sim[i, ids]
        # The first positive's pessimistic rank: every negative tied with it ranks above it (full_CPJUMP1 counts
        # tied positives too, the same rank with its single positive per query).
        rank = 1 + int(np.count_nonzero(values[~positives] >= values[positives].max()))
        records.append({"well": units.well[i], "cell_type": kind[i],
                        "average_precision": fm.average_precision(values, positives), "reciprocal_rank": 1 / rank,
                        "top1": float(rank == 1), "candidates": len(ids)})
    table = pd.DataFrame(records)
    if table.empty:
        return {"n_queries": 0, "n_types": 0, "mAP": None, "MRR": None, "top1": None}, table
    macro = table.groupby("cell_type")[["average_precision", "reciprocal_rank", "top1"]].mean().mean()
    return {"n_queries": len(table), "n_types": int(table.cell_type.nunique()), "mAP": float(macro.average_precision),
            "MRR": float(macro.reciprocal_rank), "top1": float(macro.top1)}, table


def expected_queries(units: pd.DataFrame, *, cross_well: bool) -> int:
    """The queries the labels admit: heldout_test units whose type has a heldout_fit unit (cross_well: of another
    well index)."""
    fit = units[units.pool == FIT]
    count = fit.groupby("cell_type").size()
    per_well = fit.groupby(["cell_type", "well_index"]).size()
    n = 0
    for row in units[units.pool == TEST].itertuples():
        total = int(count.get(row.cell_type, 0))
        if cross_well:
            total -= int(per_well.get((row.cell_type, row.well_index), 0))
        n += total > 0
    return n


def design_counts(meta: pd.DataFrame) -> dict:
    """The heldout design derived from the labels (RESULT.json `design`): the pools' stages and unit counts, the
    layout's folds and parts, and the queries of the retrieval secondaries."""
    units = heldout_units(meta)
    main, guard = layout_parts(units)
    fit, test = (units[units.pool == pool] for pool in (FIT, TEST))
    return {"n_units": len(meta), "n_heldout_units": len(units),
            "n_heldout_fit_units": len(fit), "n_heldout_test_units": len(test),
            "heldout_fit_stages": list(POOL_STAGES[FIT]), "heldout_test_stages": list(POOL_STAGES[TEST]),
            "n_heldout_fit_units_per_type": {t: int(n) for t, n in fit.groupby("cell_type").size().items()},
            "n_heldout_test_units_per_type": {t: int(n) for t, n in test.groupby("cell_type").size().items()},
            "layout": {"main": {"rule": MAIN_RULE, "n_folds": len(main), "n_parts": [len(f) for f in main]},
                       "guard": {"rule": GUARD_RULE, "n_folds": len(guard), "n_parts": [len(f) for f in guard]}},
            "n_queries": {"cross_time": expected_queries(units, cross_well=False),
                          "cross_time_cross_well": expected_queries(units, cross_well=True)}}


def full_metrics(run: Path, meta: pd.DataFrame, corrected: np.ndarray, delivery: dict, control: Path | None,
                 images: list[dict]) -> dict:
    """RESULT.json `full_metrics` (the reading saved as evaluation/DRAWS); with `control` (a scored control run) the
    deltas, SEs and verdicts against the control's saved reading."""
    for name in ("cells.json", "excluded_cells.json", "embedding.npy"):
        if name not in delivery["artifacts"]:
            raise ValueError(f"full metrics need the delivered {name}")
    layout, units = full_layout(meta)
    cells = read_json(run / "cells.json")
    embedding = np.load(run / "embedding.npy", mmap_mode="r")
    if len(cells) != len(embedding):
        raise ValueError("cell embedding rows disagree with cells.json")
    position = {w: i for i, w in enumerate(units.well)}
    cell_unit = np.array([position.get(c["well"], -1) for c in cells], dtype=np.int64)
    excluded = [str(c) for c in read_json(run / "excluded_cells.json")]
    well_of = {r["image_id"]: r["well"] for r in images}
    if any(excluded_image(c) not in well_of for c in excluded):
        raise ValueError("an excluded cell names an image outside the preparation")
    reading = fm.evaluate(layout, embedding, cell_unit, cell_ids=[c["cell_id"] for c in cells],
                          profiles=np.asarray(corrected)[units["index"].to_numpy()],
                          excluded_unit=[position.get(well_of[excluded_image(c)], -1) for c in excluded],
                          excluded_ids=excluded)
    fm.save(run / "evaluation" / DRAWS, reading)
    theirs = None
    if control is not None and Path(control).resolve() != Path(run).resolve():
        theirs = fm.load(Path(control) / "evaluation" / DRAWS)
    full = fm.record(layout, reading, theirs, settings={"label": "cell type (heldout units; scorer only)",
                                                        "unit": "unit id (type, well, time point)",
                                                        "cells_and_profiles": UNIT_RULE})
    has = np.bincount(cell_unit[cell_unit >= 0], minlength=len(units)) > 0
    full["n_units_without_cells_by_side"] = {pool: int((~has[(units.pool == pool).to_numpy()]).sum())
                                             for pool in (FIT, TEST)}
    full["reported"] = {"uncorrected_contract_profile_mAP": {
        **fm.map_reading(layout, fm.unit_profiles(embedding, cell_unit, len(units)), has),
        "rule": "reported, not a verdict: mAP of the contract's unit profile (the mean of the L2-normalised "
                "uncorrected cells, L2-normalised) on the same splits and draws"}}
    return full


# ---- stage GT: segmentation and crops ------------------------------------------------------------------------------

def overlap(gt: np.ndarray, pred: np.ndarray) -> dict:
    """The instance co-occurrence of two label images (one sort of the joint key): the GT and predicted labels and
    areas, and every overlapping (pred, gt) pair's intersection and IoU (indices into the label arrays)."""
    if gt.shape != pred.shape:
        raise ValueError(f"mask shape {pred.shape} != GT shape {gt.shape}")
    key = (pred.astype(np.uint64).ravel() << np.uint64(32)) | gt.astype(np.uint64).ravel()
    pairs, counts = np.unique(key, return_counts=True)
    p, g = (pairs >> np.uint64(32)).astype(np.int64), (pairs & np.uint64(0xFFFFFFFF)).astype(np.int64)
    pred_labels, gt_labels = np.unique(p[p > 0]), np.unique(g[g > 0])
    pi, gi = np.searchsorted(pred_labels, p), np.searchsorted(gt_labels, g)
    pred_area = np.bincount(pi[p > 0], weights=counts[p > 0], minlength=len(pred_labels))
    gt_area = np.bincount(gi[g > 0], weights=counts[g > 0], minlength=len(gt_labels))
    both = (p > 0) & (g > 0)
    pp, gg, inter = pi[both], gi[both], counts[both].astype(np.float64)
    return {"pred_labels": pred_labels, "gt_labels": gt_labels, "pred_area": pred_area, "gt_area": gt_area,
            "pair_pred": pp, "pair_gt": gg, "pair_iou": inter / (pred_area[pp] + gt_area[gg] - inter)}


def true_positives(o: dict, threshold: float) -> int:
    """Maximum-cardinality matching of the pairs with IoU >= threshold (the NeurIPS22-CellSeg matching's TP count);
    a full assignment only over the instances with more than one such pair."""
    keep = o["pair_iou"] >= threshold
    pp, gg = o["pair_pred"][keep], o["pair_gt"][keep]
    if len(np.unique(pp)) == len(pp) and len(np.unique(gg)) == len(gg):
        return int(len(pp))
    from scipy.optimize import linear_sum_assignment
    rows, r = np.unique(pp, return_inverse=True)
    cols, c = np.unique(gg, return_inverse=True)
    valid = np.zeros((len(rows), len(cols)))
    valid[r, c] = 1.0
    i, j = linear_sum_assignment(-valid)
    return int(valid[i, j].sum())


def segmentation_scores(o: dict) -> dict:
    n_pred, n_gt = len(o["pred_labels"]), len(o["gt_labels"])
    ap, f1 = [], []
    for threshold in IOU_THRESHOLDS:
        tp = true_positives(o, threshold)
        fp, fn = n_pred - tp, n_gt - tp
        ap.append(tp / max(1, tp + fp + fn))
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {"n_gt": n_gt, "n_pred": n_pred, "ap50": ap[0], "mean_ap": float(np.mean(ap)), "f1_50": f1[0]}


def crop_scores(o: dict, gt: np.ndarray, cells: list[dict]) -> dict:
    """The crop readings of one image's delivered cells (each with its mask instance `mask_label` and crop_bbox)."""
    height, width = gt.shape
    row = {"n_delivered": len(cells), "n_matched": 0, "n_unmatched": 0, "retention_sum": 0.0,
           "contamination_sum": 0.0, "n_full_cover": 0, "n_gt_recalled": 0}
    order = np.lexsort((o["pair_gt"], -o["pair_iou"], o["pair_pred"]))      # per pred: highest IoU, then lowest GT
    first = np.unique(o["pair_pred"][order], return_index=True)[1]
    best = {int(o["pair_pred"][order][k]): (int(o["pair_gt"][order][k]), float(o["pair_iou"][order][k]))
            for k in first}
    recalled = set()
    for cell in cells:
        label = cell["mask_label"]
        if type(label) is not int or label <= 0:
            raise ValueError(f"cell {cell['cell_id']}: mask_label must be a positive integer")
        at = int(np.searchsorted(o["pred_labels"], label))
        if at >= len(o["pred_labels"]) or o["pred_labels"][at] != label:
            raise ValueError(f"cell {cell['cell_id']}: mask_label {label} is not in its delivered mask")
        box = cell.get("crop_bbox")
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError(f"cell {cell['cell_id']}: crop_bbox must be four integers")
        r0, c0, r1, c1 = box
        if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
            raise ValueError(f"cell {cell['cell_id']}: crop_bbox outside the image")
        target, iou = best.get(at, (-1, 0.0))
        if iou < MATCH_IOU:
            row["n_unmatched"] += 1
            continue
        target_label = o["gt_labels"][target]
        inside = gt[r0:r1, c0:c1]
        in_box = int(np.count_nonzero(inside == target_label))
        retention = in_box / o["gt_area"][target]
        row["n_matched"] += 1
        row["retention_sum"] += retention
        row["contamination_sum"] += (int(np.count_nonzero(inside)) - in_box) / inside.size
        row["n_full_cover"] += int(retention >= FULL_COVER)
        recalled.add(target)
    row["n_gt_recalled"] = len(recalled)
    return row


def delivered_mask(run: Path, row: dict, artifacts: dict) -> np.ndarray:
    """An image's delivered mask (segmentation.json row): inside the run and bound to the delivery."""
    import tifffile
    try:
        relative = Path(row["mask"]).resolve().relative_to(run.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"segmentation mask outside the run: {row['image_id']}") from None
    if artifacts.get(relative) != row.get("sha256"):
        raise ValueError(f"segmentation mask is not bound to the delivery: {row['image_id']}")
    mask = tifffile.imread(within(run, relative))
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError(f"invalid segmentation mask: {row['image_id']}")
    return mask


def gt_labels(prep: Path, entry: dict) -> np.ndarray:
    """An image's GT instance label image (private/gt_index.json row), checked against its pinned SHA256."""
    import io
    payload = within(prep / "private", entry["file"]).read_bytes()        # one read: hashed, then loaded
    image_id = entry["image_id"]
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"private GT changed: {image_id}")
    with np.load(io.BytesIO(payload)) as data:
        gt = data[GT_KEY]
    if gt.ndim != 2 or gt.dtype != np.uint32:
        raise ValueError(f"private GT is not a uint32 label image: {image_id}")
    return gt


def stage_gt_table(prep: Path, run: Path, delivery: dict, prepared: dict, images: list[dict]) -> list[dict]:
    """One row per prepared image (sorted by image id): its segmentation and crop readings."""
    if digest(prep / GT_INDEX) != prepared.get("gt_index_sha256"):
        raise ValueError("private GT index changed")
    entries = read_json(prep / GT_INDEX)
    index = {r["image_id"]: r for r in entries}
    unit_of = {r["image_id"]: r["well"] for r in images}
    if len(entries) != len(index) or set(index) != set(unit_of):
        raise ValueError("private GT does not cover the images")
    if "segmentation.json" not in delivery["artifacts"]:
        raise ValueError("stage GT metrics need the delivered segmentation.json")
    masks = {r["image_id"]: r for r in read_json(run / "segmentation.json")}
    if set(masks) != set(unit_of):
        raise ValueError("segmentation.json does not cover the prepared images")
    by_image, seen = {i: [] for i in unit_of}, set()
    for cell in read_json(run / "cells.json"):
        ident = cell.get("image_id")
        if ident not in by_image or cell.get("well") != unit_of[ident]:
            raise ValueError(f"cell {cell.get('cell_id')}: image or unit outside the preparation")
        if (ident, cell.get("mask_label")) in seen:
            raise ValueError(f"cell {cell.get('cell_id')}: a mask instance delivered twice")
        seen.add((ident, cell.get("mask_label")))
        by_image[ident].append(cell)
    rows = []
    for ident in sorted(unit_of):
        gt = gt_labels(prep, index[ident])
        o = overlap(gt, delivered_mask(run, masks[ident], delivery["artifacts"]))
        # GT instances = the labels present in the label image; an annotation fully covered by earlier ones has no
        # pixels, is never a false negative and is counted apart.
        hidden = int(index[ident]["n_instances"]) - len(o["gt_labels"])
        if hidden < 0:
            raise ValueError(f"private GT holds more instances than annotated: {ident}")
        rows.append({"image_id": ident, "one": 1, **segmentation_scores(o), **crop_scores(o, gt, by_image[ident]),
                     "n_gt_without_pixels": hidden})
    return rows


def gt_groups(labels: pd.DataFrame, images: list[dict]) -> dict[str, list[str]]:
    """The aggregation groups (image ids, sorted): all, per usage (train, val, heldout; the public `usage` of
    `images`), per type, per stage, the official test images."""
    labels = labels.sort_values("image_id")
    usage = {r["image_id"]: r["usage"] for r in images}
    groups = {"all": labels.image_id.tolist()}
    for value in (TRAIN, VAL, HELDOUT):
        groups[f"usage/{value}"] = [i for i in labels.image_id if usage[i] == value]
        if not groups[f"usage/{value}"]:
            raise ValueError(f"no {value} images in the preparation")
    for kind in sorted(labels.cell_type.unique()):
        groups[f"type/{kind}"] = labels.image_id[labels.cell_type == kind].tolist()
    for stage in sorted(labels.stage.astype(int).unique()):
        groups[f"stage/{stage}"] = labels.image_id[labels.stage.astype(int) == stage].tolist()
    groups["official_test"] = labels.image_id[labels.official_split == "test"].tolist()
    if not groups["official_test"]:
        raise ValueError("no official test images in the labels")
    return groups


class GTDraws:
    """The bootstrap of the stage GT metrics, from the groups alone (identical for every arm): per group the times
    each of its images is drawn in each of BOOTSTRAP resamples (one generator, seed BOOTSTRAP_SEED, over the groups
    in order)."""

    def __init__(self, groups: dict[str, list[str]]):
        self.groups = groups
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        self.weights = {}
        for name, ids in groups.items():
            picks = rng.integers(0, len(ids), size=(BOOTSTRAP, len(ids)))
            weights = np.zeros((BOOTSTRAP, len(ids)))
            np.add.at(weights, (np.arange(BOOTSTRAP)[:, None], picks), 1)
            self.weights[name] = weights
        self.fingerprint = hashlib.sha256(json.dumps({"groups": groups, "bootstrap": [BOOTSTRAP, BOOTSTRAP_SEED]},
                                                     sort_keys=True).encode()).hexdigest()

    def reading(self, table: list[dict]) -> dict:
        """Per family, group and metric: (value, draws); value None when the group's denominator is 0."""
        frame = pd.DataFrame(table).set_index("image_id")
        out = {}
        for family, metrics in (("segmentation", SEGMENTATION), ("crop", CROP)):
            out[family] = {}
            for name, ids in self.groups.items():
                part, weights = frame.loc[ids], self.weights[name]
                out[family][name] = {}
                for metric, (num, den) in metrics.items():
                    top, bottom = part[num].to_numpy(np.float64), part[den].to_numpy(np.float64)
                    value = float(top.sum() / bottom.sum()) if bottom.sum() > 0 else None
                    with np.errstate(invalid="ignore", divide="ignore"):
                        drawn = (weights @ top) / (weights @ bottom)
                    out[family][name][metric] = (value, drawn)
        return out


def _interval(values: np.ndarray) -> list[float] | None:
    values = values[np.isfinite(values)]
    return [float(v) for v in np.percentile(values, [2.5, 97.5])] if len(values) else None


def stage_gt(prep: Path, run: Path, delivery: dict, prepared: dict, images: list[dict], labels: pd.DataFrame,
             control: Path | None) -> dict:
    """RESULT.json `stage_gt` (the per-image readings saved as evaluation/GT_TABLE); with `control` (a scored
    control run) each value's paired delta, SE and 95% interval against the control's saved per-image readings."""
    draws = GTDraws(gt_groups(labels, images))
    table = stage_gt_table(prep, run, delivery, prepared, images)
    write_json(run / "evaluation" / GT_TABLE, {"fingerprint": draws.fingerprint, "images": table})
    mine, theirs = draws.reading(table), None
    if control is not None and Path(control).resolve() != Path(run).resolve():
        saved = read_json(Path(control) / "evaluation" / GT_TABLE)
        if saved["fingerprint"] != draws.fingerprint:
            raise ValueError("the arm and the control were scored on different GT groups")
        theirs = draws.reading(saved["images"])
    frame = pd.DataFrame(table).set_index("image_id")
    out = {"rule": STAGE_GT_RULE, "fingerprint": draws.fingerprint, "control": None if theirs is None else "compared",
           "n_images": {name: len(ids) for name, ids in draws.groups.items()},
           "n_gt_instances": {name: int(frame.loc[ids, "n_gt"].sum()) for name, ids in draws.groups.items()},
           "n_gt_instances_without_pixels": {name: int(frame.loc[ids, "n_gt_without_pixels"].sum())
                                             for name, ids in draws.groups.items()}}
    for family, groups in mine.items():
        out[family] = {}
        for name, metrics in groups.items():
            out[family][name] = {}
            for metric, (value, drawn) in metrics.items():
                entry = {"value": value, "ci95": None if value is None else _interval(drawn), "vs_control": None}
                if theirs is not None:
                    other, other_drawn = theirs[family][name][metric]
                    if value is not None and other is not None:
                        diff = drawn - other_drawn
                        finite = diff[np.isfinite(diff)]
                        entry["vs_control"] = {"control": other, "delta": value - other,
                                               "se": 0.0 if np.ptp(finite) <= 1e-12 else float(np.std(finite, ddof=1)),
                                               "ci95": _interval(diff)}
                out[family][name][metric] = entry
    return out


# ---- RESULT --------------------------------------------------------------------------------------------------------

def _gt_lines(gt: dict, family: str) -> list[str]:
    metrics = list(SEGMENTATION if family == "segmentation" else CROP)
    lines = ["| group | images | " + " | ".join(metrics) + " |", "| --- | --- |" + " --- |" * len(metrics)]
    for name, entries in gt[family].items():
        cells = []
        for metric in metrics:
            e = entries[metric]
            text = fm._f(e["value"])
            if e["vs_control"]:
                vs = e["vs_control"]
                text += f" ({fm._f(vs['delta'], '+.3f')} +/- {fm._f(vs['se'])})"
            cells.append(text)
        lines.append(f"| {name} | {gt['n_images'][name]} | " + " | ".join(cells) + " |")
    return lines


def _batch_line(b: dict) -> str:
    return (f"Decision {b['decision']} (rule {b['rule']}, veto {b['veto']}, concerns {b['concerns']}); delivered "
            f"profiles {'corrected' if b['corrected'] else 'uncorrected'}; params {json.dumps(b['params'])}"
            + (f"; Codex call {b['call']}" if "call" in b else "") + ".")


def markdown(result: dict) -> str:
    full, gt = result["full_metrics"], result["stage_gt"]
    lines = [f"# full_LIVECell result ({result['arm']})", "",
             "## Verdict metrics (benchmarks/_shared/FULL_METRICS_CONTRACT.md)", "",
             f"Label = cell type ({full['layout']['splits']['main']['n_types']} types), unit = unit id over "
             f"{result['design']['n_heldout_test_units']} heldout_test units (fit on "
             f"{result['design']['n_heldout_fit_units']} heldout_fit units). {UNIT_RULE}. "
             f"{full['n_units_without_cells_by_side'][TEST]} heldout_test units without delivered cells (misses) and "
             f"{full['n_units_without_cells_by_side'][FIT]} heldout_fit units (no train rows or candidates, never "
             f"misses); <= {full['cell_cap']} cells "
             f"per unit in the cell-level probes, sampled from the delivered and the excluded cells "
             f"({full['n_excluded_sampled']} of {full['n_excluded_cells']} excluded cells sampled, each a miss).", "",
             fm.markdown(full), "",
             "Reported, not a verdict: mAP of the contract's profile (mean of the L2-normalised uncorrected cells): "
             + "; ".join(f"{split} {fm._f(v['value'])} [{fm._f(v['ci95'][0])}, {fm._f(v['ci95'][1])}]"
                         for split, v in full["reported"]["uncorrected_contract_profile_mAP"].items()
                         if isinstance(v, dict)) + ".", "",
             "## Reported retrieval endpoints (secondary)", "",
             "| endpoint | mAP | MRR | top1 | types | queries |", "| --- | --- | --- | --- | --- | --- |"]
    for name, m in result["metrics"].items():
        lines.append(f"| {name} | {fm._f(m['mAP'])} | {fm._f(m['MRR'])} | {fm._f(m['top1'])} | {m['n_types']} | "
                     f"{m['n_queries']} |")
    lines += ["", "## Stage GT metrics (reported, not verdicts)", "",
              "Value" + (" (delta vs control +/- SE)" if gt["control"] else "") + ".", "",
              "Segmentation: " + STAGE_GT_RULE["segmentation"] + ".", "", *_gt_lines(gt, "segmentation"), "",
              f"GT instances (all images): {gt['n_gt_instances']['all']} with pixels, "
              f"{gt['n_gt_instances_without_pixels']['all']} annotated without pixels.", "",
              "Crop: " + STAGE_GT_RULE["crop"] + ".", "", *_gt_lines(gt, "crop"), "", STAGE_GT_RULE["bootstrap"] + ".",
              "", "## Batch decision (reported, not a verdict)", "", _batch_line(result["batch_decision"])]
    return "\n".join(lines) + "\n"


def batch_decision(run: Path, arm: str) -> dict:
    """The arm's batch decision, reported next to the verdicts (not a verdict): the admission decision of its
    delivered batch record (SHAPE: the verifier's decision with the rule outcome, veto and concerns; control and
    Codex: the shared rule's), whether the delivered profiles differ from the uncorrected ones, and the
    correction's parameters (None when uncorrected). Codex also names its delivered call."""
    record = read_json(run / "batch_SEARCH.json")
    admission = record.get("admission") or {}
    corrected = not np.array_equal(np.load(run / "profiles.npy"), np.load(run / "corrected.npy"))
    out = {"decision": admission.get("decision"), "rule": admission.get("rule", admission.get("decision")),
           "veto": admission.get("veto"), "concerns": admission.get("concerns"), "corrected": corrected,
           "params": (record.get("selected_params") if arm == "shape" else record.get("params")) if corrected else None}
    if arm == "codex":
        root = run / "agent" / "batch"
        call = read_json(root / "SELECTED.json")["deliveries"]["dataset"]["call"]
        out.update(call=call, requested_params=read_json(root / "CALLS.json")["dataset"][call]["params"])
    return out


def score(prep: Path, run: Path, control: Path | None = None) -> dict:
    """Score a delivered run; `control`: the scored control run of the same preparation (for the verdicts and the
    stage GT deltas)."""
    delivery = read_json(run / "DELIVERY.json")
    if delivery.get("status") != "complete" or delivery.get("smoke"):
        raise ValueError("run has not delivered a complete pipeline")
    for path, sha in delivery["artifacts"].items():
        if digest(within(run, path)) != sha:
            raise ValueError(f"delivered artifact changed: {path}")
    images = load_prepared(prep)
    if delivery["prepare_sha256"] != digest(prep / "PREPARE.json"):
        raise ValueError("delivery belongs to another preparation")
    prepared = read_json(prep / "PREPARE.json")
    if digest(prep / "private" / "labels.json") != prepared["labels_sha256"]:
        raise ValueError("private evaluation labels changed")
    labels = pd.DataFrame(read_json(prep / "private" / "labels.json"))
    meta = unit_meta(images, labels, run / "profiles.csv")
    design = design_counts(meta)
    units = heldout_units(meta)
    output = run / "evaluation"
    output.mkdir(exist_ok=True)
    metrics = {}
    for variant, name in (("uncorrected", "profiles.npy"), ("delivered", "corrected.npy")):
        x = np.load(run / name)
        if len(x) != len(meta):
            raise ValueError("profile rows disagree with metadata")
        if variant == "delivered":
            corrected = x
        for cross, endpoint in ((False, "cross_time"), (True, "cross_time_cross_well")):
            key = f"{variant}_{endpoint}"
            metrics[key], table = retrieval(units, x[units["index"].to_numpy()], cross_well=cross)
            if (metrics[key]["n_types"], metrics[key]["n_queries"]) != (N_TYPES, design["n_queries"][endpoint]):
                raise ValueError(f"incomplete fixed evaluation design: {key}")
            table.to_csv(output / f"{key}.csv", index=False)
    result = {"status": "complete", "arm": delivery["arm"], "provenance": delivery["provenance"],
              "manifest_sha256": prepared["manifest_sha256"], "n_images": len(images), "n_units": len(meta),
              "design": design, "metrics": metrics, "primary": list(fm.METRICS),
              "retrieval_reading": "delivered_cross_time_cross_well",
              "full_metrics": full_metrics(run, meta, corrected, delivery, control, images),
              "stage_gt": stage_gt(prep, run, delivery, prepared, images, labels, control),
              "batch_decision": batch_decision(run, delivery["arm"]),
              "batch_mode": "transductive: all unlabeled unit profiles; batch key = culture-time stage; "
                            "calibration-only objective (train and val references, no negative controls)"}
    write_json(run / "RESULT.json", result)
    (run / "RESULT.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result
