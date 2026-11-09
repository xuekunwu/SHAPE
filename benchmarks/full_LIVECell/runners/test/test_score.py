from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.full_LIVECell.harness import score as S

TYPES = ("A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3")
GT_SHAPE = (12, 12)


# ---- small layouts ---------------------------------------------------------------------------------------------------

def small_meta(types=("A", "B"), wells=3):
    """Heldout units of `types` x `wells` wells x stages 2-5 (one time point per stage); `pool` as heldout_units
    derives it from the stage (the tests of retrieval / expected_queries take the units with it)."""
    rows = []
    for kind in types:
        for k in range(wells):
            for stage in (2, 3, 4, 5):
                rows.append(dict(well=f"u_{kind}{k}{stage}", cell_type=kind, well_index=k, real_well=f"W{k}",
                                 stage=stage, usage=S.HELDOUT, pool=S.FIT if stage < 4 else S.TEST))
    return pd.DataFrame(rows)


def test_main_split_is_one_single_shot_fold_from_heldout_fit_to_heldout_test():
    layout, units = S.full_layout(small_meta(types=TYPES))
    assert [len(fold) for fold in layout.splits["main"]] == [1]                 # one fold, one part: no reverse direction
    (train, test), = layout.splits["main"][0]
    assert not set(train) & set(test)
    assert set(units.pool[test]) == {S.TEST} and set(units.pool[train]) == {S.FIT}
    assert set(units.stage[test]) == {4, 5} and set(units.stage[train]) == {2, 3}
    assert sorted(train) == list(np.flatnonzero(units.pool == S.FIT)) and sorted(test) == list(np.flatnonzero(units.pool == S.TEST))
    assert layout.summary()["splits"]["main"]["n_test_units"] == [48]
    assert "half" not in units.columns and "half" not in small_meta().columns


def test_guard_split_never_shares_a_well_and_has_one_part_per_well_index():
    meta = small_meta(types=TYPES, wells=4)
    meta = meta[~((meta.cell_type == "Huh7") & (meta.well_index == 3))]          # a type with three wells
    layout, units = S.full_layout(meta)
    wells = list(zip(units.cell_type, units.well_index))
    assert len(layout.splits["guard"]) == 1                                     # one fold, one part per well index
    fold = layout.splits["guard"][0]
    assert len(fold) == 4                                                       # well indices 0-3
    covered = []
    for k, (train, test) in enumerate(fold):
        assert set(units.well_index[test]) == {k} and set(units.pool[test]) == {S.TEST}
        assert set(units.pool[train]) == {S.FIT} and k not in set(units.well_index[train])
        assert not {wells[i] for i in train} & {wells[i] for i in test}         # no well on both sides
        covered += list(test)
    assert sorted(covered) == list(np.flatnonzero(units.pool == S.TEST))        # every heldout_test unit, once
    assert "Huh7" not in set(units.cell_type[fold[3][1]])                       # part 3 holds seven types
    assert layout.summary()["splits"]["guard"]["n_parts"] == [4]
    assert layout.summary()["splits"]["guard"]["n_test_units"] == [len(covered)]


def test_empty_guard_parts_are_dropped():
    meta = small_meta(types=TYPES, wells=3)
    meta = meta[~((meta.well_index == 2) & (meta.pool == S.TEST))]              # no heldout_test unit of well index 2
    layout, units = S.full_layout(meta)
    assert [len(f) for f in layout.splits["guard"]] == [2]


# ---- retrieval -----------------------------------------------------------------------------------------------------

def test_retrieval_perfect_type_signal_and_expected_queries():
    units = small_meta(types=TYPES).sort_values("well").reset_index(drop=True)
    x = np.eye(8)[[TYPES.index(t) for t in units.cell_type]]
    for cross in (False, True):
        result, table = S.retrieval(units, x, cross_well=cross)
        assert result == {"n_queries": 48, "n_types": 8, "mAP": 1.0, "MRR": 1.0, "top1": 1.0}   # the heldout_test units
        assert result["n_queries"] == S.expected_queries(units, cross_well=cross)
        assert set(table.candidates) == ({48} if not cross else {46})           # the heldout_fit units
        assert set(units.stage[units.well.isin(table.well)]) == {4, 5}


def test_same_well_shortcut_is_not_cross_well_success():
    units = small_meta(types=TYPES).sort_values("well").reset_index(drop=True)
    wells = sorted(set(zip(units.cell_type, units.well_index)))
    x = np.eye(len(wells))[[wells.index(w) for w in zip(units.cell_type, units.well_index)]]
    same, _ = S.retrieval(units, x, cross_well=False)
    cross, _ = S.retrieval(units, x, cross_well=True)
    assert same["top1"] == 1.0 and same["mAP"] < 1.0
    # every candidate at cosine 0: pessimistic ties give AP = positives / candidates = 4 / 46, rank 1 + 42 negatives
    assert cross["mAP"] == pytest.approx(4 / 46) and cross["top1"] == 0 and cross["MRR"] == pytest.approx(1 / 43)


def test_ties_are_pessimistic_and_zero_norm_refused():
    units = small_meta(types=TYPES).sort_values("well").reset_index(drop=True)
    result, _ = S.retrieval(units, np.ones((len(units), 3)), cross_well=False)
    assert result["mAP"] == pytest.approx(6 / 48) and result["top1"] == 0 and result["MRR"] == pytest.approx(1 / 43)
    with pytest.raises(ValueError, match="zero-norm"):
        S.retrieval(units, np.zeros((len(units), 3)), cross_well=False)


def test_expected_queries_drop_types_without_other_well_candidates():
    units = small_meta(types=("A", "B"), wells=1).sort_values("well").reset_index(drop=True)
    assert S.expected_queries(units, cross_well=False) == 4 and S.expected_queries(units, cross_well=True) == 0
    assert S.retrieval(units, np.eye(2)[[("A", "B").index(t) for t in units.cell_type]],
                       cross_well=True)[0]["n_queries"] == 0


# ---- segmentation and crop GT ---------------------------------------------------------------------------------------

def label_image(*boxes, shape=GT_SHAPE):
    out = np.zeros(shape, dtype=np.uint32)
    for label, (r0, c0, r1, c1) in enumerate(boxes, start=1):
        out[r0:r1, c0:c1] = label
    return out


A, B = (0, 0, 4, 4), (6, 6, 10, 10)


def seg(gt, pred):
    return S.segmentation_scores(S.overlap(gt, pred))


def test_segmentation_of_known_masks():
    gt = label_image(A, B)
    assert seg(gt, gt) == {"n_gt": 2, "n_pred": 2, "ap50": 1.0, "mean_ap": 1.0, "f1_50": 1.0}
    shifted = label_image(A, (6, 7, 10, 11))                          # IoU 12 / 20 = 0.6 for the second cell
    m = seg(gt, shifted)
    assert m["ap50"] == 1.0 and m["f1_50"] == 1.0
    assert m["mean_ap"] == pytest.approx((3 * 1 + 7 * (1 / 3)) / 10)  # TP 2 at 0.50-0.60, 1 above
    extra = label_image(A, B, (10, 0, 12, 2))                          # one false positive
    assert seg(gt, extra)["ap50"] == pytest.approx(2 / 3) and seg(gt, extra)["f1_50"] == pytest.approx(4 / 5)
    relabelled = np.where(gt == 1, 7, np.where(gt == 2, 3, 0)).astype(np.uint32)
    assert seg(gt, relabelled)["ap50"] == 1.0                          # instance ids need not agree


def test_segmentation_false_positive_only_false_negative_only_and_empty():
    empty, one = np.zeros(GT_SHAPE, np.uint32), label_image(A)
    assert seg(empty, one) == {"n_gt": 0, "n_pred": 1, "ap50": 0.0, "mean_ap": 0.0, "f1_50": 0.0}
    assert seg(one, empty) == {"n_gt": 1, "n_pred": 0, "ap50": 0.0, "mean_ap": 0.0, "f1_50": 0.0}
    assert seg(empty, empty) == {"n_gt": 0, "n_pred": 0, "ap50": 0.0, "mean_ap": 0.0, "f1_50": 0.0}
    with pytest.raises(ValueError, match="shape"):
        S.overlap(one, np.zeros((4, 4), np.uint32))


def test_matching_is_maximum_cardinality_not_a_pair_count():
    gt = label_image((0, 0, 2, 4))                                    # 8 pixels
    pred = label_image((0, 0, 1, 4), (1, 0, 2, 4))                    # two halves, each IoU exactly 0.5
    o = S.overlap(gt, pred)
    assert S.true_positives(o, 0.5) == 1
    assert seg(gt, pred)["ap50"] == pytest.approx(1 / 2)


def cell(label, box, ident="c"):
    return {"cell_id": f"{ident}{label}", "mask_label": label, "crop_bbox": list(box)}


def crops(gt, pred, cells):
    return S.crop_scores(S.overlap(gt, pred), gt, cells)


def test_crop_retention_contamination_and_recall_of_known_boxes():
    gt = label_image(A, (0, 5, 4, 9))                                 # two 16-pixel cells, one column apart
    # box 1 covers cell 1 and two columns of cell 2: retention 1, contamination 8 / 28
    # box 2 cuts cell 2 to 12 pixels: retention 0.75, contamination 0
    r = crops(gt, gt, [cell(1, (0, 0, 4, 7)), cell(2, (0, 6, 4, 9))])
    assert r["n_delivered"] == 2 and r["n_matched"] == 2 and r["n_unmatched"] == 0
    assert r["retention_sum"] == pytest.approx(1 + 0.75)
    assert r["contamination_sum"] == pytest.approx(8 / 28)
    assert r["n_full_cover"] == 1 and r["n_gt_recalled"] == 2


def test_crop_target_is_the_highest_iou_gt_and_unmatched_cells_deliver_nothing():
    gt = label_image(A, B)
    pred = label_image((0, 0, 4, 2), (6, 6, 10, 10), (10, 10, 12, 12))   # half of A (IoU 0.5), B, background
    r = crops(gt, pred, [cell(1, (0, 0, 4, 4)), cell(3, (8, 8, 12, 12))])
    assert r["n_matched"] == 1 and r["n_unmatched"] == 1 and r["n_gt_recalled"] == 1
    assert r["retention_sum"] == 1.0 and r["contamination_sum"] == 0
    low = label_image((0, 0, 4, 1))                                    # IoU 0.25 with A: no match
    assert crops(gt, low, [cell(1, (0, 0, 4, 4))])["n_unmatched"] == 1
    assert crops(gt, gt, [])["n_gt_recalled"] == 0                     # B excluded: nothing delivered


@pytest.mark.parametrize("bad,match", [(dict(mask_label=5), "not in its delivered mask"),
                                       (dict(mask_label=1.0), "positive integer"),
                                       (dict(crop_bbox=[0, 0, 13, 4]), "outside the image"),
                                       (dict(crop_bbox=[4, 0, 4, 4]), "outside the image"),
                                       (dict(crop_bbox=[0, 0, 4]), "four integers")])
def test_crop_inputs_are_checked(bad, match):
    gt = label_image(A, B)
    with pytest.raises(ValueError, match=match):
        crops(gt, gt, [{**cell(1, A), **bad}])


def test_bootstrap_draws_are_deterministic_and_the_delta_is_paired():
    groups = {"all": [f"img_{i:04d}" for i in range(30)], "small": ["img_0001", "img_0002"]}
    first, second = S.GTDraws(groups), S.GTDraws(groups)
    assert first.fingerprint == second.fingerprint
    assert all(np.array_equal(first.weights[g], second.weights[g]) for g in groups)
    assert first.weights["all"].shape == (S.BOOTSTRAP, 30) and set(first.weights["all"].sum(axis=1)) == {30}
    assert S.GTDraws({"all": groups["all"][:-1]}).fingerprint != first.fingerprint

    def table(ap):
        return [{"image_id": i, "one": 1, "ap50": a, "mean_ap": a, "f1_50": a, "n_pred": 2, "n_gt": 2,
                 "n_delivered": 2, "n_matched": 2, "n_unmatched": 0, "retention_sum": 2.0, "contamination_sum": 0.0,
                 "n_full_cover": 2, "n_gt_recalled": 2} for i, a in zip(groups["all"], ap)]

    control = first.reading(table(np.linspace(0, 1, 30)))
    arm = first.reading(table(np.linspace(0, 1, 30) + 0.1))
    value, drawn = arm["segmentation"]["all"]["AP50"]
    other, other_drawn = control["segmentation"]["all"]["AP50"]
    assert value - other == pytest.approx(0.1)
    assert np.allclose(drawn - other_drawn, 0.1)              # the same draws: a constant shift has no spread
    assert arm["crop"]["all"]["retention"][0] == 1.0


# ---- the fixed design ------------------------------------------------------------------------------------------------

FULL = {"wells": 4, "times": 22, "huh7": (3, 13)}        # 7 x 4 x 22 + 3 x 13 = 655 units, the real constants
SMALL = {"wells": 3, "times": 6, "huh7": (3, 6)}          # 144 units, 288 images: the file-based tests


def design(size=FULL, seed=7):
    """(public images.json rows, private labels.json rows) of a synthetic preparation in data.py's layout; per type
    `wells` wells x `times` time points (stage = 6 r // times; well index 0 = val)."""
    from benchmarks.full_LIVECell.harness.data import PUBLIC_FIELDS
    rng = np.random.default_rng(seed)
    units = []
    for kind in TYPES:
        wells, times = size["huh7"] if kind == "Huh7" else (size["wells"], size["times"])
        for k in range(wells):
            for r in range(times):
                stage = 6 * r // times
                usage = (S.VAL if k == 0 else S.TRAIN) if stage < 2 else S.HELDOUT
                units.append(dict(kind=kind, k=k, hours=4 * r, stage=stage, usage=usage))
    unit_ids, image_ids = rng.permutation(len(units)), rng.permutation(2 * len(units))
    public, private = [], []
    for u, unit in enumerate(units):
        for location in (1, 2):
            image = f"img_{image_ids[2 * u + location - 1]:04d}"
            fields = {"image_id": image, "file": f"images/{image}.ome.tif",
                      "segment_file": f"segmentation/{image}.ome.tif", "plate": f"stage_{unit['stage']}",
                      "well": f"unit_{unit_ids[u]:04d}", "site": image, "usage": unit["usage"],
                      "calibration_group": "" if unit["usage"] == S.HELDOUT else
                      f"reference_{TYPES.index(unit['kind']):03d}", "negative_control": False, "shape": [1, 12, 12],
                      "dtype": "uint16", "sha256": "x", "segment_sha256": "x"}
            public.append({key: fields[key] for key in PUBLIC_FIELDS})
            private.append({"image_id": image, "cell_type": unit["kind"], "well": f"W{unit['k']}",
                            "well_index": unit["k"], "location": location, "crop": 1, "hours": unit["hours"],
                            "stage": unit["stage"],
                            "official_split": "test" if unit["k"] == 2 else "train", "source_file": image,
                            "annotation_source": "a.json", "annotation_sources": [{"file": "a.json"}]})
    return sorted(public, key=lambda r: r["image_id"]), sorted(private, key=lambda r: r["image_id"])


def profiles_csv(path, public):
    units = {}
    for row in public:
        units.setdefault(row["well"], row)
    pd.DataFrame([{"well": w, "plate": r["plate"], "usage": r["usage"], "calibration_group": r["calibration_group"],
                   "negative_control": False} for w, r in units.items()]).to_csv(path, index=False)
    return list(units.values())


def test_the_real_fixed_design_counts_are_derived_from_the_labels(tmp_path):
    public, private = design(FULL)
    assert len(public) == S.N_IMAGES == 1310
    profiles_csv(tmp_path / "profiles.csv", public)
    meta = S.unit_meta(public, pd.DataFrame(private), tmp_path / "profiles.csv")
    counts = S.design_counts(meta)
    # per 22-point well: stages 2-5 = time ranks 8-21 (7 in stages 2-3, 7 in stages 4-5); per 13-point well:
    # ranks 5-12 (4 + 4)
    assert counts["n_units"] == 655 and counts["n_heldout_units"] == 7 * 4 * 14 + 3 * 8 == 416
    assert (counts["n_heldout_fit_units"], counts["n_heldout_test_units"]) == (208, 208)
    assert (counts["heldout_fit_stages"], counts["heldout_test_stages"]) == ([2, 3], [4, 5])
    assert counts["n_queries"] == {"cross_time": 208, "cross_time_cross_well": 208}
    assert counts["n_heldout_fit_units_per_type"]["Huh7"] == counts["n_heldout_test_units_per_type"]["Huh7"] == 12
    assert counts["n_heldout_test_units_per_type"]["A172"] == 28
    assert counts["layout"]["main"]["n_folds"] == 1 and counts["layout"]["main"]["n_parts"] == [1]
    assert counts["layout"]["guard"]["n_folds"] == 1 and counts["layout"]["guard"]["n_parts"] == [4]
    layout, units = S.full_layout(meta)
    assert layout.summary()["splits"]["main"]["n_test_units"] == [208]
    assert [len(fold) for fold in layout.splits["guard"]] == [4]
    x = np.eye(8)[[TYPES.index(t) for t in units.cell_type]]
    assert S.retrieval(units, x, cross_well=True)[0] == {"n_queries": 208, "n_types": 8, "mAP": 1.0, "MRR": 1.0,
                                                         "top1": 1.0}


def test_unit_metadata_checks(tmp_path):
    public, private = design(FULL)
    rows = profiles_csv(tmp_path / "profiles.csv", public)
    labels = pd.DataFrame(private)
    table = pd.read_csv(tmp_path / "profiles.csv", keep_default_na=False)
    table.iloc[:-1].to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(ValueError, match="all 655 units"):
        S.unit_meta(public, labels, tmp_path / "short.csv")
    table.assign(plate="stage_0").to_csv(tmp_path / "plate.csv", index=False)
    with pytest.raises(ValueError, match="disagrees with the preparation"):
        S.unit_meta(public, labels, tmp_path / "plate.csv")
    with pytest.raises(ValueError, match="all 1310 prepared images"):
        S.unit_meta(public[:-1], labels, tmp_path / "profiles.csv")
    one = labels.copy()
    one.loc[one.image_id == public[0]["image_id"], "cell_type"] = "other"     # one site of a unit relabelled
    with pytest.raises(ValueError, match="disagree within unit"):
        S.unit_meta(public, one, tmp_path / "profiles.csv")
    leaked = [dict(r) for r in public]
    first = next(r for r in leaked if r["usage"] == S.HELDOUT)
    for r in leaked:
        if r["well"] == first["well"]:
            r["plate"] = "stage_0"
    with pytest.raises(ValueError, match="batch key"):
        S.unit_meta(leaked, labels, tmp_path / "profiles.csv")
    assert len(rows) == 655


def make_prep(root, size=SMALL):
    """A synthetic preparation on disk (data.py's layout): `design(size)`, each 12 x 12 site with GT cells A, B."""
    from benchmarks.full_LIVECell.harness.data import MANIFEST_SHA256, write_npz
    from benchmarks.full_CPJUMP1.harness.data import digest, write_json
    public, private = design(size)
    gt, index = label_image(A, B), []
    for row in public:
        path = root / "private" / "gt" / f"{row['image_id']}.npz"
        write_npz(path, labels=gt, n_instances=np.int64(2), overlap_pixels=np.int64(0))
        # the first image has a third annotation fully covered by the first two: no pixels
        index.append({"image_id": row["image_id"], "file": f"gt/{row['image_id']}.npz", "sha256": digest(path),
                      "n_instances": 3 if row is public[0] else 2, "overlap_pixels": 0})
    write_json(root / "public" / "images.json", public)
    write_json(root / "private" / "labels.json", private)
    write_json(root / "private" / "gt_index.json", index)
    write_json(root / "PREPARE.json", {"status": "complete", "dataset": "LIVECell", "manifest_sha256": MANIFEST_SHA256,
                                       "images_sha256": digest(root / "public" / "images.json"),
                                       "labels_sha256": digest(root / "private" / "labels.json"),
                                       "gt_index_sha256": digest(root / "private" / "gt_index.json")})
    return root


def small(monkeypatch):
    """The scorer's fixed sizes set to the SMALL design (data.load_prepared reads data.N_UNITS)."""
    from benchmarks.full_LIVECell.harness import data
    monkeypatch.setattr(data, "N_UNITS", 144)
    monkeypatch.setattr(S, "N_UNITS", 144)
    monkeypatch.setattr(S, "N_IMAGES", 288)


def deliver(prep, run, arm="control"):
    from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
    names = ["profiles.csv", "profiles.npy", "corrected.npy", "cells.json", "excluded_cells.json", "embedding.npy",
             "segmentation.json", "batch_SEARCH.json"]
    names += [p.relative_to(run).as_posix() for p in sorted((run / "agent").rglob("*.json"))]
    masks = [p.relative_to(run).as_posix() for p in sorted((run / "segment").rglob("*.tif"))]
    write_json(run / "DELIVERY.json", {"status": "complete", "arm": arm, "provenance": {"commit": "test"},
                                       "prepare_sha256": digest(prep / "PREPARE.json"),
                                       "artifacts": {n: digest(run / n) for n in names + masks}})
    return read_json(run / "DELIVERY.json")


def write_batch_record(run, arm):
    """The arm's delivered batch record, as pipeline.batch_correction writes it (uncorrected delivery)."""
    from benchmarks.full_CPJUMP1.harness.data import write_json
    if arm == "shape":
        write_json(run / "batch_SEARCH.json", {"admission": {"decision": "ABSTAIN", "rule": "RUN", "veto": "v",
                                                             "concerns": ["batch_group_without_negative_controls"]},
                                               "selected_params": {}})
        return
    write_json(run / "batch_SEARCH.json", {"admission": {"decision": "RUN"}, "params": {"theta": 2.0}})
    if arm == "codex":
        write_json(run / "agent" / "batch" / "SELECTED.json", {"deliveries": {"dataset": {"call": "input"}}})
        write_json(run / "agent" / "batch" / "CALLS.json", {"dataset": {"input": {"params": {}}}})


def make_run(prep, run, *, cell_noise=0.05, seed=0, miss_every=0, exclude_every=0, arm="control"):
    """A delivered run on `prep`: type-perfect unit profiles, one-hot cell embeddings with noise, masks = the GT
    (every `miss_every`-th image, by sorted id, without cell B; every `exclude_every`-th image's cell B excluded)."""
    import tifffile
    from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
    run.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    images = read_json(prep / "public" / "images.json")
    kind = {r["image_id"]: r["cell_type"] for r in read_json(prep / "private" / "labels.json")}
    rows = profiles_csv(run / "profiles.csv", images)
    profiles = np.array([np.eye(8)[TYPES.index(kind[r["image_id"]])] for r in rows])
    np.save(run / "profiles.npy", profiles)
    np.save(run / "corrected.npy", profiles)
    write_batch_record(run, arm)
    gt = label_image(A, B)
    cells, embedding, excluded, segmentation = [], [], [], []
    for n, row in enumerate(images):
        image = row["image_id"]
        mask = gt.copy()
        if miss_every and n % miss_every == 0:
            mask[mask == 2] = 0
        path = run / "segment" / image / f"cell_mask_{image}.tif"
        path.parent.mkdir(parents=True)
        tifffile.imwrite(path, mask)
        segmentation.append({"image_id": image, "mask": str(path.resolve()), "sha256": digest(path),
                             "n_cells": int(mask.max())})
        for label, box in ((1, [0, 0, 5, 5]), (2, [5, 5, 11, 11])):
            if not (mask == label).any():
                continue
            cell_id = f"default_{image}_cell_{label:04d}"
            if label == 2 and exclude_every and n % exclude_every == 0:
                excluded.append(cell_id)
                continue
            cells.append({"cell_id": cell_id, "image_id": image, "well": row["well"], "plate": row["plate"],
                          "usage": row["usage"], "crop_bbox": box, "mask_label": label})
            embedding.append(np.eye(8)[TYPES.index(kind[image])] + rng.normal(scale=cell_noise, size=8))
    write_json(run / "cells.json", cells)
    write_json(run / "excluded_cells.json", excluded)
    write_json(run / "segmentation.json", segmentation)
    np.save(run / "embedding.npy", np.array(embedding, dtype=np.float32))
    deliver(prep, run, arm)
    return run


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """The SMALL preparation, a perfect control run (scored) and a degraded arm (noisy cells, every third image
    without cell B, every second image's cell B excluded; scored against the control)."""
    root = tmp_path_factory.mktemp("small")
    prep = make_prep(root / "prep")
    control = make_run(prep, root / "control")
    arm = make_run(prep, root / "arm", cell_noise=2.0, seed=1, miss_every=3, exclude_every=2, arm="shape")
    with pytest.MonkeyPatch.context() as mp:
        small(mp)
        results = {"control": S.score(prep, control), "arm": S.score(prep, arm, control=control)}
    return {"prep": prep, "control": control, "arm": arm, "results": results}


def test_complete_design_scoring(scored):
    result, run = scored["results"]["control"], scored["control"]
    design_counts = result["design"]
    assert design_counts["n_units"] == 144 and design_counts["n_heldout_units"] == 96
    assert (design_counts["n_heldout_fit_units"], design_counts["n_heldout_test_units"]) == (48, 48)
    assert design_counts["n_queries"] == {"cross_time": 48, "cross_time_cross_well": 48}
    for key in ("uncorrected_cross_time", "uncorrected_cross_time_cross_well", "delivered_cross_time",
                "delivered_cross_time_cross_well"):
        assert result["metrics"][key] == {"n_queries": 48, "n_types": 8, "mAP": 1.0, "MRR": 1.0, "top1": 1.0}
    full = result["full_metrics"]
    assert full["n_units_without_cells_by_side"] == {S.FIT: 0, S.TEST: 0}
    assert full["layout"]["splits"]["main"]["n_test_units"] == [48] and full["layout"]["splits"]["main"]["n_parts"] == [1]
    assert full["layout"]["splits"]["guard"]["n_parts"] == [3] and full["layout"]["splits"]["guard"]["n_test_units"] == [48]
    for metric in ("mAP", "macro_f1", "balanced_accuracy"):
        assert full["metrics"][metric]["main"]["value"] == 1 and full["metrics"][metric]["guard"]["value"] == 1
        assert full["metrics"][metric]["vs_control"] is None
    gt = result["stage_gt"]
    assert gt["n_images"]["all"] == 288 and gt["n_images"]["type/Huh7"] == 36
    assert gt["n_images"]["official_test"] == 96 and gt["n_images"]["stage/0"] == 48
    assert (gt["n_images"]["usage/train"], gt["n_images"]["usage/val"], gt["n_images"]["usage/heldout"]) == (64, 32, 192)
    assert set(gt["n_images"]) == {"all", "official_test", "usage/train", "usage/val", "usage/heldout",
                                   *[f"type/{t}" for t in TYPES], *[f"stage/{s}" for s in range(6)]}
    for name in gt["segmentation"]:
        assert {m: e["value"] for m, e in gt["segmentation"][name].items()} == {
            "AP50": 1.0, "mean_AP_50_95": 1.0, "F1_50": 1.0, "count_ratio": 1.0}
        assert {m: e["value"] for m, e in gt["crop"][name].items()} == {
            "retention": 1.0, "contamination": 0.0, "full_cover_rate": 1.0, "unmatched_crop_rate": 0.0,
            "gt_recall": 1.0}
    assert gt["control"] is None and gt["segmentation"]["all"]["AP50"]["vs_control"] is None
    assert gt["n_gt_instances"]["all"] == 576 and gt["n_gt_instances_without_pixels"]["all"] == 1
    text = (run / "RESULT.md").read_text(encoding="utf-8")
    assert "| mAP (unit profiles" in text and "| official_test | 96 |" in text
    assert "576 with pixels, 1 annotated without pixels" in text
    assert "0 heldout_test units without delivered cells (misses) and 0 heldout_fit units" in text
    assert "1 fold, test units [48]" in text
    assert (run / "evaluation" / S.GT_TABLE).is_file() and (run / "evaluation" / "delivered_cross_time.csv").is_file()
    assert result["batch_decision"] == {"decision": "RUN", "rule": "RUN", "veto": None, "concerns": None,
                                        "corrected": False, "params": None}
    assert "## Batch decision" in text and "Decision RUN (rule RUN" in text and "uncorrected" in text
    assert scored["results"]["arm"]["batch_decision"]["decision"] == "ABSTAIN"


def test_batch_decision_of_each_arms_record(tmp_path):
    """Corrected profiles report their parameters (SHAPE: the selected ones; control: the tool's), Codex its call."""
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    for arm in ("control", "shape", "codex"):
        run = tmp_path / arm
        run.mkdir()
        np.save(run / "profiles.npy", x)
        np.save(run / "corrected.npy", x + 1)
        write_batch_record(run, arm)
    from benchmarks.full_CPJUMP1.harness.data import read_json, write_json
    write_json(tmp_path / "shape" / "batch_SEARCH.json", {**read_json(tmp_path / "shape" / "batch_SEARCH.json"),
               "admission": {"decision": "RUN", "rule": "RUN", "veto": None, "concerns": []},
               "selected_params": {"harmony_params": {"theta": 1.0}}})
    write_json(tmp_path / "codex" / "agent" / "batch" / "SELECTED.json", {"deliveries": {"dataset": {"call": "c1"}}})
    write_json(tmp_path / "codex" / "agent" / "batch" / "CALLS.json", {"dataset": {"c1": {"params": {"theta": 2.0}}}})
    assert S.batch_decision(tmp_path / "control", "control")["params"] == {"theta": 2.0}
    assert S.batch_decision(tmp_path / "shape", "shape") == {"decision": "RUN", "rule": "RUN", "veto": None,
                                                             "concerns": [], "corrected": True,
                                                             "params": {"harmony_params": {"theta": 1.0}}}
    codex = S.batch_decision(tmp_path / "codex", "codex")
    assert codex["corrected"] and (codex["call"], codex["requested_params"]) == ("c1", {"theta": 2.0})


def test_verdicts_and_stage_gt_deltas_against_the_control(scored):
    result = scored["results"]["arm"]
    full = result["full_metrics"]
    assert full["verdicts"]["macro_f1"] == "LOSS" and full["verdicts"]["mAP"] == "NO_RESOLVABLE_DIFFERENCE"
    from benchmarks.full_CPJUMP1.harness.data import read_json
    images = read_json(scored["prep"] / "public" / "images.json")
    excluded = [n for n in range(288) if n % 2 == 0 and n % 3]                        # B excluded where present
    assert len(read_json(scored["arm"] / "excluded_cells.json")) == len(excluded)
    # full_metrics counts the excluded cells of heldout units only
    assert full["n_excluded_cells"] == sum(images[n]["usage"] == S.HELDOUT for n in excluded)
    gt = result["stage_gt"]
    assert gt["control"] == "compared"
    missed = len(range(0, 288, 3))
    ap = gt["segmentation"]["all"]["AP50"]
    assert ap["value"] == pytest.approx(1 - 0.5 * missed / 288)
    assert ap["vs_control"]["control"] == 1.0 and ap["vs_control"]["delta"] == pytest.approx(-0.5 * missed / 288)
    assert ap["vs_control"]["se"] > 0 and ap["vs_control"]["ci95"][1] < 0
    assert gt["segmentation"]["all"]["count_ratio"]["value"] == pytest.approx(1 - missed / 576)
    # GT B recalled only where it was segmented and not excluded (n % 3 and n % 2 both nonzero)
    recalled_b = sum(1 for n in range(288) if n % 3 and n % 2)
    crop = gt["crop"]["all"]
    assert crop["gt_recall"]["value"] == pytest.approx((288 + recalled_b) / 576)
    assert crop["gt_recall"]["vs_control"]["delta"] < 0 and crop["gt_recall"]["vs_control"]["se"] > 0
    assert crop["unmatched_crop_rate"]["value"] == 0 and crop["retention"]["value"] == 1
    assert crop["retention"]["vs_control"]["delta"] == 0 and crop["retention"]["vs_control"]["se"] == 0
    assert "(-0." in (scored["arm"] / "RESULT.md").read_text(encoding="utf-8")


def test_tampered_artifacts_and_records_are_refused(scored, tmp_path, monkeypatch):
    import shutil
    from benchmarks.full_CPJUMP1.harness.data import read_json, write_json
    small(monkeypatch)
    prep = scored["prep"]
    run = tmp_path / "run"
    shutil.copytree(scored["control"], run)
    segmentation = read_json(run / "segmentation.json")
    for row in segmentation:                                   # the copy's own masks
        row["mask"] = str(run / Path(row["mask"]).relative_to(scored["control"].resolve()))
    write_json(run / "segmentation.json", segmentation)
    delivery = deliver(prep, run)
    prepared, images = read_json(prep / "PREPARE.json"), read_json(prep / "public" / "images.json")
    S.stage_gt_table(prep, run, delivery, prepared, images)   # the copy is valid

    with (run / "corrected.npy").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        S.score(prep, run)
    np.save(run / "corrected.npy", np.load(scored["control"] / "corrected.npy"))

    unbound = {**delivery, "artifacts": {k: v for k, v in delivery["artifacts"].items() if not k.startswith("segment/")}}
    with pytest.raises(ValueError, match="not bound to the delivery"):
        S.stage_gt_table(prep, run, unbound, prepared, images)
    outside = [{**segmentation[0], "mask": str(scored["control"] / "segmentation.json")}, *segmentation[1:]]
    write_json(run / "segmentation.json", outside)
    with pytest.raises(ValueError, match="outside the run"):
        S.stage_gt_table(prep, run, delivery, prepared, images)
    write_json(run / "segmentation.json", segmentation[1:])
    with pytest.raises(ValueError, match="does not cover"):
        S.stage_gt_table(prep, run, delivery, prepared, images)
    write_json(run / "segmentation.json", segmentation)

    cells = read_json(run / "cells.json")
    for edit, match in ((lambda c: c[0].update(mask_label=9), "not in its delivered mask"),
                        (lambda c: c.append(dict(c[0])), "delivered twice"),
                        (lambda c: c[0].update(well="unit_9999"), "outside the preparation"),
                        (lambda c: c[0].update(crop_bbox=[0, 0, 20, 20]), "outside the image")):
        changed = [dict(c) for c in cells]
        edit(changed)
        write_json(run / "cells.json", changed)
        with pytest.raises(ValueError, match=match):
            S.stage_gt_table(prep, run, delivery, prepared, images)
    write_json(run / "cells.json", cells)

    other = tmp_path / "prep"
    shutil.copytree(prep, other)
    first = read_json(other / "private" / "gt_index.json")[0]["image_id"]
    from benchmarks.full_LIVECell.harness.data import write_npz
    write_npz(other / "private" / "gt" / f"{first}.npz", labels=label_image(B, A))
    with pytest.raises(ValueError, match="private GT changed"):
        S.stage_gt_table(other, run, delivery, read_json(other / "PREPARE.json"), images)
    write_json(other / "private" / "gt_index.json", [])
    with pytest.raises(ValueError, match="GT index changed"):
        S.stage_gt_table(other, run, delivery, read_json(other / "PREPARE.json"), images)


def test_a_control_scored_on_other_groups_is_refused(scored, tmp_path, monkeypatch):
    import shutil
    from benchmarks.full_CPJUMP1.harness.data import read_json, write_json
    small(monkeypatch)
    control = tmp_path / "control"
    shutil.copytree(scored["control"] / "evaluation", control / "evaluation")
    saved = read_json(control / "evaluation" / S.GT_TABLE)
    write_json(control / "evaluation" / S.GT_TABLE, {**saved, "fingerprint": "other"})
    with pytest.raises(ValueError, match="different GT groups"):
        S.stage_gt(scored["prep"], scored["control"], read_json(scored["control"] / "DELIVERY.json"),
                   read_json(scored["prep"] / "PREPARE.json"), read_json(scored["prep"] / "public" / "images.json"),
                   pd.DataFrame(read_json(scored["prep"] / "private" / "labels.json")), control)


def test_a_part_whose_test_type_is_absent_from_its_train_side_fails_closed():
    meta = small_meta(types=TYPES, wells=3)
    S.full_layout(meta)                                                          # every type on both sides: fine
    only_one_well = meta[~((meta.cell_type == "BV2") & (meta.pool == S.FIT) & (meta.well_index != 0))]
    with pytest.raises(ValueError, match=r"guard part 0: test cell types \['BV2'\]"):
        S.full_layout(only_one_well)                                              # BV2's only fit well is index 0
    no_fit_type = meta[~((meta.cell_type == "BV2") & (meta.pool == S.FIT))]
    with pytest.raises(ValueError, match="main part 0: test cell types"):
        S.full_layout(no_fit_type)
