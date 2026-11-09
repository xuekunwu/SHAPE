"""extra_readouts.py on synthetic preparations built with test_score's helpers (the SMALL design, 12 x 12 images)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks._shared import full_metrics as fm
from benchmarks.full_LIVECell.harness import score as S
from benchmarks.full_LIVECell.runners import extra_readouts as X
from benchmarks.full_LIVECell.runners.test import test_score as T


# ---- pure pieces on hand-computed values ----------------------------------------------------------------------------

def test_prf_of_a_known_confusion_matrix():
    conf = np.array([[2.0, 1.0], [0.0, 3.0]])
    r = X.prf(conf, np.array([1.0, 0.0]))          # one test sample of class 0 without a prediction
    assert r["support"].tolist() == [4, 3]
    assert r["precision"].tolist() == pytest.approx([1.0, 3 / 4])
    assert r["recall"].tolist() == pytest.approx([2 / 4, 1.0])
    assert r["f1"].tolist() == pytest.approx([4 / 6, 6 / 7])
    empty = X.prf(np.zeros((2, 2)), np.zeros(2))   # zero denominators give 0, not NaN
    assert empty["precision"].tolist() == [0, 0] and empty["f1"].tolist() == [0, 0]


def ranks(score, pos, weights=None, keep=None, base=None, owner=None):
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    n = len(score)
    return X.rank_metrics(score, pos, np.ones(n, bool) if keep is None else np.asarray(keep),
                          np.ones(n) if base is None else np.asarray(base, float),
                          np.ones((1, 1)) if weights is None else weights, np.zeros(n, int) if owner is None else owner)


def test_auroc_and_auprc_of_known_scores():
    r = ranks([0.9, 0.8, 0.7, 0.6, 0.5], [1, 0, 1, 0, 1])
    assert r["auroc"][0] == pytest.approx(3 / 6)               # 3 of the 6 (positive, negative) pairs ordered
    assert r["auprc"][0] == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    assert (r["pos"][0], r["neg"][0]) == (3, 2)
    perfect = ranks([3, 2, 1, 0], [1, 1, 0, 0])
    assert perfect["auroc"][0] == 1 and perfect["auprc"][0] == 1
    tied = ranks([1, 1, 1, 1], [1, 0, 0, 0])                    # one threshold: AUROC 0.5, AP = prevalence
    assert tied["auroc"][0] == pytest.approx(0.5) and tied["auprc"][0] == pytest.approx(0.25)
    # a positive miss (score -inf) ranks below every negative; a negative miss is dropped by `keep`
    miss = ranks([0.9, 0.8, 0.3, -np.inf, -np.inf], [1, 0, 0, 1, 0], keep=[1, 1, 1, 1, 0])
    assert miss["auroc"][0] == pytest.approx(0.5) and miss["auprc"][0] == pytest.approx(0.5 * 1 + 0.5 * 0.5)
    assert np.isnan(ranks([0.2, 0.1], [1, 1])["auroc"][0])      # no negative: undefined


def test_rank_metrics_match_sklearn_with_ties_and_sample_weights():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(3)
    n, units = 80, 7
    score, owner = np.round(rng.random(n), 1), rng.integers(0, units, n)
    pos, base = rng.random(n) < 0.3, rng.integers(1, 4, n).astype(float)
    weights = np.vstack([np.ones(units), rng.integers(0, 3, (4, units))])
    r = X.rank_metrics(score, pos, np.ones(n, bool), base, weights, owner)
    for row, w in enumerate(weights):
        sample = base * w[owner]
        assert r["auroc"][row] == pytest.approx(roc_auc_score(pos, score, sample_weight=sample))
        assert r["auprc"][row] == pytest.approx(average_precision_score(pos, score, sample_weight=sample))


def test_knn_votes_and_logreg_proba_reproduce_the_verdict_probes():
    rng = np.random.default_rng(0)
    train, test = np.round(rng.normal(size=(60, 4)), 1), np.round(rng.normal(size=(25, 4)), 1)
    y = rng.integers(0, 3, 60)
    votes, predicted = X.knn_votes(train, y, test, 4)
    assert predicted.tolist() == fm.knn_predict(train, y, test, 4).tolist()
    assert votes.sum(axis=1) == pytest.approx(1.0) and set(np.unique(votes * 10)) <= set(range(11))
    proba = X.logreg_proba(train, y, test, 4)
    assert proba.sum(axis=1) == pytest.approx(1.0) and (proba[:, 3] == 0).all()
    assert proba.argmax(axis=1).tolist() == fm.logreg_predict(train, y, test)[0].tolist()
    single, labels = X.class_scores("knn", train, np.ones(60, int), test, 4)
    assert (labels == 1).all() and (single[:, 1] == 1).all()


def test_matching_picks_the_highest_iou_then_the_lowest_gt_index():
    gt = T.label_image(T.A, T.B)
    pred = T.label_image((0, 0, 4, 2), (6, 6, 10, 10), (10, 10, 12, 12))     # half of A, B, background-only
    target, iou = X.best_targets(S.overlap(gt, pred))
    assert target.tolist() == [0, 1, -1] and iou.tolist() == pytest.approx([0.5, 1.0, 0.0])


def test_regression_statistics_and_bootstrap_additivity():
    from sklearn.metrics import mean_squared_error, r2_score
    rng = np.random.default_rng(1)
    y = rng.normal(size=(40, 2))
    p = y * 0.7 + rng.normal(scale=0.3, size=(40, 2))
    unit = rng.integers(0, 5, 40)
    stats = X.group_stats(y, p, unit, 5)
    total = X.regression_scores(stats.sum(axis=0))
    for t in range(2):
        assert total["r2"][t] == pytest.approx(r2_score(y[:, t], p[:, t]))
        assert total["pearson_r"][t] == pytest.approx(np.corrcoef(y[:, t], p[:, t])[0, 1])
        assert total["mse"][t] == pytest.approx(mean_squared_error(y[:, t], p[:, t]))
    twice = X.regression_scores(2 * stats.sum(axis=0))                     # every unit drawn twice: the same values
    assert twice["r2"] == pytest.approx(total["r2"])
    constant = X.regression_scores(X.group_stats(np.ones((4, 1)), np.arange(4.0)[:, None], np.zeros(4, int), 1)[0])
    assert np.isnan(constant["r2"][0]) and np.isnan(constant["pearson_r"][0])


def test_ridge_recovers_a_linear_target_and_picks_alpha_on_the_fit_set_only():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 5)) * [1, 2, 3, 4, 5] + 7
    y = np.stack([x @ [1, 0, -1, 2, 0.5] + 3, x @ [0, 1, 0, 0, 0]], axis=1)
    groups = np.repeat(np.arange(20), 10)
    model = X.ridge_fit(x, y, groups)
    assert model["alpha"].tolist() == [min(X.RIDGE_ALPHAS)] * 2    # no noise: the least shrinkage wins
    assert np.abs(X.ridge_predict(model, x) - y).max() < 1e-2
    again = X.ridge_fit(x, y, groups)
    assert np.array_equal(again["coef"], model["coef"])
    with pytest.raises(ValueError, match="two heldout_fit units"):
        X.ridge_fit(x, y, np.zeros(200, int))


def leakage_meta(types=8, wells=3):
    rows = [dict(cell_type=f"T{k}", well_index=w, stage=s) for k in range(types) for w in range(wells)
            for s in range(6)]
    return pd.DataFrame(rows)


def test_stage_leakage_is_chance_on_random_profiles_and_one_on_stage_encoded_ones():
    meta = leakage_meta()
    rng = np.random.default_rng(5)
    random = rng.normal(size=(len(meta), 16))
    encoded = np.eye(6)[meta.stage.to_numpy()] * 5 + rng.normal(scale=0.1, size=(len(meta), 6))
    out = X.stage_leakage(meta, {"uncorrected": encoded, "delivered": random})
    assert out["chance"] == pytest.approx(1 / 6) and set(out["per_type_units"].values()) == {18}
    for classifier in ("knn", "logreg"):
        assert out["variants"]["uncorrected"][classifier]["macro_balanced_accuracy"]["value"] == pytest.approx(1.0)
        assert out["variants"]["uncorrected"][classifier]["per_type"]["T3"]["value"] == pytest.approx(1.0)
        chance = out["variants"]["delivered"][classifier]["macro_balanced_accuracy"]
        assert abs(chance["value"] - 1 / 6) < 0.1 and chance["ci95"][0] < chance["value"] < chance["ci95"][1]
        delta = out["delta_delivered_minus_uncorrected"][classifier]
        assert delta["value"] < -0.6 and delta["ci95"][1] < 0
    assert X.stage_leakage(meta, {"uncorrected": encoded, "delivered": random}) == out       # deterministic


def test_leave_one_well_index_out_never_trains_on_the_tested_well():
    meta = leakage_meta(types=2, wells=2)
    x = np.eye(6)[meta.stage.to_numpy()] + 0.0
    x[meta.well_index.to_numpy() == 1] = np.roll(x[meta.well_index.to_numpy() == 1], 1, axis=1)   # well 1: classes shifted
    out = X.stage_leakage(meta, {"uncorrected": x})
    # the shifted well is a different code: training on well 0 only, every test unit of well 1 is mispredicted
    assert out["variants"]["uncorrected"]["knn"]["macro_balanced_accuracy"]["value"] == 0.0


# ---- synthetic runs ----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    """The SMALL preparation, a degraded run (noisy cells, every third image without cell B, every second image's
    cell B excluded), scored by the harness for the parity checks, and the extra readouts."""
    root = tmp_path_factory.mktemp("extra")
    prep = T.make_prep(root / "prep")
    run = T.make_run(prep, root / "run", cell_noise=1.0, seed=1, miss_every=3, exclude_every=2, arm="shape")
    with pytest.MonkeyPatch.context() as mp:
        T.small(mp)
        scored = S.score(prep, run)
        result = X.extra_readouts(prep, run, workers=1)
    return {"root": root, "prep": prep, "run": run, "scored": scored, "result": result}


def test_the_probes_reproduce_the_harness_verdict_probes(plain):
    probes, cls = plain["scored"]["full_metrics"]["probes"], plain["result"]["classification"]["probes"]
    for level, classifier in X.PROBES:
        mine, theirs = cls[f"{level}_{classifier}"], probes[level][classifier]["main"]
        for metric, key in (("macro_f1", "macro_f1"), ("balanced_accuracy", "balanced_accuracy")):
            assert mine[metric]["value"] == pytest.approx(theirs[key]["value"], abs=1e-5)
            assert mine[metric]["ci95"] == pytest.approx(theirs[key]["ci95"], abs=1e-5)
    assert cls["cell_knn"]["macro_f1"]["value"] == pytest.approx(
        plain["scored"]["full_metrics"]["metrics"]["macro_f1"]["main"]["value"], abs=1e-5)
    assert plain["result"]["classification"]["n_excluded_sampled_test"] > 0


def test_classification_readouts_are_consistent(plain):
    cls = plain["result"]["classification"]
    assert cls["classes"] == sorted(T.TYPES) and cls["n_test_units"] == 48 and cls["n_fit_units"] == 48
    assert plain["result"]["rule"] == S.MAIN_RULE and plain["result"]["layout"]["splits"]["main"]["n_test_units"] == [48]
    for name, p in cls["probes"].items():
        conf = np.array(p["confusion"]["rows_true_columns_predicted"])
        miss = np.array(p["confusion"]["misses_per_true_class"])
        assert conf.sum() + miss.sum() == p["n_test_samples"] and miss.sum() == p["n_misses"]
        for k, c in enumerate(cls["classes"]):
            e = p["per_class"][c]
            assert e["support"] == conf[k].sum() + miss[k]
            assert e["recall"]["value"] == pytest.approx(conf[k, k] / e["support"], abs=1e-5)
            assert e["precision"]["value"] == pytest.approx(conf[k, k] / conf[:, k].sum(), abs=1e-5)
            assert 0 <= e["auroc"]["value"] <= 1 and 0 < e["auprc"]["value"] <= 1
            assert 0 < e["auprc_chance"] < 1
        assert p["macro_auroc"]["ci95"][0] <= p["macro_auroc"]["value"] <= p["macro_auroc"]["ci95"][1]
        tops = p["top_confusions"]
        assert [t["count"] for t in tops] == sorted((t["count"] for t in tops), reverse=True)
        off = conf - np.diag(np.diag(conf))
        assert (tops[0]["count"] if tops else 0) == off.max()
        assert all(t["true"] != t["predicted"] for t in tops)
    # the unit-level probes: type-perfect profiles, so a perfect classification of the units with cells
    for name in ("unit_knn", "unit_logreg"):
        p = cls["probes"][name]
        assert p["macro_auroc"]["value"] == pytest.approx(1.0) and p["n_misses"] == 0


def test_auroc_of_a_noisy_cell_probe_is_above_chance_and_below_one(plain):
    p = plain["result"]["classification"]["probes"]["cell_logreg"]
    assert 0.6 < p["macro_auroc"]["value"] < 1.0 and p["macro_auprc"]["value"] > p["chance"]["auprc_macro_prevalence"]
    assert p["chance"]["auroc"] == 0.5


def test_stage_leakage_on_the_synthetic_run_reports_both_variants(plain):
    leak = plain["result"]["stage_leakage"]
    assert set(leak["variants"]) == {"uncorrected", "delivered"} and leak["n_units"] == 144
    # type-only profiles carry no stage information: below the all-stage level
    assert leak["variants"]["delivered"]["logreg"]["macro_balanced_accuracy"]["value"] < 0.5


def test_markdown_and_json_are_complete_and_deterministic(plain, tmp_path, monkeypatch):
    T.small(monkeypatch)
    out = tmp_path / "x" / "extra.json"
    assert X.main(["--prep", str(plain["prep"]), "--run", str(plain["run"]), "--out", str(out), "--md",
                   "--workers", "1"]) == 0
    second = tmp_path / "second.json"
    assert X.main(["--prep", str(plain["prep"]), "--run", str(plain["run"]), "--out", str(second), "--workers", "1"]) == 0
    assert out.read_bytes() == second.read_bytes()
    assert json.loads(out.read_text(encoding="utf-8")) == plain["result"]
    assert b"\r" not in out.read_bytes() and b"NaN" not in out.read_bytes()
    text = out.with_suffix(".md").read_text(encoding="utf-8")
    assert b"\r" not in out.with_suffix(".md").read_bytes()
    for needle in ("## 1-2. Classification probes", "### cell_knn (the verdict probe)", "Largest confusions:",
                   "## 3. Cell morphology", "## 4. Stage leakage", "chance 0.167", "Confusion (rows true"):
        assert needle in text


def test_a_tampered_delivery_or_preparation_is_refused(plain, tmp_path, monkeypatch, capsys):
    T.small(monkeypatch)
    run, prep = tmp_path / "run", tmp_path / "prep"
    shutil.copytree(plain["run"], run)
    shutil.copytree(plain["prep"], prep)
    # a copied run keeps its original absolute mask paths: the masks are found inside the copy by their bound hashes
    X.load_run(prep, run)
    with (run / "corrected.npy").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="delivered artifact changed: corrected.npy"):
        X.load_run(prep, run)
    out = tmp_path / "never.json"
    assert X.main(["--prep", str(prep), "--run", str(run), "--out", str(out)]) == 2
    assert "refused: delivered artifact changed" in capsys.readouterr().err and not out.exists()
    np.save(run / "corrected.npy", np.load(plain["run"] / "corrected.npy"))
    delivery = json.loads((run / "DELIVERY.json").read_text(encoding="utf-8"))
    (run / "DELIVERY.json").write_text(json.dumps({**delivery, "smoke": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="complete pipeline"):
        X.load_run(prep, run)
    (run / "DELIVERY.json").write_text(json.dumps(
        {**delivery, "artifacts": {k: v for k, v in delivery["artifacts"].items() if k != "embedding.npy"}}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="lacks"):
        X.load_run(prep, run)
    (run / "DELIVERY.json").write_text(json.dumps(delivery), encoding="utf-8")
    X.load_run(prep, run)
    (prep / "private" / "labels.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="labels changed"):
        X.load_run(prep, run)
    shutil.copy(plain["prep"] / "private" / "labels.json", prep / "private" / "labels.json")
    prepared = json.loads((prep / "PREPARE.json").read_text(encoding="utf-8"))
    (prep / "PREPARE.json").write_text(json.dumps({**prepared, "note": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="another preparation"):
        X.load_run(prep, run)


# ---- morphology: matched GT instances -----------------------------------------------------------------------------

def varied_gt(n):
    """GT of image n: cell 1 a rectangle at the top left, cell 2 a rectangle (3-6 px) at the bottom right, notched
    (a 2 x 2 corner removed: solidity < 1) in every third image."""
    h1, w1, h2, w2 = 2 + n % 4, 2 + (n // 4) % 4, 3 + (n // 2) % 4, 3 + (n // 3) % 4
    gt = np.zeros(T.GT_SHAPE, np.uint32)
    gt[0:h1, 0:w1], gt[6:6 + h2, 6:6 + w2] = 1, 2
    if n % 3 == 0:
        gt[6:8, 6 + w2 - 2:6 + w2] = 0
    return gt


def expected_targets(gt):
    from skimage.measure import regionprops_table
    t = regionprops_table(gt.astype(np.int64), properties=("label", "area", "major_axis_length", "minor_axis_length",
                                                            "solidity"))
    return {int(l): [np.log(a), M / m, s] for l, a, M, m, s in zip(t["label"], t["area"], t["major_axis_length"],
                                                                    t["minor_axis_length"], t["solidity"])}


MIX = np.random.default_rng(11).normal(size=(3, 5))


@pytest.fixture(scope="module")
def varied(tmp_path_factory):
    """A SMALL preparation with a different GT per image, masks = the GT (the cell-2 mask of every fifth image moved
    off its GT instance: an unmatched cell), cell embeddings = the type one-hot and a linear mix of the matched
    GT targets (zeros for an unmatched cell)."""
    import tifffile
    from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
    from benchmarks.full_LIVECell.harness.data import write_npz
    root = tmp_path_factory.mktemp("varied")
    prep = T.make_prep(root / "prep")
    images = read_json(prep / "public" / "images.json")
    index = read_json(prep / "private" / "gt_index.json")
    for n, (row, entry) in enumerate(zip(images, index)):
        path = prep / "private" / entry["file"]
        write_npz(path, labels=varied_gt(n), n_instances=np.int64(2), overlap_pixels=np.int64(0))
        entry["sha256"] = digest(path)
    write_json(prep / "private" / "gt_index.json", index)
    prepared = read_json(prep / "PREPARE.json")
    write_json(prep / "PREPARE.json", {**prepared, "gt_index_sha256": digest(prep / "private" / "gt_index.json")})
    run = T.make_run(prep, root / "run", cell_noise=0.0)
    order = {r["image_id"]: n for n, r in enumerate(images)}
    for row in read_json(run / "segmentation.json"):
        n = order[row["image_id"]]
        mask = varied_gt(n)
        if n % 5 == 0:
            mask[mask == 2] = 0
            mask[0:3, 8:11] = 2
        tifffile.imwrite(row["mask"], mask)
    write_json(run / "segmentation.json", [
        {**r, "sha256": digest(Path(r["mask"]))} for r in read_json(run / "segmentation.json")])
    cells = read_json(run / "cells.json")
    kind = {r["image_id"]: r["cell_type"] for r in read_json(prep / "private" / "labels.json")}
    rows = []
    for c in cells:
        n = order[c["image_id"]]
        target = expected_targets(varied_gt(n))[c["mask_label"]]
        unmatched = n % 5 == 0 and c["mask_label"] == 2
        rows.append(np.concatenate([np.eye(8)[T.TYPES.index(kind[c["image_id"]])],
                                    np.zeros(5) if unmatched else np.array(target) @ MIX]))
    np.save(run / "embedding.npy", np.array(rows, dtype=np.float32))
    T.deliver(prep, run)
    with pytest.MonkeyPatch.context() as mp:
        T.small(mp)
        serial = X.extra_readouts(prep, run, workers=1)
        parallel = X.extra_readouts(prep, run, workers=2)
    return {"prep": prep, "run": run, "images": images, "serial": serial, "parallel": parallel, "order": order}


def test_morphology_counts_the_unmatched_cells_it_leaves_out(varied):
    counts = varied["serial"]["morphology"]["counts"]
    labels = pd.DataFrame(json.loads((varied["prep"] / "private" / "labels.json").read_text(encoding="utf-8")))
    stage = dict(zip(labels.image_id, labels.stage))
    for pool, stages in (("heldout_fit", (2, 3)), ("heldout_test", (4, 5))):
        ids = [i for i in varied["order"] if stage[i] in stages]
        moved = sum(varied["order"][i] % 5 == 0 for i in ids)
        assert len(ids) == 48 * 2 and counts[pool]["n_delivered"] == 2 * len(ids)
        assert counts[pool]["n_unmatched_left_out"] == moved and counts[pool]["n_used"] == 2 * len(ids) - moved
        assert counts[pool]["n_matched"] == 2 * len(ids) - moved
        assert counts[pool]["n_matched_with_undefined_target_left_out"] == 0
        assert counts[pool]["n_gt_instances"] == 2 * len(ids) and counts[pool]["gt_instances_matched"] == 2 * len(ids) - moved
    per_type = varied["serial"]["morphology"]["counts_per_type"]
    assert sum(t["heldout_test"]["n_unmatched_left_out"] for t in per_type.values()) == counts["heldout_test"]["n_unmatched_left_out"]


def test_morphology_ridge_recovers_the_matched_gt_targets(varied):
    morph = varied["serial"]["morphology"]
    assert morph["targets"] == ["log_area", "aspect_ratio", "solidity"]
    assert morph["overall"]["n_cells"] == varied["serial"]["morphology"]["counts"]["heldout_test"]["n_used"]
    for target in morph["targets"]:
        e = morph["overall"][target]
        assert e["r2"]["value"] > 0.999 and e["pearson_r"]["value"] > 0.999 and e["mse"]["value"] < 1e-3
    for kind, group in morph["per_type"].items():
        for target in ("log_area", "aspect_ratio"):
            assert group[target]["r2"]["value"] > 0.99
    assert any(g["solidity"]["r2"]["value"] is not None for g in morph["per_type"].values())


def test_matched_targets_are_the_gt_instances_values(varied):
    """A 4 x 2 rectangle: area 8, axes sqrt(1.25) and sqrt(0.25) times 4, aspect sqrt(5); a square has aspect 1."""
    gt = np.zeros(T.GT_SHAPE, np.uint32)
    gt[0:2, 0:4], gt[6:10, 6:10] = 1, 2
    t = expected_targets(gt)
    assert t[1] == pytest.approx([np.log(8), np.sqrt(5), 1.0]) and t[2] == pytest.approx([np.log(16), 1.0, 1.0])
    assert expected_targets(varied_gt(3))[2][2] < 1 and expected_targets(varied_gt(1))[2][2] == 1   # the notched one


def test_workers_do_not_change_the_result(varied):
    assert varied["parallel"] == varied["serial"]


def test_a_cell_naming_a_mask_label_outside_its_mask_is_refused(varied, tmp_path, monkeypatch):
    from benchmarks.full_CPJUMP1.harness.data import read_json, write_json
    T.small(monkeypatch)
    run = tmp_path / "run"
    shutil.copytree(varied["run"], run)
    cells = read_json(run / "cells.json")
    next(c for c in cells if c["usage"] == S.HELDOUT)["mask_label"] = 9
    write_json(run / "cells.json", cells)
    T.deliver(varied["prep"], run)
    with pytest.raises(ValueError, match="not in its delivered mask"):
        X.extra_readouts(varied["prep"], run, workers=1)


# ---- readout 5: batch integration ----------------------------------------------------------------------------------

def integration_profiles(seed=0, kinds=8, per=5, dims=16):
    """kinds x 6 stages x `per` units: a type centroid, a stage shift (the batch effect) and noise; the corrected
    version has no stage shift. Returns (meta, uncorrected, corrected)."""
    rng = np.random.default_rng(seed)
    rows = [(k, s) for k in range(kinds) for s in range(6) for _ in range(per)]
    kind, stage = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    centroid, shift = rng.normal(size=(kinds, dims)) * 3, rng.normal(size=(6, dims)) * 6
    noise = rng.normal(size=(len(rows), dims))
    corrected = centroid[kind] + noise
    meta = pd.DataFrame({"cell_type": [f"T{k}" for k in kind], "plate": [f"stage_{s}" for s in stage],
                         "usage": np.where(stage < 2, "train", S.HELDOUT)})
    return meta, corrected + shift[stage], corrected


def test_a_stage_shifted_set_integrates_worse_than_its_corrected_version():
    pytest.importorskip("scib_metrics")
    meta, shifted, corrected = integration_profiles()
    record = {"admission": {"decision": "RUN"}, "evidence": {"gate_all_rows": {
        "batch_predictability": 0.68, "chance_accuracy": 1 / 6, "effect_margin": 0.51, "detected": True,
        "reason": "r", "n_samples": 240, "n_batches": 6, "extra": 1}}}
    out = X.batch_integration(meta, {"uncorrected": shifted, "delivered": corrected}, record)
    assert out["metrics"] == ["batch_ASW", "iLISI", "graph_conn", "cLISI", "label_ASW", "NMI", "ARI"]
    assert {k: v["n_units"] for k, v in out["row_sets"].items()} == {"all_units": 240, "heldout_units": 160}
    for name, e in out["row_sets"].items():
        before, after = (e["variants"][v]["values"] for v in ("uncorrected", "delivered"))
        assert e["variants"]["uncorrected"]["unavailable"] == {} and e["n_batch_groups"] == (6 if name == "all_units" else 4)
        assert all(0 <= v <= 1 for v in {**before, **after}.values())
        assert after["batch_ASW"] > before["batch_ASW"] + 0.05 and after["iLISI"] > before["iLISI"] + 0.05
        assert after["label_ASW"] > before["label_ASW"]
        delta = e["delta_delivered_minus_uncorrected"]
        assert delta["batch_ASW"] == pytest.approx(after["batch_ASW"] - before["batch_ASW"]) and delta["iLISI"] > 0
    assert out["admission_evidence"] == {"decision": "RUN", "gate_all_rows": {
        "batch_predictability": 0.68, "chance_accuracy": 1 / 6, "effect_margin": 0.51, "detected": True, "reason": "r",
        "n_samples": 240, "n_batches": 6}}
    again = X.batch_integration(meta, {"uncorrected": shifted, "delivered": corrected}, record)
    assert again == out                                                    # deterministic


def test_an_unavailable_metric_is_null_with_its_reason_and_the_rest_is_kept(plain):
    from types import SimpleNamespace
    from shape.tools.batch_corrector import quant
    calls = []

    def unavailable(*args, **kwargs):
        raise quant.MetricUnavailable("no label has enough cells in at least two batches")

    def missing_package(*args, **kwargs):
        raise ImportError("No module named 'scib_metrics'")

    def fake(value):
        def metric(feats, *args, **kwargs):
            calls.append((type(args[0]).__name__, kwargs))
            return value + float(np.asarray(feats).mean())
        return metric

    fake_quant = SimpleNamespace(_safe=quant._safe, MetricUnavailable=quant.MetricUnavailable,
                                 silhouette_batch=unavailable, lisi_batch=fake(0.5), graph_conn=missing_package,
                                 lisi_label=fake(0.25), asw=fake(0.75), nmi_ari=unavailable)
    meta, shifted, corrected = integration_profiles(kinds=2, per=2)
    out = X.batch_integration(meta, {"uncorrected": shifted, "delivered": corrected}, None, quant=fake_quant)
    e = out["row_sets"]["all_units"]
    for variant in ("uncorrected", "delivered"):
        v = e["variants"][variant]
        assert [m for m, x in v["values"].items() if x is None] == ["batch_ASW", "graph_conn", "NMI", "ARI"]
        assert "no label has enough cells" in v["unavailable"]["batch_ASW"] and "silhouette_batch" not in v["values"]
        assert "ImportError" in v["unavailable"]["graph_conn"] and set(v["unavailable"]) == {
            "batch_ASW", "graph_conn", "NMI", "ARI"} and v["unavailable"]["NMI"] == v["unavailable"]["ARI"]
        assert v["values"]["iLISI"] == pytest.approx(0.5 + (shifted if variant == "uncorrected" else corrected).mean())
    assert e["delta_delivered_minus_uncorrected"]["batch_ASW"] is None and e["delta_delivered_minus_uncorrected"]["ARI"] is None
    assert e["delta_delivered_minus_uncorrected"]["iLISI"] == pytest.approx(corrected.mean() - shifted.mean())
    assert ("Series", {"n_neighbors": 25}) in calls                       # labels / batch as Series, k = 25
    assert out["admission_evidence"]["gate_all_rows"] is None and "no evidence.gate_all_rows" in out["admission_evidence"]["unavailable"]
    text = X.markdown({**plain["result"], "batch_integration": out})
    assert "n/a" in text and "unavailable (" in text


def test_the_run_reports_batch_integration_with_the_admission_evidence(plain):
    b = plain["result"]["batch_integration"]
    assert set(b["row_sets"]) == {"all_units", "heldout_units"} and b["admission_evidence"]["decision"] == "ABSTAIN"
    assert {k: v["n_units"] for k, v in b["row_sets"].items()} == {"all_units": 144, "heldout_units": 96}
    for e in b["row_sets"].values():
        assert set(e["variants"]) == {"uncorrected", "delivered"} and e["n_labels"] == 8
        for v in e["variants"].values():
            assert set(v["values"]) == set(b["metrics"]) and set(v["unavailable"]) <= set(b["metrics"])
            assert all(x is not None or m in v["unavailable"] for m, x in v["values"].items())
