"""CPU tests of the shared full_* metrics (full_metrics.py) on synthetic data with planted structure."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

from benchmarks._shared import full_metrics as fm


def synthetic(n_classes=4, units_per_class=6, cells_per_unit=5, signal=1.0, seed=0, dim=16):
    """Units on two 'plates' (alternating), each with cells around its class centre (signal) plus noise."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_classes, dim)) * 3
    labels = np.repeat(np.arange(n_classes), units_per_class)
    plate = np.tile(np.arange(2), len(labels) // 2)
    cell_unit = np.repeat(np.arange(len(labels)), cells_per_unit)
    cells = signal * centres[labels[cell_unit]] + rng.normal(size=(len(cell_unit), dim))
    return labels, plate, cells, cell_unit


def plate_layout(labels, plate):
    a, b = np.flatnonzero(plate == 0), np.flatnonzero(plate == 1)
    folds = [[(a, b)], [(b, a)]]
    return fm.Layout([f"u{i}" for i in range(len(labels))], [f"c{v}" for v in labels], {"main": folds, "guard": folds})


def test_planted_classes_are_recovered():
    labels, plate, cells, unit = synthetic(signal=1.0)
    reading = fm.evaluate(plate_layout(labels, plate), cells, unit)
    for key in ("main/mAP", "main/cell/knn/macro_f1", "main/cell/knn/balanced_accuracy", "main/unit/knn/macro_f1",
                "main/cell/logreg/balanced_accuracy", "main/unit/logreg/macro_f1"):
        assert reading["points"][key] == pytest.approx(1.0), key
    assert reading["points"]["main/chance/mAP"] == pytest.approx(0.25)


def test_random_embedding_is_near_chance():
    labels, plate, cells, unit = synthetic(n_classes=5, units_per_class=40, signal=0.0, seed=1)
    layout = plate_layout(labels, plate)
    reading = fm.evaluate(layout, cells, unit)
    points = reading["points"]
    assert abs(points["main/cell/knn/balanced_accuracy"] - points["main/chance/cell/balanced_accuracy"]) < 0.08
    assert abs(points["main/mAP"] - points["main/chance/mAP"]) < 0.08
    assert points["main/chance/cell/balanced_accuracy"] == pytest.approx(0.2)


def test_scores_and_ap_match_sklearn():
    rng = np.random.default_rng(3)
    true, pred = rng.integers(0, 4, 50), rng.integers(0, 4, 50)
    confusion = np.zeros((4, 4))
    np.add.at(confusion, (true, pred), 1)
    got = fm.scores(confusion)
    assert got["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(true, pred))
    assert got["macro_f1"] == pytest.approx(f1_score(true, pred, average="macro"))
    sim = np.round(rng.normal(size=30), 1)                         # rounded: equal similarities
    positives = rng.random(30) < 0.3
    assert fm.average_precision(sim, positives) == pytest.approx(average_precision_score(positives, sim))


def test_bootstrap_is_deterministic_and_resamples_whole_units():
    labels, plate, cells, unit = synthetic(signal=0.3, seed=2)
    first, second = plate_layout(labels, plate), plate_layout(labels, plate)
    assert first.fingerprint == second.fingerprint
    for f in range(2):
        draws = first.draws["main"][f]
        assert draws.shape == (fm.BOOTSTRAP, len(first.tests["main"][f]))
        assert (draws == second.draws["main"][f]).all() and (draws.sum(axis=1) == draws.shape[1]).all()
    one, two = fm.evaluate(first, cells, unit), fm.evaluate(second, cells, unit)
    for key in one["draws"]:
        assert np.array_equal(one["draws"][key], two["draws"][key])


def test_cell_draw_equals_manual_whole_unit_resample():
    labels, plate, cells, unit = synthetic(signal=0.3, seed=4)
    layout = plate_layout(labels, plate)
    reading = fm.evaluate(layout, cells, unit, classifiers=("knn",))
    n = layout.n_classes
    values = []
    for f, fold in enumerate(layout.splits["main"]):
        (train, test), = fold
        tr, te = np.isin(unit, train), np.isin(unit, test)
        predicted = fm.knn_predict(cells[tr], labels[unit[tr]], cells[te], n)
        weight = dict(zip(layout.tests["main"][f], layout.draws["main"][f][7]))     # draw 7
        confusion = np.zeros((n, n))
        np.add.at(confusion, (labels[unit[te]], predicted), [weight[u] for u in unit[te]])
        values.append(fm.scores(confusion)["macro_f1"])
    assert reading["draws"]["main/cell/knn/macro_f1"][7] == pytest.approx(np.mean(values))


def test_grouped_folds_take_the_fewest_and_keep_classes_on_both_sides():
    labels = np.repeat(["a", "b", "c"], 8)
    groups = np.array([f"g{i // 2}" for i in range(24)])              # 2 units per group
    folds, n = fm.grouped_folds(labels, groups)
    assert n == 2 and len(folds) == 2
    for (train, test), in folds:
        assert not set(groups[train]) & set(groups[test])
        assert set(labels[test]) <= set(labels[train])
    with pytest.raises(ValueError, match="fewer than 2 groups"):
        fm.grouped_folds(np.array(["a", "a", "b"]), np.array(["g0", "g1", "g2"]))


def _reading(points: dict, draws: dict) -> dict:
    return {"fingerprint": "x", "points": points, "draws": draws}


@pytest.mark.parametrize("delta, guard, expected", [(0.1, 0.0, fm.GAIN), (0.1, -0.01, fm.SHORTCUT),
                                                    (-0.1, 0.5, fm.LOSS), (0.01, 0.5, fm.NO_RESOLVABLE)])
def test_verdict_rules(delta, guard, expected):
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.02, size=fm.BOOTSTRAP)                 # SE of the delta = 0.02
    control = _reading({"main/mAP": 0.5, "guard/mAP": 0.4}, {"main/mAP": 0.5 + noise * 0, "guard/mAP": np.zeros(10)})
    arm = _reading({"main/mAP": 0.5 + delta, "guard/mAP": 0.4 + guard},
                   {"main/mAP": 0.5 + delta + noise, "guard/mAP": np.zeros(10)})
    out = fm.verdict(arm, control, "mAP")
    assert out["verdict"] == expected and out["se"] == pytest.approx(0.02, rel=0.1)
    assert out["delta"] == pytest.approx(delta) and out["guard_delta"] == pytest.approx(guard)


def test_verdict_se_zero_and_layout_mismatch():
    same = _reading({"main/mAP": 0.5, "guard/mAP": 0.5}, {"main/mAP": np.full(10, 0.5), "guard/mAP": np.zeros(10)})
    better = _reading({"main/mAP": 0.6, "guard/mAP": 0.6}, {"main/mAP": np.full(10, 0.6), "guard/mAP": np.zeros(10)})
    out = fm.verdict(better, same, "mAP")
    assert out["verdict"] == fm.NO_RESOLVABLE and out["note"] == "se_zero"
    with pytest.raises(ValueError, match="different layouts"):
        fm.verdict({**better, "fingerprint": "y"}, same, "mAP")


def test_record_save_load_and_markdown(tmp_path):
    labels, plate, cells, unit = synthetic(signal=0.5, seed=5)
    layout = plate_layout(labels, plate)
    control = fm.evaluate(layout, cells, unit)
    fm.save(tmp_path / "draws.npz", control)
    loaded = fm.load(tmp_path / "draws.npz")
    assert loaded["points"] == control["points"]
    assert all(np.array_equal(loaded["draws"][k], v) for k, v in control["draws"].items())
    arm = fm.evaluate(layout, cells * 0 + np.random.default_rng(9).normal(size=cells.shape), unit)
    full = fm.record(layout, arm, loaded)
    assert set(full["verdicts"]) == set(fm.METRICS)
    assert full["verdicts"]["macro_f1"] == fm.LOSS
    for metric in fm.METRICS:
        entry = full["metrics"][metric]
        assert entry["main"]["ci95"][0] <= entry["main"]["value"] <= entry["main"]["ci95"][1]
        assert set(entry["chance"]) == {"main", "guard"}
    assert set(full["probes"]["cell"]) == {"knn", "logreg", "chance"}
    text = fm.markdown(full)
    assert text.count("| mAP") == 1 and fm.LOSS in text and "| cell knn |" in text
    assert fm.record(layout, control)["verdicts"] is None


def test_scores_with_misses_is_sklearn_over_the_class_labels():
    rng = np.random.default_rng(6)
    true, pred = rng.integers(0, 4, 60), rng.integers(0, 4, 60)
    confusion = np.zeros((4, 4))
    np.add.at(confusion, (true, pred), 1)
    plain, none = fm.scores(confusion), fm.scores_with_misses(confusion, np.zeros(4))
    assert all(plain[m] == pytest.approx(none[m]) for m in fm.PROBE_METRICS)
    misses = np.array([3, 0, 5, 0])                                # samples of class 0 and 2 without a prediction
    missed_true = np.concatenate([true, np.repeat([0, 2], [3, 5])])
    missed_pred = np.concatenate([pred, np.full(8, -1)])            # -1: no prediction, not a class
    got = fm.scores_with_misses(confusion, misses)
    assert got["macro_f1"] == pytest.approx(f1_score(missed_true, missed_pred, labels=[0, 1, 2, 3], average="macro"))
    recall = [np.mean(missed_pred[missed_true == c] == c) for c in range(4)]
    assert got["balanced_accuracy"] == pytest.approx(np.mean(recall))


def test_cap_cells_is_deterministic_per_unit_and_matches_the_hcma_sample():
    import pandas as pd

    from benchmarks.full_HCMA.harness.score import sample_cells

    rng = np.random.default_rng(7)
    unit = rng.integers(0, 5, 900)
    ids = np.array([f"cell_{i:04d}" for i in rng.permutation(900)])
    rows = fm.cap_cells(unit, ids, cap=100)
    assert np.bincount(unit[rows], minlength=5).tolist() == [min(100, c) for c in np.bincount(unit, minlength=5)]
    hcma = sample_cells(pd.DataFrame({"cell_id": ids, "image_id": [f"im{u}" for u in unit]}), 100)
    assert set(rows.tolist()) == set(hcma.tolist())
    assert np.array_equal(rows, fm.cap_cells(unit, ids, cap=100))
    small = fm.cap_cells(np.array([0, 0, 1, -1]), cap=100)
    assert sorted(small.tolist()) == [0, 1, 2]


def test_a_unit_without_cells_counts_as_misses():
    labels, plate, cells, unit = synthetic(signal=1.0, seed=8)
    layout = plate_layout(labels, plate)
    test_unit = layout.tests["main"][0][0]
    keep = unit != test_unit
    reading = fm.evaluate(layout, cells[keep], unit[keep], cap=5)
    assert reading["n_units_without_cells"] == 1
    points = reading["points"]
    for key in ("main/cell/knn/balanced_accuracy", "main/unit/knn/macro_f1", "main/mAP"):
        assert points[key] < 1, key
    # fold 0 has 12 test units over 4 classes (3 each): its class loses one of 3 units -> recall 2/3 in fold 0
    assert points["main/cell/knn/balanced_accuracy"] == pytest.approx((1 - 1 / 12 + 1) / 2)    # 5 misses of 15
    assert points["main/mAP"] == pytest.approx((1 - 1 / 12 + 1) / 2, abs=0.02)


def test_sampled_excluded_cells_are_misses():
    labels, plate, cells, unit = synthetic(signal=1.0, seed=8)
    layout = plate_layout(labels, plate)
    test_unit = layout.tests["main"][0][0]
    ids = np.array([f"c{i:04d}" for i in range(len(unit))])
    gone = np.flatnonzero(unit == test_unit)[:2]                     # 2 of its 5 cells excluded by the agent
    keep = np.setdiff1d(np.arange(len(unit)), gone)
    reading = fm.evaluate(layout, cells[keep], unit[keep], cell_ids=ids[keep], excluded_unit=unit[gone],
                          excluded_ids=ids[gone])
    assert reading["n_excluded_cells"] == 2 and reading["n_excluded_sampled"] == 2
    assert reading["n_units_without_cells"] == 0
    # fold 0: the class has 15 test cells, 2 of them misses; fold 1 unchanged
    assert reading["points"]["main/cell/knn/balanced_accuracy"] == pytest.approx((1 - (2 / 15) / 4 + 1) / 2)
    capped = fm.evaluate(layout, cells[keep], unit[keep], cell_ids=ids[keep], excluded_unit=unit[gone],
                         excluded_ids=ids[gone], cap=3)              # the cap samples delivered and excluded cells
    assert capped["n_cells_probed"] + capped["n_excluded_sampled"] == 3 * len(layout.unit_ids) - 0
    with pytest.raises(ValueError, match="ids"):
        fm.evaluate(layout, cells[keep], unit[keep], excluded_unit=unit[gone])
