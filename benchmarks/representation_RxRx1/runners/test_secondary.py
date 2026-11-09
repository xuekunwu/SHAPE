"""CPU tests of the A9 secondary metrics (secondary.py): macro F1 and balanced accuracy of the siRNA probe across
experiments, on small synthetic layouts with a known structure, and score.final's unchanged pre-A9 outputs."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from benchmarks.representation_RxRx1.runners import control, score, secondary, shape_arm  # noqa: E402
from benchmarks.representation_RxRx1.runners.test_benchmark import _shape_engine, prepared, stub  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
PRE_A9 = "3f70c446db80c54dd045d1f125e8420442f09c72"      # the last commit before A9 (score.py at A8)


def layout(n_sirna=6, seen=3, heldout=1, plates=2, cells_per_well=8, sizes=None):
    """One well per siRNA per plate; the last `heldout` experiments held out; `sizes` overrides cells per siRNA."""
    rows = []
    for e in range(seen + heldout):
        split = "heldout" if e >= seen else None
        for p in range(plates):
            for s in range(n_sirna):
                for c in range(cells_per_well if sizes is None else sizes[s]):
                    rows.append({"experiment": f"E{e}", "plate": f"E{e}_{p}", "well": f"E{e}_{p}/W{s}",
                                 "sirna": f"S{s}",
                                 "split": split or ("train", "val", "unused")[(p + s) % 3]})
    cells = pd.DataFrame(rows)
    return cells, cells.pop("sirna").to_numpy()


def planted(cells, sirna, d=16, signal=3.0, seed=0):
    rng = np.random.default_rng(seed)
    codes = np.unique(sirna, return_inverse=True)[1]
    experiment = np.unique(cells.experiment, return_inverse=True)[1]
    return (signal * rng.normal(size=(codes.max() + 1, d))[codes] + 0.5 * rng.normal(size=(8, d))[experiment]
            + rng.normal(size=(len(cells), d)))


def groups_for(names, best="dinov3"):
    """score.compare's groups as evaluate reads them: the comparison names and the heldout group's best seed."""
    entry = {"comparisons": dict.fromkeys(names), "best_default_seed": best}
    return {t: {"seen": entry, "heldout": entry} for t in score.TABLES}


COMPARISONS = ["default_mean_vs_frozen", "shape_vs_default_mean", "shape_vs_best_default_seed", "shape_vs_frozen"]


def test_scores_match_sklearn():
    from sklearn.metrics import balanced_accuracy_score, f1_score

    rng = np.random.default_rng(0)
    for trial in range(20):
        true = rng.integers(0, 5, 60)
        pred = np.where(rng.random(60) < 0.4, true, rng.integers(0, 7, 60))     # classes 5, 6: predicted only
        confusion = np.zeros((7, 7))
        np.add.at(confusion, (true, pred), 1)
        got = secondary.scores(confusion)
        assert got["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(true, pred), abs=1e-12)
        assert got["macro_f1"] == pytest.approx(f1_score(true, pred, average="macro", zero_division=0), abs=1e-12)


def test_the_probe_recovers_planted_classes_and_random_vectors_sit_at_chance():
    cells, sirna = layout()
    probe = secondary.Probe(cells, sirna)
    assert probe.layout["well"]["n_train"] == 36 and probe.layout["well"]["n_test"] == 12
    assert probe.layout["cell"]["n_test"] == 96 and probe.layout["n_classes"] == 6
    run = probe.run(planted(cells, sirna))
    for (unit, name), entry in run.items():
        assert entry["point"]["balanced_accuracy"] > 0.95 and entry["point"]["macro_f1"] > 0.95, (unit, name)
    noise = probe.run(np.random.default_rng(1).normal(size=(len(cells), 16)))
    for entry in noise.values():
        assert entry["point"]["balanced_accuracy"] < 0.5


def test_the_bootstrap_is_deterministic_and_resamples_wells():
    cells, sirna = layout()
    first, second = secondary.Probe(cells, sirna), secondary.Probe(cells, sirna)
    assert np.array_equal(first.draws, second.draws)
    assert first.draws.shape == (secondary.BOOTSTRAP, 12) and (first.draws.sum(axis=1) == 12).all()
    vectors = planted(cells, sirna, signal=0.6)
    runs = [first.run(vectors), second.run(vectors)]
    for key in runs[0]:
        for metric in secondary.METRICS:
            assert np.array_equal(runs[0][key]["draws"][metric], runs[1][key]["draws"][metric])
    # a drawn well brings all its cells: a resample's per-class counts are the draw-weighted sum of the wells'
    tp, true, pred = first.per_well("cell", np.zeros(int(first.cell_test.sum()), dtype=np.int64))
    assert tp.shape == true.shape == pred.shape == (12, 6)
    assert (true.sum(axis=1) == 8).all() and (pred.sum(axis=1) == 8).all() and (pred[:, 0] == 8).all()
    assert (tp == np.where(true[:, :1] > 0, 8, 0) * (np.arange(6) == 0)).all()
    arms = ["frozen", "dinov3", "shape"]
    keys = {a: a for a in arms}
    tables = lambda table: {a: planted(cells, sirna, signal=0.6, seed=i) for i, a in enumerate(arms)}  # noqa: E731
    one = secondary.evaluate(tables, keys, cells, sirna, ["dinov3"], groups_for(COMPARISONS))
    two = secondary.evaluate(tables, keys, cells, sirna, ["dinov3"], groups_for(COMPARISONS))
    one.pop("seconds"), two.pop("seconds")
    assert json.dumps(one) == json.dumps(two)


def test_chance_is_the_stratified_expectation():
    sizes = [4, 6, 8, 10, 12, 14]                       # unequal classes: chance is not 1 / n_classes
    cells, sirna = layout(heldout=3, plates=8, sizes=sizes)
    probe = secondary.Probe(cells, sirna)
    n = len(probe.classes)
    for unit in secondary.UNITS:
        if unit == "well":
            train, test = probe.well_codes[probe.well_train], probe.well_codes[probe.well_test]
        else:
            train, test = probe.codes[probe.cell_train], probe.codes[probe.cell_test]
        prior = np.bincount(train, minlength=n) / len(train)
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(3000):
            confusion = np.zeros((n, n))
            np.add.at(confusion, (test, rng.choice(n, size=len(test), p=prior)), 1)
            draws.append([secondary.scores(confusion)[m] for m in secondary.METRICS])
        simulated = dict(zip(secondary.METRICS, np.mean(draws, axis=0)))
        expected = probe.layout[unit]["chance"]
        assert expected["balanced_accuracy"] == pytest.approx(prior[np.unique(test)].mean())
        assert simulated["balanced_accuracy"] == pytest.approx(expected["balanced_accuracy"], abs=0.006), unit
        assert simulated["macro_f1"] == pytest.approx(expected["macro_f1"], abs=0.02 if unit == "well" else 0.006)


def test_knn_votes_ties_to_the_nearest_neighbour_and_orders_equal_similarities_by_row():
    angles = np.radians([5, 10, 15, 20, 90, 90])
    train = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    labels = np.array([1, 0, 0, 1, 2, 2])
    test = np.array([[1.0, 0.0]])
    # k = 4: two votes each for 1 and 0; the nearest neighbour (row 0) is class 1
    assert secondary.knn_predict(train, labels, test, 3, k=4).tolist() == [1]
    assert secondary.knn_predict(train, labels, test, 3, k=3).tolist() == [0]          # 0 wins 2 to 1
    # rows 4 and 5 are equally similar to (0, 1): the earlier row is the nearest
    tie = np.array([[0.0, 1.0]])
    assert secondary.knn_predict(train, np.array([0, 0, 0, 0, 2, 1]), tie, 3, k=2).tolist() == [2]
    assert secondary.knn_predict(train, np.array([0, 0, 0, 0, 1, 2]), tie, 3, k=2).tolist() == [1]


def test_paired_deltas_use_the_same_draws_and_the_default_mean():
    cells, sirna = layout()
    arms = ["frozen", "dinov3", "dinov3_s1", "shape"]
    vectors = {a: planted(cells, sirna, signal=0.5, seed=i) for i, a in enumerate(arms)}
    vectors["shape"] = vectors["frozen"]                      # the same state: every delta exactly 0
    keys = {**{a: a for a in arms}, "shape": "frozen"}
    groups = groups_for(COMPARISONS + ["shape_top_vs_frozen", "frozen_vs_openphenom"], "dinov3_s1")
    record = secondary.evaluate(lambda table: vectors, keys, cells, sirna, ["dinov3", "dinov3_s1"], groups)
    for table in score.TABLES:
        assert record[table]["comparisons"] == COMPARISONS        # shape_top and absent arms are not A9 pairs
        for unit in secondary.UNITS:
            for name in secondary.CLASSIFIERS:
                entry = record[table][unit][name]
                same = entry["comparisons"]["shape_vs_frozen"]
                assert all(same[m] == {"delta": 0.0, "ci95": [0.0, 0.0], "ci_excludes_zero": False}
                           for m in secondary.METRICS)
                for metric in secondary.METRICS:
                    mean = np.mean([entry["arms"][a][metric]["value"] for a in ("dinov3", "dinov3_s1")])
                    band = entry["comparisons"]["default_mean_vs_frozen"][metric]
                    assert band["delta"] == pytest.approx(mean - entry["arms"]["frozen"][metric]["value"])
                    assert band["ci95"][0] <= band["delta"] <= band["ci95"][1]
                    value = entry["arms"]["shape"][metric]
                    assert value["ci95"][0] <= value["value"] <= value["ci95"][1]
    text = secondary.markdown(record)
    assert text.startswith("\n\n## Secondary metrics (contract amendment A9)\n\n" + secondary.CAVEAT)
    assert text.count("| chance | ") == 2 and "### matched table: siRNA probe" in text


def test_no_label_is_read_outside_score():
    source = (HERE / "secondary.py").read_text(encoding="utf-8")
    assert not re.search(r"open\(|read_csv|read_text|labels\.csv|private|read_labels|np\.load|Path\(", source)
    importers = [p.name for p in HERE.glob("*.py") if not p.name.startswith("test_")
                 and re.search(r"import secondary|representation_RxRx1\.runners\.secondary",
                               p.read_text(encoding="utf-8"))]
    assert importers == ["score.py"]


def _pre_a9_score(monkeypatch):
    """score.py as it was before A9 (git show PRE_A9), loaded as its own module; skipped without git."""
    try:
        text = subprocess.run(["git", "show", f"{PRE_A9}:benchmarks/representation_RxRx1/score.py"], cwd=HERE,
                              capture_output=True, text=True, check=True, encoding="utf-8").stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {PRE_A9[:8]} unavailable")
    # its lazy imports name the modules before the move into runners/
    text = text.replace("benchmarks.representation_RxRx1", "benchmarks.representation_RxRx1.runners")
    old_family = 'a.startswith("default_s")'                         # A23: the control arm is dinov3, not default_s0
    assert text.count(old_family) == 1
    text = text.replace(old_family, 'a == "dinov3" or a.startswith("dinov3_s")')
    spec =importlib.util.spec_from_loader("score_pre_a9", loader=None)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "score_pre_a9", module)        # its dataclass looks its module up
    exec(compile(text, "score_pre_a9.py", "exec"), module.__dict__)
    return module


def _without_random(value):
    """A pre-A9 RESULT.json value without the random arm (its `random_arm` record and every `random` entry)."""
    if isinstance(value, dict):
        return {k: _without_random(v) for k, v in value.items() if k not in ("random", "random_arm")}
    if isinstance(value, list):
        return [_without_random(v) for v in value]
    if isinstance(value, str) and "| random |" in value:          # the markdown tables kept in RESULT.json
        return _md_without_random(value)
    return value


def _md_without_random(text):
    """RESULT.md without the random arm's rows and the random-baseline paragraph (worded differently before)."""
    lines = [line for line in text.split("\n") if not line.startswith("| random |")]
    return "\n".join(line for line in lines if not line.startswith("Random baseline (contract amendment A7)"))


def test_every_pre_existing_output_is_unchanged(prepared, stub, tmp_path, monkeypatch):  # noqa: F811
    old_score = _pre_a9_score(monkeypatch)
    root, _, _ = prepared
    control.control(root, seeds=(0,), split_key="split")
    run_dir = tmp_path / "shape_vrr" / "HEPG2" / "runs" / "a9"
    shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=3, engine=_shape_engine(3))
    old_score.final(root, run_dir)
    old_json, old_md = (run_dir / "RESULT.json").read_text(encoding="utf-8"), (run_dir / "RESULT.md").read_text(
        encoding="utf-8")
    result = score.final(root, run_dir)
    new_json, new_md = (run_dir / "RESULT.json").read_text(encoding="utf-8"), (run_dir / "RESULT.md").read_text(
        encoding="utf-8")
    new = json.loads(new_json)
    record = new.pop("secondary")
    # the random arm is no longer scored (2026-10-11): the pre-A9 output minus it is what must be unchanged
    old = _without_random(json.loads(old_json))
    assert list(old) + ["secondary"] == list(json.loads(new_json))                     # only the key added, last
    assert json.dumps(new, indent=1, default=str) == json.dumps(old, indent=1, default=str)   # every other value
    section = secondary.markdown(record)
    assert section in new_md
    assert _md_without_random(new_md.replace(section, "", 1)) == _md_without_random(old_md)   # only the section
    assert record["arms"] == secondary.a9_arms(record["arms"]) and "shape_top" not in record["arms"]
    assert {"frozen", "dinov3", "shape"} <= set(record["arms"]) and "random" not in record["arms"]
    assert result["secondary"]["native"]["well"]["knn"]["arms"]["frozen"]["macro_f1"]["value"] is not None


# ---- the full set: negative controls out of the scoring; the scoring at its size -----------------------------------

def _with_negative_controls(cells, sirna, cells_per_well=5):
    """The layout plus one negative-control well (siRNA EMPTY) per plate, its cells interleaved at the front of each
    plate's rows (as a prep of every well type holds them)."""
    rows, labels = [], []
    for plate, block in cells.groupby("plate", sort=False):
        first = block.iloc[0]
        for c in range(cells_per_well):
            rows.append({"experiment": first.experiment, "plate": plate, "well": f"{plate}/WEMPTY",
                         "split": first.split})
            labels.append(score.NEGATIVE_CONTROL)
        rows += block.to_dict("records")
        labels += list(sirna[block.index])
    return pd.DataFrame(rows), np.array(labels)


def _old_mean_by(key, vectors, *columns):
    """score._mean_by before the full set (one pass over every row per key), the reference of the vectorised one."""
    order = np.unique(key)
    means = np.empty((len(order), vectors.shape[1]), dtype=np.float64)
    taken = [np.empty(len(order), dtype=object) for _ in columns]
    for index, value in enumerate(order):
        member = key == value
        means[index] = vectors[member].mean(axis=0)
        for column, store in zip(columns, taken):
            store[index] = column[member][0]
    return order, score.l2(means), [np.array(store.tolist()) for store in taken]


def test_the_well_profile_equals_the_per_key_loop_bitwise():
    rng = np.random.default_rng(3)
    for n_rows, n_keys in ((1, 1), (50, 7), (3000, 211)):
        key = np.array([f"P{k % 3}/W{k:03d}" for k in rng.integers(0, n_keys, n_rows)])     # unsorted, uneven
        vectors = score.l2(rng.normal(size=(n_rows, 24)) * rng.lognormal(size=(n_rows, 1)))
        column = np.array([k.split("/")[0] + str(i) for i, k in enumerate(key)])
        new, old = score._mean_by(key, vectors, column), _old_mean_by(key, vectors, column)
        assert np.array_equal(new[0], old[0]) and np.array_equal(new[1], old[1])
        assert new[2][0].tolist() == old[2][0].tolist() and new[2][0].dtype == old[2][0].dtype


def test_the_per_class_counts_give_the_confusion_matrix_scores_bitwise():
    cells, sirna = layout(n_sirna=7, heldout=2, plates=3)
    probe = secondary.Probe(cells, sirna)
    n, rng = len(probe.classes), np.random.default_rng(5)
    for unit, n_test in (("well", int(probe.well_test.sum())), ("cell", int(probe.cell_test.sum()))):
        predicted = rng.integers(0, n, n_test)
        true, wells = ((probe.well_codes[probe.well_test], np.arange(n_test)) if unit == "well"
                       else (probe.codes[probe.cell_test], probe.cell_test_well))
        dense = np.zeros((int(probe.well_test.sum()), n * n))       # the earlier per-well confusion counts
        np.add.at(dense, (wells, true * n + predicted), 1)
        counts = probe.per_well(unit, predicted)
        for metric in secondary.METRICS:
            assert (secondary.class_scores(*(c.sum(axis=0) for c in counts))[metric]
                    == secondary.scores(dense.sum(axis=0).reshape(n, n))[metric])
            assert np.array_equal(secondary.class_scores(*(probe.draws @ c for c in counts))[metric],
                                  secondary.scores((probe.draws @ dense).reshape(secondary.BOOTSTRAP, n, n))[metric])


def test_negative_controls_are_neither_queries_nor_candidates_nor_probe_classes():
    """The EMPTY wells leave the well mAP (every group, every audit group, the chance rows) and the probe exactly as if
    they were absent; a layout without them (the v3 preps) records none and RESULT.json gains no key for them."""
    cells, sirna = layout(n_sirna=6, heldout=2, plates=2)
    with_empty, labels = _with_negative_controls(cells, sirna)
    vectors = planted(with_empty, labels, signal=0.8, seed=2)
    keep = labels != score.NEGATIVE_CONTROL
    assert (~keep).sum() == 5 * 10 and score.negative_controls(cells, sirna) is None
    without = with_empty[keep].reset_index(drop=True)
    assert json.dumps(score.evaluate(vectors, with_empty, labels), sort_keys=True) == json.dumps(
        score.evaluate(vectors[keep], without, labels[keep]), sort_keys=True)
    probe, reference = secondary.Probe(with_empty, labels), secondary.Probe(without, labels[keep])
    assert score.NEGATIVE_CONTROL not in probe.classes and np.array_equal(probe.classes, reference.classes)
    assert probe.layout.pop("negative_controls") == {"sirna": "EMPTY", "cells": 50, "wells": 10,
                                                     "rule": "not a class, not trained or tested"}
    assert reference.layout.pop("negative_controls")["wells"] == 0 and probe.layout == reference.layout
    assert np.array_equal(probe.draws, reference.draws)
    runs = probe.run(vectors), reference.run(vectors[keep])
    for key in runs[0]:
        assert runs[0][key]["point"] == runs[1][key]["point"], key
        for metric in secondary.METRICS:
            assert np.array_equal(runs[0][key]["draws"][metric], runs[1][key]["draws"][metric]), key
    record = score.negative_controls(with_empty, labels)
    assert (record["sirna"], record["cells"], record["wells"]) == ("EMPTY", 50, 10)
    assert record["wells_per_group"] == {"seen": 6, "heldout": 4, "run_prep": 6}


def test_the_full_set_probes_the_well_unit_only():
    """The full-set v4 preps skip the probe's cell unit (score.PROBE_SKIPPED, user decision 2026-10-13): no cell runs,
    rows or columns, the reason in the layout and the record; the well unit is the one of both units; v3 keeps both."""
    import time

    cells, sirna = layout(n_sirna=6, heldout=2, plates=2)
    full_v4 = {"sources": {"full": {"path": "/data/agentic_ssl/prep_huvec_full_v4"}}}
    v3 = {"sources": {"full": {"path": "/data/agentic_ssl/prep_hepg2_e11_v3"}}}
    assert (score.prep_set(full_v4), score.prep_set(v3), score.prep_set({})) == ("full_v4", "v3", "v3")
    assert score.probe_units(v3) == (("well", "cell"), {})
    units, skipped = score.probe_units(full_v4)
    assert units == ("well",) and set(skipped) == {"cell"} and "1.42 M cells" in skipped["cell"]
    scorer = score.ArmScores.__new__(score.ArmScores)                   # the score cache's probe follows the prep
    scorer.summary, scorer.cells, scorer.sirna, scorer._probe = full_v4, cells, sirna, None
    assert scorer.probe().units == ("well",)
    probe, both = secondary.Probe(cells, sirna, units=units, skipped=skipped), secondary.Probe(cells, sirna)
    assert "cell" not in probe.layout and probe.layout.pop("skipped_units") == skipped
    assert probe.layout == {k: v for k, v in both.layout.items() if k != "cell"} and "skipped_units" not in both.layout
    vectors = planted(cells, sirna, signal=0.8)
    runs, reference = probe.run(vectors), both.run(vectors)
    assert sorted(runs) == [("well", "knn"), ("well", "logreg")]
    for key in runs:
        assert runs[key]["point"] == reference[key]["point"]
        for metric in secondary.METRICS:
            assert np.array_equal(runs[key]["draws"][metric], reference[key]["draws"][metric])
    probe.layout["skipped_units"] = skipped
    record = secondary.record(probe, {t: {"frozen": runs, "shape": runs} for t in score.TABLES}, [],
                              groups_for(["shape_vs_frozen"]), time.time())
    assert record["settings"]["skipped_units"] == skipped and list(record["settings"]["units"]) == ["well"]
    assert all(set(record[t]) == {"comparisons", "well"} for t in score.TABLES)
    text = secondary.markdown(record)
    assert "cell knn" not in text and "; the cell unit is skipped); kNN cosine" in text
    assert "The cell unit is skipped for this prep: the full-set preps hold up to 1.42 M cells" in text
    full_record = secondary.record(both, {t: {"frozen": reference} for t in score.TABLES}, [],
                                   groups_for([]), time.time())
    assert "skipped_units" not in full_record["settings"] and "cell knn BA" in secondary.markdown(full_record)
