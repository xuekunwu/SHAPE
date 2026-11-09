"""summarize_results.py on a stub score cache: the tables, the paired bands and the trajectories, no embedding."""
import csv
import json

import numpy as np
import pytest

from benchmarks.representation_RxRx1.runners import score, summarize_results

UNITS = [f"w{i}" for i in range(12)]
UNIT_GROUP = {u: f"s{i % 4}" for i, u in enumerate(UNITS)}            # 4 siRNA, 3 wells each
METRICS = ("balanced_accuracy", "macro_f1")
PROBE_RUNS = (("well", "knn"), ("well", "logreg"), ("cell", "knn"), ("cell", "logreg"))


def evaluation(level):
    """Every group of score.py with a per-well AP of `level` (+ a small per-well spread)."""
    per_unit = {u: level + 0.01 * i for i, u in enumerate(UNITS)}
    return {g: {"value": level, "chance": 0.05, "per_unit": per_unit, "unit_group": UNIT_GROUP}
            for g in summarize_results.GROUPS}


def probe(level, rng):
    return {f"{unit}/{name}": {"point": {m: level for m in METRICS},
                               "draws": {m: (level + 0.01 * rng.standard_normal(20)).tolist() for m in METRICS},
                               "fit": {}} for unit, name in PROBE_RUNS}


def write_cache(root, levels):
    cache = root / score.SCORE_CACHE
    cache.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for arm, level in levels.items():
        for table in score.TABLES:
            entry = {"key": {"table": table}, "arm": arm, "evaluation": evaluation(level), "probe": probe(level, rng)}
            (cache / f"{arm}-{table}.json").write_text(json.dumps(entry), encoding="utf-8")
    (cache / "broken.json").write_text("{", encoding="utf-8")          # an unreadable entry is skipped
    (cache / "other.json").write_text(json.dumps({"arm": "shape_top", "key": {"table": "native"},
                                                  "evaluation": {}}), encoding="utf-8")     # not in ARMS


def write_run(run, calls):
    (run / "logs").mkdir(parents=True)
    lines = []
    for call, parent, changes, proxy in calls:
        lines.append({"event": "start", "call": call, "parent": parent, "init": "fresh", "set": changes})
        lines.append({"event": "end", "call": call, "status": "done", "metrics": {"sibling_site_map": proxy},
                      "tool": {"training": {"epochs_trained": 9, "best_epoch": 4}}})
    (run / "logs" / "tool_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")


@pytest.fixture
def layout(tmp_path, monkeypatch):
    root = tmp_path / "cache" / "HEPG2"
    write_cache(root, {"frozen": 0.10, "dinov3": 0.12, "shape": 0.18, "react": 0.13})
    shape_run, react_run, llm_run = tmp_path / "shape_run", tmp_path / "react_run", tmp_path / "llm_run"
    write_run(shape_run, [("agent_1", "input", {"view_pairing": "same_well"}, 0.73),
                          ("agent_2", "agent_1", {"learning_rate": 1e-05}, 0.74)])
    write_run(react_run, [("agent_1", "input", {"trainable_pct": 20}, None)])
    seen = {}

    def diagnostic(cells, sirna, calls, lineage):
        seen.setdefault("calls", []).append(sorted(lineage))
        return {"points": [{"call": c, "run_prep_well_map": 0.2 + 0.01 * i} for i, c in enumerate(sorted(lineage))],
                "spearman_rho": 0.5}

    monkeypatch.setattr(summarize_results.prepare, "load", lambda r: ({"prep": str(r)}, "cells"))
    monkeypatch.setattr(score, "read_labels", lambda r, cells, summary: np.zeros(1))
    monkeypatch.setattr(score, "diagnostic", diagnostic)
    return root, tmp_path / "out", shape_run, react_run, llm_run, seen


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_summary_from_the_cache(layout):
    root, out, shape_run, react_run, llm_run, seen = layout
    compare = summarize_results.cli(["--root", str(root), "--out", str(out), "--shape-run", str(shape_run),
                                     "--react-run", str(react_run), "--llm-run", str(llm_run), "--title", "T"])
    assert compare == out / "compare"
    text = (compare / "summary.md").read_text(encoding="utf-8")
    assert text.startswith("# T\n") and "llm reused `llm_run`" in text
    assert "| shape | 0.1800 |" in text and "| chance | 0.0500 |" in text
    assert "| llm |" not in text and "shape_top" not in text      # no entry, or not a listed arm

    scores = read(compare / "scores.csv")
    assert {r["arm"] for r in scores} == {"frozen", "dinov3", "shape", "react"}
    assert len(scores) == 4 * len(score.TABLES) * len(summarize_results.GROUPS)

    paired = read(compare / "paired_vs_shape.csv")
    names = {r["comparison"] for r in paired}
    assert names == {"shape_vs_dinov3", "shape_vs_react", "shape_vs_frozen", "dinov3_vs_frozen", "react_vs_frozen"}
    row = next(r for r in paired
               if r["table"] == "native" and r["group"] == "seen" and r["comparison"] == "shape_vs_frozen")
    expected = score.paired_band(evaluation(0.10)["seen"]["per_unit"], evaluation(0.18)["seen"]["per_unit"], UNIT_GROUP)
    assert float(row["delta"]) == pytest.approx(expected["delta"]) == pytest.approx(0.08)
    assert float(row["band_2se"]) == pytest.approx(expected["band_2se"])
    assert int(row["n_units_paired"]) == len(UNITS)

    probe_rows = read(compare / "probe.csv")
    assert len(probe_rows) == 4 * len(score.TABLES) * 6
    assert "- vs dinov3: well knn balanced_accuracy: +0.060" in text

    calls = read(compare / "calls.csv")
    assert [(c["arm"], c["call"]) for c in calls] == [("shape", "agent_1"), ("shape", "agent_2"), ("react", "agent_1")]
    assert [float(c["run_prep_well_map"]) for c in calls] == pytest.approx([0.20, 0.21, 0.20])
    assert json.loads(calls[1]["changes"]) == {"learning_rate": 1e-05} and calls[2]["val_proxy_comparison_draws"] == ""
    assert seen["calls"] == [["agent_1", "agent_2"], ["agent_1"]]
    assert "shape: Spearman rho (val proxy vs run-prep well mAP over the calls) = 0.5" in text


def test_cli_requires_every_path():
    with pytest.raises(SystemExit):
        summarize_results.cli(["--root", "r", "--out", "o", "--shape-run", "s", "--react-run", "x"])


def test_summary_without_the_cell_unit(layout):
    """A full-set cache (the probe's cell unit skipped, score.PROBE_SKIPPED): the probe table has the well columns
    only and says so; the rest is unchanged."""
    root, out, shape_run, react_run, llm_run, _ = layout
    for path in (root / score.SCORE_CACHE).glob("*-*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        entry["probe"] = {k: v for k, v in entry["probe"].items() if k.startswith("well/")}
        path.write_text(json.dumps(entry), encoding="utf-8")
    compare = summarize_results.cli(["--root", str(root), "--out", str(out), "--shape-run", str(shape_run),
                                     "--react-run", str(react_run), "--llm-run", str(llm_run)])
    text = (compare / "summary.md").read_text(encoding="utf-8")
    header = "| arm | well knn BA | well knn F1 | well logreg BA | well logreg F1 |"
    assert header + "\n| --- | --- | --- | --- | --- |" in text
    assert "cell knn" not in text and "The cell unit is not probed for this prep" in text
    probe_rows = read(compare / "probe.csv")
    assert len(probe_rows) == 4 * len(score.TABLES) * 4 and {r["unit"] for r in probe_rows} == {"well"}
    assert "- vs dinov3: well knn balanced_accuracy: +0.060" in text
