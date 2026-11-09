"""`harness table` (harness/table.py) on synthetic run dirs (RESULT.json, PROCESS.json)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.full_HCMA.harness import table
from benchmarks.full_HCMA.harness.cli import main
from benchmarks.full_HCMA.harness.score import CROSS, PRIMARY


def _run(base: Path, name: str, *, arm: str = "agent", model: str = "m1", primary: float | None = 0.3,
         normalized: float | None = 0.4, end: str = "delivered", timed_out: bool = False, claimed: bool = True,
         delta: float | None = None, usd: float | None = 2.0, gpu: str = "gpu-a",
         tokens: dict | None = None) -> Path:
    run = base / name
    run.mkdir(parents=True)
    result = {"prep_name": "pairs", "metadata": "full", "status": "OK", PRIMARY: primary, CROSS: 0.1,
              f"{PRIMARY}_fine": 0.05, "comparison": {"delta": delta} if delta is not None else None}
    if normalized is not None:
        result["reference"] = {"normalized_primary": normalized, "chance": {"primary": 0.1}}
    if arm == "control":
        result["arm"] = "control"
    (run / "RESULT.json").write_text(json.dumps(result))
    agent = arm == "agent"
    llm = {"tokens": tokens or {"input_tokens": 10, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 100,
                                "output_tokens": 50}, "total_cost_usd": usd} if agent else {"applicable": False}
    process = {"agent_run": agent, "end_state": {"end_state": end, "timed_out": timed_out, "refused": False},
               "claims": {"claimed_complete": claimed} if agent else {"applicable": False},
               "E1_cost": {"wall_hours": 1.0, "llm": llm, "tools": {"gpu_hours": 0.5, "n_calls": 12}}}
    (run / "PROCESS.json").write_text(json.dumps(process))
    if agent:
        (run / "RUN_META.json").write_text(json.dumps({"model": model, "gpu_slug": gpu, "prep_name": "pairs",
                                                       "metadata": "full"}))
    return run


def test_table_aggregates_configurations(tmp_path):
    runs = tmp_path / "runs"
    dirs = [_run(runs, "a1", primary=0.30, delta=0.02), _run(runs, "a2", primary=0.40, delta=0.04),
            _run(runs, "b1", model="m2", primary=0.10, normalized=0.05, claimed=True),
            _run(runs, "b2", model="m2", primary=None, normalized=None, end="no_deliverable", timed_out=True,
                 claimed=False, usd=None, tokens={})]
    control = _run(tmp_path / "control" / "pairs" / "full", "gpu-a", arm="control", primary=0.25, normalized=0.3)
    summary = table.table(dirs, tmp_path / "out", control_root=tmp_path / "control")
    configs = {(c["arm"], c["model"]): c for c in summary["configurations"]}
    a, b, ctrl = configs[("agent", "m1")], configs[("agent", "m2")], configs[("control", "-")]
    assert a["n_runs"] == 2 and a["primary"]["mean"] == pytest.approx(0.35)
    assert a["primary"]["sd"] == pytest.approx(np.std([0.3, 0.4], ddof=1)) and (a["primary"]["min"], a["primary"]["max"]) == (0.3, 0.4)
    assert a["delta_vs_control"]["mean"] == pytest.approx(0.03)
    assert a["median_tokens_in"] == 1110 and a["median_tokens_out"] == 50 and a["median_usd_estimate"] == 2.0
    assert (b["delivered"], b["no_deliverable"], b["crashed"], b["timed_out"]) == (1, 1, 0, 1)
    assert b["primary"]["n"] == 1 and b["primary"]["sd"] is None
    assert ctrl["gpu"] == "gpu-a" and ctrl["n_runs"] == 1
    assert ctrl["median_usd_estimate"] is None and ctrl["median_tokens_in"] is None   # none, never NaN or 0
    # only m1 has >= 2 scored runs: no variance decomposition
    assert summary["variance"] is None
    assert "process_outcome" not in summary and not any("process_score" in c or "abstention_rate_mean" in c
                                                         for c in summary["configurations"])
    claims = summary["claims_low_outcome"]
    assert claims["runs"] == ["b1"] and claims["n_claimed_complete"] == 1
    out = tmp_path / "out"
    for name in ("TABLE.md", "table.csv", "runs.csv", "pareto.png", "TABLE.json"):
        assert (out / name).exists(), name
    text = (out / "TABLE.md").read_text()
    assert "nan" not in text and "0.350 +- 0.071 [0.300, 0.400]" in text and "1/1/0/1" in text
    assert len(pd.read_csv(out / "runs.csv")) == 5 and len(pd.read_csv(out / "table.csv")) == 3
    assert str(control) in set(pd.read_csv(out / "runs.csv")["path"])


def test_variance_decomposition_and_low_outcome_without_normalized(tmp_path):
    runs = tmp_path / "runs"
    values = {"m1": [0.30, 0.34], "m2": [0.10, 0.16]}
    dirs = [_run(runs, f"{m}_{i}", model=m, primary=v, normalized=None) for m, vs in values.items() for i, v in enumerate(vs)]
    summary = table.table(dirs, tmp_path / "out")
    variance = summary["variance"]
    grand = np.mean([x for vs in values.values() for x in vs])
    between = sum(len(vs) * (np.mean(vs) - grand) ** 2 for vs in values.values())
    within = sum(sum((x - np.mean(vs)) ** 2 for x in vs) for vs in values.values())
    assert variance["n_configurations"] == 2 and variance["ss_between"] == pytest.approx(between)
    assert variance["share_between"] == pytest.approx(between / (between + within))
    # no normalized_primary: no chance reference either, so these runs are unclassified, not "low"
    assert summary["claims_low_outcome"]["n_unclassified"] == 4 and summary["claims_low_outcome"]["n_low_delivered"] == 0


def test_cli_table(tmp_path, capsys):
    run = _run(tmp_path / "runs", "a1")
    main(["table", "--runs", str(run), "--out", str(tmp_path / "out")])
    assert "TABLE.md" in capsys.readouterr().out and (tmp_path / "out" / "TABLE.md").exists()
    rows = pd.read_csv(tmp_path / "out" / "runs.csv")
    assert rows.loc[0, "process_from"] == "PROCESS.json" and not math.isnan(rows.loc[0, "primary"])
