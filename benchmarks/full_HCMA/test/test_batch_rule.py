"""workspace_template/tools/batch_correct.py: the calibration rule (SHAPE's metadata profile decides whether the
tool runs).

A synthetic embed call (test/_synthetic_chain.Chain) over eight images in two sources with a strong source
offset; the batch_correct CLI runs in this process with the SHAPE tool classes replaced by test/tool_stubs.py
(the real Batch_Corrector_Tool with a mean-centring Harmony stand-in). Without label-free reference groups in
images.csv the input comes back unchanged and the tool never runs; with groups replicated across the sources
SHAPE's batch stage runs (its batch-metrics extra and calibration scorer stood in: scib-metrics / jax are not in
the test env)."""
from __future__ import annotations

import csv
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.full_HCMA.test._synthetic_chain import Chain

HERE = Path(__file__).resolve().parent
TEMPLATE_TOOLS = HERE.parent / "workspace_template" / "tools"
REPO = HERE.parents[2]
# image -> (source, group, control): groups g1 and g2 twice in both sources, one control image per source (the
# same composition in both sources: removing the source offset keeps the group separation)
DESIGN = {"img_0001": ("s0", "g1", ""), "img_0002": ("s0", "g2", ""), "img_0003": ("s0", "g1", ""),
          "img_0004": ("s0", "g2", ""), "img_0005": ("s0", "ctrl", "true"), "img_0006": ("s1", "g1", ""),
          "img_0007": ("s1", "g2", ""), "img_0008": ("s1", "g1", ""), "img_0009": ("s1", "g2", ""),
          "img_0010": ("s1", "ctrl", "true")}
CELLS = 8


def _embedding(cells: pd.DataFrame) -> np.ndarray:
    """Group signal in dims 0-1, a large source offset in dim 2, noise."""
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.3, size=(len(cells), 8))
    for i, image_id in enumerate(cells["image_id"]):
        source, group, _ = DESIGN[image_id]
        x[i, {"g1": 0, "g2": 1, "ctrl": 3}[group]] += 3.0
        x[i, 2] += 6.0 if source == "s1" else 0.0
    return x.astype(np.float32)


def _workspace(run: Path, groups: bool) -> Path:
    chain = Chain(run, traced=False)
    s = chain.segment(list(DESIGN))
    c = chain.crop(s, {i: CELLS for i in DESIGN})
    chain.embed(c, _embedding)
    ws = run / "workspace"
    columns = ["image_id", "file", "parent_id", "source", "image_modality", "magnification", "height", "width",
               "channels"] + (["group", "control"] if groups else [])
    with open(ws / "data" / "images.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for n, (image_id, (source, group, control)) in enumerate(DESIGN.items()):
            writer.writerow([image_id, f"images/{image_id}.png", f"par_{n:04d}", source, "brightfield", "", 96, 96, 1]
                            + ([group, control] if groups else []))
    shutil.rmtree(ws / "tools")
    shutil.copytree(TEMPLATE_TOOLS, ws / "tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    config = {"run": str(run), "ledger": str(run / "logs" / "tool_calls.jsonl"), "images_csv": "data/images.csv",
              "images_dir": "data/images", "gpu_slots": 1, "deadline_file": ".deadline", "min_start_seconds": 600,
              "repo": str(REPO), "env": {}, "allow_tool_stubs": True}
    (ws / "tools" / "_config.json").write_text(json.dumps(config), encoding="utf-8")
    return ws


@pytest.fixture()
def cli(tmp_path, monkeypatch):
    """(make(groups) -> workspace, call(*argv) -> exit code, the newest batch_correct record)."""
    saved = {name: sys.modules.pop(name) for name in ("_common", "batch_correct") if name in sys.modules}
    monkeypatch.setenv("FULL_HCMA_TOOL_STUBS", str(HERE / "tool_stubs.py"))
    state = {}

    def make(groups: bool) -> Path:
        ws = _workspace(tmp_path / ("groups" if groups else "plain"), groups)
        for name in ("_common", "batch_correct"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(ws / "tools"))
        importlib.import_module("_common")
        sys.path.insert(0, str(ws / "tools"))
        state["module"] = importlib.import_module("batch_correct")
        sys.path[:] = [p for p in sys.path if p != str(ws / "tools")]
        state["ws"] = ws
        return ws

    def call(*argv) -> int:
        env = dict(os.environ)
        try:
            return state["module"].main([str(a) for a in argv])
        finally:
            os.environ.clear()
            os.environ.update(env)

    def last() -> dict:
        lines = [json.loads(line) for line in (state["ws"].parent / "logs" / "tool_calls.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        start = [e for e in lines if e["event"] == "start" and e["tool"] == "batch_correct"][-1]
        end = next(e for e in lines if e["event"] == "end" and e["call"] == start["call"])
        return {"start": start, "end": end, "dir": state["ws"] / start["call_dir"],
                "result": json.loads((state["ws"] / start["call_dir"] / "tool_result.json").read_text(encoding="utf-8"))
                if (state["ws"] / start["call_dir"] / "tool_result.json").exists() else None}

    yield make, call, last, state
    for name in ("_common", "batch_correct"):
        sys.modules.pop(name, None)
    sys.modules.update(saved)


def _input(ws: Path) -> np.ndarray:
    return np.load(next((ws / "calls").glob("E001_*")) / "embedding.npy")


def _fake_scorer(meta, feats, label_key, batch_key, plate_types):
    """Stands in for quant.composite_all (scib-metrics): batch removal and group separation, higher = better."""
    feats, batch, label = np.asarray(feats), meta[batch_key].astype(str).to_numpy(), meta[label_key].astype(str).to_numpy()
    offset = np.linalg.norm(feats[batch == batch[0]].mean(0) - feats[batch != batch[0]].mean(0))
    centres = np.stack([feats[label == g].mean(0) for g in sorted(set(label))])
    bio = float(np.linalg.norm(centres[:, None] - centres[None], axis=-1).mean())
    batch_mean = 1.0 / (1.0 + offset)
    return {"overall": (batch_mean + bio / (1 + bio)) / 2, "batch_mean": batch_mean, "bio_mean": bio / (1 + bio),
            "nonrep_mean_map": 0.5, "negcon_mean_map": 0.5}


def test_no_reference_groups_return_the_input_unchanged(cli, monkeypatch):
    make, call, last, state = cli
    ws = make(groups=False)
    import shape.tools.batch_corrector.tool as tool

    def refuse(*args, **kwargs):
        raise AssertionError("the tool must not run without a calibration view")

    monkeypatch.setattr(tool.Batch_Corrector_Tool, "execute", refuse)
    assert call("--name", "rule", "--embedding", "E001", "--batch-key", "source") == 0
    rec = last()
    assert rec["end"]["status"] == "done"
    assert np.array_equal(np.load(rec["dir"] / "embedding.npy"), _input(ws))
    summary = rec["end"]["summary"]
    assert summary["corrected"] is False and summary["tier"] == "partial"
    assert summary["reason"] == "no calibration reference: no label-free groups replicated across batches"
    assert rec["result"]["status"] == "UNCORRECTED_INPUT" and rec["result"]["levers_applied"] is False
    assert any("fewer than 2 non-control groups" in r for r in rec["result"]["tier_reasons"])
    # a lever set without a calibration view: recorded, not applied
    assert call("--name", "rule_theta", "--embedding", "E001", "--batch-key", "source", "--set", "theta=4") == 0
    rec = last()
    assert rec["start"]["set_levers"] == ["theta"] and rec["result"]["levers_applied"] is False
    assert np.array_equal(np.load(rec["dir"] / "embedding.npy"), _input(ws))


def test_replicated_groups_make_one_tool_call_at_the_defaults(cli, monkeypatch):
    """A calibration view and no lever set: one tool call at the tool's DEFAULT_CONFIG (the tool's own gate
    decides); SHAPE's agentic batch loop needs a reasoner, which this CLI does not have, and is never run."""
    make, call, last, state = cli
    ws = make(groups=True)
    from shape.agent.verification import batch_stage, metadata_profile
    from shape.tools.batch_corrector.tool import DEFAULT_CONFIG

    monkeypatch.setattr(metadata_profile, "batch_metrics_problem", lambda: None)
    monkeypatch.setattr(batch_stage, "run_batch_stage", lambda *a, **k: pytest.fail("the agentic loop must not run"))
    assert call("--name", "calibrated", "--embedding", "E001", "--batch-key", "source") == 0
    rec = last()
    assert rec["result"]["tier"] == "full" and rec["end"]["summary"]["tier"] == "full"
    assert rec["result"]["levers_applied"] is False and rec["result"]["harmony_params_passed"] == dict(DEFAULT_CONFIG)
    assert rec["end"]["summary"]["corrected"] is True and rec["result"]["tool"]["batch_effect"]["detected"]
    corrected = np.load(rec["dir"] / "embedding.npy")
    x = _input(ws)
    cells = pd.read_csv(rec["dir"] / "cells.csv")
    s1 = cells["image_id"].map(lambda i: DESIGN[i][0] == "s1").to_numpy()
    assert abs(corrected[s1, 2].mean() - corrected[~s1, 2].mean()) < 0.5 < abs(x[s1, 2].mean() - x[~s1, 2].mean())
    assert not (rec["dir"] / "tool").exists()


def test_levers_with_a_calibration_view_make_one_tool_call(cli, monkeypatch):
    make, call, last, state = cli
    make(groups=True)
    from shape.agent.verification import batch_stage, metadata_profile

    monkeypatch.setattr(metadata_profile, "batch_metrics_problem", lambda: None)
    monkeypatch.setattr(batch_stage, "run_batch_stage", lambda *a, **k: pytest.fail("the stage must not run"))
    assert call("--name", "theta", "--embedding", "E001", "--batch-key", "source", "--set", "theta=3") == 0
    rec = last()
    assert rec["result"]["levers_applied"] is True and rec["result"]["harmony_params_passed"]["theta"] == 3.0
    # nclust within the CARD range but not below the number of rows: the tool itself refuses, the call fails
    assert call("--name", "toomany", "--embedding", "E001", "--batch-key", "source", "--set", "nclust=5000") == 1
    rec = last()
    assert rec["end"]["status"] == "failed" and "nclust" in rec["end"]["error"]
    assert (rec["dir"] / "error.txt").exists()


def test_dry_run_reports_the_calibration_tier(cli, capsys):
    make, call, last, state = cli
    make(groups=False)
    capsys.readouterr()
    assert call("--name", "dry", "--embedding", "E001", "--batch-key", "source", "--dry-run") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["calibration_tier"] == "partial" and out["action"].startswith("input returned unchanged")
