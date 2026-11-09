"""CPU tests of the JUMP benchmark on a synthetic scenario prep.

harmonypy and the scib-metrics / copairs metric functions are replaced by small stand-ins; the rest
-- control, the SHAPE arm (BatchCorrectionVerifier through StepVerifier, without a reasoner), the
ledger, the decision, scoring with plate-half bands, the report and the record export -- runs unchanged.
The real stack runs on the cluster.
"""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from benchmarks.batch_correction_JUMP.runners import control as control_mod  # noqa: E402
from benchmarks.batch_correction_JUMP.runners.prepare import PROTECTED, assign_split, load_scenario, sha256_file  # noqa: E402
from shape.tools.batch_corrector import quant  # noqa: E402
from shape.tools.batch_corrector.tool import DEFAULT_CONFIG  # noqa: E402
from benchmarks.batch_correction_JUMP.runners.benchmark_contract import LUNA_MODEL  # noqa: E402


def _between(feats, groups):
    x = np.asarray(feats, dtype=float)
    g = np.asarray(groups).astype(str)
    tot = ((x - x.mean(0)) ** 2).sum() + 1e-12
    return float(sum((g == u).sum() * ((x[g == u].mean(0) - x.mean(0)) ** 2).sum() for u in np.unique(g)) / tot)


def _fake_harmonypy():
    module = types.ModuleType("harmonypy")

    def run_harmony(data, meta, vars_use, theta=2.0, sigma=0.1, nclust=None, random_state=0, **kw):
        x = np.asarray(data, dtype=float)
        batch = pd.Series(meta[vars_use]).astype(str).to_numpy()
        z = x.copy()
        w = min(1.0, float(theta) / 4.0) * (1.0 if nclust is None else min(1.0, nclust / 60))
        for b in np.unique(batch):
            m = batch == b
            z[m] -= w * (x[m].mean(axis=0) - x.mean(axis=0))
        out = types.SimpleNamespace(Z_corr=z.T, objective_harmony=[3.0, 2.0, 1.5],
                                    R=np.random.default_rng(0).dirichlet(np.ones(int(nclust or 4)), size=len(x)).T)
        return out

    module.run_harmony = run_harmony
    return module


def _fake_scanorama():
    module = types.ModuleType("scanorama")
    module.assemble = lambda datasets, batch_size=5000: [np.asarray(data) for data in datasets]
    return module


@pytest.fixture()
def stand_ins(monkeypatch):
    monkeypatch.setitem(sys.modules, "harmonypy", _fake_harmonypy())
    monkeypatch.setitem(sys.modules, "scanorama", _fake_scanorama())
    monkeypatch.setattr(quant, "silhouette_batch", lambda f, l, b, metric="cosine": 1 - _between(f, b))
    monkeypatch.setattr(quant, "lisi_batch", lambda f, b, n_neighbors=25: 1 - _between(f, b))
    monkeypatch.setattr(quant, "graph_conn", lambda f, l, n_neighbors=25: _between(f, l))
    monkeypatch.setattr(quant, "lisi_label", lambda f, l, n_neighbors=25: _between(f, l))
    monkeypatch.setattr(quant, "asw", lambda f, l, metric="cosine": _between(f, l))
    monkeypatch.setattr(quant, "nmi_ari", lambda f, l, n_neighbors=25, resolution=1.0:
                        {"nmi": _between(f, l), "ari": _between(f, l)})
    monkeypatch.setattr(quant, "average_precision_negcon", lambda meta, f, lk, pt, **kw: _between(
        np.asarray(f)[quant._map_index(meta, pt, lk)], meta.loc[quant._map_index(meta, pt, lk), lk]))
    monkeypatch.setattr(quant, "average_precision_nonrep", lambda meta, f, lk, pt, **kw: _between(
        np.asarray(f)[quant._map_index(meta, pt, lk, True)], meta.loc[quant._map_index(meta, pt, lk, True), lk]))
    monkeypatch.setattr(quant, "replicate_retrieval", lambda meta, f, batch_key, plate_types, cross: 0.3 if cross else 0.6)
    monkeypatch.setattr(control_mod, "N_HALVES", 4)


def _synthetic_prep(prep, n_sources=3, plates_per_source=2, n_trt=24, n_pos=2, n_dmso=6, seed=0):
    rng = np.random.default_rng(seed)
    trt = [f"JCP2022_{i:06d}" for i in range(n_trt)]
    pos = [f"JCP2022_9{i:05d}" for i in range(n_pos)]
    effect = {c: rng.normal(0, 1.0, 8) for c in trt + pos}
    sources = [f"source_{s}" for s in (2, 6, 10, 3, 8)[:n_sources]]
    offset = {s: rng.normal(0, 2.0, 8) for s in sources}
    rows, feats = [], []
    for si, s in enumerate(sources):
        for p in range(plates_per_source):
            layout = trt + pos + ["DMSO"] * n_dmso
            for w, c in enumerate(layout):
                pert = "negcon" if c == "DMSO" else ("poscon" if c in pos else "trt")
                rows.append({"Metadata_Source": s, "Metadata_Batch": f"{s}_B{p}", "Metadata_Plate": f"{s}_P{p}",
                             "Metadata_Well": f"W{(w + 7 * (si * plates_per_source + p)) % len(layout):02d}",
                             "Metadata_JCP2022": c, "Metadata_PlateType": "TARGET2", "Metadata_PertType": pert})
                feats.append((0 if c == "DMSO" else effect[c]) + offset[s] + rng.normal(0, 0.3, 8))
    meta = pd.DataFrame(rows)
    split_of = assign_split(meta.loc[meta.Metadata_PertType == "trt", "Metadata_JCP2022"].unique())
    meta["split"] = [split_of[c] if t == "trt" else t for c, t in zip(meta.Metadata_JCP2022, meta.Metadata_PertType)]
    prep.mkdir(parents=True)
    np.save(prep / "features.npy", np.asarray(feats, dtype=np.float32))
    meta.to_parquet(prep / "meta.parquet")
    (prep / "PREPARE.json").write_text(json.dumps({
        "scenario": {"name": "scenario_2", "scenario": "scenario_2", "sources": sources, "plate_types": ["TARGET2"],
                     "batch_key": "Metadata_Source", "label_key": "Metadata_JCP2022"},
        "split_rule": "test", "sha256": {n: sha256_file(prep / n) for n in PROTECTED}}))
    return meta


@pytest.fixture()
def prepared(tmp_path, stand_ins):
    prep = tmp_path / "prep"
    meta = _synthetic_prep(prep)
    control_mod.control(prep)
    return prep, meta


def test_the_slurm_time_left_becomes_a_deadline():
    from benchmarks.batch_correction_JUMP.runners.shape_arm import _deadline_from

    assert _deadline_from("1-02:03:04", 0.0) == 93784 and _deadline_from("02:03:04", 0.0) == 7384
    assert _deadline_from("03:04", 0.0) == 184
    assert _deadline_from("UNLIMITED", 0.0) is None and _deadline_from("", 0.0) is None


def test_scenarios_and_split():
    ids = [f"JCP2022_{i}" for i in range(11)]
    a, b = assign_split(ids), assign_split(list(reversed(ids)))
    assert a == b and sum(v == "val" for v in a.values()) == 5
    for i in range(1, 6):
        assert load_scenario(f"scenario_{i}")["batch_key"] == ("Metadata_Batch" if i == 1 else "Metadata_Source")


def test_public_roster_is_raw_plus_five_official_arms():
    from benchmarks.batch_correction_JUMP.runners import benchmark_contract

    assert benchmark_contract.arm_ids(include_raw=True) == (
        "raw", "harmony", "scanorama", "gpt_luna",
        "react_luna_harmony", "shape_luna_harmony",
    )
    assert [arm.display_name for arm in benchmark_contract.OFFICIAL_ARMS] == [
        "Harmony", "Scanorama", "GPT-5.6 luna",
        "ReAct (GPT-5.6 luna + Harmony)",
        "SHAPE (GPT-5.6 luna, Harmony)",
    ]


class ReactEngine:
    """A fake engine: GPT-only's plan, and the ReAct replies `policy(n, prompt)` (n: the ReAct reply index)."""

    def __init__(self, policy=None):
        self.policy = policy or (lambda n, prompt: react_act("call", [{}]) if n == 0 else react_act("finish",
                                                                                                    deliver="agent_1"))
        self.prompts = []

    def generate(self, content, response_format=None, **kwargs):
        from benchmarks._shared.react_arm import ReactAction
        from benchmarks.batch_correction_JUMP.runners.llm_arms import GPTPlan
        from shape.engine import usage

        usage.record({"total_tokens": 10})          # as a real engine records its usage
        if response_format is GPTPlan:
            return {"content": GPTPlan(reasoning="partial centering", action="batch_center", strength=0.5),
                    "usage": {"total_tokens": 10}}
        assert response_format is ReactAction
        prompt = content[0] if isinstance(content, list) else content
        self.prompts.append(prompt)
        return {"content": self.policy(len(self.prompts) - 1, prompt), "usage": {"total_tokens": 10}}


def react_act(action, calls=None, deliver=None, thought="t"):
    from benchmarks._shared.react_arm import ParamValue, ReactAction, ReactCall

    return ReactAction(thought=thought, action=action, deliver=deliver,
                       calls=[ReactCall(params=[ParamValue(name=k, value=v) for k, v in c.items()])
                              for c in (calls or [])])


def test_gpt_only_and_react_have_distinct_tool_access(prepared, tmp_path):
    from benchmarks.batch_correction_JUMP.runners.llm_arms import run_gpt_only, run_react

    prep, _ = prepared
    engine = ReactEngine()
    gpt = run_gpt_only(prep, tmp_path / "gpt", engine=engine, require_tokens=False)
    react = run_react(prep, tmp_path / "react", engine=engine, cores=4, require_tokens=False)
    assert gpt["arm_id"] == "gpt_luna" and gpt["tool_calls"] == 0
    assert react["arm_id"] == "react_luna_harmony" and react["tool_calls"] == 1
    assert Path(gpt["array"]).is_file() and Path(react["array"]).is_file()


def test_the_calibration_view_holds_no_evaluation_labels(prepared):
    from benchmarks.batch_correction_JUMP.runners.shape_arm import calibration_view

    _, meta = prepared
    view = calibration_view(meta, batch_key="Metadata_Source", plate_types=["TARGET2"])
    assert set(meta.loc[view.rows, "split"]) == {"val", "negcon", "poscon"}
    test_ids = set(meta.loc[meta.split == "test", "Metadata_JCP2022"])
    assert not test_ids & set(view.meta["Metadata_JCP2022"]) and "split" not in view.meta


class Reasoner:
    """The SHAPE arm's reasoner: the admission follows its rule, one round of two moves from the input, then no
    moves (satisfied), and the final choice agent_1."""

    def __init__(self):
        self.questions = []

    def generate(self, content, response_format=None, **kwargs):
        from shape.agent.verification.verifiers.batch_correction_unified import (
            AdmissionVerdict, FinalChoice, LeverChange, WaveMove, WavePlan,
        )
        prompt = content[0] if isinstance(content, list) else content
        self.questions.append(response_format.__name__)
        usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        from shape.engine import usage as token_usage

        token_usage.record(usage)                  # as a real engine records its usage
        if response_format is AdmissionVerdict:
            rule = prompt.split("Deterministic rule outcome: ", 1)[1].split(".", 1)[0]
            return {"content": AdmissionVerdict(decision=rule, reason_code="rule", reasoning="the measured rule"),
                    "usage": usage}
        if response_format is FinalChoice:
            return {"content": FinalChoice(call="agent_1", reasoning="the better calibration"), "usage": usage}
        rounds = sum(q == response_format.__name__ for q in self.questions)
        moves = [WaveMove(parent="input", changes=[LeverChange(lever=k, value=v)], diagnosis="explore")
                 for k, v in (("theta", "2.5"), ("sigma", "0.07"))] if rounds == 1 else []
        return {"content": response_format(analysis="one broad round, then stop", moves=moves), "usage": usage}


def test_a_run_end_to_end(prepared, tmp_path, monkeypatch):
    from benchmarks.batch_correction_JUMP.runners import score, suite
    from benchmarks.batch_correction_JUMP.runners.llm_arms import run_gpt_only, run_react
    from benchmarks.batch_correction_JUMP.runners.shape_arm import run
    from shape.engine.errors import NoEngineError

    for name in ("SHAPE_SEARCH_STRATEGY", "SHAPE_SEARCH_STRATEGY_BATCH_CORRECTION"):
        monkeypatch.delenv(name, raising=False)
    prep, _ = prepared
    with pytest.raises(NoEngineError):                      # no reasoner is a fatal error, never a rule fallback
        run(prep, tmp_path / "no_engine", model="none", max_rounds=8, noise_band=0.002, cores=16, engine=None)
    suite_root = tmp_path / "six-arm"
    run_dir = suite_root / "shape_luna_harmony"
    reasoner = Reasoner()
    final = run(prep, run_dir, model=LUNA_MODEL, max_rounds=8, noise_band=0.002, cores=16, engine=reasoner)
    assert "WavePlan" in reasoner.questions and "FinalChoice" in reasoner.questions
    ledger = score.jsonl(run_dir / "logs" / "tool_calls.jsonl")
    started = sum(r["event"] == "start" for r in ledger)
    assert started == 2 and started == sum(r["event"] == "end" for r in ledger)     # the reasoner's one round
    starts = [r for r in ledger if r["event"] == "start"]
    assert {r["parent"] for r in starts} == {"input"}
    deltas = [{k: v for k, v in r["config"].items() if v != DEFAULT_CONFIG.get(k)} for r in starts]
    assert sorted(deltas, key=json.dumps) == [{"sigma": 0.07}, {"theta": 2.5}]
    trial0 = json.loads((run_dir / "logs" / "input_trial.json").read_text())
    # the small synthetic input resolves to 1 BLAS thread per call: 16 cores hold 16 calls at once
    assert trial0["status"] == "UNCORRECTED_INPUT" and trial0["concurrency_cap"] == 16
    assert trial0["noise"]["overall"]["used"] == 0.002                         # the stand-in metrics are deterministic
    assert final["decision"] == "RUN" and final["from_call"] in {r["call"] for r in ledger if r["event"] == "end"}
    decision = json.loads((run_dir / "submission" / "decision.json").read_text())
    assert decision == json.loads(json.dumps(final, default=str))
    end = next(r for r in ledger if r["event"] == "end")
    assert {"val", "diagnostics", "delta_vs_parent", "reconciliation", "tool_report"} <= set(end)
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text())
    assert agent["stopped_because"] == "satisfied"                          # the reasoner stopped
    assert agent["n_rounds"] <= 8 and "total_tokens" in agent["tokens"]

    result = score.final(run_dir, prep)
    assert result["decision"] == "RUN" and set(result["scores"]) == {"agent", "harmony", "scanorama", "raw"}
    assert result["comparisons_test"]["agent_vs_harmony"]["overall"]["n_halves"] == 4
    assert result["usage"]["tool_calls_started"] == started

    engine = ReactEngine(lambda n, prompt: react_act("finish", deliver="default"))   # keeps the input, no call
    run_gpt_only(prep, suite_root / "gpt_luna", engine=engine, require_tokens=False)
    run_react(prep, suite_root / "react_luna_harmony", engine=engine, cores=16, require_tokens=False)
    report = suite.write_report(prep, suite_root)
    assert tuple(report["roster"]) == (
        "raw", "harmony", "scanorama", "gpt_luna",
        "react_luna_harmony", "shape_luna_harmony",
    )
    assert all(row["ap50_gain_per_tool_call"] is None for row in report["rows"])
    # the validation: ReAct delivering the input with no tool call is valid; its records must exist and agree
    from benchmarks.batch_correction_JUMP.runners import validate_suite

    argv = ["validate_suite", "--run", str(suite_root), "--expected-commit", report["git_commit"]]
    monkeypatch.setattr(sys, "argv", argv)
    rows = {row["arm_id"]: row for row in report["rows"]}
    assert rows["react_luna_harmony"]["tool_calls"] == 0 and validate_suite.main() == 0

    # the record set for results/: everything small, no matrices, a manifest naming what is missing
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_run_records",
                                                  Path(__file__).parents[1] / "export_run_records.py")
    records = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(records)
    (tmp_path / "run.log").write_text("VALID_RUN=OK\n", encoding="utf-8")
    run_root = tmp_path / "run"
    (run_root / "prep" / "shape_control").mkdir(parents=True)
    for name in ("PREPARE.json", "shape_control/CONTROL.json"):
        (run_root / "prep" / name).write_bytes((prep / name).read_bytes())
    (run_root / "RUN_META.json").write_text("{}", encoding="utf-8")
    (run_root / "COMPLETE").write_text("VALID_RUN=OK\n", encoding="utf-8")
    import shutil

    shutil.copytree(suite_root, run_root / "suite")
    manifest = records.export(run_root, tmp_path / "records", tmp_path / "run.log")
    files = set(manifest["files"])
    assert {"shape_luna_harmony/logs/tool_calls.jsonl", "compare/compare.json", "compare/summary.md",
            "compare/paired_vs_shape.csv", "audit/index.json", "audit/shape_luna_harmony.json", "logs/run.log",
            "control/CONTROL.json"} <= files
    assert not any(name.endswith((".npy", ".parquet")) or name.startswith("suite/") for name in files)
    assert manifest["missing"] == [] and "suite/BENCHMARK_REPORT.md" not in files
    index = json.loads((tmp_path / "records" / "audit" / "index.json").read_text(encoding="utf-8"))
    assert [r["arm"] for r in index["records"]] == list(report["roster"])
    assert all(r["delivered"] and r["missing_sources"] == 0 for r in index["records"])
    shape_audit = json.loads((tmp_path / "records" / "audit" / "shape_luna_harmony.json").read_text(encoding="utf-8"))
    assert [s["call"] for s in shape_audit["steps"] if s["kind"] == "tool_call"] == [
        json.loads(line)["call"] for line in (suite_root / "shape_luna_harmony" / "logs" / "tool_calls.jsonl")
        .read_text(encoding="utf-8").splitlines() if json.loads(line)["event"] == "start"]
    # an earlier record set of the run-directory layout, re-exported: its run's path and its files, byte for byte
    (run_root / "MANIFEST.json").write_text(json.dumps({"run_dir": "/scratch/run"}), encoding="utf-8")
    again = records.export(run_root, tmp_path / "again", tmp_path / "run.log")
    assert again["run_dir"] == "/scratch/run"
    assert {k: v for k, v in again["files"].items() if k != "audit/index.json"} == {
        k: v for k, v in manifest["files"].items() if k != "audit/index.json"}
    (run_root / "MANIFEST.json").unlink()
    with pytest.raises(SystemExit, match="exists"):
        records.export(run_root, tmp_path / "records")
    (suite_root / "react_luna_harmony" / "react" / "scenario_2" / "SELECTED.json").unlink()
    with pytest.raises(RuntimeError, match="ReAct record missing"):
        validate_suite.main()


def test_abstain_scores_the_uncorrected_input(prepared, tmp_path):
    from benchmarks.batch_correction_JUMP.runners import score

    prep, _ = prepared
    run_dir = tmp_path / "abstain"
    (run_dir / "submission").mkdir(parents=True)
    (run_dir / "submission" / "decision.json").write_text(json.dumps({"decision": "ABSTAIN"}))
    result = score.final(run_dir, prep)
    raw, agent = result["scores"]["raw"]["test"], result["scores"]["agent"]["test"]
    assert result["tool_run"]["status"] == "UNCORRECTED_INPUT" and agent["overall"] == pytest.approx(raw["overall"])


def test_gpt_only_uses_the_shared_token_limit(prepared, tmp_path):
    from benchmarks.batch_correction_JUMP.runners.benchmark_contract import LUNA_MAX_TOKENS
    from benchmarks.batch_correction_JUMP.runners.llm_arms import GPTPlan, run_gpt_only

    limits = []

    class Engine:
        def generate(self, prompt, response_format=None, max_tokens=None, **kwargs):
            limits.append(max_tokens)
            return {"content": GPTPlan(reasoning="none", action="identity", strength=0.0),
                    "usage": {"total_tokens": 10}}

    prep, _ = prepared
    run_gpt_only(prep, tmp_path / "gpt", engine=Engine(), require_tokens=False)
    assert limits == [LUNA_MAX_TOKENS]
    record = json.loads((tmp_path / "gpt" / "ARM_OUTPUT.json").read_text(encoding="utf-8"))
    summary = record["matrix_summary"]                     # what the model saw, kept with its answer
    assert summary["n_profiles"] == 192 and set(summary["batch_counts"]) == {"source_2", "source_6", "source_10"}
    assert 0 <= summary["batch_centroid_variance_ratio"] <= 1
    assert record["prompt"].endswith(json.dumps(summary, sort_keys=True))


def test_a_stale_control_is_refused(prepared, tmp_path, monkeypatch):
    from benchmarks.batch_correction_JUMP.runners import score

    prep, _ = prepared
    run_dir = tmp_path / "stale"
    (run_dir / "submission").mkdir(parents=True)
    (run_dir / "submission" / "decision.json").write_text(json.dumps({"decision": "ABSTAIN"}))
    monkeypatch.setattr(control_mod, "provenance", lambda: {"tool_sha256": "other"})
    with pytest.raises(SystemExit, match="CONTROL_STALE"):
        score.final(run_dir, prep)


class StubTool:
    """A stand-in for `control.run_tool`: the tool's result keys, its matrix the batch centroids shifted by
    theta / 10 (a stronger correction, a higher stand-in calibration overall)."""

    def __init__(self, features, batch, fail=False):
        self.features, self.batch, self.fail, self.calls = np.asarray(features, dtype=float), batch, fail, []

    def __call__(self, params):
        self.calls.append(dict(params))
        if self.fail:
            raise MemoryError("out of memory")
        x, out = self.features, self.features.copy()
        for b in np.unique(self.batch):
            rows = self.batch == b
            out[rows] -= min(1.0, float(params["theta"]) / 10) * (x[rows].mean(0) - x.mean(0))
        return {"status": "CORRECTED_HARMONY", "corrected_cls_output": out, "batch_effect": {"detected": True},
                "harmony": {"applied": True, "parameters": dict(params)}, "preprocess": {}, "telemetry": {},
                "gt_free_metrics": {"batch_effect_detected": True, "displacement": 0.1}, "seconds": 0.0}


def test_react_is_the_shared_runner_with_shapes_tool_levers_and_cap(prepared, tmp_path):
    from benchmarks._shared.react_arm import BUDGET, TUNING_LINE
    from benchmarks.batch_correction_JUMP.runners import llm_arms, score
    from shape.agent.verification.verifiers.batch_correction_unified import BatchCorrectionVerifier, concurrency_cap

    prep, meta = prepared
    features = np.load(prep / "features.npy")
    tool = StubTool(features, meta["Metadata_Source"].astype(str).to_numpy())
    replies = [react_act("call", [{}, {}, {}]),                                       # over the cap of 2
               react_act("call", [{"theta": 8.0}, {"theta": 1.5, "input_scale": "zscore"}]),
               react_act("call", [{"bogus": 1.0}]),                                    # not a parameter
               react_act("finish", deliver="agent_2")]
    engine = ReactEngine(lambda n, prompt: replies[n])
    run_dir = tmp_path / "react"
    react = llm_arms.run_react(prep, run_dir, engine=engine, cores=2, tool=tool, require_tokens=False)

    # parameters: SHAPE's lever set, name / type / default only
    verifier = BatchCorrectionVerifier()
    names = [p["name"] for p in react["parameters"]]
    assert names == verifier.levers and set(names) == set(DEFAULT_CONFIG)
    assert {p["name"]: p["default"] for p in react["parameters"]} == DEFAULT_CONFIG
    types_ = {p["name"]: p["type"] for p in react["parameters"]}
    assert types_["theta"] == "number" and types_["nclust"] == "integer or null"
    assert types_["input_pca"] == "integer" and types_["input_scale"] == 'choice of "none", "zscore", "l2"'
    assert types_["lambda_value"] == "number or null"
    # the prompt: the SHAPE arm's brief, the table, the factual line; no tool documentation, metric definitions,
    # CARD text or SHAPE readouts
    prompt = engine.prompts[0]
    from benchmarks.batch_correction_JUMP.runners.shape_arm import task_brief

    brief = task_brief({"name": "scenario_2"}, "source", BUDGET)               # the SHAPE arm's task text
    assert f"## TASK\n{brief}\n" in prompt and react["task_text"] == brief
    assert f"At most {BUDGET} rounds of tool calls" in prompt and TUNING_LINE in prompt
    assert "- theta | number | 2.0" in prompt and '- input_scale | choice of "none", "zscore", "l2" | "none"' in prompt
    assert "- nclust | integer or null | null" in prompt
    for absent in ("harmonypy run_harmony arguments", "diversity penalty", "overall = 0.4", "batch_mean averages",
                   "## Levers", "Family:", "Range:", "Effective values", "Noise per calibration readout",
                   "admissible", "pareto", "higher = better"):
        assert absent not in prompt
    assert "stated noise of the calibration readouts: 0.002" in prompt and llm_arms.TRIAL0_TEXT in prompt
    assert llm_arms.CALIBRATION_TEXT in prompt
    import inspect

    from benchmarks._shared import react_arm

    label = llm_arms.BatchToolAdapter.trial0_label
    assert label == "the uncorrected input (no batch correction)"
    if "trial0_label" in inspect.getsource(react_arm):                # the shared runner reads the adapter's label
        assert f"Round 0 ({label}; not a round)" in prompt and "the tool at its defaults" not in prompt
    assert react["round_cap_source"] == llm_arms.CAP_FROM_SHAPE
    # trial 0: the uncorrected input, no tool call, not a round; no observation carries SHAPE's calibration scores
    calls = json.loads((run_dir / "react" / "scenario_2" / "CALLS.json").read_text(encoding="utf-8"))
    assert [c["call_id"] for c in calls] == ["default", "agent_1", "agent_2"]
    trial0 = calls[0]
    assert trial0["round"] == 0 and trial0["output"]["status"] == "UNCORRECTED_INPUT"
    assert "calibration" not in trial0["output"] and "gt_free_metrics" in trial0["output"]
    assert not any('"calibration"' in p or '"bio_mean"' in p for p in engine.prompts)
    assert "corrected_cls_output" not in trial0["output"]
    assert '"status": "UNCORRECTED_INPUT"' in engine.prompts[0]
    # the cap: SHAPE's size-aware concurrency_cap on these cores; over it, an error that uses its round
    assert react["round_cap"] == concurrency_cap(2, features) == 2
    assert "3 calls requested; at most 2 calls per round" in engine.prompts[1]
    assert "unknown parameters ['bogus']" in engine.prompts[3]
    # the tool: called as the SHAPE arm calls it, the full lever configuration, only for the agent's calls
    assert len(tool.calls) == 2 == react["tool_calls"]
    assert tool.calls[0] == {**DEFAULT_CONFIG, "theta": 8.0}
    assert tool.calls[1] == {**DEFAULT_CONFIG, "theta": 1.5, "input_scale": "zscore"}
    observed = calls[1]["output"]
    assert observed["status"] == "CORRECTED_HARMONY" and "calibration" not in observed
    assert observed["gt_free_metrics"] == {"batch_effect_detected": True, "displacement": 0.1}
    # delivery: the agent's finish, never the harness's pick by score
    ledger = {r["call"]: r for r in score.jsonl(run_dir / "logs" / "tool_calls.jsonl") if r["event"] == "end"}
    assert ledger["agent_1"]["val"]["overall"] > ledger["agent_2"]["val"]["overall"]
    assert react["selected_call"] == "agent_2" and react["source"] == "agent_finish"
    assert np.array_equal(np.load(react["array"]), np.load(run_dir / "trials" / "agent_2.npy"))
    # rounds: three (the invalid ones count), the finish is not a round; the records
    assert react["rounds"] == 3 and react["budget"] == BUDGET and react["stop_reason"] == "finish"
    for name in ("TRACE.jsonl", "CALLS.json", "SELECTED.json"):
        assert (run_dir / "react" / "scenario_2" / name).is_file()
    summary = json.loads((run_dir / "react" / "REACT_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["default_counts_as_round"] is False and summary["round_cap"] == 2
    assert summary["totals"]["calls"] == 3 and summary["totals"]["rounds"] == 3
    record = json.loads((run_dir / "ARM_OUTPUT.json").read_text(encoding="utf-8"))
    assert record["arm_id"] == "react_luna_harmony" and record["tool"] == "Batch_Corrector_Tool"
    assert [h["call"] for h in record["history"]] == ["default", "agent_1", "agent_2"]
    assert record["history"][1]["calibration"] == ledger["agent_1"]["val"]      # the records keep the scores
    assert record["history"][2]["config"]["input_scale"] == "zscore" and record["history"][2]["step"] == 2


@pytest.mark.parametrize("multimodal", [False, True])
def test_a_truncated_structured_parse_keeps_its_token_usage(monkeypatch, multimodal):
    openai = pytest.importorskip("openai")
    from openai.types.chat import ChatCompletion

    from shape.engine.openai import ChatOpenAI

    truncated = ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "gpt-5.6-luna",
        "choices": [{"index": 0, "finish_reason": "length",
                     "message": {"role": "assistant", "content": None}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}})
    fallback = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="{}"))],
        usage=types.SimpleNamespace(prompt_tokens=50, completion_tokens=30, total_tokens=80))

    def parse_truncated():
        raise openai.LengthFinishReasonError(completion=truncated)

    # the engine's call (`_parse_structured`): with_raw_response.parse(...), then .parse() of the raw response,
    # where the SDK raises LengthFinishReasonError for a completion cut off at the token limit
    raw = types.SimpleNamespace(parse=parse_truncated, text=truncated.model_dump_json())
    engine = ChatOpenAI.__new__(ChatOpenAI)
    engine.model_string, engine.system_prompt, engine.enable_cache = "gpt-5.6-luna", "system", False
    engine._structured, engine._extra_body, engine._sanitize_messages = True, {}, False
    engine.client = types.SimpleNamespace(
        beta=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            with_raw_response=types.SimpleNamespace(parse=lambda **kwargs: raw)))),
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kwargs: fallback)))
    monkeypatch.delenv("SHAPE_REQUIRE_AGENTIC", raising=False)
    out = (engine._generate_multimodal(["prompt"], response_format=dict, max_tokens=200) if multimodal
           else engine._generate_text("prompt", response_format=dict, max_tokens=200))
    assert out["usage"] == {"prompt_tokens": 100, "completion_tokens": 230, "total_tokens": 330}


def test_a_cached_stage_is_reused_only_with_the_same_fingerprint(tmp_path):
    from benchmarks.batch_correction_JUMP.runners.prepare import preprocess

    runs = []

    def stage(name):
        def write(*args):
            runs.append(name)
            Path(args[-1]).write_text(name, encoding="utf-8")
        return write

    modules = {
        "io": types.SimpleNamespace(write_parquet=stage("raw")),
        "stats": types.SimpleNamespace(compute_negcon_stats=stage("neg_stats"),
                                       select_variant_features=stage("variant_feats")),
        "normalize": types.SimpleNamespace(mad=stage("mad")),
        "transform": types.SimpleNamespace(rank_int=stage("mad_int")),
        "feature_selection": types.SimpleNamespace(select_features=stage("mad_int_featselect")),
    }
    scenario = {"name": "scenario_2", "sources": [], "plate_types": []}
    base, cache = tmp_path / "base", tmp_path / "cache"
    base.mkdir()
    first = {"scenario": "scenario_2", "vendor_sha256": "a"}
    preprocess(base, cache, scenario, modules, first)
    assert len(runs) == 6
    runs.clear()
    assert preprocess(base, cache, scenario, modules, first)["reused_stages"] == [
        "raw", "neg_stats", "variant_feats", "mad", "mad_int", "mad_int_featselect"] and runs == []
    (cache / "mad.done").write_text(json.dumps({"scenario": "scenario_2"}), encoding="utf-8")   # old marker
    stamp = preprocess(base, cache, scenario, modules, first)
    assert stamp["reused_stages"] == ["raw", "neg_stats", "variant_feats"]
    assert runs == ["mad", "mad_int", "mad_int_featselect"]
    runs.clear()
    preprocess(base, cache, scenario, modules, {**first, "vendor_sha256": "b"})
    assert len(runs) == 6


def test_a_crashed_call_is_an_observation_and_an_unfinished_session_delivers_the_input(prepared, tmp_path):
    from benchmarks._shared.react_arm import BUDGET
    from benchmarks.batch_correction_JUMP.runners import llm_arms

    prep, meta = prepared
    features = np.load(prep / "features.npy")
    tool = StubTool(features, meta["Metadata_Source"].astype(str).to_numpy(), fail=True)
    engine = ReactEngine(lambda n, prompt: react_act("call", [{"theta": 4.0}]) if n == 0
                         else react_act("finish", deliver="agent_1"))         # a failed call: never delivered
    react = llm_arms.run_react(prep, tmp_path / "react", engine=engine, cores=2, tool=tool, require_tokens=False)
    assert "MemoryError: out of memory" in engine.prompts[1]
    assert react["tool_calls"] == 1 and react["rounds"] == BUDGET and len(engine.prompts) == BUDGET + 1
    assert react["selected_call"] == "default" and react["source"] == "default_session_unfinished"
    assert np.array_equal(np.load(react["array"]), features.astype(np.float32))
    assert react["history"][1]["error"] == "MemoryError: out of memory"


def test_react_alone_on_a_finished_runs_prep_is_scored_into_a_report_row(prepared, tmp_path, monkeypatch):
    from benchmarks.batch_correction_JUMP.runners import benchmark_contract, llm_arms
    from shape.engine.errors import NoEngineError

    prep, _ = prepared
    with pytest.raises(NoEngineError):
        llm_arms.run_react(prep, tmp_path / "none", engine=None, cores=2, require_tokens=False)
    report = llm_arms.run_react_only(prep, tmp_path / "out", engine=ReactEngine(), cores=2)
    row = report["row"]
    benchmark_contract.validate_report_row(row)
    assert row["arm_id"] == "react_luna_harmony" and row["tool_calls"] == 1 and row["n_images"] == 192
    assert row["overall"] == report["scores"]["test"]["overall"] and len(report["scores"]["test_halves"]) == 4
    assert report["control_provenance"] == {"matches_current": True, "differs": {}}
    assert report["compare_run"] is None and report["round_cap_source"] == llm_arms.CAP_FROM_SHAPE
    assert (tmp_path / "out" / "react_luna_harmony" / "ARM_OUTPUT.json").is_file()
    with pytest.raises(SystemExit, match="exists"):
        llm_arms.run_react_only(prep, tmp_path / "out", engine=ReactEngine(), cores=2)
    # a control made with other tool code (the reported runs' commit): recorded, not refused
    monkeypatch.setattr(control_mod, "provenance", lambda: {"tool_sha256": "other"})
    again = llm_arms.run_react_only(prep, tmp_path / "out2", engine=ReactEngine(), cores=2)
    assert again["control_provenance"]["matches_current"] is False
    assert again["control_provenance"]["differs"]["tool_sha256"]["current"] == "other"
    # compared with a run: the per-round cap its SHAPE arm recorded, not this machine's
    compared = tmp_path / "reported"
    (compared / "suite" / "shape_luna_harmony").mkdir(parents=True)
    meta = compared / "suite" / "shape_luna_harmony" / "RUN_META.json"
    meta.write_text(json.dumps({"concurrency_cap": 3, "cores": 16}), encoding="utf-8")
    third = llm_arms.run_react_only(prep, tmp_path / "out3", engine=ReactEngine(), cores=2, compare_run=compared)
    assert third["round_cap"] == 3 and str(meta) in third["round_cap_source"]
    summary = json.loads((tmp_path / "out3" / "react_luna_harmony" / "react" / "REACT_SUMMARY.json").read_text())
    assert summary["round_cap"] == 3
    meta.write_text(json.dumps({"cores": 16}), encoding="utf-8")
    with pytest.raises(SystemExit, match="NO_COMPARISON_CAP"):
        llm_arms.run_react_only(prep, tmp_path / "out4", engine=ReactEngine(), cores=2, compare_run=compared)
    with pytest.raises(SystemExit, match="NO_COMPARISON_CAP"):
        llm_arms.run_react_only(prep, tmp_path / "out5", engine=ReactEngine(), cores=2, compare_run=tmp_path / "none")


def test_every_scenario_is_a_valid_report_scenario_and_missing_inputs_stop_prepare(tmp_path):
    from benchmarks.batch_correction_JUMP.runners import benchmark_contract
    from benchmarks.batch_correction_JUMP.runners.prepare import prepare

    row = {"scenario": "scenario_4", "arm_id": "raw", **{m: None for m in benchmark_contract.REPORT_METRICS}}
    benchmark_contract.validate_report_row(row)
    with pytest.raises(ValueError, match="must be one of"):
        benchmark_contract.validate_report_row({**row, "scenario": "scenario_9"})
    (tmp_path / "inputs" / "source_2" / "workspace" / "profiles").mkdir(parents=True)
    missing = "['source_3', 'source_6', 'source_8', 'source_10']"   # source_2 has its profiles
    with pytest.raises(SystemExit, match=re.escape(f"MISSING_INPUTS: scenario_4 needs the profiles of {missing}")):
        prepare(tmp_path, tmp_path / "prep", scenario="scenario_4", microscope_config=tmp_path / "m.csv",
                arevalo_cache=tmp_path / "cache")


def test_react_python_inputs_hold_the_label_free_inputs_and_the_call_outputs(prepared, tmp_path):
    from benchmarks._shared.react_arm import ReactAction
    from benchmarks.batch_correction_JUMP.runners import llm_arms

    prep, meta = prepared
    features = np.load(prep / "features.npy")
    tool = StubTool(features, meta["Metadata_Source"].astype(str).to_numpy())
    code = ("import json, numpy as np, pandas as pd\n"
            "print(json.dumps(sorted(e['path'] for e in json.load(open('manifest.json')))))\n"
            "print(json.dumps(list(pd.read_csv('inputs/metadata.csv').columns)))\n"
            "cal = pd.read_csv('inputs/calibration_labels.csv')\n"
            "print(json.dumps({'columns': list(cal.columns), 'rows': [int(r) for r in cal['row']],\n"
            "                  'labels': sorted(set(cal['Metadata_JCP2022'].astype(str)))}))\n"
            "print(np.load('calls/agent_1/corrected.npy').shape == np.load('inputs/uncorrected.npy').shape)\n")
    replies = [react_act("call", [{"theta": 8.0}]), ReactAction(thought="t", action="python", code=code),
               react_act("finish", deliver="agent_1")]
    run_dir = tmp_path / "react"
    react = llm_arms.run_react(prep, run_dir, engine=ReactEngine(lambda n, prompt: replies[n]), cores=2, tool=tool,
                               require_tokens=False)
    assert react["selected_call"] == "agent_1" and react["totals"]["python_steps"] == 1
    rows = [json.loads(line) for line in
            (run_dir / "react" / "scenario_2" / "TRACE.jsonl").read_text(encoding="utf-8").splitlines()]
    [step] = [r for r in rows if r["kind"] == "python"]
    assert step["python"]["exit"] == 0, step["python"]["output"]
    listed, columns, calibration, same = step["python"]["output"].strip().splitlines()
    assert json.loads(listed) == ["calls/agent_1/corrected.npy", "inputs/calibration_labels.csv",
                                  "inputs/metadata.csv", "inputs/uncorrected.npy"]
    from benchmarks.batch_correction_JUMP.runners.shape_arm import calibration_view

    view = calibration_view(meta, batch_key="Metadata_Source", plate_types=["TARGET2"])
    calibration = json.loads(calibration)
    assert calibration["columns"][0] == "row" and "split" not in calibration["columns"]
    assert calibration["rows"] == [int(r) for r in view.rows]               # the calibration view, SHAPE's rows
    test_ids = set(meta.loc[meta.split == "test", "Metadata_JCP2022"].astype(str))
    assert not test_ids & set(calibration["labels"])                        # no evaluation compound
    assert set(json.loads(columns)) <= set(llm_arms.PYTHON_META_COLUMNS) and "Metadata_Plate" in json.loads(columns)
    assert same == "True" and step["python"]["violations"] == []
    adapter = llm_arms.BatchToolAdapter(features, meta["Metadata_Source"].astype(str).to_numpy(),
                                        batch_key="Metadata_Source", view=None, cores=1, unit="u",
                                        run_dir=tmp_path / "leak", meta=meta, private_roots=[prep / "meta.parquet"])
    adapter.paths["leak"] = prep / "meta.parquet"
    with pytest.raises(AssertionError, match="private"):
        adapter.python_inputs("u")


def test_react_round_calls_compute_one_at_a_time(tmp_path):
    """The batch ReAct adapter runs a round's calls one after another (two concurrent calls deadlocked on the cluster)."""
    import threading
    import time as _time
    import numpy as np
    from benchmarks.batch_correction_JUMP.runners import llm_arms

    active, peak, guard = [0], [0], threading.Lock()

    def tool(params):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        _time.sleep(0.05)
        with guard:
            active[0] -= 1
        return {"corrected_cls_output": np.zeros((4, 2), dtype=np.float32), "status": "ok"}

    class View:
        def score(self, matrix):
            return {"overall": 0.5}

    adapter = llm_arms.BatchToolAdapter(np.zeros((4, 2), dtype=np.float32), np.array(["a", "a", "b", "b"]),
                                        batch_key="batch", view=View(), cores=8, unit="s", run_dir=tmp_path,
                                        tool=tool, cap=4)
    threads = [threading.Thread(target=adapter.call, args=("s", {}, f"agent_{i}")) for i in range(1, 5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert peak[0] == 1 and adapter.n_tool_calls == 4


def test_the_effort_sweep_runs_each_level_with_its_effort_and_the_shared_token_cap(prepared, tmp_path, monkeypatch):
    """One SHAPE-arm run per level: the engine sees the level (SHAPE_REASONING_EFFORT, read per request) and the same
    per-reply cap; each run is scored and logs its replies; a recorded level is kept; the environment is restored."""
    from benchmarks.batch_correction_JUMP.runners import effort_sweep as sweep

    for name in ("SHAPE_SEARCH_STRATEGY", "SHAPE_SEARCH_STRATEGY_BATCH_CORRECTION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SHAPE_REASONING_EFFORT", "medium")
    monkeypatch.delenv("SHAPE_LLM_TIMEOUT", raising=False)
    prep, _ = prepared
    seen = []

    class Effortful(Reasoner):
        def generate(self, content, response_format=None, **kwargs):
            import os

            seen.append((os.environ["SHAPE_REASONING_EFFORT"], kwargs.get("max_tokens"),
                         os.environ["SHAPE_LLM_TIMEOUT"]))
            reply = super().generate(content, response_format=response_format, **kwargs)
            reply["usage"] = {**reply["usage"], "reasoning_tokens": 5 * len(seen)}
            return reply

    out = tmp_path / "sweep"
    report = sweep.run_sweep(prep, out, model=LUNA_MODEL, max_rounds=8, noise_band=0.002, cores=16,
                             efforts=("low", "high"), max_tokens=32000, timeout=900, engine_factory=Effortful)
    assert {e for e, _, _ in seen} == {"low", "high"} and {(m, t) for _, m, t in seen} == {(32000, "900")}
    assert [r["effort"] for r in report["rows"]] == ["low", "high"]
    for r in report["rows"]:
        run_dir = out / f"shape_{r['effort']}"
        assert json.loads((run_dir / "EFFORT.json").read_text())["reasoning_effort"] == r["effort"]
        assert r["decision"] == "RUN" and r["test_overall"] is not None and r["agent_vs_harmony_overall_2se"] >= 0
        assert r["replies"] == sum(1 for e, _, _ in seen if e == r["effort"]) and r["errors"] == 0
        assert r["reasoning_tokens_max"] >= r["reasoning_tokens_mean"] > 0
    import os

    assert os.environ["SHAPE_REASONING_EFFORT"] == "medium" and "SHAPE_LLM_TIMEOUT" not in os.environ
    assert (out / "SWEEP_REPORT.md").read_text().count("| low |") == 1 and (out / "SWEEP_REPORT.csv").exists()
    n = len(seen)
    sweep.run_sweep(prep, out, model=LUNA_MODEL, max_rounds=8, noise_band=0.002, cores=16,
                    efforts=("low", "high"), engine_factory=Effortful)              # both recorded: nothing reruns
    assert len(seen) == n
    with pytest.raises(SystemExit, match="unknown effort"):
        sweep.run_sweep(prep, out, model=LUNA_MODEL, max_rounds=8, noise_band=0.002, efforts=("max",),
                        engine_factory=Effortful)
