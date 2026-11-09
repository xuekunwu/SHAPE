"""Per-stage reasoning settings of the full harness (pipeline.REASONING, user decision 2026-10-13)."""
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.full_CPJUMP1.harness import cli, codex_arm as agent, compare, pipeline
from benchmarks.full_CPJUMP1.harness.data import read_json, write_json
from benchmarks.full_CPJUMP1.harness.pipeline import (EFFORTS, REASONING, CappedEngine, reasoning_protocol,
                                                      reasoning_record, stage_env, stage_reasoning)

from .test_codex import FakeCodex, d, gateway, image_stage   # noqa: F401  (pytest fixtures)

HIGH_EMBED = {"effort": "high", "max_tokens": 16000, "timeout_seconds": 900}
HIGH_BATCH = {"effort": "high", "max_tokens": 16000, "timeout_seconds": 900}
DEFAULT = {"effort": "medium", "max_tokens": 4000, "timeout_seconds": 120}
REASONER_ENV = ("SHAPE_REASONING_EFFORT", "SHAPE_LLM_TIMEOUT")


def test_the_table_resolves_per_stage_and_arm():
    assert stage_reasoning("shape", "segment") == stage_reasoning("shape", "crop") == DEFAULT
    assert stage_reasoning("shape", "embed") == HIGH_EMBED and stage_reasoning("shape", "batch") == HIGH_BATCH
    assert [stage_reasoning("codex", s) for s in ("segment", "crop", "embed", "batch")] == [
        {"model_reasoning_effort": "medium"}, {"model_reasoning_effort": "medium"},
        {"model_reasoning_effort": "high"}, {"model_reasoning_effort": "high"}]
    for arm in ("control", "claude", "biomni"):           # not in REASONING_ARMS: their own settings
        assert all(stage_reasoning(arm, s) is None for s in cli.STAGES)
    assert stage_reasoning("shape", "cluster") is None and stage_reasoning("codex", "cluster") is None
    assert set(REASONING) == set(cli.STAGES[:-1])
    assert all(e["effort"] in EFFORTS and e["codex_effort"] in (*EFFORTS, None) for e in REASONING.values())
    record = reasoning_record("shape")
    assert record["protocol"] == reasoning_protocol() and record["stages"]["batch"] == HIGH_BATCH
    with pytest.raises(ValueError, match="no column"):
        stage_reasoning("claude", "embed", {**reasoning_protocol(), "arms": ["shape", "codex", "claude"]})


def test_stage_env_is_the_shape_arms_only():
    assert stage_env("shape", "embed") == {"SHAPE_REASONING_EFFORT": "high", "SHAPE_LLM_TIMEOUT": "900"}
    assert stage_env("shape", "segment") == {"SHAPE_REASONING_EFFORT": "medium", "SHAPE_LLM_TIMEOUT": "120"}
    assert stage_env("shape", "cluster") == {}
    assert all(stage_env(arm, "embed") == {} for arm in ("control", "codex", "claude", "biomni"))


def test_segment_and_crop_settings_are_the_engines_defaults(monkeypatch):
    """CPJUMP1 default behaviour: segment and crop run what the engine does without any setting."""
    import inspect
    from shape.engine.openai import ChatOpenAI
    for name in REASONER_ENV:
        monkeypatch.delenv(name, raising=False)
    engine = ChatOpenAI(model_string="gpt-5.6-luna", api_key="offline-dummy", base_url="http://127.0.0.1:9/v1")
    params = {}
    engine._apply_backend_quirks(params)
    for stage in ("segment", "crop"):
        assert params["reasoning_effort"] == REASONING[stage]["effort"]
        assert engine.client.timeout == REASONING[stage]["timeout_seconds"]
        for method in (ChatOpenAI._generate_text, ChatOpenAI._generate_multimodal):
            assert inspect.signature(method).parameters["max_tokens"].default == REASONING[stage]["max_tokens"]
    monkeypatch.setenv("SHAPE_REASONING_EFFORT", "high")
    monkeypatch.setenv("SHAPE_LLM_TIMEOUT", "900")
    engine = ChatOpenAI(model_string="gpt-5.6-luna", api_key="offline-dummy", base_url="http://127.0.0.1:9/v1")
    params = {}
    engine._apply_backend_quirks(params)
    assert params["reasoning_effort"] == "high" and engine.client.timeout == 900


class Recorder:
    model_string = "gpt-5.6-luna"

    def __init__(self):
        self.kwargs = []

    def generate(self, content, **kwargs):
        self.kwargs.append(kwargs)
        return {"content": "OK"}


def test_the_cap_wrapper_passes_max_tokens_unless_the_caller_sets_one():
    raw = Recorder()
    engine = CappedEngine(raw, 16000)
    engine.generate(["question"], response_format=dict)
    engine.generate(["question"], max_tokens=500)
    assert raw.kwargs == [{"response_format": dict, "max_tokens": 16000}, {"max_tokens": 500}]
    assert engine.model_string == "gpt-5.6-luna"


@pytest.mark.parametrize("stage,expected", [("segment", DEFAULT), ("crop", DEFAULT), ("embed", HIGH_EMBED),
                                            ("batch", HIGH_BATCH)])
def test_engine_for_caps_the_stage_and_refuses_a_worker_without_its_env(monkeypatch, stage, expected):
    from benchmarks.full_HCMA.harness import shape_arm
    from shape.agent.verification import vision_engine
    raw = Recorder()
    monkeypatch.setattr(vision_engine, "build_vision_engine", lambda model: raw)
    monkeypatch.setattr(shape_arm, "build_engine", lambda model, engine=None: engine)
    config = {"arm": "shape", "model": "gpt-5.6-luna"}
    for name in REASONER_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="reasoning environment"):
        pipeline.engine_for(config, stage)
    monkeypatch.setenv("SHAPE_REASONING_EFFORT", "xhigh" if expected["effort"] != "xhigh" else "low")
    monkeypatch.setenv("SHAPE_LLM_TIMEOUT", str(expected["timeout_seconds"]))
    with pytest.raises(RuntimeError, match="reasoning environment"):        # another stage's effort leaked in
        pipeline.engine_for(config, stage)
    for name, value in stage_env("shape", stage).items():
        monkeypatch.setenv(name, value)
    engine = pipeline.engine_for(config, stage)
    assert isinstance(engine, CappedEngine) and engine.engine is raw and engine.max_tokens == expected["max_tokens"]
    engine.generate(["question"])
    assert raw.kwargs[-1]["max_tokens"] == expected["max_tokens"]
    assert pipeline.engine_for({**config, "arm": "codex"}, stage) is None


def test_child_passes_its_environment_to_the_worker(tmp_path):
    log = tmp_path / "child.log"
    cli.child([sys.executable, "-c", "import os; print(os.environ['SHAPE_REASONING_EFFORT'], "
               "os.environ['SHAPE_LLM_TIMEOUT'])"], log, 120, env={**os.environ, **stage_env("shape", "batch")})
    assert log.read_text(encoding="utf-8").split() == ["high", "900"]


@pytest.mark.parametrize("arm", ["shape", "codex", "control"])
def test_run_gives_each_stage_worker_its_own_env_and_records_the_table(tmp_path, monkeypatch, arm):
    rows = [{"image_id": f"i{k}", "usage": usage} for k, usage in enumerate(
        ("development_train", "development_val", "evaluation_only"))]
    prep = tmp_path / "prep"
    prep.mkdir()
    write_json(prep / "PREPARE.json", {})
    monkeypatch.setattr(cli, "load_prepared", lambda root, verify_pixels=False: rows)
    monkeypatch.setenv("SHAPE_REASONING_EFFORT", "ambient")
    monkeypatch.delenv("SHAPE_LLM_TIMEOUT", raising=False)
    seen = {}

    def child(command, log, timeout, *, new_session=True, env=None):
        if command[3] == "preflight":
            write_json(Path(command[command.index("--out") + 1]), {"commit": "a" * 40})
            return
        stage = command[4]
        seen[stage] = {k: (env or {}).get(k) for k in REASONER_ENV}
        write_json(tmp_path / "run" / f"{stage}_COMPLETE.json", {"status": "complete", "stage": stage})

    class Stop(Exception):
        pass

    monkeypatch.setattr(cli, "child", child)
    monkeypatch.setattr(cli, "code_identity", lambda *args: (_ for _ in ()).throw(Stop()))
    args = SimpleNamespace(out=tmp_path / "run", budget_hours=1.0, train_cells_per_site=4, inference_chunk=16,
                           cores=2, prep=prep, expected_commit="a" * 40, expected_branch="e2e",
                           assets_root=tmp_path, arm=arm, model="gpt-5.6-luna", smoke=False, seed=0)
    with pytest.raises(Stop):
        cli.run(args)
    if arm == "shape":
        assert seen == {"segment": {"SHAPE_REASONING_EFFORT": "medium", "SHAPE_LLM_TIMEOUT": "120"},
                        "crop": {"SHAPE_REASONING_EFFORT": "medium", "SHAPE_LLM_TIMEOUT": "120"},
                        "embed": {"SHAPE_REASONING_EFFORT": "high", "SHAPE_LLM_TIMEOUT": "900"},
                        "batch": {"SHAPE_REASONING_EFFORT": "high", "SHAPE_LLM_TIMEOUT": "900"},
                        "cluster": {"SHAPE_REASONING_EFFORT": "ambient", "SHAPE_LLM_TIMEOUT": None}}
    else:                                       # no SHAPE reasoner: the stage workers inherit the driver's env
        assert all(v == {"SHAPE_REASONING_EFFORT": "ambient", "SHAPE_LLM_TIMEOUT": None} for v in seen.values())
    assert os.environ["SHAPE_REASONING_EFFORT"] == "ambient" and "SHAPE_LLM_TIMEOUT" not in os.environ
    config = read_json(tmp_path / "run" / "CONFIG.json")
    assert config["reasoning"] == reasoning_record(arm)
    meta = read_json(tmp_path / "run" / "RUN_META.json")
    assert meta["config"]["reasoning"] == reasoning_record(arm) and meta["status"] == "failed"


def test_a_shape_stage_records_its_settings_in_usage_and_refuses_a_wrong_env(tmp_path, monkeypatch):
    from shape.engine import usage
    monkeypatch.setattr(cli, "asset_environment", lambda root: {})
    monkeypatch.setattr(cli, "load_prepared", lambda root: [])
    write_json(tmp_path / "CONFIG.json", {"arm": "shape", "model": "gpt-5.6-luna", "assets_root": str(tmp_path),
                                          "images": [], "reasoning": reasoning_record("shape")})
    monkeypatch.setitem(pipeline.STAGES, "batch", lambda prep, run, config, rows: usage.absorb(
        {"total_tokens": 9, "prompt_tokens": 5, "completion_tokens": 4, "calls": 1}))
    monkeypatch.setenv("SHAPE_REASONING_EFFORT", "medium")
    monkeypatch.setenv("SHAPE_LLM_TIMEOUT", "120")
    with pytest.raises(RuntimeError, match="reasoning environment"):
        cli.stage(SimpleNamespace(stage="batch", prep=tmp_path, run=tmp_path))
    for name, value in stage_env("shape", "batch").items():
        monkeypatch.setenv(name, value)
    usage.reset()
    try:
        cli.stage(SimpleNamespace(stage="batch", prep=tmp_path, run=tmp_path))
    finally:
        usage.reset()
    recorded = read_json(tmp_path / "batch_USAGE.json")
    assert recorded["reasoning"] == HIGH_BATCH and recorded["total_tokens"] == 9
    assert read_json(tmp_path / "batch_COMPLETE.json")["usage"] == recorded
    config = read_json(tmp_path / "CONFIG.json")
    config["reasoning"]["stages"]["batch"] = DEFAULT
    write_json(tmp_path / "CONFIG.json", config)
    with pytest.raises(RuntimeError, match="differs from pipeline.REASONING"):
        cli.stage(SimpleNamespace(stage="batch", prep=tmp_path, run=tmp_path))


def test_codex_config_carries_the_stage_effort_and_binds_every_round(tmp_path, gateway):
    assert "model_reasoning_effort" not in agent._config(agent.MODEL)
    high = agent._config(agent.MODEL, "high")
    assert high.splitlines()[:3] == ['model = "gpt-5.6-luna"', 'model_provider = "stanford"',
                                     'model_reasoning_effort = "high"']
    assert high.replace('model_reasoning_effort = "high"\n', "") == agent._config(agent.MODEL)
    with pytest.raises(ValueError, match="effort"):
        agent._config(agent.MODEL, "maximal")
    settings = agent.setup(tmp_path / "auth", agent.MODEL, effort="high")
    assert settings.read_text(encoding="utf-8") == high
    with pytest.raises(ValueError, match="configuration changed"):     # a round must name the effort it was set with
        agent.invoke(tmp_path / "root", "p", agent.MODEL, settings, time.time() + 30, 0)


@pytest.mark.parametrize("stage,effort", [("segment", "medium"), ("crop", "medium"), ("embed", "high"), ("batch", "high")])
def test_codex_stage_gets_its_effort_in_setup_and_every_round(tmp_path, monkeypatch, gateway, stage, effort):
    monkeypatch.delenv("SHAPE_SAM_SERVER_URL", raising=False)
    config = {"arm": "codex", "model": agent.MODEL, "deadline": time.time() + 600, "cores": 2, "images": ["img0"]}
    run = agent.StageRun(stage, tmp_path / "prep", tmp_path / "run", config)
    assert run.reasoning == {"model_reasoning_effort": effort}
    assert run.settings.read_text(encoding="utf-8") == agent._config(agent.MODEL, effort)
    codex = FakeCodex({"s": [d("finish", {})]})
    monkeypatch.setattr(agent, "invoke", codex)
    decision, _ = run.ask(agent.Session(tmp_path / "s", "dataset", ["dataset"]), "prompt", 0)
    assert decision["action"] == "finish" and codex.efforts == [effort]


def test_codex_stage_records_its_reasoning_in_selected(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=1)
    codex = FakeCodex({"group": [d("finish", {})]})
    monkeypatch.setattr(agent, "invoke", codex)
    agent.run_stage("segment", prep, run, config)
    assert read_json(run / "agent/segment/SELECTED.json")["reasoning"] == {"model_reasoning_effort": "medium"}
    assert set(codex.efforts) == {"medium"}


def records():
    config = dict(seed=0, cores=8, budget_hours=11, train_cells_per_site=128, inference_chunk=2048,
                  images=["i1", "i2"], model="gpt-5.6-luna")
    return {a: {"config": {**deepcopy(config), "reasoning": reasoning_record(a)}, "delivery": {"prepare_sha256": "d"},
                "provenance": {"assets": {"sha": "assets"}, "gpu": "same"}} for a in compare.ARMS}


@pytest.mark.parametrize("module", ["cpjump1", "livecell"])
def test_compare_refuses_arms_with_different_or_missing_reasoning_tables(module):
    from benchmarks.full_LIVECell.harness import compare as livecell
    C = compare if module == "cpjump1" else livecell
    assert C.compatible(records())
    report = compare.reasoning_protocol(records())
    assert report == {"protocol": reasoning_protocol(),
                      "stages": {arm: reasoning_record(arm)["stages"] for arm in compare.ARMS}}
    r = records()
    r["codex"]["config"]["reasoning"]["protocol"]["table"]["batch"]["codex_effort"] = "medium"
    with pytest.raises(ValueError, match="reasoning tables differ"):
        C.compatible(r)
    r = records()
    for arm in r:                    # the same changed table on every arm is comparable, each arm per its settings
        r[arm]["config"]["reasoning"]["protocol"]["table"]["embed"]["effort"] = "medium"
    with pytest.raises(ValueError, match="incomparable shape: its stages"):
        C.compatible(r)
    r["shape"]["config"]["reasoning"]["stages"]["embed"]["effort"] = "medium"
    assert C.compatible(r)
    r = records()
    del r["control"]["config"]["reasoning"]
    with pytest.raises(ValueError, match="no per-stage reasoning protocol"):
        C.compatible(r)
