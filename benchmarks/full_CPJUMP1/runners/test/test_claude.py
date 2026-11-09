import json
from pathlib import Path
import time

import pytest

from benchmarks.full_CPJUMP1.harness import claude_arm as agent
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
from shape.agent.arms.read_guard import allowed


def decision(action="finish", selected="t000", params=None, parent=""):
    return dict(action=action, selected=selected, params_json=json.dumps(params or {}), parent=parent, reason="observed quality")


def trace(path, **updates):
    result = dict(type="result", subtype="success", usage={"input_tokens": 3, "output_tokens": 5},
                  structured_output=decision(), modelUsage={"claude-test": {}})
    result.update(updates)
    path.write_text("\n".join(json.dumps(e) for e in [
        {"type": "system", "subtype": "init", "model": "claude-test"}, result]), encoding="utf-8")


def test_cli_is_real_claude_code_without_shell_or_shape_engine():
    cmd = agent.command("go", "claude-test", Path("settings.json"), "session")
    assert cmd[:3] == ["claude", "-p", "go"]
    assert cmd[cmd.index("--tools") + 1] == "Read"
    assert cmd[cmd.index("--allowedTools") + 1] == "Read"
    assert "--json-schema" in cmd and "--strict-mcp-config" in cmd
    assert "--dangerously-skip-permissions" not in cmd
    assert cmd[cmd.index("--resume") + 1] == "session"


def test_response_requires_exact_model_positive_usage_and_structured_request(tmp_path):
    p = tmp_path / "trace.jsonl"
    trace(p)
    choice, usage = agent.response(p, "claude-test")
    assert choice["selected"] == "t000" and usage["total_tokens"] == 8
    with pytest.raises(RuntimeError, match="model"):
        agent.response(p, "claude-other")


@pytest.mark.parametrize("updates", [
    {"is_error": True}, {"subtype": "error_max_turns"}, {"usage": {}},
    {"modelUsage": {"fallback-model": {}}}, {"structured_output": None},
    {"structured_output": {**decision(), "reason": ""}},
])
def test_failed_claude_response_never_becomes_a_default_decision(tmp_path, updates):
    p = tmp_path / "trace.jsonl"
    trace(p, **updates)
    with pytest.raises((ValueError, RuntimeError)):
        agent.response(p, "claude-test")


def test_read_hook_blocks_labels_secrets_sibling_escape_and_write(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    image = public / "image.png"
    image.write_bytes(b"image")
    secret = tmp_path / "private.json"
    secret.write_text("private")
    event = lambda path, tool="Read": {"tool_name": tool, "tool_input": {"file_path": str(path)}}
    assert allowed(event(image), [public])
    assert not allowed(event(secret), [public])
    assert not allowed(event(public / ".." / "private.json"), [public])
    assert not allowed(event(image, "Write"), [public])
    assert not allowed(event(image, "Bash"), [public])


@pytest.mark.parametrize("key", ["split_key", "cell_crops", "seed", "init_checkpoint", "embed_only", "query_cache_dir"])
def test_agent_cannot_override_ssl_data_or_runtime_contract(key):
    with pytest.raises(ValueError, match="unknown"):
        agent.parameters("embed", {key: "forbidden"})


def test_native_lever_types_and_ranges():
    assert agent.parameters("embed", {"max_epochs": 2, "adaptation_depth": "trained"})["max_epochs"] == 2
    with pytest.raises(Exception):
        agent.parameters("embed", {"max_epochs": -1})
    with pytest.raises(Exception):
        agent.parameters("embed", {"augment_color": "false"})


def test_trial_tampering_rejected(tmp_path):
    p = tmp_path / "checkpoint.pt"
    p.write_bytes(b"original")
    record = {"artifacts": {p.name: digest(p)}}
    agent.verify_trial(tmp_path, record)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        agent.verify_trial(tmp_path, record)


def test_native_worker_resume_is_same_run_completed_checkpoint(tmp_path, monkeypatch):
    from benchmarks.full_CPJUMP1.harness import cli
    run, prep = tmp_path / "run", tmp_path / "prep"
    run.mkdir()
    write_json(run / "cells.json", [])
    calls = []

    def worker(command, log, timeout, **kwargs):
        folder = Path(command[command.index("--run") + 1])
        cfg = read_json(folder / "CONFIG.json")
        calls.append(cfg)
        cp = folder / "best.pt"
        cp.write_bytes(b"trained")
        write_json(folder / "SSL_SELECTION.json", {"checkpoint": str(cp), "checkpoint_sha256": digest(cp)})
        write_json(folder / "FEEDBACK.json", {"admission": {"decision": "RUN"}})

    monkeypatch.setattr(cli, "child", worker)
    cfg = {"arm": "claude", "deadline": time.time() + 1000}
    trials = {}
    agent.run_trial("embed", prep, run, cfg, trials, "t000", {"adaptation_depth": "trained"})
    agent.run_trial("embed", prep, run, cfg, trials, "t001", {"adaptation_depth": "trained", "max_epochs": 2}, "t000")
    assert calls[-1]["fit_only"] is True
    assert calls[-1]["resume_checkpoint"].endswith("t000/best.pt") or calls[-1]["resume_checkpoint"].endswith("t000\\best.pt")
    with pytest.raises(ValueError, match="already completed"):
        agent.run_trial("embed", prep, run, cfg, trials, "t002", {}, "../../external")


def test_gate_skip_prevents_ssl_trial(tmp_path):
    trials = {"t000": {"feedback": {"admission": {"decision": "SKIP"}}}}
    with pytest.raises(ValueError, match="gate"):
        agent.run_trial("embed", tmp_path, tmp_path, {}, trials, "t001", {"adaptation_depth": "trained"})


def test_autonomous_loop_selects_an_existing_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "setup", lambda *args: tmp_path / "settings.json")
    seen = []

    def worker(stage, prep, run, config, trials, name, params, parent=""):
        folder = run / "agent" / stage / "trials" / name
        folder.mkdir(parents=True)
        write_json(folder / "segmentation.json", [{"n_cells": 10, "chosen": name}])
        trials[name] = {"artifacts": {"segmentation.json": digest(folder / "segmentation.json")}, "params": params}
        seen.append(params)

    decisions = iter([decision("trial", "", {"diameter": 40}), decision("finish", "t001")])
    monkeypatch.setattr(agent, "run_trial", worker)
    monkeypatch.setattr(agent, "invoke", lambda *a, **k: (next(decisions), {"total_tokens": 10}, "session"))
    agent.run_stage("segment", tmp_path, tmp_path, {"model": "claude-test", "deadline": time.time() + 1000})
    assert seen == [{}, {"diameter": 40}]
    assert read_json(tmp_path / "segmentation.json")[0]["chosen"] == "t001"
    assert read_json(tmp_path / "agent/segment/SELECTED.json")["total_tokens"] == 20


def test_crop_preview_contains_five_columns_without_changing_scientific_pixels(tmp_path):
    import numpy as np
    import tifffile
    from PIL import Image
    image = tmp_path / "crop.ome.tif"
    raw = np.stack([np.arange(256, dtype=np.uint16).reshape(16, 16) + k * 1000 for k in range(5)])
    tifffile.imwrite(image, raw, photometric="minisblack")
    before = digest(image)
    write_json(tmp_path / "cells.json", [{"cell_id": "c1", "image_id": "i1", "crop_path": str(image)}])
    previews = agent.visual_evidence("crop", tmp_path)
    assert len(previews) == 1
    with Image.open(previews[0]) as preview:
        assert preview.size == (800, 208)
    assert digest(image) == before


def test_worker_cleanup_reaps_process_group_after_leader_exit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmarks.full_CPJUMP1.harness import cli
    killed, flags = [], []

    def popen(*args, **kwargs):
        flags.append(kwargs["start_new_session"])
        return SimpleNamespace(pid=123, wait=lambda **kw: 0, poll=lambda: 0)

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli, "os", SimpleNamespace(name="posix", killpg=lambda pid, sig: killed.append(pid)))
    monkeypatch.setattr(cli, "signal", SimpleNamespace(SIGKILL=9))
    cli.child(["worker"], tmp_path / "outer.log", 10)
    cli.child(["worker"], tmp_path / "inner.log", 10, new_session=False)
    assert flags == [True, False]
    assert killed == [123]  # inner workers remain in the outer watchdog's process group
