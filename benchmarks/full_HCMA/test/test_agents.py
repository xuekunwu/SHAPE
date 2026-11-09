"""The headless Claude loop (deadline watchdog, rounds, FINAL check), the live trace and the
export, with a fake agent command and a stub final_check; no agent CLI, no GPU."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
import types
from pathlib import Path

import pytest

from benchmarks.full_HCMA.harness import agents, export, trace_md
from benchmarks.full_HCMA.harness.cli import main

KEY = "sk-" + "abcdefghijklmnopqrstuvwxyz0123456789"
CLAUDE_EVENTS = [
    {"type": "system", "subtype": "init", "session_id": "S1", "model": "m", "cwd": "/w", "tools": ["Bash"]},
    {"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "Segment everything at the defaults first."},
        {"type": "text", "text": "Running segment."},
        {"type": "tool_use", "id": "t1", "name": "Bash",
         "input": {"command": "python tools/segment.py --name first --set diameter=45", "description": "segment"}}]}},
    {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1",
                                              "content": f"S001 started\nenv: {KEY}\n" + "progress\n" * 2000}]}},
    {"type": "result", "subtype": "success", "num_turns": 2, "duration_ms": 10, "total_cost_usd": 0.1,
     "usage": {"input_tokens": 5}},
]


def _jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _fake_run(run: Path, t0: float = 1000.0) -> None:
    _jsonl(run / "traces" / "round_00.jsonl", CLAUDE_EVENTS)
    (run / "traces" / "round_00.times").write_text("".join(f"{t0 + 10 + i}\n" for i in range(len(CLAUDE_EVENTS))))
    _jsonl(run / "logs" / "rounds.jsonl", [
        {"time": t0, "event": "round_start", "round": 0, "prompt": "# Task\nDo it."},
        {"time": t0 + 50, "event": "round_end", "round": 0, "returncode": 0, "timed_out": False, "seconds": 50},
        {"time": t0 + 60, "event": "final_rejected", "round": 1, "missing": ["cluster.resolution"],
         "chain_problems": ["K001 not done"]},
        {"time": t0 + 70, "event": "agent_done", "stopped_because": "final_flag"}])
    _jsonl(run / "logs" / "tool_calls.jsonl", [
        {"event": "start", "time": t0 + 13, "call": "S001", "tool": "segment", "name": "first",
         "call_dir": "calls/S001_first", "set": {"diameter": 45}, "set_levers": ["diameter"],
         "inputs": {"images": "all"}},
        {"event": "end", "time": t0 + 40, "call": "S001", "tool": "segment", "status": "done", "seconds": 12.5,
         "summary": {"n_images": 3, "n_cells": 120}, "error": None},
        {"event": "start", "time": t0 + 45, "call": "C001", "tool": "crop", "name": "c", "set": {}, "set_levers": []}])
    meta = run / "workspace" / "calls" / "S001_first" / "meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"note": "try larger cells"}))
    (run / "workspace" / "submission").mkdir(parents=True)
    (run / "workspace" / "submission" / "choice.json").write_text(json.dumps({"cluster_call": "K001", "resolution": 1.0}))
    (run / "AGENT_RUN.json").write_text(json.dumps({"agent": "claude", "model": "m", "started": t0, "n_rounds": 1,
                                                    "stopped_because": "final_flag"}))
    (run / "RESULT.json").write_text(json.dumps({"mode": "run", "status": "OK", "primary": {"value": 0.5}}))
    (run / "RESULT.md").write_text("# full_HCMA result (run): OK\n\nPrimary 0.5\n")


def test_claude_cmd_and_env(tmp_path, monkeypatch):
    cmd = agents.claude_cmd("go", "claude-opus-5", "S9", ["/extra"], "/s/settings.json")
    assert cmd[:3] == ["claude", "-p", "go"]
    flags = {cmd[i]: cmd[i + 1] for i in range(3, len(cmd) - 1)}
    assert flags["--allowedTools"] == "Bash Read Write Edit Glob Grep TodoWrite"
    assert flags["--disallowedTools"] == "WebFetch WebSearch"
    assert flags["--settings"] == "/s/settings.json" and flags["--resume"] == "S9" and flags["--model"] == "claude-opus-5"
    assert flags["--add-dir"] == "/extra" and "--dangerously-skip-permissions" not in cmd
    assert "--resume" not in agents.claude_cmd("go", None, None, [])
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://x")
    monkeypatch.setenv("PYTHONPATH", "/home/users/x/py/SHAPE-agentic-vrr")
    env = agents._env(tmp_path)
    assert "LLM_API_KEY" not in env and "LLM_BASE_URL" not in env and "PYTHONPATH" not in env
    assert env[agents.MARKER_ENV] == str(tmp_path.resolve())
    assert env["CLAUDE_CONFIG_DIR"] == str((tmp_path / "claude_config").resolve())


def test_continuation_prompt_and_open_calls(tmp_path):
    text = agents.continuation_prompt(time.time() + 2 * 3600, 3)
    assert text.startswith("Round 3") and "2.00 h" in text and "tools/time_left.py" in text
    assert "TASK.md" in text and "NOTES.md" in text and "submission/choice.json" in text and "submission/FINAL" in text
    assert "not accepted" not in text
    text = agents.continuation_prompt(time.time() + 3600, 4, {"missing": ["segment.diameter", "crop.cell_overrides"],
                                                              "chain_problems": ["K002 did not end done"]})
    assert "not accepted" in text and "segment.diameter, crop.cell_overrides" in text
    assert "K002 did not end done" in text and "tools/lever_coverage.py" in text
    _fake_run(tmp_path)
    assert agents.open_tool_calls(tmp_path) == ["C001"]


def test_trace_renders_steps_rounds_and_tool_calls(tmp_path):
    _fake_run(tmp_path)
    steps = trace_md.load_steps(tmp_path)
    kinds = [s["kind"] for s in steps]
    assert kinds.count("tool_call") == 1 and kinds.count("tool") == 3 and "thinking" in kinds and "usage" in kinds
    trace_md.write_all(tmp_path)
    text = (tmp_path / "trace.md").read_text(encoding="utf-8")
    assert "## Round 0" in text and "python tools/segment.py --name first" in text and "chars omitted" in text
    assert "tool call S001 segment `first` started" in text and "levers set: ['diameter']" in text
    assert "note: 'try larger cells'" in text and "status=done 12.5 s" in text and '"n_cells": 120' in text
    assert "| segment | 1 | 1 | 0 | 0 | 0.2 |" in text and "| crop | 1 | 0 | 0 | 0 | 0.0 |" in text
    assert "FINAL rejected before round 1" in text and "K001 not done" in text
    assert "| choice | cluster_call=K001 resolution=1.0 |" in text and "| RESULT status | OK |" in text
    assert "# full_HCMA result (run): OK" in text
    assert KEY not in text and "[REDACTED]" in text
    payload = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert payload["agent_run"]["agent"] == "claude" and KEY not in json.dumps(payload)


def test_u2028_does_not_split_a_trace_line(tmp_path):
    event = {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "a b"}]}}
    path = tmp_path / "traces" / "round_00.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(event, ensure_ascii=False) + "\n" + json.dumps(CLAUDE_EVENTS[0]) + "\n",
                    encoding="utf-8")
    steps = trace_md.parse_round(path, 0)
    assert [s["kind"] for s in steps] == ["tool_result", "system"] and steps[1]["line"] == 2


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
def test_stream_enforces_deadline(tmp_path):
    code = "import json,time\nprint(json.dumps({'type':'system','session_id':'X'}),flush=True)\ntime.sleep(60)"
    started = time.time()
    res = agents.stream([sys.executable, "-c", code], tmp_path, tmp_path / "t.jsonl", time.time() + 3)
    assert res["timed_out"] and res["session_id"] == "X" and time.time() - started < 45
    assert (tmp_path / "t.times").read_text().strip()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc (Linux)")
def test_kill_run_processes_finds_marked_leftovers(tmp_path):
    (tmp_path / "workspace").mkdir()
    env = {**os.environ, agents.MARKER_ENV: str(tmp_path.resolve())}
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], env=env, start_new_session=True)
    time.sleep(0.5)
    assert proc.pid in agents.kill_run_processes(tmp_path, grace=5)
    assert proc.wait(timeout=10) != 0


def test_run_loop_rounds_and_final_flag(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "workspace" / "submission").mkdir(parents=True)
    (run / "workspace" / "TASK.md").write_text("# Task\nDo it.\n")

    def final_check(path):
        ok = (Path(path) / "workspace" / "submission" / "choice.json").exists()
        return {"accepted": ok, "tested": {}, "skipped": {},
                "missing": [] if ok else ["segment.diameter", "crop.cell_overrides"],
                "chain_problems": [] if ok else ["no readable submission/choice.json"]}

    fake = types.ModuleType("benchmarks.full_HCMA.harness.coverage")
    fake.final_check = final_check
    monkeypatch.setitem(sys.modules, "benchmarks.full_HCMA.harness.coverage", fake)
    calls = []

    def fake_cmd(prompt, model, session_id, add_dirs, settings=None):
        calls.append((prompt, session_id))
        # round 2: the "agent" declares it is done without a choice (rejected); round 3: with one
        body = ("import json, time; print(json.dumps({'type':'system','subtype':'init','session_id':'S'}), flush=True)"
                "\ntime.sleep(1.5)"   # long enough for the live renderer to fire mid-round
                + ("\nopen('submission/choice.json','w').write('{}')" if len(calls) == 3 else "")
                + ("\nopen('submission/FINAL','w').close()" if len(calls) in (2, 3) else ""))
        return [sys.executable, "-c", body]

    monkeypatch.setattr(agents, "claude_cmd", fake_cmd)
    summary = agents.run_loop(run, budget_hours=1, model=None, add_dirs=[], min_round_gap=0, live_interval=0.5)
    assert summary["n_rounds"] == 3 and summary["stopped_because"] == "final_flag"
    assert calls[0][0].startswith("# Task") and calls[1][0].startswith("Round 1") and calls[1][1] == "S"
    assert "not accepted" in calls[2][0] and "segment.diameter, crop.cell_overrides" in calls[2][0]
    assert "no readable submission/choice.json" in calls[2][0]
    events = [json.loads(line) for line in (run / "logs" / "rounds.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds.count("round_start") == 3 and kinds[-1] == "agent_done"
    rejected = [e for e in events if e["event"] == "final_rejected"]
    assert len(rejected) == 1 and rejected[0]["missing"] == ["segment.diameter", "crop.cell_overrides"]
    for path in (run / "workspace" / ".deadline", run / "logs" / ".deadline"):
        assert float(path.read_text()) == pytest.approx(time.time() + 3600, abs=60)
    live = json.loads((run / "trace.json").read_text())
    assert live["agent_run"]["stopped_because"] == "final_flag"
    assert sum(1 for x in live["steps"] if x["kind"] == "system") == 3
    assert (run / "trace.md").read_text(encoding="utf-8").count("## Round") == 3
    assert json.loads((run / "AGENT_RUN.json").read_text())["final_flag"] is True


def test_a_failing_final_check_rejects_final_instead_of_ending_the_run(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "workspace" / "submission").mkdir(parents=True)
    (run / "workspace" / "TASK.md").write_text("# Task\n")
    checks = []

    def final_check(path):
        checks.append(path)
        if len(checks) == 1:
            raise TypeError("cluster_call must be a string")
        return {"accepted": True, "missing": [], "chain_problems": []}

    fake = types.ModuleType("benchmarks.full_HCMA.harness.coverage")
    fake.final_check = final_check
    monkeypatch.setitem(sys.modules, "benchmarks.full_HCMA.harness.coverage", fake)
    prompts = []

    def fake_cmd(prompt, model, session_id, add_dirs, settings=None):
        prompts.append(prompt)
        return [sys.executable, "-c", "open('submission/FINAL','w').close()"]

    monkeypatch.setattr(agents, "claude_cmd", fake_cmd)
    summary = agents.run_loop(run, budget_hours=1, model=None, add_dirs=[], min_round_gap=0, live_interval=30)
    assert summary["stopped_because"] == "final_flag" and summary["n_rounds"] == 2
    assert "the FINAL check failed: TypeError: cluster_call must be a string" in prompts[1]


def test_export_leaves_bulk_outputs_behind(tmp_path, monkeypatch):
    run = tmp_path / "claude-1-x"
    _fake_run(run)
    trace_md.write_all(run)
    files = {
        "workspace/calls/S001_first/masks/img_0001.tif": b"mask", "workspace/calls/S001_first/overlays/img_0001.png": b"o",
        "workspace/calls/S001_first/per_image.csv": b"image_id\n", "workspace/calls/C001_c/crops/img_0001/a.png": b"c",
        "workspace/calls/C001_c/figures/img_0001.png": b"f", "workspace/calls/C001_c/crops.csv": b"cell_id\n",
        "workspace/calls/E001_e/embedding.npy": b"npy", "workspace/calls/E001_e/cell_state_analyzed.h5ad": b"h5",
        "workspace/calls/E001_e/cells.csv": b"cell_id\n", "workspace/calls/K001_k/umap.png": b"u",
        "workspace/analysis/figure.png": b"agent figure", "workspace/NOTES.md": f"key {KEY}".encode(),
        "workspace/big.csv": b"x" * 2000, "venv/lib/site.py": b"x", "cache/tmp/t.txt": b"x",
        "logs/big.log": b"y" * 2000, "per_image.csv": b"image_id\n", "TOOLS_SHA256.json": b"{}"}
    for rel, data in files.items():
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_bytes(data)
    images = tmp_path / "prep_images"
    images.mkdir()
    (images / "img_0001.jpg").write_bytes(b"jpg")
    (run / "workspace" / "data").mkdir()
    (run / "workspace" / "data" / "images").symlink_to(images, target_is_directory=True)
    monkeypatch.setattr(export, "EXPORT_MAX_BYTES", 1000)
    out = export.export(run)
    with tarfile.open(out) as tar:
        names = {m.name.split("/", 1)[1] for m in tar.getmembers()}
        notes = tar.extractfile(f"{run.name}/workspace/NOTES.md").read().decode()
        manifest = json.loads(tar.extractfile(f"{run.name}/EXPORT_MANIFEST.json").read())
    for rel in ("trace.md", "trace.json", "RESULT.json", "RESULT.md", "AGENT_RUN.json", "TOOLS_SHA256.json",
                "per_image.csv", "logs/tool_calls.jsonl", "logs/big.log", "traces/round_00.jsonl",
                "workspace/calls/S001_first/per_image.csv", "workspace/calls/S001_first/meta.json",
                "workspace/calls/C001_c/crops.csv", "workspace/calls/E001_e/cells.csv",
                "workspace/NOTES.md"):
        assert rel in names, rel
    for rel in names:
        assert not any(part in rel.split("/") for part in ("masks", "overlays", "crops", "figures", "venv", "cache"))
        assert not rel.endswith((".npy", ".h5ad", ".tif", ".jpg", ".png")), rel
    assert "workspace/big.csv" not in names and KEY not in notes and "[REDACTED]" in notes
    assert {"path": "workspace/big.csv", "bytes": 2000} in manifest["skipped_not_exported"]
    assert manifest["redactions"]["workspace/NOTES.md"] == 1


def test_the_exact_gateway_key_is_masked_in_trace_and_export(tmp_path):
    key = "plainGatewayValue0042"                 # no credential shape: only the exact value catches it
    run = tmp_path / "claude-3-x"
    _fake_run(run)
    helper = tmp_path / "helper.py"
    helper.write_text(f"print({key!r})\n")
    (run / "claude_config").mkdir()
    (run / "claude_config" / "settings.json").write_text(json.dumps({"apiKeyHelper": f"{sys.executable} {helper}"}))
    leak = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1",
                                                     "content": f"bash key_helper.sh -> {key}"}]}}
    with open(run / "traces" / "round_00.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(leak) + "\n")
    (run / "workspace" / "NOTES.md").write_text(f"found {key}\n")
    trace_md.write_all(run)
    for name in ("trace.md", "trace.json"):
        text = (run / name).read_text(encoding="utf-8")
        assert key not in text and "[REDACTED]" in text
    out = export.export(run)
    with tarfile.open(out) as tar:
        for member in tar.getmembers():
            assert key.encode() not in tar.extractfile(member).read(), member.name
        names = [m.name for m in tar.getmembers()]
    assert f"{run.name}/traces/round_00.jsonl" in names and not any("/claude_config/" in n for n in names)
    assert trace_md.run_secrets(tmp_path / "no_such_run") == ()


def test_cli_trace_and_export(tmp_path, capsys):
    run = tmp_path / "claude-2-x"
    _fake_run(run)
    main(["trace", "--run", str(run)])
    main(["export", "--run", str(run)])
    assert (run / "trace.md").exists() and (run / f"EXPORT_{run.name}.tar.gz").exists()
    assert "EXPORT" in capsys.readouterr().out
