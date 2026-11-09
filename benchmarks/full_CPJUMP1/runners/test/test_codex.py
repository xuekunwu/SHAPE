"""CPU-only CLI contract tests, including real offline subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from benchmarks.full_CPJUMP1.harness import codex_arm as agent
from shape.agent.arms import codex as codex_backend, protocol
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json


def decision(action="finish", request=None, reason="observed quality"):
    return dict(action=action, request_json=json.dumps({"all": "c0"} if request is None else request), reason=reason)


def events(choice=None):
    return [{"type": "thread.started", "thread_id": "offline-thread"}, {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "i1", "type": "agent_message",
                                                "text": json.dumps(choice or decision())}},
            {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 3,
                                                  "output_tokens": 5, "reasoning_output_tokens": 2}}]


def trace(tmp_path, stream=None, choice=None):
    path, last = tmp_path / "trace.jsonl", tmp_path / "last.json"
    path.write_text("\n".join(json.dumps(e) for e in (events(choice) if stream is None else stream)), encoding="utf-8")
    write_json(last, choice or decision())
    return path, last


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "offline-dummy-key-not-a-credential")
    monkeypatch.setenv("LLM_API_BASE", codex_backend.BASE_URL)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)


STUB = '''import json, os, pathlib, sys, time
mode = MODE
args = sys.argv[1:]
home = pathlib.Path(os.environ["CODEX_HOME"])
cfg = (home / "config.toml").read_text(encoding="utf-8")
assert 'env_key = "LLM_API_KEY"' in cfg
assert 'requires_openai_auth = false' in cfg
assert 'wire_api = "responses"' in cfg
assert 'model = "gpt-5.6-luna"' in cfg
assert 'base_url = "https://llm-gateway.example.edu/v1"' in cfg
assert 'view_image = false' in cfg
assert '[tools]' not in cfg
assert 'shell_tool = false' in cfg
assert 'unified_exec = false' in cfg
assert 'project_root_markers = [".cpjump1-root"]' in cfg
assert (pathlib.Path.cwd() / ".cpjump1-root").is_file()
assert not (home / "auth.json").exists()
assert pathlib.Path(os.environ["FULL_HCMA_RUN"]).is_absolute()
assert all(k not in os.environ for k in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY",
    "SHAPE_API_ENV_FILE", "PYTHONPATH", "ANTHROPIC_API_KEY", "CODEX_MANAGED_CONFIG_PATH"])
if args == ["exec", "--help"]:
    print("--json --strict-config --output-schema --output-last-message --ephemeral --sandbox --skip-git-repo-check --image --model")
    sys.exit(0)
if args == ["--version"]:
    print(mode[len("version:"):] if mode.startswith("version:") else "codex-cli 0.162.0")
    sys.exit(0)
if args == ["features", "list"]:
    assert "LLM_API_KEY" not in os.environ
    if mode != "missing_shell":
        print("shell_tool stable " + ("true" if mode in ["unsafe_tools", "unsafe_forced_unified_exec"] else "false"))
    print("unified_exec stable " + ("true" if mode in ["forced_unified_exec", "unsafe_forced_unified_exec"] else "false"))
    print("view_image stable " + ("true" if mode == "unsafe_view_image" else "false"))
    sys.exit(0)
assert args[0] == "exec" and args[-1] == "-"
assert args[args.index("--sandbox") + 1] == "read-only"
assert args[args.index("--model") + 1] == "gpt-5.6-luna"
assert "--ephemeral" in args and "--json" in args
assert "--strict-config" in args
schema = json.loads(pathlib.Path(args[args.index("--output-schema") + 1]).read_text())
assert schema["additionalProperties"] is False
prompt = sys.stdin.read()
assert "PRIVATE_SENTINEL" not in prompt
assert os.environ["LLM_API_KEY"] not in prompt
if "--image" in args:
    image = pathlib.Path(args[args.index("--image") + 1])
    assert image.parent == pathlib.Path.cwd()
    if "Connectivity check" in prompt:
        assert image.read_bytes().startswith(bytes([137, 80, 78, 71, 13, 10, 26, 10]))
    else:
        assert image.read_bytes() == b"published-png"
if "Connectivity check" in prompt:
    assert "--image" in args and "green" not in prompt
if mode == "image_failure":
    print(json.dumps({"type": "error", "message": "image inputs unsupported"}))
    sys.exit(1)
if mode == "timeout":
    time.sleep(30)
if mode == "http_failure":
    print(json.dumps({"type": "error", "message": "HTTP 404 /responses"}))
    print(os.environ["LLM_API_KEY"], file=sys.stderr)
    sys.exit(1)
choice = {"action": "finish", "request_json": "{}" if "Connectivity check" in prompt else json.dumps({"all": "c0"}),
          "reason": "green" if "Connectivity check" in prompt else "observed quality"}
if mode == "image_ignored":
    choice["reason"] = "unknown"
stale = dict(choice, reason="different") if mode == "stale_choice" else choice
last = pathlib.Path(args[args.index("--output-last-message") + 1])
last.write_text(json.dumps(stale), encoding="utf-8")
stream = [{"type": "thread.started", "thread_id": "offline-thread"}, {"type": "turn.started"},
          {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": json.dumps(choice)}},
          {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 3, "output_tokens": 5}}]
if mode == "failed_event":
    stream.insert(2, {"type": "turn.failed", "error": {"message": "HTTP 400"}})
if mode == "disabled_host_warning":
    stream.insert(1, {"type": "item.completed", "item": {"id": "startup-warning", "type": "error",
        "message": "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`."}})
for event in stream:
    print(json.dumps(event))
'''


@pytest.fixture
def offline_cli(tmp_path, monkeypatch, gateway):
    original = protocol._process

    def install(mode="success"):
        script = tmp_path / "offline_codex.py"
        script.write_text(STUB.replace("MODE", repr(mode)), encoding="utf-8")

        calls = []

        def execute(cmd, *args, **kwargs):
            assert cmd[0] == "codex"
            calls.append(cmd)
            return original([sys.executable, str(script), *cmd[1:]], *args, **kwargs)

        monkeypatch.setattr(protocol, "_process", execute)
        return calls
    install()
    return install


def test_command_is_native_cli_with_explicit_images_and_stdin():
    cmd = codex_backend.command(codex_backend.MODEL, Path("schema.json"), Path("last.json"), [Path("preview.png")])
    assert cmd[:3] == ["codex", "exec", "--json"] and cmd[-1] == "-"
    assert cmd[cmd.index("--model") + 1] == codex_backend.MODEL
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in cmd and "--output-schema" in cmd and "--output-last-message" in cmd
    assert "--strict-config" in cmd
    assert "--image" in cmd and "--dangerously-bypass-approvals-and-sandbox" not in cmd


def test_usage_counts_no_cache_double_count_and_no_model_attestation(tmp_path):
    path, last = trace(tmp_path)
    choice, usage = codex_backend.response(path, codex_backend.MODEL, last)
    assert choice == decision() and usage["total_tokens"] == 15
    assert usage["requested_model"] == codex_backend.MODEL
    assert usage["response_model"] is None and usage["response_model_attested"] is False


def startup_warning(message=codex_backend.DISABLED_HOST_DIAGNOSTIC):
    return {"type": "item.completed", "item": {"id": "warning-1", "type": "error", "message": message}}


def test_live_0162_disabled_host_warning_and_cache_write_usage(tmp_path):
    stream = events()
    stream.insert(1, startup_warning())
    stream[-1]["usage"] = {"input_tokens": 6709, "cached_input_tokens": 0,
                           "cache_write_input_tokens": 6706, "output_tokens": 51,
                           "reasoning_output_tokens": 21}
    path, last = trace(tmp_path, stream)
    _, usage = codex_backend.response(path, codex_backend.MODEL, last)
    assert usage["total_tokens"] == 6760
    assert usage["startup_diagnostics"] == [codex_backend.DISABLED_HOST_DIAGNOSTIC]


@pytest.mark.parametrize("variant", ["changed", "configuration_warning", "during_turn", "duplicate",
                                    "top_level_error", "failed_status"])
def test_startup_exception_never_swallows_other_errors(tmp_path, variant):
    stream = events()
    warning = startup_warning()
    index = 1
    if variant == "changed":
        warning["item"]["message"] += " fallback enabled"
    elif variant == "configuration_warning":
        warning["item"]["message"] = "Codex is ignoring 1 unrecognized configuration setting"
    elif variant == "during_turn":
        index = 2
    elif variant == "duplicate":
        stream.insert(1, startup_warning())
    elif variant == "top_level_error":
        warning = {"type": "error", "message": codex_backend.DISABLED_HOST_DIAGNOSTIC}
    elif variant == "failed_status":
        warning["item"]["status"] = "failed"
    stream.insert(index, warning)
    path, last = trace(tmp_path, stream)
    with pytest.raises(RuntimeError):
        codex_backend.response(path, codex_backend.MODEL, last)


@pytest.mark.parametrize("change", [
    lambda e: e.insert(2, {"type": "turn.failed", "error": {"message": "failed"}}),
    lambda e: e.insert(2, {"type": "error", "message": "retry later"}),
    lambda e: e[0].update(model="fallback-model"),
    lambda e: e[2]["item"].update(model="fallback-model"),
    lambda e: e[2]["item"].update(type="command_execution", status="completed"),
    lambda e: e[2]["item"].update(type="mcp_tool_call", status="failed"),
    lambda e: e[2]["item"].update(type="file_change"),
    lambda e: e[-1].update(usage={}),
    lambda e: e[-1].update(usage={"input_tokens": True, "output_tokens": 1}),
    lambda e: e[-1].update(usage={"input_tokens": -2, "output_tokens": 10}),
    lambda e: e[-1].update(usage={"input_tokens": 2, "cached_input_tokens": 3}),
    lambda e: e.pop(),
    lambda e: e.append(e[-1]),
    lambda e: e.append({"type": "unknown_event"}),
])
def test_bad_stream_never_becomes_default_selection(tmp_path, change):
    stream = events()
    change(stream)
    path, last = trace(tmp_path, stream)
    with pytest.raises((ValueError, RuntimeError)):
        codex_backend.response(path, codex_backend.MODEL, last)


@pytest.mark.parametrize("choice", [
    {**decision(), "action": "execute"}, {**decision(), "reason": " "},
    {**decision(), "request_json": "not json"}, {**decision(), "request_json": 1},
    {**decision(), "extra": "ignored?"}, {"action": "finish", "reason": "no payload"},
])
def test_malformed_structured_decision_rejected(tmp_path, choice):
    path, last = trace(tmp_path, choice=choice)
    with pytest.raises(ValueError):
        codex_backend.response(path, codex_backend.MODEL, last)


def test_stale_last_message_rejected(tmp_path):
    path, last = trace(tmp_path)
    write_json(last, decision(request={"all": "c1"}))
    with pytest.raises(ValueError, match="match"):
        codex_backend.response(path, codex_backend.MODEL, last)


def test_configuration_and_environment_are_clean_without_mutating_parent(tmp_path, monkeypatch, gateway):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY", "PYTHONPATH", "SHAPE_API_ENV_FILE"):
        monkeypatch.setenv(key, "PRIVATE_SENTINEL")
    before = dict(os.environ)
    settings = codex_backend.setup(tmp_path / "auth", codex_backend.MODEL)
    cfg = settings.read_text()
    env = codex_backend._env(tmp_path / "isolated")
    assert dict(os.environ) == before
    assert os.environ["LLM_API_KEY"] not in cfg
    assert "PRIVATE_SENTINEL" not in cfg and "PRIVATE_SENTINEL" not in env.values()
    assert env["LLM_API_KEY"] == before["LLM_API_KEY"]
    assert env["CODEX_HOME"] != before.get("CODEX_HOME")
    with pytest.raises(ValueError, match="empty"):
        codex_backend.setup(settings.parent, codex_backend.MODEL)


@pytest.mark.parametrize("variable,value", [("LLM_API_KEY", ""), ("LLM_API_BASE", "https://api.openai.com/v1"),
                                           ("LLM_BASE_URL", "https://elsewhere.invalid/v1")])
def test_missing_credentials_or_wrong_route_fail_before_process(tmp_path, monkeypatch, gateway, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises((ValueError, RuntimeError)):
        codex_backend.preflight(tmp_path / "preflight", codex_backend.MODEL)


def test_wrong_model_fails_before_process(tmp_path, gateway):
    with pytest.raises(ValueError, match="gpt-5.6-luna"):
        codex_backend.preflight(tmp_path, "gpt-5.6")


def test_real_offline_preflight_subprocess(tmp_path, offline_cli):
    result = codex_backend.preflight(tmp_path / "preflight", codex_backend.MODEL)
    assert result["preflight_tokens"] == 15 and result["wire_api"] == "responses"
    assert result["codex_version"] == "codex-cli 0.162.0"
    assert result["image_preflight"] == "passed"
    assert result["evidence_policy"] == codex_backend.EVIDENCE_POLICY
    assert result["response_model_attested"] is False


def test_codex_0162_forced_unified_backend_with_shell_disabled(tmp_path, offline_cli):
    # Mirrors 0.162.0's managed_features normalization and shell registration gate.
    offline_cli("forced_unified_exec")
    result = codex_backend.preflight(tmp_path / "preflight", codex_backend.MODEL)
    assert result["preflight_tokens"] == 15
    assert result["codex_version"] == "codex-cli 0.162.0"


def test_real_offline_disabled_host_warning_is_recorded_in_preflight(tmp_path, offline_cli):
    offline_cli("disabled_host_warning")
    result = codex_backend.preflight(tmp_path / "preflight", codex_backend.MODEL)
    assert result["startup_diagnostics"] == [codex_backend.DISABLED_HOST_DIAGNOSTIC]
    assert result["image_preflight"] == "passed"


@pytest.mark.parametrize("version", ["codex-cli 0.161.0", "codex-cli 0.163.0",
                                     "codex-cli 0.162.0-alpha.1", ""])
@pytest.mark.parametrize("entrypoint", ["preflight", "invoke"])
def test_unaudited_version_rejected_before_features_or_inference(tmp_path, offline_cli, version, entrypoint):
    calls = offline_cli("version:" + version)
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="audited CLI version"):
        if entrypoint == "preflight":
            codex_backend.preflight(root, codex_backend.MODEL)
        else:
            settings = codex_backend.setup(root / "auth", codex_backend.MODEL)
            codex_backend.invoke(root, "feedback", codex_backend.MODEL, settings, time.time() + 30, 0)
    assert calls == [["codex", "--version"]]


@pytest.mark.parametrize("mode", ["http_failure", "failed_event", "unsafe_tools",
                                 "unsafe_forced_unified_exec", "missing_shell", "unsafe_view_image",
                                 "image_failure", "image_ignored"])
def test_real_offline_preflight_rejects_provider_or_tool_failures(tmp_path, offline_cli, mode):
    offline_cli(mode)
    with pytest.raises(RuntimeError):
        codex_backend.preflight(tmp_path / "preflight", codex_backend.MODEL)
    for path in (tmp_path / "preflight").rglob("*"):
        if path.is_file():
            assert os.environ["LLM_API_KEY"].encode() not in path.read_bytes()


def test_real_subprocess_explicit_images_and_timeout(tmp_path, offline_cli):
    root = tmp_path / "run"
    settings = codex_backend.setup(root / "auth", codex_backend.MODEL)
    preview = tmp_path / "preview.png"
    preview.write_bytes(b"published-png")
    choice, usage, session = codex_backend.invoke(root, "published feedback", codex_backend.MODEL, settings,
                                        time.time() + 20, 0, images=[preview])
    assert json.loads(choice["request_json"]) == {"all": "c0"} and usage["total_tokens"] == 15 and session is None
    offline_cli("timeout")
    with pytest.raises(RuntimeError, match="deadline"):
        codex_backend.invoke(root, "feedback", codex_backend.MODEL, settings, time.time() + 1, 1)


def test_config_tampering_and_expired_deadline_fail_closed(tmp_path, gateway, offline_cli):
    root = tmp_path / "run"
    settings = codex_backend.setup(root / "auth", codex_backend.MODEL)
    original = settings.read_text(encoding="utf-8")
    settings.write_text(original.replace('requires_openai_auth = false', 'requires_openai_auth = true'), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        codex_backend.invoke(root, "feedback", codex_backend.MODEL, settings, time.time() + 60, 0)
    settings.write_text(original, encoding="utf-8")
    with pytest.raises(RuntimeError, match="deadline"):
        codex_backend.invoke(root, "feedback", codex_backend.MODEL, settings, time.time() - 1, 0)
    assert not list((root / "traces").glob("*.jsonl"))


def test_environment_marks_exact_stage_root_for_outer_watchdog(tmp_path, monkeypatch):
    from benchmarks.full_HCMA.harness.agents import MARKER_ENV
    monkeypatch.setenv(MARKER_ENV, "unrelated-run")
    root = tmp_path / "run/agent/embed"
    env = codex_backend._env(tmp_path / "temporary/codex_home", authenticated=False, run=root, marker_env=MARKER_ENV)
    assert env[MARKER_ENV] == str(root.resolve())
    assert os.environ[MARKER_ENV] == "unrelated-run"
    assert "LLM_API_KEY" not in env


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="Linux /proc scanner")
def test_watchdog_finds_and_kills_detached_descendant_outside_stage_cwd(tmp_path):
    from benchmarks.full_HCMA.harness.agents import _run_pids, kill_run_processes
    import signal
    script = ("import subprocess,sys,time; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
              "print(p.pid,flush=True); time.sleep(30)")
    stage = tmp_path / "run/agent/embed"
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=tmp_path,
                            env=codex_backend._env(tmp_path / "codex_home", authenticated=False, run=stage),
                            start_new_session=True, stdout=subprocess.PIPE, text=True)
    descendant = None
    try:
        descendant = int(proc.stdout.readline())
        assert {proc.pid, descendant}.issubset(_run_pids(stage))
        killed = kill_run_processes(stage, grace=0)
        proc.wait(timeout=5)
        assert {proc.pid, descendant}.issubset(killed)
        assert not ({proc.pid, descendant} & set(_run_pids(stage)))
    finally:
        for pid in (proc.pid, descendant):
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        proc.wait(timeout=5)
        proc.stdout.close()


# ---------------------------------------------------------------- the stage protocol, with a fake Codex and stub tools

REMOVED_GUIDANCE = ("Stop tuning", "quality suffices", "evidence is weak", "time is short", "Reserve time",
                    "Evidence policy", "Prefer broad coverage", "admission", "measurab")


def d(action, request, reason="r"):
    return {"action": action, "request_json": json.dumps(request), "reason": reason}


class FakeCodex:
    """Scripted decisions per session (group, dataset or an image ID); records prompts, attachments, concurrency."""

    def __init__(self, scripts, delay=0.0):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.prompts, self.images, self.markers, self.efforts = {}, {}, [], []
        self.delay, self.active, self.peak = delay, 0, 0
        self.lock = threading.Lock()

    def __call__(self, root, prompt, model, settings, deadline, index, session=None, images=(), marker=None,
                 effort=None):
        key = Path(root).name
        with self.lock:
            self.efforts.append(effort)
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.prompts.setdefault(key, []).append(prompt)
            self.images.setdefault(key, []).append([Path(p).read_bytes() for p in images])
            self.markers.append(marker)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
            choice = self.scripts[key].pop(0)
        if isinstance(choice, BaseException):
            raise choice
        return choice, {"total_tokens": 15}, None


def png(path, value=128):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), value).save(path)
    return path


class FakeSegmenter:
    device = type("Device", (), {"type": "cuda"})()
    kwargs = []

    def execute(self, image, image_id, query_cache_dir, **kwargs):
        import numpy as np
        type(self).kwargs.append({"image_id": image_id, **kwargs})
        folder = Path(query_cache_dir)
        folder.mkdir(parents=True, exist_ok=True)
        labels = np.zeros((16, 16), np.int32)
        if kwargs["diameter"] != 5.0:
            labels[2:6, 2:6], labels[9:14, 9:14] = 1, 2
        np.save(folder / "mask.npy", labels)
        overlay = png(folder / "cell_overlay.png", 200)
        return {"summary": "cells", "status": "success", "cell_count": int(labels.max()),
                "mask_path": str(folder / "mask.npy"), "overlay_path": str(overlay), "visual_outputs": [str(overlay)],
                "flow_error": 0.25, "overlay_qc": {"passed": True, "cell_count": int(labels.max())},
                "parameters": kwargs, "seconds": 0.0}


class FakeCropper:
    kwargs, active, peak = [], 0, 0
    lock = threading.Lock()

    def execute(self, original_image, nuclei_mask, source_image_id, query_cache_dir, **kwargs):
        import numpy as np
        import tifffile
        with type(self).lock:
            type(self).kwargs.append({"image_id": source_image_id, "original_image": original_image,
                                      "nuclei_mask": nuclei_mask, **kwargs})
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
        time.sleep(0.2)
        with type(self).lock:
            type(self).active -= 1
        folder = Path(query_cache_dir) / "single_cell_cropper"
        folder.mkdir(parents=True, exist_ok=True)
        cells = []
        for k in range(2):
            tile = folder / f"cell_{k}.tif"
            tifffile.imwrite(tile, np.full((8, 8, 5), 10 * k, np.uint8))
            cells.append({"cell_id": f"default_{source_image_id}_cell_{k:04d}", "crop_path": str(tile),
                          "crop_bbox": [k * 8, 0, k * 8 + 8, 8], "area": 20})
        metadata = folder / "cell_crops_metadata.json"
        write_json(metadata, {"cell_metadata": cells, "cell_crop_objects": ["object"] * 2})
        return {"summary": "2 crops", "cell_count": 2, "total_detected_objects": 2,
                "cell_crops_metadata_path": str(metadata), "cell_crop_objects": [object(), object()],
                "processing_statistics": {"final_cell_count": 2}, "seconds": 0.2}


@pytest.fixture
def image_stage(tmp_path, monkeypatch, gateway):
    """(prep, run, config, rows) for an image stage; cli.child runs the native worker in this process."""
    import numpy as np
    import tifffile
    from benchmarks.full_CPJUMP1.harness import cli, pipeline
    from shape.tools.cell_segmenter import tool as segmenter_module
    from shape.tools.single_cell_cropper import tool as cropper_module

    FakeSegmenter.kwargs, FakeCropper.kwargs, FakeCropper.peak = [], [], 0
    monkeypatch.delenv("SHAPE_SAM_SERVER_URL", raising=False)
    monkeypatch.setattr(segmenter_module, "Cell_Segmenter_Tool", FakeSegmenter)
    monkeypatch.setattr(cropper_module, "Single_Cell_Cropper_Tool", FakeCropper)
    waves = []

    def build(stage, n=3, cores=4):
        prep, run = tmp_path / "prep", tmp_path / "run"
        (prep / "public").mkdir(parents=True)
        (run / "logs").mkdir(parents=True)
        rows, segmentation = [], []
        for i in range(n):
            ident = f"img{i}"
            tifffile.imwrite(prep / "public" / f"{ident}_seg.tif", np.full((2, 16, 16), 100 * (i + 1), np.uint16),
                             photometric="minisblack")
            tifffile.imwrite(prep / "public" / f"{ident}.tif",
                             np.arange(5 * 16 * 16, dtype=np.uint16).reshape(5, 16, 16), photometric="minisblack")
            rows.append({"image_id": ident, "segment_file": f"{ident}_seg.tif", "file": f"{ident}.tif",
                         "plate": "p1", "well": f"w{i}", "usage": "evaluation_only"})
            mask = run / "masks" / f"cell_mask_{ident}.tif"
            mask.parent.mkdir(exist_ok=True)
            tifffile.imwrite(mask, np.ones((16, 16), np.uint32))
            segmentation.append({"image_id": ident, "mask": str(mask), "sha256": digest(mask), "min_area": 15,
                                 "n_cells": 1})
        write_json(run / "segmentation.json", segmentation)
        monkeypatch.setattr(agent, "load_prepared", lambda root: rows)

        def child(command, log, timeout, **kwargs):
            folder = Path(command[command.index("--run") + 1])
            config = read_json(folder / "CONFIG.json")
            waves.append(config["calls"])
            log.write_text("", encoding="utf-8")
            pipeline.codex_calls(command[command.index("native-trial") + 1], prep, folder, config,
                                 [r for r in rows if r["image_id"] in config["images"]], threads=True)   # stubs

        monkeypatch.setattr(cli, "child", child)
        config = {"arm": "codex", "model": codex_backend.MODEL, "deadline": time.time() + 600, "cores": cores,
                  "images": [r["image_id"] for r in rows]}
        return prep, run, config, rows, waves

    return build


def test_round_budget_and_per_round_cap_reject_the_whole_request_and_tell_the_agent(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=2)
    script = [d("run", [{"images": ["img0"], "params": {"diameter": 10}}, {"images": ["img0"], "params": {"diameter": 11}}])]
    script += [d("run", [{"images": ["img0"], "params": {"diameter": 20 + k}}]) for k in range(7)]
    script += [d("run", [{"images": ["img0", "img1"], "params": {"diameter": 50}}]), d("finish", {"all": "c0"})]
    codex = FakeCodex({"group": script})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["rejected"] + ["executed"] * 7 + ["rejected", "finished"]
    assert "more than 1 call(s) in one round" in rounds[0]["message"] and "nothing was run" in rounds[8]["message"]
    assert "already used all 8 tool-call rounds" in rounds[8]["message"] and "img0" in rounds[8]["message"]
    assert "more than 1 call(s) in one round" in codex.prompts["group"][1]
    calls = read_json(run / "agent/segment/CALLS.json")
    assert sorted(calls["img0"]) == [f"c{k}" for k in range(8)] and sorted(calls["img1"]) == ["c0"]
    assert len(waves) == 8  # the default round and the seven accepted rounds only


@pytest.mark.parametrize("stage,cores,images,expected", [("segment", 8, 2, 1), ("crop", 8, 1000, 1), ("crop", 8, 2, 2),
                                                        ("embed", 8, 2, 1), ("batch", 32, 2, 4)])
def test_round_cap_is_shapes_verifier_concurrency(monkeypatch, stage, cores, images, expected):
    from shape.agent.verification.verifiers import batch_correction_unified, crop, representation, segmentation
    monkeypatch.delenv("SHAPE_SAM_SERVER_URL", raising=False)
    config = {"cores": cores, "images": [f"i{k}" for k in range(images)]}
    assert agent.round_cap(stage, config) == expected
    verifier = {"segment": segmentation.SegmentationAgenticVerifier, "crop": crop.CropAgenticVerifier,
                "embed": representation.RepresentationAgenticVerifier,
                "batch": batch_correction_unified.BatchCorrectionVerifier}[stage]
    monkeypatch.setattr(verifier, "concurrency", lambda self: 7)
    assert agent.round_cap(stage, config) == 7
    assert agent.MAX_TOOL_ROUNDS == __import__("shape.agent.verification.base", fromlist=["BUDGET"]).BUDGET == 8


def test_subset_round_reruns_only_named_images_in_parallel(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("crop", n=4, cores=4)
    codex = FakeCodex({"group": [d("run", [{"images": ["img1", "img2", "img3"], "params": {"margin_mode": "tight"}}]),
                                 d("finish", {"img1": "c1"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    FakeCropper.peak = 0
    agent.run_stage("crop", prep, run, config)
    assert [c["image_id"] for c in waves[1]] == ["img1", "img2", "img3"]
    assert {c["call"] for c in waves[1]} == {"c1"}
    assert FakeCropper.peak > 1
    assert sorted(read_json(run / "agent/crop/CALLS.json")["img0"]) == ["c0"]


def test_per_image_params_and_cell_overrides_reach_the_tool_kwargs(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("crop", n=2)
    override = {"default_img0_cell_0001": {"margin_mode": "mask_bbox", "auto_tighten": True}}
    codex = FakeCodex({"group": [
        d("run", [{"images": "all", "params": {"margin_frac": 0.3},
                   "per_image": {"img0": {"margin_mode": "tight", "cell_overrides": override}}}]),
        d("finish", {"all": "c1"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("crop", prep, run, config)
    second = {k["image_id"]: k for k in FakeCropper.kwargs[2:]}
    assert second["img0"]["margin_frac"] == 0.3 and second["img0"]["margin_mode"] == "tight"
    assert second["img0"]["cell_overrides"] == override
    assert second["img1"]["margin_frac"] == 0.3 and "cell_overrides" not in second["img1"]
    for kwargs in FakeCropper.kwargs:
        assert kwargs["group"] == "default" and kwargs["output_format"] == "tif" and kwargs["min_area"] == 15
    cells = read_json(run / "cells.json")
    assert len(cells) == 4 and read_json(run / "excluded_cells.json") == []
    import tifffile
    assert tifffile.imread(cells[0]["crop_path"]).shape == (5, 8, 8)


def test_per_image_sessions_run_concurrently_with_their_own_feedback_and_the_cap(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=3, cores=3)
    scripts = {"group": [d("handoff", "all")]}
    for i in range(3):
        scripts[f"img{i}"] = [d("run", [{"images": "all", "params": {"diameter": 40 + i}}]),
                              d("run", [{"images": "all", "params": {"diameter": 60 + k}} for k in range(2)]),
                              d("finish", {"all": "c1"})]
    codex = FakeCodex(scripts, delay=0.2)
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("segment", prep, run, config)
    assert codex.peak >= 2
    for i in range(3):
        prompts = codex.prompts[f"img{i}"]
        assert all(f"handles image img{i} only" in p for p in prompts)
        assert not any(f"img{j}" in p for p in prompts for j in range(3) if j != i)
        assert "more than 1 call(s) in one round" in prompts[2]
    assert len(waves) == 2 and sorted(c["image_id"] for c in waves[1]) == ["img0", "img1", "img2"]
    assert {c["params"]["diameter"] for c in waves[1]} == {40.0, 41.0, 42.0}
    selected = read_json(run / "agent/segment/SELECTED.json")
    assert {k: (v["call"], v["source"], v["session"]) for k, v in selected["deliveries"].items()} == {
        f"img{i}": ("c1", "agent", f"image:img{i}") for i in range(3)}
    assert selected["total_tokens"] == 15 * 10
    assert set(codex.markers) == {run / "agent/segment"}
    assert [r["call"] for r in read_json(run / "segmentation.json")] == ["c1"] * 3


def test_inspect_is_not_a_call_and_attaches_only_raw_inputs_and_tool_written_files(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=1)
    codex = FakeCodex({"group": [
        d("inspect", [{"image": "img0", "call": "input"}, {"image": "img0", "call": "c0", "file": "overlay_path"}]),
        d("inspect", [{"image": "img0", "call": "c0", "file": "cell_mask_img0.tif"}]),
        d("inspect", [{"image": "img0", "call": "c0", "file": "evidence/panel.png"}]),
        d("inspect", [{"image": "img0", "call": "c0", "file": "../../../../../prep/public/img0.tif"}]),
        d("finish", {})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    original = agent.StageRun.wave

    def wave(self, requests):
        names = original(self, requests)
        png(self.root / "waves/w000/img0/c0/evidence/panel.png")   # a harness-made panel is never a tool file
        return names

    monkeypatch.setattr(agent.StageRun, "wave", wave)
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["attached", "rejected", "rejected", "rejected", "finished"]
    attached = codex.images["group"][1]
    assert len(attached) == 2 and all(b.startswith(b"\x89PNG") for b in attached)
    overlay = run / "agent/segment/waves/w000/img0/c0/tool/cell_overlay.png"
    assert attached[1] == overlay.read_bytes()
    assert codex.images["group"][2] == [] and "Attached images: image 1: img0 input" in codex.prompts["group"][1]
    assert sorted(read_json(run / "agent/segment/CALLS.json")["img0"]) == ["c0"] and len(waves) == 1
    selected = read_json(run / "agent/segment/SELECTED.json")
    assert selected["deliveries"]["img0"]["source"] == "default_unnamed"


def test_finish_with_partial_choices_delivers_the_default_call_for_unnamed_images(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=3)
    codex = FakeCodex({"group": [d("run", [{"images": "all", "params": {"diameter": 40}}]),
                                 d("run", [{"images": ["img2"], "params": {"diameter": 5}}]),
                                 d("finish", {"img2": "c2"}),
                                 d("finish", {"img1": "c1"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert rounds[2]["outcome"] == "rejected" and "img2/c2 has no cells" in rounds[2]["message"]
    assert [(r["image_id"], r["call"]) for r in read_json(run / "segmentation.json")] == [
        ("img0", "c0"), ("img1", "c1"), ("img2", "c0")]
    deliveries = read_json(run / "agent/segment/SELECTED.json")["deliveries"]
    assert {k: v["source"] for k, v in deliveries.items()} == {
        "img0": "default_unnamed", "img1": "agent", "img2": "default_unnamed"}
    assert "img0" in rounds[3]["message"] and "img2" in rounds[3]["message"]
    assert sorted(k["diameter"] for k in FakeSegmenter.kwargs if k["image_id"] == "img2") == [5.0, 30.0, 40.0]
    assert all(k["model_type"] == "cpsam" and k["bsize"] == 256 for k in FakeSegmenter.kwargs)


@pytest.mark.parametrize("stage", ["segment", "crop"])
def test_prompts_state_facts_without_removed_guidance(image_stage, monkeypatch, stage):
    from benchmarks.full_CPJUMP1.harness import pipeline
    prep, run, config, rows, waves = image_stage(stage, n=2)
    codex = FakeCodex({"group": [d("finish", {})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage(stage, prep, run, config)
    prompt = codex.prompts["group"][0]
    assert "Tuning the tool's parameters can improve the tool's performance." in prompt
    assert f"Task: {agent.TASKS[stage]}" in prompt and "at most 8 tool-call rounds per image" in prompt
    assert "at most 1 call(s)" in prompt and "any number of parameters" in prompt
    assert "can process many images at once" in prompt and "batched in one call spec" in prompt
    assert not any(text.lower() in prompt.lower() for text in REMOVED_GUIDANCE)
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert all(part in source for part in agent.TASKS[stage].split("; "))
    if stage == "crop":
        assert "cell_overrides" in prompt and '"per_cell"' in prompt


def test_a_session_without_finish_delivers_the_default_call_and_records_why(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=1)
    config["smoke_policy"] = {"codex_rounds": 2}
    codex = FakeCodex({"group": [d("run", [{"images": "all", "params": {"diameter": 40}}]),
                                 d("run", [{"images": "all", "params": {"diameter": 50}}])]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("segment", prep, run, config)
    assert len(waves) == 2 and "only finish is accepted" in codex.prompts["group"][-1]
    delivery = read_json(run / "agent/segment/SELECTED.json")["deliveries"]["img0"]
    assert delivery["call"] == "c0" and delivery["source"] == "default_session_unfinished"
    assert "not finish" in delivery["reason"]
    assert read_json(run / "segmentation.json")[0]["call"] == "c0"


def test_a_malformed_per_image_reply_is_handled_as_a_rejected_request_and_the_session_goes_on(image_stage,
                                                                                                monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=2, cores=2)
    codex = FakeCodex({"group": [d("handoff", "all")],
                       "img0": [d("run", [{"images": "all", "params": {"diameter": 40}}]),
                                agent.MalformedDecision("Codex decision JSON is malformed", tokens=9),
                                d("finish", {"all": "c1"})],
                       "img1": [d("run", [{"images": "all", "params": {"diameter": 41}}]),
                                d("finish", {"all": "c1"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("segment", prep, run, config)
    selected = read_json(run / "agent/segment/SELECTED.json")
    assert selected["deliveries"]["img0"] == {"call": "c1", "source": "agent", "session": "image:img0"}
    assert selected["deliveries"]["img1"] == {"call": "c1", "source": "agent", "session": "image:img1"}
    assert selected["total_tokens"] == 15 * 5 + 9
    assert [(r["image_id"], r["call"]) for r in read_json(run / "segmentation.json")] == [("img0", "c1"), ("img1", "c1")]
    rounds = [r for r in read_json(run / "agent/segment/ROUNDS.json") if r["session"] == "image:img0"]
    assert [(r["round"], r["outcome"]) for r in rounds] == [(0, "executed"), (1, "malformed"), (2, "finished")]
    assert rounds[1]["message"] == "Codex decision JSON is malformed" and rounds[1]["action"] is None
    after = codex.prompts["img0"][2]
    assert ("Your reply in round 2 was not a valid decision and was not executed: Codex decision JSON is malformed. "
            f"Every response must be {agent.RESPONSE_FORMAT}.") in after
    assert '"outcome": "malformed", "message": "Codex decision JSON is malformed"' in after
    assert "was not a valid decision" not in codex.prompts["img0"][1]


def test_a_round_timeout_still_delivers_that_images_default_and_the_arm_goes_on(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=2, cores=2)
    codex = FakeCodex({"group": [d("handoff", "all")],
                       "img0": [d("run", [{"images": "all", "params": {"diameter": 40}}]),
                                agent.RoundTimeout("Codex exceeded its deadline")],
                       "img1": [d("run", [{"images": "all", "params": {"diameter": 41}}]),
                                d("finish", {"all": "c1"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("segment", prep, run, config)
    selected = read_json(run / "agent/segment/SELECTED.json")
    assert selected["deliveries"]["img0"]["source"] == "default_session_unfinished"
    assert "RoundTimeout" in selected["deliveries"]["img0"]["reason"]
    assert selected["deliveries"]["img1"] == {"call": "c1", "source": "agent", "session": "image:img1"}
    assert [(r["image_id"], r["call"]) for r in read_json(run / "segmentation.json")] == [("img0", "c0"), ("img1", "c1")]
    failed = [r for r in read_json(run / "agent/segment/ROUNDS.json") if r["outcome"] == "failed_round"]
    assert [r["session"] for r in failed] == ["image:img0"]


def test_malformed_and_rejected_rounds_use_session_rounds_but_no_tool_call_round(image_stage, monkeypatch):
    """A malformed reply is counted as a rejected request is: one session round, no tool-call round (no wave)."""
    prep, run, config, rows, waves = image_stage("segment", n=1)
    codex = FakeCodex({"group": [d("run", [{"images": "all", "params": {"diameter": 40}}]),
                                 agent.MalformedDecision("invalid Codex decision types", tokens=3),
                                 d("finish", {"img0": "c9"}),
                                 d("run", [{"images": "all", "params": {"diameter": 50}}]),
                                 d("finish", {"all": "c2"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["executed", "malformed", "rejected", "executed", "finished"]
    assert len(waves) == 3                                     # c0, c1, c2: the malformed and rejected rounds ran none
    calls = read_json(run / "agent/segment/CALLS.json")["img0"]
    assert sorted(calls) == ["c0", "c1", "c2"] and len({r["wave"] for r in calls.values()}) == 3
    assert "Round 4 of 32" in codex.prompts["group"][3]
    assert read_json(run / "agent/segment/SELECTED.json")["deliveries"]["img0"]["call"] == "c2"


def test_malformed_replies_are_bounded_by_the_sessions_rounds(image_stage, monkeypatch):
    """No other cap: the session's rounds bound malformed replies as they bound rejected requests; a malformed last
    round ends the session unfinished (the default call), as a rejected last round does."""
    prep, run, config, rows, waves = image_stage("segment", n=1)
    config["smoke_policy"] = {"codex_rounds": 3}
    codex = FakeCodex({"group": [agent.MalformedDecision("Codex returned malformed JSON or no final message")] * 3})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["malformed"] * 3 and len(codex.prompts["group"]) == 3
    delivery = read_json(run / "agent/segment/SELECTED.json")["deliveries"]["img0"]
    assert delivery["call"] == "c0" and delivery["source"] == "default_session_unfinished"
    assert delivery["reason"] == ("the last round's reply was not a valid decision: Codex returned malformed JSON or "
                                  "no final message")
    assert len(waves) == 1


def test_a_malformed_dataset_stage_reply_is_handled_as_a_rejected_request(dataset_stage, monkeypatch):
    run, config, workers = dataset_stage("batch")
    codex = FakeCodex({"dataset": [agent.MalformedDecision("Codex decision must give a reason", tokens=4),
                                   d("finish", {"all": "input"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("batch", run, run, config)
    rounds = read_json(run / "agent/batch/ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["malformed", "finished"]
    assert "Codex decision must give a reason. Every response must be" in codex.prompts["dataset"][1]
    assert read_json(run / "agent/batch/SELECTED.json")["deliveries"]["dataset"]["source"] == "agent"


def test_a_basic_llm_failure_still_fails_the_arm(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=1)
    codex = FakeCodex({"group": [d("handoff", "all")],
                       "img0": [RuntimeError("Codex CLI failed; Stanford Responses support/auth/model preflight is required")]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    with pytest.raises(RuntimeError, match="Codex CLI failed"):
        agent.run_stage("segment", prep, run, config)
    assert not (run / "agent/segment/SELECTED.json").exists()


def test_a_malformed_reply_keeps_its_usage_and_its_trace_out_of_the_decision_traces(tmp_path, offline_cli):
    offline_cli("stale_choice")
    root = tmp_path / "run"
    settings = codex_backend.setup(root / "auth", codex_backend.MODEL)
    with pytest.raises(agent.MalformedDecision) as caught:
        codex_backend.invoke(root, "feedback", codex_backend.MODEL, settings, time.time() + 20, 0)
    assert caught.value.tokens == 15
    assert not list((root / "traces").glob("round_*")) and (root / "traces/failed_round_000.jsonl").is_file()
    assert codex_backend.stream_usage(root / "traces/failed_round_000.jsonl", codex_backend.MODEL)["total_tokens"] == 15


@pytest.fixture
def dataset_stage(tmp_path, monkeypatch, gateway):
    """A dataset stage whose native worker writes the files the real one writes (no SHAPE gate fields)."""
    from benchmarks.full_CPJUMP1.harness import cli
    workers = []

    def build(stage, rule="RUN"):
        run = tmp_path / "run"
        (run / "logs").mkdir(parents=True)
        for name in agent.native.INPUTS[stage]:
            (run / name).write_text("input", encoding="utf-8")

        def child(command, log, timeout, **kwargs):
            assert timeout > 0 and kwargs["new_session"] is False
            folder = Path(command[command.index("--run") + 1])
            config = read_json(Path(command[command.index("--config") + 1]) if "--config" in command
                               else folder / "CONFIG.json")
            workers.append((folder, config))
            log.write_text("", encoding="utf-8")
            if "ssl_selection" in config:
                return
            if stage == "embed":
                trained = config["tool_params"].get("adaptation_depth") != "zero_shot"
                checkpoint = folder / "resume_best.pt"
                if trained:
                    checkpoint.write_bytes(b"state")
                write_json(folder / "SSL_SELECTION.json", {"mode": "frozen_checkpoint" if trained else "zero_shot",
                                                          "checkpoint": str(checkpoint) if trained else None})
                write_json(folder / "ssl_result.json", {"mode": "training" if trained else "zero-shot",
                                                       "val_history": [{"epoch": 1, "proxy": 0.5}],
                                                       "resume_best": str(checkpoint) if trained else None})
            else:
                (folder / "corrected.npy").write_bytes(b"x")
                write_json(folder / "batch_SEARCH.json", {"admission": {"decision": rule}})
                write_json(folder / "FEEDBACK.json", {"admission": rule, "calibration": {"SHAPE_ONLY": 1}})
                if not config["uncorrected"]:
                    write_json(folder / "batch/call_000/result.json", {"status": "corrected", "seconds": 1.0,
                                                                     "gt_free_metrics": {"batch_effect_detected": True}})

        monkeypatch.setattr(cli, "child", child)
        config = {"arm": "codex", "model": codex_backend.MODEL, "deadline": time.time() + 3 * 3600, "cores": 2,
                  "images": ["img0"], "train_cells_per_site": 128,
                  "ssl_epoch_budget": {"n_train": 128, "max_epochs": 12}}
        return run, config, workers

    return build


def test_ssl_gate_no_longer_blocks_codex_and_inference_follows_selection(dataset_stage, monkeypatch):
    run, config, workers = dataset_stage("embed")
    codex = FakeCodex({"dataset": [
        d("run", [{"params": {"max_epochs": 2, "view_pairing": "same_well"}}]),     # max_epochs: not a parameter
        d("run", [{"params": {"view_pairing": "same_well"}}]),
        d("run", [{"params": {"init_from": "parent", "trainable_pct": 50}, "parent": "c1"}]),
        d("run", [{"params": {"init_from": "parent"}, "parent": "input"}]),
        d("finish", {"all": "c2"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("embed", run, run, config)
    rounds = read_json(run / "agent/embed/ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["rejected", "executed", "executed", "rejected", "finished"]
    assert "max_epochs" in rounds[0]["message"] and "max_epochs" not in agent.registry("embed")
    assert "max_epochs is fixed at 12 for every call" in codex.prompts["dataset"][0]
    first, second = workers[1][1], workers[2][1]
    assert workers[0][1]["tool_params"] == {"adaptation_depth": "zero_shot"}
    assert first["tool_params"] == {"view_pairing": "same_well"} and first["fit_only"]
    assert second["tool_params"] == {"trainable_pct": 50} and second["resume_checkpoint"].endswith("resume_best.pt")
    for _, call in workers[:3]:
        assert abs(call["ssl_search_deadline"] - (time.time() + 3 * 3600 * 0.65)) < 120
    assert "Training time: training calls must end by this stage's search deadline" in codex.prompts["dataset"][0]
    assert "kept for frozen inference" in codex.prompts["dataset"][0]
    inference = workers[-1][1]
    assert inference["ssl_selection"]["checkpoint"] == second["resume_checkpoint"].replace("c1", "c2")
    assert read_json(run / "agent/embed/SELECTED.json")["deliveries"]["dataset"]["call"] == "c2"
    assert not any(text.lower() in p.lower() for p in codex.prompts["dataset"] for text in REMOVED_GUIDANCE)
    assert '"val_history"' in codex.prompts["dataset"][1]


@pytest.mark.parametrize("rule", ["ABSTAIN", "SKIP", "RUN"])
def test_batch_calibration_rule_still_blocks_correction(dataset_stage, monkeypatch, rule):
    run, config, workers = dataset_stage("batch", rule)
    codex = FakeCodex({"dataset": [d("run", [{"params": {"theta": 1.0}}]),
                                   d("finish", {"all": "c1" if rule == "RUN" else "input"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("batch", run, run, config)
    rounds = read_json(run / "agent/batch/ROUNDS.json")
    if rule == "RUN":
        assert rounds[0]["outcome"] == "executed" and workers[1][1]["tool_params"] == {"theta": 1.0}
    else:
        assert rounds[0]["outcome"] == "rejected" and rule in rounds[0]["message"] and len(workers) == 1
        assert rule in codex.prompts["dataset"][1]
    assert "SHAPE_ONLY" not in "".join(codex.prompts["dataset"])
    assert (run / "corrected.npy").read_bytes() == b"x"


def test_delivery_verification_rejects_changed_call_artifacts(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=1)
    monkeypatch.setattr(codex_backend, "invoke", FakeCodex({"group": [d("finish", {})]}))
    agent.run_stage("segment", prep, run, config)
    root = run / "agent/segment"
    selected, calls = read_json(root / "SELECTED.json"), read_json(root / "CALLS.json")
    agent.verify_delivery(root, selected, calls)
    (root / "waves/w000/img0/c0/cell_mask_img0.tif").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        agent.verify_delivery(root, selected, calls)


def test_ssl_calls_share_shapes_search_deadline_and_stop_when_its_window_ends(dataset_stage, monkeypatch):
    from benchmarks.full_CPJUMP1.harness import pipeline
    run, config, workers = dataset_stage("embed")
    monkeypatch.setattr(pipeline, "SSL_SEARCH_SHARE", 0.01)    # the same constant SHAPE's search uses
    codex = FakeCodex({"dataset": [d("run", [{"params": {"view_pairing": "same_well"}}]), d("finish", {"all": "input"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("embed", run, run, config)
    rounds = read_json(run / "agent/embed/ROUNDS.json")
    assert rounds[0]["outcome"] == "rejected" and "training window" in rounds[0]["message"]
    assert len(workers) == 2 and workers[0][1]["ssl_search_deadline"] < time.time() + 3 * 3600 * 0.02


def test_dataset_rounds_hold_at_most_the_verifiers_calls(dataset_stage, monkeypatch):
    run, config, workers = dataset_stage("embed")
    codex = FakeCodex({"dataset": [d("run", [{"params": {"view_pairing": "same_well"}},
                                             {"params": {"view_pairing": "same_site"}}]),
                                   d("finish", {"all": "input"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("embed", run, run, config)
    rounds = read_json(run / "agent/embed/ROUNDS.json")
    assert rounds[0]["outcome"] == "rejected" and "more than 1" in rounds[0]["message"]
    assert "at most 8 tool-call rounds in this stage" in codex.prompts["dataset"][0]


def test_shapes_cell_rules_reach_the_tool_as_compiled_cell_overrides(image_stage, monkeypatch):
    """SHAPE's cell rules in a Codex crop call (2026-10-12): compiled per image into cell_overrides; a rules_parent must
    be a finished call of that image, and its folder goes to the worker."""
    prep, run, config, rows, waves = image_stage("crop", n=2)
    rule = [{"when": [{"readout": "area", "op": ">", "value": 10}],
             "apply": [{"key": "margin_mode", "value": "mask_bbox"}, {"key": "auto_tighten", "value": True}]}]
    codex = FakeCodex({"group": [
        d("run", [{"images": "all", "params": {"cell_rules": rule, "rules_parent": "c9"}}]),
        d("run", [{"images": "all", "params": {"cell_rules": rule}}]),
        d("run", [{"images": ["img0"], "params": {"cell_rules": rule, "rules_parent": "c0"}}]),
        d("finish", {"all": "c1"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("crop", prep, run, config)
    rounds = read_json(run / "agent/crop/ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["rejected", "executed", "executed", "finished"]
    assert "rules_parent 'c9' is not a finished call" in rounds[0]["message"]
    compiled = {k["image_id"]: k["cell_overrides"] for k in FakeCropper.kwargs[2:4]}
    assert compiled["img0"] == {"default_img0_cell_0000": {"margin_mode": "mask_bbox", "auto_tighten": True}}
    assert "cell_rules" not in FakeCropper.kwargs[2] and "rules_parent" not in FakeCropper.kwargs[2]
    assert waves[-1][0]["rules_parent_folder"].replace("\\", "/").endswith("agent/crop/waves/w000/img0/c0")
    calls = read_json(run / "agent/crop/CALLS.json")
    assert calls["img0"]["c1"]["params"]["cell_rules"][0]["when"][0]["value"] == "10"
    assert "cell_rules" in codex.prompts["group"][0] and "rules_parent" in codex.prompts["group"][0]


# ---- the group prompt's call table: full when it fits PROMPT_TABLE_BUDGET, else aggregated ------------------------
CODEX_INPUT_LIMIT = 1_048_576


def hash_int(text):
    return sum(ord(c) * 31 ** i for i, c in enumerate(text))


def livecell_record(k, image, status="complete", qc_ok=True):
    """One synthetic segmentation call record shaped like Cell_Segmenter_Tool's result (about 170 characters of
    table row per call)."""
    seed = (hash_int(image) % 997) / 997
    view = {"status": "success" if status == "complete" else "failed", "cell_count": 100 + int(seed * 400),
            "image_id": image, "attempt": k + 1, "param_sig": "sig" + str(k), "model_used": "CellposeModel (cpsam)",
            "flow_error": round(0.2 + seed / 10, 6),
            "overlay_qc": {"cell_count": 100 + int(seed * 400), "foreground_coverage": round(0.1 + seed / 5, 6),
                           "overlay_max": 255.0, "overlay_std": round(40 + seed, 6),
                           "overlay_dynamic_range": 255.0, "overlay_is_black": False,
                           "coverage_implausible": False, "passed": qc_ok},
            "seconds": round(1.5 + seed, 1), "parameters": {"diameter": 10 + k}}
    return {"params": {"diameter": 10 + k, "flow_threshold": round(0.4 - 0.01 * k, 2)}, "status": status,
            "error": None if status == "complete" else "Segmentation output failed visual QC: cells=0",
            "seconds": 1.5, "wave": f"w{k}", "folder": f"calls/c{k}", "artifacts": {}, "view": view,
            "delivery": None}


def table_run(n_images, n_calls, bad=(), qc_bad=(), facts_chars=40_000):
    """A StageRun without a tool or Codex: only what `prompt` reads (calls, images, config, facts)."""
    run = object.__new__(agent.StageRun)
    run.images = [f"IMG_{i:05d}" for i in range(n_images)]
    run.config = {"deadline": time.time() + 600}
    run.facts = lambda rounds: "FACTS " + "f" * facts_chars
    run.calls = {image: {f"c{k}": livecell_record(k, image, "error" if (image, k) in bad else "complete",
                                                  (image, k) not in qc_bad)
                         for k in range(n_calls)} for image in run.images}
    return run


def old_group_prompt(run, session, index, rounds):
    """The group prompt as it was before the aggregated table (the reference text path)."""
    parts = [run.facts(rounds),
             f"This group session: {len(session.units)} image(s); "
             f"{len(run.images) - len(session.units)} image(s) left the group for per-image mode.",
             f"Calls of this group (compact): {json.dumps(run.table(session.units), separators=(',', ':'))}",
             f"Your previous decisions in this session: {json.dumps(session.decisions)}"]
    parts.append(f"Round {index + 1} of {rounds}. Seconds remaining for the entire arm: "
                 f"{run.config['deadline'] - time.time():.0f}.")
    return "\n".join(parts)


def aggregated_table(prompt):
    return json.loads(prompt.split("Aggregated table: ", 1)[1].split("\nYour previous decisions", 1)[0])


def test_a_table_within_the_budget_gives_the_unchanged_prompt(monkeypatch):
    monkeypatch.setattr(agent.time, "time", lambda: 1000.0)
    run = table_run(5, 3)
    session = agent.Session(Path("x"), "group", run.images)
    prompt = run.prompt(session, 2, 32)
    assert prompt == old_group_prompt(run, session, 2, 32)
    assert session.table_info["table_mode"] == "full" and "aggregated" not in prompt
    assert session.table_info["table_chars"] == len(json.dumps(run.table(run.images), separators=(",", ":")))


def test_an_over_budget_table_is_aggregated_and_the_prompt_stays_below_the_codex_limit():
    # every third image fails call c3: the flagged images' rows fill their whole budget (the largest prompt)
    run = table_run(1310, 8, bad={(f"IMG_{i:05d}", 3) for i in range(0, 1310, 3)})
    one_row = len(json.dumps(run.table(run.images)["rows"]["IMG_00000"][0], separators=(",", ":")))
    assert 130 <= one_row <= 220      # about the real per-call row size
    session = agent.Session(Path("x"), "group", run.images)
    session.decisions = [{"round": k, "action": "run", "request_json": "x" * 3000, "reason": "r", "outcome": "executed",
                          "message": "ran 1310 call(s): " + "c1" * 1000} for k in range(7)]
    biggest, modes = 0, []
    every = run.calls
    for n_calls in range(1, 9):
        run.calls = {image: dict(list(records.items())[:n_calls]) for image, records in every.items()}
        prompt = run.prompt(session, n_calls - 1, 32)
        modes.append(session.table_info["table_mode"])
        assert len(prompt) < CODEX_INPUT_LIMIT
        biggest = max(biggest, len(prompt))
    print(f"LARGEST_PROMPT_CHARS={biggest}")
    shown = json.dumps(aggregated_table(prompt)["flagged_rows"], separators=(",", ":"))
    assert agent.FLAGGED_ROWS_BUDGET - 5000 < len(shown) <= agent.FLAGGED_ROWS_BUDGET
    print(f"LAST_AGGREGATED_PROMPT_CHARS={len(prompt)}")
    assert modes == ["full"] + ["aggregated"] * 7
    assert biggest < CODEX_INPUT_LIMIT - 400_000      # a wide margin for the facts, decisions and inspected JSON
    assert session.table_info["table_chars"] < agent.PROMPT_TABLE_BUDGET < session.table_info["table_full_chars"]


def test_the_aggregated_prompt_tells_the_agent_what_it_sees_and_keeps_the_flagged_images():
    bad, qc_bad = {("IMG_00007", 1), ("IMG_00009", 1), ("IMG_00009", 2)}, {("IMG_00011", 2)}
    run = table_run(1310, 3, bad=bad, qc_bad=qc_bad)
    session = agent.Session(Path("x"), "group", run.images)
    prompt = run.prompt(session, 2, 32)
    full = len(json.dumps(run.table(run.images), separators=(",", ":")))
    assert f"would be {full} characters, above the prompt-table budget of {agent.PROMPT_TABLE_BUDGET} characters" in prompt
    assert "inspect" in prompt and "handoff" in prompt and "per-image mode" in prompt
    table = aggregated_table(prompt)
    assert table["aggregated"] is True and sorted(table["calls"]) == ["c0", "c1", "c2"]
    c1, c2 = table["calls"]["c1"], table["calls"]["c2"]
    assert (c1["images"], c1["ran"], c1["failed"], c1["qc_flagged"]) == (1310, 1308, 2, 0)
    assert (c2["images"], c2["ran"], c2["failed"], c2["qc_flagged"]) == (1310, 1309, 1, 1)
    assert c1["flagged_image_ids"] == ["IMG_00007", "IMG_00009"]
    assert c2["flagged_image_ids"] == ["IMG_00009", "IMG_00011"]
    stats = c1["stats"]["cell_count"]
    assert set(stats) == {"n", "mean", "median", "p10", "p90", "min", "max"} and stats["n"] == 1310
    assert stats["min"] <= stats["p10"] <= stats["median"] <= stats["p90"] <= stats["max"]
    assert c1["stats"]["status"]["values"]["failed"] == 2
    assert set(table["flagged_rows"]) == {"IMG_00007", "IMG_00009", "IMG_00011"} and table["flagged_rows_omitted"] == 0
    assert len(table["flagged_rows"]["IMG_00009"]) == 3
    assert {row[1] for rows in table["flagged_rows"].values() for row in rows} <= set(table["params"])
    assert table["columns"] == run.table(run.images[:1])["columns"]


def test_flagged_rows_beyond_their_budget_are_counted_not_shown(monkeypatch):
    monkeypatch.setattr(agent, "FLAGGED_ROWS_BUDGET", 3000)
    run = table_run(1310, 3, bad={(f"IMG_{i:05d}", 1) for i in range(0, 1310, 2)})
    session = agent.Session(Path("x"), "group", run.images)
    table = aggregated_table(run.prompt(session, 2, 32))
    shown = sorted(table["flagged_rows"])
    assert 0 < len(shown) < 655 and shown == [f"IMG_{i:05d}" for i in range(0, 2 * len(shown), 2)]    # first by image ID
    assert table["flagged_images"] == 655 and table["flagged_rows_omitted"] == 655 - len(shown)
    assert len(table["calls"]["c1"]["flagged_image_ids"]) == agent.FLAGGED_IDS_PER_CALL
    assert table["calls"]["c1"]["flagged_total"] == 655


def test_the_aggregated_prompt_is_deterministic():
    run = table_run(1310, 3, bad={("IMG_00005", 2)})
    first = agent.Session(Path("x"), "group", run.images)
    second = agent.Session(Path("x"), "group", list(run.images))
    a = run.prompt(first, 2, 32).split("Seconds remaining")[0]
    b = run.prompt(second, 2, 32).split("Seconds remaining")[0]
    assert a == b and "aggregated" in a


def test_group_rounds_record_the_table_mode_and_the_run_is_unchanged(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=3)
    codex = FakeCodex({"group": [d("run", [{"images": "all", "params": {"diameter": 12}}]), d("finish", {"all": "c1"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    monkeypatch.setattr(agent, "PROMPT_TABLE_BUDGET", 450)      # round 1: c0 only (386 chars), full; round 2: c0 + c1 (562), aggregated
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert [(r["table_mode"], r["outcome"]) for r in rounds] == [("full", "executed"), ("aggregated", "finished")]
    assert rounds[1]["table_full_chars"] > 450 and rounds[1]["table_chars"] > 0
    assert "Aggregated table:" in codex.prompts["group"][1] and "Aggregated table:" not in codex.prompts["group"][0]
    print(f"REAL_FACTS_PROMPT_CHARS={len(codex.prompts['group'][0])}")
    assert [r["call"] for r in read_json(run / "segmentation.json")] == ["c1"] * 3
    from benchmarks._shared import agentic_process_readouts as readouts
    record = readouts.Rec()
    readouts.agent_p3(record, "segment", rounds, read_json(run / "agent/segment/CALLS.json"))
    assert record.data["P3"]["group_table_mode_rounds"] == {"aggregated": 1, "full": 1}


def test_image_sessions_have_no_table_info(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=2)
    scripts = {"group": [d("handoff", "all")], "img0": [d("finish", {"all": "c0"})], "img1": [d("finish", {"all": "c0"})]}
    monkeypatch.setattr(codex_backend, "invoke", FakeCodex(scripts))
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    group = [r for r in rounds if r["session"] == "group"]
    assert len(group) == 1 and group[0]["table_mode"] == "full" and len(rounds) == 3
    assert all("table_mode" not in r for r in rounds if r["session"] != "group")


# ---- the other parts of the prompt: inspected JSON, previous decisions, the last-line limit -----------------------
def inspect_run(tmp_path, n_images=1310, n_calls=3, n_json=7, json_chars=300_000, facts_chars=40_000):
    """table_run plus `n_json` images whose call c1 wrote tool/meta.json of `json_chars` characters (like the cropper's
    per-cell metadata, about 1000 cells) for `inspect`; returns (run, session, items)."""
    run = table_run(n_images, n_calls, facts_chars=facts_chars)
    run.stage, run.root = "crop", tmp_path / "stage"
    items = []
    for image in run.images[:n_json]:
        folder = run.root / "calls" / image / "c1"
        (folder / "tool").mkdir(parents=True)
        write_json(folder / "tool" / "meta.json", {"cells": ["x" * 90 for _ in range(json_chars // 94)]})
        run.calls[image]["c1"]["folder"] = f"calls/{image}/c1"
        run.calls[image]["c1"]["artifacts"] = {"tool/meta.json": digest(folder / "tool" / "meta.json")}
        items.append({"image": image, "call": "c1", "file": "tool/meta.json"})
    return run, agent.Session(Path("x"), "group", run.images), items


def test_inspected_json_files_of_one_round_share_one_budget(tmp_path):
    run, session, items = inspect_run(tmp_path)
    run.inspect(session, items, 0)
    assert len(session.texts) == 7
    limit = agent.INSPECT_JSON_BUDGET // 7
    assert all(f"(first {limit} of " in t.split(": ", 1)[0] and "shared inspect budget" in t.split(": ", 1)[0]
               for t in session.texts)
    assert sum(len(t) for t in session.texts) < agent.INSPECT_JSON_BUDGET + 7 * 200
    prompt = run.prompt(session, 2, 32)
    assert len(prompt) < 600_000 and session.prompt_chars == len(prompt)


def test_one_small_inspected_json_is_unchanged_and_one_large_keeps_the_old_note(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.time, "time", lambda: 1000.0)
    run, session, items = inspect_run(tmp_path, n_images=5, n_calls=2, n_json=1, json_chars=2_000)
    run.inspect(session, items, 0)
    text = json.dumps({"cells": ["x" * 90 for _ in range(2_000 // 94)]}, separators=(",", ":"))
    assert session.texts == [f"IMG_00000/c1/tool/meta.json: {text}"]
    old = old_group_prompt(run, session, 2, 32)
    before, after = old.rsplit("\nRound ", 1)
    assert run.prompt(session, 2, 32) == before + f"\nInspected JSON {session.texts[0]}\nRound " + after
    run, session, items = inspect_run(tmp_path / "big", n_images=5, n_calls=2, n_json=1, json_chars=300_000)
    run.inspect(session, items, 0)
    import re
    head = session.texts[0].split(": ", 1)[0]
    assert re.fullmatch(r"IMG_00000/c1/tool/meta.json \(first 200000 of \d+ characters\)", head)


def test_the_last_line_limit_cuts_the_inspected_json_then_refuses_before_the_backend(tmp_path, monkeypatch):
    run, session, items = inspect_run(tmp_path, n_images=20, n_calls=2, n_json=4, json_chars=100_000)
    run.inspect(session, items, 0)
    uncut = len(run.prompt(session, 1, 32))
    monkeypatch.setattr(agent, "PROMPT_CHAR_LIMIT", uncut - 60_000)
    cut = run.prompt(session, 1, 32)
    assert len(cut) <= agent.PROMPT_CHAR_LIMIT and session.prompt_chars == len(cut)
    assert cut.count("cut to fit the prompt limit") == 4
    monkeypatch.setattr(agent, "PROMPT_CHAR_LIMIT", 5_000)
    with pytest.raises(RuntimeError, match="exceeds the harness limit 5000") as error:
        run.prompt(session, 1, 32)
    assert "'facts':" in str(error.value) and "'table':" in str(error.value) and "'inspected_json':" in str(error.value)


def test_a_prompt_above_the_limit_never_reaches_the_codex_cli(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=2)
    codex = FakeCodex({"group": [d("finish", {"all": "c0"})]})
    monkeypatch.setattr(codex_backend, "invoke", codex)
    monkeypatch.setattr(agent, "PROMPT_CHAR_LIMIT", 2_000)
    with pytest.raises(RuntimeError, match="exceeds the harness limit 2000"):
        agent.run_stage("segment", prep, run, config)
    assert codex.prompts == {}


def test_previous_decisions_are_echoed_within_bounds_and_small_ones_unchanged():
    small = [{"round": k, "action": "run", "request_json": "{}", "reason": "r", "outcome": "executed", "message": "ok"}
             for k in range(5)]
    assert json.dumps(agent.StageRun.echo_decisions(small)) == json.dumps(small)
    huge = {"round": 5, "action": "run", "request_json": json.dumps({"per_image": {f"IMG_{i:05d}": {"d": 1}
                                                                                for i in range(1310)}} | {"p": "y" * 90_000}),
            "reason": "r", "outcome": "executed", "message": "ran 1310 call(s): " + str([f"IMG_{i:05d}/c1" for i in range(1310)])}
    echoed = agent.StageRun.echo_decisions([*small, huge])
    assert echoed[:5] == small and len(json.dumps(echoed[5])) < agent.DECISION_ECHO_LIMIT + 1_000
    assert "cut in this echo" in echoed[5]["request_json"] and echoed[5]["request_json"].startswith(huge["request_json"][:100])
    assert len(huge["request_json"]) > 90_000       # the decision itself is not changed
    many = [dict(huge, round=k) for k in range(32)]
    assert len(json.dumps(agent.StageRun.echo_decisions(many))) <= agent.DECISIONS_ECHO_BUDGET


def test_every_round_records_its_prompt_chars_and_decisions_stay_complete(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=2)
    long_reason = "because " * 6_000      # 48k characters
    scripts = {"group": [d("run", [{"images": "all", "params": {"diameter": 12}}], reason=long_reason),
                         d("handoff", "all")],
               "img0": [d("finish", {"all": "c1"})], "img1": [d("finish", {"all": "c1"})]}
    codex = FakeCodex(scripts)
    monkeypatch.setattr(codex_backend, "invoke", codex)
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert all(isinstance(r["prompt_chars"], int) for r in rounds) and len(rounds) == 4
    group = [r for r in rounds if r["session"] == "group"]
    assert [r["prompt_chars"] for r in group] == [len(p) for p in codex.prompts["group"]]
    for name in ("img0", "img1"):
        assert [r["prompt_chars"] for r in rounds if r["session"] == f"image:{name}"] == [len(codex.prompts[name][0])]
    assert group[0]["reason"] == long_reason and "cut in this echo" in codex.prompts["group"][1]
    assert long_reason not in codex.prompts["group"][1]


def test_worst_case_sizes_of_the_prompt_parts_for_1310_images_8_calls_7_inspected_json(tmp_path):
    run, session, items = inspect_run(tmp_path, n_calls=8, facts_chars=30_000)
    run.calls = {image: {**records} for image, records in run.calls.items()}
    for image in run.images[::3]:                     # the flagged rows fill their budget
        run.calls[image]["c3"] = livecell_record(3, image, "error")
    run.inspect(session, items, 0)
    session.decisions = [{"round": k, "action": "run", "request_json": "x" * 60_000, "reason": "r", "outcome": "executed",
                          "message": "m" * 30_000} for k in range(8)]
    prompt = run.prompt(session, 7, 32)
    table = len(run.table_text(run.images)[0])
    decisions = len(json.dumps(run.echo_decisions(session.decisions)))
    inspected = sum(len(t) for t in session.texts)
    print(f"PARTS table={table} inspected_json={inspected} decisions={decisions} facts=30000 total={len(prompt)}")
    assert len(prompt) < agent.PROMPT_CHAR_LIMIT < CODEX_INPUT_LIMIT
