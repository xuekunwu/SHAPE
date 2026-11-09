"""The claude and biomni agent arms: Codex's stage protocol with another backend. CPU-only, with a
fake `claude` CLI and a fake Biomni driver run as real subprocesses; the driver's own pieces run on fake Biomni
modules, and only the check against the installed Biomni needs the real package."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
import types

import pytest

from benchmarks.full_CPJUMP1.harness import cli
from shape.agent.arms import BACKENDS, biomni as biomni_backend, biomni_driver, claude as claude_backend, codex as codex_backend, protocol
from benchmarks.full_CPJUMP1.harness import codex_arm as agent
from benchmarks.full_CPJUMP1.harness.data import read_json, write_json

from .test_codex import FakeCodex, d, gateway, image_stage, png   # noqa: F401  (pytest fixtures)

KEY = "offline-dummy-key-not-a-credential"


def result(decision=None, **updates):
    """A `claude -p --output-format json` result as the VM's 2.1.296 writes it."""
    choice = decision if decision is not None else d("finish", {"all": "c0"}, "observed quality")
    record = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "result": json.dumps(choice),
              "structured_output": choice, "permission_denials": [], "total_cost_usd": 0.01,
              "usage": {"input_tokens": 10, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 2,
                        "output_tokens": 5},
              "modelUsage": {claude_backend.MODEL: {"inputTokens": 10, "outputTokens": 5}}}
    record.update(updates)
    return record


@pytest.fixture
def claude_env(tmp_path, monkeypatch, gateway):
    """The gateway variables, an absolute api.env and no project instructions above the OS temporary directory (the
    Windows test home has a .claude; the check itself has its own test)."""
    api = tmp_path / "api.env"
    api.write_text(f"export LLM_API_KEY={KEY}\nexport LLM_BASE_URL={codex_backend.BASE_URL}\n", encoding="utf-8")
    monkeypatch.setenv("SHAPE_API_ENV_FILE", str(api))
    monkeypatch.setattr(claude_backend, "INSTRUCTION_FILES", ("CLAUDE.md", "CLAUDE.local.md", ".claude-cpj-test"))
    return api


# ---------------------------------------------------------------- the backend registry and the shared protocol

def test_every_agent_arm_has_a_backend_with_its_model():
    assert set(BACKENDS) == set(cli.AGENT_ARMS) == set(agent.ARM_NAMES)
    for arm in cli.AGENT_ARMS:
        module = agent.backend(arm)
        assert module.MODEL == cli.ARM_MODELS[arm]
        assert all(callable(getattr(module, name)) for name in ("route", "setup", "invoke", "preflight"))
    assert agent.backend("codex") is codex_backend and codex_backend.POLICY == "codex-cli-native-tools-v2"
    assert len({agent.backend(arm).POLICY for arm in cli.AGENT_ARMS}) == 3
    with pytest.raises(ValueError, match="unknown agent arm"):
        agent.backend("shape")


@pytest.mark.parametrize("arm,model", [("claude", "gpt-5.6-luna"), ("biomni", "claude-sonnet-5"),
                                       ("codex", "claude-sonnet-5")])
def test_each_backend_routes_only_its_own_model(claude_env, arm, model):
    with pytest.raises(ValueError, match="requires"):
        agent.backend(arm).route(model)


@pytest.mark.parametrize("arm", ["claude", "biomni"])
@pytest.mark.parametrize("variable", ["LLM_API_BASE", "LLM_BASE_URL"])
def test_new_backends_refuse_another_gateway_route(claude_env, monkeypatch, arm, variable):
    monkeypatch.setenv(variable, "https://elsewhere.invalid/v1")
    with pytest.raises(ValueError, match="Stanford Gateway"):
        agent.backend(arm).route(cli.ARM_MODELS[arm])


def test_claude_route_needs_an_absolute_api_environment_file(claude_env, monkeypatch):
    for value in ("", "api.env", str(claude_env) + ".missing"):
        monkeypatch.setenv("SHAPE_API_ENV_FILE", value)
        with pytest.raises(RuntimeError, match="SHAPE_API_ENV_FILE"):
            claude_backend.route(claude_backend.MODEL)


def test_the_prompt_names_the_arm_and_is_otherwise_codexs(image_stage, claude_env, monkeypatch, tmp_path):
    """The first line names the arm (byte-identical for Codex); every other line of every stage prompt is the same."""
    prep, run, config, rows, waves = image_stage("crop", n=2)
    facts = {}
    for arm in cli.AGENT_ARMS:
        stage = agent.StageRun("crop", prep, tmp_path / f"run_{arm}", {**config, "arm": arm,
                                                                        "model": cli.ARM_MODELS[arm]})
        # The remaining seconds are the only time-dependent text.
        facts[arm] = re.sub(r"\d+ seconds remain", "N seconds remain", stage.facts(32)).split("\n")
    assert facts["codex"][0] == "You are the Codex arm of the CPJUMP1 benchmark, stage crop, tool Single_Cell_Cropper_Tool."
    assert facts["claude"][0] == facts["codex"][0].replace("the Codex arm", "the Claude Code arm")
    assert facts["biomni"][0] == facts["codex"][0].replace("the Codex arm", "the Biomni arm")
    assert facts["claude"][1:] == facts["codex"][1:] == facts["biomni"][1:]
    assert codex_backend.SCHEMA["additionalProperties"] is False


# ---------------------------------------------------------------- the Claude backend, its pieces

def test_claude_command_is_the_audited_headless_cli():
    cmd = claude_backend.command(claude_backend.MODEL, Path("settings.json"))
    assert cmd[:4] == ["claude", "-p", "--output-format", "json"]
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == codex_backend.SCHEMA
    assert " " not in cmd[cmd.index("--json-schema") + 1]
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5"
    assert cmd[cmd.index("--settings") + 1] == "settings.json"
    assert cmd[cmd.index("--tools") + 1] == "Read" and cmd[cmd.index("--max-turns") + 1] == "24"
    assert "--strict-mcp-config" in cmd and "--dangerously-skip-permissions" not in cmd
    assert not any("prompt" in part for part in cmd)          # the prompt goes on stdin


def test_claude_environment_has_no_key_and_every_alias_set_without_mutating_parent(tmp_path, claude_env, monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "PYTHONPATH", "CLAUDE_CONFIG_DIR"):
        monkeypatch.setenv(key, "PRIVATE_SENTINEL")
    before = dict(os.environ)
    config_dir = tmp_path / "workspace/config"
    env = claude_backend._env(config_dir, tmp_path / "workspace", run=tmp_path / "run")
    assert dict(os.environ) == before
    assert "LLM_API_KEY" not in env and "ANTHROPIC_API_KEY" not in env and "SHAPE_API_ENV_FILE" not in env
    assert "PRIVATE_SENTINEL" not in env.values() and KEY not in json.dumps(env)
    assert env["CLAUDE_CONFIG_DIR"] == env["HOME"] == str(config_dir)
    assert env["ANTHROPIC_BASE_URL"] == "https://llm-gateway.example.edu"
    assert all(env[alias] == "claude-sonnet-5" for alias in claude_backend.MODEL_ALIASES)
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == env["DISABLE_TELEMETRY"] == "1"
    first = env["PATH"].split(os.pathsep)[:2]
    assert first == [str(Path.home() / "shape/cache/envs/claude-code-2.1.296/bin"),
                     str(Path.home() / "shape/cache/envs/node-v22.11.0/bin")]


def test_claude_setup_writes_a_key_helper_not_a_key(tmp_path, claude_env):
    settings = claude_backend.setup(tmp_path / "auth", claude_backend.MODEL)
    helper = settings.with_name("key_helper.sh")
    assert sorted(p.name for p in settings.parent.iterdir()) == ["key_helper.sh", "settings.json"]
    assert read_json(settings) == {"apiKeyHelper": shlex.quote(str(helper))}
    text = helper.read_text(encoding="ascii")
    assert KEY not in text and KEY not in settings.read_text(encoding="utf-8")
    assert shlex.quote(str(claude_env)) in text and codex_backend.BASE_URL in text and b"\r" not in helper.read_bytes()
    with pytest.raises(ValueError, match="empty"):
        claude_backend.setup(settings.parent, claude_backend.MODEL)
    claude_backend._check_settings(settings)
    write_json(settings, {"apiKeyHelper": "echo key"})
    with pytest.raises(ValueError, match="configuration changed"):
        claude_backend._check_settings(settings)


@pytest.mark.skipif(os.name != "posix", reason="the key helper is a bash script run by the CLI on Linux")
def test_claude_key_helper_prints_only_the_key_and_checks_the_gateway(tmp_path, claude_env):
    import subprocess
    helper = claude_backend.setup(tmp_path / "auth", claude_backend.MODEL).with_name("key_helper.sh")
    clean = {"PATH": os.environ["PATH"]}
    assert subprocess.run([str(helper)], env=clean, capture_output=True, text=True).stdout == KEY
    claude_env.write_text(f"export LLM_API_KEY={KEY}\nexport LLM_BASE_URL=https://elsewhere.invalid/v1\n")
    failed = subprocess.run([str(helper)], env=clean, capture_output=True, text=True)
    assert failed.returncode == 1 and KEY not in failed.stdout + failed.stderr


def test_claude_round_settings_add_only_the_read_guard_and_deny_rules(tmp_path, claude_env):
    settings = claude_backend.setup(tmp_path / "auth", claude_backend.MODEL)
    roots = tmp_path / "roots.json"
    record = claude_backend._round_settings(settings, roots)
    assert record["apiKeyHelper"] == read_json(settings)["apiKeyHelper"]
    assert set(record) == {"apiKeyHelper", "hooks", "permissions"}
    deny = record["permissions"]["deny"]                  # @-mentions bypass the hook; deny rules stop them
    assert "Read(//home/**)" in deny and "Read(//etc/**)" in deny
    workspace = roots.resolve().parent.as_posix()
    assert f"Read(/{workspace}/config/**)" in deny and f"Read(/{workspace}/read_roots.json)" in deny
    command = record["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert shlex.split(command) == [sys.executable, "-I", str(claude_backend.READ_GUARD), str(roots)]


def write_trace(path, record):
    path.write_text(json.dumps(record) if isinstance(record, dict) else record, encoding="utf-8")
    return path


def test_claude_structured_output_is_the_decision_and_usage_sums_all_four_counts(tmp_path):
    choice, usage = claude_backend.response(write_trace(tmp_path / "r.json", result()), claude_backend.MODEL)
    assert choice == d("finish", {"all": "c0"}, "observed quality")
    assert usage["total_tokens"] == 10 + 3 + 2 + 5
    assert usage["usage"] == {"input_tokens": 10, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 2,
                              "output_tokens": 5}
    assert usage["requested_model"] == "claude-sonnet-5" and usage["response_model_attested"] is False
    assert usage["num_turns"] == 2 and usage["permission_denials"] == [] and "result" not in usage


@pytest.mark.parametrize("updates", [
    {"subtype": "error_max_turns", "is_error": True}, {"subtype": "error_max_structured_output_retries"},
    {"structured_output": None}, {"structured_output": {"action": "finish", "reason": "no payload"}},
    {"structured_output": {**d("finish", {}), "extra": "x"}}, {"structured_output": d("execute", {})},
    {"structured_output": {**d("finish", {}), "reason": " "}},
    {"structured_output": {**d("finish", {}), "request_json": "not json"}},
    {"structured_output": {**d("finish", {}), "request_json": 1}},
])
def test_claude_error_results_and_invalid_replies_are_failed_rounds_with_their_tokens(tmp_path, updates):
    with pytest.raises(agent.MalformedDecision) as caught:
        claude_backend.response(write_trace(tmp_path / "r.json", result(**updates)), claude_backend.MODEL)
    assert caught.value.tokens == 20


@pytest.mark.parametrize("updates", [
    {"is_error": True}, {"subtype": "error_during_execution", "is_error": True}, {"api_error_status": 529},
])
def test_claude_api_errors_fail_the_arm_as_codex_error_events_do(tmp_path, updates):
    """An error result that is not the agent's own (max turns, structured-output retries) is an API or gateway error:
    the arm fails, as a Codex error event fails it (user, 2026-10-12)."""
    with pytest.raises(RuntimeError, match="API error"):
        claude_backend.response(write_trace(tmp_path / "r.json", result(**updates)), claude_backend.MODEL)


def test_claude_total_counts_every_call_of_the_model(tmp_path):
    record = result(modelUsage={claude_backend.MODEL: {"inputTokens": 30, "outputTokens": 9, "cacheReadInputTokens": 1,
                                                       "cacheCreationInputTokens": 0}})
    _, usage = claude_backend.response(write_trace(tmp_path / "r.json", record), claude_backend.MODEL)
    assert usage["total_tokens"] == 40                    # a background call of the model counted (usage alone: 20)


@pytest.mark.parametrize("record", [
    "not json", json.dumps([1]), json.dumps({"type": "system"}),
    result(usage={}), result(usage={"input_tokens": 0, "output_tokens": 0}),
    result(usage={"input_tokens": True, "output_tokens": 2}), result(usage={"input_tokens": -1, "output_tokens": 9}),
    result(usage={"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 1.5}),
    result(modelUsage={}), result(modelUsage={"claude-haiku-4": {}}),
    result(modelUsage={claude_backend.MODEL: {}, "claude-haiku-4": {}}),
    result(is_error=True, usage={"input_tokens": 0, "output_tokens": 0}),     # no model reply: route, auth, gateway
])
def test_claude_results_without_valid_usage_or_model_are_basic_failures(tmp_path, record):
    with pytest.raises(RuntimeError):
        claude_backend.response(write_trace(tmp_path / "r.json", record), claude_backend.MODEL)


# ---------------------------------------------------------------- the Claude backend through a fake CLI

FAKE_CLAUDE = r'''import json, os, pathlib, shlex, subprocess, sys
here = pathlib.Path(__file__).parent
state = json.loads((here / "claude_fake.json").read_text(encoding="utf-8"))
args = sys.argv[1:]
mode = state.get("mode", "success")
if args == ["--version"]:
    print(state.get("version", "2.1.296") + " (Claude Code)")
    sys.exit(0)
if args == ["--help"]:
    print("Usage: claude [options]  -p, --print  --output-format  --json-schema  --model  --settings  --tools  "
          "--strict-mcp-config")
    sys.exit(0)
env = os.environ
for name in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "SHAPE_API_ENV_FILE", "PYTHONPATH",
             "OPENAI_API_KEY"):
    assert name not in env, name
for name in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
             "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"):
    assert env[name] == "claude-sonnet-5", name
assert env["ANTHROPIC_BASE_URL"] == "https://llm-gateway.example.edu"
assert env["DISABLE_TELEMETRY"] == "1" and env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
config = pathlib.Path(env["CLAUDE_CONFIG_DIR"])
assert pathlib.Path(env["HOME"]) == config and [p.name for p in config.iterdir()] == ["settings.json"]
cwd = pathlib.Path.cwd()
assert cwd.name.startswith("cpj-claude-") and config.parent == cwd
assert args[:3] == ["-p", "--output-format", "json"] and args[args.index("--model") + 1] == "claude-sonnet-5"
assert args[args.index("--tools") + 1] == "Read" and args[args.index("--max-turns") + 1] == "24"
assert "--strict-mcp-config" in args
schema = json.loads(args[args.index("--json-schema") + 1])
assert schema["additionalProperties"] is False and schema["required"] == ["action", "request_json", "reason"]
settings_path = pathlib.Path(args[args.index("--settings") + 1])
assert settings_path == config / "settings.json"
settings = json.loads(settings_path.read_text(encoding="utf-8"))
assert "key_helper.sh" in settings["apiKeyHelper"]
prompt = sys.stdin.read()
evidence = sorted(p.name for p in cwd.glob("evidence_*.png"))
if evidence:
    assert prompt.splitlines()[-1] == ("Attached images are the files " + ", ".join(evidence)
                                       + " in your working directory; view them with the Read tool.")
else:
    assert "evidence_" not in prompt
# The Read guard: the round's evidence files only, never the settings, the key helper or another file.
hook = shlex.split(settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"])
def read(path, tool="Read"):
    event = json.dumps({"tool_name": tool, "tool_input": {"file_path": str(path)}})
    return subprocess.run(hook, input=event, text=True, capture_output=True).returncode
assert all(read(cwd / name) == 0 for name in evidence)
assert read(settings_path) == 2 and read(cwd / "read_roots.json") == 2
assert read(cwd / evidence[0], "Bash") == 2 if evidence else True
with open(here / "claude_prompts.jsonl", "a", encoding="utf-8") as log:
    log.write(json.dumps({"prompt": prompt, "evidence": [(cwd / n).read_bytes().hex() for n in evidence]}) + "\n")
if mode == "crash":
    print("fatal: not logged in", file=sys.stderr)
    sys.exit(1)
if "Connectivity check" in prompt:
    from PIL import Image
    green = Image.open(cwd / evidence[0]).convert("RGB").getpixel((0, 0)) == (0, 255, 0)
    choice = {"action": "finish", "request_json": "{}", "reason": "green" if green else "unknown"}
else:
    choice = state["decisions"].pop(0)
    (here / "claude_fake.json").write_text(json.dumps(state), encoding="utf-8")
usage = {"input_tokens": 10, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 2, "output_tokens": 5}
record = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "result": json.dumps(choice),
          "structured_output": choice, "permission_denials": [], "total_cost_usd": 0.01, "usage": usage,
          "modelUsage": {"claude-sonnet-5": {"inputTokens": 10, "outputTokens": 5}}}
if mode == "max_turns":
    record.update(subtype="error_max_turns", is_error=True, structured_output=None)
print(json.dumps(record))
print("warning: key " + state.get("leak", "none"), file=sys.stderr)
sys.exit(1 if mode == "max_turns" else 0)
'''


@pytest.fixture
def fake_claude(tmp_path, monkeypatch, claude_env):
    """A fake `claude` run as a real subprocess; returns (configure(mode, decisions, version), the prompt log)."""
    original = protocol._process
    script = tmp_path / "fake" / "claude.py"
    script.parent.mkdir()
    script.write_text(FAKE_CLAUDE, encoding="utf-8")
    calls = []

    def execute(cmd, *args, **kwargs):
        assert cmd[0] == "claude"
        calls.append(cmd)
        return original([sys.executable, str(script), *cmd[1:]], *args, **kwargs)

    def configure(mode="success", decisions=(), version="2.1.296", leak="none"):
        write_json(script.with_name("claude_fake.json"), {"mode": mode, "decisions": list(decisions),
                                                          "version": version, "leak": leak})

    monkeypatch.setattr(protocol, "_process", execute)
    configure()
    configure.calls = calls
    configure.prompts = lambda: [json.loads(s) for s in
                                 script.with_name("claude_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    return configure


def no_key_in(root):
    for path in Path(root).rglob("*"):
        if path.is_file():
            assert KEY.encode() not in path.read_bytes(), path


def test_claude_preflight_runs_the_version_help_and_green_image_checks(tmp_path, fake_claude):
    record = claude_backend.preflight(tmp_path / "claude_preflight", claude_backend.MODEL)
    assert record["image_preflight"] == "passed" and record["claude_version"] == "2.1.296"
    assert record["preflight_tokens"] == 20 and record["model"] == "claude-sonnet-5"
    assert record["key_variable"] == "apiKeyHelper:LLM_API_KEY" and record["policy"] == claude_backend.POLICY
    assert [c[1] for c in fake_claude.calls[:2]] == ["--version", "--help"]
    assert (tmp_path / "claude_preflight/traces/round_000.json").is_file()
    no_key_in(tmp_path / "claude_preflight")


@pytest.mark.parametrize("version", ["2.1.295", "2.1.297", ""])
def test_claude_unaudited_version_is_refused_before_inference(tmp_path, fake_claude, version):
    fake_claude(version=version)
    with pytest.raises(RuntimeError, match="audited CLI version"):
        claude_backend.preflight(tmp_path / "preflight", claude_backend.MODEL)
    assert [c[1] for c in fake_claude.calls] == ["--version"]


def test_claude_round_copies_images_adds_the_read_line_and_keeps_its_trace(tmp_path, fake_claude):
    root = tmp_path / "run"
    settings = claude_backend.setup(root / "auth", claude_backend.MODEL)
    images = [png(tmp_path / "a.png", 10), png(tmp_path / "b.png", 200)]
    # A CLI that echoes the key on stderr is redacted.
    fake_claude(decisions=[d("run", [{"images": "all", "params": {}}])], leak=KEY)
    choice, usage, session = claude_backend.invoke(root, "feedback", claude_backend.MODEL, settings,
                                                   time.time() + 60, 0, images=images, marker=root)
    assert choice["action"] == "run" and usage["total_tokens"] == 20 and session is None
    assert usage["claude_version"] == "2.1.296"
    logged = fake_claude.prompts()[-1]
    assert logged["prompt"].startswith("feedback\n")
    assert logged["evidence"] == [p.read_bytes().hex() for p in images]
    assert read_json(root / "traces/round_000.json")["structured_output"] == choice
    assert "[REDACTED]" in (root / "traces/round_000.stderr").read_text(encoding="utf-8")
    no_key_in(root)
    with pytest.raises(ValueError, match="already exists"):
        claude_backend.invoke(root, "again", claude_backend.MODEL, settings, time.time() + 60, 0)


def test_claude_error_result_is_a_failed_round_and_a_crash_a_basic_failure(tmp_path, fake_claude):
    root = tmp_path / "run"
    settings = claude_backend.setup(root / "auth", claude_backend.MODEL)
    fake_claude("max_turns", [d("finish", {})])
    with pytest.raises(agent.MalformedDecision, match="error result") as caught:
        claude_backend.invoke(root, "feedback", claude_backend.MODEL, settings, time.time() + 60, 0)
    assert caught.value.tokens == 20
    assert not list((root / "traces").glob("round_*")) and (root / "traces/failed_round_000.json").is_file()
    assert claude_backend.trace_tokens(root / "traces/failed_round_000.json", claude_backend.MODEL) == 20
    fake_claude("crash")
    with pytest.raises(RuntimeError, match="Claude Code CLI failed"):
        claude_backend.invoke(root, "feedback", claude_backend.MODEL, settings, time.time() + 60, 1)


def test_claude_refuses_inherited_project_instructions(tmp_path, fake_claude, monkeypatch):
    root = tmp_path / "run"
    settings = claude_backend.setup(root / "auth", claude_backend.MODEL)
    temporary = tmp_path / "os_temp" / "nested"
    temporary.mkdir(parents=True)
    (tmp_path / "os_temp" / "CLAUDE.md").write_text("inherited", encoding="utf-8")
    monkeypatch.setattr(claude_backend, "TEMP_ROOT", str(temporary))
    with pytest.raises(ValueError, match="inherited Claude instructions"):
        claude_backend.invoke(root, "feedback", claude_backend.MODEL, settings, time.time() + 60, 0)
    assert fake_claude.calls == []


def test_claude_tampered_settings_are_refused(tmp_path, fake_claude):
    root = tmp_path / "run"
    settings = claude_backend.setup(root / "auth", claude_backend.MODEL)
    settings.with_name("key_helper.sh").write_text("#!/usr/bin/env bash\necho other\n", encoding="ascii")
    with pytest.raises(ValueError, match="configuration changed"):
        claude_backend.invoke(root, "feedback", claude_backend.MODEL, settings, time.time() + 60, 0)


def native_configs(run, stage):
    return [read_json(p) for p in sorted((run / "agent" / stage).rglob("CONFIG.json"))]


def test_claude_segment_stage_end_to_end_through_the_fake_cli(image_stage, fake_claude):
    prep, run, config, rows, waves = image_stage("segment", n=2)
    config.update(arm="claude", model=claude_backend.MODEL)
    fake_claude(decisions=[d("inspect", [{"image": "img0", "call": "input"}]),
                           d("run", [{"images": "all", "params": {"diameter": 40}}]),
                           d("finish", {"all": "c1"})])
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["attached", "executed", "finished"]
    prompts = fake_claude.prompts()
    assert all(p["prompt"].startswith("You are the Claude Code arm of the CPJUMP1 benchmark, stage segment")
               for p in prompts)
    assert len(prompts[1]["evidence"]) == 1 and prompts[0]["evidence"] == prompts[2]["evidence"] == []
    assert "Attached images: image 1: img0 input" in prompts[1]["prompt"]
    selected = read_json(run / "agent/segment/SELECTED.json")
    assert selected["policy"] == claude_backend.POLICY and selected["total_tokens"] == 60
    assert {k: v["call"] for k, v in selected["deliveries"].items()} == {"img0": "c1", "img1": "c1"}
    assert [r["call"] for r in read_json(run / "segmentation.json")] == ["c1", "c1"]
    configs = native_configs(run, "segment")
    assert len(configs) == 2 and all(c["arm"] == "codex" and c["agent_arm"] == "claude" for c in configs)
    assert config["arm"] == "claude"                                          # the arm's own config keeps its name
    traces = sorted(p.name for p in (run / "agent/segment/group/traces").iterdir())
    assert traces == [f"round_00{i}.{s}" for i in range(3) for s in ("json", "stderr")]
    no_key_in(run)


# ---------------------------------------------------------------- the Biomni driver, its pieces (no Biomni needed)

SCHEMA = codex_backend.SCHEMA


def test_driver_instruction_and_refusal_are_the_specs():
    line = biomni_driver.instruction(SCHEMA)
    assert line.startswith("Return your decision as one JSON object with exactly the keys action, request_json, "
                           "reason (schema: ")
    assert line.endswith(") inside <solution></solution>.")
    assert json.loads(line[line.index("schema: ") + 8:line.index(") inside")]) == SCHEMA
    assert biomni_driver.REFUSAL == "Code execution is not available in this benchmark. Give your decision inside solution tags."
    assert "<solution>" not in biomni_driver.REFUSAL            # a refusal is never read as a malformed solution
    assert biomni_driver.parse_solution(["<SOLUTION>" + json.dumps(d("finish", {})) + "</SOLUTION>"], SCHEMA) == {
        "decision": d("finish", {})}                          # A1 ends its loop on any case
    assert biomni_driver.RECURSION_LIMIT == 30


def solution(value):
    return f"Thinking.\n<solution>{value if isinstance(value, str) else json.dumps(value)}</solution>"


@pytest.mark.parametrize("texts,expected", [
    ([solution(d("finish", {"all": "c0"}))], {"decision": d("finish", {"all": "c0"})}),
    ([solution(d("run", [])), "more", solution(d("finish", {}))], {"decision": d("finish", {})}),
    ([solution("```json\n" + json.dumps(d("inspect", [])) + "\n```")], {"decision": d("inspect", [])}),
    (["<solution>a</solution> then " + solution(d("handoff", "all"))], {"decision": d("handoff", "all")}),
])
def test_driver_takes_the_last_solution_as_the_decision(texts, expected):
    assert biomni_driver.parse_solution(texts, SCHEMA) == expected


@pytest.mark.parametrize("texts,why", [
    (["no tags at all"], "no <solution>"), ([solution("{not json")], "not one JSON"),
    ([solution([1, 2])], "keys must be exactly"), ([solution({"action": "finish", "reason": "x"})], "keys"),
    ([solution({**d("finish", {}), "extra": 1})], "keys"), ([solution(d("execute", {}))], "action must be one of"),
    ([], "no <solution>"),
])
def test_driver_reports_why_a_reply_is_not_a_decision(texts, why):
    outcome = biomni_driver.parse_solution(texts, SCHEMA)
    assert set(outcome) == {"malformed"} and why in outcome["malformed"]


def test_driver_text_redaction_and_usage():
    content = [{"type": "text", "text": f"key {KEY}"}, {"type": "image_url", "image_url": {"url": "data:..."}}]
    assert biomni_driver.text_of(content) == f"key {KEY}\n[image]"
    assert biomni_driver.redact(biomni_driver.text_of(content), KEY) == "key [REDACTED]\n[image]"
    assert biomni_driver.redact("text", None) == "text"
    messages = [types.SimpleNamespace(usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}),
                types.SimpleNamespace(usage_metadata=None),
                types.SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})]
    assert biomni_driver.usage_of(messages) == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
    with pytest.raises(ValueError):
        biomni_driver.usage_of([types.SimpleNamespace(usage_metadata={"input_tokens": -1})])


class Message:
    def __init__(self, kind, content, usage=None):
        self.type, self.content, self.usage_metadata = kind, content, usage


class FakeClient:
    def __init__(self, retries):
        self.max_retries = retries
        self.chat = types.SimpleNamespace(completions=("completions", retries))

    def with_options(self, max_retries):
        return FakeClient(max_retries)


@pytest.fixture
def fake_biomni(monkeypatch):
    """biomni.agent.a1, langchain_core.messages and langgraph.errors as fakes: an A1 that prints its masked key and
    configuration while constructed, and whose graph asks once to execute code before giving a solution."""
    seen = {}

    class GraphRecursionError(RuntimeError):
        pass

    class App:
        """A1's graph: like Biomni 0.0.8, each reply in the state is a rebuilt message without usage; the usage
        reaches only the LLM's callbacks."""

        def __init__(self, llm):
            self.llm = llm

        def generated(self, kind, content, usage):
            for callback in self.llm.callbacks or []:
                callback.on_llm_end(types.SimpleNamespace(
                    generations=[[types.SimpleNamespace(message=Message(kind, content, usage))]]))
            return Message(kind, content)

        def stream(self, inputs, stream_mode=None, config=None):
            seen.update(inputs=inputs, stream_mode=stream_mode, config=config)
            ask = self.generated("ai", "<execute>print(1)</execute>", {"input_tokens": 50, "output_tokens": 5,
                                                                     "total_tokens": 55})
            yield {"messages": [*inputs["messages"], ask]}
            observation = Message("human", f"<observation>{module.run_with_timeout(print, ['x'], timeout=5)}"
                                           "</observation>")
            if seen.get("mode") == "limit":
                raise GraphRecursionError("recursion limit")
            reply = self.generated("ai", f"leaked {KEY} " + solution(d("finish", {"all": "c0"}, "green")),
                                   {"input_tokens": 60, "output_tokens": 7, "total_tokens": 67})
            yield {"messages": [*inputs["messages"], ask, observation, reply]}

    class A1:
        def __init__(self, **kwargs):
            print(f"API key: {KEY[:4]}****  config: {kwargs['llm']}")
            print("Biomni configuration dump", file=sys.stderr)
            os.write(1, b"native stdout\n")
            seen["kwargs"] = kwargs
            self.module2api, self.library_content_dict, self.data_lake_dict = {"t": 1}, {"l": 1}, {"d": 1}
            self.llm = types.SimpleNamespace(callbacks=None, max_tokens=8192, max_retries=None,
                                             root_client=FakeClient(2), root_async_client=None)
            self.app = App(self.llm)
            seen["llm"] = self.llm

        def configure(self):
            seen["configured"] = (self.module2api, self.library_content_dict, self.data_lake_dict)

    module = types.ModuleType("biomni.agent.a1")
    module.A1, module.run_with_timeout = A1, lambda *a, **k: "ran code"
    messages = types.ModuleType("langchain_core.messages")
    messages.HumanMessage = lambda content: Message("human", content)
    errors = types.ModuleType("langgraph.errors")
    errors.GraphRecursionError = GraphRecursionError
    callbacks = types.ModuleType("langchain_core.callbacks")
    callbacks.BaseCallbackHandler = type("BaseCallbackHandler", (), {})
    for name, value in {"biomni": types.ModuleType("biomni"), "biomni.agent": types.ModuleType("biomni.agent"),
                        "biomni.agent.a1": module, "langchain_core": types.ModuleType("langchain_core"),
                        "langchain_core.messages": messages, "langchain_core.callbacks": callbacks,
                        "langgraph": types.ModuleType("langgraph"),
                        "langgraph.errors": errors}.items():
        monkeypatch.setitem(sys.modules, name, value)
    sys.modules["biomni"].agent = sys.modules["biomni.agent"]
    sys.modules["biomni.agent"].a1 = module
    return seen


def request(tmp_path, images=()):
    path = tmp_path / "request.json"
    write_json(path, {"prompt": "You are the Biomni arm.", "images": [str(p) for p in images], "schema": SCHEMA,
                      "model": "gpt-5.6-luna", "base_url": codex_backend.BASE_URL, "timeout_seconds": 600})
    return path


def test_driver_builds_a_decision_only_a1_silently_and_refuses_execution(tmp_path, monkeypatch, capfd, fake_biomni):
    monkeypatch.setenv("LLM_API_KEY", KEY)
    image = png(tmp_path / "evidence_000.png")
    biomni_driver.main([str(request(tmp_path, [image])), str(tmp_path / "result.json")])
    captured = capfd.readouterr()
    assert "API key" not in captured.out + captured.err and "native stdout" not in captured.out
    assert "configuration dump" not in captured.err
    kwargs = fake_biomni["kwargs"]
    assert kwargs == {"path": str(tmp_path / "biomni"), "llm": "gpt-5.6-luna", "source": "Custom",
                      "base_url": "https://llm-gateway.example.edu/v1", "api_key": KEY, "use_tool_retriever": False,
                      "expected_data_lake_files": [], "commercial_mode": False, "timeout_seconds": 600}
    assert fake_biomni["configured"] == ({}, {}, {})
    llm = fake_biomni["llm"]                              # as Codex: no output cap, no retries (user, 2026-10-12)
    assert llm.max_tokens is None and llm.max_retries == 0 and llm.root_client.max_retries == 0
    assert llm.client == ("completions", 0)
    assert fake_biomni["config"] == {"recursion_limit": 30, "configurable": {"thread_id": "cpjump1"}} and fake_biomni["inputs"]["next_step"] is None
    content = fake_biomni["inputs"]["messages"][0].content
    assert content[0] == {"type": "text", "text": "You are the Biomni arm.\n" + biomni_driver.instruction(SCHEMA)}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,iVBOR")
    out = read_json(tmp_path / "result.json")
    assert out["decision"] == d("finish", {"all": "c0"}, "green") and "malformed" not in out
    assert out["usage"] == {"input_tokens": 110, "output_tokens": 12, "total_tokens": 122}
    assert out["steps"] == 2 and out["refused_executions"] == 1
    assert out["messages"][2] == {"role": "human", "content": f"<observation>{biomni_driver.REFUSAL}</observation>"}
    assert out["messages"][0]["content"].endswith("[image]")
    text = (tmp_path / "result.json").read_text(encoding="utf-8")
    assert KEY not in text and "[REDACTED]" in text and "API key" not in text


def test_driver_step_limit_is_a_malformed_reply_with_its_usage(tmp_path, monkeypatch, fake_biomni):
    monkeypatch.setenv("LLM_API_KEY", KEY)
    fake_biomni["mode"] = "limit"
    biomni_driver.main([str(request(tmp_path)), str(tmp_path / "result.json")])
    out = read_json(tmp_path / "result.json")
    assert "decision" not in out and "recursion_limit 30" in out["malformed"]
    assert out["usage"]["total_tokens"] == 55 and out["refused_executions"] == 1


@pytest.mark.skipif(importlib.util.find_spec("biomni") is None, reason="needs the real biomni package")
def test_installed_biomni_takes_the_drivers_construction_arguments():
    import inspect
    from importlib.metadata import version
    import biomni.agent.a1 as a1
    assert version("biomni") == biomni_backend.BIOMNI_VERSION
    parameters = inspect.signature(a1.A1.__init__).parameters
    for name in ("path", "llm", "source", "base_url", "api_key", "use_tool_retriever", "expected_data_lake_files",
                 "commercial_mode", "timeout_seconds"):
        assert name in parameters
    assert callable(a1.run_with_timeout) and callable(a1.A1.configure)


# ---------------------------------------------------------------- the Biomni backend through a fake driver

FAKE_DRIVER = r'''import json, os, pathlib, sys
assert sys.flags.isolated, "the driver runs with -I"
allowed = {"PATH", "HOME", "LANG", "LC_ALL", "LLM_API_KEY", "SYSTEMROOT", "WINDIR", "FULL_HCMA_RUN"}
assert set(k.upper() for k in os.environ) <= allowed, sorted(os.environ)
here = pathlib.Path(__file__).parent
state = json.loads((here / "biomni_fake.json").read_text(encoding="utf-8"))
request_path, result_path = map(pathlib.Path, sys.argv[1:3])
request = json.loads(request_path.read_text(encoding="utf-8"))
assert pathlib.Path(os.environ["HOME"]).parent == request_path.parent
assert request_path.parent.name.startswith("cpj-biomni-")
assert set(request) == {"prompt", "images", "schema", "model", "base_url", "timeout_seconds"}
assert request["model"] == "gpt-5.6-luna" and request["base_url"] == "https://llm-gateway.example.edu/v1"
assert request["schema"]["additionalProperties"] is False and 0 < request["timeout_seconds"] <= 600
for image in request["images"]:
    path = pathlib.Path(image)
    assert path.parent == request_path.parent and path.name.startswith("evidence_")
    assert path.read_bytes().startswith(bytes([137, 80, 78, 71]))
with open(here / "biomni_requests.jsonl", "a", encoding="utf-8") as log:
    log.write(json.dumps(request) + "\n")
mode = state.get("mode", "success")
if mode == "crash":
    print("Traceback: gateway refused", file=sys.stderr)
    sys.exit(3)
if "Connectivity check" in request["prompt"]:
    from PIL import Image
    green = Image.open(request["images"][0]).convert("RGB").getpixel((0, 0)) == (0, 255, 0)
    outcome = {"decision": {"action": "finish", "request_json": "{}", "reason": "green" if green else "unknown"}}
else:
    choice = state["decisions"].pop(0)                 # {"malformed": why}: this round's reply is not a decision
    outcome = choice if "malformed" in choice else {"decision": choice}
    (here / "biomni_fake.json").write_text(json.dumps(state), encoding="utf-8")
if mode == "malformed":
    outcome = {"malformed": "no <solution></solution> in the replies"}
key = os.environ["LLM_API_KEY"]
result = {**outcome, "usage": {"input_tokens": 30, "output_tokens": 4, "total_tokens": 34}, "steps": 1,
          "refused_executions": 0, "messages": [{"role": "ai", "content": "unredacted " + key}]}
result_path.write_text(json.dumps(result), encoding="utf-8")
print("biomni says " + key, file=sys.stderr)
'''


@pytest.fixture
def fake_driver(tmp_path, monkeypatch, gateway):
    driver = tmp_path / "fake_driver" / "biomni_driver.py"
    driver.parent.mkdir()
    driver.write_text(FAKE_DRIVER, encoding="utf-8")
    monkeypatch.setattr(biomni_backend, "DRIVER", driver)
    monkeypatch.setenv("FULL_CPJUMP1_BIOMNI_PYTHON", sys.executable)

    def configure(mode="success", decisions=()):
        write_json(driver.with_name("biomni_fake.json"), {"mode": mode, "decisions": list(decisions)})

    configure()
    configure.requests = lambda: [json.loads(s) for s in
                                  driver.with_name("biomni_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    return configure


def test_biomni_python_is_configurable_and_absolute(monkeypatch):
    monkeypatch.delenv("FULL_CPJUMP1_BIOMNI_PYTHON", raising=False)
    assert biomni_backend.python() == Path.home() / "shape/cache/envs/biomni-0.0.8/bin/python"
    monkeypatch.setenv("FULL_CPJUMP1_BIOMNI_PYTHON", "relative/python")
    with pytest.raises(ValueError, match="absolute"):
        biomni_backend.python()


def test_biomni_round_maps_the_driver_result_and_redacts_its_traces(tmp_path, fake_driver):
    root = tmp_path / "run"
    settings = biomni_backend.setup(root / "auth", biomni_backend.MODEL)
    assert KEY not in settings.read_text(encoding="utf-8")
    fake_driver(decisions=[d("run", [{"images": "all", "params": {}}])])
    image = png(tmp_path / "x.png")
    choice, usage, session = biomni_backend.invoke(root, "feedback", biomni_backend.MODEL, settings,
                                                   time.time() + 60, 0, images=[image], marker=root)
    assert choice["action"] == "run" and session is None
    assert usage["total_tokens"] == 34 and usage["steps"] == 1 and usage["refused_executions"] == 0
    assert fake_driver.requests()[-1]["prompt"] == "feedback"              # the driver adds its own line
    trace = read_json(root / "traces/round_000.json")
    assert trace["messages"] == [{"role": "ai", "content": "unredacted [REDACTED]"}]
    assert "[REDACTED]" in (root / "traces/round_000.stderr").read_text(encoding="utf-8")
    no_key_in(root)
    assert biomni_backend.trace_tokens(root / "traces/round_000.json", biomni_backend.MODEL) == 34


def test_biomni_malformed_reply_is_a_failed_round_and_a_driver_crash_a_basic_failure(tmp_path, fake_driver):
    root = tmp_path / "run"
    settings = biomni_backend.setup(root / "auth", biomni_backend.MODEL)
    fake_driver("malformed", [d("finish", {})])
    with pytest.raises(agent.MalformedDecision, match="no <solution>") as caught:
        biomni_backend.invoke(root, "feedback", biomni_backend.MODEL, settings, time.time() + 60, 0)
    assert caught.value.tokens == 34 and (root / "traces/failed_round_000.json").is_file()
    assert not list((root / "traces").glob("round_*"))
    assert biomni_backend.trace_tokens(root / "traces/failed_round_000.json", biomni_backend.MODEL) == 34
    fake_driver("crash")
    with pytest.raises(RuntimeError, match="Biomni driver failed"):
        biomni_backend.invoke(root, "feedback", biomni_backend.MODEL, settings, time.time() + 60, 1)
    write_json(settings, {**read_json(settings), "model": "other"})
    with pytest.raises(ValueError, match="configuration changed"):
        biomni_backend.invoke(root, "feedback", biomni_backend.MODEL, settings, time.time() + 60, 2)


@pytest.mark.parametrize("record", [
    {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 0}, "messages": [], "decision": {}},
    {"usage": {"input_tokens": 1, "output_tokens": 1}, "messages": [], "decision": {}},
    {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, "messages": []},
    {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, "messages": [], "decision": {},
     "malformed": "both"},
    [1, 2],
])
def test_biomni_incomplete_driver_results_are_basic_failures(tmp_path, record):
    path = tmp_path / "round_000.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError):
        biomni_backend.response(path, biomni_backend.MODEL)


@pytest.mark.parametrize("decision", [{"action": "finish", "request_json": "{}", "reason": " "},
                                      {"action": "finish", "request_json": "{", "reason": "r"}])
def test_biomni_invalid_decisions_are_failed_rounds(tmp_path, decision):
    path = tmp_path / "round_000.json"
    write_json(path, {"decision": decision, "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                      "messages": []})
    with pytest.raises(agent.MalformedDecision):
        biomni_backend.response(path, biomni_backend.MODEL)


def test_biomni_preflight_checks_the_version_then_the_green_image(tmp_path, fake_driver, monkeypatch):
    monkeypatch.setattr(biomni_backend, "_version", lambda deadline: "0.0.8")
    record = biomni_backend.preflight(tmp_path / "biomni_preflight", biomni_backend.MODEL)
    assert record["image_preflight"] == "passed" and record["biomni_version"] == "0.0.8"
    assert record["model"] == "gpt-5.6-luna" and record["preflight_tokens"] == 34
    assert record["policy"] == biomni_backend.POLICY
    no_key_in(tmp_path / "biomni_preflight")


@pytest.mark.skipif(importlib.util.find_spec("biomni") is not None, reason="this interpreter has biomni")
def test_biomni_version_check_fails_without_the_package(gateway, monkeypatch):
    monkeypatch.setenv("FULL_CPJUMP1_BIOMNI_PYTHON", sys.executable)
    with pytest.raises(RuntimeError, match="audited version 0.0.8"):
        biomni_backend._version(time.time() + 60)


def test_biomni_crop_stage_end_to_end_through_the_fake_driver(image_stage, fake_driver):
    prep, run, config, rows, waves = image_stage("crop", n=2)
    config.update(arm="biomni", model=biomni_backend.MODEL)
    rule = [{"when": [{"readout": "area", "op": ">", "value": 10}],
             "apply": [{"key": "margin_mode", "value": "mask_bbox"}, {"key": "auto_tighten", "value": True}]}]
    # The handoff runs img0's own session at once (its finish), then the group goes on.
    fake_driver(decisions=[d("run", [{"images": ["img1"], "params": {"cell_rules": rule}}]),
                           d("handoff", ["img0"]), d("finish", {"all": "c0"}), d("finish", {"img1": "c1"})])
    agent.run_stage("crop", prep, run, config)
    rounds = read_json(run / "agent/crop/ROUNDS.json")
    assert [(r["session"], r["outcome"]) for r in rounds] == [("group", "executed"), ("image:img0", "finished"),
                                                              ("group", "executed"), ("group", "finished")]
    requests = fake_driver.requests()
    assert all(r["prompt"].startswith("You are the Biomni arm of the CPJUMP1 benchmark, stage crop") for r in requests)
    assert "This session handles image img0 only" in requests[2]["prompt"]
    selected = read_json(run / "agent/crop/SELECTED.json")
    assert selected["policy"] == biomni_backend.POLICY and selected["total_tokens"] == 4 * 34
    assert {k: (v["call"], v["session"]) for k, v in selected["deliveries"].items()} == {
        "img0": ("c0", "image:img0"), "img1": ("c1", "group")}
    configs = native_configs(run, "crop")
    assert len(configs) == 2 and all(c["arm"] == "codex" and c["agent_arm"] == "biomni" for c in configs)
    assert configs[1]["calls"][0]["params"]["cell_rules"][0]["when"][0]["value"] == "10"
    assert len(read_json(run / "cells.json")) == 4
    no_key_in(run)


def test_a_malformed_biomni_reply_is_handled_as_a_rejected_request_like_codex(image_stage, fake_driver):
    """Biomni's "the solution is not one JSON object" is recorded (outcome malformed, the parse error), shown with the
    required format in the next round, and the session goes on; no structured output or other help is added."""
    prep, run, config, rows, waves = image_stage("segment", n=1)
    config.update(arm="biomni", model=biomni_backend.MODEL)
    fake_driver(decisions=[{"malformed": "the solution is not one JSON object"},
                           d("run", [{"images": "all", "params": {"diameter": 40}}]),
                           d("finish", {"all": "c1"})])
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["malformed", "executed", "finished"]
    error = "Biomni reply is not a decision: the solution is not one JSON object"
    assert rounds[0]["message"] == error and len(waves) == 2
    prompts = [r["prompt"] for r in fake_driver.requests()]
    assert f"Your reply in round 1 was not a valid decision and was not executed: {error}." in prompts[1]
    assert f"Every response must be {agent.RESPONSE_FORMAT}." in prompts[1]
    assert set(fake_driver.requests()[1]) == set(fake_driver.requests()[0])      # the same request, no new field
    selected = read_json(run / "agent/segment/SELECTED.json")
    assert selected["deliveries"]["img0"] == {"call": "c1", "source": "agent", "session": "group"}
    assert selected["total_tokens"] == 3 * 34
    traces = run / "agent/segment/group/traces"
    assert (traces / "failed_round_000.json").is_file() and (traces / "round_001.json").is_file()


def test_a_malformed_claude_reply_is_handled_as_a_rejected_request_like_codex(image_stage, fake_claude):
    prep, run, config, rows, waves = image_stage("segment", n=1)
    config.update(arm="claude", model=claude_backend.MODEL)
    fake_claude(decisions=[{"action": "run", "reason": "no request"}, d("finish", {"all": "c0"})])
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["malformed", "finished"]
    error = "Claude Code structured tool request is missing or malformed"
    assert rounds[0]["message"] == error
    assert f"Your reply in round 1 was not a valid decision and was not executed: {error}." in (
        fake_claude.prompts()[1]["prompt"])
    selected = read_json(run / "agent/segment/SELECTED.json")
    assert selected["deliveries"]["img0"] == {"call": "c0", "source": "agent", "session": "group"}
    assert selected["total_tokens"] == 2 * 20


def test_a_claude_error_result_is_a_malformed_round_and_the_session_goes_on(image_stage, fake_claude):
    prep, run, config, rows, waves = image_stage("segment", n=1)
    config.update(arm="claude", model=claude_backend.MODEL)
    config["smoke_policy"] = {"codex_rounds": 2}
    fake_claude("max_turns", [d("finish", {"all": "c0"}), d("finish", {"all": "c0"})])
    agent.run_stage("segment", prep, run, config)
    rounds = read_json(run / "agent/segment/ROUNDS.json")
    assert [(r["outcome"], r["message"]) for r in rounds] == [
        ("malformed", "Claude Code returned an error result (error_max_turns)")] * 2
    delivery = read_json(run / "agent/segment/SELECTED.json")["deliveries"]["img0"]
    assert delivery["source"] == "default_session_unfinished"
    assert delivery["reason"].startswith("the last round's reply was not a valid decision")


def test_codex_arm_still_records_codex_and_its_worker_configs_name_it(image_stage, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=1)
    monkeypatch.setattr(codex_backend, "invoke", FakeCodex({"group": [d("finish", {})]}))
    agent.run_stage("segment", prep, run, config)
    assert read_json(run / "agent/segment/SELECTED.json")["policy"] == "codex-cli-native-tools-v2"
    assert [(c["arm"], c["agent_arm"]) for c in native_configs(run, "segment")] == [("codex", "codex")]
    assert (run / "agent/segment/auth/config.toml").read_text(encoding="utf-8") == codex_backend._config(codex_backend.MODEL, "medium")
