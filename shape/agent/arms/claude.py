"""Claude Code decisions for the agent-arm stage protocol (codex_arm.StageRun).

The task, inputs, tools, budget, prompt, actions and decision schema are the Codex arm's; only the agent answering
each round differs. Each round is one headless `claude -p` in a fresh OS temporary workspace, Read its only tool,
limited by a PreToolUse hook to the round's attached evidence files. The Stanford gateway key reaches the CLI only
through the apiKeyHelper of its settings file; it is never in the CLI's environment, a command line or a trace.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import time

from . import protocol
from .protocol import read_json, write_json

MODEL = "claude-opus-5"
# The Claude Code arm's allowed reasoners (the CLI alias env vars and the --model flag are all set to the
# one picked per run; a run may not mix models -- modelUsage must name exactly it). Add new ids here.
ALLOWED_MODELS = ("claude-opus-5", "claude-sonnet-5-5")
CLAUDE_CLI_VERSION = "2.1.296"
POLICY = "claude-code-native-tools-v2"
# the result subtypes of the agent's own failure (a malformed reply); any other error result is an API error
AGENT_ERRORS = ("error_max_turns", "error_max_structured_output_retries")
MODEL_USAGE_KEYS = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens")
# The base_url handed to the Claude Code CLI. Defaults to the Stanford gateway but follows
# LLM_BASE_URL when set (the repo .env), so the claude arm can run against any
# OpenAI/Anthropic-compatible gateway. The route guard (protocol.BASE_URL) is
# already env-driven, so both stay consistent.
ANTHROPIC_BASE_URL = (os.environ.get("LLM_BASE_URL")
                      or "https://llm-gateway.example.edu").rstrip("/")
MAX_TURNS = 24


def tool_bins() -> list[str]:
    """The Claude Code CLI and its Node runtime bin dirs, put first on PATH. Discover the NEWEST
    installed version of each under ~/shape/cache/envs (never pin a version -- it breaks the moment
    the CLI/Node is updated); override with $SHAPE_CLAUDE_BIN / $SHAPE_NODE_BIN (absolute bin dirs).
    A missing env is simply left off PATH."""
    import glob
    import re
    envs = Path.home() / "shape" / "cache" / "envs"

    def newest(prefix: str, override: str) -> list[str]:
        chosen = os.environ.get(override)
        if chosen:
            return [chosen]
        found = glob.glob(str(envs / f"{prefix}*" / "bin"))
        key = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]  # noqa: E731
        return [sorted(found, key=key)[-1]] if found else []

    return newest("claude-code-", "SHAPE_CLAUDE_BIN") + newest("node-", "SHAPE_NODE_BIN")
# Every model alias the CLI may resolve, background calls included, is the arm's model.
MODEL_ALIASES = ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                 "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL")
# Project instructions the CLI would load from the workspace or any ancestor.
INSTRUCTION_FILES = ("CLAUDE.md", "CLAUDE.local.md", ".claude")
HELP_FLAGS = ("--print", "--output-format", "--json-schema", "--model", "--settings", "--tools",
              "--strict-mcp-config")
USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
READ_GUARD = Path(__file__).with_name("read_guard.py")
# Round workspaces live in /tmp, not under the arm's TMPDIR (a launcher may put it in the run cache, below a home that
# may hold a .claude); elsewhere (tests) the OS default.
TEMP_ROOT = "/tmp" if os.name == "posix" else None


def evidence_line(names) -> str:
    """The one prompt line Claude gets beyond the Codex prompt: it reads attached images through Read."""
    return (f"Attached images are the files {', '.join(names)} in your working directory; "
            "view them with the Read tool.")


def api_env() -> Path:
    path = os.environ.get("SHAPE_API_ENV_FILE", "")
    if not path or not Path(path).is_absolute() or not Path(path).is_file():
        raise RuntimeError("Claude Code preflight blocked: SHAPE_API_ENV_FILE must name the absolute api.env file")
    return Path(path)


def route(model):
    if model not in ALLOWED_MODELS:
        raise ValueError(f"CPJUMP1 Claude Code requires one of {ALLOWED_MODELS}")
    for variable in ("LLM_API_BASE", "LLM_BASE_URL"):
        if os.environ.get(variable, protocol.BASE_URL).rstrip("/") != protocol.BASE_URL:
            raise ValueError("Claude Code requires the Stanford Gateway route")
    api_env()


def _helper(source: Path) -> str:
    """The apiKeyHelper: source the API environment file silently, check its gateway URL, print only the key."""
    return ("#!/usr/bin/env bash\n"
            f"source {shlex.quote(str(source))} >/dev/null 2>&1\n"
            f'[[ "${{LLM_BASE_URL%/}}" == {protocol.BASE_URL} && -n "$LLM_API_KEY" ]] || exit 1\n'
            'printf "%s" "$LLM_API_KEY"\n')


def _settings(helper: Path) -> dict:
    return {"apiKeyHelper": shlex.quote(str(helper))}


def setup(root: Path, model: str) -> Path:
    """Write only nonsecret configuration: the settings file and its key helper, no key."""
    route(model)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Claude Code setup requires an empty configuration directory")
    helper = root / "key_helper.sh"
    helper.write_text(_helper(api_env()), encoding="ascii", newline="\n")
    helper.chmod(0o700)
    settings = root / "settings.json"
    write_json(settings, _settings(helper))
    return settings


def _check_settings(settings: Path) -> None:
    helper = Path(settings).with_name("key_helper.sh")
    try:
        unchanged = (read_json(settings) == _settings(helper)
                     and helper.read_text(encoding="ascii") == _helper(api_env()))
    except (ValueError, OSError):
        unchanged = False
    if not unchanged:
        raise ValueError("Claude Code configuration changed")


# Read deny rules beside the hook (QC 2026-10-12, verified on the VM with 2.1.296): an @-file mention in the prompt
# is expanded by the CLI itself, without a Read call, so the hook never sees it; a deny rule stops it. The agent's own
# reasons return verbatim in later prompts, so a mention could name the key file. Denied: the system and user trees
# (the key, the data, the code) and the round's own configuration; the evidence files stay readable.
DENIED_ROOTS = ("/etc", "/home", "/root", "/mnt", "/proc", "/sys", "/var", "/opt", "/srv")


def _round_settings(settings: Path, roots: Path) -> dict:
    """The round's settings: setup's, plus the hook that lets Read open only the round's evidence files and the Read
    deny rules for everything else that matters (`DENIED_ROOTS`, the round's config dir and its read roots)."""
    hook = shlex.join([sys.executable, "-I", str(READ_GUARD), str(roots)])
    workspace = Path(roots).resolve().parent
    deny = [f"Read(/{root}/**)" for root in DENIED_ROOTS]
    deny += [f"Read(/{(workspace / 'config').as_posix()}/**)", f"Read(/{(workspace / 'read_roots.json').as_posix()})"]
    return {**read_json(settings), "permissions": {"deny": deny},
            "hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"type": "command", "command": hook}]}]}}


def _inherited(workspace: Path) -> list[Path]:
    return [parent / name for parent in (workspace, *workspace.parents) for name in INSTRUCTION_FILES
            if (parent / name).exists()]


def _env(config_dir: Path, workspace: Path, run=None, marker_env=None, model=MODEL) -> dict:
    """From scratch, as Codex's: no key variable, a run-private config dir and home, every alias the arm's model."""
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL",
               "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"}
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    bins = tool_bins()
    env["PATH"] = os.pathsep.join([*bins, env.get("PATH", "")])
    env.update(CLAUDE_CONFIG_DIR=str(config_dir), HOME=str(config_dir), USERPROFILE=str(config_dir),
               TMP=str(workspace), TEMP=str(workspace), TMPDIR=str(workspace),
               ANTHROPIC_BASE_URL=ANTHROPIC_BASE_URL,
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_TELEMETRY="1",
               # no CLI retries, as Codex (request/stream retries 0): an API error fails the arm (user, 2026-10-12)
               CLAUDE_CODE_MAX_RETRIES="0",
               **{alias: model for alias in MODEL_ALIASES})
    if run is not None and marker_env:
        # An outer watchdog (the caller's harness) scans this marker even after a nested setsid/chdir; the
        # caller names the variable, so this backend knows nothing of any particular harness.
        env[marker_env] = str(Path(run).resolve())
    return env


def command(model, settings) -> list[str]:
    """The headless CLI; the prompt goes on stdin."""
    return ["claude", "-p", "--output-format", "json",
            "--json-schema", json.dumps(protocol.SCHEMA, separators=(",", ":")), "--model", model,
            "--settings", str(settings), "--tools", "Read", "--strict-mcp-config", "--max-turns", str(MAX_TURNS)]


def _version(cwd, env, deadline) -> str:
    rc, out, _ = protocol._process(["claude", "--version"], cwd, env, deadline, name="Claude Code")
    version = out.strip().split()[0] if out.strip() else ""
    # Not pinned by default (pinning broke on every CLI update); CLAUDE_CLI_VERSION is the audited
    # version, enforced only when $FULL_CPJUMP1_PIN_CLI is set (reproducible submission run).
    required = CLAUDE_CLI_VERSION if os.environ.get("FULL_CPJUMP1_PIN_CLI") else None
    if rc or not version or (required and version != required):
        raise RuntimeError(f"claude --version failed (rc={rc}) or != pinned {required!r}; got {version!r}")
    return version


def stream_usage(trace: Path, model: str) -> dict:
    """The CLI's documented JSON result and its token usage (any failure here is a basic CLI/LLM failure; a result
    without tokens means the model never answered: route, auth or gateway). Keeps the result under "result"."""
    if model not in ALLOWED_MODELS:
        raise ValueError("unexpected requested Claude Code model")
    try:
        result = json.loads(Path(trace).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        raise RuntimeError("Claude Code returned no JSON result") from None
    if not isinstance(result, dict) or result.get("type") != "result" or not isinstance(result.get("usage"), dict):
        raise RuntimeError("Claude Code returned no result with token usage")
    usage = result["usage"]
    if not {"input_tokens", "output_tokens"}.issubset(usage):
        raise RuntimeError("Claude Code token usage is incomplete")
    for key in USAGE_KEYS:
        value = usage.get(key, 0)
        if type(value) is not int or value < 0:
            raise RuntimeError("Claude Code token usage is malformed")
    tokens = sum(usage.get(key, 0) for key in USAGE_KEYS)
    if tokens <= 0:
        raise RuntimeError("Claude Code reported no positive token usage (route, auth or gateway failure)")
    models = result.get("modelUsage")
    if not isinstance(models, dict) or set(models) != {model}:
        raise RuntimeError("Claude Code reported a different or a background model")
    # every call of the model, background ones included (QC F8): the larger of the two totals
    counted = sum(int(models[model].get(key) or 0) for key in MODEL_USAGE_KEYS)
    tokens = max(tokens, counted)
    return {"total_tokens": tokens, "usage": {k: usage.get(k, 0) for k in USAGE_KEYS}, "model": model,
            "requested_model": model, "model_usage": models, "response_model": None,
            "response_model_attested": False, "model_provenance": "requested_cli_aliases",
            "provider": "stanford", "base_url": ANTHROPIC_BASE_URL, "num_turns": result.get("num_turns"),
            "permission_denials": result.get("permission_denials"), "total_cost_usd": result.get("total_cost_usd"),
            "result": result}


def response(trace: Path, model: str):
    """The decision of a validated result (MalformedDecision: an error result or no valid structured output, a
    malformed reply, handled as a rejected request)."""
    usage = stream_usage(trace, model)
    result, tokens = usage.pop("result"), usage["total_tokens"]
    subtype = result.get("subtype")
    if subtype in AGENT_ERRORS:            # the agent's own failure: as a malformed Codex reply
        raise protocol.MalformedDecision(f"Claude Code returned an error result ({subtype})", tokens)
    if result.get("is_error") or subtype != "success" or result.get("api_error_status") is not None:
        # an API or gateway error, as Codex's error event: the arm fails (user, 2026-10-12)
        raise RuntimeError(f"Claude Code API error ({subtype}, status {result.get('api_error_status')})")
    decision = result.get("structured_output")
    problem = protocol.decision_problem(decision)
    if problem:
        raise protocol.MalformedDecision(f"Claude Code {problem}", tokens)
    return decision, usage


def trace_tokens(trace: Path, model: str) -> int:
    """The tokens of a kept round trace: a decision trace must still hold a valid decision; a failed round's only
    valid usage."""
    if Path(trace).name.startswith("failed_"):
        return stream_usage(trace, model)["total_tokens"]
    return response(trace, model)[1]["total_tokens"]


def invoke(root, prompt, model, settings, deadline, index, images=(), marker=None, marker_env=None):
    """One round; traces in root/traces. `marker`: the stage root the outer watchdog scans (named by `marker_env`,
    else not exported)."""
    route(model)
    _check_settings(settings)
    traces = Path(root) / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    trace = traces / f"round_{index:03d}.json"
    if trace.exists() or trace.with_suffix(".stderr").exists():
        raise ValueError("Claude Code round output already exists")
    # OS temporary directory: no repository or ancestor project instructions.
    with tempfile.TemporaryDirectory(prefix="cpj-claude-", dir=TEMP_ROOT) as temporary:
        workspace = Path(temporary)
        found = _inherited(workspace)
        if found:
            raise ValueError(f"unexpected inherited Claude instructions: {found[0]}")
        config_dir = workspace / "config"
        config_dir.mkdir(mode=0o700)
        names = []
        for i, path in enumerate(images):
            names.append(f"evidence_{i:03d}.png")
            shutil.copy2(path, workspace / names[-1])
        roots = workspace / "read_roots.json"
        write_json(roots, [str((workspace / name).resolve()) for name in names])
        write_json(config_dir / "settings.json", _round_settings(settings, roots))
        env = _env(config_dir, workspace, run=marker or root, marker_env=marker_env, model=model)
        version = _version(workspace, env, deadline)
        text = prompt + ("\n" + evidence_line(names) if names else "")
        rc, stdout, stderr = protocol._process(command(model, config_dir / "settings.json"), workspace, env,
                                               deadline, text, name="Claude Code")
        trace.write_text(stdout, encoding="utf-8")
        trace.with_suffix(".stderr").write_text(stderr, encoding="utf-8")
    failure = RuntimeError("Claude Code CLI failed; Stanford gateway route/auth/model preflight is required")
    try:
        decision, usage = response(trace, model)
    except protocol.MalformedDecision:
        # An error result (the CLI exits nonzero) or an invalid reply is a malformed reply; it keeps its trace
        # under failed_round_*, outside the decision traces.
        for path in (trace, trace.with_suffix(".stderr")):
            if path.exists():
                path.replace(path.with_name("failed_" + path.name))
        raise
    except RuntimeError:
        if rc:
            raise failure from None
        raise
    if rc:
        raise failure
    usage["claude_version"] = version
    return decision, usage, None


def preflight(root, model):
    """The Codex arm's check: the audited CLI and its flags, then a plain finish that names the attached image's
    colour."""
    route(model)
    root = Path(root)
    settings = setup(root / "auth", model)
    deadline = time.time() + 180
    with tempfile.TemporaryDirectory(prefix="cpj-claude-check-", dir=TEMP_ROOT) as temporary:
        cwd = Path(temporary)
        (cwd / "config").mkdir()
        env = _env(cwd / "config", cwd, run=root, model=model)
        version = _version(cwd, env, deadline)
        rc, help_text, _ = protocol._process(["claude", "--help"], cwd, env, deadline, name="Claude Code")
        if rc or any(flag not in help_text for flag in HELP_FLAGS):
            raise RuntimeError("installed Claude Code CLI lacks required flags")
    from PIL import Image
    preview = root / "preflight.png"
    Image.new("RGB", (32, 32), (0, 255, 0)).save(preview)
    decision, usage, _ = invoke(root, 'Connectivity check. Return action="finish", request_json="{}". '
                               'Inspect the attached image and set reason to '
                               'its dominant color, using only the lowercase English color name.',
                               model, settings, deadline, 0, images=[preview])
    if decision["action"] != "finish" or json.loads(decision["request_json"]) != {}:
        raise ValueError("unexpected Claude Code preflight response")
    if decision["reason"].strip().lower() != "green":
        raise RuntimeError("Claude Code image preflight failed: attached image was not identified")
    return {**usage, "claude_version": version, "preflight_tokens": usage["total_tokens"],
            "image_preflight": "passed", "key_variable": "apiKeyHelper:LLM_API_KEY", "policy": POLICY,
            "evidence_policy": protocol.EVIDENCE_POLICY}
