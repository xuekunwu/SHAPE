"""Codex CLI decisions for the agent-arm stage protocol.

The agent gets SHAPE's task, inputs, tool levers and per-image call budget; it sees only what the tools
return and write, plus the raw inputs it asks to inspect. CLI contracts:
https://learn.chatgpt.com/docs/non-interactive-mode and
https://learn.chatgpt.com/docs/config-file/config-reference . No SDK fallback.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from . import protocol
from .protocol import ACTIONS, BASE_URL, EFFORTS, EVIDENCE_POLICY, MalformedDecision, SCHEMA, read_json, write_json
# `_process` / `_redact` are referenced through `protocol` (not imported by name) so every backend shares ONE
# patchable subprocess seam (`protocol._process`), the way the tests fake the CLI for all three arms at once.

# CPJUMP1's Codex arm ran gpt-5.6-luna before the new-model matrix moved to gpt-6-sol (git history); both
# still serve on the Stanford Gateway, so re-running codex x luna (e.g. a seed2 reproducibility check) only
# needs this set widened -- not a new routing path, unlike claude.py's Anthropic-direct split.
ALLOWED_MODELS = ("gpt-6-sol", "gpt-5.6-luna")
CLI_VERSION = "codex-cli 0.162.0"
POLICY = "codex-cli-native-tools-v2"
PROVIDER = "stanford"
DISABLED_HOST_DIAGNOSTIC = (
    "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; "
    "enable `features.code_mode_host` and install `codex-code-mode-host`."
)
DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "shell_snapshot", "view_image", "apps", "plugins",
    "multi_agent", "multi_agent_v2", "hooks", "memories", "code_mode", "code_mode_host",
    "js_repl", "apply_patch_freeform", "browser_use", "browser_use_external",
    "in_app_browser", "computer_use", "image_generation", "remote_plugin",
    "skill_search", "skill_mcp_dependency_install", "workspace_dependencies",
    "tool_suggest", "goals", "unbounded_connection_retries",
)


def _route(model):
    if model not in ALLOWED_MODELS:
        raise ValueError(f"CPJUMP1 Codex requires one of {ALLOWED_MODELS}")
    for variable in ("LLM_API_BASE", "LLM_BASE_URL"):
        if os.environ.get(variable, BASE_URL).rstrip("/") != BASE_URL:
            raise ValueError("Codex requires the Stanford Gateway Responses route")
    if not os.environ.get("LLM_API_KEY", "").strip():
        raise RuntimeError("Codex preflight blocked: LLM_API_KEY is missing")


route = _route


def _config(model, effort=None):
    """config.toml; `effort`: its model_reasoning_effort (None: not written, the CLI's default)."""
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"unknown Codex reasoning effort {effort!r}")
    return (f'model = {json.dumps(model)}\nmodel_provider = "{PROVIDER}"\n'
            + (f'model_reasoning_effort = "{effort}"\n' if effort is not None else '')
            + 'approval_policy = "never"\nsandbox_mode = "read-only"\nweb_search = "disabled"\n'
            'project_doc_max_bytes = 0\nproject_doc_fallback_filenames = []\n'
            'project_root_markers = [".cpjump1-root"]\n'
            'cli_auth_credentials_store = "ephemeral"\ncheck_for_update_on_startup = false\n'
            '[history]\npersistence = "none"\n'
            '[shell_environment_policy]\ninherit = "none"\n'
            '[mcp_servers]\n[features]\n'
            + ''.join(f'{name} = false\n' for name in DISABLED_FEATURES)
            + f'[model_providers.{PROVIDER}]\nname = "Stanford Gateway"\n'
            + f'base_url = "{BASE_URL}"\nenv_key = "LLM_API_KEY"\n'
            'wire_api = "responses"\nrequires_openai_auth = false\nsupports_websockets = false\n'
            'request_max_retries = 0\nstream_max_retries = 0\n')


def setup(root: Path, model: str, evidence=(), effort=None) -> Path:
    """Write only nonsecret configuration; no inherited Codex home or auth file. `effort`: the stage's
    model_reasoning_effort (pipeline.REASONING)."""
    _route(model)
    root.mkdir(parents=True, exist_ok=True)
    settings = root / "config.toml"
    # Refuse an old home rather than reuse its auth, hooks, skills or MCP settings.
    if any(root.iterdir()):
        raise ValueError("Codex setup requires an empty configuration directory")
    settings.write_text(_config(model, effort), encoding="utf-8")
    return settings


def _env(home: Path, *, authenticated=True, run=None, marker_env=None):
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL",
               "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"}
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    env.update(CODEX_HOME=str(home), HOME=str(home.parent), USERPROFILE=str(home.parent),
               XDG_CONFIG_HOME=str(home.parent / "xdg"), TMP=str(home.parent),
               TEMP=str(home.parent), TMPDIR=str(home.parent))
    if authenticated:
        env["LLM_API_KEY"] = os.environ["LLM_API_KEY"]
    if run is not None and marker_env:
        # An outer watchdog (the caller's harness) scans this marker even after a nested setsid/chdir; the
        # caller names the variable, so this backend knows nothing of any particular harness.
        env[marker_env] = str(Path(run).resolve())
    return env


def command(model, schema, last_message, images=()):
    cmd = ["codex", "exec", "--json", "--strict-config", "--output-schema", str(schema),
           "--output-last-message", str(last_message), "--ephemeral", "--sandbox", "read-only",
           "--skip-git-repo-check", "--model", model, "--color", "never"]
    for path in images:
        cmd += ["--image", str(path)]
    return cmd + ["-"]


def stream_usage(trace: Path, model: str) -> dict:
    """Validate the documented exec JSONL event stream (any failure here is a basic CLI/LLM failure)."""
    if model not in ALLOWED_MODELS:
        raise ValueError("unexpected requested Codex model")
    try:
        events = [json.loads(s) for s in trace.read_text(encoding="utf-8").splitlines() if s.strip()]
    except (ValueError, OSError):
        raise ValueError("Codex returned a malformed event stream") from None
    state, thread_id, usage, messages = "new", None, None, []
    startup_diagnostics = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("invalid Codex event")
        kind = event.get("type")
        if event.get("model") is not None and event["model"] != model:
            raise RuntimeError("Codex event reported a different model")
        if kind in ("error", "turn.failed") or event.get("error"):
            raise RuntimeError("Codex reported a failed event")
        if kind == "thread.started" and state == "new":
            thread_id = event.get("thread_id")
            if not isinstance(thread_id, str) or not thread_id:
                raise ValueError("Codex thread ID missing")
            state = "thread"
        elif kind == "item.completed" and state == "thread":
            item = event.get("item")
            # 0.162.0 exec's collect_warning emits Error items. Its disabled
            # CodeModeSessionProvider rejects both availability and session creation.
            if (not isinstance(item, dict) or set(item) != {"id", "type", "message"}
                    or not isinstance(item["id"], str) or not item["id"]
                    or item["type"] != "error" or item["message"] != DISABLED_HOST_DIAGNOSTIC
                    or startup_diagnostics):
                raise RuntimeError("Codex reported an unexpected startup diagnostic")
            startup_diagnostics.append(item["message"])
        elif kind == "turn.started" and state == "thread":
            state = "turn"
        elif kind in ("item.started", "item.updated", "item.completed") and state == "turn":
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") not in ("agent_message", "reasoning", "todo_list"):
                raise RuntimeError("Codex emitted a forbidden tool or unknown item")
            if item.get("status") in ("failed", "error") or item.get("error"):
                raise RuntimeError("Codex reported a failed item")
            if item.get("model") is not None and item["model"] != model:
                raise RuntimeError("Codex item reported a different model")
            if kind == "item.completed" and item["type"] == "agent_message":
                messages.append(item.get("text"))
        elif kind == "turn.completed" and state == "turn":
            usage, state = event.get("usage"), "done"
        else:
            raise RuntimeError("Codex event stream is unknown, incomplete or out of order")
    if state != "done" or not isinstance(usage, dict):
        raise RuntimeError("Codex did not complete with token usage")
    if not {"input_tokens", "output_tokens"}.issubset(usage):
        raise RuntimeError("Codex token usage is incomplete")
    for key in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_input_tokens", "reasoning_output_tokens"):
        value = usage.get(key, 0)
        if type(value) is not int or value < 0:
            raise RuntimeError("Codex token usage is malformed")
    # Cached inputs and reasoning outputs are subsets, not extra tokens.
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    if tokens <= 0 or usage.get("cached_input_tokens", 0) > usage.get("input_tokens", 0):
        raise RuntimeError("Codex reported no positive valid token usage")
    return {"total_tokens": tokens, "usage": usage, "model": model,
            "requested_model": model, "response_model": None,
            "response_model_attested": False, "model_provenance": "requested_cli_config",
            "provider": PROVIDER, "base_url": BASE_URL, "wire_api": "responses",
            "thread_id": thread_id, "startup_diagnostics": startup_diagnostics, "messages": messages}


def response(trace: Path, model: str, last_message: Path):
    """Validate the documented exec JSONL API, never infer a response model; then the decision itself
    (MalformedDecision: a malformed reply, handled as a rejected request)."""
    usage = stream_usage(trace, model)
    messages, tokens = usage.pop("messages"), usage["total_tokens"]
    try:
        decision = read_json(last_message)
    except (ValueError, OSError):
        raise MalformedDecision("Codex returned malformed JSON or no final message", tokens) from None
    if not isinstance(decision, dict) or set(decision) != set(SCHEMA["required"]):
        raise MalformedDecision("Codex structured tool request is missing or malformed", tokens)
    if any(not isinstance(v, str) for v in decision.values()) or decision["action"] not in ACTIONS:
        raise MalformedDecision("invalid Codex decision types", tokens)
    if not decision["reason"].strip():
        raise MalformedDecision("Codex decision must give a reason", tokens)
    try:
        matches = bool(messages) and json.loads(messages[-1]) == decision
        json.loads(decision["request_json"])
    except (ValueError, TypeError):
        raise MalformedDecision("Codex decision JSON is malformed", tokens) from None
    if not matches:
        raise MalformedDecision("Codex final message does not match the event stream", tokens)
    return decision, usage


def _check_tools(cwd, home, deadline, run, marker_env=None):
    env = _env(home, authenticated=False, run=run, marker_env=marker_env)
    rc, version, _ = protocol._process(["codex", "--version"], cwd, env, deadline)
    version = version.strip()
    # The feature-flag audit below (shell_tool etc. disabled) is the execution-safety guard and
    # always runs. The exact CLI version is NOT pinned by default -- that broke every time the CLI
    # was updated; CLI_VERSION is the version this arm was audited against, enforced only when
    # $FULL_CPJUMP1_CODEX_CLI_VERSION is set (e.g. a reproducible submission run).
    required = os.environ.get("FULL_CPJUMP1_CODEX_CLI_VERSION", CLI_VERSION) \
        if os.environ.get("FULL_CPJUMP1_PIN_CLI") else None
    if rc or (required and version != required):
        raise RuntimeError(f"codex --version failed (rc={rc}) or != pinned {required!r}; got {version!r}")
    rc, features, _ = protocol._process(["codex", "features", "list"], cwd,
                               env, deadline)
    flags = {parts[0]: parts[-1] for line in features.splitlines() if len(parts := line.split()) >= 3}
    if rc or flags.get("shell_tool") != "false":
        raise RuntimeError("Codex cannot verify execution tools are disabled")
    # rust-v0.162.0 core/src/config/managed_features.rs forces unified_exec on.
    # core/src/tools/spec_plan.rs::add_shell_tools returns before registering
    # exec_command/write_stdin when ShellTool is disabled, regardless of backend.
    if any(flags.get(k, "false") != "false" for k in DISABLED_FEATURES if k != "unified_exec"):
        raise RuntimeError("Codex tool configuration was overridden")
    return version.strip()


def invoke(root, prompt, model, settings, deadline, index, session=None, images=(), marker=None, effort=None,
           marker_env=None):
    """One ephemeral round; traces in root/traces. `marker`: the stage root the outer watchdog scans (named by
    `marker_env`, else not exported); `effort`: the model_reasoning_effort `settings` was written with (setup)."""
    _route(model)
    if session is not None:
        raise ValueError("Codex uses ephemeral rounds with explicit evidence, not session resume")
    if Path(settings).read_text(encoding="utf-8") != _config(model, effort):
        raise ValueError("Codex configuration changed")
    traces = root / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    trace, final = traces / f"round_{index:03d}.jsonl", traces / f"round_{index:03d}.json"
    if trace.exists() or final.exists():
        raise ValueError("Codex round output already exists")
    # OS temporary directory avoids all repository/ancestor project configuration.
    with tempfile.TemporaryDirectory(prefix="cpjump1-codex-") as temporary:
        workspace = Path(temporary)
        (workspace / ".cpjump1-root").touch()
        home = workspace / "codex_home"
        home.mkdir(mode=0o700)
        shutil.copy2(settings, home / "config.toml")
        version = _check_tools(workspace, home, deadline, marker or root, marker_env)
        schema, last = workspace / "schema.json", workspace / "last.json"
        write_json(schema, SCHEMA)
        attachments = []
        for i, path in enumerate(images):
            target = workspace / f"evidence_{i:03d}.png"
            shutil.copy2(path, target)
            attachments.append(target)
        rc, stdout, stderr = protocol._process(command(model, schema, last, attachments), workspace,
                                     _env(home, run=marker or root, marker_env=marker_env), deadline, prompt)
        trace.write_text(stdout, encoding="utf-8")
        trace.with_suffix(".stderr").write_text(stderr, encoding="utf-8")
        if rc:
            if "413 Payload Too Large" in stdout:
                # The gateway refuses a request above its size limit (base64 attachments plus the prompt can pass
                # it although the attachments are within max_attachment_bytes): a rejected request, not a broken
                # CLI. The round's traces move aside as for a malformed reply; the next round goes without them.
                for path in (trace, trace.with_suffix(".stderr")):
                    path.replace(path.with_name("failed_" + path.name))
                raise MalformedDecision("the gateway rejected this round's request as too large (HTTP 413: the "
                                        "attached images plus the prompt exceed its size limit); nothing was "
                                        "attached to the reply; inspect fewer or smaller items")
            raise RuntimeError("Codex CLI failed; Stanford Responses support/auth/model preflight is required")
        if last.is_file():
            final.write_text(protocol._redact(last.read_text(encoding="utf-8")), encoding="utf-8")
    try:
        decision, usage = response(trace, model, final)
    except MalformedDecision:
        # A malformed reply keeps its trace under failed_round_*, outside the decision traces.
        for path in (trace, final, trace.with_suffix(".stderr")):
            if path.exists():
                path.replace(path.with_name("failed_" + path.name))
        raise
    usage["codex_version"] = version
    usage["model_reasoning_effort"] = effort
    return decision, usage, None


def preflight(root, model):
    _route(model)
    root = Path(root)
    settings = setup(root / "auth", model)
    deadline = time.time() + 180
    with tempfile.TemporaryDirectory(prefix="cpjump1-codex-check-") as temporary:
        cwd = Path(temporary)
        (cwd / ".cpjump1-root").touch()
        home = cwd / "codex_home"
        home.mkdir()
        shutil.copy2(settings, home / "config.toml")
        env = _env(home, authenticated=False, run=root)
        version = _check_tools(cwd, home, deadline, root)
        rc, help_text, _ = protocol._process(["codex", "exec", "--help"], cwd, env, deadline)
        required = ("--json", "--output-schema", "--output-last-message", "--ephemeral",
                    "--sandbox", "--skip-git-repo-check", "--image", "--model", "--strict-config")
        if rc or any(flag not in help_text for flag in required):
            raise RuntimeError("installed Codex CLI lacks required exec flags")
    from PIL import Image
    preview = root / "preflight.png"
    Image.new("RGB", (32, 32), (0, 255, 0)).save(preview)
    decision, usage, _ = invoke(root, 'Connectivity check. Return action="finish", request_json="{}". '
                               'Inspect the attached image and set reason to '
                               'its dominant color, using only the lowercase English color name.',
                               model, settings, deadline, 0, images=[preview])
    if decision["action"] != "finish" or json.loads(decision["request_json"]) != {}:
        raise ValueError("unexpected Codex preflight response")
    if decision["reason"].strip().lower() != "green":
        raise RuntimeError("Codex image preflight failed: attached image was not identified")
    return {**usage, "codex_version": version.strip(), "preflight_tokens": usage["total_tokens"],
            "image_preflight": "passed", "key_variable": "LLM_API_KEY", "policy": POLICY,
            "evidence_policy": EVIDENCE_POLICY}
