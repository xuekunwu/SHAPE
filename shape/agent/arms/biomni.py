"""Biomni decisions for the agent-arm stage protocol (codex_arm.StageRun).

The task, inputs, tools, budget, prompt, actions and decision schema are the Codex arm's; only the agent answering
each round differs. Biomni is decision-only, as Codex is: each round runs `biomni_driver.py` under Biomni's own
virtual environment (never imported into this one), in a fresh OS temporary directory, with an environment of only
PATH, a temporary HOME, the locale and LLM_API_KEY (Biomni's gateway client needs it).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from . import protocol
from .protocol import read_json, write_json

MODEL = "gpt-6-sol"                                   # default reasoner
ALLOWED = frozenset({"gpt-6-sol", "gpt-5.6-luna", "claude-opus-5", "claude-sonnet-5-5",  # gateway (gpt-*) / Anthropic-direct (claude-*)
                     "zai-org/GLM-5.3-Flash", "deepseek-ai/DeepSeek-V4.1-Flash"})  # Together.ai OpenAI-compatible (Custom source, LLM_API_KEY)
BIOMNI_VERSION = "0.0.8"
POLICY = "biomni-a1-decision-only-v1"
DRIVER = Path(__file__).with_name("biomni_driver.py")
PYTHON_ENV = "FULL_CPJUMP1_BIOMNI_PYTHON"
USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def _provider(model: str) -> str:
    """'anthropic' for claude-* (Biomni A1 source "Anthropic", ANTHROPIC_API_KEY, native endpoint), else 'openai'
    (source "Custom" against the OpenAI-compatible gateway with LLM_API_KEY). The gateway serves no claude model."""
    return "anthropic" if model.startswith("claude") else "openai"


def python() -> Path:
    """Biomni's own interpreter: $FULL_CPJUMP1_BIOMNI_PYTHON, else the NEWEST installed
    ~/shape/cache/envs/biomni-*/bin/python (never pin a version -- it breaks on update)."""
    chosen = os.environ.get(PYTHON_ENV)
    if chosen:
        path = Path(chosen)
        if not path.is_absolute():
            raise ValueError(f"{PYTHON_ENV} must be an absolute path")
        return path
    import glob
    import re
    found = glob.glob(str(Path.home() / "shape" / "cache" / "envs" / "biomni-*" / "bin" / "python"))
    if not found:
        raise FileNotFoundError("no ~/shape/cache/envs/biomni-*/bin/python; set FULL_CPJUMP1_BIOMNI_PYTHON")
    key = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]  # noqa: E731
    return Path(sorted(found, key=key)[-1])


def route(model):
    if model not in ALLOWED:
        raise ValueError(f"CPJUMP1 Biomni accepts {sorted(ALLOWED)}, not {model!r}")
    if _provider(model) == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise RuntimeError("Biomni (Anthropic) preflight blocked: ANTHROPIC_API_KEY is missing")
        return
    for variable in ("LLM_API_BASE", "LLM_BASE_URL"):
        if os.environ.get(variable, protocol.BASE_URL).rstrip("/") != protocol.BASE_URL:
            raise ValueError("Biomni requires the configured OpenAI-compatible gateway route")
    if not os.environ.get("LLM_API_KEY", "").strip():
        raise RuntimeError("Biomni preflight blocked: LLM_API_KEY is missing")


def _settings(model) -> dict:
    # claude-* is Biomni's native Anthropic source (no base_url: the anthropic SDK's default endpoint);
    # gpt-* is the "Custom" OpenAI-compatible source pointed at the gateway.
    if _provider(model) == "anthropic":
        return {"backend": "biomni", "model": model, "source": "Anthropic", "base_url": None,
                "python": str(python()), "driver": str(DRIVER), "biomni_version": BIOMNI_VERSION}
    return {"backend": "biomni", "model": model, "source": "Custom", "base_url": protocol.BASE_URL,
            "python": str(python()), "driver": str(DRIVER), "biomni_version": BIOMNI_VERSION}


def setup(root: Path, model: str) -> Path:
    """Write only nonsecret configuration."""
    route(model)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Biomni setup requires an empty configuration directory")
    settings = root / "settings.json"
    write_json(settings, _settings(model))
    return settings


def _env(home: Path, run=None, marker_env=None, model: str = MODEL) -> dict:
    """Only PATH, a temporary HOME, the locale and the reasoner's key (plus what Windows needs to start Python).
    gpt-* carries LLM_API_KEY (the gateway), claude-* carries ANTHROPIC_API_KEY (Anthropic-direct). The key is
    set only when present, so the keyless version probe runs for either provider."""
    allowed = {"PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR"}
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    env.update(HOME=str(home))
    key = "ANTHROPIC_API_KEY" if _provider(model) == "anthropic" else "LLM_API_KEY"
    if os.environ.get(key):
        env[key] = os.environ[key]
    if run is not None and marker_env:
        # An outer watchdog (the caller's harness) scans this marker even after a nested setsid/chdir; the
        # caller names the variable, so this backend knows nothing of any particular harness.
        env[marker_env] = str(Path(run).resolve())
    return env


def stream_usage(trace: Path, model: str) -> dict:
    """The driver's result and its token usage (any failure here is a basic driver/LLM failure). Keeps the result
    under "result"."""
    if model not in ALLOWED:
        raise ValueError("unexpected requested Biomni model")
    try:
        result = read_json(trace)
    except (ValueError, OSError):
        raise RuntimeError("Biomni driver returned no JSON result") from None
    usage = result.get("usage") if isinstance(result, dict) else None
    if not isinstance(usage, dict) or not isinstance(result.get("messages"), list) or (
            ("decision" in result) == ("malformed" in result)):
        raise RuntimeError("Biomni driver result is incomplete")
    for key in USAGE_KEYS:
        value = usage.get(key)
        if type(value) is not int or value < 0:
            raise RuntimeError("Biomni token usage is malformed")
    if usage["total_tokens"] <= 0:
        raise RuntimeError("Biomni reported no positive token usage")
    anthropic = _provider(model) == "anthropic"
    return {"total_tokens": usage["total_tokens"], "usage": {k: usage[k] for k in USAGE_KEYS}, "model": model,
            "requested_model": model, "response_model": None, "response_model_attested": False,
            "model_provenance": "requested_driver_config",
            "provider": "anthropic" if anthropic else "stanford",
            "base_url": "https://api.anthropic.com" if anthropic else protocol.BASE_URL,
            "steps": result.get("steps"), "refused_executions": result.get("refused_executions"), "result": result}


def response(trace: Path, model: str):
    """The decision of a validated result (MalformedDecision: no valid decision, a malformed reply, handled as a
    rejected request)."""
    usage = stream_usage(trace, model)
    result, tokens = usage.pop("result"), usage["total_tokens"]
    if "malformed" in result:
        raise protocol.MalformedDecision(f"Biomni reply is not a decision: {result['malformed']}", tokens)
    problem = protocol.decision_problem(result["decision"])
    if problem:
        raise protocol.MalformedDecision(f"Biomni {problem}", tokens)
    return result["decision"], usage


def trace_tokens(trace: Path, model: str) -> int:
    """The tokens of a kept round trace: a decision trace must still hold a valid decision; a failed round's only
    valid usage."""
    if Path(trace).name.startswith("failed_"):
        return stream_usage(trace, model)["total_tokens"]
    return response(trace, model)[1]["total_tokens"]


def invoke(root, prompt, model, settings, deadline, index, images=(), marker=None, marker_env=None):
    """One round; the driver's result (messages included) in root/traces. `marker`: the stage root the outer watchdog
    scans (named by `marker_env`, else not exported)."""
    route(model)
    try:
        unchanged = read_json(settings) == _settings(model)
    except (ValueError, OSError):
        unchanged = False
    if not unchanged:
        raise ValueError("Biomni configuration changed")
    traces = Path(root) / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    trace = traces / f"round_{index:03d}.json"
    if trace.exists() or trace.with_suffix(".stderr").exists():
        raise ValueError("Biomni round output already exists")
    with tempfile.TemporaryDirectory(prefix="cpj-biomni-") as temporary:
        workspace = Path(temporary)
        home = workspace / "home"
        home.mkdir(mode=0o700)
        attachments = []
        for i, path in enumerate(images):
            attachments.append(workspace / f"evidence_{i:03d}.png")
            shutil.copy2(path, attachments[-1])
        request, result = workspace / "request.json", workspace / "result.json"
        anthropic = _provider(model) == "anthropic"
        write_json(request, {"prompt": prompt, "images": [str(p) for p in attachments], "schema": protocol.SCHEMA,
                             "model": model, "source": "Anthropic" if anthropic else "Custom",
                             "base_url": None if anthropic else protocol.BASE_URL,
                             "timeout_seconds": max(1, int(deadline - time.time()))})
        # The driver's stdout is dropped: whatever Biomni prints after construction is not a record.
        rc, _, stderr = protocol._process([str(python()), "-I", str(DRIVER), str(request), str(result)], workspace,
                                          _env(home, run=marker or root, marker_env=marker_env, model=model),
                                          deadline, name="Biomni")
        trace.with_suffix(".stderr").write_text(stderr, encoding="utf-8")
        if rc or not result.is_file():
            raise RuntimeError(f"Biomni driver failed (exit {rc}); see {trace.with_suffix('.stderr')}")
        trace.write_text(protocol._redact(result.read_text(encoding="utf-8")), encoding="utf-8")
    try:
        decision, usage = response(trace, model)
    except protocol.MalformedDecision:
        # A malformed reply keeps its trace under failed_round_*, outside the decision traces.
        for path in (trace, trace.with_suffix(".stderr")):
            if path.exists():
                path.replace(path.with_name("failed_" + path.name))
        raise
    usage["biomni_version"] = BIOMNI_VERSION
    return decision, usage, None


def _version(deadline) -> str:
    """Biomni imports in its own interpreter and is the audited version (metadata and, if set, __version__)."""
    probe = ("import json, importlib.metadata as m, biomni.agent.a1, biomni; "
             "print(json.dumps([m.version('biomni'), getattr(biomni, '__version__', None)]))")
    with tempfile.TemporaryDirectory(prefix="cpj-biomni-check-") as temporary:
        home = Path(temporary)
        rc, out, _ = protocol._process([str(python()), "-I", "-c", probe], home, _env(home), deadline, name="Biomni")
    try:
        metadata, attribute = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        metadata = attribute = None
    if rc or metadata != BIOMNI_VERSION or attribute not in (None, BIOMNI_VERSION):
        raise RuntimeError(f"Biomni requires audited version {BIOMNI_VERSION} in {python()}")
    return metadata


def preflight(root, model):
    """The Codex arm's check: the audited Biomni, then a plain finish that names the attached image's colour."""
    route(model)
    root = Path(root)
    settings = setup(root / "auth", model)
    deadline = time.time() + 180
    version = _version(deadline)
    from PIL import Image
    preview = root / "preflight.png"
    Image.new("RGB", (32, 32), (0, 255, 0)).save(preview)
    decision, usage, _ = invoke(root, 'Connectivity check. Return action="finish", request_json="{}". '
                               'Inspect the attached image and set reason to '
                               'its dominant color, using only the lowercase English color name.',
                               model, settings, deadline, 0, images=[preview])
    if decision["action"] != "finish" or json.loads(decision["request_json"]) != {}:
        raise ValueError("unexpected Biomni preflight response")
    if decision["reason"].strip().lower() != "green":
        raise RuntimeError("Biomni image preflight failed: attached image was not identified")
    return {**usage, "biomni_version": version, "preflight_tokens": usage["total_tokens"],
            "image_preflight": "passed", "key_variable": "LLM_API_KEY", "policy": POLICY,
            "evidence_policy": protocol.EVIDENCE_POLICY}
