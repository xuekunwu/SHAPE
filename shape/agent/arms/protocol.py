"""The agent-arm stage protocol shared by every backend (codex, claude, biomni).

A backend answers one decision round: given SHAPE's task, inputs, tool levers and call budget, it returns a decision
of `SCHEMA` (action, request_json, reason); the stage orchestration (benchmarks/full_CPJUMP1/harness/codex_arm.py
StageRun) runs the requested tool calls. This module holds only what is backend-agnostic: the decision schema and its
validation, the subprocess launcher and its redaction, the two reasoning efforts, the gateway base URL, the evidence
policy, a small atomic JSON io, and the `Backend` interface each backend module satisfies. It imports nothing from any
benchmark, so a backend can live under shape/ and be reused across datasets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Protocol

# Gateway base URL: defaults to the Stanford gateway, but an operator can point every agent
# arm at any OpenAI-compatible gateway by exporting LLM_BASE_URL (e.g. the repo .env). The
# route check in each backend then simply requires LLM_BASE_URL/LLM_API_BASE to agree with this
# value, so a local gateway is honoured without weakening the "creds must match the configured
# route" guard.
BASE_URL = (os.environ.get("LLM_BASE_URL") or os.environ.get("LLM_API_BASE")
            or "https://llm-gateway.example.edu/v1").rstrip("/")
ACTIONS = ("run", "inspect", "handoff", "finish")
# The payload is a JSON string: image IDs are dynamic keys, which strict structured output cannot declare.
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"action": {"type": "string", "enum": list(ACTIONS)},
                         "request_json": {"type": "string"}, "reason": {"type": "string"}},
          "required": ["action", "request_json", "reason"]}
EFFORTS = ("minimal", "low", "medium", "high", "xhigh")       # the values both reasoners take
MAX_INSPECT = 16
EVIDENCE_POLICY = {"images": "agent_inspect_only", "max_images_per_round": MAX_INSPECT,
                   "max_attachment_bytes": 18 * 1024 * 1024,
                   "tiff": "png_rendering_channels_side_by_side_p1_p99",
                   "feedback": "tool_returned_dict_and_tool_written_files"}


def write_json(path: Path, value) -> None:
    """Atomic JSON write (temp-and-replace, with the Windows retry); backends write plain dicts, so there is no
    json_value normalisation here (the CPJUMP1 harness keeps its own in data.py for structured diagnostics)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    for attempt in range(5):     # Windows refuses to replace a file another process briefly holds open
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _redact(text):
    secret = os.environ.get("LLM_API_KEY")
    return text.replace(secret, "[REDACTED]") if secret else text


class MalformedDecision(ValueError):
    """The agent's reply in a well-formed event stream is not a valid decision: handled as a rejected request (the
    session goes on), not a basic LLM failure. `tokens`: the round's reported usage."""

    def __init__(self, message, tokens=0):
        super().__init__(message)
        self.tokens = tokens


class RoundTimeout(RuntimeError):
    """An agent process exceeded its deadline (the round's cap, or the arm's when that is earlier)."""


def _process(cmd, cwd, env, deadline, prompt=None, name="Codex"):
    """(returncode, stdout, stderr) of one agent process, both redacted; `name`: the agent in the error messages."""
    remaining = deadline - time.time()
    if remaining <= 0:
        raise RuntimeError(f"{name} exceeded its deadline")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace",
                            start_new_session=os.name == "posix")
    try:
        out, err = proc.communicate(prompt, timeout=remaining)
    except subprocess.TimeoutExpired:
        raise RoundTimeout(f"{name} exceeded its deadline") from None
    finally:
        # Reap descendants even when the leader already exited with pipes open.
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif proc.poll() is None:
            proc.kill()
        proc.communicate()
    return proc.returncode, _redact(out), _redact(err)


def decision_problem(decision) -> str | None:
    """Why a reply is not a valid decision of SCHEMA (None: it is one); the checks `codex.response` makes."""
    if not isinstance(decision, dict) or set(decision) != set(SCHEMA["required"]):
        return "structured tool request is missing or malformed"
    if any(not isinstance(v, str) for v in decision.values()) or decision["action"] not in ACTIONS:
        return "invalid decision types"
    if not decision["reason"].strip():
        return "decision must give a reason"
    try:
        json.loads(decision["request_json"])
    except ValueError:
        return "decision JSON is malformed"
    return None


class Backend(Protocol):
    """The interface every agent-arm backend MODULE satisfies (shape.agent.arms.{codex,claude,biomni}); the stage
    orchestration dispatches each round through it. The backends are modules, not instances, so `MODEL`/`POLICY` are
    module constants and the methods are module-level functions -- this Protocol documents that contract (it is not
    runtime-enforced). `route`/`setup`/`invoke`/`preflight`/`response`/`stream_usage` carry the round."""

    MODEL: str
    POLICY: str

    def route(self, model: str) -> None: ...
    def setup(self, root: Path, model: str, **options) -> Path: ...
    def invoke(self, root, prompt, model, settings, deadline, index, **kwargs): ...
    def preflight(self, root, model) -> dict: ...
    def response(self, trace: Path, model: str): ...
    def stream_usage(self, trace: Path, model: str) -> dict: ...
