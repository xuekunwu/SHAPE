"""Agent-runner abstraction: launch a coding agent (Claude Code, Codex, ...) headless in an
ISOLATED working directory, capturing its full event trace (every tool call + reasoning).

A runner knows nothing about the benchmark -- it just runs a prompt in a workdir and records the
trace. The benchmark adapter builds the workdir + prompt and scores the deliverable afterwards.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    ok: bool
    agent: str
    workdir: str
    trace_path: str
    session_id: str | None = None          # for resuming in the outer feedback loop
    returncode: int | None = None
    error: str = ""
    meta: dict = field(default_factory=dict)


class AgentRunner:
    """Base class. Subclasses implement ``run`` (fresh turn) and ``resume`` (feedback turn)."""

    name = "agent"

    def __init__(self, api_key: str, model: str | None = None, timeout: int = 3600,
                 extra_env: dict | None = None):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.extra_env = extra_env or {}

    # --- interface ---------------------------------------------------------- #
    def run(self, prompt: str, workdir: Path, trace_path: Path) -> RunResult:
        """First turn: run ``prompt`` in ``workdir``; append the event stream to ``trace_path``."""
        raise NotImplementedError

    def resume(self, prompt: str, workdir: Path, trace_path: Path,
               session_id: str | None) -> RunResult:
        """Feedback turn: continue the SAME session with new ``prompt`` (the scored feedback)."""
        raise NotImplementedError

    # --- helpers ------------------------------------------------------------ #
    def _env(self) -> dict:
        env = dict(os.environ)
        env.update(self.extra_env)
        # Keep the agent's cwd-isolated view clean: never leak SHAPE's gateway creds.
        for k in ("LLM_API_KEY", "LLM_BASE_URL", "SHAPE_SAM_SERVER_URL"):
            env.pop(k, None)
        return env
