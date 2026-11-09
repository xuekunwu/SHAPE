"""Agent-arm backends shared across benchmarks.

The agent arms share one stage protocol (`protocol`); only the backend answering each round differs. A backend module
provides MODEL, POLICY, route(model), setup(root, model), invoke(root, prompt, model, settings, deadline, index,
images=(), marker=None) and preflight(root, model). `get_backend(arm)` returns the backend module for an arm; the
registry (AGENT_ARMS, ARM_MODELS, ARM_NAMES) names the arms, their exact model IDs and display names. The protocol
symbols are re-exported so a stage orchestration imports everything it needs from `shape.agent.arms`.
"""
from __future__ import annotations

import importlib

from .protocol import (BASE_URL, EFFORTS, EVIDENCE_POLICY, MAX_INSPECT, SCHEMA, Backend, MalformedDecision,
                       RoundTimeout, _process, _redact, decision_problem, read_json, write_json)

# arm -> backend module; imported lazily (a backend pulls in heavy clients only when its arm runs).
BACKENDS = {"codex": "shape.agent.arms.codex", "claude": "shape.agent.arms.claude",
            "biomni": "shape.agent.arms.biomni"}
AGENT_ARMS = ("codex", "claude", "biomni")
ARM_MODELS = {"codex": "gpt-5.6-luna", "claude": "claude-sonnet-5", "biomni": "gpt-5.6-luna"}
ARM_NAMES = {"codex": "Codex", "claude": "Claude Code", "biomni": "Biomni"}


def get_backend(arm: str):
    """The backend module of an agent arm."""
    if arm not in BACKENDS:
        raise ValueError(f"unknown agent arm {arm!r}")
    return importlib.import_module(BACKENDS[arm])


backend = get_backend          # the stage orchestration's existing name for the registry lookup

__all__ = ["BACKENDS", "AGENT_ARMS", "ARM_MODELS", "ARM_NAMES", "get_backend", "backend",
           "Backend", "SCHEMA", "decision_problem", "_process", "_redact", "MalformedDecision",
           "RoundTimeout", "BASE_URL", "EVIDENCE_POLICY", "EFFORTS", "MAX_INSPECT", "read_json", "write_json"]
