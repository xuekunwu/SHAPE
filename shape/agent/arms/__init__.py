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
# Default model per arm (used when a run does not pass --model) and the full set each arm accepts.
# A single arm may run several reasoners (biomni: gpt-* on the gateway or claude-* on Anthropic-direct);
# ARM_ALLOWED is the membership check, ARM_MODELS[arm] the default. Each backend's route(model) enforces
# the same set at invoke time (the only gate in staged mode, where preflight is not run).
ARM_MODELS = {"codex": "gpt-6-sol", "claude": "claude-opus-5", "biomni": "gpt-6-sol"}
ARM_ALLOWED = {"codex": frozenset({"gpt-6-sol"}),
               "claude": frozenset({"claude-opus-5"}),
               "biomni": frozenset({"gpt-6-sol", "claude-opus-5", "zai-org/GLM-5.3-Flash",
                                    "deepseek-ai/DeepSeek-V4.1-Flash"})}
ARM_NAMES = {"codex": "Codex", "claude": "Claude Code", "biomni": "Biomni"}


def get_backend(arm: str):
    """The backend module of an agent arm."""
    if arm not in BACKENDS:
        raise ValueError(f"unknown agent arm {arm!r}")
    return importlib.import_module(BACKENDS[arm])


backend = get_backend          # the stage orchestration's existing name for the registry lookup

__all__ = ["BACKENDS", "AGENT_ARMS", "ARM_MODELS", "ARM_ALLOWED", "ARM_NAMES", "get_backend", "backend",
           "Backend", "SCHEMA", "decision_problem", "_process", "_redact", "MalformedDecision",
           "RoundTimeout", "BASE_URL", "EVIDENCE_POLICY", "EFFORTS", "MAX_INSPECT", "read_json", "write_json"]
