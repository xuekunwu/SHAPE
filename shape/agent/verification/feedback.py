"""Cross-stage feedback: turn ``escalate_upstream`` into an actual upstream rerun.

The per-stage VLRR loop (``step_verifier``) closes the loop WITHIN a stage. When a
stage instead localizes the fault to a PRIOR stage it returns
``action="escalate_upstream"`` with an ``upstream_target``. Without this module that
verdict is a dead end (the agent stops, the benchmark skips the image). Here we close
the loop ACROSS stages: nudge the named upstream stage's params, re-run it, then
re-run the dependent stage -- bounded by a rerun budget and anti-oscillation, exactly
like the within-stage loop but one level up.
"""

from __future__ import annotations

from typing import Dict, Optional

from .base import apply_bounded

# How to push an upstream stage to fix a fault a downstream stage localized to it.
# No automatic upstream rerun is defined for image loading: after removing the
# preprocessor tool, input-stage concessions are evidence for the planner rather
# than a parameterized tool retry.
_UPSTREAM_NUDGE: Dict[str, Dict[str, str]] = {}


def upstream_nudge(upstream_target: str) -> Dict[str, str]:
    """Directional param delta that pushes ``upstream_target`` toward fixing a fault a
    downstream stage localized to it. Empty when we have no defined remedy."""
    return dict(_UPSTREAM_NUDGE.get(upstream_target, {}))


def can_feedback(upstream_target: Optional[str]) -> bool:
    """Whether escalate_upstream(``upstream_target``) has a defined upstream remedy."""
    return bool(upstream_target) and bool(_UPSTREAM_NUDGE.get(upstream_target))


def nudge_params(params: dict, upstream_target: str) -> dict:
    """Apply the upstream nudge to an upstream stage's params (bounded/clamped)."""
    return apply_bounded(dict(params or {}), upstream_nudge(upstream_target))
