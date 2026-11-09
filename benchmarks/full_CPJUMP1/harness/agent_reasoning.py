"""Opt-in reasoning override for native agent backends; existing runs keep defaults."""
import os

EFFORT_ENV = "FULL_CPJUMP1_AGENT_REASONING_EFFORT"


def effort():
    value = os.environ.get(EFFORT_ENV)
    if value is not None and value not in ("low", "medium", "high"):
        raise ValueError(f"invalid {EFFORT_ENV}")
    return value
