"""Prompt-template convention for VRR (task 032): `shape/agent/prompts/<tool>/
{verify,recommend}.md` -- extracted from the tool's inline prompt-building code, so
the prompt text is versioned/reviewable on its own instead of buried in a Python
f-string. Loaded, formatted with the caller's placeholders (e.g. `{metrics_json}`,
`{card_section}`), and concatenated exactly as the pre-extraction code built the
same string (see `llm_recommend._build_prompt`'s byte-parity test).

A tool that has no LLM-based Reconcile phase (the current architecture keeps
Reconcile deterministic, see `base.classify_lever` -- vrr.md sec 3 "Reconcile is the
verifier's measurement, reused, not a new model") has no `reconcile.md`; this loader
does not assume every one of the three template names exists for every tool.
"""

from __future__ import annotations

import os
from functools import lru_cache

_PROMPTS_DIR = os.path.dirname(__file__)


@lru_cache(maxsize=None)
def load_prompt(tool: str, name: str) -> str:
    """Raw text of `shape/agent/prompts/<tool>/<name>.md`, or "" if it does not exist."""
    path = os.path.join(_PROMPTS_DIR, tool, f"{name}.md")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()
