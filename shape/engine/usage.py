"""Process-level LLM token accounting.

Each engine records its per-call `usage` here; a caller (e.g. a benchmark) reads the running
`total_tokens()` BEFORE and AFTER a unit of work to attribute tokens to it -- e.g. one image's
`solve()`. Process-global on purpose: engines are created internally by `solve()`, so the caller
cannot reach them, but it CAN read this counter's delta.
"""

from __future__ import annotations

import threading

_TOKEN_FIELDS = (
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "thinking_tokens",
)
_CORE_NORMALIZED_FIELDS = ("input_tokens", "output_tokens", "total_tokens")
_TOTAL = {**{key: 0 for key in _TOKEN_FIELDS}, "calls": 0}
# New normalized fields are omitted from snapshots until a provider actually reports
# them.  This distinguishes "not reported" from a real zero-token count.
_OBSERVED = set()
_FIELD_CALLS = {key: 0 for key in _TOKEN_FIELDS}

# Fail-fast health: how many engine calls have FAILED back-to-back (any single healthy
# response clears it). The solver aborts a run whose LLM is persistently unavailable -- e.g.
# every call 1301-blocked -- instead of thrashing to max_steps.
_CONSEC_FAILURES = 0
# Engines may be called from several threads at once (e.g. the crop mask check's concurrent montages).
_LOCK = threading.Lock()


def record(usage) -> None:
    """Add one engine call's usage dict ({prompt,completion,total}_tokens). Best-effort/no-op
    on anything malformed, so accounting never breaks a generate()."""
    if not isinstance(usage, dict):
        return
    # Treat legacy prompt/completion fields as exact aliases of input/output.  This
    # keeps old engines compatible with the normalized completeness signal without
    # estimating any provider value (especially never total_tokens).
    observed = dict(usage)
    if observed.get("input_tokens") is None and observed.get("prompt_tokens") is not None:
        observed["input_tokens"] = observed["prompt_tokens"]
    if observed.get("output_tokens") is None and observed.get("completion_tokens") is not None:
        observed["output_tokens"] = observed["completion_tokens"]
    values = {}
    for k in _TOKEN_FIELDS:
        if k not in observed or observed[k] is None:
            continue
        try:
            values[k] = int(observed[k])
        except (TypeError, ValueError, OverflowError):
            continue
    with _LOCK:
        for k, value in values.items():
            _TOTAL[k] += value
            _OBSERVED.add(k)
            _FIELD_CALLS[k] += 1
        _TOTAL["calls"] += 1


def total_tokens() -> int:
    return _TOTAL["total_tokens"]


def snapshot() -> dict:
    # Retain the original three counters for backward compatibility.  Optional
    # normalized counters appear only if observed, never as fabricated zeros.
    always = {"total_tokens", "prompt_tokens", "completion_tokens", "calls"}
    with _LOCK:
        result = {key: value for key, value in _TOTAL.items()
                  if key in always or key in _OBSERVED}
        calls = _TOTAL["calls"]
        # Machine-readable coverage lets the harness map a compatibility zero to null
        # whenever the provider did not report that field on every recorded call.
        result["usage_observed_fields"] = sorted(_OBSERVED)
        result["usage_field_observed_calls"] = dict(_FIELD_CALLS)
        result["usage_complete"] = bool(
            calls and all(_FIELD_CALLS[key] == calls for key in _CORE_NORMALIZED_FIELDS)
        )
    return result


def is_failure(out) -> bool:
    """Classify an engine ``generate()`` return as a FAILED call (for the fail-fast guard).

    Failure = ``None``, an ``{"error": ...}`` dict, or a ``{"content": ...}`` whose content is one
    of the engines' error sentinels: they return content starting with ``"Error in "`` when a
    structured parse AND its plain fallback both fail (e.g. a GLM ``1301`` content-filter block on
    both the ``.parse`` and ``.create`` calls). Anything else is a healthy response."""
    if out is None:
        return True
    if not isinstance(out, dict):
        return False
    if out.get("error"):
        return True
    content = out.get("content")
    return isinstance(content, str) and content.startswith("Error in ")


def note_call(ok: bool) -> None:
    """Record one engine call's health: a healthy response clears the failure streak; a failed
    call extends it. The solver reads ``consecutive_failures()`` to stop a run whose LLM is
    persistently unavailable rather than looping to max_steps."""
    global _CONSEC_FAILURES
    with _LOCK:
        _CONSEC_FAILURES = 0 if ok else _CONSEC_FAILURES + 1


def consecutive_failures() -> int:
    return _CONSEC_FAILURES


def absorb(other: dict) -> None:
    """Add another process's `snapshot()` (e.g. a worker process's calls) to this process's totals."""
    with _LOCK:
        for k in _TOKEN_FIELDS:
            if isinstance(other.get(k), int):
                _TOTAL[k] += other[k]
        _TOTAL["calls"] += int(other.get("calls") or 0)
        _OBSERVED.update(k for k in other.get("usage_observed_fields") or () if k in _FIELD_CALLS)
        for k, n in (other.get("usage_field_observed_calls") or {}).items():
            if k in _FIELD_CALLS:
                _FIELD_CALLS[k] += int(n)


def reset() -> None:
    global _CONSEC_FAILURES
    for k in _TOTAL:
        _TOTAL[k] = 0
    _OBSERVED.clear()
    for k in _FIELD_CALLS:
        _FIELD_CALLS[k] = 0
    _CONSEC_FAILURES = 0
