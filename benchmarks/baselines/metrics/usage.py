"""Token-usage + cost extraction from agent traces (Claude Code stream-json, Codex --json).

Both agents report token usage in their event stream; we sum it per iteration and price it so a
baseline run self-reports cost, exactly like the SHAPE benchmark runs.
"""
from __future__ import annotations

import json

# (input, output, cached_input) USD per 1M tokens. Cached input = cache-read rate (~10% of input).
# Verify against provider pricing pages; override per run with --price-in/--price-out.
PRICES = {
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
    # gpt-5.6-sol / -luna are gateway models with no public list price -> pass --price-* to price them.
}


def extract_usage(trace_path: str, agent: str) -> dict:
    """Total {input_tokens, output_tokens, cached_input_tokens} for one iteration's trace, plus
    ``reported_cost_usd`` when the agent itself reports one (Claude Code's result event)."""
    tin = tout = tcached = 0
    reported_cost = None
    try:
        lines = open(trace_path, encoding="utf-8").read().splitlines()
    except Exception:
        return {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
                "reported_cost_usd": None}
    for line in lines:
        try:
            e = json.loads(line)
        except Exception:
            continue
        if agent == "claude-code":
            if e.get("type") == "result":               # final event: authoritative totals + cost
                u = e.get("usage") or {}
                tin = u.get("input_tokens", tin)
                tout = u.get("output_tokens", tout)
                tcached = u.get("cache_read_input_tokens", tcached)
                reported_cost = e.get("total_cost_usd", reported_cost)
        else:  # codex: sum per-turn usage
            if e.get("type") == "turn.completed":
                u = e.get("usage") or {}
                tin += u.get("input_tokens", 0)
                tout += u.get("output_tokens", 0)
                tcached += u.get("cached_input_tokens", 0)
    return {"input_tokens": int(tin), "output_tokens": int(tout),
            "cached_input_tokens": int(tcached), "reported_cost_usd": reported_cost}


def cost_usd(model: str, usage: dict, price_in=None, price_out=None, price_cached=None):
    """USD cost from usage. Explicit prices win; else the PRICES table; else None (tokens only).
    If the agent reported its own cost (Claude Code) and no table/override applies, use that."""
    tab = PRICES.get(model)
    pin = price_in if price_in is not None else (tab[0] if tab else None)
    pout = price_out if price_out is not None else (tab[1] if tab else None)
    if pin is None or pout is None:
        return usage.get("reported_cost_usd")
    pcached = price_cached if price_cached is not None else (tab[2] if tab else pin)
    tin, tout, tc = usage["input_tokens"], usage["output_tokens"], usage["cached_input_tokens"]
    non_cached = max(0, tin - tc)
    return round(non_cached / 1e6 * pin + tc / 1e6 * pcached + tout / 1e6 * pout, 6)
