"""`harness table`: the BIABench-style summary over run directories (outcome +- s.d. over repeated
runs, run-to-run range, end states, median cost, the paired difference to the control, the
variance decomposition, and the completion claims of delivered runs with a low outcome).

A configuration = (arm kind control | agent | shape, its model, prep, metadata level, GPU slug). Each run dir
gives RESULT.json (the scorer) and PROCESS.json (harness/process.py; computed here when absent).
Fields that a run does not have are empty, never 0.
Writes TABLE.md, table.csv (one row per configuration), runs.csv (one row per run) and pareto.png.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LOW_NORMALIZED = 0.1           # a delivered run with normalized_primary below this has a low outcome
LOW_ABOVE_CHANCE = 0.05        # else: primary below the chance reference + this
CONFIG_KEYS = ("arm", "model", "prep", "metadata", "gpu")


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _get(d, *keys):
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def _num(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def run_row(run: Path) -> dict:
    """One run's numbers (RESULT.json, PROCESS.json -- computed when absent)."""
    from .process import process
    from .score import CROSS, PRIMARY

    run = Path(run)
    result = _json(run / "RESULT.json") or {}
    report = _json(run / "PROCESS.json")
    process_from = "PROCESS.json"
    if not isinstance(report, dict):
        report, process_from = process(run), "computed by table"
    meta = _json(run / "RUN_META.json") or {}
    agent_run = _json(run / "AGENT_RUN.json") or {}
    agent = bool(report.get("agent_run")) and result.get("arm") != "control"
    arm = "shape" if report.get("arm") == "shape" else "agent" if agent else "control"
    gpu = meta.get("gpu_slug") or _get(result, "control_slug", "slug") or (None if arm != "control" else run.name)
    e1 = report.get("E1_cost") or {}
    llm = e1.get("llm") if isinstance(e1.get("llm"), dict) and e1["llm"].get("applicable") is not False else {}
    tokens = llm.get("tokens") or {}
    split = {k: _num(tokens.get(k)) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
    tokens_in = sum(v for v in split.values() if v is not None) if any(v is not None for v in split.values()) else None
    end = report.get("end_state") or {}
    claim = report.get("claims") if isinstance(report.get("claims"), dict) else {}
    reference = result.get("reference") if isinstance(result.get("reference"), dict) else {}
    wall = _num(e1.get("wall_hours"))
    return {
        "run": run.name, "path": str(run), "arm": arm,
        "model": (meta.get("model") or agent_run.get("model") or "-") if arm != "control" else "-",
        "prep": result.get("prep_name") or meta.get("prep_name") or "?",
        "metadata": result.get("metadata") or meta.get("metadata") or "?", "gpu": gpu or "?",
        "score_status": result.get("status"), "primary": _num(result.get(PRIMARY)),
        "cross_source": _num(result.get(CROSS)), "normalized_primary": _num(reference.get("normalized_primary")),
        "chance_primary": _num(_get(reference, "chance", "primary")),
        "fine_primary": _num(result.get(f"{PRIMARY}_fine")),
        "delta_vs_control": _num(_get(result, "comparison", "delta")),
        "end_state": end.get("end_state"), "timed_out": bool(end.get("timed_out")), "refused": bool(end.get("refused")),
        "claimed_complete": claim.get("claimed_complete") if claim.get("applicable") is not False else None,
        "wall_minutes": wall * 60 if wall is not None else None,
        "tokens_in": tokens_in, **{k: split[k] for k in split}, "tokens_out": _num(tokens.get("output_tokens")),
        "usd_estimate": _num(llm.get("total_cost_usd")),
        "gpu_hours": _num(_get(e1, "tools", "gpu_hours")), "tool_calls": _num(_get(e1, "tools", "n_calls")),
        "process_from": process_from}


def _stats(values) -> dict:
    v = [x for x in values if x is not None]
    return {"mean": float(np.mean(v)) if v else None, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "min": float(min(v)) if v else None, "max": float(max(v)) if v else None, "n": len(v)}


def _median(values) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def _low_outcome(row: dict) -> bool | None:
    if row["normalized_primary"] is not None:
        return row["normalized_primary"] < LOW_NORMALIZED
    if row["primary"] is not None and row["chance_primary"] is not None:
        return row["primary"] < row["chance_primary"] + LOW_ABOVE_CHANCE
    return None


def summarize(rows: list[dict]) -> dict:
    by_config: dict[tuple, list[dict]] = {}
    for row in rows:                    # plain dicts: a pandas round trip would turn a missing value into NaN
        by_config.setdefault(tuple(str(row[k]) for k in CONFIG_KEYS), []).append(row)
    configs = []
    for key, g in sorted(by_config.items()):
        configs.append({
            **dict(zip(CONFIG_KEYS, key)), "n_runs": len(g),
            "primary": _stats([r["primary"] for r in g]), "cross_source": _stats([r["cross_source"] for r in g]),
            "normalized_primary": _stats([r["normalized_primary"] for r in g]),
            "fine_primary_mean": _stats([r["fine_primary"] for r in g])["mean"],
            "delivered": sum(r["end_state"] == "delivered" for r in g),
            "no_deliverable": sum(r["end_state"] == "no_deliverable" for r in g),
            "crashed": sum(r["end_state"] == "crashed" for r in g), "timed_out": sum(r["timed_out"] for r in g),
            "median_wall_minutes": _median([r["wall_minutes"] for r in g]),
            "median_tokens_in": _median([r["tokens_in"] for r in g]),
            "median_input_tokens": _median([r["input_tokens"] for r in g]),
            "median_cache_read_tokens": _median([r["cache_read_input_tokens"] for r in g]),
            "median_cache_creation_tokens": _median([r["cache_creation_input_tokens"] for r in g]),
            "median_tokens_out": _median([r["tokens_out"] for r in g]),
            "median_usd_estimate": _median([r["usd_estimate"] for r in g]),
            "median_gpu_hours": _median([r["gpu_hours"] for r in g]),
            "median_tool_calls": _median([r["tool_calls"] for r in g]),
            "delta_vs_control": _stats([r["delta_vs_control"] for r in g])})
    # variance decomposition over the configurations with >= 2 scored runs
    groups = [[r["primary"] for r in g if r["primary"] is not None] for g in by_config.values()]
    groups = [g for g in groups if len(g) >= 2]
    variance = None
    if len(groups) >= 2:
        grand = float(np.mean([x for g in groups for x in g]))
        between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
        within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
        total = between + within
        variance = {"n_configurations": len(groups), "n_runs": sum(map(len, groups)), "ss_between": float(between),
                    "ss_within": float(within), "share_between": float(between / total) if total > 0 else None,
                    "share_within": float(within / total) if total > 0 else None,
                    "rule": "one-way ANOVA sums of squares of the primary over configurations with >= 2 scored runs"}
    low = [r for r in rows if r["arm"] == "agent" and r["end_state"] == "delivered" and _low_outcome(r)]
    unknown = [r for r in rows if r["arm"] == "agent" and r["end_state"] == "delivered" and _low_outcome(r) is None]
    return {"configurations": configs, "variance": variance,
            "claims_low_outcome": {"n_low_delivered": len(low), "n_claimed_complete": sum(bool(r["claimed_complete"]) for r in low),
                                   "runs": [r["run"] for r in low], "n_unclassified": len(unknown),
                                   "rule": f"delivered agent runs with normalized_primary < {LOW_NORMALIZED} (else primary "
                                           f"< chance + {LOW_ABOVE_CHANCE}); unclassified without either reference"}}


def _fmt(stats: dict | None, spec: str = ".3f") -> str:
    if not stats or stats.get("mean") is None:
        return "n/a"
    text = format(stats["mean"], spec) + (f" +- {format(stats['sd'], spec)}" if stats.get("sd") is not None else "")
    return text + (f" [{format(stats['min'], spec)}, {format(stats['max'], spec)}]" if stats.get("n", 0) > 1 else "")


def _v(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def markdown(summary: dict) -> str:
    lines = ["# full_HCMA summary table", "",
             "Outcome = mean +- s.d. [min, max] over runs. USD is Claude Code's list-price estimate (not gateway billing). "
             "D / N / C / T = delivered / no deliverable / crashed / timed out.", "",
             "| arm | model | prep | metadata | GPU | n | primary | cross-source | normalized | fine | D/N/C/T "
             "| wall min | tokens in / out | USD (est.) | GPU-h | tool calls | vs control |",
             "| " + " | ".join(["---"] * 17) + " |"]
    for c in summary["configurations"]:
        tokens = (f"{_v(c['median_tokens_in'], ',.0f')} / {_v(c['median_tokens_out'], ',.0f')}"
                  if c["median_tokens_in"] is not None else "n/a")
        lines.append(f"| {c['arm']} | {c['model']} | {c['prep']} | {c['metadata']} | {c['gpu']} | {c['n_runs']} | "
                     f"{_fmt(c['primary'])} | {_fmt(c['cross_source'])} | {_fmt(c['normalized_primary'])} | "
                     f"{_v(c['fine_primary_mean'])} | {c['delivered']}/{c['no_deliverable']}/"
                     f"{c['crashed']}/{c['timed_out']} | {_v(c['median_wall_minutes'], '.0f')} | {tokens} | "
                     f"{_v(c['median_usd_estimate'], '.2f')} | {_v(c['median_gpu_hours'], '.2f')} | "
                     f"{_v(c['median_tool_calls'], '.0f')} | {_fmt(c['delta_vs_control'], '+.3f')} |")
    variance = summary["variance"]
    lines += ["", "Variance decomposition: " + (
        f"{_v(variance['share_between'], '.2f')} of the primary's variance between configurations, "
        f"{_v(variance['share_within'], '.2f')} within (run to run), over {variance['n_runs']} runs in "
        f"{variance['n_configurations']} configurations." if variance else "n/a (needs >= 2 configurations with >= 2 runs).")]
    cl = summary["claims_low_outcome"]
    lines.append(f"Delivered agent runs with a low outcome: {cl['n_low_delivered']}, of which {cl['n_claimed_complete']} "
                 f"claimed completion ({cl['n_unclassified']} unclassified without a reference).")
    return "\n".join(lines) + "\n"


def pareto(summary: dict, path: Path) -> Path | None:
    """Outcome mean vs median USD (estimate) per configuration with a USD value."""
    points = [c for c in summary["configurations"]
              if c["primary"]["mean"] is not None and c["median_usd_estimate"] is not None]
    from .figures import _apply_nature_style, distinct_colours

    _apply_nature_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    colours = distinct_colours()
    for n, c in enumerate(points):
        label = f"{c['arm']} {c['model']} {c['prep']}/{c['metadata']}/{c['gpu']}"
        ax.errorbar(c["median_usd_estimate"], c["primary"]["mean"], yerr=c["primary"]["sd"] or 0, fmt="o", ms=3,
                    lw=0.5, capsize=1.5, color=colours[n % len(colours)], label=label)
    ax.set_xlabel("median USD per run (Claude Code estimate)")
    ax.set_ylabel("primary (mean +- s.d.)")
    if points:
        ax.legend(frameon=False, fontsize=5, loc="best")
    else:
        ax.text(0.5, 0.5, "no configuration with a USD estimate", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def table(runs: list[Path], out: Path, control_root: Path | None = None) -> dict:
    """Rows of `runs` (and of every control dir under control_root holding a RESULT.json); writes the outputs."""
    dirs = [Path(r) for r in runs]
    if control_root is not None:
        dirs += sorted(p.parent for p in Path(control_root).rglob("RESULT.json") if p.parent not in dirs)
    rows = [run_row(d) for d in dirs]
    summary = summarize(rows)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    flat = []
    for c in summary["configurations"]:
        row = {k: v for k, v in c.items() if not isinstance(v, dict)}
        for name in ("primary", "cross_source", "normalized_primary", "delta_vs_control"):
            row.update({f"{name}_{k}": v for k, v in c[name].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(out / "table.csv", index=False)
    (out / "TABLE.md").write_text(markdown(summary), encoding="utf-8")
    (out / "TABLE.json").write_text(json.dumps({**summary, "runs": rows}, indent=1, default=str), encoding="utf-8")
    pareto(summary, out / "pareto.png")
    return summary
