"""The SHAPE arm at each reasoning-effort level on one prepared scenario.

The question: how sensitive SHAPE's batch-correction tuning is to the reasoner's effort. One SHAPE-arm run per level
(`EFFORTS`, the levels gpt-5.6-luna accepts through the gateway), each scored by `score.final` against the same
control stage (raw, harmony, scanorama; computed once); nothing else changes between the runs. The level reaches the
engine through SHAPE_REASONING_EFFORT, which `shape/engine/openai.py` reads on every request; every level gets the
same per-reply token cap (`max_tokens`, reasoning included; the engine's default of 4000 cuts high-effort replies)
and request timeout (SHAPE_LLM_TIMEOUT). A level whose RESULT.json exists is kept (the resume point).

Layout of OUT:
    shape_<effort>/          the SHAPE arm's run dir (shape_arm.run + score.final), plus EFFORT.json and
                             logs/reasoner_calls.jsonl (one line per reply: usage, finish, error)
    SWEEP_REPORT.json/.csv/.md   one row per level
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

EFFORTS = ("none", "low", "medium", "high", "xhigh")
MAX_TOKENS = 32000
TIMEOUT_SECONDS = 900
KEYS = ("overall", "batch_mean", "bio_mean", "nonrep_mean_map", "negcon_mean_map")


class CappedEngine:
    """The engine with a per-reply token cap (unless the caller sets one) that records every reply's usage to
    `log_path`: tokens (reasoning included when the API reports them) and an error reply (e.g. the token limit)."""

    def __init__(self, engine, max_tokens: int, log_path: Path):
        self.engine, self.max_tokens, self.log_path = engine, max_tokens, log_path

    def generate(self, content, *args, **kwargs):
        kwargs.setdefault("max_tokens", self.max_tokens)
        started = time.time()
        error = None
        try:
            response = self.engine.generate(content, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- recorded, then re-raised unchanged
            response, error = None, f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            usage = response.get("usage") if isinstance(response, dict) else None
            if error is None and isinstance(response, dict) and response.get("error") and not response.get("content"):
                error = str(response.get("error"))[:300]
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"seconds": round(time.time() - started, 2), "max_tokens": kwargs["max_tokens"],
                                         "usage": usage, "error": error}, default=str) + "\n")
        return response

    def __getattr__(self, name):
        return getattr(self.engine, name)


def _ensure_control(prep: Path) -> dict:
    """The control stage (raw, harmony, scanorama), computed when missing; a stale one is refused (as `suite`)."""
    from benchmarks.batch_correction_JUMP.runners import control as control_mod

    path = prep / control_mod.CONTROL_DIR / "CONTROL.json"
    if not path.exists():
        control_mod.control(prep)
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("provenance") != control_mod.provenance():
        raise SystemExit(f"CONTROL_STALE: move {path.parent} aside and rerun with the current code")
    if tuple(existing.get("arms", {})) != ("raw", "harmony", "scanorama"):
        raise SystemExit(f"CONTROL_ROSTER_STALE: move {path.parent} aside and rerun controls")
    return existing


def reply_stats(log_path: Path) -> dict:
    """Per-reply usage of one level's run: replies, errors, reasoning and completion tokens (mean / max)."""
    rows = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()] \
        if log_path.exists() else []
    usages = [r["usage"] for r in rows if isinstance(r.get("usage"), dict)]

    def series(key):
        return [u[key] for u in usages if isinstance(u.get(key), int)]

    reasoning, completion = series("reasoning_tokens"), series("completion_tokens")
    return {"replies": len(rows), "errors": sum(1 for r in rows if r.get("error")),
            "token_limit_errors": sum(1 for r in rows if "token_limit" in str(r.get("error"))),
            "reasoning_tokens_mean": round(sum(reasoning) / len(reasoning), 1) if reasoning else None,
            "reasoning_tokens_max": max(reasoning) if reasoning else None,
            "completion_tokens_max": max(completion) if completion else None,
            "reply_seconds_mean": round(sum(r["seconds"] for r in rows) / len(rows), 2) if rows else None}


def row(effort: str, run_dir: Path) -> dict:
    """One level's readouts: the test-view scores, the deltas against harmony and raw (+- 2SE), cost and stop."""
    result = json.loads((run_dir / "RESULT.json").read_text(encoding="utf-8"))
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text(encoding="utf-8"))
    test = result["scores"]["agent"]["test"]
    out = {"effort": effort, "decision": result["decision"], "from_call": result["from_call"],
           "non_default_parameters": result["non_default_parameters"],
           **{f"test_{k}": test.get(k) for k in KEYS}}
    for name in ("agent_vs_harmony", "agent_vs_raw"):
        band = result["comparisons_test"][name]["overall"]
        out[f"{name}_overall"], out[f"{name}_overall_2se"] = band["delta"], band["band_2se"]
    usage, tokens = result["usage"], agent.get("tokens") or {}
    out.update(rounds=agent.get("n_rounds"), stopped_because=agent.get("stopped_because"),
               tool_calls=usage.get("tool_calls_finished"), wall_seconds=round(agent["ended"] - agent["started"], 1),
               total_tokens=tokens.get("total_tokens"), reasoning_tokens=tokens.get("reasoning_tokens"),
               best_calibration_overall=(usage.get("best_calibration_overall") or {}).get("value"),
               **reply_stats(run_dir / "logs" / "reasoner_calls.jsonl"))
    return out


def run_sweep(prep: Path, output: Path, *, model: str, max_rounds: int, noise_band: float, cores: int | None = None,
              efforts=EFFORTS, max_tokens: int = MAX_TOKENS, timeout: int = TIMEOUT_SECONDS,
              engine_factory=None) -> dict:
    """Every level in `efforts`, one after the other, then the sweep report. `engine_factory()` builds a fresh
    engine (default `build_vision_engine(model)`), after SHAPE_LLM_TIMEOUT and SHAPE_REASONING_EFFORT are set."""
    from benchmarks.batch_correction_JUMP.runners import score
    from benchmarks.batch_correction_JUMP.runners.shape_arm import run as run_shape

    unknown = [e for e in efforts if e not in EFFORTS]
    if unknown:
        raise SystemExit(f"unknown effort levels {unknown}; use {EFFORTS}")
    if engine_factory is None:
        from shape.agent.verification.vision_engine import build_vision_engine

        def engine_factory():
            return build_vision_engine(model)
    output.mkdir(parents=True, exist_ok=True)
    _ensure_control(prep)
    saved = {k: os.environ.get(k) for k in ("SHAPE_REASONING_EFFORT", "SHAPE_LLM_TIMEOUT")}
    try:
        for effort in efforts:
            run_dir = output / f"shape_{effort}"
            if (run_dir / "RESULT.json").exists():
                print(f"[effort-sweep] {effort}: recorded, kept", flush=True)
                continue
            if run_dir.exists():                 # a level killed mid-run starts again from an empty dir
                import shutil
                shutil.rmtree(run_dir)
            (run_dir / "logs").mkdir(parents=True)
            os.environ["SHAPE_REASONING_EFFORT"], os.environ["SHAPE_LLM_TIMEOUT"] = effort, str(timeout)
            engine = engine_factory()
            if engine is None:
                raise SystemExit(f"Luna engine unavailable for {model}")
            (run_dir / "EFFORT.json").write_text(json.dumps(
                {"reasoning_effort": effort, "max_tokens": max_tokens, "timeout_seconds": timeout, "model": model},
                indent=1), encoding="utf-8")
            print(f"[effort-sweep] {effort}: start {time.strftime('%H:%M:%S')}", flush=True)
            run_shape(prep, run_dir, model=model, max_rounds=max_rounds, noise_band=noise_band, cores=cores,
                      engine=CappedEngine(engine, max_tokens, run_dir / "logs" / "reasoner_calls.jsonl"))
            score.final(run_dir, prep)
            print(f"[effort-sweep] {effort}: done {time.strftime('%H:%M:%S')}", flush=True)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return write_report(output, efforts, max_tokens=max_tokens, timeout=timeout, model=model)


def write_report(output: Path, efforts=EFFORTS, *, max_tokens: int, timeout: int, model: str) -> dict:
    rows = [row(e, output / f"shape_{e}") for e in efforts if (output / f"shape_{e}" / "RESULT.json").exists()]
    report = {"model": model, "max_tokens": max_tokens, "timeout_seconds": timeout, "efforts": list(efforts),
              "rows": rows,
              "rule": "one SHAPE-arm run per level (n = 1); test view; deltas vs harmony / raw on the paired plate "
                      "halves (+- 2SE); every level the same control, token cap and timeout"}
    (output / "SWEEP_REPORT.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if rows:
        with open(output / "SWEEP_REPORT.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
                             for r in rows)

    def f(value, spec=".4f"):
        return "n/a" if value is None else format(value, spec)

    lines = [f"# SHAPE arm by reasoning effort ({model}; cap {max_tokens} tokens per reply, timeout {timeout} s)", "",
             "| effort | overall | batch | bio | nonrep mAP | vs harmony (+-2SE) | rounds | calls | stop | tokens | "
             "reasoning tokens / reply (max) | errors | wall (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['effort']} | {f(r['test_overall'])} | {f(r['test_batch_mean'])} | {f(r['test_bio_mean'])} | "
            f"{f(r['test_nonrep_mean_map'])} | {f(r['agent_vs_harmony_overall'], '+.4f')} +- "
            f"{f(r['agent_vs_harmony_overall_2se'])} | {r['rounds']} | {r['tool_calls']} | {r['stopped_because']} | "
            f"{r['total_tokens']} | {f(r['reasoning_tokens_mean'], '.0f')} ({r['reasoning_tokens_max']}) | "
            f"{r['errors']} | {f(r['wall_seconds'], '.0f')} |")
    lines += ["", report["rule"] + "."]
    (output / "SWEEP_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
