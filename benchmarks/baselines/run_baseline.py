"""Run a coding-agent BASELINE (Claude Code / Codex) on a SHAPE benchmark, as a SHAPE peer.

This is the new launch command (NOT the raw `claude`/`codex`): it sandboxes the agent in an
isolated workdir OUTSIDE the SHAPE repo (only inputs + TASK.md -- never SHAPE's code), runs it
headless while saving the full tool-call/reasoning trace, scores the delivered masks/crops with
OUR copy of SHAPE's metrics, and drives an outer FEEDBACK LOOP: each iteration's score + critique
is fed back to the same agent session for another attempt. Keep-best across iterations.

    python -m benchmarks.baselines.run_baseline \
        --agent claude-code --benchmark segmentation \
        --data-root data/NeurIPS22-CellSeg --limit 5 --iters 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_REPO = Path(__file__).resolve().parents[2]
# Run workdirs live OUTSIDE the repo so the agent's cwd never sees SHAPE's code (cwd isolation).
_DEFAULT_RUN_ROOT = Path("/mnt/filesystem-r8/shape_baseline_runs")
_RESULTS = Path(__file__).resolve().parent / "results"


def _make_agent(name, model, timeout):
    if name == "claude-code":
        from benchmarks.baselines.agents.claude_code import ClaudeCodeRunner
        return ClaudeCodeRunner(api_key=os.environ["ANTHROPIC_API_KEY"], model=model, timeout=timeout)
    if name == "codex":
        from benchmarks.baselines.agents.codex import CodexRunner
        return CodexRunner(api_key=os.environ["OPENAI_API_KEY"], model=model, timeout=timeout)
    raise SystemExit(f"unknown agent: {name}")


def _make_adapter(name, data_root, sam_url=None):
    if name == "segmentation":
        from benchmarks.baselines.adapters.segmentation import SegmentationAdapter
        return SegmentationAdapter(data_root, sam_url=sam_url)
    if name == "cropper":
        from benchmarks.baselines.adapters.cropper import CropperAdapter
        return CropperAdapter(data_root)
    raise SystemExit(f"unknown benchmark: {name}")


def main():
    load_dotenv(_REPO / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, choices=["claude-code", "codex"])
    ap.add_argument("--benchmark", required=True, choices=["segmentation", "cropper"])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5, help="#items (images).")
    ap.add_argument("--iters", type=int, default=3, help="feedback-loop iterations (>=1).")
    ap.add_argument("--min-iters", type=int, default=3,
                    help="minimum iterations before an agent-written STOP is honored, so it "
                         "cannot bail out while the score is still improving. Default 3.")
    ap.add_argument("--model", default=None,
                    help="agent model. Default: claude-sonnet-5 (claude-code) / gpt-5.6-sol (codex).")
    ap.add_argument("--price-in", type=float, default=None, help="USD/1M input tokens (cost calc).")
    ap.add_argument("--price-out", type=float, default=None, help="USD/1M output tokens.")
    ap.add_argument("--price-cached", type=float, default=None, help="USD/1M cached-input tokens.")
    ap.add_argument("--timeout", type=int, default=5400, help="per-iteration agent timeout (s).")
    ap.add_argument("--run-root", type=Path, default=_DEFAULT_RUN_ROOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sam-url", default=None,
                    help="If set (segmentation), TASK.md tells the agent to CALL this Cellpose-SAM "
                         "service instead of installing cellpose (e.g. http://gpu-node.example.net:8137).")
    args = ap.parse_args()
    # per-agent model defaults: CC -> claude-sonnet-5, codex -> gpt-5.6-sol
    if args.model is None:
        args.model = "claude-sonnet-5" if args.agent == "claude-code" else "gpt-5.6-sol"

    adapter = _make_adapter(args.benchmark, args.data_root, sam_url=args.sam_url)
    agent = _make_agent(args.agent, args.model, args.timeout)
    items = adapter.select_items(args.limit, args.seed)
    if not items:
        raise SystemExit("no items selected")

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{args.agent}_{args.benchmark}_{ts}"
    run_dir = args.run_root / run_name
    workdir = run_dir / "workdir"
    traces = run_dir / "traces"
    workdir.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    adapter.prepare_workdir(workdir, items)

    print(f"[baseline] {run_name}: {len(items)} items, {args.iters} iters -> {run_dir}", flush=True)
    history, best, session_id = [], None, None
    for it in range(args.iters):
        trace_path = traces / f"iter{it}.jsonl"
        prompt = (adapter.task_prompt(workdir, items) if it == 0
                  else adapter.feedback_prompt(best["score"], history=history))
        res = (agent.run(prompt, workdir, trace_path) if it == 0
               else agent.resume(prompt, workdir, trace_path, session_id))
        session_id = res.session_id or session_id
        score = adapter.score(workdir, items)
        from benchmarks.baselines.metrics.usage import extract_usage, cost_usd
        usage = extract_usage(str(trace_path), args.agent)
        cost = cost_usd(args.model, usage, args.price_in, args.price_out, args.price_cached)
        rec = {"iteration": it, "agent_ok": res.ok, "returncode": res.returncode,
               "error": res.error[:500], "session_id": session_id,
               "trace": str(trace_path), "n_delivered": score.n_delivered,
               "n_expected": score.n_expected, "summary": score.summary,
               "usage": usage, "cost_usd": cost}
        history.append(rec)
        rec_full = {"score": score, **rec}
        # canonical optimization key: cropper -> "score" (retention-contamination); seg -> ap50
        key = score.summary.get("score")
        if key is None:
            key = score.summary.get("ap50", 0)
        if best is None or key > (best["key"]):
            best = {"key": key, "iteration": it, "score": score}
        # snapshot this iteration's deliverable
        snap = run_dir / f"deliver_iter{it}"
        if (workdir / "deliver").exists():
            shutil.copytree(workdir / "deliver", snap, dirs_exist_ok=True)
        print(f"[baseline] iter {it}: delivered {score.n_delivered}/{score.n_expected} | "
              f"{score.summary} | tokens in={usage['input_tokens']} out={usage['output_tokens']} "
              f"| cost={'$%.4f' % cost if cost is not None else 'n/a (no price)'}", flush=True)
        _dump(run_dir, args, run_name, items, history, best)
        # agent-decided early stop: honored only AFTER a minimum number of iterations, so
        # the agent cannot bail out after one lucky round while the score is still rising
        # (measured: CC stopped at iter1 still climbing 0.70->0.77). Before the floor, a
        # STOP is cleared and ignored so the loop keeps pushing.
        stop_hit = (workdir / "deliver" / "STOP").exists() or (workdir / "STOP").exists()
        if stop_hit and (it + 1) >= args.min_iters:
            print(f"[baseline] agent signalled STOP at iter {it} (>= min_iters "
                  f"{args.min_iters}); ending loop early.", flush=True)
            break
        elif stop_hit:
            for sp in (workdir / "deliver" / "STOP", workdir / "STOP"):
                if sp.exists():
                    sp.unlink()
            print(f"[baseline] STOP at iter {it} ignored (< min_iters {args.min_iters}); "
                  f"keep iterating.", flush=True)

    # keep the best iteration's per-item detail
    (run_dir / "best_per_item.json").write_text(
        json.dumps(best["score"].per_item, indent=2, default=str), encoding="utf-8")
    # SHAPE-parallel per-image deliver channel (images/<id>/{metrics.json, deliverable_seg.png})
    best_deliver = run_dir / f"deliver_iter{best['iteration']}"
    if best_deliver.exists():
        adapter.materialize(best_deliver, items, run_dir, best["score"])
    _RESULTS.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, _RESULTS / run_name, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("workdir"))
    print(f"[baseline] DONE. best iter={best['iteration']} {best['score'].summary} "
          f"-> results/{run_name}", flush=True)


def _dump(run_dir, args, run_name, items, history, best):
    tin = sum((h.get("usage") or {}).get("input_tokens", 0) for h in history)
    tout = sum((h.get("usage") or {}).get("output_tokens", 0) for h in history)
    costs = [h.get("cost_usd") for h in history if h.get("cost_usd") is not None]
    total_cost = round(sum(costs), 6) if costs else None
    (run_dir / "run_summary.json").write_text(json.dumps({
        "run": run_name, "agent": args.agent, "benchmark": args.benchmark,
        "model": args.model, "n_items": len(items), "iters": args.iters,
        "items": items, "history": history,
        "best_iteration": best["iteration"] if best else None,
        "best_summary": best["score"].summary if best else None,
        "total_input_tokens": tin, "total_output_tokens": tout, "total_cost_usd": total_cost,
    }, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
