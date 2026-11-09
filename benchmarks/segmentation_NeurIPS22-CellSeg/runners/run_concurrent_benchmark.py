"""Concurrent full-pipeline benchmark for NeurIPS22-CellSeg (default 32-way).

Runs the COMPLETE SHAPE agent (``shape.solver.solve``) on every image, N images at a
time, and delivers three things per run:

1. **An enriched deliverable CSV** (``cell_image_metrics.csv``) — the usual instance
   metrics (AP50/AP75/mAP/precision/recall/F1/count-error) PLUS per-image **agentic
   activity**: how many times each tool was called, how many VLRR *verifier iteration
   rounds* happened (total and per stage), vision judgements, escalations, plan
   rejections, tokens, and the terminal ``stop_reason``.

2. **A structured per-image trace** (``images/<id>/trace.json``) — an ordered,
   step-numbered record where every step names each MODULE's input and output:
   Planner (tool selected + sub-goal + reasoning), Executor (command + params +
   result), Verifier (per-stage verdicts + VLRR trials). This is deliberately more
   detailed than a flat log: you can read exactly what each module saw and produced at
   step *k*.

3. **The delivered visualization** (``images/<id>/deliverable_seg.png``) — raw image,
   predicted instance mask, and prediction-vs-GT contour overlay (plus a crop montage if
   the pipeline ever emits crops; this segmentation-only arm normally does not).

Why processes, not threads: ``shape.engine.usage`` accounts tokens in a **process-global**
counter, so per-image token/'activity' attribution is only correct when each image runs in
its own process. Processes also isolate matplotlib and give real parallelism. CUDA-safe
``spawn`` start method is used. On a single GPU, 32 concurrent model loads can exhaust VRAM
— dial ``--workers`` down for GPU-bound runs; 32 is ideal for the LLM-latency-bound agentic
path.

Reuses the canonical scoring/activity helpers from ``run_shape_neurips_cellseg`` so metric
definitions never drift from the sequential runner.

Example
-------
    export SHAPE_BENCH_DATA_ROOT=/path/to/shape_benchmark
    export LLM_API_KEY=...            # provider key for the config's model
    python run_concurrent_benchmark.py --limit 64 --workers 32

    # Validate all the plumbing (CSV/trace/viz) with NO GPU/LLM/data:
    python run_concurrent_benchmark.py --dry-run --limit 8 --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

# This benchmark dir must be importable in BOTH the parent and every spawned worker so
# `import run_shape_neurips_cellseg` (the canonical scoring helpers) and `config` resolve.
_THIS_DIR = Path(__file__).resolve().parent
_BENCH = _THIS_DIR.parent          # benchmarks/segmentation_NeurIPS22-CellSeg
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

# Convenience list prices (USD per 1M tokens) so a run can self-report cost. These CHANGE —
# verify against OpenAI's official pricing page. Override any run with --price-in/--price-out
# (and optionally --price-cached). A model NOT in this table (e.g. a private gateway alias
# like gpt-5.6-luna) yields cost=null unless you pass explicit prices. Tuple = (input,
# output, cached_input) per 1M tokens.
_OPENAI_PRICES = {
    "gpt-4o": (2.50, 10.00, 1.25),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "gpt-4.1": (2.00, 8.00, 0.50),
    "gpt-4.1-mini": (0.40, 1.60, 0.10),
    "o3": (2.00, 8.00, 0.50),
    "o4-mini": (1.10, 4.40, 0.275),
}


def _resolve_prices(model, price_in, price_out, price_cached):
    """(input, output, cached) USD/1M — explicit flags win, else the table, else all None."""
    if price_in is not None and price_out is not None:
        return price_in, price_out, (price_cached if price_cached is not None else price_in)
    tab = _OPENAI_PRICES.get(model)
    if tab:
        return tab[0], tab[1], (price_cached if price_cached is not None else tab[2])
    return None, None, None


def _cost_usd(tin, tout, tcached, pin, pout, pcached):
    """OpenAI-style cost: cached input is billed at the (cheaper) cached rate, the rest of
    the input at the input rate, output at the output rate. None when prices are unknown."""
    if pin is None or pout is None:
        return None
    tin, tout, tcached = int(tin or 0), int(tout or 0), int(tcached or 0)
    non_cached_in = max(0, tin - tcached)
    rate_cached = pcached if pcached is not None else pin
    return round(non_cached_in / 1e6 * pin + tcached / 1e6 * rate_cached
                 + tout / 1e6 * pout, 6)


def _load_local_config():
    """Load THIS benchmark's runners/cellseg_config.py by explicit path (named apart from the
    `config.py` several benchmark dirs ship)."""
    spec = importlib.util.spec_from_file_location(
        "_neurips_concurrent_cfg", _THIS_DIR / "cellseg_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DEFAULT


# --------------------------------------------------------------------------- #
# Per-image agentic-activity extraction (tool calls + verifier iteration rounds)
# --------------------------------------------------------------------------- #
def extract_tool_activity(memory, verification_summary, nb) -> dict:
    """How many times each tool ran, and how many VLRR *verifier iteration rounds* fired.

    - ``tool_calls`` counts every EXECUTED planner action per tool (plan-rejected /
      not-executed steps are excluded, but cross-stage-feedback re-runs are counted —
      they are real tool executions).
    - ``verifier_iterations_total`` sums the within-stage VLRR trials across every stage
      (this is "how many verify->retune rounds happened"); ``verifier_rounds_by_stage``
      breaks it down per stage. Reuses the sequential runner's ``_iter_stages`` so the
      accounting matches ``run_summary.json``.
    """
    actions = memory.get_actions(llm_safe=False) if hasattr(memory, "get_actions") else []
    tool_calls: dict = {}
    n_executed = 0
    plan_rejections = 0
    for a in actions:
        cmd = str(a.get("command") or "")
        tool = a.get("tool_name")
        if cmd.startswith("[plan-rejected"):
            plan_rejections += 1
            continue
        if cmd.startswith("[not executed"):
            continue
        if tool:
            tool_calls[tool] = tool_calls.get(tool, 0) + 1
            n_executed += 1

    rounds_by_stage: dict = {}
    retunes_by_stage: dict = {}
    verifier_iters = 0
    retunes_total = 0
    for stage in nb._iter_stages(verification_summary):
        n = stage.get("n_trials")
        if n is None:
            n = len(stage.get("trials") or [])
        n = int(n or 0)
        st = stage.get("stage") or "?"
        rounds_by_stage[st] = rounds_by_stage.get(st, 0) + n
        verifier_iters += n
        # a RETUNE is a re-run AFTER the mandatory initial trial: max(0, n_trials - 1).
        # The first trial is the unavoidable tool call, not a retune decision.
        r = max(0, n - 1)
        retunes_by_stage[st] = retunes_by_stage.get(st, 0) + r
        retunes_total += r

    summary = nb._summarize_agent_activity(verification_summary)
    return {
        "n_steps_executed": n_executed,
        "tool_calls": tool_calls,
        "tool_calls_total": sum(tool_calls.values()),
        "segmenter_calls": tool_calls.get("Cell_Segmenter_Tool", 0),
        "plan_rejections": plan_rejections,
        "verifier_iterations_total": verifier_iters,   # Σ n_trials (INCLUDES initial trial)
        "verifier_rounds_by_stage": rounds_by_stage,
        "retunes_total": retunes_total,                # Σ max(0, n_trials-1) -- true retunes
        "retunes_by_stage": retunes_by_stage,
        "vision_judgements": summary.get("vision_judged", 0),
        "escalations": summary.get("escalations", 0),
        "segmentation_retuned": summary.get("segmentation_retuned", 0),
    }


def build_structured_trace(image_id, question, memory, verification_summary,
                           stop_reason, metrics, activity) -> dict:
    """Ordered, step-numbered trace naming each MODULE's input & output at every step.

    The agent loop per step is: Planner -> PlanVerifier -> Executor -> ResultVerifier;
    within-stage VLRR verify->retune rounds happen *inside* the Executor+Verifier of a
    managed step (their count is in ``activity['verifier_iterations_total']`` and each
    step's ``verifier`` block carries the per-trial detail).
    """
    from shape.utils.data_persistence import make_json_serializable

    actions = memory.get_actions(llm_safe=False) if hasattr(memory, "get_actions") else []
    steps = []
    for a in actions:
        steps.append({
            "step": a.get("step_count"),
            "tool": a.get("tool_name"),
            "modules": {
                "planner": {
                    "input": {"question": question},
                    "output": {
                        "tool_selected": a.get("tool_name"),
                        "sub_goal": a.get("sub_goal"),
                        "reasoning": a.get("planner_reasoning"),
                    },
                },
                "executor": {
                    "input": {"command": a.get("command"), "params": a.get("params")},
                    "output": a.get("result"),
                },
                "verifier": {
                    # per-stage verdicts + VLRR trials for this step (None on unmanaged steps)
                    "output": a.get("verification"),
                },
            },
        })
    trace = {
        "schema": "shape.neurips.concurrent.trace.v1",
        "image_id": image_id,
        "question": question,
        "pipeline": ("Planner -> PlanVerifier(plan gate) -> Executor -> "
                     "ResultVerifier, repeated per step; VLRR verify->retune rounds run "
                     "inside a managed step's Executor+Verifier"),
        "stop_reason": stop_reason,
        "n_steps": len(steps),
        "activity": activity,
        "final_metrics": metrics,
        "steps": steps,
    }
    return make_json_serializable(trace)


# --------------------------------------------------------------------------- #
# Deliverable visualization (seg; + crops if ever present)
# --------------------------------------------------------------------------- #
def _load_mask(path):
    import numpy as np
    p = str(path)
    if p.endswith(".npy"):
        return np.load(p)
    from skimage.io import imread
    return imread(p)


def _to_display(img):
    """Normalize an image to a 0..1 float RGB-ish array for imshow."""
    import numpy as np
    arr = np.asarray(img).astype("float32")
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def _read_image_for_display(path):
    if str(path).endswith(".npy"):
        return _to_display(_load_mask(path))
    from skimage.io import imread
    return _to_display(imread(str(path)))


def render_deliverable(image_path, gt_path, pred_mask_path, out_png, meta,
                       crop_dir=None) -> str | None:
    """Raw | predicted instance mask | pred-vs-GT contour overlay, with a metrics title.

    Returns the written path, or None on any failure (never raises — visualization must
    not fail a scored run)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.color import label2rgb
    from skimage.segmentation import find_boundaries

    try:
        img = _read_image_for_display(image_path)
        pred = _load_mask(pred_mask_path).astype("int64")
        gt = None
        if gt_path and Path(gt_path).exists():
            try:
                gt = _load_mask(gt_path).astype("int64")
            except Exception:
                gt = None
        base = img if img.ndim == 3 else np.stack([img] * 3, axis=-1)

        # (title, kind) — kind is a plain string so we never compare a numpy array to a str.
        panels = [("input", "raw"), ("prediction", "label")]
        if gt is not None:
            panels.append(("overlay: pred=cyan, GT=magenta", "overlay"))
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
        axes = np.atleast_1d(axes).ravel()
        for ax, (title, kind) in zip(axes, panels):
            if kind == "raw":
                ax.imshow(img, cmap="gray")
            elif kind == "label":
                ax.imshow(label2rgb(pred, bg_label=0))
            else:  # overlay: pred contours cyan, GT contours magenta
                ov = base.copy()
                ov[find_boundaries(pred, mode="outer")] = [0.0, 1.0, 1.0]
                if gt is not None:
                    ov[find_boundaries(gt, mode="outer")] = [1.0, 0.0, 1.0]
                ax.imshow(ov)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        title = (f"{meta.get('image_id')}  AP50={_fmt(meta.get('ap50'))}  "
                 f"F1@50={_fmt(meta.get('f1_at_50'))}  "
                 f"pred={meta.get('predicted_cell_count')} gt={meta.get('ground_truth_cell_count')}")
        fig.suptitle(title, fontsize=9)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # If the pipeline ever emitted single-cell crops, tile a small montage alongside.
        if crop_dir and Path(crop_dir).exists():
            crops = sorted(Path(crop_dir).glob("*.png"))[:16]
            if crops:
                m = int(np.ceil(np.sqrt(len(crops))))
                figc, axc = plt.subplots(m, m, figsize=(2 * m, 2 * m))
                axc = np.atleast_2d(axc).ravel()
                for a in axc:
                    a.axis("off")
                for a, cp in zip(axc, crops):
                    try:
                        a.imshow(__import__("skimage.io", fromlist=["imread"]).imread(str(cp)))
                    except Exception:
                        pass
                figc.suptitle(f"{meta.get('image_id')} crops (n={len(crops)})", fontsize=9)
                figc.savefig(str(out_png).replace(".png", "_crops.png"), dpi=130,
                             bbox_inches="tight")
                plt.close(figc)
        return str(out_png)
    except Exception as exc:  # never fail a scored run on a plotting error
        print(f"  [viz] {meta.get('image_id')}: {exc}")
        return None


def _fmt(v):
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "n/a"


# NOTE: the post-hoc color/class filter (a VLM refining the FINAL mask, formerly opt-in
# via --color-filter) was removed. The color/class filter now lives ONLY inside the VLRR
# loop as a retune action (shape/agent/verification/verifiers/segmentation.py), where it
# passes through keep-best + reconcile. The old CLI path re-ran it on the final mask
# OUTSIDE the loop (no reconcile, no keep-best) -- a duplicate, so it was deleted.


# --------------------------------------------------------------------------- #
# The worker: one image, one process (top-level & picklable for spawn)
# --------------------------------------------------------------------------- #
def _zero_metrics(gt_path, image_id: str) -> dict:
    """All-false-negative metrics for an image that HAS ground truth but got no valid
    prediction (the agent ran but produced no mask): AP=0 etc. Used ONLY in the
    "including failures" aggregate so a no-mask image scores 0 instead of silently
    dropping out of the denominator. Keys match evaluate_instance_segmentation's
    output for a zero-cell prediction, so macro_metrics sees a consistent schema."""
    import numpy as np
    import run_shape_neurips_cellseg as nb
    gt = nb.load_label_mask(gt_path)
    n_gt = int(len([u for u in np.unique(gt) if int(u) != 0]))
    return {
        "image_name": image_id,
        "ground_truth_path": str(gt_path),
        "ground_truth_cell_count": n_gt,
        "predicted_cell_count": 0,
        "ap50": 0.0, "ap75": 0.0, "map_50_95": 0.0,
        "precision_at_50": 0.0, "recall_at_50": 0.0, "f1_at_50": 0.0,
        "mean_matched_iou_at_50": 0.0,
        "cell_count_error_fraction": 1.0 if n_gt else 0.0,
    }


# Metrics tracked at each cumulative SAM call for the ap50-per-call scaling record.
_CALL_METRIC_KEYS = ("ap50", "ap75", "map_50_95", "precision_at_50",
                     "recall_at_50", "f1_at_50", "predicted_cell_count")


def _attempt_call_metrics(img_out, image_id: str, gt_path) -> list[dict]:
    """Ordered per-tool-call metric trajectory from the saved VLRR attempt masks
    (`shape_cache/output_visualizations/cell_mask_*_a{k}.npy`, one .npy per segmenter
    call, k=1..N). Entry k holds the FULL metrics (_CALL_METRIC_KEYS) of the k-th SAM
    call: entry[0] is the initial/default-param call (the baseline), entry[-1] the last.
    Returns [] if there is no GT or no attempt mask on disk."""
    import glob
    import re
    import run_shape_neurips_cellseg as nb
    vis = Path(img_out) / "shape_cache" / "output_visualizations"
    if not (gt_path and Path(gt_path).is_file() and vis.is_dir()):
        return []
    indexed = []
    for p in glob.glob(str(vis / "cell_mask_*_a*.npy")):
        m = re.search(r"_a(\d+)\.npy$", p)
        if m:
            indexed.append((int(m.group(1)), p))
    indexed.sort()
    out = []
    for _, p in indexed:
        try:
            m = nb.evaluate_instance_segmentation(p, gt_path)
            out.append({k: float(m.get(k, 0) or 0) for k in _CALL_METRIC_KEYS})
        except Exception:
            pass
    return out


def run_one_image(job: dict) -> dict:
    """Run the full agent on ONE image, score it, dump its trace + viz, return a CSV row."""
    # --- Set process env BEFORE importing numpy/torch/cellpose (they read it at LOAD time) ---
    # Cap BLAS/OpenMP threads to 1 PER WORKER: otherwise each of N worker processes spawns
    # ~n_core threads for numpy/cellpose CPU ops, so N procs x n_core threads massively
    # oversubscribe the box (8 x 8 = 64 threads on 16 cores) -> context-switch thrash ->
    # aggregate throughput collapses to ~one worker. One thread/worker => true parallelism.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"
    # Fast per-attempt figures: PIL write instead of 300dpi matplotlib (~1.8s -> ~0.1s each).
    os.environ["SHAPE_FAST_FIGURES"] = "1"
    # Shared SAM: delegate the heavy segmentation model to the server (no local cpsam load).
    # NOTE: do NOT hide CUDA from the worker. The QC path compute_flow_error() calls cellpose
    # masks_to_flows_GPU; with CUDA hidden it silently runs the GPU kernel on CPU (~30-100s/eval
    # burning 2 cores, GPU idle) -- the concurrency killer. Keeping CUDA visible costs only a
    # ~600MB context per worker (8 x 600MB on a 143GB H200 is nothing) and keeps flow_error on
    # GPU (~0.4s). The tool still skips loading its own cpsam model in remote mode.
    if job.get("sam_server_url"):
        os.environ["SHAPE_SAM_SERVER_URL"] = job["sam_server_url"]

    import run_shape_neurips_cellseg as nb  # canonical scoring/activity helpers (dir on sys.path)
    from shape.engine import usage as token_usage
    from shape.utils.data_persistence import CustomEncoder
    try:
        import torch
        torch.set_num_threads(1)          # definitive per-worker cap (overrides torch's default)
    except Exception:
        pass

    image_path = Path(job["image_path"])
    gt_path = Path(job["gt_path"]) if job.get("gt_path") else None
    image_id = image_path.stem
    img_out = Path(job["run_root"]) / "images" / image_id
    img_out.mkdir(parents=True, exist_ok=True)

    row = {"image_name": image_id, "status": "ok", "error": ""}
    t0 = time.perf_counter()
    snap0 = token_usage.snapshot()   # per-process counter -> correct per-image delta
    try:
        if job["dry_run"]:
            result = _fake_solve(image_id, img_out, image_path, gt_path)
        else:
            from shape.solver import solve
            solve_kwargs = dict(
                question=job["question"],
                images=[{"path": str(image_path), "group": "default"}],
                llm_engine_name=job["model"], api_key=job.get("api_key"),
                max_steps=job["max_steps"],
                query_cache_dir=str(img_out / "shape_cache"),
                verify_refine=job["verify_refine"],
                enabled_tools=job["enabled_tools"],
            )
            result = solve(**solve_kwargs)

        runtime = time.perf_counter() - t0
        snap1 = token_usage.snapshot()

        def _d(key):
            return int(snap1.get(key, 0) or 0) - int(snap0.get(key, 0) or 0)
        tokens = _d("total_tokens")
        tok_in = _d("input_tokens") or _d("prompt_tokens")
        tok_out = _d("output_tokens") or _d("completion_tokens")
        tok_cached = _d("cached_tokens")
        memory = result["memory"]
        ver = result.get("verification_summary") or []
        stop_reason = result.get("stop_reason")

        pred = nb._collect_masks_by_image(memory).get(image_id)

        # Distinguish the failure modes so the aggregate can report BOTH a scored-only
        # macro (valid predictions only) and an including-failures macro (no-mask images
        # count as AP=0), and never silently drop an image with GT from the denominator:
        #   metrics      -> valid pred + GT  (drives the scored-only aggregate)
        #   metrics_all  -> valid pred OR a no-mask-with-GT zero (drives include-failures)
        # status: ok | no_mask (ran, no mask, GT present) | no_gt (unscoreable) | error
        metrics: dict = {}
        metrics_all: dict = {}
        has_gt = bool(gt_path and gt_path.is_file())
        if not has_gt:
            row["status"] = "no_gt"          # cannot score -> excluded from BOTH macros
        elif pred:
            metrics = nb.evaluate_instance_segmentation(pred, gt_path)
            metrics["image_name"] = image_id
            metrics_all = metrics
        else:
            row["status"] = "no_mask"        # ran but produced no mask -> AP=0
            metrics_all = _zero_metrics(gt_path, image_id)

        activity = extract_tool_activity(memory, ver, nb)
        trace = build_structured_trace(image_id, job["question"], memory, ver,
                                       stop_reason, metrics, activity)
        (img_out / "trace.json").write_text(
            json.dumps(trace, indent=2, cls=CustomEncoder), encoding="utf-8")
        (img_out / "metrics.json").write_text(
            json.dumps({**metrics, **activity, "runtime_seconds": runtime,
                        "tokens": tokens, "stop_reason": stop_reason},
                       indent=2, default=str), encoding="utf-8")

        viz = render_deliverable(
            image_path, gt_path, pred,
            img_out / "deliverable_seg.png",
            {"image_id": image_id, **metrics},
            crop_dir=img_out / "shape_cache") if pred else None

        # Real per-tool-call metric trajectory (one entry per segmenter call) from the saved
        # VLRR attempt masks -> trustworthy efficiency (tool_calls_per_image, retunes,
        # gain = last - baseline) AND the ap50-per-call scaling record. Call 1 = default params.
        call_metrics = _attempt_call_metrics(img_out, image_id, gt_path)
        call_ap50 = [m["ap50"] for m in call_metrics]

        row.update({
            "ground_truth_cell_count": metrics.get("ground_truth_cell_count"),
            "predicted_cell_count": metrics.get("predicted_cell_count"),
            "ap50": metrics.get("ap50"), "ap75": metrics.get("ap75"),
            "map_50_95": metrics.get("map_50_95"),
            "precision_at_50": metrics.get("precision_at_50"),
            "recall_at_50": metrics.get("recall_at_50"),
            "f1_at_50": metrics.get("f1_at_50"),
            "mean_matched_iou_at_50": metrics.get("mean_matched_iou_at_50"),
            "cell_count_error_fraction": metrics.get("cell_count_error_fraction"),
            "runtime_seconds": round(runtime, 3),
            "tokens": tokens, "tokens_input": tok_in, "tokens_output": tok_out,
            "tokens_cached": tok_cached,
            "n_steps_executed": activity["n_steps_executed"],
            "tool_calls_total": activity["tool_calls_total"],
            "tool_calls": json.dumps(activity["tool_calls"], sort_keys=True),
            "segmenter_calls": activity["segmenter_calls"],
            "verifier_iterations_total": activity["verifier_iterations_total"],
            "verifier_rounds_by_stage": json.dumps(activity["verifier_rounds_by_stage"],
                                                   sort_keys=True),
            "retunes_total": activity["retunes_total"],
            "vision_judgements": activity["vision_judgements"],
            "escalations": activity["escalations"],
            "plan_rejections": activity["plan_rejections"],
            "stop_reason": stop_reason,
            "trace_path": str(img_out / "trace.json"),
            "viz_path": viz or "",
            "_metrics": metrics,          # scored-only macro (stripped from CSV)
            "_metrics_all": metrics_all,  # include-failures macro (no_mask -> AP=0)
            "_call_metrics": call_metrics,  # per-SAM-call metric trajectory (ap50-per-call)
            "_efficiency": {"image_id": image_id,
                            "baseline_ap50": call_ap50[0] if call_ap50 else None,
                            "call_ap50": call_ap50,
                            "tokens": tokens, "seconds": round(runtime, 3)},
        })
    except Exception as exc:
        import traceback
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        (img_out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return row


# --------------------------------------------------------------------------- #
# Dry-run fake (exercises the CSV / trace / viz plumbing with no GPU/LLM/data)
# --------------------------------------------------------------------------- #
class _FakeMemory:
    def __init__(self, actions):
        self._actions = actions

    def get_actions(self, llm_safe=False):
        return self._actions


def _fake_solve(image_id, img_out, image_path, gt_path):
    """Fabricate a realistic solve() return + a tiny mask/image so the whole delivery
    path (scoring, activity extraction, trace, viz) runs without heavy deps."""
    import numpy as np
    from skimage.io import imsave

    rng = np.random.default_rng(abs(hash(image_id)) % (2**32))
    lab = np.zeros((64, 64), dtype="int32")
    for i in range(1, 6):
        y, x = int(rng.integers(6, 58)), int(rng.integers(6, 58))
        lab[max(0, y - 4):y + 4, max(0, x - 4):x + 4] = i
    mask_path = img_out / "pred_mask.npy"
    np.save(mask_path, lab)
    # Write the fake INPUT image at the job's image_path so the viz can read it.
    img = (lab > 0).astype("float32") * 0.7 + rng.random((64, 64)).astype("float32") * 0.3
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    imsave(str(image_path), (img * 255).astype("uint8"))
    if gt_path is not None and not Path(gt_path).exists():
        np.save(str(gt_path), lab)   # gt_path already ends in .npy in dry-run

    # Two-step trace: a first segmentation run + one VLRR-retuned re-run (2 trials).
    actions = [
        {"step_count": 1, "tool_name": "Cell_Segmenter_Tool",
         "sub_goal": "segment all cells", "planner_reasoning": "estimate scale, segment",
         "command": "[verified] Cell_Segmenter_Tool.execute(**{'diameter': 30})",
         "params": {"diameter": 30},
         "result": {"image_id": image_id, "mask_path": str(mask_path),
                    "cell_count": int(lab.max())},
         "verification": {"tool": "Cell_Segmenter_Tool", "stages": [
             {"stage": "segmentation", "action": "retune", "n_trials": 2,
              "trials": [{"notes": "[vision] under-detection", "param_delta": {"diameter": -6}},
                         {"notes": "[vision] good"}]}]}},
    ]
    return {"memory": _FakeMemory(actions),
            "verification_summary": actions[0]["verification"] and [actions[0]["verification"]],
            "stop_reason": "task_complete",
            # scoring uses the real mask we just wrote:
            "_pred": str(mask_path)}


# --------------------------------------------------------------------------- #
# Aggregate + CSV
# --------------------------------------------------------------------------- #
_CSV_COLUMNS = [
    "image_name", "status",
    "ground_truth_cell_count", "predicted_cell_count",
    "ap50", "ap75", "map_50_95", "precision_at_50", "recall_at_50", "f1_at_50",
    "mean_matched_iou_at_50", "cell_count_error_fraction",
    "runtime_seconds",
    "tokens", "tokens_input", "tokens_output", "tokens_cached", "cost_usd",
    "n_steps_executed", "tool_calls_total", "tool_calls", "segmenter_calls",
    "verifier_iterations_total", "verifier_rounds_by_stage", "retunes_total",
    "vision_judgements", "escalations", "plan_rejections", "stop_reason",
    "trace_path", "viz_path", "error",
]


def write_metrics_csv(rows: list[dict], path: Path) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in _CSV_COLUMNS})


def _wait_sam_healthy(url: str, timeout_s: int = 600) -> bool:
    """Poll GET /health until the SAM server answers (model load can take a while)."""
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def _autostart_sam(port: int, replicas: int = 0):
    """Launch the shared Cellpose-SAM server (pool of `replicas`, 0=auto) once healthy."""
    import subprocess
    url = f"http://127.0.0.1:{port}"
    print(f"[concurrent] autostarting shared SAM server on {url} (replicas={replicas or 'auto'}) ...")
    cmd = [sys.executable, "-m", "shape.tools.cell_segmenter.sam_server", "--port", str(port)]
    if replicas:
        cmd += ["--replicas", str(replicas)]
    proc = subprocess.Popen(cmd)
    if not _wait_sam_healthy(url):
        proc.terminate()
        raise SystemExit("[concurrent] SAM server did not become healthy in time.")
    print("[concurrent] shared SAM server is healthy.")
    return proc, url


def _stop_proc(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def _discover_images(image_dir: Path, label_dir: Path, suffix: str, ext: str,
                     limit: int | None, sample_frac: float | None, seed: int):
    import random
    imgs = sorted(p for p in image_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
                  and not p.stem.endswith(suffix))
    if sample_frac is not None:
        rng = random.Random(seed)
        k = max(1, int(len(imgs) * sample_frac))
        imgs = sorted(rng.sample(imgs, k), key=lambda p: p.name)
    if limit is not None:
        imgs = imgs[:limit]

    def gt_for(p: Path) -> Path | None:
        for e in (ext, ".npy", ".tiff", ".tif"):
            cand = label_dir / f"{p.stem}{suffix}{e}"
            if cand.exists():
                return cand
        return None

    return [(p, gt_for(p)) for p in imgs]


def main() -> None:
    # Load repo-root .env so LLM_API_KEY / LLM_BASE_URL (the gpt-5.6-luna gateway creds) land
    # in os.environ for THIS process and are inherited by every spawned worker.
    try:
        from dotenv import load_dotenv
        load_dotenv(_BENCH.parents[1] / ".env")
    except Exception:
        pass
    cfg = _load_local_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = os.environ.get("SHAPE_BENCH_DATA_ROOT")
    ap.add_argument("--data-root", type=Path,
                    default=(Path(default_root) / "NeurIPS22-CellSeg") if default_root else None,
                    help="Dataset root with images/ + labels/. Default $SHAPE_BENCH_DATA_ROOT/NeurIPS22-CellSeg.")
    ap.add_argument("--image-subdir", default=cfg.image_subdir)
    ap.add_argument("--label-subdir", default=cfg.label_subdir)
    ap.add_argument("--label-suffix", default=cfg.label_suffix)
    ap.add_argument("--label-ext", default=cfg.label_ext)
    ap.add_argument("--output-root", type=Path, default=_BENCH / "results",
                    help="Run artifacts root (a timestamped subdir is created). Default: ./results")
    ap.add_argument("--workers", type=int, default=32,
                    help="Concurrent worker PROCESSES (default 32). Lower it on a single GPU.")
    ap.add_argument("--limit", type=int, help="Cap #images after sampling.")
    ap.add_argument("--sample-frac", type=float, help="Random subset fraction, e.g. 0.2.")
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--model", default=cfg.llm_model,
                    help="LLM the agent uses. Default from cellseg_config.py (llm_model).")
    ap.add_argument("--max-steps", type=int, default=cfg.max_steps)
    ap.add_argument("--verify-refine", action="store_true", default=cfg.verify_refine)
    ap.add_argument("--sam-server-url", default=None,
                    help="Delegate Cellpose-SAM to a shared server at this URL (workers do "
                         "NOT load their own model). Start it with "
                         "`python -m shape.tools.cell_segmenter.sam_server`.")
    ap.add_argument("--autostart-sam", action="store_true",
                    help="Launch the shared SAM server automatically for this run and stop it "
                         "afterwards (one GPU-resident model for all workers).")
    ap.add_argument("--sam-port", type=int, default=8137,
                    help="Port for --autostart-sam (default 8137).")
    ap.add_argument("--sam-replicas", type=int, default=0,
                    help="Model instances in the autostarted SAM pool (max concurrent evals). "
                         "0 = auto-size from free VRAM. Raise for high --workers (K=4 ~ 42 workers).")
    ap.add_argument("--price-in", type=float, default=None,
                    help="USD per 1M INPUT tokens (overrides the built-in table). Required to "
                         "cost a private model like gpt-5.6-luna. See OpenAI's pricing page.")
    ap.add_argument("--price-out", type=float, default=None,
                    help="USD per 1M OUTPUT tokens (overrides the built-in table).")
    ap.add_argument("--price-cached", type=float, default=None,
                    help="USD per 1M CACHED-input tokens (optional; defaults to the input rate).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fabricate solve() output + a tiny mask/image to validate the "
                         "CSV / trace / viz plumbing with NO GPU/LLM/data.")
    args = ap.parse_args()

    if args.data_root is None and not args.dry_run:
        ap.error("--data-root is required (or set $SHAPE_BENCH_DATA_ROOT), unless --dry-run.")

    run_root = args.output_root / (time.strftime("%Y-%m-%d_%H%M%S") + "_concurrent")
    run_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # Fabricate a handful of image ids; no real files needed (worker synthesizes them).
        n = args.limit or 8
        pairs = [(run_root / "images" / f"cell_{i:05d}" / f"cell_{i:05d}.tif",
                  run_root / "images" / f"cell_{i:05d}" / f"cell_{i:05d}_label.npy")
                 for i in range(1, n + 1)]
        for img, _ in pairs:
            img.parent.mkdir(parents=True, exist_ok=True)
    else:
        image_dir = args.data_root / args.image_subdir
        label_dir = args.data_root / args.label_subdir
        if not image_dir.is_dir():
            ap.error(f"images dir not found: {image_dir}")
        pairs = _discover_images(image_dir, label_dir, args.label_suffix, args.label_ext,
                                 args.limit, args.sample_frac, args.seed)
        if not pairs:
            ap.error(f"no images found under {image_dir}")

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not args.dry_run and not api_key:
        print("[warn] no LLM_API_KEY/ANTHROPIC_API_KEY set — the agent's LLM planner/verifier "
              "cannot run.")

    question = cfg.question_fallback
    qfile = _BENCH / cfg.question_file
    if qfile.exists():
        question = qfile.read_text(encoding="utf-8")

    # Shared Cellpose-SAM: one GPU-resident model for all workers (no per-worker load).
    sam_proc, sam_url = None, args.sam_server_url
    if args.autostart_sam and not args.dry_run:
        sam_proc, sam_url = _autostart_sam(args.sam_port, args.sam_replicas)
    if sam_url:
        print(f"[concurrent] workers will call the shared SAM server at {sam_url}")

    jobs = [{
        "image_path": str(img), "gt_path": (str(gt) if gt else None),
        "run_root": str(run_root), "question": question, "model": args.model,
        "api_key": api_key, "max_steps": args.max_steps,
        "enabled_tools": [cfg.enabled_tools[0]] if cfg.enabled_tools else None,
        "verify_refine": args.verify_refine, "dry_run": args.dry_run,
        "sam_server_url": (None if args.dry_run else sam_url),
    } for img, gt in pairs]

    # Self-document the run: dump EVERY knob so any later run is reproducible/comparable.
    # (run_summary.json records results; run_params.json records the inputs that produced them.)
    _pin, _pout, _pcached = _resolve_prices(args.model, args.price_in, args.price_out,
                                            args.price_cached)
    try:
        from dataclasses import asdict as _asdict
        cfg_snap = _asdict(cfg)
    except Exception:
        cfg_snap = {k: getattr(cfg, k) for k in vars(cfg)} if hasattr(cfg, "__dict__") else {}
    try:
        import subprocess
        _git = subprocess.run(["git", "-C", str(_THIS_DIR), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        _git = None
    run_params = {
        "timestamp": run_root.name,
        "argv": sys.argv,
        "git_commit": _git,
        "n_images": len(jobs),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "agent": {  # the flags each worker's solve() actually receives
            "model": args.model, "max_steps": args.max_steps,
            "enabled_tools": list(cfg.enabled_tools) if cfg.enabled_tools else None,
            "enable_verification": cfg.enable_verification,
            "enable_llm_verifier": cfg.enable_llm_verifier,
            "verify_refine": args.verify_refine,
        },
        "sam": {"server_url": sam_url, "autostart": args.autostart_sam,
                "port": args.sam_port, "replicas": args.sam_replicas},
        "pricing_usd_per_1M": {"input": _pin, "output": _pout, "cached_input": _pcached},
        "config_snapshot": cfg_snap,
        "env": {k: os.environ.get(k) for k in
                ("LLM_BASE_URL", "SHAPE_SAM_SERVER_URL", "SHAPE_FAST_FIGURES",
                 "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                 "SHAPE_BENCH_DATA_ROOT")},
    }
    (run_root / "run_params.json").write_text(
        json.dumps(run_params, indent=2, default=str), encoding="utf-8")
    print(f"[concurrent] wrote run_params.json ({run_root / 'run_params.json'})")

    print(f"[concurrent] {len(jobs)} image(s) x {args.workers} workers -> {run_root}")
    t0 = time.perf_counter()
    rows: list[dict] = []
    # spawn: CUDA-safe and gives each worker its own process-global token counter.
    ctx = get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=ctx) as ex:
            futs = {ex.submit(run_one_image, job): job["image_path"] for job in jobs}
            for i, fut in enumerate(as_completed(futs), start=1):
                try:
                    r = fut.result()
                except Exception as exc:  # a worker that died hard still gets a row
                    r = {"image_name": Path(futs[fut]).stem, "status": "worker_crash",
                         "error": f"{type(exc).__name__}: {exc}"}
                rows.append(r)
                print(f"  [{i}/{len(jobs)}] {r['image_name']}: {r['status']} "
                      f"AP50={_fmt(r.get('ap50'))} calls={r.get('tool_calls_total','-')} "
                      f"vlrr_iters={r.get('verifier_iterations_total','-')}")
    finally:
        if sam_proc is not None:
            print("[concurrent] stopping shared SAM server")
            _stop_proc(sam_proc)

    rows.sort(key=lambda r: r["image_name"])

    # Cost (OpenAI-style: input/output/cached USD per 1M tokens). None when the model's
    # price is unknown and no --price-in/--price-out was given.
    pin, pout, pcached = _resolve_prices(args.model, args.price_in, args.price_out,
                                         args.price_cached)
    for r in rows:
        r["cost_usd"] = _cost_usd(r.get("tokens_input"), r.get("tokens_output"),
                                  r.get("tokens_cached"), pin, pout, pcached)
    if pin is None:
        print(f"[cost] no price for model '{args.model}' — cost_usd left null. "
              f"Pass --price-in/--price-out (USD per 1M tokens) to compute it.")

    csv_path = run_root / "cell_image_metrics.csv"
    write_metrics_csv(rows, csv_path)

    # Aggregate (reuse the canonical macro + efficiency helpers).
    import run_shape_neurips_cellseg as nb
    import segmentation_metrics as sm  # noqa: E402  (benchmarks/_shared on path via nb)
    scored = [r["_metrics"] for r in rows if r.get("_metrics")]
    scored_all = [r["_metrics_all"] for r in rows if r.get("_metrics_all")]
    eff = [r["_efficiency"] for r in rows if r.get("_efficiency")]
    # ap50-per-call scaling record: mean of each metric at the cumulative k-th SAM call,
    # with last-observation-carry-forward -- an image that converged in n<k calls
    # contributes its call min(k, n), so every image is counted at every k. This is the
    # SHAPE analogue of the baselines' per-iteration curve; from it the AP50-vs-#calls
    # scaling figure follows directly.
    import numpy as _np
    per_call = [r["_call_metrics"] for r in rows if r.get("_call_metrics")]
    _maxk = max((len(c) for c in per_call), default=0)
    ap50_per_call = []
    for k in range(1, _maxk + 1):
        entry = {"call": k, "n_images": len(per_call)}
        for mk in _CALL_METRIC_KEYS:
            vals = [c[min(k, len(c)) - 1][mk] for c in per_call if c]
            entry[mk] = round(float(_np.mean(vals)), 6) if vals else 0.0
        ap50_per_call.append(entry)
    n_no_mask = sum(1 for r in rows if r["status"] == "no_mask")
    n_no_gt = sum(1 for r in rows if r["status"] == "no_gt")
    n_errors = sum(1 for r in rows if r["status"] in ("error", "worker_crash"))
    summary = {
        "dataset": "NeurIPS22-CellSeg",
        "task": "agentic_segmentation (concurrent)",
        "workers": args.workers,
        "model": args.model,
        "dry_run": args.dry_run,
        "n_images": len(rows),
        "n_scored": len(scored),
        "n_errors": n_errors,
        # explicit failure ledger: every image is accounted for in exactly one bucket.
        "failure_breakdown": {
            "ok": len(scored), "no_mask": n_no_mask, "no_gt": n_no_gt, "error": n_errors,
        },
        "wall_clock_seconds": round(time.perf_counter() - t0, 2),
        # BOTH aggregates: scored-only (valid predictions) AND including-failures
        # (no_mask counts as AP=0). `segmentation_macro` stays the scored-only value
        # for backward compatibility with existing plot/report consumers.
        "segmentation_macro": nb.macro_metrics(scored),
        "segmentation_macro_excluding_failures": nb.macro_metrics(scored),
        "segmentation_macro_including_failures": nb.macro_metrics(scored_all),
        "ap50_per_call": ap50_per_call,   # metrics at cumulative SAM call k (LOCF)
        "efficiency_metrics": sm.compute_metrics(eff) if eff else {},
        "agentic_activity_totals": {
            # OUTER LOOP: agent planner steps actually executed (segmenter re-runs)
            "n_steps_executed_total": sum(int(r.get("n_steps_executed") or 0) for r in rows),
            "tool_calls_total": sum(int(r.get("tool_calls_total") or 0) for r in rows),
            # VERIFY: Σ n_trials (incl. initial trial); retunes = Σ max(0, n_trials-1)
            "verifier_iterations_total": sum(int(r.get("verifier_iterations_total") or 0) for r in rows),
            "retunes_total": sum(int(r.get("retunes_total") or 0) for r in rows),
            "vision_judgements": sum(int(r.get("vision_judgements") or 0) for r in rows),
            "escalations": sum(int(r.get("escalations") or 0) for r in rows),
        },
        "cost": {
            "model": args.model,
            "price_per_1M": {"input": pin, "output": pout, "cached_input": pcached},
            "tokens_input_total": sum(int(r.get("tokens_input") or 0) for r in rows),
            "tokens_output_total": sum(int(r.get("tokens_output") or 0) for r in rows),
            "tokens_cached_total": sum(int(r.get("tokens_cached") or 0) for r in rows),
            "tokens_total": sum(int(r.get("tokens") or 0) for r in rows),
            "cost_usd_total": (round(sum(r["cost_usd"] for r in rows
                                         if r.get("cost_usd") is not None), 4)
                               if pin is not None else None),
        },
        "csv": str(csv_path),
    }
    (run_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    macro = summary["segmentation_macro"]
    macro_all = summary["segmentation_macro_including_failures"]
    cost = summary["cost"]
    cost_str = (f"${cost['cost_usd_total']}" if cost["cost_usd_total"] is not None else "n/a")
    print(f"[done] {summary['n_scored']}/{summary['n_images']} scored "
          f"(no_mask={n_no_mask} no_gt={n_no_gt} err={n_errors}) in "
          f"{summary['wall_clock_seconds']}s | "
          f"macro AP50 excl_fail={_fmt(macro.get('ap50'))} "
          f"incl_fail={_fmt(macro_all.get('ap50'))} "
          f"F1@50={_fmt(macro.get('f1_at_50'))} | "
          f"tool_calls={summary['agentic_activity_totals']['tool_calls_total']} "
          f"vlrr_iters={summary['agentic_activity_totals']['verifier_iterations_total']}")
    print(f"[cost] tokens in={cost['tokens_input_total']} out={cost['tokens_output_total']} "
          f"cached={cost['tokens_cached_total']} total={cost['tokens_total']} -> {cost_str}")
    print(f"[done] CSV -> {csv_path}")
    print(f"[done] per-image traces + viz under {run_root / 'images'}")


if __name__ == "__main__":
    main()
