#!/usr/bin/env python3
"""Per-image audit records (JSON) of every arm of a run_cellseg_benchmark.py run dir.

One file per arm x image, `<out>/<arm>/<band>/<image>.json` (schema `cellseg-run-record/1`), built only from
the run's own artifacts: `<arm>/<band>/records.jsonl` (score, cost, status), `<arm>/<band>/<image>/trace.json`
(shape: the stop reason and stage trace; the tools: the call's settings; gpt: the reply's size),
`gpt/.../gpt_reply.json`, react's TRACE.jsonl / CALLS.json / SELECTED.json (the shared ReAct runner's records;
a run of the earlier ReAct design: its trace.json turns),
`shape/<band>/<image>/verification_summary.json` (the agentic VRR record), `<arm>/run_meta.json`
(provenance), and the arm's log (`logs/<arm>.log`: the warnings / errors printed while that image ran).
Each record holds:

    input            image id, band, image path (the ground truth is used for scoring only)
    provenance       commit, host, env, packages, model, max_steps, start / end (run_meta.json)
    config           what the arm was given (tool, tunable inputs, prompt / enabled tools)
    steps            every step in order: the tool call (tool + arguments), its result, the decision and the
                     rationale the agent stated for it (as recorded by the run), the system's rules where they
                     acted (shape: preflight, admission, reference call, override check)
    final_output     the delivered attempt / call, its parameters, mask path, predicted cell count
    score / cost     AP50, AP75, mAP, precision, recall, F1, counts / seconds, tool calls, tokens
    status           delivered, error, stop reasons, reasoner failures, fallback rounds, log warnings / errors
    sources          the files the record was built from, and the ones that were missing

Usage:
    python export_run_records.py --run-dir <run> [--arms ...] [--logs-dir <run>/logs] [--out <run>/audit]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCHEMA = "cellseg-run-record/1"
BANDS = ("hard", "medium", "easy")
ARMS = ("cellpose_sam", "cellpose3", "microsam", "stardist", "gpt", "react", "shape")
# what each single-call arm runs (cellseg_arms.py); the per-image settings come from its trace.json
_TOOL_CONFIG = {
    "cellpose_sam": {"tool": "Cell_Segmenter_Tool (Cellpose-SAM, cpsam)", "settings": "the tool's default parameters"},
    "cellpose3": {"tool": "Cellpose 3 (models.Cellpose)", "settings": "model cyto3, diameter=None (estimated by "
                  "its size model), channels=[0, 0] on the highest-variance channel"},
    "microsam": {"tool": "micro-sam automatic instance segmentation", "settings": "model vit_b_lm, defaults; "
                 "tiled (tile 1024, halo 256) when the long side exceeds 2048 px"},
    "stardist": {"tool": "StarDist2D (pretrained)", "settings": "2D_versatile_he for an RGB image with a bright "
                 "background, else 2D_versatile_fluo on the highest-variance channel (inverted when the background "
                 "is bright); percentile normalization 1 / 99.8; CPU TensorFlow"},
}
_SCORE_KEYS = ("ap50", "ap75", "map_50_95", "precision_at_50", "recall_at_50", "f1_at_50",
               "ground_truth_cell_count", "predicted_cell_count", "cell_count_error_fraction")
_LOG_EVENT = re.compile(r"(WARNING|ERROR|Traceback|CUDA tensor allocation failed|Using device: cpu)")


def _load(path: Path, sources: dict):
    if path.is_file():
        sources["used"].append(str(path))
        return json.loads(path.read_text(encoding="utf-8"))
    sources["missing"].append(str(path))
    return None


def _records(path: Path) -> dict:
    if not path.is_file():
        return {}
    rows = (json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
    return {r["image"]: r for r in rows}


def log_events(log_path: Path, arm: str) -> dict:
    """image id -> the warning / error lines its run printed (the lines before its result line)."""
    if not log_path.is_file():
        return {}
    out, block = {}, []
    result = re.compile(rf"^\[{re.escape(arm)}/\w+ \d+/\d+\] (\S+):")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = result.match(line)
        if m:
            events = [ln.strip() for ln in block if _LOG_EVENT.search(ln)]
            out[m.group(1)] = list(dict.fromkeys(events))           # deduplicated, in order
            block = []
        else:
            block.append(line)
    return out


def _common(arm: str, band: str, image: str, rec: dict, meta: dict | None, events: list) -> dict:
    return {
        "schema": SCHEMA, "arm": arm,
        "input": {"image_id": image, "band": band, "ground_truth": "used for scoring only, never shown to the arm"},
        "provenance": {k: (meta or {}).get(k) for k in ("commit", "host", "env", "python", "packages", "model",
                                                       "max_steps", "data_root", "started_utc", "ended_utc")},
        "score": {k: rec.get(k) for k in _SCORE_KEYS},
        "cost": {"seconds": rec.get("seconds"), "tool_calls": rec.get("tool_calls"), "tokens": rec.get("tokens")},
        "status": {"delivered": rec.get("delivered"), "error": rec.get("error"), "log_events": events},
    }


# --------------------------------------------------------------------------- #
# the single-call domain tools, and gpt
# --------------------------------------------------------------------------- #
def tool_record(arm: str, band: str, image: str, rec: dict, trace: dict | None, meta, events) -> dict:
    out = _common(arm, band, image, rec, meta, events)
    out["config"] = dict(_TOOL_CONFIG[arm])
    out["steps"] = [{"step": "tool_call", "by": "the benchmark (no agent, no LLM)",
                     "tool_call": {"tool": _TOOL_CONFIG[arm]["tool"], "arguments": trace},
                     "tool_result": {"status": "failed" if rec.get("error") else "success", "error": rec.get("error"),
                                     "predicted_cell_count": rec.get("predicted_cell_count")}}]
    out["final_output"] = {"mask": f"{arm}/{band}/{image}/mask.tif", "settings": trace,
                           "predicted_cell_count": rec.get("predicted_cell_count")}
    return out


def gpt_record(band: str, image: str, rec: dict, trace: dict | None, reply: dict | None, meta, events) -> dict:
    out = _common("gpt", band, image, rec, meta, events)
    trace = trace or {}
    out["config"] = {"agent": "the LLM alone (no tool)", "output": "one polygon per cell as JSON, rasterized",
                     "image_shown": {"size": trace.get("shown_size"), "scale_to_original": trace.get("scale")},
                     "model": (meta or {}).get("model")}
    result = {"status": "parsed" if trace.get("parsed", trace.get("n_polygons") is not None) else "unparsed",
              "polygons_returned": trace.get("n_polygons"), "polygons_skipped": trace.get("n_skipped")}
    if reply:
        result["reply"] = reply.get("reply")
    elif trace.get("reply_head"):
        result["reply_head"] = trace.get("reply_head")
    else:
        result["reply"] = "not recorded by this run (only its polygon count)"
    out["steps"] = [{"step": "llm_call", "by": "gpt", "prompt": (reply or {}).get("prompt"),
                     "result": result}]
    out["final_output"] = {"mask": f"gpt/{band}/{image}/mask.tif", "predicted_cell_count": rec.get("predicted_cell_count")}
    return out


# --------------------------------------------------------------------------- #
# react
# --------------------------------------------------------------------------- #
def _jsonl(path: Path, sources: dict) -> list | None:
    if path.is_file():
        sources["used"].append(str(path))
        return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sources["missing"].append(str(path))
    return None


def react_record(band: str, image: str, rec: dict, rows: list | None, calls: list | None, selected: dict | None,
                 meta, events) -> dict:
    """The react arm on the shared ReAct runner (benchmarks/_shared/react_arm.py): every round of TRACE.jsonl in
    order (the default call, each agent round's thought, action, calls and observations, the finish question),
    each call's parameters and tool result from CALLS.json, the delivery from SELECTED.json."""
    out = _common("react", band, image, rec, meta, events)
    meta = meta or {}
    by_call = {c.get("call_id"): c for c in calls or []}
    out["config"] = {"agent": "an LLM plus the tool (benchmarks/_shared/react_arm.py, REACT_ARM_CONTRACT.md)",
                     "tool": "Cell_Segmenter_Tool", "parameters": meta.get("parameters"),
                     "task_text": "results/Prompt.txt (SHAPE's question in this benchmark)",
                     "observation": "the tool's returned dict + the overlay PNG the tool wrote for each call",
                     "budget": meta.get("budget"), "round_cap": meta.get("round_cap"),
                     "default_counts_as_round": meta.get("default_counts_as_round"), "workers": meta.get("workers"),
                     "model": meta.get("model")}
    steps = []
    for row in rows or []:
        step = {"round": row.get("round"), "kind": row.get("kind"), "prompt_sha256": row.get("prompt_sha256"),
                "images": row.get("images"), "reply": row.get("reply")}
        action = row.get("action")
        if row.get("kind") != "default":
            step["parsed"] = action is not None
            if action:
                step.update(rationale=action.get("thought"), action=action.get("action"),
                            deliver_requested=action.get("deliver"))
        step["tool_calls"] = [{"call": c.get("call_id"), "tool": "Cell_Segmenter_Tool", "arguments": c.get("params"),
                               "tool_result": {k: (by_call.get(c.get("call_id")) or {}).get(k)
                                               for k in ("output", "images", "error", "seconds")}}
                              for c in row.get("calls") or []]
        step["observations"] = row.get("observations")
        steps.append(step)
    out["steps"] = steps
    selected = selected or {}
    delivered = by_call.get(selected.get("call_id")) or {}
    out["final_output"] = {
        "delivered_call": selected.get("call_id"),
        "how": {"agent_finish": "the call the agent named at finish",
                "default_session_unfinished": "the default call (the session ended without a valid finish)"}.get(
                    selected.get("source"), "none"),
        "parameters": selected.get("params"),
        "mask_path": (delivered.get("output") or {}).get("mask_path"),
        "predicted_cell_count": rec.get("predicted_cell_count")}
    out["status"].update({"delivery": selected.get("source"), "stop": selected.get("stop"),
                          "delivered_call_error": selected.get("call_error"), "rounds": selected.get("rounds"),
                          "calls": selected.get("calls"), "tokens": selected.get("tokens"),
                          "unparsed_replies": sum(1 for s in steps if s.get("parsed") is False)})
    return out


def react_record_legacy(band: str, image: str, rec: dict, trace: dict | None, meta, events) -> dict:
    """The earlier ReAct design (superseded; runs before benchmarks/_shared/react_arm.py): its trace.json turns."""
    out = _common("react", band, image, rec, meta, events)
    trace = trace or {}
    attempts = trace.get("attempts") or []
    out["config"] = {"agent": "ReAct (Thought -> Action -> Observation)", "tool": "Cell_Segmenter_Tool",
                     "tunable_inputs": ["diameter", "flow_threshold", "cellprob_threshold", "min_size", "niter",
                                        "augment", "tile_overlap"],
                     "observation": "the tool's text result + the attempt's overlay image",
                     "max_steps": (meta or {}).get("max_steps")}
    steps, attempt = [], 0
    for s in trace.get("steps") or []:
        step = {"turn": s.get("turn"), "parsed": s.get("parsed", True)}
        for k in ("prompt", "images", "usage"):          # written by runs from this commit on
            if s.get(k) is not None:
                step[k] = s[k]
        if s.get("parsed") is False:
            step["status"] = "the reply could not be parsed as the requested JSON; the turn was used"
            step["reply_head"] = s.get("reply_head")
            steps.append(step)
            continue
        step["rationale"] = s.get("thought")
        step["action"] = s.get("action")
        if s.get("action") == "segment":
            args = {k: s.get(k) for k in ("diameter", "flow_threshold", "cellprob_threshold", "min_size", "niter",
                                          "augment", "tile_overlap")}
            args["niter"] = args["niter"] or None
            obs = s.get("observation") or ""
            ok = obs.startswith("attempt ")
            attempt += ok
            step["tool_call"] = {"tool": "Cell_Segmenter_Tool", "arguments": args}
            step["tool_result"] = {"status": "success" if ok else "failed", "observation": obs,
                                   "attempt": attempt if ok else None,
                                   "mask_path": attempts[attempt - 1].get("mask_path") if ok and attempt <= len(attempts) else None}
        else:
            step["deliver_attempt_requested"] = s.get("deliver_attempt")
            step["deliver_attempt_accepted"] = s.get("delivered")
        steps.append(step)
    out["steps"] = steps
    delivered = trace.get("delivered_attempt")
    finished = any(s.get("action") == "finish" for s in steps)
    out["final_output"] = {
        "delivered_attempt": delivered,
        "how": ("the attempt the agent named at finish" if finished and delivered
                else "the last successful attempt (no finish within the step budget)" if delivered else "none"),
        "mask_path": attempts[delivered - 1].get("mask_path") if delivered and delivered <= len(attempts) else None,
        "parameters": next((st["tool_call"]["arguments"] for st in steps
                            if (st.get("tool_result") or {}).get("attempt") == delivered), None),
        "predicted_cell_count": rec.get("predicted_cell_count")}
    out["status"].update({"finished_by_agent": finished,
                          "turns": len(steps), "unparsed_turns": sum(1 for s in steps if s.get("parsed") is False)})
    return out


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #
def call_names(details: dict) -> list[str]:
    """Trial index -> the call's name, from the record's rounds: "input", then each round's calls in order (the
    names the run used: "default", "reference", "agent_<n>", ..., or c01, c02, ... in records before the role
    names)."""
    return ["input"] + [c for calls in details.get("rounds") or [] for c in calls]


def _call(names: list[str], index: int) -> str:
    return names[index] if 0 <= index < len(names) else f"trial_{index}"


def shape_record(band: str, image: str, rec: dict, trace: dict | None, vs: list | None, meta, events) -> dict:
    out = _common("shape", band, image, rec, meta, events)
    trace = trace or {}
    out["config"] = {"agent": "SHAPE (shape.solver.solve: planner + agentic VRR verifier)",
                     "prompt": "results/Prompt.txt", "enabled_tools": ["Cell_Segmenter_Tool"],
                     "max_steps": (meta or {}).get("max_steps")}
    seg = None
    for group in vs or []:
        for pi in (group.get("per_image") or [group]):
            for stage in pi.get("stages") or []:
                if stage.get("stage") == "segmentation":
                    seg = (pi, stage)
    if seg is None:
        out["steps"] = []
        out["final_output"] = {"mask_path": trace.get("mask_path"),
                               "predicted_cell_count": rec.get("predicted_cell_count")}
        out["status"].update({"solver_stop_reason": trace.get("stop_reason"),
                              "note": "no segmentation VRR record"})
        return out
    pi, stage = seg
    d = stage.get("details") or {}
    trials = stage.get("trials") or []
    plans = {p.get("round"): p for p in d.get("plans") or []}
    assessments = d.get("assessments") or {}
    problems = d.get("measured_problems") or {}
    colours = d.get("color_populations") or {}
    round_of = {c: i + 1 for i, calls in enumerate(d.get("rounds") or []) for c in calls}
    names = call_names(d)

    steps = [{"step": "preflight_read", "by": "vision LLM (image_context preflight)",
              "input": d.get("input"), "result": d.get("preflight")},
             {"step": "target", "by": "the request", "result": d.get("target")}]
    for i, t in enumerate(trials):
        if i == 0:
            continue
        call = _call(names, i)
        r = t.get("result") or {}
        steps.append({
            "step": "tool_call", "call": call, "round": round_of.get(call),
            "parent": _call(names, t["parent"]) if isinstance(t.get("parent"), int) else t.get("parent"),
            "origin": t.get("notes"),
            "tool_call": {"tool": "Cell_Segmenter_Tool", "arguments": t.get("params"), "change": t.get("param_delta")},
            "tool_result": {"status": "failed" if r.get("error") else "success", "error": r.get("error"),
                            "cell_count": r.get("cell_count"), "attempt": r.get("attempt"),
                            "mask_path": r.get("mask_path"), "metrics": t.get("metrics"), "qc_score": t.get("score"),
                            "measured_problems": problems.get(call), "color_populations": colours.get(call)},
            "agent_assessment": assessments.get(call),
        })
        if call == (d.get("admission") or {}).get("call"):
            steps.append({"step": "admission", "by": "system rule (admit_defaults)", "result": d.get("admission")})
    for rnd in sorted(k for k in plans if k is not None):
        p = plans[rnd]
        steps.append({"step": "plan", "round": rnd, "by": "agent", "question": p.get("question"),
                      "rationale": p.get("analysis"), "hypotheses": p.get("hypotheses"),
                      "focus_levers": p.get("focus_levers"), "moves": p.get("moves"),
                      "assessments": p.get("assessments")})
    steps.append({"step": "final_choice", "by": "agent, checked by the system's override rule",
                  "agent_answers": d.get("final_answers"), "reference_default": d.get("reference_default"),
                  "choice": d.get("choice"),
                  "selected": _call(names, d["selected"]) if isinstance(d.get("selected"), int) else None})
    out["steps"] = steps
    out["final_output"] = {"delivered_call": (_call(names, d["selected"]) if isinstance(d.get("selected"), int)
                                              else None),
                           "how": d.get("choice"), "parameters": pi.get("final_params"),
                           "mask_path": trace.get("mask_path"), "predicted_cell_count": rec.get("predicted_cell_count"),
                           "verdict": {k: stage.get(k) for k in ("decision", "action", "status", "reason", "score")}}
    out["status"].update({"solver_stop_reason": trace.get("stop_reason"), "vrr_stop_reason": d.get("stop_reason"),
                          "verifier_stop": d.get("verifier_stop"), "reasoner_failures": d.get("reasoner_failures"),
                          "fallback_rounds": d.get("fallback_rounds"), "failed_calls": d.get("failed"),
                          "admissible_calls": d.get("admissible"), "hypotheses": d.get("hypotheses"),
                          "focus_history": d.get("focus_history")})
    return out


# --------------------------------------------------------------------------- #
def export(run_dir: Path, arms=ARMS, logs_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    logs_dir = logs_dir or run_dir / "logs"
    out_dir = out_dir or run_dir / "audit"
    index = []
    for arm in arms:
        meta_sources = {"used": [], "missing": []}
        meta = _load(run_dir / arm / "run_meta.json", meta_sources)
        events = log_events(logs_dir / f"{arm}.log", arm)
        for band in BANDS:
            for image, rec in sorted(_records(run_dir / arm / band / "records.jsonl").items()):
                sources = {"used": list(meta_sources["used"]) + [str(run_dir / arm / band / "records.jsonl")],
                           "missing": list(meta_sources["missing"])}
                img_dir = run_dir / arm / band / image
                trace = _load(img_dir / "trace.json", sources)
                if arm == "react" and (trace or {}).get("steps") is not None:    # the earlier ReAct design
                    record = react_record_legacy(band, image, rec, trace, meta, events.get(image, []))
                elif arm == "react":
                    record = react_record(band, image, rec, _jsonl(img_dir / "TRACE.jsonl", sources),
                                          _load(img_dir / "CALLS.json", sources),
                                          _load(img_dir / "SELECTED.json", sources), meta, events.get(image, []))
                elif arm == "shape":
                    vs = _load(img_dir / "verification_summary.json", sources)
                    record = shape_record(band, image, rec, trace, vs, meta, events.get(image, []))
                elif arm == "gpt":
                    reply = _load(img_dir / "gpt_reply.json", sources)
                    record = gpt_record(band, image, rec, trace, reply, meta, events.get(image, []))
                elif arm in _TOOL_CONFIG:
                    record = tool_record(arm, band, image, rec, trace, meta, events.get(image, []))
                else:
                    raise ValueError(f"no audit record builder for arm {arm!r}")
                record["sources"] = sources
                path = out_dir / arm / band / f"{image}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
                index.append({"arm": arm, "band": band, "image": image, "path": str(path.relative_to(out_dir)),
                              "ap50": rec.get("ap50"), "delivered": rec.get("delivered"), "error": rec.get("error"),
                              "steps": len(record["steps"]), "missing_sources": len(sources["missing"])})
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema": SCHEMA, "run_dir": str(run_dir), "arms": list(arms), "n_records": len(index),
               "records": index}
    (out_dir / "index.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="holds <arm>/<band>/records.jsonl")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
                    help="default: every arm (an arm without records is skipped)")
    ap.add_argument("--logs-dir", type=Path, help="default <run-dir>/logs")
    ap.add_argument("--out", type=Path, help="default <run-dir>/audit")
    args = ap.parse_args()
    s = export(args.run_dir, tuple(args.arms), args.logs_dir, args.out)
    missing = sum(r["missing_sources"] for r in s["records"])
    print(f"[audit] {s['n_records']} records -> {args.out or args.run_dir / 'audit'} "
          f"({missing} missing source file(s) across records)")


if __name__ == "__main__":
    main()
