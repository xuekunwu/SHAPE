#!/usr/bin/env python3
"""Per-image audit records (JSON) of every arm of a run_benchmark.py run dir, and its copy into data/.

One file per arm x image, `<out>/<arm>/<image>.json` (schema `cropper-run-record/1`), built only from the
run's own artifacts: `selection.json` (the input, the seed), `provenance.json` (the job's checks and status),
`<arm>/run_meta.json` (the arm's settings; host, python, packages and the `llm` block when the run wrote them),
`<arm>/records.jsonl` (score, cost, status, the arm's trace: gpt's requests, react's rounds and tool calls),
`shape/images/<image>/agent_log.jsonl` + `agent_details.json` (every SHAPE question, reply, system rule, call and
the mask check), and the job logs (`<job>.out` / `.err` whose header names this run dir, `<job>-<arm>.log`: the
warnings / errors printed while that image ran). Each record holds:

    input            image id, image / mask path, seed, selection index (the mask is the ground truth)
    provenance       commit, model, host, python, packages, llm block, seed, times, the job's status
    config           what the arm was given (tool, settings, model, budget)
    steps            every step in order: tool calls with arguments and results; every model request with its
                     prompt, attached images (label, sha256, bytes; never the bytes), reply as returned, usage,
                     error and attempt; the rationale the agent stated (react `thought`, SHAPE `analysis`,
                     hypotheses, moves, final reasoning); SHAPE's system rules, rejected / ignored moves, the
                     delivery and the mask check (montages and verdicts). Provider reasoning: token counts only.
    final_output     the delivered call / boxes, crops count, delivery
    score / cost     the registered readouts, coverage, triage rejects / runtime, tool calls, tokens, model calls
    status           status, error, delivery, fallbacks, retries, the log's warnings / errors for that image
    sources          the files the record was built from; missing files and fields ("not recorded in this run")

`--into-data DATA` also copies the run's non-pixel artifacts to `DATA/runs/<run_id>/` (never crop TIFFs, zips,
montage PNGs or other images), replaces the run's rows in `DATA/cropper-run_image_metrics.csv`, and updates
`DATA/runs/index.json` and `DATA/seeds.json`.

Usage:
    python export_run_records.py --run-dir R [--arms ...] [--logs-dir D] [--out R/audit] [--into-data DATA]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

SCHEMA = "cropper-run-record/1"
ARMS = ("maskcrop", "cellprofiler", "gpt", "react", "shape")
NOT_RECORDED = "not recorded in this run"
METRICS = ("neighbor_contamination", "target_retention", "runtime_per_image", "tool_calls_per_image",
           "tokens_per_image")
_SCORE_KEYS = ("neighbor_contamination", "target_retention", "crop_coverage", "n_crops", "n_expected_crops",
               "n_triage_rejected")
_COST_KEYS = ("runtime_per_image", "tool_calls_per_image", "tokens_per_image", "model_calls_per_image",
              "input_tokens_per_image", "output_tokens_per_image", "cached_tokens_per_image",
              "reasoning_tokens_per_image", "reasoner_calls_per_image", "vision_calls_per_image",
              "reasoner_tokens_per_image", "vision_tokens_per_image")
CSV_FIELDS = ("run_id", "commit", "seed", "n_images", "arm", "image_id", "status", *METRICS, "crop_coverage",
              "n_crops", *_COST_KEYS[3:], "delivery", "n_triage_rejected")
# known seeds: seed -> (drawn, how); seeds.json keeps any drawn / how already written there
KNOWN_SEEDS = {0: (None, "default, the original pilot selection"),
               1881694274: ("2026-10-06", "drawn with Python's secrets module")}
PIXEL_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".zip", ".npy", ".npz"}
_LOG_EVENT = re.compile(r"WARNING|Warning|ERROR|Error|Traceback|Exception|FAIL|[Ff]ailed|Killed|timed out")
_RUN_EVENT = re.compile(r"VALID_RUN|PROBE_FAIL|ARM_FAILED|ARM_DONE|Traceback|ERROR|Error|CANCELLED|TIME LIMIT")
_JOB_HEADER = re.compile(r"^== stage=(\S+) arm=(\S*) .*?\brun_dir=(\S+)", re.M)
_TOOL = "Single_Cell_Cropper_Tool (MaskCrop)"


def _relative(entry: str, run_dir: Path) -> str:
    """A sources entry with its leading path relative to the run dir (posix), so the record reads the same
    wherever the run dir is."""
    for root, prefix in ((str(run_dir), ""), (str(run_dir.parent), "../")):
        if entry.startswith(root):
            path, sep, tail = entry[len(root):].lstrip("\\/").partition(" ")
            return prefix + path.replace("\\", "/") + sep + tail
    return entry


def _load(path: Path, sources: dict):
    if path.is_file():
        sources["used"].append(str(path))
        return json.loads(path.read_text(encoding="utf-8"))
    sources["missing"].append(str(path))
    return None


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _parsed(text):
    """A logged JSON text as its value; other text as is."""
    if isinstance(text, str) and text[:1] in "{[":
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


# --------------------------------------------------------------------------- #
# job logs
# --------------------------------------------------------------------------- #
def find_jobs(run_dir: Path, logs_dirs) -> list[dict]:
    """The jobs that ran this run dir: every `<job>.out` whose header line names a run dir of this name, with its
    `.err` and its `<job>-<arm>.log` files (a STAGE=arm job's .out is that arm's log), in job-id order; then the
    run dir's own per-arm stage logs, `<run_dir>/logs/<arm>.log`, when present."""
    jobs = {}
    for d in logs_dirs:
        if not d or not Path(d).is_dir():
            continue
        for out in sorted(Path(d).glob("*.out")):
            m = _JOB_HEADER.search(out.read_text(encoding="utf-8", errors="replace"))
            if not m or m.group(3).rstrip("/").split("/")[-1] != run_dir.name or out.stem in jobs:
                continue
            arm_logs = {arm: out.with_name(f"{out.stem}-{arm}.log") for arm in ARMS}
            arm_logs = {arm: p for arm, p in arm_logs.items() if p.is_file()}
            if m.group(1) == "arm" and m.group(2) in ARMS:
                arm_logs.setdefault(m.group(2), out)
            err = out.with_suffix(".err")
            jobs[out.stem] = {"job": out.stem, "stage": m.group(1), "out": out,
                              "err": err if err.is_file() else None, "arm_logs": arm_logs}
    key = lambda j: int(re.search(r"(\d+)$", j["job"]).group(1)) if re.search(r"(\d+)$", j["job"]) else 0
    found = sorted(jobs.values(), key=key)
    stage_logs = {arm: run_dir / "logs" / f"{arm}.log" for arm in ARMS}      # per-arm stage logs in the run dir
    stage_logs = {arm: p for arm, p in stage_logs.items() if p.is_file()}
    if stage_logs:
        found.append({"job": "logs/<arm>.log", "stage": None, "out": None, "err": None, "arm_logs": stage_logs})
    return found


def log_events(log_path: Path, arm: str) -> dict:
    """image id -> the warning / error lines its run printed (the lines before its result line)."""
    out, block = {}, []
    result = re.compile(rf"^{re.escape(arm)} (\S+): (ok|error|kept)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = result.match(line)
        if m:
            events = [ln.strip()[:300] for ln in block if _LOG_EVENT.search(ln)]
            out.setdefault(m.group(1), []).extend(dict.fromkeys(events))     # deduplicated, in order
            block = []
        else:
            block.append(line)
    return out


def _run_events(jobs: list[dict]) -> list[str]:
    lines = []
    for job in jobs:
        for path in (job["out"], job["err"]):
            if path is not None:
                lines += [f"{path.name}: {ln.strip()[:300]}" for ln in
                          path.read_text(encoding="utf-8", errors="replace").splitlines() if _RUN_EVENT.search(ln)]
    return list(dict.fromkeys(lines))


# --------------------------------------------------------------------------- #
# the common blocks
# --------------------------------------------------------------------------- #
def _common(run_id: str, arm: str, image_id: str, item: dict | None, index: int | None, selection: dict,
            rec: dict, meta: dict | None, prov: dict | None, events: list, sources: dict) -> dict:
    for key in ("host", "python", "packages", "seed") + (("llm",) if arm in ("gpt", "react", "shape") else ()):
        if meta is not None and key not in meta:
            sources["missing"].append(f"{arm}/run_meta.json [{key}]: {NOT_RECORDED}")
    meta, prov, metrics = meta or {}, prov or {}, rec.get("metrics") or {}
    provenance = {"commit": meta.get("git_commit") or prov.get("git_commit"),
                  "branches_containing_commit": prov.get("branches_containing_commit"),
                  "model": meta.get("model"), "host": meta.get("host"), "env": meta.get("env"),
                  "python": meta.get("python"), "packages": meta.get("packages"), "llm": meta.get("llm"),
                  "seed": meta.get("seed", selection.get("seed")), "selection_created_utc": selection.get("created_utc"),
                  "arm_started_utc": meta.get("started_utc"), "arm_resumed_utc": meta.get("resumed_utc"),
                  "selection_sha256": meta.get("selection_sha256"), "run_status": prov.get("status"),
                  "run_problems": prov.get("problems")}
    if arm == "cellprofiler":
        provenance.update({k: prov.get(k) for k in ("cellprofiler_image", "cellprofiler_image_sha256",
                                                     "cellprofiler_version", "cellprofiler_module")})
    if arm in ("maskcrop", "react", "shape"):
        provenance.update({k: prov.get(k) for k in ("opencv_distribution", "opencv_version")})
    return {
        "schema": SCHEMA, "run_id": run_id, "arm": arm,
        "input": {"image_id": image_id, "image_path": (item or {}).get("image_path"),
                  "mask_path": (item or {}).get("mask_path"), "gt_path": (item or {}).get("gt_path"),
                  "mask_source": selection.get("mask_source"), "seed": selection.get("seed"),
                  "selection_index": index, "min_area": meta.get("min_area"),
                  "ground_truth": "the input mask is the ground-truth instance mask; the same mask scores the crops"},
        "provenance": provenance,
        "score": {k: metrics.get(k) for k in _SCORE_KEYS},
        "cost": {k: metrics.get(k) for k in _COST_KEYS},
        "status": {"status": rec.get("status"), "error": rec.get("error"),
                   "delivery": metrics.get("delivery", (rec.get("trace") or {}).get("delivery")),
                   "fallbacks": [], "retries": 0, "log_events": events},
    }


# --------------------------------------------------------------------------- #
# the domain tools
# --------------------------------------------------------------------------- #
def tool_record(out: dict, arm: str, rec: dict, meta: dict | None) -> dict:
    meta, trace, metrics = meta or {}, rec.get("trace") or {}, rec.get("metrics") or {}
    settings = {"min_area": meta.get("min_area"), "margin": meta.get("margin"),
                "margin_frac": meta.get("maskcrop_margin_frac"), "margin_mode": "fraction", "recenter": False}
    if arm == "maskcrop":
        out["config"] = {"tool": _TOOL, "settings": "the tool at its fraction-mode defaults, one call", **settings}
        result = trace.get("result") or {}
        tool_result = {k: result.get(k) for k in ("summary", "cell_count", "total_detected_objects",
                                                   "filtered_by_area", "filtered_by_border", "invalid_crop_data",
                                                   "seconds")}
    else:
        out["config"] = {"tool": "CellProfiler SaveCroppedObjects (native), then the MaskCrop square-geometry "
                                 "adapter (a control: its scored crops share MaskCrop's geometry)",
                         "cellprofiler_python": meta.get("cellprofiler_python"), **settings,
                         "square_normalization": trace.get("square_normalization")}
        tool_result = {k: trace.get(k) for k in ("backend", "module", "cellprofiler_version",
                                                  "cellprofiler_core_version", "native_crop_count", "stdout")}
    out["steps"] = [{"step": "tool_call", "by": "the benchmark (no agent, no LLM)",
                     "tool_call": {"tool": out["config"]["tool"], "arguments": settings},
                     "tool_result": {"status": "failed" if rec.get("error") else "success",
                                     "error": rec.get("error"), **tool_result}}]
    out["final_output"] = {"how": "every target's crop from the single call", "n_crops": metrics.get("n_crops")}
    return out


# --------------------------------------------------------------------------- #
# gpt
# --------------------------------------------------------------------------- #
def gpt_record(out: dict, rec: dict, meta: dict | None, sources: dict, where: str) -> dict:
    meta, trace, metrics = meta or {}, rec.get("trace") or {}, rec.get("metrics") or {}
    calls = trace.get("calls")
    out["config"] = {"agent": "the model alone, one shot (no tool, no feedback)",
                     "sees": "the input image with the mask's object boundaries, every object of the mask "
                             "(cell_label, bbox, area), the image shape and the targets of its request",
                     "output": "one square box per target as JSON", "model": meta.get("model"),
                     "batch_size": meta.get("gpt_batch_size"), "image_attached": trace.get("image_attached"),
                     "params": (calls[0].get("params") if calls else NOT_RECORDED)}
    steps = []
    if calls is not None:
        for c in calls:
            steps.append({"step": "llm_call", "by": "gpt", **{k: c.get(k) for k in (
                "batch_index", "attempt", "targets", "prompt", "images", "params", "reply", "usage", "error",
                "seconds")}, "result": "rejected: " + str(c["error"]) if c.get("error") else "accepted"})
        out["status"]["retries"] = sum(1 for c in calls if (c.get("attempt") or 1) > 1)
    else:
        sources["missing"].append(f"{where} [trace.calls: every request's prompt, images, reply]: {NOT_RECORDED}")
        for b in trace.get("batches") or []:
            steps.append({"step": "llm_batch", "by": "gpt", "batch_index": b.get("index"),
                          "n_targets": b.get("n_targets"), "attempts": b.get("attempts"),
                          "note": f"the batch's prompt, images and replies: {NOT_RECORDED}"})
        out["status"]["retries"] = sum(max(0, (b.get("attempts") or 1) - 1) for b in trace.get("batches") or [])
        if not trace:
            sources["missing"].append(f"{where} [trace]: {NOT_RECORDED} (a failed image)")
    out["steps"] = steps
    out["final_output"] = {"how": "the boxes of the model's accepted replies, one per target",
                           "n_crops": metrics.get("n_crops"),
                           "n_batches": len(trace.get("batches") or []) if trace else None}
    return out


# --------------------------------------------------------------------------- #
# react
# --------------------------------------------------------------------------- #
def _thought(content):
    """The `thought` the reply stated (a schema reply as JSON, or as its recorded text)."""
    if isinstance(content, str):
        content = _parsed(content)
    if isinstance(content, dict):
        return content.get("thought")
    if isinstance(content, str):
        m = re.match(r"thought=(['\"])(.*?)\1 action=", content, re.S)
        return m.group(2) if m else None
    return None


_REACT_TOOL_KEYS = ("step", "action", "call", "parent", "change", "cell_rules", "thought", "readouts", "guard",
                    "error", "seconds")


def react_runner_record(out: dict, rec: dict, meta: dict | None, sources: dict, where: str) -> dict:
    """The record of a ReAct image run on the shared runner (benchmarks/_shared/react_arm.py): its rounds in order,
    each model reply (prompt digest, image refs, reply as returned, the action and thought) followed by its tool
    calls (parameters, the tool's output, images, error, seconds)."""
    meta, trace, metrics = meta or {}, rec.get("trace") or {}, rec.get("metrics") or {}
    out["config"] = {"agent": "an LLM plus the tool (the shared ReAct runner)", "runner": trace.get("runner"),
                     "tool": _TOOL, "model": meta.get("model"), "cores": meta.get("cores"),
                     "round_cap": trace.get("round_cap"), "budget": trace.get("budget"),
                     "default_counts_as_round": trace.get("default_counts_as_round"),
                     "tool_budget": meta.get("react_tool_budget"), "deadline": meta.get("deadline"),
                     "parameters": trace.get("parameters"),
                     "sees": "SHAPE's task text, the input (image, mask, min_area), the parameter table (name, type, "
                             "default), the budget and the time left; every call's tool output and its crop tiles"}
    calls = {c.get("call_id"): c for c in trace.get("tool_calls") or []}
    steps = []
    for r in trace.get("rounds_trace") or []:
        if r.get("kind") != "default":
            reply = r.get("reply")
            action = r.get("action") or {}
            steps.append({"step": "llm_call", "by": "react", "round": r.get("round"), "kind": r.get("kind"),
                          "prompt": NOT_RECORDED, "prompt_sha256": r.get("prompt_sha256"), "images": r.get("images"),
                          "reply": reply.get("content") if isinstance(reply, dict) and "content" in reply else reply,
                          "usage": reply.get("usage") if isinstance(reply, dict) else None,
                          "rationale": action.get("thought"), "action": action.get("action"),
                          "deliver": action.get("deliver"),
                          "error": "; ".join(o["error"] for o in r.get("observations") or [] if o.get("error")
                                             and "call_id" not in o) or None})
        for c in r.get("calls") or []:
            call = calls.get(c.get("call_id")) or {}
            steps.append({"step": "tool_call", "round": r.get("round"), "call": c.get("call_id"),
                          "by": "the benchmark (the tool at its defaults)" if r.get("kind") == "default" else "react",
                          "tool_call": {"tool": _TOOL, "params": c.get("params")},
                          "tool_result": {"status": "failed" if call.get("error") else "success",
                                          "error": call.get("error"), "output": call.get("output"),
                                          "images": call.get("images"), "seconds": call.get("seconds")}})
    if not trace:
        sources["missing"].append(f"{where} [trace]: {NOT_RECORDED} (a failed image)")
    elif trace.get("rounds_trace"):
        sources["missing"].append(f"{where} [prompt text]: {NOT_RECORDED} (the runner records its sha256)")
    out["steps"] = steps
    delivery = trace.get("delivery") or metrics.get("delivery")
    out["final_output"] = {"delivered_call": trace.get("delivered_call"), "delivery": delivery,
                           "how": ("the call the agent's finish named" if delivery == "agent_finish" else
                                   "no valid finish: the default call" if delivery else None),
                           "n_crops": metrics.get("n_crops"), "rounds": trace.get("rounds")}
    out["status"]["fallbacks"] = [delivery] if delivery and delivery != "agent_finish" else []
    out["status"]["step_errors"] = [s["error"] for s in steps if s.get("error")] + [
        s["tool_result"]["error"] for s in steps if s["step"] == "tool_call" and s["tool_result"]["error"]]
    return out


def react_record(out: dict, rec: dict, meta: dict | None, sources: dict, where: str) -> dict:
    meta, trace, metrics = meta or {}, rec.get("trace") or {}, rec.get("metrics") or {}
    if "rounds_trace" in trace or (meta.get("react_runner") and not trace.get("trace")):
        return react_runner_record(out, rec, meta, sources, where)
    out["config"] = {"agent": "ReAct (think, act, observe) with MaskCrop as its only tool",
                     "tool": _TOOL, "model": meta.get("model"), "cores": meta.get("cores"),
                     "concurrency_cap": meta.get("concurrency_cap"), "max_rounds": meta.get("max_rounds"),
                     "tool_budget": meta.get("react_tool_budget", trace.get("tool_budget")),
                     "deadline": meta.get("deadline"), "image_attached": trace.get("image_attached"),
                     "sees": "SHAPE's task, objectives and admissibility rule, the CARD's levers, the input facts, "
                             "every call's readouts and guard, SHAPE's round images; replies in ReActAction"}
    steps, no_prompt = [], 0
    for s in trace.get("trace") or []:
        if s.get("action") == "maskcrop":
            first = s.get("parent") == "input" and s.get("step") == 0
            steps.append({
                "step": "tool_call", "by": "the benchmark (the tool at its defaults)" if first else "react (CALL)",
                "call": s.get("call"), "parent": s.get("parent"), "rationale": s.get("thought"),
                "tool_call": {"tool": _TOOL, "change": s.get("change"), "cell_rules": s.get("cell_rules"),
                              **{k: v for k, v in s.items() if k not in _REACT_TOOL_KEYS}},
                "tool_result": {"status": "failed" if s.get("error") else "success", "error": s.get("error"),
                                "readouts": s.get("readouts"), "guard": s.get("guard"), "seconds": s.get("seconds")}})
            continue
        raw = s.get("raw_reply")
        content = raw.get("content") if isinstance(raw, dict) and "content" in raw else raw
        no_prompt += "prompt" not in s
        steps.append({"step": "llm_call", "by": "react", "attempt": s.get("attempt"),
                      "prompt": s.get("prompt", NOT_RECORDED), "images": s.get("images", NOT_RECORDED),
                      "reply": content, "usage": raw.get("usage") if isinstance(raw, dict) else None,
                      "seconds": s.get("seconds"), "rationale": _thought(content), "action": s.get("action"),
                      "deliver": s.get("deliver"), "error": s.get("error")})
    if no_prompt:
        sources["missing"].append(f"{where} [trace.trace[].prompt, images, seconds of {no_prompt} model step(s)]: "
                                  f"{NOT_RECORDED}")
    if not trace:
        sources["missing"].append(f"{where} [trace]: {NOT_RECORDED} (a failed image)")
    out["steps"] = steps
    delivery = trace.get("delivery") or metrics.get("delivery")
    out["final_output"] = {"delivered_call": trace.get("delivered_call"), "delivery": delivery,
                           "how": ("the call FINISH named" if delivery == "finish" else delivery),
                           "n_crops": metrics.get("n_crops")}
    out["status"]["fallbacks"] = [delivery] if str(delivery or "").startswith("fallback") else []
    out["status"]["retries"] = sum(1 for s in steps if s["step"] == "llm_call" and (s.get("attempt") or 1) > 1)
    out["status"]["step_errors"] = [s["error"] for s in steps if s.get("error")]
    return out


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #
_SHAPE_BY = {"admission": "system rule (the admission on the defaults call)",
             "rejected_move": "system rule (move check: the move was rejected)",
             "ignored_change": "system rule (move check: a change was dropped)",
             "dropped_move": "system rule (the round's call cap)",
             "dropped_auto_tighten": "system rule (cell rules: auto_tighten acts only in mask_bbox)",
             "fallback": "system (deterministic fallback moves: the reasoner gave no plan)",
             "plateau": "system rule (plateau stop)", "plan_delivery": "system (the plan's delivery checked)",
             "choice": "system (the delivered call)", "reference": "system (reference calls)",
             "reference_rejected": "system (reference calls)"}


def call_index(label: str, details: dict) -> int | None:
    """A call's trial index, from the record's rounds: "input" is trial 0, then each round's calls in run order (the
    names the run used: "default", "reference", "agent_<n>", "fallback_<n>", ..., or c01, c02, ... in the records
    before the role names, e.g. the reported run made at ea78bb4b); None for a name no round holds."""
    names = ["input"] + [c for calls in details.get("rounds") or [] for c in calls]
    return names.index(label) if label in names else None


def _shape_call_step(label: str, details: dict, round_of: dict) -> dict:
    calls, entries = details.get("calls") or {}, details.get("entries") or []
    index = call_index(label, details)
    entry = (entries[index] if index is not None and index < len(entries) else None) or {}
    call = calls.get(label) or {}
    failed = next((f for f in details.get("failed") or [] if isinstance(f, dict) and f.get("call") == label), None)
    r = round_of.get(label)
    return {"step": "tool_call", "by": "the loop (round 1: the tool at its defaults)" if r == 1 else
            "the loop (the agent's move)" if not str(entry.get("hypothesis") or "").startswith("fallback")
            else "the loop (a deterministic fallback move)",
            "call": label, "round": r, "parent": call.get("parent"),
            "tool_call": {"tool": _TOOL, "change": entry.get("param_delta"),
                          "configuration": call.get("configuration")},
            "tool_result": {"status": "failed" if failed else "success",
                            "error": (failed or {}).get("error"), "readouts": call.get("readouts"),
                            "admissible": label in (details.get("admissible") or []),
                            "metadata_path": call.get("metadata_path"), "outcome": entry.get("outcome"),
                            "score_delta_vs_parent": entry.get("score_delta_vs_parent")}}


def shape_steps(details: dict, log: list[dict] | None, sources: dict, where: str) -> list[dict]:
    rounds = details.get("rounds") or []
    round_of = {c: i + 1 for i, calls in enumerate(rounds) for c in calls}
    steps, done = [], set()

    def flush(upto: int) -> None:
        for i in range(min(upto, len(rounds))):
            if i not in done:
                done.add(i)
                steps.extend(_shape_call_step(c, details, round_of) for c in rounds[i])

    flush(1)                                            # round 1: the defaults call, before the admission
    if log is None:                                     # no log: the calls, then the plans the details hold
        flush(len(rounds))
        for p in details.get("plans") or []:
            steps.append({"step": "plan", "by": "the agent (reasoner)", "round": p.get("round"),
                          "question": p.get("question"), "rationale": p.get("analysis"),
                          "hypotheses": p.get("hypotheses"), "moves": p.get("moves"), "deliver": p.get("deliver"),
                          "note": f"the question's prompt and reply: {NOT_RECORDED} (no agent_log.jsonl)"})
        return steps
    n_round, current, no_images = 0, None, 0
    for row in log:
        kind, label = row.get("kind"), row.get("label")
        if kind == "prompt":
            if label == "round":
                n_round += 1
                flush(n_round)                          # the calls the previous round question planned
            elif label != "repair":
                flush(len(rounds))
            no_images += "images" not in row
            current = {"step": "llm_call", "by": "the agent (reasoner)", "question": label,
                       "round": n_round + 1 if label in ("round", "repair") else None, "t": row.get("t"),
                       "prompt": row.get("text"), "images": row.get("images", NOT_RECORDED),
                       "usage": [], "reply": None, "rationale": None, "errors": []}
            steps.append(current)
        elif kind in ("usage", "reasoning", "reasoner_error") and current is not None:
            if kind == "usage":
                current["usage"].append(_parsed(row.get("text")))
            elif kind == "reasoning":
                reply = current["reply"] = _parsed(row.get("text"))
                if isinstance(reply, dict):
                    current["rationale"] = reply.get("analysis") or reply.get("reasoning")
            else:
                current["errors"].append({"error": row.get("text"), "reply": row.get("reply", NOT_RECORDED)})
        else:
            if kind in ("plateau", "plan_delivery", "choice"):
                flush(len(rounds))
            steps.append({"step": kind, "by": _SHAPE_BY.get(kind, "the loop"), "t": row.get("t"),
                          "result": _parsed(row.get("text"))})
    flush(len(rounds))
    if no_images:
        sources["missing"].append(f"{where} [prompt images of {no_images} question(s)]: {NOT_RECORDED}")
    return steps


def shape_record(out: dict, rec: dict, meta: dict | None, details: dict | None, log: list[dict] | None,
                 sources: dict, img_dir: Path) -> dict:
    meta, trace, metrics = meta or {}, rec.get("trace") or {}, rec.get("metrics") or {}
    out["config"] = {"agent": "SHAPE (CropAgenticVerifier, the shared agentic VRR loop) with MaskCrop",
                     "tool": _TOOL, "model": meta.get("model"), "shape_loop": trace.get("shape_loop"),
                     "cores": meta.get("cores"), "concurrency_cap": meta.get("concurrency_cap"),
                     "max_rounds": meta.get("max_rounds"), "deadline": meta.get("deadline")}
    d = details or {}
    steps = shape_steps(d, log, sources, str(img_dir / "agent_log.jsonl")) if details or log is not None else []
    if details:
        steps.append({"step": "final_choice", "by": "the agent (the plan's delivery or the final question), "
                                                   "checked by the system", "call": d.get("call"),
                      "choice": d.get("choice"), "plan_delivery": d.get("plan_delivery"),
                      "final_answers": d.get("final_answers"), "stop_reason": d.get("stop_reason"),
                      "verifier_stop": d.get("verifier_stop")})
        mc = d.get("mask_check")
        if mc:
            steps.append({"step": "mask_check", "by": "vision LLM on the delivered call (a measurement; no rule "
                                                       "or prompt reads it)",
                          **{k: v for k, v in mc.items() if k != "montages"},
                          "verdicts": mc.get("verdicts", NOT_RECORDED)})
            for m in mc.get("montages") or []:
                steps.append({"step": "vision_call", "by": "the mask check", **m})
            if "montages" not in mc:
                sources["missing"].append(f"{img_dir / 'agent_details.json'} [mask_check.montages, verdicts]: "
                                          f"{NOT_RECORDED}")
    else:
        steps.append({"step": "final_choice", "by": "the agent, checked by the system",
                      "call": trace.get("delivered_call"), "choice": trace.get("delivery"),
                      "note": f"the questions, calls and mask check: {NOT_RECORDED} (no agent_details.json)"})
    out["steps"] = steps
    out["final_output"] = {"delivered_call": trace.get("delivered_call", d.get("call")),
                           "delivery": trace.get("delivery", d.get("choice")), "n_crops": metrics.get("n_crops"),
                           "n_triage_rejected": metrics.get("n_triage_rejected"),
                           "delivered_metadata_path": trace.get("delivered_metadata_path")}
    errors = [e for s in steps if s.get("step") == "llm_call" for e in s.get("errors") or []]
    out["status"].update({
        "fallbacks": [s["result"] for s in steps if s.get("step") == "fallback"],
        "retries": len(errors), "reasoner_errors": [e["error"] for e in errors],
        "loop_status": trace.get("status", d.get("status")), "loop_reason": trace.get("reason", d.get("reason")),
        "stop_reason": d.get("stop_reason"), "reasoner_failures": d.get("reasoner_failures"),
        "fallback_rounds": d.get("fallback_rounds"), "failed_calls": d.get("failed")})
    return out


# --------------------------------------------------------------------------- #
def export(run_dir: Path, arms=ARMS, logs_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "audit"
    run_id = run_dir.name
    run_sources = {"used": [], "missing": []}
    selection = _load(run_dir / "selection.json", run_sources) or {}
    prov = _load(run_dir / "provenance.json", run_sources)
    items = selection.get("selected_images") or []
    item_by_id = {it["image_id"]: (i, it) for i, it in enumerate(items)}
    jobs = find_jobs(run_dir, [logs_dir] if logs_dir else [run_dir / "logs", run_dir.parent])
    if not jobs:
        run_sources["missing"].append(f"job logs (a <job>.out naming run dir {run_id}): none found")
    index, csv_rows = [], []
    for arm in arms:
        records_path = run_dir / arm / "records.jsonl"
        if not records_path.is_file():
            continue
        meta_sources = {"used": [], "missing": []}
        meta = _load(run_dir / arm / "run_meta.json", meta_sources)
        events: dict = {}
        for job in jobs:
            if arm in job["arm_logs"]:
                for image, lines in log_events(job["arm_logs"][arm], arm).items():
                    events.setdefault(image, []).extend(lines)
        grouped: dict = {}
        for rec in _jsonl(records_path):
            grouped.setdefault(str(rec.get("image_id")), []).append(rec)
        ids = [it["image_id"] for it in items] + sorted(set(grouped) - set(item_by_id))
        for image in ids:
            sources = {"used": [*run_sources["used"], *meta_sources["used"], str(records_path)],
                       "missing": [*run_sources["missing"], *meta_sources["missing"]]}
            sources["used"] += [str(j["arm_logs"][arm]) for j in jobs if arm in j["arm_logs"]]
            recs = grouped.get(image) or [{"image_id": image, "status": "missing",
                                           "error": "no record for the selected image", "metrics": {}}]
            rec = recs[-1]
            idx, item = item_by_id.get(image, (None, None))
            out = _common(run_id, arm, image, item, idx, selection, rec, meta, prov,
                          list(dict.fromkeys(events.get(image, []))), sources)
            if len(recs) > 1:
                out["status"]["n_records_for_image"] = len(recs)
            where = f"{records_path}"
            if arm in ("maskcrop", "cellprofiler"):
                record = tool_record(out, arm, rec, meta)
            elif arm == "gpt":
                record = gpt_record(out, rec, meta, sources, where)
            elif arm == "react":
                record = react_record(out, rec, meta, sources, where)
            else:
                img_dir = run_dir / "shape" / "images" / image
                details = _load(img_dir / "agent_details.json", sources)
                log_path = img_dir / "agent_log.jsonl"
                log = _jsonl(log_path) if log_path.is_file() else None
                sources["used" if log is not None else "missing"].append(str(log_path))
                record = shape_record(out, rec, meta, details, log, sources, img_dir)
            record["sources"] = {k: [_relative(e, run_dir) for e in v] for k, v in sources.items()}
            path = out_dir / arm / f"{image}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            metrics = rec.get("metrics") or {}
            index.append({"arm": arm, "image_id": image, "path": path.relative_to(out_dir).as_posix(),
                          "status": rec.get("status"), "error": rec.get("error"),
                          "delivery": record["status"]["delivery"],
                          **{k: metrics.get(k) for k in ("neighbor_contamination", "target_retention")},
                          "steps": len(record["steps"]), "missing_sources": len(sources["missing"])})
            csv_rows.append({"arm": arm, "image_id": image, "status": rec.get("status"),
                             **{k: metrics.get(k) for k in (*METRICS, *_COST_KEYS[3:], "crop_coverage", "n_crops",
                                                            "delivery", "n_triage_rejected")}})
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema": SCHEMA, "run_id": run_id, "run_dir": str(run_dir),
               "commit": (prov or {}).get("git_commit"), "seed": selection.get("seed"), "n_images": len(items),
               "arms": sorted({r["arm"] for r in index}, key=ARMS.index),
               "logs": {"jobs": [j["job"] for j in jobs], "run_events": _run_events(jobs)},
               "run_sources": {k: [_relative(e, run_dir) for e in v] for k, v in run_sources.items()},
               "n_records": len(index), "records": index}
    (out_dir / "index.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    summary["_csv_rows"], summary["_jobs"] = csv_rows, jobs
    return summary


# --------------------------------------------------------------------------- #
# into data/
# --------------------------------------------------------------------------- #
def _copy(src: Path, dst: Path) -> None:
    if src.is_file() and src.suffix.lower() not in PIXEL_SUFFIXES:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    for path in sorted(src.rglob("*")) if src.is_dir() else []:
        _copy(path, dst / path.relative_to(src))


def into_data(run_dir: Path, data_dir: Path, summary: dict, out_dir: Path) -> Path:
    """Copies the run's non-pixel artifacts to DATA/runs/<run_id>/, replaces its rows in the per-image CSV and
    updates runs/index.json and seeds.json."""
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    run_id = run_dir.name
    dest = data_dir / "runs" / run_id
    if dest.resolve() != run_dir.resolve():
        if dest.exists():
            shutil.rmtree(dest)                                     # a re-export replaces the run's copy
        for name in ("selection.json", "provenance.json", "RUN_META.json", "COMPLETE"):
            _copy(run_dir / name, dest / name)
        for path in sorted(run_dir.glob("RUN_META_resume_*.json")):        # one per resume job
            _copy(path, dest / path.name)
        for arm in ARMS:
            for name in ("run_meta.json", "records.jsonl"):
                _copy(run_dir / arm / name, dest / arm / name)
        _copy_tree(run_dir / "compare", dest / "compare")
        for img_dir in sorted((run_dir / "shape" / "images").glob("*")):
            for name in ("agent_log.jsonl", "agent_details.json"):
                _copy(img_dir / name, dest / "shape" / "images" / img_dir.name / name)
        for job in summary["_jobs"]:
            for path in [job["out"], job["err"], *job["arm_logs"].values()]:
                if path is not None:
                    _copy(path, dest / "logs" / path.name)
        _copy_tree(out_dir, dest / "audit")
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    prov_path = run_dir / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
    seed, n_images, commit = selection.get("seed"), len(selection.get("selected_images") or []), summary["commit"]

    csv_path = data_dir / "cropper-run_image_metrics.csv"
    rows = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("run_id") != run_id]
    rows += [{"run_id": run_id, "commit": commit, "seed": seed, "n_images": n_images, **r}
             for r in summary["_csv_rows"]]
    rows.sort(key=lambda r: r["run_id"])        # stable: a re-exported run keeps its place and its rows' order
    data_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    index_path = data_dir / "runs" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"runs": []}
    runs = [r for r in index.get("runs", []) if r.get("run_id") != run_id]
    runs.append({"run_id": run_id, "commit": commit, "seed": seed, "n_images": n_images,
                 "date": selection.get("created_utc"), "arms": summary["arms"], "status": prov.get("status"),
                 "problems": prov.get("problems", [])})
    runs.sort(key=lambda r: (str(r.get("date") or ""), r["run_id"]))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"schema": "cropper-data-runs/1", "runs": runs}, indent=1) + "\n",
                          encoding="utf-8")

    seeds_path = data_dir / "seeds.json"
    seeds = json.loads(seeds_path.read_text(encoding="utf-8")).get("seeds", []) if seeds_path.is_file() else []
    for entry in seeds:
        entry["runs"] = [r for r in entry.get("runs", []) if r != run_id]
    entry = next((e for e in seeds if e.get("seed") == seed), None)
    if entry is None:
        drawn, how = KNOWN_SEEDS.get(seed, (None, NOT_RECORDED))
        entry = {"seed": seed, "drawn": drawn, "how": how, "runs": []}
        seeds.append(entry)
    entry["runs"] = sorted({*entry["runs"], run_id})
    seeds = sorted((e for e in seeds if e.get("runs") or e.get("seed") in KNOWN_SEEDS),
                   key=lambda e: (e.get("seed") is None, e.get("seed") or 0))
    seeds_path.write_text(json.dumps({"seeds": seeds}, indent=1) + "\n", encoding="utf-8")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="holds selection.json and <arm>/records.jsonl")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
                    help="default: every arm (an arm without records is skipped)")
    ap.add_argument("--logs-dir", type=Path,
                    help="where the job logs are (default: <run-dir>/logs and the run dir's parent)")
    ap.add_argument("--out", type=Path, help="default <run-dir>/audit")
    ap.add_argument("--into-data", type=Path, help="the benchmark's data/ dir: copy the run's non-pixel artifacts")
    args = ap.parse_args()
    out_dir = args.out or args.run_dir / "audit"
    s = export(args.run_dir, tuple(args.arms), args.logs_dir, out_dir)
    missing = sum(r["missing_sources"] for r in s["records"])
    print(f"[audit] {s['n_records']} records -> {out_dir} ({missing} missing source(s) across records; "
          f"job logs: {', '.join(s['logs']['jobs']) or 'none'})")
    if args.into_data:
        dest = into_data(args.run_dir, args.into_data, s, out_dir)
        print(f"[data] {args.run_dir.name} -> {dest}")


if __name__ == "__main__":
    main()
