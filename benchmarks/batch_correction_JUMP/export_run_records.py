#!/usr/bin/env python3
"""The record set of one six-arm run, for `results/<run_id>/` (no matrices), laid out as the CellSeg benchmark's.

From a run directory (`<run>/` holding `prep/` and `suite/`; an
earlier record set of this layout works too), copies every small file a reader needs to check the reported numbers
and writes the comparison and the audit records built from them:

    <run_id>/RUN_META.json, COMPLETE         the run's provenance and completion marker
    <run_id>/prep/PREPARE.json               the scenario's wells, plates, sources and the val / test split
    <run_id>/control/CONTROL.json            raw, harmony, scanorama: configuration, tool record, scores, halves
    <run_id>/<arm>/                          each LLM arm's own files (ARM_OUTPUT.json, tool-call ledgers; SHAPE's
                                             traces, decision, notes and result), as the suite wrote them
    <run_id>/compare/                        compare.json and readouts.csv (the suite's report), paired_vs_shape.csv
                                             and summary.md
    <run_id>/audit/<arm>.json                one audit record per arm, + index.json
    <run_id>/logs/                           the run's log (`--log`)
    <run_id>/MANIFEST.json                   every file with its size and sha256, the expected files missing

Matrices (`*.npy`, `*.parquet`) stay on scratch. The suite's BENCHMARK_REPORT.md is replaced by compare/summary.md
(the same table with the overall score and the paired deltas); suite/COMPLETE is the run's COMPLETE.

Usage:
    python benchmarks/batch_correction_JUMP/export_run_records.py --run-dir <run> --log <log> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

SCHEMA = "batch-jump-run-record/1"
ARMS = ("raw", "harmony", "scanorama", "gpt_luna", "react_luna_harmony", "shape_luna_harmony")
CONTROL_ARMS = ("raw", "harmony", "scanorama")
SHAPE = "shape_luna_harmony"
METRICS = ("overall", "batch_mean", "bio_mean", "nonrep_mean_map", "negcon_mean_map")
SKIP = (".npy", ".parquet")
RENAMED = {"suite/BENCHMARK_REPORT.json": "compare/compare.json", "suite/BENCHMARK_REPORT.csv": "compare/readouts.csv",
           "prep/shape_control/CONTROL.json": "control/CONTROL.json"}
DROPPED = ("suite/BENCHMARK_REPORT.md", "suite/COMPLETE", "MANIFEST.json")
EXPECTED = ("RUN_META.json", "COMPLETE", "prep/PREPARE.json", "control/CONTROL.json", "compare/compare.json",
            "compare/readouts.csv", "compare/paired_vs_shape.csv", "compare/summary.md", "audit/index.json",
            "gpt_luna/ARM_OUTPUT.json", "react_luna_harmony/ARM_OUTPUT.json",
            "react_luna_harmony/logs/tool_calls.jsonl", f"{SHAPE}/RESULT.json", f"{SHAPE}/logs/tool_calls.jsonl",
            f"{SHAPE}/traces/round_00.jsonl", f"{SHAPE}/submission/decision.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(rel: str) -> str:
    if rel in RENAMED:
        return RENAMED[rel]
    return rel[len("suite/"):] if rel.startswith("suite/") else rel


def record_files(run_dir: Path) -> list[Path]:
    """The run's files the record set holds, relative to `run_dir`."""
    picked = [p for p in run_dir.iterdir() if p.is_file()]
    picked += [run_dir / name for name in ("prep/PREPARE.json", "prep/shape_control/CONTROL.json")
               if (run_dir / name).is_file()]
    picked += [p for p in (run_dir / "suite").rglob("*") if p.is_file() and p.suffix not in SKIP]
    return sorted(p.relative_to(run_dir) for p in picked if p.relative_to(run_dir).as_posix() not in DROPPED)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _signed(value) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _rows(out: Path) -> dict:
    report = _load(out / "compare" / "compare.json") or {}
    return {row["arm_id"]: row for row in report.get("rows", [])}


def paired_rows(result: dict | None) -> list[dict]:
    """SHAPE minus each control arm on the test view (the report's plate-half band) and on all wells."""
    if not result:
        return []
    rows = []
    for arm in CONTROL_ARMS:
        test = (result.get("comparisons_test") or {}).get(f"agent_vs_{arm}") or {}
        all_wells = (result.get("comparisons_all_wells") or {}).get(f"agent_vs_{arm}") or {}
        for metric in METRICS:
            if metric in test:
                cell = test[metric]
                rows.append({"arm": arm, "metric": metric, "n_halves": cell.get("n_halves"),
                             "mean_delta (shape - arm)": cell.get("delta"), "band_2se": cell.get("band_2se"),
                             "beyond_band": cell.get("significant"), "all_wells_delta": all_wells.get(metric)})
    return rows


def write_compare(out: Path) -> None:
    rows = _rows(out)
    report = _load(out / "compare" / "compare.json") or {}
    paired = paired_rows(_load(out / SHAPE / "RESULT.json"))
    columns = ("arm", "metric", "n_halves", "mean_delta (shape - arm)", "band_2se", "beyond_band", "all_wells_delta")
    with (out / "compare" / "paired_vs_shape.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired)
    n = next((row.get("n_images") for row in rows.values()), None)
    lines = [f"# JUMP batch correction, {report.get('scenario', '?')}: readouts per arm", "",
             f"## test view (n = {n:,} well profiles in the run; scored on the evaluation compounds and the controls)"
             if n else "## test view", "",
             "| arm | overall | batch_mean | bio_mean | nonrep mAP | runtime (s) | tool calls | tokens "
             "| nonrep mAP gain / tool call |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm in report.get("roster", list(rows)):
        row = rows.get(arm)
        if row is None:
            continue
        gain = row.get("nonrep_map_gain_per_tool_call")
        lines.append(f"| {arm} | {row['overall']:.4f} | {row['batch_correction']:.4f} | {row['biometrics']:.4f} "
                     f"| {row['nonrep_mean_map']:.4f} | {row['runtime_seconds']:.0f} | {row['tool_calls']} "
                     f"| {row['tokens']:,} | {'n/a' if gain is None else f'{gain:.4f}'} |")
    if paired:
        lines += ["", f"SHAPE minus arm, test view (delta, 2SE band from {paired[0]['n_halves']} plate halves; "
                      "beyond / within the band; all-wells delta in brackets):", ""]
        for arm in CONTROL_ARMS:
            cells = [r for r in paired if r["arm"] == arm]
            if cells:
                lines.append(f"- vs {arm}: " + "; ".join(
                    f"{r['metric']} {_signed(r['mean_delta (shape - arm)'])} ({r['band_2se']:.4f}, "
                    f"{'beyond' if r['beyond_band'] else 'within'}) [{_signed(r['all_wells_delta'])}]"
                    for r in cells))
        lines += ["", "The report computes no band for SHAPE against gpt_luna or react_luna_harmony."]
    if report.get("metric_note"):
        lines += ["", report["metric_note"]]
    (out / "compare" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _readouts(view: dict | None) -> dict | None:
    return None if view is None else {metric: view.get(metric) for metric in METRICS}


def _shape_steps(out: Path) -> list[dict]:
    """SHAPE's plans (admission, rounds, stop checks, final choice) and tool calls, in run order."""
    events = []
    for event in _jsonl(out / SHAPE / "traces" / "round_00.jsonl"):
        label = event.get("label")
        if event.get("kind") == "thinking":
            try:
                plan = json.loads(event.get("text") or "null")
            except json.JSONDecodeError:
                plan = event.get("text")
            events.append((event.get("time", 0), {"kind": label, "plan": plan}))
        elif event.get("kind") == "message" and label in ("ignored_change", "plateau", "choice", "stopped"):
            events.append((event.get("time", 0), {"kind": label, "text": event.get("text")}))
    calls: dict[str, dict] = {}
    for event in _jsonl(out / SHAPE / "logs" / "tool_calls.jsonl"):
        call = calls.setdefault(event["call"], {"kind": "tool_call", "call": event["call"]})
        if event.get("event") == "start":
            call.update({"time": event.get("time", 0), "parent": event.get("parent"),
                         "rationale": event.get("diagnosis"), "expected": event.get("expected"),
                         "tool_call": {"set": event.get("set"), "config": event.get("config")}})
        else:
            call["tool_result"] = {"status": event.get("status"), "tool_status": event.get("tool_status"),
                                   "calibration": event.get("val"), "delta_vs_parent": event.get("delta_vs_parent"),
                                   "seconds": event.get("seconds")}
    events += [(call.pop("time", 0), call) for call in calls.values()]
    return [step for _, step in sorted(events, key=lambda item: item[0])]


def audit_record(out: Path, arm: str, run_meta: dict, prepare: dict, control: dict, rows: dict) -> dict:
    used = ["RUN_META.json", "prep/PREPARE.json", "compare/compare.json"]
    row = rows.get(arm) or {}
    scenario = prepare.get("scenario") or {}
    record = {
        "schema": SCHEMA,
        "arm": arm,
        "input": {"scenario": scenario.get("scenario"), "sources": scenario.get("sources"),
                  "plate_types": scenario.get("plate_types"), "batch_key": scenario.get("batch_key"),
                  "label_key": scenario.get("label_key"), "n_wells": prepare.get("n_wells"),
                  "n_features": prepare.get("n_features"), "n_plates": prepare.get("n_plates"),
                  "n_compounds": prepare.get("n_compounds"), "n_wells_by_split": prepare.get("n_wells_by_split"),
                  "split_rule": prepare.get("split_rule"),
                  "evaluation": "the test compounds score the arm and are never shown to it"},
        "provenance": {**run_meta, "packages": ((prepare.get("preprocessing") or {}).get("fingerprint") or {})
                       .get("packages"), "tools": control.get("provenance")},
        "score": {"test": {"overall": row.get("overall"), "batch_mean": row.get("batch_correction"),
                           "bio_mean": row.get("biometrics"), "nonrep_mean_map": row.get("nonrep_mean_map")}},
        "cost": {"seconds": row.get("runtime_seconds"), "tool_calls": row.get("tool_calls"),
                 "tokens": row.get("tokens")},
    }
    if arm in CONTROL_ARMS:
        used.append("control/CONTROL.json")
        source = (control.get("arms") or {}).get(arm) or {}
        record["score"].update({view: _readouts(source.get(view)) for view in ("val", "all")})
        record["config"] = {"tool": arm, "configuration": source.get("config")}
        record["steps"] = [] if arm == "raw" else [{"kind": "tool_call", "tool_call": {"config": source.get("config")},
                                                     "tool_result": {"tool_status": source.get("tool_status"),
                                                                     "seconds": source.get("tool_seconds"),
                                                                     "telemetry": source.get("telemetry")}}]
        record["final_output"] = {"representation": "the uncorrected input" if arm == "raw" else f"{arm}'s output",
                                  "tool_status": source.get("tool_status")}
        record["status"] = {"delivered": bool(source)}
    elif arm == "gpt_luna":
        used.append("gpt_luna/ARM_OUTPUT.json")
        output = _load(out / arm / "ARM_OUTPUT.json") or {}
        plan = output.get("plan") or {}
        record["config"] = {"agent": "GPT-only: one generic linear operation, no domain tool", "model": output.get("model")}
        record["steps"] = [{"kind": "reply", "prompt": output.get("prompt"), "observation": output.get("matrix_summary"),
                            "rationale": plan.get("reasoning"), "action": plan.get("action"),
                            "strength": plan.get("strength"), "usage": output.get("tokens")}]
        record["final_output"] = plan
        record["status"] = {"delivered": bool(output)}
    elif arm == "react_luna_harmony":
        used += ["react_luna_harmony/ARM_OUTPUT.json", "react_luna_harmony/logs/tool_calls.jsonl"]
        output = _load(out / arm / "ARM_OUTPUT.json") or {}
        seconds = {e["call"]: e.get("seconds") for e in _jsonl(out / arm / "logs" / "tool_calls.jsonl")
                   if e.get("event") == "end"}
        history = output.get("history") or []
        record["config"] = {"agent": "ReAct: the shared ReAct runner (an LLM plus Batch_Corrector_Tool)"
                            if output.get("tool") else "ReAct: plain sequential harmonypy, native parameters",
                            "model": output.get("model"), "max_steps": output.get("max_steps"),
                            "noise_band": output.get("noise_band"), "round_cap": output.get("round_cap")}
        record["steps"] = [{"kind": "trial0" if h.get("call") == "default" else "tool_call", "step": h.get("step"),
                            "call": h.get("call"), "rationale": h.get("reasoning"), "tool_call": {"config": h.get("config")},
                            "tool_result": {"calibration": h.get("calibration"), "seconds": seconds.get(h.get("call")),
                                            "error": h.get("error")}}
                           for h in history]
        chosen = next((h for h in history if h.get("call") == output.get("selected_call")), {})
        record["final_output"] = {"call": output.get("selected_call"), "parameters": chosen.get("config")}
        record["status"] = {"delivered": bool(output), "stop_reason": output.get("stop_reason"),
                            "source": output.get("source"), "steps": output.get("steps"),
                            "reasoner_errors": output.get("reasoner_errors")}
    else:
        used += [f"{SHAPE}/RESULT.json", f"{SHAPE}/AGENT_RUN.json", f"{SHAPE}/RUN_META.json",
                 f"{SHAPE}/logs/tool_calls.jsonl", f"{SHAPE}/traces/round_00.jsonl", f"{SHAPE}/submission/decision.json"]
        result = _load(out / SHAPE / "RESULT.json") or {}
        agent_run = _load(out / SHAPE / "AGENT_RUN.json") or {}
        evidence = ((result.get("submission") or {}).get("evidence")) or {}
        record["provenance"]["arm_run_meta"] = _load(out / SHAPE / "RUN_META.json")
        record["score"].update({view: _readouts(((result.get("scores") or {}).get("agent") or {}).get(view))
                                for view in ("val", "all")})
        record["config"] = {"agent": "SHAPE: BatchCorrectionVerifier through StepVerifier wave mode",
                            "tool": "shape.tools.batch_corrector.Batch_Corrector_Tool", "model": agent_run.get("model"),
                            "max_rounds": agent_run.get("max_rounds"),
                            "prompts": f"{SHAPE}/traces/round_00.jsonl"}
        record["steps"] = _shape_steps(out)
        record["final_output"] = {"decision": result.get("decision"), "call": result.get("from_call"),
                                  "parameters": result.get("chosen_config"),
                                  "non_default_parameters": result.get("non_default_parameters")}
        record["status"] = {"delivered": bool(result), "stopped_because": agent_run.get("stopped_because"),
                            "rounds": agent_run.get("n_rounds"),
                            **{key: evidence.get(key) for key in ("stop_reason", "reason", "reasoner_failures",
                                                                  "fallback_rounds", "stop_checks", "focus_history")}}
    record["sources"] = {"used": used, "missing": [name for name in used if not (out / name).is_file()]}
    return record


def write_audit(out: Path, run_dir: str) -> None:
    run_meta = _load(out / "RUN_META.json") or {}
    prepare = _load(out / "prep" / "PREPARE.json") or {}
    control = _load(out / "control" / "CONTROL.json") or {}
    rows = _rows(out)
    (out / "audit").mkdir(parents=True, exist_ok=True)
    index = []
    for arm in ARMS:
        record = audit_record(out, arm, run_meta, prepare, control, rows)
        (out / "audit" / f"{arm}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        index.append({"arm": arm, "path": f"audit/{arm}.json", "overall": record["score"]["test"]["overall"],
                      "delivered": record["status"]["delivered"], "steps": len(record["steps"]),
                      "missing_sources": len(record["sources"]["missing"])})
    (out / "audit" / "index.json").write_text(json.dumps({"schema": SCHEMA, "run_dir": run_dir, "arms": list(ARMS),
                                                          "n_records": len(index), "records": index}, indent=1),
                                              encoding="utf-8")


def export(run_dir: Path, out: Path, log: Path | None = None) -> dict:
    run_dir, out = Path(run_dir), Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; export into a fresh directory")
    earlier = _load(run_dir / "MANIFEST.json") or {}
    source = earlier.get("run_dir") or str(run_dir.resolve())     # a re-exported record set keeps its run's path
    copied = []
    for rel in record_files(run_dir):
        target = out / _target(rel.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / rel, target)
        copied.append(_target(rel.as_posix()))
    if log is not None:
        (out / "logs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, out / "logs" / Path(log).name)
        copied.append(f"logs/{Path(log).name}")
    (out / "compare").mkdir(parents=True, exist_ok=True)
    write_compare(out)
    write_audit(out, source)
    copied += ["compare/paired_vs_shape.csv", "compare/summary.md", "audit/index.json"]
    copied += [f"audit/{arm}.json" for arm in ARMS]
    manifest = {
        "run_dir": source,
        "files": {name: {"bytes": (out / name).stat().st_size, "sha256": _sha256(out / name)}
                  for name in sorted(copied)},
        "missing": [name for name in EXPECTED if not (out / name).is_file()],
        "skipped_suffixes": list(SKIP),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--log", type=Path, help="the run's log (e.g. the Slurm .out or the interactive tee log)")
    args = parser.parse_args()
    manifest = export(args.run_dir, args.out, args.log)
    print(json.dumps({"out": str(args.out), "files": len(manifest["files"]), "missing": manifest["missing"]}))


if __name__ == "__main__":
    main()
