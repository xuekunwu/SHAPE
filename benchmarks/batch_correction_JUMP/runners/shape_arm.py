"""The SHAPE arm of the JUMP batch-correction benchmark: SHAPE's budgeted VRR (BatchCorrectionVerifier
through StepVerifier's wave mode) on SHAPE's Batch_Corrector_Tool.

Information: the tool sees every well's features and the batch column; the verifier scores only the calibration
view -- the calibration (val) compounds and every control well -- and never sees the evaluation (test)
compounds' labels; the stated noise (0.002) and the budget (at most 8 rounds) are the task's. Trial 0 is
the uncorrected input (not a tool call); every tool call is chosen by the reasoner, a round being the calls
it runs at once: at most cores // threads, each with threads = call_threads(the input) BLAS threads.

A run directory holds:
  RUN_META.json, AGENT_RUN.json   provenance; timing, stop reason, reasoner tokens
  logs/tool_calls.jsonl           one start / end record per call (parent, diagnosis, expected effect,
                                  configuration, calibration readouts, GT-free readouts, deltas vs the
                                  parent, reconciliation, tool report); logs/input_trial.json = trial 0
  traces/round_00.jsonl           the arm's events ({"type": "shape", "kind": ...})
  trials/<config sha256>.npy      each call's corrected matrix (float32, prep row order)
  submission/decision.json        the decision (RUN / SKIP / ABSTAIN), from_call, evidence, claims;
                                  provisional after every round
  NOTES.md                        admission, resolution and the calls in words
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MAP_COLUMNS = ("Metadata_Plate", "Metadata_Well", "Metadata_PlateType", "Metadata_PertType")


def config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def reconcile(expected: dict, delta: dict, noise) -> dict:
    """Per expected metric: confirmed (the observed delta has the predicted sign beyond its noise;
    `preserve`: within it), contradicted (the opposite), inconclusive (within the noise)."""
    out = {}
    for metric, direction in expected.items():
        d, band = delta.get(metric), noise(metric)
        if d is None:
            out[metric] = {"expected": direction, "observed_delta": None, "label": "not_computable"}
            continue
        if direction in ("preserve", "same"):
            label_ = "confirmed" if abs(d) <= band else "contradicted"
        elif abs(d) <= band:
            label_ = "inconclusive"
        else:
            label_ = "confirmed" if (d > 0) == (direction == "up") else "contradicted"
        out[metric] = {"expected": direction, "observed_delta": d, "label": label_}
    return out


def calibration_view(meta, *, batch_key: str, plate_types):
    """The calibration (val) compounds and every control well, with the columns the metric needs."""
    from shape.agent.verification.verifiers.batch_correction_unified import CalibrationView
    from shape.tools.batch_corrector import quant

    mask = quant.split_mask(meta, "val")
    columns = list(dict.fromkeys([quant.LABEL_KEY, batch_key, *MAP_COLUMNS]))
    return CalibrationView(rows=np.flatnonzero(mask), meta=quant.subset(meta[columns], mask), batch_key=batch_key,
                           plate_types=tuple(plate_types))


def task_brief(scenario: dict, batch_column: str, max_rounds: int) -> str:
    from benchmarks.batch_correction_JUMP.runners.config import SCENARIO_DESIGN

    design = SCENARIO_DESIGN.get(scenario["name"], scenario["name"])
    return (f"JUMP Cell Painting (cpg0016), {scenario['name']}: {design}. Batch column: {batch_column}. "
            f"At most {max_rounds} rounds of tool calls. Calibration readouts come from the calibration compounds and all control "
            "wells; the evaluation compounds are held out.")


class Events:
    """The arm's event stream: traces/round_00.jsonl (+ arrival times) and stdout."""

    def __init__(self, run: Path):
        (run / "traces").mkdir(parents=True, exist_ok=True)
        self.path, self.times = run / "traces" / "round_00.jsonl", run / "traces" / "round_00.times"

    def __call__(self, kind: str, **fields) -> None:
        now = time.time()
        line = json.dumps({"type": "shape", "kind": kind, "time": now, **fields}, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        with self.times.open("a", encoding="utf-8") as handle:
            handle.write(f"{now}\n")
        print(line[:2000], flush=True)


def _numeric(values: dict) -> dict:
    return {k: v for k, v in (values or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


class Ledger:
    """StepVerifier observer: the tool-call ledger, the provisional decision after every round."""

    def __init__(self, run: Path, verifier, emit, checkpoint):
        self.run, self.verifier, self.emit, self.checkpoint = run, verifier, emit, checkpoint
        self.path = run / "logs" / "tool_calls.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trials: dict[int, tuple] = {}          # index -> (params, vres, result)
        self.parents: dict[int, int | None] = {}
        self.entries: list = [None]

    def _append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), **record}, default=str) + "\n")

    @staticmethod
    def calibration(vres, gt_free_keys) -> dict:
        return {k: v for k, v in (vres.metrics or {}).items() if k not in gt_free_keys}

    def noise(self, metric: str) -> float:
        """The verifier's noise of a calibration readout; the stated noise for a GT-free readout."""
        value = self.verifier._noise(metric)
        return self.verifier.stated_noise if value is None else value

    def __call__(self, event: dict) -> None:
        from shape.agent.verification.verifiers.batch_correction_unified import _READOUTS, tool_summary

        kind, index = event["event"], event.get("index")
        if kind == "trial_started":
            flat = event["params"]["harmony_params"]
            call, parent = self.verifier.label(index, event["params"]), self.verifier.label(event["parent"])
            self._append({"event": "start", "call": call, "parent": parent,
                          "diagnosis": event["notes"], "expected": event["expected_effect"], "config": flat,
                          "config_sha256": config_sha256(flat), "set": event["param_delta"]})
            self.emit("tool_call", name="Batch_Corrector_Tool",
                      input={"parent": parent, "set": event["param_delta"],
                             "expected": event["expected_effect"], "diagnosis": event["notes"]})
        elif kind == "trial_finished":
            params, vres, result = event["params"], event["vres"], event["result"]
            call = self.verifier.label(index, params)
            self.trials[index], self.parents[index] = (params, vres, result), event["parent"]
            val = self.calibration(vres, _READOUTS)
            diagnostics = dict(result.get("gt_free_metrics") or {}) if not event["failed"] else {}
            report = ({k: result.get(k) for k in ("batch_effect", "harmony", "telemetry", "preprocess")}
                      if not event["failed"] else {})
            if index == 0:
                record = {"call": "input", "status": result.get("status"), "val": val, "diagnostics": diagnostics,
                          "gate": result.get("batch_effect"), "admission": self.verifier.admission,
                          "noise": self.verifier._noise_table(), "cores": self.verifier.cores,
                          "concurrency_cap": self.verifier.cap}
                (self.run / "logs" / "input_trial.json").write_text(json.dumps(record, indent=1, default=str),
                                                                    encoding="utf-8")
                self.emit("message", label="input", text=json.dumps(
                    {"calibration": val, "gate": result.get("batch_effect"), "noise": record["noise"]}, default=str))
                return
            if event["failed"]:
                self._append({"event": "end", "call": call, "status": "crashed",
                              "error": result.get("error"), "seconds": event["seconds"]})
                self.emit("tool_result", name="Batch_Corrector_Tool", error=True, text=str(result.get("error")))
                return
            parent_vres = self.trials[event["parent"]][1]
            delta = {k: v - parent_vres.metrics[k] for k, v in _numeric(vres.metrics).items()
                     if isinstance(parent_vres.metrics.get(k), (int, float))}
            self._append({"event": "end", "call": call, "status": "done", "tool_status": result.get("status"),
                          "val": val, "diagnostics": diagnostics, "delta_vs_parent": delta,
                          "reconciliation": reconcile(vres.expected_effect, delta, self.noise), "tool_report": report,
                          "seconds": event["seconds"], "corrected": result.get("_path")})
            self.emit("tool_result", name="Batch_Corrector_Tool", error=False,
                      text=json.dumps({"call": call, "tool_status": result.get("status"),
                                       "overall": val.get("overall"), "nonrep_mean_map": val.get("nonrep_mean_map"),
                                       "tool": tool_summary(result)}, default=str))
        elif kind == "wave_finished":
            for entry in event["entries"]:
                self.entries.append(entry)
            self.checkpoint(self)
        elif kind in ("wave", "candidate_dropped", "stopped"):
            self.emit("message", label=kind, text=json.dumps({k: v for k, v in event.items() if k != "event"}, default=str))


def claims(verifier, evidence: dict, decision: str, call: str | None, vres, inp) -> list[dict]:
    """Claims the evidence supports; batch_reduced / biology_preserved compare the submitted call's
    calibration readouts with the uncorrected input's (trial 0), as the evaluator checks them."""
    out = []
    gate = evidence.get("gate_all_rows") or {}
    if gate.get("detected"):
        out.append({"type": "batch_effect_present", "statement": f"gate: {gate}", "support": "tool detector on all rows"})
    if decision == "RUN":
        out.append({"type": "separable_from_biology",
                    "statement": f"{evidence.get('reference_groups_in_all_batch_groups')} of {evidence.get('reference_groups')} "
                                 "calibration compounds occur in every batch group",
                    "support": "design (natural metadata)"})
    if decision == "RUN" and call is not None and vres is not None and inp is not None:
        for claim, key, holds in (("batch_reduced", "batch_mean", lambda new, raw, n: new > raw + n),
                                  ("biology_preserved", "bio_mean", lambda new, raw, n: new >= raw - n)):
            raw, new = inp.metrics.get(key), vres.metrics.get(key)
            if raw is not None and new is not None and holds(new, raw, verifier._noise(key)):
                out.append({"type": claim, "statement": f"calibration {key} {raw:.4f} (uncorrected) -> {new:.4f} ({call})",
                            "support": "calibration view"})
    return out


def submission(verifier, call: str | None, vres, inp, resolution=None) -> dict:
    """decision.json: RUN with the call to submit, else the admission's decision (ABSTAIN when admission
    ran but no call is submitted). Before the search ends (resolution None) the call is the highest-overall
    admissible call (provisional)."""
    admission = verifier.admission or {"decision": "ABSTAIN", "rule": "none"}
    evidence = verifier.evidence
    decision = "RUN" if call else ("ABSTAIN" if admission["decision"] == "RUN" else admission["decision"])
    reasoner, veto, rule = admission.get("reasoner") or {}, admission.get("veto"), admission.get("rule")
    if resolution is None:
        reason_code = "PROVISIONAL"
        reason = "the search is running; the admissible call with the highest calibration overall is submitted"
    elif admission["decision"] == "RUN" and call is None:
        reason_code, reason = f"NO_CORRECTION_{resolution.reason.upper()}", f"resolution: {resolution.reason}"
    elif veto and not veto["accepted"]:
        reason_code = "RUN_VETO_REJECTED"
        reason = (f"admission rule {rule}; the reasoner's ABSTAIN was rejected because none of its cited concerns "
                  f"{veto['cited']} holds (reasoner: {reasoner.get('reasoning')})")
    elif reasoner and reasoner.get("decision") != admission["decision"]:
        reason_code = f"{rule}_RULE"
        reason = (f"admission rule {rule}; the reasoner's {reasoner.get('decision')} does not change it "
                  f"(reasoner: {reasoner.get('reasoning')})")
    else:
        reason_code = reasoner.get("reason_code") or rule
        reason = reasoner.get("reasoning") or f"admission rule {rule}"
    out = {"decision": decision, "reason_code": reason_code,
           "information_used": [evidence.get("batch_column"), "negcon", "poscon", "calibration compounds",
                                "calibration readouts"],
           "evidence": {"admission": evidence, "admission_concerns": admission.get("concerns"), "veto": veto},
           "claims": claims(verifier, evidence, decision, call, vres, inp), "reason": reason}
    if resolution is not None:
        details = resolution.details
        out["evidence"].update({key: details.get(key) for key in (
            "stop_reason", "verifier_stop", "admissible", "choice", "final_answers", "hypotheses", "rounds",
            "noise", "failed", "reasoner_failures", "fallback_rounds", "stop_checks", "focus_history")})
        out["evidence"].update({"status": resolution.status, "reason": resolution.reason,
                                "lever_ledger": resolution.ledger})
    if decision == "RUN":
        out["from_call"] = call
    return out


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def slurm_deadline() -> float | None:
    """Epoch seconds at which this Slurm job's time limit ends (`squeue %L`, the time left), or None
    outside Slurm or without a limit."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        left = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return _deadline_from(left, time.time())


def _deadline_from(left: str, now: float) -> float | None:
    import re

    m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", left or "")
    if not m:
        return None                       # UNLIMITED, NOT_SET, INVALID or no answer
    days, hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())
    return now + ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(prep: Path, run_dir: Path, *, model: str, max_rounds: int, noise_band: float, cores: int | None = None,
        engine=None) -> dict:
    """One SHAPE-arm run on a prepared scenario (the control stage must have run)."""
    import shape
    from shape.agent.verification.step_verifier import StepVerifier
    from shape.agent.verification.verifiers.batch_correction_unified import (
        BatchCorrectionVerifier, BatchTask, adapt_batch_correction, available_cores, blas_limits, call_threads,
        concurrency_cap, uncorrected_result)
    from shape.engine import usage
    from shape.utils.schemas import WaveState

    from benchmarks.batch_correction_JUMP.runners.control import CONTROL_DIR, provenance, run_tool
    from benchmarks.batch_correction_JUMP.runners.prepare import load

    if REPO not in Path(shape.__file__).resolve().parents:
        raise SystemExit(f"`import shape` resolves to {shape.__file__}, not inside {REPO}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "trials", "submission"):
        (run_dir / sub).mkdir(exist_ok=True)
    summary, meta, feats = load(prep)
    scenario = summary["scenario"]
    batch_key, plate_types = scenario["batch_key"], scenario["plate_types"]
    control_path = prep / CONTROL_DIR / "CONTROL.json"
    control = json.loads(control_path.read_text(encoding="utf-8")) if control_path.exists() else {}
    if control and control.get("provenance") != provenance():
        raise SystemExit(f"CONTROL_STALE: {control_path} was computed with other tool or metric code")
    batch_column = "source" if batch_key == "Metadata_Source" else "batch"
    cores = cores or available_cores()
    deadline = slurm_deadline()
    started = time.time()
    _write_json(run_dir / "RUN_META.json", {
        "commit": _git_commit(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
        "scenario": scenario["name"], "max_rounds": max_rounds, "agent": "shape", "model": model,
        "threads": os.environ.get("SLURM_CPUS_PER_TASK"), "noise_band": noise_band, "cores": cores,
        "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline)) if deadline else None,
        "threads_per_call": call_threads(feats, cores), "concurrency_cap": concurrency_cap(cores, feats),
        "cli_version": f"run_batch_benchmark @ {REPO} {_git_commit()}", "shape_package": str(Path(shape.__file__).parent),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))})
    emit = Events(run_dir)
    tokens_before = usage.snapshot()
    emit("system", text=f"SHAPE arm (BatchCorrectionVerifier, StepVerifier wave mode), model={model}, "
                        f"engine={'ready' if engine else 'none'}, max_rounds={max_rounds}, stated noise={noise_band}, "
                        f"cores={cores}")

    batch = meta[batch_key].astype(str).to_numpy()
    view = calibration_view(meta, batch_key=batch_key, plate_types=plate_types)
    task = BatchTask(batch=batch, features=feats, calibration=view, noise_band=noise_band,
                     brief=task_brief(scenario, batch_column, max_rounds), batch_column=batch_column, cores=cores,
                     deadline=deadline)

    def tool_runner(params):
        flat = dict(params["harmony_params"])
        result = run_tool(feats, batch, flat, batch_key=batch_key)
        path = run_dir / "trials" / f"{config_sha256(flat)}.npy"
        np.save(path, np.asarray(result["corrected_cls_output"], dtype=np.float32))
        result["corrected_cls_output"], result["_path"] = np.load(path, mmap_mode="r"), str(path)
        return result

    verifier = BatchCorrectionVerifier(log=lambda kind, **f: emit(
        "thinking" if kind == "reasoning" else "message", label=f.get("label", kind), text=f.get("text", "")))
    verifier.llm_engine = engine
    decision_path = run_dir / "submission" / "decision.json"

    def checkpoint(ledger: Ledger) -> None:     # decision.json stays valid while the search runs
        indices = sorted(ledger.trials)
        state = WaveState(trials=[ledger.trials[i] for i in indices], parents=[ledger.parents[i] for i in indices],
                          entries=ledger.entries[:len(indices)], remaining=0, context={})
        if (verifier.admission or {}).get("decision") != "RUN":
            return
        inp, attempts = verifier._bind(state)
        call = verifier.fallback_choice(inp, attempts)
        _write_json(decision_path, submission(verifier, call, attempts[call][2] if call else None, inp))

    ledger = Ledger(run_dir, verifier, emit, checkpoint)
    with blas_limits(feats, cores):
        resolution = StepVerifier(max_iters=1 + max_rounds, observer=ledger).resolve_stage(
            verifier, adapt_batch_correction, tool_runner, {}, uncorrected_result(feats, batch),
            upstream_ctx={"batch_task": task})

    selected = resolution.details.get("selected")
    inp = resolution.trials[0][1]
    call = verifier.label(selected) if selected and verifier.admission.get("decision") == "RUN" else None
    final = submission(verifier, call, resolution.verdict if call else None, inp, resolution)
    _write_json(decision_path, final)
    tokens = {k: v - tokens_before.get(k, 0) for k, v in usage.snapshot().items()
              if isinstance(v, int) and not isinstance(v, bool)}
    ended = time.time()
    _write_json(run_dir / "AGENT_RUN.json", {
        "agent": "shape", "model": model, "started": started, "ended": ended,
        "wall_hours_used": round((ended - started) / 3600, 3), "n_rounds": len(verifier.rounds),
        "stopped_because": verifier.stop_reason or resolution.details.get("stop_reason"), "max_rounds": max_rounds,
        "tokens": tokens})
    details = resolution.details
    notes = [f"# NOTES (SHAPE arm, {model})", "",
             f"Admission: {verifier.admission.get('decision')} (rule {verifier.admission.get('rule')}); "
             f"submitted decision {final['decision']}.",
             f"Resolution: {resolution.status} ({resolution.reason}); stop {verifier.stop_reason or details.get('stop_reason')}; "
             f"submitted {call} ({details.get('choice')}); admissible {details.get('admissible')}; "
             f"rounds {details.get('rounds')}; failed calls {len(details.get('failed') or [])}.",
             f"Noise (stated / measured / used): {details.get('noise')}.",
             f"Reasoner tokens: {tokens}.", "", "Calls:"]
    for entry in details.get("entries") or []:
        if entry is not None:
            notes.append(f"- {verifier.label(entry['index'])} <- {verifier.label(entry['parent'])}: "
                         f"{entry['param_delta']} -> outcome "
                         f"{entry['outcome']}, hypothesis {entry['hypothesis']}, score delta vs parent "
                         f"{entry['score_delta_vs_parent']}" + (" (failed)" if entry["failed"] else ""))
    notes += ["", "Hypotheses:"] + [f"- {h.get('id')} [{h.get('status')}] {h.get('statement')} ({h.get('evidence')})"
                                    for h in details.get("hypotheses") or []]
    (run_dir / "NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    emit("message", label="final", text=json.dumps(final, default=str)[:4000])
    return final
