"""The SHAPE arm: RepresentationAgenticVerifier through StepVerifier's wave mode on Cell_State_Analyzer_Tool, with the
gpt-5.6-luna reasoner (SHAPE's engine factory, the Stanford AI Gateway).

Information: the tool sees the run prep's crops (train + val wells) with their plate, plate-qualified well, site,
experiment and split, and group "default" -- never a label. Trial 0 is the analyzer at adaptation_depth="zero_shot"
with the task's split in ssl_calls/input (the verifier's `frozen_params`; the control's frozen call), not a round; a
tool exception there is a failed trial 0 (the verifier's `initial_trial_failed`), not a crash. Every later call is
the agent's (a parent and CARD levers), run through SHAPE's managed-tool invocation (`stage_map.invoke_managed_tool`,
as solve() runs a tool), one call per round (concurrency 1). The verifier measures each call on the val wells and
delivers the agent's final choice among the calls its guard admits, else frozen. The deadline is the Slurm job's
(`squeue %L`); on the VM (contract amendment A8, `resume.deadline`) RXRX_DEADLINE_UTC or none. The arm refuses to
start without a reasoner engine (`no_engine=True` runs the deterministic fallback: a smoke only); with an engine
SHAPE_REQUIRE_AGENTIC=1 makes a reasoner failure an error. On the VM the arm also writes its replay records
(logs/reasoner.jsonl: every request and its complete reply; each call's CALL_RECORD.json) and `resume=True` resumes an
unfinished run in place by replay (resume.py).

Expert start (contract amendments A18, A19, A21, A24, user decisions; SHAPE only): `EXPERT_CONFIG` (the verifier's
`expert_config`, off by default in the verifier), two rounds of progressive unfreezing (A24, the full benchmarks'
SSL start): round 1 sets view_pairing same_well, contrastive_mask same_well, batch_sampling per_plate,
intensity_norm per_plate, augmentation light and trainable_pct 33 (the last third of the backbone); round 2 resumes a
round-1 call (init_from parent) and deepens to trainable_pct 66, no other lever change. The round prompts state it in
one block; a move that does not follow it is rejected through the repair question. From round 3 the loop is the
agent's (A20: no forced exploration at a plateau; A21: no untried-lever list). The calls are the agent's own
(agent_1, agent_2). RUN_META.json records `expert_config`.

Run seed (A18): `seed` (run_rxrx1.py --seed, the launcher's RXRX_SEED, default 0) is the analyzer's `seed` kwarg of
every call of the arm (trial 0 included), recorded in RUN_META.json as `seed`; repeats of the arm differ by it.

A run directory holds:
  RUN_META.json, AGENT_RUN.json   provenance; timing, stop reason, reasoner tokens, the agentic record
  logs/tool_calls.jsonl           one start / end record per call: parent, the levers it set, the expected effect,
                                  the diagnosis, the kwargs the tool received, the verifier's measurements, the
                                  tool's facts (control.TOOL_FACTS); logs/input_trial.json = trial 0
  logs/agent_log.jsonl            the verifier's log (every prompt, answer, rejection and stop) and the loop's events
  logs/reasoner.jsonl             VM only: every reasoner request (schema, prompt sha256) and its reply as returned
  cache/                          the tool's outputs (cell_state_analysis/ssl_calls/<call>/)
  submission/decision.json        use (a call or "frozen"), its checkpoint (resume_best), the case; provisional
                                  after trial 0 and after every round, overwritten by the delivery when the
                                  verifier's select returns
Every write of this module retries a transient filesystem failure (`control.fs_retry`; the analyzer's own cache
writes are not retried); a log or ledger write that still fails is
recorded in AGENT_RUN.json (write_errors) and the search goes on.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# A21/A24: the expert start has one entry per round; same_well pairing goes with per_plate batches (A18 diagnostic:
# without them the run-prep target fell below frozen), then the second round resumes with deeper unfreezing.
EXPERT_CONFIG = [{"view_pairing": "same_well", "contrastive_mask": "same_well", "batch_sampling": "per_plate",
                  "intensity_norm": "per_plate", "augmentation": "light", "trainable_pct": 33},
                 {"init_from": "parent", "trainable_pct": 66}]   # A24: two rounds of progressive unfreezing


def task_brief(summary: dict, cell_type: str, max_rounds: int) -> str:
    """The agent's task as facts about its input (no label, no result)."""
    t = summary["tool_input"]
    return (f"RxRx1 {cell_type}: {t['cells']} single-cell crops (6 fluorescence channels, 64 x 64 px, centred on a "
            f"nucleus) from {t['sites']} sites of {t['wells']} wells on {t['plates']} plates of {t['experiments']} "
            f"experiments; every crop carries its plate, plate-qualified well and site. Training uses the train wells "
            f"({t['train_cells']} cells); the val wells ({t['val_cells']} cells, at least "
            f"{t['min_val_wells_per_plate']} per plate, chosen without labels) never enter training and measure every "
            f"call. At most {max_rounds} rounds of tool calls.")


def visible_gpus() -> int:
    """The GPUs this process sees, at least 1 (one call at a time without one)."""
    import torch

    return max(1, torch.cuda.device_count())


def _write_failed(errors: list, what: str, error: OSError) -> None:
    errors.append({"time": time.time(), "what": what, "error": f"{type(error).__name__}: {error}"})
    print(f"WRITE_FAILED {what}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)


class AgentLog:
    """logs/agent_log.jsonl: the verifier's log and the loop's events, one JSON line each (shortened on stdout)."""

    def __init__(self, path: Path):
        self.path, self.lock, self.write_errors = path, threading.Lock(), []

    def __call__(self, kind: str, **fields) -> None:
        from benchmarks.representation_RxRx1.runners.control import append_jsonl

        record = {"time": time.time(), "kind": kind, **fields}
        try:
            with self.lock:
                append_jsonl(self.path, record)
        except OSError as error:
            _write_failed(self.write_errors, self.path.name, error)
        print(json.dumps(record, default=str)[:2000], flush=True)


class Ledger:
    """StepVerifier observer: the tool-call ledger, trial 0's record, the provisional decision after trial 0 and after
    every round."""

    def __init__(self, run_dir: Path, verifier, log, checkpoint, available: set[str], calls=None):
        self.run_dir, self.verifier, self.log, self.checkpoint = run_dir, verifier, log, checkpoint
        self.calls = calls                          # resume.CallCache on the VM: a served call keeps its wall seconds
        self.available = available                  # the tool's execute() kwargs
        self.path = run_dir / "logs" / "tool_calls.jsonl"
        self.trials: dict[int, tuple] = {}          # index -> (params, vres, result)
        self.parents: dict[int, int | None] = {}
        self.entries: list = [None]
        self.facts: dict[str, dict] = {}            # "input" / call -> the tool's facts of a finished trial
        self.write_errors: list = []

    def _append(self, record: dict) -> None:
        from benchmarks.representation_RxRx1.runners.control import append_jsonl

        append_jsonl(self.path, {"time": time.time(), **record})

    def __call__(self, event: dict) -> None:
        try:
            self._observe(event)
        except OSError as error:                    # after fs_retry: the search goes on, the failure is recorded
            _write_failed(self.write_errors, f"{event.get('event')} {event.get('index')}", error)

    def _observe(self, event: dict) -> None:
        from shape.agent.verification.stage_map import map_tool_params

        from benchmarks.representation_RxRx1.runners.control import TOOL_FACTS, TOOL_NAME, write_json

        kind, index = event["event"], event.get("index")
        if kind == "trial_started":
            params = event["params"]
            kwargs = map_tool_params(TOOL_NAME, params, self.available)
            call, parent = self.verifier.label(index, params), self.verifier.label(event["parent"])
            self._append({"event": "start", "call": call, "parent": parent,
                          "init": self.verifier.start_state(parent, params),
                          "set": event["param_delta"], "expected": event["expected_effect"],
                          "diagnosis": event["notes"], "kwargs": kwargs,
                          "not_forwarded": {k: v for k, v in params.items() if k not in kwargs}})
        elif kind == "trial_finished":
            params, vres, result = event["params"], event["vres"], event["result"]
            self.trials[index], self.parents[index] = (params, vres, result), event["parent"]
            call = self.verifier.label(index, params)
            record = {"score": vres.scalar_score, "metrics": vres.metrics, "action": vres.action}
            if event["failed"]:
                record.update(status="crashed", error=result.get("error"))
            else:
                self.facts[call] = {k: result.get(k) for k in TOOL_FACTS}
                record.update(status="done", tool=self.facts[call])
            if index == 0:
                v = self.verifier
                write_json(self.run_dir / "logs" / "input_trial.json",
                           {"call": call, "params": params, **record, "admission": v.admission, "cores": v.cores,
                            "concurrency_cap": v.cap})
                self.checkpoint(self)               # frozen, provisional: a deliverable before round 1 ends
                return
            if not event["failed"]:
                record.update(score_delta_vs_parent=vres.scalar_score - self.trials[event["parent"]][1].scalar_score,
                              score_delta_vs_input=vres.scalar_score - self.trials[0][1].scalar_score)
            served = self.calls.served.get(params.get("output_subdir")) if self.calls is not None else None
            seconds = self.calls.seconds_of(served) if served else event["seconds"]
            if self.calls is not None and not served:    # a served call reports these seconds again (A8)
                self.calls.note_outer(params, seconds)
            self._append({"event": "end", "call": call, **record, "seconds": seconds})
        elif kind == "wave_finished":
            self.entries.extend(event["entries"])
            self.checkpoint(self)
        else:                                       # wave, candidate_dropped, stopped
            self.log(kind, text=json.dumps({k: v for k, v in event.items() if k != "event"}, default=str))


def submission(ledger: Ledger, call: str | None, resolution=None) -> dict:
    """decision.json. `use`: the delivered call, else "frozen" (trial 0, the pretrained backbone); `checkpoint`: the
    call's state at its best val point (resume_best), with which score.py embeds every FULL cell. Before the search
    ends (resolution None) the call is the admissible call with the best score (provisional). `case`: the verifier's
    decision record's, else its choice, else the resolution's reason."""
    facts = ledger.facts.get(call or "input", {})
    out = {"use": call or "frozen", "provisional": resolution is None,
           "checkpoint": facts.get("resume_best") if call else None, "adata_path": facts.get("adata_path")}
    if resolution is not None:
        details = resolution.details
        out.update(status=resolution.status, reason=resolution.reason,
                   case=(details.get("decision") or {}).get("case") or details.get("choice") or resolution.reason,
                   details=details)
    return out


# SHAPE-ES (early-stopping monitoring, default off): the search-level rule of the verifier and the within-call rule
# of the analyzer (val-proxy patience + the overfitting gap).
EARLY_STOP_SEARCH = {"plateau_rounds": 2}
EARLY_STOP_TOOL = {"stop_by": "val_proxy", "patience": 2, "overfit_gap": True, "gap_evals": 2, "train_wells": 264}


def run(root: Path, run_dir: Path, *, cell_type: str, model: str, max_rounds: int, engine=None,
        no_engine: bool = False, resume: bool = False, seed: int = 0, early_stop: bool = False) -> dict:
    """One SHAPE-arm run on the prepared cell type under `root`; `resume`: the unfinished run in `run_dir`, resumed
    by replay (resume.py); `seed`: the analyzer's seed of every call (A18)."""
    import inspect

    import shape
    from shape.agent.verification.base import available_cores
    from shape.agent.verification.stage_map import invoke_managed_tool
    from shape.agent.verification.step_verifier import StepVerifier
    from shape.agent.verification.verifiers.representation import RepresentationTask, frozen_params
    from shape.engine import usage
    from shape.utils.schemas import WaveState

    from benchmarks._shared.runner import agentic_record
    from benchmarks.representation_RxRx1.runners import control, resume as resume_mod
    from benchmarks.representation_RxRx1.runners.epoch_budget import FixedEpochVerifier, shared_budget
    from benchmarks.representation_RxRx1.runners.prepare import load, tool_input

    if engine is None and not no_engine:
        raise SystemExit(f"NOT_READY: no reasoner engine for {model}; the SHAPE arm does not start without one")
    if REPO not in Path(shape.__file__).resolve().parents:
        raise SystemExit(f"`import shape` resolves to {shape.__file__}, not inside {REPO}")
    state = resume_mod.Resume.open(run_dir, "shape") if resume else None
    if state is None and (run_dir / "RUN_META.json").exists():
        raise SystemExit(f"{run_dir} holds a run already; use a new run directory")
    recording = resume_mod.on_vm()                  # the replay records (A8), VM only: the cluster's run is unchanged
    summary, cells = load(root)
    epoch_budget = shared_budget(root, summary, cells)
    provenance = {**control.provenance(), "prep_cells_sha256": summary["cells_sha256"]}
    control_path = root / control.CONTROL_DIR / "CONTROL.json"
    if control_path.exists() and json.loads(control_path.read_text(encoding="utf-8"))["provenance"] != provenance:
        raise SystemExit(f"CONTROL_STALE: {control_path} was computed with other code or another prep")
    if engine is not None:
        os.environ["SHAPE_REQUIRE_AGENTIC"] = "1"   # a reasoner failure is an error, never the deterministic fallback
    for sub in ("logs", "submission", "cache"):
        control.fs_retry((run_dir / sub).mkdir, parents=True, exist_ok=True)
    crop_array, meta = tool_input(summary, cells)
    base_kwargs = {"crop_array": crop_array, "cell_metadata": meta, "query_cache_dir": str(run_dir / "cache")}
    if early_stop:
        base_kwargs["early_stop"] = dict(EARLY_STOP_TOOL)
    deadline = resume_mod.deadline()
    task = RepresentationTask(cores=available_cores(), gpus=visible_gpus(), deadline=deadline,
                              brief=task_brief(summary, cell_type, max_rounds))
    started = time.time()
    run_meta = {
        "commit": control.git_commit(), "expected_commit": os.environ.get("RXRX_EXPECTED_COMMIT"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(), "cell_type": cell_type,
        "agent": "shape", "model": model if engine is not None else "none", "max_rounds": max_rounds,
        "training_budget": epoch_budget, "fixed_levers": {"max_epochs": epoch_budget["max_epochs"]},
        "cores": task.cores, "gpus": task.gpus, "split_key": task.split_key,
        "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline)) if deadline else None,
        "shape_file": str(Path(shape.__file__).resolve()), "crop_array": crop_array, "provenance": provenance,
        "card_sha256": control.card_sha256(), "require_agentic": os.environ.get("SHAPE_REQUIRE_AGENTIC"),
        "expert_config": [dict(r) for r in EXPERT_CONFIG], "seed": int(seed),
        "early_stop": ({"search": dict(EARLY_STOP_SEARCH), "tool": dict(EARLY_STOP_TOOL)} if early_stop else None),
        "reasoning_effort": os.environ.get("SHAPE_REASONING_EFFORT") if engine is not None else None,
        "max_completion_tokens": getattr(engine, "max_tokens", None),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), **resume_mod.platform_keys()}
    if state is None:
        control.write_json(run_dir / "RUN_META.json", run_meta)
    else:
        state.write_meta(run_meta)
    log = AgentLog(run_dir / "logs" / "agent_log.jsonl")
    tokens_before = usage.snapshot()
    log("system", text=f"SHAPE arm (RepresentationAgenticVerifier, StepVerifier wave mode), model={model}, "
                       f"engine={'ready' if engine is not None else 'none'}, max_rounds={max_rounds}, "
                       f"cores={task.cores}, gpus={task.gpus}, crops={len(meta)}")

    write_errors: list = []                         # the replay records' write failures (VM)
    calls = resume_mod.CallCache(run_dir / "cache", state, write_errors) if recording else None

    def tool_runner(params):
        params = {**params, "max_epochs": epoch_budget["max_epochs"]}

        def invoke():
            return invoke_managed_tool(control.analyzer(), control.TOOL_NAME,
                                       {**base_kwargs, "cell_metadata": [dict(m) for m in meta], "seed": int(seed)},
                                       params)

        return invoke() if calls is None else calls.run(params, invoke)

    verifier = FixedEpochVerifier(epoch_budget, log=log, expert_config=EXPERT_CONFIG,
                                  early_stop=dict(EARLY_STOP_SEARCH) if early_stop else None)
    if early_stop and verifier.early_stop != EARLY_STOP_SEARCH:
        raise RuntimeError(f"SHAPE-ES: the verifier did not take early_stop ({verifier.early_stop!r})")
    if engine is not None and recording:
        recorder = resume_mod.RecordingEngine(engine, run_dir / "logs" / "reasoner.jsonl", write_errors)
        engine = recorder
        if state is not None:
            replies = resume_mod.shape_replies(state.source / "reasoner.jsonl")
            recorder.count = len(replies)           # a live request continues the logged numbering
            engine = resume_mod.ReplayEngine(replies, recorder, state,
                                             echo=lambda r, content: recorder.append({
                                                 **{k: v for k, v in r.items() if k != "time"},
                                                 "prompt_sha256": resume_mod._prompt_digest(content),
                                                 "prompt_masked_sha256": resume_mod.masked_digest(content),
                                                 "replayed": state.number}),
                                             prompt_check=resume_mod.shape_prompt_check)
    if state is not None:                           # the prompts' wall time counts the run's active time before
        begin, offset = verifier.begin, state.entry["active_seconds_before"]

        def shifted_begin(task=None):
            begin(task)
            verifier.started -= offset

        verifier.begin = shifted_begin
    verifier.llm_engine = engine
    decision_path = run_dir / "submission" / "decision.json"

    def checkpoint(ledger: Ledger) -> None:         # decision.json stays valid while the search runs
        indices = sorted(ledger.trials)
        state = WaveState(trials=[ledger.trials[i] for i in indices], parents=[ledger.parents[i] for i in indices],
                          entries=ledger.entries[:len(indices)], remaining=0, context={})
        inp, attempts = verifier._bind(state)
        control.write_json(decision_path, submission(ledger, verifier.fallback_choice(inp, attempts)))

    ledger = Ledger(run_dir, verifier, log, checkpoint, set(inspect.signature(control.analyzer().execute).parameters),
                    calls)
    initial_params = frozen_params(task)
    initial_params["max_epochs"] = epoch_budget["max_epochs"]
    try:
        initial_result = tool_runner(initial_params)
    except Exception as error:  # noqa: BLE001 -- a failed trial 0 is the verifier's initial_trial_failed, not a crash
        initial_result = {"error": f"{type(error).__name__}: {error}"}
    resolution = StepVerifier(max_iters=1 + max_rounds, observer=ledger).resolve_stage(
        verifier, verifier.make_adapter(base_kwargs), tool_runner, initial_params, initial_result,
        upstream_ctx={"representation_task": task})

    selected = resolution.details.get("selected")
    final = submission(ledger, verifier.label(selected) if selected else None, resolution)
    control.write_json(decision_path, final)
    tokens = {k: v - tokens_before.get(k, 0) for k, v in usage.snapshot().items() if isinstance(v, int)}
    if state is not None:
        state.close()
    ended = time.time()
    details = resolution.details
    control.write_json(run_dir / "AGENT_RUN.json", {
        "agent": "shape", "model": model, "started": started, "ended": ended,
        "wall_hours_used": resume_mod.active_hours(state.meta if state else None, started, ended),
        "n_rounds": len(verifier.rounds),
        "stopped_because": verifier.stop_reason or details.get("stop_reason"), "max_rounds": max_rounds,
        "tokens": tokens, "write_errors": ledger.write_errors + log.write_errors + write_errors,
        "agentic": agentic_record([{"stages": [{"stage": "representation", "details": details}]}], "representation"),
        **({"resumes": resume_mod.resumes_record(state.meta)} if state else {})})
    log("final", text=json.dumps({k: final.get(k) for k in ("use", "case", "status", "reason", "checkpoint")},
                                 default=str))
    return final
