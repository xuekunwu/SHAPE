"""StepVerifier: the framework-level verifier component.

Division of labour (as specified for the SHAPE integration):
  - the **planner** recommends sensible *initial* parameters for a tool;
  - the **StepVerifier** (here) judges the tool's execution result and, when a
    LOCAL fault is found, adjusts parameters and **reruns the tool** until it
    converges or concedes.

This is the live counterpart of ``base.Verifier.run()``: instead of calling
``tool(**params)`` directly, it drives a ``tool_runner`` callable (supplied by
the Executor in the live pipeline, or a fake in tests) and uses an ``adapter`` to
turn each tool result into the verifier's input shape. The adaptive-iteration,
expected-vs-observed reconciliation and anti-oscillation logic are shared with
``base`` so behaviour stays identical to the reference loop.

VRR migration (task 032): the loop's terminal vocabulary collapses to two outcomes,
``satisfied`` (was ``converged``) and ``conceded`` (was ``escalate_upstream`` /
``escalate_human`` / ``hypothesis_invalidated`` / ``judge_no_improvement`` /
``oscillation_stop`` / ``max_iters_reached``), plus ``exclude_samples`` retained as
a DISTINCT in-loop data-op status (not a concede -- see docs/vrr.md). ``conceded``
carries a ``fault`` tag (task 045 refinement: ``ceiling`` | ``upstream`` -- the old
third value ``human`` is retired, folded into ``ceiling``) and, for ``upstream``, an
``upstream_target`` -- this is what `shape.solver` keys on instead of the old action
string (see solver.py's `_vlrr_loop_decision` and friends). The original,
finer-grained reason (e.g. "hypothesis_invalidated" or "escalate_ceiling") is
preserved in the new ``reason`` field for tracing/observability; it does not drive
control flow.

Reconcile-as-diagnostic (vrr.md sec 4): every reconcile pass -- INCLUDING on the
`accept` exit path (task 021's bug: a retune whose predicted effect never really
materialized was silently accepted) -- writes a per-lever `ledger` entry via
`base.classify_lever`: effective / no-headroom / inert.

The ledger is observational-only this task: nothing reads it to skip/prefer a
lever, so it does NOT by itself enforce "an inert lever is never proposed again"
across separate calls -- that enforcement (Recommend actively skipping levers the
ledger marked inert) is a follow-on, once the ledger is wired to Recommend. Within
a single `resolve_stage` call, though, a lever the ledger marks inert cannot be
re-proposed by THIS call: `classify_lever`'s "did it move" check now uses the SAME
epsilon as the stop-gate's `reconcile()` below (pre-merge fix, QC N1 -- see
`base.py`'s ledger section for the full reasoning), so whenever the ledger marks a
lever inert, the identical `reconcile()` check on the same data also fails to
confirm the metric moved, and (on the `retune_here` path) the loop concedes
(`hypothesis_invalidated`) that same iteration -- there is no next iteration in
which this call could re-propose it.

`StageResolution` lives in `shape.utils.schemas` (pre-merge fix, single schema
home) -- imported below.

Reconciliation (task 040): task 028/030 (origin/main) independently added a
plateau stop -- `patience`/`best_rank`/`stall`, tracking the independent judge's
per-attempt severity and stopping once it has not improved for `patience`
consecutive judged retunes, rather than on the first non-improving one. This
REPLACES the single-step `judge_rank(vres) >= judge_rank(prev_vres)` check task 032
had (both are the same weaker "judge severity plateaued" path; keeping both would
re-impose patience=1). The reconcile stop-gate (`hypothesis_invalidated`) is
patience-1: a failed expected-effect hypothesis concedes `fault="ceiling"` on the
spot (the v8g publication-snapshot semantics, restored deliberately). The remaining
adaptive stop paths are unchanged -- convergence, no legal new action, duplicate or
oscillating parameters, judge plateau, or an explicit safety budget.

Wave mode: a verifier that sets `uses_waves` (the batch corrector's budgeted VRR) proposes several
candidates per round, each derived from any earlier trial, and selects the final trial itself; the
loop runs at most `max_iters - 1` rounds after the initial trial (the linear loop's unit: one iteration =
one round; a round may hold several calls), stops early on an empty round, runs each wave concurrently
and keeps the
per-trial ledger against each trial's parent (`base.classify_trial`). `StageResolution.parents` /
`details` record the tree. The linear loop above is untouched for every other verifier.
"""

from __future__ import annotations

import copy
import math
import os
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, Callable, Optional

from shape.engine.errors import is_fatal_llm_error
from shape.utils.schemas import (EXECUTION_ERROR, ExecutionError, StageResolution, VLRRResult, WaveState,
                                 is_failed_result)

from .base import BaseVerifier, apply_bounded, classify_lever, classify_trial, judge_rank, reconcile

def execution_failure(result: Any, error: BaseException) -> dict:
    """A call whose verification hit an execution error (`schemas.ExecutionError`: an in-run vision call failed):
    a failed result with the status EXECUTION_ERROR and the cause, the call's own result kept under
    `call_result`."""
    return {"_attempt_status": "FAILED", "status": EXECUTION_ERROR, "error": f"execution error: {error}",
            "call_result": result}


def is_execution_error(result: Any) -> bool:
    return isinstance(result, Mapping) and result.get("status") == EXECUTION_ERROR


def _never_ran(result: Any) -> bool:
    """A call an execution error failed before it ran (`WaveCandidate.error`): it tried no configuration."""
    return isinstance(result, Mapping) and result.get("ran") is False


# adapter: (tool_result, upstream_context) -> (verifier_input, verifier_context)
Adapter = Callable[[Any, dict], tuple]
# tool_runner: (params) -> tool_result
ToolRunner = Callable[[dict], Any]


def _zero_detection_rescue_env_enabled() -> bool:
    return os.getenv("SHAPE_ZERO_DETECTION_RESCUE", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


_ZERO_DETECTION_RESCUE_LEVERS = {
    "cellprob_threshold",
    "augment",
    "niter",
    "tile_overlap",
    "diameter",
    "flow_threshold",
    "resample",
}


def _is_zero_detection_retry(prev_vres: VLRRResult, vres: VLRRResult) -> bool:
    prev_metrics = prev_vres.metrics or {}
    metrics = vres.metrics or {}
    prev_delta = prev_vres.param_delta or {}
    delta = vres.param_delta or {}
    return (
        prev_vres.action == "retune_here"
        and set(prev_delta).issubset(_ZERO_DETECTION_RESCUE_LEVERS)
        and prev_vres.expected_effect.get("cell_count") == "up"
        and prev_metrics.get("cell_count") == 0
        and metrics.get("cell_count") == 0
        and vres.action == "retune_here"
        and bool(delta)
        and set(delta).issubset(_ZERO_DETECTION_RESCUE_LEVERS)
    )


class StepVerifier:
    def __init__(self, max_iters: int | None = None, patience: int = 2,
                 zero_detection_rescue: bool = False, *, wave_workers: int | None = None,
                 observer: Callable[[dict], None] | None = None):
        # `max_iters` is a hard SAFETY ceiling; the plateau stop (patience) is the primary, earlier
        # stop. patience > 1 lets the loop survive a temporarily-worse retune to reach a better
        # optimum (e.g. expand-then-tighten a crop) instead of stopping on the first non-improving
        # tweak -- the deep-learning early-stopping idea (a plateau, not a single down-step).
        self.max_iters = max_iters
        self.patience = patience
        self.zero_detection_rescue = zero_detection_rescue
        # wave mode only (a verifier with `uses_waves`): concurrent tool calls per wave (default: the
        # wave size) and an observer called with one event dict per trial / wave / stop
        self.wave_workers = wave_workers
        self.observer = observer

    def _zero_detection_rescue_enabled(self) -> bool:
        return self.zero_detection_rescue or _zero_detection_rescue_env_enabled()

    def resolve_stage(self, verifier: BaseVerifier, adapter: Adapter,
                      tool_runner: ToolRunner, initial_params: dict,
                      initial_result: Any, upstream_ctx: Optional[dict] = None) -> StageResolution:
        """Verify a stage and retune+rerun while the fault is local.

        Returns a StageResolution. On `conceded` the caller (the main loop /
        planner) takes over (e.g. re-runs an upstream stage on fault="upstream").
        On accept/converge the stage's result flows downstream.
        """
        if getattr(verifier, "uses_waves", False):
            return self._resolve_waves(verifier, adapter, tool_runner, initial_params, initial_result,
                                       upstream_ctx or {})
        upstream_ctx = upstream_ctx or {}
        params = dict(initial_params)
        result = initial_result
        history: list[tuple[dict, VLRRResult, Any]] = []
        ledger: dict[str, object] = {}
        best_rank = None   # best (lowest) judge severity seen so far; the plateau is measured vs THIS
        stall = 0          # consecutive judged retunes that did not improve on best_rank

        iteration_index = 0
        while True:
            try:
                vinput, ctx = adapter(result, upstream_ctx)
                vres = verifier.verify(vinput, ctx)
            except ExecutionError as exc:
                if is_fatal_llm_error(exc):      # no key / permission / model / endpoint: a system error
                    raise
                # the call is failed (an execution error, as a tool call that failed): the loop ends with the
                # best usable call so far; a stage left without one raises
                failed = execution_failure(result, exc)
                history.append((dict(params), VLRRResult(
                    "FAIL", 0.0, {}, locus="tool_mismatch", triggered=["execution_error"], action="conceded",
                    fault="ceiling", notes=failed["error"]), failed))
                usable = [t for t in history if not is_failed_result(t[2])]
                if not usable:
                    raise ExecutionError(f"{verifier.name}: no usable call left ({failed['error']})") from exc
                res = self._best(usable, ledger, "conceded", fault="ceiling", reason="execution_error")
                res.trials = self._trials(history)
                return res
            history.append((dict(params), vres, result))

            # Reconcile-as-diagnostic: classify the PREVIOUS attempt's lever against
            # THIS attempt's actual metrics/score and WRITE the ledger, on EVERY exit
            # path (accept included -- task 021's field_aware bug: an ineffective
            # lever must be recorded honestly, not silently accepted).
            if len(history) >= 2:
                _, prev_vres, _ = history[-2]
                state = classify_lever(prev_vres, vres)
                if state:
                    for lever in prev_vres.param_delta:
                        ledger[lever] = state

            if vres.action == "accept":
                # task 021 Stage 2: adopt the label-free BEST trial seen so far, not
                # necessarily the current (accept'ed) one -- shares the SAME keep-best
                # implementation as the concede paths below (`_best`), just with status
                # "satisfied" instead of "conceded". In the common case (monotonic
                # tighten) the current trial already IS the best, so this returns
                # exactly what the old code did. `conceded`/`exclude_samples` are NOT
                # routed through this -- they keep their own current-trial semantics.
                return self._best(history, ledger, "satisfied", reason="converged")
            if vres.action == "exclude_samples":
                # D3: a distinct in-loop data-op, NOT a concede -- byte-identical to the
                # pre-migration behavior (current trial, no keep-best reselection).
                return self._resolution(result, "exclude_samples", params, vres, history,
                                        ledger, reason="exclude_samples")
            if vres.action == "conceded":
                # the verifier itself gave up (fault is "upstream" or "ceiling" --
                # task 045: a verifier-emitted "ceiling" now also covers what used
                # to be the separate "human" fault, i.e. the tool/data is unfit, not
                # just tuned-out. StepVerifier's OWN loop-stall reasons below assign
                # "ceiling" too, so the same tag can come from either source; only
                # "upstream" is exclusively verifier-emitted).
                if any(
                    getattr(trial[1], "selection_priority", 0) > 0
                    for trial in history
                ):
                    return self._best(
                        history, ledger, "conceded", fault=vres.fault,
                        reason=f"escalate_{vres.fault}")
                return self._resolution(
                    result, "conceded", params, vres, history, ledger,
                    fault=vres.fault, upstream_target=vres.upstream_target,
                    reason=f"escalate_{vres.fault}")

            # plateau bookkeeping (origin/main task 028/030): track the best judge
            # severity and how long it has stalled. Only judged attempts count; a
            # deterministic stage never stalls here and falls back to the reconcile /
            # oscillation / max_iters stops exactly as before.
            if vres.judge_severity is not None:
                rank = judge_rank(vres)
                if best_rank is None or rank < best_rank:
                    best_rank, stall = rank, 0
                else:
                    stall += 1

            # reconcile the previous prediction before applying a new tweak (the
            # EXISTING stop-gate -- loose epsilon, patience-1). A failed
            # expected-effect hypothesis TERMINATES tuning with conceded/ceiling:
            # the v8g publication-snapshot semantics, restored deliberately. The
            # `__hypothesis_invalidated__` counter is still recorded before the stop
            # so ledger consumers keep seeing it.
            if len(history) >= 2:
                _, prev_vres, _ = history[-2]
                if prev_vres.expected_effect:
                    ok = reconcile(prev_vres.expected_effect, prev_vres.metrics, vres.metrics)
                    if ok and not all(ok.values()):
                        if (
                            self._zero_detection_rescue_enabled()
                            and _is_zero_detection_retry(prev_vres, vres)
                        ):
                            # Organoid rescue pilot: zero-mask Cellpose outputs can need a
                            # larger threshold sweep than the default single retry. Keep
                            # all other reconcile stops unchanged.
                            pass
                        else:
                            ledger["__hypothesis_invalidated__"] = int(
                                ledger.get("__hypothesis_invalidated__", 0)
                            ) + 1
                            return self._best(history, ledger, "conceded", fault="ceiling",
                                              reason="hypothesis_invalidated")
                # PLATEAU stop: the independent judge's severity has not improved for
                # `patience` consecutive retunes -> the search has plateaued; stop and
                # keep the best attempt. REPLACES the single-step "did this one retune
                # improve" judge check (patience>1 lets a temporarily-worse retune
                # continue toward a better optimum instead of stopping on the first
                # down-step). Same status/reason string as that check.
                if stall >= self.patience:
                    return self._best(history, ledger, "conceded", fault="ceiling",
                                      reason="judge_no_improvement")

            for lever in (vres.param_delta or {}):
                ledger.setdefault(lever, "untried")
            new_params = apply_bounded(params, vres.param_delta or {})
            if any(new_params == p for p, _, _ in history):
                return self._best(history, ledger, "conceded", fault="ceiling",
                                  reason="oscillation_stop")
            # Do not execute an unobservable (max_iters + 1)th tool call.  The old
            # loop reran here even on its final iteration, then immediately exited
            # without verifying or selecting that artifact.
            if self.max_iters is not None and iteration_index + 1 >= self.max_iters:
                return self._best(
                    history, ledger, "conceded", fault="ceiling",
                    reason="max_iters_reached")
            params = new_params
            result = tool_runner(params)
            iteration_index += 1

        return self._best(history, ledger, "conceded", fault="ceiling", reason="max_iters_reached")

    # -- wave mode ---------------------------------------------------------------

    def _resolve_waves(self, verifier: BaseVerifier, adapter: Adapter, tool_runner: ToolRunner,
                       initial_params: dict, initial_result: Any, upstream_ctx: dict) -> StageResolution:
        """The loop for a verifier with `uses_waves` (see BaseVerifier). Trial 0 is the initial result;
        if its verify() says retune_here, the verifier's waves run until the budget -- `max_iters - 1`
        rounds after the initial trial, each round of any size -- is spent or a wave is empty; then the
        verifier selects. Each wave runs concurrently; results are verified on this thread and
        recorded in the wave's order. A candidate whose parent is unknown or failed, or whose params
        equal an earlier trial's (failed ones included) or a wave-mate's, is dropped. A tool call that
        raises is a failed trial: it counts, is never re-run, never a parent and never selected."""
        if self.max_iters is None:
            raise ValueError("wave mode needs max_iters (1 + the number of rounds)")
        emit = self.observer or (lambda event: None)
        trials: list[tuple[dict, VLRRResult, Any]] = []
        parents: list[Optional[int]] = []
        entries: list[Optional[dict]] = []

        def judge(index, parent, result, candidate):
            """(result, vres, ctx) of one trial; an execution error while verifying it makes it a failed trial
            (`execution_failure`), which the verifier sees as a failed call."""
            vres = None
            if not is_failed_result(result):
                try:
                    vinput, ctx = adapter(result, upstream_ctx)
                    ctx = {**ctx, "trial_index": index, "parent_index": parent}
                    vres = verifier.verify(vinput, ctx)
                except ExecutionError as exc:
                    if is_fatal_llm_error(exc):  # no key / permission / model / endpoint: a system error
                        raise
                    result = execution_failure(result, exc)
            if vres is None:
                vres = VLRRResult("FAIL", 0.0, {}, locus="tool_mismatch",
                                  triggered=["execution_error" if is_execution_error(result) else "tool_error"],
                                  action="conceded", fault="ceiling", notes=str(result.get("error", "")))
                ctx = dict(upstream_ctx)
            if candidate is not None:     # the move that produced this trial, measured from its parent
                vres.param_delta = dict(candidate.param_delta) or None
                vres.expected_effect = dict(candidate.expected_effect)
                vres.notes = candidate.notes or vres.notes
            return result, vres, ctx

        def record(index, parent, params, vres, ctx, result):
            trials.append((dict(params), vres, result))
            parents.append(parent)
            if parent is None:
                entries.append(None)
                return
            failed = is_failed_result(result)
            call_error = getattr(verifier, "call_error", None)
            if not failed and callable(call_error):
                # Native tools can retain masks/paths beside an error. Use the
                # verifier's same failure rule here as in its candidate selection,
                # rather than crediting a failed call in the efficacy ledger.
                trial_state = self._wave_state(trials, parents, [*entries, None],
                                               self.max_iters - 1 - rounds, upstream_ctx)
                failed = call_error(index, trial_state) is not None
            entry = ({"param_delta": dict(vres.param_delta or {}), "outcome": None, "hypothesis": None,
                      "score_delta_vs_parent": None} if failed
                     else classify_trial(trials[parent][1], vres, *verifier.noise_floors(vres, ctx)))
            gain = vres.scalar_score - trials[0][1].scalar_score
            entries.append({"index": index, "parent": parent, **entry, "failed": failed,
                            "score_gain_vs_initial": gain if not failed and math.isfinite(gain) else None})

        initial_result, vres, ctx = judge(0, None, initial_result, None)
        record(0, None, initial_params, vres, ctx, initial_result)
        emit({"event": "trial_finished", "index": 0, "parent": None, "params": dict(initial_params),
              "vres": vres, "result": initial_result, "failed": is_failed_result(initial_result), "seconds": None})
        stop = "max_iters_reached" if vres.action == "retune_here" else f"initial_{vres.action}"
        rounds = 0
        while vres.action == "retune_here" and rounds < self.max_iters - 1:
            remaining = self.max_iters - 1 - rounds
            proposed = verifier.recommend_wave(self._wave_state(trials, parents, entries, remaining, upstream_ctx))
            wave = self._admissible(list(proposed or []), trials, emit)
            if not wave:
                stop = "no_new_configuration"
                break
            rounds += 1
            emit({"event": "wave", "size": len(wave), "round": rounds, "remaining": remaining})
            base = len(trials)
            for i, candidate in enumerate(wave):
                emit({"event": "trial_started", "index": base + i, "parent": candidate.parent,
                      "params": copy.deepcopy(candidate.params), "param_delta": dict(candidate.param_delta),
                      "expected_effect": dict(candidate.expected_effect), "notes": candidate.notes})
            judged = {}
            with ThreadPoolExecutor(max_workers=self.wave_workers or len(wave)) as pool:
                futures = {pool.submit(self._run_candidate, tool_runner, c): i for i, c in enumerate(wave)}
                for future in as_completed(futures):
                    i = futures[future]
                    result, seconds = future.result()
                    judged[i] = judge(base + i, wave[i].parent, result, wave[i])
                    result = judged[i][0]
                    emit({"event": "trial_finished", "index": base + i, "parent": wave[i].parent,
                          "params": copy.deepcopy(wave[i].params), "vres": judged[i][1], "result": result,
                          "failed": is_failed_result(result), "seconds": seconds})
            for i, candidate in enumerate(wave):
                result, trial_vres, trial_ctx = judged[i]
                record(base + i, candidate.parent, candidate.params, trial_vres, trial_ctx, result)
            emit({"event": "wave_finished", "entries": list(entries[base:])})

        state = self._wave_state(trials, parents, entries, self.max_iters - 1 - rounds, upstream_ctx)
        selection = verifier.select(state, stop)
        index = selection.index
        if index is not None and (not 0 <= index < len(trials) or is_failed_result(trials[index][2])):
            raise ValueError(f"select() chose trial {index}, which is unknown or failed")
        errors = [t[2]["error"] for t in trials if is_execution_error(t[2])]
        usable = [i for i, t in enumerate(trials)
                  if not is_failed_result(t[2]) and (i > 0 or not hasattr(verifier, "input_record"))]
        if index is None and errors and not usable:      # execution errors left the stage without a usable call
            raise ExecutionError(f"{verifier.name}: no usable call left ({'; '.join(errors)[:1000]})")
        ledger: dict[str, object] = {}
        for entry in entries:
            for lever in (entry or {}).get("param_delta") or {}:
                if entry["failed"]:
                    ledger.setdefault(lever, "untried")
                elif ledger.get(lever) != "effective":
                    ledger[lever] = entry["outcome"] or ledger.get(lever, "untried")
        chosen = trials[index] if index is not None else None
        # the stage's verdict: the delivered trial's (the input's when none is delivered) with the action
        # the selection concluded -- a trial's own action (retune_here keeps the waves going) is not the
        # stage's, and a planner reading "retune_here" in the record would invoke the tool again
        verdict = replace(chosen[1] if chosen else trials[0][1],
                          action="accept" if selection.status == "satisfied" else "conceded", fault=selection.fault)
        emit({"event": "stopped", "reason": stop, "selected": index, "status": selection.status})
        return StageResolution(
            result=chosen[2] if chosen else None, status=selection.status,
            params=dict(chosen[0]) if chosen else {}, verdict=verdict,
            trials=self._trials(trials), fault=selection.fault, reason=selection.reason or stop,
            ledger=ledger, parents=list(parents),
            details={"stop_reason": stop, "selected": index, "entries": list(entries), **selection.details})

    @staticmethod
    def _wave_state(trials, parents, entries, remaining, context) -> WaveState:
        return WaveState(trials=list(trials), parents=list(parents), entries=list(entries),
                         remaining=remaining, context=dict(context))

    @staticmethod
    def _admissible(wave, trials, emit) -> list:
        kept = []
        for candidate in wave:
            why = None
            if not isinstance(candidate.parent, int) or not 0 <= candidate.parent < len(trials):
                why = "unknown parent"
            elif is_failed_result(trials[candidate.parent][2]):
                why = "failed parent"
            elif getattr(candidate, "error", None) is None and (
                    any(candidate.params == p for p, _, r in trials if not _never_ran(r))
                    or any(candidate.params == k.params for k in kept if getattr(k, "error", None) is None)):
                why = "configuration already run or already in this wave"
            if why:
                emit({"event": "candidate_dropped", "parent": candidate.parent,
                      "param_delta": dict(candidate.param_delta), "why": why})
            else:
                kept.append(candidate)
        return kept

    @staticmethod
    def _run_candidate(tool_runner: ToolRunner, candidate) -> tuple[Any, float]:
        if getattr(candidate, "error", None) is not None:      # failed before it could run: not run
            return {"_attempt_status": "FAILED", "status": EXECUTION_ERROR, "ran": False,
                    "error": f"execution error: {candidate.error}"}, 0.0
        started = time.time()
        try:
            result = tool_runner(copy.deepcopy(candidate.params))
        except Exception as exc:  # noqa: BLE001 -- a failed call is a trial, not the end of the loop
            result = {"_attempt_status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        return result, round(time.time() - started, 1)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _trials(history):
        # keep the per-attempt result too, so the record can trace each parameter
        # change's actual segmentation output (cell_count, mask path), not only its
        # score/metrics.
        return [(p, v, r) for p, v, r in history]

    def _resolution(self, result, status, params, vres, history, ledger, *,
                    fault=None, upstream_target=None, reason=""):
        return StageResolution(result=result, status=status, params=params,
                               verdict=vres, trials=self._trials(history),
                               upstream_target=upstream_target, fault=fault,
                               reason=reason, ledger=dict(ledger))

    def _best(self, history, ledger, status, *, fault=None, reason=""):
        # Training/evidence runners may mark a failed attempt INCONCLUSIVE. If at
        # least one attempt is READY, failed attempts are never eligible for keep-best;
        # this preserves the framework judge ordering without selecting a bad artifact.
        def attempt_status(result):
            if isinstance(result, Mapping):
                return result.get("_attempt_status")
            return getattr(result, "_attempt_status", None)

        statuses = [attempt_status(trial[2]) for trial in history]
        statusful = any(value is not None for value in statuses)
        ready_history = [
            trial for trial, value in zip(history, statuses)
            if value == "READY" or (not statusful and value is None)
        ]
        if statusful and not ready_history:
            _, vres, _ = history[-1]
            return StageResolution(
                result=None, status="conceded", params={}, verdict=vres,
                trials=self._trials(history), upstream_target=vres.upstream_target,
                fault="ceiling", reason="no_ready_trials", ledger=dict(ledger),
            )
        selection_history = ready_history or history
        # keep the attempt the independent judge found least-problematic -- but ONLY
        # when every attempt in history was judged. `judge_rank` maps None ("judge did
        # not run") to 0, the SAME rank as an actual "none" verdict -- so if even one
        # attempt is unjudged, ranking by (judge_rank, -score) could let that unjudged
        # attempt's rank-0 outrank a judged `major` (rank 2) attempt and win keep-best.
        # This mirrors the stop guard above, which already requires BOTH severities
        # non-None before trusting judge_rank. When any attempt is unjudged, fall back
        # to mask-shape score alone, consistent with this function's original comment
        # ("the sole key when no judge ran").
        judged = all(t[1].judge_severity is not None for t in selection_history)
        has_selection_hint = any(
            getattr(t[1], "selection_priority", 0) != 0 for t in selection_history)
        if judged and has_selection_hint:
            key = lambda t: (
                judge_rank(t[1]),
                -getattr(t[1], "selection_priority", 0),
                -t[1].scalar_score,
            )
        elif has_selection_hint:
            key = lambda t: (
                -getattr(t[1], "selection_priority", 0),
                -t[1].scalar_score,
            )
        elif judged:
            key = lambda t: (judge_rank(t[1]), -t[1].scalar_score)
        else:
            key = lambda t: -t[1].scalar_score
        params, vres, result = min(selection_history, key=key)
        return StageResolution(result=result, status=status, params=params,
                               verdict=vres, trials=self._trials(history),
                               upstream_target=vres.upstream_target, fault=fault,
                               reason=reason, ledger=dict(ledger))
