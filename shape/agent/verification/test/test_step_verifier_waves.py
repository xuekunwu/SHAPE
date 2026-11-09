"""StepVerifier wave mode (a verifier with `uses_waves`) on a toy tool: score = a function of x."""
from __future__ import annotations

import threading
import time

import pytest

from shape.agent.verification.base import BaseVerifier
from shape.agent.verification.step_verifier import StepVerifier
from shape.utils.schemas import VLRRResult, WaveCandidate, WaveSelection


def tool(params):
    x = params["x"]
    if x == "boom":
        raise RuntimeError("tool crashed")
    time.sleep(params.get("sleep", 0))
    return {"score": float(x) / 10}


def identity(result, upstream):
    return result, dict(upstream)


class WaveVerifier(BaseVerifier):
    uses_waves = True

    def __init__(self, plan, admit="retune_here", chooser=None):
        self.plan, self.admit, self.chooser, self.states, self.stops = plan, admit, chooser, [], []

    def quantify(self, out, ctx):
        return {"score": out["score"]}

    def localize(self, metrics, ctx):
        return "this_step", [], []

    def adjudicate(self, metrics, locus, triggered, ctx):
        action = self.admit if ctx["trial_index"] == 0 else "retune_here"
        return VLRRResult("PASS", metrics["score"], dict(metrics), action=action)

    def recommend_wave(self, state):
        self.states.append(state)
        return self.plan(state)

    def select(self, state, stop):
        self.stops.append(stop)
        if self.chooser is not None:
            return self.chooser(state)
        best = max((i for i in range(len(state.trials)) if not state.failed(i)),
                   key=lambda i: state.trials[i][1].scalar_score)
        return WaveSelection(best, "satisfied", "best score")

    def noise_floors(self, vres, ctx):
        return {"score": 0.05}, 0.05


def move(parent, x, **extra):
    return WaveCandidate(parent=parent, params={"x": x, **extra}, param_delta={"x": x},
                         expected_effect={"score": "up"}, notes=f"x={x}")


def by_round(*rounds):
    """A plan that returns rounds[k] for the k-th wave (then nothing)."""
    def plan(state):
        k = len(plan.calls)
        plan.calls.append(state)
        return rounds[k](state) if k < len(rounds) else []
    plan.calls = []
    return plan


def resolve(verifier, max_iters=6, **kw):
    return StepVerifier(max_iters=max_iters, **kw).resolve_stage(verifier, identity, tool, {"x": 1}, tool({"x": 1}))


def test_the_budget_counts_rounds_and_trials_keep_submission_order():
    events = []
    plan = by_round(lambda s: [move(0, 5, sleep=0.3), move(0, 2), move(0, 3)],
                    lambda s: [move(1, 7), move(1, 6)], lambda s: [move(4, 9)])
    res = resolve(WaveVerifier(plan), max_iters=3, observer=events.append)      # 2 rounds after trial 0
    assert [p["x"] for p, _, _ in res.trials] == [1, 5, 2, 3, 7, 6]       # recorded in the wave's order
    assert res.parents == [None, 0, 0, 0, 1, 1]
    assert res.details["stop_reason"] == "max_iters_reached" and len(plan.calls) == 2
    assert [s.remaining for s in plan.calls] == [2, 1]
    assert res.result == {"score": 0.7} and res.params == {"x": 7} and res.status == "satisfied"
    finished = [e["index"] for e in events if e["event"] == "trial_finished"]
    assert finished[0] == 0 and finished.index(1) > finished.index(2)       # c1 slept: it finished last
    assert [e["event"] for e in events].count("wave") == 2 and events[-1]["event"] == "stopped"


def test_the_ledger_is_measured_against_the_parent_with_the_noise_floors():
    plan = by_round(lambda s: [move(0, 5), move(0, 1.2)], lambda s: [move(1, 4)])
    res = resolve(WaveVerifier(plan), max_iters=4)
    entries = res.details["entries"]
    assert entries[0] is None
    assert entries[1]["outcome"] == "effective" and entries[1]["hypothesis"] == "effective"
    assert entries[2]["outcome"] == "inert" and entries[2]["hypothesis"] == "inert"    # +0.02 < floor 0.05
    assert entries[3]["parent"] == 1 and entries[3]["outcome"] == "inert"               # 0.4 < its parent 0.5
    assert entries[3]["score_delta_vs_parent"] == pytest.approx(-0.1)
    assert entries[3]["score_gain_vs_initial"] == pytest.approx(0.3)
    assert res.ledger == {"x": "effective"}                                            # sticky effective
    trial = res.trials[3][1]
    assert trial.param_delta == {"x": 4} and trial.expected_effect == {"score": "up"} and trial.notes == "x=4"


def test_duplicates_unknown_and_failed_parents_are_dropped_and_failed_calls_count():
    events = []
    plan = by_round(lambda s: [move(0, "boom"), move(0, 1), move(0, 3), move(0, 3)],
                    lambda s: [move(1, 4), move(9, 4), move(2, 4)])
    verifier = WaveVerifier(plan)
    res = resolve(verifier, max_iters=5, observer=events.append)
    dropped = [e["why"] for e in events if e["event"] == "candidate_dropped"]
    assert dropped == ["configuration already run or already in this wave",       # x=1 is the initial call
                       "configuration already run or already in this wave",       # the second x=3
                       "failed parent", "unknown parent"]
    assert [p["x"] for p, _, _ in res.trials] == [1, "boom", 3, 4]
    failed = res.trials[1]
    assert failed[2]["_attempt_status"] == "FAILED" and "tool crashed" in failed[2]["error"]
    assert failed[1].action == "conceded" and res.details["entries"][1]["failed"] is True
    assert verifier.states[1].failed(1) and not verifier.states[1].failed(2)
    assert res.details["stop_reason"] == "no_new_configuration"          # the third wave is empty


def test_selecting_a_failed_trial_is_an_error_and_none_keeps_the_input():
    crash = by_round(lambda s: [move(0, "boom")])
    with pytest.raises(ValueError, match="failed"):
        resolve(WaveVerifier(crash, chooser=lambda s: WaveSelection(1, "satisfied")), max_iters=2)
    keep = WaveVerifier(by_round(lambda s: [move(0, 5)]),
                        chooser=lambda s: WaveSelection(None, "conceded", "separability_unverifiable", "ceiling"))
    res = resolve(keep, max_iters=2)
    assert res.result is None and res.params == {} and res.status == "conceded" and res.fault == "ceiling"
    assert res.reason == "separability_unverifiable"
    # the stage's verdict is the input's, with the stage's conclusion; trial 0 keeps its own action
    assert res.verdict.scalar_score == res.trials[0][1].scalar_score and res.verdict is not res.trials[0][1]
    assert (res.verdict.action, res.verdict.fault) == ("conceded", "ceiling")
    assert res.trials[0][1].action == "retune_here"


def test_no_wave_runs_unless_the_initial_trial_says_retune_here():
    verifier = WaveVerifier(by_round(lambda s: [move(0, 5)]), admit="accept")
    res = resolve(verifier, max_iters=4)
    assert len(res.trials) == 1 and verifier.states == [] and verifier.stops == ["initial_accept"]


def test_a_round_is_not_cut_by_the_rounds_left():
    res = resolve(WaveVerifier(by_round(lambda s: [move(0, 2), move(0, 3), move(0, 4)])), max_iters=2)
    assert [p["x"] for p, _, _ in res.trials] == [1, 2, 3, 4] and res.details["stop_reason"] == "max_iters_reached"


def test_wave_mode_needs_a_budget():
    with pytest.raises(ValueError, match="max_iters"):
        resolve(WaveVerifier(by_round()), max_iters=None)


def test_a_wave_runs_concurrently():
    barrier = threading.Barrier(3, timeout=5)

    def meeting(params):
        barrier.wait()                 # every call of the wave must be running at the same time
        return {"score": float(params["x"]) / 10}

    plan = by_round(lambda s: [move(0, 2), move(0, 3), move(0, 4)])
    res = StepVerifier(max_iters=4).resolve_stage(WaveVerifier(plan), identity, meeting, {"x": 1}, {"score": 0.1})
    assert not any(isinstance(r, dict) and r.get("_attempt_status") == "FAILED" for _, _, r in res.trials)
    serial = StepVerifier(max_iters=4, wave_workers=1).resolve_stage(
        WaveVerifier(by_round(lambda s: [move(0, 2), move(0, 3), move(0, 4)])), identity,
        lambda p: {"score": float(p["x"]) / 10}, {"x": 1}, {"score": 0.1})
    assert [p["x"] for p, _, _ in serial.trials] == [1, 2, 3, 4]


def test_the_linear_loop_never_enters_wave_mode(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("wave mode entered")

    monkeypatch.setattr(StepVerifier, "_resolve_waves", boom)

    class Linear(BaseVerifier):
        def verify(self, out, ctx):
            return VLRRResult("PASS", 1.0, {"m": 1.0}, action="accept")

    res = StepVerifier(max_iters=3).resolve_stage(Linear(), identity, tool, {"x": 1}, {"score": 0.1})
    assert res.status == "satisfied" and res.parents == [] and res.details == {}


def test_verdict_and_record_of_a_wave_resolution():
    from shape.agent.verification.integration import make_verification_record
    from shape.agent.verifier import Verifier

    plan = by_round(lambda s: [move(0, 5), move(0, 2)])
    res = resolve(WaveVerifier(plan), max_iters=3)
    verdict = Verifier().verify_result(res)
    assert verdict.observed_effect == {"score": True}                    # the selected trial vs its parent
    record = make_verification_record("Batch_Corrector_Tool", [res], "ok")
    assert [t["parent"] for t in record["stages"][0]["trials"]] == [None, 0, 0]
    # the record carries the stage's conclusion, never a call's "retune_here" (a planner reading it would
    # invoke the tool again); each trial keeps its own action
    assert (res.verdict.action, res.verdict.fault) == ("accept", None) and record["stages"][0]["action"] == "accept"
    assert [t["action"] for t in record["stages"][0]["trials"]] == ["retune_here"] * 3
