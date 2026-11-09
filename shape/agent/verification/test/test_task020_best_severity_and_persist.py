"""Task 020: fix `StepVerifier._best`'s keep-best None-conflation + persist judge_severity.

Bug (see BRIEF.md): `judge_rank` maps BOTH "judge said none" and "judge did not run"
(None) to rank 0 (best). The old keep-best key `(judge_rank(t[1]), -t[1].scalar_score)`
therefore lets an UNJUDGED attempt (rank 0) beat a JUDGED `major` attempt (rank 2) --
even though the unjudged attempt might have a much worse actual score. The stop guard
(step_verifier.py:87-90) already requires BOTH severities non-None before trusting
judge_rank; `_best` did not have the equivalent guard. This test file proves:
  1. all-judged history: severity-primary ranking is unchanged vs the old key.
  2. mixed history (the bug): the NEW `_best` picks the higher-score judged attempt,
     while the OLD key would have picked the unjudged (None) one -- shown explicitly.
  3. deterministic all-None history: `_best` picks max scalar_score, identical to the
     old key's pick (regression guard for the no-judge-ran path).
  4. `judge_severity` is now persisted in each trial dict of `make_verification_record`.

Recovered verbatim from `.agent/archives/020_keepbest-severity-conflation/results/` for
task 032's "020 preserved" whole-loop regression gate (the 026 approach). Adapted ONLY
for `StepVerifier._best`'s new signature (`_best(history, ledger, status, *, fault=,
reason=)` -- task 032 added the lever ledger + fault/reason fields, BRIEF sec A); the
picked-attempt assertions themselves (the actual behavior under test) are unchanged.
"""

from __future__ import annotations

from shape.agent.verification.base import judge_rank
from shape.agent.verification.integration import make_verification_record
from shape.agent.verification.step_verifier import StepVerifier
from shape.utils.schemas import StageResolution, VLRRResult


def _vres(scalar_score, judge_severity=None, decision="WARNING", action="retune_here"):
    return VLRRResult(decision, scalar_score, {}, action=action,
                       judge_severity=judge_severity)


def _old_key(t):
    """The pre-fix keep-best key, kept here ONLY to prove the flip in test 2."""
    return (judge_rank(t[1]), -t[1].scalar_score)


# ---- (1) all-judged: severity primary, unchanged vs the old key --------------

def test_all_judged_severity_is_primary_unchanged_vs_old_key():
    params_a, params_b = {"p": 1}, {"p": 2}
    result_a, result_b = {"r": "a"}, {"r": "b"}
    # a: worse severity ("major", rank 2) but the HIGHER scalar_score
    # b: better severity ("none", rank 0) but a LOWER scalar_score
    history = [
        (params_a, _vres(0.9, judge_severity="major"), result_a),
        (params_b, _vres(0.4, judge_severity="none"), result_b),
    ]

    sv = StepVerifier()
    resolution = sv._best(history, {}, "conceded", fault="ceiling", reason="max_iters_reached")

    old_pick_params, _, _ = min(history, key=_old_key)

    assert resolution.params == params_b == old_pick_params
    assert resolution.verdict.judge_severity == "none"


# ---- (2) mixed history: THE fix -- prove the flip explicitly -----------------

def test_mixed_history_new_best_flips_pick_vs_old_key():
    params_major, params_none = {"p": "major_attempt"}, {"p": "unjudged_attempt"}
    result_major, result_none = {"r": "major"}, {"r": "unjudged"}
    history = [
        (params_major, _vres(0.9, judge_severity="major"), result_major),
        (params_none, _vres(0.5, judge_severity=None), result_none),
    ]

    # OLD key: judge_rank(None) == judge_rank("none") == 0, so the unjudged (rank 0,
    # score 0.5) attempt beats the judged major (rank 2, score 0.9) attempt.
    old_pick_params, old_pick_vres, _ = min(history, key=_old_key)
    assert old_pick_params == params_none
    assert old_pick_vres.judge_severity is None
    assert old_pick_vres.scalar_score == 0.5

    # NEW _best: not all attempts judged (one is None) -> falls back to score-only ->
    # picks the 0.9 major attempt instead.
    sv = StepVerifier()
    resolution = sv._best(history, {}, "conceded", fault="ceiling", reason="max_iters_reached")

    assert resolution.params == params_major
    assert resolution.verdict.judge_severity == "major"
    assert resolution.verdict.scalar_score == 0.9
    # the flip: new pick != old pick
    assert resolution.params != old_pick_params


# ---- (3) deterministic all-None history: identical to the old key's pick ----

def test_deterministic_all_none_picks_max_score_identical_to_old_key():
    params_a, params_b, params_c = {"p": 1}, {"p": 2}, {"p": 3}
    result_a, result_b, result_c = {"r": "a"}, {"r": "b"}, {"r": "c"}
    history = [
        (params_a, _vres(0.3, judge_severity=None), result_a),
        (params_b, _vres(0.8, judge_severity=None), result_b),
        (params_c, _vres(0.6, judge_severity=None), result_c),
    ]

    old_pick_params, old_pick_vres, _ = min(history, key=_old_key)

    sv = StepVerifier()
    resolution = sv._best(history, {}, "conceded", fault="ceiling", reason="max_iters_reached")

    assert resolution.params == params_b == old_pick_params
    assert resolution.verdict.scalar_score == 0.8 == old_pick_vres.scalar_score


# ---- (4) persist: judge_severity now appears in each trial dict --------------

def _resolution_with_one_trial(judge_severity):
    params = {"p": 1}
    result = {"r": "x", "cell_count": 5}
    verdict = _vres(0.7, judge_severity=judge_severity, decision="PASS", action="accept")
    verdict.judge_problem = "cells merged" if judge_severity else ""
    return StageResolution(result=result, status="satisfied", params=params,
                            verdict=verdict, trials=[(params, verdict, result)])


def test_make_verification_record_persists_judge_severity():
    res_major = _resolution_with_one_trial(judge_severity="major")
    res_none = _resolution_with_one_trial(judge_severity=None)

    # this tool maps to exactly two verification stages (representation +
    # morphological_state, per stage_map.py), so zip(stages, resolutions) pairs each
    # resolution with its own stage 1:1.
    record = make_verification_record(
        "Cell_State_Analyzer_Tool", [res_major, res_none], "satisfied")

    trials_major = record["stages"][0]["trials"]
    trials_none = record["stages"][1]["trials"]

    assert len(trials_major) == 1
    assert trials_major[0]["judge_severity"] == "major"
    assert len(trials_none) == 1
    assert trials_none[0]["judge_severity"] is None
