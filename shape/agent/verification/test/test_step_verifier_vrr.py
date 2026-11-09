"""Task 032 (VRR): unit tests for `step_verifier.StepVerifier`'s single-exit loop,
lever ledger, and the escalate_*->conceded migration's control flow.

Co-located with the module under test (committed -- AGENTS.md sec 6 P5), not in the
gitignored `tests/` dir. Uses a `ScriptedVerifier` stub (returns a pre-built
VLRRResult sequence, ignoring its (vinput, ctx) inputs) so only StepVerifier's OWN
control flow is exercised -- no real tool/adapter needed.

Task 040 reconciliation: origin/main independently added a plateau-stop
(`patience`, default 2) to the judge-severity stop path, merged on top of this
task's `conceded`/ledger vocabulary (see step_verifier.py's module docstring).
`test_judge_no_improvement_concedes_ceiling` is REWRITTEN below for patience=2
(was patience=1); a new positive regression,
`test_judge_plateau_patience_2_survives_one_worse_retune_then_reaches_better_optimum`,
locks in the plateau-stop's actual point (a temporarily-worse retune survives and a
later better optimum is kept) so the merge cannot silently revert to patience=1.
"""

from shape.agent.verification.step_verifier import StepVerifier
from shape.utils.schemas import VLRRResult


class ScriptedVerifier:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def verify(self, vinput, ctx):
        vres = self.script[self.calls]
        self.calls += 1
        return vres


def _identity_adapter(result, upstream_ctx):
    return result, {}


def _noop_runner(params):
    return params


# ---- single-exit / golden path ----------------------------------------------

def test_accept_first_trial_returns_satisfied():
    vres = VLRRResult("PASS", 0.9, {"m": 1.0}, action="accept")
    verifier = ScriptedVerifier([vres])
    sv = StepVerifier(max_iters=3)
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner, {"p": 1}, "init_result")
    assert res.status == "satisfied"
    assert res.reason == "converged"
    assert res.fault is None
    assert res.ledger == {}
    assert res.result == "init_result"
    assert res.verdict is vres


# ---- lever ledger, incl. the accept path (task 021's bug) --------------------

def test_ledger_effective_on_a_retune_that_actually_helped():
    v0 = VLRRResult("WARNING", 0.50, {"tiny_mask_fraction": 0.30}, action="retune_here",
                    param_delta={"min_size": "up"}, expected_effect={"tiny_mask_fraction": "down"})
    v1 = VLRRResult("PASS", 0.85, {"tiny_mask_fraction": 0.05}, action="accept")
    verifier = ScriptedVerifier([v0, v1])
    sv = StepVerifier(max_iters=3)
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner, {"min_size": 15}, "r0")
    assert res.status == "satisfied"
    assert res.ledger == {"min_size": "effective"}


def test_ledger_no_headroom_on_a_retune_that_moved_but_did_not_improve_score():
    v0 = VLRRResult("WARNING", 0.60, {"tiny_mask_fraction": 0.30}, action="retune_here",
                    param_delta={"min_size": "up"}, expected_effect={"tiny_mask_fraction": "down"})
    v1 = VLRRResult("WARNING", 0.58, {"tiny_mask_fraction": 0.05}, action="accept")
    verifier = ScriptedVerifier([v0, v1])
    sv = StepVerifier(max_iters=3)
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner, {"min_size": 15}, "r0")
    assert res.status == "satisfied"
    assert res.ledger == {"min_size": "no-headroom"}


def test_021_field_aware_bug_accepted_ledger_records_no_headroom():
    """Task 021's actual bug: field_aware's predicted silhouette_by_batch DOWN moved
    only ~0.8% (0.503 -> 0.499), yet the loop accepted the result. The loop STATUS is
    unchanged (still accepts -- the stop-gate's bare epsilon says "moved" and this was
    never a regression to fix in the accept/stop control flow); what changes is the
    LEDGER, which now records the lever honestly instead of silent acceptance.

    Pre-merge fix (QC N1): the ledger's `inert` gate is now the SAME bare epsilon as
    the stop-gate (see base.py), so a real-but-small move like this one is no longer
    mislabeled `inert` (that would falsely claim "no measurable change" happened) --
    it lands in `no-headroom` (measurable, but too small to credit as a win).
    """
    v0 = VLRRResult("WARNING", 0.60, {"silhouette_by_batch": 0.503}, action="retune_here",
                    param_delta={"field_aware": True},
                    expected_effect={"silhouette_by_batch": "down"})
    v1 = VLRRResult("PASS", 0.61, {"silhouette_by_batch": 0.499}, action="accept")
    verifier = ScriptedVerifier([v0, v1])
    sv = StepVerifier(max_iters=3)
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner,
                           {"field_aware": False}, "r0")
    assert res.status == "satisfied"          # behavior-preserving: still accepts
    assert res.ledger == {"field_aware": "no-headroom"}   # honest, not silently accepted


# ---- D1: ceiling faults (was hypothesis_invalidated/judge_no_improvement/
#          oscillation_stop/max_iters_reached) -> conceded(fault="ceiling") ----------

def test_hypothesis_invalidated_concedes_ceiling_and_records_inert():
    """A failed expected-effect hypothesis is a terminal stop (v8g publication-snapshot
    semantics, restored): the reconcile stop-gate stays patience-1 and concedes
    `fault="ceiling"` on the same iteration, so THIS call never re-proposes the lever.
    The `__hypothesis_invalidated__` counter is still recorded before the stop."""
    v0 = VLRRResult("WARNING", 0.5, {"m": 10.0}, action="retune_here",
                    param_delta={"cellprob_threshold": "down"}, expected_effect={"m": "up"})
    # m predicted to go UP, but it went DOWN -> the hypothesis is invalidated.
    v1 = VLRRResult("WARNING", 0.4, {"m": 9.0}, action="retune_here",
                    param_delta={"cellprob_threshold": "down"}, expected_effect={"m": "up"})
    v2 = VLRRResult("PASS", 0.9, {"m": 8.0}, action="accept")
    verifier = ScriptedVerifier([v0, v1, v2])
    sv = StepVerifier()
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner,
                           {"cellprob_threshold": 0.0}, "r0")
    assert res.status == "conceded"
    assert res.fault == "ceiling"
    assert res.reason == "hypothesis_invalidated"
    assert res.ledger["cellprob_threshold"] == "inert"
    assert res.ledger["__hypothesis_invalidated__"] == 1
    assert verifier.calls == 2   # v2 is never reached within THIS resolve_stage call

def test_zero_detection_rescue_allows_recall_lever_sequence(monkeypatch):
    monkeypatch.setenv("SHAPE_ZERO_DETECTION_RESCUE", "1")
    v0 = VLRRResult("WARNING", 0.0, {"cell_count": 0}, action="retune_here",
                    param_delta={"cellprob_threshold": "down"},
                    expected_effect={"cell_count": "up"})
    v1 = VLRRResult("WARNING", 0.0, {"cell_count": 0}, action="retune_here",
                    param_delta={"cellprob_threshold": "down", "augment": True},
                    expected_effect={"cell_count": "up"})
    v2 = VLRRResult("PASS", 0.9, {"cell_count": 2}, action="accept")
    verifier = ScriptedVerifier([v0, v1, v2])
    sv = StepVerifier(max_iters=5)

    res = sv.resolve_stage(
        verifier, _identity_adapter, _noop_runner,
        {"cellprob_threshold": 0.0, "augment": False},
        "r0",
    )

    assert res.status == "satisfied"
    assert res.reason == "converged"
    assert verifier.calls == 3


def test_zero_detection_rescue_can_be_enabled_without_environment(monkeypatch):
    monkeypatch.delenv("SHAPE_ZERO_DETECTION_RESCUE", raising=False)
    v0 = VLRRResult("WARNING", 0.0, {"cell_count": 0}, action="retune_here",
                    param_delta={"cellprob_threshold": -2.0},
                    expected_effect={"cell_count": "up"})
    v1 = VLRRResult("WARNING", 0.0, {"cell_count": 0}, action="retune_here",
                    param_delta={"cellprob_threshold": -4.0},
                    expected_effect={"cell_count": "up"})
    v2 = VLRRResult("PASS", 0.9, {"cell_count": 2}, action="accept")
    verifier = ScriptedVerifier([v0, v1, v2])
    sv = StepVerifier(max_iters=5, zero_detection_rescue=True)

    res = sv.resolve_stage(
        verifier, _identity_adapter, _noop_runner,
        {"cellprob_threshold": 0.0},
        "r0",
    )

    assert res.status == "satisfied"
    assert res.reason == "converged"
    assert verifier.calls == 3


def test_judge_no_improvement_concedes_ceiling():
    """Task 040 reconciliation: origin/main's plateau-stop (patience=2, the
    StepVerifier default) means the judge-severity plateau path no longer stops
    on the FIRST non-improving retune -- it stops once severity has failed to
    improve for `patience` CONSECUTIVE retunes. A single non-improving retune
    (v0 -> v1, both "major") must NOT concede yet; only the SECOND consecutive
    non-improving retune (v1 -> v2) trips `stall >= patience` and concedes.
    """
    v0 = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here",
                    param_delta={"flow_threshold": "up"}, judge_severity="major")
    v1 = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here",
                    param_delta={"flow_threshold": "up"}, judge_severity="major")
    v2 = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here",
                    param_delta={"flow_threshold": "up"}, judge_severity="major")
    verifier = ScriptedVerifier([v0, v1, v2])
    sv = StepVerifier(max_iters=5)
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner,
                           {"flow_threshold": 0.4}, "r0")
    assert res.status == "conceded"
    assert res.fault == "ceiling"
    assert res.reason == "judge_no_improvement"
    # the concede only fires on the 3rd verify() call (stall reaches patience=2
    # only after v1->v2's second consecutive non-improvement) -- not the 2nd.
    assert verifier.calls == 3


def test_judge_plateau_patience_2_survives_one_worse_retune_then_reaches_better_optimum():
    """Positive plateau regression (the BRIEF's D1 gate): origin/main shipped the
    plateau-stop's widening (patience=2) with NO test proving its actual point --
    that a TEMPORARILY WORSE retune survives (the loop does not stop at the first
    down-step) so the search can go on to reach a BETTER later optimum, which
    keep-best then returns. This locks that behavior so the merge cannot silently
    revert the plateau-stop back to patience=1 (which would instead concede at v1
    and keep-best would wrongly return v0, never reaching v2).

    v0: judge_severity="minor" (rank 1) -- the current best.
    v1: judge_severity="major" (rank 2) -- WORSE than v0; a naive patience=1 stop
        would concede right here, keeping v0 as best. With patience=2 the loop
        must instead continue (stall=1 < patience=2).
    v2: judge_severity="none" (rank 0) -- BETTER than the v0 best; the tool
        accepts here, so resolve_stage returns via the accept/keep-best path with
        v2 as the (best AND current) trial.
    """
    v0 = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here",
                    param_delta={"flow_threshold": "up"}, judge_severity="minor")
    v1 = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here",
                    param_delta={"flow_threshold": "up"}, judge_severity="major")
    v2 = VLRRResult("PASS", 0.9, {"m": 1.0}, action="accept", judge_severity="none")
    verifier = ScriptedVerifier([v0, v1, v2])
    sv = StepVerifier(max_iters=5)
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner,
                           {"flow_threshold": 0.4}, "r0")
    # survived the v0->v1 down-step: all 3 scripted attempts were consumed
    # (a patience=1 plateau would have conceded after v1, only 2 calls).
    assert verifier.calls == 3
    assert res.status == "satisfied"
    assert res.reason == "converged"
    # keep-best returns the actual best (v2, judge_severity="none"), not v0/v1.
    assert res.verdict is v2
    assert res.verdict.judge_severity == "none"


def test_oscillation_stop_concedes_ceiling():
    # a categorical (non-numeric) lever avoids float-rounding ambiguity in the
    # revisited-params check: True -> False -> True is an exact oscillation.
    v0 = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here",
                    param_delta={"flag": False})
    v1 = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here",
                    param_delta={"flag": True})
    verifier = ScriptedVerifier([v0, v1])
    sv = StepVerifier(max_iters=5)
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner, {"flag": True}, "r0")
    assert res.status == "conceded"
    assert res.fault == "ceiling"
    assert res.reason == "oscillation_stop"


def test_max_iters_reached_concedes_ceiling():
    v0 = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here", param_delta={"diameter": "up"})
    v1 = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here", param_delta={"diameter": "up"})
    verifier = ScriptedVerifier([v0, v1])
    runner_calls = []

    def counted_runner(params):
        runner_calls.append(dict(params))
        return params

    sv = StepVerifier(max_iters=2)
    res = sv.resolve_stage(
        verifier, _identity_adapter, counted_runner,
        {"diameter": 30.0}, "r0")
    assert res.status == "conceded"
    assert res.fault == "ceiling"
    assert res.reason == "max_iters_reached"
    assert len(res.trials) == 2
    assert len(runner_calls) == 1


# ---- verifier-emitted conceded (was escalate_upstream / escalate_human, now
#      escalate_upstream / escalate_ceiling -- task 045 retires "human") -----------

def test_verifier_conceded_upstream_carries_target():
    vres = VLRRResult("FAIL", 0.2, {"flow_error": 0.9}, action="conceded", fault="upstream",
                      upstream_target="image_loading")
    verifier = ScriptedVerifier([vres])
    sv = StepVerifier(max_iters=3)
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner, {"diameter": 30.0}, "r0")
    assert res.status == "conceded"
    assert res.fault == "upstream"
    assert res.upstream_target == "image_loading"
    assert res.reason == "escalate_upstream"
    assert verifier.calls == 1        # conceded on the first verify -- no retune spent


def test_verifier_conceded_ceiling_carries_no_target():
    """Task 045: a verifier-emitted "unfit tool/data" finding now carries
    fault="ceiling" (the retired "human" value folds into this bucket) -- StepVerifier
    only forwards it, it does not distinguish the source of a "ceiling" fault."""
    vres = VLRRResult("FAIL", 0.1, {}, action="conceded", fault="ceiling")
    verifier = ScriptedVerifier([vres])
    sv = StepVerifier(max_iters=3)
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner, {}, "r0")
    assert res.status == "conceded"
    assert res.fault == "ceiling"
    assert res.upstream_target is None
    assert res.reason == "escalate_ceiling"


# ---- D3: exclude_samples is a DISTINCT status, not a concede -----------------------

def test_exclude_samples_is_distinct_not_a_concede():
    vres = VLRRResult("WARNING", 0.5, {}, action="exclude_samples")
    verifier = ScriptedVerifier([vres])
    sv = StepVerifier(max_iters=3)
    res = sv.resolve_stage(verifier, _identity_adapter, _noop_runner, {"p": 1}, "r0")
    assert res.status == "exclude_samples"
    assert res.fault is None
    assert res.reason == "exclude_samples"
    assert res.result == "r0"    # current-trial result -- no keep-best reselection
