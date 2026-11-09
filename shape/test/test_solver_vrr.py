"""Task 032 (VRR): whole-loop regression tests for `solver.py`'s escalate_*->conceded
migration -- the mandatory whole-loop regression gate (BRIEF sec "Validation",
reusing the 026 approach). Co-located with the module under test (committed --
AGENTS.md sec 6 P5), not in the gitignored `tests/` dir.

Task 045 refinement: the fault taxonomy retires its third value "human" -- folded
into "ceiling" (a verifier's own "this tool/data can't do better" finding now
carries the SAME tag as StepVerifier's own tuning-ceiling stall). Consequently
`_vlrr_loop_decision`/`_result_stop_reason`/`_worst_outcome` no longer special-case
a fault value at this per-stage layer beyond "upstream" -- a hard tool error is the
only thing that stops the loop immediately here; every other conceded outcome
(ceiling OR upstream, within budget) continues, carrying the best result forward,
per the D1 invariant. The task-completion planner prompt (`verificate_memory`,
tested in `shape/agent/test_completion_conceded_answered.py`) is the one place that
decides whether the OVERALL task should stop on a "ceiling" concession.

Covers:
  (b) cross-stage recovery still fires on a fault="upstream" case (feedback.py path)
  (d) the escalation-budget bound still caps retries
  (e) D1: a fault="ceiling" case -> solver CONTINUES carrying best (the trap) --
      now also covers the RETIRED "human" case, folded into the same bucket
  (g) D3: exclude_samples behavior byte-identical (continue, severity-0)
"""

from shape.solver import (
    _cross_stage_recovery_loop,
    _result_stop_reason,
    _severity_of,
    _try_cross_stage_recovery,
    _vlrr_loop_decision,
    _worst_outcome,
)


# ---- D1 (the trap): fault="ceiling" -> CONTINUE, carrying best ---------------------
# Covers BOTH origins of a "ceiling" fault (task 045): StepVerifier's own tuning-
# ceiling stall, and a verifier's own "unfit tool/data" finding (formerly the
# separate "human" fault, now folded into the SAME bucket -- see module docstring).

def test_D1_ceiling_fault_continues_not_stops():
    decision, n = _vlrr_loop_decision("conceded", "ceiling", {"cell_count": 5}, 0)
    assert decision == "continue"
    assert n == 0
    # `_result_stop_reason` is only ever consulted with the decision _vlrr_loop_decision
    # returned; for "continue" it must be None regardless of fault.
    assert _result_stop_reason(decision, {"cell_count": 5}) is None


def test_ex_human_ceiling_fault_also_continues_not_stops():
    """The retired "human" fault's old scenarios (tool/data judged unfit) now carry
    fault="ceiling" and take the SAME continue path -- there is no longer a distinct
    immediate-stop tier for them at this per-stage layer. Uses a result WITHOUT a
    hard-error marker (no "error"/"status": "failed") so only the fault value is
    under test, not the separate hard_error branch."""
    decision, n = _vlrr_loop_decision("conceded", "ceiling", {"cell_count": 0}, 0)
    assert decision == "continue"
    assert n == 0


def test_D1_ceiling_severity_is_1_not_a_hard_stop_tier():
    assert _severity_of("conceded", "ceiling") == 1
    assert _severity_of("conceded", "upstream") == 1  # uniform severity across conceded faults
    assert _severity_of("satisfied", None) == 0


def test_hard_error_stops_regardless_of_fault():
    decision, _ = _vlrr_loop_decision("conceded", "ceiling", {"error": "boom"}, 0)
    assert decision == "stop"
    assert _result_stop_reason(decision, {"error": "boom"}) == "hard_error"


# ---- fault="upstream" -> bounded continue, then stop once budget exhausted ---------

def test_upstream_fault_continues_within_budget_then_stops():
    decision, n = _vlrr_loop_decision("conceded", "upstream", {}, 0, max_escalation_retries=2)
    assert decision == "continue" and n == 1
    decision, n = _vlrr_loop_decision("conceded", "upstream", {}, n, max_escalation_retries=2)
    assert decision == "continue" and n == 2
    decision, n = _vlrr_loop_decision("conceded", "upstream", {}, n, max_escalation_retries=2)
    assert decision == "stop" and n == 3
    assert _result_stop_reason(decision, {}) == "escalation_budget_exhausted"


# ---- D3: exclude_samples is byte-identical -- continue, severity 0 -----------------

def test_exclude_samples_continues_severity_zero():
    decision, n = _vlrr_loop_decision("exclude_samples", None, {}, 0)
    assert decision == "continue"
    assert _severity_of("exclude_samples", None) == 0


# ---- _worst_outcome: any conceded fault > exclude_samples/satisfied; hard error wins -

def test_worst_outcome_prefers_any_conceded_fault_over_satisfied():
    outcomes = [
        ("satisfied", None, {}),
        ("conceded", "ceiling", {}),
    ]
    assert _worst_outcome(outcomes) == ("conceded", "ceiling", {})


def test_worst_outcome_hard_error_beats_even_a_conceded_fault():
    outcomes = [
        ("conceded", "ceiling", {}),
        ("satisfied", None, {"error": "boom"}),
    ]
    assert _worst_outcome(outcomes) == ("satisfied", None, {"error": "boom"})


def test_worst_outcome_default_is_satisfied():
    assert _worst_outcome([]) == ("satisfied", None, {})


def test_no_human_fault_literal_remains_reachable():
    """Task 045: "human" is retired from the fault taxonomy -- assert the dispatcher
    no longer treats it specially (it now falls through to the generic "continue",
    identical to any unrecognized/no-op fault value, never an immediate stop)."""
    decision, n = _vlrr_loop_decision("conceded", "human", {"cell_count": 0}, 0)
    assert decision == "continue"
    assert _result_stop_reason("continue", {"cell_count": 0}) is None


# ---- (b) cross-stage recovery fires on an upstream fault (feedback.py path) --------

class _FakeExecutor:
    def __init__(self, up_result):
        self.up_result = up_result
        self.calls = []

    def run_managed_tool(self, tool_name, base_kwargs, params):
        self.calls.append((tool_name, dict(base_kwargs), dict(params)))
        return self.up_result


class _FakeMemory:
    def add_action(self, **kwargs):
        pass

    def get_actions(self, llm_safe=False):
        return []


class _FakeUpstreamResolution:
    """Minimal stand-in for a StageResolution: only `.upstream_target` is read by
    `_try_cross_stage_recovery` to decide whether/how to recover."""
    def __init__(self, upstream_target):
        self.upstream_target = upstream_target


def test_cross_stage_recovery_fires_and_reruns_the_named_upstream_tool(monkeypatch):
    """Construct a fault="upstream" case (upstream_target="crop") and assert the
    bounded upstream nudge+rerun mechanism actually fires: the upstream tool
    (Single_Cell_Cropper_Tool) is invoked, then the downstream tool is re-verified.

    Production `feedback._UPSTREAM_NUDGE` defines NO remedy for any stage (task 082
    retired its one remedy, `preprocessing.correction_strength`, with the
    CellProfiler-faithful deterministic substrate; the `preprocessing` stage itself was
    later removed from the DAG entirely -- see `stage_map.py`'s module docstring). This
    test's own subject is the cross-stage RECOVERY MECHANISM (does a defined remedy
    actually fire and rerun the right tool), not any specific knob value, so it injects
    its OWN placeholder remedy on a still-managed stage (`crop`) rather than depending
    on production config or a stage that no longer exists.
    """
    import shape.agent.verification.feedback as feedback
    import shape.agent.verification.integration as integration

    monkeypatch.setattr(feedback, "_UPSTREAM_NUDGE", {"crop": {"__test_probe__": "boosted"}})

    recovered_status = {"n": 0, "request": None, "metadata_profile": None}

    def fake_verify_managed_tool(executor, tool_name, base_kwargs, tool_params, upstream_metrics, request=None,
                                 metadata_profile=None):
        # simulate: the re-verify converges clean on the first recovery pass
        recovered_status["n"] += 1
        recovered_status["request"] = request
        recovered_status["metadata_profile"] = metadata_profile
        return (["resolutions-after-recovery"], {"cell_count": 42}, "satisfied")

    monkeypatch.setattr(integration, "verify_managed_tool", fake_verify_managed_tool)
    monkeypatch.setattr(integration, "build_base_kwargs", lambda tool_name, image, memory: {})
    monkeypatch.setattr(integration, "latest_stage_metrics", lambda memory: {})

    executor = _FakeExecutor(up_result={"cell_crops_metadata_path": "/tmp/p.json", "status": "success"})
    memory = _FakeMemory()
    resolutions = [_FakeUpstreamResolution(upstream_target="crop")]

    result = _try_cross_stage_recovery(
        executor, "Cell_State_Analyzer_Tool", {"learning_rate": 3e-5},
        {"path": "/tmp/img.tif", "image_id": "img1"}, memory, step=1, resolutions=resolutions,
        request="## Target\nthe nuclei", metadata_profile="the-profile")

    assert result == (["resolutions-after-recovery"], {"cell_count": 42}, "satisfied")
    assert recovered_status["n"] == 1
    # the re-verification carries the user's request, as the first verification did
    assert recovered_status["request"] == "## Target\nthe nuclei"
    assert recovered_status["metadata_profile"] == "the-profile"     # and the run's metadata profile
    # the UPSTREAM tool (mapped from stage "crop") was actually re-run.
    assert len(executor.calls) == 1
    up_tool_name, _, up_params = executor.calls[0]
    assert up_tool_name == "Single_Cell_Cropper_Tool"
    # feedback.nudge_params applied the injected test-only remedy (a categorical value --
    # apply_bounded's non-directional passthrough, so it needs no PARAM_BOUNDS entry).
    assert up_params.get("__test_probe__") == "boosted"


def test_cross_stage_recovery_loop_pure_control_flow_stops_on_convergence(monkeypatch):
    """`_cross_stage_recovery_loop` in isolation (per its own docstring): stops the
    FIRST time `attempt` reports "satisfied", not exhausting the rerun budget. Same
    injected placeholder remedy convention as the sibling test above (production has no
    remedy defined for any stage) -- this test's subject is the pure control-flow
    stop-on-convergence behaviour, independent of which stage/remedy is configured."""
    import shape.agent.verification.feedback as feedback
    monkeypatch.setattr(feedback, "_UPSTREAM_NUDGE", {"crop": {"__test_probe__": "boosted"}})

    calls = []

    def attempt(n):
        calls.append(n)
        return (["r"], {}, "satisfied")

    out = _cross_stage_recovery_loop("crop", attempt, max_reruns=2)
    assert out == (["r"], {}, "satisfied")
    assert calls == [0]     # stopped after the first pass -- did not spend the 2nd


def test_cross_stage_recovery_loop_returns_none_with_no_defined_remedy():
    def attempt(n):
        raise AssertionError("must not be called -- no remedy for this target")
    assert _cross_stage_recovery_loop("no_such_stage", attempt) is None


# ---- a failed managed stage: continue (re-plan), and the planner is told no output was produced ---------

def test_a_failed_managed_stage_continues_tells_the_planner_no_output_and_ends_within_max_steps(tmp_path,
                                                                                                monkeypatch):
    """The planner picks Batch_Corrector_Tool on the solve() path, which hands it no inputs: the call raises in
    the executor, the stage concedes with no output (`initial_trial_failed`), the loop continues
    (`('continue', 0)`), the planner's memory digest says NO OUTPUT produced, and the run ends at max_steps.
    The batch corrector is system-routed on the solve() path (`stage_map.SYSTEM_ROUTED_TOOLS`); this test takes
    it off that list so the planner's pick goes through the managed path, as any managed tool's would."""
    from types import SimpleNamespace

    import shape.solver as solver_module
    from shape.agent.executor import Executor
    from shape.agent.memory import Memory
    from shape.agent.planner import Planner
    from shape.agent.verification import stage_map

    monkeypatch.setattr(stage_map, "SYSTEM_ROUTED_TOOLS", ())
    decisions = []
    real_decision = solver_module._vlrr_loop_decision

    def recording_decision(*args, **kwargs):
        decisions.append(real_decision(*args, **kwargs))
        return decisions[-1]

    monkeypatch.setattr(solver_module, "_vlrr_loop_decision", recording_decision)
    digests = []

    class _Planner:
        recommended_params = {"harmony_params": {}}
        task_spec = None

        def verificate_memory(self, memory, **kwargs):
            return SimpleNamespace(stop_signal=False, analysis="not answered yet")

        def generate_next_step(self, memory, **kwargs):
            digests.append(Planner._format_memory(object.__new__(Planner), memory))
            return SimpleNamespace(tool_name="Batch_Corrector_Tool", sub_goal="correct the batch effect",
                                   context="", justification="")

    executor = Executor("unused-model", query_cache_dir=str(tmp_path))
    executor.enable_llm_verifier = False
    memory = Memory()
    solver = {"planner": _Planner(), "executor": executor, "memory": memory, "initializer": None,
              "available_tools": ["Batch_Corrector_Tool"]}
    stop = solver_module._run_steps(solver, "correct the batch effect", None, "analysis", 3, True)
    assert stop == "max_steps_reached" and decisions == [("continue", 0)] * 3
    actions = memory.get_actions(llm_safe=False)
    assert len(actions) == 3 and all(a["result"] is None for a in actions)
    stage = actions[0]["verification"]["stages"][0]
    assert (stage["status"], stage["reason"], stage["no_output"]) == ("conceded", "initial_trial_failed", True)
    assert "run_managed_tool failed" in stage["notes"]
    # the planner sees it: the steps after the first carry the no-output digest
    assert "NO OUTPUT produced (initial_trial_failed)" not in digests[0]
    assert all("batch_correction=FAIL/conceded fault=ceiling NO OUTPUT produced (initial_trial_failed)" in d
               for d in digests[1:])


# ---- the planner naming no tool is the planner's stop, not an unregistered tool ------------------
def test_planner_choosing_no_tool_is_recognised():
    from shape.solver import _planner_chose_no_tool
    for name in ("No matched tool given: None", "No matched tool given: NONE",
                 "No matched tool given: No further tool call",
                 "No matched tool given: NONE — task complete",
                 "No matched tool given: None — terminal stage reached",
                 "No matched tool given: None (task complete)", "None", "", None):
        assert _planner_chose_no_tool(name), name
    for name in ("No matched tool given: Nuclei_Segmenter_Tool", "No matched tool given: Image_Denoiser",
                 "Cell_Segmenter_Tool"):
        assert not _planner_chose_no_tool(name), name
