"""Task 045 Phase-2 (completion-logic fix + refinement) unit tests: the STOP
prompt's resolved A/B conflict in `Planner.verificate_memory` (planner.py, the
"CRITICAL: Each step may carry a `QC:` line..." block).

Round 1 root cause: rule A ("the query's specific ask is answered -> STOP") and
rule B ("a stage conceded -> prefer continuing") could both fire on a
conceded-but-answered stage (e.g. a segmentation that conceded on a hard image but
still produced a cell count), and the model followed B -> escalated out of scope
(preprocess -> re-seg -> crop -> analyze -> UMAP crash on 2 cells, per FIX_BRIEF).

Refinement-round root cause (the coordinator's spot-run of round 1): on a "robust
segmentation"-style ask, a TUNING-CEILING concession (`fault=ceiling` -- the tool
exhausted its tunable parameters) is still merely best-effort, not "no usable
output" -- so round 1's "no usable output -> continue" carve-out did not
obviously exclude it, and re-invoking the same tool just reproduces the same
concession forever (a hang, not a crash). The refinement makes the stage's own
`fault=` tag (rendered verbatim in the `QC:` digest by `_format_qc`) the EVIDENCE
the completion prompt reasons over: the tool already judged its own mask quality
(its VRR loop); the planner's separate job is only whether re-invoking would
plausibly change anything.

Phase-2 refinement v3 (this round -- see CODING_LOG.md): three further corrections,
directed mid-task:
  1. The fault taxonomy retires its third value "human" entirely -- folded into
     "ceiling" (every VERIFIER site that used to emit fault="human" now emits
     fault="ceiling", same `notes`). The planner no longer distinguishes a
     "ceiling" vs a "human" concession -- both were already STOP outcomes, so this
     is a pure vocabulary simplification: a `conceded` with NO `upstream_target` is
     now the ONE "stop-here" bucket, regardless of why the tool/data can't do
     better. See `shape/agent/verification/base.py`, `shape/solver.py` for the
     non-prompt sites; `shape/agent/verification/verifiers/*.py` for the verifier
     renames.
  2. `fault=upstream` is reframed as a bounded, VERIFICATION-oriented look-back
     (not a from-scratch re-run of the upstream stage), skeptical by default, and
     STRICTLY bounded to at most ONE look-back per upstream stage -- if the named
     upstream stage already ran again since the concession and the verdict is
     unchanged, STOP instead of looping again (test `..._bounded_once_then_stop`
     below).
  3. cell_00136 over-stop regression: a non-terminal stage's `satisfied` verdict
     (e.g. image_loading=PASS) does NOT itself answer a segmentation/count query --
     it only means "proceed to the required terminal stage". Test
     `test_continues_when_only_non_terminal_stage_satisfied` below reproduces it.

Design (state assumptions, per Karpathy guideline): no live LLM call is made here --
this is a CPU unit test. A fake engine returns a FIXED `MemoryVerification`
simulating what a prompt-compliant reader should conclude, so what is actually
under test is (1) the wiring -- the fake's `stop_signal` propagates unchanged
through `verificate_memory` -> `_verify_task_complete` -> `decide_stop`, matching
`shape/agent/test_verificate_memory_cpu_replay.py`'s CPU-tier convention -- and
(2) that the fault type + the answering field (or its absence) actually reach the
prompt text via the real `_format_memory`/`_format_qc` pipeline, so a real LLM
would have the information the rule needs. Whether a live LLM actually follows
the wording is the GPU spot-run's job (FIX_BRIEF M2), out of this tier's reach by
construction.
"""

from shape.agent import planner as planner_module
from shape.agent.memory import Memory
from shape.agent.planner import Planner
from shape.schemas.formatters import MemoryVerification
from shape.solver import _verify_task_complete, decide_stop


class _FakeEngine:
    """Records every `generate` call's prompt and returns a canned response."""

    def __init__(self, response):
        self.calls = []
        self._response = response

    def generate(self, prompt, response_format=None, **kwargs):
        self.calls.append(prompt)
        return self._response


def _make_planner(monkeypatch, response):
    fake_text = _FakeEngine(response)
    fake_mm = _FakeEngine(response)

    def fake_create_engine(model_string, is_multimodal=False, api_key=None, **kw):
        return fake_mm if is_multimodal else fake_text

    monkeypatch.setattr(planner_module, "create_engine", fake_create_engine)
    planner = Planner(llm_engine_name="gpt-4o", api_key="test-key")
    monkeypatch.setattr(planner, "get_image_info", lambda *a, **kw: {})
    return planner, fake_text, fake_mm


# ---- (a) conceded-but-answered "how many cells" query -> STOP -----------------

def test_stops_on_conceded_but_answered_count_query(monkeypatch):
    question = "How many cells are in this image?"
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(analysis="best-effort count available", stop_signal=True))

    memory = Memory()
    memory.set_query(question)
    memory.add_action(
        step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
        command="Cell_Segmenter_Tool.execute(...)",
        result={"cell_count": 12, "status": "success"},
        verification={
            "status": "conceded",
            "stages": [{"stage": "segmentation", "decision": "FAIL", "action": "conceded",
                        "status": "conceded", "fault": "ceiling",
                        "notes": "target cells clearly MISSED on a difficult image"}],
        },
    )

    task_verdict = _verify_task_complete(planner, question, "image.png", "", memory)

    assert decide_stop(task_verdict) is True
    assert task_verdict.stop_reason == "task_complete"
    # the fixed engine is the TEXT engine (no image bytes attached)
    assert fake_mm.calls == []
    assert len(fake_text.calls) == 1
    prompt = fake_text.calls[0]
    assert isinstance(prompt, str)
    # the ingredients the new rule needs are actually in the prompt: the per-step
    # `QC:` digest's dynamic "stage=decision/action" marker for THIS concession,
    # its `fault=` evidence tag (the refinement's digest addition), and the count
    # that already answers the ask (all come from the memory content, not the
    # static instruction text).
    assert "segmentation=FAIL/conceded" in prompt
    assert "fault=ceiling" in prompt
    assert "'cell_count': 12" in prompt


# ---- (b) genuinely-incomplete multi-tool query -> CONTINUE ---------------------

def test_continues_on_genuinely_incomplete_multi_tool_query(monkeypatch):
    """Segmentation succeeded (no concession) but the query also asks for
    morphological-state characterization, which requires cropping + the analyzer --
    neither has run yet. This mirrors the referee's scope-D1 failure case: a
    required-for-the-ask stage has produced no output at all, so rule B's
    "prefer continuing" (not rule A) governs."""
    question = "Segment the cells and characterize their morphological states."
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(
            analysis="segmentation done but no morphological-state analysis yet",
            stop_signal=False))

    memory = Memory()
    memory.set_query(question)
    memory.add_action(
        step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
        command="Cell_Segmenter_Tool.execute(...)",
        result={"mask_path": "mask.npy", "cell_count": 12, "status": "success"},
    )

    task_verdict = _verify_task_complete(planner, question, "image.png", "", memory)

    assert decide_stop(task_verdict) is False
    assert task_verdict.stop_reason is None
    assert len(fake_text.calls) == 1
    # no verification was recorded for this step -- the formatted memory carries no
    # per-step `  QC:` digest line at all, so there is nothing for the new "conceded
    # but answered" carve-out to latch onto; this is a plain incomplete-chain case
    # (checked on the memory formatting directly, not the full prompt, since the
    # static instructions legitimately mention "QC:" themselves).
    assert "\n  QC:" not in planner._format_memory(memory)


# ---- (c) conceded stage with a CEILING fault and NO usable output at all -> STOP,
#          not continue. RENAMED AND FLIPPED from round 1's
#          `test_continues_on_conceded_stage_with_no_usable_output`; re-tagged from
#          `fault=human` to `fault=ceiling` this round (Phase-2 refinement v3 #1 --
#          "human" is retired, folded into "ceiling").
#
# Investigation-first justification for the flip: this scenario (the tool/data is
# judged unfit, no upstream fix available) means no tool in this system can
# auto-resolve the limitation (verified against the pre-existing, UNEDITED
# `planner.py` ESCALATION HANDLING rule, which already says exactly this for the
# next-step/tool-selection prompt: "action=conceded with NO upstream_target ...
# stop and report it rather than looping"). Continuing would just burn steps
# trying tools that cannot fix an unfit-tool/no-usable-data fault -- the honest
# behavior is to stop and report the failure, not loop hoping something changes.
# Round 1's rule (a bare "no usable output -> continue" carve-out) never actually
# checked the fault TYPE, so it disagreed with the tool-selection prompt's
# own, already-correct rule for this exact case -- this refinement removes that
# inconsistency rather than introducing a new one.

def test_stops_on_conceded_ceiling_fault_even_with_no_usable_output(monkeypatch):
    question = "How many cells are in this image?"
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(
            analysis="tool/data judged unfit, cannot be auto-resolved -- reporting the "
                     "limitation", stop_signal=True))

    memory = Memory()
    memory.set_query(question)
    memory.add_action(
        step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
        command="Cell_Segmenter_Tool.execute(...)",
        result={"status": "failed", "error": "image quality too low to segment"},
        verification={
            "status": "conceded",
            "stages": [{"stage": "segmentation", "decision": "FAIL", "action": "conceded",
                        "status": "conceded", "fault": "ceiling",
                        "notes": "image quality too low to segment; no upstream fix available"}],
        },
    )

    task_verdict = _verify_task_complete(planner, question, "image.png", "", memory)

    assert decide_stop(task_verdict) is True
    assert task_verdict.stop_reason == "task_complete"
    prompt = fake_text.calls[0]
    assert "segmentation=FAIL/conceded" in prompt
    assert "fault=ceiling" in prompt
    # no count anywhere in the recorded result -- the ask has nothing to answer
    # with, yet the fault type alone (not "is there usable output") is what
    # decides STOP here.
    assert "cell_count" not in prompt
    # Phase-2 refinement v3 #1: the retired literal must not resurface anywhere.
    assert "human" not in prompt.lower()
    # 045 P3 Fix 3: the completion prompt no longer hard-codes downstream tool names
    # (Analysis_Visualizer_Tool / Cell_State_Analyzer completion rules removed); the
    # answering stage is expressed as the DERIVED REQUIRED TERMINAL stage instead.
    assert "Analysis_Visualizer_Tool" not in prompt
    assert "TERMINAL" in prompt.upper()


# ---- (d) conceded TUNING-CEILING fault on a demanding "robust segmentation"
#          ask -> STOP (the coordinator's spot-run regression: this exact case
#          used to re-segment forever, since round 1's rule never checked the
#          fault type and a "robust" ask made a merely best-effort ceiling
#          concession look "not fully answered") ----------------------------

def test_stops_on_conceded_tuning_ceiling_for_robust_segmentation_query(monkeypatch):
    question = ("Perform robust single-cell instance segmentation on this "
                "microscopy image.")
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(
            analysis="tuning ceiling reached; best-effort mask/count reported",
            stop_signal=True))

    memory = Memory()
    memory.set_query(question)
    memory.add_action(
        step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
        command="Cell_Segmenter_Tool.execute(...)",
        result={"cell_count": 10, "status": "success"},
        verification={
            "status": "conceded",
            "stages": [{"stage": "segmentation", "decision": "FAIL", "action": "retune_here",
                        "status": "conceded", "fault": "ceiling",
                        "notes": "target cells clearly MISSED on a difficult image"}],
        },
    )

    task_verdict = _verify_task_complete(planner, question, "image.png", "", memory)

    assert decide_stop(task_verdict) is True
    assert task_verdict.stop_reason == "task_complete"
    prompt = fake_text.calls[0]
    assert "segmentation=FAIL/retune_here" in prompt
    assert "fault=ceiling" in prompt
    assert "'cell_count': 10" in prompt


# ---- (e) conceded UPSTREAM fault, FIRST occurrence -> CONTINUE (the one
#          legitimate, bounded recovery case: the tool blames its INPUT, not its
#          own tuning ceiling; proves the refinement does not over-correct into
#          "always stop on any concession") -----------------------------------

def test_continues_on_conceded_upstream_fault_bounded_recovery(monkeypatch):
    question = "How many cells are in this image?"
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(
            analysis="segmentation blames input quality; re-check image loading once, "
                     "then re-try segmentation", stop_signal=False))

    memory = Memory()
    memory.set_query(question)
    memory.add_action(
        step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
        command="Cell_Segmenter_Tool.execute(...)",
        result={"status": "failed", "error": "input contrast too low"},
        verification={
            "status": "conceded",
            "upstream_target": "image_loading",
            "stages": [{"stage": "segmentation", "decision": "FAIL", "action": "conceded",
                        "status": "conceded", "fault": "upstream",
                        "notes": "High flow_error traced to poor input quality."}],
        },
    )

    task_verdict = _verify_task_complete(planner, question, "image.png", "", memory)

    assert decide_stop(task_verdict) is False
    assert task_verdict.stop_reason is None
    prompt = fake_text.calls[0]
    assert "segmentation=FAIL/conceded" in prompt
    assert "fault=upstream" in prompt
    assert "upstream_target=image_loading" in prompt


# ---- (c) [Phase-2 refinement v3 #2] fault=upstream, SECOND occurrence -> STOP,
#          not another loop. The look-back already happened once (image loading
#          re-ran in response to the FIRST concession, visible in PREVIOUS STEPS)
#          and segmentation conceded upstream toward the SAME target again with an
#          unchanged verdict -- strictly bounded to at most ONE look-back per
#          upstream stage, so this must STOP rather than recommend re-running
#          image loading a second time (the vicious-cycle guard). ---------------

def test_stops_on_conceded_upstream_fault_after_bounded_lookback_already_done(monkeypatch):
    question = "How many cells are in this image?"
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(
            analysis="image loading already re-ran once for this upstream target and "
                     "segmentation still concedes the same way -- do not loop again, "
                     "accept the best-effort limitation and stop", stop_signal=True))

    memory = Memory()
    memory.set_query(question)
    upstream_verification = {
        "status": "conceded",
            "upstream_target": "image_loading",
        "stages": [{"stage": "segmentation", "decision": "FAIL", "action": "conceded",
                    "status": "conceded", "fault": "upstream",
                    "notes": "High flow_error traced to poor input quality."}],
    }
    # 1) first concession, blaming image loading
    memory.add_action(
        step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
        command="Cell_Segmenter_Tool.execute(...)",
        result={"status": "failed", "error": "input contrast too low"},
        verification=upstream_verification,
    )
    # 2) the bounded look-back: image loading re-checked once in response to (1)
    memory.add_action(
        step_count=2, tool_name="Cell_Segmenter_Tool",
        sub_goal="cross-stage feedback: re-check image loading",
        command="Cell_Segmenter_Tool.execute(...)",
        result={"image_path": "image.png", "status": "success"},
    )
    # 3) segmentation re-tried after the look-back -- STILL conceded, same target,
    #    unchanged verdict
    memory.add_action(
        step_count=3, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
        command="Cell_Segmenter_Tool.execute(...)",
        result={"status": "failed", "error": "input contrast still too low"},
        verification=upstream_verification,
    )

    task_verdict = _verify_task_complete(planner, question, "image.png", "", memory)

    assert decide_stop(task_verdict) is True
    assert task_verdict.stop_reason == "task_complete"
    # the evidence a prompt-compliant reader needs is actually present: the
    # look-back step ran (visible as its own step), and BOTH the original and the
    # post-look-back concession are visible (two occurrences), not just one.
    # Checked on the formatted PREVIOUS STEPS directly (not the full prompt --
    # the static CRITICAL/CHECKLIST instructions legitimately mention
    # "fault=upstream" themselves, as literal vocabulary, so counting on the full
    # prompt would double-count instruction text, not memory evidence).
    formatted_memory = planner._format_memory(memory)
    assert "re-check image loading" in formatted_memory
    assert formatted_memory.count("fault=upstream") == 2
    assert formatted_memory.count("upstream_target=image_loading") == 2


# ---- (d) [Phase-2 refinement v3 #3, cell_00136 regression] a non-terminal stage's
#          `satisfied` verdict (image_loading=PASS) does NOT answer a "segment
#          cells" query -- segmentation (the required terminal stage) has not run
#          at all yet -> CONTINUE, never STOP just because an earlier stage passed.

def test_continues_when_only_non_terminal_stage_satisfied(monkeypatch):
    question = "Segment and count the cells in this image."
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(
            analysis="image loading passed but segmentation (the query's required "
                     "terminal stage) has not run yet -- continue", stop_signal=False))

    memory = Memory()
    memory.set_query(question)
    memory.add_action(
        step_count=1, tool_name="Image_Loading", sub_goal="load image",
        command="load_image(...)",
        result={"image_path": "image.png", "status": "success"},
        verification={
            "status": "satisfied",
            "stages": [{"stage": "image_loading", "decision": "PASS", "action": "accept",
                        "fault": None, "notes": "", "metrics": {}}],
        },
    )

    task_verdict = _verify_task_complete(planner, question, "image.png", "", memory)

    assert decide_stop(task_verdict) is False
    assert task_verdict.stop_reason is None
    prompt = fake_text.calls[0]
    # the non-terminal stage's PASS is visible in the per-step memory digest.
    assert "image_loading=PASS/accept" in prompt
    # the ACTUAL recorded steps (not the prompt's static pipeline instructions,
    # which legitimately mention "segmentation" themselves) carry no segmentation
    # step at all -- checked on the formatted memory directly, matching test (b)'s
    # convention above.
    formatted_memory = planner._format_memory(memory)
    assert "segmentation" not in formatted_memory.lower()
    assert "Cell_Segmenter_Tool" not in formatted_memory


# ---- (f) [045 P3 Fix 1] tool self-signals are EVIDENCE, not deterministic overrides.
#          Previously a tool's can_terminate_after_chain RETURNED stop_signal=True by
#          fiat (bypassing the completion LLM), and an unmet required_next_tools
#          SUPPRESSED the stop-check. Both are now surfaced to the single stop
#          authority (the completion LLM), which decides. -----------------------------

def test_can_terminate_signal_is_evidence_not_a_forced_stop(monkeypatch):
    question = "Analyze the cell states in this image."
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(analysis="weigh it; continue", stop_signal=False))
    memory = Memory()
    memory.set_query(question)
    memory.add_action(
        step_count=1, tool_name="Cell_State_Analyzer_Tool", sub_goal="analyze",
        command="Cell_State_Analyzer_Tool.execute(...)",
        result={"status": "success", "can_terminate_after_chain": True,
                "termination_reason": "too few cells for clustering"},
    )
    task_verdict = _verify_task_complete(planner, question, "image.png", "", memory)
    # the flag no longer force-stops; the LLM (here declining) governs.
    assert decide_stop(task_verdict) is False
    prompt = fake_text.calls[0]
    assert "TOOL SELF-SIGNALS" in prompt
    assert "LIMITATION" in prompt
    assert "too few cells for clustering" in prompt


def test_can_terminate_signal_the_llm_may_still_stop_on_it(monkeypatch):
    question = "Analyze the cell states in this image."
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(analysis="limitation reported; stop", stop_signal=True))
    memory = Memory()
    memory.set_query(question)
    memory.add_action(
        step_count=1, tool_name="Cell_State_Analyzer_Tool", sub_goal="analyze",
        command="Cell_State_Analyzer_Tool.execute(...)",
        result={"status": "success", "can_terminate_after_chain": True,
                "termination_reason": "too few cells"},
    )
    task_verdict = _verify_task_complete(planner, question, "image.png", "", memory)
    assert decide_stop(task_verdict) is True          # via the LLM, weighing the evidence
    assert "LIMITATION" in fake_text.calls[0]


def test_unmet_required_next_tools_surface_as_evidence(monkeypatch):
    question = "Analyze the cell states in this image."
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(analysis="continue to the visualizer", stop_signal=False))
    memory = Memory()
    memory.set_query(question)
    memory.add_action(
        step_count=1, tool_name="Cell_State_Analyzer_Tool", sub_goal="analyze",
        command="Cell_State_Analyzer_Tool.execute(...)",
        result={"status": "success", "required_next_tools": ["Analysis_Visualizer_Tool"]},
    )
    task_verdict = _verify_task_complete(planner, question, "image.png", "", memory)
    prompt = fake_text.calls[0]
    assert "TOOL SELF-SIGNALS" in prompt
    assert "Analysis_Visualizer_Tool" in prompt
    assert "not complete until" in prompt.lower()


# ---- (g) [045 P3, Fix-4 close-out] the lever ledger is EVIDENCE toward stopping a
#          ceiling concede -- surfaced to the completion LLM, which still DECIDES. NOT a
#          deterministic ledger->stop branch (that would re-introduce the kind of
#          override Fix 1 removed and violate "no hard thresholds"). --------------------

def _ceiling_inert_memory(question):
    memory = Memory()
    memory.set_query(question)
    memory.add_action(
        step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
        command="Cell_Segmenter_Tool.execute(...)",
        result={"cell_count": 41, "status": "success"},
        verification={
            "status": "conceded",
            "stages": [{"stage": "segmentation", "decision": "FAIL", "action": "retune_here",
                        "status": "conceded", "fault": "ceiling", "notes": "faint cells missed",
                        "ledger": {"cellprob_threshold": "inert", "augment": "inert"}}],
        },
    )
    return memory


def test_inert_ledger_is_evidence_not_a_hard_stop(monkeypatch):
    """A ceiling concede with an INERT ledger surfaces the ledger + the weigh-toward-stop
    guidance in the prompt -- but does NOT force a stop: the LLM here declines and the run
    continues, proving there is no deterministic ledger->stop branch."""
    question = "Perform robust single-cell instance segmentation on this image."
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(analysis="one more in-scope try", stop_signal=False))
    task_verdict = _verify_task_complete(
        planner, question, "image.png", "", _ceiling_inert_memory(question))
    prompt = fake_text.calls[0]
    assert "ledger={" in prompt                       # the ledger is rendered as evidence
    assert "evidence to weigh, not a rule" in prompt  # the agentic guidance, not a command
    assert decide_stop(task_verdict) is False         # the LLM decided; ledger did NOT force stop


def test_inert_ledger_the_llm_may_weigh_it_and_stop(monkeypatch):
    """Same inert-ledger ceiling concede; the LLM weighs the evidence and STOPS -- via the
    single stop authority, driven by the evidence, not a gate."""
    question = "Perform robust single-cell instance segmentation on this image."
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(
            analysis="ceiling + inert ledger -> re-running reproduces it -> best-effort stop",
            stop_signal=True))
    task_verdict = _verify_task_complete(
        planner, question, "image.png", "", _ceiling_inert_memory(question))
    assert decide_stop(task_verdict) is True
    assert task_verdict.stop_reason == "task_complete"


# ---- (h) [045 P3] mechanistic RE-INVOCATION-progress evidence -- the planner-level
#          analog of the tool loop's plateau/oscillation stop. A stage re-invoked into
#          the same ceiling N times surfaces "re-invoking is futile" evidence; the LLM
#          still decides (NOT a 'max N tries' cap). ------------------------------------

def _n_ceiling_memory(question, n):
    memory = Memory()
    memory.set_query(question)
    for i in range(1, n + 1):
        memory.add_action(
            step_count=i, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
            command="Cell_Segmenter_Tool.execute(...)",
            result={"cell_count": 40, "status": "success"},
            verification={
                "status": "conceded",
                "stages": [{"stage": "segmentation", "decision": "FAIL", "action": "retune_here",
                            "status": "conceded", "fault": "ceiling", "notes": "faint cells missed",
                            "ledger": {"cellprob_threshold": "inert"}}],
            },
        )
    return memory


def test_reinvocation_plateau_surfaces_as_evidence(monkeypatch):
    question = "Perform robust single-cell instance segmentation on this image."
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(analysis="ceiling reproduced -> stop", stop_signal=True))
    task_verdict = _verify_task_complete(
        planner, question, "image.png", "", _n_ceiling_memory(question, 2))
    prompt = fake_text.calls[0]
    assert "RE-INVOCATION PROGRESS" in prompt
    assert "re-invoked 2x" in prompt
    assert "NOT a cap" in prompt                       # framed as evidence, not a threshold
    assert decide_stop(task_verdict) is True


def test_reinvocation_evidence_is_not_a_hard_cap(monkeypatch):
    """Same 2x-ceiling history, but the LLM judges a further try warranted -> continue.
    The re-invocation count NEVER force-stops (mechanistic evidence, agentic)."""
    question = "Perform robust single-cell instance segmentation on this image."
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(analysis="one more targeted try", stop_signal=False))
    task_verdict = _verify_task_complete(
        planner, question, "image.png", "", _n_ceiling_memory(question, 2))
    assert "RE-INVOCATION PROGRESS" in fake_text.calls[0]
    assert decide_stop(task_verdict) is False


def test_reinvocation_evidence_silent_on_single_invocation(monkeypatch):
    question = "Perform robust single-cell instance segmentation on this image."
    planner, fake_text, fake_mm = _make_planner(
        monkeypatch, MemoryVerification(analysis="continue", stop_signal=False))
    _verify_task_complete(planner, question, "image.png", "", _n_ceiling_memory(question, 1))
    assert "RE-INVOCATION PROGRESS" not in fake_text.calls[0]
