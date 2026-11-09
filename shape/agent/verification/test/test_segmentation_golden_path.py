"""Task 032 (VRR) whole-loop regression (a): the segmentation golden path is
behavior-identical to `main` -- a clean mask still accepts on the FIRST attempt
(status="satisfied"), driving the REAL `SegmentationVerifier` (deterministic-only,
no LLM engine) through the REAL `StepVerifier.resolve_stage`, no fakes/stubs in
between. Co-located with the code under test (committed -- AGENTS.md sec 6 P5).

The second part drives Cell_Segmenter_Tool's agentic VRR (`SegmentationAgenticVerifier`, a
`base.AgenticVerifier`) through StepVerifier's wave mode on a simulated segmenter (real mask /
overlay files, so `adapt_segmentation` and `mask_qc` run) with a scripted reasoner. Rules measure
and guard: the tool runs once at its defaults (the default call) and a measured problem of that call starts the
tuning (then the system's reference call with arm A's corrections that hold on the default call: the measured cell
scale, arm A's colour rule), and that call is the default delivery unless the agent's final choice has
strictly fewer measured problems; every other admission, every later call, the round sizes, the stop
and the final choice are the agent's. The target comes from the request's "## Target" section, else
the CARD's default convention.
"""

from __future__ import annotations

import io
import json
import re

import numpy as np
import pytest
from PIL import Image

from shape.agent.verification import base
from shape.agent.verification.image_context import ImageContextProfile
from shape.agent.verification.initial_params import diameter_from_mask_area
from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.verifiers import segmentation as seg
from shape.agent.verification.verifiers.segmentation import (
    CallAssessment, SegChoice, SegmentationAgenticVerifier, SegmentationVerifier, SegPlan, input_record,
)
from shape.tools.cell_segmenter import color_class_filter as ccf
from shape.utils.schemas import AdmissionAnswer, FinalChoice, LeverChange, VLRRResult, WaveMove


@pytest.fixture(autouse=True)
def historical_search_policy(monkeypatch):
    # These scripted searches exercise the previous full-search contract, including otherwise
    # unbeatable references. The default efficient policy has its own end-to-end regression tests.
    monkeypatch.setenv("SHAPE_SEGMENTATION_EARLY_STOP", "0")
    monkeypatch.setenv("SHAPE_SEGMENTATION_CONFIRM_STOP", "always")
    monkeypatch.setenv("SHAPE_SEGMENTATION_COMPACT_PROMPTS", "0")


def _clean_masks() -> np.ndarray:
    """5 same-sized, non-touching, non-border square blobs -- a well-formed
    segmentation with no tiny/large/border/solidity/under-detection complaint."""
    masks = np.zeros((200, 200), dtype=np.int32)
    for i in range(5):
        r0 = 20 + i * 30
        masks[r0:r0 + 10, 20:30] = i + 1
    return masks


def _adapter(result, upstream_ctx):
    masks, flow_error = result
    image = np.zeros(masks.shape, dtype=np.float32)
    context = {
        "upstream": {},
        "overlay_qc": {"overlay_is_black": False, "coverage_implausible": False},
        "coverage_qc": {},
        "overlay_path": None,
    }
    return (masks, image, flow_error), context


def test_clean_segmentation_accepts_on_first_attempt_status_satisfied():
    verifier = SegmentationVerifier()   # no llm_engine wired -> deterministic-only
    sv = StepVerifier(max_iters=3)

    def tool_runner(params):
        raise AssertionError("a clean golden-path mask must not need a retune/rerun")

    res = sv.resolve_stage(
        verifier, _adapter, tool_runner,
        initial_params={"diameter": 30.0, "flow_threshold": 0.4, "cellprob_threshold": 0.0,
                        "min_size": 15},
        initial_result=(_clean_masks(), 0.1))

    assert res.status == "satisfied"
    assert res.reason == "converged"
    assert res.fault is None
    assert res.verdict.decision == "PASS"
    assert res.verdict.action == "accept"
    assert len(res.trials) == 1
    assert res.verdict.metrics["cell_count"] == 5.0


def test_zero_detection_rescue_switches_beyond_threshold_sweep(monkeypatch):
    monkeypatch.setenv("SHAPE_ZERO_DETECTION_RESCUE", "1")
    verifier = SegmentationVerifier()
    masks = np.zeros((32, 32), dtype=np.int32)
    image = np.zeros_like(masks, dtype=np.float32)
    context = {
        "upstream": {},
        "overlay_qc": {"overlay_is_black": False, "coverage_implausible": False},
        "coverage_qc": {},
        "overlay_path": None,
        "parameters": {
            "cellprob_threshold": 0.0,
            "augment": False,
        },
    }

    vres = verifier.verify((masks, image, 0.1), context)

    assert vres.action == "retune_here"
    assert vres.param_delta == {"cellprob_threshold": -2.0}
    assert vres.expected_effect == {"cell_count": "up"}


def test_organoid_zero_detection_rescue_is_tool_default(monkeypatch):
    monkeypatch.delenv("SHAPE_ZERO_DETECTION_RESCUE", raising=False)
    verifier = SegmentationVerifier()
    verifier.tool_name = "Organoid_Segmenter_Tool"
    masks = np.zeros((32, 32), dtype=np.int32)
    image = np.zeros_like(masks, dtype=np.float32)
    context = {
        "upstream": {},
        "overlay_qc": {"overlay_is_black": False, "coverage_implausible": False},
        "coverage_qc": {},
        "overlay_path": None,
        "parameters": {
            "diameter": None,
            "cellprob_threshold": 0.0,
            "augment": False,
        },
    }

    vres = verifier.verify((masks, image, 0.1), context)

    assert vres.action == "retune_here"
    assert vres.param_delta == {"cellprob_threshold": -2.0}
    assert vres.expected_effect == {"cell_count": "up"}


def test_organoid_zero_detection_restores_native_scale_after_a_large_diameter(monkeypatch):
    # Cellpose resizes by 30 / diameter: a diameter set to the organoid's pixel width shrinks it away.
    monkeypatch.delenv("SHAPE_ZERO_DETECTION_RESCUE", raising=False)
    verifier = SegmentationVerifier()
    verifier.tool_name = "Organoid_Segmenter_Tool"
    masks = np.zeros((32, 32), dtype=np.int32)
    image = np.zeros_like(masks, dtype=np.float32)
    context = {
        "upstream": {},
        "overlay_qc": {"overlay_is_black": False, "coverage_implausible": False},
        "coverage_qc": {},
        "overlay_path": None,
        "parameters": {"diameter": 100.0, "cellprob_threshold": 0.0, "augment": False},
    }

    vres = verifier.verify((masks, image, 0.1), context)

    assert vres.action == "retune_here"
    assert vres.param_delta == {"diameter": None, "cellprob_threshold": -3.0}


def test_cell_segmenter_zero_detection_default_stays_single_threshold(monkeypatch):
    monkeypatch.delenv("SHAPE_ZERO_DETECTION_RESCUE", raising=False)
    verifier = SegmentationVerifier()
    verifier.tool_name = "Cell_Segmenter_Tool"
    masks = np.zeros((32, 32), dtype=np.int32)
    image = np.zeros_like(masks, dtype=np.float32)
    context = {
        "upstream": {},
        "overlay_qc": {"overlay_is_black": False, "coverage_implausible": False},
        "coverage_qc": {},
        "overlay_path": None,
        "parameters": {"cellprob_threshold": 0.0},
    }

    vres = verifier.verify((masks, image, 0.1), context)

    assert vres.action == "retune_here"
    assert vres.param_delta == {"cellprob_threshold": "down"}


def test_zero_detection_rescue_tests_an_enlarged_scale_after_thresholds(monkeypatch):
    monkeypatch.setenv("SHAPE_ZERO_DETECTION_RESCUE", "1")
    verifier = SegmentationVerifier()
    masks = np.zeros((32, 32), dtype=np.int32)
    image = np.zeros_like(masks, dtype=np.float32)
    context = {
        "upstream": {},
        "overlay_qc": {"overlay_is_black": False, "coverage_implausible": False},
        "coverage_qc": {},
        "overlay_path": None,
        "parameters": {
            "diameter": None,
            "cellprob_threshold": -4.0,
            "augment": False,
        },
    }

    vres = verifier.verify((masks, image, 0.1), context)

    assert vres.action == "retune_here"
    assert vres.param_delta == {"diameter": 15.0, "cellprob_threshold": -3.0}


def test_zero_detection_rescue_returns_to_native_after_enlarged_scale(monkeypatch):
    monkeypatch.setenv("SHAPE_ZERO_DETECTION_RESCUE", "1")
    verifier = SegmentationVerifier()
    masks = np.zeros((32, 32), dtype=np.int32)
    image = np.zeros_like(masks, dtype=np.float32)
    context = {
        "upstream": {},
        "overlay_qc": {"overlay_is_black": False, "coverage_implausible": False},
        "coverage_qc": {},
        "overlay_path": None,
        "parameters": {
            "diameter": 15.0,
            "cellprob_threshold": -3.0,
            "augment": False,
        },
    }

    vres = verifier.verify((masks, image, 0.1), context)

    assert vres.action == "retune_here"
    assert vres.param_delta == {"diameter": None, "cellprob_threshold": -3.0}


def test_zero_detection_rescue_does_not_oscillate_after_native_probe(monkeypatch):
    monkeypatch.setenv("SHAPE_ZERO_DETECTION_RESCUE", "1")
    verifier = SegmentationVerifier()
    masks = np.zeros((32, 32), dtype=np.int32)
    image = np.zeros_like(masks, dtype=np.float32)
    context = {
        "upstream": {},
        "overlay_qc": {"overlay_is_black": False, "coverage_implausible": False},
        "coverage_qc": {},
        "overlay_path": None,
        "parameters": {
            "diameter": None,
            "cellprob_threshold": -3.0,
            "augment": False,
        },
    }

    vres = verifier.verify((masks, image, 0.1), context)

    assert vres.action == "retune_here"
    assert vres.param_delta == {"diameter": 60.0, "cellprob_threshold": -3.0}


# --------------------------------------------------------------------------- #
# The agentic VRR (`SegmentationAgenticVerifier`) through StepVerifier's wave mode
# --------------------------------------------------------------------------- #
RETRY_SECONDS = base.REASONER_RETRY_SECONDS     # as imported (the fixture below patches it)


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)


@pytest.fixture()
def image(tmp_path):
    path = tmp_path / "cells.png"
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((64, 64)) * 200).astype(np.uint8)).save(path)
    return str(path)


def segmenter(tmp_path, calls, fixed_side=None, invalid=None, crash=None, flow_error=0.2):
    """Masks of 8-px cells when diameter <= 25, merged 16-px blobs above (`fixed_side`: cells of that side at
    any diameter); a black overlay in the overlay QC (the guard's invalid visual output) when flow_threshold >
    1.0 (`invalid(params)` instead, when given); `flow_error` for every call; a call for which `crash(params)`
    holds raises. Every call writes its mask and overlay and reports 0.5 s."""
    def run(params):
        p = {k: v for k, v in params.items() if not k.startswith("_")}
        calls.append(p)
        if crash is not None and crash(p):
            raise RuntimeError("the segmenter crashed")
        n = len(calls)
        side = fixed_side or (8 if float(p.get("diameter", 30)) <= 25 else 16)
        mask, k = np.zeros((64, 64), np.int32), 0
        for r in range(0, 64, side):
            for c in range(0, 64, side):
                k += 1
                mask[r:r + side - 1, c:c + side - 1] = k
        mask_path, overlay_path = tmp_path / f"mask_{n}.npy", tmp_path / f"overlay_{n}.png"
        np.save(mask_path, mask)
        Image.fromarray(((mask % 7) * 30).astype(np.uint8)).save(overlay_path)
        black = invalid(p) if invalid else float(p.get("flow_threshold", 0.4)) > 1.0
        return {"status": "success", "mask_path": str(mask_path), "overlay_path": str(overlay_path),
                "flow_error": flow_error, "overlay_qc": {"passed": True, "overlay_is_black": black}, "parameters": p,
                "cell_count": k, "seconds": 0.5}
    return run


def move(parent="input", expected=(), **changes):
    return WaveMove(parent=parent, changes=[LeverChange(lever=k, value=json.dumps(v)) for k, v in changes.items()],
                    diagnosis="d", expected=[{"metric": m, "direction": d} for m, d in expected])


# The admission is the linear loop's deterministic one: only a measured problem of the default call starts the
# tuning. On the toy images the default call has none, so a search that should tune gets this stand-in measured
# problem (`with_problem`).
TEST_PROBLEM = "under-detection (the test's measured problem)"


def with_problem(verifier, engine):
    """`engine.admission == "major"` (the Engine's default): when the default call has no measured problem (neither
    the guard nor its own admission problems), TEST_PROBLEM, so the tuning starts -- and the default call's measured
    problems in the calls table (`call_problems`) carry it too, as production's do (both are the default call's
    guard + measured problems); any other value (or no engine): none."""
    if engine is not None and getattr(engine, "admission", None) == "major":
        own, own_calls = verifier.admission_problems, verifier.call_problems
        verifier.admission_problems = lambda inp, vres: (own(inp, vres)
                                                         or ([] if verifier.guard(inp, vres) else [TEST_PROBLEM]))
        verifier.call_problems = lambda attempts, call: (own_calls(attempts, call)
                                                         or ([TEST_PROBLEM] if call == "default" else []))
    return verifier


class Engine:
    """`admission`: whether the search gets `TEST_PROBLEM` ("major", the default: the tuning starts; see
    `with_problem`) -- the loop asks no admission question. Answers each round with the next plan (then no
    moves), assessing every
    overlay attached for assessment with `severity[call]` (default minor); a repair prompt with `repair`; the
    final prompt with the next of `final`; the preflight read of the input image with `profile` (None: no
    answer -> the preflight is unavailable); arm A's colour question (`ColorClassVerdict`) with `color` (default:
    the model's "none", keep every mask; None: no answer). `contents` holds the loop's questions in order (`asked`:
    those of one schema), `preflights` the preflight's, `colors` the colour questions."""

    def __init__(self, rounds=(), final=(), severity=None, repair=(), profile="unknown", admission="major", color="none"):
        self.rounds, self.final, self.severity = list(rounds), list(final), dict(severity or {})
        self.repair = repair if isinstance(repair, dict) else list(repair)
        self.profile = ImageContextProfile() if profile == "unknown" else profile
        self.admission = admission
        self.color = ccf.ColorClassVerdict(scene="s", is_mixed_class=False, action="none", confidence="high",
                                           reason="one class") if color == "none" else color
        self.contents, self.schemas, self.preflights, self.colors = [], [], [], []

    def asked(self, schema) -> list:
        return [c for c, s in zip(self.contents, self.schemas) if s is schema]

    def prompts(self) -> str:
        """Every text the loop sent (each question's prompt and its attachment labels)."""
        return "\n".join(item for c in self.contents for item in c if isinstance(item, str))

    def generate(self, content, response_format=None):
        if response_format is ImageContextProfile:
            self.preflights.append(content)
            return {"content": self.profile}
        if response_format is ccf.ColorClassVerdict:
            self.colors.append(content)
            return {"content": self.color}
        self.contents.append(content)
        self.schemas.append(response_format)
        if response_format is AdmissionAnswer:
            if self.admission is None:
                raise TimeoutError("gateway down")
            problem = "none" if self.admission == "none" else "masks miss half the cells"
            return {"content": AdmissionAnswer(problem=problem, severity=self.admission, reasoning="t")}
        if response_format is SegChoice:
            todo = re.search(r"severity none \| minor \| major, confidence\): (\[[^\]]*\])", content[0])
            listed = json.loads(todo.group(1).replace("'", '"')) if todo else []
            return {"content": SegChoice(call=self.final.pop(0) if self.final else "default", reasoning="t",
                                         assessments=[CallAssessment(call=c, problem="p",
                                                                     severity=self.severity.get(c, "minor"))
                                                      for c in listed])}
        shown = [s.split()[1] for s in content if isinstance(s, str) and s.startswith("OVERLAY") and "(to assess)" in s]
        assessments = [CallAssessment(call=c, problem="p", severity=self.severity.get(c, "minor")) for c in shown]
        if "were rejected" in content[0]:                         # `repair`: moves, or {"moves", "focus"}
            moves, focus = ((self.repair["moves"], self.repair["focus"]) if isinstance(self.repair, dict)
                            else (self.repair, []))
            return {"content": SegPlan(analysis="repair", focus_levers=focus, moves=moves)}
        step = self.rounds.pop(0) if self.rounds else []          # a round's moves, or {"moves", "focus"}
        moves, focus = (step["moves"], step["focus"]) if isinstance(step, dict) else (step, [])
        return {"content": SegPlan(analysis="t", assessments=assessments, focus_levers=focus, moves=moves)}


def no_schedule(verifier, schedule: bool):
    """`verifier` without the narrowing (its focus never widened) unless `schedule` (the tests of the other
    parts)."""
    verifier.narrowing = schedule
    return verifier


def run(image, tmp_path, engine=None, budget=8, cap=1, monkeypatch=None, verifier=None, schedule=False, **tool):
    """One agentic search of `image` (`verifier`, default a fresh one) on the simulated segmenter (`tool`: its
    options); without the schedule unless `schedule`."""
    calls = []
    verifier = no_schedule(with_problem(verifier or SegmentationAgenticVerifier(), engine), schedule)
    verifier.llm_engine = engine
    if monkeypatch is not None:
        monkeypatch.setattr(seg, "sam_concurrency", lambda: cap)
    record = input_record({"image": image, "image_id": "cells"})
    resolution = StepVerifier(max_iters=1 + budget).resolve_stage(
        verifier, verifier.make_adapter({"image": image}), segmenter(tmp_path, calls, **tool), {}, record)
    return resolution, verifier, calls


def test_agentic_trial_0_is_the_input_default_the_defaults_and_every_later_call_the_agents(image, tmp_path,
                                                                                          monkeypatch):
    engine = Engine([[move(diameter=20.0)], [move("agent_1", cellprob_threshold=-1.0, min_size=20)]],
                    final=["agent_2"])
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch)
    defaults = verifier.defaults()                                           # the CARD lever Defaults
    assert resolution.trials[0][0] == {} and resolution.trials[0][2]["status"] == "INPUT"
    assert "sharpness" in resolution.trials[0][1].metrics                    # the image QC
    assert len(calls) == 3 and calls[0] == defaults                          # default: the admission call
    assert resolution.trials[1][1].notes == "the tool at its defaults (the admission call)"
    assert calls[1] == {**defaults, "diameter": 20.0}
    assert calls[2] == {**defaults, "diameter": 20.0, "cellprob_threshold": -1.0, "min_size": 20}
    assert resolution.trials[3][1].param_delta == {"cellprob_threshold": -1.0, "min_size": 20}
    # no reference call: the default call is the default, and agent_2 has no fewer measured problems -> the default
    # call is delivered
    assert resolution.parents == [None, 0, 0, 2] and resolution.params == calls[0]
    assert resolution.status == "satisfied" and resolution.reason == "satisfied"
    assert resolution.trials[1][1].judge_severity == "minor"                 # assessed in round 1
    assert resolution.trials[2][1].judge_severity == "minor"                 # the agent's assessment


def test_four_stage_segmentation_keeps_assessments_and_reference_delivery(image, tmp_path, monkeypatch):
    from shape.agent.verification.test.test_four_stage import CoveringEngine

    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY_SEGMENTATION", "four_stage")
    inner = Engine([[move(diameter=20.0)]], final=["agent_1"])
    resolution, verifier, calls = run(image, tmp_path, CoveringEngine(inner), monkeypatch=monkeypatch)
    assert len(calls) == 2 and resolution.details["admission"]["decision"] == "RUN"
    assert verifier.assessments
    assert resolution.details["call"] == "default"        # no weakening of the reference override rule
    assert resolution.details["search_strategy"]["confirmation"]["call"] == "default"


def test_agentic_admission_skip_delivers_the_call_at_the_defaults_without_a_question(image, tmp_path,
                                                                                    monkeypatch):
    """No measured problem of the default call (the linear loop's deterministic admission): SKIP, the default call
    delivered as it is, and the agent is asked nothing -- no admission question, no round, no final question."""
    engine = Engine([[move(diameter=20.0)]], admission="none")
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch)
    assert calls == [verifier.defaults()] and engine.contents == []
    assert resolution.status == "satisfied" and resolution.reason == "no_tuning_needed"
    assert resolution.verdict.action == "accept" and resolution.result is resolution.trials[1][2]
    assert resolution.params == verifier.defaults() and resolution.details["call"] == "default"
    assert resolution.details["choice"] == resolution.details["verifier_stop"] == "no_tuning_needed"
    admission = resolution.details["admission"]
    assert admission["rule"] == admission["decision"] == "SKIP" and admission["problems"] == []
    assert admission["answer"] is None and verifier.assessments == {} and resolution.trials[1][1].judge_severity is None
    assert resolution.details["color_evidence"] is None and verifier.reference_calls == {}
    other = tmp_path / "no_reasoner"                                  # without a reasoner: the same SKIP
    other.mkdir()
    resolution, verifier, calls = run(image, other, None, monkeypatch=monkeypatch)
    assert resolution.reason == "no_tuning_needed" and calls == [verifier.defaults()]
    assert verifier.reasoner_failures == 0


def test_agentic_admission_a_measured_problem_runs_and_the_default_is_assessed_in_round_1(image, tmp_path,
                                                                                         monkeypatch):
    engine = Engine([[move("default", min_size=20)]], final=["agent_1"],
                    severity={"default": "major"})                            # TEST_PROBLEM on the default call
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch)
    admission = resolution.details["admission"]
    assert admission["rule"] == admission["decision"] == "RUN" and admission["problems"] == [TEST_PROBLEM]
    assert admission["answer"] is None and engine.asked(AdmissionAnswer) == [] and len(calls) == 2
    first = engine.asked(SegPlan)[0]
    assert ("Task: the instance segmentation this image's analysis needs, as accurate as possible.\n"
            f"Target (default convention): {DEFAULT_TARGET}\n"
            f"Admission: the tool ran once at its defaults (default); the tuning started on {TEST_PROBLEM}.\n"
            "Tool defaults" in first[0] and "\nRounds: 1 done, at most 7 more" in first[0])
    # the default call's overlay is attached to assess in round 1, and the agent's assessment is recorded
    assert "OVERLAY default (to assess):" in first and verifier.plans[0]["shown"] == ["default"]
    assert verifier.assessments["default"]["severity"] == "major" and resolution.trials[1][1].judge_severity == "major"
    assert verifier.plans[1]["shown"] == ["agent_1"]
    # no reference call: the default call is the default, named so from the first round prompt on
    assert "Reference default: default (the call at the defaults) is delivered unless" in first[0]
    # agent_1 has no measured problem, the default call has TEST_PROBLEM: strictly fewer -> the override holds,
    # agent_1 delivered
    problems = resolution.details["measured_problems"]
    assert problems == {"default": [TEST_PROBLEM], "agent_1": []} and len(engine.asked(SegChoice)) == 1
    assert resolution.reason == "satisfied" and resolution.details["call"] == "agent_1"
    assert resolution.details["choice"] == "agent_choice" and resolution.details["reference_default"] == {
        "call": "default", "overridden": True,
        "reason": f"agent_1 has 0 measured problem(s) (none) and default has 1 ({TEST_PROBLEM}): strictly fewer; "
                  "your assessment of agent_1 is minor and of default major: better"}
    assert verifier.reference_calls == {} and "Then the system ran" not in first[0]


def test_agentic_to_assess_includes_the_call_at_the_defaults():
    verifier = SegmentationAgenticVerifier()
    verifier.rounds = [[1]]
    verifier._names = {0: "input", 1: "default"}                     # trial 1 named as the loop names it
    assert verifier._to_assess() == ["default"]
    verifier.failed = [{"call": "default"}]
    assert verifier._to_assess() == []


class Measured(SegmentationAgenticVerifier):
    """The call at the defaults (trial 1) with the readouts of `readouts` set, the others measured."""
    readouts: dict = {}

    def quantify(self, tool_output, context):
        metrics = super().quantify(tool_output, context)
        return {**metrics, **self.readouts} if context.get("trial_index") == 1 else metrics


def measured(readouts: dict) -> Measured:
    verifier = Measured()
    verifier.readouts = dict(readouts)
    return verifier


SCALE = "x the diameter used (30), off by a factor of 2 or more"
DEFAULT_TARGET = ("Whole cells: each cell is one instance, over its full extent. In a field where a few stained /\n"
                  "nucleated cells lie among many unstained or red cells, the stained cells. For a nuclear stain\n"
                  "only, the nuclei.")                                          # the CARD's default convention


def reference(diameter: float) -> str:
    """The diagnosis of the system's reference call holding arm A's diameter correction only (the agent reads it as
    the system's rule)."""
    return f"the system's diameter rule (mask_diameter_px {diameter:.1f})"


@pytest.mark.parametrize("readouts, problems", [
    ({"tiny_mask_fraction": 0.2}, ["tiny_mask_fraction 0.200 above 0.15"]),        # the shape gates
    ({"large_mask_fraction": 0.2}, ["large_mask_fraction 0.200 above 0.1"]),
    ({"mean_solidity": 0.3}, ["mean_solidity 0.300 below 0.35"]),
    ({"tiny_mask_fraction": 0.2, "missed_foreground_fraction": 0.12},           # under-detection: information only
     ["tiny_mask_fraction 0.200 above 0.15"]),
    ({"mask_diameter_px": 60.0}, [f"mask_diameter_px 60.00 is 2.00 {SCALE}"]),     # arm A's scale check, at
    ({"mask_diameter_px": 15.0}, [f"mask_diameter_px 15.00 is 0.50 {SCALE}"]),     # its bounds (>= 2, <= 0.5)
    ({"overlay_is_black": 1.0}, ["visual output invalid (black overlay or implausible coverage)"]),   # the guard's
    ({"cell_count": 0.0}, ["no cells detected"]),
    ({"flow_error": 0.9, "tiny_mask_fraction": 0.2}, ["tiny_mask_fraction 0.200 above 0.15"]),   # no flow gate
])
def test_agentic_admission_a_measured_problem_of_the_defaults_starts_the_tuning_without_asking(
        image, tmp_path, monkeypatch, readouts, problems):
    scale = "mask_diameter_px" in readouts
    # the second call: the system's reference call on the scale check, else the agent's
    engine = Engine([[move("default", min_size=20)]], final=["reference" if scale else "agent_1"])
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, verifier=measured(readouts))
    admission = resolution.details["admission"]
    assert engine.asked(AdmissionAnswer) == [] and admission["rule"] == admission["decision"] == "RUN"
    assert admission["problems"] == problems and admission["answer"] is None
    # the scale check: the system's reference call (the default call at the measured diameter) before the agent's
    # call
    assert list(verifier.reference_calls) == (["reference"] if scale else []) and len(calls) == 2 + scale
    if scale:
        assert calls[1] == {**verifier.defaults(), "diameter": readouts["mask_diameter_px"]}
    then = f" Then the system ran: reference ({reference(readouts['mask_diameter_px'])})." if scale else ""
    first = engine.asked(SegPlan)[0]
    assert (f"Admission: the tool ran once at its defaults (default); the tuning started on {'; '.join(problems)}."
            f"{then}\n" in first[0])
    assert "OVERLAY default (to assess):" in first and verifier.assessments["default"]["severity"] == "minor"


@pytest.mark.parametrize("readouts", [
    {"missed_foreground_fraction": 0.12, "border_mask_fraction": 0.0},     # under-detection is no problem, with
    {"count_miss_fraction": 0.2, "border_mask_fraction": 0.0},             # any qc_score (1 here, 0.75 below):
    {"missed_foreground_fraction": 0.12}, {"count_miss_fraction": 0.2},    # it is shown as information
    {"mask_diameter_px": 59.0}, {"mask_diameter_px": 15.5},                # within a factor of 2 of 30
    {"flow_error": 0.9},                                                   # flow_error is no admission check
])
def test_agentic_admission_the_linear_loops_accept_and_a_scale_within_the_factor_deliver_the_default(
        image, tmp_path, monkeypatch, readouts):
    engine = Engine(admission="minor")
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, verifier=measured(readouts))
    assert engine.contents == [] and resolution.details["admission"]["problems"] == []      # nothing asked
    assert resolution.reason == "no_tuning_needed" and calls == [verifier.defaults()]


def test_agentic_admission_small_cells_at_the_defaults_are_the_scale_problem(image, tmp_path, monkeypatch):
    # 8-px cells at diameter 30, measured: the scale is off; the uncovered texture reads as under-detection,
    # alone and with qc_score >= 0.8, so it is no problem (the linear loop accepts it)
    engine = Engine([[move("default", diameter=8.0)]], final=["agent_1"])
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, fixed_side=8)
    default = resolution.trials[1][1]
    assert default.metrics["missed_foreground_fraction"] > verifier._linear.miss_gate
    assert default.scalar_score >= verifier._linear.accept_score
    diameter = default.metrics["mask_diameter_px"]
    assert resolution.details["admission"]["problems"] == [
        f"mask_diameter_px {diameter:.2f} is {diameter / 30:.2f} {SCALE}"]
    assert engine.asked(AdmissionAnswer) == [] and calls[1]["diameter"] == diameter      # the reference call
    assert calls[2]["diameter"] == 8.0                                                    # the agent's


def test_agentic_reference_call_runs_after_a_run_admission_when_the_scale_check_holds(image, tmp_path, monkeypatch):
    # 12-px cells at diameter 30: the scale check holds on the default call -> the system runs it at its measured
    # diameter (reference), one round of the budget, before the agent's first round; the agent reads it like any
    # call. It is the default delivery: the agent's agent_1 has no fewer measured problems, so its choice is answered
    # and asked again
    engine = Engine([[move("reference", min_size=20)]], final=["agent_1", "reference"])
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, fixed_side=12)
    measured = resolution.trials[1][1].metrics["mask_diameter_px"]
    assert calls == [verifier.defaults(), {**verifier.defaults(), "diameter": measured},
                     {**verifier.defaults(), "diameter": measured, "min_size": 20}]
    assert list(verifier.reference_calls) == ["reference"] and resolution.parents[:3] == [None, 0, 1]   # from default
    assert resolution.trials[2][1].notes == reference(measured)
    assert resolution.trials[2][1].param_delta == {"diameter": measured}
    assert resolution.details["rounds"] == [["default"], ["reference"], ["agent_1"]] and len(engine.asked(SegPlan)) == 3   # + the stop check
    first = engine.asked(SegPlan)[0]
    assert (f"the tuning started on mask_diameter_px {measured:.2f} is {measured / 30:.2f} {SCALE}. Then the system "
            f"ran: reference ({reference(measured)}).\nTool defaults" in first[0]
            and "\nRounds: 2 done, at most 6 more" in first[0])
    assert f'"diagnosis": "{reference(measured)}"' in first[0]                     # its row in the calls table
    # no question has shown the default call (the admission measured its problem): both overlays are assessed now
    assert "OVERLAY default (to assess):" in first and "OVERLAY reference (to assess):" in first
    assert verifier.plans[0]["shown"] == ["default", "reference"]
    assert set(verifier.assessments) == {"default", "reference", "agent_1"}
    assert resolution.details["call"] == "reference" and resolution.details["choice"] == "agent_choice"
    assert resolution.reason == "satisfied" and resolution.details["reference_default"] == {
        "call": "reference", "overridden": False,
        "reason": "agent_1 has 0 measured problem(s) (none) and reference has 0 (none): not strictly fewer; your "
                  "assessment of agent_1 is minor and of reference minor: not better"}


class PerTrial(SegmentationAgenticVerifier):
    """Trial i with the readouts of `readouts[i]` set, the others measured."""
    readouts: dict = {}

    def quantify(self, tool_output, context):
        return {**super().quantify(tool_output, context), **self.readouts.get(context.get("trial_index"), {})}


RULE = ("Reference default: reference (the system's reference call) is delivered unless you choose a call that\n"
        "has strictly fewer measured_problems than reference (the calls table) and your assessment of it\nis better "
        "(a lower severity) than your assessment of reference; a choice that does not is answered with the reason and "
        "asked once more; if it\nstill does not, reference is delivered.")


def test_agentic_an_override_of_the_reference_default_needs_fewer_problems_and_a_better_assessment(image, tmp_path,
                                                                                                 monkeypatch):
    # 12-px cells at diameter 30: reference is the reference call (the measured diameter); the agent's agent_1 from it
    engine = Engine([[move("reference", min_size=20)]], final=["agent_1", "agent_1"])
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, fixed_side=12)
    measured = resolution.trials[1][1].metrics["mask_diameter_px"]
    # the rule is stated in the round prompt and the final prompt; the calls table shows each call's problems
    first, final = engine.asked(SegPlan)[0][0], engine.asked(SegChoice)[0][0]
    assert RULE in first and RULE in final
    assert "the reference default\nreference is delivered." in first             # a choice still not qualifying
    assert f'"measured_problems": ["mask_diameter_px {measured:.2f} is {measured / 30:.2f} {SCALE}"]' in first
    assert '"call": "reference"' in first and '"measured_problems": []' in first
    assert "measured_problems = its guard failures and the admission's checks on the call" in first
    assert "arm A" not in "\n".join(c[0] for c in engine.contents)         # the system's rules, by that name
    # agent_1 has no fewer problems (0 and 0): answered with both calls' problems and assessments, asked again;
    # still agent_1 -> reference
    why = ("agent_1 has 0 measured problem(s) (none) and reference has 0 (none): not strictly fewer; your assessment "
           "of agent_1 is minor and of reference minor: not better")
    assert (f"Your choice 'agent_1' does not override the reference default reference: {why}. Choose reference, or a "
            "call that has strictly fewer measured_problems than reference (the calls table) and your assessment of "
            "it\nis better (a lower severity) than your assessment of reference." in engine.asked(SegChoice)[1][0])
    assert resolution.details["call"] == "reference" and resolution.details["choice"] == "reference_default"
    assert resolution.status == "satisfied" and resolution.result is resolution.trials[2][2]
    assert resolution.details["reference_default"] == {"call": "reference", "overridden": False, "reason": why}
    assert resolution.details["measured_problems"]["reference"] == [] and calls[2]["min_size"] == 20
    # reference with a measured problem agent_1 does not have, but assessed alike: still no override
    verifier = PerTrial()
    verifier.readouts = {2: {"tiny_mask_fraction": 0.2}}
    other = tmp_path / "alike"
    other.mkdir()
    engine = Engine([[move("reference", min_size=20)]], final=["agent_1", "agent_1"])
    resolution, verifier, _ = run(image, other, engine, monkeypatch=monkeypatch, verifier=verifier, fixed_side=12)
    assert resolution.details["call"] == "reference" and resolution.details["reference_default"]["reason"].endswith(
        "strictly fewer; your assessment of agent_1 is minor and of reference minor: not better")
    # ... and with reference assessed major, agent_1 minor: the override holds
    verifier = PerTrial()
    verifier.readouts = {2: {"tiny_mask_fraction": 0.2}}
    other = tmp_path / "allowed"
    other.mkdir()
    engine = Engine([[move("reference", min_size=20)]], final=["agent_1"], severity={"reference": "major"})
    resolution, verifier, _ = run(image, other, engine, monkeypatch=monkeypatch, verifier=verifier, fixed_side=12)
    assert resolution.details["call"] == "agent_1" and resolution.details["choice"] == "agent_choice"
    assert len(engine.asked(SegChoice)) == 1 and resolution.details["reference_default"] == {
        "call": "reference", "overridden": True,
        "reason": "agent_1 has 0 measured problem(s) (none) and reference has 1 (tiny_mask_fraction 0.200 above 0.15): "
                  "strictly fewer; your assessment of agent_1 is minor and of reference major: better"}


def test_agentic_the_final_question_never_replaces_an_assessment_made_in_a_round(image, tmp_path, monkeypatch):
    """The final question's assessments count only for calls no round has assessed: an agent re-assessing its own
    choice as better at the final question cannot make it override the reference default."""
    class Reassessing(Engine):
        def generate(self, content, response_format=None):
            out = super().generate(content, response_format)
            if response_format is SegChoice:                          # its choice re-assessed as flawless
                out["content"].assessments = [CallAssessment(call="agent_1", problem="none", severity="none")]
            return out

    verifier = PerTrial()
    verifier.readouts = {2: {"tiny_mask_fraction": 0.2}}           # agent_1 has fewer measured problems than reference
    engine = Reassessing([[move("reference", min_size=20)]], final=["agent_1", "agent_1"],
                         severity={"reference": "major", "agent_1": "major"})
    resolution, verifier, _ = run(image, tmp_path, engine, monkeypatch=monkeypatch, verifier=verifier, fixed_side=12)
    assert verifier.assessments["agent_1"]["severity"] == "major"
    assert verifier.assessments["reference"]["severity"] == "major"
    assert resolution.details["call"] == "reference" and resolution.details["choice"] == "reference_default"
    assert resolution.details["reference_default"]["reason"].endswith(
        "your assessment of agent_1 is major and of reference major: not better")


def test_agentic_the_agent_paces_the_search_and_its_focus_never_widens(image, tmp_path, monkeypatch):
    """The narrowing (`narrowing`): the agent explores as it likes (several levers in one call), names a focus when it
    chooses -- that round's moves change only its focus_levers -- and keeps or narrows it in every later plan; a plan
    that widens it is rejected with the reason and repaired; the prompts state the rule and where the search
    stands; the last call is assessed in the final question."""
    assert SegmentationAgenticVerifier.narrowing is True
    engine = Engine([[move("reference", min_size=20, cellprob_threshold=-1.0, flow_threshold=0.6)],
                     {"moves": [move("agent_1", flow_threshold=0.8, min_size=25)],
                      "focus": ["cellprob_threshold", "flow_threshold", "min_size"]},
                     {"moves": [move("agent_2", augment=True)], "focus": ["min_size", "augment"]},  # widens: rejected
                     {"moves": [move("agent_3", min_size=35)], "focus": ["min_size"]}],
                    repair={"moves": [move("agent_2", min_size=30)], "focus": ["min_size"]}, final=["reference"],
                    severity={"default": "major", "agent_1": "none"})      # progress: no plateau before agent_5
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, schedule=True, fixed_side=12)
    assert resolution.details["rounds"] == [["default"], ["reference"], ["agent_1"], ["agent_2"], ["agent_3"],
                                            ["agent_4"]]
    assert calls[2]["min_size"] == 20 and calls[2]["cellprob_threshold"] == -1.0 and calls[2]["flow_threshold"] == 0.6
    assert calls[3]["flow_threshold"] == 0.8 and calls[3]["min_size"] == 25                  # agent_2: a combination
    assert calls[4]["min_size"] == 30 and calls[5]["min_size"] == 35                         # agent_3: the repair's
    assert verifier.focus_history == [["cellprob_threshold", "flow_threshold", "min_size"], ["min_size"], ["min_size"]]
    plans = [c[0] for c in engine.asked(SegPlan)]
    assert all("The search goes broad, then narrow; its pace is yours." in p for p in plans)   # the shared narrowing
    assert "Now: exploring -- no focus named yet." in plans[0] and "Now: exploring" in plans[1]
    assert "Now: focused on ['cellprob_threshold', 'flow_threshold', 'min_size']" in plans[2]
    assert "Now: focused on ['min_size']" in plans[4]
    repair = next(p for p in plans if "were rejected" in p)
    assert ("focus_levers ['min_size', 'augment'] widen your previous focus ['cellprob_threshold', 'flow_threshold', "
            "'min_size']: keep it or narrow it") in repair
    assert resolution.details["focus_history"] == verifier.focus_history
    final = engine.asked(SegChoice)[0][0]
    assert "Then\nchoose." in final and resolution.details["call"] == "reference"


def test_agentic_the_narrowing_rejects_only_a_widened_or_missing_focus():
    verifier = SegmentationAgenticVerifier()
    verifier.rounds = [[1], [2]]

    def problem(parent="agent_1", **changes):
        return verifier._schedule_problem(move(parent, **changes).model_dump(), {})   # no CARD Family: no coverage

    # no focus yet: any move, any number of levers, from any call
    verifier._pending_focus = []
    assert problem(min_size=30, augment=True, flow_threshold=0.8) is None and problem("default", niter=400) is None
    # the agent's first focus is its own choice, of any size; its moves change only those levers
    verifier._pending_focus = ["min_size", "augment", "flow_threshold", "cellprob_threshold"]
    assert problem(min_size=30, augment=True) is None
    assert problem(niter=400) == ("['niter'] not in this round's focus_levers ['min_size', 'augment', "
                                  "'flow_threshold', 'cellprob_threshold']")
    verifier._pending_focus = ["no_such_lever"]
    assert problem(min_size=30) == "focus_levers ['no_such_lever'] are not levers"
    # after a round under the focus ["min_size", "augment"]: the same or fewer, named again
    verifier.focus_history, verifier._focus_round = [["min_size", "augment"]], len(verifier.rounds)
    verifier.rounds += [[3]]
    verifier._pending_focus = []
    assert problem(min_size=30) == "name focus_levers: your previous focus ['min_size', 'augment'], or fewer of them"
    verifier._pending_focus = ["min_size", "niter"]
    assert problem(min_size=30) == ("focus_levers ['min_size', 'niter'] widen your previous focus ['min_size', "
                                    "'augment']: keep it or narrow it")
    verifier._pending_focus = ["min_size", "augment"]
    assert problem(min_size=30, augment=True) is None
    verifier._pending_focus = ["min_size"]
    assert problem(min_size=40) is None


def test_agentic_without_the_schedule_the_prompts_state_none():
    verifier = no_schedule(SegmentationAgenticVerifier(), False)
    assert verifier.schedule_text() == "" and "Schedule" not in verifier.move_text()


def test_agentic_images_are_attached_and_prompts_carry_information_only(image, tmp_path, monkeypatch):
    engine = Engine([[move(diameter=20.0)], [move("agent_1", min_size=20)]], final=["agent_2"])
    run(image, tmp_path, engine, monkeypatch=monkeypatch)
    assert engine.asked(AdmissionAnswer) == []                                     # no admission question
    first, second = engine.asked(SegPlan)[:2]
    for label in ("INPUT IMAGE:", "OVERLAY default (to assess):"):                 # default assessed in round 1
        assert isinstance(first[first.index(label) + 1], bytes), label
    assert "OVERLAY agent_1 (to assess):" in second
    assert "seconds of each finished call: {'default': 0.5, 'agent_1': 0.5}" in second[0]    # the tool's report
    final = engine.contents[-1]
    assert final[0].startswith("Choose the delivered segmentation") and "OVERLAY agent_2:" in final
    text = "\n".join(c[0] for c in engine.contents)
    for rule in ("SINGLE", "choose ONE", "screen", "at least one untried", "ap50", "_label"):
        assert rule not in text, rule
    assert "more; calls so far" in first[0] and "### diameter" in first[0]
    assert "asked once more" in first[0] and "Reference default: default (the call at the defaults)" in first[0]


def test_agentic_reference_overlay_is_the_best_earlier_call_the_guard_admits(image, tmp_path, monkeypatch):
    # default (the defaults, an invalid overlay at diameter 30) fails the guard but is assessed best; agent_1, agent_2
    # admissible
    engine = Engine([[move(diameter=20.0)], [move("agent_1", min_size=20)]],
                    severity={"default": "none", "agent_1": "minor", "agent_2": "minor"})
    run(image, tmp_path, engine, monkeypatch=monkeypatch, invalid=lambda p: p["diameter"] > 25)
    assert engine.asked(AdmissionAnswer) == []                   # the guard's problem started the tuning
    first, second, third, *_ = engine.asked(SegPlan)                       # (then the stop check)
    assert "OVERLAY default (to assess):" in first and not any("for comparison" in str(c) for c in first)
    assert "OVERLAY default (your best call so far, not admissible, for comparison):" in second     # none admissible
    assert "OVERLAY agent_1 (your best admissible call so far, for comparison):" in third
    assert not any(isinstance(c, str) and c.startswith("OVERLAY default") for c in third)


SMEAR_PROFILE = ImageContextProfile(modality="brightfield", specimen_context="blood_smear",
                                    visible_object_classes=["red blood cells", "white blood cells"],
                                    provisional_target_hint="nucleated_cells", confidence="high")


def smear(tmp_path, n_purple=3, side=12):
    """A stained-smear-like RGB field: 36 objects of `side` px in a 6 x 6 grid, `n_purple` purple (bluer) and the
    rest pink (redder), and the label mask of all 36."""
    pitch = side + 4
    rgb = np.full((6 * pitch, 6 * pitch, 3), 235, np.uint8)
    labels, k = np.zeros((6 * pitch, 6 * pitch), np.int32), 0
    for r in range(6):
        for c in range(6):
            k += 1
            box = (slice(r * pitch + 2, r * pitch + 2 + side), slice(c * pitch + 2, c * pitch + 2 + side))
            labels[box] = k
            rgb[box] = (120, 40, 170) if k <= n_purple else (225, 120, 140)
    path = tmp_path / "smear.png"
    Image.fromarray(rgb).save(path)
    return str(path), labels


def smear_segmenter(tmp_path, labels, rgb_path, calls):
    """Every object masked at any configuration; color_class_filter applied as the tool applies it, the mask before
    the filter recorded (`prefilter_mask_path`) when the filter acted."""
    from shape.tools.cell_segmenter import color_class_filter as ccf

    rgb = np.asarray(Image.open(rgb_path))

    def run(params):
        p = {k: v for k, v in params.items() if not k.startswith("_")}
        calls.append(p)
        mask = ccf.apply(rgb, labels.copy(), p["color_class_filter"]) if p.get("color_class_filter") else labels.copy()
        mask_path, overlay_path = tmp_path / f"smear_mask_{len(calls)}.npy", tmp_path / f"smear_overlay_{len(calls)}.png"
        np.save(mask_path, mask)
        Image.fromarray(((mask % 7) * 30).astype(np.uint8)).save(overlay_path)
        result = {"status": "success", "mask_path": str(mask_path), "overlay_path": str(overlay_path),
                  "flow_error": 0.2, "overlay_qc": {"passed": True}, "parameters": p,
                  "cell_count": int(len(np.unique(mask)) - 1)}
        if not np.array_equal(mask, labels):                                    # the filter acted
            result["prefilter_mask_path"] = str(tmp_path / f"smear_prefilter_mask_{len(calls)}.npy")
            np.save(result["prefilter_mask_path"], labels)
        return result
    return run


def smear_search(tmp_path, monkeypatch, engine, verifier=None, field=None, schedule=False, **fixture):
    """One agentic search (`verifier`, default a fresh one) of a smear (`field(**fixture)`, default `smear`) on
    `smear_segmenter`: (resolution, verifier, calls, the raw image, the labels)."""
    path, labels = (field or smear)(tmp_path, **fixture)
    monkeypatch.setattr(seg, "sam_concurrency", lambda: 1)
    verifier, calls = no_schedule(with_problem(verifier or SegmentationAgenticVerifier(), engine), schedule), []
    verifier.llm_engine = engine
    resolution = StepVerifier(max_iters=9).resolve_stage(
        verifier, verifier.make_adapter({"image": path}), smear_segmenter(tmp_path, labels, path, calls), {},
        input_record({"image": path, "image_id": "smear"}))
    return resolution, verifier, calls, np.asarray(Image.open(path)), labels


def test_agentic_prompt_carries_the_preflight_read_and_the_color_populations(tmp_path, monkeypatch):
    engine = Engine([[move("default", color_class_filter="keep_high_br")]], final=["agent_1"], profile=SMEAR_PROFILE)
    resolution, verifier, calls, _, labels = smear_search(tmp_path, monkeypatch, engine)
    path = resolution.trials[0][2]["image"]
    first, second = (c[0] for c in engine.asked(SegPlan)[:2])
    assert ('"preflight": {"specimen_context": "blood_smear", "visible_object_classes": ["red blood cells", '
            '"white blood cells"], "provisional_target_hint": "nucleated_cells", "confidence": "high"}') in first
    assert "the instance segmentation this image's analysis needs" in first and "every cell" not in first
    assert ('"color_populations": {"objects": 36, "cut_b_minus_r": ' in first
            and '"high": {"masks": 3, "fraction": 0.0833' in first and '"low": {"masks": 33, "fraction": 0.9167' in first)
    assert '"mask_area_median": 144.0, "mask_diameter_px": 13.5406' in first        # 12 x 12 px objects
    assert resolution.trials[1][1].metrics["mask_diameter_px"] == diameter_from_mask_area(144.0)   # arm A's measure
    # default, the defaults: the objects measure 13.54 px at diameter 30 (arm A's scale check) and 3 of the 36 are
    # purple (arm A's colour gates) -> no admission question; the reference call at the measured diameter
    assert engine.asked(AdmissionAnswer) == [] and resolution.details["admission"]["problems"] == [
        "mask_diameter_px 13.54 is 0.45 x the diameter used (30), off by a factor of 2 or more",
        "colour minority: 3 of 36 masks (8.3%) in the high (bluer) population"]
    assert calls[1] == {**verifier.defaults(), "diameter": diameter_from_mask_area(144.0)}
    assert calls[2]["color_class_filter"] == "keep_high_br" and '"cell_count": 3.0' in second   # the agent's filter
    assert resolution.details["preflight"]["specimen_context"] == "blood_smear"
    assert resolution.details["color_populations"]["default"]["high"]["masks"] == 3
    again = Engine([[]], profile=SMEAR_PROFILE)            # a second search of the same image: no second read
    other = SegmentationAgenticVerifier()
    other.llm_engine = again
    StepVerifier(max_iters=9).resolve_stage(other, other.make_adapter({"image": path}),
                                            smear_segmenter(tmp_path, labels, path, []), {},
                                            input_record({"image": path, "image_id": "smear"}))
    assert len(engine.preflights) == 1 and not again.preflights and other.preflight == verifier.preflight


EVIDENCE = ("COLOUR POPULATIONS default (top row: high, 3 masks, 8.3%, mean B-R +50.0; bottom row: low, 33 masks, "
            "91.7%, mean B-R -85.0):")


def test_agentic_a_colour_minority_starts_the_tuning_with_the_colour_evidence_and_asks_the_colour_rule_once(
        tmp_path, monkeypatch):
    # 24-px objects (mask_diameter_px 27.1 at diameter 30: no scale problem), 3 of 36 purple: the colour minority only
    engine = Engine([[move("default", min_size=20)]], final=["agent_1"],
                    profile=SMEAR_PROFILE)                                   # colour: the model's none
    resolution, verifier, calls, rgb, labels = smear_search(tmp_path, monkeypatch, engine, side=24)
    admission = resolution.details["admission"]
    assert engine.asked(AdmissionAnswer) == [] and admission["decision"] == "RUN" and admission["problems"] == [
        "colour minority: 3 of 36 masks (8.3%) in the high (bluer) population"]
    # arm A's colour rule asked once, with its own question on its own evidence image; the model's none: no filter,
    # no correction holds, so no reference call
    assert len(engine.colors) == 1 and engine.colors[0][0].startswith(ccf._PROMPT.split("{clusters}")[0])
    assert resolution.details["color_decision"] == {"action": "none", "confidence": "high", "reason": "one class",
                                                    "applied": "none"}
    assert verifier.reasoner_failures == 0
    assert verifier.reference_calls == {} and len(calls) == 2
    # so the default call is the default: agent_1 (min_size 20) has the same colour minority, not strictly fewer
    # problems
    assert resolution.details["reference_default"]["call"] == "default"
    assert resolution.details["reference_default"]["overridden"] is False
    # its prompt's rules reach no agentic prompt
    rules = [s.strip() for s in re.split(r"(?<=[.:])\s+", ccf._PROMPT) if len(s.strip()) > 30]
    assert rules and not any(rule in engine.prompts() for rule in rules)
    # the image right after the input image, in every round prompt and the final one
    for content in engine.asked(SegPlan) + engine.asked(SegChoice):
        assert content[1] == "INPUT IMAGE:" and content[3] == EVIDENCE and isinstance(content[4], bytes)
    shown = resolution.details["color_evidence"]          # the 3 purple masks; 8 of the 33 pink, evenly spaced
    assert shown == {"high": [1, 2, 3], "low": [4, 9, 13, 18, 22, 27, 31, 36]}
    # the raw colours: each tile's centre is its mask's pixel as arm A's evidence image shows it; the mask outlined
    evidence = np.asarray(Image.open(io.BytesIO(engine.asked(SegPlan)[0][4])))
    arm_a = np.asarray(Image.open(io.BytesIO(ccf.render_evidence_png(rgb, labels))))
    tile, gap = seg.COLOR_TILE_PX, 2
    assert evidence.shape == (2 * tile + gap, 8 * tile + 7 * gap, 3)
    for row, ids in enumerate((shown["high"], shown["low"])):
        for col, i in enumerate(ids):
            y, x = row * (tile + gap), col * (tile + gap)
            r, c = (int(np.mean(axis)) for axis in np.nonzero(labels == i))
            assert tuple(evidence[y + tile // 2, x + tile // 2]) == tuple(arm_a[r, c]), (row, i)
            assert (evidence[y:y + tile, x:x + tile] == (30, 30, 30)).all(axis=-1).any(), (row, i)
    assert tuple(evidence[tile // 2, tile // 2]) != tuple(evidence[tile + gap + tile // 2, tile // 2])  # purple, pink
    # arm A's rule did not apply: no call has the colour filter (the agent's call changed min_size only)
    assert all(c["color_class_filter"] is None for c in calls) and resolution.details["call"] == "default"


def verdict(action="keep_high_br", confidence="high"):
    return ccf.ColorClassVerdict(scene="a stained smear", is_mixed_class=True, action=action, confidence=confidence,
                                 reason="the few purple cells are the target")


COLOUR_RULE = "the system's colour rule: keep_high_br (the few purple cells are the target)"


@pytest.mark.parametrize("side, scale", [(12, True), (24, False)])
def test_agentic_the_reference_call_holds_every_arm_a_correction_that_holds(tmp_path, monkeypatch, side, scale):
    # 3 of 36 purple; arm A's colour rule keeps the purple population (high confidence, 8.3% <= 10%, 3 >= 3 masks).
    # 12-px objects: also the scale (13.54 px at diameter 30) -> ONE reference call with both; 24-px: the colour only
    engine = Engine([[move("reference", min_size=20)]], final=["reference"], profile=SMEAR_PROFILE, color=verdict())
    resolution, verifier, calls, _, _ = smear_search(tmp_path, monkeypatch, engine, side=side)
    diameter = resolution.trials[1][1].metrics["mask_diameter_px"]
    change = {**({"diameter": diameter} if scale else {}), "color_class_filter": "keep_high_br"}
    assert list(verifier.reference_calls) == ["reference"] and calls[1] == {**verifier.defaults(), **change}
    assert resolution.trials[2][1].param_delta == change and resolution.parents[2] == 1
    notes = f"{reference(diameter)}; {COLOUR_RULE}" if scale else COLOUR_RULE
    assert resolution.trials[2][1].notes == notes
    assert f"Then the system ran: reference ({notes})." in engine.asked(SegPlan)[0][0]
    assert resolution.trials[2][1].metrics["cell_count"] == 3                      # the purple cells
    assert len(engine.colors) == 1 and resolution.details["color_decision"] == {
        "action": "keep_high_br", "confidence": "high", "reason": "the few purple cells are the target",
        "applied": "keep_high_br"}
    assert resolution.details["call"] == "reference" and resolution.details["reference_default"]["call"] == "reference"
    assert "arm A" not in "\n".join(c[0] for c in engine.contents)         # the system's rules, by that name


def test_agentic_without_a_reference_call_the_call_at_the_defaults_is_the_default(image, tmp_path, monkeypatch):
    """No reference call ran (no correction holds on the default call): the default call is the default delivery,
    under the reference default's rule -- an override needs strictly fewer measured problems, the same retry, tag and
    details."""
    # default (diameter 30) has TEST_PROBLEM; agent_1 (diameter 20) the scale problem: one each, refused twice ->
    # default
    engine = Engine([[move(diameter=20.0)]], final=["agent_1", "agent_1"])
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch)
    problems = resolution.details["measured_problems"]
    assert verifier.reference_calls == {} and problems["default"] == [TEST_PROBLEM] and len(problems["agent_1"]) == 1
    # the first round prompt already names the default call as the default (bound before the admission ran)
    assert ("Reference default: default (the call at the defaults) is delivered unless you choose a call that\n"
            "has strictly fewer measured_problems than default") in engine.asked(SegPlan)[0][0]
    assert resolution.status == "satisfied" and resolution.details["call"] == "default"
    assert resolution.details["choice"] == "reference_default" and resolution.params == calls[0]
    default = resolution.details["reference_default"]
    assert default["call"] == "default" and default["overridden"] is False and "not strictly fewer" in default["reason"]
    final, retry = engine.asked(SegChoice)
    assert ("Reference default: default (the call at the defaults) is delivered unless you choose a call that\n"
            "has strictly fewer measured_problems than default (the calls table)") in final[0]
    assert "the system's reference call" not in final[0]
    assert "does not override the reference default default" in retry[0]
    # the agent's call with strictly fewer measured problems than the default call is delivered (colour minority on
    # the default call only)
    other = tmp_path / "fewer"
    other.mkdir()
    engine = Engine([[move("default", color_class_filter="keep_high_br")]], final=["agent_1"],
                    severity={"default": "major"}, profile=SMEAR_PROFILE)
    resolution, verifier, _, _, _ = smear_search(other, monkeypatch, engine, side=24)
    problems = resolution.details["measured_problems"]
    assert verifier.reference_calls == {} and len(problems["default"]) == 1 and problems["agent_1"] == []
    assert resolution.details["call"] == "agent_1" and resolution.details["choice"] == "agent_choice"
    assert resolution.details["reference_default"] == {"call": "default", "overridden": True,
                                                       "reason": resolution.details["reference_default"]["reason"]}
    assert len(engine.asked(SegChoice)) == 1
    # SKIP: the default call delivered as no_tuning_needed, no final question, no default recorded
    skip = tmp_path / "skip"
    skip.mkdir()
    engine = Engine([[move(diameter=20.0)]], admission="none")
    resolution, verifier, _ = run(image, skip, engine, monkeypatch=monkeypatch)
    assert resolution.reason == "no_tuning_needed" and resolution.details["call"] == "default"
    assert engine.asked(SegChoice) == [] and "reference_default" not in resolution.details


def test_agentic_a_reference_call_stays_the_default_over_the_call_at_the_defaults(tmp_path, monkeypatch):
    # the colour rule applies on the default call (24-px objects, no scale problem): the reference call is the
    # default; the agent's choice of the default call has more measured problems -> refused, then reference delivered
    engine = Engine([], final=["default", "default"], profile=SMEAR_PROFILE, color=verdict())
    resolution, verifier, _, _, _ = smear_search(tmp_path, monkeypatch, engine, side=24)
    assert list(verifier.reference_calls) == ["reference"] and verifier.default_call == "reference"
    assert resolution.details["call"] == "reference" and resolution.details["choice"] == "reference_default"
    assert resolution.details["reference_default"]["call"] == "reference"
    final = engine.asked(SegChoice)[0][0]
    assert ("Reference default: reference (the system's reference call)" in final
            and "the call at the defaults)" not in final)


@pytest.mark.parametrize("color", [verdict("keep_low_br"), verdict(confidence="low")])
def test_agentic_arm_as_colour_verdict_is_not_applied_when_its_gates_fail(tmp_path, monkeypatch, color):
    # keeping the pink 92% fails the kept-fraction gate; a low-confidence verdict fails the confidence gate: the
    # reference call holds the diameter correction only (12-px objects), and no reference call without a scale problem
    for side in (12, 24):
        other = tmp_path / f"side_{side}"
        other.mkdir()
        engine = Engine([[move("default", min_size=20)]], final=["default"], color=color)
        resolution, verifier, calls, _, _ = smear_search(other, monkeypatch, engine, side=side)
        decision = resolution.details["color_decision"]
        assert decision["action"] == color.action and decision["applied"] == "none" and len(engine.colors) == 1
        assert all(c["color_class_filter"] is None for c in calls)
        if side == 12:
            diameter = resolution.trials[1][1].metrics["mask_diameter_px"]
            assert list(verifier.reference_calls) == ["reference"]
            assert resolution.trials[2][1].notes == reference(diameter)
        else:
            assert verifier.reference_calls == {}


def speckled(tmp_path, n=19, side=24):
    """A textured smear-like field: n x n objects of `side` px with pixel noise (the texture the coverage readouts
    see) on a flat background -- 3 blue, 31 purple, the rest pink --, and the label mask of all of them. The Otsu
    split of the default call's 361 masks puts 35 in the high population (one pink at the cut), the colour minority;
    within those 35, the 3 blue are a minority again."""
    rng = np.random.default_rng(0)
    pitch = side + 4
    rgb = np.full((n * pitch, n * pitch, 3), 235, np.uint8)
    labels, k = np.zeros((n * pitch, n * pitch), np.int32), 0
    for r in range(n):
        for c in range(n):
            k += 1
            box = (slice(r * pitch + 2, r * pitch + 2 + side), slice(c * pitch + 2, c * pitch + 2 + side))
            labels[box] = k
            colour = (40, 40, 200) if k <= 3 else (120, 40, 170) if k <= 34 else (225, 120, 140)
            rgb[box] = np.clip(np.array(colour) + rng.integers(-25, 26, (side, side, 3)), 0, 255)
    path = tmp_path / "speckled.png"
    Image.fromarray(rgb).save(path)
    return str(path), labels


MINORITY = "colour minority: 35 of 361 masks (9.7%) in the high (bluer) population"
BALANCE = "colour filter keeps 326 of 361 masks (90.3%), above 10%"


def test_agentic_a_filtered_call_counts_the_removed_masks_as_covered_and_is_not_re_split(tmp_path, monkeypatch):
    # default has the colour minority; the system's colour rule keeps it (reference, 35 masks). On the reference
    # call's kept masks the removed pink cells read as missed foreground and the kept masks re-split into a minority
    # (3 of 35); the reference call's qc_score is set below accept_score, so no waiver hides the under-detection
    from shape.agent.verification.loaders import _load_gray, _uncovered_foreground

    verifier = PerTrial()
    verifier.readouts = {2: {"border_mask_fraction": 0.4}}
    engine = Engine([[move("reference", min_size=20)]], final=["default", "default"], color=verdict())
    resolution, verifier, calls, _, labels = smear_search(tmp_path, monkeypatch, engine, verifier=verifier,
                                                          field=speckled)
    (_, at_defaults, first), (_, ref, second) = resolution.trials[1], resolution.trials[2]
    assert calls[1]["color_class_filter"] == "keep_high_br" and ref.metrics["cell_count"] == 35
    assert ref.scalar_score < verifier._linear.accept_score and np.array_equal(np.load(second["prefilter_mask_path"]),
                                                                              np.load(first["mask_path"]))
    linear = verifier._linear
    on_kept = _uncovered_foreground(_load_gray(resolution.trials[0][2]["image"]), np.load(second["mask_path"]))
    assert on_kept["missed_foreground_fraction"] > linear.miss_gate
    assert on_kept["count_miss_fraction"] > linear.count_gate
    # measured with the masks before the filter (here the default call's): the removed masks count as covered, and
    # the missed blobs' size comes from the same masks; the other readouts stay on the kept masks
    assert ({k: ref.metrics[k] for k in seg.PREFILTER_READOUTS}
            == {k: at_defaults.metrics[k] for k in seg.PREFILTER_READOUTS})
    assert ref.metrics["missed_foreground_fraction"] == ref.metrics["count_miss_fraction"] == 0.0
    assert on_kept["missed_blob_size_ratio"] != at_defaults.metrics["missed_blob_size_ratio"]   # the kept masks' differ
    # the kept masks' re-split is shown, not counted; the split the filter made passes the balance gate (9.7%)
    assert verifier._color_minority(verifier._colors[2]) == (
        "colour minority: 3 of 35 masks (8.6%) in the high (bluer) population")
    assert verifier._prefilter_colors[2] == verifier._colors[1]
    problems = resolution.details["measured_problems"]
    assert problems == {"default": [MINORITY], "reference": [], "agent_1": []}
    # before these rules (the kept masks' coverage, the re-split counted) the reference call had 3 problems, and the
    # default call overrode it; the kept masks' coverage no longer counts at all (under-detection is information),
    # the re-split would
    old = VLRRResult("PASS", ref.scalar_score, {**ref.metrics, **on_kept})
    assert verifier.measured_problems(old, calls[1]["diameter"], verifier._colors[2]) == [
        "colour minority: 3 of 35 masks (8.6%) in the high (bluer) population"]
    assert resolution.details["call"] == "reference" and resolution.details["choice"] == "reference_default"
    assert "default has 1 measured problem(s) (" + MINORITY + ") and reference has 0 (none)" in \
        resolution.details["reference_default"]["reason"]


@pytest.mark.parametrize("readouts, default", [
    ({"border_mask_fraction": 0.4}, []),                                   # reference's qc_score below accept_score
    ({"tiny_mask_fraction": 0.2}, ["tiny_mask_fraction 0.200 above 0.15"]),   # reference with a real problem
])
def test_agentic_neither_the_unfiltered_call_nor_a_majority_filter_overrides_the_filtered_default(
        tmp_path, monkeypatch, readouts, default):
    # the QC's departure 6: the reference default (reference) keeps the bluer minority; the agent chooses the
    # default call (the colour minority unfiltered), then agent_1 (keep_low_br from the default call: the pink
    # majority, arm A's balance gate fails); neither has fewer measured problems, so reference is delivered
    verifier = PerTrial()
    verifier.readouts = {2: readouts}
    engine = Engine([[move("default", color_class_filter="keep_low_br")]], final=["default", "agent_1"],
                    color=verdict())
    resolution, verifier, calls, _, _ = smear_search(tmp_path, monkeypatch, engine, verifier=verifier, field=speckled)
    assert calls[2]["color_class_filter"] == "keep_low_br" and resolution.trials[3][1].metrics["cell_count"] == 326
    assert resolution.details["measured_problems"] == {"default": [MINORITY], "reference": default,
                                                       "agent_1": [BALANCE]}
    assert len(engine.asked(SegChoice)) == 2 and resolution.details["call"] == "reference"
    assert resolution.details["choice"] == "reference_default" and resolution.status == "satisfied"
    assert f'"measured_problems": ["{BALANCE}"]' in engine.asked(SegChoice)[0][0]      # agent_1's row, in the table
    assert resolution.details["reference_default"]["reason"].startswith(
        f"agent_1 has 1 measured problem(s) ({BALANCE})")


def test_agentic_the_colour_rule_is_tried_as_every_reasoner_question(tmp_path, monkeypatch):
    class ColourDown(Engine):
        """The colour question fails `failures` times (a gateway error), then is answered."""

        def __init__(self, *args, failures=1, **kwargs):
            super().__init__(*args, **kwargs)
            self.failures = failures

        def generate(self, content, response_format=None):
            if response_format is ccf.ColorClassVerdict and self.failures:
                self.failures -= 1
                self.colors.append(content)
                raise TimeoutError("gateway down")
            return super().generate(content, response_format)

    # one failed try, then the answer: the rule applies
    engine = ColourDown([[move("reference", min_size=20)]], final=["reference"], color=verdict())
    resolution, verifier, calls, _, _ = smear_search(tmp_path, monkeypatch, engine, side=24)
    assert len(engine.colors) == 2 and verifier.reasoner_failures == 0
    assert resolution.details["color_decision"]["applied"] == calls[1]["color_class_filter"] == "keep_high_br"
    # every try fails, or no usable answer: unavailable, a reasoner failure, no filter (no reference call here)
    for name, engine, why in (("down", ColourDown([[move("default", min_size=20)]], color=verdict(), failures=99),
                               "TimeoutError: gateway down"),
                              ("empty", Engine([[move("default", min_size=20)]], color=None), "no usable answer")):
        other = tmp_path / name
        other.mkdir()
        resolution, verifier, calls, _, _ = smear_search(other, monkeypatch, engine, side=24)
        assert len(engine.colors) == base.REASONER_ATTEMPTS and resolution.details["reasoner_failures"] == 1
        assert resolution.details["color_decision"] == f"unavailable ({why})"
        assert resolution.details["failed"][0]["call"] == "reference"
        assert resolution.details["execution_errors"][0]["question"] == "color_rule"
        assert all(c["color_class_filter"] is None for c in calls)
    # a run that requires the agent fails with the last error
    monkeypatch.setenv("SHAPE_REQUIRE_AGENTIC", "1")
    strict = tmp_path / "strict"
    strict.mkdir()
    resolution, _, _, _, _ = smear_search(strict, monkeypatch, ColourDown(color=verdict(), failures=99), side=24)
    assert resolution.details["execution_errors"][0]["question"] == "color_rule"


def test_agentic_measured_problems_of_a_call_the_guard_rejects_are_its_guard_failures():
    verifier = SegmentationAgenticVerifier()
    verifier._labels = {"default": 1, "agent_1": 2}
    attempts = {"default": ("input", verifier.defaults(), VLRRResult("PASS", 0.0, {"cell_count": 0.0})),
                "agent_1": ("input", verifier.defaults(), VLRRResult("PASS", 0.0, {"cell_count": 5.0,
                                                                                    "overlay_is_black": 1.0}))}
    assert verifier.call_problems(attempts, "default") == ["no cells detected"]
    assert verifier.call_problems(attempts, "agent_1") == [
        "visual output invalid (black overlay or implausible coverage)"]


def test_agentic_the_under_detection_readouts_are_the_ones_the_linear_gates_read():
    import inspect

    source = inspect.getsource(SegmentationVerifier._under_detected)
    assert set(re.findall(r'm\.get\("(\w+)"', source)) == set(seg.UNDER_DETECTION_READOUTS)


def test_agentic_colour_evidence_is_attached_only_on_a_colour_minority(image, tmp_path, monkeypatch):
    # arm A's gates: 6 of 36 (16.7%) is above the kept fraction, 2 of 36 (5.6%) below the kept count, a grey image
    # has no color split -> no colour minority: the agent is asked, and no question carries the colour evidence
    def attached(engine) -> list[str]:
        return [item for c in engine.contents for item in c[1:] if isinstance(item, str)]   # the image labels

    for n_purple in (6, 2):
        other = tmp_path / f"purple_{n_purple}"
        other.mkdir()
        engine = Engine([[move("default", min_size=20)]], final=["agent_1"],
                        severity={"default": "major"})                      # TEST_PROBLEM: the tuning starts
        resolution, _, _, _, _ = smear_search(other, monkeypatch, engine, n_purple=n_purple, side=24)
        assert resolution.details["admission"]["problems"] == [TEST_PROBLEM] and engine.asked(AdmissionAnswer) == []
        # 2 rounds, the stop check and the final question (agent_1, without the default call's TEST_PROBLEM,
        # overrides it)
        assert resolution.details["color_evidence"] is None and len(engine.contents) == 4
        assert resolution.details["call"] == "agent_1"
        assert all(c[1] == "INPUT IMAGE:" and c[3].startswith("OVERLAY") for c in engine.contents)
        assert not any(label.startswith("COLOUR POPULATIONS") for label in attached(engine))
    engine = Engine([[move("default", min_size=20)]], final=["agent_1"], severity={"default": "major"})
    resolution, _, _ = run(image, tmp_path, engine, monkeypatch=monkeypatch)
    assert resolution.details["color_evidence"] is None and attached(engine)
    assert not any(label.startswith("COLOUR POPULATIONS") for label in attached(engine))


def test_agentic_colour_minority_without_a_reasoner_is_fatal(tmp_path, monkeypatch):
    from shape.engine.errors import NoEngineError
    with pytest.raises(NoEngineError):
        smear_search(tmp_path, monkeypatch, None, side=24)


def test_agentic_flow_error_is_no_guard_and_never_reaches_the_agent(image, tmp_path, monkeypatch):
    # every call at flow_error 0.9 with cells and a valid overlay: no admission problem, admissible, and not shown
    engine = Engine([[move("default", min_size=20)]], final=["agent_1"],
                    severity={"default": "major"})                          # TEST_PROBLEM: the tuning starts
    resolution, verifier, _ = run(image, tmp_path, engine, monkeypatch=monkeypatch, flow_error=0.9)
    assert resolution.details["admission"]["problems"] == [TEST_PROBLEM] and engine.asked(AdmissionAnswer) == []
    assert resolution.details["admissible"] == ["default", "agent_1"]
    assert verifier.guard(None, resolution.trials[1][1]) == []
    assert resolution.status == "satisfied" and resolution.details["call"] == "agent_1"   # overrides the default
    assert [resolution.trials[i][1].metrics["flow_error"] for i in (1, 2)] == [0.9, 0.9]     # still measured
    assert engine.asked(SegChoice) and "flow_error" not in engine.prompts()          # admission, rounds, final
    # an expectation on flow_error is left out of the predictions shown
    parent = VLRRResult("PASS", 1.0, {"flow_error": 0.9, "cell_count": 2.0})
    vres = VLRRResult("PASS", 1.0, {"flow_error": 0.5, "cell_count": 3.0},
                      expected_effect={"flow_error": "down", "cell_count": "up"})
    assert [p["metric"] for p in verifier._prediction(parent, vres)] == ["cell_count"]


def test_agentic_preflight_preserves_fatal_provider_error(image):
    import httpx
    import openai
    from shape.engine.errors import is_fatal_llm_error
    from shape.utils.schemas import ExecutionError

    class InvalidKey:
        def generate(self, *args, **kwargs):
            raise openai.AuthenticationError("invalid key", body=None,
                response=httpx.Response(401, request=httpx.Request("POST", "https://test.invalid")))

    verifier = SegmentationAgenticVerifier()
    verifier.llm_engine = InvalidKey()
    with pytest.raises(ExecutionError) as caught:
        verifier._preflight(image)
    assert is_fatal_llm_error(caught.value)
    assert caught.value.shape_question == "preflight"
    assert verifier.reasoner_failures == 1


def test_agentic_preflight_unavailable_is_shown_as_such(image, tmp_path):
    verifier = SegmentationAgenticVerifier()
    assert verifier._preflight(image) == "unavailable (no VLM engine available)"      # no engine
    verifier.llm_engine = Engine(profile=None)                                        # no usable answer
    assert verifier._preflight(image) == "unavailable (malformed VLM response)"
    assert verifier.reasoner_failures == 1
    assert verifier.execution_errors[0]["question"] == "preflight"
    verifier._input_path = image                                                      # a grey image: no split
    mask = np.zeros((64, 64), np.int32)
    for k, (r, c) in enumerate(((0, 0), (20, 0), (0, 20), (40, 40)), start=1):
        mask[r:r + 8, c:c + 8] = k
    assert verifier._color_populations(mask) is None
    assert verifier._color_populations(np.zeros((8, 8), np.int32)) == "unavailable (no raw image of the mask's size)"
    np.save(tmp_path / "empty.npy", np.zeros((64, 64), np.int32))                   # a call without masks
    metrics = verifier.quantify({"status": "success", "mask_path": str(tmp_path / "empty.npy"), "flow_error": 0.0,
                                 "overlay_qc": {}}, {"trial_index": 1, "image_path": image})
    assert metrics["cell_count"] == 0 and "mask_diameter_px" not in metrics


def test_agentic_round_holds_the_agents_calls_within_the_cap(image, tmp_path, monkeypatch):
    engine = Engine([[move(diameter=20.0), move(diameter=24.0), move(diameter=18.0)]], final=["default"])
    resolution, _, calls = run(image, tmp_path, engine, cap=2, monkeypatch=monkeypatch)
    assert len(calls) == 3 and [p["diameter"] for p in calls] == [30.0, 20.0, 24.0]    # default: the defaults
    assert resolution.details["rounds"] == [["default"], ["agent_1", "agent_2"]]


def test_agentic_failed_prediction_does_not_stop_the_search(image, tmp_path, monkeypatch):
    # agent_2 predicts more cells but merges them (64 -> 16 masks); the prediction is a fact, the search goes on
    engine = Engine([[move(diameter=20.0)], [move("agent_1", diameter=40.0, expected=[("cell_count", "up")])],
                     [move("agent_1", min_size=20)]], final=["agent_3"])
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch)
    assert len(calls) == 4 and resolution.reason == "satisfied"
    third = engine.asked(SegPlan)[2][0]
    assert '"expected": "up", "observed": "-48.0000 (moved)"' in third


def test_agentic_plateau_stops_the_search(image, tmp_path, monkeypatch):
    # default major (assessed in round 1); agent_1 improves the best to minor; agent_2, agent_3, agent_4 do not: 3
    # tool calls
    engine = Engine([[move(diameter=20.0)], [move("agent_1", min_size=20)], [move("agent_2", min_size=25)],
                     [move("agent_3", min_size=30)], [move("agent_4", min_size=35)]],
                    severity={"default": "major", "agent_1": "minor", "agent_2": "minor", "agent_3": "minor",
                              "agent_4": "major"})
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch)
    assert verifier.patience == base.PATIENCE == 3
    assert len(calls) == 5 and verifier.stop_reason == "plateau" and resolution.reason == "plateau"
    assert [p["round"] for p in verifier.plans] == [2, 3, 4, 5, 6]     # the last plan is kept, not run
    assert ("Stop: return no moves when you are satisfied, at any round; you may then name in deliver the finished\n"
            "call to deliver, which must meet the final rule (otherwise you are asked to choose separately). A stop\n"
            "with rounds left is asked once more: the deliver of the answer that confirms it counts. As a\n"
            "backstop, the loop also stops when the rounds are spent or when 3 rounds in a row have not "
            "improved your best assessed severity."
            in engine.asked(SegPlan)[0][0])


def test_agentic_plateau_counts_rounds_a_round_with_a_crashed_call_included(image, tmp_path, monkeypatch):
    # agent_1, agent_2 improve the best (default major); then the rounds of agent_3 and agent_4 (crashed), of agent_5
    # and of agent_6 do not: 3 rounds without progress -> the plateau (counted in rounds, not tool calls: 4 calls in
    # those rounds)
    engine = Engine([[move(diameter=20.0), move(diameter=24.0)],
                     [move("agent_1", min_size=20), move("agent_1", min_size=25)],
                     [move("agent_3", min_size=30)], [move("agent_5", min_size=40)], [move("agent_6", min_size=45)]],
                    severity={"default": "major"})               # the last plan is made, then the plateau stops
    resolution, verifier, calls = run(image, tmp_path, engine, cap=2, monkeypatch=monkeypatch,
                                      crash=lambda p: p["min_size"] == 25)
    assert resolution.details["rounds"] == [["default"], ["agent_1", "agent_2"], ["agent_3", "agent_4"], ["agent_5"],
                                            ["agent_6"]]
    assert [f["call"] for f in resolution.details["failed"]] == ["agent_4"]
    assert verifier.stop_reason == "plateau" and verifier._stall(None, None) == 3 and len(calls) == 7


def test_agentic_budget_counts_rounds(image, tmp_path, monkeypatch):
    engine = Engine([[move(diameter=float(d))] for d in (20, 21, 22, 23)], final=["agent_1"])
    resolution, _, calls = run(image, tmp_path, engine, budget=2, monkeypatch=monkeypatch)
    assert len(calls) == 2 and resolution.reason == "max_iters_reached"   # default, the admission call, is round 1
    assert resolution.details["rounds"] == [["default"], ["agent_1"]]
    assert "Rounds: 1 done, at most 1 more" in engine.asked(SegPlan)[0][0]
    other = tmp_path / "one_round"
    other.mkdir()
    with pytest.raises(ValueError, match="at least 2 rounds"):          # the admission call leaves no round
        run(image, other, Engine(), budget=1, monkeypatch=monkeypatch)


def test_agentic_values_are_validated_and_the_repair_names_the_accepted_moves(image, tmp_path, monkeypatch):
    verifier = SegmentationAgenticVerifier()
    assert verifier.coerce("flow_threshold", ".6") == 0.6 and verifier.coerce("augment", "true") is True
    assert verifier.coerce("niter", "null") is None and verifier.coerce("min_size", "20.4") == 20
    assert verifier.coerce("color_class_filter", "keep_high_br") == "keep_high_br"
    assert verifier.coerce("color_class_filter", "null") is None
    for lever, value, message in (("diameter", "500", "outside"), ("augment", "2", "true or false"),
                                  ("theta", "2", "not a CARD lever"), ("diameter", ":4", "not a JSON value"),
                                  ("color_class_filter", "keep_all", "must be one of")):
        with pytest.raises(ValueError, match=message):
            verifier.coerce(lever, value)
    bad = WaveMove(changes=[LeverChange(lever="diameter", value="")])
    engine = Engine([[move(diameter=20.0), bad]], repair=[move(diameter=20.0), move(diameter=24.0)],
                    final=["default"])
    resolution, _, calls = run(image, tmp_path, engine, cap=2, monkeypatch=monkeypatch)
    repair = next(c[0] for c in engine.contents if "were rejected" in c[0])
    assert "do not return them again" in repair and "{'diameter': 20.0}" in repair
    assert [p["diameter"] for p in calls] == [30.0, 20.0, 24.0]                 # default: the defaults


def test_agentic_inadmissible_final_choice_is_asked_once_more_then_the_best_is_delivered(image, tmp_path,
                                                                                          monkeypatch):
    # no default: default (the defaults) and agent_1 (flow_threshold 1.2) fail the guard (an invalid overlay at
    # diameter 30), so neither the reference call nor the default call is a default; agent_2 (diameter 20) is
    # admissible
    invalid = lambda p: float(p.get("flow_threshold", 0.4)) > 1.0 or float(p.get("diameter", 30)) == 30  # noqa: E731
    engine = Engine([[move(flow_threshold=1.2)], [move(diameter=20.0)]], final=["agent_1", "agent_2"])
    resolution, _, _ = run(image, tmp_path, engine, monkeypatch=monkeypatch, invalid=invalid)
    assert resolution.details["admissible"] == ["agent_2"] and resolution.details["call"] == "agent_2"
    assert resolution.details["choice"] == "agent_choice" and "reference_default" not in resolution.details
    twice = Engine([[move(flow_threshold=1.2)], [move(diameter=20.0)]], final=["agent_1", "agent_1"])
    other = tmp_path / "second"
    other.mkdir()
    resolution, _, _ = run(image, other, twice, monkeypatch=monkeypatch, invalid=invalid)
    # the admissible call with the best assessment
    assert resolution.status == "conceded" and resolution.details["call"] == "agent_2"
    assert resolution.reason == "final_choice_inadmissible"
    # with the default call admissible it is the default: an inadmissible choice, then agent_2 (not strictly fewer
    # measured problems than the default call) -> the default call is delivered
    again = Engine([[move(flow_threshold=1.2)], [move(diameter=20.0)]], final=["agent_1", "agent_2"])
    third = tmp_path / "third"
    third.mkdir()
    resolution, _, _ = run(image, third, again, monkeypatch=monkeypatch)
    assert resolution.details["admissible"] == ["default", "agent_2"] and resolution.details["call"] == "default"
    assert resolution.status == "satisfied" and resolution.details["choice"] == "reference_default"


def test_agentic_no_admissible_call_delivers_the_best_finished_call(image, tmp_path, monkeypatch):
    # every call fails the guard (an invalid overlay): the admission starts the tuning on it, without a question
    engine = Engine([[move(flow_threshold=1.2)]], severity={"default": "none"})
    resolution, _, _ = run(image, tmp_path, engine, monkeypatch=monkeypatch, invalid=lambda p: True)
    assert resolution.details["admission"]["problems"] == [
        "visual output invalid (black overlay or implausible coverage)"]
    assert resolution.reason == "no_admissible_call" and resolution.result is resolution.trials[1][2]
    assert resolution.status == "conceded" and not engine.asked(SegChoice)


def test_clean_segmentation_needs_no_reasoner_but_tuning_requires_one(image, tmp_path, monkeypatch):
    from shape.engine.errors import NoEngineError
    resolution, verifier, calls = run(image, tmp_path, None, monkeypatch=monkeypatch)
    assert calls == [verifier.defaults()] and resolution.reason == "no_tuning_needed"
    other = tmp_path / "second"
    other.mkdir()
    with pytest.raises(NoEngineError):
        run(image, other, None, monkeypatch=monkeypatch, fixed_side=8)


def test_agentic_without_a_reasoner_a_scale_only_problem_re_segments_at_the_measured_diameter(image, tmp_path,
                                                                                               monkeypatch):
    # 12-px cells at diameter 30: arm A's scale check and no linear trigger; the system's reference call makes arm
    # A's correction, and the fallback's own (the same configuration) is not run again
    resolution, verifier, calls = run(image, tmp_path, Engine(final=["reference"]), monkeypatch=monkeypatch, fixed_side=12)
    default = resolution.trials[1][1]
    measured = default.metrics["mask_diameter_px"]
    assert verifier._linear.localize(dict(default.metrics), {"upstream": {}})[1] == []         # the scale only
    assert resolution.details["admission"]["problems"] == [
        f"mask_diameter_px {measured:.2f} is {measured / 30:.2f} {SCALE}"]
    assert calls == [verifier.defaults(), {**verifier.defaults(), "diameter": measured}]  # once, at the measure
    assert resolution.trials[2][1].notes == reference(measured) and list(verifier.reference_calls) == ["reference"]
    assert resolution.details["fallback_rounds"] == 0 and resolution.details["choice"] == "agent_choice"
    # the fallback delivers the reference default; its own rule (the best assessment, then QC score) would pick
    # the default call, which ties the reference call on both and comes first
    attempts = {"default": ("input", calls[0], default), "reference": ("default", calls[1], resolution.trials[2][1])}
    assert verifier.fallback_choice(None, attempts) == "default"
    assert resolution.details["call"] == "reference" and resolution.status == "satisfied"
    assert resolution.details["reference_default"] == {"call": "reference", "overridden": False, "reason": None}
    assert verifier.color_decision is None                                   # no colour minority: no decision


def test_agentic_fallback_steps_a_null_niter_from_its_effective_default(monkeypatch):
    # large missed cells -> SegmentationVerifier's diagnosis raises niter; the defaults leave it null: the step
    # starts from the value the tool then acts with, base.EFFECTIVE_DEFAULTS["niter"] (patched here to show it)
    monkeypatch.setitem(base.EFFECTIVE_DEFAULTS, "niter", 300.0)
    verifier = SegmentationAgenticVerifier()
    metrics = {"cell_count": 20.0, "flow_error": 0.1, "tiny_mask_fraction": 0.0, "large_mask_fraction": 0.0,
               "mean_solidity": 0.9, "border_mask_fraction": 0.0, "mask_area_median": 700.0,
               "mask_diameter_px": diameter_from_mask_area(700.0), "missed_foreground_fraction": 0.2,
               "count_miss_fraction": 0.0, "missed_blob_size_ratio": 2.0}
    attempts = {"default": ("input", verifier.defaults(), VLRRResult("PASS", 1.0, metrics, action="retune_here"))}
    added = []
    verifier.fallback_moves(None, attempts, lambda parent, changes, why: added.append((parent, changes)))
    step = base.apply_bounded({"niter": 300.0}, {"niter": "up"})["niter"]
    assert added == [("default", [{"lever": "niter", "value": json.dumps(float(f"{step:.6g}"))}])]


def test_agentic_fallback_moves_do_not_depend_on_flow_error():
    # normal-sized missed foreground: SegmentationVerifier's diagnosis turns on flow_error (at least 0.25 -> raise
    # flow_threshold); the agentic fallback sets flow_error aside, as its admission does, and arm A keeps its rule
    verifier, moves = SegmentationAgenticVerifier(), {}
    for flow in (0.1, 0.9):
        metrics = {"cell_count": 20.0, "flow_error": flow, "tiny_mask_fraction": 0.0, "large_mask_fraction": 0.0,
                   "mean_solidity": 0.9, "border_mask_fraction": 0.0, "mask_area_median": 700.0,
                   "mask_diameter_px": diameter_from_mask_area(700.0), "missed_foreground_fraction": 0.2,
                   "count_miss_fraction": 0.0, "missed_blob_size_ratio": 1.0}
        attempts = {"default": ("input", verifier.defaults(),
                                VLRRResult("PASS", 1.0, metrics, action="retune_here"))}
        moves[flow] = []
        verifier.fallback_moves(None, attempts, lambda parent, changes, why: moves[flow].append((changes, why)))
    assert moves[0.9] == moves[0.1] and moves[0.1]
    assert all(c["lever"] != "flow_threshold" and "flow" not in why for changes, why in moves[0.9] for c in changes)
    assert verifier._linear._diagnose(metrics, {"parameters": verifier.defaults()})[0] == {"flow_threshold": "up"}


def test_agentic_failed_reasoner_question_is_tried_again(image, tmp_path, monkeypatch):
    class Flaky(Engine):
        failures = 1

        def generate(self, content, response_format=None):
            if response_format is SegPlan and self.failures:
                self.failures -= 1
                raise TimeoutError("gateway timeout")
            return super().generate(content, response_format)

    engine = Flaky([[move(diameter=20.0)]], final=["default"])
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch)
    assert len(calls) == 2 and verifier.reasoner_failures == 0 and resolution.details["call"] == "default"


class Executor:
    def __init__(self, runner, engine):
        self.runner, self.verifier_llm_engine_mm, self.enable_llm_verifier = runner, engine, True
        self.verify_refine = False

    def run_managed_tool(self, tool_name, base_kwargs, params):
        return self.runner(params)


def test_agentic_solve_path_runs_no_call_at_the_planner_defaults_and_no_diameter_pass(image, tmp_path,
                                                                                    monkeypatch):
    from shape.agent.verification.integration import verify_managed_tool

    monkeypatch.setattr(seg, "sam_concurrency", lambda: 1)
    monkeypatch.setattr(SegmentationAgenticVerifier, "admission_problems", lambda self, inp, vres: [TEST_PROBLEM])
    calls = []
    engine = Engine([[move(diameter=40.0)]], final=["default"])
    resolutions, result, status = verify_managed_tool(
        Executor(segmenter(tmp_path, calls), engine), "Cell_Segmenter_Tool", {"image": image, "image_id": "cells"},
        {"diameter": 30.0, "flow_threshold": 0.4, "cellprob_threshold": 0.0, "min_size": 15}, {})
    # the default call at the CARD defaults (the admission call, not the planner's params), then merged masks at
    # diameter 40 (16.9 px measured), still no diameter pass
    defaults = SegmentationAgenticVerifier().defaults()
    assert calls == [defaults, {**defaults, "diameter": 40.0}]
    assert result is resolutions[0].trials[1][2] and status == "satisfied"


REQUEST = ("# Task\nSegment the image.\n\n## TARGET\nOnly the round yeast cells, one instance each.\n"
           "Buds count with their mother.\n\n---\n\n## Scope\nNothing downstream.\n")
TARGET_TEXT = "Only the round yeast cells, one instance each.\nBuds count with their mother."
FROM_REQUEST = f"Target (from the request): {TARGET_TEXT}"


def test_agentic_the_target_is_the_requests_target_section_shown_before_the_preflight(image, tmp_path, monkeypatch):
    verifier = SegmentationAgenticVerifier()
    verifier.request = REQUEST
    engine = Engine([[move("default", min_size=20)]], final=["agent_1"], severity={"default": "major"},
                    profile=SMEAR_PROFILE)                                  # TEST_PROBLEM: the tuning starts
    resolution, verifier, _ = run(image, tmp_path, engine, monkeypatch=monkeypatch, verifier=verifier)
    questions = engine.asked(AdmissionAnswer) + engine.asked(SegPlan) + engine.asked(SegChoice)
    assert len(questions) == 4              # 2 rounds, the stop check and the final question: agent_1 overrides the default call
    for content in questions:                         # the round prompts, the final prompts
        text = content[0]
        assert ("Task: the instance segmentation this image's analysis needs, as accurate as possible.\n"
                f"{FROM_REQUEST}\n") in text
        assert "Nothing downstream" not in text and "default convention" not in text
        assert text.index(FROM_REQUEST) < text.index('"provisional_target_hint": "nucleated_cells"')   # then the read
    assert resolution.details["target"] == {"source": "Target (from the request)", "text": TARGET_TEXT}


@pytest.mark.parametrize("request_text", [None, "", "Segment every cell.", "### Target\nthe nuclei",
                                          "## Target\n\n## Scope\nthe nuclei", "## Targets\nthe nuclei"])
def test_agentic_without_a_target_section_the_target_is_the_cards_default_convention(request_text):
    verifier = SegmentationAgenticVerifier()
    verifier.request = request_text
    assert verifier.target() == ("Target (default convention)", DEFAULT_TARGET)
    assert verifier.task_text().endswith(f"\nTarget (default convention): {DEFAULT_TARGET}")
    verifier.request = "Intro.\n## target   \nthe nuclei\n# Next"                       # any case, up to the next H1
    assert verifier.target() == ("Target (from the request)", "the nuclei")


def test_agentic_a_crlf_or_cr_request_states_its_target_as_the_lf_one():
    verifier = SegmentationAgenticVerifier()
    for newline in ("\r\n", "\r"):
        verifier.request = REQUEST.replace("\n", newline)
        assert verifier.target() == ("Target (from the request)", TARGET_TEXT), repr(newline)


def test_agentic_the_benchmarks_request_states_the_approved_target():
    from pathlib import Path

    verifier = SegmentationAgenticVerifier()
    verifier.request = (Path(__file__).resolve().parents[4] / "benchmarks" / "segmentation_NeurIPS22-CellSeg"
                        / "results" / "Prompt.txt").read_text(encoding="utf-8")
    assert verifier.target() == ("Target (from the request)", (
        "Each whole cell is one instance, over its full extent (including faint cytoplasm, not only the nucleus).\n"
        "In a field where a few stained (nucleated, blue / purple) cells lie among many unstained or red cells,\n"
        "e.g. a blood smear, only the stained cells are cells to segment."))


def test_agentic_the_request_reaches_the_stage_verifier_from_the_solver_step(image, tmp_path, monkeypatch):
    # the solver's executor step (`_execute_and_verify`, what solve()'s loop runs per planned step) -> the managed
    # path -> `verify_managed_tool(request=)` -> `Verifier.resolve_managed_stage(request=)` -> the verifier's prompts
    from types import SimpleNamespace

    from shape.agent.memory import Memory
    from shape.agent.verifier import Verdict, Verifier
    from shape.solver import _execute_and_verify

    monkeypatch.setattr(seg, "sam_concurrency", lambda: 1)
    monkeypatch.setattr(SegmentationAgenticVerifier, "admission_problems", lambda self, inp, vres: [TEST_PROBLEM])
    planner = SimpleNamespace(recommended_params={"diameter": 30.0})
    for question, sub_goal, shown in ((REQUEST, "segment the cells", FROM_REQUEST),
                                      ("", "## Target\nthe nuclei", "Target (from the request): the nuclei"),
                                      ("Segment the image.", "## Target\nthe nuclei",
                                       f"Target (default convention): {DEFAULT_TARGET}")):    # the query first
        other = tmp_path / f"q{len(question)}"
        other.mkdir()
        calls, engine = [], Engine(final=["default"])                   # RUN; the agent's plan: no moves
        step = SimpleNamespace(tool_name="Cell_Segmenter_Tool", sub_goal=sub_goal, context="", justification="")
        _execute_and_verify({"planner": planner, "executor": Executor(segmenter(other, calls), engine)}, step, 1,
                            question, image, Memory(), True, {}, Verifier(), Verdict(scope="plan", status="pass"))
        assert len(calls) == 1 and engine.asked(AdmissionAnswer) == []
        assert f"{shown}\n" in engine.asked(SegPlan)[0][0], question


def test_agentic_levers_are_the_card_lines_within_the_linear_loops_bounds():
    from shape.agent.verification.stage_map import stage_by_name, verifier_cls_for

    verifier = SegmentationAgenticVerifier()
    spec = stage_by_name("segmentation")
    assert set(verifier.levers) == {k for k in spec.param_map if not k.startswith("_")}   # all reach the tool
    assert all(set(verifier.lever_fields[lever]) == {"Type", "Range", "Default"} for lever in verifier.levers)
    for lever in verifier.levers:
        kind, bounds, _ = verifier.lever_spec(lever)
        if lever in base.PARAM_BOUNDS:
            assert kind in ("number", "integer") and bounds == base.PARAM_BOUNDS[lever][:2], lever
    assert verifier.lever_spec("niter") == ("integer", (50.0, 3000.0), True)
    assert verifier.lever_spec("augment")[0] == verifier.lever_spec("resample")[0] == "bool"
    assert verifier.lever_spec("color_class_filter") == ("choice", (None, "keep_high_br", "keep_low_br"), True)
    assert verifier_cls_for(spec, "Cell_Segmenter_Tool") is SegmentationAgenticVerifier
    from shape.agent.verification.verifiers.organoid_segmentation import OrganoidSegmentationAgenticVerifier
    assert verifier_cls_for(spec, "Organoid_Segmenter_Tool") is OrganoidSegmentationAgenticVerifier


def test_agentic_card_loop_settings_match_the_code():
    from shape.tools.base import load_tuning_card

    where = {"`base.py`": base, "`verifiers/segmentation.py`": seg,
             "`SegmentationAgenticVerifier`": SegmentationAgenticVerifier, "`color_class_filter.py`": ccf}
    rows, owner = {}, None
    for line in (load_tuning_card("cell_segmenter") or "").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith("`"):
            owner = owner if cells[2] == "same" else where[cells[2]]
            value = cells[1] == "true" if cells[1] in ("true", "false") else float(cells[1])
            rows[cells[0].strip("`")] = (value, owner)
    assert set(rows) == {"BUDGET", "PATIENCE", "REASONER_ATTEMPTS", "REASONER_RETRY_SECONDS", "SCALE_FACTOR",
                         "COLOR_TILES", "COLOR_TILE_PX", "COLOR_TILE_MIN_PX", "PROMPT_IMAGE_PX", "FINAL_IMAGE_PX",
                         "admission_call", "_MINORITY_MAX_FRAC", "_MINORITY_MIN_COUNT"}
    assert not hasattr(seg, "FLOW_GATE")                 # flow_error is no guard of the agentic loop
    for name, (value, owner) in rows.items():
        assert value == (RETRY_SECONDS if name == "REASONER_RETRY_SECONDS" else getattr(owner, name)), name
    assert (SegmentationAgenticVerifier.budget, SegmentationAgenticVerifier.patience) == (base.BUDGET, base.PATIENCE)
    assert base.PATIENCE == 3 and SegmentationAgenticVerifier.admission_call is True


# ---- the segmentation cohort's modes (`segmentation_cohort`): cohort_levers, start_params ----------------------

def cohort_verifier(**options):
    verifier = SegmentationAgenticVerifier()
    for name, value in options.items():
        setattr(verifier, name, value)
    return verifier


def test_the_cohort_mode_runs_the_images_corrections_with_the_consensus_once_and_asks_the_agent_nothing(
        image, tmp_path, monkeypatch):
    """12-px cells at diameter 30: the scale check holds; the one reference call carries the measured diameter AND
    the group's consensus, then the search stops and that call is delivered unasked."""
    engine = Engine([[move("reference", min_size=20)]], final=["agent_1"])
    verifier = cohort_verifier(cohort_levers={"cellprob_threshold": -1.0})
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, verifier=verifier,
                                      fixed_side=12)
    measured = resolution.trials[1][1].metrics["mask_diameter_px"]
    assert calls == [verifier.defaults(), {**verifier.defaults(), "diameter": measured, "cellprob_threshold": -1.0}]
    assert engine.asked(SegPlan) == [] and engine.asked(SegChoice) == []
    assert resolution.details["rounds"] == [["default"], ["reference"]] and resolution.details["call"] == "reference"
    assert resolution.details["choice"] == resolution.details["verifier_stop"] == "cohort_applied"
    assert resolution.status == "satisfied" and resolution.params["cellprob_threshold"] == -1.0
    notes = resolution.trials[2][1].notes
    assert reference(measured) in notes and 'the group\'s calibrated parameters {"cellprob_threshold": -1.0}' in notes
    assert "reference" in resolution.details["measured_problems"]


def test_the_cohort_mode_without_a_correction_runs_the_consensus_alone(image, tmp_path, monkeypatch):
    """A RUN admission no system rule corrects (TEST_PROBLEM): the reference call is the consensus alone."""
    engine = Engine([[move(diameter=20.0)]])
    verifier = cohort_verifier(cohort_levers={"cellprob_threshold": -1.0, "min_size": 20})
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, verifier=verifier)
    assert calls == [verifier.defaults(), {**verifier.defaults(), "cellprob_threshold": -1.0, "min_size": 20}]
    assert engine.contents == [] and resolution.details["call"] == "reference"


def test_the_cohort_mode_with_no_consensus_and_no_correction_delivers_the_default_call(image, tmp_path, monkeypatch):
    engine = Engine([[move(diameter=20.0)]])
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch,
                                      verifier=cohort_verifier(cohort_levers={}))
    assert calls == [verifier.defaults()] and engine.contents == [] and verifier.reference_calls == {}
    assert resolution.details["call"] == "default" and resolution.details["choice"] == "cohort_applied"
    assert resolution.details["measured_problems"]["default"] == [TEST_PROBLEM]      # the cohort routes it


def test_the_cohort_mode_keeps_the_admissions_skip(image, tmp_path, monkeypatch):
    engine = Engine([[move(diameter=20.0)]], admission="none")
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch,
                                      verifier=cohort_verifier(cohort_levers={"cellprob_threshold": -1.0}))
    assert calls == [verifier.defaults()] and engine.contents == []
    assert resolution.reason == "no_tuning_needed" and resolution.details["call"] == "default"


def test_a_routed_image_starts_its_own_vrr_from_the_consensus(image, tmp_path, monkeypatch):
    """`start_params`: the group's consensus in place of the CARD defaults; the VRR is otherwise the same."""
    engine = Engine([[move("default", min_size=20)]], final=["default"])
    verifier = cohort_verifier(start_params={"cellprob_threshold": -1.0, "not_a_lever": 1})
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, verifier=verifier)
    assert verifier.defaults()["cellprob_threshold"] == -1.0 and "not_a_lever" not in verifier.defaults()
    assert calls[0]["cellprob_threshold"] == -1.0 and calls[1] == {**calls[0], "min_size": 20}
    assert len(engine.asked(SegPlan)) >= 1                                         # the agent's rounds run


def test_a_stage_verifier_option_it_does_not_declare_is_an_error():
    from shape.agent.verification.stage_map import stage_by_name
    from shape.agent.verifier import Verifier

    spec = stage_by_name("segmentation")
    with pytest.raises(ValueError, match="has no option 'no_such_option'"):
        Verifier().resolve_managed_stage(spec, lambda p: {}, {}, {}, tool_name="Cell_Segmenter_Tool",
                                         verifier_options={"no_such_option": 1})
