"""Task 018 (Phase A): the vision judge's severity must drive the loop, not be discarded.

Bug (see BRIEF.md): `rec.severity` can arrive None/invalid at runtime. Before this fix,
`_verdict_to_recommendation` gated on `rec.severity != "major"` directly -- with
severity=None that's True, so the vision judge's own retune proposal was silently
thrown away (accepted instead), the clamp branch kept the deterministic `_diagnose`
retune, and `judge_severity=None` was stamped -- which also disables the
`judge_no_improvement` anti-oscillation guard in `step_verifier.py` (it requires
non-None severity on both attempts). This is the task-006-class failure recurring in
the segmentation loop: an LLM's reasoning is computed but its decision never takes
effect downstream.

Recovered verbatim from `.agent/archives/018_judge-severity-honor-vision/results/` for
task 032's "018 preserved" whole-loop regression gate (the 026 approach); the ONE
adaptation is `test_step_verifier_guard_fires_on_repeated_normalized_severity`'s status
assertion, updated for task 032's escalate_*->conceded migration (D1: this ceiling stop
is now `status="conceded", fault="ceiling"`, with the original label carried in
`reason` instead of `status`) -- same guard, same behavior, new vocabulary. Every other
test/assertion below is unchanged.

These tests use a stub multimodal engine (no API) that returns a duck-typed verdict
object -- NOT a real pydantic `SegOverlayVerdict` for the invalid-severity cases, since
`SegOverlayVerdict.severity` is a required `Literal["none","minor","major"]` and
pydantic construction would reject `severity=None` outright. This mirrors what the code
itself actually trusts: `vision_recommend_segmentation` only checks
`hasattr(rec, "severity")` structurally, never that the VALUE is one of the three
literals -- which is exactly the gap A1 found and A2 closes.
"""

from __future__ import annotations

import os
import tempfile
import types

import pytest

from shape.agent.verification.base import BaseVerifier
from shape.agent.verification.llm_recommend import (
    _normalize_severity,
    vision_recommend_segmentation,
)
from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.verifiers.segmentation import SegmentationVerifier
from shape.utils.schemas import VLRRResult


# ---- fixtures ----------------------------------------------------------------

@pytest.fixture()
def overlay_context(tmp_path):
    """A context dict with a real (dummy) overlay file so `_overlay_bytes` succeeds."""
    overlay_path = tmp_path / "overlay.png"
    overlay_path.write_bytes(b"not-a-real-png-but-bytes-is-all-thats-read")
    return {"overlay_path": str(overlay_path)}


def _verdict(**overrides):
    """A duck-typed stand-in for a `SegOverlayVerdict` -- attributes only, no pydantic
    validation, so an invalid `severity` (e.g. None) can be constructed to reproduce the
    runtime shape the BRIEF traced (severity=None, problem/reasoning populated)."""
    base = dict(
        specimen="HeLa cells on glass",
        cell_size="medium",
        cell_density="moderate",
        image_usable="yes",
        problem="none",
        severity="none",
        param="none",
        direction="none",
        confidence="high",
        reasoning="masks look fine",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _StubEngine:
    """Stub `engine_mm`: `.generate([prompt, overlay_bytes], response_format=...)` returns
    a fixed verdict, ignoring the actual prompt/image (no API call)."""

    def __init__(self, verdict, prompt_tokens=800):
        self._verdict = verdict
        self._prompt_tokens = prompt_tokens

    def generate(self, content, response_format=None, **kwargs):
        return {"content": self._verdict, "usage": {"prompt_tokens": self._prompt_tokens}}


def _det(action="retune_here", param_delta=None, expected_effect=None, decision="WARNING"):
    return VLRRResult(decision, 0.5, {}, action=action,
                       param_delta=param_delta, expected_effect=expected_effect or {})


# ---- (a) valid severity="major" is honored ------------------------------------

def test_valid_major_severity_is_honored(overlay_context):
    verdict = _verdict(problem="cells merged", severity="major",
                        param="flow_threshold", direction="up")
    engine = _StubEngine(verdict)
    det = _det(param_delta={"cellprob_threshold": "up", "min_size": "up"})

    out = vision_recommend_segmentation(engine, overlay_context, {}, det)

    assert out.judge_severity == "major"
    assert out.action == "retune_here"
    assert out.param_delta == {"flow_threshold": "up"}


# ---- (b) severity=None + named problem + concrete param -> normalized "major", ----
# ---- and the VISION param wins over what `_diagnose` would have picked (006-class) --

def test_none_severity_with_named_problem_normalizes_to_major_and_vision_param_wins(
    overlay_context,
):
    # Real deterministic diagnosis for an over-segmentation profile (tiny_mask_fraction
    # high): `_diagnose` picks cellprob_threshold/min_size up -- NOT flow_threshold.
    verifier = SegmentationVerifier()
    metrics = {"tiny_mask_fraction": 0.30, "large_mask_fraction": 0.0,
               "cell_count": 50, "mean_solidity": 0.9}
    det_delta, det_expect, det_why = verifier._diagnose(metrics)
    assert det_delta == {"cellprob_threshold": "up", "min_size": "up"}

    det = VLRRResult("WARNING", 0.5, {}, action="retune_here",
                      param_delta=det_delta, expected_effect=det_expect, notes=det_why)

    # The vision judge names a real problem + a DIFFERENT concrete fix, but its
    # `severity` arrives None (the bug's runtime shape).
    verdict = _verdict(problem="crowded cells merged into large blobs", severity=None,
                        param="flow_threshold", direction="up")
    engine = _StubEngine(verdict)

    out = vision_recommend_segmentation(engine, overlay_context, metrics, det)

    assert out.judge_severity == "major"                      # normalized, non-null
    assert out.action == "retune_here"
    assert out.param_delta == {"flow_threshold": "up"}         # vision's pick...
    assert out.param_delta != det_delta                        # ...not _diagnose's


# ---- (c) severity=None + problem "none" / no param -> normalized "none", accept ----

def test_none_severity_no_problem_normalizes_to_none_and_accepts(overlay_context):
    verdict = _verdict(problem="none", severity=None, param="none", direction="none")
    engine = _StubEngine(verdict)
    det = _det(action="accept", param_delta=None)   # deterministic also found nothing

    out = vision_recommend_segmentation(engine, overlay_context, {}, det)

    assert out.judge_severity == "none"
    assert out.action == "accept"
    assert out.param_delta is None


# ---- direct unit coverage of the normalization rule itself --------------------

@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # already valid -> kept as-is
        (dict(severity="minor", problem="a few missed cells", param="none", direction="none"), "minor"),
        # missing attribute entirely -> duck-typed object without `severity` at all
        (dict(problem="cells merged", param="flow_threshold", direction="up"), "major"),
        # invalid non-literal value + named problem + concrete fix -> major
        (dict(severity="unknown", problem="cells merged", param="flow_threshold", direction="up"), "major"),
        # None severity but no real problem named -> none
        (dict(severity=None, problem="", param="none", direction="none"), "none"),
        # None severity, problem named but NO concrete param -> none (nothing to honor)
        (dict(severity=None, problem="cells merged", param="none", direction="none"), "none"),
    ],
)
def test_normalize_severity_rule(kwargs, expected):
    base = dict(specimen="x", cell_size="medium", cell_density="moderate",
                image_usable="yes", confidence="high", reasoning="r",
                severity="none", problem="none", param="none", direction="none")
    base.update(kwargs)
    rec = types.SimpleNamespace(**base)
    if "severity" not in kwargs:
        del rec.severity
    assert _normalize_severity(rec) == expected


# ---- (d) the judge_no_improvement guard now fires once severity is non-null --------

class _JudgedVerifier(BaseVerifier):
    """Minimal verifier: always retunes deterministically, then defers to the vision
    judge via `vision_recommend_segmentation` (same wiring as SegmentationVerifier).
    Relies on `BaseVerifier.verify()`'s existing quantify/localize/adjudicate composition."""

    def __init__(self, engine):
        self.engine = engine

    def quantify(self, tool_output, context):
        return {}

    def localize(self, metrics, context):
        return "this_step", ["x"], []

    def adjudicate(self, metrics, locus, triggered, context):
        det = VLRRResult("WARNING", 0.5, {}, action="retune_here",
                          param_delta={"cellprob_threshold": "up"}, expected_effect={})
        return vision_recommend_segmentation(self.engine, context, metrics, det)


def test_step_verifier_guard_fires_on_repeated_normalized_severity(overlay_context):
    # Same bug shape as (b): severity arrives None on every attempt, but the judge
    # names a real problem+fix each time, so it normalizes to "major" every attempt --
    # not improving -- which should now trip the anti-oscillation guard.
    verdict = _verdict(problem="crowded cells merged", severity=None,
                        param="flow_threshold", direction="up")
    engine = _StubEngine(verdict)
    verifier = _JudgedVerifier(engine)

    def adapter(result, upstream_ctx):
        return result, overlay_context

    def tool_runner(params):
        return {"params": params}

    # `flow_threshold` (the vision judge's proposed knob) must be a real, bounded
    # param on the initial params so each retune actually moves it (apply_bounded is a
    # no-op for an unrecognized key) -- otherwise the loop would spuriously hit
    # `oscillation_stop` on iteration 1 before the judge guard ever gets to iteration 2.
    step_verifier = StepVerifier(max_iters=3)
    resolution = step_verifier.resolve_stage(
        verifier, adapter, tool_runner,
        initial_params={"flow_threshold": 0.5},
        initial_result={"params": {"flow_threshold": 0.5}},
    )

    # task 032: the status vocabulary collapsed ("judge_no_improvement" is now a
    # `reason`, carried alongside `status="conceded", fault="ceiling"`) -- same guard,
    # same trigger, same keep-best behavior; only the label moved.
    assert resolution.status == "conceded"
    assert resolution.fault == "ceiling"
    assert resolution.reason == "judge_no_improvement"
    assert resolution.verdict.judge_severity == "major"


# ---- deterministic-only (no llm_engine) path is unchanged --------------------

def test_no_engine_path_raises(overlay_context):
    det = _det(param_delta={"cellprob_threshold": "up", "min_size": "up"})
    with pytest.raises(RuntimeError, match="no vision engine is configured"):                # never the deterministic verdict
        vision_recommend_segmentation(None, overlay_context, {}, det)


def test_no_overlay_path_returns_deterministic_unchanged():
    det = _det(param_delta={"cellprob_threshold": "up", "min_size": "up"})
    engine = _StubEngine(_verdict(severity="major"))
    out = vision_recommend_segmentation(engine, {"overlay_path": None}, {}, det)
    assert out is det
