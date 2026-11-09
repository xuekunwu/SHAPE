"""Task 044 (M1a): the segmenter Verify refinement -- scale_mismatch evidence (scope
A) + dropping flow_error from the VISION prompt only (scope B) -- gated by the
`verify_refine` flag (scope C). Flag OFF must be code-path-identical to base
`origin/main @ 6087637`: every test below either asserts the flag-off behavior
matches base (no scale_mismatch key, flow_error still in the vision prompt) or the
flag-on refinement (scale_mismatch present/correct/omitted per the degenerate
cases; flow_error dropped from the vision prompt copy only, never from `mask_qc`'s
dict / `vres.metrics`). Co-located with the code under test (AGENTS.md sec 6 P5).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from shape.agent.verification.initial_params import diameter_from_mask_area
from shape.agent.verification.llm_recommend import _build_prompt
from shape.agent.verification.verifiers.segmentation import (
    SegmentationVerifier,
    adapt_segmentation,
    mask_qc,
)


def _clean_masks() -> np.ndarray:
    """5 same-sized (10x10 = 100px^2), non-touching, non-border square blobs --
    identical fixture to `test_segmentation_golden_path.py` (same known area)."""
    masks = np.zeros((200, 200), dtype=np.int32)
    for i in range(5):
        r0 = 20 + i * 30
        masks[r0:r0 + 10, 20:30] = i + 1
    return masks


def _zero_masks() -> np.ndarray:
    return np.zeros((200, 200), dtype=np.int32)


_KNOWN_AREA = 100.0   # each blob in _clean_masks() is exactly 10x10 px


# --------------------------------------------------------------------------- #
# Scope A -- scale_mismatch: correct value + degenerate-case omission (mask_qc)
# --------------------------------------------------------------------------- #

def test_scale_mismatch_correct_on_normal_case():
    diameter_used = 20.0
    metrics = mask_qc(_clean_masks(), _clean_masks(), 0.1, diameter_used)
    expected = diameter_from_mask_area(_KNOWN_AREA) / diameter_used
    assert metrics["scale_mismatch"] == pytest.approx(expected)
    # sanity: the known-area blob's measured diameter is NOT the used diameter, so
    # this is a genuine (non-trivial) ratio, not an accidental ~1.0
    assert metrics["scale_mismatch"] != pytest.approx(1.0)


def test_scale_mismatch_matches_ratio_of_one_when_diameter_used_equals_measured():
    measured = diameter_from_mask_area(_KNOWN_AREA)
    metrics = mask_qc(_clean_masks(), _clean_masks(), 0.1, measured)
    assert metrics["scale_mismatch"] == pytest.approx(1.0)


def test_scale_mismatch_omitted_when_cell_count_zero():
    """Referee D2: cell_count==0 -> mask_qc has no mask_area_median -> the metric
    must be OMITTED, not a misleading ~1.0 from the helper's `default=30` fallback."""
    metrics = mask_qc(_zero_masks(), _zero_masks(), 0.1, 20.0)
    assert metrics["cell_count"] == 0.0
    assert "mask_area_median" not in metrics
    assert "scale_mismatch" not in metrics


def test_scale_mismatch_omitted_when_diameter_used_is_none():
    """diameter=None (tool.py:369's auto-detect default) -> no diameter was actually
    applied to compare against -> OMIT, do not divide by None."""
    metrics = mask_qc(_clean_masks(), _clean_masks(), 0.1, None)
    assert "scale_mismatch" not in metrics


def test_scale_mismatch_omitted_when_diameter_used_is_auto_string():
    metrics = mask_qc(_clean_masks(), _clean_masks(), 0.1, "auto")
    assert "scale_mismatch" not in metrics
    # case/whitespace-insensitive
    metrics2 = mask_qc(_clean_masks(), _clean_masks(), 0.1, " Auto ")
    assert "scale_mismatch" not in metrics2


def test_scale_mismatch_absent_by_default_when_diameter_used_not_given():
    """`mask_qc`'s new 4th param defaults to None -- an unmodified 3-arg caller (the
    pre-044 call shape) gets base-identical output."""
    metrics = mask_qc(_clean_masks(), _clean_masks(), 0.1)
    assert "scale_mismatch" not in metrics


# --------------------------------------------------------------------------- #
# adapt_segmentation threads the used diameter (tool.py:537) into context
# --------------------------------------------------------------------------- #

def test_adapt_segmentation_threads_diameter_used_into_context(tmp_path):
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, _clean_masks())
    result = {"mask_path": str(mask_path), "parameters": {"diameter": 42.5}}
    (_, _, _), context = adapt_segmentation(result)
    assert context["diameter_used"] == 42.5


def test_adapt_segmentation_diameter_used_none_when_parameters_missing():
    (_, _, _), context = adapt_segmentation({"mask_path": None})
    assert context["diameter_used"] is None


# --------------------------------------------------------------------------- #
# Scope C -- the verify_refine flag: OFF = base-identical, ON = refined
# --------------------------------------------------------------------------- #

def test_quantify_flag_off_never_adds_scale_mismatch_even_with_diameter_in_context():
    """Flag-off identity (scope A side): SegmentationVerifier() defaults
    verify_refine=False, so `quantify` must never thread diameter_used into
    `mask_qc` -- base-identical output -- even though context carries one."""
    verifier = SegmentationVerifier()
    assert verifier.verify_refine is False
    context = {"diameter_used": 20.0, "overlay_qc": {}, "coverage_qc": {}}
    metrics = verifier.quantify((_clean_masks(), _clean_masks(), 0.1), context)
    assert "scale_mismatch" not in metrics


def test_quantify_flag_on_adds_correct_scale_mismatch():
    verifier = SegmentationVerifier()
    verifier.verify_refine = True
    context = {"diameter_used": 20.0, "overlay_qc": {}, "coverage_qc": {}}
    metrics = verifier.quantify((_clean_masks(), _clean_masks(), 0.1), context)
    expected = diameter_from_mask_area(_KNOWN_AREA) / 20.0
    assert metrics["scale_mismatch"] == pytest.approx(expected)


def test_build_prompt_flag_off_keeps_flow_error():
    """Flag-off identity (scope B side): an unset/False verify_refine keeps
    flow_error in the vision prompt's metrics copy, exactly like base."""
    metrics = {"flow_error": 0.55, "cell_count": 3.0}
    out = _build_prompt(metrics, card=None)
    assert '"flow_error": 0.55' in out


def test_build_prompt_flag_on_drops_flow_error_but_keeps_other_metrics():
    metrics = {"flow_error": 0.55, "cell_count": 3.0}
    out = _build_prompt(metrics, card=None, verify_refine=True)
    assert '"flow_error"' not in out
    assert '"cell_count": 3.0' in out


class _CapturingEngine:
    """Stub multimodal engine: records the prompt string it was given (the vision
    judge's actual input) instead of calling a real API, mirroring
    `test_task018_judge_severity.py`'s `_StubEngine`."""

    def __init__(self, verdict):
        self._verdict = verdict
        self.received_prompt = None

    def generate(self, content, response_format=None, **kwargs):
        self.received_prompt = content[0]
        return {"content": self._verdict, "usage": {"prompt_tokens": 800}}


def _clean_verdict():
    return types.SimpleNamespace(
        specimen="cells", cell_size="medium", cell_density="moderate",
        image_usable="yes", problem="none", severity="none",
        param="none", direction="none", confidence="high",
        reasoning="masks look fine",
    )


def test_verify_flag_on_drops_flow_error_from_vision_prompt_but_keeps_it_in_result_metrics(
    tmp_path,
):
    """End-to-end (BaseVerifier.verify): flow_error stays in `result.metrics` (the
    FULL dict base.py:245 attaches -- the gate / _diagnose_under_detection /
    Reconcile all keep seeing it) while the vision judge's own prompt never sees it."""
    overlay_path = tmp_path / "overlay.png"
    overlay_path.write_bytes(b"not-a-real-png-but-bytes-is-all-thats-read")
    context = {"upstream": {}, "overlay_qc": {}, "coverage_qc": {},
              "overlay_path": str(overlay_path), "diameter_used": 20.0}
    engine = _CapturingEngine(_clean_verdict())

    verifier = SegmentationVerifier()
    verifier.llm_engine = engine
    verifier.verify_refine = True

    # flow_error below the verifier's flow_gate (default 0.4) so localize() routes
    # to "this_step" and adjudicate() actually reaches the vision judge below.
    result = verifier.verify((_clean_masks(), _clean_masks(), 0.15), context)

    assert engine.received_prompt is not None, "the vision engine must have been called"
    assert '"flow_error"' not in engine.received_prompt
    assert result.metrics["flow_error"] == pytest.approx(0.15)
    assert result.metrics["scale_mismatch"] == pytest.approx(
        diameter_from_mask_area(_KNOWN_AREA) / 20.0)


def test_verify_flag_off_keeps_flow_error_in_vision_prompt_and_omits_scale_mismatch(
    tmp_path,
):
    """Flag-off identity, full loop: with verify_refine left at its False default,
    the vision prompt is base-identical (flow_error present, no scale_mismatch)."""
    overlay_path = tmp_path / "overlay.png"
    overlay_path.write_bytes(b"not-a-real-png-but-bytes-is-all-thats-read")
    context = {"upstream": {}, "overlay_qc": {}, "coverage_qc": {},
              "overlay_path": str(overlay_path), "diameter_used": 20.0}
    engine = _CapturingEngine(_clean_verdict())

    verifier = SegmentationVerifier()
    verifier.llm_engine = engine
    # verify_refine left unset -> class default False

    result = verifier.verify((_clean_masks(), _clean_masks(), 0.15), context)

    assert engine.received_prompt is not None
    assert '"flow_error"' in engine.received_prompt
    assert "scale_mismatch" not in result.metrics
