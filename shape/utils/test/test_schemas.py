"""Task 032 (M3 scaffolding): `shape/utils/schemas.py`'s generic VRR structured
outputs. Co-located with the module under test (committed -- AGENTS.md sec 6 P5).
"""

from shape.utils.schemas import RecommendDecision, ReconcileVerdict, VerifyReport, \
    card_levers, validate_lever


class _FakeEngine:
    """No-API stand-in: `.generate(prompt, response_format=...)` returns a fixed,
    schema-valid `RecommendDecision`, ignoring the prompt -- proves a fake engine's
    structured decision is honored end-to-end (parsed, fields intact)."""

    def __init__(self, decision: RecommendDecision):
        self._decision = decision

    def generate(self, content, response_format=None, **kwargs):
        assert response_format is RecommendDecision
        return {"content": self._decision, "usage": {"prompt_tokens": 42}}


def test_fake_engine_recommend_decision_is_honored():
    decision = RecommendDecision(lever="diameter", value="down", reasoning="cells too small")
    engine = _FakeEngine(decision)

    resp = engine.generate(["prompt"], response_format=RecommendDecision)
    out = resp["content"]

    assert isinstance(out, RecommendDecision)
    assert out.lever == "diameter"
    assert out.value == "down"
    assert out.concede is False
    assert out.reasoning == "cells too small"


def test_recommend_decision_lever_validated_against_the_card():
    decision = RecommendDecision(lever="diameter", reasoning="x")
    assert validate_lever(decision.lever, "cell_segmenter") is True

    bogus = RecommendDecision(lever="not_a_real_knob", reasoning="x")
    assert validate_lever(bogus.lever, "cell_segmenter") is False


def test_card_levers_extracts_all_segmenter_tunable_params():
    levers = card_levers("cell_segmenter")
    assert levers == {"diameter", "cellprob_threshold", "flow_threshold", "min_size", "niter", "tile_overlap",
                      "augment", "resample", "color_class_filter"}


def test_card_levers_empty_for_a_tool_with_no_card():
    assert card_levers("no_such_tool_dir") == set()


def test_recommend_decision_concede_carries_fault_and_upstream_target():
    decision = RecommendDecision(lever="none", concede=True, fault="upstream",
                                 upstream_target="preprocessing", reasoning="collapse")
    assert decision.concede is True
    assert decision.fault == "upstream"
    assert decision.upstream_target == "preprocessing"


def test_reconcile_verdict_states():
    v = ReconcileVerdict(lever="min_size", state="effective", notes="tiny_mask_fraction dropped")
    assert v.state == "effective"


def test_verify_report_severity_literal():
    r = VerifyReport(problem="merged cells", severity="major")
    assert r.severity == "major"
