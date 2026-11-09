"""Segmentation cost reductions preserve guards, delivery rules and evidence."""
import json
from types import SimpleNamespace

import pytest

from shape.agent.verification import base
from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier, SegPlan, SegChoice
from shape.agent.verification.test.test_segmentation_golden_path import image, run, Engine, move
from shape.agent.verification.test.test_segmentation_cohort import Images
from shape.utils.schemas import AdmissionAnswer


@pytest.fixture(autouse=True)
def efficient_defaults(monkeypatch):
    """These tests are about the cost-saving stop rules, the organoid verifier's defaults; Cellpose-SAM keeps the historical
    rules unless the environment asks (`test_defaults_follow_the_verifier`)."""
    monkeypatch.setenv("SHAPE_SEGMENTATION_EARLY_STOP", "1")
    monkeypatch.setenv("SHAPE_SEGMENTATION_CONFIRM_STOP", "valid_choice")


def test_defaults_follow_the_verifier(monkeypatch):
    from shape.agent.verification.segmentation_policy import EFFICIENT, LEGACY
    from shape.agent.verification.verifiers.organoid_segmentation import OrganoidSegmentationAgenticVerifier
    for key in ("SHAPE_SEGMENTATION_EARLY_STOP", "SHAPE_SEGMENTATION_CONFIRM_STOP", "SHAPE_SEGMENTATION_COMPACT_PROMPTS"):
        monkeypatch.delenv(key, raising=False)
    assert SegmentationAgenticVerifier().efficiency_policy == LEGACY
    assert OrganoidSegmentationAgenticVerifier().efficiency_policy == EFFICIENT


@pytest.mark.parametrize("strategy", ["legacy", "four_stage"])
def test_zero_problem_reference_stops_before_search_and_final(image, tmp_path, monkeypatch, strategy):
    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY", strategy)
    engine = Engine([[move("reference", min_size=20)]], final=["agent_1"])
    resolution, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, fixed_side=12)
    assert len(calls) == 2 and resolution.details["call"] == "reference"
    assert resolution.details["measured_problems"]["reference"] == []
    assert resolution.details["verifier_stop"] == "zero_problem_reference"
    assert engine.asked(SegPlan) == engine.asked(SegChoice) == []
    assert resolution.status == "satisfied"
    assert resolution.details["efficiency_events"][0]["action"] == "stop_search"
    # The preflight remains; no fabricated zero usage when the test engine reports none.
    event = resolution.details["llm_calls"][0]
    assert event["phase"] == "preflight" and event["usage"] is None


def test_early_stop_can_be_disabled_for_matched_ablation(image, tmp_path, monkeypatch):
    monkeypatch.setenv("SHAPE_SEGMENTATION_EARLY_STOP", "0")
    engine = Engine([[move("reference", min_size=20)]], final=["reference"])
    resolution, _, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch, fixed_side=12)
    assert len(calls) == 3 and resolution.details["call"] == "reference"
    assert resolution.details["efficiency_policy"]["zero_problem_reference"] == "0"


def test_remaining_problem_still_searches(image, tmp_path, monkeypatch):
    engine = Engine([[move("default", min_size=20)]], final=["default"])
    resolution, _, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch)
    assert len(calls) >= 2 and resolution.details["plans"]
    assert resolution.details["measured_problems"]["default"]


def test_inadmissible_reference_never_triggers_early_stop(monkeypatch):
    verifier = SegmentationAgenticVerifier()
    monkeypatch.setattr(verifier, "reference_default", lambda i, a: "reference")
    monkeypatch.setattr(verifier, "call_problems", lambda a, c: [])
    monkeypatch.setattr(verifier, "guard", lambda i, v: ["invalid overlay"])
    assert verifier.unbeatable_reference(None, {"reference": (None, {}, None)}) is None
    assert verifier.unbeatable_reference(None, {}) is None


def test_stop_confirmation_requires_valid_delivery_and_override(monkeypatch):
    v = SegmentationAgenticVerifier()
    monkeypatch.setattr(v, "admissible", lambda i, a: ["reference", "agent_1"])
    monkeypatch.setattr(v, "reference_default", lambda i, a: "reference")
    monkeypatch.setattr(v, "override_allowed", lambda *a: (False, "not better"))
    assert not v.confirm_stop_required(SimpleNamespace(deliver="reference"), None, {})
    for call in (None, "missing", "agent_1"):
        assert v.confirm_stop_required(SimpleNamespace(deliver=call), None, {})
    monkeypatch.setattr(v, "override_allowed", lambda *a: (True, "better"))
    assert not v.confirm_stop_required(SimpleNamespace(deliver="agent_1"), None, {})
    v.efficiency_policy["stop_confirmation"] = "always"
    assert v.confirm_stop_required(SimpleNamespace(deliver="reference"), None, {})


def test_compact_table_retains_every_evidence_field(monkeypatch):
    monkeypatch.setenv("SHAPE_SEGMENTATION_COMPACT_PROMPTS", "1")
    v = SegmentationAgenticVerifier()
    rows = [{"call": "reference", "readouts": {"tiny": .1}, "guard": "admissible", "change": {"min_size": 15}},
            {"call": "agent_1", "readouts": {"tiny": .2}, "guard": "invalid", "change": {"min_size": 100}}]
    monkeypatch.setattr(v, "_rows", lambda *a: rows)
    encoded = v._table(None, {"reference": 1, "agent_1": 2})
    table = json.loads(encoded)
    assert [dict(zip(table["columns"], row)) for row in table["rows"]] == rows
    assert len(encoded) < len("\n".join(json.dumps(row) for row in rows))


def test_skip_calibration_still_routes_problem_images_and_preserves_small_groups(monkeypatch):
    monkeypatch.setenv("SHAPE_SEGMENTATION_CALIBRATION", "skip")
    imgs = Images({("cell_culture", "fluorescence"): 12, ("blood_smear", "brightfield"): 2, None: 1},
                  problems=lambda i, o: ["bad"] if i.endswith("000") and o is not None else [])
    _, record = imgs.go()
    assert record["calibration_mode"] == "skip" and "calibration" not in record["counts"]
    big = next(g for g in record["groups"] if len(g["images"]) == 12)
    assert big["calibration"] == [] and big["cohort_levers"] == {}
    assert len(big["routed"]) == 1 and record["counts"] == {"per_image_vrr": 4, "cohort": 11}
    assert imgs.options(big["routed"][0]) == [{"cohort_levers": {}}, None]


def test_llm_cost_records_include_failed_attempts_without_counting_retry_sleep(monkeypatch):
    monkeypatch.setattr(base.time, "sleep", lambda *a: None)
    events = []
    v = SegmentationAgenticVerifier(log=lambda kind, **kw: events.append((kind, kw)))
    replies = iter([RuntimeError("transient"), {"content": AdmissionAnswer(problem="none", severity="none", reasoning="ok"),
                   "usage": {"total_tokens": 15, "prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 3}}])
    def generate(*a, **kw):
        x = next(replies)
        if isinstance(x, Exception): raise x
        return x
    v.llm_engine = SimpleNamespace(generate=generate)
    v._ask(["private prompt"], AdmissionAnswer, "round")
    assert [r["outcome"] for r in v.llm_calls] == ["error", "ok"]
    assert v.llm_calls[0]["usage"] is None
    assert v.llm_calls[1]["usage"]["total_tokens"] == 15
    assert all(r["seconds"] >= 0 and r["phase"] == "round" for r in v.llm_calls)
    persisted = [json.loads(e["text"]) for k,e in events if k == "llm_call"]
    assert persisted == v.llm_calls and "private prompt" not in json.dumps(persisted)


@pytest.mark.parametrize("key", ["SHAPE_SEGMENTATION_EARLY_STOP", "SHAPE_SEGMENTATION_CONFIRM_STOP",
                                  "SHAPE_SEGMENTATION_COMPACT_PROMPTS"])
def test_invalid_efficiency_policy_fails_closed(monkeypatch, key):
    monkeypatch.setenv(key, "typo")
    with pytest.raises(ValueError, match=key): SegmentationAgenticVerifier()


@pytest.mark.parametrize("policy, rounds", [("valid_choice", 1), ("always", 2)])
def test_valid_stopping_plan_is_delivered_without_extra_final_question(image, tmp_path, monkeypatch, policy, rounds):
    monkeypatch.setenv("SHAPE_SEGMENTATION_CONFIRM_STOP", policy)
    class StopEngine(Engine):
        def generate(self, content, response_format=None, **kwargs):
            if response_format is SegPlan:
                self.contents.append(content)
                self.schemas.append(response_format)
                return {"content": SegPlan(analysis="retain valid default", moves=[], deliver="default"),
                        "usage": {"total_tokens": 15, "prompt_tokens": 10, "completion_tokens": 5}}
            return super().generate(content, response_format=response_format, **kwargs)
    engine = StopEngine()
    result, verifier, calls = run(image, tmp_path, engine, monkeypatch=monkeypatch)
    assert len(calls) == 1 and result.details["call"] == "default"
    assert len(engine.asked(SegPlan)) == rounds and not engine.asked(SegChoice)
    assert result.details["plan_delivery"]["delivered"]
    assert sum(e["usage"]["total_tokens"] for e in result.details["llm_calls"] if e["usage"]) == 15*rounds
    assert result.details["stop_checks"] == rounds-1


def test_invalid_calibration_mode_is_rejected_before_work(monkeypatch):
    monkeypatch.setenv("SHAPE_SEGMENTATION_CALIBRATION", "typo")
    with pytest.raises(ValueError, match="SHAPE_SEGMENTATION_CALIBRATION"):
        Images({("cell_culture", "fluorescence"): 12}).go()
