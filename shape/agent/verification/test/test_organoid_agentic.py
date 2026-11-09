"""Organoid CARD, real managed VRR, native-scale and cohort regressions.

Synthetic native tool results exercise the real adapters/search/delivery without
requiring CO weights, a GPU or live model calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from shape.agent.verification import segmentation_cohort as cohort
from shape.agent.verification.integration import verify_managed_tool
from shape.agent.verification.stage_map import invoke_managed_tool, stage_by_name, verifier_cls_for
from shape.agent.verification.test.test_four_stage import CoveringEngine as CellCoveringEngine
from shape.agent.verification.test.test_segmentation_golden_path import Engine as CellEngine, move
from shape.agent.verification.verifiers.organoid_segmentation import OrganoidSegmentationAgenticVerifier
from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier, SegPlan, SegChoice
from shape.solver import _segmentation_cohort, _uses_segmentation_cohort
from shape.utils.schemas import StageResolution, VLRRResult


class CoveringEngine(CellCoveringEngine):
    def generate(self, content, response_format=None, **kwargs):
        from shape.agent.verification.verifiers.organoid_segmentation import EmptyOrganoidReview
        if response_format is EmptyOrganoidReview:
            return self.inner.generate(content,response_format=response_format,**kwargs)
        return super().generate(content,response_format=response_format)


class Engine(CellEngine):
    """Synthetic positive raw images: the independent review sees their target."""
    def generate(self, content, response_format=None, **kwargs):
        from shape.agent.verification.verifiers.organoid_segmentation import EmptyOrganoidReview
        if response_format is EmptyOrganoidReview:
            return {"content":EmptyOrganoidReview(target_presence="present",interpretable=True,confidence="high",evidence="A synthetic organoid is visible in the raw image"),"usage":{"total_tokens":17}}
        return super().generate(content,response_format=response_format,**kwargs)


TOOL = "Organoid_Segmenter_Tool"


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "organoids.png"
    Image.fromarray(np.random.default_rng(7).integers(10, 240, (600, 600), dtype=np.uint8)).save(path)
    return str(path)


class NativeOrganoidTool:
    """Write actual masks and overlays; optionally return a zero-detection failure."""

    def __init__(self, root, side=80, fail_default=False, debris=False, invalid_reference=False):
        self.root, self.side, self.fail_default = root, side, fail_default
        self.debris = debris
        self.invalid_reference = invalid_reference
        self.calls = []

    def execute(self, image, diameter=None, flow_threshold=.4, cellprob_threshold=0,
                min_size=15, niter=None, tile_overlap=None, augment=None, resample=None,
                attempt_id=None, **kwargs):
        params = dict(diameter=diameter, flow_threshold=flow_threshold,
                      cellprob_threshold=cellprob_threshold, min_size=min_size,
                      niter=niter, tile_overlap=tile_overlap, augment=augment, resample=resample)
        self.calls.append({**params, "attempt_id": attempt_id})
        mask = np.zeros((600, 600), dtype=np.int32)
        empty = self.fail_default and cellprob_threshold == 0
        if not empty:
            if self.debris:
                for i, (r, c) in enumerate(((80, 80), (80, 260), (80, 440), (300, 80), (300, 260)), 1):
                    mask[r:r + 80, c:c + 80] = i
                if min_size <= 15 or self.debris == "always":
                    mask[10:12, 10:12] = 6
            else:
                mask[100:100 + self.side, 100:100 + self.side] = 1
        count = len(np.unique(mask)) - 1
        invalid = self.invalid_reference and attempt_id not in (None, 1)   # every call after the default
        path = self.root / f"organoid_mask_attempt_{attempt_id}.npy"
        overlay = self.root / f"organoid_overlay_attempt_{attempt_id}.png"
        np.save(path, mask)
        Image.fromarray((mask * 150 + 40).astype(np.uint8)).save(overlay)
        result = {"status": "failed" if empty or invalid else "success", "parameters": params,
                  "organoid_mask_path": str(path), "overlay_path": str(overlay),
                  "organoid_count": count, "cell_count": count,
                  "flow_error": .1,
                  "overlay_qc": {"passed": not (empty or invalid), "overlay_is_black": invalid,
                                 "foreground_coverage": float(np.mean(mask > 0))}}
        if empty:
            result["error"] = "organoids=0"
        elif invalid:
            result["error"] = "invalid reference overlay"
        return result


def managed(image, tool, engine=None, **options):
    executor = SimpleNamespace(enable_llm_verifier=engine is not None,
                               verifier_llm_engine_mm=engine,
                               run_managed_tool=lambda name, bk, params: invoke_managed_tool(tool, name, bk, params))
    return verify_managed_tool(executor, TOOL, {"image": image},
                               request="Segment whole organoids.", verifier_options=options or None)


def test_organoid_card_reaches_only_supported_levers_and_keeps_tool_defaults(monkeypatch):
    verifier = OrganoidSegmentationAgenticVerifier()
    assert verifier_cls_for(stage_by_name("segmentation"), TOOL) is type(verifier)
    assert verifier.defaults() == {"diameter": None, "flow_threshold": .4, "cellprob_threshold": 0,
                                   "min_size": 15, "niter": None, "tile_overlap": None,
                                   "augment": None, "resample": None}
    assert verifier.coerce("diameter", "null") is None   # the checkpoint's native scale
    assert verifier.coerce("diameter", "600") == 600
    assert verifier.coerce("min_size", "5000") == 5000
    with pytest.raises(ValueError, match="not a CARD lever"):
        verifier.coerce("color_class_filter", "keep_high_br")
    with pytest.raises(ValueError, match="outside"):
        verifier.coerce("diameter", "2001")
    monkeypatch.setenv("SHAPE_SAM_SERVER_URL", "http://unrelated-cpsam-server")
    assert verifier.concurrency() == 1
    assert "whole organoid" in verifier.target()[1]
    verifier.request = "Segment the image.\n## Target\nOnly intact spheroids.\n## Output\nMask"
    assert verifier.target()[1] == "Only intact spheroids."
    # No change to the cell model's own action space/default scale.
    cell = SegmentationAgenticVerifier()
    assert cell.defaults()["diameter"] == 30 and "color_class_filter" in cell.levers


def test_managed_default_uses_organoid_parameters_and_one_audited_call(image, tmp_path):
    tool = NativeOrganoidTool(tmp_path)
    resolutions, result, status = managed(image, tool)
    res = resolutions[0]
    assert status == "satisfied" and res.reason == "no_tuning_needed"
    assert len(tool.calls) == 1 and tool.calls[0]["diameter"] is None
    assert tool.calls[0]["attempt_id"] == 1
    assert res.trials[0][2]["status"] == "INPUT"
    assert result["organoid_count"] == 1 and res.details["segmentation_tool"] == TOOL
    assert "color_populations" not in res.details


def test_large_native_organoid_is_delivered_without_a_scale_reference(image, tmp_path):
    # Cellpose resizes by 30 / diameter: a 406 px mask at native scale is no reason to set diameter 406.
    tool = NativeOrganoidTool(tmp_path, side=360)
    engine = Engine(final=["reference"], admission="none")
    resolutions, result, status = managed(image, tool, engine)
    res = resolutions[0]
    assert status == "satisfied" and res.reason == "no_tuning_needed"
    assert len(tool.calls) == 1 and tool.calls[0]["diameter"] is None
    assert not engine.contents and result["organoid_mask_path"].endswith("attempt_1.npy")


def test_a_numeric_diameter_far_from_the_mask_size_is_not_a_scale_problem(image, tmp_path):
    tool = NativeOrganoidTool(tmp_path, side=360, fail_default=True)
    engine = Engine([[move(diameter=15, cellprob_threshold=-2, expected=[("cell_count", "up")])]],
                    final=["agent_1"], severity={"agent_1": "none"})
    resolutions, _, status = managed(image, tool, engine)
    res = resolutions[0]
    assert status == "satisfied" and res.details["call"] == "agent_1"
    assert [c["diameter"] for c in tool.calls] == [None, 15]
    assert res.details["measured_problems"]["agent_1"] == []
    assert "scale_error" not in engine.prompts() and "30 / diameter" in engine.prompts()


@pytest.mark.parametrize("strategy", ["legacy", "four_stage"])
def test_zero_detection_uses_agentic_search_and_keeps_attempts(image, tmp_path, monkeypatch, strategy):
    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY_SEGMENTATION", strategy)
    tool = NativeOrganoidTool(tmp_path, fail_default=True)
    inner = Engine([[move(cellprob_threshold=-2, expected=[("cell_count", "up")])]],
                   final=["agent_1"], severity={"agent_1": "none"})
    engine = CoveringEngine(inner) if strategy == "four_stage" else inner
    resolutions, result, status = managed(image, tool, engine)
    res = resolutions[0]
    assert status == "satisfied" and result["organoid_count"] == 1
    assert [c["cellprob_threshold"] for c in tool.calls] == [0, -2]
    assert [c["attempt_id"] for c in tool.calls] == [1, 2]
    assert res.details["call"] == "agent_1" and res.details["failed"]
    assert res.details["admission"]["decision"] == "RUN" and not inner.colors
    assert (tmp_path / "organoid_mask_attempt_1.npy").exists()
    assert "fixed custom CO Cellpose model" in inner.prompts()
    assert "whole organoid" in inner.prompts()
    audit = res.details["reconciliation"][0]
    assert audit["status"] == "not_computable" and audit["before"]["cell_count"] is None
    assert audit["hypothesis"] is None  # the failed default/input is no measured count baseline
    if strategy == "four_stage":
        assert res.details["search_strategy"]["confirmation"]["call"] == "agent_1"


def test_organoid_cohort_applies_its_own_consensus_without_a_scale_change(image, tmp_path):
    tool = NativeOrganoidTool(tmp_path, debris=True)   # a measured problem (debris) starts the reference
    resolutions, _, status = managed(image, tool, cohort_levers={"min_size": 100})
    assert status == "satisfied" and len(tool.calls) == 2
    assert tool.calls[1]["min_size"] == 100 and tool.calls[1]["diameter"] is None
    assert resolutions[0].reason == "cohort_applied"


@pytest.mark.parametrize("calibration", ["full", "skip"])
def test_organoid_cohort_records_tool_identity_and_calibrates_only_organoid_levers(calibration):
    card = OrganoidSegmentationAgenticVerifier()
    specs = [{"image_id": f"o{i}", "path": f"o{i}.png"} for i in range(12)]
    calls = []

    def run_one(spec, options):
        calls.append((spec["image_id"], options))
        params = {**card.defaults(), "diameter": 600, "cellprob_threshold": -2}
        res = StageResolution(result={"organoid_mask_path": "m.npy"}, status="satisfied", params=params,
                              verdict=VLRRResult("PASS", 1, {}),
                              details={"call": "reference", "measured_problems": {"reference": []}})
        return [res], res.result, res.status, None

    runs, record = cohort.run(specs, run_one, lambda items, fn: [fn(s) for s in items],
                              lambda _: SimpleNamespace(confidence="high", specimen_context="organoid",
                                                        modality="brightfield"),
                              card.defaults(), card.levers, tool_name=TOOL, calibration=calibration)
    assert len(runs) == 12 and record["tool"] == TOOL
    assert record["calibration_mode"] == calibration
    consensus = {"cellprob_threshold": -2} if calibration == "full" else {}
    assert record["counts"] == ({"calibration": 3, "cohort": 9} if calibration == "full" else {"cohort": 12})
    assert record["groups"][0]["cohort_levers"] == consensus
    assert all(o is None or o == {"cohort_levers": consensus} for _, o in calls)


def test_solver_cohort_dispatch_uses_organoid_card_and_tool(monkeypatch):
    solver = {"executor": SimpleNamespace(enable_llm_verifier=False), "segmentation_cohort": True}
    specs = [{"image_id": "o1"}, {"image_id": "o2"}]
    assert _uses_segmentation_cohort(solver, TOOL, specs)
    assert not _uses_segmentation_cohort({**solver, "segmentation_cohort": False}, TOOL, specs)
    seen = {}

    def capture(items, run_one, map_images, read_profile, defaults, levers, **kwargs):
        seen.update(defaults=defaults, levers=levers, **kwargs)
        return [], {"tool": kwargs["tool_name"]}

    monkeypatch.setattr(cohort, "run", capture)
    _, record = _segmentation_cohort(solver, specs, None, TOOL)
    assert record["tool"] == TOOL and seen["defaults"]["diameter"] is None
    assert "color_class_filter" not in seen["levers"]


@pytest.mark.parametrize("strategy", ["legacy", "four_stage"])
def test_organoid_early_stop_switch_does_not_change_a_cohort_reference_delivery(
        image, tmp_path, monkeypatch, strategy):
    # Without a scale rule, an organoid reference call comes only from a cohort's calibrated levers;
    # a zero-problem one is delivered under either early-stop policy, with no decision question.
    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY_SEGMENTATION", strategy)
    monkeypatch.setenv("SHAPE_SEGMENTATION_CONFIRM_STOP", "always")  # isolate the early-stop ablation
    outcomes = {}
    for policy in ("0", "1"):
        monkeypatch.setenv("SHAPE_SEGMENTATION_EARLY_STOP", policy)
        folder = tmp_path / policy
        folder.mkdir()
        tool = NativeOrganoidTool(folder, debris=True)
        inner = Engine(final=["reference"], admission="none")
        engine = CoveringEngine(inner) if strategy == "four_stage" else inner
        resolutions, result, status = managed(image, tool, engine, cohort_levers={"min_size": 100})
        res = resolutions[0]
        outcomes[policy] = res
        assert status == "satisfied" and len(tool.calls) == 2 and not inner.contents
        assert res.details["call"] == "reference" and not res.details["measured_problems"]["reference"]
        assert res.details["verifier_stop"] == "cohort_applied"
        assert [(r["call"], r["status"]) for r in res.details["reconciliation"]] == [("reference", "unpredicted")]
        np.testing.assert_array_equal(np.load(result["organoid_mask_path"]), np.load(res.trials[2][2]["organoid_mask_path"]))
    assert outcomes["0"].params == outcomes["1"].params
    np.testing.assert_array_equal(np.load(outcomes["0"].result["organoid_mask_path"]),
                                  np.load(outcomes["1"].result["organoid_mask_path"]))


@pytest.mark.parametrize("strategy", ["legacy", "four_stage"])
@pytest.mark.parametrize("policy, stop_checks", [("valid_choice", 0), ("always", 1)])
def test_organoid_remaining_problem_keeps_vrr_and_valid_delivery_skips_only_confirmation(
        image, tmp_path, monkeypatch, strategy, policy, stop_checks):
    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY_SEGMENTATION", strategy)
    monkeypatch.setenv("SHAPE_SEGMENTATION_CONFIRM_STOP", policy)

    class DeliveryEngine(Engine):
        def generate(self, content, response_format=None):
            reply = super().generate(content, response_format=response_format)
            if response_format is SegPlan and not reply["content"].moves:
                reply["content"].deliver = "agent_1"
            return reply

    inner = DeliveryEngine([[move("default", min_size=100, expected=[("tiny_mask_fraction", "down")])]],
                           severity={"default": "major", "agent_1": "none"})
    engine = CoveringEngine(inner) if strategy == "four_stage" else inner
    tool = NativeOrganoidTool(tmp_path, debris=True)
    resolutions, _, status = managed(image, tool, engine)
    res = resolutions[0]
    assert status == "satisfied" and len(tool.calls) == 2 and res.details["call"] == "agent_1"
    assert res.details["measured_problems"]["default"] and not res.details["measured_problems"]["agent_1"]
    assert res.details["stop_checks"] == stop_checks and res.details["plan_delivery"]["delivered"]
    assert len(inner.asked(SegPlan)) == 2 + stop_checks and not inner.asked(SegChoice)
    audit = res.details["reconciliation"][0]
    assert audit["status"] == "reconciled" and audit["hypothesis"] == "effective"
    assert audit["before"]["tiny_mask_fraction"] > audit["after"]["tiny_mask_fraction"] == 0
    assert res.ledger["min_size"] == "effective"
    assert [e["phase"] for e in res.details["llm_calls"] if e["phase"] != "preflight"] == (
        ["round", "round"] + (["stop_check"] if stop_checks else []))
    assert all(e["usage"] is None for e in res.details["llm_calls"])   # unknown usage is not zero


def test_organoid_unfulfilled_prediction_does_not_replace_reference(image, tmp_path):
    tool = NativeOrganoidTool(tmp_path, debris="always")
    engine = Engine([[move("default", min_size=100, expected=[("tiny_mask_fraction", "down")])]],
                    final=["agent_1", "default"], severity={"default": "major", "agent_1": "none"})
    resolutions, _, status = managed(image, tool, engine)
    res = resolutions[0]
    assert status == "conceded" and len(tool.calls) == 2
    assert res.details["needs_review"] and res.details["visual_review_required"]
    assert res.details["call"] == "default" and res.ledger["min_size"] == "inert"
    audit = res.details["reconciliation"][0]
    assert audit["hypothesis"] == "inert" and audit["before"] == audit["after"]
    assert "unchanged" in audit["observed"][0]["observed"]


def test_skip_organoid_calibration_still_routes_an_image_with_remaining_problems(monkeypatch):
    monkeypatch.setenv("SHAPE_SEGMENTATION_CALIBRATION", "skip")
    card = OrganoidSegmentationAgenticVerifier()
    specs = [{"image_id": f"o{i}", "path": f"o{i}.png"} for i in range(12)]
    options_seen = []

    def run_one(spec, options):
        options_seen.append((spec["image_id"], options))
        problems = ["remaining organoid mask problem"] if spec["image_id"] == "o0" and options is not None else []
        res = StageResolution(result={"organoid_mask_path": "m.npy"}, status="satisfied", params=card.defaults(),
                              verdict=VLRRResult("PASS", 1, {}),
                              details={"call": "default", "measured_problems": {"default": problems}})
        return [res], res.result, res.status, None

    _, record = cohort.run(specs, run_one, lambda items, fn: [fn(s) for s in items],
                          lambda _: SimpleNamespace(confidence="high", specimen_context="organoid", modality="brightfield"),
                          card.defaults(), card.levers, tool_name=TOOL)
    assert record["tool"] == TOOL and record["calibration_mode"] == "skip"
    assert record["counts"] == {"per_image_vrr": 1, "cohort": 11}
    assert record["groups"][0]["calibration"] == [] and record["groups"][0]["cohort_levers"] == {}
    assert [o for i, o in options_seen if i == "o0"] == [{"cohort_levers": {}}, None]


@pytest.mark.parametrize("strategy", ["legacy", "four_stage"])
def test_failed_organoid_reference_cannot_shortcut_verify_or_reconcile(image, tmp_path, monkeypatch, strategy):
    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY_SEGMENTATION", strategy)
    tool = NativeOrganoidTool(tmp_path, debris=True, invalid_reference=True)
    inner = Engine(final=["default"])
    engine = CoveringEngine(inner) if strategy == "four_stage" else inner
    resolutions, result, _ = managed(image, tool, engine, cohort_levers={"min_size": 100})
    res = resolutions[0]
    assert len(tool.calls) == 2 and res.details["call"] == "default"
    assert result["organoid_mask_path"].endswith("attempt_1.npy")
    assert not any(e.get("reason") == "zero_problem_reference" or e.get("action") == "stop_search"
                   for e in res.details["efficiency_events"])
    audit = res.details["reconciliation"][0]
    assert audit["call"] == "reference" and audit["status"] == "failed" and audit["hypothesis"] is None
    # This mask was measured, but its invalid overlay excludes it from delivery:
    # a measured reference must not be promoted to a successful call.
    assert audit["score_outcome"] is None and audit["error"] == "invalid reference overlay"
    entry = res.details["entries"][2]
    assert entry["failed"] and entry["hypothesis"] is None and entry["outcome"] is None
    assert res.ledger["min_size"] == "untried"  # the loop ledger agrees with the failed-call audit


def test_organoid_rules_do_not_hide_a_missing_reasoner_when_tuning_is_needed(image, tmp_path):
    from shape.engine.errors import NoEngineError

    tool = NativeOrganoidTool(tmp_path, fail_default=True)
    with pytest.raises(NoEngineError):
        managed(image, tool)
    assert len(tool.calls) == 1  # no silent rule-based rescue/search replacing the required reasoning
