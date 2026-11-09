"""Whole-target anomaly routing and representative-only visual cost regressions."""
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from shape.tools.organoid_segmenter.body_qc import body_qc, problems
from shape.agent.verification.test.test_organoid_agentic import NativeOrganoidTool, managed, Engine
from shape.agent.verification.test.test_segmentation_golden_path import move
from shape.agent.verification.verifiers.organoid_segmentation import OrganoidRepresentativeReview
from shape.engine.errors import NoEngineError


def scene():
    raw = np.full((600, 600), 230, dtype=np.uint8)
    raw[80:400, 80:400] = 25
    return raw


def test_internal_core_triggers_but_whole_small_target_does_not():
    raw = scene(); mask = np.zeros(raw.shape, dtype=np.int32); mask[100:180, 100:180] = 1
    original = mask.copy()
    assert problems(body_qc(raw, mask))
    assert np.array_equal(mask, original)
    mask[100:400, 100:400] = 1
    assert not problems(body_qc(raw, mask))
    raw[:] = 230; raw[30:90, 470:530] = 25; mask[:] = 0; mask[30:90, 470:530] = 1
    assert not problems(body_qc(raw, mask))  # no fixed size/centre assumption


def test_fragmentation_triggers_without_changing_labels():
    raw = scene(); mask = np.zeros(raw.shape, dtype=np.int32)
    for i in range(16):
        y, x = 80 + (i // 4) * 80, 80 + (i % 4) * 80
        mask[y:y+80, x:x+80] = i + 1
    assert any("fragments" in p for p in problems(body_qc(raw, mask)))
    assert len(np.unique(mask)) == 17


@pytest.mark.parametrize("raw", [np.ones((40, 40)), np.full((40, 40), np.nan)])
def test_ambiguous_input_is_not_confident(raw):
    assert body_qc(raw, np.ones((40, 40), dtype=np.int32))["organoid_body_confident"] == 0


class RepairTool(NativeOrganoidTool):
    def execute(self, image, cellprob_threshold=0, attempt_id=None, **params):
        self.side = 300 if cellprob_threshold < 0 else 80
        return super().execute(image, cellprob_threshold=cellprob_threshold, attempt_id=attempt_id, **params)


class RepresentativeEngine(Engine):
    def __init__(self, severity="none", bad_review=False, **kwargs):
        super().__init__(**kwargs); self.reviews = 0; self.severity = severity; self.bad_review = bad_review
    def generate(self, content, response_format=None, **kwargs):
        if response_format is OrganoidRepresentativeReview:
            self.reviews += 1
            assert sum(isinstance(x, bytes) for x in content) == 2
            return {"content": "malformed" if self.bad_review else OrganoidRepresentativeReview(
                problem="Whole target reviewed against the raw image", severity=self.severity, confidence="high"),
                "usage": {"total_tokens": 99}}
        return super().generate(content, response_format=response_format, **kwargs)


def run_scene(tmp_path, tool, engine=None, **options):
    p = tmp_path / "scene.png"; Image.fromarray(scene()).save(p)
    return managed(str(p), tool, engine, preflight_override="synthetic", **options)


def test_normal_fast_path_adds_no_per_image_visual_call(tmp_path):
    tool = NativeOrganoidTool(tmp_path, side=300)
    engine = RepresentativeEngine()
    res, _, status = run_scene(tmp_path, tool, engine, organoid_selective_review=True, cohort_levers={})
    assert status == "satisfied" and len(tool.calls) == 1
    assert engine.reviews == 0 and not engine.contents


def test_internal_core_enters_real_vrr_and_can_be_repaired(tmp_path):
    tool = RepairTool(tmp_path)
    engine = Engine([[move(cellprob_threshold=-1, expected=[("foreground_coverage", "up")])]],
                    final=["agent_1"], severity={"default": "major", "agent_1": "none"})
    res, result, status = run_scene(tmp_path, tool, engine, organoid_selective_review=True)
    assert len(tool.calls) >= 2 and engine.contents
    assert result["parameters"]["cellprob_threshold"] == -1
    assert status == "satisfied" and res[0].details["selected_visual_assessment"]["severity"] == "none"


def test_representative_checks_once_and_records_assessment(tmp_path):
    tool = NativeOrganoidTool(tmp_path, side=300); engine = RepresentativeEngine()
    res, _, status = run_scene(tmp_path, tool, engine, organoid_selective_review=True, organoid_representative=True)
    assert status == "satisfied" and engine.reviews == 1 and len(tool.calls) == 1
    assert res[0].details["selected_visual_assessment"]["confidence"] == "high"


def test_malformed_representative_review_cannot_authorize_calibration(tmp_path, monkeypatch):
    from shape.agent.verification import base
    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)
    tool = NativeOrganoidTool(tmp_path, side=300); engine = RepresentativeEngine(bad_review=True)
    res, _, status = run_scene(tmp_path, tool, engine, organoid_selective_review=True, organoid_representative=True)
    assert status == "conceded" and res[0].details["needs_review"]
    assert not res[0].details["selected_visual_assessment"]


def test_representative_missing_engine_propagates(tmp_path):
    with pytest.raises(NoEngineError):
        run_scene(tmp_path, NativeOrganoidTool(tmp_path, side=300), organoid_selective_review=True, organoid_representative=True)


def test_cohort_visual_option_applies_only_to_representatives():
    from shape.agent.verification import segmentation_cohort as cohort
    from shape.agent.verification.verifiers.organoid_segmentation import OrganoidSegmentationAgenticVerifier
    from shape.utils.schemas import StageResolution, VLRRResult
    specs=[dict(image_id=str(i),path=str(i)) for i in range(12)]; seen=[]
    def run_one(spec, options):
        seen.append((spec["image_id"], options))
        d=dict(call="default", measured_problems={"default":[]}, selected_visual_assessment=dict(severity="none",confidence="high"))
        r=StageResolution(result={"mask_path":"test"},status="satisfied",params={},verdict=VLRRResult("PASS",0,{}),details=d)
        return [r],r.result,r.status,None
    card=OrganoidSegmentationAgenticVerifier()
    _, record=cohort.run(specs,run_one,lambda xs,f:[f(x) for x in xs],
        lambda _:SimpleNamespace(confidence="high",specimen_context="organoid",modality="phase"),
        card.defaults(),card.levers,tool_name="Organoid_Segmenter_Tool",organoid_selective_review=True)
    representatives=set(record["groups"][0]["calibration"])
    assert len(representatives)==3
    assert all(o["organoid_selective_review"] and o["organoid_representative"]==(i in representatives) for i,o in seen)
    assert record["counts"]=={"calibration":3,"cohort":9}


@pytest.mark.parametrize("all_bad", [False, True])
def test_unusable_representative_replaced_with_bounded_reused_samples(all_bad):
    from shape.agent.verification import segmentation_cohort as cohort
    from shape.agent.verification.verifiers.organoid_segmentation import OrganoidSegmentationAgenticVerifier
    from shape.utils.schemas import StageResolution, VLRRResult
    specs=[dict(image_id=str(i),path=str(i)) for i in range(12)]; seen=[]
    initial=cohort.draw([x["image_id"] for x in specs],3,("organoid","phase"));bad=set(initial[:1])
    def run_one(spec, options):
        seen.append(spec["image_id"]);failed=all_bad or spec["image_id"] in bad
        d=dict(call="default", measured_problems={"default":[]},
               selected_visual_assessment=dict(severity="major" if failed else "none", confidence="high"))
        r=StageResolution(result={"mask_path":"test"},status="conceded" if failed else "satisfied",
            reason="visual_review_required" if failed else "",params={},verdict=VLRRResult("PASS",0,{}),details=d)
        return [r],r.result,r.status,None
    card=OrganoidSegmentationAgenticVerifier()
    _,record=cohort.run(specs,run_one,lambda xs,f:[f(x) for x in xs],
        lambda _:SimpleNamespace(confidence="high",specimen_context="organoid",modality="phase"),
        card.defaults(),card.levers,tool_name="Organoid_Segmenter_Tool",organoid_selective_review=True)
    g=record["groups"][0]
    assert len(g["calibration_replacements"]) == (3 if all_bad else 1)
    assert len(seen)==len(set(seen))==12
    if not all_bad:
        assert len(g["calibration_eligible"])==3 and record["counts"]=={"calibration":4,"cohort":8}
    else:assert g["routing"]=="per_image_insufficient_clean_calibration"
