"""Task 057 Group A (M1/M4): `image_context_profile` propagation through
`integration.py`'s `latest_stage_metrics` (the channel that reaches
`adapt_segmentation`'s `context["upstream"]`, and hence the segmentation vision
prompt -- see `test_llm_recommend_image_context_adapter.py`) and through
`make_verification_record`'s per-trial result summary (the verification-trail
persistence). Co-located with the module under test (committed -- AGENTS.md sec 6
P5), not in the gitignored `tests/` dir.
"""

from __future__ import annotations

from shape.agent.verification.integration import latest_stage_metrics, make_verification_record
from shape.utils.schemas import StageResolution, VLRRResult


class _FakeMemory:
    def __init__(self, actions):
        self._actions = actions

    def get_actions(self, llm_safe=False):
        return self._actions


def test_latest_stage_metrics_carries_the_profile_from_a_tool_result():
    profile = {"specimen_context": "blood_smear", "provisional_target_hint": "nucleated_cells"}
    actions = [{
        "tool_name": "Cell_Segmenter_Tool",
        "result": {"mask_path": "m.npy", "image_context_profile": profile},
        "verification": {"metrics": {"illumination_CV": 0.1}},
    }]
    metrics = latest_stage_metrics(_FakeMemory(actions))
    assert metrics["illumination_CV"] == 0.1
    assert metrics["image_context_profile"] == profile


def test_latest_stage_metrics_handles_per_image_batch_shape():
    profile = {"specimen_context": "organoid"}
    actions = [{
        "tool_name": "Cell_Segmenter_Tool",
        "result": {"per_image": [{"image_context_profile": profile}]},
        "verification": {"metrics": {"illumination_CV": 0.2}},
    }]
    metrics = latest_stage_metrics(_FakeMemory(actions))
    assert metrics["image_context_profile"] == profile


def test_latest_stage_metrics_absent_profile_leaves_metrics_unchanged():
    """A stage whose raw result carries no profile (e.g. the segmenter's own
    result) must not gain a spurious key -- confirms no crash / no pollution for
    every OTHER `latest_stage_metrics` consumer (crop's seg_metrics, final_review's
    `_KEY_METRICS`-gated filter, `_analysis_stage_ran`)."""
    actions = [{
        "tool_name": "Cell_Segmenter_Tool",
        "result": {"mask_path": "m.npy", "overlay_path": "o.png"},
        "verification": {"metrics": {"cell_count": 12.0}},
    }]
    metrics = latest_stage_metrics(_FakeMemory(actions))
    assert "image_context_profile" not in metrics
    assert metrics == {"cell_count": 12.0}


def test_latest_stage_metrics_empty_memory_unchanged():
    assert latest_stage_metrics(_FakeMemory([])) == {}
    assert latest_stage_metrics(None) == {}


# ---- make_verification_record: the trials[].result persistence ----------------

def test_make_verification_record_persists_profile_in_trial_result():
    from shape.agent.verification.stage_map import stage_by_name

    profile = {"specimen_context": "blood_smear", "confidence": "high"}
    result = {"mask_path": "m.npy", "image_context_profile": profile}
    verdict = VLRRResult("PASS", 0.9, {"illumination_CV": 0.1}, action="accept")
    resolution = StageResolution(
        result=result, status="satisfied", params={}, verdict=verdict,
        trials=[({}, verdict, result)],
    )
    vrecord = make_verification_record("Cell_Segmenter_Tool", [resolution], "satisfied")
    stage = vrecord["stages"][0]
    assert stage["stage"] == stage_by_name("segmentation").stage
    assert stage["trials"][0]["result"]["image_context_profile"] == profile


def test_make_verification_record_other_stage_result_has_no_spurious_key():
    result = {"mask_path": "m.npy"}
    verdict = VLRRResult("PASS", 0.9, {"cell_count": 5.0}, action="accept")
    resolution = StageResolution(
        result=result, status="satisfied", params={}, verdict=verdict,
        trials=[({}, verdict, result)],
    )
    vrecord = make_verification_record("Cell_Segmenter_Tool", [resolution], "satisfied")
    assert "image_context_profile" not in vrecord["stages"][0]["trials"][0]["result"]
