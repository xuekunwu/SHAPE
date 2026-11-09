"""Task 045 (Phase-2: `consolidate_vision` flag removed, this behavior is now the
UNCONDITIONAL default) unit tests: `_finalize_with_vision_review`'s structural gate
(`_analysis_stage_ran`).

Fires ONLY when an analysis stage actually ran -- an analyzer/visualizer tool in
memory, or `latest_stage_metrics` carrying a `_KEY_METRICS` key. Does NOT fire on a
segmentation-only memory (both directions).

Co-located with the module under test (committed -- AGENTS.md sec 6 P5), not the
gitignored `tests/` dir. No LLM/network: `vision_review_final` is monkeypatched to a
call-recording fake.
"""

from types import SimpleNamespace

from PIL import Image

from shape.agent.memory import Memory
from shape.agent.verification.final_review import FinalFigureVerdict
from shape.solver import _analysis_stage_ran, _finalize_with_vision_review

_ACCEPT_VERDICT = FinalFigureVerdict(
    cluster_separation="clear", group_structure="distinct",
    figure_text_consistency="consistent", action="accept",
    confidence="high", reasoning="looks fine",
)


def _segmentation_only_memory(tmp_path):
    overlay = tmp_path / "overlay.png"
    Image.new("RGB", (4, 4)).save(overlay)
    memory = Memory()
    memory.set_query("how many cells?")
    memory.add_action(step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
                      command="...", result={"overlay_path": str(overlay), "status": "success"})
    return memory


def _analysis_memory(tmp_path):
    umap = tmp_path / "umap.png"
    Image.new("RGB", (4, 4)).save(umap)
    memory = Memory()
    memory.set_query("what cell states are present?")
    memory.add_action(step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
                      command="...", result={"mask_path": "mask.npy", "status": "success"})
    memory.add_action(step_count=2, tool_name="Analysis_Visualizer_Tool", sub_goal="visualize",
                      command="...", result={"umap_path": str(umap), "status": "success"})
    return memory


def _patch_fake_judge(monkeypatch, verdict=_ACCEPT_VERDICT):
    calls = []

    def fake_vision_review_final(engine, figures, conclusion, metrics=None):
        calls.append((figures, conclusion, metrics))
        return verdict

    monkeypatch.setattr(
        "shape.agent.verification.final_review.vision_review_final", fake_vision_review_final)
    return calls


# ---- _analysis_stage_ran: direct, deterministic (no LLM) --------------------------

def test_analysis_stage_ran_false_for_segmentation_only():
    memory = Memory()
    memory.add_action(step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
                      command="...", result={"status": "success"})
    assert _analysis_stage_ran(memory, {}) is False


def test_analysis_stage_ran_true_for_analyzer_tool():
    memory = Memory()
    memory.add_action(step_count=1, tool_name="Cell_State_Analyzer_Tool",
                      sub_goal="analyze", command="...", result={"status": "success"})
    assert _analysis_stage_ran(memory, {}) is True


def test_analysis_stage_ran_true_for_visualizer_tool():
    memory = Memory()
    memory.add_action(step_count=1, tool_name="Analysis_Visualizer_Tool",
                      sub_goal="visualize", command="...", result={"status": "success"})
    assert _analysis_stage_ran(memory, {}) is True


def test_analysis_stage_ran_true_via_key_metrics_even_without_analyzer_tool_name():
    memory = Memory()
    memory.add_action(step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
                      command="...", result={"status": "success"})
    assert _analysis_stage_ran(memory, {"silhouette_by_group": 0.4}) is True


# ---- _finalize_with_vision_review: both directions, unconditional gate -----------

def test_does_not_fire_on_segmentation_only(tmp_path, monkeypatch):
    calls = _patch_fake_judge(monkeypatch)
    memory = _segmentation_only_memory(tmp_path)
    executor = SimpleNamespace(enable_llm_verifier=True, verifier_llm_engine_mm=object())

    conclusion, record = _finalize_with_vision_review(executor, memory, "5 cells found.")

    assert conclusion == "5 cells found."     # untouched
    assert record is None
    assert calls == []                        # judge never invoked -- gated out


def test_does_fire_on_analysis_visualizer(tmp_path, monkeypatch):
    calls = _patch_fake_judge(monkeypatch)
    memory = _analysis_memory(tmp_path)
    executor = SimpleNamespace(enable_llm_verifier=True, verifier_llm_engine_mm=object())

    conclusion, record = _finalize_with_vision_review(executor, memory, "3 cell states found.")

    assert len(calls) == 1
    assert record is not None
    assert record["tool"] == "final_review"


# ---- a failed review is recorded as failed, never returned as if reviewed -----------------------------------

class _DownEngine:
    def generate(self, content, response_format=None):
        raise TimeoutError("gateway down")


def test_a_failed_final_review_is_recorded_as_an_execution_error_and_noted_in_the_conclusion(tmp_path):
    from shape.utils.schemas import EXECUTION_ERROR

    memory = _analysis_memory(tmp_path)
    executor = SimpleNamespace(enable_llm_verifier=True, verifier_llm_engine_mm=_DownEngine())
    conclusion, record = _finalize_with_vision_review(executor, memory, "two clear states")
    assert record["status"] == EXECUTION_ERROR and record["stages"][0]["status"] == EXECUTION_ERROR
    assert "final vision review failed: TimeoutError: gateway down" in record["error"]
    assert conclusion.startswith("two clear states" + chr(10) * 2) and "The final vision review failed" in conclusion
    assert "were not checked" in conclusion


def test_without_a_vision_engine_the_final_review_raises_a_system_error(tmp_path):
    import pytest

    from shape.engine.errors import NoEngineError

    memory = _analysis_memory(tmp_path)
    executor = SimpleNamespace(enable_llm_verifier=True, verifier_llm_engine_mm=None)
    with pytest.raises(NoEngineError, match="no vision engine is configured"):     # fatal: never recorded as reviewed
        _finalize_with_vision_review(executor, memory, "two clear states")
