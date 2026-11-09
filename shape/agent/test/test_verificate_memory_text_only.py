"""Task 045 (Phase-2: `consolidate_vision` flag removed, text-only is now the
UNCONDITIONAL default) unit tests: `Planner.verificate_memory` never attaches image
bytes and always routes to the TEXT engine (`self.llm_engine`, not `llm_engine_mm`).

- No image bytes attached (neither the input-image attach nor `_append_figure_bytes`),
  routed to the TEXT engine.
- A multi-step / analysis-path completion case still stops correctly through the
  text-only task gate.

Co-located with the module under test (committed -- AGENTS.md sec 6 P5), not the
gitignored `tests/` dir. Uses fake engines (no network / API key needed) so this is a
pure unit test.
"""

from PIL import Image

from shape.agent import planner as planner_module
from shape.agent.memory import Memory
from shape.agent.planner import Planner
from shape.schemas.formatters import MemoryVerification
from shape.solver import _verify_task_complete, decide_stop


class _FakeEngine:
    """Records every `generate` call's input and returns a canned response."""

    def __init__(self, response=None):
        self.calls = []
        self._response = response if response is not None else MemoryVerification(
            analysis="fake", stop_signal=False)

    def generate(self, input_data, response_format=None, **kwargs):
        self.calls.append(input_data)
        return self._response


def _make_planner(monkeypatch, text_response=None, mm_response=None):
    fake_text = _FakeEngine(text_response)
    fake_mm = _FakeEngine(mm_response)

    def fake_create_engine(model_string, is_multimodal=False, api_key=None, **kw):
        return fake_mm if is_multimodal else fake_text

    monkeypatch.setattr(planner_module, "create_engine", fake_create_engine)
    planner = Planner(llm_engine_name="gpt-4o", api_key="test-key")
    return planner, fake_text, fake_mm


def _memory_with_one_action(tmp_path):
    overlay = tmp_path / "overlay.png"
    Image.new("RGB", (4, 4)).save(overlay)
    memory = Memory()
    memory.set_query("how many cells?")
    memory.add_action(
        step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
        command="Cell_Segmenter_Tool.execute(...)",
        result={"overlay_path": str(overlay), "cell_count": 5, "status": "success"},
    )
    return memory


# ---- text-only: no image bytes, TEXT engine ---------------------------------------

def test_no_image_bytes_uses_text_engine(tmp_path, monkeypatch):
    planner, fake_text, fake_mm = _make_planner(monkeypatch)
    input_png = tmp_path / "input.png"
    Image.new("RGB", (4, 4)).save(input_png)
    # simulate an image_info carrying 'image_path', so this test actually exercises
    # a memory state that (pre-045) would have triggered an image attach.
    monkeypatch.setattr(planner, "get_image_info",
                        lambda image, *a, **kw: {"image_path": str(input_png)})
    memory = _memory_with_one_action(tmp_path)

    result = planner.verificate_memory(
        question="how many cells?", image=str(input_png), query_analysis="",
        memory=memory,
    )

    assert isinstance(result, MemoryVerification)
    assert fake_mm.calls == []                         # no overlay look at all
    assert len(fake_text.calls) == 1
    assert isinstance(fake_text.calls[0], str)          # plain prompt, no bytes list


# ---- multi-step / analysis-path completion still stops via the text-only gate ----

def test_multi_step_analysis_path_stops_correctly_with_text_only_gate(tmp_path, monkeypatch):
    """A completed Segmentation -> Cropping -> Analyzer -> Visualizer memory: the
    text-only task gate (fed only the structured QC digest) must still be able to
    recognize completion and stop -- i.e. the wiring from `verificate_memory` through
    `_verify_task_complete` to the single `decide_stop` authority is unaffected by
    dropping the overlay look."""
    stop_response = MemoryVerification(
        analysis="Analysis_Visualizer_Tool completed; pipeline done.", stop_signal=True)
    planner, fake_text, fake_mm = _make_planner(monkeypatch, text_response=stop_response)
    monkeypatch.setattr(planner, "get_image_info", lambda *a, **kw: {})

    crop_meta = tmp_path / "crop1.png"
    Image.new("RGB", (4, 4)).save(crop_meta)
    memory = Memory()
    memory.set_query("what cell states are present?")
    memory.add_action(step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
                      command="...", result={"mask_path": "mask.npy", "status": "success"})
    memory.add_action(step_count=2, tool_name="Single_Cell_Cropper_Tool", sub_goal="crop",
                      command="...",
                      result={"cell_crops_metadata_path": str(crop_meta), "status": "success"})
    memory.add_action(step_count=3, tool_name="Cell_State_Analyzer_Tool",
                      sub_goal="analyze", command="...",
                      result={"n_clusters": 3, "status": "success"},
                      verification={"status": "satisfied",
                                    "stages": [{"stage": "morphological_state",
                                                "decision": "PASS", "action": "accept"}],
                                    "metrics": {"n_clusters": 3}})
    memory.add_action(step_count=4, tool_name="Analysis_Visualizer_Tool",
                      sub_goal="visualize", command="...",
                      result={"umap_path": str(crop_meta), "status": "success"})

    task_verdict = _verify_task_complete(
        planner, "what cell states are present?", str(crop_meta), "", memory)

    assert decide_stop(task_verdict) is True
    assert task_verdict.stop_reason == "task_complete"
    assert fake_mm.calls == []              # still no overlay look, even multi-step
    assert len(fake_text.calls) == 1
