"""Task 045 unit tests: `Planner.generate_final_output`'s `text_only` routing.

`generate_final_output` takes the routing decision as a plain `text_only: bool`
argument -- the caller (`solve()`, solver.py) precomputes it unconditionally from
`not _analysis_stage_ran(memory, latest_stage_metrics(memory))` (Phase-2: the
`consolidate_vision` flag this predicate used to be gated behind was removed; the
re-org is now the unconditional default).

- `text_only=True` (no analysis stage ran): no image bytes attached (neither the raw
  input nor the output figures), routed to the TEXT engine (`self.llm_engine`, not
  `llm_engine_mm`), and the prompt's "VISUAL EVIDENCE -- inspect the figures" clause
  is ABSENT (referee Cond. C-2 -- the text-only answer must not claim to have looked
  at a figure it never saw).
- `text_only=False` (analysis path -- the caller's precomputed value once an analysis
  stage ran): unchanged base behavior -- multimodal engine, input image + output
  figure bytes attached, the VISUAL EVIDENCE clause present.
- `_analysis_stage_ran` (imported from `shape.solver`, not re-derived here) is
  exercised directly against a segmentation-only vs an analysis/visualizer memory to
  confirm the predicate the caller reuses actually discriminates the two cases.

Co-located with the module under test (committed -- AGENTS.md sec 6 P5), not the
gitignored `tests/` dir. Uses fake engines (no network / API key needed) so this is a
pure unit test.
"""

from PIL import Image

from shape.agent import planner as planner_module
from shape.agent.memory import Memory
from shape.agent.planner import Planner
from shape.solver import _analysis_stage_ran

_VISUAL_EVIDENCE_MARKER = "VISUAL EVIDENCE"


class _FakeEngine:
    """Records every `generate` call's input and returns a canned response."""

    def __init__(self, response="fake final answer"):
        self.calls = []
        self._response = response

    def generate(self, input_data, **kwargs):
        self.calls.append(input_data)
        return self._response


def _make_planner(monkeypatch):
    fake_text = _FakeEngine()
    fake_mm = _FakeEngine()

    def fake_create_engine(model_string, is_multimodal=False, api_key=None, **kw):
        return fake_mm if is_multimodal else fake_text

    monkeypatch.setattr(planner_module, "create_engine", fake_create_engine)
    planner = Planner(llm_engine_name="gpt-4o", api_key="test-key")
    return planner, fake_text, fake_mm


def _segmentation_only_memory(tmp_path):
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


# ---- (a) text_only=True + segmentation-only memory: no bytes, text engine, no clause

def test_text_only_true_no_image_bytes_uses_text_engine_no_visual_clause(tmp_path, monkeypatch):
    planner, fake_text, fake_mm = _make_planner(monkeypatch)
    monkeypatch.setattr(planner, "get_image_info", lambda *a, **kw: {})
    memory = _segmentation_only_memory(tmp_path)

    result = planner.generate_final_output(
        question="how many cells?", image="unused.png", memory=memory, text_only=True,
    )

    assert result == "fake final answer"
    assert fake_mm.calls == []                          # no multimodal call at all
    assert len(fake_text.calls) == 1
    prompt = fake_text.calls[0]
    assert isinstance(prompt, str)                       # plain prompt, no bytes list
    assert _VISUAL_EVIDENCE_MARKER not in prompt
    assert "Inspect them, not just the text" not in prompt


# ---- (b) text_only=False + analysis/visualizer memory: unchanged multimodal --------

def test_text_only_false_on_analysis_memory_unchanged_multimodal_with_clause(tmp_path, monkeypatch):
    planner, fake_text, fake_mm = _make_planner(monkeypatch)
    input_png = tmp_path / "input.png"
    Image.new("RGB", (4, 4)).save(input_png)
    monkeypatch.setattr(planner, "get_image_info",
                        lambda image, *a, **kw: {"image_path": str(input_png)})
    memory = _analysis_memory(tmp_path)

    # An analysis stage ran, so the caller's precomputed `text_only` (mirroring
    # solver.py's `not _analysis_stage_ran(...)`) is False -- unchanged multimodal
    # behavior.
    assert _analysis_stage_ran(memory, {}) is True
    result = planner.generate_final_output(
        question="what cell states are present?", image=str(input_png), memory=memory,
        text_only=False,
    )

    assert result == "fake final answer"
    assert fake_text.calls == []
    assert len(fake_mm.calls) == 1
    input_data = fake_mm.calls[0]
    assert isinstance(input_data, list)
    assert isinstance(input_data[0], str)
    assert _VISUAL_EVIDENCE_MARKER in input_data[0]
    image_bytes = [x for x in input_data[1:] if isinstance(x, bytes)]
    assert len(image_bytes) == 2                        # input image + output figure


# ---- (c) text_only default (OFF): byte-identical multimodal base behavior ---------

def test_text_only_default_off_is_base_identical_multimodal(tmp_path, monkeypatch):
    planner, fake_text, fake_mm = _make_planner(monkeypatch)
    input_png = tmp_path / "input.png"
    Image.new("RGB", (4, 4)).save(input_png)
    monkeypatch.setattr(planner, "get_image_info",
                        lambda image, *a, **kw: {"image_path": str(input_png)})
    memory = _segmentation_only_memory(tmp_path)

    result = planner.generate_final_output(
        question="how many cells?", image=str(input_png), memory=memory,
    )   # text_only defaults False -- pre-045 call signature, base-identical

    assert result == "fake final answer"
    assert fake_text.calls == []
    assert len(fake_mm.calls) == 1
    input_data = fake_mm.calls[0]
    assert isinstance(input_data, list)
    prompt = input_data[0]
    assert _VISUAL_EVIDENCE_MARKER in prompt
    image_bytes = [x for x in input_data[1:] if isinstance(x, bytes)]
    assert len(image_bytes) == 2                        # input image + output figure


# ---- the reused predicate itself discriminates the two memory fixtures (Cond. C-1) -

def test_analysis_stage_ran_predicate_false_for_segmentation_only_memory(tmp_path):
    memory = _segmentation_only_memory(tmp_path)
    assert _analysis_stage_ran(memory, {}) is False


def test_analysis_stage_ran_predicate_true_for_analysis_memory(tmp_path):
    memory = _analysis_memory(tmp_path)
    assert _analysis_stage_ran(memory, {}) is True
