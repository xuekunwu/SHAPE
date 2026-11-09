"""Task 057 Group A (M4): `Planner._format_image_context_profile` -- the dedicated
digest line that surfaces a tool result's `ImageContextProfile` to the
planner's memory-formatting prompt, reliably (not subject to `_format_memory`'s
generic 500-char `result_str` truncation). Deterministic, constructed-dict
rendering (no LLM/engine needed) -- mirrors `test_planner_format_qc.py`'s pattern
of calling the method unbound. Co-located with the module under test (committed --
AGENTS.md sec 6 P5), not in the gitignored `tests/` dir.
"""

from __future__ import annotations

from shape.agent.planner import Planner

_format_image_context_profile = Planner._format_image_context_profile


def test_no_profile_renders_empty_string():
    assert _format_image_context_profile(None, {"mask_path": "m.npy"}) == ""
    assert _format_image_context_profile(None, "not a dict") == ""


def test_blood_smear_profile_renders_target_hint_and_visible_classes():
    result = {
        "mask_path": "m.npy",
        "image_context_profile": {
            "modality": "brightfield",
            "specimen_context": "blood_smear",
            "visible_object_classes": ["red blood cell", "platelet", "nucleated white blood cell"],
            "provisional_target_hint": "nucleated_cells",
            "confidence": "high",
            "limitations": [],
        },
    }
    digest = _format_image_context_profile(None, result)
    assert "specimen_context=blood_smear" in digest
    assert "nucleated_cells" in digest
    assert "advisory only" in digest
    assert "red blood cell" in digest
    assert "platelet" in digest


def test_unknown_profile_still_renders_modality_and_confidence_but_no_hint():
    result = {
        "image_context_profile": {
            "modality": "unknown", "specimen_context": "unknown",
            "visible_object_classes": [], "provisional_target_hint": "unknown",
            "confidence": "low", "limitations": ["no VLM engine available"],
        },
    }
    digest = _format_image_context_profile(None, result)
    assert "modality=unknown" in digest
    assert "confidence=low" in digest
    assert "provisional_target_hint" not in digest  # gated: hint omitted when unknown
    assert "limitations=" in digest


def test_per_image_batch_shape_reads_first_entry():
    result = {
        "per_image": [
            {"image_context_profile": {"modality": "fluorescence", "specimen_context": "cell_culture",
                                       "confidence": "medium", "visible_object_classes": [],
                                       "provisional_target_hint": "unknown", "limitations": []}},
        ],
    }
    digest = _format_image_context_profile(None, result)
    assert "modality=fluorescence" in digest
    assert "specimen_context=cell_culture" in digest


def test_format_memory_includes_the_image_context_line():
    """End-to-end (no LLM): `_format_memory` renders the dedicated ImageContext line
    for a tool action, reachable even if the generic Result: string would
    have truncated it away."""
    from shape.agent.memory import Memory

    memory = Memory()
    big_padding = {"noise_field": "x" * 600}  # forces result_str truncation at 500 chars
    memory.add_action(
        step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
        command="tool.execute(...)",
        result={
            **big_padding,
            "image_context_profile": {
                "modality": "brightfield", "specimen_context": "blood_smear",
                "visible_object_classes": ["platelet"], "provisional_target_hint": "nucleated_cells",
                "confidence": "high", "limitations": [],
            },
        },
    )
    formatted = Planner._format_memory(_UnboundSelf(), memory)
    assert "ImageContext:" in formatted
    assert "blood_smear" in formatted


class _UnboundSelf:
    """Minimal stand-in so `_format_memory`/`_format_qc`/`_format_image_context_profile`
    (none of which read instance state beyond calling each other) can be exercised
    without constructing a real `Planner` (which builds live LLM engines in
    `__init__`)."""

    _format_qc = Planner._format_qc
    _format_image_context_profile = Planner._format_image_context_profile
