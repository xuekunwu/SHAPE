"""Task 057 Group A (M1): the ONE-WAY `ImageContextProfile` -> segmentation-prompt
adapter (`_render_image_context_hint`/`_build_prompt`'s `context_hint`/
`vision_recommend_segmentation`). Co-located with the module under test (committed
-- AGENTS.md sec 6 P5), not in the gitignored `tests/` dir.

Confirms: (a) the adapter is gated on `specimen_context != "unknown"` so an
unknown/default profile leaves the prompt BYTE-IDENTICAL to before this task (BRIEF
M1); (b) a known profile's
fields actually reach the rendered prompt text; (c) `provisional_target_hint` is
rendered as informational text only, never a schema field on `SegOverlayVerdict`.
"""

from __future__ import annotations

from shape.agent.verification.llm_recommend import (
    SegOverlayVerdict,
    _build_prompt,
    _render_image_context_hint,
    vision_recommend_segmentation,
)
from shape.utils.schemas import VLRRResult


# ---- _render_image_context_hint ------------------------------------------------

def test_hint_is_none_for_missing_profile():
    assert _render_image_context_hint(None) is None


def test_hint_is_none_when_specimen_context_unknown():
    assert _render_image_context_hint({"specimen_context": "unknown"}) is None


def test_hint_renders_blood_smear_fields():
    profile = {
        "specimen_context": "blood_smear",
        "visible_object_classes": ["red blood cell", "platelet", "nucleated white blood cell"],
        "provisional_target_hint": "nucleated_cells",
    }
    hint = _render_image_context_hint(profile)
    assert hint is not None
    assert "blood_smear" in hint
    assert "nucleated_cells" in hint
    assert "advisory only" in hint
    assert "PRIOR CONTEXT" in hint


# ---- _build_prompt byte-identity (default path unchanged) ---------------------

def test_build_prompt_byte_identical_with_no_context_hint():
    """Pins the SAME golden string the pre-057 extraction test pins, called via the
    new keyword-only `context_hint` default (None) -- confirms zero drift for every
    existing caller that does not pass it."""
    out_new_default = _build_prompt({"cell_count": 1.0}, None)
    out_explicit_none = _build_prompt({"cell_count": 1.0}, None, context_hint=None)
    assert out_new_default == out_explicit_none
    assert out_new_default.startswith("You are the visual check")
    assert "PRIOR CONTEXT" not in out_new_default


def test_build_prompt_prepends_hint_when_given():
    out = _build_prompt({"cell_count": 1.0}, None, context_hint="PRIOR CONTEXT: X")
    assert out.startswith("PRIOR CONTEXT: X\n\n")
    assert "You are the visual check" in out


# ---- vision_recommend_segmentation end-to-end -----------------------------------

class _CapturingEngine:
    """Captures the prompt text actually sent, then returns a fixed, schema-valid
    verdict that accepts (mirrors the project's established fake-engine convention,
    see test_task021_representation_batch_driven.py)."""

    def __init__(self):
        self.last_prompt = None

    def generate(self, content, response_format=None, **kwargs):
        self.last_prompt = content[0]
        return SegOverlayVerdict(
            specimen="blood smear", cell_size="small", cell_density="moderate",
            image_usable="yes", problem="none", severity="none",
            param="none", direction="none", confidence="high",
            reasoning="masks look complete",
        )


def _fake_context(overlay_path, image_context_profile=None):
    return {
        "overlay_path": overlay_path,
        "upstream": ({"image_context_profile": image_context_profile}
                     if image_context_profile is not None else {}),
    }


def test_segmentation_prompt_carries_the_hint_when_profile_known(tmp_path):
    overlay = tmp_path / "overlay.png"
    overlay.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)  # content is irrelevant here
    engine = _CapturingEngine()
    det = VLRRResult("PASS", 0.9, {"cell_count": 5})
    profile = {
        "specimen_context": "blood_smear",
        "visible_object_classes": ["red blood cell", "platelet"],
        "provisional_target_hint": "nucleated_cells",
    }
    vision_recommend_segmentation(
        engine, _fake_context(str(overlay), profile), {"cell_count": 5.0}, det)
    assert engine.last_prompt is not None
    assert "blood_smear" in engine.last_prompt
    assert "nucleated_cells" in engine.last_prompt


def test_segmentation_prompt_unchanged_when_profile_absent(tmp_path):
    overlay = tmp_path / "overlay2.png"
    overlay.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    engine = _CapturingEngine()
    det = VLRRResult("PASS", 0.9, {"cell_count": 5})
    vision_recommend_segmentation(
        engine, _fake_context(str(overlay)), {"cell_count": 5.0}, det)
    assert engine.last_prompt is not None
    assert "PRIOR CONTEXT" not in engine.last_prompt


def test_segoverlayverdict_schema_unchanged_no_image_context_field():
    """BRIEF M1: no schema merge -- `SegOverlayVerdict` gains no new field for the
    profile; the adapter is prompt TEXT only."""
    assert "image_context" not in SegOverlayVerdict.model_fields
    assert "provisional_target_hint" not in SegOverlayVerdict.model_fields
