"""Task 057 Group A (M1): regression tests for the preflight `ImageContextProfile`
read (`image_context.py`). Co-located with the module under test (committed --
AGENTS.md sec 6 P5), not in the gitignored `tests/` dir.

Covers the BRIEF's Validation Requirements relevant to Group A: schema completeness,
the VLM-missing/malformed/low-confidence/contradictory fallback matrix, the
blood-smear structured response, and the process-lifetime cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from shape.agent.verification.image_context import (
    ChannelRoleEntry,
    ImageContextProfile,
    _is_internally_consistent,
    default_unknown_profile,
    read_image_context_profile,
)
from shape.utils.image_data import ImageData


def _fake_img_data(num_channels=1, source_path=None):
    h, w = 32, 32
    data = np.random.randint(0, 255, size=(h, w, num_channels), dtype=np.uint8)
    names = (["grayscale"] if num_channels == 1
             else [f"channel_{i}" for i in range(num_channels)])
    return ImageData(data=data, dtype=data.dtype, channel_names=names, source_path=source_path)


class _StubEngine:
    """No-API stand-in mirroring the project's established fake-engine convention
    (see `test_task021_representation_batch_driven.py`): returns a fixed,
    schema-valid verdict, or a plain string to simulate a malformed response."""

    def __init__(self, verdict_or_raise):
        self._payload = verdict_or_raise

    def generate(self, content, response_format=None, **kwargs):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


# ---- schema completeness -----------------------------------------------------

def test_schema_has_all_brief_fields_with_safe_defaults():
    profile = ImageContextProfile()
    assert profile.modality == "unknown"
    assert profile.specimen_context == "unknown"
    assert profile.channel_roles == []
    assert profile.visible_object_classes == []
    assert profile.provisional_target_hint == "unknown"
    assert profile.preprocessing_action == "uncertain"
    assert profile.structures_to_preserve == []
    assert profile.confidence == "low"
    assert profile.evidence == []
    assert profile.limitations == []


def test_the_preflight_prompt_states_the_target_rule_and_the_schema_its_three_targets():
    """The provisional target follows the rule: nuclei only for a fluorescence image whose signal is a nuclear
    stain alone; the nucleated (white) cells of a stained blood smear; whole cells for every other image. The
    schema's values are these three targets and the fallback's unknown."""
    from typing import get_args

    from shape.agent.prompts import load_prompt

    step5 = load_prompt("image_preprocessor", "preflight").split("STEP 5", 1)[1].split("STEP 6", 1)[0]
    for statement in ("one of nuclei / nucleated_cells / whole_cells",
                      "nuclei: ONLY a fluorescence image whose signal is a nuclear stain alone (DAPI, Hoechst",
                      "with no cytoplasm or membrane channel",
                      "nucleated_cells: a stained blood smear -- its nucleated (white) cells, not the red blood cells",
                      "whole_cells: every other image, including fluorescence with a nuclear and a cytoplasm or "
                      "membrane channel, brightfield, phase contrast and DIC"):
        assert statement in step5, statement
    assert not any(old in step5 for old in ("all_cells", "organoids", "task_defined"))
    assert get_args(ImageContextProfile.model_fields["provisional_target_hint"].annotation) == (
        "nuclei", "nucleated_cells", "whole_cells", "unknown")
    load_prompt("image_preprocessor", "preflight").format(channel_info="1 channel(s)")   # still a valid template


def test_default_unknown_profile_is_conservative_and_records_the_reason():
    profile = default_unknown_profile("no VLM engine available")
    assert profile.modality == "unknown"
    assert profile.specimen_context == "unknown"
    assert profile.provisional_target_hint == "unknown"
    assert profile.preprocessing_action == "uncertain"
    assert profile.confidence == "low"
    assert profile.limitations == ["no VLM engine available"]


def test_profile_round_trips_through_model_dump_json_safe():
    profile = ImageContextProfile(
        modality="fluorescence", specimen_context="cell_culture",
        channel_roles=[ChannelRoleEntry(index=0, name="GFP", role="fluorescence",
                                        evidence="diffuse green signal")],
        visible_object_classes=["nucleus"], provisional_target_hint="nuclei",
        preprocessing_action="background_subtract",
        structures_to_preserve=["faint low-intensity nuclei"],
        confidence="high", evidence=["bright diffuse signal"], limitations=[],
    )
    dumped = profile.model_dump()
    assert dumped["channel_roles"] == [
        {"index": 0, "name": "GFP", "role": "fluorescence", "evidence": "diffuse green signal"}
    ]
    import json
    json.dumps(dumped)  # must not raise -- this is what tool.py persists into the result


# ---- fallback matrix (BRIEF Validation Requirements #9) -----------------------

def test_a_missing_engine_raises():
    with pytest.raises(RuntimeError, match="no vision engine is configured"):                # never a default profile
        read_image_context_profile(_fake_img_data(), engine=None)


def test_unreadable_degenerate_image_falls_back_to_unknown(monkeypatch):
    import shape.agent.verification.image_context as ic
    monkeypatch.setattr(ic, "_preview_png_bytes", lambda img_data: None)
    engine = _StubEngine(ImageContextProfile(modality="brightfield", confidence="high"))
    profile = read_image_context_profile(_fake_img_data(), engine=engine, use_cache=False)
    assert profile.modality == "unknown"
    assert "unreadable image" in profile.limitations[0]


def test_a_malformed_vlm_response_is_an_execution_error():
    from shape.utils.schemas import ExecutionError

    engine = _StubEngine("not a structured verdict")  # no .modality attribute
    with pytest.raises(ExecutionError, match="malformed preflight answer"):
        read_image_context_profile(_fake_img_data(), engine=engine, use_cache=False)


def test_an_engine_exception_is_an_execution_error():
    from shape.utils.schemas import ExecutionError

    engine = _StubEngine(RuntimeError("API timeout"))
    with pytest.raises(ExecutionError, match="preflight VLM call failed: RuntimeError: API timeout"):
        read_image_context_profile(_fake_img_data(), engine=engine, use_cache=False)


def test_low_confidence_falls_back_to_unknown_even_with_populated_fields():
    verdict = ImageContextProfile(
        modality="brightfield", specimen_context="cell_culture",
        visible_object_classes=["fibroblast"], confidence="low",
    )
    engine = _StubEngine(verdict)
    profile = read_image_context_profile(_fake_img_data(), engine=engine, use_cache=False)
    assert profile.modality == "unknown"
    assert profile.specimen_context == "unknown"
    assert "low confidence" in profile.limitations[0]


def test_contradictory_modality_vs_channel_role_falls_back_to_unknown():
    """The CODING_PROMPT's own probe example: modality=fluorescence but a
    channel_roles entry names a brightfield channel."""
    verdict = ImageContextProfile(
        modality="fluorescence", specimen_context="cell_culture",
        channel_roles=[ChannelRoleEntry(index=0, name="ch0", role="brightfield",
                                        evidence="looks like brightfield")],
        confidence="high",
    )
    engine = _StubEngine(verdict)
    profile = read_image_context_profile(_fake_img_data(num_channels=2), engine=engine,
                                         use_cache=False)
    assert profile.modality == "unknown"
    assert "inconsistent" in profile.limitations[0]


def test_consistent_multi_channel_response_is_not_flagged_contradictory():
    verdict = ImageContextProfile(
        modality="mixed", specimen_context="cell_culture",
        channel_roles=[
            ChannelRoleEntry(index=0, name="BF", role="brightfield", evidence="grayscale"),
            ChannelRoleEntry(index=1, name="GFP", role="fluorescence", evidence="green"),
        ],
        confidence="high",
    )
    assert _is_internally_consistent(verdict) is True


# ---- the happy path: blood-smear structured response (BRIEF Validation
# Requirements #5) --------------------------------------------------------------

def test_blood_smear_high_confidence_response_passes_through_verbatim():
    verdict = ImageContextProfile(
        modality="brightfield", specimen_context="blood_smear",
        visible_object_classes=["nucleated white blood cell", "red blood cell", "platelet"],
        provisional_target_hint="nucleated_cells",
        preprocessing_action="brightness_only",
        structures_to_preserve=["nuclear boundaries"],
        confidence="high",
        evidence=["small biconcave red blood cells", "larger nucleated leukocytes"],
    )
    engine = _StubEngine(verdict)
    profile = read_image_context_profile(_fake_img_data(), engine=engine, use_cache=False)
    assert profile.specimen_context == "blood_smear"
    assert profile.provisional_target_hint == "nucleated_cells"
    assert "red blood cell" in profile.visible_object_classes
    assert "platelet" in profile.visible_object_classes
    assert profile.confidence == "high"


# ---- process-lifetime cache -----------------------------------------------------

def test_cache_avoids_a_second_vlm_call_for_the_same_file(tmp_path):
    from PIL import Image as PILImage
    img_path = tmp_path / "sample.png"
    PILImage.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(img_path)
    img_data = ImageData.from_path(str(img_path))

    calls = {"n": 0}

    class _CountingEngine:
        def generate(self, content, response_format=None, **kwargs):
            calls["n"] += 1
            return ImageContextProfile(modality="brightfield", confidence="high")

    engine = _CountingEngine()
    p1 = read_image_context_profile(img_data, engine=engine, use_cache=True)
    p2 = read_image_context_profile(img_data, engine=engine, use_cache=True)
    assert calls["n"] == 1
    assert p1.modality == p2.modality == "brightfield"


def test_use_cache_false_calls_the_engine_every_time(tmp_path):
    from PIL import Image as PILImage
    img_path = tmp_path / "sample2.png"
    PILImage.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(img_path)
    img_data = ImageData.from_path(str(img_path))

    calls = {"n": 0}

    class _CountingEngine:
        def generate(self, content, response_format=None, **kwargs):
            calls["n"] += 1
            return ImageContextProfile(modality="brightfield", confidence="high")

    engine = _CountingEngine()
    read_image_context_profile(img_data, engine=engine, use_cache=False)
    read_image_context_profile(img_data, engine=engine, use_cache=False)
    assert calls["n"] == 2
