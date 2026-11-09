"""Task 057 M1: the preflight `ImageContextProfile` -- a typed, abstention-capable
VLM read of the RAW input image, run BEFORE any lossy pixel transform.

Scope (Group A only, per BRIEF/CODING_PROMPT): this module produces the profile and
falls back safely. It does NOT branch any pixel/mask/segmentation decision on
`provisional_target_hint` -- that consumer is explicitly deferred to a later,
separate task (BRIEF M1: "provisional_target_hint has no runtime consumer in this
task"). The profile is advisory context only; M2's modality-safe pixel actions and
M3's structure-preservation gate (Group B) are a separate, later increment.

Structurally separate from (never merged into) the post-segmentation
`SegOverlayVerdict.specimen` free-text field (`llm_recommend.py`) -- see
`llm_recommend._render_image_context_hint` for the ONE-WAY adapter that renders a
profile produced here as prior/hint text for that OTHER prompt, never the reverse.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Literal, Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from shape.engine.errors import NoEngineError
from shape.utils.schemas import ExecutionError

logger = logging.getLogger("shape.agent.verification")

_Modality = Literal[
    "brightfield", "phase_contrast", "fluorescence", "histology", "mixed", "unknown"
]
_SpecimenContext = Literal[
    "cell_culture", "blood_smear", "tissue_section", "organoid", "microorganism", "unknown"
]
# The provisional target by the preflight prompt's rule (STEP 5): nuclei only for a fluorescence image whose
# signal is a nuclear stain alone; nucleated_cells for a stained blood smear (its white cells, not the red
# ones); whole_cells for every other image. unknown: the conservative fallback profile only.
_TargetHint = Literal[
    "nuclei", "nucleated_cells", "whole_cells", "unknown"
]
_PreprocessingAction = Literal[
    "skip", "brightness_only", "flat_field_divide", "background_subtract",
    "per_channel", "uncertain",
]
_ChannelRole = Literal["brightfield", "phase_contrast", "fluorescence", "unknown"]
_Confidence = Literal["high", "medium", "low"]


class ChannelRoleEntry(BaseModel):
    """One channel's imaging role + the visual evidence for it (BRIEF M1:
    "channel_roles: per-channel role + evidence")."""

    index: int = 0
    name: str = ""
    role: _ChannelRole = "unknown"
    evidence: str = ""


class ImageContextProfile(BaseModel):
    """Preflight, pre-segmentation read of an image's modality/specimen/target
    context (BRIEF M1's exact field list). Used directly as the VLM structured-
    output schema (mirrors `SegOverlayVerdict`/`RepresentationVerdict`'s convention
    in `llm_recommend.py`) AND as the persisted profile object."""

    modality: _Modality = "unknown"
    specimen_context: _SpecimenContext = "unknown"
    channel_roles: list[ChannelRoleEntry] = Field(default_factory=list)
    visible_object_classes: list[str] = Field(default_factory=list)
    # Advisory forward-contract field for the deferred full target-class-grounding
    # task. NO code in this module -- or anywhere in this task's diff -- branches
    # deterministic behavior on its value; see the module docstring.
    provisional_target_hint: _TargetHint = "unknown"
    preprocessing_action: _PreprocessingAction = "uncertain"
    structures_to_preserve: list[str] = Field(default_factory=list)
    confidence: _Confidence = "low"
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def default_unknown_profile(reason: str = "") -> ImageContextProfile:
    """The conservative, abstention fallback: every field at its least-committal
    value. Used on a missing engine, an unreadable/degenerate image, a malformed VLM
    response, low VLM-reported confidence, or an internally-inconsistent response
    (BRIEF M1: "return an explicit conservative unknown/uncertain profile on missing
    engine, malformed response, low confidence, or unreadable image")."""
    return ImageContextProfile(limitations=[reason] if reason else [])


# ---- internal-consistency check ("contradictory -> fail safe", BRIEF Validation
# Requirements #9 / CODING_PROMPT's probe example) -----------------------------------
# Per single-modality classification, which per-channel roles are consistent with it.
# `mixed`/`unknown` modalities impose no such constraint (a mixed BF+fluorescence
# image is EXPECTED to carry channels of different roles).
_MODALITY_ALLOWED_CHANNEL_ROLES: dict[str, set[str]] = {
    "brightfield": {"brightfield", "unknown"},
    "phase_contrast": {"phase_contrast", "unknown"},
    "fluorescence": {"fluorescence", "unknown"},
    "histology": {"brightfield", "unknown"},
}


def _is_internally_consistent(profile: ImageContextProfile) -> bool:
    """False when the reported single-modality classification contradicts a
    per-channel role it also reported (e.g. modality=fluorescence naming a
    brightfield channel)."""
    allowed = _MODALITY_ALLOWED_CHANNEL_ROLES.get(profile.modality)
    if allowed is None:
        return True
    return all(cr.role in allowed for cr in profile.channel_roles)


# ---- preview image bytes (ephemeral API-call bytes; never written to disk / never
# the quantitative analysis image) --------------------------------------------------

def _preview_png_bytes(img_data) -> Optional[bytes]:
    """A small in-memory PNG preview of the RAW image for the VLM call."""
    try:
        if img_data.is_multi_channel:
            preview = (img_data.create_merged_rgb() * 255).astype(np.uint8)
        else:
            preview = img_data.to_uint8(0)
        buf = io.BytesIO()
        Image.fromarray(preview).save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning("image_context: could not build preview bytes (%s)", e)
        return None


# ---- process-lifetime cache --------------------------------------------------------
# Mirrors `shape.agent.prompts.load_prompt`'s / `shape.tools.base._read_card_file`'s
# `@lru_cache` precedent. Keyed by the source file's identity (path+mtime+size), NOT
# content-hashed. This avoids one VLM call per VLRR retune re-run of the SAME image
# while still satisfying M1's "one [VLM] read before downstream analysis" per image,
# per process.

_PROFILE_CACHE: dict[tuple, ImageContextProfile] = {}


def _cache_key(img_data) -> Optional[tuple]:
    path = getattr(img_data, "source_path", None)
    if not path:
        return None
    try:
        st = os.stat(path)
        return (os.path.abspath(path), st.st_mtime, st.st_size)
    except OSError:
        return None


def read_image_context_profile(img_data, engine=None, use_cache: bool = True) -> ImageContextProfile:
    """The M1 preflight read: a loaded `ImageData` + channel metadata -> an
    `ImageContextProfile`.

    Conservative on every failure mode named in the BRIEF: no engine, an unreadable/
    degenerate image, a malformed VLM response, low VLM-reported confidence, or an
    internally-inconsistent response all fall back to `default_unknown_profile`.
    `img_data` is an already-loaded `shape.utils.image_data.ImageData`; this function
    never re-reads the file itself, so it stays agnostic of the loader).
    """
    if engine is None:                    # a system error, never a default profile
        raise NoEngineError("image_context: the preflight read: no vision engine is configured")
    key = _cache_key(img_data) if use_cache else None
    if key is not None and key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
    profile = _read_uncached(img_data, engine)
    if key is not None:
        _PROFILE_CACHE[key] = profile
    return profile


def _read_uncached(img_data, engine) -> ImageContextProfile:
    image_bytes = _preview_png_bytes(img_data)
    if image_bytes is None:
        return default_unknown_profile("unreadable image")

    from shape.agent.prompts import load_prompt

    channel_names = list(getattr(img_data, "channel_names", None) or ["grayscale"])
    channel_info = f"{len(channel_names)} channel(s), named: {', '.join(channel_names)}"
    template = load_prompt("image_preprocessor", "preflight")
    prompt = (template.format(channel_info=channel_info) if template else
              f"Describe this microscopy image's imaging modality, specimen, and "
              f"visible object classes. Channels: {channel_info}.")

    try:
        resp = engine.generate([prompt, image_bytes], response_format=ImageContextProfile)
        rec = resp.get("content") if isinstance(resp, dict) else resp
    except Exception as e:  # an execution error of the measurement it belongs to, never a default profile
        raise ExecutionError(f"image_context: preflight VLM call failed: {type(e).__name__}: {e}") from e

    if not hasattr(rec, "modality"):
        raise ExecutionError(f"image_context: malformed preflight answer ({type(rec).__name__})")
    if rec.confidence == "low":
        return default_unknown_profile("VLM reported low confidence")
    if not _is_internally_consistent(rec):
        return default_unknown_profile(
            "VLM response internally inconsistent (modality vs channel_roles)")
    return rec
