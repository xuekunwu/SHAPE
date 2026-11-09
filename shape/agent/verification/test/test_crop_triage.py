"""Task 041: revives the 019 crop-triage vision judge as `CropVerifier`'s LLM Recommend
seam. Co-located with `crop_triage.py` (committed -- AGENTS.md sec 6 P5; 019's own logic
tests were run ad hoc, never committed -- this closes that regression-net gap too).

Covers:
  - `triage_crop`/`triage_crops`: keep-bias + confidence gate, graceful fallback
    (no engine / unreadable / error), tighten-only + sample-cap (019's core logic).
  - `adapt_crop`/`crop_qc`: `crop_path` threading + `vision_reject` exclusion.
  - `vision_recommend_crop`: the Recommend-phase merge -- tighten-only (never loosens,
    never skips a notch), deterministic fallback on no-engine/no-crops/no-reject/failure.
  - `CropVerifier`: the `uses_llm` wiring + the PARITY guarantee (no `llm_engine` ->
    byte-identical to `main`'s deterministic-only path, load-bearing per the brief).

Task 038 restructure (human-locked, v7 the cluster finding -- `cell_00994`, base commit
25070bc): sending EVERY crop (262/262) to the judge cost 12.8M tokens for ~0 net
geometry gain. Task 038 v9 (a FOLLOW-UP round, this file's CURRENT state): a v8 the cluster
run confirmed the geometry gain stayed ~0 even after the v7 fix (square-tight cannot
isolate an adjacent neighbour) -- THREE further, coherent fixes, tested in their own
dedicated sections below:
  - OFF-CENTER geometry stays 100% DATA-DRIVEN and fully DECOUPLED from vision (v7,
    unchanged) -- `CropVerifier.adjudicate` reads `centroid_offset` directly, no judge
    involved, no engine needed.
  - DOUBLET geometry is now VLM-CONFIRMED (v9, a deliberate reversal of v7's pure-data
    doublet trigger): the data still DETECTS the candidate (`n_complete_cells>=2`), but
    the VLM -- seeing the crop GROUNDED with its target mask outlined -- decides
    tighten (a NEW `margin_mode="mask_bbox"` crop -- in v9 a rectangle tight to the
    target's own bbox, since the human rule of 2026-09-30 the square around it, every crop
    being square) or leave. See
    "GEOMETRY (task 038 v9 restructure)" below.
  - REJECT: the mask DATA SCREENS which crops are even worth asking about
    (`crop_triage._vlm_candidate_reasons`, now RELAXED and covering doublet UNION
    cut_off UNION empty_debris); a non-candidate is auto-kept, NEVER sent. A candidate
    gets a TARGETED confirm instruction (`_scenario_confirm_text`) naming exactly its
    own flagged scenario(s), grounded on the SAME mask overlay, and the judge's verdict
    on it is AUTHORITATIVE -- no second-guessing data check after it answers. See
    "DATA-SCREENED, VLM-DECIDED, MASK-GROUNDED reject+doublet triage" below.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

from shape.agent.verification.crop_triage import (
    ABPositionVerdict,
    ABVerdict,
    MaskCheckPositionVerdict,
    MaskCheckVerdict,
    CropTriageVerdict,
    MontagePositionVerdict,
    MontageVerdict,
    S1PositionVerdict,
    S1Verdict,
    _apply_ab_verdicts,
    _apply_mask_check_verdicts,
    _apply_montage_verdicts,
    _apply_s1_verdicts,
    _build_ab_montage,
    _build_mask_check_montage_prompt,
    _build_montage,
    _build_s1_montage_prompt,
    _crop_identifier,
    _judge_reject_montages,
    _montages_dir_for,
    _quant_evidence,
    _MONTAGE_CELL_PX,
    ab_recommend_crop,
    mask_check_recommend_crop,
    crop_kept,
    s1_recommend_crop,
    tighten_for_confirmed_rejects,
    triage_crop,
    triage_crops,
    vision_recommend_crop,
)
from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.verifiers.crop import (
    CropVerifier,
    _build_crop_decision_trace,
    _gate_arms,
    _gate_candidate,
    _gate_image_stats,
    _target_mask_area,
    _target_mask_std,
    _target_solidity,
    adapt_crop,
    crop_qc,
    is_crop_valid,
)
from shape.utils.schemas import VLRRResult


def _fabricate_montage_verdict(prompt: str, verdict: str, confidence: str, problem: str) -> MontageVerdict:
    """Test-only helper (task 038 montage round): fabricates a plausible `MontageVerdict`
    from a canned verdict/confidence/problem, applied UNIFORMLY to every labeled position
    the real `_build_montage_prompt` actually printed (a stub obviously cannot literally
    SEE the montage image, so it reads back the same "Position N:" list a real judge
    reads from the prompt text instead). `crop_id` is deliberately left blank -- an empty
    string skips `_apply_montage_verdicts`' cross-check (falsy), so a stub never
    spuriously logs a mismatch; the dedicated mismatch test below builds its own verdict
    by hand instead."""
    indices = [int(i) for i in re.findall(r"Position (\d+):", prompt)]
    effective_problem = problem if verdict == "reject" else "none"
    return MontageVerdict(verdicts=[
        MontagePositionVerdict(index=i, crop_id="", problem=effective_problem,
                               severity="severe" if effective_problem != "none" else "none",
                               confidence=confidence)
        for i in indices
    ])


class _ABStubEngine:
    """No-API stand-in for the task 038 v13 BLINDED A/B engine: returns a fixed
    `ABVerdict` ("left_better"/"right_better"/"abstain") applied UNIFORMLY to every
    labeled position the real `_build_ab_montage_prompt` actually printed -- reads
    back the same "Position N:" list a real judge reads from the prompt text (mirrors
    `_fabricate_montage_verdict`'s own test-stub pattern for the reject channel).
    `crop_id` left blank -- skips `_apply_ab_verdicts`' own cross-check (falsy). This
    stub cannot know which side the candidate randomly landed on (that mapping is
    internal to `_build_ab_montage`) -- tests that need a DETERMINISTIC de-
    randomization outcome use `_build_ab_montage`/`_apply_ab_verdicts` directly with a
    `_FixedRng` instead of this stub."""

    def __init__(self, verdict: str = "left_better", confidence: str = "high"):
        self._verdict = verdict
        self._confidence = confidence
        self.calls: list = []

    def generate(self, content, response_format=None, **kwargs):
        self.calls.append(content)
        if response_format is ABVerdict:
            indices = [int(i) for i in re.findall(r"Position (\d+):", content[0])]
            return {"content": ABVerdict(verdicts=[
                ABPositionVerdict(index=i, crop_id="", verdict=self._verdict,
                                  confidence=self._confidence)
                for i in indices
            ])}
        raise RuntimeError("_ABStubEngine: unexpected response_format")


class _FixedRng:
    """A deterministic stand-in for `_build_ab_montage`'s own `rng` param (task 038
    v13) -- `.random()` always returns the SAME fixed value, forcing the candidate to
    a KNOWN side every tile (`< 0.5` -> left, per `_build_ab_montage`'s own check), so
    a test can assert the L/R de-randomization mapping precisely instead of depending
    on true randomness."""

    def __init__(self, value: float):
        self._value = value

    def random(self) -> float:
        return self._value


class _S1StubEngine:
    """No-API stand-in for the task 038 Round B S1 engine: returns a fixed `S1Verdict`
    ("cell_present"/"target_region_empty"/"abstain") applied UNIFORMLY to every
    labeled position the real `_build_s1_montage_prompt` actually printed (mirrors
    `_fabricate_montage_verdict`/`_ABStubEngine`'s own established test-stub pattern
    for the reject/A-B channels). `crop_id` left blank -- skips `_apply_s1_verdicts`'
    own cross-check (falsy)."""

    def __init__(self, verdict: str = "cell_present", confidence: str = "high"):
        self._verdict = verdict
        self._confidence = confidence
        self.calls: list = []

    def generate(self, content, response_format=None, **kwargs):
        self.calls.append(content)
        if response_format is S1Verdict:
            indices = [int(i) for i in re.findall(r"Position (\d+):", content[0])]
            return {"content": S1Verdict(verdicts=[
                S1PositionVerdict(index=i, crop_id="", verdict=self._verdict,
                                  confidence=self._confidence)
                for i in indices
            ])}
        raise RuntimeError("_S1StubEngine: unexpected response_format")


class _MaskCheckStubEngine:
    """No-API stand-in for the task 058 M3 mask-check engine: returns a fixed `MaskCheckVerdict`
    ("cell_present"/"target_region_empty"/"wrong_object"/"merged"/"abstain") applied
    UNIFORMLY to every labeled position the real `_build_mask_check_montage_prompt`
    actually printed -- mirrors `_S1StubEngine`'s own established test-stub pattern
    (the channel this one replaces). `crop_id` left blank -- skips
    `_apply_mask_check_verdicts`' own cross-check (falsy)."""

    def __init__(self, verdict: str = "cell_present", confidence: str = "high"):
        self._verdict = verdict
        self._confidence = confidence
        self.calls: list = []

    def generate(self, content, response_format=None, **kwargs):
        self.calls.append(content)
        if response_format is MaskCheckVerdict:
            indices = [int(i) for i in re.findall(r"Position (\d+):", content[0])]
            return {"content": MaskCheckVerdict(verdicts=[
                MaskCheckPositionVerdict(index=i, crop_id="", verdict=self._verdict,
                                    confidence=self._confidence)
                for i in indices
            ])}
        raise RuntimeError("_MaskCheckStubEngine: unexpected response_format")


class _MaskCheckPartialStubEngine:
    """No-API stand-in for task 058 M8's own QC-reviewer probe: returns an
    `MaskCheckVerdict` covering ONLY `answer_indices` (default: none) regardless of how
    many positions the real `_build_mask_check_montage_prompt` actually printed --
    `MaskCheckVerdict` explicitly permits this SUBSET shape (`answer_indices=(0,)` drives
    the "sent but omitted" case; `answer_indices=()` drives a successful call that
    returns an empty verdict list). `_MaskCheckStubEngine`'s own UNIFORM-verdict stub
    cannot reach either case."""

    def __init__(self, answer_indices: tuple[int, ...] = (),
                verdict: str = "cell_present", confidence: str = "high"):
        self._answer_indices = answer_indices
        self._verdict = verdict
        self._confidence = confidence
        self.calls: list = []

    def generate(self, content, response_format=None, **kwargs):
        self.calls.append(content)
        if response_format is MaskCheckVerdict:
            return {"content": MaskCheckVerdict(verdicts=[
                MaskCheckPositionVerdict(index=i, crop_id="", verdict=self._verdict,
                                    confidence=self._confidence)
                for i in self._answer_indices
            ])}
        raise RuntimeError("_MaskCheckPartialStubEngine: unexpected response_format")


class _StubEngine:
    """No-API stand-in for the multimodal engine: returns a fixed, schema-valid
    `CropTriageVerdict` for every crop it is asked to judge (the DOUBLET channel, still
    one call per crop). `doublet` (task 038 v9 restructure, default "leave"): the
    SEPARATE geometry-confirmation axis, orthogonal to verdict/problem -- lets a test
    drive a doublet CANDIDATE's own confirm/deny.

    Task 038 montage round: when asked for a `MontageVerdict` instead (the REJECT
    channel, now batched), returns the SAME canned verdict/confidence/problem applied to
    every labeled position in the montage (`_fabricate_montage_verdict`) -- lets the
    EXISTING per-crop reject fixtures below keep exercising real reject/rescue behavior
    through the new batched mechanism, unchanged in spirit."""

    def __init__(self, verdict="keep", confidence="high", problem="none", doublet="leave",
                raise_error=False):
        self._verdict, self._confidence, self._problem = verdict, confidence, problem
        self._doublet = doublet
        self._raise = raise_error

    def generate(self, content, response_format=None, **kwargs):
        if self._raise:
            raise RuntimeError("engine failure")
        if response_format is MontageVerdict:
            return {"content": _fabricate_montage_verdict(
                content[0], self._verdict, self._confidence, self._problem)}
        rec = CropTriageVerdict(verdict=self._verdict, confidence=self._confidence,
                                problem=self._problem, doublet=self._doublet, reasoning="stub")
        return {"content": rec}


def _crop_image(tmp_path, name="cell.png", **fields) -> dict:
    """A crop dict pointing at a real (tiny) on-disk image, so `_crop_bytes` can read it."""
    from skimage.io import imsave
    path = tmp_path / name
    imsave(str(path), (np.random.rand(16, 16) * 255).astype(np.uint8))
    crop = {"path": str(path), "mask_containment": 1.0, "centroid_offset": 0.0}
    crop.update(fields)
    return crop


# ---------------------------------------------------------------------------
# triage_crop / triage_crops (019's core logic, ported to a committed test)
# ---------------------------------------------------------------------------

def test_triage_crop_high_confidence_reject(tmp_path):
    crop = _crop_image(tmp_path)
    reject, verdict = triage_crop(_StubEngine(verdict="reject", confidence="high",
                                              problem="cut_off"), crop)
    assert reject is True
    assert verdict.problem == "cut_off"


def test_triage_crop_low_confidence_reject_is_kept(tmp_path):
    """Confidence-gated: a LOW-confidence reject does not reject (keep-bias)."""
    crop = _crop_image(tmp_path)
    reject, verdict = triage_crop(_StubEngine(verdict="reject", confidence="low"), crop)
    assert reject is False


def test_triage_crop_keep_verdict_is_kept(tmp_path):
    crop = _crop_image(tmp_path)
    reject, verdict = triage_crop(_StubEngine(verdict="keep", confidence="high"), crop)
    assert reject is False


def test_triage_crop_no_engine_is_noop(tmp_path):
    crop = _crop_image(tmp_path)
    assert triage_crop(None, crop) == (False, None)


def test_triage_crop_unreadable_path_is_noop():
    assert triage_crop(_StubEngine(), {"path": "/does/not/exist.png"}) == (False, None)


def test_triage_crop_engine_error_falls_back_to_keep(tmp_path):
    crop = _crop_image(tmp_path)
    assert triage_crop(_StubEngine(raise_error=True), crop) == (False, None)


def test_triage_crops_tighten_only_never_rescues_metric_invalid(tmp_path):
    """A crop the metric gate already rejected (truncated) is never re-examined."""
    valid_crop = _crop_image(tmp_path, "valid.png")
    invalid_crop = _crop_image(tmp_path, "invalid.png", mask_containment=0.5)  # truncated
    crops = [valid_crop, invalid_crop]
    summary = triage_crops(_StubEngine(verdict="reject", confidence="high", problem="cut_off"), crops)
    assert summary == {"judged": 1, "rejected": 1, "engine": True}
    assert valid_crop.get("vision_reject") is True
    assert "vision_reject" not in invalid_crop


def test_triage_crops_sample_cap_evenly_spaced(tmp_path):
    crops = [_crop_image(tmp_path, f"c{i}.png") for i in range(4)]
    summary = triage_crops(_StubEngine(verdict="reject", confidence="high"), crops, sample_cap=2)
    assert summary["judged"] == 2


def test_triage_crops_no_engine_is_noop(tmp_path):
    crops = [_crop_image(tmp_path)]
    assert triage_crops(None, crops) == {"judged": 0, "rejected": 0, "engine": False}
    assert "vision_reject" not in crops[0]


def test_crop_kept_requires_both_metric_and_vision():
    good = {"mask_containment": 1.0, "centroid_offset": 0.0}
    assert crop_kept(good) is True
    assert crop_kept(dict(good, vision_reject=True)) is False


# ---------------------------------------------------------------------------
# adapt_crop / crop_qc wiring (crop_path threading + vision_reject exclusion)
# ---------------------------------------------------------------------------

def test_adapt_crop_carries_crop_path(tmp_path):
    meta = {"cell_metadata": [{
        "qc": {"mask_containment": 1.0, "occupancy": 0.5, "centroid_offset": 0.0,
              "border_touch": 0.0, "neighbor_contamination": 0.0},
        "crop_path": "/some/crop.png",
    }]}
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))
    crops, _ = adapt_crop(str(meta_path))
    assert crops[0]["crop_path"] == "/some/crop.png"


def test_adapt_crop_carries_qc_block_and_metadata_path(tmp_path):
    """Task 038 C1/C4 plumbing: `adapt_crop` also threads the raw `qc` block onto each crop
    dict (so `_quant_evidence`'s mask-based branch can actually fire on real-tool crops --
    the flattened fields alone do not preserve a nested `crop["qc"]`) and the metadata
    file's own resolved path into `context["metadata_path"]` (so the crop-stage Recommend
    can locate it to write the kept-crops manifest, task 038 C4)."""
    qc = {"mask_containment": 1.0, "occupancy": 0.5, "centroid_offset": 0.0,
         "border_touch": 0.0, "neighbor_contamination": 0.0}
    meta = {"cell_metadata": [{"qc": qc, "crop_path": "/some/crop.png"}]}
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))
    crops, context = adapt_crop(str(meta_path))
    assert crops[0]["qc"] == qc
    assert context["metadata_path"] == str(meta_path)


def test_adapt_crop_carries_mask_path(tmp_path):
    """Task 038 v11 FIX (human-locked, found inspecting the v10 round's own saved
    montages -- they came out GRAYSCALE): `adapt_crop` also threads `mask_path` (the
    per-cell mask file `tool.py` has written into the metadata since the v9 restructure,
    `crop_metadata["mask_path"]`) onto each crop dict. WITHOUT this,
    `crop_triage._target_mask_pixels` always returns None end-to-end (it reads
    `crop["mask_path"]`), so the montage's own colorful grounding overlay
    (`_crop_overlay_array`) silently falls back to plain grayscale for every REAL crop
    -- even though the overlay function itself was already correct and already tested
    with a mask (`test_crop_overlay_array_grounds_with_a_mask_when_available`, direct
    unit test, mask supplied by hand -- exactly why QC did not catch this: the isolated
    unit test never exercised `adapt_crop`'s own real threading). See
    `test_adapt_crop_to_montage_end_to_end_renders_colored_overlay` below for the full
    crop-metadata-json -> adapt_crop -> montage-build regression proof."""
    qc = {"mask_containment": 1.0, "occupancy": 0.5, "centroid_offset": 0.0,
         "border_touch": 0.0, "neighbor_contamination": 0.0}
    meta = {"cell_metadata": [{"qc": qc, "crop_path": "/some/crop.png",
                               "mask_path": "/some/mask.png"}]}
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))
    crops, _ = adapt_crop(str(meta_path))
    assert crops[0]["mask_path"] == "/some/mask.png"


def test_adapt_crop_to_montage_end_to_end_renders_colored_overlay(tmp_path):
    """Task 038 v11 FIX -- the END-TO-END regression proof for the exact gap that let a
    grayscale montage ship: crop-metadata-JSON (the REAL on-disk shape `tool.py`
    writes, `mask_path` included) -> `adapt_crop` (the REAL adapter, not a hand-built
    crop dict) -> `vision_recommend_crop` (the REAL production entry point
    `CropVerifier.adjudicate` calls) -> the montage bytes actually sent to the engine.
    Confirms the montage's own labeled cell is genuinely COLORED over the target
    region (R/G/B not all equal), i.e. the mask overlay actually rendered -- NOT
    grayscale, which is what the pre-fix `adapt_crop` (missing `mask_path`) silently
    shipped despite the overlay function itself working correctly in isolation."""
    import cv2
    from skimage.io import imsave

    tool_cache_dir = tmp_path / "tool_cache"
    tool_cache_dir.mkdir()
    crop_path = tool_cache_dir / "img1_cell_0000_crop.png"
    mask_path = tool_cache_dir / "img1_cell_0000_mask.png"
    imsave(str(crop_path), (np.random.rand(16, 16) * 255).astype(np.uint8))
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255   # covers the crop's own center
    imsave(str(mask_path), mask)

    qc = {"mask_containment": 1.0, "occupancy": 0.5, "centroid_offset": 0.0,
         "border_touch": 0.6, "neighbor_contamination": 0.0}   # border_touch=0.6 -> a
                                                                # genuine cut_off candidate,
                                                                # so it IS actually sent
    meta = {
        "execution_status": "success", "source_image_id": "img1", "group": "default",
        "cell_crops_paths": [str(crop_path)],
        "cell_metadata": [{"cell_id": "cellA", "qc": qc, "crop_path": str(crop_path),
                           "mask_path": str(mask_path)}],
    }
    meta_path = tool_cache_dir / "cell_crops_metadata_e2e0001.json"
    meta_path.write_text(json.dumps(meta))

    crops, _context = adapt_crop(str(meta_path))
    assert crops[0]["mask_path"] == str(mask_path)   # the fix itself, on the real path

    engine = _RecordingEngine(verdict="keep", confidence="high")
    vision_recommend_crop(engine, crops, _det())
    assert len(engine.calls) == 1
    _, montage_bytes = engine.calls[0]
    decoded = cv2.imdecode(np.frombuffer(montage_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    center = decoded[_MONTAGE_CELL_PX // 2, _MONTAGE_CELL_PX // 2]
    r, g, b = int(center[2]), int(center[1]), int(center[0])
    assert r != g or g != b   # genuinely COLORED over the target region, not grayscale


def test_crop_qc_excludes_vision_rejected_crop():
    crops = [
        {"mask_containment": 1.0, "centroid_offset": 0.0},
        {"mask_containment": 1.0, "centroid_offset": 0.0, "vision_reject": True},
    ]
    assert crop_qc(crops)["valid_crop_rate"] == 0.5


def test_crop_qc_no_vision_reject_key_is_unaffected():
    """Strict no-op: crops with no vision_reject key score exactly as before."""
    crops = [{"mask_containment": 1.0, "centroid_offset": 0.0} for _ in range(3)]
    assert crop_qc(crops)["valid_crop_rate"] == 1.0


def test_crop_qc_exempts_a_deliberately_tightened_crop_from_truncation_and_off_center():
    """Task 038 FIX round #1 (coordinator-directed): a DELIBERATELY tightened crop
    (`margin_mode=="mask_bbox"` AND a non-empty APPLIED `bbox_inset_px`) truncates the
    target (down to the retention floor) and off-centers it (a directional inset on
    the neighbour side only) BY DESIGN -- a real headless run showed `localize()`
    reporting exactly this intended effect as a WARNING/FAIL defect ("target cell
    touches/exceeds the crop border" / "off-center") on a cleanly-tightened image. This
    crop's own bad containment/offset must NOT count toward `truncation_fraction`/
    `off_center_fraction`; an UNTIGHTENED crop with the SAME bad numbers keeps the
    existing defect semantics, completely unchanged. (0.85: below the 0.90 retention
    floor, the truncation gate since 2026-10-01 -- a crop AT the floor is not truncated.)"""
    tightened = {"mask_containment": 0.85, "centroid_offset": 0.35,
                "margin_mode": "mask_bbox", "bbox_inset_px": {"right": 10}}
    untightened = {"mask_containment": 0.85, "centroid_offset": 0.35,
                   "margin_mode": "fraction", "bbox_inset_px": {}}
    metrics = crop_qc([tightened, untightened])
    assert metrics["truncation_fraction"] == 0.5    # only the untightened crop counts
    assert metrics["off_center_fraction"] == 0.5


def test_crop_qc_plain_mask_bbox_with_no_applied_inset_is_not_exempt():
    """The exemption is gated on a GENUINE applied inset, not just `margin_mode==
    "mask_bbox"` -- a PLAIN mask_bbox crop with NO inset (`bbox_inset_px=={}`, nothing
    was actually cut) is not exempt (defensive: a real plain mask_bbox crop's own
    containment/offset are already good by construction -- see
    `test_mask_bbox_is_inherently_centered_subsumes_recenter`, `tool.py`'s own test
    suite -- so this exact scenario should not arise in practice, but the exemption's
    own GATE must be verified directly, not just its usual no-op outcome). Its containment
    (0.85) is below the 0.90 retention floor, the truncation gate."""
    crop = {"mask_containment": 0.85, "centroid_offset": 0.35,
           "margin_mode": "mask_bbox", "bbox_inset_px": {}}
    metrics = crop_qc([crop])
    assert metrics["truncation_fraction"] == 1.0
    assert metrics["off_center_fraction"] == 1.0


# ---------------------------------------------------------------------------
# vision_recommend_crop (the Recommend-phase merge). Task 038 restructure: a crop is
# only ever SENT to the judge when the data screens it as a reject candidate
# (`border_touch=0.6` below -> `not_truncated=0.4 < 0.5`, CARD.md SS3) -- fixtures here
# carry that field so the StubEngine's own verdict is actually reachable; the data
# SCREEN itself has its own dedicated proofs in the "DATA-SCREENED, VLM-DECIDED" section
# further below.
# ---------------------------------------------------------------------------

def _det(decision="PASS", action="accept", score=0.9, notes="", param_delta=None):
    return VLRRResult(decision, score, {}, action=action, notes=notes, param_delta=param_delta)


def test_vision_recommend_crop_no_engine_returns_deterministic_unchanged():
    det = _det()
    assert vision_recommend_crop(None, [{"path": "x"}], det) is det


def test_vision_recommend_crop_no_crops_returns_deterministic_unchanged():
    det = _det()
    engine = _StubEngine(verdict="reject", confidence="high")
    assert vision_recommend_crop(engine, [], det) is det
    assert vision_recommend_crop(engine, None, det) is det


def test_vision_recommend_crop_no_rejects_returns_deterministic_unchanged(tmp_path):
    det = _det()
    crops = [_crop_image(tmp_path, border_touch=0.6)]   # a genuine candidate
    out = vision_recommend_crop(_StubEngine(verdict="keep", confidence="high"), crops, det)
    assert out is det


def test_vision_recommend_crop_tightens_pass_to_warning(tmp_path):
    det = _det(decision="PASS", action="accept", score=0.9)
    crops = [_crop_image(tmp_path, border_touch=0.6)]
    out = vision_recommend_crop(
        _StubEngine(verdict="reject", confidence="high", problem="cut_off"), crops, det)
    assert out.decision == "WARNING"
    assert out.scalar_score == det.scalar_score      # score untouched -- evidence, not a gate
    assert out.action == det.action                  # action untouched
    assert out.param_delta == det.param_delta
    assert "cut_off" in out.notes


def test_vision_recommend_crop_tightens_warning_to_fail(tmp_path):
    det = _det(decision="WARNING", action="retune_here", score=0.5,
              param_delta={"padding": "up"})
    crops = [_crop_image(tmp_path, border_touch=0.6)]
    out = vision_recommend_crop(
        _StubEngine(verdict="reject", confidence="high", problem="cut_off"), crops, det)
    assert out.decision == "FAIL"
    assert out.action == "retune_here"
    assert out.param_delta == {"padding": "up"}


def test_vision_recommend_crop_never_loosens_or_skips_past_fail(tmp_path):
    det = _det(decision="FAIL", action="conceded", score=0.1)
    crops = [_crop_image(tmp_path, border_touch=0.6)]
    out = vision_recommend_crop(
        _StubEngine(verdict="reject", confidence="high", problem="cut_off"), crops, det)
    assert out.decision == "FAIL"


# ---------------------------------------------------------------------------
# CropTriageVerdict.problem's enum + _VALID_REJECT_REASONS -- the CARD's exhaustive 2
# reject reasons (`cut_off`/`empty_debris`); a real run once showed the judge rejecting
# on off_center/target_dominance/doublet/saturated, which CARD.md SS2/SS4 explicitly
# forbid. Two layers: (1) the schema's own closed enum -- the model cannot even NAME an
# invalid reason; (2) `problem_gate` in `llm_recommend.judge_item` -- a runtime downgrade
# to keep, defense in depth for an engine that does not enforce structured output.
# ---------------------------------------------------------------------------

def test_crop_triage_verdict_problem_enum_excludes_off_center_and_dominance():
    """Layer 1 (structural): `CropTriageVerdict.problem` is a closed enum that does NOT
    include off_center/target_dominance/swamped/doublet/saturated -- the model literally
    cannot construct a verdict naming one of these as the reject reason."""
    import pytest as _pytest
    from pydantic import ValidationError

    for bad in ("off_center", "target_dominance", "dominance", "swamped", "low_dominance",
               "doublet", "saturated"):
        with _pytest.raises(ValidationError):
            CropTriageVerdict(verdict="reject", confidence="high", problem=bad,
                              reasoning="x")
    # the 2 CARD reasons + "none" all construct fine
    for ok in ("cut_off", "empty_debris", "none"):
        CropTriageVerdict(verdict="reject", confidence="high", problem=ok, reasoning="x")


def test_crop_triage_verdict_problem_enum_excludes_removed_blur_reason():
    """`severe_blur` is not in the taxonomy at all -- the model literally cannot
    construct a verdict naming it as the reject reason, mirroring the proof above."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CropTriageVerdict(verdict="reject", confidence="high", problem="severe_blur",
                          reasoning="x")


def test_valid_reject_reasons_is_exactly_the_card_2():
    """Only the two defects the cropper genuinely CANNOT fix by geometry. `doublet` is
    geometry's job (a data-driven tighten, never a reject); `saturated`/blur are
    acquisition issues, not this tool's call."""
    from shape.agent.verification.crop_triage import _VALID_REJECT_REASONS
    assert _VALID_REJECT_REASONS == {"cut_off", "empty_debris"}
    assert "doublet" not in _VALID_REJECT_REASONS
    assert "saturated" not in _VALID_REJECT_REASONS
    assert "off_center" not in _VALID_REJECT_REASONS
    assert "target_dominance" not in _VALID_REJECT_REASONS
    assert "severe_blur" not in _VALID_REJECT_REASONS
    assert "none" not in _VALID_REJECT_REASONS   # "none" is not itself a reject reason


def test_vision_recommend_crop_downgrades_off_center_reject_to_keep(tmp_path):
    """Layer 2 (runtime, defense in depth), task 038 montage round: even if an engine's
    response bypasses `MontageVerdict`'s own enum (e.g. a non-conforming engine that does
    not enforce structured output), `_apply_montage_verdicts`' `_VALID_REJECT_REASONS`
    gate downgrades a reject naming off_center to keep -- the deterministic verdict
    passes through UNCHANGED, exactly as if the engine had never rejected at all.
    `border_touch=0.6` makes this crop a genuine `cut_off` CANDIDATE (so it IS actually
    sent, to the reject-montage channel), independent of the off-topic reason the stub
    names."""
    class _NonConformingEngine:
        """Simulates an engine whose `response_format` enforcement is imperfect: returns
        a MontageVerdict-shaped duck type (NOT a real, pydantic-validated `MontageVerdict`)
        naming a reason outside the CARD's 2 -- `_apply_montage_verdicts` reads
        `.index`/`.problem`/`.confidence` via plain attribute access, so this reaches the
        `_VALID_REJECT_REASONS` check without ever going through pydantic validation.
        `index=0` -- the single crop below is the ONLY (and therefore position-0) entry
        in its own montage."""
        def generate(self, content, response_format=None, **kwargs):
            class _Pos:
                index = 0
                crop_id = ""
                problem = "off_center"
                severity = "severe"
                confidence = "high"
            class _Rec:
                verdicts = [_Pos()]
            return {"content": _Rec()}

    det = _det(decision="PASS", action="accept", score=0.9)
    crops = [_crop_image(tmp_path, border_touch=0.6)]
    out = vision_recommend_crop(_NonConformingEngine(), crops, det)
    assert out.decision == "PASS"          # NOT tightened -- the reject was downgraded
    assert out is det                      # deterministic verdict passed through UNCHANGED
    assert "vision_reject" not in crops[0]  # never marked rejected


def test_vision_recommend_crop_downgrades_target_dominance_reject_to_keep(tmp_path):
    """Same proof, target_dominance/swamped reason -- CARD.md SS4's OTHER explicitly
    forbidden reject reason."""
    class _NonConformingEngine:
        def generate(self, content, response_format=None, **kwargs):
            class _Pos:
                index = 0
                crop_id = ""
                problem = "target_dominance"
                severity = "severe"
                confidence = "high"
            class _Rec:
                verdicts = [_Pos()]
            return {"content": _Rec()}

    det = _det(decision="PASS", action="accept", score=0.9)
    crops = [_crop_image(tmp_path, border_touch=0.6)]
    out = vision_recommend_crop(_NonConformingEngine(), crops, det)
    assert out.decision == "PASS"
    assert out is det
    assert "vision_reject" not in crops[0]


def test_vision_recommend_crop_still_rejects_on_a_valid_card_reason(tmp_path):
    """Control: the SAME non-conforming-engine shape, but with a VALID CARD reason
    (cut_off), still tightens/rejects normally -- proves the montage's own
    `_VALID_REJECT_REASONS` gate discriminates, not a blanket no-op."""
    class _NonConformingEngine:
        def generate(self, content, response_format=None, **kwargs):
            class _Pos:
                index = 0
                crop_id = ""
                problem = "cut_off"
                severity = "severe"
                confidence = "high"
            class _Rec:
                verdicts = [_Pos()]
            return {"content": _Rec()}

    det = _det(decision="PASS", action="accept", score=0.9)
    crops = [_crop_image(tmp_path, border_touch=0.6)]
    out = vision_recommend_crop(_NonConformingEngine(), crops, det)
    assert out.decision == "WARNING"
    assert crops[0].get("vision_reject") is True


def test_vision_recommend_crop_internal_failure_falls_back(tmp_path, monkeypatch):
    """A failure inside the SHARED judging step itself (distinct from a per-crop failure,
    which `llm_recommend.judge_item` already catches) must not break the loop -- proves the
    cropper's thin adapter correctly delegates to `llm_recommend.judge_items`'s OWN
    graceful-degrade contract."""
    det = _det()
    crops = [_crop_image(tmp_path)]

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("shape.agent.verification.llm_recommend.judge_items", _boom)
    assert vision_recommend_crop(_StubEngine(), crops, det) is det


# ---------------------------------------------------------------------------
# CropVerifier wiring + the no-engine PARITY guarantee (load-bearing)
# ---------------------------------------------------------------------------

def test_crop_verifier_uses_llm_is_true():
    assert CropVerifier.uses_llm is True
    assert CropVerifier.llm_engine is None


def test_crop_verifier_quantify_stashes_crops_in_context():
    verifier = CropVerifier()
    crops = [{"mask_containment": 1.0, "centroid_offset": 0.0}]
    context = {}
    verifier.quantify(crops, context)
    assert context["crops"] is crops


def _clean_crops(n=10):
    return [{"mask_containment": 1.0, "centroid_offset": 0.0} for _ in range(n)]


def _truncated_crops():
    crops = _clean_crops(9)
    crops.append({"mask_containment": 0.5, "centroid_offset": 0.0})   # 1/10 truncated
    return crops


def test_no_engine_adjudicate_pass_unchanged():
    """PARITY (load-bearing): no llm_engine -> byte-identical to `main`'s deterministic PASS.
    These crops carry no `cell_id`, so the data-driven geometry step (task 038
    restructure) also has nothing to key an override on -- fully inert here too."""
    verifier = CropVerifier()   # no llm_engine wired -> deterministic-only
    crops = _clean_crops()
    context = {}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    out = verifier.adjudicate(metrics, locus, triggered, context)
    assert out.decision == "PASS"
    assert out.action == "accept"
    assert out.scalar_score == 1.0
    assert out.param_delta is None
    assert out.notes == ""


def test_no_engine_adjudicate_truncation_is_note_only():
    """`truncation` is NOTE-ONLY, mirroring `off_center` -- padding is already INERT in
    tight mode (`effective_margin`'s tight branch never reads `margin`), so `adjudicate`
    no longer proposes it. `triggered`/`expected_effect`/the QC notes text still surface
    (useful evidence for a human/the trace); only `param_delta`/`action` change -> no
    scalar param is proposed here either way. Geometry (task 038 restructure) has no
    `cell_id` to key on for these crops, so it stays inert too -> `action` stays
    "accept"."""
    verifier = CropVerifier()
    crops = _truncated_crops()
    context = {}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    out = verifier.adjudicate(metrics, locus, triggered, context)
    assert triggered == ["truncation"]
    assert out.action == "accept"
    assert out.param_delta is None
    assert out.expected_effect == {"mean_containment": "up"}
    assert "target cell touches/exceeds the crop border (incomplete cell)" in out.notes
    assert "Target cells are truncated" in out.notes
    assert out.decision == "WARNING"   # rate = 9/10 = 0.9 > 0.80


def test_engine_wired_but_no_rejects_matches_deterministic(tmp_path):
    verifier = CropVerifier()
    verifier.llm_engine = _StubEngine(verdict="keep", confidence="high")
    crops = [_crop_image(tmp_path, f"c{i}.png") for i in range(5)]
    context = {}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    out = verifier.adjudicate(metrics, locus, triggered, context)
    assert out.decision == "PASS"
    assert out.action == "accept"


def test_engine_wired_tightens_end_to_end(tmp_path):
    """Consulted at the Recommend phase (adjudicate) -- evidence the loop weighs, not a
    hard gate: action stays the deterministic gate's own; only decision tightens.
    Every crop here carries `border_touch=0.6` (not_truncated=0.4 < the 0.8 screening
    floor), making all 5 genuine `cut_off` CANDIDATES -- so all 5 are actually sent.

    `scalar_score` (task 038 v13, human-directed) is now the PARETO-CONSISTENT
    objective (`CropVerifier._pareto_scalar`, not `mean_crop_quality_score_proxy` --
    disqualified as an acceptance authority) -- none of these bare crops set
    `cell_id`/`target_dominance`/`neighbor_contamination`, so each falls back to
    `1.0 - 0.0 = 1.0` (none vetoed, none floor-violating): `scalar_score == 1.0`,
    UNTOUCHED by the REJECT channel either way (a SEPARATE VLM channel, its own
    schema, never writes `scalar_score` itself). Task 058 M3 round 2: (B) REJECT's
    own candidate screening is off by default (`enable_reject_channel`, mirrors
    `enable_ab_channel`) -- this test exercises the channel directly, so it opts
    back in explicitly."""
    verifier = CropVerifier()
    verifier.llm_engine = _StubEngine(verdict="reject", confidence="high", problem="cut_off")
    verifier.enable_reject_channel = True
    crops = [_crop_image(tmp_path, f"c{i}.png", border_touch=0.6) for i in range(5)]
    context = {}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    out = verifier.adjudicate(metrics, locus, triggered, context)
    assert triggered == []                # the deterministic gate saw nothing wrong
    assert out.decision == "WARNING"       # vision tightened PASS -> WARNING
    assert out.action == "accept"          # action untouched
    assert out.scalar_score == pytest.approx(1.0)
    assert "cut_off" in out.notes
    assert all(c.get("vision_reject") for c in crops)


# ---------------------------------------------------------------------------
# KEEP-BEST (task 038 v9 keep-best fix, coordinator-traced, human-locked): a real v9
# cluster run showed the 3 densest images -- the ones where the agent's own
# `mask_bbox` doublet repair did the MOST good -- all CONCEDED back to their NAIVE
# crops (target_dominance +0.22 / crop_quality_score +0.0153 genuinely gained, then
# silently discarded). Root cause: the SHARED `StepVerifier._best` keep-best ranks an
# unjudged trial by `-scalar_score`; the OLD `adjudicate()` scalar_score
# ({valid_rate, containment, centering}) never included `target_dominance` at all, so
# a mask_bbox trial that raised dominance left scalar_score ~UNCHANGED -- a tie (or
# near-tie) that Python's stable `min()` breaks by keeping the FIRST (naive) trial,
# every time. The fix makes `scalar_score` the SAME `mean_crop_quality_score_proxy`
# the geometry triggers actually optimize (task 038 v9, above) -- these tests prove
# `StepVerifier._best` (the SHARED, unmodified keep-best mechanism) now genuinely
# selects the higher-quality trial.
# ---------------------------------------------------------------------------

def _crop_metrics_vres(crops: list[dict]) -> VLRRResult:
    """One batch of synthetic crop-metric dicts through the REAL `CropVerifier.
    quantify -> localize -> adjudicate` (no engine, no tool, no StepVerifier) --
    returns the REAL VLRRResult `adjudicate` produces for exactly these crops' own
    metrics, so `scalar_score` is the production value, not a hand-computed stand-in."""
    verifier = CropVerifier()
    context = {}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    return verifier.adjudicate(metrics, locus, triggered, context)


def test_keep_best_selects_the_higher_crop_quality_trial_not_the_first():
    """VALIDATE the v9 keep-best fix (STILL holds under the v13 Pareto scalar): a
    doublet's NAIVE crop (target_dominance diluted by a tolerated neighbour, e.g. 0.5)
    vs the SAME cell's CONFIRMED mask_bbox-tightened crop (neighbour excluded,
    target_dominance=1.0) -- `mask_containment`/`centroid_offset`/`border_touch` are
    IDENTICAL between the two, so ONLY `target_dominance` differs. Task 038 v13,
    human-directed: `scalar_score` is now `CropVerifier._pareto_scalar` (`_WEIGHTS`-
    composite DISQUALIFIED as an acceptance authority) -- its own per-crop objective
    IS `target_dominance` (when present) for a non-vetoed, floor-satisfying crop, so
    the tightened trial's genuine dominance gain still produces a real score
    difference, and `StepVerifier._best` (the shared, UNMODIFIED keep-best mechanism)
    must still select the higher-quality (tightened) trial, not the first (naive)
    one -- the EXACT v9 the cluster root cause this test isolates."""
    naive_crops = [{"mask_containment": 1.0, "centroid_offset": 0.0, "border_touch": 0.0,
                    "target_dominance": 0.5}]
    tightened_crops = [{"mask_containment": 1.0, "centroid_offset": 0.0, "border_touch": 0.0,
                        "target_dominance": 1.0}]

    vres_naive = _crop_metrics_vres(naive_crops)
    vres_tightened = _crop_metrics_vres(tightened_crops)
    # sanity: these two trials really do differ ONLY on target_dominance -- both are
    # otherwise-identical PASS/accept verdicts (no truncation/off-center trigger).
    assert vres_naive.decision == vres_tightened.decision == "PASS"
    # task 038 v13: scalar_score == target_dominance directly (no cell_id -> never
    # vetoed; mask_containment=1.0 -> never floor-violating) for a single-crop batch.
    assert vres_naive.scalar_score == pytest.approx(0.5)
    assert vres_tightened.scalar_score == pytest.approx(1.0)
    assert vres_tightened.scalar_score > vres_naive.scalar_score

    history = [({"trial": 1}, vres_naive, "naive_result"),
              ({"trial": 2}, vres_tightened, "tightened_result")]
    resolution = StepVerifier()._best(history, {}, "satisfied", reason="test")
    assert resolution.result == "tightened_result"


def test_clean_crop_batch_accepts_on_first_attempt_through_step_verifier(tmp_path):
    """Whole-loop regression, no llm_engine wired: mirrors
    `test_segmentation_golden_path.py`'s golden-path shape for the crop stage."""
    meta = {"cell_metadata": [
        {"qc": {"mask_containment": 1.0, "occupancy": 0.5, "centroid_offset": 0.0,
               "border_touch": 0.0, "neighbor_contamination": 0.0}}
        for _ in range(10)
    ]}
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))

    verifier = CropVerifier()   # no llm_engine wired -> deterministic-only
    sv = StepVerifier(max_iters=3)

    def adapter(result, upstream_ctx):
        return adapt_crop(result)

    def tool_runner(params):
        raise AssertionError("a clean crop batch must not need a retune/rerun")

    res = sv.resolve_stage(verifier, adapter, tool_runner,
                           initial_params={"padding": 25}, initial_result=str(meta_path))
    assert res.status == "satisfied"
    assert res.reason == "converged"
    assert res.verdict.decision == "PASS"
    assert res.verdict.action == "accept"
    assert len(res.trials) == 1


# ---------------------------------------------------------------------------
# DATA-SCREENED, VLM-DECIDED, MASK-GROUNDED reject+doublet triage (task 038 v9
# restructure, human-locked). The SHARED mechanism (`build_judge_prompt`/`judge_item`/
# `judge_items`) has its own generic-fixture tests in
# `test_llm_recommend_population_triage.py`; this section covers the CROPPER'S OWN
# contribution: which crops are even candidates (`_vlm_candidate_reasons` -- doublet
# UNION cut_off UNION empty_debris), and what happens once one is sent. Doublet-specific
# confirm/deny landing is covered in "GEOMETRY (task 038 v9 restructure)" below (it lands
# as a geometry override, read back by `CropVerifier.adjudicate`, not here).
# ---------------------------------------------------------------------------

class _RecordingEngine:
    """No-API stand-in that RECORDS the exact `content` it was called with, so a test can
    assert the fused prompt + image were both actually sent -- and how many times.

    Task 038 montage round: also answers a `MontageVerdict` ask (the REJECT channel) the
    same way `_StubEngine` does (`_fabricate_montage_verdict`, applied uniformly to every
    labeled position it was actually asked about) -- `self.calls` still records the RAW
    `[prompt, montage_bytes]` content either way, regardless of what is returned."""

    def __init__(self, verdict="keep", confidence="high", problem="none"):
        self._verdict, self._confidence, self._problem = verdict, confidence, problem
        self.calls: list = []

    def generate(self, content, response_format=None, **kwargs):
        self.calls.append(content)
        if response_format is MontageVerdict:
            return {"content": _fabricate_montage_verdict(
                content[0], self._verdict, self._confidence, self._problem)}
        rec = CropTriageVerdict(verdict=self._verdict, confidence=self._confidence,
                                problem=self._problem, reasoning="stub")
        return {"content": rec}


# ---------------------------------------------------------------------------
# RELAXED reject screening thresholds (task 038 v9 restructure): `_SEVERE_NOT_TRUNCATED_MAX`
# 0.5 -> 0.8, `_EMPTY_DEBRIS_FOREGROUND_MAX` 0.02 -> 0.2 -- CARD.md SS3 notes these two
# flag ~0% of crops on real NeurIPS22 data; `doublet` (~7%) is the real VLM load now.
# ---------------------------------------------------------------------------

def test_relaxed_thresholds_match_the_v9_values():
    from shape.agent.verification.crop_triage import (
        _EMPTY_DEBRIS_FOREGROUND_MAX,
        _SEVERE_NOT_TRUNCATED_MAX,
    )
    assert _SEVERE_NOT_TRUNCATED_MAX == 0.8
    assert _EMPTY_DEBRIS_FOREGROUND_MAX == 0.2


def test_relaxed_cut_off_floor_flags_a_crop_the_old_0_5_floor_would_not(tmp_path):
    """not_truncated=0.75 (border_touch=0.25) is NOT a candidate under the OLD 0.5
    floor, but IS one under the RELAXED 0.8 floor."""
    from shape.agent.verification.crop_triage import _vlm_candidate_reasons
    crop = _crop_image(tmp_path, border_touch=0.25)
    assert "cut_off" in _vlm_candidate_reasons(crop)


def test_relaxed_cut_off_floor_still_excludes_a_crop_above_it(tmp_path):
    from shape.agent.verification.crop_triage import _vlm_candidate_reasons
    crop = _crop_image(tmp_path, border_touch=0.1)   # not_truncated = 0.9, above 0.8
    assert "cut_off" not in _vlm_candidate_reasons(crop)


def test_relaxed_empty_debris_floor_flags_a_crop_the_old_0_02_floor_would_not(tmp_path):
    """foreground_fraction=0.15 is NOT a candidate under the OLD 0.02 floor, but IS one
    under the RELAXED 0.2 floor."""
    from shape.agent.verification.crop_triage import _vlm_candidate_reasons
    crop = _crop_image(tmp_path, qc={"foreground_fraction": 0.15})
    assert "empty_debris" in _vlm_candidate_reasons(crop)


def test_relaxed_empty_debris_floor_still_excludes_a_crop_above_it(tmp_path):
    from shape.agent.verification.crop_triage import _vlm_candidate_reasons
    crop = _crop_image(tmp_path, qc={"foreground_fraction": 0.25})
    assert "empty_debris" not in _vlm_candidate_reasons(crop)


# ---------------------------------------------------------------------------
# THE GROUNDING FIX (task 038 v9 restructure). Task 038 v10: geometry has NO VLM channel
# at all any more, so grounding now applies ONLY to REJECT candidates, exclusively via
# the montage's own `_crop_overlay_array` -- there is no more per-crop, single-image
# `vision_recommend_crop` grounding call to test at that level (the earlier doublet
# channel that used to exercise it is gone); the array-level fallback contract is
# covered directly below instead.
# ---------------------------------------------------------------------------

def test_crop_overlay_array_grounds_with_a_mask_when_available(tmp_path):
    """The MONTAGE channel's own grounding fallback (task 038 montage round,
    `_crop_overlay_array`, reused by `_build_montage`) -- this tool's ONLY remaining
    grounding consumer since task 038 v10 removed the doublet channel: WITH a readable
    `mask_path`, the overlaid array differs from a plain grayscale->RGB conversion of
    the same raw pixels (the mask tint changed real pixel values); WITHOUT one, it
    gracefully degrades to the plain conversion instead."""
    from skimage.io import imread, imsave

    from shape.agent.verification.crop_triage import _crop_overlay_array
    from shape.utils.segmentation_qc import to_display_uint8

    mask_path = tmp_path / "mask2.png"
    imsave(str(mask_path), (np.ones((16, 16)) * 255).astype(np.uint8))
    crop_with_mask = _crop_image(tmp_path, "with_mask.png", mask_path=str(mask_path))
    overlaid = _crop_overlay_array(crop_with_mask)
    pixels = imread(crop_with_mask["path"])
    plain = to_display_uint8(pixels)
    assert overlaid is not None
    assert not np.array_equal(overlaid[..., 1], plain)   # green channel tinted by the mask

    crop_no_mask = _crop_image(tmp_path, "no_mask.png")
    fallback = _crop_overlay_array(crop_no_mask)
    assert fallback is not None
    assert fallback.shape == (16, 16, 3)


def test_overlay_contour_is_semi_transparent_not_opaque(tmp_path):
    """Human directive (folded into task 038 Round B): the white contour must be
    SEMI-TRANSPARENT, not opaque -- an opaque contour completely destroys the pixel
    information exactly where a judge needs to look (S1: is there a cell UNDER the
    coloured region; S3: is the target cut off at an edge). Uses the SAME multi-
    channel/small-crop fixture the Round C channel-0 tests use (`_multi_channel_
    crop_image`, defined below -- NOT a large synthetic square): the real delivered
    crop sizes (8x12, 21x21, 28x13, 44x44) are all this small, where an opaque
    contour eats a much larger fraction of the boundary band once upscaled to the
    160px montage tile."""
    from skimage.io import imsave

    from shape.agent.verification.crop_triage import _crop_overlay_array

    crop = _multi_channel_crop_image(tmp_path, name="contour_test.png")
    mask_path = tmp_path / "contour_mask.png"
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255   # an INTERNAL boundary -- does not fill the whole crop
    imsave(str(mask_path), mask)
    crop["mask_path"] = str(mask_path)

    overlay = _crop_overlay_array(crop)
    assert overlay is not None

    contour_band = overlay[4, 4:12]   # the mask's own top edge (row 4 of the 4:12 box)
    # NOT a constant colour -- an OPAQUE line paints every contour pixel identically
    # (255, 255, 255) regardless of what was underneath; a semi-transparent one lets
    # the (varying, real) underlying signal show through, so distinct pixels differ.
    assert len(set(map(tuple, contour_band))) > 1
    # still genuinely a bright/whitish marking, not merely faint background noise
    assert contour_band.astype(np.int32).mean() > 150


# ---------------------------------------------------------------------------
# Task 038 Round C (coordinator-directed, a REAL bug -- confirmed against the actual
# delivered v15 crop files): `_crop_pixels` used to hardcode channel 0
# (`arr[..., 0]`), but NeurIPS22-CellSeg puts its real signal in a DIFFERENT channel
# per image -- `cell_00860`'s own delivered crops are ~92% empty on channel 0 (signal
# sits in channel 2). Every EARLIER fixture in this file is either single-channel or
# puts its signal in channel 0, so none of them could ever have caught this -- this
# fixture deliberately mirrors the REAL failure layout instead: channel 0 EMPTY,
# signal in channel 2.
# ---------------------------------------------------------------------------

def _multi_channel_crop_image(tmp_path, name="cell_multi.png") -> dict:
    """A crop dict pointing at a real, on-disk MULTI-CHANNEL image whose channel 0 is
    EMPTY (all zero) and whose signal lives in channel 2 only -- mirrors `cell_00860`'s
    own real delivered layout (measured on the v15 headless run: ch0 frac<10=0.90,
    ch1 frac<10=1.00/empty, ch2 frac<10=0.68/the signal). A ch0-only fixture (as every
    EXISTING crop-triage fixture in this file already is) cannot exercise this bug at
    all -- channel 0 IS the signal for those."""
    from skimage.io import imsave
    h, w = 16, 16
    ch0 = np.zeros((h, w), dtype=np.uint8)
    ch1 = np.zeros((h, w), dtype=np.uint8)
    ch2 = (np.random.RandomState(0).rand(h, w) * 200 + 55).astype(np.uint8)   # the real signal
    arr = np.stack([ch0, ch1, ch2], axis=-1)
    path = tmp_path / name
    imsave(str(path), arr)
    return {"path": str(path), "mask_containment": 1.0, "centroid_offset": 0.0}


def test_crop_pixels_max_projects_when_signal_is_not_in_channel_0(tmp_path):
    """`_crop_pixels` reads a real signal that lives OUTSIDE channel 0 -- the direct
    unit-level proof of the fix (max-projection, not `arr[..., 0]`)."""
    from shape.agent.verification.crop_triage import _crop_pixels
    crop = _multi_channel_crop_image(tmp_path)
    pixels = _crop_pixels(crop)
    assert pixels is not None
    assert pixels.ndim == 2                # the 2D contract downstream consumers expect
    assert pixels.max() > 100               # the signal (channel 2) survives the read
    assert not np.all(pixels < 10)          # NOT reduced to a near-black frame


def test_crop_overlay_array_shows_signal_when_channel_0_is_empty(tmp_path):
    """End to end, per the coordinator's own explicit instruction: driven through the
    REAL `_crop_pixels` -> `_crop_overlay_array` path (the montage tile a live VLM run
    would actually be shown), not just the pixel reader in isolation -- the tile
    genuinely contains the signal, not a black frame."""
    from shape.agent.verification.crop_triage import _crop_overlay_array
    crop = _multi_channel_crop_image(tmp_path, name="cell_multi2.png")
    overlay = _crop_overlay_array(crop)
    assert overlay is not None
    assert overlay.max() > 100              # the montage tile is not a black frame


def test_non_candidate_crop_never_sent_to_judge_and_is_auto_kept(tmp_path):
    """VALIDATE (a): a crop crossing NEITHER data-screen floor (`not_truncated>=0.5` AND
    `foreground_fraction>=0.02`) is NEVER sent to the judge at all -- auto-kept,
    regardless of what the judge would have said."""
    crop = _crop_image(tmp_path, border_touch=0.0, qc={"foreground_fraction": 0.5})
    engine = _RecordingEngine(verdict="reject", confidence="high", problem="cut_off")
    det = _det()
    out = vision_recommend_crop(engine, [crop], det)
    assert len(engine.calls) == 0          # never even asked
    assert out is det                      # the deterministic verdict, unchanged
    assert crop.get("vision_reject") is not True


def test_cut_off_candidate_is_sent_and_confirm_lands_the_reject(tmp_path):
    """VALIDATE (b): `not_truncated<0.5` makes a crop a `cut_off` CANDIDATE -- IS sent
    to the judge, and a CONFIRMING verdict lands the reject as-is, no second-guessing
    data check afterward."""
    det = _det(decision="PASS", action="accept", score=0.9)
    crop = _crop_image(tmp_path, border_touch=0.6)   # not_truncated=0.4 < 0.5
    engine = _RecordingEngine(verdict="reject", confidence="high", problem="cut_off")
    out = vision_recommend_crop(engine, [crop], det)
    assert len(engine.calls) == 1          # genuinely asked
    assert out.decision == "WARNING"
    assert crop.get("vision_reject") is True


def test_cut_off_candidate_is_sent_and_deny_keeps(tmp_path):
    """VALIDATE (b), the other half: the SAME candidate, but the judge DISAGREES
    (verdict=keep) -- the crop stays kept, landed as-is."""
    det = _det(decision="PASS", action="accept", score=0.9)
    crop = _crop_image(tmp_path, border_touch=0.6)
    engine = _RecordingEngine(verdict="keep", confidence="high", problem="none")
    out = vision_recommend_crop(engine, [crop], det)
    assert len(engine.calls) == 1
    assert out is det
    assert crop.get("vision_reject") is not True


def test_empty_debris_candidate_is_sent_and_confirm_lands_the_reject(tmp_path):
    """The `empty_debris` mirror of the cut_off proof above -- `foreground_fraction<0.02`
    makes a crop a candidate."""
    det = _det(decision="PASS", action="accept", score=0.9)
    crop = _crop_image(tmp_path, qc={"foreground_fraction": 0.01})
    engine = _RecordingEngine(verdict="reject", confidence="high", problem="empty_debris")
    out = vision_recommend_crop(engine, [crop], det)
    assert len(engine.calls) == 1
    assert out.decision == "WARNING"
    assert crop.get("vision_reject") is True


def test_mixed_population_only_candidates_are_sent(tmp_path):
    """A batch of 3: one clean crop (non-candidate), one cut_off candidate, one
    empty_debris candidate -- only the 2 candidates are ever sent to the judge; the clean
    crop is auto-kept without ever reaching it. Task 038 montage round: the 2 reject
    candidates are now BATCHED into a SINGLE montage call (not 2 separate per-crop
    calls) -- the whole point of the montage mechanism."""
    clean = _crop_image(tmp_path, "clean.png", border_touch=0.0, qc={"foreground_fraction": 0.5})
    cut_off = _crop_image(tmp_path, "cutoff.png", border_touch=0.6)
    empty = _crop_image(tmp_path, "empty.png", qc={"foreground_fraction": 0.01})
    engine = _RecordingEngine(verdict="keep", confidence="high")
    vision_recommend_crop(engine, [clean, cut_off, empty], _det())
    assert len(engine.calls) == 1          # ONE montage call, batching both candidates
    assert clean.get("vision_reject") is not True


def test_targeted_scenario_confirm_text_matches_flagged_reasons(tmp_path):
    """VALIDATE (d): the evidence actually sent matches exactly the reason(s)
    `_vlm_candidate_reasons` flagged for that crop. Task 038 montage round: cut_off/
    empty_debris go through the BATCHED montage prompt (`_build_montage_prompt`/
    `_montage_position_detail`). Task 038 v10: `doublet` is REMOVED from
    `_vlm_candidate_reasons` entirely (geometry is now pure-data, SS1/SS5) -- there is
    no third scenario to check any more."""
    # NOTE: CARD.md's own prose (appended to every montage prompt, `card_section`) now
    # legitimately discusses "cut_off candidate"/"empty_debris candidate" as CONCEPTS
    # (SS2's "THE MONTAGE FIX" bullet) -- a bare substring check on those phrases alone
    # would false-positive. The EXACT `_montage_position_detail` phrasing ("floor -> ...
    # candidate)") is unique to the per-position evidence line, so THAT is what
    # discriminates here.
    cut_off_crop = _crop_image(tmp_path, "a.png", border_touch=0.6)
    engine = _RecordingEngine()
    vision_recommend_crop(engine, [cut_off_crop], _det())
    prompt = engine.calls[0][0]
    assert "GRID MONTAGE" in prompt
    assert "floor -> cut_off candidate)" in prompt
    assert "floor -> empty_debris candidate)" not in prompt
    assert "not_truncated=0.400" in prompt

    empty_crop = _crop_image(tmp_path, "b.png", qc={"foreground_fraction": 0.01})
    engine2 = _RecordingEngine()
    vision_recommend_crop(engine2, [empty_crop], _det())
    prompt2 = engine2.calls[0][0]
    assert "floor -> empty_debris candidate)" in prompt2
    assert "floor -> cut_off candidate)" not in prompt2
    assert "foreground_fraction=0.0100" in prompt2

    both_crop = _crop_image(tmp_path, "e.png", border_touch=0.6,
                            qc={"foreground_fraction": 0.01})
    engine3 = _RecordingEngine()
    vision_recommend_crop(engine3, [both_crop], _det())
    # Both reject reasons flagged on the SAME crop -> ONE montage call, BOTH named.
    assert len(engine3.calls) == 1
    prompt3 = engine3.calls[0][0]
    assert "floor -> cut_off candidate)" in prompt3
    assert "floor -> empty_debris candidate)" in prompt3


def test_montage_prompt_carries_card_and_rescue_framing(tmp_path):
    """The MONTAGE channel's own prompt (task 038 montage round, `_build_montage_prompt`)
    -- built directly in this module, NOT via `llm_recommend.build_judge_prompt` (see
    the module docstring's (D) for the layering reason) -- still carries the real
    CARD.md text and a KEEP-BIASED RESCUE framing (the cell_00554 false-positive this
    round fixes), even though it is not the shared judge-prompt shape."""
    crop = _crop_image(tmp_path, border_touch=0.6)   # a cut_off candidate
    engine = _RecordingEngine()
    vision_recommend_crop(engine, [crop], _det())
    assert len(engine.calls) == 1
    prompt, montage_bytes = engine.calls[0]
    assert isinstance(prompt, str) and isinstance(montage_bytes, bytes) and len(montage_bytes) > 0
    assert "GRID MONTAGE" in prompt
    assert "RESCUE" in prompt
    # The montage judge carries only the judge's slice of the CARD (`_cropper_judge_rules`:
    # purpose + calibration + readout meanings), not the whole CARD (levers/loop/dev-notes it
    # does not use) -- a token-cost fix. The real CARD.md calibration text is still present:
    assert "0.004" in prompt and "swamped" in prompt                    # CARD's own calibration
    assert "QUANT evidence" not in prompt         # the shared build_judge_prompt is NOT used here
    # `_TRIAGE_RECONCILE_INSTRUCTION`'s own closing text (NOT "RECONCILE" the bare word --
    # CARD.md's own prose legitimately says "RECONCILE hooks" elsewhere, SS "5." heading)
    # is absent -- the montage prompt is NOT built via the shared `build_judge_prompt`.
    assert "CONTRADICTS must NOT drive a reject" not in prompt


def test_vision_recommend_crop_never_asks_about_off_center_recentering(tmp_path):
    """Task 038 v7/v10: OFF-CENTER geometry (recenter) is NEVER asked about at all -- it
    is decided entirely from data in `CropVerifier.adjudicate`, which since task 038 v10
    is true of ALL geometry (mask_bbox too, not just recenter -- SS1/SS5). Regression-
    guards this via the strongest possible proof: a crop that is severely off-center but
    flags NEITHER of the 2 VLM candidate reasons (cut_off/empty_debris) is not even sent
    to the judge at all -- so nothing is ever asked about it, geometry or otherwise
    (`_vlm_candidate_reasons` has no off-center or contamination scenario at all)."""
    crop = _crop_image(tmp_path, border_touch=0.0, centroid_offset=0.9,
                       qc={"foreground_fraction": 0.5})
    engine = _RecordingEngine()
    vision_recommend_crop(engine, [crop], _det())
    assert len(engine.calls) == 0


# ---------------------------------------------------------------------------
# Task 038 M-A: `_quant_evidence` reads the crop's own mask-based `qc` block when
# present (target_dominance/not_truncated/single_object/centered), falling back to pixel
# `{sharpness, target_dominance}` when absent. (A later round temporarily made this
# PIXEL-ONLY in all cases; a follow-up round REVERSED that -- human-locked: the mask is a
# GIVEN input, not a GT yardstick, so there is no circularity to avoid.)
# ---------------------------------------------------------------------------

def test_quant_evidence_mask_based_reads_qc_target_dominance_not_truncated_single_object_and_centered(tmp_path):
    """When the crop carries a mask-based `qc` block (adapt_crop's real-tool shape),
    target_dominance/not_truncated/single_object/centered are ALL read straight off it --
    NOT re-derived from pixels -- while sharpness stays pixel-computed (no mask version);
    the returned keys stay the same 5 the prompt builder expects. `single_object` uses a
    distinctive value (0.5) a pixel-Otsu re-derivation on a random-noise crop would not
    coincidentally match, proving it is MASK-sourced, not pixel-recomputed."""
    crop = _crop_image(tmp_path, qc={"target_dominance": 0.42, "border_touch": 0.1,
                                     "single_object": 0.5, "centroid_offset": 0.2})
    ev = _quant_evidence(crop)
    assert ev["target_dominance"] == 0.42
    assert ev["not_truncated"] == pytest.approx(0.9)
    assert ev["single_object"] == 0.5
    assert ev["centered"] == pytest.approx(0.8)
    assert isinstance(ev["sharpness"], float)
    assert set(ev) == {"sharpness", "single_object", "target_dominance", "not_truncated", "centered"}


def test_quant_evidence_falls_back_to_pixel_sharpness_and_target_dominance_without_qc(tmp_path):
    """Fallback: a crop with no `qc` block (bare/HPA crops via `adapt_bare_crops`) gets
    the PIXEL-only pair `{sharpness, target_dominance}`, computed straight from
    `quant.sharpness`/`quant.target_dominance` on its own pixels."""
    from skimage.io import imread

    from shape.tools.single_cell_cropper import quant

    crop = _crop_image(tmp_path)
    pixels = imread(crop["path"])
    expected = {"sharpness": quant.sharpness(pixels), "target_dominance": quant.target_dominance(pixels)}
    assert _quant_evidence(crop) == expected


# ---------------------------------------------------------------------------
# Task 038 C4 (M-A design): `vision_reject`'s downstream teeth (the kept-crops manifest),
# exclusion rule = `not vision_reject` ONLY -- a gate-invalid crop the VLM RESCUES must
# stay in the downstream analyzer set (NOT `crop_kept`'s `is_crop_valid AND`, which would
# silently drop the rescue).
# ---------------------------------------------------------------------------

class _OrderedEngine:
    """Returns a DIFFERENT verdict per call, in call order -- lets a test selectively
    reject one crop and keep another (a fixed-verdict `_StubEngine` cannot).

    Task 038 montage round: when asked for a `MontageVerdict` (the batched reject
    channel), walks the SAME `self._verdicts` queue ONE ENTRY PER LABELED POSITION
    instead of one per call -- position 0 in the montage gets the next queued entry,
    position 1 the one after, etc. -- so a test can still selectively reject one crop
    and keep another even though both may now be BATCHED into the same montage call."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self._i = 0

    def generate(self, content, response_format=None, **kwargs):
        if response_format is MontageVerdict:
            indices = [int(i) for i in re.findall(r"Position (\d+):", content[0])]
            pos_verdicts = []
            for i in indices:
                verdict, confidence, problem = self._verdicts[self._i]
                self._i += 1
                effective_problem = problem if verdict == "reject" else "none"
                pos_verdicts.append(MontagePositionVerdict(
                    index=i, crop_id="", problem=effective_problem,
                    severity="severe" if effective_problem != "none" else "none",
                    confidence=confidence))
            return {"content": MontageVerdict(verdicts=pos_verdicts)}
        verdict, confidence, problem = self._verdicts[self._i]
        self._i += 1
        rec = CropTriageVerdict(verdict=verdict, confidence=confidence, problem=problem,
                                reasoning="stub")
        return {"content": rec}


def _write_two_crop_metadata(tool_cache_dir, meta_filename):
    """Two real on-disk crops + their metadata file, mirroring `Single_Cell_Cropper_Tool`'s
    own output shape (`cell_crops_metadata_*.json` / `cell_crops_paths` / `cell_metadata`
    with mask-based `qc` blocks). Crop 0 is gate-VALID; crop 1 is gate-INVALID (truncated,
    `mask_containment=0.5` below the 0.90 floor) -- so the two crops probe BOTH halves of
    the manifest's `not vision_reject` ONLY exclusion rule (task 038 M-A): a gate-valid
    crop the VLM rejects must be EXCLUDED, and a gate-INVALID crop the VLM RESCUES (keeps)
    must still be INCLUDED. Returns (meta_path, crop_paths).

    `border_touch=0.6` on BOTH crops (`not_truncated=0.4 < 0.5`) makes both genuine
    `cut_off` CANDIDATES (task 038 restructure's data screen) -- `is_crop_valid` never
    reads `border_touch` directly (only `mask_containment`), so this does not affect
    "gate-VALID" below."""
    from skimage.io import imsave

    crop_paths = []
    for name in ("img1_cell_0000_crop.png", "img1_cell_0001_crop.png"):
        p = tool_cache_dir / name
        imsave(str(p), (np.random.rand(16, 16) * 255).astype(np.uint8))
        crop_paths.append(str(p))
    # target_dominance=0.6 -- is_crop_valid never reads target_dominance (see
    # verifiers/crop.py::is_crop_valid), so this does not affect "gate-VALID" below;
    # target_dominance is purely informational, never a reject driver either way.
    qc_valid = {"mask_containment": 1.0, "occupancy": 0.5, "centroid_offset": 0.0,
               "border_touch": 0.6, "neighbor_contamination": 0.4, "target_dominance": 0.6,
               "single_object": 1.0}
    qc_gate_invalid = dict(qc_valid, mask_containment=0.5)   # truncated: fails is_crop_valid
    meta = {
        "execution_status": "success", "source_image_id": "img1", "group": "default",
        "cell_crops_paths": crop_paths,
        "cell_metadata": [dict(qc=qc_valid, crop_path=crop_paths[0]),
                          dict(qc=qc_gate_invalid, crop_path=crop_paths[1])],
    }
    meta_path = tool_cache_dir / meta_filename
    meta_path.write_text(json.dumps(meta))
    return meta_path, crop_paths


def test_c4_vision_reject_gains_downstream_teeth_via_kept_crops_manifest(tmp_path):
    """C4 (M-A design): confirms, then closes, the wiring gap the 038 sign-off traced --
    AND proves the manifest's exclusion rule is `not vision_reject` ONLY, preserving a
    rescue.

    Crop 0 is gate-VALID but the VLM REJECTS it (high confidence) -> must be EXCLUDED from
    the downstream analyzer set. Crop 1 is gate-INVALID (truncated) but the VLM KEEPS it
    (a rescue) -> must still be PRESENT -- this is the assertion that would fail under the
    superseded `crop_kept` (`is_crop_valid` AND not `vision_reject`) formula, which would
    silently exclude the rescued crop 1 too. Without `write_kept_manifest` + the
    analyzer-side manifest read at all, BOTH assertions fail (the wiring gap).

    Crop 0's reject reason is `cut_off` (`doublet` is not a valid reject reason at all) --
    `_write_two_crop_metadata` sets `border_touch=0.6` on both crops so both are genuine
    data-screened candidates and the judge's own verdict on each lands as-is.

    Task 058 M3 round 2: (B) REJECT's own candidate screening is off by default
    (`enable_reject_channel`) -- this test exercises the channel directly (its own
    `cut_off` candidacy, not the mask check's), so it opts back in explicitly.
    """
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    tool_cache_dir = tmp_path / "tool_cache"
    tool_cache_dir.mkdir()
    meta_path, crop_paths = _write_two_crop_metadata(tool_cache_dir, "cell_crops_metadata_test0001.json")

    # Run the crop-stage VLRR verifier with an engine that REJECTS crop 0 (high confidence)
    # and KEEPS crop 1 -- mirrors the deployed CropVerifier wiring (task 038/041).
    crops, context = adapt_crop(str(meta_path))
    from shape.agent.verification.verifiers.crop import is_crop_valid
    assert is_crop_valid(crops[0])[0] is True    # crop 0 is gate-VALID
    assert is_crop_valid(crops[1])[0] is False   # crop 1 is gate-INVALID (truncated)

    verifier = CropVerifier()
    verifier.llm_engine = _OrderedEngine([("reject", "high", "cut_off"), ("keep", "high", "none")])
    verifier.enable_reject_channel = True
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)

    assert crops[0].get("vision_reject") is True     # gate-valid crop 0 WAS vision-rejected...
    assert not crops[1].get("vision_reject")          # ...gate-invalid crop 1 was RESCUED (kept)

    # Investigation fact (documented; not what this test's pass/fail hinges on): the
    # verdict's valid_crop_rate is computed in quantify(), BEFORE adjudicate() sets
    # vision_reject, so it never reflects the reject on a first-iteration accept -- crop_qc
    # is vision_reject's only OTHER reader, and this is a structural ordering fact the
    # manifest fix does not change. (0.5 here = only crop 0 passes is_crop_valid at
    # quantify()-time; crop 1's gate-invalidity is a separate, pre-existing fact.)
    assert metrics["valid_crop_rate"] == 0.5

    # The load-bearing assertions: the downstream analyzer must NOT ingest the
    # vision-rejected crop, but MUST still ingest the gate-invalid-but-rescued one --
    # `not vision_reject` ONLY, never `is_crop_valid AND` -- non-destructive (the crop
    # FILE itself is untouched either way, never deleted).
    analyzer = Cell_State_Analyzer_Tool()
    loaded_crops, loaded_metadata, skipped = analyzer._load_cell_data_from_metadata(str(tmp_path))
    assert crop_paths[0] not in loaded_crops   # gate-valid but vision-rejected -> excluded
    assert crop_paths[1] in loaded_crops        # gate-INVALID but vision-RESCUED -> present
    assert os.path.exists(crop_paths[0])              # non-destructive: file still on disk


def test_c4_no_engine_analyzer_ingestion_unaffected(tmp_path):
    """No-engine parity (FIX round validation item d): with no vision engine wired, no
    manifest is written, so the analyzer's set stays byte-identical to before task 038 --
    every crop PNG in the metadata file, unfiltered."""
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    tool_cache_dir = tmp_path / "tool_cache"
    tool_cache_dir.mkdir()
    meta_path, crop_paths = _write_two_crop_metadata(tool_cache_dir, "cell_crops_metadata_test0002.json")

    crops, context = adapt_crop(str(meta_path))
    verifier = CropVerifier()          # no llm_engine wired -> deterministic-only
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)

    manifest_path = tool_cache_dir / "cell_crops_kept_manifest_test0002.json"
    assert not manifest_path.exists()

    analyzer = Cell_State_Analyzer_Tool()
    loaded_crops, loaded_metadata, skipped = analyzer._load_cell_data_from_metadata(str(tmp_path))
    assert set(loaded_crops) == set(crop_paths)


# ---------------------------------------------------------------------------
# GEOMETRY (task 038 v10, human-locked). BOTH `mask_bbox` AND off-center `recenter` are
# now 100% PURE DATA, fully DECOUPLED from vision -- `CropVerifier.adjudicate` reads
# `neighbor_contamination`/`centroid_offset` directly, no judge, no engine needed for
# EITHER trigger. SUPERSEDES the earlier v9 design (a VLM-CONFIRMED doublet, gated by
# `n_complete_cells>=2`) -- a real v10 QC audit found that trigger missed most real
# contamination in practice (only 12.5% of crops isolated despite 79% carrying
# `neighbor_contamination>0.2`, since most contamination is a PARTIAL neighbour clipped
# at the crop edge, `n_complete_cells=1`, invisible to a "genuinely WHOLE second cell"
# trigger). A contamination-triggered cell (`neighbor_contamination >
# _MASK_BBOX_CONTAMINATION_MIN`) lands `mask_bbox` DIRECTLY and SKIPS the off-center
# check entirely -- `mask_bbox` is inherently centered on the target, SUBSUMES recenter.
# ---------------------------------------------------------------------------

def _adjudicate_geometry(crop, cell_overrides=None, engine=None, enable_ab_channel=False):
    """One crop through the REAL `CropVerifier.adjudicate` (quantify -> localize ->
    adjudicate, mirroring the production call sequence). `engine=None` (default):
    geometry (task 038 v10) never needs one -- both `mask_bbox` and `recenter` are pure
    data; `engine` only matters for the SEPARATE reject channel (exercised elsewhere).
    `enable_ab_channel` (task 058 M3, default False -- matches `CropVerifier`'s own
    production default): the S3/A-B blinded-comparison gate tests below pass
    `enable_ab_channel=True` explicitly to exercise the (now opt-in) channel. Returns
    (out, crop)."""
    out, crops = _adjudicate_geometry_multi([crop], cell_overrides=cell_overrides, engine=engine,
                                            enable_ab_channel=enable_ab_channel)
    return out, crops[0]


def _adjudicate_geometry_multi(crops, cell_overrides=None, engine=None, enable_ab_channel=False):
    """Task 038 FIX round: the multi-crop sibling of `_adjudicate_geometry` above --
    SEVERAL crops through the REAL `CropVerifier.adjudicate` in ONE round (needed to
    exercise a POPULATION-level prediction, e.g. `expected_effect`, when different
    cells trigger different geometry levers in the SAME round). Returns (out, crops)."""
    verifier = CropVerifier()
    verifier.llm_engine = engine
    verifier.enable_ab_channel = enable_ab_channel
    context = {"cell_overrides": dict(cell_overrides or {})}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    out = verifier.adjudicate(metrics, locus, triggered, context)
    return out, crops


def test_geometry_off_center_only_gives_recenter_without_tight(tmp_path):
    """`centroid_offset>0.20` alone (no doublet) -> `{"recenter": True}` ONLY --
    `margin_mode` is NOT bundled. No engine needed -- pure data."""
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.25)
    out, crop = _adjudicate_geometry(crop)
    assert out.action == "retune_here"
    assert out.param_delta == {"cell_overrides": {"cellA": {"recenter": True}}}


def test_geometry_neither_trigger_proposes_no_override(tmp_path):
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.0)
    out, crop = _adjudicate_geometry(crop)
    assert out.action == "accept"
    assert out.param_delta is None


def test_geometry_unchanged_recommendation_does_not_repropose(tmp_path):
    """The 'changed' detection (`CropVerifier.adjudicate`): a cell ALREADY carrying the
    exact override its OWN current data would produce (e.g. re-adjudicating an
    already-overridden crop on a later VLRR pass) must NOT re-trigger `retune_here` --
    the loop reaches `action='accept'` (StepVerifier's own 'satisfied' path) instead of
    relying on the oscillation-stop safety net to catch a redundant re-proposal."""
    already = {"cellA": {"recenter": True}}
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.25)
    out, crop = _adjudicate_geometry(crop, cell_overrides=already)
    assert out.action == "accept"      # data still flags off-center, but nothing CHANGED
    assert out.param_delta is None


def test_geometry_contamination_lands_mask_bbox_pure_data_no_engine(tmp_path):
    """VALIDATE (task 038 v10): a crop with `neighbor_contamination>0.2` (the exact v10
    blind-spot case -- a PARTIAL neighbour, `n_complete_cells=1`, that the OLD
    `n_complete_cells>=2` trigger structurally missed) now gets `{"margin_mode":
    "mask_bbox", "auto_tighten": True}` DIRECTLY, PURE DATA -- no engine wired at all,
    no VLM confirmation needed or asked. `auto_tighten: True` (task 038 architectural
    FIX round): makes the tool iterate the directional inset lever internally, to
    convergence, in this SAME retune -- round 1 and the tightening are now ONE tool
    call."""
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.0,
                       neighbor_contamination=0.3, qc={"n_complete_cells": 1})
    out, crop = _adjudicate_geometry(crop)
    assert out.action == "retune_here"
    assert out.param_delta == {"cell_overrides": {
        "cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}}


def test_geometry_low_contamination_gives_no_mask_bbox_override(tmp_path):
    """The other half: `neighbor_contamination` AT the 0.2 threshold (not above it) does
    NOT trigger `mask_bbox` -- a clean crop, no override at all."""
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.0,
                       neighbor_contamination=0.2)
    out, crop = _adjudicate_geometry(crop)
    assert out.action == "accept"
    assert out.param_delta is None


def test_geometry_contamination_subsumes_off_center_no_separate_recenter_key(tmp_path):
    """A cell that is BOTH contamination-triggered AND off-center: the off-center check
    is SKIPPED entirely for it -- `mask_bbox`(+`auto_tighten`) alone lands (no separate
    `recenter` key), since `mask_bbox` is inherently centered on the target. Pure data,
    no engine."""
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.25,
                       neighbor_contamination=0.3)
    out, crop = _adjudicate_geometry(crop)
    assert out.action == "retune_here"
    assert out.param_delta == {"cell_overrides": {
        "cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}}


def test_geometry_contamination_preserves_a_prior_recenter_key(tmp_path):
    """A cell PREVIOUSLY overridden with `{"recenter": True}` only (an earlier
    off-center round), whose data NOW also crosses the contamination trigger ->
    upgrades to the mask_bbox+auto_tighten keys (`recenter` preserved via key-merge), a
    genuinely NEW state -> `retune_here` fires again. Mirrors the QC key-merge fix
    (`verifiers/crop.py`): flags only ever ADD keys, never drop a prior one. Pure data,
    no engine."""
    already = {"cellA": {"recenter": True}}
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.25,
                       neighbor_contamination=0.3)
    out, crop = _adjudicate_geometry(crop, cell_overrides=already)
    assert out.action == "retune_here"
    assert out.param_delta == {"cell_overrides": {
        "cellA": {"recenter": True, "margin_mode": "mask_bbox", "auto_tighten": True}}}


def test_geometry_trigger_never_calls_the_engine_even_when_wired(tmp_path):
    """Task 038 v10: geometry (mask_bbox AND recenter alike) never reaches the engine at
    all, even when one IS wired -- proves geometry is genuinely never asked about, not
    merely defaulting to no-op for lack of an engine. This crop is contamination- AND
    off-center-triggered, but flags NEITHER reject reason (no border_touch/
    foreground_fraction set), so the SEPARATE reject channel also finds no candidates --
    the `_RecordingEngine` sees ZERO calls total."""
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.25,
                       neighbor_contamination=0.3)
    engine = _RecordingEngine(verdict="reject", confidence="high", problem="cut_off")
    out, crop = _adjudicate_geometry(crop, engine=engine)
    assert out.action == "retune_here"
    assert out.param_delta == {"cell_overrides": {
        "cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}}
    assert len(engine.calls) == 0


def test_geometry_off_center_pass_preserves_a_carried_key_from_elsewhere(tmp_path):
    """The off-center PURE-DATA pass itself key-merges onto whatever this cell already
    carries (e.g. a prior `mask_bbox` doublet confirmation) -- it must ADD `recenter`
    without dropping the carried key, the same QC-fixed mechanism
    (`verifiers/crop.py`) proven from the doublet side above."""
    already = {"cellA": {"margin_mode": "mask_bbox"}}
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.25)
    out, crop = _adjudicate_geometry(crop, cell_overrides=already)
    assert out.action == "retune_here"
    assert out.param_delta == {"cell_overrides": {"cellA": {"margin_mode": "mask_bbox", "recenter": True}}}


def test_geometry_preserves_a_prior_iterations_override_for_an_unflagged_cell(tmp_path):
    """A DIFFERENT, PREVIOUSLY-overridden cell (`cellB`) not present in THIS iteration's
    crop list must still survive in `merged_overrides` (the accumulated `cell_overrides`
    dict, carried via `context["cell_overrides"]`) -- `apply_bounded`'s categorical
    passthrough REPLACES the whole dict on every retune, so silently omitting a past
    entry would UNDO it on the next tool call."""
    already = {"cellB": {"margin_mode": "tight"}}
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.25)
    out, crop = _adjudicate_geometry(crop, cell_overrides=already)
    assert out.action == "retune_here"
    assert out.param_delta == {"cell_overrides": {
        "cellB": {"margin_mode": "tight"},
        "cellA": {"recenter": True},
    }}


def test_geometry_contamination_driven_predicts_dominance_only_even_with_a_simultaneous_recenter(tmp_path):
    """Task 038 FIX round #3 (coordinator-directed): a real headless run showed 2 of 3
    images `conceded` because an image with SOME contamination-triggered cells AND SOME
    off-center-triggered cells in the SAME round predicted BOTH `mean_dominance: up`
    AND `mean_centroid_offset: down` -- but the tighten OFF-CENTERS the contamination
    cells BY DESIGN (a directional inset on the neighbour side only) and swamps the
    recenter cells' own improvement at the POPULATION level, so `mean_centroid_offset`
    never actually goes down even though a genuine recenter also committed. This test
    builds exactly that mixed-round scenario directly (`cellA` contamination-triggered,
    `cellB` off-center-only) and asserts `expected_effect` predicts ONLY
    `mean_dominance`, dropping the unfulfillable `mean_centroid_offset` claim entirely
    -- `elif`, not two independent `if`s."""
    cell_a = _crop_image(tmp_path, name="a.png", cell_id="cellA", mask_containment=1.0,
                         centroid_offset=0.0, neighbor_contamination=0.3)
    cell_b = _crop_image(tmp_path, name="b.png", cell_id="cellB", mask_containment=1.0,
                         centroid_offset=0.25, neighbor_contamination=0.0)
    out, crops = _adjudicate_geometry_multi([cell_a, cell_b])
    assert out.action == "retune_here"
    assert out.param_delta == {"cell_overrides": {
        "cellA": {"margin_mode": "mask_bbox", "auto_tighten": True},
        "cellB": {"recenter": True},
    }}
    # the DEFINING assertion: mean_dominance ONLY, mean_centroid_offset dropped even
    # though cellB's own recenter genuinely committed this SAME round.
    assert out.expected_effect == {"mean_dominance": "up"}


def test_geometry_recenter_only_still_predicts_centroid_offset_when_no_contamination(tmp_path):
    """The OTHER half of the `elif`: when NO contamination-driven cut commits this
    round, a genuine recenter still predicts `mean_centroid_offset: down` exactly as
    before (this branch is UNCHANGED by the FIX round #3 fix -- only the "both fired"
    case changes)."""
    cell_b = _crop_image(tmp_path, name="b.png", cell_id="cellB", mask_containment=1.0,
                         centroid_offset=0.25, neighbor_contamination=0.0)
    out, crops = _adjudicate_geometry_multi([cell_b])
    assert out.action == "retune_here"
    assert out.expected_effect == {"mean_centroid_offset": "down"}


# ---------------------------------------------------------------------------
# Task 038 v13 (human-directed): the ROUND-2+ tighten loop -- a cell already on
# `margin_mode="mask_bbox"` this iteration, still contaminated, proposes growing
# `bbox_inset_px[contamination_side]` by the NEIGHBOUR-AWARE, data-computed depth
# `tool.py::_compute_crop_qc` already reports as `qc["contamination_inset_px"]`
# (already floor-capped THERE, UNCHANGED since v12 R2). What CHANGED this round: the
# ACCEPTANCE decision is now a PARETO/CONSTRAINT rule, not a per-round absolute VLM QC
# verdict -- a COMFORTABLE cut (high retention, headroom remaining) commits on data
# alone, no VLM call; a genuinely UNCERTAIN cut (VoI-screened: retention within
# `_VOI_RETENTION_MARGIN` of the floor, OR stuck at the floor still contaminated) goes
# to a BLINDED A/B comparison against its own mask_bbox baseline
# (`crop_triage.ab_recommend_crop`) -- the judge preferring the CANDIDATE commits the
# cut and continues; preferring the BASELINE or ABSTAINING reverts fully to baseline
# and FREEZES (`tighten_frozen`, KEPT, never discarded). This RETIRES v12 R2's own
# per-round absolute-QC gate (good_single_cell/over_cropped/still_multi_cell), which
# itself replaced an even earlier fixed-fraction step -- see CODING_LOG.md for the
# full v13 rationale (the self-proving-score critique) and CARD.md SS1/SS2/SS5.
# ---------------------------------------------------------------------------

def test_geometry_no_warning_or_fail_on_a_cleanly_tightened_image_population(tmp_path):
    """Task 038 FIX round #1, full-pipeline proof (coordinator-directed): a real
    headless run showed `localize()` reporting WARNING/FAIL on images whose crops were
    ALL correctly, deliberately auto-tightened -- `crop_qc()`'s own exemption (tested
    directly above) must actually reach `localize()`'s own `triggered`/`decision`
    output through the real `quantify -> localize -> adjudicate` sequence, not just the
    metric in isolation. A population of ONLY deliberately-tightened crops (bad raw
    containment/offset, exempted) must NOT trigger `"truncation"`/`"off_center"` and
    must NOT decision WARNING/FAIL."""
    tightened_crops = [
        {"cell_id": f"cell{i}", "mask_containment": 0.90, "centroid_offset": 0.35,
         "margin_mode": "mask_bbox", "bbox_inset_px": {"right": 10},
         "neighbor_contamination": 0.0, "qc": {"contamination_inset_px": 0}}
        for i in range(5)
    ]
    out, crops = _adjudicate_geometry_multi(tightened_crops)
    assert out.decision == "PASS"
    assert out.decision not in ("WARNING", "FAIL")


# ---------------------------------------------------------------------------

def test_geometry_tighten_freezes_when_data_has_no_more_headroom(tmp_path):
    """A cell already on `mask_bbox`, still contaminated per `neighbor_contamination`,
    whose data-computed `contamination_inset_px` is 0 (the floor was already reached on
    its VERY FIRST attempt, before any `baseline_crop_path` was ever rendered -- so it
    cannot be A/B'd either) freezes SILENTLY (KEPT, `action="accept"`, no further
    override proposed, no override change at all) -- pure data, no engine needed to
    make this call, and no VLM call was ever at risk of being wastefully repeated."""
    already = {"cellA": {"margin_mode": "mask_bbox"}}
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=0.85, centroid_offset=0.0,
                       neighbor_contamination=0.3, margin_mode="mask_bbox",
                       qc={"contamination_side": "right", "contamination_inset_px": 0})
    out, crop = _adjudicate_geometry(crop, cell_overrides=already)
    assert out.action == "accept"
    assert out.param_delta is None


def test_geometry_tighten_freezes_when_no_contamination_side(tmp_path):
    """A cell still flagged contaminated by `neighbor_contamination` alone, but whose
    `contamination_side` is None (nothing for the data cut to point at) also freezes
    SILENTLY -- KEPT, not discarded."""
    already = {"cellA": {"margin_mode": "mask_bbox"}}
    crop = _crop_image(tmp_path, cell_id="cellA", mask_containment=1.0, centroid_offset=0.0,
                       neighbor_contamination=0.3, margin_mode="mask_bbox",
                       qc={"contamination_side": None, "contamination_inset_px": 0})
    out, crop = _adjudicate_geometry(crop, cell_overrides=already)
    assert out.action == "accept"
    assert out.param_delta is None


def test_geometry_full_vlrr_loop_converges_in_two_calls_with_auto_tighten(tmp_path):
    """Task 038 architectural FIX round (coordinator-directed): "assert the trial count
    drops vs the old per-round design." Drives the FULL `StepVerifier.resolve_stage`
    (real reconcile path) with a tool stub whose OWN "mask_bbox" state ALREADY reflects
    a fully auto-tightened cell -- mirroring what `tool.py::_auto_tighten_mask_bbox`
    now computes INSIDE one `execute()` call, not recomputed round-by-round here (that
    per-round growth no longer exists at the verifier level at all, see the module's
    own `CropVerifier.adjudicate` geometry section). A contaminated cell now converges
    in TWO tool calls (naive -> mask_bbox+auto_tighten), down from the prior FIX
    round's own THREE (naive -> mask_bbox -> inset) -- and reaches the SAME real final
    contamination `test_tool.py::
    test_auto_tighten_converges_to_clean_on_a_single_small_neighbour` measures for the
    identical single-small-neighbour scenario (genuinely clean, `neighbor_contamination
    ==0.0`, `target_retention==1.0`)."""
    def _write_meta(state: str, cell_overrides: dict) -> str:
        if state == "naive":
            margin_mode = "fraction"
            qc = {"mask_containment": 1.0, "centroid_offset": 0.0, "border_touch": 0.0,
                 "neighbor_contamination": 0.6, "contamination_side": "right",
                 "contamination_inset_px": 10}
        else:   # "mask_bbox_auto_tightened" -- the TOOL already iterated internally,
                # to full convergence, in ONE call (real numbers, see the docstring).
            margin_mode = "mask_bbox"
            qc = {"mask_containment": 1.0, "centroid_offset": 0.0, "border_touch": 0.0,
                 "neighbor_contamination": 0.0, "contamination_side": None,
                 "contamination_inset_px": 0}
        # `parameters.cell_overrides` echoes back exactly what was applied to PRODUCE
        # this result -- mirrors the real tool.py's own contract.
        meta = {"execution_status": "success", "source_image_id": "imgX", "group": "default",
               "cell_metadata": [{"cell_id": "cellA", "qc": qc, "crop_path": "cellA.png",
                                  "margin_mode": margin_mode,
                                  "bbox_inset_px": {"right": 10} if state != "naive" else {}}],
               "parameters": {"cell_overrides": cell_overrides}}
        path = tmp_path / f"meta_{state}_{len(list(tmp_path.glob('meta_*.json')))}.json"
        path.write_text(json.dumps(meta))
        return str(path)

    def tool_runner(params):
        cell_overrides = (params or {}).get("cell_overrides") or {}
        cell = cell_overrides.get("cellA") or {}
        state = "mask_bbox_auto_tightened" if cell.get("auto_tighten") else "naive"
        return {"cell_crops_metadata_path": _write_meta(state, cell_overrides)}

    def adapter(result, upstream_ctx):
        return adapt_crop(result)

    verifier = CropVerifier()   # headless -- no llm_engine, the VoI/A-B channel never fires
    sv = StepVerifier(max_iters=6)
    initial_result = tool_runner({"cell_overrides": {}})
    res = sv.resolve_stage(verifier, adapter, tool_runner,
                           initial_params={"cell_overrides": {}}, initial_result=initial_result)

    assert res.status == "satisfied", (res.status, res.reason)
    assert res.reason == "converged"
    assert len(res.trials) == 2   # naive -> mask_bbox+auto_tighten, THEN converges

    mask_bbox_round_vres = res.trials[0][1]
    assert mask_bbox_round_vres.action == "retune_here"
    assert mask_bbox_round_vres.param_delta == {"cell_overrides": {
        "cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}}
    # the prior FIX round's own reconcile prediction, unaffected by this round's
    # architectural change: a contamination-driven cut still claims `mean_dominance`.
    assert mask_bbox_round_vres.expected_effect == {"mean_dominance": "up"}

    final_params = res.trials[-1][0]
    assert final_params["cell_overrides"]["cellA"] == {
        "margin_mode": "mask_bbox", "auto_tighten": True}


def test_geometry_full_vlrr_loop_mixed_contamination_and_recenter_converges_headless_no_warning(tmp_path):
    """Task 038 FIX round #4 -- holistic proof combining Bugs (1)/(2)/(3) together
    through the FULL `StepVerifier.resolve_stage`, matching the exact real-trace shape
    the coordinator's own headless run exposed: ONE image with a contamination cell
    (`cellA`, triggers `mask_bbox`+`auto_tighten`) AND an off-center-only cell (`cellB`,
    triggers `recenter`) in the SAME round. Real headless run evidence this reproduces:
    decision WARNING/FAIL on trial1 of a cleanly-tightened image (Bug 1), 2 of 3 images
    conceding `hypothesis_invalidated` because `expected_effect` predicted an
    unfulfillable `mean_centroid_offset: down` alongside `mean_dominance: up` in the
    SAME round (Bug 2), and an extra, pointless `retune_here` on the headless VoI path
    (Bug 3). Asserts the coordinator's own literal "Expected outcome": exactly 2 tool
    calls, `satisfied` (never `conceded`), no WARNING/FAIL decision on either trial."""
    def _write_meta(cell_overrides: dict) -> str:
        cell_a_ov = cell_overrides.get("cellA") or {}
        cell_b_ov = cell_overrides.get("cellB") or {}
        if cell_a_ov.get("auto_tighten"):
            # the tool's own auto_tighten result: floor-capped retention, deliberately
            # off-centered (a directional inset on the neighbour side) -- BOTH would
            # trip the raw truncation/off_center thresholds if not exempted (Bug 1).
            qc_a = {"mask_containment": 0.90, "centroid_offset": 0.35, "border_touch": 0.0,
                   "neighbor_contamination": 0.05, "contamination_side": None,
                   "contamination_inset_px": 0}
            cell_a = {"cell_id": "cellA", "qc": qc_a, "crop_path": "cellA.png",
                     "margin_mode": "mask_bbox", "bbox_inset_px": {"right": 10},
                     "baseline_crop_path": str(tmp_path / "baseline_cellA.png")}
        else:
            qc_a = {"mask_containment": 1.0, "centroid_offset": 0.0, "border_touch": 0.0,
                   "neighbor_contamination": 0.6, "contamination_side": "right",
                   "contamination_inset_px": 10}
            cell_a = {"cell_id": "cellA", "qc": qc_a, "crop_path": "cellA.png",
                     "margin_mode": "fraction", "bbox_inset_px": {}, "baseline_crop_path": None}
        if cell_b_ov.get("recenter"):
            qc_b = {"mask_containment": 1.0, "centroid_offset": 0.05, "border_touch": 0.0,
                   "neighbor_contamination": 0.0}
        else:
            qc_b = {"mask_containment": 1.0, "centroid_offset": 0.25, "border_touch": 0.0,
                   "neighbor_contamination": 0.0}
        cell_b = {"cell_id": "cellB", "qc": qc_b, "crop_path": "cellB.png",
                 "margin_mode": "fraction", "bbox_inset_px": {}, "baseline_crop_path": None}
        meta = {"execution_status": "success", "source_image_id": "imgZ", "group": "default",
               "cell_metadata": [cell_a, cell_b], "parameters": {"cell_overrides": cell_overrides}}
        path = tmp_path / f"meta_{len(list(tmp_path.glob('meta_*.json')))}.json"
        path.write_text(json.dumps(meta))
        return str(path)

    def tool_runner(params):
        cell_overrides = (params or {}).get("cell_overrides") or {}
        return {"cell_crops_metadata_path": _write_meta(cell_overrides)}

    def adapter(result, upstream_ctx):
        return adapt_crop(result)

    verifier = CropVerifier()   # headless -- no llm_engine
    sv = StepVerifier(max_iters=6)
    initial_result = tool_runner({"cell_overrides": {}})
    res = sv.resolve_stage(verifier, adapter, tool_runner,
                           initial_params={"cell_overrides": {}}, initial_result=initial_result)

    assert res.status == "satisfied", (res.status, res.reason)
    assert res.reason == "converged"          # NEVER "conceded"/"hypothesis_invalidated"
    assert len(res.trials) == 2               # exactly 2 tool calls

    trial0_vres = res.trials[0][1]
    assert trial0_vres.action == "retune_here"
    assert trial0_vres.decision not in ("WARNING", "FAIL")
    # Bug 2: mean_dominance ONLY -- mean_centroid_offset dropped even though cellB's
    # own recenter also committed this SAME round.
    assert trial0_vres.expected_effect == {"mean_dominance": "up"}

    trial1_vres = res.trials[1][1]
    assert trial1_vres.action == "accept"     # Bug 3: headless VoI path stays SILENT
    # Bug 1: the tightened cellA's own deliberate truncation/off-centering must NOT
    # decision WARNING/FAIL.
    assert trial1_vres.decision not in ("WARNING", "FAIL")

    # the delivered, kept-best result also carries a clean decision.
    assert res.verdict.decision not in ("WARNING", "FAIL")


def _ab_gate_crop(tmp_path, *, retention: float = 0.93, inset_px: int = 3, side: str = "right",
                  contamination: float = 0.3):
    """A round-2+ candidate crop that is genuinely VoI-RISKY (`retention` within
    `_VOI_RETENTION_MARGIN` of the `_TARGET_RETENTION_FLOOR`, default args land at
    0.93 <= 0.95) AND carries a REAL, readable `baseline_crop_path` (a SEPARATE
    on-disk image, deliberately a different size than the candidate -- the common
    case, since a directional inset SHRINKS the candidate relative to its own
    baseline) -- so it is genuinely eligible for the blinded A/B gate."""
    baseline_path = tmp_path / "baseline.png"
    from skimage.io import imsave
    imsave(str(baseline_path), (np.random.rand(20, 20) * 255).astype(np.uint8))
    return _crop_image(tmp_path, cell_id="cellA", mask_containment=retention, centroid_offset=0.0,
                       neighbor_contamination=contamination, margin_mode="mask_bbox",
                       baseline_crop_path=str(baseline_path),
                       qc={"contamination_side": side, "contamination_inset_px": inset_px})


def test_geometry_voi_gate_skips_a_comfortable_high_retention_cut(tmp_path):
    """A cut with COMFORTABLE (high) retention, well above `_VOI_RETENTION_MARGIN`'s
    own band above the floor, is accepted on DATA ALONE -- the engine is NEVER called,
    even though it is wired, because the VoI screen never classifies it as
    uncertain/risky in the first place (CARD.md: "a comfortable, clean cut is accepted
    on data alone -- no VLM call"). Task 038 architectural FIX round: with NOTHING
    risky about this cell, there is also nothing further to propose (`auto_tighten`
    already tightened it maximally inside the tool in round 1) -- the loop converges
    SILENTLY, `action=="accept"`."""
    already = {"cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}
    crop = _ab_gate_crop(tmp_path, retention=1.0, inset_px=5)   # comfortably above 0.95
    engine = _ABStubEngine(verdict="left_better")
    out, crop = _adjudicate_geometry(crop, cell_overrides=already, engine=engine)
    assert out.action == "accept"
    assert out.param_delta is None
    assert len(engine.calls) == 0   # the DEFINING VoI-gate assertion: no engine call at all
    assert "ab_verdict" not in crop


def test_geometry_voi_gate_triggers_ab_for_a_floor_hitting_low_retention_cut(tmp_path):
    """A cut whose retention has landed within `_VOI_RETENTION_MARGIN` of the floor
    (genuinely uncertain from data alone) DOES trigger the blinded A/B gate -- the
    engine IS called exactly once, PROVIDED `enable_ab_channel` is explicitly turned
    on (task 058 M3 -- off by default, see `test_ab_channel_off_by_default_...`
    below)."""
    already = {"cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}
    crop = _ab_gate_crop(tmp_path, retention=0.93)   # within the 0.90-0.95 VoI band
    engine = _ABStubEngine(verdict="left_better")
    out, crop = _adjudicate_geometry(crop, cell_overrides=already, engine=engine,
                                     enable_ab_channel=True)
    assert len(engine.calls) == 1
    assert crop.get("ab_verdict") in ("candidate", "baseline")   # a real, resolved verdict


def test_ab_channel_off_by_default_veto_never_fires_on_a_risky_cut(tmp_path):
    """Task 058 M3 (human-authorized): `CropVerifier.enable_ab_channel` defaults to
    False -- a VoI-risky cut that WOULD have triggered the A/B gate (same crop as the
    test above) is instead accepted on data alone even with an `llm_engine` wired:
    the engine is NEVER called, no `ab_verdict` is ever set, and the tool's own
    already-auto-tightened geometry stands (no veto/revert -- the exact "off is
    correct" default the human directed, 038's own A/B-harmful finding)."""
    already = {"cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}
    crop = _ab_gate_crop(tmp_path, retention=0.93)   # the SAME VoI-risky cut
    engine = _ABStubEngine(verdict="left_better")
    verifier = CropVerifier()
    assert verifier.enable_ab_channel is False   # the production default itself
    out, crop = _adjudicate_geometry(crop, cell_overrides=already, engine=engine)
    assert len(engine.calls) == 0   # A/B never called -- the veto cannot fire
    assert "ab_verdict" not in crop
    assert out.action == "accept"   # nothing further proposed -- the SAME silent-freeze
    assert out.param_delta is None  # path a headless run already used for this cell


def test_geometry_ab_gate_candidate_wins_commits_the_next_cut(tmp_path):
    """The judge's de-randomized verdict resolves to `"candidate"` (the tightened crop
    wins) -> the tool's own already-tightened geometry IS the accepted result (task 038
    architectural FIX round: there is no further per-round cut for the verifier to
    propose any more) -- FREEZE so this cell is never re-judged again, verified across
    several trials since the L/R side is randomized per call; the resolved `ab_verdict`
    is what matters, not the raw judge answer."""
    already = {"cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}
    seen_candidate = False
    for _ in range(20):
        crop = _ab_gate_crop(tmp_path)
        engine = _ABStubEngine(verdict="left_better")
        out, crop = _adjudicate_geometry(crop, cell_overrides=already, engine=engine,
                                         enable_ab_channel=True)
        if crop.get("ab_verdict") == "candidate":
            seen_candidate = True
            assert out.action == "retune_here"
            assert out.param_delta == {"cell_overrides": {
                "cellA": {"margin_mode": "mask_bbox", "auto_tighten": True,
                         "tighten_frozen": True}}}
            break
    assert seen_candidate, "candidate never won across 20 trials -- de-randomization likely broken"


def test_geometry_ab_gate_baseline_wins_reverts_and_freezes(tmp_path):
    """The judge's de-randomized verdict resolves to `"baseline"` (the plain mask_bbox
    crop wins -- no clear improvement) -> the verifier REVERTS to the plain mask_bbox
    baseline and FREEZES (`tighten_frozen`) -- the crop is still KEPT (`retune_here`
    re-crops at the safer baseline geometry), never discarded. Task 038 architectural
    FIX round: the revert is now `auto_tighten: False` (`tool.py` renders the plain,
    zero-inset box whenever `auto_tighten` is falsy -- there is no externally-tracked
    `bbox_inset_px` to clear any more, the tool computed it internally)."""
    already = {"cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}
    seen_baseline = False
    for _ in range(20):
        crop = _ab_gate_crop(tmp_path)
        engine = _ABStubEngine(verdict="left_better")
        out, crop = _adjudicate_geometry(crop, cell_overrides=already, engine=engine,
                                         enable_ab_channel=True)
        if crop.get("ab_verdict") == "baseline":
            seen_baseline = True
            assert out.action == "retune_here"
            assert out.param_delta == {"cell_overrides": {
                "cellA": {"margin_mode": "mask_bbox", "auto_tighten": False,
                         "tighten_frozen": True}}}
            break
    assert seen_baseline, "baseline never won across 20 trials -- de-randomization likely broken"


def test_geometry_ab_gate_abstain_reverts_and_freezes_same_as_baseline(tmp_path):
    """An `"abstain"` verdict (not clearly distinguishable) is treated the SAME as a
    baseline-preferred verdict -- no clear winner means KEEP the safer (baseline)
    state, per the Pareto acceptance rule's own "a trade-off has no clear winner ->
    KEEP the baseline" wording."""
    already = {"cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}
    crop = _ab_gate_crop(tmp_path)
    engine = _ABStubEngine(verdict="abstain")
    out, crop = _adjudicate_geometry(crop, cell_overrides=already, engine=engine,
                                     enable_ab_channel=True)
    assert crop.get("ab_verdict") == "abstain"
    assert out.action == "retune_here"
    assert out.param_delta == {"cell_overrides": {
        "cellA": {"margin_mode": "mask_bbox", "auto_tighten": False, "tighten_frozen": True}}}
    assert len(engine.calls) == 1


def test_geometry_ab_gate_veto_never_wins_keep_best_over_the_accepted_trial():
    """`scalar_score` PARETO-CONSISTENCY (task 038 v13's own core requirement): a
    CRAFTED history where the MOST-TIGHTENED trial (lower `neighbor_contamination`,
    which would have scored HIGHEST under the old, disqualified composite) is the ONE
    CURRENTLY BEING VETOED this round must NOT win `StepVerifier._best`'s own
    keep-best -- the shared, UNMODIFIED mechanism must instead pick the trial where
    that SAME cell sits at its own (higher, un-vetoed) baseline dominance."""
    # Trial 1 (accepted baseline state, cellA untouched): dominance 0.5.
    vetoed_now = set()
    accepted_crops = [{"cell_id": "cellA", "mask_containment": 1.0, "neighbor_contamination": 0.5}]
    verifier = CropVerifier()
    vres_accepted = verifier._pareto_scalar(accepted_crops, vetoed_now)
    # Trial 2 (the SAME cell, MORE tightened -- neighbor_contamination genuinely LOWER,
    # 0.1 -- but THIS round's own A/B judge is CURRENTLY VETOING it, about to revert).
    tightened_crops = [{"cell_id": "cellA", "mask_containment": 1.0, "neighbor_contamination": 0.1}]
    vetoed_this_round = {"cellA"}
    vres_vetoed = verifier._pareto_scalar(tightened_crops, vetoed_this_round)

    # The DEFINING assertion: despite genuinely LOWER contamination, the vetoed state
    # scores WORSE (sentinel-bad) than the accepted baseline state.
    assert vres_vetoed < vres_accepted

    history = [({"trial": 1}, VLRRResult("PASS", vres_accepted, {}, action="accept"), "accepted_result"),
              ({"trial": 2}, VLRRResult("PASS", vres_vetoed, {}, action="accept"), "vetoed_result")]
    resolution = StepVerifier()._best(history, {}, "satisfied", reason="test")
    assert resolution.result == "accepted_result"   # NOT "vetoed_result" -- the whole point


def test_geometry_tighten_headless_commits_next_cut_without_any_engine(tmp_path):
    """No `llm_engine` wired at all -- a VoI-risky cell gets NO verdict (A/B never ran)
    and stays SILENT (task 038 FIX round #3: previously committed a `tighten_frozen`
    marker here even headless, an extra, pointless tool call -- there is no "next cut"
    for `auto_tighten` to prevent and no VLM call to protect in a headless run, so that
    commit bought nothing; the SAME "silent unless it prevents something more
    expensive" reasoning already applied to clean/stuck cells, FIX round #2). The
    headless fallback CARD.md documents ("headless -> pure data, accept every
    constraint-satisfying cut") -- `action` stays `"accept"`, nothing proposed."""
    already = {"cellA": {"margin_mode": "mask_bbox", "auto_tighten": True}}
    crop = _ab_gate_crop(tmp_path)   # genuinely risky by retention -- but no engine wired
    out, crop = _adjudicate_geometry(crop, cell_overrides=already)
    assert out.action == "accept"
    assert out.param_delta is None
    assert "ab_verdict" not in crop


def test_is_crop_valid_exempts_a_low_containment_mask_bbox_crop():
    """§2.5 hazard fix (task 038 v12, human-locked): a directionally-tightened
    `mask_bbox` crop's `mask_containment` legitimately drops below `_CONTAINMENT_MIN`
    -- that is the INTENDED effect of the lever, not a truncation defect -- so it must
    NOT be auto-dropped by `is_crop_valid`'s own containment gate."""
    tightened = {"mask_containment": 0.6, "margin_mode": "mask_bbox"}
    ok, reasons = is_crop_valid(tightened)
    assert ok is True
    assert "truncated" not in reasons

    # a NON-mask_bbox crop at the same low containment is still correctly gated --
    # the exemption is specific to mask_bbox mode, not a global loosening.
    not_tightened = {"mask_containment": 0.6, "margin_mode": None}
    ok2, reasons2 = is_crop_valid(not_tightened)
    assert ok2 is False
    assert "truncated" in reasons2


# ---------------------------------------------------------------------------
# THE MONTAGE FIX (task 038 montage round, human-locked): the REJECT confirm
# (cut_off/empty_debris) is now BATCHED into labeled grid montages instead of one VLM
# call per candidate -- motivated by a real v9 cluster run (cell_00554: 5 GOOD spindle
# cells falsely flagged `empty_debris`, judged in isolation, because a genuinely
# thin/elongated single cell's own `foreground_fraction` is naturally low without the
# cell being empty/debris at all). Doublet confirmation stays UNCHANGED (per-crop, v9
# mechanism) -- covered above, not here.
# ---------------------------------------------------------------------------

def test_build_montage_produces_n_labeled_cells_and_a_correct_index_map(tmp_path):
    """`_build_montage` places each candidate crop's own GROUNDED overlay into a labeled
    grid cell, in order, and records the STRICT index->crop map the judge's own verdicts
    are landed back through -- the montage image itself is the expected grid canvas
    size (rows*cols cells of `_MONTAGE_CELL_PX` each)."""
    import cv2

    crops = [_crop_image(tmp_path, f"g{i}.png", border_touch=0.6) for i in range(3)]
    montage_bytes, index_to_crop = _build_montage(crops, (2, 2))
    assert montage_bytes is not None
    decoded = cv2.imdecode(np.frombuffer(montage_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (2 * _MONTAGE_CELL_PX, 2 * _MONTAGE_CELL_PX)
    assert set(index_to_crop) == {0, 1, 2}
    for i, crop in enumerate(crops):
        assert index_to_crop[i] is crop
        assert _crop_identifier(index_to_crop[i]) == _crop_identifier(crop)


def test_build_montage_empty_chunk_returns_none():
    """No readable crops in the chunk -> `(None, {})`, the graceful-degrade contract
    `_judge_reject_montages` relies on to skip a chunk without crashing."""
    montage_bytes, index_to_crop = _build_montage([{"path": "/does/not/exist.png"}], (2, 2))
    assert montage_bytes is None
    assert index_to_crop == {}


def test_montage_verdict_problem_none_rescues_a_flagged_crop(tmp_path):
    """A crop DATA-FLAGGED as a reject candidate (cut_off) whose montage verdict reports
    `problem='none'` is a RESCUE -- `_apply_montage_verdicts` must NOT set
    `vision_reject` on it. The exact cell_00554 case this round fixes: a data-screen
    false-positive corrected once the judge sees it and disagrees."""
    crop = _crop_image(tmp_path, border_touch=0.6)   # data-flagged cut_off candidate
    index_to_crop = {0: crop}
    mv = MontageVerdict(verdicts=[MontagePositionVerdict(
        index=0, crop_id=_crop_identifier(crop), problem="none", severity="none",
        confidence="high")])
    verdicts = _apply_montage_verdicts(mv, index_to_crop)
    assert crop.get("vision_reject") is not True
    assert verdicts[0]["reject"] is False
    assert verdicts[0]["verdict"] == "keep"


def test_montage_verdict_empty_debris_high_confidence_rejects(tmp_path):
    """The other half: a CONFIRMED `empty_debris` at high confidence lands the reject
    as-is, on the crop the index actually points to."""
    crop = _crop_image(tmp_path, qc={"foreground_fraction": 0.01})
    index_to_crop = {0: crop}
    mv = MontageVerdict(verdicts=[MontagePositionVerdict(
        index=0, crop_id=_crop_identifier(crop), problem="empty_debris", severity="severe",
        confidence="high")])
    verdicts = _apply_montage_verdicts(mv, index_to_crop)
    assert crop.get("vision_reject") is True
    assert crop.get("vision_problem") == "empty_debris"
    assert verdicts[0]["reject"] is True


def test_montage_verdict_low_confidence_reject_is_kept(tmp_path):
    """Keep-biased, confidence-gated -- mirrors the per-crop mechanism: a LOW-confidence
    reject does not reject."""
    crop = _crop_image(tmp_path, border_touch=0.6)
    index_to_crop = {0: crop}
    mv = MontageVerdict(verdicts=[MontagePositionVerdict(
        index=0, crop_id=_crop_identifier(crop), problem="cut_off", severity="moderate",
        confidence="low")])
    verdicts = _apply_montage_verdicts(mv, index_to_crop)
    assert crop.get("vision_reject") is not True


def test_montage_verdict_crop_id_mismatch_trusts_the_index(tmp_path):
    """A verdict naming the WRONG `crop_id` for its own `index` still lands BY INDEX --
    the grid position is what the judge actually looked at; `crop_id` is only a restated
    cross-check, never an independent identification (task 038 montage round,
    human-specified landing rule)."""
    crop = _crop_image(tmp_path, border_touch=0.6)
    index_to_crop = {0: crop}
    mv = MontageVerdict(verdicts=[MontagePositionVerdict(
        index=0, crop_id="some/other/path.png", problem="cut_off", severity="severe",
        confidence="high")])
    verdicts = _apply_montage_verdicts(mv, index_to_crop)
    assert crop.get("vision_reject") is True     # landed by INDEX despite the mismatch
    assert verdicts[0]["reject"] is True


def test_montage_verdict_unknown_index_is_skipped_not_a_crash():
    """A verdict naming an index outside this montage (e.g. a hallucinated extra
    position) is skipped -- keep-biased, no crash, no crop mutated."""
    mv = MontageVerdict(verdicts=[MontagePositionVerdict(
        index=5, crop_id="", problem="cut_off", severity="severe", confidence="high")])
    verdicts = _apply_montage_verdicts(mv, {0: {"path": "x"}})
    assert verdicts == []


def test_judge_reject_montages_chunks_10_candidates_into_9_plus_1(tmp_path):
    """`montage_grid=(3, 3)` (chunk size 9) chunks 10 reject candidates into TWO montage
    calls -- 9, then the remaining 1 -- a partial LAST montage is fine (task 038 montage
    round, human-specified grid-size behavior)."""
    crops = [_crop_image(tmp_path, f"r{i}.png", border_touch=0.6) for i in range(10)]
    engine = _RecordingEngine(verdict="keep", confidence="high")
    summary = _judge_reject_montages(engine, crops, montage_grid=(3, 3))
    assert len(engine.calls) == 2                     # two montages: 9 + 1
    prompt1, prompt2 = engine.calls[0][0], engine.calls[1][0]
    assert len(re.findall(r"Position \d+:", prompt1)) == 9
    assert len(re.findall(r"Position \d+:", prompt2)) == 1
    assert summary["judged"] == 10


def test_recording_engine_receives_the_montage_not_a_single_crop(tmp_path):
    """The image bytes actually sent for a batch of reject candidates decode to a GRID
    canvas (rows*cols cells of `_MONTAGE_CELL_PX` each), not a single crop's own native
    16x16 size -- proves the MONTAGE, not a bare per-crop image, is what reaches the
    engine (task 038 montage round)."""
    import cv2

    crops = [_crop_image(tmp_path, f"m{i}.png", border_touch=0.6) for i in range(3)]
    engine = _RecordingEngine(verdict="keep", confidence="high")
    vision_recommend_crop(engine, crops, _det(), montage_grid=(2, 2))
    assert len(engine.calls) == 1
    _, montage_bytes = engine.calls[0]
    decoded = cv2.imdecode(np.frombuffer(montage_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (2 * _MONTAGE_CELL_PX, 2 * _MONTAGE_CELL_PX)


def test_build_montage_cell_shows_a_colored_mask_fill(tmp_path):
    """Task 038 v10 ("overlay 要显示 mask 为 colorful 像 segmentation 那样"): the
    montage's own per-cell overlay is a genuinely COLORED fill over the target region,
    not a flat grayscale tint -- `label2rgb`'s default color blend makes the R channel
    (BGR index 2, since this module's own arrays are cv2-native) clearly dominant over
    G/B inside the masked region, by a constant margin regardless of the underlying
    (random) pixel brightness (`0.5*red + 0.5*gray` vs `0.5*gray` -- R-G/B = a constant
    ~0.5, see `_overlay_target_on_crop`'s own math)."""
    import cv2
    from skimage.io import imsave

    mask_path = tmp_path / "mask.png"
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255   # covers the crop's own center
    imsave(str(mask_path), mask)
    crop = _crop_image(tmp_path, "colored.png", mask_path=str(mask_path))

    montage_bytes, index_to_crop = _build_montage([crop], (1, 1))
    assert montage_bytes is not None
    decoded = cv2.imdecode(np.frombuffer(montage_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    center = decoded[_MONTAGE_CELL_PX // 2, _MONTAGE_CELL_PX // 2]
    assert int(center[2]) > int(center[1]) + 20   # R (BGR idx 2) clearly above G -- a
                                                   # genuine COLOR, not a gray/green tint
    assert int(center[2]) > int(center[0]) + 20   # R clearly above B too


def test_sent_montage_is_persisted_to_disk_and_decodes_to_the_same_grid(tmp_path):
    """Task 038 v10 ("把 montage 的图也都给保持"): every montage actually SENT to the
    engine is ALSO written to disk under `_montages_dir_for(metadata_path)` -- the
    saved file exists, is BYTE-IDENTICAL to what was actually sent, and decodes to the
    SAME grid dimensions."""
    import cv2

    meta_path = tmp_path / "cell_crops_metadata_abc123.json"
    meta_path.write_text("{}")   # only this file's own PATH/naming is used here
    crops = [_crop_image(tmp_path, f"m{i}.png", border_touch=0.6) for i in range(3)]
    engine = _RecordingEngine(verdict="keep", confidence="high")
    vision_recommend_crop(engine, crops, _det(), montage_grid=(2, 2),
                          metadata_path=str(meta_path))
    assert len(engine.calls) == 1
    _, sent_bytes = engine.calls[0]

    saved_path = _montages_dir_for(str(meta_path)) + os.sep + "montage_00.png"
    assert os.path.exists(saved_path)
    with open(saved_path, "rb") as fh:
        saved_bytes = fh.read()
    assert saved_bytes == sent_bytes   # byte-identical to what was actually sent
    decoded = cv2.imdecode(np.frombuffer(saved_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (2 * _MONTAGE_CELL_PX, 2 * _MONTAGE_CELL_PX)


def test_no_candidates_writes_no_montage_file(tmp_path):
    """No REJECT candidates at all -> `_judge_reject_montages` never builds/sends a
    montage, so it never touches `montages_dir` either -- no directory, no file."""
    meta_path = tmp_path / "cell_crops_metadata_def456.json"
    meta_path.write_text("{}")
    clean_crop = _crop_image(tmp_path, "clean.png", border_touch=0.0,
                             qc={"foreground_fraction": 0.5})
    engine = _RecordingEngine(verdict="reject", confidence="high", problem="cut_off")
    vision_recommend_crop(engine, [clean_crop], _det(), metadata_path=str(meta_path))
    assert len(engine.calls) == 0
    assert not os.path.exists(_montages_dir_for(str(meta_path)))


def test_no_engine_writes_no_montage_file(tmp_path):
    """No engine wired at all -> no montage judged, no file written."""
    meta_path = tmp_path / "cell_crops_metadata_ghi789.json"
    meta_path.write_text("{}")
    crop = _crop_image(tmp_path, border_touch=0.6)
    vision_recommend_crop(None, [crop], _det(), metadata_path=str(meta_path))
    assert not os.path.exists(_montages_dir_for(str(meta_path)))


def test_montages_dir_for_derives_from_metadata_path_basename():
    """`_montages_dir_for` mirrors `llm_recommend.write_kept_manifest`'s own sidecar-
    naming idiom -- swap the `cell_crops_metadata_` prefix, keep the unique hash suffix
    -- so montages from a DIFFERENT trial (a different `metadata_path`, same
    `tool_cache_dir`) never collide."""
    meta = os.path.join("x", "y", "cell_crops_metadata_abc123.json")
    expected = os.path.join("x", "y", "montages_abc123")
    assert _montages_dir_for(meta) == expected
    assert _montages_dir_for(None) is None
    assert _montages_dir_for(os.path.join("x", "y", "unexpected_name.json")) is None


def test_montages_dir_for_ab_prefix_is_a_separate_subdirectory():
    """Task 038 v13: the A/B channel's own montages persist under a DIFFERENT
    subdirectory (`prefix="montages_ab"`) than the REJECT channel's (the default
    `prefix="montages"`) for the SAME trial -- never overwriting each other."""
    meta = os.path.join("x", "y", "cell_crops_metadata_abc123.json")
    reject_dir = _montages_dir_for(meta)
    ab_dir = _montages_dir_for(meta, prefix="montages_ab")
    assert reject_dir != ab_dir
    assert ab_dir == os.path.join("x", "y", "montages_ab_abc123")


def _crop_with_baseline(tmp_path, name="cell.png", **fields) -> dict:
    """A crop dict with BOTH a real, readable `path` (the candidate) AND a real,
    readable `baseline_crop_path` (a SEPARATE on-disk image, deliberately a different
    size -- the common case for a directionally-inset candidate vs. its own baseline)
    -- so it is a genuinely pairable A/B candidate."""
    from skimage.io import imsave
    baseline_path = tmp_path / f"baseline_{name}"
    imsave(str(baseline_path), (np.random.rand(20, 20) * 255).astype(np.uint8))
    crop = _crop_image(tmp_path, name, **fields)
    crop["baseline_crop_path"] = str(baseline_path)
    return crop


# ---------------------------------------------------------------------------
# Task 038 v13 (human-directed): `ab_recommend_crop` -- the per-round, VoI-gated
# BLINDED A/B comparison gate, RETIRING v12 R2's own absolute-QC tighten gate.
# `verifiers/crop.py::CropVerifier.adjudicate`'s own round-2+ pass is the ONLY caller
# in production (its own tests, in the GEOMETRY section above, cover the full
# end-to-end wiring); these tests exercise `ab_recommend_crop` directly, mirroring how
# `_judge_reject_montages` gets its own dedicated tests too.
# ---------------------------------------------------------------------------

def test_ab_recommend_crop_sets_verdict_in_place(tmp_path):
    crop = _crop_with_baseline(tmp_path, cell_id="cellA")
    engine = _ABStubEngine(verdict="left_better")
    summary = ab_recommend_crop(engine, [crop])
    assert summary["judged"] == 1
    assert crop["ab_verdict"] in ("candidate", "baseline")   # de-randomized, either is valid
    assert summary["verdicts"][0]["verdict"] == crop["ab_verdict"]
    assert summary["verdicts"][0]["raw_verdict"] == "left_better"


def test_ab_recommend_crop_no_engine_is_noop(tmp_path):
    crop = _crop_with_baseline(tmp_path, cell_id="cellA")
    summary = ab_recommend_crop(None, [crop])
    assert summary == {"judged": 0, "verdicts": []}
    assert "ab_verdict" not in crop


def test_ab_recommend_crop_no_candidates_is_noop():
    engine = _ABStubEngine(verdict="left_better")
    summary = ab_recommend_crop(engine, [])
    assert summary == {"judged": 0, "verdicts": []}
    assert len(engine.calls) == 0


def test_ab_recommend_crop_no_baseline_is_skipped(tmp_path):
    """A candidate with NO `baseline_crop_path` (or an unreadable one) cannot be
    PAIRED -- skipped entirely (not a one-sided tile), so it is never judged."""
    crop = _crop_image(tmp_path, cell_id="cellA")   # no baseline_crop_path at all
    engine = _ABStubEngine(verdict="left_better")
    summary = ab_recommend_crop(engine, [crop])
    assert summary == {"judged": 0, "verdicts": []}
    assert len(engine.calls) == 0
    assert "ab_verdict" not in crop


def test_ab_recommend_crop_internal_failure_falls_back(tmp_path, monkeypatch):
    """A failure inside the A/B-montage judging step itself must not break the loop --
    unlike the reject channel (guarded by an OUTER try/except in the
    `vision_recommend_crop` adapter around `_judge_reject_montages`), this function is
    called DIRECTLY from `verifiers/crop.py::adjudicate` with no adapter wrapping it, so
    its own try/except must cover montage-BUILDING too, not just the engine call."""
    crop = _crop_with_baseline(tmp_path, cell_id="cellA")

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("shape.agent.verification.crop_triage._build_ab_montage", _boom)
    summary = ab_recommend_crop(_ABStubEngine(), [crop])
    assert summary == {"judged": 0, "verdicts": []}
    assert "ab_verdict" not in crop


def test_ab_montage_is_persisted_to_its_own_subdirectory(tmp_path):
    """Every A/B montage actually SENT is persisted under
    `_montages_dir_for(metadata_path, prefix="montages_ab")` -- a SEPARATE
    subdirectory from the reject channel's own montages for the same trial."""
    meta_path = tmp_path / "cell_crops_metadata_ab001.json"
    meta_path.write_text("{}")
    crop = _crop_with_baseline(tmp_path, cell_id="cellA")
    engine = _ABStubEngine(verdict="left_better")
    ab_recommend_crop(engine, [crop], metadata_path=str(meta_path))
    assert len(engine.calls) == 1
    saved_path = os.path.join(_montages_dir_for(str(meta_path), prefix="montages_ab"),
                              "montage_00.png")
    assert os.path.exists(saved_path)


# ---------------------------------------------------------------------------
# Task 038 v13: L/R randomization + de-randomization -- `_build_ab_montage`'s own
# per-tile side assignment (via `rng`) and `_apply_ab_verdicts`'s own mapping back to
# candidate/baseline. Tested DIRECTLY (not through `ab_recommend_crop`/a stub engine)
# with a `_FixedRng` so the side is deterministic, since the prompt text alone cannot
# reveal which side a real judge's answer refers to.
# ---------------------------------------------------------------------------

def test_ab_montage_left_right_derandomization_candidate_on_left(tmp_path):
    """`_FixedRng(0.0)` forces the candidate to the LEFT side of its own tile -- a
    `left_better` verdict then correctly de-randomizes to `ab_verdict="candidate"`."""
    crop = _crop_with_baseline(tmp_path, cell_id="cellA")
    montage_bytes, index_to_pair = _build_ab_montage([crop], (3, 3), _FixedRng(0.0))
    assert montage_bytes is not None
    assert index_to_pair[0]["candidate_side"] == "left"

    verdict = ABVerdict(verdicts=[
        ABPositionVerdict(index=0, crop_id="", verdict="left_better", confidence="high")])
    applied = _apply_ab_verdicts(verdict, index_to_pair)
    assert crop["ab_verdict"] == "candidate"
    assert applied[0]["verdict"] == "candidate"
    assert applied[0]["raw_verdict"] == "left_better"


def test_ab_montage_left_right_derandomization_candidate_on_right(tmp_path):
    """`_FixedRng(0.9)` forces the candidate to the RIGHT side -- the SAME
    `left_better` verdict now correctly de-randomizes to `ab_verdict="baseline"` (the
    LEFT panel, which is the baseline here, is what the judge actually preferred)."""
    crop = _crop_with_baseline(tmp_path, cell_id="cellA")
    montage_bytes, index_to_pair = _build_ab_montage([crop], (3, 3), _FixedRng(0.9))
    assert index_to_pair[0]["candidate_side"] == "right"

    verdict = ABVerdict(verdicts=[
        ABPositionVerdict(index=0, crop_id="", verdict="left_better", confidence="high")])
    applied = _apply_ab_verdicts(verdict, index_to_pair)
    assert crop["ab_verdict"] == "baseline"
    assert applied[0]["verdict"] == "baseline"


def test_ab_montage_overlay_present_on_both_panels(tmp_path):
    """BOTH panels of a tile carry the grounded, colourful overlay (human-required --
    no overlay tanks VLM efficiency) -- verified indirectly via a successful,
    correctly-sized montage build (the overlay function itself is direct-unit-tested
    elsewhere; this proves the A/B tile-assembly path actually CALLS it for both
    images, not just one)."""
    crop = _crop_with_baseline(tmp_path, cell_id="cellA")
    montage_bytes, index_to_pair = _build_ab_montage([crop], (1, 1), _FixedRng(0.0))
    assert montage_bytes is not None
    import cv2
    decoded = cv2.imdecode(np.frombuffer(montage_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (_MONTAGE_CELL_PX, _MONTAGE_CELL_PX)


def test_vision_recommend_crop_montage_internal_failure_falls_back(tmp_path, monkeypatch):
    """A failure inside the MONTAGE judging step itself must not break the loop either --
    mirrors `test_vision_recommend_crop_internal_failure_falls_back`'s own proof for the
    doublet channel, now for the reject channel's own try/except in
    `vision_recommend_crop`."""
    det = _det()
    crops = [_crop_image(tmp_path, border_touch=0.6)]

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("shape.agent.verification.crop_triage._judge_reject_montages", _boom)
    assert vision_recommend_crop(_StubEngine(), crops, det) is det


def test_crop_verifier_triage_montage_grid_default_and_threading(tmp_path):
    """`CropVerifier.triage_montage_grid` defaults to (3, 3) and is threaded through to
    `vision_recommend_crop`'s own `montage_grid` parameter end-to-end -- verified via the
    montage call's own grid-chunking behavior (a 1x1 grid forces ONE montage call per
    candidate, so 3 candidates -> 3 calls). Task 058 M3 round 2: (B) REJECT's own
    candidate screening is off by default -- opts back in explicitly to exercise it."""
    verifier = CropVerifier()
    assert verifier.triage_montage_grid == (3, 3)

    verifier.llm_engine = _RecordingEngine(verdict="keep", confidence="high")
    verifier.enable_reject_channel = True
    verifier.triage_montage_grid = (1, 1)
    crops = [_crop_image(tmp_path, f"t{i}.png", border_touch=0.6) for i in range(3)]
    context = {}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)
    assert len(verifier.llm_engine.calls) == 3


# ---------------------------------------------------------------------------
# Task 038 Round B, S1 (coordinator-directed, evidence-driven -- NOT a preference):
# "is the mask even pointing at a real cell". The human flagged 5 kept crops
# (cell_00299/0037, cell_00994/0001, /0011, /0068, /0323) as obviously-should-be-
# discarded: the mask claims a cell, and the image at that location is empty -- the
# visible cell material sits OUTSIDE the target mask. Their mask-derived metrics
# look fine (contamination 0.10-0.26, retention 0.90-1.0), so no mask-derived check
# can ever catch this. See `verifiers/crop.py`'s own module comment above
# `_GATE_DIM_FRAC` for the full THREE-ARM gate rule (task 058 M4, REPLACES the old
# single p90-only screen this section originally tested) + its derivation against
# the real delivered v15 crops, and `crop_triage.py`'s own S1 section banner for the
# VLM channel this section tests.
# ---------------------------------------------------------------------------

def _s1_crop_image(tmp_path, name="s1.png", *, target_level: int | None = None) -> dict:
    """A crop dict with REAL on-disk pixels + a REAL mask, where the TARGET region's
    OWN brightness is directly controllable -- arm (a)'s (the dim arm's) own screen
    operand. `target_level=None` (default) makes a BRIGHT, uniform target region
    (220/255); an explicit low value (e.g. 50) makes a genuinely DIM one. Background
    is always dim (0-30) so the target/background contrast itself stays realistic
    regardless of which regime is under test.

    NOTE (task 058 M4): the target region is perfectly UNIFORM (no per-pixel
    texture) -- by construction this ALSO trips the debris/texture arm
    (`_GATE_FLAT_TEXTURE_MAX`) regardless of brightness, so a crop built here is
    NEVER a clean "fires no arm at all" fixture; the dedicated per-arm isolation
    tests below (`_gate_crop_image`) use REALISTIC per-pixel texture instead so
    each arm can be tested in true isolation. This fixture stays useful for tests
    that only need SOME candidate signal (e.g. feeding `mask_check_recommend_crop`
    directly) or that predate the three-arm gate and assert exact real `target_p90`
    values off a uniform patch."""
    from skimage.io import imsave
    rng = np.random.RandomState(0)
    size = 16
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    img = (rng.rand(size, size) * 30).astype(np.uint8)
    img[4:12, 4:12] = 220 if target_level is None else target_level
    path = tmp_path / name
    mask_path = tmp_path / f"mask_{name}"
    imsave(str(path), img)
    imsave(str(mask_path), mask)
    return {"path": str(path), "crop_path": str(path), "mask_path": str(mask_path),
           "cell_id": name, "mask_containment": 1.0, "centroid_offset": 0.0}


# `ref_p90`/`median_area` mirror the OLD screen's own implicit constants (the fixed
# 160 floor, and "no other crops" i.e. the area arm is a no-op) -- used directly
# (not derived via `_gate_image_stats`) so these tests probe `_gate_candidate`
# itself, independent of the per-image aggregation it is normally fed by.
_LEGACY_REF_P90 = 160.0
_LEGACY_MEDIAN_AREA = 0.0


def test_gate_candidate_flags_a_dim_target_region(tmp_path):
    crop = _s1_crop_image(tmp_path, "dim.png", target_level=50)
    assert _gate_candidate(crop, _LEGACY_REF_P90, _LEGACY_MEDIAN_AREA) is True
    assert "dim" in _gate_arms(crop, _LEGACY_REF_P90, _LEGACY_MEDIAN_AREA)


def test_gate_candidate_false_on_missing_mask(tmp_path):
    """No `mask_path` at all (e.g. a bare/HPA crop) -> conservatively NOT a candidate
    -- the gate fails CLOSED (never ask) on data it cannot read, not open (none of
    the three arms can read anything without a mask)."""
    crop = _s1_crop_image(tmp_path, "dim2.png", target_level=50)
    del crop["mask_path"]
    assert _gate_candidate(crop, _LEGACY_REF_P90, _LEGACY_MEDIAN_AREA) is False
    assert _gate_arms(crop, _LEGACY_REF_P90, _LEGACY_MEDIAN_AREA) == []


def test_gate_candidate_false_on_unreadable_pixels():
    """A crop with no readable image file at all -> False, no crash."""
    crop = {"path": "/no/such/file.png", "mask_path": "/no/such/mask.png"}
    assert _gate_candidate(crop, _LEGACY_REF_P90, _LEGACY_MEDIAN_AREA) is False
    assert _gate_arms(crop, _LEGACY_REF_P90, _LEGACY_MEDIAN_AREA) == []


# --------------------------------------------------------------------------- #
# Task 058 M4 (human-directed, 2026-07-17): THREE-ARM OR gate isolation tests --
# each arm is exercised on a crop crafted to trip ONLY that one signature. Unlike
# `_s1_crop_image` (a perfectly FLAT target patch -- see its own docstring, it
# ALWAYS also trips the debris/texture arm), `_gate_crop_image` below draws
# REALISTIC per-pixel texture so the three arms can be isolated from each other.
# `_GATE_REF_P90`/`_GATE_MEDIAN_AREA` are explicit, representative per-image stats
# (mirrors a typical image's own `_gate_image_stats` output) fed directly to
# `_gate_arms`/`_gate_candidate`, so these tests probe the per-crop predicate
# itself, independent of the per-image aggregation (tested separately below).
# --------------------------------------------------------------------------- #

def _round_mask(size: int, radius: int) -> np.ndarray:
    """A near-convex disk (discretization keeps solidity ~0.95-0.96, safely above
    `_GATE_SOLIDITY_MIN`) -- the "normal cell" / non-doublet shape."""
    mask = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size // 2, size // 2)
    mask[(yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= radius * radius] = 255
    return mask


def _dumbbell_mask(size: int = 36) -> np.ndarray:
    """Two touching, overlapping disks under ONE mask label -- a genuine merged-
    doublet SHAPE (a concave "waist"), solidity ~0.88, safely below `_GATE_
    SOLIDITY_MIN`. The mask is a SINGLE connected component throughout (mirrors
    `tool.py`'s own doublet docstring: a merged cell is one label,
    `single_object` scores it 1.0) -- this is a SHAPE check, not a
    fragmentation check."""
    mask = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    r = 8
    for cy, cx in ((14, 12), (14, 24)):
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 255
    return mask


def _gate_crop_image(tmp_path, name: str, mask: np.ndarray, *, level: int,
                     noise_std: float, seed: int = 0) -> dict:
    """A crop dict with REAL on-disk pixels + a REAL mask, built with a CALLER-
    SUPPLIED mask SHAPE and REALISTIC per-pixel texture (unlike `_s1_crop_image`'s
    own perfectly-flat target patch) -- lets each arm of the three-arm gate be
    exercised in true isolation, only the crafted signature fires."""
    from skimage.io import imsave
    rng = np.random.RandomState(seed)
    size = mask.shape[0]
    img = (rng.rand(size, size) * 30).astype(np.float64)
    img[mask > 0] = level
    img = np.clip(img + rng.normal(0, noise_std, size=(size, size)), 0, 255).astype(np.uint8)
    path = tmp_path / name
    mask_path = tmp_path / f"mask_{name}"
    imsave(str(path), img)
    imsave(str(mask_path), mask)
    return {"path": str(path), "crop_path": str(path), "mask_path": str(mask_path),
           "cell_id": name, "mask_containment": 1.0, "centroid_offset": 0.0}


_GATE_REF_P90 = 160.0
_GATE_MEDIAN_AREA = 1000.0


def test_gate_arms_dim_only(tmp_path):
    """A round (convex), normal-area, TEXTURED but genuinely dim target -- fires
    ONLY the dim arm."""
    crop = _gate_crop_image(tmp_path, "dim_only.png", _round_mask(60, 20),
                            level=40, noise_std=15)
    assert _gate_arms(crop, _GATE_REF_P90, _GATE_MEDIAN_AREA) == ["dim"]
    assert _gate_candidate(crop, _GATE_REF_P90, _GATE_MEDIAN_AREA) is True


def test_gate_arms_doublet_only(tmp_path):
    """A dumbbell (concave "waist") mask, bright + textured + normal-area -- fires
    ONLY the doublet arm (an intensity-bimodality signal was tried and REJECTED
    per the module comment above `_GATE_SOLIDITY_MIN` -- this is a pure SHAPE
    check, so a UNIFORM-intensity doublet still fires it)."""
    crop = _gate_crop_image(tmp_path, "doublet_only.png", _dumbbell_mask(),
                            level=200, noise_std=25)
    assert _gate_arms(crop, _GATE_REF_P90, _GATE_MEDIAN_AREA) == ["doublet"]
    assert _gate_candidate(crop, _GATE_REF_P90, _GATE_MEDIAN_AREA) is True


def test_gate_arms_debris_only_small_area(tmp_path):
    """A round (convex), bright + textured but genuinely TINY target (well below
    the image's own median crop area) -- fires ONLY the debris arm, via area."""
    crop = _gate_crop_image(tmp_path, "debris_area.png", _round_mask(20, 7),
                            level=200, noise_std=25)
    assert _gate_arms(crop, _GATE_REF_P90, _GATE_MEDIAN_AREA) == ["debris"]
    assert _gate_candidate(crop, _GATE_REF_P90, _GATE_MEDIAN_AREA) is True


def test_gate_arms_debris_only_flat_texture(tmp_path):
    """A round (convex), bright, normal-area but genuinely FLAT (featureless)
    target -- fires ONLY the debris arm, via texture."""
    crop = _gate_crop_image(tmp_path, "debris_texture.png", _round_mask(60, 20),
                            level=200, noise_std=2)
    assert _gate_arms(crop, _GATE_REF_P90, _GATE_MEDIAN_AREA) == ["debris"]
    assert _gate_candidate(crop, _GATE_REF_P90, _GATE_MEDIAN_AREA) is True


def test_gate_arms_normal_single_cell_fires_none(tmp_path):
    """A round (convex), bright, normal-area, genuinely TEXTURED target -- a
    plain, unremarkable single mid-size cell -- fires NO arm at all (not an
    mask-check candidate, the required negative control)."""
    crop = _gate_crop_image(tmp_path, "normal.png", _round_mask(60, 20),
                            level=200, noise_std=25)
    assert _gate_arms(crop, _GATE_REF_P90, _GATE_MEDIAN_AREA) == []
    assert _gate_candidate(crop, _GATE_REF_P90, _GATE_MEDIAN_AREA) is False


def test_gate_image_stats_ref_p90_uses_real_p75_when_stable(tmp_path):
    """>=`_GATE_REF_MIN_CROPS` crops with a usable `target_p90` -> `ref_p90` is
    the REAL p75 over the image's own crops, not the fixed fallback."""
    levels = [90, 110, 130, 150, 170]   # 5 crops -- exactly at the stability floor
    crops = [_s1_crop_image(tmp_path, f"ref_{i}.png", target_level=lvl)
            for i, lvl in enumerate(levels)]
    ref_p90, median_area = _gate_image_stats(crops)
    assert ref_p90 == pytest.approx(float(np.percentile(levels, 75)))
    assert ref_p90 != 160.0   # genuinely the real p75 (150), NOT coincidentally the fallback


def test_gate_image_stats_falls_back_when_ref_unstable(tmp_path):
    """<`_GATE_REF_MIN_CROPS` crops with a usable `target_p90` -> `ref_p90` falls
    back to the fixed `_S1_TARGET_P90_MAX` (160), not a p75 computed over too few
    points."""
    crops = [_s1_crop_image(tmp_path, f"unstable_{i}.png", target_level=lvl)
            for i, lvl in enumerate([100, 200, 250])]
    ref_p90, median_area = _gate_image_stats(crops)
    assert ref_p90 == 160.0


def test_gate_image_stats_median_area_over_the_image(tmp_path):
    """`median_area` is the median target-mask AREA over the image's own crops."""
    crops = [_gate_crop_image(tmp_path, f"area_{i}.png", _round_mask(60, r),
                              level=200, noise_std=25)
            for i, r in enumerate([10, 15, 20, 25, 30])]
    ref_p90, median_area = _gate_image_stats(crops)
    expected = sorted(_target_mask_area(c) for c in crops)[2]   # middle of 5
    assert median_area == pytest.approx(expected)


def test_build_s1_montage_prompt_carries_the_outside_region_narrative():
    """The coordinator's own exact instruction is LOAD-BEARING in the prompt's own
    NARRATIVE text, not only the schema field's description
    ([[judge_prompt_narrative_not_schema_field]]) -- and the grid convention + the
    index->crop_id position list, mirroring `_build_montage_prompt`'s own
    established framing."""
    index_to_crop = {0: {"cell_id": "cellA", "crop_path": "/a.png"},
                     1: {"cell_id": "cellB", "crop_path": "/b.png"}}
    prompt = _build_s1_montage_prompt(index_to_crop)
    assert "Position 0: crop_id='cellA'" in prompt
    assert "Position 1: crop_id='cellB'" in prompt
    assert "OUTSIDE the coloured region" in prompt
    assert "target_region_empty" in prompt
    assert "cell_present" in prompt


def test_apply_s1_verdicts_lands_target_region_empty_as_reject():
    crop = {"crop_path": "/a.png", "cell_id": "cellA"}
    index_to_crop = {0: crop}
    verdict = S1Verdict(verdicts=[
        S1PositionVerdict(index=0, crop_id="cellA", verdict="target_region_empty",
                          confidence="low")])   # confidence NOT gated, per the coordinator's
                                                # own literal 3-way landing spec
    verdicts = _apply_s1_verdicts(verdict, index_to_crop)
    assert crop["vision_reject"] is True
    assert crop["vision_problem"] == "target_region_empty"
    assert verdicts[0]["reject"] is True


def test_apply_s1_verdicts_cell_present_and_abstain_are_kept():
    crop_present = {"crop_path": "/a.png"}
    crop_abstain = {"crop_path": "/b.png"}
    index_to_crop = {0: crop_present, 1: crop_abstain}
    verdict = S1Verdict(verdicts=[
        S1PositionVerdict(index=0, crop_id="", verdict="cell_present", confidence="high"),
        S1PositionVerdict(index=1, crop_id="", verdict="abstain", confidence="high"),
    ])
    _apply_s1_verdicts(verdict, index_to_crop)
    assert not crop_present.get("vision_reject")
    assert not crop_abstain.get("vision_reject")


def test_apply_s1_verdicts_unmentioned_position_stays_kept():
    """A crop in the montage the judge simply never mentions -> conservative default,
    NOT rejected (mirrors `MontageVerdict`'s own "subset of positions" contract)."""
    crop = {"crop_path": "/a.png"}
    index_to_crop = {0: crop}
    verdict = S1Verdict(verdicts=[])   # judge said nothing about position 0 at all
    _apply_s1_verdicts(verdict, index_to_crop)
    assert not crop.get("vision_reject")


def test_apply_s1_verdicts_unknown_index_is_skipped(caplog):
    verdict = S1Verdict(verdicts=[
        S1PositionVerdict(index=99, crop_id="", verdict="target_region_empty", confidence="high")])
    verdicts = _apply_s1_verdicts(verdict, {0: {"crop_path": "/a.png"}})
    assert verdicts == []   # unknown index 99 -> skipped, no crash


def test_s1_recommend_crop_no_engine_or_no_candidates_is_noop(tmp_path):
    crop = _s1_crop_image(tmp_path, "dim3.png", target_level=50)
    assert s1_recommend_crop(None, [crop]) == {"judged": 0, "rejected": 0, "engine": False,
                                                "verdicts": []}
    engine = _S1StubEngine(verdict="target_region_empty")
    assert s1_recommend_crop(engine, []) == {"judged": 0, "rejected": 0, "engine": True,
                                             "verdicts": []}   # engine given, just no candidates
    assert not crop.get("vision_reject")   # never touched either way


def test_s1_recommend_crop_rejects_confirmed_candidates_and_persists_its_own_montage(tmp_path):
    meta_path = tmp_path / "cell_crops_metadata_s1test.json"
    meta_path.write_text("{}")
    crops = [_s1_crop_image(tmp_path, f"s1_{i}.png", target_level=50) for i in range(2)]
    engine = _S1StubEngine(verdict="target_region_empty", confidence="high")

    summary = s1_recommend_crop(engine, crops, metadata_path=str(meta_path))

    assert summary["judged"] == 2
    assert summary["rejected"] == 2
    assert all(c["vision_reject"] for c in crops)
    assert all(c["vision_problem"] == "target_region_empty" for c in crops)

    montages_dir = _montages_dir_for(str(meta_path), prefix="montages_s1")
    assert os.path.exists(os.path.join(montages_dir, "montage_00.png"))
    # the REJECT channel's own montages, if any, would land in a DIFFERENT subdir --
    # confirms S1 never collides with (or overwrites) that channel's own evidence.
    assert _montages_dir_for(str(meta_path), prefix="montages_s1") != _montages_dir_for(str(meta_path))


def test_s1_recommend_crop_sample_cap_evenly_samples(tmp_path):
    crops = [_s1_crop_image(tmp_path, f"cap_{i}.png", target_level=50) for i in range(6)]
    engine = _S1StubEngine(verdict="cell_present")
    summary = s1_recommend_crop(engine, crops, sample_cap=2, montage_grid=(1, 1))
    assert summary["judged"] == 2   # capped down from 6


def test_crop_verifier_s1_candidate_cap_default_is_27():
    """The coordinator's own explicit "state the cap" ask -- 27 = exactly 3 full 3x3
    montages (`triage_montage_grid`'s own default), a reasoned per-image budget."""
    assert CropVerifier().s1_candidate_cap == 27


def test_adjudicate_mask_check_cap_keeps_the_most_suspicious_candidates_not_iteration_order(tmp_path):
    """Task 058 M3: `s1_candidate_cap` (the data screen's OWN cap, name unchanged --
    "keep that gate") still applies to the mask check exactly as it did to S1 -- Task 038
    Round D's own v17-measured fix (SORTED by suspicion before capping) is UNCHANGED
    by the judge swap. Builds a population where the MOST suspicious crops are
    deliberately placed LAST in iteration order, caps at a small number, and asserts
    the cap keeps the LOWEST-`target_p90` candidates regardless of their original
    list position."""
    # 5 candidates, all below the 160 screening floor, p90 DESCENDING in list order
    # (so a crop-iteration-order cap would keep the LEAST suspicious ones first --
    # exactly backwards from what Round D fixes).
    levels = [150, 130, 110, 90, 55]
    crops = [_s1_crop_image(tmp_path, f"cap_order_{i}.png", target_level=lvl)
            for i, lvl in enumerate(levels)]
    for c, lvl in zip(crops, levels):
        c["cell_id"] = f"p90_{lvl}"

    verifier = CropVerifier()
    verifier.s1_candidate_cap = 2   # small, deliberately forces a real truncation
    verifier.llm_engine = _MaskCheckStubEngine(verdict="target_region_empty", confidence="high")
    context = {}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)

    rejected_ids = {c["cell_id"] for c in crops if c.get("vision_reject")}
    # the 2 MOST suspicious (lowest p90: 55 and 90) must be the ones judged+rejected,
    # NOT the first 2 in iteration order (150 and 130).
    assert rejected_ids == {"p90_55", "p90_90"}


def test_mask_check_candidate_ordering_lets_0068_and_0323_survive_a_27_cap_on_cell_00994_scale(tmp_path):
    """The coordinator's own explicit verification ask (task 038 Round D), still
    valid after the task 058 M3 judge swap: with the SAME `s1_candidate_cap=27` a
    real `cell_00994` run uses, do the two crops a v17 LIVE run actually MISSED
    (`cell_00994/0068`, target_p90=54.9, and `/0323`, target_p90=89.0 -- the
    outstanding 2 of the human's own 5 named crops) now land inside the cap? Mirrors
    the real v17 scale: ~55 raw candidates competing for the SAME 27-wide cap (not a
    toy population of 5), with 0068/0323 placed LAST in iteration order (worst case
    for the OLD, unsorted cap) and every OTHER candidate's own `target_p90` set
    ABOVE both of theirs (so a suspicion-blind cap would have filled up on those
    first, exactly reproducing the v17 failure)."""
    # 60 "ordinary" candidates (p90 in [140, 159], all below the 160 floor, all
    # LESS suspicious than 0068/0323) placed FIRST, then 0068/0323 placed LAST.
    ordinary = [_s1_crop_image(tmp_path, f"ord_{i}.png", target_level=140 + (i % 20))
               for i in range(60)]
    for i, c in enumerate(ordinary):
        c["cell_id"] = f"ordinary_{i}"
    crop_0068 = _s1_crop_image(tmp_path, "0068.png", target_level=55)   # matches the real 54.9
    crop_0068["cell_id"] = "cell_00994_cell_0068"
    crop_0323 = _s1_crop_image(tmp_path, "0323.png", target_level=89)   # matches the real 89.0
    crop_0323["cell_id"] = "cell_00994_cell_0323"
    crops = ordinary + [crop_0068, crop_0323]

    verifier = CropVerifier()
    assert verifier.s1_candidate_cap == 27   # the SAME real cap, unchanged this round
    verifier.llm_engine = _MaskCheckStubEngine(verdict="target_region_empty", confidence="high")
    context = {}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)

    assert crop_0068.get("vision_reject") is True, "0068 (the MOST suspicious) must survive the cap"
    assert crop_0323.get("vision_reject") is True, "0323 must survive the cap"


def test_adjudicate_headless_never_calls_mask_check_and_crops_are_byte_identical(tmp_path):
    """"headless must behave EXACTLY as today" (the coordinator's own explicit
    requirement, task 038 Round B -- STILL true after the task 058 M3 judge swap):
    no `llm_engine` -> the mask check never even LOOKS at the crops (the screen itself is
    pure data, always runs -- but with no engine, `mask_check_recommend_crop` is never
    invoked at all, per the SAME `getattr(self, "llm_engine", None) is not None`
    gate every other channel uses), so no crop is ever touched -- every prior
    mechanism number stays reproducible."""
    verifier = CropVerifier()   # no llm_engine wired
    crops = [_s1_crop_image(tmp_path, f"h{i}.png", target_level=50) for i in range(3)]
    before = [dict(c) for c in crops]
    context = {}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)
    assert crops == before   # byte-identical -- not even a "vision_reject" key added
    assert all("vision_reject" not in c for c in crops)


def test_adjudicate_mask_check_reject_excludes_crop_from_kept_manifest(tmp_path):
    """End-to-end wiring (mirrors `test_c4_...`'s own established pattern): a crop
    with a signal-poor target region, run through the REAL `CropVerifier.adjudicate`
    with an engine that confirms `target_region_empty`, is excluded from the
    downstream kept-crops manifest -- the mask check's own landing IS a population-membership
    constraint, verified through the actual persisted artifact, not just the
    in-memory flag."""
    tool_cache_dir = tmp_path / "tool_cache"
    tool_cache_dir.mkdir()
    meta_path, crop_paths = _write_two_crop_metadata(tool_cache_dir, "cell_crops_metadata_s1e2e.json")
    crops, context = adapt_crop(str(meta_path))

    # crop 0 (gate-valid) gets a REAL dim target region + mask -- a mask-check candidate;
    # crop 1 is left alone (bright/no mask -- never becomes a candidate at all).
    dim_crop = _s1_crop_image(tmp_path, "e2e_dim.png", target_level=50)
    crops[0]["path"] = dim_crop["path"]
    crops[0]["crop_path"] = dim_crop["path"]
    crops[0]["mask_path"] = dim_crop["mask_path"]

    verifier = CropVerifier()
    verifier.llm_engine = _MaskCheckStubEngine(verdict="target_region_empty", confidence="high")
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)

    assert crops[0].get("vision_reject") is True
    assert crops[0].get("vision_problem") == "target_region_empty"
    assert crops[0].get("mask_check_verdict") == "target_region_empty"

    manifest_path = tool_cache_dir / "cell_crops_kept_manifest_s1e2e.json"
    kept = json.loads(manifest_path.read_text())["kept_crop_paths"]
    assert crops[0]["crop_path"] not in kept


def test_adjudicate_mask_check_verdict_lands_in_the_rich_per_crop_triage_trace(tmp_path):
    """Task 058 M3 (human directive: "过程要体现清晰"): `verifiers/crop.py::adjudicate`
    OVERWRITES the SAME `*_triage_trace.json` sidecar `vision_recommend_crop` already
    writes with the richer, per-crop-complete trace (`_build_crop_decision_trace`) --
    still ONE sidecar file (extends it, per CARD.md's own naming idiom -- no new
    format), now carrying an entry for EVERY crop (not just the ones a channel
    judged), each with `mask_check`/`final` reflecting the crop's own actual outcome."""
    tool_cache_dir = tmp_path / "tool_cache"
    tool_cache_dir.mkdir()
    meta_path, crop_paths = _write_two_crop_metadata(tool_cache_dir, "cell_crops_metadata_s1trace.json")
    crops, context = adapt_crop(str(meta_path))

    dim_crop = _s1_crop_image(tmp_path, "trace_dim.png", target_level=50)
    crops[0]["path"] = dim_crop["path"]
    crops[0]["crop_path"] = dim_crop["path"]
    crops[0]["mask_path"] = dim_crop["mask_path"]
    crops[0]["cell_id"] = "dim_target_cell"
    # crop 1 already has border_touch=0.6 (see _write_two_crop_metadata) -> a genuine
    # REJECT (cut_off) candidate too, independent of the mask check.

    verifier = CropVerifier()
    verifier.llm_engine = _MaskCheckStubEngine(verdict="target_region_empty", confidence="high")
    # _MaskCheckStubEngine only answers MaskCheckVerdict asks -- the REJECT channel's own
    # MontageVerdict ask would raise on it; that is fine here, `vision_recommend_crop`
    # catches the failure and falls back gracefully, so crop 1 is simply never
    # rejected by (B) -- only the mask check's own verdict for crop 0 needs to land in the trace.
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)

    trace_path = tool_cache_dir / "cell_crops_triage_trace_s1trace.json"
    trace = json.loads(trace_path.read_text())["verdicts"]
    assert len(trace) == 2   # ONE entry per crop, not just the judged one
    dim_entry = next(e for e in trace if e["cell_id"] == "dim_target_cell")
    assert dim_entry["mask_check"]["sent_to_vlm"] is True
    assert dim_entry["mask_check"]["verdict"] == "target_region_empty"
    assert dim_entry["final"]["kept"] is False
    assert dim_entry["final"]["reason"] == "target_region_empty"


# --------------------------------------------------------------------------- #
# ACCEPTANCE BENCHMARK (coordinator-directed): the human's 5 named crops ARE the
# gold standard. `_S1_MEASURED_TARGET_P90` below is NOT a synthetic guess -- it is
# the EXACT `target_p90` measured directly off the real delivered v15 crop files
# for cell_00299/0037, cell_00994/0001, /0011, /0068, /0323 (see CODING_LOG.md for
# the measurement script + full derivation). This benchmark asserts the DATA SCREEN
# genuinely covers all 5 real values (not a re-derived/rounded approximation), and
# that a CORRECT VLM verdict on a signal-poor target lands as a rejection through
# the REAL, wired `CropVerifier.adjudicate` path with a MOCKED engine. Per the
# coordinator's own explicit instruction, the screen was NOT tuned to these 5 --
# `_S1_TARGET_P90_MAX` (160) was fixed BEFORE checking how tightly it could be
# shrunk around them (see `verifiers/crop.py`'s own module comment).
# --------------------------------------------------------------------------- #
_S1_MEASURED_TARGET_P90 = {
    "cell_00299_cell_0037": 138.0,
    "cell_00994_cell_0001": 112.8,
    "cell_00994_cell_0011": 113.4,
    "cell_00994_cell_0068": 54.9,
    "cell_00994_cell_0323": 89.0,
}


def test_gate_candidate_covers_all_5_human_flagged_crops_real_measured_values(tmp_path):
    """The THREE-ARM gate genuinely flags all 5 -- built from crops whose OWN target
    region is a UNIFORM patch at the crop's real, measured `target_p90` (a 90th
    percentile of a uniform patch equals that patch's own value), so `_gate_candidate`
    exercises the EXACT real numbers, not a reconstruction. `_s1_crop_image`'s target
    patch is perfectly flat (see its own docstring), so every one of these 5 is ALSO
    caught by the debris/texture arm regardless of the dim arm's own verdict -- an
    even stronger (redundant-OR) coverage guarantee than the old single screen had."""
    for cell_id, p90 in _S1_MEASURED_TARGET_P90.items():
        crop = _s1_crop_image(tmp_path, f"{cell_id}.png", target_level=int(p90))
        assert _gate_candidate(crop, _LEGACY_REF_P90, _LEGACY_MEDIAN_AREA) is True, (
            f"{cell_id} (target_p90={p90}) was NOT screened in -- the gate must "
            f"cover every one of the coordinator's own 5 named crops")


def test_mask_check_pipeline_rejects_the_5_human_flagged_crop_pattern_end_to_end(tmp_path):
    """The FULL, REAL `CropVerifier.adjudicate` path (data screen -> mask-check montage ->
    landing), driven with a MOCKED engine that answers exactly what a CORRECT VLM
    should for this pattern (`target_region_empty`) -- all 5 crops, built at their
    OWN real measured `target_p90`, are screened in AND end up excluded from the
    kept population. If a live VLM run cannot reproduce `target_region_empty` on
    these crops, that is an honest negative result about the VLM's OWN judgment --
    this test only proves the PIPELINE (screen -> montage -> landing) correctly
    excludes a crop when the judge DOES confirm it, not that a live judge will."""
    crops = [_s1_crop_image(tmp_path, f"{cell_id}.png", target_level=int(p90))
            for cell_id, p90 in _S1_MEASURED_TARGET_P90.items()]
    for c, cell_id in zip(crops, _S1_MEASURED_TARGET_P90):
        c["cell_id"] = cell_id

    verifier = CropVerifier()
    verifier.llm_engine = _MaskCheckStubEngine(verdict="target_region_empty", confidence="high")
    context = {}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)

    for c, cell_id in zip(crops, _S1_MEASURED_TARGET_P90):
        assert c.get("vision_reject") is True, f"{cell_id} was not rejected end-to-end"
        assert c.get("vision_problem") == "target_region_empty"
        assert c.get("mask_check_verdict") == "target_region_empty"


# --------------------------------------------------------------------------- #
# Task 058 M3 (human-authorized controller change): MASK CHECK's own dedicated tests --
# the contour-only rendering, the mask-derived stretch, the richer verdict schema
# (cell_present/target_region_empty/wrong_object/merged/abstain), and
# `mask_check_recommend_crop`'s own wiring. Mirrors the S1 block above's own test
# structure and fixtures (`_s1_crop_image`) -- the mask check is a sibling channel, not a
# reimplementation.
# --------------------------------------------------------------------------- #

def test_mask_check_thin_contour_overlay_marks_edge_only_no_fill():
    """The human directive this round supersedes the fill entirely (contour-only,
    `_PROBE_FILL_ALPHA=0.0` in the validated probe this is ported from): the
    interior fill pixels are UNCHANGED from the plain base image (label2rgb's own
    fill blend at alpha=0.0 is a mathematical no-op, +-1 float-rounding noise only --
    measured, not just argued, mirroring the M2 round's own direct pixel check), and
    the contour itself is EXACTLY pure green (BGR), OPAQUE -- never blended toward
    the underlying pixel value."""
    from shape.agent.verification.crop_triage import _mask_check_thin_contour_overlay

    base = (np.random.RandomState(0).rand(16, 16) * 200 + 20).astype(np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255

    overlay = _mask_check_thin_contour_overlay(base, mask)
    assert overlay.shape == (16, 16, 3)

    interior = overlay[6, 6]   # well inside the mask box, away from any edge
    expected = int(base[6, 6])
    assert all(abs(int(v) - expected) <= 1 for v in interior)   # no fill tint

    contour_pixel = overlay[4, 6]   # the mask's own top edge
    assert tuple(int(v) for v in contour_pixel) == (0, 255, 0)   # pure green, opaque


def test_mask_check_stretched_pixels_reveals_a_dim_target_masked_by_a_bright_neighbour():
    """LOAD-BEARING for the mask check specifically (see the module's own section banner):
    a genuinely dim target region, embedded in a tile that ALSO contains a bright,
    unrelated neighbour, is revealed by the MASK-DERIVED window stretch -- a
    WHOLE-TILE stretch (the plain `normalize_for_overlay` the other channels' own
    overlay relies on via `to_display_uint8`) is dominated by the bright neighbour
    and leaves the SAME dim target barely moved, the exact regression the mask
    window exists to prevent (ported from the cropper probe's own measured SCOPE
    FIX)."""
    from shape.agent.verification.crop_triage import _mask_check_stretched_pixels
    from shape.utils.segmentation_qc import normalize_for_overlay

    rng = np.random.RandomState(0)
    size = 32
    pixels = (rng.rand(size, size) * 20).astype(np.uint8)   # dim background, 0-20
    mask_bool = np.zeros((size, size), dtype=bool)
    mask_bool[10:20, 10:20] = True
    pixels[mask_bool] = (35 + rng.rand(100) * 10).astype(np.uint8)   # target, ~35-45
    pixels[0:5, 0:5] = 250   # a BRIGHT, unrelated neighbour elsewhere in the tile

    mask_window_stretched = _mask_check_stretched_pixels(pixels, mask_bool)
    mask_window_p90 = float(np.percentile(mask_window_stretched[mask_bool], 90))

    whole_tile_stretched = normalize_for_overlay(pixels)
    whole_tile_p90 = float(np.percentile(whole_tile_stretched[mask_bool], 90))

    assert mask_window_p90 > whole_tile_p90 + 100   # genuinely reveals it
    assert mask_window_p90 > 200


def test_mask_check_overlay_array_falls_back_to_plain_grayscale_without_mask(tmp_path):
    from shape.agent.verification.crop_triage import _mask_check_overlay_array
    crop = _crop_image(tmp_path, "no_mask_mask_check.png")
    overlay = _mask_check_overlay_array(crop)
    assert overlay is not None
    assert overlay.shape == (16, 16, 3)


def test_mask_check_overlay_array_none_on_unreadable_pixels():
    from shape.agent.verification.crop_triage import _mask_check_overlay_array
    assert _mask_check_overlay_array({"path": "/no/such/file.png"}) is None


def test_build_mask_check_montage_prompt_carries_all_four_verdict_narrative():
    """The four-way mask-error question is LOAD-BEARING in the prompt's own
    NARRATIVE text, not only the schema field's description
    ([[judge_prompt_narrative_not_schema_field]])."""
    index_to_crop = {0: {"cell_id": "cellA", "crop_path": "/a.png"},
                     1: {"cell_id": "cellB", "crop_path": "/b.png"}}
    prompt = _build_mask_check_montage_prompt(index_to_crop)
    assert "Position 0: crop_id='cellA'" in prompt
    assert "Position 1: crop_id='cellB'" in prompt
    assert "cell_present" in prompt
    assert "target_region_empty" in prompt
    assert "wrong_object" in prompt
    assert "merged" in prompt
    assert "TWO touching cells" in prompt


# The mask check prompt and its response schema are the judge's whole information, so a change to them must be
# deliberate (as the batch prompt lock, `test_batch_budgeted_loop.py`): the sha256 of the rendered prompt for
# fixed positions ("no_positions" is the text `benchmarks/cropper_NeurIPS22-CellSeg/benchmark.py::_prompt_hashes`
# records as `mask_check_prompt_sha256`) and of the schema. A deliberate change updates the hashes with it.
MASK_CHECK_PROMPT_SHA256 = {
    # 2026-10-17: prompt-cache layout -- CARD rules + fixed instructions moved to the FRONT
    # (stable prefix), the varying candidate count + positions to the END. Token-cost fix;
    # wording unchanged. Was b727f4b6.../bb4209ee...
    "no_positions": "a86283c1236b8323d8b46b82ce3f42a7914b4f5d4c9fc95c1c608ce65591f664",
    "two_positions": "51dc15837a1a8fc959f13325f9dabb8e91b109704801f79b47f9637ee118938b",
}
# 2026-10-05: "arm C" renamed "mask check" (class titles and one description); was d605cce6...b9c185
MASK_CHECK_SCHEMA_SHA256 = "69340aa15347916fe7243b90cf60e4aa929000838edb768954b03226f36e5b9e"


def test_the_mask_check_prompt_is_locked_and_carries_only_the_cards_rule_text():
    import hashlib

    from shape.agent.verification.crop_triage import _cropper_judge_rules

    prompts = {"no_positions": _build_mask_check_montage_prompt({}),
               "two_positions": _build_mask_check_montage_prompt({0: {"cell_id": "cellA", "crop_path": "/a.png"},
                                                              1: {"cell_id": "cellB", "crop_path": "/b.png"}})}
    assert {name: hashlib.sha256(p.encode("utf-8")).hexdigest() for name, p in prompts.items()} == MASK_CHECK_PROMPT_SHA256
    schema = json.dumps(MaskCheckVerdict.model_json_schema(), default=str)
    assert hashlib.sha256(schema.encode("utf-8")).hexdigest() == MASK_CHECK_SCHEMA_SHA256
    # the CARD's rule text the judge needs: the purpose and quality order, the readouts' definitions
    rules = _cropper_judge_rules()
    # prompt-cache layout: the CARD rules now lead the prompt (stable prefix), not trail it
    assert rules and rules in prompts["no_positions"] and prompts["no_positions"].lstrip().startswith("TOOL CARD")
    assert rules.startswith("## Purpose") and "\n## 3. CALIBRATION" in rules
    assert "A high-quality crop is, in this order:" in rules and "- **neighbor_contamination" in rules
    assert all(h not in rules for h in ("## Agentic VRR", "## Levers", "## 2.", "## 4.", "## Development notes"))
    # no history, dates, task references, token counts or findings, and no pointer to a section it does not carry
    for pattern in (r"\b20\d\d-\d\d-\d\d\b", r"\btask \d", r"\bv\d+\b", r"Sherlock", r"token",
                    r"human[- ](locked|decision|rule|redirect)", r"\d+/\d+", r"archives/", r"\bSS\d", r"Levers"):
        assert not re.search(pattern, rules), pattern


def test_apply_mask_check_verdicts_sets_verdict_and_confidence_in_place_always():
    """`crop["mask_check_verdict"]`/`crop["mask_check_confidence"]` are set IN PLACE
    UNCONDITIONALLY, even on a keep verdict -- the raw call is recorded for the
    trace + the benchmark's own GT-adjudication validation, not only a reject
    reason."""
    crop = {"crop_path": "/a.png"}
    index_to_crop = {0: crop}
    verdict = MaskCheckVerdict(verdicts=[
        MaskCheckPositionVerdict(index=0, crop_id="", verdict="cell_present", confidence="low")])
    _apply_mask_check_verdicts(verdict, index_to_crop)
    assert crop["mask_check_verdict"] == "cell_present"
    assert crop["mask_check_confidence"] == "low"
    assert not crop.get("vision_reject")


def test_apply_mask_check_verdicts_rejects_target_region_empty_wrong_object_and_merged():
    """The richer verdict space the mask check adds over S1: `wrong_object` and `merged` are
    BOTH reject reasons, same as `target_region_empty` -- REGARDLESS of confidence
    (mirrors the S1 landing spec this channel replaces, no confidence qualifier)."""
    order = ("target_region_empty", "wrong_object", "merged")
    crops = {v: {"crop_path": f"/{v}.png"} for v in order}
    index_to_crop = {i: crops[v] for i, v in enumerate(order)}
    verdict = MaskCheckVerdict(verdicts=[
        MaskCheckPositionVerdict(index=i, crop_id="", verdict=v, confidence="low")
        for i, v in enumerate(order)])
    _apply_mask_check_verdicts(verdict, index_to_crop)
    for v in order:
        assert crops[v]["vision_reject"] is True, v
        assert crops[v]["vision_problem"] == v
        assert crops[v]["mask_check_verdict"] == v


def test_apply_mask_check_verdicts_cell_present_and_abstain_are_kept():
    crop_present = {"crop_path": "/a.png"}
    crop_abstain = {"crop_path": "/b.png"}
    index_to_crop = {0: crop_present, 1: crop_abstain}
    verdict = MaskCheckVerdict(verdicts=[
        MaskCheckPositionVerdict(index=0, crop_id="", verdict="cell_present", confidence="high"),
        MaskCheckPositionVerdict(index=1, crop_id="", verdict="abstain", confidence="high"),
    ])
    _apply_mask_check_verdicts(verdict, index_to_crop)
    assert not crop_present.get("vision_reject")
    assert not crop_abstain.get("vision_reject")


def test_apply_mask_check_verdicts_unmentioned_position_stays_kept():
    crop = {"crop_path": "/a.png"}
    index_to_crop = {0: crop}
    verdict = MaskCheckVerdict(verdicts=[])   # judge said nothing about position 0 at all
    _apply_mask_check_verdicts(verdict, index_to_crop)
    assert not crop.get("vision_reject")


def test_apply_mask_check_verdicts_unknown_index_is_skipped(caplog):
    verdict = MaskCheckVerdict(verdicts=[
        MaskCheckPositionVerdict(index=99, crop_id="", verdict="merged", confidence="high")])
    verdicts = _apply_mask_check_verdicts(verdict, {0: {"crop_path": "/a.png"}})
    assert verdicts == []   # unknown index 99 -> skipped, no crash


def test_mask_check_recommend_crop_no_engine_or_no_candidates_is_noop(tmp_path):
    crop = _s1_crop_image(tmp_path, "mask_check_noop.png", target_level=50)
    # `reused` counts the verdicts taken from the mask-check verdict cache: none here
    assert mask_check_recommend_crop(None, [crop]) == {"judged": 0, "rejected": 0, "reused": 0,
                                                   "engine": False, "verdicts": [], "montages": []}
    engine = _MaskCheckStubEngine(verdict="target_region_empty")
    assert mask_check_recommend_crop(engine, []) == {"judged": 0, "rejected": 0, "reused": 0,
                                                "engine": True, "verdicts": [], "montages": []}
    assert not crop.get("vision_reject")   # never touched either way


def test_mask_check_recommend_crop_rejects_confirmed_candidates_and_persists_its_own_montage(tmp_path):
    meta_path = tmp_path / "cell_crops_metadata_armctest.json"
    meta_path.write_text("{}")
    crops = [_s1_crop_image(tmp_path, f"mask_check_{i}.png", target_level=50) for i in range(2)]
    engine = _MaskCheckStubEngine(verdict="merged", confidence="high")

    summary = mask_check_recommend_crop(engine, crops, metadata_path=str(meta_path))

    assert summary["judged"] == 2
    assert summary["rejected"] == 2
    assert all(c["vision_reject"] for c in crops)
    assert all(c["vision_problem"] == "merged" for c in crops)
    assert all(c["mask_check_verdict"] == "merged" for c in crops)

    montages_dir = _montages_dir_for(str(meta_path), prefix="montages_mask_check")
    assert os.path.exists(os.path.join(montages_dir, "montage_00.png"))
    # a SEPARATE subdirectory from every other channel's own montages for the same trial.
    assert montages_dir != _montages_dir_for(str(meta_path))
    assert montages_dir != _montages_dir_for(str(meta_path), prefix="montages_s1")


def test_mask_check_recommend_crop_sample_cap_evenly_samples(tmp_path):
    crops = [_s1_crop_image(tmp_path, f"mask_check_cap_{i}.png", target_level=50) for i in range(6)]
    engine = _MaskCheckStubEngine(verdict="cell_present")
    summary = mask_check_recommend_crop(engine, crops, sample_cap=2, montage_grid=(1, 1))
    assert summary["judged"] == 2


def test_mask_check_records_each_montage_question_and_each_verdict(tmp_path):
    """Per montage: its index, the prompt sent, the image as a ref (never the bytes), the saved montage, the
    verdict object as JSON and the error (a gateway error keeps the batch); per judged crop: crop_id, verdict,
    confidence, reject, montage_path."""
    import hashlib

    meta_path = tmp_path / "cell_crops_metadata_record.json"
    meta_path.write_text("{}")
    crops = [_s1_crop_image(tmp_path, f"mask_check_rec_{i}.png", target_level=50) for i in range(2)]
    for i, crop in enumerate(crops):
        crop["cell_id"] = f"cell{i}"

    class OneFails(_MaskCheckStubEngine):
        def generate(self, content, response_format=None, **kwargs):
            if "crop_id='cell1'" in content[0]:
                self.calls.append(content)
                return {"error": "rate_limit", "message": "slow down"}
            return super().generate(content, response_format=response_format, **kwargs)

    engine = OneFails(verdict="merged")
    summary = mask_check_recommend_crop(engine, crops, metadata_path=str(meta_path), montage_grid=(1, 1))
    assert summary["judged"] == 1 and summary["rejected"] == 1 and not crops[1].get("vision_reject")
    sent = {c[0]: c[1] for c in engine.calls}
    first, second = summary["montages"]
    assert set(first) == {"index", "prompt", "image", "montage_path", "reply", "error"}
    assert [first["index"], second["index"]] == [0, 1] and first["prompt"] in sent and second["prompt"] in sent
    png = sent[first["prompt"]]
    assert first["image"] == {"label": "MASK CHECK MONTAGE 0", "sha256": hashlib.sha256(png).hexdigest(),
                              "bytes": len(png)}
    assert os.path.exists(first["montage_path"]) and first["error"] is None
    assert first["reply"] == {"verdicts": [{"index": 0, "crop_id": "", "verdict": "merged", "confidence": "high"}]}
    assert second["error"] == "rate_limit: slow down"
    assert second["reply"] == {"error": "rate_limit", "message": "slow down"}
    assert [{k: v[k] for k in ("crop_id", "verdict", "confidence", "reject", "montage_path")}
            for v in summary["verdicts"]] == [{"crop_id": "cell0", "verdict": "merged", "confidence": "high",
                                               "reject": True, "montage_path": first["montage_path"]}]
    assert "PNG" not in json.dumps(summary["montages"])


def test_mask_check_judges_its_montages_at_once_with_the_serial_outcome(tmp_path):
    """The montages are asked concurrently (the stub only answers once all three questions are in flight)
    and the verdicts are applied in chunk order: the result is the serial one."""
    import threading

    crops = [_s1_crop_image(tmp_path, f"mask_check_par_{i}.png", target_level=50) for i in range(3)]
    for i, crop in enumerate(crops):
        crop["cell_id"] = f"cell{i}"
    barrier = threading.Barrier(3, timeout=10)

    class Concurrent(_MaskCheckStubEngine):
        def generate(self, content, response_format=None, **kwargs):
            barrier.wait()                                # BrokenBarrierError unless all three overlap
            self.calls.append(content)
            crop_id = re.search(r"crop_id='([^']+)'", content[0]).group(1)
            return {"content": MaskCheckVerdict(verdicts=[MaskCheckPositionVerdict(
                index=0, crop_id="", verdict="merged" if crop_id == "cell1" else "cell_present",
                confidence="high")])}

    cache: dict = {}
    summary = mask_check_recommend_crop(Concurrent(), crops, montage_grid=(1, 1), verdict_cache=cache)
    assert summary["judged"] == 3 and summary["rejected"] == 1
    assert [v["verdict"] for v in summary["verdicts"]] == ["cell_present", "merged", "cell_present"]
    assert [bool(c.get("vision_reject")) for c in crops] == [False, True, False]
    assert all(c["mask_check_sent"] for c in crops)
    assert set(cache) == {"cell0", "cell2"}              # merged is geometry-dependent: never cached


def test_usage_record_counts_every_concurrent_call():
    import threading

    from shape.engine import usage

    before = usage.snapshot()
    threads = [threading.Thread(target=lambda: [usage.record({"total_tokens": 1}) for _ in range(2000)])
               for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    after = usage.snapshot()
    assert after["total_tokens"] - before["total_tokens"] == 16000
    assert after["calls"] - before["calls"] == 16000


# --------------------------------------------------------------------------- #
# Task 058 M3 requirement 3 ("过程要体现清晰"): the CLEAR per-crop decision trace --
# {cell_id, geometry, data_screen, mask_check, final} for EVERY crop, not just the ones a
# channel judged.
# --------------------------------------------------------------------------- #

def test_crop_decision_trace_has_all_four_stages_for_every_outcome(tmp_path):
    """ONE round, THREE distinct outcomes: never-flagged (bright + textured +
    normal-area + convex -- fires no arm, auto-kept), flagged+sent+rejected (dim,
    the mask check confirms), and flagged-but-CAPPED (dim, but `s1_candidate_cap` drops it
    before it ever reaches the judge -- visibly different from never-flagged:
    `data_screen.verdict=="flagged"` but `mask_check.sent_to_vlm is False`).

    Task 058 M4: `bright` uses `_gate_crop_image` (REALISTIC texture), not
    `_s1_crop_image` (a perfectly FLAT patch that would ALSO trip the debris/
    texture arm regardless of brightness -- see that fixture's own docstring) --
    otherwise this population's own "never-flagged" outcome could not exist at
    all under the three-arm gate."""
    tool_cache_dir = tmp_path / "tool_cache"
    tool_cache_dir.mkdir()
    meta_path = tool_cache_dir / "cell_crops_metadata_tracetest.json"
    meta_path.write_text("{}")

    bright = _gate_crop_image(tmp_path, "bright_c.png", _round_mask(60, 20),
                              level=200, noise_std=25)
    bright["cell_id"] = "bright_cell"
    bright["margin_mode"] = "mask_bbox"
    bright["qc"] = {"neighbor_contamination": 0.0, "target_retention": 1.0,
                    "contamination_inset_px": 0}

    dim_rejected = _s1_crop_image(tmp_path, "dim_reject_c.png", target_level=50)
    dim_rejected["cell_id"] = "dim_rejected_cell"
    dim_rejected["margin_mode"] = "mask_bbox"
    dim_rejected["qc"] = {"neighbor_contamination": 0.1, "target_retention": 0.95,
                          "contamination_inset_px": 2}

    dim_capped = _s1_crop_image(tmp_path, "dim_capped_c.png", target_level=60)
    dim_capped["cell_id"] = "dim_capped_cell"
    dim_capped["margin_mode"] = "mask_bbox"
    dim_capped["qc"] = {"neighbor_contamination": 0.05, "target_retention": 0.97,
                        "contamination_inset_px": 1}

    crops = [bright, dim_rejected, dim_capped]

    verifier = CropVerifier()
    verifier.s1_candidate_cap = 1   # forces exactly ONE of the two dim crops through
    verifier.llm_engine = _MaskCheckStubEngine(verdict="target_region_empty", confidence="high")
    context = {"metadata_path": str(meta_path)}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)

    trace_path = tool_cache_dir / "cell_crops_triage_trace_tracetest.json"
    trace = {e["cell_id"]: e for e in json.loads(trace_path.read_text())["verdicts"]}
    assert set(trace) == {"bright_cell", "dim_rejected_cell", "dim_capped_cell"}
    # the new nested keys are present on EVERY entry; legacy-flat compatibility keys
    # (item_id/reject/verdict/confidence/problem/reasoning/montage_path) ALSO
    # coexist -- another benchmark (run_shape_crop.py, outside this task's own
    # authorized files) reads those directly off every trace entry (see
    # `_build_crop_decision_trace`'s own COMPATIBILITY section for the real
    # cross-benchmark test failure this guards against).
    for entry in trace.values():
        assert {"cell_id", "geometry", "data_screen", "mask_check", "final"} <= set(entry)
        assert {"item_id", "reject", "verdict", "confidence", "problem",
               "reasoning", "montage_path"} <= set(entry)

    bright_e = trace["bright_cell"]
    assert bright_e["geometry"] == {"margin_mode": "mask_bbox", "inset_px": 0,
                                    "neighbor_contamination": 0.0, "target_retention": 1.0}
    assert bright_e["data_screen"]["verdict"] == "not_flagged"
    assert bright_e["data_screen"]["arms_fired"] == []   # task 058 M4: none of the 3 arms fired
    assert bright_e["mask_check"] == {"sent_to_vlm": False, "verdict": None, "confidence": None,
                                 "montage_path": None}
    assert bright_e["final"] == {"kept": True, "reason": "kept"}
    assert bright_e["reject"] is False and bright_e["verdict"] == "keep"   # legacy keys agree

    rejected_e = trace["dim_rejected_cell"]
    assert rejected_e["data_screen"]["verdict"] == "flagged"
    assert "dim" in rejected_e["data_screen"]["arms_fired"]   # task 058 M4: WHICH arm(s) fired
    assert rejected_e["mask_check"]["sent_to_vlm"] is True
    assert rejected_e["mask_check"]["verdict"] == "target_region_empty"
    assert rejected_e["mask_check"]["confidence"] == "high"
    assert rejected_e["mask_check"]["montage_path"] is not None
    assert rejected_e["final"] == {"kept": False, "reason": "target_region_empty"}
    # legacy-flat keys (run_shape_crop.py's own reader) must agree with `final`
    assert rejected_e["reject"] is True
    assert rejected_e["verdict"] == "reject"
    assert rejected_e["problem"] == "target_region_empty"

    capped_e = trace["dim_capped_cell"]
    assert capped_e["data_screen"]["verdict"] == "flagged"   # the screen DID flag it...
    assert "dim" in capped_e["data_screen"]["arms_fired"]
    assert capped_e["mask_check"]["sent_to_vlm"] is False          # ...but the cap dropped it
    assert capped_e["mask_check"]["verdict"] is None
    assert capped_e["final"]["kept"] is True   # never rejected -- keep-biased default


def test_crop_decision_trace_sent_to_vlm_false_when_render_skips_unreadable_overlay(tmp_path):
    """Task 058 M7 (Codex QC finding 2 fix): `sent_to_vlm` must reflect what
    ACTUALLY reached a montage, not the pre-render candidate list. A dumbbell
    (doublet) mask fires the "doublet" arm PURELY from mask solidity
    (`_target_solidity` -- no pixel read at all, see `_gate_arms_doublet_only`
    above), so this crop IS a genuine mask-check candidate -- but its own crop-image
    path is then pointed at a file that does not exist, so `_crop_pixels`/
    `_mask_check_overlay_array` return None and `_build_mask_check_montage` SKIPS it
    (`crop_triage.py`'s own `if overlay is None: continue`) -- it never reaches
    `index_to_crop`, so `_apply_mask_check_verdicts` never sets `crop["mask_check_verdict"]`.
    The OLD `cell_id in mask_check_sent_ids` check could not see this render-time drop
    (the data screen DID flag it as a candidate) and would have reported
    `sent_to_vlm=True` here -- wrong, since the montage image was never actually
    built/sent."""
    tool_cache_dir = tmp_path / "tool_cache"
    tool_cache_dir.mkdir()
    meta_path = tool_cache_dir / "cell_crops_metadata_rendertest.json"
    meta_path.write_text("{}")

    crop = _gate_crop_image(tmp_path, "render_skip.png", _dumbbell_mask(),
                            level=200, noise_std=25)
    crop["crop_path"] = crop["path"] = str(tmp_path / "does_not_exist.png")
    crop["qc"] = {"neighbor_contamination": 0.0, "target_retention": 1.0,
                 "contamination_inset_px": 0}
    crop["margin_mode"] = "mask_bbox"

    verifier = CropVerifier()
    verifier.llm_engine = _MaskCheckStubEngine(verdict="target_region_empty", confidence="high")
    context = {"metadata_path": str(meta_path)}
    crops = [crop]
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)

    assert crops[0].get("mask_check_verdict") is None   # never judged -- skipped at render

    trace_path = tool_cache_dir / "cell_crops_triage_trace_rendertest.json"
    trace = {e["cell_id"]: e for e in json.loads(trace_path.read_text())["verdicts"]}
    entry = trace["render_skip.png"]
    assert entry["data_screen"]["verdict"] == "flagged"       # the screen DID flag it...
    assert "doublet" in entry["data_screen"]["arms_fired"]
    assert entry["mask_check"]["sent_to_vlm"] is False              # ...but render skipped it
    assert entry["mask_check"]["verdict"] is None


def test_mask_check_recommend_crop_marks_sent_true_for_subset_response_omission(tmp_path):
    """Task 058 M8 (QC-blocking fix, the reviewer's OWN exact probe): a genuine
    2-crop candidate batch renders into ONE 2-position montage and is dispatched to
    `engine.generate`, but the (schema-valid) response names position 0 ONLY --
    `MaskCheckVerdict` explicitly permits this SUBSET shape. Both crops were genuinely
    SENT; the omitted one must still read `mask_check_sent=True` even though
    `mask_check_verdict` stays None (never discard on silence)."""
    crops = [_gate_crop_image(tmp_path, f"subset_{i}.png", _dumbbell_mask(), level=200,
                              noise_std=25, seed=i) for i in range(2)]
    engine = _MaskCheckPartialStubEngine(answer_indices=(0,), verdict="cell_present")

    summary = mask_check_recommend_crop(engine, crops, montage_grid=(1, 2))

    assert len(engine.calls) == 1                       # ONE montage call, both positions in it
    assert summary["judged"] == 2                        # both counted as judged/sent
    assert crops[0]["mask_check_sent"] is True
    assert crops[0]["mask_check_verdict"] == "cell_present"
    assert crops[1]["mask_check_sent"] is True                 # genuinely sent...
    assert crops[1].get("mask_check_verdict") is None            # ...but omitted by the subset response
    assert not crops[1].get("vision_reject")                 # keep-biased: never discard on silence


def test_mask_check_recommend_crop_marks_sent_true_for_empty_verdict_list(tmp_path):
    """A successful engine call that returns an EMPTY verdict list (a malformed/
    no-verdict response `MaskCheckVerdict` still permits schema-wise) must still mark
    every crop in that batch `mask_check_sent=True` -- the trace must never read the
    response's own completeness back INTO whether the crop was sent."""
    crops = [_gate_crop_image(tmp_path, f"empty_{i}.png", _dumbbell_mask(), level=200,
                              noise_std=25, seed=i) for i in range(2)]
    engine = _MaskCheckPartialStubEngine(answer_indices=())

    summary = mask_check_recommend_crop(engine, crops, montage_grid=(1, 2))

    assert summary["judged"] == 2
    assert summary["rejected"] == 0
    assert all(c["mask_check_sent"] is True for c in crops)
    assert all(c.get("mask_check_verdict") is None for c in crops)


def test_crop_decision_trace_sent_to_vlm_true_when_subset_response_omits_position(tmp_path):
    """Task 058 M8 -- THE regression test the reviewer required (their own exact
    probe): two genuine mask-check candidates render into ONE 2-position montage and are
    both SENT to `engine.generate`, but the (schema-valid) response names position 0
    ONLY. The omitted crop's decision-trace entry must read `sent_to_vlm=True` with
    `verdict=None` -- the M7 fix (`mask_check_verdict is not None`) wrongly read
    `sent_to_vlm=False` here, falsely claiming it was never sent."""
    tool_cache_dir = tmp_path / "tool_cache"
    tool_cache_dir.mkdir()
    meta_path = tool_cache_dir / "cell_crops_metadata_subsettest.json"
    meta_path.write_text("{}")

    crops = [_gate_crop_image(tmp_path, f"subset_trace_{i}.png", _dumbbell_mask(),
                              level=200, noise_std=25, seed=i) for i in range(2)]
    for c in crops:
        c["qc"] = {"neighbor_contamination": 0.0, "target_retention": 1.0,
                   "contamination_inset_px": 0}
        c["margin_mode"] = "mask_bbox"

    verifier = CropVerifier()
    verifier.triage_montage_grid = (1, 2)              # forces BOTH crops into ONE montage
    verifier.llm_engine = _MaskCheckPartialStubEngine(answer_indices=(0,), verdict="cell_present")
    context = {"metadata_path": str(meta_path)}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    verifier.adjudicate(metrics, locus, triggered, context)

    assert len(verifier.llm_engine.calls) == 1
    assert crops[0]["mask_check_verdict"] == "cell_present"
    assert crops[1].get("mask_check_verdict") is None          # omitted by the subset response

    trace_path = tool_cache_dir / "cell_crops_triage_trace_subsettest.json"
    trace = {e["cell_id"]: e for e in json.loads(trace_path.read_text())["verdicts"]}
    judged_entry = trace["subset_trace_0.png"]
    omitted_entry = trace["subset_trace_1.png"]
    assert judged_entry["mask_check"]["sent_to_vlm"] is True
    assert judged_entry["mask_check"]["verdict"] == "cell_present"
    assert omitted_entry["mask_check"]["sent_to_vlm"] is True   # sent...
    assert omitted_entry["mask_check"]["verdict"] is None         # ...but omitted -- NOT False
    assert omitted_entry["final"]["kept"] is True             # keep-biased: never discard on silence


# --------------------------------------------------------------------------- #
# Task 058 M3 ROUND 2 (coordinator-directed fix, 2026-07-16): `adjudicate` was
# calling TWO VLM reject channels (mask_check_recommend_crop AND vision_recommend_crop's
# own (B) REJECT), not one -- confirmed by grepping `adjudicate` directly (see the
# coding log). `enable_reject_channel` (new, default False, mirrors
# `enable_ab_channel`) makes the mask check the SOLE default VLM reject channel; the shared
# `tighten_for_confirmed_rejects` helper (extracted from `vision_recommend_crop`'s
# own tail) lets the mask check's OWN confirmed rejects tighten `det` the SAME way (B)
# REJECT's always did (a real gap-fix -- before this round, neither S1 nor the mask check's
# own rejects ever tightened `det` at all).
# --------------------------------------------------------------------------- #

def test_tighten_for_confirmed_rejects_noop_when_zero_rejected():
    det = VLRRResult("PASS", 0.0, {}, action="accept")
    out = tighten_for_confirmed_rejects(det, [], rejected=0, judged=3, tag="mask-check")
    assert out is det   # byte-identical no-op, not just an equal-valued copy


def test_tighten_for_confirmed_rejects_tightens_one_notch_and_names_the_tag():
    det = VLRRResult("PASS", 0.0, {}, action="accept")
    crops = [{"vision_reject": True, "vision_problem": "merged"}]
    out = tighten_for_confirmed_rejects(det, crops, rejected=1, judged=1, tag="mask-check")
    assert out.decision == "WARNING"           # PASS -> WARNING, one notch
    assert out.action == "accept"              # action untouched
    assert "[mask-check] 1/1" in out.notes
    assert "merged" in out.notes
    assert out.judge_problem == "merged"


def test_tighten_for_confirmed_rejects_never_exceeds_fail():
    det = VLRRResult("FAIL", 0.0, {}, action="accept")
    crops = [{"vision_reject": True, "vision_problem": "wrong_object"}]
    out = tighten_for_confirmed_rejects(det, crops, rejected=1, judged=1, tag="mask-check")
    assert out.decision == "FAIL"   # already at the ceiling -- stays there, never errors


def test_mask_check_reject_tightens_det_severity_through_adjudicate(tmp_path):
    """the mask check's OWN confirmed reject correctly tightens `det` (PASS -> WARNING) through
    the REAL `CropVerifier.adjudicate` path -- the gap-fix this round makes: before
    it, this exact scenario left `det` at PASS even though a crop was genuinely
    excluded from the kept population."""
    crop = _s1_crop_image(tmp_path, "tighten_dim.png", target_level=50)
    crop["cell_id"] = "dim_cell"
    verifier = CropVerifier()
    verifier.llm_engine = _MaskCheckStubEngine(verdict="target_region_empty", confidence="high")
    context = {}
    metrics = verifier.quantify([crop], context)
    locus, triggered, affected = verifier.localize(metrics, context)
    out = verifier.adjudicate(metrics, locus, triggered, context)
    assert out.decision == "WARNING"
    assert "[mask-check]" in out.notes
    assert "target_region_empty" in out.notes


def test_mask_check_is_sole_default_vlm_reject_channel_one_montage_dir(tmp_path):
    """Task 058 M3 round 2's own defining check: BOTH a mask-check candidate (dim
    target) AND a genuine (B)-REJECT candidate (cut_off, high border_touch) exist
    in the SAME batch -- by DEFAULT (`enable_reject_channel=False`), only the mask check's
    own montage fires: its own `montages_mask_check_<hash>/` directory is created with a
    real montage file; (B) REJECT's own `montages_<hash>/` directory (no prefix) is
    NEVER created at all -- proving ONE VLM reject channel by default, not two, by
    inspecting the actual montage directories a real run produces, not just an
    in-memory call count."""
    tool_cache_dir = tmp_path / "tool_cache"
    tool_cache_dir.mkdir()
    meta_path = tool_cache_dir / "cell_crops_metadata_soletest.json"
    meta_path.write_text("{}")

    dim_crop = _s1_crop_image(tmp_path, "dim_mask_check.png", target_level=50)
    dim_crop["cell_id"] = "dim_cell"
    dim_crop["margin_mode"] = "mask_bbox"

    cutoff_crop = _crop_image(tmp_path, "cutoff_reject.png", cell_id="cutoff_cell",
                              border_touch=0.6, mask_containment=1.0, centroid_offset=0.0)
    cutoff_crop["crop_path"] = cutoff_crop["path"]

    crops = [dim_crop, cutoff_crop]
    verifier = CropVerifier()
    assert verifier.enable_reject_channel is False   # the production default itself
    verifier.llm_engine = _MaskCheckStubEngine(verdict="target_region_empty", confidence="high")
    context = {"metadata_path": str(meta_path)}
    metrics = verifier.quantify(crops, context)
    locus, triggered, affected = verifier.localize(metrics, context)
    out = verifier.adjudicate(metrics, locus, triggered, context)

    mask_check_dir = _montages_dir_for(str(meta_path), prefix="montages_mask_check")
    reject_dir = _montages_dir_for(str(meta_path))   # (B) REJECT's own, no prefix
    assert os.path.exists(mask_check_dir)
    assert os.path.exists(os.path.join(mask_check_dir, "montage_00.png"))
    assert not os.path.exists(reject_dir)   # REJECT never even built a montage

    # the mask check's own confirmed reject still correctly tightens severity -- NOT lost by
    # gating (B) REJECT off (the coordinator's own explicit "do not loosen any gate").
    assert out.decision in ("WARNING", "FAIL")
    assert "[mask-check]" in out.notes

    # cutoff_crop, though a genuine (B)-REJECT candidate, is UNTOUCHED (never asked
    # -- the SCREENING is what is disabled, proving the gate actually bites).
    assert "vision_reject" not in cutoff_crop
    assert dim_crop.get("vision_reject") is True

    # the kept-crops manifest still correctly reflects the mask check's OWN rejection --
    # (B) REJECT being disabled must not also silently break the manifest write
    # (`vision_recommend_crop` is still called every round for its OTHER role).
    manifest_path = tool_cache_dir / "cell_crops_kept_manifest_soletest.json"
    kept = json.loads(manifest_path.read_text())["kept_crop_paths"]
    assert dim_crop["crop_path"] not in kept
    assert cutoff_crop["crop_path"] in kept
