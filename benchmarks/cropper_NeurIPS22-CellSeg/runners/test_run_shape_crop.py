"""Task 038: offline validation of `run_shape_crop.py` on hand-built SYNTHETIC image+mask
pairs (committed -- AGENTS.md sec 6 P5), with a MOCK/stub triage engine (no real API
call). Covers: the two-arm structure (MaskCrop control vs SHAPE agentic use different
geometry), `crop_quality_score` reporting per crop scored over each arm's ACTUAL OUTPUT
population (SHAPE = kept-only, excludes triage-rejected crops -- the real fix this
follow-up round made), the triage judge's tokens actually flow through
`shape.engine.usage` into the per-image record, all 6 (+more) efficiency fields populate
via the generalized `compute_metrics(metric_key="crop_quality_score")`, the full-process
JSONL trace, and persistent per-arm crop saving. The real gpt-4o cluster run is a
separate, manual step (see the module docstring).

Task 038 v13 (Honest reporting, human decision): also covers the MECHANISM metrics this
round added (`_qc_values_for_result`, `process_one_image`'s new record fields, and the
summary's new `"mechanism"` section + the `"quality"` section's own `"authoritative":
False` marker) -- see `run_shape_crop.py`'s own module docstring ("QUALITY vs.
MECHANISM") for the full rationale.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_shape_crop as rsc  # noqa: E402
import segmentation_metrics as sm  # noqa: E402

from shape.agent.verification.crop_triage import (  # noqa: E402
    CropTriageVerdict,
    MontagePositionVerdict,
    MontageVerdict,
)
from shape.agent.verification.step_verifier import StepVerifier  # noqa: E402
from shape.engine import usage as token_usage  # noqa: E402
from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool  # noqa: E402
from shape.utils.schemas import VLRRResult  # noqa: E402


def _montage_verdict_for(prompt: str, verdict: str, confidence: str, problem: str) -> MontageVerdict:
    """Task 038 montage round: fabricates a `MontageVerdict` applying the SAME canned
    verdict/confidence/problem to every labeled position the real `_build_montage_prompt`
    printed (read back from its own "Position N:" list) -- lets the stub engines below
    keep answering the now-BATCHED reject channel meaningfully (mirrors
    `test_crop_triage.py`'s own `_fabricate_montage_verdict`)."""
    indices = [int(i) for i in re.findall(r"Position (\d+):", prompt)]
    effective_problem = problem if verdict == "reject" else "none"
    return MontageVerdict(verdicts=[
        MontagePositionVerdict(index=i, crop_id="", problem=effective_problem,
                               severity="severe" if effective_problem != "none" else "none",
                               confidence=confidence)
        for i in indices
    ])


class _TokenStubEngine:
    """A minimal stub vision engine: always KEEPs, and records a fixed token cost via
    `shape.engine.usage` -- mirroring how a REAL engine's `generate()` does internally
    (`shape/engine/{openai,anthropic,litellm}.py`) -- so a test can verify tokens flow
    through end to end without a real API key.

    Task 038 montage round: also answers a `MontageVerdict` ask (the now-batched reject
    channel) via `_montage_verdict_for`, `self.calls` still counts EVERY `generate()`
    call regardless of shape."""

    def __init__(self, tokens_per_call: int = 37):
        self._tokens_per_call = tokens_per_call
        self.calls = 0

    def generate(self, content, response_format=None, **kwargs):
        self.calls += 1
        token_usage.record({"total_tokens": self._tokens_per_call,
                            "prompt_tokens": self._tokens_per_call - 5,
                            "completion_tokens": 5})
        if response_format is MontageVerdict:
            return {"content": _montage_verdict_for(content[0], "keep", "high", "none")}
        rec = CropTriageVerdict(verdict="keep", confidence="high", problem="none",
                                reasoning="stub")
        return {"content": rec}


class _NoVerdictEngine(_TokenStubEngine):
    """The vision engine of a run that judges nothing: every vision call answered with no verdict (keep) and no
    tokens (a run without an engine raises: the vision calls have no deterministic substitute)."""

    def __init__(self):
        super().__init__(tokens_per_call=0)


class _AlwaysRejectEngine:
    """Stub that always rejects with high confidence -- proves the kept-only scoring fix
    (the real bug it found): SHAPE's headline `crop_quality_score` population must
    EXCLUDE triage-rejected crops, while MaskCrop (no triage at all) is completely
    unaffected. Task 038 v9 restructure: a crop is only ever SENT to the judge when the
    DATA screens it as a candidate (`crop_triage._vlm_candidate_reasons`,
    `n_complete_cells>=2`, `not_truncated<0.8`, or `foreground_fraction<0.2`) -- these
    synthetic fixture crops are well-formed (fully contained, real cell mask, single
    cell), so a bare "always reject" stub would never even be asked. The tests below
    monkeypatch `_vlm_candidate_reasons` to force every crop into candidacy, isolating
    the KEPT-ONLY-SCORING/PERSISTENCE concern this fixture is actually for from the
    (separately, fully tested in `test_crop_triage.py`) data-screen mechanism itself.

    Task 038 montage round: also answers a `MontageVerdict` ask (the now-batched reject
    channel) via `_montage_verdict_for`, rejecting every labeled position it is asked
    about."""

    def generate(self, content, response_format=None, **kwargs):
        if response_format is MontageVerdict:
            return {"content": _montage_verdict_for(content[0], "reject", "high", "cut_off")}
        rec = CropTriageVerdict(verdict="reject", confidence="high", problem="cut_off",
                                reasoning="stub-always-reject")
        return {"content": rec}


def _write_pair(data_dir: Path, name: str, mask: np.ndarray, seed: int = 0) -> tuple[Path, Path]:
    """A STRUCTURED image (mask regions bright, background dark + mild noise -- the
    `test_quant.py::_disk_crop` convention), not pure random noise: `sharpness` needs a
    real edge to measure, so a purely-random crop makes it noise that can mask the
    geometry signal a test cares about (a small crop-window difference then shifts
    `sharpness` by more than the `centered` axis it's supposed to isolate)."""
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    image = np.full(mask.shape, 40.0)
    image[mask > 0] = 200.0
    image = np.clip(image + rng.normal(0, 3, size=mask.shape), 0, 255).astype(np.uint8)
    image_path = images_dir / f"{name}.tif"
    label_path = labels_dir / f"{name}_label.tiff"
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(label_path, mask.astype(np.uint16))
    return image_path, label_path


def _asymmetric_cell_mask() -> np.ndarray:
    """An L-shaped target (bbox midpoint far from its true centroid -- centroid-recenter
    should measurably help here) plus a nearby neighbour, comfortably inside the image."""
    mask = np.zeros((200, 200), dtype=np.int64)
    mask[40:100, 40:70] = 1        # target: tall arm
    mask[70:100, 40:100] = 1       # target: short arm (bottom-right heavy)
    mask[50:65, 108:123] = 2       # neighbour cell
    return mask


def _severely_asymmetric_cell_mask() -> np.ndarray:
    """Task 038 follow-up (VLM-hallucination-fix round): a MORE asymmetric L-shape than
    `_asymmetric_cell_mask` -- at the naive (10% fraction, no-recenter) starting margin
    this target's own `centroid_offset` (0.2658, MEASURED) genuinely exceeds the NEW
    off-center tighten-corroboration floor (`CropVerifier._OFF_CENTER_MAX=0.20`), unlike
    `_asymmetric_cell_mask`'s own target (0.1974, just BELOW it -- would get its tighten
    VETOED under the new hard mask-corroboration design). Used ONLY by
    `test_shape_raises_centered_axis_on_asymmetric_cell` -- every other test in this file
    keeps `_asymmetric_cell_mask` unchanged."""
    mask = np.zeros((200, 200), dtype=np.int64)
    mask[40:110, 40:60] = 1        # target: tall, narrow arm (70x20)
    mask[90:110, 40:90] = 1        # target: short, wide arm (20x50, bottom-right heavy)
    mask[50:65, 108:123] = 2       # neighbour cell (unchanged)
    return mask


def test_process_one_image_without_verdicts_is_structurally_complete(tmp_path):
    """An engine that judges nothing (every crop kept, 0 tokens): every field present, both arms produce crops,
    nothing is excluded (SHAPE's kept population == its full population). (Without an engine the vision calls
    raise: they have no deterministic substitute.)

    Task 038 restructure: SHAPE's arm starts from the SAME naive geometry as MaskCrop's
    own (margin_mode="fraction", recenter=False) -- the per-cell geometry override is now
    100% DATA-DRIVEN and fires regardless of whether an engine is wired (`CropVerifier.
    adjudicate` reads `centroid_offset`/`n_complete_cells` straight off each crop's own
    `qc` block). It does NOT fire here because `_asymmetric_cell_mask`'s own naive
    `centroid_offset` (0.1974, MEASURED) sits just BELOW the off-center trigger
    (`CropVerifier._OFF_CENTER_MAX=0.20`), and there is no doublet (`n_complete_cells<2`)
    -- so the loop terminates in exactly ONE trial, same as before, just for a DATA
    reason now (a genuinely off-center/doublet fixture would retune even with no engine
    at all -- see `test_shape_raises_centered_axis_on_asymmetric_cell`)."""
    image_path, mask_path = _write_pair(tmp_path / "data", "img1", _asymmetric_cell_mask())
    tool = Single_Cell_Cropper_Tool()

    record, trace = rsc.process_one_image(
        tool, image_path, mask_path, tmp_path / "out", rsc.DEFAULT_MIN_AREA, rsc.DEFAULT_MARGIN,
        rsc.DEFAULT_MASKCROP_MARGIN_FRAC, _NoVerdictEngine())

    assert record["tokens"] == 0
    assert record["n_crops_maskcrop"] == 2
    assert record["n_crops_shape_final_total"] == 2
    assert record["n_crops_shape_final_kept"] == 2   # no engine -> nothing rejected
    assert len(record["call_crop_quality_score"]) >= 1
    assert record["baseline_crop_quality_score"] is not None
    assert all(0.0 <= s <= 1.0 for s in record["maskcrop_scores"])
    assert all(0.0 <= s <= 1.0 for s in record["shape_final_scores"])
    assert record["n_vlrr_trials"] == 1   # no engine -> no geometry retune can fire
    assert trace["image_id"] == "img1"
    assert len(trace["vlrr_trials"]) == 1
    assert not any(v.get("reject") for v in trace["crop_verdicts"])   # judged, nothing rejected


def test_two_arms_share_the_same_fair_start_geometry(tmp_path):
    """Task 038 follow-up (part B redesign, FAIR START): SHAPE's arm no longer starts
    from a different (tight+recenter) config than MaskCrop's own. Both arms' FIRST
    execute() must read the IDENTICAL naive tool-default (margin_mode="fraction",
    recenter=False, same `margin`/`min_area`) -- verified from each arm's own written
    metadata, so a future edit that reintroduces a starting-config asymmetry shows up
    here. SHAPE's own per-crop geometry override only happens AFTER this shared start,
    driven entirely by data (`CropVerifier.adjudicate`, no engine needed) -- this
    fixture's own naive `centroid_offset` (0.1974) sits below the off-center trigger and
    there is no doublet, so `cell_overrides` stays empty and the persisted crops are
    still literally the naive-start geometry, same as MaskCrop's)."""
    image_path, mask_path = _write_pair(tmp_path / "data", "img1", _asymmetric_cell_mask())
    tool = Single_Cell_Cropper_Tool()

    maskcrop_result = rsc.run_maskcrop_arm(
        tool, image_path, mask_path, tmp_path / "maskcrop", rsc.DEFAULT_MIN_AREA,
        rsc.DEFAULT_MARGIN, rsc.DEFAULT_MASKCROP_MARGIN_FRAC)
    maskcrop_meta = json.loads(Path(maskcrop_result["cell_crops_metadata_path"]).read_text())
    assert maskcrop_meta["parameters"]["margin_mode"] == "fraction"
    assert maskcrop_meta["parameters"]["recenter"] is False

    resolution = rsc.run_shape_arm(
        tool, image_path, mask_path, tmp_path / "shape", rsc.DEFAULT_MIN_AREA,
        rsc.DEFAULT_MARGIN, rsc.DEFAULT_MASKCROP_MARGIN_FRAC, _NoVerdictEngine())
    shape_meta = json.loads(Path(resolution.result["cell_crops_metadata_path"]).read_text())
    assert shape_meta["parameters"]["margin_mode"] == "fraction"
    assert shape_meta["parameters"]["recenter"] is False
    assert shape_meta["parameters"]["cell_overrides"] == {}
    # same starting config on both arms -- literally the same `margin`/`margin_mode`
    assert shape_meta["parameters"]["margin"] == maskcrop_meta["parameters"]["margin"]


def test_shape_raises_centered_axis_on_asymmetric_cell(tmp_path):
    """The headline geometry claim, isolated to the axis it actually drives: on a
    SEVERELY asymmetric cell (bbox midpoint far from the true centroid), SHAPE's
    `centered` component ends up higher than MaskCrop's naive one -- centroid-recenter
    measurably helps (mirrors the earlier asymmetric-cell proof, now
    through the crop-stage VLRR loop rather than a bare execute() call). Checked on
    `centered` directly (not the blended `crop_quality_score`, which also weighs
    `sharpness` -- crop-size-sensitive pixel noise on a tiny neighbour crop can swamp the
    `centered` axis's small overall weight (0.5/4.0), see the code comment below).

    Task 038 restructure: geometry is now 100% DATA-DRIVEN and fully decoupled from
    vision (`CropVerifier.adjudicate` reads `centroid_offset`/`n_complete_cells` straight
    off each crop's own `qc` block) -- so this test drives the loop with `engine=None`,
    no judge involved at all. Uses `_severely_asymmetric_cell_mask`, NOT
    `_asymmetric_cell_mask` -- the ORIGINAL fixture's own naive `centroid_offset`
    (0.1974, MEASURED) sits just BELOW the off-center geometry trigger
    (`CropVerifier._OFF_CENTER_MAX=0.20`), so its tighten would never fire; this
    fixture's own naive offset (0.2658, MEASURED) genuinely exceeds it, so the
    data-driven `recenter=True` override fires for the target cell (area 2000) but not
    the already-centered neighbour (area 225, offset well within bound) -- an honest,
    per-cell demonstration of the mechanism, not a blanket flip. `len(resolution.trials)
    > 1` proves the retune genuinely FIRED (not a one-shot default)."""
    image_path, mask_path = _write_pair(tmp_path / "data", "img1", _severely_asymmetric_cell_mask())
    tool = Single_Cell_Cropper_Tool()

    maskcrop_result = rsc.run_maskcrop_arm(
        tool, image_path, mask_path, tmp_path / "maskcrop", rsc.DEFAULT_MIN_AREA,
        rsc.DEFAULT_MARGIN, rsc.DEFAULT_MASKCROP_MARGIN_FRAC)
    resolution = rsc.run_shape_arm(
        tool, image_path, mask_path, tmp_path / "shape", rsc.DEFAULT_MIN_AREA,
        rsc.DEFAULT_MARGIN, rsc.DEFAULT_MASKCROP_MARGIN_FRAC, _NoVerdictEngine())

    assert len(resolution.trials) > 1   # the retune genuinely fired, not a one-shot default

    # match crops by AREA (2000 = the severely asymmetric L-shaped TARGET, the SAME cell
    # in both arms -- NOT by score rank: the small, already-centered neighbour (area=225)
    # can outscore the target in either arm on raw crop_quality_score, since sharpness
    # (weight 0.5) is sensitive to the exact crop size/window and can swamp the small
    # `centered`-axis (weight 0.5/4.0) gain on a tiny, easy neighbour crop -- picking by
    # `max()` would silently compare two DIFFERENT cells across arms.
    def _target_centered(result):
        meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
        target = max(meta["cell_metadata"], key=lambda m: m["area"])
        assert target["area"] == 2000   # sanity: this IS the severely asymmetric L-shape
        return 1.0 - target["qc"]["centroid_offset"]

    maskcrop_centered = _target_centered(maskcrop_result)
    shape_centered = _target_centered(resolution.result)
    assert shape_centered > maskcrop_centered
    # MEASURED values for `_severely_asymmetric_cell_mask` at the task 038 v9 naive
    # starting margin (20%, was 10%) -- naive `centered`=0.7638 (MaskCrop, and SHAPE's
    # own naive starting point); SHAPE's final `centered`=1.0 (essentially perfect),
    # reached via ONLY the off-center trigger firing (`recenter=True`, no `margin_mode`
    # override -- the target's own `n_complete_cells<2`, the neighbour sits outside this
    # crop box, so no doublet). The WIDER 20% naive box gives even MORE slack to sit
    # exactly at the true centroid while still containing the asymmetric bbox than the
    # 10% baseline did (0.9756) -- a small "tight"/"mask_bbox" box is more constrained by
    # containment and leaves a nontrivial residual offset (CARD.md SS1's own documented
    # tradeoff) -- decoupling the two triggers avoids paying that cost when only
    # recentering, not tightening, was actually needed.
    assert shape_centered == pytest.approx(1.0, abs=5e-3)
    assert maskcrop_centered == pytest.approx(0.7638, abs=5e-3)


def test_triage_engine_tokens_flow_into_the_record(tmp_path, monkeypatch):
    """A stub triage engine that records real token usage -- the per-image record's
    `tokens` field must reflect the triage judge's LLM call(s), proving the wiring from
    `CropVerifier.adjudicate` through to `token_usage` and into this benchmark's own
    per-image accounting. Geometry itself (per-crop `cell_overrides`) is PURELY
    DATA-DRIVEN and never calls the judge at all (task 038 restructure), so `engine.calls`
    here is exactly the reject-triage judge's own per-image call count. Task 038
    restructure: a crop is only ever SENT to the judge when the DATA screens it as a
    candidate -- this fixture's real crops are well-formed (fully contained, real cell
    mask), so `_vlm_candidate_reasons` is monkeypatched to force candidacy, isolating
    the TOKEN-ACCOUNTING wiring this test is actually for from the (separately, fully
    tested in `test_crop_triage.py`) data-screen mechanism itself.

    Task 058 M3 round 2 (coordinator-directed fix): `CropVerifier`'s (B) REJECT
    channel (cut_off/empty_debris) is now off by default (`enable_reject_channel`,
    mirrors `enable_ab_channel` -- the mask check is the sole default VLM reject channel).
    This test exercises (B) REJECT directly (its own `cut_off` candidacy, not the mask
    check's), so it opts back in via a class-level monkeypatch (no `run_shape_crop.py`
    source change needed -- that file stays untouched, only this test file's own
    setup changes)."""
    monkeypatch.setattr("shape.agent.verification.crop_triage._vlm_candidate_reasons",
                        lambda crop: ["cut_off"])
    monkeypatch.setattr(rsc.CropVerifier, "enable_reject_channel", True)
    image_path, mask_path = _write_pair(tmp_path / "data", "img1", _asymmetric_cell_mask())
    tool = Single_Cell_Cropper_Tool()
    engine = _TokenStubEngine(tokens_per_call=37)

    record, trace = rsc.process_one_image(
        tool, image_path, mask_path, tmp_path / "out", rsc.DEFAULT_MIN_AREA, rsc.DEFAULT_MARGIN,
        rsc.DEFAULT_MASKCROP_MARGIN_FRAC, engine)

    assert engine.calls >= 1
    assert record["tokens"] == 37 * engine.calls
    assert record["tokens"] > 0
    # a keep-everything engine still produced a full triage trace (task 038 follow-up)
    assert len(trace["crop_verdicts"]) == record["n_crops_shape_final_total"]
    assert all(v["verdict"] == "keep" and not v["reject"] for v in trace["crop_verdicts"])


def test_shape_kept_only_scoring_excludes_rejected_crops_maskcrop_unaffected(tmp_path, monkeypatch):
    """The real bug this follow-up round fixed: `_crop_quality_scores_for_result` used to
    score EVERY crop in a result regardless of triage, so SHAPE's own triage had ZERO
    effect on the headline quality metric. With an engine that rejects every crop, SHAPE's
    KEPT population must be EMPTY (n_crops_shape_final_kept == 0, shape_final_scores == [])
    while the crop COUNT (n_crops_shape_final_total, descriptive provenance only) still
    reports 2, and MaskCrop's own scores (no triage stage exists for that arm at all) are
    completely unaffected by the engine.

    Task 038 restructure: a crop is only ever SENT to the judge when the DATA screens it
    as a candidate (`crop_triage._vlm_candidate_reasons`) -- monkeypatched here to
    force every crop into candidacy, since this fixture's real crops are well-formed
    (fully contained, real cell mask) and would otherwise never even be asked -- an
    orthogonal concern this test does not intend to exercise (the screen itself has its
    own dedicated proofs in `test_crop_triage.py`).

    Task 058 M3 round 2: (B) REJECT is off by default -- opts back in explicitly
    (see `test_triage_engine_tokens_flow_into_the_record`'s own updated docstring)."""
    monkeypatch.setattr("shape.agent.verification.crop_triage._vlm_candidate_reasons",
                        lambda crop: ["cut_off"])
    monkeypatch.setattr(rsc.CropVerifier, "enable_reject_channel", True)
    image_path, mask_path = _write_pair(tmp_path / "data", "img1", _asymmetric_cell_mask())
    tool = Single_Cell_Cropper_Tool()
    engine = _AlwaysRejectEngine()

    record, trace = rsc.process_one_image(
        tool, image_path, mask_path, tmp_path / "out", rsc.DEFAULT_MIN_AREA, rsc.DEFAULT_MARGIN,
        rsc.DEFAULT_MASKCROP_MARGIN_FRAC, engine)

    assert record["n_crops_shape_final_total"] == 2
    assert record["n_crops_shape_final_kept"] == 0
    assert record["shape_final_scores"] == []
    assert record["n_crops_maskcrop"] == 2
    assert len(record["maskcrop_scores"]) == 2
    # the triage trace records the rejection with full per-crop detail
    assert len(trace["crop_verdicts"]) >= 1
    assert all(v["reject"] for v in trace["crop_verdicts"])
    assert all(v["problem"] == "cut_off" for v in trace["crop_verdicts"])


def test_crops_persist_under_out_and_result_manifest_written(tmp_path):
    """SAVE + ORGANIZE (task 038 follow-up, Item 2): crop PNGs for BOTH arms survive
    under the given `crops_dir` in `tool_cache/` (no longer a `tempfile.TemporaryDirectory`
    that gets wiped after every image), organized by cell id (the tool's own
    `<image>_cell_<idx>_crop.png` naming); BOTH arms also get a `result_manifest.json` a
    human can read to see which crops were kept vs rejected. No engine -> nothing
    rejected on either arm, so everything lands in kept/ and rejected/ is empty."""
    image_path, mask_path = _write_pair(tmp_path / "data", "img1", _asymmetric_cell_mask())
    tool = Single_Cell_Cropper_Tool()
    crops_dir = tmp_path / "out" / "crops" / "img1"

    rsc.process_one_image(
        tool, image_path, mask_path, crops_dir, rsc.DEFAULT_MIN_AREA, rsc.DEFAULT_MARGIN,
        rsc.DEFAULT_MASKCROP_MARGIN_FRAC, _NoVerdictEngine())

    maskcrop_crops = list((crops_dir / "maskcrop" / "tool_cache").glob("*_crop.png"))
    shape_crops = list((crops_dir / "shape" / "tool_cache").glob("*_crop.png"))
    assert len(maskcrop_crops) == 2
    assert len(shape_crops) >= 2

    for arm_dir in (crops_dir / "maskcrop", crops_dir / "shape"):
        manifest_path = arm_dir / "result_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["kept_crop_paths"]) == 2
        assert manifest["rejected_crop_paths"] == []   # no engine -> nothing rejected
        assert len(list((arm_dir / "kept").glob("*_crop.png"))) == 2
        assert list((arm_dir / "rejected").glob("*_crop.png")) == []


def test_shape_rejected_crops_land_in_rejected_dir_maskcrop_unaffected(tmp_path, monkeypatch):
    """The PHYSICAL separation itself (task 038 follow-up, Item 2's real point -- "so a
    human can eyeball them directly, esp. the ~165 blur rejects"): with an engine that
    rejects every crop, SHAPE's `rejected/` dir gets ALL its crops (byte-identical files,
    same basenames as `tool_cache/`), `kept/` stays empty; MaskCrop (no triage at all) is
    completely unaffected -- ALL of its crops still land in `kept/`.

    Task 038 restructure: the data screen (`crop_triage._vlm_candidate_reasons`) is
    monkeypatched to force candidacy -- see the sibling kept-only-scoring test above for
    why (orthogonal concern, its own dedicated proofs elsewhere).

    Task 058 M3 round 2: (B) REJECT is off by default -- opts back in explicitly."""
    monkeypatch.setattr("shape.agent.verification.crop_triage._vlm_candidate_reasons",
                        lambda crop: ["cut_off"])
    monkeypatch.setattr(rsc.CropVerifier, "enable_reject_channel", True)
    image_path, mask_path = _write_pair(tmp_path / "data", "img1", _asymmetric_cell_mask())
    tool = Single_Cell_Cropper_Tool()
    crops_dir = tmp_path / "out" / "crops" / "img1"
    engine = _AlwaysRejectEngine()

    rsc.process_one_image(
        tool, image_path, mask_path, crops_dir, rsc.DEFAULT_MIN_AREA, rsc.DEFAULT_MARGIN,
        rsc.DEFAULT_MASKCROP_MARGIN_FRAC, engine)

    shape_dir = crops_dir / "shape"
    assert len(list((shape_dir / "rejected").glob("*_crop.png"))) == 2
    assert list((shape_dir / "kept").glob("*_crop.png")) == []
    shape_manifest = json.loads((shape_dir / "result_manifest.json").read_text())
    assert len(shape_manifest["rejected_crop_paths"]) == 2
    assert shape_manifest["kept_crop_paths"] == []

    maskcrop_dir = crops_dir / "maskcrop"
    assert len(list((maskcrop_dir / "kept").glob("*_crop.png"))) == 2
    assert list((maskcrop_dir / "rejected").glob("*_crop.png")) == []
    maskcrop_manifest = json.loads((maskcrop_dir / "result_manifest.json").read_text())
    assert len(maskcrop_manifest["kept_crop_paths"]) == 2
    assert maskcrop_manifest["rejected_crop_paths"] == []


def test_compute_metrics_reports_the_six_headline_efficiency_fields():
    """The dispatch's explicit 6: tool_calls_per_image / positive_tool_call_rate /
    wasted_tool_call_rate / crop_quality_score_gain_per_tool_call / time_per_image_s /
    token_cost_per_image -- all populate (not None) on a small hand-built record set
    with a real retune (so the rate metrics have a denominator)."""
    records = [
        {"image_id": "a", "baseline_crop_quality_score": 0.60,
         "call_crop_quality_score": [0.75, 0.80], "tokens": 40, "seconds": 3.0},
        {"image_id": "b", "baseline_crop_quality_score": 0.55,
         "call_crop_quality_score": [0.70], "tokens": 0, "seconds": 1.5},
    ]
    m = sm.compute_metrics(records, metric_key="crop_quality_score")
    for key in ("tool_calls_per_image", "positive_tool_call_rate", "wasted_tool_call_rate",
                "crop_quality_score_gain_per_tool_call", "time_per_image_s",
                "token_cost_per_image"):
        assert m[key] is not None, f"{key} unexpectedly None"
    assert m["tool_calls_per_image"] == pytest.approx(1.5)   # (2+1)/2
    assert m["token_cost_per_image"] == pytest.approx(20.0)  # (40+0)/2


def test_cli_end_to_end_writes_summary_trace_and_persisted_crops(tmp_path, monkeypatch):
    """Full CLI smoke test (no engine -- no API key needed): the summary JSON carries
    both the QUALITY section (per-arm mean + delta, MaskCrop/SHAPE naming) and the
    EFFICIENCY section (all 6 headline fields present as keys, even if some are None with
    0 retunes); the full-process JSONL trace and persisted per-arm crops (task 038
    follow-up) both exist on disk after the run."""
    monkeypatch.setattr(rsc, "build_vision_engine", lambda *a, **k: _NoVerdictEngine())
    data_dir = tmp_path / "data"
    _write_pair(data_dir, "img1", _asymmetric_cell_mask(), seed=1)

    out_dir = tmp_path / "out"
    argv = ["run_shape_crop.py", "--data-dir", str(data_dir), "--out", str(out_dir),
            "--sample-cap", "10"]
    monkeypatch.setattr(sys, "argv", argv)
    rsc.main()

    summary_path = out_dir / "run_shape_crop_summary.json"
    csv_path = out_dir / "per_image_crop_agentic.csv"
    trace_path = out_dir / "trace.jsonl"
    assert summary_path.exists() and csv_path.exists() and trace_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["provenance"]["n_images_processed"] == 1
    assert "quality" in summary and "efficiency" in summary
    assert summary["quality"]["maskcrop_crop_quality_score_mean"] is not None
    assert summary["quality"]["shape_crop_quality_score_mean"] is not None
    assert summary["quality"]["quality_delta"] is not None
    for key in ("tool_calls_per_image", "positive_tool_call_rate", "wasted_tool_call_rate",
                "crop_quality_score_gain_per_tool_call", "time_per_image_s",
                "token_cost_per_image"):
        assert key in summary["efficiency"]

    trace_lines = trace_path.read_text().strip().splitlines()
    assert len(trace_lines) == 1
    trace = json.loads(trace_lines[0])
    assert trace["image_id"] == "img1"
    assert "vlrr_trials" in trace and len(trace["vlrr_trials"]) >= 1
    assert "crop_verdicts" in trace

    crops_dir = out_dir / "crops" / "img1"
    assert list((crops_dir / "maskcrop" / "tool_cache").glob("*_crop.png"))
    assert list((crops_dir / "shape" / "tool_cache").glob("*_crop.png"))
    assert (crops_dir / "shape" / "result_manifest.json").exists()
    assert list((crops_dir / "maskcrop" / "kept").glob("*_crop.png"))
    assert list((crops_dir / "shape" / "kept").glob("*_crop.png"))


# ---------------------------------------------------------------------------
# Task 038 v13 (Honest reporting, human decision -- a coordinator follow-up on top of
# the v13 commit): the runner no longer headlines a `crop_quality_score` delta as
# evidence of "better crops" -- it reports MECHANISM metrics instead
# (`neighbor_contamination` reduction, `target_retention` distribution, plus the
# existing cost fields), and marks the still-computed `crop_quality_score` section
# `"authoritative": False`. See `run_shape_crop.py`'s own module docstring ("QUALITY
# vs. MECHANISM") for the full rationale -- this is a SURGICAL reporting change only;
# the core VLRR/verifier decision logic is unchanged (frozen for QC) and these tests
# exercise ONLY the new reporting wiring.
# ---------------------------------------------------------------------------

def test_qc_values_for_result_reads_the_given_key(tmp_path):
    """`_qc_values_for_result` reads an arbitrary `qc[key]` per crop -- proven here on
    `neighbor_contamination` and `target_retention`, both real fields
    `tool.py::_compute_crop_qc` always populates, no pixel I/O needed (unlike the
    quality-score reader, this never opens the crop's own PNG)."""
    image_path, mask_path = _write_pair(tmp_path / "data", "img1", _asymmetric_cell_mask())
    tool = Single_Cell_Cropper_Tool()
    result = rsc.run_maskcrop_arm(
        tool, image_path, mask_path, tmp_path / "maskcrop", rsc.DEFAULT_MIN_AREA,
        rsc.DEFAULT_MARGIN, rsc.DEFAULT_MASKCROP_MARGIN_FRAC)

    contamination = rsc._qc_values_for_result(result, "neighbor_contamination")
    retention = rsc._qc_values_for_result(result, "target_retention")
    assert len(contamination) == 2
    assert len(retention) == 2
    assert all(0.0 <= c <= 1.0 for c in contamination)
    assert all(0.0 <= r <= 1.0 for r in retention)


def test_qc_values_for_result_kept_only_respects_the_manifest(tmp_path, monkeypatch):
    """`kept_only=True` excludes triage-rejected crops, the SAME population-selection
    logic `_crop_quality_scores_for_result` already uses -- proven here with the SAME
    always-reject engine fixture the existing kept-only-scoring test uses.

    Task 058 M3 round 2: (B) REJECT is off by default -- opts back in explicitly."""
    monkeypatch.setattr("shape.agent.verification.crop_triage._vlm_candidate_reasons",
                        lambda crop: ["cut_off"])
    monkeypatch.setattr(rsc.CropVerifier, "enable_reject_channel", True)
    image_path, mask_path = _write_pair(tmp_path / "data", "img1", _asymmetric_cell_mask())
    tool = Single_Cell_Cropper_Tool()
    resolution = rsc.run_shape_arm(
        tool, image_path, mask_path, tmp_path / "shape", rsc.DEFAULT_MIN_AREA,
        rsc.DEFAULT_MARGIN, rsc.DEFAULT_MASKCROP_MARGIN_FRAC, _AlwaysRejectEngine())

    all_contamination = rsc._qc_values_for_result(resolution.result, "neighbor_contamination",
                                                   kept_only=False)
    kept_contamination = rsc._qc_values_for_result(resolution.result, "neighbor_contamination",
                                                    kept_only=True)
    assert len(all_contamination) == 2
    assert kept_contamination == []   # everything rejected -- nothing kept


def test_qc_values_for_result_empty_on_missing_metadata():
    assert rsc._qc_values_for_result({}, "neighbor_contamination") == []
    assert rsc._qc_values_for_result({"cell_crops_metadata_path": "/no/such/file.json"},
                                     "target_retention") == []


class _ScriptedVerifier:
    """A minimal REAL `BaseVerifier`-shaped stub (mirrors `step_verifier.py`'s own test
    file, `test_step_verifier_vrr.py::ScriptedVerifier` -- reproduced locally, not
    cross-imported from another test module, to keep this file self-contained): returns
    a pre-built `VLRRResult` sequence, ignoring its own `(vinput, ctx)` inputs, so a
    test can drive the REAL `StepVerifier.resolve_stage` loop deterministically."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def verify(self, vinput, ctx):
        vres = self.script[self.calls]
        self.calls += 1
        return vres


def test_converged_result_reflects_the_real_disk_state_on_a_max_iters_reached_run(tmp_path):
    """Task 038 QC N1 (an independent QC of the geometry release candidate,
    PASS_WITH_NOTES -- falsified this function's own earlier claim that `trials[-1]`
    matches disk on EVERY exit path): drives the REAL `StepVerifier.resolve_stage`
    (not a hand-built `SimpleNamespace` standing in for a `StageResolution` -- the
    PRIOR version of this test was exactly that, a TAUTOLOGY QC correctly flagged as
    unable to ever catch this, since it never exercised `step_verifier.py`'s own real
    loop mechanics at all) to `max_iters_reached`, with a REAL `tool_runner` that
    writes a genuinely uniquely-named metadata file to disk on EVERY call (mirroring
    `tool.py`'s own `uuid4`-suffixed naming -- confirmed by reading `tool.py` directly,
    not assumed).

    QC found then that the final loop iteration's call was never appended to `history`, so
    `trials[-1]` was the second-to-last call. StepVerifier now records every call as a trial
    and counts the caller's initial call in `max_iters`, so that gap is closed there. The test
    keeps guarding the delivered state: `trials[-1]` and `_converged_result` (which reads the
    newest-mtime metadata file in the same `tool_cache/`, `_newest_metadata_file`) are both the
    newest file genuinely on disk."""
    import time

    tool_cache_dir = tmp_path / "tool_cache"
    tool_cache_dir.mkdir()
    call_count = [0]

    def tool_runner(params):
        call_count[0] += 1
        meta_path = tool_cache_dir / f"cell_crops_metadata_call{call_count[0]}.json"
        meta_path.write_text(json.dumps({"cell_metadata": [], "call_index": call_count[0]}))
        time.sleep(0.02)   # guards against a coarse filesystem mtime resolution tying
                            # two rapid-succession writes -- production `tool.execute()`
                            # calls are never this close together (real disk I/O for
                            # every crop image + QC computation), so this only matters
                            # for this test's own synthetic speed, not production.
        return {"cell_crops_metadata_path": str(meta_path)}

    # every retune genuinely changes `params` (diameter grows each call) -- avoids
    # oscillation_stop; no `expected_effect` -- avoids the reconcile-fail stop; no
    # `judge_severity` -- avoids the plateau stop. So the loop runs the FULL
    # `max_iters=3`: 3 tool calls in all (1 initial call, made by the caller BEFORE
    # `resolve_stage` per its own real contract and counted in `max_iters`, + 2 more
    # inside the loop -- exactly the shape a real `run_shape_arm` call has).
    vres = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here",
                      param_delta={"diameter": "up"})
    verifier = _ScriptedVerifier([vres, vres, vres])
    sv = StepVerifier(max_iters=3)
    initial_params = {"diameter": 30.0}
    initial_result = tool_runner(initial_params)

    resolution = sv.resolve_stage(
        verifier=verifier, adapter=lambda result, ctx: (result, {}), tool_runner=tool_runner,
        initial_params=initial_params, initial_result=initial_result)

    assert resolution.reason == "max_iters_reached"
    assert call_count[0] == 3   # 1 initial + 2 loop iterations, one real call each

    on_disk_files = sorted(tool_cache_dir.glob("cell_crops_metadata_*.json"))
    assert len(on_disk_files) == 3   # every call's own metadata file survives -- none overwritten
    newest_on_disk = max(on_disk_files, key=lambda p: p.stat().st_mtime)

    assert len(resolution.trials) == 3   # every call is a trial
    assert resolution.trials[-1][2]["cell_crops_metadata_path"] == str(newest_on_disk)

    converged = rsc._converged_result(resolution)
    assert converged["cell_crops_metadata_path"] == str(newest_on_disk)   # the delivered, on-disk state


def test_converged_result_falls_back_to_resolution_result_when_trials_empty():
    """Defensive fallback: `resolve_stage` always appends at least one trial before
    returning on every real code path, so `trials` should never actually be empty --
    but `_converged_result` degrades gracefully to `resolution.result` rather than
    raising an IndexError if it somehow is."""
    from types import SimpleNamespace
    fallback_result = {"cell_crops_metadata_path": "/fake/only_result.json"}
    resolution = SimpleNamespace(result=fallback_result, trials=[])
    assert rsc._converged_result(resolution) == fallback_result


def test_process_one_image_reports_mechanism_metrics(tmp_path):
    """`process_one_image`'s own `record` carries the new task 038 v13 mechanism
    fields, correctly populated (not just present) -- `maskcrop_contamination`/
    `shape_contamination` (one value per crop, both arms have 2 crops here) and
    `shape_retention` (SHAPE's own kept population)."""
    image_path, mask_path = _write_pair(tmp_path / "data", "img1", _asymmetric_cell_mask())
    tool = Single_Cell_Cropper_Tool()

    record, _trace = rsc.process_one_image(
        tool, image_path, mask_path, tmp_path / "out", rsc.DEFAULT_MIN_AREA, rsc.DEFAULT_MARGIN,
        rsc.DEFAULT_MASKCROP_MARGIN_FRAC, _NoVerdictEngine())

    assert len(record["maskcrop_contamination"]) == 2
    assert len(record["shape_contamination"]) == 2   # no engine -> nothing rejected -> all kept
    assert len(record["shape_retention"]) == 2
    assert all(0.0 <= c <= 1.0 for c in record["maskcrop_contamination"])
    assert all(0.0 <= c <= 1.0 for c in record["shape_contamination"])
    assert all(0.0 <= r <= 1.0 for r in record["shape_retention"])
    assert "n_geometry_reverts" not in record  # task 058 M7: zombie metric, removed


def test_cli_summary_headlines_mechanism_and_marks_quality_non_authoritative(tmp_path, monkeypatch):
    """Task 038 v13 (Honest reporting): the CLI's own written summary carries a NEW
    `"mechanism"` section (contamination reduction, retention distribution, the cost
    fields) and the EXISTING `"quality"` section now carries an explicit
    `"authoritative": False` marker -- the machine-readable proof this benchmark no
    longer claims a `crop_quality_score` delta as evidence of "better crops" (the
    `crop_quality_score` numbers themselves are UNCHANGED/still present, just
    relabeled -- CARD.md's own "Honest reporting" section, `crop_quality_score` "if
    still computed, keep it clearly labeled as a NON-authoritative...descriptor").

    Task 058 M7 (Codex QC finding 1): `n_geometry_reverts` was a zombie metric (always
    0 -- geometry retune no longer writes `bbox_inset_px`) and is REMOVED; assert it no
    longer appears in the mechanism section."""
    monkeypatch.setattr(rsc, "build_vision_engine", lambda *a, **k: _NoVerdictEngine())
    data_dir = tmp_path / "data"
    _write_pair(data_dir, "img1", _asymmetric_cell_mask(), seed=1)

    out_dir = tmp_path / "out"
    argv = ["run_shape_crop.py", "--data-dir", str(data_dir), "--out", str(out_dir),
            "--sample-cap", "10"]
    monkeypatch.setattr(sys, "argv", argv)
    rsc.main()

    summary = json.loads((out_dir / "run_shape_crop_summary.json").read_text())

    assert "mechanism" in summary
    for key in ("maskcrop_neighbor_contamination_mean", "shape_neighbor_contamination_mean",
                "neighbor_contamination_reduction", "shape_target_retention_mean",
                "shape_target_retention_min",
                "tool_calls_per_image", "token_cost_per_image", "time_per_image_s"):
        assert key in summary["mechanism"], f"{key} missing from mechanism section"
    assert "n_geometry_reverts" not in summary["mechanism"]
    assert "geometry_reverts_note" not in summary["mechanism"]

    assert summary["quality"]["authoritative"] is False
    assert "note" in summary["quality"]
    # the quality numbers THEMSELVES are still present/unchanged -- only relabeled
    assert summary["quality"]["maskcrop_crop_quality_score_mean"] is not None
    assert summary["quality"]["shape_crop_quality_score_mean"] is not None


def test_the_benchmarks_default_min_area_is_the_tools_one_value():
    """The cropper inherits the segmenter's object definition (2026-10-01): `DEFAULT_MIN_AREA` (benchmark.py's
    `--min-area`, which both arms -- MaskCrop and SHAPE -- receive), config.py's `default_min_area` and
    pairing's `DEFAULT_MIN_AREA` is the tool's default, the cell segmenter's min_size (15; 50 before).
    config.py is loaded by path: every benchmark directory has a `config` module."""
    import importlib.util

    import pairing

    from shape.tools.single_cell_cropper.tool import SEGMENTER_DEFAULT_MIN_SIZE

    spec = importlib.util.spec_from_file_location("_cropper_config", Path(__file__).resolve().parent / "cropper_config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    assert (rsc.DEFAULT_MIN_AREA == config.DEFAULT.default_min_area == pairing.DEFAULT_MIN_AREA
            == SEGMENTER_DEFAULT_MIN_SIZE == 15)
