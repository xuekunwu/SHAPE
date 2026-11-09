"""Vision crop-TRIAGE: the AUTHORITY on a data-screened REJECT candidate, GROUNDED on a
mask overlay, batched into labeled MONTAGES -- and (task 038 v12, human-locked, see (E)
below) the per-round DIRECTION-only gate on the iterative geometry TIGHTEN loop. GEOMETRY
was 100% PURE DATA (task 038 v10, human-locked) with NO VLM channel at all in this
module; task 038 v12 DELIBERATELY REVERSES that -- by explicit human directive, not a
silent violation, see (E) and CARD.md SS1/SS5's own updated rationale -- while the
DECISION itself (whether/how much to grow, the hard retention floor) still lives
entirely in `verifiers/crop.py::CropVerifier.adjudicate`; this module is STILL only ever
the VLM's DIRECTION-only say, never the magnitude.

DATA SCREENS, VLM DECIDES (task 038 restructure, human-locked) -- for REJECT only. A v7
cluster run (cell_00994, base commit 25070bc) sent EVERY crop (262/262) to the judge --
12.8M tokens for a run whose ONLY effect was 26 rejects, all corrected back to keep, and
~0 net geometry gain. A FOLLOW-UP v8 run then confirmed the geometry gain stays ~0 even
after that restructure: SQUARE-tight geometry cannot ISOLATE an adjacent neighbour (it
sizes half_side from the bbox's own longer axis, then pads the SHORTER axis out to match
-- that pad can still admit a neighbour sitting just outside the target's own bbox), and
the 10%-margin naive baseline both arms started from was already close to its own
ceiling. Task 038 v9 made TWO independent, deliberately separate fixes on top of that:

  (A) [v9, SUPERSEDED task 038 v10 -- see below] GEOMETRY was VLM-CONFIRMED for a
  DOUBLET candidate: the data DETECTED it (`n_complete_cells>=2`), the VLM (seeing the
  crop WITH its target mask outlined) decided tighten or leave. **Task 038 v10
  (human-locked) REMOVES this VLM channel entirely.** A real v10 QC audit found only
  12.5% of crops got isolated even though 79% carry `neighbor_contamination>0.2` --
  because most contamination comes from a PARTIAL neighbour (`n_complete_cells=1`, a
  neighbour merely clipped at the crop edge) that a "genuinely WHOLE second cell"
  trigger structurally cannot see. `mask_bbox` -- in v10 a crop tight to the target's
  own mask bbox, since the human rule of 2026-09-30 the SQUARE around it (every crop is
  square), whose square-preserving `auto_tighten` cuts remove a neighbour inside it
  (whole OR partial neighbours alike) while keeping at least 0.90 of the target -- is a
  SAFE, deterministic isolation that needs no VLM judgment to apply: it is now landed
  DIRECTLY from `neighbor_contamination > _MASK_BBOX_CONTAMINATION_MIN` (0.2,
  `verifiers/crop.py::CropVerifier`), 100% PURE DATA, exactly like OFF-CENTER already
  was. `verifiers/crop.py::CropVerifier.adjudicate` decides BOTH geometry triggers now
  -- this module (`crop_triage.py`) has NO geometry code path left at all.

  (B) REJECT: the mask DATA SCREENS which crops are even worth asking about, RELAXED in
  the v9 round -- `not_truncated < _SEVERE_NOT_TRUNCATED_MAX` (0.8, was 0.5) or
  `foreground_fraction < _EMPTY_DEBRIS_FOREGROUND_MAX` (0.2, was 0.02), CARD.md SS3 (on
  real NeurIPS22 data these two screens flag ~0% of crops). A crop flagging NEITHER of
  the 2 reject reasons is auto-kept, NEVER sent to the judge. The judge's verdict on a
  candidate is AUTHORITATIVE -- confirms the flagged defect -> reject; disagrees ->
  keep. There is no second-guessing pass after the judge answers.

  (C) THE GROUNDING FIX (the Q3 finding: a vision judge does not hallucinate when it can
  SEE which object is the target -- mirrors why the segmentor's own VLM judge, which
  always sees a mask overlay, does not hallucinate either). Every REJECT candidate's OWN
  target mask (the per-cell mask `tool.py` already persists per crop) is drawn on top as
  a semi-transparent COLORED fill + bright contour -- a genuinely COLORFUL,
  segmentation-style overlay (`_overlay_target_on_crop`, task 038 v10, human-locked)
  BEFORE it is sent, as one labeled cell in a montage (`_crop_overlay_array`, (D) below)
  -- so the judge can reliably tell the OUTLINED target apart from a neighbour or
  background, instead of guessing from raw pixels alone.

  (D) THE MONTAGE FIX (task 038 montage round, human-locked -- REPLACES an earlier
  per-crop reject confirm). A real v9 cluster run found the per-crop reject ask still
  FALSE-REJECTS a clear, good cell: cell_00554's 5 rejects were all GOOD spindle cells
  flagged `empty_debris`, because a genuinely thin/elongated single cell's own
  `foreground_fraction` is naturally low (< 0.2, the screening floor) WITHOUT the cell
  being empty/debris at all -- judged in ISOLATION, the per-crop ask has no way to
  notice this. `_judge_reject_montages` batches EVERY cut_off/empty_debris candidate
  into labeled grid montages (`_build_montage`, `montage_grid=(rows, cols)`, default 3x3
  -- `CropVerifier.triage_montage_grid`) instead: each grid cell is that crop's own
  GROUNDED overlay (C, above) plus a drawn INDEX label; ONE VLM call judges the WHOLE
  montage (`MontageVerdict`, one `MontagePositionVerdict` per labeled position), so the
  judge can SEE many candidates AT ONCE and COMPARE them -- catching a data-screen
  false-positive a per-crop ask structurally cannot. `problem="none"` on a flagged
  position is a RESCUE (kept, not rejected) -- exactly the cell_00554 case v9's own
  montage fixed. This channel calls the ENGINE DIRECTLY (`engine.generate([prompt,
  montage_bytes], response_format=MontageVerdict)`), NOT `llm_recommend.judge_item` --
  see `_judge_reject_montages`'s own docstring for the layering reason. Every montage
  actually sent is ALSO persisted to disk (task 038 v10, human-locked --
  `_save_montage_png`, under `_montages_dir_for(metadata_path)`) so a human can inspect
  EXACTLY what the judge saw; a no-engine/no-candidate run writes no montage files.

  (E) THE BLINDED A/B GATE (task 038 v13, human-directed -- RETIRES the v12 R2
  absolute-QC tighten gate, `tighten_recommend_crop`/`TightenVerdict`, which itself
  reversed (A)/(D)'s own "geometry stays pure-data" framing for this ONE lever). A
  methodology critique settled that `crop_quality_score` cannot be its own validator
  (v12 reweighted it toward the conclusion; even un-tuned, mask geometry is not the
  task objective -- only downstream SSL, a SEPARATE task OUT of 038, can validate
  that). Using the SAME score to both SELECT a tightened crop and PROVE it better is
  the 048 self-proving trap. `tool.py::_compute_crop_qc` still computes a
  NEIGHBOUR-AWARE `contamination_inset_px` exactly as R2 did (UNCHANGED this round):
  the exact pixel depth, read straight off the mask, that would exclude the non-target
  foreground reaching furthest toward the target on `contamination_side`, CAPPED so
  `target_retention` never drops below the hard floor. `verifiers/crop.py::
  CropVerifier.adjudicate`'s round-2+ pass VoI-SCREENS (below) which committed cuts
  are genuinely UNCERTAIN from data alone (low retention, or floor-stuck still
  contaminated) -- ONLY those go to `ab_recommend_crop` (this module): a BLINDED
  PAIRWISE comparison, the tightened CANDIDATE side by side with its own plain
  mask_bbox BASELINE (`tool.py` renders BOTH as real images, no history threading),
  L/R order RANDOMIZED per tile, BOTH panels grounded with the SAME colourful overlay
  (C, above). The judge answers ONE comparative question, `ABVerdict` -- which panel
  keeps the target MORE COMPLETE while containing FEWER other cells -- de-randomized
  back into `"candidate"` (commit the next data-computed cut), `"baseline"` (revert
  fully to the plain mask_bbox box + freeze), or `"abstain"` (treated the SAME as
  `"baseline"` -- no clear winner, keep the safer state). THE v6-HALLUCINATION
  GUARDRAIL still holds (the validated [[vlm_proposes_data_constrains_triage]]
  pattern): `ABVerdict`'s schema remains structurally unable to report a number or a
  magnitude, only one of the 3 comparative labels -- DATA alone measures the cut depth
  and owns the hard retention floor, both computed BEFORE the judge is ever asked.
  A comfortable, clean cut (VoI screen says NOT uncertain) is accepted on DATA ALONE,
  no VLM call at all -- the v7 lesson (12.8M tokens, ~0 gain) applied to THIS channel.
  Opt-in (same `llm_engine` hook as (B)/(D)); a HEADLESS run commits every
  data-screened cut directly (no A/B possible without an engine) -- the pure-data path
  still fully works without a judge, converging to data-clean or the floor. Montages
  persist into their OWN subdirectory (`_montages_dir_for(metadata_path,
  prefix="montages_ab")`), separate from (D)'s own reject montages for the same
  trial.

  (F) S1 -- "IS THE MASK EVEN POINTING AT A REAL CELL" (task 038 Round B,
  coordinator-directed, evidence-driven -- NOT a preference): a real audit of 5
  human-flagged kept crops found ONE pattern no mask-derived metric can see --
  the mask claims a cell, and the image at that location is empty; the visible
  cell material sits OUTSIDE the target mask (their own contamination/retention
  numbers looked fine, 0.10-0.26 / 0.90-1.0). Three cheap data-only detectors
  (mean contrast, boundary adherence, intensity histogram) all failed to catch
  this reliably -- it is the VLM's job. The DATA screen deciding which crops are
  even worth asking (`verifiers/crop.py::_s1_target_region_signal_poor`,
  DELIBERATELY LOOSE, VoI-capped at `CropVerifier.s1_candidate_cap`) lives in the
  VERIFIER, not here -- mirrors the A/B gate's own candidate-selection split
  (data-driven selection is the verifier's job; this module owns the montage/
  prompt/schema machinery). `s1_recommend_crop` judges candidates in the SAME
  batched, labeled-grid montages every other channel uses (`_build_montage`/
  `_crop_overlay_array`, reused not reimplemented), persisted to their own
  `montages_s1_<hash>/` subdirectory. ONE narrative question per position: is
  there really a cell under the coloured region, or is that region empty --
  INCLUDING when real cell material IS visible but lies outside the coloured
  region (load-bearing IN THE PROMPT TEXT, not only the schema field's own
  description, per [[judge_prompt_narrative_not_schema_field]]). LANDING is a
  CONSTRAINT, never a score (mirrors REJECT's own framing): a confirmed
  `target_region_empty` sets `crop["vision_reject"]` -- the SAME flag REJECT (B)
  already sets, so S1 needs no separate "kept" concept and `crop_kept`/
  `_rewrite_kept_manifest`'s existing `not vision_reject` predicate picks it up
  automatically, with ZERO changes to either. `cell_present`/`abstain`/an
  unmentioned position/any failure all KEEP (never discard on silence). Also
  folded into this round (human directive, not cosmetic): `_overlay_target_on_
  crop`'s own contour is now ALPHA-BLENDED (`_OVERLAY_ALPHA`, the SAME 0.5 the
  fill already uses), not opaque -- an opaque contour destroyed the pixel
  information exactly where S1 (and S3's cut-off check) needs to look,
  disproportionately so on the tiny real crop sizes this tool produces (8x12,
  21x21, 28x13, 44x44 px).

This module now has THREE live VLM channels -- REJECT (B/C/D, via
`vision_recommend_crop` -> `_judge_reject_montages`), the BLINDED A/B GATE (E, via
`ab_recommend_crop`), and S1 (F, via `s1_recommend_crop`) -- all calling the engine
DIRECTLY, NOT through `llm_recommend.judge_item`/`build_judge_prompt` (those helpers
are shaped around ONE item's own evidence, not a multi-item labeled grid).
`llm_recommend.py` itself is used ONLY for its two GENERIC, already tool-agnostic
landers -- `write_kept_manifest`/`write_triage_trace` -- both reused unchanged (by
the REJECT channel's own `vision_recommend_crop` call ONLY; A/B and S1 do not write
a kept-manifest or triage-trace of their own -- S1's own verdicts instead thread
through `vision_recommend_crop`'s `extra_verdicts` parameter, task 038 Round B, so
its evidence lands in the SAME trace file rather than a second, colliding sidecar
write); `judge_item`/`judge_items`/`build_judge_prompt` are NOT called by this
module's live path at all any more (task 038 v10 removed their one caller here, the
doublet channel) -- they remain used by `run_arbitration_spotcheck.py`'s own
standalone comparison harness, unaffected by this round. `llm_recommend.py` is NOT
touched by this round either (human-locked layering). This module's own
contribution, beyond CARD.md + quant.py, is WHAT evidence a candidate carries
(`_crop_overlay_array`, `_quant_evidence` -- the latter still used by the standalone
spot-check harness too), WHICH crops are even worth asking (`_vlm_candidate_reasons`
for REJECT, task 038 v10; the caller's own VoI screens in `verifiers/crop.py` for
the A/B gate (task 038 v13) and S1 (task 038 Round B)), WHAT to ask about each
montage (`_build_montage_prompt` / `_build_ab_montage_prompt` / `_build_s1_montage_
prompt`), and WHAT "kept" means for this tool (`not vision_reject`).

History (task 038, abbreviated -- earlier rounds tried: sending every crop with the
judge's own reject checked against the SAME two mask floors used here, under 4- then
2-reason taxonomies; a per-crop geometry question bundled into the same call, gated by
the same two floors plus a third; then (v7) screening candidates by data first and never
asking about geometry at all; then (v9) grounding every candidate's image with its own
mask overlay AND VLM-confirming an `n_complete_cells>=2` doublet candidate; then (v9
montage) batching the reject ask; then (v10) REMOVING the VLM doublet channel entirely
in favor of a pure-data `neighbor_contamination` trigger. Each was a real, human-approved
design, and each got corrected by a real cluster run or audit: v4 over-rejected 86% of
crops (178 false `severe_blur` claims); v6 hallucinated 69% of its `empty_debris` rejects
(`target_dominance` was wrongly read as a signal for "is a cell present at all") and
blanket-retuned 128 cells net-negative; v7 spent 12.8M tokens for ~0 quality gain; v8
confirmed the geometry gain STAYS ~0 even after the token fix, because square-tight
cannot isolate an adjacent neighbour and the 10% naive baseline was already near-optimal
(task 038 v9's own two fixes, A/B above); v9 itself, even after grounding, still
FALSE-REJECTED thin/elongated good cells on the per-crop reject ask, judged in isolation
-- the direct motivation for the montage batching (D, above); a v10 QC AUDIT then found
the v9 doublet trigger's own real-world coverage was poor -- only 12.5% of crops
isolated despite 79% carrying `neighbor_contamination>0.2` -- because `n_complete_cells
>=2` structurally misses a PARTIAL neighbour (the dominant real-world case), the direct
motivation for this round's pure-data contamination trigger (A, above). Git history
holds the detail of each round; this module documents only the CURRENT, human-locked
design. `saturated`/`severe_blur` are acquisition issues, not this tool's call to
reject on either.

CORE module: imports stdlib + pydantic + numpy (+ the engine/quant/gate/CARD/cv2 loader
at call time). Must NOT import benchmarks.
"""
from __future__ import annotations

import logging
import os
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from shape.utils.schemas import VLRRResult

logger = logging.getLogger("shape.agent.verification")

# The CARD's own 2 REJECT reasons (CARD.md SS2 VERDICT DECLARATION) -- EXHAUSTIVE, not
# illustrative: a doublet/contaminated crop is NOT a reject reason -- it is a PURE-DATA
# geometry action (`verifiers/crop.py::CropVerifier.adjudicate`'s `mask_bbox` override,
# triggered directly from `neighbor_contamination`, task 038 v10 -- no judge involved at
# all), a categorically different axis from reject/keep; `saturated`/`severe_blur` are
# acquisition/exposure/focus issues this tool neither causes nor can fix. `problem`'s enum
# below makes the model structurally unable to NAME an out-of-taxonomy REJECT reason;
# `_apply_montage_verdicts` additionally checks this same set at runtime as a second,
# defense-in-depth layer (for an engine that does not enforce structured output).
_VALID_REJECT_REASONS = frozenset({"cut_off", "empty_debris"})


class CropTriageVerdict(BaseModel):
    """What the judge reports for ONE candidate crop (its image GROUNDED with the
    target's own mask drawn on top) -- confirm or deny the flagged reject scenario named
    in the prompt's "SCENARIO TO CONFIRM" evidence (`_scenario_confirm_text`).

    Task 038 v10: geometry (`doublet`/`mask_bbox`) is no longer a VLM-judged axis at
    all -- this schema is now population-triage ONLY (`verdict`/`problem`), the same
    shape it held before the v9 restructure briefly added a `doublet` field (now
    removed). It is NOT used by the live production path any more
    (`vision_recommend_crop` -> `_judge_reject_montages` uses `MontageVerdict`, below,
    exclusively) -- this schema stays for the LEGACY `triage_crop`/`triage_crops` (task
    041, exercised only by tests) and `run_arbitration_spotcheck.py`'s own standalone
    fused-judge comparison."""
    verdict: Literal["keep", "reject"]
    confidence: Literal["high", "low"]
    problem: Literal["cut_off", "empty_debris", "none"] = Field(
        description="The SPECIFIC reason, if verdict=reject -- one of the CARD's 2 reject "
                    "reasons ONLY, and ONLY the one(s) named in the SCENARIO TO CONFIRM "
                    "evidence for THIS crop (the flagged defect this crop was actually "
                    "screened for). Blur and SATURATION are NEVER valid problem values "
                    "(acquisition/exposure issues, not the cropper's to fix). A doublet/"
                    "contaminated crop is NEVER a reject reason -- geometry (mask_bbox) "
                    "is a separate, pure-data mechanism, never this judge's call. "
                    "Off-center or ordinary target_dominance are also never reject "
                    "grounds. Use 'none' when verdict=keep.")
    reasoning: str


class MontagePositionVerdict(BaseModel):
    """What the judge reports for ONE labeled position in a REJECT-candidate montage
    (task 038 montage round, human-specified schema -- see `_build_montage`/
    `_build_montage_prompt`). `index`/`crop_id` together identify WHICH crop this
    verdict is about; `_apply_montage_verdicts` lands it back onto that crop's own
    dict, trusting `index` over `crop_id` on a mismatch (see that function's own
    docstring for why)."""
    index: int = Field(description="The grid position (0..N-1) this verdict is about "
                                    "-- the SAME number drawn on that position's own "
                                    "label in the montage image.")
    crop_id: str = Field(description="The crop identifier printed alongside this "
                                     "position in the prompt's own position list -- "
                                     "restate it here as a cross-check; if it does not "
                                     "match what you were told for this index, the "
                                     "INDEX still wins.")
    problem: Literal["empty_debris", "cut_off", "none"] = Field(
        description="'empty_debris' or 'cut_off' ONLY if the outlined target at this "
                    "position is GENUINELY that defect. 'none' if the outlined target "
                    "is actually a FINE, usable single cell that was wrongly "
                    "data-flagged (e.g. a thin/elongated cell -- its own "
                    "foreground_fraction is naturally low without the cell being "
                    "empty/debris at all) -- this is a RESCUE, the exact case this "
                    "montage exists to catch.")
    severity: Literal["severe", "moderate", "none"] = Field(
        description="How bad the confirmed problem is ('none' whenever problem='none') "
                    "-- reported for the trace/notes; does NOT itself gate the "
                    "reject/keep landing (only `problem`+`confidence` do, see "
                    "`_apply_montage_verdicts`).")
    confidence: Literal["high", "low"]


class MontageVerdict(BaseModel):
    """One `MontagePositionVerdict` per labeled position in the montage (task 038
    montage round). May be a SUBSET of the montage's own positions (a position the
    judge does not mention is treated as NOT rejected, the same keep-biased default
    every other verdict shape in this module uses on any gap/failure)."""
    verdicts: list[MontagePositionVerdict]


class ABPositionVerdict(BaseModel):
    """What the judge reports for ONE labeled position in a blinded A/B-candidate
    montage (task 038 v13, human-directed -- RETIRES `TightenPositionVerdict`'s own
    absolute-QC ask; a SIBLING of `MontagePositionVerdict`, SAME landing rule: the grid
    INDEX is the ground truth, `crop_id` only a restated cross-check). A COMPARATIVE
    pairwise ask, not an absolute quality-control question -- and the v6-hallucination
    GUARDRAIL in schema form regardless: the judge is structurally unable to report
    anything OTHER than one of 3 labels -- it can never name a dominance number, a
    pixel count, or any other MAGNITUDE. Data (the caller, `verifiers/
    crop.py::CropVerifier.adjudicate`, reading `tool.py::_compute_crop_qc`'s own
    `contamination_inset_px`) computes the cut depth and owns the hard retention floor
    entirely outside this schema; the judge only ever says WHICH panel it prefers."""
    index: int = Field(description="The grid position (0..N-1) this verdict is about "
                                    "-- the SAME number drawn on that position's own "
                                    "label in the montage image.")
    crop_id: str = Field(description="The crop identifier printed alongside this "
                                     "position in the prompt's own position list -- "
                                     "restate it here as a cross-check; if it does not "
                                     "match what you were told for this index, the "
                                     "INDEX still wins.")
    verdict: Literal["left_better", "right_better", "abstain"] = Field(
        description="'left_better': the LEFT panel of this position's own side-by-side "
                    "pair keeps the coloured target more complete while containing "
                    "fewer other cells. 'right_better': same judgment, the RIGHT panel "
                    "wins instead. 'abstain': the two panels are not clearly "
                    "distinguishable -- genuinely too close to call, no forced pick.")
    confidence: Literal["high", "low"]


class ABVerdict(BaseModel):
    """One `ABPositionVerdict` per labeled position in the montage (task 038 v13,
    human-directed -- a sibling of `MontageVerdict`). May be a SUBSET of the montage's
    own positions (a position the judge does not mention is treated as NO opinion --
    the caller's own data-screened default then applies, the same keep/tighten-biased
    default every other verdict shape in this module uses on any gap/failure)."""
    verdicts: list[ABPositionVerdict]


class S1PositionVerdict(BaseModel):
    """What the judge reports for ONE labeled position in an S1 montage (task 038
    Round B, coordinator-directed -- the "does the mask even point at a real cell"
    channel: a sibling of `MontagePositionVerdict`/`ABPositionVerdict`, SAME landing
    rule -- the grid INDEX is ground truth, `crop_id` only a restated cross-check.
    Structurally CANNOT express a magnitude (the same v6-hallucination guardrail
    `ABPositionVerdict` already documents) -- only one of 3 labels."""
    index: int = Field(description="The grid position (0..N-1) this verdict is about "
                                    "-- the SAME number drawn on that position's own "
                                    "label in the montage image.")
    crop_id: str = Field(description="The crop identifier printed alongside this "
                                     "position in the prompt's own position list -- "
                                     "restate it here as a cross-check; if it does not "
                                     "match what you were told for this index, the "
                                     "INDEX still wins.")
    verdict: Literal["cell_present", "target_region_empty", "abstain"] = Field(
        description="'cell_present': the coloured region genuinely contains cell "
                    "material. 'target_region_empty': the coloured region is empty / "
                    "background / noise -- INCLUDING when a real cell IS visible "
                    "somewhere in this tile but it lies OUTSIDE the coloured region "
                    "(the mask is pointing at the wrong place, not just an empty "
                    "crop). 'abstain': genuinely too ambiguous to call either way.")
    confidence: Literal["high", "low"]


class S1Verdict(BaseModel):
    """One `S1PositionVerdict` per labeled position in the montage (task 038 Round B).
    May be a SUBSET of the montage's own positions -- a position the judge does not
    mention is treated as NOT rejected (the SAME keep-biased default every other
    verdict shape in this module uses on any gap/failure -- CARD.md's own "never
    discard on silence" mandate, applied here too)."""
    verdicts: list[S1PositionVerdict]


# The calibrated, KEEP-biased prompt validated in task 017 (run_cropper_triage.py);
# reject taxonomy is 2 (see `_VALID_REJECT_REASONS` for the full history/why). This
# constant is ONLY used by the LEGACY `triage_crop`/`triage_crops` (task 041, called only
# by tests -- NOT the live judge path, which is `vision_recommend_crop` ->
# `llm_recommend.judge_item` -> `build_judge_prompt`, below).
_TRIAGE_PROMPT = (
    "This is ONE single-cell microscopy crop (grayscale, small) for morphology analysis. "
    "DEFAULT TO KEEP. Only 'reject' if you are CLEARLY confident it is unusable, for one of these "
    "TWO reasons ONLY -- neither is fixable by this tool's own geometry: the cell is obviously "
    "CUT OFF at the crop edge with a large piece missing; or there is essentially NO cell mask "
    "in the crop at all -- empty/debris-looking, not just a tolerated minority-share target next "
    "to a neighbour (a genuinely low-dominance-but-present target is NOT empty_debris, it is a "
    "tolerated, normal crop). A doublet (two or more separate cells) is NEVER a reject reason -- "
    "this tool's own geometry can isolate the target from it instead. Blur and SATURATION are "
    "NOT valid reject reasons either (acquisition/exposure issues this crop cannot fix, not this "
    "judge's call). A plausible single cell -- even if small, dim, slightly off-center, "
    "low-contrast, blurry, or lightly touching a neighbour at the very edge -- is KEEP. Report "
    "verdict (keep/reject), confidence (high/low), a one-phrase problem (or 'none'), and one "
    "sentence of reasoning.")


def _crop_bytes(crop: dict) -> bytes | None:
    """Read a crop's image bytes from its recorded path (`path` for bare crops, `crop_path`
    for cropper-metadata crops)."""
    path = crop.get("path") or crop.get("crop_path")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except Exception as exc:
        logger.warning("crop-triage: cannot read crop %s (%s)", path, exc)
        return None


def triage_crop(engine_mm, crop: dict) -> tuple[bool, CropTriageVerdict | None]:
    """Judge ONE crop. Returns (reject, verdict). reject=True only on a HIGH-confidence clear
    defect; any failure / no engine / unreadable -> (False, None) so the deterministic gate stands."""
    if engine_mm is None:
        return False, None
    img = _crop_bytes(crop)
    if img is None:
        return False, None
    try:
        resp = engine_mm.generate([_TRIAGE_PROMPT, img], response_format=CropTriageVerdict)
        rec = resp.get("content") if isinstance(resp, dict) else resp
        if not hasattr(rec, "verdict"):
            return False, None
        reject = (rec.verdict == "reject" and rec.confidence == "high")
        return reject, rec
    except Exception as exc:  # pragma: no cover - defensive: triage must never break QC
        logger.warning("crop-triage failed on one crop (%s); keeping", exc)
        return False, None


def triage_crops(engine_mm, crops: list[dict], *, sample_cap: int | None = None,
                 valid_kwargs: dict | None = None) -> dict:
    """Annotate crops in place with `vision_reject` (bool) + `vision_problem` for the judged ones.

    Only ADDS rejections: a crop already invalid by the metric gate is not re-examined. Judges up
    to `sample_cap` of the metric-valid crops (None = all). `valid_kwargs` is forwarded to
    `is_crop_valid` so the "metric-valid" candidate set matches the caller's gate (gate_groups /
    derived bounds). Returns a small summary. No engine -> no-op (nothing annotated), so downstream
    validity == the deterministic gate exactly.
    """
    if engine_mm is None:
        return {"judged": 0, "rejected": 0, "engine": False}
    from .verifiers.crop import is_crop_valid
    vk = valid_kwargs or {}
    # only consider crops the metric gate KEPT (the judge tightens, never rescues)
    candidates = [c for c in crops if is_crop_valid(c, **vk)[0]]
    if sample_cap is not None and len(candidates) > sample_cap:
        # deterministic sample: evenly spaced, so the summary is reproducible
        step = len(candidates) / float(sample_cap)
        candidates = [candidates[int(i * step)] for i in range(sample_cap)]
    judged = rejected = 0
    for crop in candidates:
        reject, verdict = triage_crop(engine_mm, crop)
        judged += 1
        if reject:
            rejected += 1
            crop["vision_reject"] = True
            crop["vision_problem"] = getattr(verdict, "problem", "") if verdict else ""
    return {"judged": judged, "rejected": rejected, "engine": True}


def crop_kept(crop: dict) -> bool:
    """A crop is kept iff the metric gate passes AND the vision judge did not reject it."""
    from .verifiers.crop import is_crop_valid
    return is_crop_valid(crop)[0] and not crop.get("vision_reject", False)


# --------------------------------------------------------------------------- #
# Task 038 restructure (v9), then v10 (this round): DATA-SCREENED, VLM-DECIDED,
# MASK-GROUNDED reject triage -- the cropper's THIN adapter (`vision_recommend_crop`,
# below), reusing `llm_recommend.py`'s generic `write_kept_manifest`/`write_triage_trace`
# landers (task 038 M-B) but judging candidates itself via the montage mechanism (D,
# below), not `llm_recommend.judge_item`. NOT `triage_crops` above -- see the module
# docstring for why the two coexist.
# --------------------------------------------------------------------------- #


def _cropper_card() -> str | None:
    """The Single_Cell_Cropper_Tool CARD body (cached by `load_tuning_card`), or None if it
    cannot be loaded -- the CARD channel of the judge prompt; loaded lazily/defensively
    (a CARD read failure must not break the judge, same "any failure -> no rejection"
    contract as the rest of this module)."""
    try:
        from shape.tools.base import load_tuning_card
        return load_tuning_card("single_cell_cropper")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("crop-triage: CARD load failed (%s)", exc)
        return None


# The CARD sections the mask-check judge reads: what the crops are for (the purpose and the quality order) and
# what the crop readouts mean. The rest of the CARD -- the loops, the levers, the other channels' verdict
# rules and the development notes (history, dates, task references, findings) -- is not its information.
_MASK_CHECK_CARD_SECTIONS = ("## Purpose", "## 3. CALIBRATION")


def _cropper_judge_rules() -> str | None:
    """The mask-check judge's part of the cropper CARD: each `_MASK_CHECK_CARD_SECTIONS` section, from its H2 line to
    the next H2 (without the `---` rule before it), or None when the CARD cannot be loaded or holds none of
    them -- the same "any failure -> no CARD section" contract as `_cropper_card`."""
    card = _cropper_card()
    if not card:
        return None
    lines = card.splitlines()
    sections = []
    for title in _MASK_CHECK_CARD_SECTIONS:
        start = next((i for i, line in enumerate(lines) if line.startswith(title)), None)
        if start is None:
            logger.warning("crop-triage: CARD section %r not found", title)
            continue
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        body = "\n".join(lines[start:end]).strip()
        sections.append(body[:-3].rstrip() if body.endswith("\n---") else body)
    return "\n\n".join(sections) or None


def _crop_pixels(crop: dict) -> np.ndarray | None:
    """The crop's raw pixel array, read from the same `path`/`crop_path` `_crop_bytes`
    reads -- the QUANT channel needs pixels, not just bytes. None on any read failure.

    Task 038 Round C (coordinator-directed, a REAL bug -- verified against the actual
    delivered v15 crop files, not assumed): a multi-channel crop is MAX-PROJECTED across
    channels (`arr.max(axis=2)`), NOT reduced to channel 0. NeurIPS22-CellSeg puts its
    real signal in a DIFFERENT channel per image -- e.g. `cell_00860`'s own delivered
    crops are 92%+ empty on channel 0 (its signal sits in channel 2), while
    `cell_00994`'s signal splits between channels 0 and 1 with channel 2 empty --
    `arr[..., 0]` was silently feeding the VLM/quant an effectively BLANK image for
    `cell_00860` and a partial one for `cell_00994`, exactly the kind of silent
    grounding failure the v13 uint16-mask-lattice bug was (a different mechanism, same
    class of failure: the code and a prior audit both read `im[..., 0]` and so both
    missed it identically). Max-projection preserves whichever channel actually carries
    signal for THIS image, with one consistent rule across every image (a per-crop
    "pick the strongest channel" rule would vary WITHIN an image and complicate the
    otherwise-simple 2D contract `_overlay_target_on_crop`/the quant channel already
    expect) -- verified on the real delivered crops before shipping: `cell_00860`'s own
    mean fraction of pixels below 10/255 falls from 0.934 (channel 0 alone) to 0.618
    (max-projected), and `cell_00994`'s falls from 0.219 to 0.119 (an IMPROVEMENT, not
    a degradation, since its channel 1 carries some signal channel 0 alone misses)."""
    path = crop.get("path") or crop.get("crop_path")
    if not path or not os.path.exists(path):
        return None
    try:
        from skimage import io as skio
        arr = np.asarray(skio.imread(path))
        if arr.ndim == 3:
            arr = arr.max(axis=2)
        return arr
    except Exception as exc:
        logger.warning("crop-triage: cannot read crop pixels %s (%s)", path, exc)
        return None


def _target_mask_pixels(crop: dict) -> np.ndarray | None:
    """The TARGET's own per-cell mask array for this crop, read from `crop["mask_path"]`
    (`tool.py::_generate_cell_crops` persists it per crop and threads the path into the
    JSON metadata, task 038 v9). None when no path is recorded or the file is
    unreadable -- the overlay then gracefully degrades to no overlay
    (`_crop_overlay_array`)."""
    path = crop.get("mask_path")
    if not path or not os.path.exists(path):
        return None
    try:
        from skimage import io as skio
        arr = np.asarray(skio.imread(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr
    except Exception as exc:
        logger.warning("crop-triage: cannot read target mask %s (%s)", path, exc)
        return None


_OVERLAY_ALPHA = 0.5   # task 038 Round B: the ONE alpha both the fill AND (now) the
                       # contour blend at -- a single named constant so the two stay
                       # in sync by construction, not by two matching literals.


def _overlay_target_on_crop(crop_pixels: np.ndarray, mask_pixels: np.ndarray) -> np.ndarray:
    """Draws the TARGET's own mask onto its crop as a semi-transparent COLORED fill +
    a bright contour -- a genuinely COLORFUL, segmentation-style overlay (task 038 v10
    round, human-locked: "overlay 要显示 mask 为 colorful 像 segmentation 那样" --
    REPLACES the earlier flat green tint; the grounding fix itself, Q3, is UNCHANGED: a
    vision judge reliably tells target from neighbour/background when it can SEE which
    object is the target, the same reason the segmentor's own VLM judge does not
    hallucinate).

    Reuses `skimage.color.label2rgb` -- the genuinely SHARED, correct-granularity
    colored-mask primitive: `cellpose.plot.mask_overlay` (the segmentor's OWN overlay
    routine, `cell_segmenter/tool.py`/`organoid_segmenter/tool.py`) was investigated
    first (per the coordinator's own instruction) and NOT reused -- it colors MULTIPLE
    distinct FULL-FIELD instances by RANDOM hue, the wrong granularity for a SINGLE
    target mask within one already-cropped image (the SAME reason the v9 round gave for
    writing this small, cropper-specific function instead of reusing it in the first
    place); `shape/utils/segmentation_qc.py` has no colored-mask drawer of its own
    (only `to_display_uint8`/`normalize_for_overlay`, still reused below for the base
    image). `label2rgb` is the correct-granularity, ALREADY-a-dependency (via
    `_crop_pixels`/`_target_mask_pixels`'s own `skimage.io` use) primitive instead of
    hand-rolling the color-blend math. Its own DEFAULT color cycle ('red' first, for a
    single binary label) is bright and DETERMINISTIC -- distinct from the montage's own
    yellow index label (`_label_montage_cell`), so the two never visually clash, and
    legible at the small montage-cell size. `label2rgb` returns RGB-ORDERED floats;
    converted to a BGR uint8 array (`cv2.cvtColor`) before the contour is drawn on top
    and before this array enters the REST of this module's own cv2-native pipeline
    (`_build_montage`'s canvas, `cv2.imencode`) -- otherwise the fill's red/blue
    channels would render swapped in the final PNG. Returns a cv2-native (BGR-ordered)
    RGB uint8 array; the crop's raw pixels are otherwise UNCHANGED (no mask-out --
    CARD.md SS1: neighbours/background stay visible, only the target is marked).

    Task 038 Round B (human directive, folded into the S1 round -- NOT cosmetic): the
    contour is now ALPHA-BLENDED with the underlying pixels (`_OVERLAY_ALPHA`, the SAME
    0.5 the fill above already uses -- "the same way the fill already is"), not drawn
    OPAQUE. An opaque contour completely destroys the pixel information exactly where a
    judge needs to look -- S1 ("is there a cell UNDER the coloured region, or is the
    visible material OUTSIDE it") and S3 ("is the coloured target cut off at an edge")
    both hinge on the boundary band. This matters MORE than it looks: real delivered
    crop sizes from the v16 run include 8x12, 21x21, 28x13, 44x44 px -- the contour is
    drawn HERE, in NATIVE crop pixels, BEFORE `_label_montage_cell`'s own resize to the
    fixed 160px montage tile (confirmed by reading the call chain, not assumed) -- so a
    1px-native line on an 8x12 crop is upscaled ~20x to the montage, occupying a much
    larger fraction of the tile's own boundary band than the SAME 1px would on a
    128x128 native crop upscaled only ~1.25x. Thickness itself is already at cv2's own
    technical floor (1px, cannot be drawn thinner) -- there is no "more proportional"
    thickness to set below that floor, so semi-transparency is the ONLY available lever
    against this, and it applies uniformly regardless of native crop size (a 1px-wide
    band that lets ~50% of the true pixel value through is exactly as informative at
    8x12 as at 512x512)."""
    import cv2
    from skimage.color import label2rgb

    from shape.utils.segmentation_qc import to_display_uint8

    base = to_display_uint8(np.asarray(crop_pixels))
    m = (np.asarray(mask_pixels) > 0)
    if m.shape[:2] != base.shape[:2]:
        # shape mismatch -- safety net, no overlay rather than a crash
        return (cv2.cvtColor(base, cv2.COLOR_GRAY2BGR) if base.ndim == 2
                else np.asarray(base)[..., :3].copy())
    colored = label2rgb(m.astype(np.int32), image=base, bg_label=0, alpha=_OVERLAY_ALPHA, image_alpha=1.0)
    tint = cv2.cvtColor((np.clip(colored, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros(tint.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, 1)   # 1px -- cv2's own floor, see docstring
    contour_bool = contour_mask > 0
    white = np.full_like(tint, 255)
    blended = (tint.astype(np.float32) * (1.0 - _OVERLAY_ALPHA)
              + white.astype(np.float32) * _OVERLAY_ALPHA)
    tint = np.where(contour_bool[..., None], blended, tint).astype(np.uint8)
    return tint


def _quant_evidence(crop: dict) -> dict:
    """Mask-based quant numbers for this crop when available -- the mask is a GIVEN input
    to this tool (the task is "given the mask, crop the highest-quality crop"), not a GT
    yardstick it is being scored against, so there is no circularity to avoid.
    `target_dominance`, `not_truncated`, `single_object`, AND `centered` are all read off
    the crop's own MASK-based `qc` block (`crop["qc"]`, from `tool.py::_compute_crop_qc`,
    threaded onto each real-tool crop dict by `adapt_crop`) when present -- the reliable,
    full-field-aware values, not `quant.py`'s own weaker pixel-Otsu re-derivation.
    `sharpness` stays PIXEL-computed either way (blur has no mask version). Falls back to
    the PIXEL-only `{sharpness, target_dominance}` pair when `crop` has no `qc` block --
    bare/HPA crops via `adapt_bare_crops`, which carry no full-field mask context. Empty
    dict on any failure (no pixels, scoring error, malformed `qc`) -- the judge then
    arbitrates on vision + the CARD alone."""
    pixels = _crop_pixels(crop)
    if pixels is None:
        return {}
    try:
        from shape.tools.single_cell_cropper import quant
        qc = crop.get("qc")
        if qc:
            return {
                "sharpness": quant.sharpness(pixels),
                "not_truncated": float(1.0 - qc["border_touch"]),
                "single_object": float(qc["single_object"]),
                "target_dominance": float(qc["target_dominance"]),
                "centered": float(1.0 - qc["centroid_offset"]),
            }
        return {"sharpness": quant.sharpness(pixels),
               "target_dominance": quant.target_dominance(pixels)}
    except Exception as exc:
        logger.warning("crop-triage: quant scoring failed (%s)", exc)
        return {}


# CARD.md SS3's own data-screen thresholds (task 038 v9 restructure -- RELAXED; task 038
# v10 -- doublet REMOVED from this screen entirely, see the module docstring's (A)/(B)):
# a crop crossing either floor is a REJECT CANDIDATE -- its image (grounded with the
# target's own mask overlay, `_crop_overlay_array`) is placed into a labeled montage
# (`_judge_reject_montages`, (D)) and judged; a crop crossing NEITHER is auto-kept and
# NEVER sent. The judge's verdict on a candidate is AUTHORITATIVE (CARD.md SS2) -- there
# is no second-guessing pass after it answers. `target_dominance` plays NO role in this
# REJECT screen -- it measures population-share among cells that ARE present, not whether
# a cell is here at all (a DIFFERENT question, CARD.md SS3); a real cluster run (v6)
# found it wrongly driving 59/86 hallucinated `empty_debris` rejects when used for this.
# (`neighbor_contamination`, its near-complement, DOES drive geometry now -- task 038
# v10 -- but as a PURE-DATA trigger the judge is never asked about, see
# `verifiers/crop.py`.)
_SEVERE_NOT_TRUNCATED_MAX = 0.8      # below this: cut_off is a candidate (CARD.md SS3; was 0.5)
_EMPTY_DEBRIS_FOREGROUND_MAX = 0.2   # below this: empty_debris is a candidate (CARD.md SS3; was 0.02)


def _vlm_candidate_reasons(crop: dict) -> list[str]:
    """CARD.md SS2/SS3's data screen: which of the CARD's 2 REJECT scenarios this crop's
    OWN mask data flags as worth asking the judge about -- NOT a verdict, just candidacy.
    Task 038 v10: `doublet` is REMOVED from this list entirely -- geometry (mask_bbox/
    recenter) is now 100% pure-data (`verifiers/crop.py::CropVerifier.adjudicate`), no
    VLM candidacy at all. A crop flagging NEITHER reason is never sent to the judge at
    all (`vision_recommend_crop`, below) -- it is auto-kept. A crop flagging BOTH is
    asked about both (`_scenario_confirm_text` joins every flagged instruction into one
    prompt, one judge call). Missing data -> that scenario simply does not flag
    (conservative: no data, no candidacy)."""
    reasons: list[str] = []
    qc = crop.get("qc") or {}
    border_touch = crop.get("border_touch")
    if border_touch is not None and (1.0 - float(border_touch)) < _SEVERE_NOT_TRUNCATED_MAX:
        reasons.append("cut_off")
    fg = qc.get("foreground_fraction")
    if fg is not None and float(fg) < _EMPTY_DEBRIS_FOREGROUND_MAX:
        reasons.append("empty_debris")
    return reasons


def _cut_off_confirm_text(crop: dict) -> str:
    """The TARGETED confirm instruction for a cut_off candidate -- names this crop's own
    `not_truncated` value against the CARD's screening floor (CARD.md SS3), grounded on
    the mask overlay: the target is the OUTLINED object."""
    border_touch = crop.get("border_touch")
    not_truncated = 1.0 - float(border_touch) if border_touch is not None else float("nan")
    return (f"This crop's own mask data shows the OUTLINED target cell touching the crop "
            f"border, with a potentially large fraction cut off (not_truncated="
            f"{not_truncated:.3f}, below the CARD's {_SEVERE_NOT_TRUNCATED_MAX} screening "
            f"floor -- CARD.md SS3). Confirm 'cut_off' ONLY if the image shows a "
            f"genuinely large piece of the outlined target actually missing at the edge; "
            f"if the target looks essentially whole, verdict=keep.")


def _empty_debris_confirm_text(crop: dict) -> str:
    """The TARGETED confirm instruction for an empty_debris candidate -- names this
    crop's own `foreground_fraction` value against the CARD's screening floor (CARD.md
    SS3)."""
    fg = (crop.get("qc") or {}).get("foreground_fraction")
    fg = float(fg) if fg is not None else float("nan")
    return (f"This crop's own mask data shows almost no cell foreground in the crop "
            f"(foreground_fraction={fg:.4f}, below the CARD's {_EMPTY_DEBRIS_FOREGROUND_MAX} "
            f"screening floor -- CARD.md SS3). Confirm 'empty_debris' ONLY if the image is "
            f"genuinely empty/debris-looking with no real cell present at all; if a real "
            f"(even small, even minority-share) cell is visible, verdict=keep.")


_SCENARIO_CONFIRM_TEXT_FN = {"cut_off": _cut_off_confirm_text,
                            "empty_debris": _empty_debris_confirm_text}


def _scenario_confirm_text(crop: dict) -> str:
    """Joins the TARGETED confirm instruction(s) for every reason `_vlm_candidate_reasons`
    flagged on THIS crop -- a crop flagging multiple scenarios gets ALL of their
    instructions, in the same evidence channel (`vision_recommend_crop`'s own
    `evidence_fn`, below), one judge call. Only ever called on a crop that IS a candidate
    (`vision_recommend_crop` only judges candidates), so this is never empty in
    practice."""
    reasons = _vlm_candidate_reasons(crop)
    return "\n".join(_SCENARIO_CONFIRM_TEXT_FN[r](crop) for r in reasons)


# --------------------------------------------------------------------------- #
# Task 038 montage round (human-locked): the REJECT confirm (cut_off/empty_debris)
# moves from ONE VLM call per candidate to BATCHED, labeled grid MONTAGES -- this is
# now this module's ONLY VLM channel (task 038 v10 removed the doublet channel
# entirely, see the module docstring's (A)). Motivation (a real v9 cluster run):
# cell_00554's 5 rejects were all GOOD spindle cells FALSELY flagged `empty_debris`,
# because a genuinely thin/elongated single cell's own `foreground_fraction` is
# naturally low (< 0.2) without the cell being empty/debris at all -- the per-crop ask,
# judging ONE candidate in isolation, had no way to notice this. A montage lets the
# judge SEE many candidates AT ONCE and COMPARE them -- catching exactly this
# false-positive.
#
# Calls the ENGINE DIRECTLY (`engine.generate([prompt, montage_bytes],
# response_format=MontageVerdict)`), NOT `llm_recommend.judge_item`/`build_judge_prompt`
# -- human-locked layering: `llm_recommend.py` stays tool-AGNOSTIC, and those helpers are
# shaped around ONE item's own quant numbers/evidence, not a multi-item labeled grid, so
# reusing them would mean cramming montage-specific structure into a shared, supposedly
# tool-agnostic seam. This module builds its OWN prompt instead.
# --------------------------------------------------------------------------- #

_MONTAGE_CELL_PX = 160   # fixed per-cell thumbnail size (px) -- every montage is a
                        # uniform grid regardless of how different each crop's own
                        # native size is (a doublet crop vs a tiny debris candidate).


def _crop_identifier(crop: dict) -> str:
    """The STABLE identifier for a crop used in the montage's own index->crop_id map
    (task 038 montage round) -- `cell_id` (the tool's own stable per-cell id) when
    present, falling back to `crop_path`/`path` (still unique per crop)."""
    return crop.get("cell_id") or crop.get("crop_path") or crop.get("path") or ""


def _crop_overlay_array(crop: dict) -> np.ndarray | None:
    """Reuses `_overlay_target_on_crop` (task 038 v9's grounding fix) to draw a crop's
    OWN target mask on top, as a raw RGB array -- the per-cell image a montage grid
    pastes in (task 038 montage round). Falls back to a plain grayscale (converted to
    RGB, ungrounded) array when there is no readable mask -- graceful degrade. None only
    when the crop's OWN pixels cannot be read at all -- that crop is simply OMITTED from
    the montage (`_build_montage`), not a hard failure."""
    pixels = _crop_pixels(crop)
    if pixels is None:
        return None
    mask = _target_mask_pixels(crop)
    if mask is not None:
        try:
            return _overlay_target_on_crop(pixels, mask)
        except Exception as exc:
            logger.warning("crop-triage: overlay draw failed for a montage cell (%s)", exc)
    import cv2

    from shape.utils.segmentation_qc import to_display_uint8
    base = to_display_uint8(np.asarray(pixels))
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR) if base.ndim == 2 else np.asarray(base)[..., :3].copy()


def _label_montage_cell(overlay_rgb: np.ndarray, index: int) -> np.ndarray:
    """Resizes ONE crop's own overlay to the fixed per-cell thumbnail size and draws a
    clearly-legible INDEX label (a filled black box + a yellow number) in the top-left
    corner -- the SAME number `MontagePositionVerdict.index` reports back (task 038
    montage round)."""
    import cv2
    cell = cv2.resize(np.asarray(overlay_rgb), (_MONTAGE_CELL_PX, _MONTAGE_CELL_PX),
                      interpolation=cv2.INTER_AREA)
    label = str(index)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(cell, (0, 0), (tw + 8, th + baseline + 6), (0, 0, 0), -1)
    cv2.putText(cell, label, (4, th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return cell


def _build_montage(crops_chunk: list[dict], grid: tuple[int, int]) -> tuple[bytes | None, dict[int, dict]]:
    """Builds ONE labeled grid montage from up to `grid[0]*grid[1]` reject-candidate
    crops (task 038 montage round) -- each grid cell = that crop's own GROUNDED overlay
    (`_crop_overlay_array`, reusing task 038 v9's per-crop grounding fix) resized to a
    common thumbnail size, with a clearly-drawn INDEX label (`_label_montage_cell`).
    Records the STRICT index->crop MAP as each cell is placed (top-left to
    bottom-right, row-major) -- the judge reports verdicts BY INDEX
    (`MontagePositionVerdict.index`), and this map is what `_apply_montage_verdicts`
    uses to land each verdict back onto the right crop dict. A crop whose own pixels
    cannot be read is SKIPPED (not placed, not indexed) -- graceful degrade, mirrors
    the rest of this module's "any failure -> keep going" contract. Returns
    `(None, {})` when NOTHING in the chunk was readable at all."""
    import cv2

    rows, cols = grid
    cell_imgs: list[np.ndarray] = []
    index_to_crop: dict[int, dict] = {}
    for crop in crops_chunk:
        if len(cell_imgs) >= rows * cols:
            break
        overlay = _crop_overlay_array(crop)
        if overlay is None:
            continue
        index = len(cell_imgs)
        cell_imgs.append(_label_montage_cell(overlay, index))
        index_to_crop[index] = crop
    if not cell_imgs:
        return None, {}

    montage = np.zeros((rows * _MONTAGE_CELL_PX, cols * _MONTAGE_CELL_PX, 3), dtype=np.uint8)
    for idx, cell in enumerate(cell_imgs):
        r, c = divmod(idx, cols)
        montage[r * _MONTAGE_CELL_PX:(r + 1) * _MONTAGE_CELL_PX,
               c * _MONTAGE_CELL_PX:(c + 1) * _MONTAGE_CELL_PX] = cell

    ok, buf = cv2.imencode(".png", montage)
    if not ok:
        return None, {}
    return buf.tobytes(), index_to_crop


def _montage_position_detail(crop: dict) -> str:
    """The per-position evidence line for the montage prompt (task 038 montage round)
    -- names THIS crop's own flagged reject scenario(s) and their triggering value(s),
    the same numbers `_cut_off_confirm_text`/`_empty_debris_confirm_text` name for the
    single-crop path."""
    reasons = [r for r in _vlm_candidate_reasons(crop) if r in ("cut_off", "empty_debris")]
    detail = []
    if "cut_off" in reasons:
        border_touch = crop.get("border_touch")
        if border_touch is not None:
            detail.append(f"not_truncated={1.0 - float(border_touch):.3f} "
                          f"(< {_SEVERE_NOT_TRUNCATED_MAX} floor -> cut_off candidate)")
    if "empty_debris" in reasons:
        fg = (crop.get("qc") or {}).get("foreground_fraction")
        if fg is not None:
            detail.append(f"foreground_fraction={float(fg):.4f} "
                          f"(< {_EMPTY_DEBRIS_FOREGROUND_MAX} floor -> empty_debris candidate)")
    return "; ".join(detail) if detail else "; ".join(reasons)


def _build_montage_prompt(index_to_crop: dict[int, dict]) -> str:
    """Assembles the montage judge's OWN prompt (task 038 montage round) -- built
    directly in this module (see the section banner above for why it does not go
    through `llm_recommend.build_judge_prompt`). States the grid layout convention,
    the keep-biased RESCUE framing (the exact v9 cell_00554 false-positive this round
    fixes), each labeled position's own flagged reason(s)/value(s), and the real
    CARD.md text."""
    n = len(index_to_crop)
    positions = "\n".join(
        f"  Position {i}: crop_id={_crop_identifier(index_to_crop[i])!r} -- flagged for "
        f"{_montage_position_detail(index_to_crop[i])}"
        for i in sorted(index_to_crop))
    # CARD SIZE (token cost): carry only the judge's rule text (_cropper_judge_rules: purpose +
    # calibration + readout meanings, ~3.4k tok) rather than the whole CARD (~24.4k tok: levers,
    # loop settings, dev notes the mask-check judge does not use). The 4-way judge below already
    # does this; the three primary montage judges now match it.
    card = _cropper_judge_rules()
    # PROMPT-CACHE LAYOUT (token cost): the montage judge is called thousands of times across
    # the VLRR loop; most of each prompt is the FIXED instruction block + the (identical) tool
    # CARD rules. Put every byte that is identical call-to-call FIRST, as a stable prefix, and
    # the per-call variable part (the crop count + the position list) LAST. Prefix-caching
    # providers then bill the fixed instruction+CARD prefix at the cached rate on every repeat
    # call. Previously the leading "...MONTAGE of {n} candidates" put a varying number at byte
    # 58, breaking the prefix so the whole block was re-billed every call (measured ~1% cache
    # hit). Wording/semantics are unchanged -- only the order.
    card_section = (f"TOOL CARD -- this item's own verdict semantics, calibration, "
                    f"and complementarity rules:\n{card.strip()}\n\n" if card else "")
    fixed_prefix = (
        f"{card_section}"
        f"You are the POPULATION-TRIAGE judge for a GRID MONTAGE of single-cell "
        f"crop candidates (labeled positions, indexed 0..N-1, top-left to "
        f"bottom-right, row-major). EACH position's own TARGET cell is marked with a "
        f"COLORED overlay + a bright contour; the small yellow-on-black number in that "
        f"position's own top-left corner is its INDEX -- report each verdict against "
        f"THAT index (restate the crop_id you were told for it as a cross-check).\n\n"
        f"Each position was flagged by this tool's OWN mask data as a POSSIBLE defect "
        f"(named below, per position) -- but that data screen can be WRONG: a "
        f"genuinely thin/elongated single cell naturally has a LOW foreground_fraction "
        f"(looks 'empty' by the numeric floor alone) without actually being empty or "
        f"debris. SEEING ALL the positions together, and COMPARING them, lets you tell "
        f"a genuinely defective crop apart from a merely thin/oddly-shaped but "
        f"PERFECTLY FINE single cell -- catching exactly this false-positive is why "
        f"this montage exists.\n\n"
        f"For EACH labeled position, report:\n"
        f"  - problem: 'empty_debris' or 'cut_off' ONLY if the outlined target at that "
        f"position is GENUINELY that defect; 'none' if the outlined target is actually "
        f"a FINE, usable single cell that was wrongly flagged (a RESCUE).\n"
        f"  - severity: 'severe' (unmistakably bad), 'moderate' (borderline), or "
        f"'none' (whenever problem='none').\n"
        f"  - confidence: 'high' or 'low'.\n"
        f"KEEP-BIASED: default to problem='none' unless you are CLEARLY confident the "
        f"crop is genuinely unusable -- neither defect is fixable by this tool's own "
        f"geometry, so only a clear, unfixable defect should be flagged.\n\n"
    )
    return (
        f"{fixed_prefix}"
        f"THIS MONTAGE has {n} candidates (indices 0..{n - 1}).\n"
        f"Positions and their own flagged reason(s):\n{positions}"
    )


def _apply_montage_verdicts(montage_verdict, index_to_crop: dict[int, dict], *,
                            montage_path: str | None = None) -> list[dict]:
    """Maps each labeled position's verdict back onto its own crop dict, BY INDEX
    (task 038 montage round, human-specified landing rule) -- validates `crop_id`
    against the index's own crop and, on a MISMATCH, LOGS it and trusts the INDEX (the
    grid position is the ground truth -- it is what was actually drawn on the image the
    judge looked at; `crop_id` is only a cross-check the judge restates from the
    prompt's own text, not an independent identification). A position naming an
    UNKNOWN index (outside this montage) is skipped (logged) -- keep-biased, no crash.
    Sets `vision_reject`/`vision_problem` on a crop IN PLACE when `problem!='none' AND
    confidence=='high'` -- otherwise the crop is simply left un-rejected (RESCUED, if
    it was wrongly flagged by the data screen -- the exact v9 cell_00554 case this
    round fixes). Returns the per-position verdict-trace dicts, the SAME shape
    `llm_recommend.judge_items`'s own summary uses (`{item_id, reject, verdict,
    confidence, problem, reasoning}`) PLUS `montage_path` (task 038 v10, human-locked --
    the PERSISTED montage PNG this verdict's own index/crop_id maps back into, so a
    human/analysis script reading `write_triage_trace`'s sidecar can open the EXACT
    montage image the judge saw for this verdict; None when nothing was persisted, e.g.
    no `metadata_path`), for `write_triage_trace`."""
    verdicts = []
    for pos in montage_verdict.verdicts:
        crop = index_to_crop.get(pos.index)
        if crop is None:
            logger.warning("crop-triage: montage verdict named an unknown index %s; skipping",
                           pos.index)
            continue
        expected_id = _crop_identifier(crop)
        if pos.crop_id and pos.crop_id != expected_id:
            logger.warning(
                "crop-triage: montage verdict index %s crop_id mismatch (got %r, expected "
                "%r) -- trusting the index", pos.index, pos.crop_id, expected_id)
        # `problem in _VALID_REJECT_REASONS` (not just `!= "none"`) -- defense in depth,
        # mirroring `llm_recommend.judge_item`'s own `problem_gate` for the doublet
        # channel: `MontagePositionVerdict.problem`'s enum already excludes any
        # out-of-taxonomy value structurally, but a non-conforming engine that bypasses
        # pydantic validation (duck-typed response) could still name one -- this second,
        # runtime layer downgrades it to keep, same as the doublet channel's own gate.
        reject = pos.problem in _VALID_REJECT_REASONS and pos.confidence == "high"
        if reject:
            crop["vision_reject"] = True
            crop["vision_problem"] = pos.problem
        verdicts.append({
            "item_id": crop.get("crop_path"),
            "reject": reject,
            "verdict": "reject" if pos.problem != "none" else "keep",
            "confidence": pos.confidence,
            "problem": pos.problem,
            "reasoning": f"[montage index {pos.index}, severity={pos.severity}]",
            "montage_path": montage_path,
        })
    return verdicts


_DEFAULT_MONTAGE_GRID = (3, 3)   # human wants 2x2/3x3/4x4 tested -- CropVerifier's own
                                 # `triage_montage_grid` attribute overrides this.


def _montages_dir_for(metadata_path: str | None, *, prefix: str = "montages") -> str | None:
    """The per-trial montage output directory (task 038 v10, human-locked: "把 montage
    的图也都给保持" -- persist every montage actually sent, so a human can see exactly
    what the judge saw), derived from `metadata_path`'s own basename -- the SAME
    sidecar-naming idiom `llm_recommend.write_kept_manifest`/`write_triage_trace`
    already use (swap the `cell_crops_metadata_` prefix): `cell_crops_metadata_
    <hash>.json` -> `<tool_cache_dir>/<prefix>_<hash>/`. This keeps montages from a
    DIFFERENT trial/attempt for the SAME image (a different `metadata_path`, same
    `tool_cache_dir` across VLRR retune iterations) from colliding/overwriting each
    other -- a flat, un-keyed `montages/` folder would NOT have this property. `prefix`
    (task 038 v12, human-locked: "own subdir e.g. montages_tighten_<hash>/") lets the
    REJECT channel (`prefix="montages"`, the default, UNCHANGED from v10) and the
    directional-TIGHTEN channel (`prefix="montages_tighten"`) persist into SEPARATE
    subdirectories for the SAME trial, never overwriting each other's montages either.
    None when `metadata_path` is falsy or does not match the expected naming
    convention -- graceful degrade, no montages persisted (mirrors
    `write_kept_manifest`'s own naming guard)."""
    if not metadata_path:
        return None
    basename = os.path.basename(metadata_path)
    if "cell_crops_metadata_" not in basename:
        return None
    stem = basename.replace("cell_crops_metadata_", f"{prefix}_").rsplit(".", 1)[0]
    return os.path.join(os.path.dirname(metadata_path), stem)


def _save_montage_png(montages_dir: str, chunk_idx: int, montage_bytes: bytes) -> str | None:
    """Persists ONE montage's own PNG bytes to disk, exactly as SENT to the engine
    (task 038 v10, human-locked). Creates `montages_dir` lazily -- ONLY when there is
    genuinely a montage to save, so a no-candidate/no-engine run creates nothing at
    all. Filename (`montage_{chunk_idx:02d}.png`) is the SAME chunk index
    `_apply_montage_verdicts` records in each verdict's own `montage_path` field, so a
    verdict in the triage trace and its own source montage image are directly
    recoverable from each other. Any failure (permission error, disk full, ...) is
    logged and swallowed -- persisting evidence must never break judging itself, same
    "any failure -> keep going" contract as the rest of this module. Returns the path
    written, or None on failure."""
    try:
        os.makedirs(montages_dir, exist_ok=True)
        path = os.path.join(montages_dir, f"montage_{chunk_idx:02d}.png")
        with open(path, "wb") as fh:
            fh.write(montage_bytes)
        return path
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("crop-triage: failed to persist montage %d (%s)", chunk_idx, exc)
        return None


def _judge_reject_montages(engine_mm, reject_candidates: list[dict], *,
                           sample_cap: int | None = None,
                           montage_grid: tuple[int, int] = _DEFAULT_MONTAGE_GRID,
                           montages_dir: str | None = None,
                           verdict_cache: dict | None = None) -> dict:
    """Judges ALL reject candidates (cut_off/empty_debris, task 038 montage round) in
    BATCHED, labeled grid montages instead of one VLM call per candidate -- see the
    module docstring's (D) and this section's own banner for WHY. `sample_cap`
    (optional): an evenly-spaced deterministic sample of AT MOST `sample_cap`
    candidates is judged, mirroring `llm_recommend.judge_items`'s own sampling
    contract (reproduced here, not imported, so this stays a self-contained
    direct-to-engine call -- see the section banner for the layering reason).
    `montage_grid=(rows, cols)` (default 3x3): candidates are CHUNKED into montages of
    `rows*cols`, in order; a partial LAST montage is fine. `montages_dir` (task 038
    v10, optional -- from `_montages_dir_for`): when given, EVERY montage actually
    built and sent is ALSO persisted to disk (`_save_montage_png`) before the engine
    call, so the saved file reflects EXACTLY the bytes sent regardless of whether the
    engine call itself then succeeds or raises. Returns the SAME `{judged, rejected,
    engine, verdicts}` summary shape `llm_recommend.judge_items` returns, so
    `vision_recommend_crop` can consume it uniformly. No engine / no candidates -> a
    no-op summary, same graceful-degrade contract as the rest of this module (and
    persists nothing -- `montages_dir` is never even touched)."""
    if engine_mm is None or not reject_candidates:
        return {"judged": 0, "rejected": 0, "reused": 0,
                "engine": engine_mm is not None, "verdicts": []}

    candidates = reject_candidates
    # Cross-round cost saver (task 059): `verdict_cache` (a dict on the CropVerifier,
    # persistent across VLRR iterations) memoises GEOMETRY-INVARIANT rejects by the
    # crop's stable cell id. `empty_debris` (the target region is empty/debris -- there
    # is no real cell there) cannot be fixed by tightening/recentering the box, so once
    # the VLM says it we NEVER re-send that crop in a later round -- we re-apply the
    # cached reject and skip the montage. `cut_off` is NOT cached (recentering can fix
    # it), so it is always re-judged. Kept crops are never cached either (a geometry
    # change could make them cut_off), so correctness is unchanged -- only redundant,
    # provably-identical re-judgements of known-empty crops are eliminated.
    reused = 0
    if verdict_cache is not None:
        fresh = []
        for c in candidates:
            cached = verdict_cache.get(_crop_identifier(c))
            if cached:
                c["vision_reject"] = True
                c["vision_problem"] = cached["problem"]
                reused += 1
            else:
                fresh.append(c)
        candidates = fresh
    if not candidates:
        return {"judged": 0, "rejected": reused, "reused": reused,
                "engine": True, "verdicts": []}
    if sample_cap is not None and len(candidates) > sample_cap:
        step = len(candidates) / float(sample_cap)
        candidates = [candidates[int(i * step)] for i in range(sample_cap)]

    rows, cols = montage_grid
    chunk_size = max(1, rows * cols)
    judged = rejected = 0
    all_verdicts: list[dict] = []
    for start in range(0, len(candidates), chunk_size):
        chunk_idx = start // chunk_size
        chunk = candidates[start:start + chunk_size]
        montage_bytes, index_to_crop = _build_montage(chunk, montage_grid)
        if montage_bytes is None:
            continue
        montage_path = (_save_montage_png(montages_dir, chunk_idx, montage_bytes)
                        if montages_dir else None)
        prompt = _build_montage_prompt(index_to_crop)
        try:
            resp = engine_mm.generate([prompt, montage_bytes], response_format=MontageVerdict)
            rec = resp.get("content") if isinstance(resp, dict) else resp
            if not hasattr(rec, "verdicts"):
                continue
        except Exception as exc:  # pragma: no cover - defensive: triage must never break QC
            logger.warning("crop-triage: montage judging failed (%s); keeping this batch", exc)
            continue
        judged += len(index_to_crop)
        batch_verdicts = _apply_montage_verdicts(rec, index_to_crop, montage_path=montage_path)
        rejected += sum(1 for v in batch_verdicts if v["reject"])
        all_verdicts.extend(batch_verdicts)
        # Remember only the geometry-INVARIANT rejects (empty_debris) so later rounds
        # skip re-sending them. `_apply_montage_verdicts` set vision_reject/vision_problem
        # on each crop in place, so read them straight off the crop.
        if verdict_cache is not None:
            for crop in index_to_crop.values():
                if crop.get("vision_reject") and crop.get("vision_problem") == "empty_debris":
                    verdict_cache[_crop_identifier(crop)] = {"problem": "empty_debris"}
    return {"judged": judged, "rejected": rejected + reused, "reused": reused,
            "engine": True, "verdicts": all_verdicts}


# --------------------------------------------------------------------------- #
# Task 038 v13 (human-directed -- RETIRES the v12 R2 absolute-QC tighten gate above,
# `tighten_recommend_crop`/`TightenVerdict`, in favour of a BLINDED PAIRWISE A/B
# comparison): a methodology critique settled that `crop_quality_score` cannot be its
# own validator (v12 reweighted it toward the conclusion; even un-tuned, a mask-
# geometry score is not the task objective -- only downstream SSL can validate that,
# and SSL is a SEPARATE task, OUT of 038). Using the SAME score to both SELECT a
# tightened crop (keep-best) and PROVE it better is the 048 self-proving trap. v13's
# fix: DATA still drives the geometry (`tool.py::_compute_crop_qc`'s own
# `contamination_inset_px`, unchanged from R2); the VLM's job becomes a BLINDED A/B
# comparison of the tightened CANDIDATE against its own mask_bbox BASELINE (the
# un-inset box, `tool.py` renders both as REAL images -- no history threading needed)
# -- "which of these two keeps the target more complete while containing fewer other
# cells?" -- a genuinely comparative, harder-to-game question than an absolute
# "does this look good" QC ask. VoI-GATED (`verifiers/crop.py::CropVerifier.
# adjudicate`'s own round-2+ pass, NOT this module) -- only a cut whose outcome is
# genuinely uncertain from data (low retention, or floor-stuck still contaminated) is
# ever A/B'd; a comfortable, clean cut is accepted on data alone, no VLM call (the v7
# lesson: all-crops-to-VLM = 12.8M tokens, ~0 gain). L/R order is RANDOMIZED PER TILE
# (a position-bias guard -- a judge asked "left or right" repeatedly could otherwise
# learn a spurious position preference) -- the mapping is recorded per position so the
# judge's own left_better/right_better answer de-randomizes back into candidate/
# baseline. THE v6-HALLUCINATION GUARDRAIL still holds structurally: the schema can
# express only a LABEL (left_better/right_better/abstain), never a magnitude or a
# number -- DATA alone measures the cut depth and owns the hard retention floor,
# entirely outside this schema. Reuses the SAME `_crop_overlay_array`/
# `_save_montage_png` machinery (v10/v11's own colourful overlay + persistence apply
# unchanged to BOTH panels -- human-required: no overlay tanks VLM efficiency); only
# the TILE assembly (`_pair_tile`/`_build_ab_montage`), prompt
# (`_build_ab_montage_prompt`), and verdict schema (`ABVerdict`, above) are new.
# Montages persist to their OWN `montages_ab_<hash>/` subdirectory. Calls the ENGINE
# DIRECTLY, same layering reason as `_judge_reject_montages` (a multi-item labeled
# grid does not fit `llm_recommend.judge_item`'s own one-item shape).
# --------------------------------------------------------------------------- #

def _pair_tile(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Horizontally concatenates two overlay arrays into one [left|right] tile (task
    038 v13) -- pads the shorter of the two to the taller one's own height first (a
    directional inset SHRINKS the candidate relative to its own baseline, so a height
    mismatch is the COMMON case here, not an edge case), so `np.hstack` never raises."""
    import cv2

    h = max(left.shape[0], right.shape[0])

    def _pad(img: np.ndarray) -> np.ndarray:
        pad = h - img.shape[0]
        return img if pad <= 0 else cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return np.hstack([_pad(left), _pad(right)])


def _build_ab_montage(ab_candidates: list[dict], grid: tuple[int, int], rng) -> tuple[bytes | None, dict[int, dict]]:
    """Builds ONE labeled grid montage where EACH tile is a RANDOMIZED [left|right]
    side-by-side pair -- one cell's own mask_bbox BASELINE crop vs. its tightened
    CANDIDATE, BOTH grounded with the SAME colourful overlay `_crop_overlay_array`
    already draws (reused verbatim on a small pseudo-crop dict pointing at the
    baseline's own `crop_path`/`mask_path` pair -- `tool.py` renders the baseline
    with its OWN correctly-sized mask specifically so this reuse works without a
    shape-mismatch fallback to no overlay). `rng` (a `random.Random`-shaped object,
    injected so tests can force a deterministic side) decides, PER TILE, whether the
    candidate lands left or right -- recorded in the returned index map
    (`{"crop": crop, "candidate_side": "left"|"right"}`) so `_apply_ab_verdicts` can
    de-randomize the judge's own left_better/right_better answer. A cell whose
    baseline OR candidate pixels cannot be read (no `baseline_crop_path`, unreadable
    file) is SKIPPED entirely -- a one-sided tile would be meaningless. Returns
    `(None, {})` when nothing in the chunk was pairable at all."""
    rows, cols = grid
    cell_imgs: list[np.ndarray] = []
    index_to_pair: dict[int, dict] = {}
    for crop in ab_candidates:
        if len(cell_imgs) >= rows * cols:
            break
        baseline_pseudo = {"crop_path": crop.get("baseline_crop_path"),
                           "mask_path": crop.get("baseline_mask_path")}
        candidate_overlay = _crop_overlay_array(crop)
        baseline_overlay = _crop_overlay_array(baseline_pseudo)
        if candidate_overlay is None or baseline_overlay is None:
            continue
        candidate_side = "left" if rng.random() < 0.5 else "right"
        left, right = ((candidate_overlay, baseline_overlay) if candidate_side == "left"
                      else (baseline_overlay, candidate_overlay))
        index = len(cell_imgs)
        cell_imgs.append(_label_montage_cell(_pair_tile(left, right), index))
        index_to_pair[index] = {"crop": crop, "candidate_side": candidate_side}
    if not cell_imgs:
        return None, {}

    import cv2
    montage = np.zeros((rows * _MONTAGE_CELL_PX, cols * _MONTAGE_CELL_PX, 3), dtype=np.uint8)
    for idx, cell in enumerate(cell_imgs):
        r, c = divmod(idx, cols)
        montage[r * _MONTAGE_CELL_PX:(r + 1) * _MONTAGE_CELL_PX,
               c * _MONTAGE_CELL_PX:(c + 1) * _MONTAGE_CELL_PX] = cell
    ok, buf = cv2.imencode(".png", montage)
    if not ok:
        return None, {}
    return buf.tobytes(), index_to_pair


def _build_ab_montage_prompt(index_to_pair: dict[int, dict]) -> str:
    """Assembles the blinded A/B judge's OWN prompt (task 038 v13, human-directed) -- a
    NARRATIVE, COMPARATIVE ask grounded on the SAME colourful overlay `_build_montage`
    already draws, applied to BOTH panels of each tile. States the grid + PAIR-TILE
    layout convention (random L/R order, explicitly called out so the judge does not
    assume a fixed side), the label-only framing (never a magnitude), each labeled
    position's own identity, and the real CARD.md text."""
    n = len(index_to_pair)
    positions = "\n".join(
        f"  Position {i}: crop_id={_crop_identifier(index_to_pair[i]['crop'])!r}"
        for i in sorted(index_to_pair))
    card = _cropper_judge_rules()   # judge rule text only (~3.4k tok), not the full CARD
    # PROMPT-CACHE LAYOUT: fixed instructions + CARD first (stable prefix), the varying
    # pair count + position list last. See _build_montage_prompt for the full rationale.
    card_section = (f"TOOL CARD -- this item's own verdict semantics, calibration, "
                    f"and complementarity rules:\n{card.strip()}\n\n" if card else "")
    fixed_prefix = (
        f"{card_section}"
        f"You are comparing PAIRS of single-cell crops in a GRID MONTAGE of pairs "
        f"(labeled positions, indexed 0..N-1, top-left to bottom-right, row-major). "
        f"EACH position shows TWO crops SIDE BY SIDE (a LEFT half and a RIGHT half) of "
        f"the SAME cell -- the coloured cell (a COLORED overlay + bright contour) IS "
        f"the target in BOTH halves; anything else visible is background or another "
        f"cell. The order (which crop is on the left vs. the right) is RANDOM and "
        f"DIFFERENT for every position -- do not assume a fixed side means anything. "
        f"The small yellow-on-black number in each position's own top-left corner is "
        f"its INDEX -- report each verdict against THAT index (restate the crop_id you "
        f"were told for it as a cross-check).\n\n"
        f"For EACH labeled position, compare the LEFT and RIGHT panels and answer: "
        f"which panel keeps the coloured target MORE COMPLETE (not clipped/cut off) "
        f"while containing FEWER other cells?\n\n"
        f"Report verdict:\n"
        f"  - 'left_better': the LEFT panel is the better crop.\n"
        f"  - 'right_better': the RIGHT panel is the better crop.\n"
        f"  - 'abstain': not clearly distinguishable -- genuinely too close to call.\n\n"
        f"You are reporting a LABEL ONLY -- do NOT estimate how much of either crop is "
        f"target vs. neighbour, do NOT report any number or percentage. Exactly HOW "
        f"MUCH was cut, and whether it was safe to do so at all, was decided separately "
        f"from measured data; your answer is never allowed to override that safety "
        f"floor.\n\n"
    )
    return (
        f"{fixed_prefix}"
        f"THIS MONTAGE has {n} pairs (indices 0..{n - 1}).\n"
        f"Positions:\n{positions}"
    )


def _apply_ab_verdicts(ab_verdict, index_to_pair: dict[int, dict], *,
                       montage_path: str | None = None) -> list[dict]:
    """Maps each labeled position's verdict back onto its own crop dict, BY INDEX (task
    038 v13 -- the SAME landing rule `_apply_montage_verdicts` uses: `crop_id`
    mismatch is logged and the INDEX still wins), DE-RANDOMIZING the judge's own
    left_better/right_better answer through THIS position's own `candidate_side`
    mapping first. Sets `crop["ab_verdict"]` IN PLACE to `"candidate"` (the judge
    preferred the tightened crop), `"baseline"` (preferred the plain mask_bbox crop),
    or `"abstain"` -- `verifiers/crop.py::CropVerifier.adjudicate` reads this back to
    decide whether to commit the next data-computed cut (`"candidate"`) or revert to
    baseline + freeze (`"baseline"`/`"abstain"`). A position naming an UNKNOWN index is
    skipped (logged), no crash. Returns the per-position verdict-trace dicts
    (`{item_id, verdict, raw_verdict, confidence, montage_path}`) -- `raw_verdict`
    keeps the judge's own un-de-randomized answer for audit."""
    verdicts = []
    for pos in ab_verdict.verdicts:
        pair = index_to_pair.get(pos.index)
        if pair is None:
            logger.warning("crop-triage: A/B-montage verdict named an unknown index "
                           "%s; skipping", pos.index)
            continue
        crop = pair["crop"]
        expected_id = _crop_identifier(crop)
        if pos.crop_id and pos.crop_id != expected_id:
            logger.warning(
                "crop-triage: A/B-montage verdict index %s crop_id mismatch (got "
                "%r, expected %r) -- trusting the index", pos.index, pos.crop_id, expected_id)
        if pos.verdict == "abstain":
            resolved = "abstain"
        else:
            candidate_won = ((pos.verdict == "left_better" and pair["candidate_side"] == "left")
                             or (pos.verdict == "right_better" and pair["candidate_side"] == "right"))
            resolved = "candidate" if candidate_won else "baseline"
        crop["ab_verdict"] = resolved
        verdicts.append({
            "item_id": crop.get("crop_path"),
            "verdict": resolved,
            "raw_verdict": pos.verdict,
            "confidence": pos.confidence,
            "montage_path": montage_path,
        })
    return verdicts


def ab_recommend_crop(engine_mm, ab_candidates: list[dict], *,
                      sample_cap: int | None = None,
                      metadata_path: str | None = None,
                      montage_grid: tuple[int, int] = _DEFAULT_MONTAGE_GRID,
                      rng=None) -> dict:
    """The per-round, VoI-GATED blinded A/B comparison gate for the tighten loop (task
    038 v13, human-directed -- RETIRES `tighten_recommend_crop`) --
    `verifiers/crop.py::CropVerifier.adjudicate`'s own round-2+ pass has ALREADY
    data-screened `ab_candidates` down to the UNCERTAIN subset (low retention, or
    floor-stuck still contaminated -- comfortable cuts never reach this function at
    all); this function is the VLM's comparative say on that screened set, batched
    into labeled PAIR-TILE grid montages (`_build_ab_montage`, reusing the SAME
    `_crop_overlay_array`/`_save_montage_png` machinery the reject/tighten channels
    already use -- montages persist under `_montages_dir_for(metadata_path,
    prefix="montages_ab")`, a SEPARATE subdirectory from every other channel's own
    montages for the same trial).

    Sets `crop["ab_verdict"]` IN PLACE on each judged candidate (`_apply_ab_verdicts`,
    already de-randomized) -- the CALLER reads it back to decide whether to commit the
    next data-computed cut (`"candidate"`) or revert to baseline + freeze
    (`"baseline"`/`"abstain"`); a candidate this function never judges (no engine, no
    `baseline_crop_path` to pair against, sampled out, or a failed call) is simply left
    with NO `ab_verdict` key at all, so the caller's own data-only fallback naturally
    applies (headless -> pure data, accept every constraint-satisfying cut).

    No engine / no candidates -> a no-op (nothing annotated, nothing persisted) --
    identical in shape to `_judge_reject_montages`'s own graceful-degrade contract.
    Returns `{"judged": int, "verdicts": [...]}` (no "rejected"/"engine" keys -- this
    channel does not reject anything itself, it only gates a GEOMETRY proposal the
    caller owns end to end)."""
    if engine_mm is None or not ab_candidates:
        return {"judged": 0, "verdicts": []}

    import random
    rng = rng or random.Random()

    montages_dir = _montages_dir_for(metadata_path, prefix="montages_ab")
    candidates = ab_candidates
    if sample_cap is not None and len(candidates) > sample_cap:
        step = len(candidates) / float(sample_cap)
        candidates = [candidates[int(i * step)] for i in range(sample_cap)]

    rows, cols = montage_grid
    chunk_size = max(1, rows * cols)
    judged = 0
    all_verdicts: list[dict] = []
    for start in range(0, len(candidates), chunk_size):
        chunk_idx = start // chunk_size
        chunk = candidates[start:start + chunk_size]
        try:
            # `ab_recommend_crop` is called DIRECTLY from `verifiers/crop.py::adjudicate`
            # with no outer adapter wrapping it (unlike `_judge_reject_montages`, which
            # `vision_recommend_crop` itself guards) -- so the try/except here must
            # cover montage-BUILDING too, not only the engine call, to uphold the SAME
            # "triage must never break QC" contract (the exact gap task 038 v12 R2's own
            # own failure-injection test found and fixed for the retired tighten gate).
            montage_bytes, index_to_pair = _build_ab_montage(chunk, montage_grid, rng)
            if montage_bytes is None:
                continue
            montage_path = (_save_montage_png(montages_dir, chunk_idx, montage_bytes)
                            if montages_dir else None)
            prompt = _build_ab_montage_prompt(index_to_pair)
            resp = engine_mm.generate([prompt, montage_bytes], response_format=ABVerdict)
            rec = resp.get("content") if isinstance(resp, dict) else resp
            if not hasattr(rec, "verdicts"):
                continue
        except Exception as exc:  # pragma: no cover - defensive: triage must never break QC
            logger.warning("crop-triage: A/B-montage judging failed (%s); keeping "
                           "this batch", exc)
            continue
        judged += len(index_to_pair)
        all_verdicts.extend(_apply_ab_verdicts(rec, index_to_pair, montage_path=montage_path))
    return {"judged": judged, "verdicts": all_verdicts}


# --------------------------------------------------------------------------- #
# Task 038 Round B, S1 (coordinator-directed, evidence-driven): the human flagged 5
# kept crops (cell_00299/0037, cell_00994/0001, /0011, /0068, /0323) as obviously-
# should-be-discarded -- re-examined in the CORRECT channel (Round C's max-projection
# fix) with the CORRECT mask (the ab352fa lattice fix), they share ONE pattern: the
# mask CLAIMS a cell, and the image at that location is EMPTY -- the visible cell
# material sits OUTSIDE the target mask. Their mask-derived metrics look fine
# (contamination 0.10-0.26, retention 0.90-1.0) -- NO mask-derived metric can ever
# catch this, since the mask is the thing that is wrong. Three cheap data-only
# detectors were tried (mean contrast, boundary adherence, intensity histogram) and
# ALL THREE failed (modality-confounded, density-confounded, and simply missed
# cell_0323 outright). This is the one job that is provably the VLM's -- S1 asks it
# ONE narrative question: is there really a cell under the coloured region?
#
# The DATA SCREEN deciding which crops are even worth asking (`verifiers/
# crop.py::_s1_target_region_signal_poor`, DELIBERATELY LOOSE -- a false candidate
# costs one montage slot, a missed one ships a garbage crop straight through) lives
# in `verifiers/crop.py`, not here -- mirrors the A/B gate's own VoI screen living in
# the VERIFIER (data-driven candidate selection is the verifier's job; this module
# owns the montage/prompt/schema machinery for ANY channel that needs it). Reuses
# the SAME `_build_montage`/`_crop_overlay_array`/`_save_montage_png` machinery
# every other channel already uses (not reimplemented); montages persist to their
# OWN `montages_s1_<hash>/` subdirectory (mirrors `montages_ab_<hash>/`).
#
# LANDING is a CONSTRAINT, never a score (mirrors REJECT's own framing): confirmed
# `target_region_empty` -> the crop is REJECTED (sets `crop["vision_reject"]`, the
# SAME flag REJECT/`crop_kept` already read -- S1 does not need its own separate
# "kept" concept, it is simply ANOTHER reason a crop can fail the same population
# gate). `cell_present` -> kept (unchanged). `abstain`, an unmentioned position, or
# ANY failure (no engine, montage-build error, engine exception) -> KEPT -- the
# conservative default every other channel in this module already uses, never
# discard on silence.
# --------------------------------------------------------------------------- #

def _build_s1_montage_prompt(index_to_crop: dict[int, dict]) -> str:
    """Assembles the S1 judge's own prompt (task 038 Round B) -- states the grid
    layout convention (mirrors `_build_montage_prompt`'s own framing exactly) and
    asks ONE narrative question per labeled position: is there a real cell under the
    coloured region, or is that region empty/background/noise -- INCLUDING the case
    where real cell material IS visible in the tile but sits OUTSIDE the coloured
    region (the coordinator's own exact wording, load-bearing: a real S1 VLM run
    must be told this explicitly, in the narrative, not only via the schema field's
    own description -- [[judge_prompt_narrative_not_schema_field]])."""
    n = len(index_to_crop)
    positions = "\n".join(
        f"  Position {i}: crop_id={_crop_identifier(index_to_crop[i])!r}"
        for i in sorted(index_to_crop))
    card = _cropper_judge_rules()   # judge rule text only (~3.4k tok), not the full CARD
    # PROMPT-CACHE LAYOUT: fixed instructions + CARD first (stable prefix), the varying
    # candidate count + position list last. See _build_montage_prompt for the full rationale.
    card_section = (f"TOOL CARD -- this item's own verdict semantics, calibration, "
                    f"and complementarity rules:\n{card.strip()}\n\n" if card else "")
    fixed_prefix = (
        f"{card_section}"
        f"You are checking whether a segmentation MASK actually points at a real cell, "
        f"for a GRID MONTAGE of single-cell crop candidates (labeled positions, "
        f"indexed 0..N-1, top-left to bottom-right, row-major). EACH position's own "
        f"target region is marked with a COLORED overlay + a bright contour; the small "
        f"yellow-on-black number in that position's own top-left corner is its INDEX -- "
        f"report each verdict against THAT index (restate the crop_id you were told for "
        f"it as a cross-check).\n\n"
        f"The coloured region is where the segmentation says the target cell is. For "
        f"EACH labeled position, look at the IMAGE UNDER the colour: is there actually "
        f"a cell there, or is that region empty / background / noise? If you can see "
        f"cell material in this tile but it lies OUTSIDE the coloured region, that "
        f"STILL counts as target_region_empty -- the mask is pointing at the wrong "
        f"place, which is exactly the defect this check exists to catch.\n\n"
        f"For EACH labeled position, report:\n"
        f"  - verdict: 'cell_present' if the coloured region genuinely contains cell "
        f"material; 'target_region_empty' if that region is empty/background/noise OR "
        f"if the real cell material lies outside it; 'abstain' only if genuinely too "
        f"ambiguous to call either way.\n"
        f"  - confidence: 'high' or 'low'.\n\n"
    )
    return (
        f"{fixed_prefix}"
        f"THIS MONTAGE has {n} candidates (indices 0..{n - 1}).\n"
        f"Positions:\n{positions}"
    )


def _apply_s1_verdicts(s1_verdict, index_to_crop: dict[int, dict], *,
                       montage_path: str | None = None) -> list[dict]:
    """Maps each labeled position's verdict back onto its own crop dict, BY INDEX
    (task 038 Round B -- the SAME landing rule `_apply_montage_verdicts`/`_apply_ab_
    verdicts` use: a `crop_id` mismatch is logged and the INDEX still wins). Sets
    `crop["vision_reject"]`/`crop["vision_problem"]` IN PLACE -- the SAME flag the
    REJECT channel (`_apply_montage_verdicts`) already sets, so `crop_kept`/
    `_rewrite_kept_manifest`'s existing `not vision_reject` predicate picks up an
    S1 rejection with NO changes needed to either -- ONLY when `verdict ==
    'target_region_empty'` (regardless of confidence -- the coordinator's own
    landing spec names no confidence qualifier for this channel, unlike REJECT's
    `confidence=='high'` gate; taken at face value, not silently added). A position
    naming an UNKNOWN index is skipped (logged), no crash. Returns the per-position
    verdict-trace dicts, the SAME shape the other channels' own trace entries use."""
    verdicts = []
    for pos in s1_verdict.verdicts:
        crop = index_to_crop.get(pos.index)
        if crop is None:
            logger.warning("crop-triage: S1-montage verdict named an unknown index "
                           "%s; skipping", pos.index)
            continue
        expected_id = _crop_identifier(crop)
        if pos.crop_id and pos.crop_id != expected_id:
            logger.warning(
                "crop-triage: S1-montage verdict index %s crop_id mismatch (got %r, "
                "expected %r) -- trusting the index", pos.index, pos.crop_id, expected_id)
        reject = pos.verdict == "target_region_empty"
        if reject:
            crop["vision_reject"] = True
            crop["vision_problem"] = "target_region_empty"
        verdicts.append({
            "item_id": crop.get("crop_path"),
            "reject": reject,
            "verdict": pos.verdict,
            "confidence": pos.confidence,
            "problem": "target_region_empty" if reject else "none",
            "reasoning": f"[S1 montage index {pos.index}]",
            "montage_path": montage_path,
        })
    return verdicts


def s1_recommend_crop(engine_mm, s1_candidates: list[dict], *,
                      sample_cap: int | None = None,
                      metadata_path: str | None = None,
                      montage_grid: tuple[int, int] = _DEFAULT_MONTAGE_GRID) -> dict:
    """The S1 channel (task 038 Round B, coordinator-directed): judges `s1_candidates`
    (already data-screened, SUSPICION-SORTED, and VoI-CAPPED by the CALLER,
    `verifiers/crop.py::CropVerifier.adjudicate` -- see the section banner above and
    `CropVerifier.s1_candidate_cap`'s own docstring, task 038 Round D) in BATCHED,
    labeled grid montages, the SAME `_build_montage` machinery every other channel
    uses -- NOT one call per crop. `sample_cap` (optional, default None): a FURTHER,
    generic evenly-spaced deterministic reduction (the SAME sampling contract
    `_judge_reject_montages`/`ab_recommend_crop` already use) -- NOT the primary VoI
    cap for S1 any more (task 038 Round D moved that to the caller, so it can sort
    by suspicion BEFORE capping; a real v17 run found the OLD unsorted cap here
    discarding the single MOST suspicious of the human's own 5 named crops). Most
    callers should leave this None and let `adjudicate`'s own suspicion-ordered cap
    be the only reduction.

    Sets `crop["vision_reject"]`/`crop["vision_problem"]` IN PLACE on a confirmed
    `target_region_empty` position (`_apply_s1_verdicts`) -- the caller's LATER
    `vision_recommend_crop` call (which runs the (B) REJECT channel and THEN writes
    the shared kept-crops manifest + triage trace over the FULL population) picks
    this up automatically as long as this function runs FIRST, with no changes
    needed to the manifest-writing predicate itself. Does NOT write its own kept-
    manifest or triage-trace sidecar (mirrors `ab_recommend_crop`'s own precedent --
    those sidecars are single, shared files keyed off `metadata_path`; writing a
    second one here would just overwrite the REJECT channel's own SAME-named file,
    not merge with it). Returns `{"judged": int, "rejected": int, "engine": bool,
    "verdicts": [...]}`, the SAME shape `_judge_reject_montages` returns, so a
    caller (`vision_recommend_crop`'s own `extra_verdicts` parameter) can fold
    `["verdicts"]` into the SAME triage-trace write REJECT already performs.

    No engine / no candidates -> a no-op summary, same graceful-degrade contract as
    every other channel in this module."""
    if engine_mm is None or not s1_candidates:
        return {"judged": 0, "rejected": 0, "engine": engine_mm is not None, "verdicts": []}

    candidates = s1_candidates
    if sample_cap is not None and len(candidates) > sample_cap:
        step = len(candidates) / float(sample_cap)
        candidates = [candidates[int(i * step)] for i in range(sample_cap)]

    montages_dir = _montages_dir_for(metadata_path, prefix="montages_s1")
    rows, cols = montage_grid
    chunk_size = max(1, rows * cols)
    judged = rejected = 0
    all_verdicts: list[dict] = []
    for start in range(0, len(candidates), chunk_size):
        chunk_idx = start // chunk_size
        chunk = candidates[start:start + chunk_size]
        try:
            montage_bytes, index_to_crop = _build_montage(chunk, montage_grid)
            if montage_bytes is None:
                continue
            montage_path = (_save_montage_png(montages_dir, chunk_idx, montage_bytes)
                            if montages_dir else None)
            prompt = _build_s1_montage_prompt(index_to_crop)
            resp = engine_mm.generate([prompt, montage_bytes], response_format=S1Verdict)
            rec = resp.get("content") if isinstance(resp, dict) else resp
            if not hasattr(rec, "verdicts"):
                continue
        except Exception as exc:  # pragma: no cover - defensive: triage must never break QC
            logger.warning("crop-triage: S1-montage judging failed (%s); keeping this batch", exc)
            continue
        judged += len(index_to_crop)
        batch_verdicts = _apply_s1_verdicts(rec, index_to_crop, montage_path=montage_path)
        rejected += sum(1 for v in batch_verdicts if v["reject"])
        all_verdicts.extend(batch_verdicts)
    return {"judged": judged, "rejected": rejected, "engine": True, "verdicts": all_verdicts}


# --------------------------------------------------------------------------- #
# Task 058 M3 (human-authorized controller change, 2026-07-16): MASK CHECK -- the
# MASK-ERROR VLM channel, promoted from a benchmark-only probe/eval instrument
# (the cropper benchmark's own S1/A-C probe) to a FIRST-CLASS production
# channel that REPLACES S1's own reject role above (`s1_recommend_crop` stays in
# this file for other callers/tests, but `verifiers/crop.py::CropVerifier.adjudicate`
# no longer calls it -- it calls `mask_check_recommend_crop` on the SAME data-screened
# candidates instead). The task 038 M2 pilot (this task's own inherited context,
# BRIEF.md) measured the mask check's raw-image judgment catching merged/split/false-positive
# Cellpose mask errors (73%/51% precision in the hard/medium AP50 bands) that the
# mask-derived data screen is STRUCTURALLY blind to (a merged cell reads
# `neighbor_contamination=0` -- "perfect single cell" -- from the mask alone).
#
# CONTOUR-ONLY RENDERING (human directive 2026-07-16, ported byte-for-byte from
# the cropper probe's `_thin_contour_overlay`, NOT `_overlay_target_on_crop` above):
# a fixed, OPAQUE, saturated-green 1px contour marks the target's own mask edge, with
# NO interior fill at all (`label2rgb`'s own fill blend called at alpha=0.0, a
# mathematical no-op) -- the crop's own cell must stay fully, un-tinted visible so the
# judge can tell a genuinely coherent single cell from a doublet/wrong object. Opaque
# (not alpha-blended toward white, unlike the REJECT/A-B channels' own contour) because
# grayscale microscopy carries zero saturation of its own -- a saturated hue never
# washes into either a near-black or a near-white background, while a blended contour
# (this rendering's own first, REJECTED attempt in that probe) vanishes on a
# tile whose background happens to sit near the blend target.
#
# MASK-DERIVED CONTRAST STRETCH (ported from the SAME probe's own
# `_target_window_diagnostics`/`_apply_window` -- LOAD-BEARING, not an optional
# refinement, for THIS channel specifically): every crop `tool.py` delivers is
# ALREADY uint8 (`tool.py::normalize_source_image` stretches the whole SOURCE image
# once, before cropping), so `_overlay_target_on_crop`'s own `to_display_uint8` call
# is a NO-OP on it -- a crop from a genuinely dim region of an already-globally-
# stretched image still renders dim/black without its OWN, per-tile stretch. The mask check's
# candidates INCLUDE the data-screened DIM crops (task 058 M4's `_gate_candidate`
# three-arm OR gate, dim being one of the three arms -- `verifiers/crop.py`, module
# comment above `_GATE_DIM_FRAC`), so reusing the OTHER channels' plain overlay
# machinery unmodified would ship a broken instrument for a large share of its own real
# candidates -- the exact failure the probe's own repair round measured and fixed
# (wrong_object 75%->44%, still-dark 42%->28% on the SAME crops). The percentile
# WINDOW is derived from the pixels INSIDE the target mask (falling back to a padded
# bbox, then the whole tile, on too few mask pixels for a stable estimate), then
# APPLIED to the WHOLE tile -- so a genuinely dim target is revealed even when a
# bright, unrelated neighbour elsewhere in the crop would otherwise dominate a
# whole-tile stretch.
#
# CAPPED, PADDED UPSCALE (ported from the SAME probe's own `_resize_capped_padded`,
# FIX 2): real delivered crop sizes are commonly tiny (8x12, 21x21, 28x13, 44x44 px,
# per this module's own S1-round docstring above) -- the OTHER channels' shared
# `_label_montage_cell` force-resizes to fill the full 160px montage cell regardless
# of native size, a >=10x interpolated upscale on these that the probe's own
# audit found driving spurious `wrong_object` over-calls. The mask check instead caps the
# upscale at `_MASK_CHECK_MAX_MONTAGE_UPSCALE`x with NEAREST-neighbour interpolation (every
# displayed pixel a REAL source value, never a blended guess) and pads the remainder
# with a neutral gray, distinct from both the montage canvas's own black and the
# contour's own bright green.
#
# VERDICT SCHEMA (richer than S1, subsumes it): `cell_present` / `target_region_empty`
# / `wrong_object` (a DIFFERENT real cell than the one the contour intends, or the
# contour straddles a neighbour) / `merged` (a doublet -- TWO touching cells under the
# ONE outline, an under-segmentation the mask-only channels are structurally blind to)
# / `abstain`. `wrong_object` and `merged` are SEPARATE verdicts (not folded together)
# -- they name two different, actionable mask-error MODES a human/the trace should be
# able to tell apart, mirroring the GT-adjudication categories (`merged`/`split`/
# `false_positive`) this channel exists to catch in production, GT-free.
#
# NO GT in the controller (CARD.md's own "no self-scoring" mandate, applied here too)
# -- the mask check makes the keep/reject call from the image alone; GT adjudication of the
# call it made stays BENCHMARK-only (the cropper benchmark's own probe).
# --------------------------------------------------------------------------- #

_MASK_CHECK_MIN_WINDOW_PIXELS = 100   # ported from the cropper probe's _MIN_WINDOW_PIXELS --
                                 # below this many pixels a percentile estimate is unstable.
_MASK_CHECK_STRETCH_RING_PX = 2       # ported from the cropper probe's _STRETCH_RING_PX -- the
                                 # bbox-ring fallback padding when the mask itself is too small.
_MASK_CHECK_CONTOUR_COLOR_BGR = (0, 255, 0)   # pure green, cv2-native BGR order -- ported from
                                         # the cropper probe's _PROBE_CONTOUR_COLOR_BGR.
_MASK_CHECK_MAX_MONTAGE_UPSCALE = 3.0   # ported from the cropper probe's _MAX_MONTAGE_UPSCALE.
_MASK_CHECK_MONTAGE_PAD_VALUE = 96      # ported from the cropper probe's _MONTAGE_PAD_VALUE.


def _mask_check_percentile_window(values: np.ndarray) -> tuple[float, float] | None:
    """1st/99th-percentile window of `values` -- the SAME degenerate-handling contract
    `shape.utils.segmentation_qc.normalize_for_overlay` uses (min/max fallback, then
    None on a genuinely constant input). Ported from
    the cropper probe's `_percentile_window`."""
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return None
    return float(lo), float(hi)


def _mask_check_mask_bbox_ring_pixels(pixels: np.ndarray, mask_bool: np.ndarray, ring_px: int) -> np.ndarray:
    """The target mask's own bounding box, padded by `ring_px` on every side and
    clipped to the tile's own bounds -- the FALLBACK window source region used only
    when the mask itself has too few pixels for a stable percentile estimate. Ported
    from the cropper probe's `_mask_bbox_ring_pixels`."""
    rows = np.where(np.any(mask_bool, axis=1))[0]
    cols = np.where(np.any(mask_bool, axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    r0 = max(0, r0 - ring_px)
    r1 = min(pixels.shape[0] - 1, r1 + ring_px)
    c0 = max(0, c0 - ring_px)
    c1 = min(pixels.shape[1] - 1, c1 + ring_px)
    return pixels[r0:r1 + 1, c0:c1 + 1]


def _mask_check_stretched_pixels(pixels: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """The mask check's own per-tile contrast stretch (see the section banner above for why
    this is load-bearing, not optional, for this channel) -- ported from
    the cropper probe's `_target_window_diagnostics`/`_apply_window`, minus that
    probe's own diagnostic-only bookkeeping (`stretch_window_source` etc., useful for
    a probe's own reporting, not needed by a production judge). Falls back to the
    window source's own min/max, then to flat black, on a genuinely degenerate
    (constant) region -- `normalize_for_overlay`'s own tested degenerate contract,
    reused via `_mask_check_percentile_window`."""
    mask_pixels = pixels[mask_bool]
    window_pixels = mask_pixels.reshape(-1)
    if window_pixels.size < _MASK_CHECK_MIN_WINDOW_PIXELS:
        ring_pixels = _mask_check_mask_bbox_ring_pixels(
            pixels, mask_bool, _MASK_CHECK_STRETCH_RING_PX).reshape(-1)
        window_pixels = (ring_pixels if ring_pixels.size >= _MASK_CHECK_MIN_WINDOW_PIXELS
                         else pixels.reshape(-1))
    window = _mask_check_percentile_window(window_pixels)
    if window is None:
        window = _mask_check_percentile_window(pixels.reshape(-1))
    if window is None:
        return np.zeros(pixels.shape, dtype=np.uint8)
    lo, hi = window
    arr = np.asarray(pixels, dtype=np.float32)
    return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _mask_check_thin_contour_overlay(crop_pixels: np.ndarray, mask_pixels: np.ndarray) -> np.ndarray:
    """Draws the target's own mask edge as a fixed, OPAQUE, saturated-green 1px
    contour -- NO interior fill. A byte-for-byte port of
    the cropper probe's `_thin_contour_overlay` (see the section banner above for the
    contour-color rationale); `crop_pixels` is expected ALREADY STRETCHED
    (`_mask_check_stretched_pixels`) -- `to_display_uint8` here is only a dtype safety net
    (a no-op on an already-uint8 array). Returns a cv2-native (BGR-ordered) uint8
    array, the SAME contract `_overlay_target_on_crop` uses."""
    import cv2
    from skimage.color import label2rgb

    from shape.utils.segmentation_qc import to_display_uint8

    base = to_display_uint8(np.asarray(crop_pixels))
    m = (np.asarray(mask_pixels) > 0)
    if m.shape[:2] != base.shape[:2]:
        return (cv2.cvtColor(base, cv2.COLOR_GRAY2BGR) if base.ndim == 2
                else np.asarray(base)[..., :3].copy())
    colored = label2rgb(m.astype(np.int32), image=base, bg_label=0, alpha=0.0, image_alpha=1.0)
    tint = cv2.cvtColor((np.clip(colored, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros(tint.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, 1)   # 1px -- cv2's own floor
    tint[contour_mask > 0] = _MASK_CHECK_CONTOUR_COLOR_BGR
    return tint


def _mask_check_overlay_array(crop: dict) -> np.ndarray | None:
    """The mask check's own per-crop montage-cell image: reuses `_crop_pixels`/
    `_target_mask_pixels` VERBATIM, inserts the mask-derived contrast stretch
    (`_mask_check_stretched_pixels`), then draws the CONTOUR-ONLY mark
    (`_mask_check_thin_contour_overlay`) -- the SAME graceful-degrade contract
    `_crop_overlay_array` uses: falls back to a plain (unstretched) grayscale->RGB
    array when there is no readable/usable mask; None only when the crop's OWN
    pixels cannot be read at all (that crop is simply OMITTED from the montage, not
    a hard failure)."""
    pixels = _crop_pixels(crop)
    if pixels is None:
        return None
    mask = _target_mask_pixels(crop)
    if mask is not None:
        mask_bool = np.asarray(mask) > 0
        if mask_bool.shape == pixels.shape and mask_bool.any():
            try:
                stretched = _mask_check_stretched_pixels(pixels, mask_bool)
                return _mask_check_thin_contour_overlay(stretched, mask)
            except Exception as exc:
                logger.warning("crop-triage: the mask check overlay draw failed for a montage cell (%s)", exc)
    import cv2

    from shape.utils.segmentation_qc import to_display_uint8
    base = to_display_uint8(np.asarray(pixels))
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR) if base.ndim == 2 else np.asarray(base)[..., :3].copy()


def _mask_check_resize_capped_padded(overlay_rgb: np.ndarray) -> np.ndarray:
    """Resizes `overlay_rgb` (native crop size) into a fixed `_MONTAGE_CELL_PX`-square
    canvas WITHOUT unconstrained upscaling -- ported from
    the cropper probe's `_resize_capped_padded` (see the section banner above)."""
    import cv2
    h, w = overlay_rgb.shape[:2]
    native_max = max(h, w, 1)
    scale = min(_MASK_CHECK_MAX_MONTAGE_UPSCALE, _MONTAGE_CELL_PX / native_max)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(np.asarray(overlay_rgb), (new_w, new_h), interpolation=interp)
    canvas = np.full((_MONTAGE_CELL_PX, _MONTAGE_CELL_PX, 3), _MASK_CHECK_MONTAGE_PAD_VALUE, dtype=np.uint8)
    r0, c0 = (_MONTAGE_CELL_PX - new_h) // 2, (_MONTAGE_CELL_PX - new_w) // 2
    canvas[r0:r0 + new_h, c0:c0 + new_w] = resized
    return canvas


def _mask_check_label_montage_cell(overlay_rgb: np.ndarray, index: int) -> np.ndarray:
    """FIX 2's drop-in replacement for `_label_montage_cell`, ported from
    the cropper probe's `_capped_pad_montage_cell` -- the ONLY difference from the
    shared function: the resize/pad step is `_mask_check_resize_capped_padded`. The index
    label itself (font/size/box) is drawn identically."""
    import cv2
    cell = _mask_check_resize_capped_padded(overlay_rgb)
    label = str(index)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(cell, (0, 0), (tw + 8, th + baseline + 6), (0, 0, 0), -1)
    cv2.putText(cell, label, (4, th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return cell


def _build_mask_check_montage(crops_chunk: list[dict], grid: tuple[int, int]) -> tuple[bytes | None, dict[int, dict]]:
    """Builds ONE labeled grid montage from up to `grid[0]*grid[1]` mask-check candidate
    crops -- a sibling of `_build_montage`, using the mask-check-specific overlay/label
    functions above instead of the shared ones (the OTHER channels' own montages are
    untouched by this substitution). Same index->crop map / graceful-degrade contract
    as `_build_montage`."""
    import cv2

    rows, cols = grid
    cell_imgs: list[np.ndarray] = []
    index_to_crop: dict[int, dict] = {}
    for crop in crops_chunk:
        if len(cell_imgs) >= rows * cols:
            break
        overlay = _mask_check_overlay_array(crop)
        if overlay is None:
            continue
        index = len(cell_imgs)
        cell_imgs.append(_mask_check_label_montage_cell(overlay, index))
        index_to_crop[index] = crop
    if not cell_imgs:
        return None, {}

    montage = np.zeros((rows * _MONTAGE_CELL_PX, cols * _MONTAGE_CELL_PX, 3), dtype=np.uint8)
    for idx, cell in enumerate(cell_imgs):
        r, c = divmod(idx, cols)
        montage[r * _MONTAGE_CELL_PX:(r + 1) * _MONTAGE_CELL_PX,
               c * _MONTAGE_CELL_PX:(c + 1) * _MONTAGE_CELL_PX] = cell

    ok, buf = cv2.imencode(".png", montage)
    if not ok:
        return None, {}
    return buf.tobytes(), index_to_crop


class MaskCheckPositionVerdict(BaseModel):
    """What the judge reports for ONE labeled position in a mask-check montage (task 058
    M3). A sibling of `S1PositionVerdict`, richer verdict space (subsumes it -- see
    the section banner above): the grid INDEX is ground truth, `crop_id` only a
    restated cross-check."""
    index: int = Field(description="The grid position (0..N-1) this verdict is about "
                                    "-- the SAME number drawn on that position's own "
                                    "label in the montage image.")
    crop_id: str = Field(description="The crop identifier printed alongside this "
                                     "position in the prompt's own position list -- "
                                     "restate it here as a cross-check; if it does not "
                                     "match what you were told for this index, the "
                                     "INDEX still wins.")
    verdict: Literal["cell_present", "target_region_empty", "wrong_object", "merged", "abstain"] = Field(
        description="'cell_present': the green contour outlines a coherent SINGLE "
                    "cell. 'target_region_empty': the contour outlines empty "
                    "background or noise, with NO real cell visible anywhere in this "
                    "tile. 'wrong_object': a real cell genuinely IS visible somewhere "
                    "in this tile, but either it lies mostly OUTSIDE the contour (a "
                    "DIFFERENT cell than the one the contour intends) or the contour "
                    "straddles a neighbouring cell. 'merged': the contour outlines "
                    "TWO touching cells at once -- a doublet under ONE outline, not "
                    "cleanly a single cell (an under-segmentation the mask alone "
                    "cannot see). 'abstain': genuinely too ambiguous to call.")
    confidence: Literal["high", "low"]


class MaskCheckVerdict(BaseModel):
    """One `MaskCheckPositionVerdict` per labeled position in the montage. May be a SUBSET
    of the montage's own positions -- a position the judge does not mention is
    treated as NOT rejected (the same keep-biased default every other verdict shape
    in this module uses on any gap/failure)."""
    verdicts: list[MaskCheckPositionVerdict]


_MASK_CHECK_REJECT_VERDICTS = frozenset({"target_region_empty", "wrong_object", "merged"})
# GEOMETRY-INVARIANT mask-check verdicts, safe to cache across VLRR rounds by cell id: they
# describe the cell/target-region IDENTITY, not the box tightness, so recentering /
# tightening cannot change them. `merged` is EXCLUDED (two touching cells CAN separate
# when the box is recentered -> must be re-judged); `abstain` is not a real verdict.
_MASK_CHECK_INVARIANT_VERDICTS = frozenset({"cell_present", "wrong_object", "target_region_empty"})


def _build_mask_check_montage_prompt(index_to_crop: dict[int, dict]) -> str:
    """Assembles the mask check's own prompt (task 058 M3) -- mirrors
    `_build_s1_montage_prompt`'s grid/position/index protocol, but asks the FOUR-WAY
    mask-error question this channel exists to answer, IN THE NARRATIVE (not only
    implied by the schema field's own doc,
    [[judge_prompt_narrative_not_schema_field]]). Of the cropper CARD it carries only the
    rule text the judge needs (`_cropper_judge_rules`: the purpose and quality order, and
    the readouts' definitions)."""
    n = len(index_to_crop)
    positions = "\n".join(
        f"  Position {i}: crop_id={_crop_identifier(index_to_crop[i])!r}"
        for i in sorted(index_to_crop))
    card = _cropper_judge_rules()
    # PROMPT-CACHE LAYOUT (token cost): this is the mask-check judge the cropper benchmark
    # actually calls, thousands of times. Fixed instructions + CARD rules FIRST as a stable
    # prefix; the per-call candidate count + position list LAST. Previously the leading
    # "GRID MONTAGE of {n} candidates" put a varying number near byte 50 and the CARD at the
    # tail, so the prefix never matched call-to-call (measured 0% cache hit). See
    # _build_montage_prompt for the full rationale. Wording/semantics unchanged.
    card_section = (f"TOOL CARD -- what the crops are for and what the crop readouts "
                    f"mean:\n{card}\n\n" if card else "")
    fixed_prefix = (
        f"{card_section}"
        f"You are checking whether a segmentation MASK actually points at the RIGHT "
        f"cell, for a GRID MONTAGE of single-cell crop candidates (labeled "
        f"positions, indexed 0..N-1, top-left to bottom-right, row-major). EACH "
        f"position's own target region is marked with a THIN, SATURATED GREEN "
        f"contour ONLY (no fill -- the cell's own texture is fully visible under and "
        f"around it); the small yellow-on-black number in that position's own "
        f"top-left corner is its INDEX -- report each verdict against THAT index "
        f"(restate the crop_id you were told for it as a cross-check).\n\n"
        f"For EACH labeled position, decide which ONE of these best describes what "
        f"is INSIDE the green contour:\n"
        f"  (a) cell_present -- one coherent SINGLE cell (a real cell shape/texture "
        f"you can make out).\n"
        f"  (b) target_region_empty -- empty background or noise, and you do NOT "
        f"see a real cell anywhere else in this tile either.\n"
        f"  (c) wrong_object -- a real cell IS visible in this tile, but either it "
        f"lies mostly OUTSIDE the contour (a DIFFERENT cell than the one the "
        f"contour intends), or the contour straddles a neighbouring cell.\n"
        f"  (d) merged -- the contour outlines TWO touching cells at once -- a "
        f"doublet under ONE outline, not cleanly a single cell.\n\n"
        f"For EACH labeled position, report:\n"
        f"  - verdict: 'cell_present' / 'target_region_empty' / 'wrong_object' / "
        f"'merged' -- or 'abstain' only if genuinely too ambiguous to call between "
        f"them.\n"
        f"  - confidence: 'high' or 'low'.\n\n"
    )
    return (
        f"{fixed_prefix}"
        f"THIS MONTAGE has {n} candidates (indices 0..{n - 1}).\n"
        f"Positions:\n{positions}"
    )


def _apply_mask_check_verdicts(mask_check_verdict, index_to_crop: dict[int, dict], *,
                          montage_path: str | None = None) -> list[dict]:
    """Maps each labeled position's verdict back onto its own crop dict, BY INDEX
    (the SAME landing rule `_apply_s1_verdicts`/`_apply_montage_verdicts` use: a
    `crop_id` mismatch is logged and the INDEX still wins). Sets
    `crop["mask_check_verdict"]`/`crop["mask_check_confidence"]` IN PLACE UNCONDITIONALLY
    (mirrors `s1_recommend_crop`'s own de-randomized in-place contract -- the raw
    verdict is recorded regardless of reject/keep, for the trace + the benchmark's
    own GT-adjudication validation), and additionally sets
    `crop["vision_reject"]`/`crop["vision_problem"]` -- the SAME flag the (B) REJECT
    /S1 channels already set, so `crop_kept`/`_rewrite_kept_manifest`'s existing
    `not vision_reject` predicate picks up a mask-check rejection automatically, with NO
    changes needed to either -- ONLY when `verdict` is one of
    `_MASK_CHECK_REJECT_VERDICTS` (`target_region_empty`/`wrong_object`/`merged`),
    REGARDLESS of confidence (mirrors `_apply_s1_verdicts`'s own landing spec, the
    exact channel this one replaces -- no confidence qualifier was ever part of that
    landing rule). `cell_present`/`abstain`/an unmentioned position/any failure all
    KEEP (never discard on silence). Returns the per-position verdict-trace dicts,
    the SAME shape the other channels' own trace entries use."""
    verdicts = []
    for pos in mask_check_verdict.verdicts:
        crop = index_to_crop.get(pos.index)
        if crop is None:
            logger.warning("crop-triage: mask-check montage verdict named an unknown "
                           "index %s; skipping", pos.index)
            continue
        expected_id = _crop_identifier(crop)
        if pos.crop_id and pos.crop_id != expected_id:
            logger.warning(
                "crop-triage: mask-check montage verdict index %s crop_id mismatch (got "
                "%r, expected %r) -- trusting the index", pos.index, pos.crop_id, expected_id)
        crop["mask_check_verdict"] = pos.verdict
        crop["mask_check_confidence"] = pos.confidence
        reject = pos.verdict in _MASK_CHECK_REJECT_VERDICTS
        if reject:
            crop["vision_reject"] = True
            crop["vision_problem"] = pos.verdict
        verdicts.append({
            "item_id": crop.get("crop_path"),
            "crop_id": expected_id,
            "reject": reject,
            "verdict": pos.verdict,
            "confidence": pos.confidence,
            "problem": pos.verdict if reject else "none",
            "reasoning": f"[mask-check montage index {pos.index}]",
            "montage_path": montage_path,
        })
    return verdicts


def mask_check_recommend_crop(engine_mm, mask_check_candidates: list[dict], *,
                         sample_cap: int | None = None,
                         metadata_path: str | None = None,
                         montage_grid: tuple[int, int] = _DEFAULT_MONTAGE_GRID,
                         verdict_cache: dict | None = None) -> dict:
    """The mask check channel (task 058 M3, human-authorized -- REPLACES `s1_recommend_crop`
    as the cropper's data-screened mask-error VLM channel): judges `mask_check_candidates`
    (already data-screened by the CALLER, `verifiers/crop.py::CropVerifier.adjudicate`
    -- task 058 M4's own THREE-ARM `_gate_candidate` OR gate (dim / doublet / bright-
    debris, REPLACES the old p90-only screen) + the SAME suspicion-sorted
    `s1_candidate_cap` S1 always used) in BATCHED, labeled grid montages
    (`_build_mask_check_montage`, the mask-check-specific contour-only + mask-stretched
    rendering -- see the section banner above), NOT one call per crop.

    Sets `crop["mask_check_verdict"]`/`crop["mask_check_confidence"]`/`crop["vision_reject"]`/
    `crop["vision_problem"]` IN PLACE on every JUDGED candidate (`_apply_mask_check_
    verdicts`) -- the caller's LATER `vision_recommend_crop` call (which runs the (B)
    REJECT channel and THEN writes the shared kept-crops manifest + triage trace over
    the FULL population) picks this up automatically as long as this function runs
    FIRST, mirroring `s1_recommend_crop`'s own established wiring contract exactly
    (same signature shape, same `extra_verdicts` threading).

    Task 058 M8 (QC fix): ALSO sets `crop["mask_check_sent"] = True` IN PLACE on every
    crop actually placed into a montage batch dispatched to `engine_mm.generate`,
    set just before that call so it is True regardless of whether a verdict comes
    back for it -- `MaskCheckVerdict` explicitly permits a SUBSET response, and
    `_apply_mask_check_verdicts` only sets `mask_check_verdict` for positions the response
    names. A crop can legitimately end up `mask_check_sent=True, mask_check_verdict=None`
    (sent but omitted). A crop skipped at render (`_build_mask_check_montage`'s own
    `overlay is None` drop) or capped out before this function ran never gets
    `mask_check_sent` at all. This is the ACTUAL-sent signal `_build_crop_decision_
    trace`'s own `sent_to_vlm` now reads, instead of `mask_check_verdict is not None`.

    `sample_cap` (optional, default None): a FURTHER, generic evenly-spaced
    deterministic reduction (mirrors `s1_recommend_crop`'s own contract) -- most
    callers leave this None and let the caller's own suspicion-ordered cap
    (`CropVerifier.s1_candidate_cap`) be the only reduction.

    Returns `{"judged": int, "rejected": int, "engine": bool, "verdicts": [...]}`,
    the SAME shape `s1_recommend_crop`/`_judge_reject_montages` return, plus `"montages"`: per montage sent
    `{index, prompt, image (its ref: label, sha256, bytes), montage_path, reply (the verdict object as JSON, or
    the reply as returned), error}` (the record of each question; never the image bytes). No engine /
    no candidates -> a no-op summary, same graceful-degrade contract as every other
    channel in this module."""
    if engine_mm is None or not mask_check_candidates:
        return {"judged": 0, "rejected": 0, "reused": 0,
                "engine": engine_mm is not None, "verdicts": [], "montages": []}

    candidates = mask_check_candidates
    # Cross-round cost saver (task 059): reuse a GEOMETRY-INVARIANT mask-check verdict
    # (cell_present / wrong_object / target_region_empty -- see _MASK_CHECK_INVARIANT_VERDICTS)
    # cached by the crop's stable cell id, so a later VLRR round re-applies it WITHOUT
    # a VLM call. `merged`/`abstain` are never cached, so a genuinely geometry-dependent
    # verdict is always re-judged -- triage correctness is unchanged, only redundant,
    # provably-identical re-judgements of identity-settled crops are removed.
    reused = reused_rejects = 0
    if verdict_cache is not None:
        fresh = []
        for c in candidates:
            cached = verdict_cache.get(_crop_identifier(c))
            if cached:
                c["mask_check_verdict"] = cached["verdict"]
                c["mask_check_confidence"] = cached["confidence"]
                if cached["verdict"] in _MASK_CHECK_REJECT_VERDICTS:
                    c["vision_reject"] = True
                    c["vision_problem"] = cached["verdict"]
                    reused_rejects += 1
                reused += 1
            else:
                fresh.append(c)
        candidates = fresh
    if not candidates:
        return {"judged": 0, "rejected": reused_rejects, "reused": reused,
                "engine": True, "verdicts": [], "montages": []}
    if sample_cap is not None and len(candidates) > sample_cap:
        step = len(candidates) / float(sample_cap)
        candidates = [candidates[int(i * step)] for i in range(sample_cap)]

    montages_dir = _montages_dir_for(metadata_path, prefix="montages_mask_check")
    rows, cols = montage_grid
    chunk_size = max(1, rows * cols)
    judged = rejected = 0
    all_verdicts: list[dict] = []
    batches = []          # (prompt, montage_bytes, index_to_crop, montage_path, chunk_idx), in chunk order
    for start in range(0, len(candidates), chunk_size):
        chunk_idx = start // chunk_size
        chunk = candidates[start:start + chunk_size]
        try:
            montage_bytes, index_to_crop = _build_mask_check_montage(chunk, montage_grid)
            if montage_bytes is None:
                continue
            montage_path = (_save_montage_png(montages_dir, chunk_idx, montage_bytes)
                            if montages_dir else None)
            batches.append((_build_mask_check_montage_prompt(index_to_crop), montage_bytes, index_to_crop,
                            montage_path, chunk_idx))
        except Exception as exc:  # pragma: no cover - defensive: triage must never break QC
            logger.warning("crop-triage: mask-check montage judging failed (%s); keeping this batch", exc)

    def judge(batch):
        prompt, montage_bytes, index_to_crop, _, _ = batch
        # Task 058 M8 (QC fix): mark every crop in THIS batch as sent, in place,
        # right before dispatch -- true regardless of whether a verdict comes
        # back for it, or the response is a subset/malformed/empty one (see this
        # function's own docstring).
        for crop in index_to_crop.values():
            crop["mask_check_sent"] = True
        try:
            resp = engine_mm.generate([prompt, montage_bytes], response_format=MaskCheckVerdict)
        except Exception as exc:  # pragma: no cover - defensive: triage must never break QC
            from shape.engine.errors import is_fatal_llm_error
            if is_fatal_llm_error(exc):
                exc.shape_question = "mask_check"
                raise
            logger.warning("crop-triage: mask-check montage judging failed (%s); keeping this batch", exc)
            return None, f"{type(exc).__name__}: {exc}", None
        rec = resp.get("content") if isinstance(resp, dict) else resp
        error = None
        if not hasattr(rec, "verdicts"):     # kept, as on any failure: a gateway error or an invalid reply
            error = (f"{resp.get('error')}: {resp.get('message', '')}" if isinstance(resp, dict) and resp.get("error")
                     else f"no MaskCheckVerdict in the reply ({type(rec).__name__})")
        return rec, error, (resp if isinstance(resp, dict) and "content" not in resp else rec)

    # the montages are judged at once (independent questions, one per montage); the verdicts are applied
    # below in chunk order, so the outcome is the one a serial pass gives
    if len(batches) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(batches)) as pool:
            records = list(pool.map(judge, batches))
    else:
        records = [judge(batch) for batch in batches]
    from .base import image_refs, reply_record
    montages = [{"index": chunk_idx, "prompt": prompt,
                 "image": image_refs([prompt, montage_bytes], f"MASK CHECK MONTAGE {chunk_idx}")[0],
                 "montage_path": montage_path, "reply": reply_record(reply), "error": error}
                for (prompt, montage_bytes, _, montage_path, chunk_idx), (_, error, reply) in zip(batches, records)]
    for (_, _, index_to_crop, montage_path, _), (rec, _, _) in zip(batches, records):
        if not hasattr(rec, "verdicts"):
            continue
        judged += len(index_to_crop)
        batch_verdicts = _apply_mask_check_verdicts(rec, index_to_crop, montage_path=montage_path)
        rejected += sum(1 for v in batch_verdicts if v["reject"])
        all_verdicts.extend(batch_verdicts)
        # Remember only geometry-INVARIANT verdicts so a later round skips re-sending
        # them. `_apply_mask_check_verdicts` set mask_check_verdict/mask_check_confidence in place.
        if verdict_cache is not None:
            for crop in index_to_crop.values():
                v = crop.get("mask_check_verdict")
                if v in _MASK_CHECK_INVARIANT_VERDICTS:
                    verdict_cache[_crop_identifier(crop)] = {
                        "verdict": v, "confidence": crop.get("mask_check_confidence")}
    return {"judged": judged, "rejected": rejected + reused_rejects, "reused": reused,
            "engine": True, "verdicts": all_verdicts, "montages": montages}


def _rewrite_kept_manifest(crops: list[dict], metadata_path: str | None) -> None:
    """Writes the kept-crops manifest (`llm_recommend.write_kept_manifest`) over the FULL
    `crops` population (not just the judged candidates) -- `kept_predicate=not
    vision_reject` naturally includes every non-candidate crop (never given `vision_reject`
    at all) as kept, and every judged candidate by its own verdict. No-op if
    `metadata_path` is falsy (mirrors the shared lander's own graceful-degrade contract)."""
    if not metadata_path:
        return
    from . import llm_recommend
    llm_recommend.write_kept_manifest(
        crops, metadata_path, kept_predicate=lambda c: not c.get("vision_reject", False))


_DECISION_RANK = {"PASS": 0, "WARNING": 1, "FAIL": 2}
_RANK_DECISION = {v: k for k, v in _DECISION_RANK.items()}


def tighten_for_confirmed_rejects(deterministic: VLRRResult, crops: list[dict], rejected: int,
                                  judged: int, *, tag: str) -> VLRRResult:
    """Shared severity-tightening step (task 058 M3, EXTRACTED from
    `vision_recommend_crop`'s own tail -- byte-identical logic for that function's
    own existing callers, see the coding log's own before/after check -- so the mask check's
    OWN confirmed rejects can tighten `deterministic` the SAME way (B) REJECT's own
    confirmed rejects always did). This EXTRACTION also closes a real, PRE-EXISTING
    gap, not just preserves the status quo: before this round, `vision_recommend_
    crop`'s own early-return checked ONLY its OWN local `rejected` count, so a
    confirmed S1/mask-check reject with zero (B)-REJECT-channel rejects THIS round never
    tightened `det` at all, even though the crop really was excluded from the kept
    population -- calling this helper separately, once per channel, with THAT
    channel's own rejected/judged counts, fixes it for the mask check going forward.

    A confirmed reject may only TIGHTEN the deterministic decision by ONE severity
    notch (PASS->WARNING->FAIL), never loosen it; `scalar_score`/`action`/
    `param_delta` stay the deterministic gate's own. `rejected==0` -> `deterministic`
    UNCHANGED (nothing to tighten for) -- `tighten_for_confirmed_rejects` is a no-op
    when a channel judged crops but confirmed none of them unusable."""
    if not rejected:
        return deterministic
    problems = sorted({c.get("vision_problem", "") for c in crops if c.get("vision_reject")} - {""})
    tightened_rank = min(_DECISION_RANK[deterministic.decision] + 1, _DECISION_RANK["FAIL"])
    decision = _RANK_DECISION[max(_DECISION_RANK[deterministic.decision], tightened_rank)]
    notes = (f"{deterministic.notes} " if deterministic.notes else "") + (
        f"[{tag}] {rejected}/{judged} data-screened candidate(s) confirmed unusable "
        f"({', '.join(problems) or 'unspecified'}).")
    return VLRRResult(
        decision, deterministic.scalar_score, {},
        action=deterministic.action,
        param_delta=deterministic.param_delta,
        expected_effect=deterministic.expected_effect,
        fault=deterministic.fault,
        upstream_target=deterministic.upstream_target,
        judge_problem="; ".join(problems),
        notes=notes.strip(),
    )


def vision_recommend_crop(engine_mm, crops: list[dict] | None, deterministic: VLRRResult, *,
                          sample_cap: int | None = None,
                          metadata_path: str | None = None,
                          montage_grid: tuple[int, int] = _DEFAULT_MONTAGE_GRID,
                          extra_verdicts: list[dict] | None = None,
                          screen_reject_candidates: bool = True,
                          verdict_cache: dict | None = None) -> VLRRResult:
    """The cropper's THIN adapter (task 038 M-B; restructured v9, then the montage
    round, then v10) for REJECT (cut_off/empty_debris) ONLY -- geometry (mask_bbox/
    recenter) is 100% pure-data (task 038 v10, human-locked) and has NO VLM channel
    here at all any more; see `verifiers/crop.py::CropVerifier.adjudicate`.

    `reject_candidates` = crops flagged for cut_off/empty_debris
    (`_vlm_candidate_reasons`) -- judged in BATCHED, labeled grid MONTAGES
    (`_judge_reject_montages`), NOT one call per crop -- see that function's own
    docstring and the module docstring's (D) for why (a real v9 cluster run: a
    thin/elongated cell's naturally-low `foreground_fraction` false-triggered
    `empty_debris` on isolated per-crop asks; a montage lets the judge compare many
    candidates at once and catch it).

    `crops` (the FULL population, not just the candidates) is what the kept-crops
    manifest + triage-trace are written against (`_rewrite_kept_manifest`/
    `write_triage_trace`) -- a non-candidate crop never gets `vision_reject` set at
    all, so it is correctly counted "kept" by the SAME `kept_predicate=not
    vision_reject` (CARD.md SS2's VERDICT DECLARATION) the manifest lander uses for a
    judged crop.

    No engine / no crops / any failure in the montage channel -> a no-op
    (`deterministic` returned UNCHANGED) -- parity-safe, byte-identical to the
    deterministic-only path when nothing happens at all. A confirmed REJECT may only
    TIGHTEN the deterministic decision by ONE severity notch (PASS->WARNING->FAIL),
    never loosen it; `scalar_score`/`action`/`param_delta` stay the deterministic
    gate's own -- this function judges KEEP/REJECT per crop, it does not propose a
    tunable parameter itself.

    `montage_grid=(rows, cols)` (default 3x3, `CropVerifier.triage_montage_grid`'s own
    default -- HUMAN wants 2x2/3x3/4x4 tested): the reject-montage chunk size, passed
    straight through to `_judge_reject_montages`.

    `metadata_path` (from `adapt_crop`'s context): when the engine actually ran (crops is
    non-empty and `engine_mm` is not None), also persists the kept-crops manifest and the
    full-verdict triage trace so the downstream analyzer/a human can see exactly what
    was judged and why -- unconditionally, even when there were zero candidates (a
    batch with nothing worth asking about is itself a valid, reportable outcome). No
    `metadata_path` -> no manifest/trace, unchanged behavior. It ALSO derives WHERE any
    sent montage gets persisted (task 038 v10, human-locked -- `_montages_dir_for`,
    `<tool_cache_dir>/montages_<hash>/`): a no-engine or no-reject-candidate run writes
    no montage files at all (`_judge_reject_montages` never touches `montages_dir`
    unless it genuinely builds+sends one).

    `extra_verdicts` (task 038 Round B, optional, default None -- backward compatible,
    every EXISTING caller is unaffected): another channel's own per-position verdict
    dicts (e.g. `s1_recommend_crop`'s own `["verdicts"]`) to fold into the SAME triage-
    trace write this function already performs -- `write_kept_manifest`/`write_triage_
    trace` derive a FIXED sidecar path from `metadata_path` alone, so a second,
    independent write from a caller would silently OVERWRITE this one rather than
    merge with it; threading the other channel's verdicts through here instead keeps
    ONE trace file with every channel's own reasoning. The CALLER (`verifiers/
    crop.py::CropVerifier.adjudicate`) is responsible for running that other channel
    BEFORE this function, so its own `vision_reject` flags are already set on `crops`
    by the time `_rewrite_kept_manifest` (below) reads them -- this parameter only
    affects the TRACE text, never population membership itself.

    `screen_reject_candidates` (task 058 M3, default True -- EVERY existing caller
    unaffected by the default): when False, this function's OWN cut_off/empty_debris
    candidate screen (`_vlm_candidate_reasons`) is skipped entirely -- NO reject
    montage is ever built or sent (no SECOND VLM pass) -- but the manifest-write/
    trace-write side effects below still run UNCONDITIONALLY, exactly as this
    function's own pre-existing "unconditional, even zero candidates" contract
    already promised. This matters: a crop already rejected by a DIFFERENT channel
    (the mask check, `verifiers/crop.py::CropVerifier.adjudicate`, which sets
    `crop["vision_reject"]` IN PLACE BEFORE this function is ever called) still
    needs `_rewrite_kept_manifest` to run so that rejection reaches the persisted
    kept-crops manifest sidecar every downstream reader (this benchmark, `Cell_
    State_Analyzer_*_Tool`) trusts -- disabling THIS function's own screening must
    NOT also silently stop it from being the manifest writer other code depends on.
    `verifiers/crop.py`'s own production caller passes this as `enable_reject_
    channel` (new attr, default False -- mirrors `enable_ab_channel`: the mask check is the
    sole DEFAULT VLM reject channel, task 058 M3 human-directed; this channel's own
    cut_off/empty_debris coverage stays recoverable via the flag, not deleted).
    """
    from . import llm_recommend

    if engine_mm is None or not crops:
        return deterministic

    reject_candidates = ([c for c in crops
                          if set(_vlm_candidate_reasons(c)) & {"cut_off", "empty_debris"}]
                         if screen_reject_candidates else [])

    try:
        reject_summary = _judge_reject_montages(
            engine_mm, reject_candidates, sample_cap=sample_cap, montage_grid=montage_grid,
            montages_dir=_montages_dir_for(metadata_path), verdict_cache=verdict_cache)
    except Exception as exc:  # pragma: no cover - defensive: triage must never break QC
        logger.warning("crop-triage: reject-montage judging failed (%s); using deterministic", exc)
        reject_summary = {"judged": 0, "rejected": 0, "engine": False, "verdicts": []}

    _rewrite_kept_manifest(crops, metadata_path)
    if metadata_path:
        trace_verdicts = list(extra_verdicts or []) + reject_summary.get("verdicts", [])
        llm_recommend.write_triage_trace(trace_verdicts, metadata_path)

    total_judged = reject_summary.get("judged", 0)
    total_rejected = reject_summary.get("rejected", 0)
    return tighten_for_confirmed_rejects(deterministic, crops, total_rejected, total_judged,
                                         tag="crop-triage")
