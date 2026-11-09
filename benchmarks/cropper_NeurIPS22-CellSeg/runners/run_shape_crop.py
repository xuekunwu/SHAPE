#!/usr/bin/env python3
"""Task 038: the agentic CROPPER eval on NeurIPS22-CellSeg GT-masked cells -- MaskCrop's
analog of `segmentation_NeurIPS22-CellSeg/runners/run_shape_neurips_cellseg.py`.

INVESTIGATION FIRST (before building this): read the segmenter benchmark +
`benchmarks/_shared/segmentation_metrics.py::compute_metrics`. Finding: the segmenter
benchmark does NOT literally isolate its stage via a direct verifier call -- it runs the
FULL agent (`shape.solver.solve`, real planner/executor/LLM) with a query
(`Prompt.txt`) worded to stay scoped to segmentation, so the planner naturally stops
there. Replicating that pattern for the cropper would need the LLM PLANNER to reach a
working segmentation -> crop hand-off, with no natural way to inject the REAL GT mask
directly (the planner reasons from images, not pre-made masks) -- expensive (a live
segmentation LLM call per image) and entangles the cropper's OWN agentic value with the
segmenter's. Given a GT mask is available here, this script instead isolates the
crop-stage VLRR loop DIRECTLY: `CropVerifier` + `StepVerifier.resolve_stage` +
`crop_triage.adapt_crop` + `Single_Cell_Cropper_Tool.execute()`, called explicitly rather
than through the full planner -- genuine reuse of the REAL production verifier/loop
machinery (no verifier logic reimplemented), just orchestrated directly. This is also
MUCH cheaper (deterministic geometry -> zero LLM calls; the ONLY tokens spent are the
triage vision judge's, inside `CropVerifier.adjudicate`) and a cleaner isolation of the
cropper's own agentic contribution specifically.

TWO ARMS (one crop-stage run per image, per arm) -- named for what they actually ARE, not
"baseline/agentic":
  MaskCrop : the deterministic CONTROL -- naive fixed margin (20% of the mask's own
             equivalent diameter, margin_mode="fraction", margin_frac=0.20,
             recenter=False) -- ONE `execute()` call, NO agent (no VLRR retune, no
             triage judge).
  SHAPE    : the FULL AGENTIC flow (task 038 follow-up, part B redesign, human directive
             -- FAIR START). Its FIRST `execute()` call reads the EXACT SAME naive
             margin as MaskCrop's own control arm (margin_mode="fraction", margin_frac=
             0.20, recenter=False) -- since a LATER round's own human directive, this is
             now ALSO the tool's own shipped `execute()` default (`tool.py`), so this arm
             explicitly hard-codes it rather than relying on that default, to stay
             correct regardless of the tool's own default -- so SHAPE's measured gain is
             the PURE agentic contribution on top of an identical starting point, not a
             "starts already-optimal" head start. This then runs through the REAL
             crop-stage VLRR loop (CropVerifier -> StepVerifier.resolve_stage) + the
             triage vision judge (`vision_recommend_crop`, inside
             `CropVerifier.adjudicate`, when an engine is wired), which now ALSO reasons
             per KEPT crop whether/how to tighten that cell toward the tight+recenter
             geometry (Part A's per-cell optimum, still present in `tool.py` as an
             opt-in, and now the AGENT's own optimization target rather than any
             caller's default) via a per-cell `cell_overrides` retune (see
             `run_shape_arm`).

QUALITY vs. MECHANISM (task 038 v13, human decision -- REFRAMED this round; read
CARD.md's own "Task 038 v13 SCOPE"/"Honest reporting" sections first): a methodology
critique settled that `crop_quality_score` is DISQUALIFIED as a validator (v12 reweighted
it toward the conclusion it was supposed to prove; even un-tuned, a mask-geometry score
is not the actual task objective -- only a downstream SSL/analyzer run, a SEPARATE,
not-yet-built task, can validate that). This script therefore no longer HEADLINES a
`crop_quality_score` delta as evidence SHAPE's crops are "better" -- the "quality"
section below is STILL computed (harmless, informational) but explicitly marked
non-authoritative in its own JSON (`"authoritative": False`), and the PRIMARY reported
content is now a "mechanism" section: `neighbor_contamination` reduction (SHAPE's own
final-kept population vs. MaskCrop's, mean per arm, from each crop's own MASK-based
`qc` block), `target_retention` distribution (SHAPE's own kept population -- how close
to `_TARGET_RETENTION_FLOOR` the tightened crops actually landed), plus the SAME
`tool_calls_per_image`/`token_cost_per_image`/`time_per_image_s` cost fields the efficiency section already
reports (these aggregate BOTH VLM channels -- reject and the blinded A/B gate -- combined;
this script has no separate per-channel call/abstain instrumentation without touching the
verifier internals, which are OUT of this round's own surgical scope). `crop_quality_score`
(when still read) is scored over each arm's ACTUAL OUTPUT population (task 038 follow-up
fix -- the real bug that round found: scoring EVERY crop in a result, including SHAPE's
own triage-REJECTED ones, made SHAPE's triage have ZERO effect on the metric): MaskCrop
has no triage, so its mean is over ALL its crops; SHAPE's mean is over the crops it
actually KEEPS (excludes `vision_reject`, read from the fused judge's own kept-crops
manifest, `llm_recommend.write_kept_manifest` -- no manifest, e.g. no engine wired or
nothing rejected, falls back to ALL crops, its own documented graceful-degrade contract).
This SAME kept-only population selection is reused for the mechanism metrics too
(`_qc_values_for_result`). `n_crops_*` fields are DESCRIPTIVE PROVENANCE only, not a
scored axis.

EFFICIENCY: reuses `benchmarks/_shared/segmentation_metrics.py::compute_metrics`,
generalized this task with a `metric_key` param (see that module's own diff) --
called here with `metric_key="crop_quality_score"`. The geometry is deterministic, so
retune-efficiency (positive/wasted rate, gain-per-call) is expected to show FEW padding
retunes: under `margin_mode="tight"`, containment is already guaranteed by construction
(`compute_square_crop_containing_bbox`), so a "truncation" trigger is almost always a
genuine image-edge case padding cannot fix either -- reported anyway for schema parity
(the dispatch's explicit ask), not hidden. ONLY these 7 fields (the already-defined set --
`crop_quality_score` + the 6 headline efficiency metrics from `compute_metrics`) are
reported; no reject-rate/kept-mean/yield axis is invented.

FULL-PROCESS TRACE (default on): `<out>/trace.jsonl`, one JSON line per image -- each
VLRR trial's params/decision/action/expected_effect/notes (SHAPE's own retune search),
plus the FINAL result's own per-crop triage verdicts (keep/reject/confidence/problem/
reasoning, read from `llm_recommend.write_triage_trace`'s sidecar -- task 038 follow-up
addition to the shared judge, since `judge_items` previously discarded confidence/
reasoning for kept crops), plus tokens/seconds. "Final result" (not every trial) because
that is what actually determined SHAPE's kept/rejected population and headline score --
the per-trial params/status list already traces the retune search's own trajectory
separately.

SAVED CROPS (persistent, both arms, PHYSICALLY separated -- task 038 follow-up, Item 2):
each arm's `query_cache_dir` now points at `<out>/crops/<image_id>/{maskcrop,shape}/`
(was a `tempfile.TemporaryDirectory` that got cleaned up after every image) -- the raw
crop PNGs + metadata JSON survive the run in `tool_cache/`, organized by cell id (the
tool's own `<image>_cell_<idx>_crop.<ext>` naming). On top of that, `_organize_kept_
rejected_crops` COPIES each arm's own final-result crops into browsable `kept/`/
`rejected/` subdirs (not just a manifest a human has to cross-reference by hand --
directly eyeballable, especially the bulk of blur rejects) plus a `result_manifest.json`
(`final_metadata_path`/`kept_crop_paths`/`rejected_crop_paths`, kept for machine-readable
cross-reference). MaskCrop never triages, so its `kept/` gets ALL its crops and
`rejected/` stays empty -- the SAME function handles both arms with no special-casing
(`_read_kept_paths` returns None, "no filtering", when no kept-manifest sidecar exists).
SHAPE's own VLRR loop may write more than one trial's crops into `tool_cache/` (harmless
residue from earlier, non-final retune attempts) -- only the FINAL trial's own crops get
organized into `kept/`/`rejected/`.

Usage (an allocated GPU node; needs an LLM key for the triage judge in the SHAPE arm):
    conda run -n shape_bench python benchmarks/cropper_NeurIPS22-CellSeg/runners/run_shape_crop.py \\
        --data-dir /scratch/groups/your-group/your-user/shape_benchmark/NeurIPS22-CellSeg \\
        --out /scratch/groups/your-group/your-user/shape_benchmark/results/cropper_NeurIPS22-CellSeg/<run_id> \\
        --sample-cap 60 --model gpt-4o

Offline validation: see the co-located `test_run_shape_crop.py` (synthetic cells, a
mock/stub engine -- no real API call).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))  # repo root, for `shape.*`
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "_shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling module, hyphenated dir

import segmentation_metrics as sm  # noqa: E402  -- benchmarks/_shared/ (generalized, task 038)
from pairing import _find_image_mask_pairs  # noqa: E402  -- reused, not reimplemented

from shape.agent.verification.step_verifier import StepVerifier  # noqa: E402
from shape.agent.verification.verifiers.crop import CropVerifier, adapt_crop  # noqa: E402
from shape.agent.verification.vision_engine import build_vision_engine  # noqa: E402
from shape.engine import usage as token_usage  # noqa: E402
from shape.tools.single_cell_cropper.quant import crop_quality_from_qc  # noqa: E402
from shape.tools.single_cell_cropper.tool import SEGMENTER_DEFAULT_MIN_SIZE, Single_Cell_Cropper_Tool  # noqa: E402

# The tool's own default min_area: the cell segmenter's default min_size (`tool.py::SEGMENTER_DEFAULT_MIN_SIZE`;
# the cropper inherits the segmenter's object definition, 2026-10-01 -- it was 50). Both arms use this value.
DEFAULT_MIN_AREA = SEGMENTER_DEFAULT_MIN_SIZE
DEFAULT_MARGIN = 1
DEFAULT_MASKCROP_MARGIN_FRAC = 0.20   # "naive fixed margin = 20% of equivalent diameter"
# Task 038 follow-up (part B redesign, human directive -- FAIR START): SHAPE's arm now
# starts EVERY crop at this SAME naive margin (margin_mode="fraction", recenter=False) --
# the IDENTICAL geometry MaskCrop's own control arm uses, no separate "shape" constant
# to risk drifting out of sync. SHAPE's own agentic contribution is then the PURE,
# per-cell result-driven optimization ON TOP of this shared starting point (see
# `run_shape_arm`) -- so the comparison isolates exactly what the agent adds, and
# retunes genuinely FIRE (a loose, non-recentered crop has real tightening/recentering
# work for the agent to reason about, unlike starting from the tool's own tight+recenter
# fixed point, task 038 follow-up's earlier "one-shot" design). Raised 0.10 -> 0.20 (task
# 038 v9 restructure, human-locked): a v8 cluster run confirmed the square-tight
# geometry lever's own quality gain stays ~0 (it cannot isolate an adjacent neighbour --
# `tool.py`'s new `margin_mode="mask_bbox"` is the fix), and the 10% baseline was
# already close to its own ceiling -- this gives BOTH arms' shared starting point
# genuinely more headroom.


def _kept_manifest_path(metadata_path: str) -> Path | None:
    """The kept-crops manifest sidecar path `llm_recommend.write_kept_manifest` writes
    next to `metadata_path`, by the SAME naming transform (never re-derived ad hoc)."""
    basename = Path(metadata_path).name
    if "cell_crops_metadata_" not in basename:
        return None
    return Path(metadata_path).parent / basename.replace(
        "cell_crops_metadata_", "cell_crops_kept_manifest_")


def _triage_trace_path(metadata_path: str) -> Path | None:
    """The full-verdict triage-trace sidecar path `llm_recommend.write_triage_trace`
    writes next to `metadata_path` (task 038 follow-up addition)."""
    basename = Path(metadata_path).name
    if "cell_crops_metadata_" not in basename:
        return None
    return Path(metadata_path).parent / basename.replace(
        "cell_crops_metadata_", "cell_crops_triage_trace_")


def _read_kept_paths(metadata_path: str) -> set[str] | None:
    """The fused judge's own KEPT crop_paths for this metadata file, or None when no
    manifest exists (no engine wired, nothing rejected, or the write failed) -- callers
    must treat None as "no filtering, every crop in this result is kept", matching
    `write_kept_manifest`'s own graceful-degrade contract."""
    manifest_path = _kept_manifest_path(metadata_path)
    if manifest_path is None or not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return set(data.get("kept_crop_paths", []))
    except Exception:
        return None


def _read_triage_trace(metadata_path: str) -> list[dict]:
    """The fused judge's full per-crop verdicts for this metadata file ([] when no
    engine ran, nothing was judged, or no trace sidecar exists)."""
    trace_path = _triage_trace_path(metadata_path)
    if trace_path is None or not trace_path.exists():
        return []
    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        return data.get("verdicts", [])
    except Exception:
        return []


def _crop_quality_scores_for_result(execute_result: dict, *, kept_only: bool = False) -> list[float]:
    """Per-crop `crop_quality_score` for every crop in ONE `execute()` result, read from
    its metadata JSON (mask-based qc + pixel sharpness). `kept_only=True` (SHAPE's own
    arm): scores only the crops the fused judge actually KEEPS (excludes
    `vision_reject`), per `_read_kept_paths` -- this is the real fix (task 038 follow-up):
    scoring every crop regardless of triage made SHAPE's own triage have ZERO effect on
    the headline metric. `kept_only=False` (MaskCrop, no triage): scores ALL its crops,
    unchanged. [] when no metadata/crops (e.g. a failed or empty call)."""
    metadata_path = execute_result.get("cell_crops_metadata_path")
    if not metadata_path or not Path(metadata_path).exists():
        return []
    from skimage.io import imread

    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    kept_paths = _read_kept_paths(metadata_path) if kept_only else None
    scores = []
    for m in meta.get("cell_metadata", []) or []:
        qc = m.get("qc")
        crop_path = m.get("crop_path")
        if not qc or not crop_path or not Path(crop_path).exists():
            continue
        if kept_paths is not None and crop_path not in kept_paths:
            continue
        crop_pixels = np.asarray(imread(crop_path))
        if crop_pixels.ndim == 3:
            crop_pixels = crop_pixels[..., 0]
        scores.append(crop_quality_from_qc(crop_pixels, qc))
    return scores


def _qc_values_for_result(execute_result: dict, key: str, *, kept_only: bool = False) -> list[float]:
    """Per-crop `qc[key]` for every crop in ONE `execute()` result (task 038 v13, Honest
    reporting) -- the SAME population-selection logic `_crop_quality_scores_for_result`
    uses (`kept_only=True` excludes triage-rejected crops for SHAPE's own arm, via
    `_read_kept_paths`), reused here for MECHANISM metrics (`neighbor_contamination`/
    `target_retention`) instead of the disqualified composite quality score. No pixel
    I/O needed (unlike `_crop_quality_scores_for_result`, which reads the crop's own PNG
    for pixel sharpness) -- purely metadata-based. [] when no metadata/crops or the key
    is absent from a crop's own `qc` block."""
    metadata_path = execute_result.get("cell_crops_metadata_path")
    if not metadata_path or not Path(metadata_path).exists():
        return []
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    kept_paths = _read_kept_paths(metadata_path) if kept_only else None
    values = []
    for m in meta.get("cell_metadata", []) or []:
        qc = m.get("qc")
        crop_path = m.get("crop_path")
        if not qc or key not in qc or qc[key] is None:
            continue
        if kept_paths is not None and crop_path not in kept_paths:
            continue
        values.append(float(qc[key]))
    return values


def _newest_metadata_file(tool_cache_dir: Path) -> Path | None:
    """The metadata file with the NEWEST mtime under a `tool.py`-managed `tool_cache/`
    directory (task 038 QC N1 fix, below) -- `tool.execute()` writes a UNIQUELY
    hash-suffixed `cell_crops_metadata_<hash>.json` on EVERY call (confirmed by
    reading `tool.py` directly: `uuid4().hex[:8]`), so every call within one
    `resolve_stage` run accumulates its OWN metadata file in the SAME directory,
    never overwriting an earlier one -- unlike the per-cell crop IMAGE files
    (deterministic names, genuinely overwritten in place each call). The newest
    metadata file is therefore always the one describing whatever the crop image
    files CURRENTLY on disk actually are, regardless of which call's own result
    dict a caller happens to be holding. None if the directory does not exist or
    has no matching files (defensive; never raises)."""
    if not tool_cache_dir.is_dir():
        return None
    candidates = list(tool_cache_dir.glob("cell_crops_metadata_*.json"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _converged_result(resolution) -> dict:
    """Task 038 Round A2 (coordinator-directed, QC's own v13 note in QC_DECISION.md):
    `resolution.result` is `StepVerifier._best`'s own SCALAR-SCORE re-ranking across the
    WHOLE trial history (`step_verifier.py`, shared/frozen, NOT touched by this fix --
    layering) -- on the "satisfied" convergence path (the common headless outcome) `_best`
    can return an EARLIER trial's params/metrics while the on-disk crop files
    (deterministic filenames the tool overwrites on EVERY `tool_runner` call) reflect the
    LATEST trial instead. Headless never triggers this divergence (no A/B veto ever
    reverts a comfortable cut back to baseline, so there is nothing for a later trial to
    differ FROM); a LIVE A/B veto -> revert will genuinely produce this exact divergence.

    `resolution.trials[-1]`'s own `result` (the third element of each `(params,
    VLRRResult, result)` tuple `StepVerifier._trials` collects) is what the loop's own
    FINAL `tool_runner` call produced on **every exit path EXCEPT `max_iters_reached`**
    (task 038 QC N1 -- an independent QC of the geometry release candidate falsified this
    function's own earlier claim that the invariant holds UNCONDITIONALLY; corrected
    here, not silently re-asserted). Read (not modified) `step_verifier.py` directly to
    confirm the exact mechanism: `resolve_stage`'s loop (`:100-181`) appends the CURRENT
    trial to `history` at `:103`, THEN (if no exit condition fires) calls `tool_runner`
    again at `:179` to produce the NEXT trial's own result -- assigned to the loop-local
    `result` variable, NOT yet appended to `history` (that append happens at the START of
    the NEXT iteration, `:103` again). On six of the seven `return` statements inside
    `resolve_stage` (`:124` satisfied, `:128` exclude_samples, `:137` conceded/verifier-
    emitted, `:159` conceded/hypothesis_invalidated, `:169` conceded/judge_no_improvement,
    `:176` conceded/oscillation_stop), the `return` sits INSIDE the loop body, strictly
    BEFORE `:179` would run for that iteration -- so no extra `tool_runner` call ever
    happens after the trial that triggers the return, and `history[-1]`'s own result IS
    the true, on-disk, delivered state. On the SEVENTH, `max_iters_reached` (`:181`,
    reached only when the `for` loop itself is exhausted, i.e. `:179` DID run one more
    time in the FINAL iteration with nothing left to append it), `history[-1]`'s own
    result is the SECOND-TO-LAST `tool_runner` call, while the crop image files
    currently on disk (and the metadata file describing them) reflect the LAST one --
    QC proved this against the real loop (a 4-call run: disk held call_4, `trials[-1][2]`
    held call_3).

    **The fix** (chosen over documenting the gap, per the coordinator's own "prefer a
    cheap honest fix" instruction -- `tool.py`'s own crop IMAGE files use deterministic,
    overwritten-in-place names, but its METADATA JSON files do NOT: `tool.execute()`
    writes a freshly `uuid4`-suffixed `cell_crops_metadata_<hash>.json` on EVERY call
    into the SAME `tool_cache/` directory, so unlike the crop images, every call's own
    metadata file survives on disk, not just the latest one): read the METADATA FILE
    WITH THE NEWEST MTIME from the SAME `tool_cache/` directory `trials[-1][2]`'s own
    metadata path already lives in (`_newest_metadata_file`, above), rather than trusting
    `trials[-1][2]`'s own path unconditionally. On the six exit paths where the
    invariant already held, `trials[-1][2]`'s own metadata path IS already the newest
    file (nothing to correct -- confirmed by the existing regression tests, unchanged
    behavior). On `max_iters_reached`, the newest file is the FINAL, unrecorded
    `tool_runner` call's own output -- exactly the delivered on-disk state -- so this
    closes the gap rather than merely documenting it. Only `cell_crops_metadata_path`
    is patched onto a COPY of `trials[-1][2]`'s own dict (confirmed via `grep`: every
    consumer in THIS file reads only that one key off a `converged_result`, so no other
    field needs reconciling). Falls back to `trials[-1][2]` unchanged whenever its own
    metadata path is missing/unreadable or already the newest file (the common case,
    and every existing hand-built/fake-path test in `test_run_shape_crop.py`, which
    point at nonexistent paths on purpose) -- and to `resolution.result` only if
    `trials` is somehow empty (defensive; `resolve_stage` always appends at least one
    trial before returning on every real code path)."""
    if not resolution.trials:
        return resolution.result
    last_result = resolution.trials[-1][2]
    metadata_path = last_result.get("cell_crops_metadata_path") if isinstance(last_result, dict) else None
    if not metadata_path or not Path(metadata_path).exists():
        return last_result
    newest = _newest_metadata_file(Path(metadata_path).parent)
    if newest is None or str(newest) == str(metadata_path):
        return last_result
    return {**last_result, "cell_crops_metadata_path": str(newest)}


def run_maskcrop_arm(tool: Single_Cell_Cropper_Tool, image_path: Path, mask_path: Path,
                     out_dir: Path, min_area: int, margin: int,
                     maskcrop_margin_frac: float) -> dict:
    """MaskCrop, the deterministic CONTROL: ONE `execute()` call, naive fixed margin (20%
    of equivalent diameter), legacy bbox-midpoint centering, no retune, no triage -- the
    "no agent" foil."""
    return tool.execute(
        original_image=str(image_path), nuclei_mask=str(mask_path),
        min_area=min_area, margin=margin, margin_frac=maskcrop_margin_frac,
        margin_mode="fraction", recenter=False, query_cache_dir=str(out_dir),
    )


def run_shape_arm(tool: Single_Cell_Cropper_Tool, image_path: Path, mask_path: Path,
                  out_dir: Path, min_area: int, initial_margin: int,
                  naive_margin_frac: float, engine):
    """SHAPE, the FULL AGENTIC flow (task 038 follow-up, part B redesign, human
    directive -- FAIR START): starts from the EXACT SAME naive geometry as MaskCrop's
    own control arm (margin_mode="fraction", recenter=False, `naive_margin_frac` -- the
    SAME value `run_maskcrop_arm` uses) -- so SHAPE's measured gain is the PURE agentic
    contribution, not a "starts already-optimal" head start. The crop-stage VLRR loop
    (`CropVerifier` + `StepVerifier.resolve_stage`) then runs the triage vision judge
    (`vision_recommend_crop`, inside `CropVerifier.adjudicate`, when `engine` is not
    None) -- which now ALSO reasons, per KEPT crop, whether/how to move THAT cell toward
    the tool's own tight+recenter geometry (Part A's per-cell optimum -- still present in
    `tool.py`, no longer the SHAPE arm's OWN starting default) via a PER-CELL
    `cell_overrides` retune (`params["cell_overrides"]`, accumulated across iterations --
    see `CropVerifier.adjudicate`'s own docstring for the mechanics). Direction is
    TIGHTEN/RECENTER only, per cell, never a global/aggregate retune (the earlier,
    REMOVED `margin_frac` widening lever's own mistake).

    Returns the `StageResolution`. `crop_triage.adapt_crop` is passed AS-IS as the
    adapter (its own `(result_or_path, seg_metrics=None)` signature already matches
    `StepVerifier`'s `Adapter` shape) -- not wrapped or reimplemented.
    """
    def tool_runner(params: dict) -> dict:
        return tool.execute(
            original_image=str(image_path), nuclei_mask=str(mask_path),
            min_area=min_area, margin=params["padding"], margin_frac=naive_margin_frac,
            margin_mode="fraction", recenter=False, query_cache_dir=str(out_dir),
            cell_overrides=params.get("cell_overrides", {}),
        )

    initial_params = {"padding": float(initial_margin), "cell_overrides": {}}
    initial_result = tool_runner(initial_params)
    verifier = CropVerifier()
    verifier.llm_engine = engine
    return StepVerifier().resolve_stage(
        verifier=verifier, adapter=adapt_crop, tool_runner=tool_runner,
        initial_params=initial_params, initial_result=initial_result,
    )


def run_shape_agentic_arm(tool: Single_Cell_Cropper_Tool, image_path: Path, mask_path: Path,
                          out_dir: Path, min_area: int, initial_margin: int,
                          naive_margin_frac: float, engine, *, cores: int | None = None,
                          deadline: float | None = None):
    """SHAPE's agentic crop loop (`CropAgenticVerifier`, the shared `base.AgenticVerifier`; CARD.md
    "Agentic VRR"): trial 0 is the input (the image and the mask, measured), round 1 is the tool at its
    defaults (the admission's call: a measured problem or the agent's major problem starts the tuning, else
    that call is delivered), every later call is the agent's, each call writes into its own
    `output_subdir`, and the delivered call is copied into `out_dir/tool_cache/` (`resolution.result`
    points at the delivered metadata). The tool defaults of a move from "input" are MaskCrop's own geometry
    (`initial_margin`, `naive_margin_frac`, fraction mode, no recenter). Without an engine, or when a reasoner
    question fails every try, the loop raises (there is no deterministic fallback). Writes the agent's log
    (`agent_log.jsonl`: every prompt's text, answer, rejection and stop) and the selection details
    (`agent_details.json`) into `out_dir`. Returns the `StageResolution`."""
    from shape.agent.verification.verifiers.crop import CropAgenticVerifier, CropTask, input_record

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "agent_log.jsonl"

    def log(kind, **fields):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **fields}, default=str) + "\n")

    def tool_runner(params: dict) -> dict:
        result = tool.execute(original_image=str(image_path), nuclei_mask=str(mask_path),
                              query_cache_dir=str(out_dir), **params)       # the result carries its seconds
        if result.get("error") or not result.get("cell_count"):
            raise RuntimeError(str(result.get("error") or "the tool produced no crops"))
        return result

    base_kwargs = {"original_image": str(image_path), "nuclei_mask": str(mask_path), "min_area": min_area,
                   "margin": initial_margin, "margin_frac": naive_margin_frac}
    verifier = CropAgenticVerifier(log=log)
    verifier.llm_engine = engine
    resolution = StepVerifier(max_iters=1 + verifier.budget).resolve_stage(
        verifier, verifier.make_adapter(base_kwargs), tool_runner, {}, input_record(base_kwargs),
        upstream_ctx={"crop_task": CropTask(cores=cores, deadline=deadline)})
    (out_dir / "agent_details.json").write_text(json.dumps(
        {"status": resolution.status, "reason": resolution.reason, "parents": resolution.parents,
         **resolution.details}, indent=2, default=str), encoding="utf-8")
    return resolution


def _organize_kept_rejected_crops(out_dir: Path, execute_result: dict) -> None:
    """Physically separate a result's own crops into browsable `kept/`/`rejected/`
    subdirs under `out_dir` (task 038 follow-up, Item 2 -- SAVE + ORGANIZE crops): today
    all crops sit together in `tool_cache/` and kept/rejected exists only inside a
    manifest JSON, not directly browsable (esp. for eyeballing the bulk of blur rejects).
    COPIES (not moves -- `tool_cache/` stays the single authoritative source both arms'
    own metadata/qc still point at) each crop file by its own basename.

    MaskCrop's arm never triages at all (its own `execute()` call never goes through
    `CropVerifier`), so its `metadata_path` has no kept-crops manifest sidecar --
    `_read_kept_paths` then returns `None`, meaning "no filtering, everything kept" -- so
    this SAME function correctly organizes BOTH arms with zero special-casing: MaskCrop's
    crops all land in `kept/`, `rejected/` stays empty (still created, for a predictable
    directory shape either arm produces).

    Also writes `result_manifest.json` (`final_metadata_path`/`kept_crop_paths`/
    `rejected_crop_paths`) -- the manifest-based bookkeeping is KEPT, this is additive,
    not a replacement (a human or script can still cross-reference the machine-readable
    form). Non-destructive, best-effort (any failure -> no files, never raises)."""
    metadata_path = execute_result.get("cell_crops_metadata_path")
    if not metadata_path or not Path(metadata_path).exists():
        return
    try:
        meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        all_paths = [m.get("crop_path") for m in meta.get("cell_metadata", []) or []
                    if m.get("crop_path") and Path(m["crop_path"]).exists()]
        kept_paths = _read_kept_paths(metadata_path)
        kept = all_paths if kept_paths is None else [p for p in all_paths if p in kept_paths]
        rejected = [] if kept_paths is None else [p for p in all_paths if p not in kept_paths]

        kept_dir, rejected_dir = out_dir / "kept", out_dir / "rejected"
        kept_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)
        for p in kept:
            shutil.copy2(p, kept_dir / Path(p).name)
        for p in rejected:
            shutil.copy2(p, rejected_dir / Path(p).name)

        (out_dir / "result_manifest.json").write_text(json.dumps({
            "final_metadata_path": metadata_path,
            "kept_crop_paths": kept,
            "rejected_crop_paths": rejected,
        }, indent=2))
    except Exception as exc:
        print(f"WARNING: failed to organize kept/rejected crops under {out_dir}: {exc}",
              file=sys.stderr)


def process_one_image(tool: Single_Cell_Cropper_Tool, image_path: Path, mask_path: Path,
                      crops_dir: Path, min_area: int, margin: int,
                      naive_margin_frac: float,
                      engine) -> tuple[dict, dict]:
    """Both arms on ONE image. Returns (record, trace):
      - `record`: ready for `sm.compute_metrics(..., metric_key="crop_quality_score")`
        PLUS the raw per-crop score lists (for the headline quality mean/delta).
      - `trace`: the full-process JSONL trace entry for this image (VLRR trials +
        SHAPE's own final per-crop triage verdicts + tokens/time).

    `naive_margin_frac` (task 038 follow-up, part B redesign -- FAIR START): the ONE
    naive `margin_frac` value BOTH arms start from (`run_maskcrop_arm`'s own CONTROL
    geometry, and now `run_shape_arm`'s own STARTING point too) -- a single shared
    parameter, not two that could drift apart, so the two arms are structurally
    guaranteed to start identical."""
    maskcrop_dir = crops_dir / "maskcrop"
    shape_dir = crops_dir / "shape"

    maskcrop_result = run_maskcrop_arm(
        tool, image_path, mask_path, maskcrop_dir, min_area, margin, naive_margin_frac)
    maskcrop_scores = _crop_quality_scores_for_result(maskcrop_result, kept_only=False)
    maskcrop_mean = float(np.mean(maskcrop_scores)) if maskcrop_scores else None
    _organize_kept_rejected_crops(maskcrop_dir, maskcrop_result)

    tok0 = token_usage.total_tokens()
    t0 = time.perf_counter()
    resolution = run_shape_arm(
        tool, image_path, mask_path, shape_dir, min_area, margin,
        naive_margin_frac, engine)
    seconds = time.perf_counter() - t0
    tokens = token_usage.total_tokens() - tok0

    # per-trial mean crop_quality_score (kept-only, REAL pixel-inclusive sharpness --
    # NOT `CropVerifier`'s own mask-only in-loop proxy, see `crop_qc`'s docstring), INDEX
    # -ALIGNED with `resolution.trials` (None for a trial with zero scoreable crops) so
    # the trajectory below can zip it 1:1; `call_scores` (compute_metrics' own dense,
    # gap-free list) drops those trials instead, unchanged.
    trial_mean_scores: list[float | None] = []
    call_scores = []
    for _params, _vres, result in resolution.trials:
        scores = _crop_quality_scores_for_result(result, kept_only=True)
        mean_score = float(np.mean(scores)) if scores else None
        trial_mean_scores.append(mean_score)
        if scores:
            call_scores.append(mean_score)
    # Task 038 Round A2: `converged_result` (the loop's own LAST trial, NOT `_best`'s
    # scalar re-ranking -- see `_converged_result`'s own docstring for the full
    # reasoning) replaces every `resolution.result` read below -- this is the state
    # actually delivered on disk, and the v13 design's own acceptance authority.
    converged_result = _converged_result(resolution)
    shape_final_all = _crop_quality_scores_for_result(converged_result, kept_only=False)
    shape_final_kept = _crop_quality_scores_for_result(converged_result, kept_only=True)

    # task 038 v13 (Honest reporting): MECHANISM metrics, over the SAME kept-only
    # population selection the (non-authoritative) quality score above already uses --
    # see the module docstring's own "QUALITY vs. MECHANISM" section.
    maskcrop_contamination = _qc_values_for_result(maskcrop_result, "neighbor_contamination",
                                                    kept_only=False)
    shape_contamination = _qc_values_for_result(converged_result, "neighbor_contamination",
                                                 kept_only=True)
    shape_retention = _qc_values_for_result(converged_result, "target_retention", kept_only=True)

    _organize_kept_rejected_crops(shape_dir, converged_result)

    # task 038 follow-up (item 3 part 4): per-iteration TRAJECTORY -- global params,
    # this trial's own mean crop_quality_score, and the judge's reasoning (`notes`,
    # already collected) -- so the dynamic optimization process is visible in
    # trace.jsonl, mirroring the segmenter's own AP50 trajectory.
    trials_trace = [
        {"params": params, "decision": vres.decision, "action": vres.action,
         "expected_effect": vres.expected_effect, "notes": vres.notes,
         "mean_crop_quality_score": trial_mean_scores[i]}
        for i, (params, vres, _result) in enumerate(resolution.trials)
    ]
    final_metadata_path = converged_result.get("cell_crops_metadata_path")
    crop_verdicts = _read_triage_trace(final_metadata_path) if final_metadata_path else []

    record = {
        "image_id": image_path.stem,
        "baseline_crop_quality_score": maskcrop_mean,   # compute_metrics' fixed field name
        "call_crop_quality_score": call_scores,          # compute_metrics' fixed field name
        "tokens": tokens,
        "seconds": seconds,
        "n_crops_maskcrop": len(maskcrop_scores),
        "n_crops_shape_final_total": len(shape_final_all),
        "n_crops_shape_final_kept": len(shape_final_kept),
        "n_vlrr_trials": len(resolution.trials),
        "vlrr_status": resolution.status,
        "maskcrop_scores": maskcrop_scores,
        "shape_final_scores": shape_final_kept,
        # task 038 v13 (Honest reporting) -- MECHANISM metrics, the primary reported
        # content this round (see the module docstring's own "QUALITY vs. MECHANISM").
        "maskcrop_contamination": maskcrop_contamination,
        "shape_contamination": shape_contamination,
        "shape_retention": shape_retention,
    }
    trace = {
        "image_id": image_path.stem,
        "tokens": tokens,
        "seconds": seconds,
        "vlrr_trials": trials_trace,
        "vlrr_status": resolution.status,
        "crop_verdicts": crop_verdicts,
    }
    return record, trace


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception as exc:
        return f"UNKNOWN ({exc})"


def _run_solver_mode(args) -> None:
    """Alternate entry point: orchestrate per-image cropping through shape.solver.solve().

    Unlike the default arm A/B/C loop (which injects GT masks directly), this path
    hands the image to the SHAPE planner and lets it call Single_Cell_Cropper_Tool
    autonomously — no GT mask is used, matching a real deployment scenario.

    Writes solver_mode_summary.json to args.out.
    """
    from shape.solver import solve  # noqa: PLC0415 — lazy to avoid circular import at module load
    from cropper_config import DEFAULT as cfg  # noqa: PLC0415 — sys.path patched above for hyphenated dir

    pairs = _find_image_mask_pairs(args.data_dir, args.label_suffix, args.label_ext)
    if not pairs:
        raise SystemExit(f"no image/label pairs found under {args.data_dir}")

    rng = np.random.RandomState(args.seed)
    if args.sample_cap and len(pairs) > args.sample_cap:
        idx = rng.choice(len(pairs), size=args.sample_cap, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]

    args.out.mkdir(parents=True, exist_ok=True)
    model = getattr(args, "model", None) or cfg.llm_model
    results = []
    for image_path, _mask_path in pairs:
        t0 = time.perf_counter()
        result = solve(
            question=cfg.solver_question,
            image_path=str(image_path),
            llm_engine_name=model,
            enabled_tools=list(cfg.solver_enabled_tools),
            max_steps=cfg.solver_max_steps,
            query_cache_dir=str(args.out / "solver_cache"),
        )
        elapsed = time.perf_counter() - t0
        results.append({
            "image_id": image_path.stem,
            "elapsed_s": round(elapsed, 3),
            "direct_output": result.get("direct_output", ""),
        })
        print(f"  [{image_path.stem}] done in {elapsed:.1f}s")

    summary = {
        "mode": "use_solver",
        "model": model,
        "n_images": len(results),
        "results": results,
    }
    out_path = args.out / "solver_mode_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[use_solver] {len(results)} images -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="NeurIPS22-CellSeg root containing images/ and labels/ subfolders.")
    ap.add_argument("--out", required=True, type=Path, help="Output dir for the CSV/JSON evidence.")
    ap.add_argument("--sample-cap", type=int, default=30,
                    help="Cap on number of IMAGES processed (every candidate cell within a "
                         "sampled image is cropped by both arms; default 30).")
    ap.add_argument("--model", default=None,
                    help="Multimodal LLM id for the triage judge. "
                         "Default (None) falls back to config.py (llm_model).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the image sample.")
    ap.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA)
    ap.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    ap.add_argument("--maskcrop-margin-frac", type=float, default=DEFAULT_MASKCROP_MARGIN_FRAC,
                    help="The ONE naive starting margin_frac BOTH arms use (task 038 "
                         "follow-up, part B redesign, FAIR START -- SHAPE's arm now "
                         "starts identical to MaskCrop's own control geometry).")
    ap.add_argument("--label-suffix", default="_label")
    ap.add_argument("--label-ext", default=".tiff")
    ap.add_argument("--use-solver", action="store_true",
                    help="Route through shape.solver.solve() (no GT mask). "
                         "Default off: uses the direct VLRR arm A/B/C loop with GT masks.")
    args = ap.parse_args()

    if args.use_solver:
        return _run_solver_mode(args)

    pairs = _find_image_mask_pairs(args.data_dir, args.label_suffix, args.label_ext)
    if not pairs:
        raise SystemExit(f"no image/label pairs found under {args.data_dir}")

    rng = np.random.RandomState(args.seed)
    if args.sample_cap and len(pairs) > args.sample_cap:
        idx = rng.choice(len(pairs), size=args.sample_cap, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "crops").mkdir(parents=True, exist_ok=True)
    engine = build_vision_engine(args.model)
    tool = Single_Cell_Cropper_Tool()

    records = []
    traces = []
    n_failed = 0
    for image_path, mask_path in pairs:
        try:
            crops_dir = args.out / "crops" / image_path.stem
            record, trace = process_one_image(
                tool, image_path, mask_path, crops_dir, args.min_area, args.margin,
                args.maskcrop_margin_frac, engine)
            records.append(record)
            traces.append(trace)
        except Exception as exc:
            print(f"WARNING: failed on {image_path.name}: {exc}", file=sys.stderr)
            n_failed += 1

    if not records:
        raise SystemExit("no records produced -- check --data-dir / --min-area")

    # --- FULL-PROCESS TRACE (default on, task 038 follow-up) -----------------------------
    trace_path = args.out / "trace.jsonl"
    with open(trace_path, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    # --- per-image CSV (recomputable raw evidence) --------------------------------------
    csv_path = args.out / "per_image_crop_agentic.csv"
    fieldnames = ["image_id", "baseline_crop_quality_score", "call_crop_quality_score",
                  "tokens", "seconds", "n_crops_maskcrop", "n_crops_shape_final_total",
                  "n_crops_shape_final_kept", "n_vlrr_trials", "vlrr_status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fieldnames})

    # --- QUALITY (task 038 v13: NON-AUTHORITATIVE -- see "Honest reporting" below) -------
    # crop_quality_score per crop -> mean per arm. STILL computed (harmless,
    # informational) but explicitly NOT the evidence this script uses to claim SHAPE's
    # crops are "better" -- see the module docstring's own "QUALITY vs. MECHANISM".
    all_maskcrop_scores = [s for r in records for s in r["maskcrop_scores"]]
    all_shape_scores = [s for r in records for s in r["shape_final_scores"]]
    maskcrop_quality_mean = float(np.mean(all_maskcrop_scores)) if all_maskcrop_scores else None
    shape_quality_mean = float(np.mean(all_shape_scores)) if all_shape_scores else None
    quality_delta = (shape_quality_mean - maskcrop_quality_mean
                     if maskcrop_quality_mean is not None and shape_quality_mean is not None
                     else None)

    # --- MECHANISM (task 038 v13, Honest reporting -- the PRIMARY reported content) ------
    # neighbor_contamination reduction (SHAPE's own final-kept population vs. MaskCrop's),
    # target_retention distribution (SHAPE's own kept population -- how close to the
    # tool's own `_TARGET_RETENTION_FLOOR` the tightened crops actually landed).
    # `tool_calls_per_image`/`token_cost_per_image`/`time_per_image_s` (the EFFICIENCY
    # section below) are the AGGREGATE VLM cost across BOTH channels (reject + A/B) --
    # this script has no separate per-channel call/abstain instrumentation (would need
    # verifier-internal changes, out of this round's own surgical scope).
    all_maskcrop_contamination = [c for r in records for c in r["maskcrop_contamination"]]
    all_shape_contamination = [c for r in records for c in r["shape_contamination"]]
    all_shape_retention = [v for r in records for v in r["shape_retention"]]
    maskcrop_contamination_mean = (float(np.mean(all_maskcrop_contamination))
                                   if all_maskcrop_contamination else None)
    shape_contamination_mean = (float(np.mean(all_shape_contamination))
                                if all_shape_contamination else None)
    contamination_reduction = (maskcrop_contamination_mean - shape_contamination_mean
                               if maskcrop_contamination_mean is not None
                               and shape_contamination_mean is not None else None)
    shape_retention_mean = float(np.mean(all_shape_retention)) if all_shape_retention else None
    shape_retention_min = float(np.min(all_shape_retention)) if all_shape_retention else None

    # --- EFFICIENCY: reuse compute_metrics, generalized (task 038) -----------------------
    efficiency_records = [
        {"image_id": r["image_id"],
         "baseline_crop_quality_score": r["baseline_crop_quality_score"],
         "call_crop_quality_score": r["call_crop_quality_score"],
         "tokens": r["tokens"], "seconds": r["seconds"]}
        for r in records
    ]
    efficiency_metrics = sm.compute_metrics(efficiency_records, metric_key="crop_quality_score")

    summary = {
        "provenance": {
            "driver": "run_shape_crop.py",
            "git_commit": _git_commit(_REPO_ROOT),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "engine_available": engine is not None,
            "data_dir": str(args.data_dir),
            "n_images_available": len(_find_image_mask_pairs(args.data_dir, args.label_suffix, args.label_ext)),
            "n_images_processed": len(records),
            "n_images_failed": n_failed,
            "sample_cap": args.sample_cap, "seed": args.seed,
            "trace_path": str(trace_path),
            "crops_dir": str(args.out / "crops"),
        },
        "parameters": {
            "min_area": args.min_area, "margin": args.margin,
            "maskcrop": {"margin_mode": "fraction", "margin_frac": args.maskcrop_margin_frac,
                        "recenter": False},
            # Task 038 follow-up (part B redesign, FAIR START): SHAPE's arm now starts
            # from this SAME naive geometry as MaskCrop's own -- the per-cell
            # `cell_overrides` retune (agent-driven, see `CropVerifier.adjudicate`) then
            # moves individual crops toward tight+recenter (or further, for a swamped
            # cell) over the course of the VLRR loop; see trace.jsonl's own vlrr_trials
            # for each image's actual per-iteration cell_overrides state.
            "shape": {"margin_mode_initial": "fraction",
                     "margin_frac_initial": args.maskcrop_margin_frac,
                     "recenter_initial": False},
        },
        # Task 038 v13 (Honest reporting, human decision): MECHANISM is the PRIMARY
        # reported content -- contamination reduction, retention distribution, plus
        # the SAME cost fields "efficiency" already reports.
        # See the module docstring's own "QUALITY vs. MECHANISM" section.
        "mechanism": {
            "n_crops_maskcrop": len(all_maskcrop_contamination),
            "n_crops_shape_kept": len(all_shape_contamination),
            "maskcrop_neighbor_contamination_mean": maskcrop_contamination_mean,
            "shape_neighbor_contamination_mean": shape_contamination_mean,
            "neighbor_contamination_reduction": contamination_reduction,
            "shape_target_retention_mean": shape_retention_mean,
            "shape_target_retention_min": shape_retention_min,
            "tool_calls_per_image": efficiency_metrics["tool_calls_per_image"],
            "token_cost_per_image": efficiency_metrics["token_cost_per_image"],
            "time_per_image_s": efficiency_metrics["time_per_image_s"],
            "vlm_cost_note": (
                "tool_calls_per_image/token_cost_per_image/time_per_image_s aggregate "
                "BOTH VLM channels (reject + the blinded A/B gate) combined -- this "
                "script has no separate per-channel call/abstain instrumentation."),
        },
        # Task 038 v13 (Honest reporting, human decision): kept for informational/
        # diagnostic purposes ONLY -- `crop_quality_score` is DISQUALIFIED as a
        # validator (CARD.md's own "Task 038 v13 SCOPE"/"Honest reporting" sections);
        # `authoritative: False` is the explicit, machine-readable marker that this
        # section is NOT evidence SHAPE's crops are "better" -- the "better" verdict is
        # deferred to a separate SSL/analyzer validation task, not answered here.
        "quality": {
            "authoritative": False,
            "note": ("NON-authoritative mechanism descriptor -- crop_quality_score is "
                    "disqualified as a validator (task 038 v13). Do NOT read a delta "
                    "here as evidence of 'better crops'; see 'mechanism' above and "
                    "CARD.md's own 'Honest reporting' section."),
            "n_crops_maskcrop": len(all_maskcrop_scores),
            "n_crops_shape": len(all_shape_scores),
            "maskcrop_crop_quality_score_mean": maskcrop_quality_mean,
            "shape_crop_quality_score_mean": shape_quality_mean,
            "quality_delta": quality_delta,
        },
        "efficiency": efficiency_metrics,
    }
    (args.out / "run_shape_crop_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"n_images={len(records)}")
    print(f"mechanism: neighbor_contamination maskcrop={maskcrop_contamination_mean}  "
          f"shape={shape_contamination_mean}  reduction={contamination_reduction}")
    print(f"mechanism: shape_target_retention mean={shape_retention_mean}  "
          f"min={shape_retention_min}")
    print(f"mechanism: tool_calls_per_image={efficiency_metrics['tool_calls_per_image']}  "
          f"token_cost_per_image={efficiency_metrics['token_cost_per_image']}  "
          f"time_per_image_s={efficiency_metrics['time_per_image_s']}")
    print(f"efficiency (metric_key=crop_quality_score, full set): "
          f"positive_tool_call_rate={efficiency_metrics['positive_tool_call_rate']}  "
          f"wasted_tool_call_rate={efficiency_metrics['wasted_tool_call_rate']}  "
          f"crop_quality_score_gain_per_tool_call={efficiency_metrics['crop_quality_score_gain_per_tool_call']}")
    print(f"quality (NON-authoritative, see 'mechanism' above): maskcrop={maskcrop_quality_mean}  "
          f"shape={shape_quality_mean}  delta={quality_delta}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
