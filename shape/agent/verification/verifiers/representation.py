"""Verifier 5: Representation (DINOv3, model not tuned; INPUT strategy tuned).

DINOv3 weights are frozen, so `retune_here` adjusts what we feed it (masked vs
full crop) and how we post-process (regress out area), not model hyperparams.
Two hard upgrades over the original:
  - PC1_area_correlation > 0.7 is NOT a soft "interpret size effect" -> it
    mandatorily regresses out area, because size dominance poisons every
    downstream morphological analysis.
  - batch-driven embeddings escalate upstream to find the batch source, rather
    than stopping at a WARNING in this step.

Two verifiers: the linear `RepresentationVerifier` and `RepresentationAgenticVerifier`, the
analyzer's `base.AgenticVerifier` (the agent tunes the self-supervised fine-tune per input; StepVerifier
wave mode), with the stage context it reads (`RepresentationTask`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Optional

import numpy as np
from pydantic import BaseModel, Field

from ..loaders import _codes, _load_adata
from shape.utils.schemas import Hypothesis, LeverChange, VLRRResult, WaveMove, WavePlan, WaveSelection

from ..base import (INPUT, AgenticVerifier, BaseVerifier, batch_dominates_group, config_key, fmt, normalize,
                    numeric, weighted_score)


def adapt_representation(adata_path: str, context=None):
    """Analyzer h5ad -> ((emb, areas, batch, group), context).

    Folded out of the shared adapters bridge (Phase 2b). emb = adata.X,
    areas/batch/group from adata.obs (written by the analyzer). Missing areas fall
    back to a constant so PC1_area_correlation is simply 0.
    """
    adata = _load_adata(adata_path)
    emb = np.asarray(adata.X, dtype=np.float64)
    n = emb.shape[0]
    if "area" in adata.obs:
        areas = np.asarray(adata.obs["area"], dtype=np.float64)
        areas = np.nan_to_num(
            areas,
            nan=float(np.nanmean(areas)) if np.isfinite(np.nanmean(areas)) else 0.0)
    else:
        areas = np.zeros(n)
    batch = _codes(adata.obs["batch"]) if "batch" in adata.obs else np.zeros(n, dtype=int)
    # SSL selection must not inspect biological treatment labels. The unified analyzer
    # writes well/site/source-derived technical_group explicitly for this verifier.
    group_key = "technical_group" if "technical_group" in adata.obs else None
    group = _codes(adata.obs[group_key]) if group_key else np.zeros(n, dtype=int)
    return (emb, areas, batch, group), dict(context or {})


_TRIAL_CONFIG_KEYS = ("augment_color", "field_aware", "adaptation_depth")


def trial_context(adata_path: str | None, base_kwargs: dict | None,
                  trial_kwargs: dict | None = None,
                  report: dict | None = None) -> dict:
    """Per-trial adapter context enrichment (task 008 P3, additive-only -- called by
    `Verifier._adapter_for`'s representation branch, NOT a change to `adapt_representation`
    itself). With `base_kwargs` supplied (the core-routed retrain loop), recompute
    `nn_visual_consistency` from THIS trial's own adata -- reusing
    `crop_vision.nn_visual_consistency_judge`'s deterministic pixel-proxy fallback (no LLM
    call here; the LLM enters only at `adjudicate`, per the VLRR contract) -- so each retrain
    is judged on its OWN embedding instead of the pre-P3 default of 1.0, and attach the
    up-front `crop_vision_profile` for the mid-loop recommend. With no `base_kwargs` (every
    pre-P3 caller, e.g. `run_shape_hpa.verify_representation`) this returns {} -- byte-identical
    to before (nn_visual_consistency then defaults to 1.0 in `quantify`).

    `trial_kwargs` (task 021 Stage 1b): THIS trial's actual analyzer `execute()` kwargs (the
    core-routed retrain runner's `result["analyzer_kwargs"]`, `stage_map.map_params`'s output --
    same key names `adjudicate`'s batch_driven floor already reads via `context.get(...)`).
    Threading it in fixes the no-op-retune bug: without it, `adjudicate` never learns a floor
    lever (e.g. augment_color) was already turned on by the UP-FRONT recommend, and re-proposes
    it, producing an identical-params retune that `StepVerifier` then kills as `oscillation_stop`
    with zero effective retunes (see THREAD.md's P4 root-cause). Only the 4 keys `adjudicate`/
    the mid-loop prompt reason about are forwarded; everything else in `analyzer_kwargs`
    (learning_rate, max_epochs, compute_alignment, ...) is irrelevant to this decision."""
    bk = base_kwargs or {}
    if not adata_path:
        return {}
    ctx: dict = {}
    if bk:
        from ..crop_vision import nn_visual_consistency_judge
        if "crop_vision_profile" in bk:
            ctx["crop_vision_profile"] = bk["crop_vision_profile"]
        nn = nn_visual_consistency_judge(None, adata_path)
        if nn.get("available"):
            ctx["nn_visual_consistency"] = nn["score"]
    if report:
        ctx["report"] = report
    tk = trial_kwargs or {}
    for key in _TRIAL_CONFIG_KEYS:
        if key in tk:
            ctx[key] = tk[key]
    return ctx


def representation_qc(emb: np.ndarray, areas: np.ndarray,
                      batch: np.ndarray, group: np.ndarray) -> dict[str, float]:
    nan_frac = float(np.isnan(emb).mean())
    clean = np.nan_to_num(emb)
    var = clean.var(0)
    # single SVD of the centered embedding, reused for both effective rank and PC1
    centered = clean - clean.mean(0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    # effective rank via participation ratio of singular values
    eff_rank = float((s.sum() ** 2) / (np.square(s).sum() + 1e-12))
    pc1 = centered @ vt[0]
    # guard against degenerate variance (e.g. a collapsed/constant PC1), which
    # would make np.corrcoef divide by zero and return NaN
    if areas.std() > 0 and pc1.std() > 0:
        pc1_area = float(np.corrcoef(pc1, areas)[0, 1])
    else:
        pc1_area = 0.0
    technical_silhouette = _silhouette(clean, group)
    return {
        "NaN_fraction": nan_frac,
        "feature_variance_mean": float(var.mean()),
        "effective_rank": eff_rank,
        "PC1_area_correlation": pc1_area,
        "silhouette_by_batch": _silhouette(clean, batch),
        "silhouette_by_technical_group": technical_silhouette,
        # Backward-compatible trace key. It has always occupied the fourth argument
        # position, but unified SSL now makes its technical meaning explicit above.
        "silhouette_by_group": technical_silhouette,
    }


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    # cheap silhouette proxy: between-group dist / within-group dist, mapped to ~[-1,1].
    # Singleton groups have no within-group cohesion to measure -- counting them drives the
    # score toward +1 artifactually (their within-distance is 0), which made batch labels
    # with ~one member per group (e.g. single-cell-per-field source IDs) read as strong
    # batch structure and falsely trip the batch_driven gate. So compute the proxy ONLY over
    # groups with >= 2 members (the convention that silhouette is undefined for singletons),
    # and return 0.0 when fewer than two such groups remain.
    uniq, counts = np.unique(labels, return_counts=True)
    multi = uniq[counts >= 2]
    if len(multi) < 2:
        return 0.0
    keep = np.isin(labels, multi)
    x, labels = x[keep], labels[keep]
    cents = {u: x[labels == u].mean(0) for u in multi}
    within = np.mean([np.linalg.norm(x[i] - cents[labels[i]]) for i in range(len(x))])
    between = np.mean([np.linalg.norm(cents[a] - cents[b])
                       for a in multi for b in multi if a != b])
    return float((between - within) / (max(between, within) + 1e-9))


# --------------------------------------------------------------------------- #
# Label-free SSL model selection (no cell-line labels, even for selection)
# --------------------------------------------------------------------------- #
def ssl_quality_components(adata_path: str, *, source_key: str = "image_name") -> dict:
    """Per-embedding LABEL-FREE quality signals for picking among SSL training configs.

    Reuses representation_qc so the selection metrics ARE the verifier's metrics:
      - effective_rank  (RankMe-style; higher = better-used representation, Garrido et al.),
      - source_silhouette = silhouette_by_batch with batch := the source-image id
        (lower = less field/batch domination; uses the source id, NEVER the cell line).
    No cell-line label is touched. Comparable across configs trained on the same data."""
    adata = _load_adata(adata_path)
    emb = np.asarray(adata.X, dtype=np.float64)
    n = emb.shape[0]
    source = _codes(adata.obs[source_key]) if source_key in adata.obs else np.zeros(n, dtype=int)
    m = representation_qc(emb, np.zeros(n), source, np.zeros(n, dtype=int))
    return {
        "effective_rank": float(m["effective_rank"]),
        "source_silhouette": float(m["silhouette_by_batch"]),
        "n": int(n),
    }


def select_by_ssl_score(components: list[dict], *,
                        rank_weight: float = 1.0, batch_weight: float = 1.0) -> dict:
    """Pick the best config purely label-free: z-score effective_rank (higher better) and
    source_silhouette (lower better) across configs, combine, argmax. With <2 configs or no
    variance in a term, that term drops out (z-score -> 0). Returns the chosen index, the
    per-config scores, and the components used."""
    ranks = np.array([c["effective_rank"] for c in components], dtype=float)
    sils = np.array([c["source_silhouette"] for c in components], dtype=float)

    def _z(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)

    scores = rank_weight * _z(ranks) - batch_weight * _z(sils)
    best = int(np.argmax(scores)) if len(scores) else -1
    return {
        "best_index": best,
        "scores": scores.tolist(),
        "rank_weight": rank_weight,
        "batch_weight": batch_weight,
    }


# --------------------------------------------------------------------------- #
# Label-free training-health (for the pre-flight lr/epoch calibration; see the
# Cell_State_Analyzer_Tool CARD.md). Soft priors, not magic numbers in the benchmark.
# --------------------------------------------------------------------------- #
_DIVERGE_TOL = 1.02        # final loss above initial*this => diverging (lr too high)
_OSCILLATE_TOL = 6.0       # std(diffs)/|mean(diffs)| above this => oscillating (lr too high)
_PLATEAU_FRAC_MAX = 0.25   # late-window drop below this fraction of total => converged
_MIN_REL_DROP = 0.05       # need at least this relative loss drop to count as trained


def _effective_rank(emb: np.ndarray) -> float:
    x = np.nan_to_num(np.asarray(emb, dtype=np.float64))
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    s = np.linalg.svd(x - x.mean(0), compute_uv=False)
    return float((s.sum() ** 2) / (np.square(s).sum() + 1e-12))


def loss_curve_health(loss_history) -> dict:
    """Label-free training-dynamics health from the per-epoch loss curve. Flags map to a
    direction the lr/epoch recommend uses: diverging/oscillating => lr down; trained-but-not-
    plateaued => epochs up; tiny total drop => lr up. score in [0,1], higher = healthier."""
    h = np.asarray([x for x in (loss_history or []) if np.isfinite(x)], dtype=np.float64)
    if h.size < 2:
        return {"score": 0.0, "n": int(h.size), "diverging": False, "oscillating": False,
                "converged": False, "underfit": True, "relative_drop": 0.0,
                "plateau_frac": 1.0, "oscillation_ratio": 0.0}
    initial, final = float(h[0]), float(h[-1])
    scale = abs(initial) + 1e-9
    total_drop = initial - final
    relative_drop = total_drop / scale
    late = h[-max(2, h.size // 2):]
    plateau_frac = abs(float(late[0] - late[-1])) / (abs(total_drop) + 1e-9)
    diffs = np.diff(h)
    osc = float(np.std(diffs) / (abs(np.mean(diffs)) + 1e-9))
    diverging = final > initial * _DIVERGE_TOL
    oscillating = osc > _OSCILLATE_TOL
    converged = (relative_drop > _MIN_REL_DROP and plateau_frac < _PLATEAU_FRAC_MAX
                 and not diverging and not oscillating)
    underfit = not converged and not diverging and not oscillating
    drop_term = float(np.clip(relative_drop / 0.3, 0.0, 1.0))
    plateau_term = float(np.clip(1.0 - plateau_frac, 0.0, 1.0))
    score = drop_term * plateau_term
    if diverging:
        score *= 0.1
    if oscillating:
        score *= 0.5
    return {"score": float(score), "n": int(h.size), "diverging": bool(diverging),
            "oscillating": bool(oscillating), "converged": bool(converged),
            "underfit": bool(underfit), "relative_drop": float(relative_drop),
            "plateau_frac": float(plateau_frac), "oscillation_ratio": float(osc)}


def training_health_score(loss_history, emb=None, alignment=None, *,
                          eff_rank_min: float = 10.0) -> dict:
    """Composite LABEL-FREE health for pre-flight lr/epoch calibration: loss-curve health +
    embedding non-collapse (effective rank) + augmentation/NN consistency (alignment). No
    cell-line labels. Higher = a more stably-trained, non-collapsed embedding."""
    loss = loss_curve_health(loss_history)
    parts = {"loss": loss["score"]}
    eff_rank = None
    if emb is not None:
        eff_rank = _effective_rank(emb)
        parts["rank"] = normalize(eff_rank, 50.0, eff_rank_min)
    if alignment is not None:
        parts["alignment"] = float(np.clip(alignment, 0.0, 1.0))
    return {"score": float(weighted_score(parts, {k: 1.0 for k in parts})),
            "loss_curve": loss, "effective_rank": eff_rank,
            "alignment": (None if alignment is None else float(alignment)), "parts": parts}


class RepresentationVerifier(BaseVerifier):
    name = "representation"
    tool_dir = "cell_state_analyzer"   # unified analyzer CARD documents the recommend
    # Opt-in to the same CARD-driven LLM recommend the segmentation verifier uses (mirrors
    # `segmentation.py:116-117`). `llm_engine` is injected by the integration layer when
    # enable_llm_verifier is on; when it is None the verifier stays purely deterministic.
    uses_llm = True
    llm_engine = None

    # config-declared gates (soft priors), not inline literals in localize()
    _VARIANCE_COLLAPSE = 1e-3
    _PC1_AREA_MAX = 0.7

    def __init__(self, eff_rank_min: float = 10.0, nn_consistency_thr: float = 0.5):
        self.eff_rank_min = eff_rank_min
        self.nn_thr = nn_consistency_thr
        # task 008 Thread A / task 021: tracks whether THIS verifier's own retune loop has
        # already turned field_aware (the reliable-source floor lever) or augment_color (the
        # unreliable-source floor lever) on. One RepresentationVerifier instance is
        # constructed per `resolve_stage` call and reused for every retune iteration within it
        # (never shared across stage resolutions), so this is safe per-loop state -- it lets
        # `adjudicate` detect "the floor lever already tried" on a later call even where the
        # caller's `context` does not (yet) carry the live per-trial value.
        # `context["field_aware"]`/`context["augment_color"]` still win when a caller/adapter
        # DOES supply them (see `adjudicate` below).
        self._field_aware_on = False
        self._augment_color_floor_on = False

    def quantify(self, tool_output, context) -> dict[str, float]:
        emb, areas, batch, group = tool_output
        m = representation_qc(emb, areas, batch, group)
        m["nn_visual_consistency"] = context.get("nn_visual_consistency", 1.0)
        m["embedding_reproducibility"] = context.get("embedding_reproducibility", 1.0)
        return m

    def localize(self, metrics, context):
        triggered = []
        if metrics["NaN_fraction"] > 0 or (
                metrics["effective_rank"] < self.eff_rank_min
                and metrics["feature_variance_mean"] < self._VARIANCE_COLLAPSE):
            triggered.append("collapse")
            return "upstream", triggered, []   # collapse = bad input/preproc, not DINOv3
        if abs(metrics["PC1_area_correlation"]) > self._PC1_AREA_MAX:
            triggered.append("size_dominated")
            return "this_step", triggered, []  # fixable by regressing area
        if batch_dominates_group(
                metrics["silhouette_by_batch"],
                metrics.get("silhouette_by_technical_group",
                            metrics.get("silhouette_by_group", 0.0))):
            triggered.append("batch_driven")
            return "upstream", triggered, []   # find batch source upstream
        if metrics["nn_visual_consistency"] < self.nn_thr:
            triggered.append("low_consistency")
            return "this_step", triggered, []
        return "this_step", triggered, []

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        technical_silhouette = metrics.get(
            "silhouette_by_technical_group", metrics.get("silhouette_by_group", 0.0))
        parts = {
            "rank": normalize(metrics["effective_rank"], 50.0, self.eff_rank_min),
            "size_indep": normalize(abs(metrics["PC1_area_correlation"]), 0.0, 0.7),
            "technical_struct": normalize(technical_silhouette, 0.5, 0.0),
        }
        score = weighted_score(parts, {k: 1.0 for k in parts})

        # collapse is a mandatory CONCEDE, not a this-step retune/accept candidate --
        # `_clamp_action` only ranks "accept"/"retune_here" (segmentation never routes its
        # analogous upstream/tool_mismatch concedes through the vision tail either, see
        # segmentation.py's locus=="upstream"/"tool_mismatch" branches), so it returns
        # directly and stays a floor regardless of any wired llm_engine.
        if "collapse" in triggered:
            return VLRRResult("FAIL", score, {}, action="conceded", fault="upstream",
                              upstream_target="image_loading",
                              notes="Embedding collapse (NaN or rank<min): the fault "
                                    "is input loading, not the frozen model.")
        # task 021: batch_driven is now EVIDENCE + a deterministic FLOOR, not a terminal
        # hard-coded script (the old task-008-Thread-A behavior this retires -- see
        # CODING_LOG). `det` below is built exactly like the size_dominated/low_consistency
        # cases and does NOT `return` directly: it falls through to the SAME shared LLM tail
        # every other locus already uses, so the LLM always gets to reason
        # input-features -> best lever / when to concede, per SHAPE's core thesis (no
        # hard-coded per-input policy). The floor lever is gated on the up-front
        # `_source_reliable` check (`initial_params._source_reliable` -- the SAME helper the
        # up-front recommend uses, single source of truth): field_aware needs a trustworthy
        # per-crop source id (real well/plate metadata + multi-cell fields) or it is
        # near-inert (masking a field of ~1 crop is close to a no-op); an unreliable source
        # instead gets augment_color (still a valid per-image intensity decorrelator on
        # single-channel data, per the CARD). `_clamp_action` still lets the LLM only
        # confirm/tighten this floor, never loosen it.
        if "batch_driven" in triggered:
            from ..initial_params import _source_reliable
            src_ok = _source_reliable(context.get("report") or {})
            if src_ok:
                lever = "field_aware"
                tried = bool(context.get("field_aware", self._field_aware_on))
            else:
                lever = "augment_color"
                tried = bool(context.get("augment_color", self._augment_color_floor_on))
            if not tried:
                if lever == "field_aware":
                    self._field_aware_on = True
                    reason = ("source id is reliable (real field/well metadata, multi-cell "
                              "fields) -> field-aware contrastive (mask same-source-field "
                              "crops from negatives)")
                else:
                    self._augment_color_floor_on = True
                    reason = ("source id is NOT reliable (filename-guessed, mostly "
                              "singleton fields) -> field_aware would be near-inert; "
                              "augment_color's brightness/contrast jitter is still a valid "
                              "per-image intensity decorrelator")
                det = VLRRResult(
                    "WARNING", score, {}, action="retune_here",
                    param_delta={lever: True},
                    expected_effect={"silhouette_by_batch": "down"},
                    notes=f"Batch silhouette > group: {reason}.")
            else:
                # the floor lever already tried and the batch confound still persists:
                # accept the trained gain (keep-best still returns it) and flag data-side,
                # rather than masking the confound with a further retune or a
                # collapse-style escalation.
                det = VLRRResult(
                    "WARNING", score, {}, action="accept",
                    notes=f"Batch silhouette still > group with {lever} already on: "
                          "confound persists -> go multi-channel / data-side, not a "
                          "further retune here.")
        # ...this_step quality zone: build the deterministic recommendation first...
        elif "size_dominated" in triggered:
            # Masking non-target pixels is the CROPPER's job, not this stage's: the analyzer
            # only ever receives finished crops. The former `input_strategy=use_masked_crop`
            # retune proposed here was a no-op -- the tool accepted the kwarg, echoed it into
            # `uns`, and never branched on it -- so the loop paid a full fine-tune and then
            # invalidated its own hypothesis when the metric did not move. Concede upstream.
            det = VLRRResult(
                "WARNING", score, {}, action="conceded", fault="upstream",
                upstream_target="crop",
                notes="PC1 dominated by cell size: the crops carry too much non-target "
                      "signal. Fixable by tighter/masked cropping upstream, plus "
                      "regress-out-area in post -- not by any lever at this stage.")
        elif "low_consistency" in triggered:
            det = VLRRResult(
                "WARNING", score, {}, action="conceded", fault="upstream",
                upstream_target="crop",
                notes="NN visual mismatch: neighbours in feature space do not look alike, "
                      "which points at crop content rather than at a training setting "
                      "reachable from this stage.")
        else:
            det = VLRRResult("PASS", score, {}, action="accept")

        # ...then let the LLM confirm/tighten (never loosen, per `_clamp_action`) the
        # accept-vs-retune decision, same as the segmentation verifier (`segmentation.py`
        # `:249-254`). Falls back to `det` on any failure or when no engine is wired.
        if getattr(self, "llm_engine", None) is not None:
            from ..llm_recommend import vision_recommend_representation
            return vision_recommend_representation(self.llm_engine, context, metrics, det,
                                                   card=self.tuning_card())
        return det


# --------------------------------------------------------------------------- #
# Agentic VRR (`RepresentationAgenticVerifier`; StepVerifier wave mode)
# --------------------------------------------------------------------------- #
# The loop itself is `base.AgenticVerifier`; below is only the analyzer's own part: trial 0 (the frozen
# backbone, `frozen_params`), the measurability gate, the readouts on the val wells (`quant`, the tool's
# measurement module: the adopted proxy on the comparison draws and the ALARMS; every other output is logged
# only), the paired 2 SE noise, the guard, the resource stop, the lineage of init_from = parent, each call's
# own directory, the prompt text, and the delivery of the chosen embedding -- trial 0's whenever no call is
# delivered.

SCORE = "sibling_site_map"           # the replicate tier's adopted proxy (quant.ADOPTED_PROXY), on the val wells
CHANCE = f"{SCORE}_chance"           # its value for a random ranking of the same queries
ALARMS = ("effective_rank", "dead_dim_frac", "image_leakage")   # shown, labelled as alarms; no rule reads them
SHOWN = (SCORE, CHANCE, *ALARMS)     # the readouts the prompts show; every other output is logged only
TRAINING_FACTS = ("epochs_trained", "best_epoch", "best_by", "batch_size", "trainable_blocks")
EARLY_STOP_FACTS = ("stop_reason", "epochs_saved", "overfit_gap_best", "overfit_gap_last")   # only when present
PROXY_K = 2                          # the proxy's site profiles: cells per site (decision P)
PROXY_DRAWS = 20                     # the proxy's seeded draws of those cells
LEAKAGE_K = 5                        # image_leakage: the nearest neighbours of a cell
MIN_VAL_WELLS = 2                    # the gate: the val wells every plate needs
SELECTION_CHECK_TOL = 0.01           # the selection-draw recompute vs val_history's best (logged only)
RESERVE_MINUTES = 15                 # kept before the deadline for the delivery (a call's time cap)
CALLS_DIR = "ssl_calls"              # the trials' own directories, inside <query_cache_dir>/cell_state_analysis/
RESUME_NAME = "resume_best.pt"       # a call's resume state at its best val point (the tool's file name)
CONTINUATION = "continuation"        # the role of a forced continuation (A12), its calls continuation_<n>
# the tool's outputs a trial's record keeps; never shown
_LOGGED_KEYS = ("training", "val_history", "split", "final_loss", "loss_history", "resume_best", "resume_end",
                "cluster_key", "clustering_representation")
# a lever value -> the metadata column every crop needs for it (tested as the tool does, `quant.identities`)
_COLUMN_OF = {("view_pairing", "same_site"): "site", ("view_pairing", "same_well"): "well",
              ("contrastive_mask", "same_site"): "site", ("contrastive_mask", "same_well"): "well",
              ("batch_sampling", "per_plate"): "plate", ("batch_sampling", "per_experiment"): "experiment",
              ("intensity_norm", "per_plate"): "plate"}
# the deterministic fallback (no engine), after the tool at its defaults: each step resumes the previous step's call
_FALLBACK_CHAIN = (
    (("view_pairing", "same_well"), ("contrastive_mask", "same_well"), ("trainable_pct", 33), ("learning_rate", 2e-5)),
    (("init_from", "parent"), ("trainable_pct", 50)),
    (("init_from", "parent"), ("trainable_pct", 75), ("learning_rate", 1e-5)),
)


@dataclass
class RepresentationTask:
    """What the caller knows about the representation stage (``upstream_ctx["representation_task"]``)."""
    cores: int                           # cores for the tool calls
    gpus: int                            # GPUs the process sees (the calls still run one at a time: `concurrency`)
    deadline: Optional[float]            # epoch seconds by which the stage must end (e.g. the job's limit)
    brief: str                           # a one-line task description shown to the agent
    split_key: Optional[str] = "split"   # the metadata key holding "train" / "val"; None -> the tool's val_fraction
    val_fraction: float = 0.1            # with split_key None: the tool's share of val wells per plate


class FinalChoice(BaseModel):
    """The final choice (``keep_input``): a call, or null to deliver the frozen input (trial 0)."""
    call: Optional[str] = Field(None, description="the chosen call id, or null to deliver the frozen input")
    reasoning: str
    cited_readouts: list[str] = Field(default_factory=list, description="the readouts the choice rests on")


class BasedChange(LeverChange):
    """A lever change of an SSL move with its basis (amendment A12): the shared change and why it is made. An empty
    basis is rejected by the verifier's validation (`_basis_problem`), not by the schema, so it reaches the repair
    question."""
    basis: str = Field("", description="why this change: an observation from this run's readouts (name the call "
                                       "and the readout) or a testable hypothesis")


class BasedMove(WaveMove):
    changes: list[BasedChange] = Field(
        default_factory=list, description='the levers changed; none from "input" = the tool at its defaults')


class BasedPlan(BaseModel):
    """The SSL round's plan: the shared ``WavePlan`` without focus_levers (no narrowing, A12) and with based moves."""
    analysis: str
    hypotheses: list[Hypothesis] = Field(default_factory=list, description="your current hypothesis list")
    moves: list[BasedMove] = Field(description="the calls of this round; none = satisfied, stop")
    deliver: Optional[str] = Field(
        None, description="with no moves: the finished call to deliver; it must meet the final rule")


class BasedFocusPlan(BasedPlan):
    """`BasedPlan` with the shared ``WavePlan.focus_levers``: the plan of the opt-in four-stage search
    (``SHAPE_SEARCH_STRATEGY_REPRESENTATION=four_stage``), whose narrow stage is the focus; the legacy search keeps
    `BasedPlan` (no narrowing, A12)."""
    focus_levers: list[str] = WavePlan.model_fields["focus_levers"]


def split_params(task: Any = None) -> dict:
    """The split every call of the task gets, trial 0's included: the task's split key, else the tool's own val
    wells (val_fraction); without a task, RepresentationTask's defaults."""
    split_key = getattr(task, "split_key", RepresentationTask.split_key)
    return {"split_key": split_key} if split_key else {
        "val_fraction": getattr(task, "val_fraction", RepresentationTask.val_fraction)}


def frozen_params(task: Any = None) -> dict:
    """Trial 0's tool kwargs: the frozen pretrained backbone (zero_shot) on all crops, with the task's split, in its
    own directory (`ssl_calls/input`, as the benchmark names it; the canonical cell_state_analysis/ paths stay free
    for the delivery)."""
    return {"adaptation_depth": "zero_shot", **split_params(task), "output_subdir": f"{CALLS_DIR}/{INPUT}"}


def visible_gpus() -> int:
    """The GPUs this process sees (torch's CUDA device count; 0 without CUDA)."""
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 -- no torch or no CUDA: no GPU
        return 0


def _link(source: str, target) -> None:
    """`target` as a hard link to `source` (a copy where the filesystem has none), replacing what is there."""
    if os.path.abspath(source) == os.path.abspath(target):
        return
    if os.path.lexists(target):
        os.remove(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _r4(value):
    return round(float(value), 4) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


def _number(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class RepresentationAgenticVerifier(AgenticVerifier):
    """Agentic VRR for Cell_State_Analyzer_Tool (``base.AgenticVerifier``, StepVerifier wave mode): the agent tunes
    the self-supervised fine-tune per input -- which configurations, from which parent, how many, when to stop and
    which call to deliver; rules measure and guard. Not registered in ``stage_map`` (no ``verifier_by_tool``
    entry): a benchmark runs it.

    Trial 0 is the tool at zero_shot (the frozen pretrained backbone) on all crops with the task's split
    (`frozen_params`), run by the caller: the input every call is compared with, and itself a valid output
    (``keep_input``). Every readout is measured on the val wells -- a label-free split by well; the training sees
    only the train wells -- with the tool's measurement module (``quant``), on each embedding's val rows in the
    order of the tool's validation set (the val crops in input order, matched by cell_id): the tier's adopted proxy,
    ``sibling_site_map``, with its chance level, on the comparison draws (``quant.FRESH_SEED``), never on the
    selection draws (``quant.SELECTION_SEED``) that picked each call's best point; as ALARMS the collapse indicators
    of ``quant.health`` (effective_rank, dead_dim_frac) and ``quant.image_leakage``; a call's training facts and
    seconds are its report. That is what the prompts show (`input_text`, `_rows`); every other output (the proxy on
    the selection draws and its check against val_history, val_history, the losses, the gradient norms,
    mean_cosine, uniformity, the clustering, the resume paths) is logged in the selection details only.

    The admission is on trial 0 (no ``admission_call``), a measurability gate (`gate`): the replicate tier
    (``quant.tier_of``, the tier with an adopted proxy), MIN_VAL_WELLS val wells on every plate and a query unit of
    the frozen proxy. It fails -> SKIP: trial 0 is delivered, satisfied, "not_measurable" (`select_on_input`). It
    holds -> the search, whose round 1 is the shared round prompt: an empty round-1 plan is the agent's judgement
    that this input needs no tuning, and trial 0 is delivered, satisfied, "no_tuning_needed" (`select`).

    The noise of a comparison is the paired 2 SE of the proxy over the val units both trials have (a bootstrap over
    wells, ``quant.beats``), in the facts (`_delta`, `_prediction`) and in the plateau (`_stall`). The guard: a
    finite, non-constant embedding whose proxy beats the frozen input's by more than 2 SE. The resource stop
    (`recommend_wave`): no call when the time left for one cannot hold a call's non-training part. Whenever the
    shared selection delivers no call while trial 0 finished, trial 0 is delivered (`select`); the details record
    the decision (`_decision`).

    A call's params are the tool's kwargs: the CARD levers (``## Levers``), init_from = parent as the parent call's
    resume state at its best val point (``init_checkpoint``, `extend_move`; a move from a finished call resumes it
    unless it sets init_from, Amendment A7), the call's own directory
    (``output_subdir``, named by its configuration key, which includes the parent's key for a continuation:
    `config_key`), the task's split and the time left before the deadline (``max_minutes``). No narrowing and no
    forced coverage of the CARD Families (amendment A12: ``narrowing`` off). The round prompt states the goal
    (`goal_text`); every lever change of the agent's moves needs a basis (`BasedPlan`, `_accept`); a stop while the
    last finished call still improved on its parent is refused (`stop_check_text`), and an empty answer to the
    refusal runs a continuation of that call (`forced_move`). With a reasoner, a plateau does not stop the search and
    starts no forced exploration (amendment A20, which withdraws A13's): whether to continue, go back to an earlier
    call, change direction or stop is the agent's; the round budget is the backstop (`_plateau`).
    One call runs at a time (`concurrency`). The delivered trial's embedding and resume state are linked to the
    canonical cell_state_analysis/ paths (`deliver`).

    Expert start (amendments A18, A19; default off): with `expert_config` ([round-1 {lever: value}, round-2 {lever:
    value}]), the round-1 and round-2 prompts state it (`admission_text`); a round-k move changes only levers of
    rounds 1..k, to their expert values (init_from aside), and its configuration has every one of them at its expert
    value (so a round-2 move from a parent other than a round-1 call sets both rounds'); a move that does not is
    rejected through the repair question (`_expert_problem`). A round entry after the first may also hold
    ``"init_from": "parent"``: that round's move resumes a finished call of the previous round (init_from parent from
    that call; when none finished, the move sets every expert value as above)."""

    name = "representation"
    tool_dir = "cell_state_analyzer"
    lever_section = "## Levers"
    score_key = SCORE
    key_readouts = (SCORE,)
    keep_input = True                    # trial 0, the frozen embedding, is a valid output (delivered when no call is)
    admission_on_input = True
    admission_skip_reason = "not_measurable"
    narrowing = False                    # A12: no broad-then-narrow rule, no forced coverage of the CARD Families
    plan_schema = BasedPlan              # A12: every lever change carries a basis
    choice_schema = FinalChoice
    tool_phrase = "self-supervised fine-tune of a DINOv3 backbone (the cell-state analyzer, contrastive)"
    unit = "set of cell crops"
    value_examples = '0.5, 10, true, null, "off"'

    def __init__(self, *, log: Callable[..., None] | None = None,
                 expert_config: Sequence[Mapping[str, Any]] | None = None,
                 early_stop: Mapping[str, Any] | None = None,
                 search_evidence_policy: str = 'legacy', **kwargs):
        from ..representation_policy import validate_policy
        self.search_evidence_policy = validate_policy(search_evidence_policy)
        self.stop_evidence: list[dict] = []
        super().__init__(log=log, **kwargs)
        # Search-level early stopping (default off, `main`'s behaviour): with {"plateau_rounds": R}, a stop is
        # refused only while the last finished call beat its parent by more than 2 SE (not by any size, A12), and
        # the search stops after R rounds in a row without a call that beats the best so far by more than 2 SE
        # (the shared plateau backstop, `_stall`, which A20 turns off with a reasoner).
        self.early_stop: dict | None = dict(early_stop) if early_stop else None
        if self.early_stop is not None:
            rounds = int(self.early_stop.get("plateau_rounds", 2))
            if rounds < 1:
                raise ValueError(f"early_stop plateau_rounds must be >= 1, got {rounds}")
            self.early_stop["plateau_rounds"] = self.patience = rounds
        if self._search is not None:                    # the four-stage search narrows by focus_levers
            from ..four_stage import staged_schema

            self.plan_schema = staged_schema(BasedFocusPlan)
        self.expert_config: list[dict] = [dict(r) for r in expert_config or ()]   # A19, default off: per round
        if any("init_from" in r and (not i or r["init_from"] != "parent") for i, r in enumerate(self.expert_config)):
            raise ValueError('an expert_config entry may set only init_from "parent", and not in round 1')
        self.expert_rejections: list[dict] = []         # the moves the expert start rejected (A18)
        self.gpus = 1
        self.split = split_params()                     # the task's split kwargs (`begin`)
        self.tier: str | None = None                    # quant.tier_of of the crops (trial 0)
        self.proxy: str | None = None                   # the tier's adopted proxy (quant.ADOPTED_PROXY)
        self.split_facts: dict = {}                     # the split sizes and the val wells per plate (trial 0)
        self.setup: dict = {}                           # the proxy's k and draws (quant.sibling_site_map)
        self._meta: list = []                           # the crops' metadata (the tool's cell_metadata)
        self._ids: list = []                            # quant.identities of the crops (plate-qualified)
        self._val: list[int] = []                       # the tool's validation set: the val crops in input order
        self._units: dict[int, dict] = {}               # trial -> the proxy's per-unit values, comparison draws
        self._logged: dict[int, dict] = {}              # trial -> its logged-only outputs (never in a prompt)
        self._resume: dict[int, Optional[str]] = {}     # trial -> its resume state at the best val point
        self._overhead: dict[int, float] = {}           # trial -> its non-training minutes (seconds - train_seconds)
        self._unusable: dict[int, str] = {}             # trial -> why it gave no measurable embedding
        self._index_of: dict[int, int] = {}             # id(trial VLRRResult) -> trial index
        self._pairs: dict[tuple, dict] = {}             # (a, b) -> quant.beats of trial b over trial a
        self._parent_keys: dict[str, str] = {}          # a resume state -> the key of the call that wrote it
        self._search_training: dict[int, dict] = {}
        self.stop_refusals: list[dict] = []             # the stops refused while the last call improved (A12)
        self.forced_continuations: list[dict] = []      # the continuations run after a refusal met no moves (A12)
        self._basis_of: dict[str, dict] = {}            # config key -> {lever: basis} of the move that proposed it

    def begin(self, task: Any = None) -> None:
        """Trial 0: also the GPUs (the task's, else the visible ones) and the task's split."""
        gpus = getattr(task, "gpus", None)
        self.gpus = int(gpus) if gpus else visible_gpus()
        self.split = split_params(task)
        super().begin(task)

    def concurrency(self) -> int:
        """One call at a time, a hardware limit: the tool runs every call on the first GPU and seeds process-global
        random streams. It lifts (one call per GPU) when the tool can pin each call to its own device."""
        return 1

    @staticmethod
    def trial0_call(base_kwargs, query_cache_dir=None, request=None) -> tuple[dict, dict, dict]:
        """Trial 0 on the solve() path (`integration.verify_managed_tool`): (base kwargs, trial 0's kwargs, stage
        context). The base kwargs add the crops and their metadata as the tool loads them from the query cache
        (its `_load_cell_data_from_metadata`: the kept-crops manifests applied, the technical identifiers the
        image specs gave and nothing else), so every call and the readouts see the same rows; when the loader
        finds no crops they are left out and trial 0 fails in the tool as before. The task: this process's
        cores and GPUs, no deadline, the request as the brief, and the tool's val wells at
        RepresentationTask.val_fraction when the crops' plates and wells allow them
        (`quant.assign_val_wells`), else no split (the gate then skips). Trial 0's kwargs: `frozen_params`."""
        from shape.tools.cell_state_analyzer import quant
        from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

        from ..base import available_cores

        kwargs = dict(base_kwargs or {})
        try:
            crops, meta, _ = Cell_State_Analyzer_Tool()._load_cell_data_from_metadata(
                query_cache_dir or "solver_cache/temp")
            kwargs.update(cell_crops=crops, cell_metadata=meta)
        except ValueError:
            meta = []
        val_fraction = RepresentationTask.val_fraction if meta else None
        try:
            if val_fraction is not None:
                quant.assign_val_wells(meta, val_fraction)
        except ValueError:
            val_fraction = None
        task = RepresentationTask(cores=available_cores(), gpus=visible_gpus(), deadline=None,
                                  brief=" ".join(str(request or "").split()), split_key=None,
                                  val_fraction=val_fraction)
        return kwargs, frozen_params(task), {"representation_task": task}

    @staticmethod
    def make_adapter(base_kwargs):
        """The native result is the verifier input; the context also carries the crops' metadata (the tool's
        `cell_metadata`), which the readouts group by plate, well and site."""
        meta = list((base_kwargs or {}).get("cell_metadata") or [])
        return lambda result, upstream: (result, {**dict(upstream or {}), "cell_metadata": meta})

    # ---- Verify -------------------------------------------------------------------------------------------
    def quantify(self, tool_output, context) -> dict[str, float]:
        """Trial 0 (the frozen input): the clock, the cores, the GPUs and the split (`begin`), the crops' metadata,
        their identities and tier, and the tool's validation set. Every trial: the readouts of its embedding
        (`_measure`); its training facts and seconds are its report (shown), its resume state at the best val point
        is kept for a continuation (`extend_move`), its non-training time for the resource stop, its other outputs
        are logged only."""
        from shape.tools.cell_state_analyzer import quant

        result = tool_output if isinstance(tool_output, Mapping) else {}
        index = int(context.get("trial_index") or 0)
        if index == 0:
            self.begin(context.get("representation_task"))
            self._meta = list(context.get("cell_metadata") or [])
            self._ids = quant.identities(self._meta)
            self.tier = quant.tier_of(self._meta) if self._meta else None
            self.proxy = quant.ADOPTED_PROXY.get(self.tier)
            split = self._split_of_crops()
            self._val = [i for i, part in enumerate(split) if part == "val"]
            self.split_facts = self._split_facts(split)
        training = result.get("training") or {}
        if self._search is not None:
            self._search_training[index] = {k: training.get(k) for k in (
                "steps_trained", "start_step", "end_step", "best_step", "epochs_trained", "batch_size", "train_seconds")}
        self._reports[index] = {**{k: training.get(k) for k in TRAINING_FACTS},
                                **{k: training[k] for k in EARLY_STOP_FACTS if k in training},
                                "seconds": result.get("seconds")}
        self._resume[index] = result.get("resume_best")
        seconds, train_seconds = _number(result.get("seconds")), _number(training.get("train_seconds"))
        if seconds is not None and train_seconds is not None:
            self._overhead[index] = (seconds - train_seconds) / 60
        self._logged[index] = {k: result.get(k) for k in _LOGGED_KEYS if k in result}
        return self._measure(index, result)

    def _split_of_crops(self) -> list[str]:
        """Each crop's split as the tool resolves it (3.3): the task's split key's value, else the tool's val wells
        (``quant.assign_val_wells``, the tool's rule for val_fraction)."""
        from shape.tools.cell_state_analyzer import quant

        key = self.split.get("split_key")
        if key:
            return [str(m.get(key)) if isinstance(m, Mapping) else "" for m in self._meta]
        if self.split.get("val_fraction") is None:
            return ["train"] * len(self._meta)
        return list(quant.assign_val_wells(self._meta, self.split["val_fraction"]))

    def _split_facts(self, split: list[str]) -> dict:
        """The split sizes (n_train, n_val, n_val_wells) and the val wells of every plate."""
        val_wells = {(self._ids[i]["plate"], self._ids[i]["well"]) for i in self._val}
        plates = {r["plate"] for r in self._ids}
        return {"n_train": split.count("train"), "n_val": len(self._val), "n_val_wells": len(val_wells),
                "val_wells_per_plate": {str(p): sum(1 for q, _ in val_wells if q == p) for p in plates}}

    def _measure(self, index: int, result: Mapping[str, Any]) -> dict[str, float]:
        """The readouts of a trial's embedding (its h5ad): whether it is finite and not constant and, in a tier with
        the adopted proxy, on its val rows in the tool's validation order: the proxy on the comparison draws (its
        per-unit values kept for the paired comparisons) and the ALARMS; logged only: the proxy on the selection
        draws, its check against val_history's best entry, mean_cosine and uniformity. A result without a readable
        h5ad, whose rows do not match the crops' metadata, or a frozen input that is not finite, is unusable
        (`call_error`)."""
        from shape.tools.cell_state_analyzer import quant

        path = result.get("adata_path")
        if result.get("error") or not path or not os.path.exists(path):
            self._unusable[index] = str(result.get("error") or f"no h5ad written ({path})")[-300:]
            return {}
        try:
            adata = _load_adata(path)
        except Exception as exc:  # noqa: BLE001 -- an unreadable h5ad is an unusable trial, not the end of the loop
            self._unusable[index] = f"unreadable h5ad: {type(exc).__name__}: {exc}"[-300:]
            return {}
        rows = self._crop_rows(adata)
        if rows is None:
            self._unusable[index] = "the embedding's rows do not match the crops' metadata"
            return {}
        x = np.asarray(adata.X, dtype=np.float64)
        finite = bool(np.isfinite(x).all())
        if not finite and index == 0:
            self._unusable[0] = "the frozen embedding is not finite"
            return {}
        metrics = {"embedding_finite": float(finite),
                   "embedding_varies": float(finite and x.size > 0 and bool((np.ptp(x, axis=0) > 0).any()))}
        self._units[index] = {}
        if self.proxy != SCORE or not finite or not self._val:
            return metrics
        vectors, meta = x[rows[self._val]], [self._meta[i] for i in self._val]
        proxy = quant.sibling_site_map(vectors, meta, k=PROXY_K, draws=PROXY_DRAWS, seed=quant.FRESH_SEED)
        selection = quant.sibling_site_map(vectors, meta, k=PROXY_K, draws=PROXY_DRAWS, seed=quant.SELECTION_SEED)
        self._units[index] = dict(proxy.get("units") or {})
        self.setup = {"k": proxy.get("k"), "draws": proxy.get("draws")}
        health = quant.health(vectors)
        metrics.update(numeric({SCORE: proxy.get("value"), CHANCE: proxy.get("chance"),
                                "effective_rank": health.get("effective_rank"),
                                "dead_dim_frac": health.get("dead_dim_frac"),
                                "image_leakage": quant.image_leakage(vectors, meta, k=LEAKAGE_K)}))
        check = self._selection_check(selection.get("value"), result.get("val_history"))
        if check["within_tol"] is False:
            self.log("selection_check", text=f"{self.label(index)}: {json.dumps(check)}")
        self._logged[index].update({"mean_cosine": health.get("mean_cosine"), "uniformity": health.get("uniformity"),
                                    f"{SCORE}_selection_draws": selection.get("value"), "selection_check": check})
        return metrics

    def _crop_rows(self, adata) -> Optional[np.ndarray]:
        """The embedding row of each crop, in the crops' (input) order: by cell_id when the rows carry it (the tool's
        stable identity), else the rows in the crops' order; None when they do not match the crops."""
        n = len(self._meta)
        if adata.n_obs != n:
            return None
        if "cell_id" not in adata.obs:
            return np.arange(n)
        row_of = {str(c): r for r, c in enumerate(adata.obs["cell_id"])}
        ids = [str(m.get("cell_id")) if isinstance(m, Mapping) else None for m in self._meta]
        if len(row_of) != n or not all(i in row_of for i in ids):
            return None
        return np.array([row_of[i] for i in ids], dtype=np.int64)

    @staticmethod
    def _selection_check(recomputed, val_history) -> dict:
        """The proxy on the selection draws, recomputed from the h5ad, against val_history's best entry: the h5ad comes
        from a separate embedding pass (other batches, AMP), so they agree only within SELECTION_CHECK_TOL. A record,
        never a call_error, never shown."""
        values = [e.get(SCORE) for e in val_history or [] if isinstance(e, Mapping)]
        best = max((v for v in map(_number, values) if v is not None), default=None)
        diff = abs(recomputed - best) if recomputed is not None and best is not None else None
        return {"recomputed": recomputed, "val_history_best": best, "abs_diff": diff,
                "within_tol": None if diff is None else diff <= SELECTION_CHECK_TOL}

    def localize(self, metrics, context):
        return "this_step", [], []

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        """A call: its proxy as the score. Trial 0: the measurability gate (`gate`) -- RUN starts the search
        (retune_here), SKIP delivers trial 0 (accept); an unusable trial 0 concedes (no output)."""
        index = int(context.get("trial_index") or 0)
        score = float(metrics.get(SCORE, float("nan")))
        if index:
            vres = VLRRResult("PASS" if metrics else "FAIL", score, dict(metrics), action="retune_here")
        elif 0 in self._unusable:
            self.set_admission({"decision": "ABSTAIN", "reasons": [self._unusable[0]]})
            vres = VLRRResult("FAIL", score, dict(metrics), action="conceded", fault="ceiling",
                              notes=f"the frozen input gave no measurable embedding: {self._unusable[0]}")
        else:
            self.set_admission(self.gate())
            run = self.admission["decision"] == "RUN"
            action, fault = self.admission_action()
            vres = VLRRResult("PASS", score, dict(metrics), action=action, fault=fault,
                              notes="the frozen input" + ("" if run else f"; not measurable: "
                                                          f"{'; '.join(self.admission['reasons'])}"))
        self._index_of[id(vres)] = index
        return vres

    def gate(self) -> dict:
        """The measurability gate on trial 0: the replicate tier (the one with an adopted proxy, sibling_site_map),
        at least MIN_VAL_WELLS val wells on every plate, and at least one query unit of the frozen proxy. RUN when all
        hold, else SKIP with the reasons.

        $SHAPE_FORCE_SSL_RUN=1 forces RUN regardless of the above -- an explicit, opt-in escape hatch for a
        dataset whose tier has no adopted proxy at all (ADOPTED_PROXY: batch/image -> None; only "replicate"
        -- >=2 sites on over half the wells -- has one, sibling_site_map). Forcing RUN here does NOT give the
        tuning loop a working proxy: `self.proxy` and `self._units` are unaffected, so every later `_compare`
        (quant.beats) between trials still has no units to compare and reports no measurable improvement.
        This is a genuinely unvalidated experiment, not a safe override -- use it to see what SSL training
        actually produces (judged externally, e.g. by this benchmark's own score()), not because the gate was
        being overly strict."""
        per_plate = self.split_facts.get("val_wells_per_plate") or {}
        units = len(self._units.get(0) or {})
        reasons = []
        if self.proxy != SCORE:
            reasons.append(f"the {self.tier} tier has no adopted proxy")
        else:
            low = {p: n for p, n in per_plate.items() if n < MIN_VAL_WELLS}
            if low or not per_plate:
                reasons.append(f"fewer than {MIN_VAL_WELLS} val wells on a plate (val wells per plate: "
                               f"{low or per_plate})")
            if not units:
                reasons.append("the frozen proxy has no query unit")
        forced = bool(reasons) and os.environ.get("SHAPE_FORCE_SSL_RUN") == "1"
        return {"decision": "RUN" if (not reasons or forced) else "SKIP", "tier": self.tier,
                "adopted_proxy": self.proxy, "val_wells_per_plate": per_plate, "frozen_units": units,
                "reasons": reasons, **({"forced": True} if forced else {})}

    # ---- the noise: the paired 2 SE of each comparison, on the comparison draws ------------------------------
    def _compare(self, a: int, b: int) -> dict:
        """quant.beats of trial b's proxy units over trial a's (cached; both on the comparison draws): delta = b - a
        over the units both have, its se (a bootstrap over wells), n units, and whether b beats a by more than 2 SE;
        not comparable when either trial has no units."""
        if (a, b) not in self._pairs:
            from shape.tools.cell_state_analyzer import quant

            units_a, units_b = self._units.get(a) or {}, self._units.get(b) or {}
            self._pairs[(a, b)] = (quant.beats(units_b, units_a) if units_a and units_b else
                                   {"beats": False, "delta": None, "se": None, "n": 0, "n_wells": 0})
        return self._pairs[(a, b)]

    @staticmethod
    def _paired_text(paired: Mapping[str, Any]) -> str:
        """A paired difference as the prompts show it: b - a +/- 2 SE, beyond or within 2 SE, the shared units."""
        d, se, n = paired.get("delta"), paired.get("se"), paired.get("n", 0)
        if d is None:
            return "no shared val units"
        if se is None:
            return f"{d:+.4f}, no SE (fewer than 2 wells), n {n}"
        return f"{d:+.4f} +/- {2 * se:.4f}, {'beyond' if abs(d) > 2 * se else 'within'} 2 SE, n {n}"

    def _delta(self, before: VLRRResult, after: VLRRResult, key: str) -> str | None:
        """`after - before`: for the proxy, the paired difference over the val units both have, +/- 2 SE
        (`_paired_text`); for another readout, the shared plain difference."""
        if key != SCORE:
            return super()._delta(before, after, key)
        a, b = self._index_of.get(id(before)), self._index_of.get(id(after))
        return None if a is None or b is None else self._paired_text(self._compare(a, b))

    def _prediction(self, parent: VLRRResult, vres: VLRRResult) -> list[dict]:
        """Each stated expectation next to the observed change from the parent (a measurement; whether it failed is
        the agent's judgement): the proxy's paired difference, up or down beyond 2 SE or within; another shown
        readout's plain difference; a readout the prompts do not show is not reported."""
        a, b = self._index_of.get(id(parent)), self._index_of.get(id(vres))
        out = []
        for metric, direction in (vres.expected_effect or {}).items():
            x, y = self._score(parent, metric), self._score(vres, metric)
            if metric not in SHOWN:
                seen = "not a shown readout"
            elif x is None or y is None:
                seen = "not computable"
            elif metric == SCORE and a is not None and b is not None:
                paired = self._compare(a, b)
                d, se = paired.get("delta"), paired.get("se")
                if d is None:
                    seen = "not computable"
                elif se is None:
                    seen = f"{d:+.4f} (no SE)"
                else:
                    seen = f"{d:+.4f} ({'up' if d > 2 * se else 'down' if d < -2 * se else 'within 2 SE'})"
            else:
                seen = f"{y - x:+.4f} ({'moved' if y != x else 'unchanged'})"
            out.append({"metric": metric, "expected": direction, "observed": seen})
        return out

    def _stall(self, inp, attempts) -> int:
        """Rounds in a row without progress: the finished rounds (a round of failed calls included) after the last
        round with a call whose proxy beats the best so far (the frozen input first) by more than 2 SE
        (`_progress`)."""
        return self._progress(attempts)[1]

    def _progress(self, attempts) -> tuple[int, int]:
        """(the best trial so far, the rounds in a row without progress). The best: the frozen input first, then each
        round's best call whose proxy beats it by more than 2 SE."""
        def score(i):
            value = self._score(attempts[self.label(i)][2])
            return -math.inf if value is None else value

        best, stall = 0, 0
        for indices in self.rounds:
            better = [i for i in indices if self.label(i) in attempts and self._compare(best, i)["beats"]]
            if better:
                best, stall = max(better, key=score), 0
            else:
                stall += 1
        return best, stall

    # ---- A20: a plateau is the agent's to judge ------------------------------------------------------------
    def _plateau(self, inp, attempts) -> bool:
        """With a reasoner, never (amendment A20): a plateau neither stops the search nor forces a move; the agent
        decides, the round budget is the backstop. Without one (the deterministic chain), the shared backstop."""
        if self.early_stop is not None:                 # search-level early stopping: the backstop holds
            return super()._plateau(inp, attempts)
        return False if self.llm_engine is not None else super()._plateau(inp, attempts)

    def guard(self, inp: VLRRResult, vres: VLRRResult) -> list[str]:
        """The conditions a final choice fails (empty = admissible): a measured embedding (a finished call that
        wrote an h5ad: `call_error`), finite and not constant, whose val proxy beats the frozen input's by more
        than 2 SE (``quant.beats``, paired over the val units both have, on the comparison draws).

        $SHAPE_FORCE_SSL_RUN=1 drops the "beats the frozen input" requirement specifically when this tier has
        no adopted proxy at all (self.proxy != SCORE -- gate()'s own "no adopted proxy" case): `_compare`
        always returns beats=False here (no sibling_site_map units exist to pair), so without this the guard
        would reject every trial unconditionally and the search would always deliver the frozen input --
        making gate()'s SHAPE_FORCE_SSL_RUN override a no-op in practice (real training runs, nothing it
        produces is ever admissible). The finite/non-constant sanity checks still apply; only the
        unattainable comparison is skipped. Still unvalidated: a trial that clears this relaxed guard is not
        confirmed better than zero-shot, only not obviously broken."""
        m, index = vres.metrics or {}, self._index_of.get(id(vres))
        if not m or index is None:
            return ["no measured embedding"]
        fails = []
        if not m.get("embedding_finite"):
            fails.append("the embedding is not finite")
        elif not m.get("embedding_varies"):
            fails.append("the embedding is constant")
        no_proxy_forced = self.proxy != SCORE and os.environ.get("SHAPE_FORCE_SSL_RUN") == "1"
        if not no_proxy_forced:
            paired = self._compare(0, index)
            if not paired["beats"]:
                fails.append(f"val {SCORE} does not beat the frozen input's by more than 2 SE: "
                             f"{self._paired_text(paired)}")
        return fails

    # ---- the round: the shared one, with the resource stop ------------------------------------------------
    def search_stop_reason(self) -> Optional[str]:
        """The shared loop's resource hook: no call (stop_reason "deadline") when the time a call may
        take (`max_minutes`) is at most 0 or below the shortest non-training time of a finished call (its seconds
        minus its training seconds; trial 0's, all of whose time is non-training, until a call has finished); the
        final question still runs."""
        left = self.max_minutes()
        if left is not None:
            shortest = min((m for i, m in self._overhead.items() if i > 0), default=self._overhead.get(0))
            if left <= 0 or (shortest is not None and left < shortest):
                self.log("deadline", text=f"{left} minute(s) left for a call; the shortest non-training time of a "
                                          f"finished call: {shortest} minute(s)")
                return "deadline"
        return None

    # ---- the loop state in the analyzer's terms ---------------------------------------------------------
    def search_dimensions(self) -> list[str]:
        return [*super().search_dimensions(), "training_history"]

    def search_changed_dimensions(self, delta: Mapping[str, Any]) -> set[str]:
        return set(delta) | ({"training_history"} if "init_from" in delta else set())

    def search_values(self, params: Mapping[str, Any], call: Optional[str]) -> dict:
        values = super().search_values(params, call)
        if params.get("trainable_pct") is not None and params["trainable_pct"] < 100:
            values.pop("patch_embed_lr_mult", None)
        if params.get("view_pairing") != "nn_other_image":
            values = {k: v for k, v in values.items() if not k.startswith("nn_")}
        elif params.get("nn_space") != "current":
            values.pop("nn_refresh_epochs", None)
        if params.get("augmentation") == "ssl116":
            values.pop("augment_color", None)
        if params.get("batch_sampling") == "per_plate":
            values.pop("plate_mean_weight", None)
        resume = params.get("init_checkpoint")
        values["training_history"] = self._parent_keys.get(resume, resume) if resume else "pretrained"
        facts = self._search_training.get(self._labels.get(call), {})
        if facts.get("batch_size") is not None:
            values["batch_size"] = facts["batch_size"]
        return values

    def search_context(self, params: Mapping[str, Any], call: str) -> Optional[dict]:
        facts = self._search_training.get(self._labels.get(call), {})
        required = ("steps_trained", "start_step", "best_step", "batch_size")
        if any(facts.get(k) is None for k in required):
            return None
        return {**{k: facts[k] for k in required}, "seed": params.get("seed"),
                "origin": self.search_values(params, None)["training_history"],
                "validation": hashlib.sha1(json.dumps([self.split, self._val, self.setup],
                                                      sort_keys=True, default=str).encode()).hexdigest()}

    def search_notes(self) -> str:
        return ("SSL candidates are multi-epoch training runs. training_history is a separate dimension.\n"
                "A move from a finished call resumes it unless the move sets init_from, so it also changes\n"
                "training_history; init_from pretrained keeps a fresh anchor's training_history.\n"
                "A resumed parent-child gain includes extra training; compare branches from the SAME checkpoint\n"
                "and inspect actual steps, best-step selection and effective batch size. Missing step telemetry\n"
                "or unequal contexts cannot establish a local control. Use contrast_waiver when unaffordable.\n"
                "The existing comparison draws are search feedback, never independent validation.\n")

    def trial_config(self, params: Mapping[str, Any]) -> dict:
        """A call's configuration from its tool kwargs: the CARD levers -- init_from = parent when it loaded a
        resume state -- and that state (`init_checkpoint`)."""
        config = {lever: params.get(lever) for lever in self.levers}
        config["init_from"] = "parent" if params.get("init_checkpoint") else "pretrained"
        config["init_checkpoint"] = params.get("init_checkpoint")
        return config

    def trial_params(self, config: Mapping[str, Any]) -> dict:
        """The tool kwargs of a configuration: its levers (init_from as the resume state it loads,
        `init_checkpoint`; init_from itself never reaches the tool), its own directory (`output_subdir`:
        ssl_calls/<sha1 of its config_key>[:12]), the task's split and the time cap (`max_minutes`, not part of the
        key)."""
        params = {lever: config.get(lever) for lever in self.levers if lever != "init_from"}
        digest = hashlib.sha1(self.config_key(config).encode()).hexdigest()
        params.update({"init_checkpoint": config.get("init_checkpoint"), "output_subdir": f"{CALLS_DIR}/{digest[:12]}",
                       **self.split, "max_minutes": self.max_minutes()})
        return params

    def config_key(self, params: Mapping[str, Any]) -> str:
        """What makes two configurations the same call (a sha1 hex digest): the CARD levers and, for init_from =
        parent, the key of the call whose resume state it loads -- the same levers from two parents are two calls."""
        resume = params.get("init_checkpoint")
        text = config_key({**{lever: params.get(lever) for lever in self.levers},
                           "parent": self._parent_keys.get(resume, resume)})
        return hashlib.sha1(text.encode()).hexdigest()

    def max_minutes(self) -> Optional[float]:
        """A call's time cap: the minutes left before the deadline minus RESERVE_MINUTES, at least 0; None without a
        deadline."""
        if not self.deadline:
            return None
        return max(0.0, round((self.deadline - time.time()) / 60 - RESERVE_MINUTES, 1))

    def check_change(self, lever: str, value: Any, base: Mapping[str, Any], parent: str) -> None:
        """A value that reads a metadata column needs it on every crop (`_COLUMN_OF`, tested with
        ``quant.identities`` as the tool tests it: `<name>_id` keys accepted, NaN and blanks missing); init_from =
        parent needs a finished call as the parent (the input is the frozen backbone, with no training state).
        Raises ValueError."""
        column = _COLUMN_OF.get((lever, value))
        if column and (not self._ids or any(ids[column] is None for ids in self._ids)):
            raise ValueError(f"{lever}={value} needs the metadata column {column!r} on every crop")
        if lever == "init_from" and value == "parent" and parent == INPUT:
            raise ValueError('init_from=parent needs a finished call as the parent ("input" is the frozen backbone, '
                             "with no training state)")

    def extend_move(self, move: Mapping[str, Any], params: dict, parent: str, attempts) -> dict:
        """init_from (Amendment A7): a move from a finished call resumes that call -- init_from = parent -- unless the
        move sets init_from itself (pretrained: the pretrained backbone with a fresh adapter and projector, the
        parent's other levers kept); a move from the input keeps its init_from (pretrained; parent is rejected by
        `check_change`). init_from = parent: the parent call's resume state at its best val point as
        `init_checkpoint`, the parent's key recorded for `config_key`, and {"init_from": "parent"} as the move's own
        change -- so a pure continuation of a finished call is a move (the path stays out of the prompts);
        pretrained: no state, nothing added. Raises ValueError when the parent left no resume state."""
        if parent != INPUT and not any(c.get("lever") == "init_from" for c in move.get("changes") or []):
            params["init_from"] = "parent"
        if params.get("init_from") != "parent":
            params["init_checkpoint"] = None
            return {}
        resume = self._resume.get(self._labels.get(parent, -1))
        if not resume or not os.path.exists(resume):
            raise ValueError(f"{parent} kept no training state to resume; set init_from pretrained to restart from "
                             "the pretrained backbone")
        params["init_checkpoint"] = resume
        self._parent_keys[resume] = self.config_key(attempts[parent][1])
        return {"init_from": "parent"}

    @staticmethod
    def start_state(parent: str, params: Mapping[str, Any]) -> str:
        """How a call starts, as each arm's ledger records it (Amendment A7): "resumed: <parent>" when it loads its
        parent's resume state (`init_checkpoint`), else "fresh" (the pretrained backbone)."""
        return f"resumed: {parent}" if params.get("init_checkpoint") else "fresh"

    def call_error(self, index: int, state) -> Optional[str]:
        """A failed call, or a finished call without a measurable embedding (an error, no readable h5ad, rows that do
        not match the crops)."""
        if index in self._unusable:
            return self._unusable[index]
        return super().call_error(index, state)

    def fallback_moves(self, inp, attempts, add: Callable[..., None]) -> None:
        """The deterministic chain, only without an engine (after the tool at its defaults): same_well pairing and
        mask at trainable_pct 33 and lr 2e-5 from the input; then, once that call finished, init_from parent at
        trainable_pct 50 from it; then trainable_pct 75 at lr 1e-5 from that one (`_FALLBACK_CHAIN`)."""
        if self.llm_engine is not None:
            return
        parent = INPUT
        for step, changes in enumerate(_FALLBACK_CHAIN, 1):
            add(parent, [{"lever": lever, "value": json.dumps(value)} for lever, value in changes],
                f"step {step} of the deterministic chain")
            parent = next((c for c, (p, config, _) in attempts.items()
                           if p == parent and all(config.get(lever) == value for lever, value in changes)), None)
            if parent is None:
                return

    # ---- the selection: trial 0 whenever no call is delivered ----------------------------------------------
    def select_on_input(self, state, details: dict) -> Optional[WaveSelection]:
        """Trial 0 failed or unusable -> no output; the gate's SKIP -> trial 0 (the frozen embedding), satisfied,
        "not_measurable"; RUN -> None (the search's selection, `select`)."""
        if state.failed(0) or 0 in self._unusable or not self.admission:
            return WaveSelection(None, "conceded", "initial_trial_failed", "ceiling", details)
        if self.admission["decision"] == "SKIP":
            details.update({"call": INPUT, "choice": "not_measurable"})
            self.deliver(state, 0, details)
            return WaveSelection(0, "satisfied", "not_measurable", None, details)
        return None

    def select(self, state, stop_reason: str) -> WaveSelection:
        """The shared selection; when it delivers no call while trial 0 finished -- no call finished (the agent's
        empty round-1 plan among them), no admissible call, a null final choice, a choice still inadmissible after
        the retry -- trial 0, the frozen embedding, is delivered under its case (`_frozen_case`): satisfied when the
        agent's round-1 plan had no moves ("no_tuning_needed", its judgement that the input needs no tuning), the
        shared status otherwise. The details record the decision (`_decision`)."""
        selection = super().select(state, stop_reason)
        details = selection.details
        case = details.get("choice") or selection.reason
        if selection.index is None and not state.failed(0) and 0 not in self._unusable:
            case = self._frozen_case(selection.reason)
            satisfied = case == "no_tuning_needed"
            details.update({"call": INPUT, "choice": case})
            self.deliver(state, 0, details)
            selection = WaveSelection(0, "satisfied" if satisfied else selection.status, case,
                                      None if satisfied else selection.fault, details)
        details["decision"] = self._decision(selection.index, case)
        return self.finalize_search(state, selection)

    def _frozen_case(self, reason: str) -> str:
        """Why trial 0 is delivered after the search: the shared selection's reason ("agent_kept_input",
        "final_choice_inadmissible", "no_admissible_call"), with "no_call_finished" split into the agent's empty
        round-1 plan ("no_tuning_needed"), "every_call_failed" and "no_call_ran"."""
        if reason != "no_call_finished":
            return reason
        if self.failed:
            return "every_call_failed"
        return "no_tuning_needed" if self.stop_reason == "satisfied" and not self.rounds else "no_call_ran"

    def _decision(self, index: Optional[int], case: str) -> dict:
        """The decision record: the delivered trial ("frozen" for trial 0, None without an output), the case, and the
        delivered call's paired proxy difference to the frozen input on the comparison draws (delta, se, n_units)."""
        record = {"use": None if index is None else ("frozen" if index == 0 else self.label(index)), "case": case,
                  "delta": None, "se": None, "n_units": 0}
        if index:
            paired = self._compare(0, index)
            record.update(delta=paired.get("delta"), se=paired.get("se"), n_units=paired.get("n", 0))
        return record

    def extra_details(self, inp, attempts) -> dict:
        """The tier, the adopted proxy, the split, the proxy's setup and every measured trial's parent, lever changes,
        readouts, report and logged-only outputs (the selection-draw proxy and its `selection_check` among them; the
        record, never shown), each call's basis per changed lever (`param_basis`), the stops refused while the last
        call improved, the forced continuations and the plateau's exploration rounds (A13); `select` adds the
        decision."""
        trials = {}
        for i in sorted(self._reports):
            label = self.label(i)
            entry = attempts.get(label)
            vres = inp if i == 0 else (entry[2] if entry else None)
            failed = next((f for f in self.failed if f["call"] == label), None)
            config = entry[1] if entry else (failed or {}).get("params")
            trials[label] = {"parent": entry[0] if entry else None,
                             "change": dict(vres.param_delta or {}) if vres is not None and i else {},
                             "param_basis": dict(self._basis_of.get(self.config_key(config), {})) if i and config
                             else {},
                             "readouts": dict(vres.metrics or {}) if vres is not None else {},
                             "report": self._reports[i], "logged": self._logged.get(i),
                             "unusable": self._unusable.get(i)}
        return {"tier": self.tier, "adopted_proxy": self.proxy, "split": self.split_facts, "proxy_setup": self.setup,
                "search_evidence_policy": self.search_evidence_policy, "stop_evidence": list(self.stop_evidence),
                "trials": trials, "stop_refusals": list(self.stop_refusals),
                "forced_continuations": list(self.forced_continuations),
                "expert_config": [dict(r) for r in self.expert_config], "expert_rejections": list(self.expert_rejections),
                **({"search_training": dict(self._search_training)} if self._search is not None else {})}

    def deliver(self, state, index: int, details: dict) -> None:
        """Links the delivered trial's h5ad and, for a call, its resume state at the best val point to the canonical
        paths in `<query_cache_dir>/cell_state_analysis/` (the trial's directory up its output_subdir), where a call
        without one writes them, and points the delivered result at them (its own paths stay as `call_adata_path` /
        `call_resume_best`); a hard link, a copy where the filesystem has none. Without a resume state (trial 0), a
        canonical one left by an earlier delivery is removed."""
        params, _, result = state.trials[index]
        source = result.get("adata_path") if isinstance(result, Mapping) else None
        if not source or not os.path.exists(source):
            return
        root = Path(source).parent
        parts = Path(str(params.get("output_subdir") or "")).parts
        if parts and len(root.parts) > len(parts) and root.parts[-len(parts):] == parts:
            root = root.parents[len(parts) - 1]
        delivered = {"trial": self.label(index), "adata_path": str(root / Path(source).name), "resume_best": None}
        _link(source, delivered["adata_path"])
        resume, target = result.get("resume_best"), root / RESUME_NAME
        if resume and os.path.exists(resume):
            _link(resume, target)
            delivered["resume_best"] = str(target)
        elif target.exists():
            target.unlink()
        result.update({"call_adata_path": source, "adata_path": delivered["adata_path"], "call_resume_best": resume,
                       "resume_best": delivered["resume_best"]})
        details["delivered"] = delivered

    # ---- prompts: the analyzer's text (the frame and the questions are the shared ones) ---------------------
    def resources_text(self) -> str:
        return (f"Resources: {self.cores} cores and {self.gpus} GPU(s); one call runs at a time (the tool runs every "
                "call on the\nfirst GPU).")

    def final_rule_text(self) -> str:
        return ("Final output: at the end you choose one finished call that the guard admits: a finite, non-constant\n"
                f"embedding whose val {SCORE} beats the frozen input's by more than 2 SE (paired over the val units\n"
                "both have, on the comparison draws). A choice that does not qualify is answered with the reason and\n"
                "asked once more. If you choose none (null), or no call qualifies, the frozen input (the pretrained\n"
                "backbone, no training) is delivered.")

    def plateau_text(self) -> str:
        if self.llm_engine is None or self.early_stop is not None:
            return super().plateau_text()
        return ("the reasoner gives no valid move (a plateau does not stop the search: whether to continue from a\n"
                "call, go back to an earlier one, change direction or stop is yours)")

    # ---- A12: the goal, a basis for every change, no stop while the last call improves --------------------
    def goal_text(self) -> str:
        """The goal as the round prompt states it (A12; this verifier's prompt only, not the task brief); with
        search-level early stopping, its rule instead of A12's."""
        if self.early_stop is not None:
            return (f"Goal: the largest improvement of the val {SCORE} (on the comparison draws) over the frozen input\n"
                    "(trial 0) that the call budget can reach. Changing levers is the means, not the goal. Early\n"
                    "stopping: a stop is refused only while the last finished call beat its parent by more than 2 SE;\n"
                    f"the search ends after {self.patience} rounds in a row without a call that beats the best so far by\n"
                    "more than 2 SE. Gains within the noise are not progress: after a plateau, a different direction\n"
                    "(pairing, mask, depth, regulariser) is more informative than a finer learning-rate decay of the\n"
                    "same chain. Each call reports why it stopped (stop_reason: val_proxy_plateau = its val proxy set\n"
                    "no new best for the patience; overfit_gap = the proxy on its training wells kept rising above\n"
                    "the val proxy while the val proxy stalled), the epochs it saved, and overfit_gap (proxy on\n"
                    "training wells - proxy on val wells; a growing gap is memorisation of the training wells).")
        return (f"Goal: the largest improvement of the val {SCORE} (on the comparison draws) over the frozen input\n"
                "(trial 0) that the call budget can reach. Changing levers is the means, not the goal; the search\n"
                "continues while gains remain. A stop while the last finished call still improved on its parent (a\n"
                f"positive {SCORE} delta to its parent, of any size) is answered with that delta, and you are\n"
                "asked for moves.")

    @staticmethod
    def basis_text() -> str:
        return ("Each lever change needs a basis: an observation from this run's readouts (name the call and the\n"
                "readout) or a testable hypothesis. A change without one is rejected.")

    def move_text(self) -> str:
        return f"{self.goal_text()}\n\n{super().move_text()}\n{self.basis_text()}"

    @staticmethod
    def _basis_problem(move: Mapping[str, Any]) -> Optional[str]:
        """Why `move` lacks a basis (None = every lever change has a non-empty one)."""
        missing = [c.get("lever") for c in move.get("changes") or [] if not str(c.get("basis") or "").strip()]
        if not missing:
            return None
        return (f"no basis for {missing}: every lever change needs one -- an observation from this run's readouts "
                "(name the call and the readout) or a testable hypothesis")

    def _accept(self, moves, attempts, target: list[dict], cap: int, rejected: list[str],
                pending: set[str] = frozenset()) -> None:
        """The shared validation, each of the agent's moves first checked for a basis on every lever change
        (`_basis_problem`, rejected with the reason as an invalid value is, so the repair question asks again); in
        rounds 1-2 of an expert start, also checked against it (`_expert_problem`)."""
        based = []
        for move in moves:
            data = move.model_dump()
            why = self._basis_problem(data)
            if why:
                rejected.append(f"{self.move_name(data)}: {why}")
                self.log("rejected_move", text=f"{why}: {move.model_dump_json()}")
            else:
                based.append(move)
        if not self._expert_round():
            super()._accept(based, attempts, target, cap, rejected, pending)
            return
        for move in based:
            before = len(target)
            super()._accept([move], attempts, target, cap, rejected, pending)
            if len(target) == before:                                              # not accepted
                continue
            why = self._expert_problem(target[-1], attempts)                     # A18: rounds 1-2
            if why:
                target.pop()
                rejected.append(f"{self.move_name(move.model_dump())}: {why}")
                self.expert_rejections.append({"round": len(self.rounds) + 1, "move": move.model_dump(), "reason": why})
                self.log("rejected_move", text=f"{why}: {move.model_dump_json()}")

    # ---- A18/A19: the expert start, fixed expert levers per round in rounds 1-2 ------------------------------
    def _expert_round(self) -> bool:
        """Whether the round being planned is one of the expert start's rounds (one per `expert_config` entry)."""
        return len(self.rounds) < len(self.expert_config)

    def _expert_problem(self, request: Mapping[str, Any], attempts) -> Optional[str]:
        """Why a validated request of round k (1 or 2) breaks the expert start (None = it holds, or a later round): a
        change of a lever outside the expert levers of rounds 1..k (init_from aside) or of one to another value; a
        configuration without every one of them at its expert value; in a round whose entry sets init_from "parent",
        a move that does not resume a finished call of the previous round (when one finished)."""
        if not self._expert_round():
            return None
        due = {k: v for r in self.expert_config[:len(self.rounds) + 1] for k, v in r.items() if k != "init_from"}
        delta = {k: v for k, v in request["param_delta"].items() if k != "init_from"}
        problems = []
        other = [k for k in delta if k not in due]
        if other:
            problems.append(f"it changes {other}: the expert start's rounds change no other lever (init_from aside)")
        wrong = {k: v for k, v in delta.items() if k in due and v != due[k]}
        if wrong:
            problems.append(f"it sets {json.dumps(wrong)} instead of {json.dumps({k: due[k] for k in wrong})}")
        off = [k for k, v in due.items() if k not in wrong and request["params"].get(k) != v]
        if off:
            problems.append(f"its configuration leaves {off} off the expert values of round {len(self.rounds) + 1}"
                            + (" (round 2 builds on a round-1 call)" if self.rounds else ""))
        if self.expert_config[len(self.rounds)].get("init_from") == "parent":
            k = len(self.rounds)
            previous = [self.label(i) for i in self.rounds[-1] if self.label(i) in attempts]
            if previous and (request["parent"] not in previous or request["params"].get("init_from") != "parent"):
                problems.append(f"it does not resume a round-{k} call: round {k + 1} resumes one of {previous} "
                                "(init_from parent)")
        return f"expert start: {'; '.join(problems)}" if problems else None

    def admission_text(self) -> str:
        """The round prompt's line before the tool defaults: in rounds 1-2 of an expert start, the expert start
        (A18/A19); else the shared line (none here)."""
        if not self._expert_round():
            return super().admission_text()
        n = len(self.expert_config)

        def basis(i, r):
            if not i:
                return ""
            if r.get("init_from") == "parent":
                return f", resuming a round-{i} call (init_from parent)"
            return f", building on a round-{i} call"

        sets = "; ".join(f"in round {i + 1}{basis(i, r)}, " + ", ".join(f"{k}={v}" for k, v in r.items()
                                                                         if k != "init_from")
                         for i, r in enumerate(self.expert_config))
        both = " (if no round-1 call finished, a round-2 move sets both rounds' values)" if n == 2 else ""
        rounds = "Round 1 changes" if n == 1 else f"Rounds 1-{n} change"
        return (f"Expert start: expert experience for this tool sets, {sets}{both}. {rounds} no other lever "
                f"(init_from aside). A move that does not follow this is rejected. This is round "
                f"{len(self.rounds) + 1}.")

    # ---- A21: no untried-lever list or reminder in the prompts ------------------------------------------------
    def untried_text(self, attempts) -> str:
        """None (amendment A21): the round prompt lists no levers no call has changed and no reminder to review
        them; which levers to change is the agent's, from the evidence."""
        return ""

    def validate(self, move: Mapping[str, Any], attempts, pending: set[str]) -> dict:
        """The shared validation; the request also carries each changed lever's basis (`param_basis`), kept by the
        configuration key for the selection details (`extra_details`)."""
        request = super().validate(move, attempts, pending)
        basis = {c.get("lever"): str(c.get("basis") or "").strip() for c in move.get("changes") or []
                 if c.get("lever") in request["param_delta"] and str(c.get("basis") or "").strip()}
        request["param_basis"] = self._basis_of[request["config_key"]] = basis
        return request

    def _last_call(self, attempts) -> Optional[tuple[str, str, dict]]:
        """The most recently finished call (run order), its parent and its paired proxy difference to the parent on
        the comparison draws (`_compare`; the input's trial 0 for a move from "input"); None before any call
        finished."""
        if not attempts:
            return None
        call = next(reversed(attempts))
        parent = attempts[call][0]
        return call, parent, self._compare(0 if parent == INPUT else self._labels[parent], self._labels[call])

    def stop_check_text(self, inp, attempts, remaining: int) -> str:
        """The question after a plan with no moves while rounds are left. While the last finished call still improved
        on its parent (paired delta > 0, any size) the stop is refused: the text states that delta +/- 2 SE and asks
        for moves (recorded in `stop_refusals` and logged). It is asked once per round, as the shared stop check; an
        answer with no moves again runs a forced continuation (`plan_round`). Otherwise the shared stop check."""
        last = self._last_call(attempts)
        delta = last[2].get("delta") if last else None
        evidence_note = ''
        if self.search_evidence_policy != 'legacy' and last:
            from ..representation_policy import continuation_evidence
            evidence = continuation_evidence(last[2])
            evidence.update(call=last[0], parent=last[1], round=len(self.rounds) + 1)
            self.stop_evidence.append(evidence)
            self.log('stop_evidence', text=json.dumps(evidence, default=str))
            if not evidence['continue_required']:
                # The agent still confirms stop or proposes another evidence-based move.
                # An uncertain gain must not manufacture a forced native training call.
                delta = None
                evidence_note = (f"\nSSL stop evidence: {evidence['reason']}; the latest gain does not establish "
                    "improvement beyond 2 SE over its parent. Consider expected benefit and additional native "
                    "training/I/O cost. You may confirm stop or propose a justified move. The final delivery "
                    "guard against the frozen input remains unchanged.\n")
        noise = self.early_stop is not None and last is not None and not last[2].get("beats")
        if delta is None or delta <= 0 or noise:         # noise: early stopping refuses only beyond 2 SE; the shared stop check, without its untried groups (A21)
            best = self.fallback_choice(inp, attempts)
            best_text = (f"{best} ({self.score_key} {fmt(self._score(attempts[best][2]))})" if best
                         else "none admissible yet")
            return (evidence_note + f"\n\nYou returned no moves: the search ends here ({remaining} rounds remain; the same result in "
                    f"fewer\nrounds costs less). The best admissible call so far: {best_text}. Return moves if gains "
                    "are left to find,\nor no moves to confirm the stop.")
        call, parent, paired = last
        record = {"round": len(self.rounds) + 1, "call": call, "parent": parent, "delta": delta,
                  "se": paired.get("se"), "n": paired.get("n", 0)}
        self.stop_refusals.append(record)
        self.log("stop_refused", text=json.dumps(record, default=str))
        return (f"\n\nYou returned no moves, but the stop is not accepted: the last finished call, {call}, still "
                f"improved on its parent ({parent}): val {SCORE} {self._paired_text(paired)}. The search continues "
                f"while the last call improves: return moves for this round ({remaining} rounds remain). If you "
                f"return no moves again, the system runs a continuation of {call} (init_from parent, no lever "
                "change) as this round's call.")

    def plan_round(self, inp, attempts, k_max: int, remaining: int) -> list[dict]:
        """The shared round; when this round's stop was refused (`stop_check_text`) and the answer to the refusal
        again has no moves, the stop still does not hold (A12, strict): the verifier schedules one continuation of the
        last finished call (`forced_move`), a call counted in the budget and the plateau like any other, so the search
        ends only by its backstops or by an empty plan when the last call did not improve."""
        refused = len(self.stop_refusals)
        requests = super().plan_round(inp, attempts, k_max, remaining)
        if requests or len(self.stop_refusals) == refused or self.stop_reason != "satisfied":
            return requests
        return self.forced_move(attempts, self.stop_refusals[-1])

    def forced_move(self, attempts, refusal: dict) -> list[dict]:
        """One continuation of the refused stop's last call: parent = that call, no lever change (init_from parent,
        `extend_move`: a new state, so a new configuration key), the basis as its diagnosis; logged as
        `forced_continuation` and recorded in `forced_continuations`. A continuation that does not validate (e.g. one
        that already failed, or a parent without a resume state) ends the search ("no_valid_move")."""
        call, delta = refusal["call"], refusal["delta"]
        basis = f"forced continuation: the last call improved by {delta:+.4f} over its parent (user rule A12)"
        record = {"round": len(self.rounds) + 1, "parent": call, "delta": delta, "basis": basis}
        self.plan_delivery = None
        try:
            request = self.validate({"parent": call, "changes": [], "diagnosis": basis}, attempts, set())
        except ValueError as exc:
            record["rejected"] = str(exc)
            self.forced_continuations.append(record)
            self.log("forced_continuation", text=json.dumps(record, default=str))
            self.stop_reason = "no_valid_move"
            return []
        self.stop_reason = None
        request["param_basis"] = self._basis_of[request["config_key"]] = {"init_from": basis}
        self.forced_continuations.append(record)
        self.log("forced_continuation", text=json.dumps(record, default=str))
        return [{**request, "role": CONTINUATION}]

    def evidence_text(self) -> str:
        f = self.split_facts
        per_plate = list((f.get("val_wells_per_plate") or {}).values()) or [0]
        return (f"Tier: {self.tier} (site and plate-qualified well on every crop, at least 2 sites in most wells); "
                f"adopted proxy: {self.proxy}.\n"
                f"Split (label-free, by well): n_train {f.get('n_train')} cells, n_val {f.get('n_val')} cells, "
                f"n_val_wells {f.get('n_val_wells')}\n({len(per_plate)} plates, {min(per_plate)}-{max(per_plate)} "
                "val wells each).")

    def defaults_note(self) -> str:
        return ('A move from "input" starts at every lever\'s Default (the pretrained backbone, init_from\n'
                "pretrained). A move from a finished call starts at that call's full configuration and resumes that\n"
                "call (init_from parent) unless the move sets init_from: init_from pretrained restarts from the\n"
                "pretrained backbone with a fresh adapter and projector and keeps the parent's other levers.\n"
                "init_from = parent resumes the parent call's training state at the parent's best val point; it needs\n"
                "a finished call as the parent.")

    def readouts_text(self) -> str:
        """The stable metric set as the prompts show it: what each shown readout measures, its scale or chance, what
        its SE covers and, in label-free terms, how far it can be trusted."""
        from shape.tools.cell_state_analyzer import quant

        return (
            "Val wells. The tool holds out a share of each plate's wells as val wells; their cells never train. At\n"
            "every evaluation the tool embeds every val cell, and every readout below is computed on these val cells\n"
            "only. No treatment or label information is used anywhere.\n\n"
            f"{SCORE} (the val proxy). Each val site gets a profile: the mean of {PROXY_K} of its cells, drawn at\n"
            "random, L2-normalised. Each site ranks the other val sites of its plate by the cosine similarity of\n"
            "their profiles; the positives are the other site(s) of its own well. The value is the average precision\n"
            f"of these rankings, averaged over {PROXY_DRAWS} seeded draws of the cells and over all query sites; 1 "
            "means every\nsite retrieves its own well's other site(s) first. chance is the average precision of a "
            "random ranking\nof the same queries; it depends on how many val sites share a plate and how many sites "
            "each well has.\nIt measures how much more alike the cells of one well are than cells of other wells on "
            "the same plate.\n"
            "How far it can be trusted: it has not been checked against any quality measure on this tool's embedding\n"
            f"protocol. A profile from {PROXY_K} cells is noisy, so the value moves between draws of the cells, and "
            "with few val\nwells per plate its SE is wide. It ranks sites only against sites of the same plate, so it "
            "says nothing\nabout how plates or experiments relate to each other. Training that pulls cells of one site "
            "or well\ntogether targets the property this readout measures, on other wells; a rise can reflect that "
            "transfer\nrather than broader quality. Continued training alone can move it. Its level depends on the "
            "val set, so\nvalues compare only between calls on the same val cells.\n\n"
            "Two sets of draws. A call's best point is its val evaluation with the highest proxy value on one fixed\n"
            "set of draws (the selection draws). Every value, delta and SE shown here, and the delivery rule, use a\n"
            "second fixed set of draws that picked no best point (the comparison draws), the same for every call and\n"
            "for trial 0. The best point's value on the selection draws is the highest of many looks and runs high;\n"
            "it is not shown. The val wells are the same in both sets, so the comparison draws remove the reuse of\n"
            "the cell draws, not of the val wells.\n\n"
            "Comparisons. Each call is compared with trial 0 (the frozen pretrained backbone) and with its parent:\n"
            "delta = mean per-site difference (this call minus the reference), SE = bootstrap over val wells, n =\n"
            "shared query sites. The SE covers the sampling of val wells and of the cell draws. It does not cover\n"
            "differences between training runs of the same configuration with other random streams, and it covers\n"
            "each comparison on its own, not the best of several calls. A delta within 2 SE is not distinguishable\n"
            "from that noise. A resumed call starts from its parent's best point. Only a call whose proxy beats\n"
            "trial 0 by more than 2 SE can be delivered; otherwise trial 0 is delivered.\n\n"
            "ALARMS. They flag a failure mode. They are not quality scores, and no rule reads them. Trial 0's value\n"
            "on the same val cells is shown next to each call's. None of them has an SE.\n"
            "- effective_rank: exp of the entropy of the normalised singular-value spectrum of the centred,\n"
            f"  L2-normalised embedding of a seeded subset of up to {quant.HEALTH_ROWS} val cells. 1 means all cells "
            "lie on one\n  direction (collapse); the maximum is the embedding dimension. How far it can be trusted: a "
            "value near 1\n  is a clear sign of collapse; moderate moves in either direction carry no quality "
            "information --\n  training terms that spread or anchor the embedding move it by construction, and it "
            "has moved both\n  ways in training that did not collapse.\n"
            "- dead_dim_frac: the share of embedding dimensions that are constant across the val cells (std < 1e-6).\n"
            "  A non-zero share signals collapsed dimensions. How far it can be trusted: it stays at zero unless\n"
            "  dimensions collapse, so it flags that and nothing else.\n"
            f"- image_leakage: within each plate, the share of a cell's {LEAKAGE_K} nearest val neighbours that come "
            "from its own\n  image (site), divided by the share expected if the image did not matter. 1 means the "
            "image adds nothing\n  beyond the plate; higher means cells of one image are each other's neighbours "
            "beyond expectation. How\n  far it can be trusted: its level depends on how many cells each site and "
            "plate holds, so only the\n  comparison with trial 0 on the same val cells carries meaning.\n\n"
            "Training facts per call: epochs trained, best epoch, what picked the best point (val proxy or training\n"
            "loss), batch size used, trainable blocks, wall-clock seconds.")

    def _shown(self, index: int, vres: VLRRResult) -> dict:
        """The readouts the prompts show of one trial: the proxy on the comparison draws (value, chance, k, draws, val
        units) and the ALARMS."""
        m = vres.metrics or {}
        return {SCORE: {"value": _r4(m.get(SCORE)), "chance": _r4(m.get(CHANCE)), "k": self.setup.get("k"),
                        "draws": self.setup.get("draws"), "units": len(self._units.get(index) or {})},
                "ALARMS": {k: _r4(m.get(k)) for k in ALARMS}}

    def input_text(self, inp: VLRRResult) -> str:
        return f"Input (trial 0: the frozen pretrained backbone, no training): {json.dumps(self._shown(0, inp))}"

    def calls_legend(self) -> str:
        return ("change = relative to the parent (init_from parent: the call resumed its parent's training state at\n"
                "the parent's best val point); configuration = the levers' non-default values;\n"
                f"{SCORE} = the val proxy on the comparison draws (value, chance, k, draws, units = query sites);\n"
                f"vs_input / vs_parent = the paired {SCORE} difference to trial 0 (the frozen input) / to the\n"
                "parent call, delta +/- 2 SE over the units both have, beyond or within 2 SE, n shared units;\n"
                "prediction = each stated expectation with the observed change from the parent; ALARMS =\n"
                "effective_rank, dead_dim_frac, image_leakage of the call next to trial 0's (alarms only, no rule\n"
                "reads them); training = the call's training facts and seconds; guard = whether it is an admissible\n"
                "final choice")

    def _rows(self, inp, attempts) -> list[dict]:
        """One row per finished call: its move and configuration, the shown readouts (the proxy, the ALARMS next to
        trial 0's, the training facts), the paired differences and predictions, the guard -- never a logged-only
        output."""
        defaults, rows, frozen = self.defaults(), [], self._shown(0, inp)["ALARMS"]
        for call in attempts:                    # in run order (`_bind`)
            parent, config, vres = attempts[call]
            index = self._labels[call]
            parent_vres = inp if parent == INPUT else attempts[parent][2]
            fails = self.guard(inp, vres)
            shown = self._shown(index, vres)
            rows.append({
                "call": call, "parent": parent, "change": vres.param_delta or "none (the tool at its defaults)",
                "configuration": {k: config.get(k) for k in self.levers if config.get(k) != defaults.get(k)},
                "diagnosis": vres.notes, SCORE: shown[SCORE], "vs_input": self._delta(inp, vres, SCORE),
                "vs_parent": "the parent is the input" if parent == INPUT else self._delta(parent_vres, vres, SCORE),
                "prediction": self._prediction(parent_vres, vres),
                "ALARMS": {k: {"call": v, "input": frozen.get(k)} for k, v in shown["ALARMS"].items()},
                "training": self.reports.get(call), "guard": "admissible" if not fails else "; ".join(fails)})
        return rows

    def choice_text(self) -> str:
        return ("Choose the delivered embedding of this set of cell crops: one finished call that the guard admits, or "
                "null to\ndeliver the frozen input.")

    def admissible_text(self) -> str:
        return (f"a finite, non-constant embedding whose val {SCORE} beats the frozen input's by more than 2 SE "
                "(paired over\nthe val units both have, on the comparison draws)")

    def final_evidence_text(self, inp, attempts, admissible: list[str]) -> str:
        return f"{self.evidence_text()}\n\n{self.readouts_text()}"
