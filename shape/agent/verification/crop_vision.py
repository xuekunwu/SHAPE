"""Crop-vision (CORE): montage construction + vision-judged crop consistency/profile.

Relocated (task 008 P2, behavior-preserving) from the HPA benchmark's `diagnostic_planner.py`,
so the reasoning layer -- not a benchmark script -- owns the mechanism every tool's up-front
and post-hoc crop-vision checks share. `diagnostic_planner.py` re-imports these names via a
thin shim; its behavior is unchanged.

Two crop-vision moments:
  - `nn_visual_consistency_judge` -- POST-HOC, per-trial: on a TRAINED embedding, do a query
    crop's nearest neighbours actually look alike? (feeds RepresentationVerifier's
    `low_consistency` trigger via the benchmark's `verify_embedding`.)
  - `crop_vision_profile` -- UP-FRONT (P2, new): before any training, a montage of a sampled,
    class-covering crop set -> a structured read of the dataset (modality / cell size / density
    / neighbour crowding / heterogeneity / whether colour carries biology). This is the vision
    evidence `recommend_initial_analyzer_config` (`initial_params.py`) reasons from.

Both vision-judge when an engine is available and fall back to a deterministic proxy otherwise
(the same graceful-degrade pattern used throughout this subsystem: no engine / any failure ->
deterministic, never a crash).

CRITICAL -- dependency direction: this module is CORE and must NOT import the benchmark
(anything under `benchmarks/`). It
may import core only (stdlib, numpy/PIL/skimage/anndata, `shape.agent.verification.*`).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from pydantic import BaseModel

from shape.engine.errors import NoEngineError
from shape.utils.schemas import ExecutionError

from .initial_params import recommend_qc_sample_size

logger = logging.getLogger("shape.agent.verification")


# --------------------------------------------------------------------------- #
# nn_visual_consistency: a REAL check that the trained embedding's nearest neighbours actually
# look alike -- the signal RepresentationVerifier's low_consistency trigger was designed for but
# never received (it silently defaulted to 1.0 = "always consistent"). Vision-judged when an
# engine is available (reuses build_vision_engine, the SAME graceful fallback pattern used
# throughout this module); a deterministic pixel-correlation proxy keeps the check alive offline.
# --------------------------------------------------------------------------- #
class _NNConsistencyVerdict(BaseModel):
    """What the vision model reports about one query-crop-vs-neighbours montage."""
    consistent: Literal["yes", "partial", "no"]
    reasoning: str


_NN_CONSISTENCY_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}
_NN_JUDGE_PROMPT = (
    "You are checking a self-supervised cell-morphology embedding. The attached montage shows a "
    "QUERY crop (leftmost) next to its {k} NEAREST NEIGHBOURS in the learned embedding space, in "
    "order. Judge: do the neighbours look morphologically SIMILAR to the query (same organelle "
    "pattern / shape / texture), or does the embedding put visually DISSIMILAR crops next to each "
    "other?\n\nReport 'yes' if most neighbours are visually consistent with the query, 'partial' "
    "if mixed, 'no' if the neighbours look nothing like the query. One-sentence reasoning.")


def _montage_bytes(paths: list) -> bytes | None:
    """Side-by-side montage image: paths[0] = query crop, paths[1:] = its neighbours."""
    import io
    from PIL import Image
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("L"))
        except Exception:
            continue
    if len(imgs) < 2:
        return None
    h = max(im.height for im in imgs)
    canvas = Image.new("L", (sum(im.width for im in imgs) + 4 * (len(imgs) - 1), h), color=255)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + 4
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _load_gray_resized(p: str):
    """Raw crop pixels, resized to a common comparison size. Shared by the null-distribution
    sampler and the query-vs-neighbour correlation, so both measure the same thing."""
    from skimage import io as skio
    from skimage.transform import resize
    im = np.asarray(skio.imread(p), dtype=np.float64)
    if im.ndim == 3:
        im = im[..., 0]
    return resize(im, (64, 64), anti_aliasing=True)


def _pixel_correlation(path_a: str, path_b: str) -> float | None:
    """Normalized cross-correlation between two crops' raw pixels, or None on any read failure."""
    try:
        a, b = _load_gray_resized(path_a), _load_gray_resized(path_b)
        ac, bc = a - a.mean(), b - b.mean()
        denom = (np.linalg.norm(ac) * np.linalg.norm(bc)) or 1e-9
        return float((ac * bc).sum() / denom)
    except Exception:
        return None


def _null_correlation_stats(paths, n_pairs: int, seed: int = 0) -> dict:
    """The 'what does an UNRELATED pair look like in THIS dataset' reference: sample `n_pairs`
    random, non-identical crop-index pairs from the pool and compute their pixel correlations.
    Mirrors the role recommend_off_center_bound/recommend_dominance_bound play elsewhere -- a
    bound/reference derived from the dataset's own distribution, not a fixed absolute number."""
    rng = np.random.default_rng(seed)
    n = len(paths)
    corrs = []
    attempts = 0
    while len(corrs) < n_pairs and attempts < n_pairs * 4:
        attempts += 1
        i, j = rng.integers(0, n), rng.integers(0, n)
        if i == j:
            continue
        c = _pixel_correlation(paths[i], paths[j])
        if c is not None:
            corrs.append(c)
    if not corrs:
        return {"null_mean": 0.0, "null_std": 0.0, "n_null_pairs": 0}
    return {"null_mean": float(np.mean(corrs)), "null_std": float(np.std(corrs)),
            "n_null_pairs": len(corrs)}


def _pixel_proxy_consistency(paths: list, null_mean: float, null_std: float) -> str:
    """Deterministic fallback (no vision engine / no network -- the common case on the GPU
    cluster): classify the query-vs-neighbours mean pixel correlation RELATIVE TO the dataset's
    own null (random-pair) distribution -- not a fixed absolute-correlation cutoff, so a globally
    low-contrast/noisy dataset (whose null itself sits low) is judged on its own terms."""
    corrs = [c for c in (_pixel_correlation(paths[0], p) for p in paths[1:]) if c is not None]
    if not paths or not corrs:
        return "partial"
    mean_corr = float(np.mean(corrs))
    if mean_corr > null_mean + null_std:
        return "yes"          # meaningfully more similar than an unrelated pair in this dataset
    if mean_corr > null_mean:
        return "partial"
    return "no"                # no better than random chance


def _vision_judge_montage(engine, group_paths: list, k: int) -> str | None:
    """One montage's verdict via the vision engine; None without a montage. A failed or malformed vision call
    is an execution error (`schemas.ExecutionError`), never a silent fallback."""
    montage = _montage_bytes(group_paths)
    if montage is None:
        return None
    try:
        resp = engine.generate([_NN_JUDGE_PROMPT.format(k=k), montage],
                               response_format=_NNConsistencyVerdict)
        rec = resp.get("content") if isinstance(resp, dict) else resp
    except Exception as exc:
        raise ExecutionError(f"crop-vision NN judge failed: {type(exc).__name__}: {exc}") from exc
    if not hasattr(rec, "consistent"):
        raise ExecutionError(f"crop-vision NN judge: malformed answer ({type(rec).__name__})")
    return rec.consistent


def nn_visual_consistency_judge(hpa, adata_path: str, *, model_string: str | None = None,
                                api_key=None, sample: int = 6, k: int = 4, seed: int = 0) -> dict:
    """Sample query cells, find their top-k cosine neighbours in the TRAINED embedding, and judge
    (vision LLM, else a pixel-correlation proxy) whether the neighbours actually look alike.
    Needs obs['crop_path'] (written by _annotate_crop_path at embed time)."""
    import anndata as ad

    try:
        adata = ad.read_h5ad(adata_path)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    if "crop_path" not in adata.obs:
        return {"available": False, "reason": "no crop_path in obs"}
    n = adata.n_obs
    if n < k + 1:
        return {"available": False, "reason": f"too few cells ({n}) for k={k} neighbours"}

    rng = np.random.default_rng(seed)
    X = np.asarray(adata.X, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    paths = adata.obs["crop_path"].astype(str).to_numpy()
    queries = rng.choice(n, size=min(sample, n), replace=False)

    # The dataset-relative reference for the pixel-proxy fallback (computed once, reused per
    # query -- see _null_correlation_stats). Cheap relative to the per-query correlations, and
    # keeping it unconditional avoids a lazy-init branch for the rare case a vision call fails
    # mid-loop after the engine built successfully.
    null_stats = _null_correlation_stats(paths, n_pairs=2 * sample, seed=seed)

    engine = None
    if model_string:                      # a vision judge was asked for: it must be built, never skipped
        from shape.agent.verification.vision_engine import build_vision_engine
        engine = build_vision_engine(model_string, api_key)
        if engine is None:
            raise NoEngineError(f"crop-vision NN judge: no vision engine for {model_string}")

    verdicts = []
    for qi in queries:
        sims = X @ X[qi]
        sims[qi] = -np.inf
        neigh = np.argsort(-sims)[:k]
        group_paths = [paths[qi]] + [paths[j] for j in neigh]
        verdict = _vision_judge_montage(engine, group_paths, k) if engine is not None else None
        if verdict is None:
            verdict = _pixel_proxy_consistency(group_paths, null_stats["null_mean"],
                                               null_stats["null_std"])
        verdicts.append(verdict)

    score = float(np.mean([_NN_CONSISTENCY_SCORE[v] for v in verdicts])) if verdicts else 1.0
    return {"available": True, "score": round(score, 4),
            "mode": "vision" if engine is not None else "pixel_proxy",
            "n_sampled": len(verdicts), "verdicts": verdicts, **null_stats}


# --------------------------------------------------------------------------- #
# crop_vision_profile (P2, new): the UP-FRONT sampled-crop VISION profile -- a montage of a
# stratified, class-covering crop sample (reuses `recommend_qc_sample_size`, does not re-sample)
# judged into a structured read of the dataset. This is the vision evidence
# `recommend_initial_analyzer_config` (`initial_params.py`) reasons from; unlike
# `nn_visual_consistency_judge` above it never needs a trained embedding.
# --------------------------------------------------------------------------- #
class CropProfileVerdict(BaseModel):
    """The vision model's structured READ of a sampled-crop montage (mirrors SegOverlayVerdict /
    RepresentationVerdict's style). `crop_vision_profile`'s deterministic fallback
    (`engine=None` or any failure) derives the same shape from the numeric `profile_dataset`
    report instead -- no vision call."""
    modality: Literal["single_channel_fluorescence_or_brightfield", "rgb_or_multichannel",
                      "unknown"]
    cell_size: Literal["tiny", "small", "medium", "large", "unknown"]
    cell_density: Literal["sparse", "moderate", "crowded", "confluent", "unknown"]
    neighbour_crowding: Literal["low", "moderate", "high", "unknown"]
    heterogeneity: Literal["low", "moderate", "high", "unknown"]
    color_informative: Literal["yes", "no", "cautious", "unknown"]
    reasoning: str


_CROP_VISION_PROMPT = (
    "You are profiling a single-cell crop dataset before any training. The attached montage "
    "shows {n} crops, sampled to cover every class in the dataset. Judge the dataset as a "
    "whole (not any one crop): modality (single-channel fluorescence/brightfield vs "
    "rgb/multichannel); typical cell_size; cell_density; neighbour_crowding (do neighbouring "
    "cells/debris commonly encroach into the crop, crowding out the target cell?); "
    "heterogeneity (how much do the crops vary in appearance); and color_informative (does "
    "colour carry biological signal here, or is it incidental).\n\n"
    "Report: modality; cell_size; cell_density; neighbour_crowding; heterogeneity; "
    "color_informative; and a one-sentence reasoning citing what you see in the montage.")


def _rule_crop_vision_profile(numeric_profile: dict, n_sampled: int = 0) -> dict:
    """Deterministic fallback for `crop_vision_profile`: the same structured shape, derived from
    the numeric `profile_dataset` report's own modality/channel/dominance/off-center numbers --
    no vision call. Honest about what it cannot see: cell_size/cell_density have no numeric proxy
    at this (pre-training) stage, so both report 'unknown' rather than a guess."""
    mod = numeric_profile.get("modality", {}) or {}
    quality = numeric_profile.get("crop_quality", {}) or {}
    modality = mod.get("likely_modality") or "unknown"
    carries = mod.get("color_carries_biology")
    color_informative = ("yes" if carries is True else "no" if carries is False
                          else "cautious" if carries == "cautious" else "unknown")

    dom_bound = (quality.get("dominance_bound") or {}).get("bound")
    mean_dom = quality.get("mean_target_dominance")
    if isinstance(mean_dom, (int, float)) and isinstance(dom_bound, (int, float)):
        # below this dataset's own data-derived dominance fence -> neighbours crowd the target
        neighbour_crowding = "high" if mean_dom < dom_bound else "low"
    else:
        neighbour_crowding = "unknown"

    off_center_frac = quality.get("off_center_fraction")
    if isinstance(off_center_frac, (int, float)):
        heterogeneity = "high" if off_center_frac > 0.3 else "low"
    else:
        heterogeneity = "unknown"

    return {
        "modality": modality, "cell_size": "unknown", "cell_density": "unknown",
        "neighbour_crowding": neighbour_crowding, "heterogeneity": heterogeneity,
        "color_informative": color_informative,
        "reasoning": ("numeric fallback (no vision engine): modality from the channel-detection "
                     "module; neighbour_crowding from mean_target_dominance vs its data-derived "
                     "Tukey bound; heterogeneity from the off-center fraction; cell_size/"
                     "cell_density have no numeric proxy at this stage -> 'unknown'"),
        "mode": "numeric_fallback", "n_sampled": n_sampled,
    }


def crop_vision_profile(records, numeric_profile: dict, *, engine=None, sample: int = 24,
                        seed: int = 0) -> dict:
    """The up-front sampled-crop VISION profile (task 008 P2): reuses
    `recommend_qc_sample_size`'s stratified, class-covering sample (does NOT re-sample) -> a
    montage of up to `sample` of those crops -> a structured `CropProfileVerdict` via the vision
    engine. No engine: a RuntimeError; a failed vision call: an `ExecutionError`. Without a usable
    montage (no readable crop): the same structured shape derived from `numeric_profile` (the
    `profile_dataset` report), no vision call. This is the up-front evidence `recommend_initial_analyzer_config` reasons from;
    it never needs a trained embedding (unlike `nn_visual_consistency_judge`, which is POST-HOC
    on a trained one).
    """
    if engine is None:                    # a system error, never the numeric-only profile
        raise NoEngineError("crop-vision profile: no vision engine is configured")

    qc = recommend_qc_sample_size(records, seed=seed)
    paths = [r["path"] for r in qc["sample"][:sample]]
    montage = _montage_bytes(paths)
    if montage is not None:
        try:
            prompt = _CROP_VISION_PROMPT.format(n=len(paths))
            resp = engine.generate([prompt, montage], response_format=CropProfileVerdict)
            rec = resp.get("content") if isinstance(resp, dict) else resp
        except Exception as e:  # an execution error, never the numeric fallback
            raise ExecutionError(f"crop-vision profile failed: {type(e).__name__}: {e}") from e
        if not hasattr(rec, "modality"):
            raise ExecutionError(f"crop-vision profile: malformed answer ({type(rec).__name__})")
        return {
            "modality": rec.modality, "cell_size": rec.cell_size,
            "cell_density": rec.cell_density,
            "neighbour_crowding": rec.neighbour_crowding,
            "heterogeneity": rec.heterogeneity,
            "color_informative": rec.color_informative,
            "reasoning": rec.reasoning, "mode": "vision", "n_sampled": len(paths),
        }
    return _rule_crop_vision_profile(numeric_profile, n_sampled=len(paths))
