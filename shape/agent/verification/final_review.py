"""Finalization vision check: does the pipeline's OUTPUT figure support the text?

`Analysis_Visualizer_Tool` is not a managed stage, so the figure the user actually
reads (UMAP / cluster composition) gets zero QC. This is the end-to-end backstop: a
multimodal judge looks at the final figures together with the written conclusion and
the deterministic QC metrics, and decides whether the visual evidence actually
supports the conclusion. Like the segmentation judge, it is deterministic-first --

Safety guarantee (can't make things worse): the judge can only ADD a caveat / lower
confidence; it never blesses a result or removes an existing hedge. If no engine is
wired, no figures exist, or anything fails, the conclusion is returned untouched.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Literal, Optional

from pydantic import BaseModel

from shape.engine.errors import NoEngineError
from shape.utils.schemas import ExecutionError

logger = logging.getLogger("shape.agent.verification")

# QC metrics worth showing the judge (kept small; others are context noise).
_KEY_METRICS = (
    "silhouette_by_group", "silhouette_by_batch", "group_enrichment_FDR",
    "effect_size", "replicate_consistency", "n_clusters", "ARI_across_seeds",
    "ARI_across_resolutions", "small_cluster_fraction",
)


class FinalFigureVerdict(BaseModel):
    cluster_separation: Literal["clear", "weak", "none", "not_applicable"]
    group_structure: Literal["distinct", "overlapping", "absent", "not_applicable"]
    figure_text_consistency: Literal["consistent", "partial", "contradicts"]
    action: Literal["accept", "add_caveat", "rerun"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


def _read_bytes(path: str) -> Optional[bytes]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning("final review: cannot read figure %s (%s)", path, e)
        return None


def _build_prompt(conclusion: str, metrics: dict) -> str:
    m = {k: round(float(v), 4) for k, v in (metrics or {}).items()
         if k in _KEY_METRICS and isinstance(v, (int, float)) and not isinstance(v, bool)}
    return (
        "You are the finalization reliability check of a single-cell morphology "
        "pipeline. The attached images are the pipeline's OUTPUT figures (e.g. "
        "segmentation overlay, UMAP, cluster-composition). Below are the deterministic "
        "QC metrics and the WRITTEN CONCLUSION the pipeline produced. Judge ONLY "
        "whether the visual evidence and metrics actually support the conclusion -- do "
        "not invent new findings.\n\n"
        f"Deterministic QC metrics: {json.dumps(m)}\n\n"
        "WRITTEN CONCLUSION:\n"
        f"{(conclusion or '')[:3000]}\n\n"
        "Decide:\n"
        "- cluster_separation: in the UMAP, are clusters visually separated? "
        "(clear / weak / none / not_applicable if no UMAP)\n"
        "- group_structure: do the cell types / groups occupy distinct regions? "
        "(distinct / overlapping / absent / not_applicable)\n"
        "- figure_text_consistency: does the written conclusion match what the figures "
        "and metrics show? (consistent / partial / contradicts)\n"
        "- action: accept (figures support the text), add_caveat (text overstates what "
        "the figures show), rerun (figures contradict the text or QC says it is "
        "unreliable).\n"
        "- confidence and a one-sentence reasoning.\n"
        "Note: a negative group silhouette or replicate_consistency near 0 means the "
        "structure is weak/absent, so the text must NOT claim strong separation."
    )


def vision_review_final(engine_mm, figure_paths: List[str], conclusion: str,
                        metrics: Optional[dict] = None) -> Optional[FinalFigureVerdict]:
    """Judge the final figures vs the conclusion. None when there is nothing to review (no figure, no readable
    figure). No engine: a RuntimeError; a failed or malformed vision call: an `ExecutionError` (the caller
    records the review as failed, never as reviewed)."""
    if not figure_paths:
        return None
    if engine_mm is None:
        raise NoEngineError("final review: no vision engine is configured")
    images = [b for b in (_read_bytes(p) for p in figure_paths) if b is not None]
    if not images:
        return None
    try:
        prompt = _build_prompt(conclusion, metrics or {})
        resp = engine_mm.generate([prompt, *images], response_format=FinalFigureVerdict)
        rec = resp.get("content") if isinstance(resp, dict) else resp
    except Exception as e:
        raise ExecutionError(f"final vision review failed: {type(e).__name__}: {e}") from e
    if not hasattr(rec, "action"):
        raise ExecutionError(f"final vision review: malformed answer ({type(rec).__name__})")
    return rec


def caveat_from_verdict(verdict: Optional[FinalFigureVerdict]) -> Optional[str]:
    """Render a caveat ONLY when the judge flags a problem; never a reassurance."""
    if verdict is None:
        return None
    issues = []
    if verdict.figure_text_consistency == "contradicts":
        issues.append("the output figures contradict the written conclusion")
    elif verdict.figure_text_consistency == "partial":
        issues.append("the figures only partially support the conclusion")
    if verdict.cluster_separation in ("weak", "none"):
        issues.append(f"cluster separation looks {verdict.cluster_separation} in the UMAP")
    if verdict.group_structure in ("overlapping", "absent"):
        issues.append(f"cell-type/group structure appears {verdict.group_structure}")
    if not issues and verdict.action == "accept":
        return None
    body = "; ".join(issues) if issues else f"flagged for {verdict.action}"
    note = (verdict.reasoning or "")[:240]
    return (f"⚠ Automated reliability check ({verdict.confidence} confidence): "
            f"{body}. {note}").strip()


def full_reasoning(verdict: Optional[FinalFigureVerdict]) -> str:
    """Full, untruncated judge rationale -- for the durable record only.

    `caveat_from_verdict` above stays truncated at 240 chars (it feeds the
    user-facing conclusion text); this is its full-text counterpart for the
    run's persisted trace (`vlrr_trace.json`)."""
    return (verdict.reasoning or "") if verdict is not None else ""
