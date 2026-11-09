"""STAGE 4 -- dataset-composition screening for HCMA, as a VLRR loop.

This is the screening the user calls out: "是否因为数据很少从而需要 batch correction --
这也是整个数据开始的时候的筛选需要的". Before any embedding is trusted, decide TWO things
from the metadata alone (no GPU, no features required):

  1. the multi-source subset -- which cell types have >= min_sources INDEPENDENT source
     images and can therefore be scored without the single-image technical-variation
     confound (guard A of the fixed evaluator);
  2. whether BATCH CORRECTION is required -- i.e. even after subsetting, is cell type so
     confounded with lab/source that a classifier could cheat by recognizing the imaging
     condition rather than the biology.

It runs as a proper VLRR loop (mirrors shape.agent.verification.BaseVerifier):
  quantify -> localize -> adjudicate -> VLRRResult, with a bounded rerun that RETUNES the
  one lever `min_sources` upward until the surviving subset clears the source-confound
  gate while still retaining enough cells/labels; if it never clears, the verdict is
  `batch_correction_required = True`.

Primary confound signal is metadata-only: NMI(cell_type, source) on the surviving subset
-- high NMI means cell type is predictable from source (confounded). If per-cell features
are supplied, detect_batch_effect() adds an embedding-level batch_predictability readout.

  python -m benchmarks.cell_state_HCMA.stage4_screening \
      --meta data/HCMA/labels.parquet \
      --label-key cell_type --image-key image_id --source-key source \
      --out benchmarks/cell_state_HCMA/results/stage4_screen.json
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from shape.agent.verification.base import (BaseVerifier, VLRRResult, normalize,
                                           weighted_score)


def _per_label_source_counts(df, label_key, image_key, source_key=None):
    """Per cell type: #distinct source images and #distinct sources/labs."""
    g = df.groupby(label_key)
    out = pd.DataFrame({"n_images": g[image_key].nunique()})
    out["n_sources"] = (g[source_key].nunique() if source_key and source_key in df
                        else out["n_images"])   # no lab col -> image is the independence unit
    out["n_cells"] = g.size()
    return out.sort_values("n_images")


def screen_metrics(df, label_key, image_key, source_key, min_sources,
                   features: Optional[np.ndarray] = None) -> dict:
    """GT-free composition metrics on the CURRENT min_sources subset."""
    counts = _per_label_source_counts(df, label_key, image_key, source_key)
    unit = "n_sources" if (source_key and source_key in df) else "n_images"
    eligible = counts.index[counts[unit] >= min_sources]
    mask = df[label_key].isin(eligible).to_numpy()
    sub = df[mask]
    n_total_labels = int(counts.shape[0])
    n_eligible = int(len(eligible))

    # label<->source confound on the SURVIVING subset (the set the benchmark will score).
    # Only meaningful with a REAL source grouping that is coarser than the image: NMI against
    # the image itself is degenerate (every image maps to one label => NMI ~ 1 by
    # construction), so we do NOT compute a confound in that case -- the verdict then rests
    # on multi-image eligibility + coverage alone.
    has_real_source = (source_key and source_key in sub and source_key != image_key
                       and sub[source_key].nunique() > 1)
    if n_eligible >= 2 and has_real_source:
        confound = float(normalized_mutual_info_score(
            sub[label_key].astype(str), sub[source_key].astype(str)))
    else:
        confound = float("nan")

    metrics = {
        "n_total_labels": float(n_total_labels),
        "n_eligible_labels": float(n_eligible),
        "n_dropped_single_source": float(n_total_labels - n_eligible),
        "frac_single_source_labels": float((counts[unit] < min_sources).mean()),
        "subset_cell_coverage": float(mask.mean()),
        "median_sources_per_label": float(counts[unit].median()),
        "label_source_confound_nmi": confound,
    }
    # optional embedding-level batch predictability (source predictable from features)
    if features is not None and (source_key and source_key in sub) and n_eligible >= 2:
        from shape.tools.batch_corrector.tool import detect_batch_effect
        rep = detect_batch_effect(features[mask], sub, batch_key=source_key)
        if rep.batch_predictability is not None:
            metrics["batch_predictability"] = float(rep.batch_predictability)
            metrics["batch_effect_margin"] = float(rep.effect_margin or 0.0)
    metrics["_eligible_labels"] = list(map(str, eligible))
    return metrics


class Stage4ScreeningVerifier(BaseVerifier):
    """VLRR verdict on dataset composition: is the multi-source subset clean, or is
    batch correction required? The single lever is `min_sources`."""

    name = "stage4_screening"
    uses_llm = False

    confound_gate = 0.35            # NMI(cell_type, source) above this -> confounded
    coverage_floor = 0.30           # subset must retain >= this fraction of cells
    min_eligible_labels = 3         # need >= this many multi-source cell types to benchmark
    predictability_gate = 0.55      # source predictable from features above this -> confounded
    accept_score = 0.70

    def quantify(self, tool_output, context):
        return {k: v for k, v in (tool_output or {}).items()
                if isinstance(v, (int, float)) and np.isfinite(v)}

    def localize(self, metrics, context):
        # confound is NaN (hence dropped by quantify) exactly when < 2 labels survive --
        # that is the insufficient-multisource case, NOT a tool mismatch. Only treat a
        # truly empty readout (no label counts at all) as a mismatch.
        if not metrics or "n_total_labels" not in metrics:
            return "tool_mismatch", ["missing_screen_metrics"], []
        triggered = []
        if metrics.get("n_eligible_labels", 0) < self.min_eligible_labels:
            triggered.append("insufficient_multisource")
        if metrics.get("subset_cell_coverage", 1.0) < self.coverage_floor:
            triggered.append("over_aggressive_subset")
        # confound is only present when a REAL source grouping was available; absent (NaN)
        # means source-confound is simply not assessable from metadata -- not a failure.
        if metrics.get("label_source_confound_nmi", 0.0) > self.confound_gate:
            triggered.append("residual_source_confound")
        if metrics.get("batch_predictability", 0.0) > self.predictability_gate:
            triggered.append("batch_predictable_embedding")
        return "this_step", triggered, []

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        min_sources = int(context.get("min_sources", 2))
        parts = {
            "coverage": normalize(metrics.get("subset_cell_coverage", 0.0),
                                  1.0, self.coverage_floor),
            "labels": normalize(metrics.get("n_eligible_labels", 0.0),
                                float(max(self.min_eligible_labels * 3, 9)),
                                float(self.min_eligible_labels)),
        }
        if "label_source_confound_nmi" in metrics:      # only score confound when assessable
            parts["confound"] = normalize(metrics["label_source_confound_nmi"],
                                          0.0, self.confound_gate)
        if "batch_predictability" in metrics:
            parts["predictability"] = normalize(metrics["batch_predictability"],
                                                0.0, self.predictability_gate)
        score = weighted_score(parts, {k: 1.0 for k in parts})

        if locus == "tool_mismatch":
            return VLRRResult("FAIL", 0.0, parts, action="conceded", fault="ceiling",
                              notes="No screening metrics produced.")
        if "insufficient_multisource" in triggered:
            return VLRRResult(
                "FAIL", score, parts, action="conceded", fault="ceiling",
                notes=(f"Only {int(metrics.get('n_eligible_labels',0))} cell types have "
                       f">= {min_sources} independent sources; the dataset cannot support a "
                       "multi-source benchmark. Batch correction cannot manufacture "
                       "independent sources -- acquire more images."))
        if "over_aggressive_subset" in triggered:
            return VLRRResult(
                "WARNING", score, parts, action="retune_here",
                param_delta={"min_sources": "down"},
                expected_effect={"subset_cell_coverage": "up", "n_eligible_labels": "up"},
                notes="min_sources too strict -- it discards most cells; relax it.")
        if "residual_source_confound" in triggered or "batch_predictable_embedding" in triggered:
            return VLRRResult(
                "FAIL", score, parts, action="retune_here",
                param_delta={"min_sources": "up"},
                expected_effect={"label_source_confound_nmi": "down"},
                notes=("Cell type remains confounded with source in the multi-source "
                       "subset -> raise min_sources; if it cannot clear, BATCH CORRECTION "
                       "is required before the embedding can be trusted."))
        return VLRRResult(
            "PASS", score, parts, action="accept",
            notes=("Multi-source subset is adequately sized and cell type is not "
                   "confounded with source -> no batch correction needed for screening."))


def run_vlrr_screening(df, *, label_key, image_key, source_key=None,
                       features=None, min_sources=2, max_sources=6) -> dict:
    """Bounded VLRR: raise the `min_sources` lever until the surviving subset clears the
    source-confound gate with adequate coverage; else declare batch correction required."""
    verifier = Stage4ScreeningVerifier()
    min_eligible = verifier.min_eligible_labels
    trace, passed, last_populated, terminal = [], None, None, None
    confound_seen = False
    ms = min_sources
    for _ in range(max_sources - min_sources + 2):
        metrics = screen_metrics(df, label_key, image_key, source_key, ms, features)
        res = verifier.verify(metrics, {"min_sources": ms})
        trace.append({"min_sources": ms, "decision": res.decision, "action": res.action,
                      "score": round(res.scalar_score, 4), "triggered": res.triggered,
                      "notes": res.notes,
                      "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in metrics.items() if not k.startswith("_")}})
        if metrics["n_eligible_labels"] >= min_eligible:
            last_populated = (ms, metrics, res)
        if "residual_source_confound" in res.triggered or "batch_predictable_embedding" in res.triggered:
            confound_seen = True               # a REAL source-confound fired at some round
        if res.decision == "PASS":
            passed = (ms, metrics, res); break
        delta = (res.param_delta or {}).get("min_sources")
        if delta == "up" and ms < max_sources:
            ms += 1; continue                  # raise the lever, try to clear the confound
        terminal = (ms, metrics, res); break

    # Classify the terminal state by the ACTUAL reason it failed, not merely "not PASS".
    warnings = []
    if passed:                                 # clean subset -> no batch correction
        ms_final, metrics_final, res_final = passed
        decision, reason, batch_needed = "PASS", res_final.notes, False
    elif confound_seen and last_populated is not None:   # real source-confound never cleared
        ms_final, metrics_final, res_final = last_populated
        decision, batch_needed = "BATCH_CORRECTION_REQUIRED", True
        reason = ("Cell type stays confounded with source in the multi-source subset "
                  f"(NMI={metrics_final['label_source_confound_nmi']:.3f} at min_sources="
                  f"{ms_final}); raising min_sources only empties the subset, so the "
                  "confound is systematic (source-specific cell types), not single-image "
                  "inflation. Batch correction is required before the embedding is trusted.")
    elif last_populated is not None:           # enough multi-image labels; only coverage was low
        ms_final, metrics_final, res_final = last_populated
        decision, batch_needed = "PASS", False
        reason = (f"{int(metrics_final['n_eligible_labels'])} labels reach >= {ms_final} "
                  "independent images -> the benchmark is runnable; source-confound was "
                  "not assessable or is acceptable.")
        cov = metrics_final["subset_cell_coverage"]
        if cov < verifier.coverage_floor:
            warnings.append(f"large single-image tail: the multi-image subset keeps only "
                            f"{cov*100:.0f}% of cells ({int(metrics_final['n_dropped_single_source'])} "
                            "single-image labels dropped).")
    else:                                      # never enough multi-image labels at all
        ms_final, metrics_final, res_final = (terminal or (ms, metrics, res))
        decision, batch_needed = "INSUFFICIENT_DATA", False
        reason = (f"Fewer than {min_eligible} labels reach {min_sources} independent "
                  "images; batch correction cannot manufacture independent sources -- "
                  "acquire more images before benchmarking.")

    verdict = {
        "batch_correction_required": bool(batch_needed),
        "screening_decision": decision,
        "warnings": warnings,
        "reason": reason,
        "recommended_min_sources": int(ms_final),
        "n_eligible_labels": int(metrics_final["n_eligible_labels"]),
        "n_dropped_single_source": int(metrics_final["n_dropped_single_source"]),
        "subset_cell_coverage": round(metrics_final["subset_cell_coverage"], 4),
        "label_source_confound_nmi": (round(metrics_final["label_source_confound_nmi"], 4)
                                      if np.isfinite(metrics_final["label_source_confound_nmi"])
                                      else None),
        "eligible_cell_types": metrics_final["_eligible_labels"],
        "vlrr_trace": trace,
    }
    if "batch_predictability" in metrics_final:
        verdict["batch_predictability_from_features"] = round(metrics_final["batch_predictability"], 4)
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, help="csv/parquet with label/image[/source] columns")
    ap.add_argument("--label-key", default="cell_type")
    ap.add_argument("--image-key", default="image_id")
    ap.add_argument("--source-key", default=None, help="lab/source column (else image = unit)")
    ap.add_argument("--min-sources", type=int, default=2)
    ap.add_argument("--max-sources", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = (pd.read_parquet(a.meta) if a.meta.endswith(".parquet") else pd.read_csv(a.meta))
    verdict = run_vlrr_screening(df, label_key=a.label_key, image_key=a.image_key,
                                 source_key=a.source_key, min_sources=a.min_sources,
                                 max_sources=a.max_sources)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    if a.out:
        import os
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(verdict, open(a.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
