"""JUMP-style scib composite for HCMA, RAW vs Harmony. Uses the same
shape.tools.batch_corrector.quant.evaluate_batch_correction as the JUMP benchmark, on a
subsample (the graph/silhouette/lisi/leiden metrics are O(n^2)-ish). map_controls/map_nonrep
are UNSUPPORTED on HCMA (no DMSO controls, no well/plate replicate structure) -> NaN and
dropped; the composite is the available-case 0.4*batch + 0.6*bio over the six transferable
metrics: batch = {graph_connectivity, lisi_batch(iLISI), silhouette_batch}, bio = {lisi_label,
leiden_ari, leiden_nmi, silhouette_label}.

  python -m benchmarks.cell_state_HCMA.score_scib --label-key cell_state --batch-key source --n 30000
"""
from __future__ import annotations

import argparse, json, os
import numpy as np, anndata as ad, pandas as pd

from benchmarks.cell_state_HCMA.evaluate import multi_source_mask, _get_embedding
from benchmarks.cell_state_HCMA.hcma_labels import attach_label
from shape.tools.batch_corrector import quant

META = "data/HCMA/Human_Cell_Morphology_Atlas_HCMA/HCMA_metadata.csv"
BATCH_M = ("graph_connectivity", "lisi_batch", "silhouette_batch")
BIO_M = ("lisi_label", "leiden_ari", "leiden_nmi", "silhouette_label")   # map_* dropped (no controls)


def _avail(scores):
    b = [scores[k] for k in BATCH_M if np.isfinite(scores.get(k, np.nan))]
    o = [scores[k] for k in BIO_M if np.isfinite(scores.get(k, np.nan))]
    batch = float(np.mean(b)) if b else float("nan")
    bio = float(np.mean(o)) if o else float("nan")
    return {**{k: scores.get(k) for k in BATCH_M + BIO_M},
            "batch_score": batch, "bio_score": bio, "overall": 0.4 * batch + 0.6 * bio}


def _scib(X, meta, label_key, batch_key):
    s = quant.evaluate_batch_correction(
        np.asarray(X, float), meta, label_key=label_key, batch_key=batch_key,
        plate_keys=["image_name"], control_label=None, well_key=None)
    return _avail(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5ad", default="benchmarks/cell_state_HCMA/results/hcma_configA_stage3.h5ad")
    ap.add_argument("--label-key", default="cell_state")
    ap.add_argument("--batch-key", default="source")
    ap.add_argument("--min-batches", type=int, default=2)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    full = ad.read_h5ad(a.h5ad)
    full.obs[a.label_key] = attach_label(full.obs, META, a.label_key)
    full = full[full.obs[a.label_key].notna() & (full.obs[a.label_key].astype(str) != "unknown")].copy()
    keep = multi_source_mask(full.obs, a.label_key, a.batch_key, a.min_batches)
    sub = full[keep].copy()
    n = min(a.n, sub.n_obs)
    idx = np.sort(np.random.RandomState(0).choice(sub.n_obs, n, replace=False))
    sub = sub[idx].copy()
    meta = sub.obs.reset_index(drop=True)
    rawX = _get_embedding(sub, "X")
    print(f"[scib] {n} cells | label={a.label_key}({sub.obs[a.label_key].nunique()}) "
          f"| batch={a.batch_key}({sub.obs[a.batch_key].nunique()})", flush=True)

    from shape.tools.batch_corrector.tool import Batch_Corrector_Tool
    corX = np.asarray(Batch_Corrector_Tool().execute(
        cls_output=rawX, metadata=meta, batch_key=a.batch_key,
        harmony_params=None)["corrected_cls_output"], float)

    raw = _scib(rawX, meta, a.label_key, a.batch_key)
    print(f"[scib] RAW overall={raw['overall']:.3f} (batch={raw['batch_score']:.3f} bio={raw['bio_score']:.3f})", flush=True)
    cor = _scib(corX, meta, a.label_key, a.batch_key)
    print(f"[scib] HARMONY overall={cor['overall']:.3f} (batch={cor['batch_score']:.3f} bio={cor['bio_score']:.3f})", flush=True)

    report = {"framing": f"label={a.label_key}, batch={a.batch_key}", "n_cells": n,
              "note": "map_controls/map_nonrep unsupported on HCMA (no controls/wells); "
                      "overall = available-case 0.4*batch + 0.6*bio",
              "RAW": raw, "HARMONY": cor,
              "delta_overall": cor["overall"] - raw["overall"]}
    out = a.out or f"benchmarks/cell_state_HCMA/results/hcma_scib_{a.label_key}_{a.batch_key}.json"
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2)); print(f"[scib] -> {out}")


if __name__ == "__main__":
    main()
