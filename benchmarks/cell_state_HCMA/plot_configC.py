"""Config-C visualization: UMAP of the cell-state-eligible cells, RAW vs organ-corrected.
Row 1 (RAW embedding):      by cell_state | by organ (batch) | by Leiden cluster
Row 2 (Harmony(organ)):     by cell_state | by organ (batch) | by Leiden cluster
Computed on a random subsample (default 30k) so Leiden/UMAP + a viz-only Harmony are fast;
the quantitative RAW-vs-Harmony numbers live in run_configC (full 230k).
"""
from __future__ import annotations

import argparse, os
import numpy as np, anndata as ad, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.cm as cm

from benchmarks.cell_state_HCMA.evaluate import multi_source_mask, _get_embedding
from benchmarks.cell_state_HCMA.hcma_labels import attach_label

META = "data/HCMA/Human_Cell_Morphology_Atlas_HCMA/HCMA_metadata.csv"


def _umap_leiden(X):
    a = ad.AnnData(X=np.ascontiguousarray(X, np.float32))
    import scanpy as sc
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X")
    sc.tl.leiden(a, resolution=0.5, key_added="leiden", flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(a)
    return np.asarray(a.obsm["X_umap"], float), a.obs["leiden"].astype(str).to_numpy()


def _scatter(ax, P, labels, title, legend=True, max_leg=20):
    cats = pd.Categorical(pd.Series(labels).astype(str))
    for i, c in enumerate(cats.categories):
        m = cats.codes == i
        ax.scatter(P[m, 0], P[m, 1], s=3, alpha=0.5, color=cm.tab20(i % 20), label=f"{c} ({m.sum()})")
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    if legend and len(cats.categories) <= max_leg:
        ax.legend(fontsize=5, markerscale=2, loc="best", ncol=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5ad", default="benchmarks/cell_state_HCMA/results/hcma_configA_stage3.h5ad")
    ap.add_argument("--out", default="benchmarks/cell_state_HCMA/results/hcma_configC_umap.png")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--batch-key", default="organ_category")
    a = ap.parse_args()

    full = ad.read_h5ad(a.h5ad)
    full.obs["cell_state"] = attach_label(full.obs, META, "cell_state")
    full = full[full.obs["cell_state"].notna() & (full.obs["cell_state"].astype(str) != "unknown")].copy()
    keep = multi_source_mask(full.obs, "cell_state", a.batch_key, 2)
    sub = full[keep].copy()
    n = min(a.n, sub.n_obs)
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(sub.n_obs, n, replace=False))
    sub = sub[idx].copy()
    state = sub.obs["cell_state"].astype(str).to_numpy()
    organ = sub.obs[a.batch_key].astype(str).to_numpy()
    print(f"[plotC] {n} cells | {len(set(state))} states | {len(set(organ))} {a.batch_key} groups", flush=True)

    rawX = _get_embedding(sub, "X")
    from shape.tools.batch_corrector.tool import Batch_Corrector_Tool
    corX = np.asarray(Batch_Corrector_Tool().execute(
        cls_output=rawX, metadata=sub.obs.reset_index(drop=True),
        batch_key=a.batch_key, harmony_params=None)["corrected_cls_output"], float)
    print("[plotC] Harmony(organ) done on subsample; running UMAPs ...", flush=True)

    Praw, lraw = _umap_leiden(rawX)
    Pcor, lcor = _umap_leiden(corX)

    fig, ax = plt.subplots(2, 3, figsize=(20, 12.5))
    _scatter(ax[0, 0], Praw, state, "RAW: by CELL STATE (label)")
    _scatter(ax[0, 1], Praw, organ, "RAW: by ORGAN (batch)")
    _scatter(ax[0, 2], Praw, lraw, f"RAW: Leiden clusters ({len(set(lraw))})", legend=False)
    _scatter(ax[1, 0], Pcor, state, "HARMONY(organ): by CELL STATE (label)")
    _scatter(ax[1, 1], Pcor, organ, "HARMONY(organ): by ORGAN (batch)")
    _scatter(ax[1, 2], Pcor, lcor, f"HARMONY(organ): Leiden clusters ({len(set(lcor))})", legend=False)
    fig.suptitle("HCMA Config-C: cell-state embedding, RAW (top) vs organ-corrected (bottom)  "
                 f"[{n}-cell subsample]", fontsize=13)
    fig.tight_layout(); fig.savefig(a.out, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"[plotC] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
