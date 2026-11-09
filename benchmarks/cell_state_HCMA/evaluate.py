"""FIXED evaluator for the HCMA end-to-end cell-state-inference benchmark.

Design contract (per the task spec):
  * The AGENT only produces output -- a label-free morphology representation. It receives
    the task "infer cell state without labels and output an AnnData with an embedding and a
    cluster assignment". It is NEVER shown the cell-type labels.
  * THIS script owns evaluation. It reads the agent's AnnData and scores it against the
    held-out cell-type labels, with the technical-variation confound controlled.

Two metric families (matching the paper's "label-free cell-state inference" claim + the
xlsx accuracy/NLL, which clustering alone cannot produce):
  1. clustering vs ground-truth cell type  -> ARI, NMI            (leiden cluster vs cell_type)
  2. representation quality (label probes)  -> kNN accuracy, linear-probe accuracy, linear NLL
     computed ON THE EMBEDDING, so they are independent of the number of clusters.

THE BIG PITFALL (R1 #18 technical-variation confound): HCMA is stitched from ~2,546 images
of varied provenance; many cell types come from only one or two images / one lab, so ANY
embedding can "recognize" a cell type by recognizing its imaging condition. Two guards,
both enforced here:
  (A) subset to cell types that appear in >= --min-sources INDEPENDENT source images;
  (B) every supervised probe uses GroupKFold grouped by image_id -- an image never appears
      in both train and test, so the probe cannot exploit per-image technical signatures.

The agent's AnnData must carry (labels are NOT among these -- the agent never sees them):
  obs[--image-key]      source image id          (the CV group + the multi-source unit)
  obs[--cluster-key]    agent's cluster label    (e.g. leiden_0.5)
  embedding             obsm[--embedding-key] if present, else falls back to adata.X

Ground-truth cell type is supplied to THIS script, never to the agent, via either:
  * --labels <file.csv|.parquet> with columns [--join-key, --cell-type-key] joined onto
    obs on --join-key (default: the image key -- one label per source image); or
  * obs[--cell-type-key] already present (only when a trusted harness injected it).

    # agent output has embedding=adata.X, cluster=leiden_0.5, image_name; GT held aside:
    python -m benchmarks.cell_state_HCMA.evaluate --adata cell_state_analyzed.h5ad \
        --image-key image_name --cluster-key leiden_0.5 --embedding-key X \
        --labels hcma_labels.parquet --join-key cell_id \
        --cell-type-key cell_type --min-sources 2 --out scores.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import anndata as ad
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             accuracy_score, log_loss)


def multi_source_mask(obs, cell_type_key: str, image_key: str, min_sources: int) -> np.ndarray:
    """Boolean mask keeping only cells whose cell type appears in >= min_sources distinct
    source images -- guard (A) against the single-image technical-variation confound."""
    n_src = obs.groupby(cell_type_key)[image_key].nunique()
    keep = n_src[n_src >= min_sources].index
    return obs[cell_type_key].isin(keep).to_numpy()


def grouped_probes(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
                   n_splits: int = 5, knn_k: int = 15, seed: int = 0) -> dict:
    """kNN + linear-probe accuracy and linear NLL under GroupKFold(image) -- guard (B).
    Test-fold classes absent from the train fold are dropped from that fold's scoring so
    accuracy/NLL stay well-defined; per-fold means are reported."""
    uniq_groups = np.unique(groups)
    n_splits = int(min(n_splits, len(uniq_groups)))
    if n_splits < 2:
        return {"error": "need >=2 image groups for GroupKFold"}
    gkf = GroupKFold(n_splits=n_splits)
    knn_acc, lin_acc, lin_nll = [], [], []
    for tr, te in gkf.split(X, y, groups):
        seen = set(np.unique(y[tr]))
        keep_te = np.isin(y[te], list(seen))
        if keep_te.sum() == 0:
            continue
        te = te[keep_te]
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        knn = KNeighborsClassifier(n_neighbors=min(knn_k, int(len(tr))), weights="distance").fit(Xtr, y[tr])
        knn_acc.append(accuracy_score(y[te], knn.predict(Xte)))
        lr = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1).fit(Xtr, y[tr])
        lin_acc.append(accuracy_score(y[te], lr.predict(Xte)))
        lin_nll.append(log_loss(y[te], lr.predict_proba(Xte), labels=lr.classes_))
    return {"knn_accuracy": float(np.mean(knn_acc)) if knn_acc else float("nan"),
            "linear_accuracy": float(np.mean(lin_acc)) if lin_acc else float("nan"),
            "linear_nll": float(np.mean(lin_nll)) if lin_nll else float("nan"),
            "n_folds": len(lin_acc)}


def _attach_labels(adata, labels_path, join_key, cell_type_key):
    """Join ground-truth cell type from an external table onto obs[cell_type_key].
    The agent's AnnData never carries the labels; this reunites them for scoring only."""
    import pandas as pd
    lab = (pd.read_parquet(labels_path) if labels_path.endswith(".parquet")
           else pd.read_csv(labels_path))
    for c in (join_key, cell_type_key):
        if c not in lab.columns:
            raise KeyError(f"--labels file missing column {c!r}")
    if join_key not in adata.obs:
        raise KeyError(f"AnnData.obs missing --join-key {join_key!r} to align labels")
    m = lab.drop_duplicates(join_key).set_index(join_key)[cell_type_key]
    adata.obs[cell_type_key] = adata.obs[join_key].astype(str).map(
        m.rename(index=str)).astype("object")
    n_missing = adata.obs[cell_type_key].isna().sum()
    if n_missing:
        raise ValueError(f"{n_missing} cells have no label after join on {join_key!r}")


def _get_embedding(adata, embedding_key):
    if embedding_key in ("X", None) or embedding_key not in adata.obsm:
        if embedding_key not in adata.obsm and embedding_key not in ("X", None):
            raise KeyError(f"embedding {embedding_key!r} not in obsm and != 'X'")
        return np.asarray(adata.X, dtype=float)
    return np.asarray(adata.obsm[embedding_key], dtype=float)


def evaluate(adata_path: str, *, cell_type_key="cell_type", image_key="image_id",
             cluster_key="leiden", embedding_key="X_emb", min_sources=2,
             labels_path=None, join_key=None) -> dict:
    adata = ad.read_h5ad(adata_path)
    if labels_path:
        _attach_labels(adata, labels_path, join_key or image_key, cell_type_key)
    for col in (cell_type_key, image_key, cluster_key):
        if col not in adata.obs:
            raise KeyError(f"AnnData.obs missing required column {col!r} "
                           "(pass --labels to inject ground-truth cell type)")

    mask = multi_source_mask(adata.obs, cell_type_key, image_key, min_sources)
    if mask.sum() == 0:
        raise ValueError("no cell type survives the multi-source subset -- the whole "
                         "dataset is single-image-confounded; cannot judge cell state")
    sub = adata[mask]
    y = LabelEncoder().fit_transform(sub.obs[cell_type_key].astype(str).to_numpy())
    groups = sub.obs[image_key].astype(str).to_numpy()
    clusters = sub.obs[cluster_key].astype(str).to_numpy()
    X = _get_embedding(sub, embedding_key)

    ari = float(adjusted_rand_score(y, clusters))
    nmi = float(normalized_mutual_info_score(y, clusters))
    probes = grouped_probes(X, y, groups)
    dropped = adata.obs.loc[~mask, cell_type_key].nunique()
    return {
        "n_cells_total": int(adata.n_obs),
        "n_cells_evaluated": int(mask.sum()),
        "n_cell_types_evaluated": int(len(np.unique(y))),
        "n_cell_types_dropped_single_source": int(dropped),
        "n_source_images": int(len(np.unique(groups))),
        "clustering_vs_celltype": {"ARI": ari, "NMI": nmi},
        "representation_image_grouped_CV": probes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--cell-type-key", default="cell_type")
    ap.add_argument("--image-key", default="image_id")
    ap.add_argument("--cluster-key", default="leiden")
    ap.add_argument("--embedding-key", default="X_emb",
                    help="obsm key holding the embedding; 'X' uses adata.X")
    ap.add_argument("--min-sources", type=int, default=2)
    ap.add_argument("--labels", default=None,
                    help="external GT table (csv/parquet) with [join-key, cell-type-key]")
    ap.add_argument("--join-key", default=None,
                    help="obs column to join --labels on (default: --image-key)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = evaluate(a.adata, cell_type_key=a.cell_type_key, image_key=a.image_key,
                   cluster_key=a.cluster_key, embedding_key=a.embedding_key,
                   min_sources=a.min_sources, labels_path=a.labels, join_key=a.join_key)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
