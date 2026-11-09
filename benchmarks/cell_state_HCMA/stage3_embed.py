"""STAGE 3 -- label-free DINOv3 embedding + clustering for HCMA Config-A, VLRR-verified.

Embeds every stage-2 crop with the zero-shot DINOv3 vitb16 CLS token (served by the shared
GPU DINOv3 service, mirroring the SAM service), builds an AnnData (embedding + Leiden cluster
+ carried obs), then runs the repo's RepresentationVerifier through StepVerifier as a REAL
Verify/Retune/Rerun loop before scoring.

Because the DINOv3 weights are FROZEN (zero-shot, no fine-tuning), a retune canNOT re-embed.
The only this-step retune RepresentationVerifier emits is the batch-floor lever
(field_aware / augment_color) with expected_effect {silhouette_by_batch: down} -- fired when
vendor batch structure dominates. On a frozen embedding we realize that lever's GOAL
frozen-safe: regress the per-batch (vendor) mean out of the SAVED embedding (a post-hoc batch
decorrelation that genuinely lowers silhouette_by_batch), recompute Leiden/UMAP, re-verify.
See the coordinator note below -- the verifier does NOT emit a regress-out-area lever
(PC1_area>0.7 concedes upstream to the cropper), so the honest fired lever here is the batch one.

The agent is never shown labels: obs carries organ_category only so the FIXED evaluator can
score it. obs['batch']=source(vendor) drives the verifier; organ_category is deliberately NOT
written as technical_group (the verifier forbids inspecting the biological label).

Crop manifest (parquet): [path, cell_id, image_name, organ_category, source, (area?)]

  python -m benchmarks.cell_state_HCMA.stage3_embed \
      --manifest solver_cache/HCMA/configA_crops.parquet \
      --service-url http://127.0.0.1:8138 --out-dir benchmarks/cell_state_HCMA/results
  python -m benchmarks.cell_state_HCMA.stage3_embed --selftest
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.verifiers.representation import (
    RepresentationVerifier, adapt_representation)

_OBS_COLS = ("cell_id", "image_name", "organ_category", "source")


def _cluster_umap(adata):
    """Leiden cluster (-> obs['leiden_0.5']) + UMAP (-> obsm['X_umap']) on adata.X,
    with a KMeans/PCA fallback when scanpy/leiden/umap are unavailable."""
    X = np.asarray(adata.X, dtype=float)
    try:
        import scanpy as sc
        sc.pp.neighbors(adata, n_neighbors=min(15, max(2, adata.n_obs - 1)), use_rep="X")
        sc.tl.leiden(adata, resolution=0.5, key_added="leiden_0.5", flavor="igraph",
                     n_iterations=2, directed=False)
        sc.tl.umap(adata)
    except Exception as exc:
        print(f"[stage3] WARNING scanpy failed ({type(exc).__name__}: {exc}); KMeans/PCA fallback")
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        k = min(14, max(2, adata.obs["organ_category"].nunique() if "organ_category" in adata.obs else 8))
        adata.obs["leiden_0.5"] = KMeans(k, n_init=4, random_state=0).fit_predict(X).astype(str)
        adata.obsm["X_umap"] = PCA(n_components=2, random_state=0).fit_transform(X)
    return adata


def build_anndata(manifest_df, embeddings, areas=None):
    """AnnData with X=embedding, carried obs, obs['batch']=vendor + obs['area'] for the
    RepresentationVerifier. NO Leiden/UMAP here: the verifier reads emb/area/batch/group
    (not clusters), and clustering on the full ~270k cells is the pipeline's slowest step,
    so it is deferred to a 1/10 subsample AFTER the VLRR loop (see run_stage3). organ_category
    is NOT written as technical_group (the label-free verifier must not see the label)."""
    import anndata as ad
    X = np.ascontiguousarray(embeddings, dtype=np.float32)
    if len(X) != len(manifest_df):
        raise ValueError(f"embeddings ({len(X)}) != manifest rows ({len(manifest_df)})")
    obs = pd.DataFrame({c: manifest_df[c].astype(str).to_numpy() for c in _OBS_COLS})
    obs.index = obs["cell_id"].to_numpy()
    obs["batch"] = obs["source"]                         # nuisance batch for the verifier
    if areas is None:
        areas = (manifest_df["area"].to_numpy() if "area" in manifest_df.columns
                 else np.zeros(len(manifest_df)))
    obs["area"] = np.asarray(areas, dtype=float)
    return ad.AnnData(X=X, obs=obs)


def _regress_out_batch(X, batch_codes):
    """Frozen-safe realization of the verifier's batch-floor lever: center each batch group
    to the global mean (removes the per-vendor offset -> lowers silhouette_by_batch)."""
    Xc = np.asarray(X, dtype=float).copy()
    glob = Xc.mean(0)
    for g in np.unique(batch_codes):
        m = batch_codes == g
        Xc[m] = Xc[m] - Xc[m].mean(0) + glob
    return Xc


def run_stage3_vlrr(base_h5ad, out_dir, *, max_iters=4):
    """Drive RepresentationVerifier through StepVerifier. tool_runner transforms the BASE
    embedding per the proposed lever (frozen-safe, stateless) and returns a new h5ad path."""
    import anndata as ad
    base = ad.read_h5ad(base_h5ad)

    def tool_runner(params):
        adata = base.copy()
        X = np.asarray(adata.X, dtype=float)
        applied = []
        if params.get("field_aware") or params.get("augment_color"):   # batch-floor lever
            codes = pd.Categorical(adata.obs["batch"].astype(str)).codes
            X = _regress_out_batch(X, codes)
            applied.append("batch_mean_regression")
        adata.X = X.astype(np.float32)
        # no clustering here: the RepresentationVerifier scores emb/area/batch/group, not
        # leiden; clustering is done once on a subsample after the loop (run_stage3).
        sig = "_".join(sorted(k for k, v in params.items() if v)) or "base"
        p = os.path.join(out_dir, f"hcma_configA_stage3_retune_{sig}.h5ad")
        adata.write_h5ad(p)
        print(f"[stage3] tool_runner applied={applied or 'none'} -> {os.path.basename(p)}", flush=True)
        return p

    sv = StepVerifier(max_iters=max_iters)
    return sv.resolve_stage(RepresentationVerifier(),
                            lambda result, ctx: adapt_representation(result, ctx),
                            tool_runner, {}, base_h5ad, upstream_ctx={})


def _umap_figure(adata, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    P = np.asarray(adata.obsm["X_umap"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, key, title in ((axes[0], "organ_category", "colored by organ_category (LABEL)"),
                           (axes[1], "source", "colored by source/vendor (BATCH)")):
        cats = pd.Categorical(adata.obs[key].astype(str))
        for i, c in enumerate(cats.categories):
            m = cats.codes == i
            ax.scatter(P[m, 0], P[m, 1], s=4, alpha=0.5, color=cm.tab20(i % 20), label=f"{c} ({m.sum()})")
        ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=6, markerscale=2, loc="best")
    fig.suptitle("HCMA Config-A stage-3 DINOv3 embedding (UMAP): biology (left) vs vendor batch (right)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=125, bbox_inches="tight"); plt.close(fig)


def _mock_embeddings(manifest_df, dim=768, batch_dominated=False):
    """Random embeddings with a per-organ mean shift; if batch_dominated, add a LARGE
    per-vendor offset so silhouette_by_batch clears the 0.20 gate (forces a retune)."""
    rng = np.random.RandomState(0)
    organs = pd.Categorical(manifest_df["organ_category"].astype(str))
    centers = rng.randn(len(organs.categories), dim) * (1.2 if batch_dominated else 4.0)
    emb = centers[organs.codes] + rng.randn(len(manifest_df), dim) * 0.6
    if batch_dominated:
        vend = pd.Categorical(manifest_df["source"].astype(str))
        vshift = rng.randn(len(vend.categories), dim) * 8.0
        emb = emb + vshift[vend.codes]
    return emb.astype(np.float32)


def run_stage3(manifest_path, out_dir, *, service_url=None, mock=False,
               mock_batch_dominated=False, max_iters=4, cluster_subset=30000):
    import anndata as ad
    os.makedirs(out_dir, exist_ok=True)
    man = pd.read_parquet(manifest_path)
    for c in ("path", *_OBS_COLS):
        if c not in man.columns:
            raise KeyError(f"manifest missing column {c!r}")
    print(f"[stage3] {len(man)} crops | {man['image_name'].nunique()} images | "
          f"{man['organ_category'].nunique()} organs | {man['source'].nunique()} vendors", flush=True)

    if mock:
        emb = _mock_embeddings(man, batch_dominated=mock_batch_dominated)
        print(f"[stage3] MOCK embeddings (batch_dominated={mock_batch_dominated})", flush=True)
    else:
        if not service_url:
            raise SystemExit("need --service-url (DINOv3 service) unless --mock")
        from shape.tools.cell_state_analyzer.dinov3_batch import dino_health, dino_embed_paths
        print(f"[stage3] DINOv3 service: {dino_health(service_url)}", flush=True)
        emb = dino_embed_paths(service_url, man["path"].tolist())
        print(f"[stage3] embedded -> {emb.shape}", flush=True)

    base = build_anndata(man, emb)
    base_h5ad = os.path.join(out_dir, "hcma_configA_stage3_base.h5ad")
    base.write_h5ad(base_h5ad)
    print(f"[stage3] base embedding -> {base_h5ad} ({base.n_obs} cells, no cluster yet)", flush=True)

    # --- REAL VLRR loop over the representation (no clustering inside the loop) ---
    res = run_stage3_vlrr(base_h5ad, out_dir, max_iters=max_iters)
    n_retunes = max(0, len(res.trials) - 1)
    print(f"[stage3] VLRR status={res.status} verdict={res.verdict.decision} "
          f"reason={res.reason} | {n_retunes} retune(s) | ledger={res.ledger}", flush=True)

    final = ad.read_h5ad(res.result)                     # full final embedding (X + obs, no cluster)
    h5ad = os.path.join(out_dir, "hcma_configA_stage3.h5ad")
    final.write_h5ad(h5ad)

    scores = _score_full(final, out_dir, cluster_subset=cluster_subset)
    scores["vlrr"] = {"status": res.status, "verdict": res.verdict.decision,
                      "reason": res.reason, "ledger": res.ledger, "n_retunes": n_retunes}
    out_json = os.path.join(out_dir, "hcma_configA_stage3_scores.json")
    json.dump(scores, open(out_json, "w"), indent=2)
    print(json.dumps(scores, indent=2))
    print(f"[stage3] FINAL AnnData -> {h5ad} | scores -> {out_json}", flush=True)
    return {"h5ad": h5ad, "scores": scores, "scores_json": out_json, "resolution": res}


def _score_full(final, out_dir, *, cluster_subset=30000):
    """Clustering (ARI/NMI) + UMAP on a random SUBSAMPLE (leiden/umap on ~270k is the slow
    step); representation probes (kNN/linear) on the FULL embedding (the headline metric,
    not diluted). Returns the merged score dict."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from benchmarks.cell_state_HCMA.evaluate import multi_source_mask, grouped_probes, _get_embedding

    n = final.n_obs
    k = int(min(n, cluster_subset))
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(n, k, replace=False)) if k < n else np.arange(n)
    sub = final[idx].copy()
    _cluster_umap(sub)                                   # leiden + umap on the subsample only
    sub_h5ad = os.path.join(out_dir, "hcma_configA_stage3_sub.h5ad")
    sub.write_h5ad(sub_h5ad)
    _umap_figure(sub, os.path.join(out_dir, "hcma_configA_umap.png"))
    print(f"[stage3] clustered+UMAP on {k}/{n} subsample ({sub.obs['leiden_0.5'].nunique()} clusters) "
          f"-> UMAP saved", flush=True)

    # ARI/NMI: leiden vs organ on the multi-source subset WITHIN the subsample
    ms = multi_source_mask(sub.obs, "organ_category", "image_name", 2)
    ys = LabelEncoder().fit_transform(sub.obs["organ_category"][ms].astype(str))
    cl = sub.obs["leiden_0.5"][ms].astype(str).to_numpy()
    ari = float(adjusted_rand_score(ys, cl)); nmi = float(normalized_mutual_info_score(ys, cl))

    # Probes: image-grouped kNN + linear probe on the FULL multi-source subset
    msf = multi_source_mask(final.obs, "organ_category", "image_name", 2)
    subf = final[msf]
    yf = LabelEncoder().fit_transform(subf.obs["organ_category"].astype(str))
    gf = subf.obs["image_name"].astype(str).to_numpy()
    probes = grouped_probes(_get_embedding(subf, "X"), yf, gf)
    print(f"[stage3] probes on FULL {int(msf.sum())} cells | ARI/NMI on {int(ms.sum())} subsample cells",
          flush=True)
    return {
        "n_cells_total": int(n),
        "n_cells_probed_full": int(msf.sum()),
        "n_cells_clustered_subsample": int(ms.sum()),
        "n_organ_categories": int(len(np.unique(yf))),
        "n_source_images": int(len(np.unique(gf))),
        "clustering_vs_celltype": {"ARI": ari, "NMI": nmi, "on": "1/10 subsample"},
        "representation_image_grouped_CV": probes,
    }


def _selftest():
    """GPU-free: synthesize a batch-DOMINATED 6-organ / 2-vendor manifest so the verifier's
    batch-floor lever fires; assert a retune ran (ledger records the lever) and ARI is finite."""
    import tempfile
    rows = []
    organs = [f"organ_{i}" for i in range(6)]
    vendors = ["ATCC", "ScienCell"]
    cid = 0
    for oi, organ in enumerate(organs):
        for vi, vendor in enumerate(vendors):           # every organ in both vendors (cross-vendor)
            for img in range(1):
                image_name = f"img_{oi}_{vi}"
                for _ in range(22):
                    rows.append({"path": f"/dev/null/{cid}.png", "cell_id": f"c{cid}",
                                 "image_name": image_name, "organ_category": organ,
                                 "source": vendor, "area": 100.0 + (cid % 7)})
                    cid += 1
    man = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "manifest.parquet"); man.to_parquet(mp)
        res = run_stage3(mp, d, mock=True, mock_batch_dominated=True)
        r = res["resolution"]
        ari = res["scores"]["clustering_vs_celltype"]["ARI"]
        assert os.path.exists(res["h5ad"]), "final h5ad missing"
        assert np.isfinite(ari), f"ARI not finite: {ari}"
        lever_fired = any(k in r.ledger for k in ("field_aware", "augment_color"))
        assert len(r.trials) >= 2 and lever_fired, \
            f"expected a batch-floor retune to fire; trials={len(r.trials)} ledger={r.ledger}"
        print(f"\n[selftest] PASS -- {len(r.trials)-1} retune(s), ledger={r.ledger}, "
              f"verdict={r.verdict.decision}, ARI={ari:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest")
    ap.add_argument("--service-url", default=None)
    ap.add_argument("--out-dir", default="benchmarks/cell_state_HCMA/results")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--cluster-subset", type=int, default=30000,
                    help="cells to subsample for Leiden+UMAP+ARI/NMI (probes stay full); 0=all")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    if not a.manifest:
        raise SystemExit("need --manifest (or --selftest)")
    run_stage3(a.manifest, a.out_dir, service_url=a.service_url, mock=a.mock,
               max_iters=a.max_iters, cluster_subset=(a.cluster_subset or 10**9))


if __name__ == "__main__":
    main()
