"""Config C (the doc's framing): recover CELL STATE (epithelial/fibroblast/endothelial/...),
treating ORGAN as the batch to correct. Cell state is the biology; organ (which tissue the
cells came from) is a domain/nuisance. Well-posed because ~23 normalized cell states span
>= 2 organs (raw cell_type is nested in organ; the normalized state is not).

Reuses the SAVED stage-3 embedding (no re-embed) + the existing VLRR batch-correction loop
(run_vlrr_batch_correction) with batch_key='organ_category'. Reports, on the subset of cell
states spanning >= min_batches organs:
  * RAW  : cell-state recovery (image-grouped kNN/linear) + organ_predictability
  * HARMONY(organ): same, after correcting organ -> does removing organ HELP cell state?
  * ARI/NMI of the 1/10-subsample leiden vs cell state.

  python -m benchmarks.cell_state_HCMA.run_configC \
     --h5ad results/hcma_configA_stage3.h5ad --sub-h5ad results/hcma_configA_stage3_sub.h5ad \
     --out-dir results
"""
from __future__ import annotations

import argparse, json, os
import numpy as np, anndata as ad
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmarks.cell_state_HCMA.evaluate import multi_source_mask, grouped_probes, _get_embedding
from benchmarks.cell_state_HCMA.hcma_labels import attach_label
from benchmarks.cell_state_HCMA.stage4_harmony import run_vlrr_batch_correction

META = "data/HCMA/Human_Cell_Morphology_Atlas_HCMA/HCMA_metadata.csv"


def _predictability(X, y, *, seed=0):
    """5-fold CV accuracy predicting `y` (the batch) from X; higher = more batch-driven."""
    ye = LabelEncoder().fit_transform(np.asarray(y).astype(str))
    Xs = StandardScaler().fit_transform(np.asarray(X, float))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return float(cross_val_score(clf, Xs, ye, cv=5).mean())


def _probe(X, obs, label_key):
    y = LabelEncoder().fit_transform(obs[label_key].astype(str).to_numpy())
    groups = obs["image_name"].astype(str).to_numpy()
    return grouped_probes(np.asarray(X, float), y, groups)


def run(h5ad, sub_h5ad, out_dir, *, label_key="cell_state", batch_key="organ_category",
        min_batches=2, max_iters=4, probe_subset=0):
    os.makedirs(out_dir, exist_ok=True)
    full = ad.read_h5ad(h5ad)
    full.obs[label_key] = attach_label(full.obs, META, label_key)
    full = full[full.obs[label_key].notna() & (full.obs[label_key].astype(str) != "unknown")].copy()

    # eligible = cell states spanning >= min_batches distinct batch groups
    keep = multi_source_mask(full.obs, label_key, batch_key, min_batches)
    sub = full[keep].copy()
    if probe_subset and sub.n_obs > probe_subset:                # speed: subsample for the probe
        idx = np.sort(np.random.RandomState(0).choice(sub.n_obs, probe_subset, replace=False))
        sub = sub[idx].copy()
        print(f"[configC] probe subsampled to {probe_subset}", flush=True)
    y_lab = sub.obs[label_key].astype(str)
    print(f"[configC] {full.n_obs} cells -> {sub.n_obs} in {y_lab.nunique()} cell states "
          f"spanning >= {min_batches} {batch_key}; batch has {sub.obs[batch_key].nunique()} groups",
          flush=True)

    rawX = _get_embedding(sub, "X")
    meta = sub.obs.reset_index(drop=True)
    batch_vals = sub.obs[batch_key].to_numpy()

    raw_probe = _probe(rawX, sub.obs, label_key)
    raw_bpred = _predictability(rawX, batch_vals)
    print(f"[configC] RAW cell-state linear={raw_probe['linear_accuracy']:.3f} "
          f"knn={raw_probe['knn_accuracy']:.3f} | organ_predictability={raw_bpred:.3f}", flush=True)

    # --- VLRR Harmony correcting the ORGAN batch ---
    res, traj = run_vlrr_batch_correction(rawX, meta, batch_key=batch_key, max_iters=max_iters)
    corrected = np.asarray(res.result["corrected_cls_output"], float)
    cor_probe = _probe(corrected, sub.obs, label_key)
    cor_bpred = _predictability(corrected, batch_vals)
    print(f"[configC] HARMONY cell-state linear={cor_probe['linear_accuracy']:.3f} "
          f"knn={cor_probe['knn_accuracy']:.3f} | organ_predictability={cor_bpred:.3f}", flush=True)

    # ARI/NMI: 1/10-subsample leiden vs cell state (on the eligible states)
    subc = ad.read_h5ad(sub_h5ad)
    subc.obs[label_key] = attach_label(subc.obs, META, label_key)
    m2 = subc.obs[label_key].isin(set(y_lab.unique())).to_numpy()
    ys = LabelEncoder().fit_transform(subc.obs[label_key][m2].astype(str))
    cl = subc.obs["leiden_0.5"][m2].astype(str).to_numpy()
    ari = float(adjusted_rand_score(ys, cl)); nmi = float(normalized_mutual_info_score(ys, cl))

    report = {
        "framing": f"label={label_key} (biology), batch={batch_key} (corrected)",
        "n_cells_eligible": int(sub.n_obs), "n_cell_states": int(y_lab.nunique()),
        "n_batch_groups": int(sub.obs[batch_key].nunique()),
        "clustering_vs_cellstate": {"ARI": ari, "NMI": nmi, "on": "1/10 subsample"},
        "RAW": {"cellstate_linear_acc": raw_probe["linear_accuracy"],
                "cellstate_knn_acc": raw_probe["knn_accuracy"],
                "cellstate_linear_nll": raw_probe["linear_nll"],
                f"{batch_key}_predictability": raw_bpred},
        "HARMONY": {"cellstate_linear_acc": cor_probe["linear_accuracy"],
                    "cellstate_knn_acc": cor_probe["knn_accuracy"],
                    "cellstate_linear_nll": cor_probe["linear_nll"],
                    f"{batch_key}_predictability": cor_bpred},
        "vlrr": {"status": res.status, "verdict": res.verdict.decision,
                 "n_retunes": max(0, len(res.trials) - 1), "trajectory": traj},
    }
    report["delta"] = {k: report["HARMONY"][k] - report["RAW"][k] for k in report["RAW"]}
    tag = f"_{globals().get('_OUT_TAG')}" if globals().get("_OUT_TAG") else ""
    out = os.path.join(out_dir, f"hcma_configC_cellstate_scores{tag}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("framing","n_cells_eligible","n_cell_states",
          "clustering_vs_cellstate","RAW","HARMONY","delta")}, indent=2))
    print(f"[configC] -> {out}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5ad", default="benchmarks/cell_state_HCMA/results/hcma_configA_stage3.h5ad")
    ap.add_argument("--sub-h5ad", default="benchmarks/cell_state_HCMA/results/hcma_configA_stage3_sub.h5ad")
    ap.add_argument("--out-dir", default="benchmarks/cell_state_HCMA/results")
    ap.add_argument("--label-key", default="cell_state")
    ap.add_argument("--batch-key", default="organ_category")
    ap.add_argument("--min-batches", type=int, default=2)
    ap.add_argument("--probe-subset", type=int, default=0, help="subsample cells for the probe (0=all)")
    ap.add_argument("--out-tag", default=None, help="suffix for the output json name")
    a = ap.parse_args()
    global _OUT_TAG; _OUT_TAG = a.out_tag
    run(a.h5ad, a.sub_h5ad, a.out_dir, label_key=a.label_key, batch_key=a.batch_key,
        min_batches=a.min_batches, probe_subset=a.probe_subset)


if __name__ == "__main__":
    main()
