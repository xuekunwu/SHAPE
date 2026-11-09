"""Score the stage-3 embedding against ANY HCMA label (cell_type / cell_name / cell_system /
organ_category), reusing the saved artifacts -- no re-embed. Joins the requested label from
HCMA_metadata.csv onto the cells by image-filename stem, then:
  * ARI/NMI = leiden (from the 1/10 subsample h5ad) vs the label, on the subsample;
  * kNN + linear probe (image-grouped GroupKFold) on the FULL embedding.
Same multi-source guard (label needs >= min_sources independent image_names) as evaluate.py.

  python -m benchmarks.cell_state_HCMA.score_label --label-key cell_type \
     --h5ad results/hcma_configA_stage3.h5ad --sub-h5ad results/hcma_configA_stage3_sub.h5ad \
     --metadata data/HCMA/Human_Cell_Morphology_Atlas_HCMA/HCMA_metadata.csv \
     --out results/hcma_configA_stage3_scores_cell_type.json
"""
from __future__ import annotations

import argparse, json, os
import numpy as np, pandas as pd, anndata as ad
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmarks.cell_state_HCMA.evaluate import multi_source_mask, grouped_probes, _get_embedding
from benchmarks.cell_state_HCMA.hcma_labels import attach_label as _join_label


def score(h5ad, sub_h5ad, meta_path, label_key, image_key="image_name", min_sources=2):
    full = ad.read_h5ad(h5ad)
    full.obs[label_key] = _join_label(full.obs, meta_path, label_key, image_key)
    full = full[full.obs[label_key].notna()].copy()          # drop unmatched cells
    sub = ad.read_h5ad(sub_h5ad)
    sub.obs[label_key] = _join_label(sub.obs, meta_path, label_key, image_key)
    sub = sub[sub.obs[label_key].notna()].copy()

    # ARI/NMI: leiden vs label on the multi-source subset WITHIN the subsample
    ms = multi_source_mask(sub.obs, label_key, image_key, min_sources)
    ys = LabelEncoder().fit_transform(sub.obs[label_key][ms].astype(str))
    cl = sub.obs["leiden_0.5"][ms].astype(str).to_numpy()
    ari = float(adjusted_rand_score(ys, cl)); nmi = float(normalized_mutual_info_score(ys, cl))

    # Probes: image-grouped kNN + linear on the FULL multi-source subset
    msf = multi_source_mask(full.obs, label_key, image_key, min_sources)
    subf = full[msf]
    yf = LabelEncoder().fit_transform(subf.obs[label_key].astype(str))
    gf = subf.obs[image_key].astype(str).to_numpy()
    probes = grouped_probes(_get_embedding(subf, "X"), yf, gf)
    return {
        "label_key": label_key,
        "n_cells_total": int(full.n_obs),
        "n_cells_probed_full": int(msf.sum()),
        "n_labels_total": int(full.obs[label_key].nunique()),
        "n_labels_evaluated": int(len(np.unique(yf))),
        "n_labels_dropped_single_source": int(full.obs[label_key].nunique() - len(np.unique(yf))),
        "n_source_images": int(len(np.unique(gf))),
        "clustering_vs_label": {"ARI": ari, "NMI": nmi, "on": "1/10 subsample"},
        "representation_image_grouped_CV": probes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-key", required=True)
    ap.add_argument("--h5ad", default="benchmarks/cell_state_HCMA/results/hcma_configA_stage3.h5ad")
    ap.add_argument("--sub-h5ad", default="benchmarks/cell_state_HCMA/results/hcma_configA_stage3_sub.h5ad")
    ap.add_argument("--metadata", default="data/HCMA/Human_Cell_Morphology_Atlas_HCMA/HCMA_metadata.csv")
    ap.add_argument("--image-key", default="image_name")
    ap.add_argument("--min-sources", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = score(a.h5ad, a.sub_h5ad, a.metadata, a.label_key, a.image_key, a.min_sources)
    print(json.dumps(s, indent=2))
    out = a.out or f"benchmarks/cell_state_HCMA/results/hcma_configA_stage3_scores_{a.label_key}.json"
    json.dump(s, open(out, "w"), indent=2)
    print(f"[score] {a.label_key} -> {out}")


if __name__ == "__main__":
    main()
