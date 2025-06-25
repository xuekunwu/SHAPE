import scanpy as sc
import numpy as np
import os

class ActivationScorerTool:
    """
    Tool to quantify fibroblast activation score by mapping adata to a reference map.
    """

    def run(self, input_adata_path: str, reference_path: str, output_adata_path: str = "adata_with_activation.h5ad") -> str:
        if not os.path.exists(input_adata_path):
            raise FileNotFoundError(f"Input adata file not found: {input_adata_path}")
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference adata file not found: {reference_path}")

        adata = sc.read(input_adata_path)
        adata_ref = sc.read(reference_path)

        try:
            sc.tl.ingest(adata, adata_ref)
        except Exception as e:
            raise RuntimeError(f"Scanpy ingest failed: {e}")

        sc.pp.neighbors(adata)
        sc.tl.diffmap(adata)
        adata.uns['iroot'] = 0  # TODO: choose root cell appropriately
        sc.tl.dpt(adata, n_dcs=10)

        if "dpt_pseudotime" not in adata_ref.obs or "dpt_pseudotime" not in adata.obs:
            raise ValueError("dpt_pseudotime missing in adata or reference.")
        pt_min = adata_ref.obs["dpt_pseudotime"].min()
        pt_max = adata_ref.obs["dpt_pseudotime"].max()
        adata.obs["norm_dpt"] = (adata.obs["dpt_pseudotime"] - pt_min) / (pt_max - pt_min)

        class_weights = {
            "dead": 0,
            "q-Fb": 0.5,
            "proto-MyoFb": 2,
            "p-MyoFb": 3,
            "np-MyoFb": 4
        }
        if "predicted_class" not in adata.obs:
            raise ValueError("predicted_class column missing in adata.obs.")
        adata.obs["class_weight"] = adata.obs["predicted_class"].map(class_weights).astype(float)
        adata.obs["activation_score"] = adata.obs["norm_dpt"] * adata.obs["class_weight"]

        act_min = adata.obs["activation_score"].min()
        act_max = adata.obs["activation_score"].max()
        adata.obs["activation_score_norm"] = (adata.obs["activation_score"] - act_min) / (act_max - act_min)

        adata.write(output_adata_path)
        return output_adata_path