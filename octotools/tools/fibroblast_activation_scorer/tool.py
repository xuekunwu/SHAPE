import scanpy as sc
import numpy as np
import os
from huggingface_hub import hf_hub_download

class FibroblastActivationScorerTool:
    """
    Tool to quantify fibroblast activation score by mapping adata to a reference map.
    """
    
    def __init__(self):
        # Hugging Face dataset info
        self.repo_id = "5xuekun/adata_reference"
        self.filename = "adata_reference.h5ad"
    
    def _download_reference_file(self) -> str:
        """
        Download reference file from Hugging Face dataset.
        
        Returns:
            str: Local path to downloaded reference file
            
        Raises:
            Exception: If download fails
        """
        try:
            print(f"Downloading reference file from {self.repo_id}...")
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                repo_type="dataset"
            )
            print(f"Reference file downloaded to: {local_path}")
            return local_path
        except Exception as e:
            raise Exception(f"Failed to download reference file from {self.repo_id}: {e}")
    
    def run(self, input_adata_path: str, output_adata_path: str = "adata_with_activation.h5ad") -> str:
        """
        Map input adata to reference, compute activation score, and save result.
        
        Args:
            input_adata_path (str): Path to input AnnData file.
            output_adata_path (str): Path to save output AnnData file.
            
        Returns:
            str: Path to output AnnData file.
        """
        # Check input file existence
        if not os.path.exists(input_adata_path):
            raise FileNotFoundError(f"Input adata file not found: {input_adata_path}")
        
        # Download reference file from Hugging Face
        reference_path = self._download_reference_file()
        print(f"Using reference file: {reference_path}")

        # 1. Load adata and reference
        print("Loading input adata...")
        adata = sc.read(input_adata_path)
        print("Loading reference adata...")
        adata_ref = sc.read(reference_path)

        # 2. Ingest mapping
        print("Performing ingest mapping...")
        try:
            sc.tl.ingest(adata, adata_ref)
        except Exception as e:
            raise RuntimeError(f"Scanpy ingest failed: {e}")

        # 3. Compute neighbors, diffmap, dpt
        print("Computing neighbors, diffmap, and dpt...")
        sc.pp.neighbors(adata)
        sc.tl.diffmap(adata)
        adata.uns['iroot'] = 0  # TODO: choose root cell appropriately
        sc.tl.dpt(adata, n_dcs=10)

        # 4. Normalize pseudotime
        if "dpt_pseudotime" not in adata_ref.obs or "dpt_pseudotime" not in adata.obs:
            raise ValueError("dpt_pseudotime missing in adata or reference.")
        pt_min = adata_ref.obs["dpt_pseudotime"].min()
        pt_max = adata_ref.obs["dpt_pseudotime"].max()
        adata.obs["norm_dpt"] = (adata.obs["dpt_pseudotime"] - pt_min) / (pt_max - pt_min)

        # 5. Define class weights
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

        # 6. Compute activation score
        print("Computing activation scores...")
        adata.obs["activation_score"] = adata.obs["norm_dpt"] * adata.obs["class_weight"]

        # 7. Normalize activation score to [0, 1]
        act_min = adata.obs["activation_score"].min()
        act_max = adata.obs["activation_score"].max()
        adata.obs["activation_score_norm"] = (adata.obs["activation_score"] - act_min) / (act_max - act_min)

        # 8. Save result
        print(f"Saving result to: {output_adata_path}")
        adata.write(output_adata_path)
        
        # Print summary
        print(f"Activation scoring completed successfully!")
        print(f"Total cells processed: {len(adata)}")
        print(f"Activation score range: {act_min:.3f} - {act_max:.3f}")
        print(f"Normalized score range: 0.000 - 1.000")
        
        return output_adata_path 