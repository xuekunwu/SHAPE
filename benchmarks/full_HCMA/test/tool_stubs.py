"""Stand-ins for the SHAPE tool classes in test_tools.py (FULL_HCMA_TOOL_STUBS, see tools/_common.py).

- Cell_Segmenter_Tool: a threshold segmenter with the real tool's file layout and return keys (the
  real one needs the Cellpose-SAM weights); an image without foreground fails like the real tool's QC;
  FULL_HCMA_STUB_CRASH=<image_id> kills the process on that image.
- Single_Cell_Cropper_Tool: the REAL cropper (CPU only).
- Cell_State_Analyzer_Tool: the real class with execute() replaced by a seeded feature stand-in (no
  DINOv3); its clustering methods stay the real ones (cluster.py uses them).
- Batch_Corrector_Tool: the real tool (its real detector) with a per-batch mean-centring runner
  instead of harmonypy.
The real modules are imported on first use, so a worker process pays only for what it uses.
"""
from __future__ import annotations

import os
import time

import numpy as np

_BUILT: dict = {}


class Cell_Segmenter_Tool:
    def __init__(self):
        self.model_type = "cpsam"
        self.device = os.environ.get("FULL_HCMA_STUB_DEVICE", "cuda")   # the device it reports, not used

    def execute(self, image, diameter=30, flow_threshold=0.4, cellprob_threshold=0, min_size=15, niter=None,
                bsize=256, augment=False, resample=True, tile_overlap=0.1, model_type="cpsam",
                query_cache_dir=None, image_id=None, color_class_filter=None):
        from PIL import Image
        from skimage.measure import label
        from skimage.morphology import remove_small_objects

        if os.environ.get("FULL_HCMA_STUB_CRASH") == image_id:      # a worker process that dies
            os._exit(3)
        started = time.time()
        img = np.asarray(Image.open(image).convert("L"), dtype=np.float32)
        mask = label(img > 100).astype(np.int32)
        if mask.max():
            mask = label(remove_small_objects(mask > 0, min_size=int(min_size))).astype(np.int32)
        out = os.path.join(query_cache_dir, "output_visualizations")
        os.makedirs(out, exist_ok=True)
        mask_path = os.path.join(out, f"cell_mask_{image_id}_a1.npy")
        overlay_path = os.path.join(out, f"cell_overlay_{image_id}_a1.png")
        viz_path = os.path.join(out, f"cell_mask_viz_{image_id}_a1.png")
        np.save(mask_path, mask)
        Image.fromarray(np.clip(img + 60 * (mask > 0), 0, 255).astype(np.uint8)).save(overlay_path)
        Image.fromarray(((mask > 0) * 255).astype(np.uint8)).save(viz_path)
        n_cells = int(len(np.unique(mask)) - 1)
        result = {"summary": f"{n_cells} cells", "status": "success" if n_cells else "failed", "cell_count": n_cells,
                  "visual_outputs": [overlay_path, viz_path], "deliverables": [overlay_path], "mask_path": mask_path,
                  "overlay_path": overlay_path, "output_path": overlay_path, "image_id": image_id, "attempt": 1,
                  "segmented_channel": None, "flow_error": 0.1, "overlay_qc": {"cell_count": n_cells},
                  "seconds": round(time.time() - started, 1),
                  "parameters": {"diameter": diameter, "flow_threshold": flow_threshold,
                                 "cellprob_threshold": cellprob_threshold, "min_size": min_size, "niter": niter,
                                 "bsize": bsize, "augment": augment, "resample": resample,
                                 "tile_overlap": tile_overlap, "color_class_filter": color_class_filter}}
        if not n_cells:
            result["error"] = "Segmentation output failed visual QC: cells=0"
        return result


def _analyzer():
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool as Real
    from shape.tools.cell_state_analyzer.tool import _validate_identity_metadata, resolve_technical_metadata

    class Cell_State_Analyzer_Tool(Real):
        calls: list = []

        def execute(self, cell_crops=None, cell_metadata=None, query_cache_dir=None, seed=0, cluster_resolution=0.5,
                    n_neighbors=15, n_pcs=None, **levers):
            import anndata as ad
            from skimage import io

            type(self).calls.append({"n": len(cell_crops), "seed": seed, "metadata": cell_metadata[0], **levers})
            ids = _validate_identity_metadata(cell_metadata)
            groups, batches, report = resolve_technical_metadata(cell_metadata, [m["image_name"] for m in cell_metadata])
            rng = np.random.default_rng(seed)
            features = []
            for path in cell_crops:
                pixels = np.asarray(io.imread(path), dtype=np.float32)
                features.append([pixels.mean(), pixels.std(), pixels.shape[0], pixels.shape[1], *rng.normal(size=12)])
            adata = ad.AnnData(X=np.asarray(features, dtype=np.float32))
            adata.obs_names = ids
            adata.obs["cell_id"] = ids
            adata.obs["crop_id"] = ids
            adata.obs["image_name"] = [m["image_name"] for m in cell_metadata]
            adata.obs["technical_group"] = groups
            adata.obs["batch"] = batches
            out = os.path.join(query_cache_dir, "cell_state_analysis")
            os.makedirs(out, exist_ok=True)
            path = os.path.join(out, "cell_state_analyzed.h5ad")
            adata.write(path)
            with open(os.path.join(out, "best_model.pth"), "wb") as handle:     # stands in for the checkpoint
                handle.write(b"0" * 1024)
            return {"summary": "stub", "cell_count": len(cell_crops), "mode": "training", "epochs_trained": 2,
                    "final_loss": 0.5, "loss_history": [1.0, 0.5], "training_config": {"stop_reason": "stub",
                    "batch_size": 4}, "analyzer_kwargs": dict(levers, seed=seed), "adata_path": path,
                    "source_reliability_report": report, "cluster_resolution": cluster_resolution,
                    "n_neighbors": n_neighbors}

    return Cell_State_Analyzer_Tool


def _mean_centring(features, metadata, batch_key, params):
    batches = metadata[batch_key].astype(str).to_numpy()
    out = np.asarray(features, dtype=float).copy()
    for batch in np.unique(batches):
        out[batches == batch] -= out[batches == batch].mean(axis=0)
    return out, {"stub": True}


def _batch():
    from shape.tools.batch_corrector.tool import Batch_Corrector_Tool as Real

    class Batch_Corrector_Tool(Real):
        def execute(self, *args, **kwargs):
            kwargs.setdefault("harmony_runner", _mean_centring)
            return super().execute(*args, **kwargs)

    return Batch_Corrector_Tool


def __getattr__(name):
    if name not in _BUILT:
        if name == "Single_Cell_Cropper_Tool":
            from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool as value
        elif name == "Cell_State_Analyzer_Tool":
            value = _analyzer()
        elif name == "Batch_Corrector_Tool":
            value = _batch()
        elif name == "DEFAULT_CONFIG":
            from shape.tools.batch_corrector.tool import DEFAULT_CONFIG as value
        else:
            raise AttributeError(name)
        _BUILT[name] = value
    return _BUILT[name]
