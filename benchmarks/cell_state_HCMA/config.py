"""
Canonical configuration for the cell_state_HCMA end-to-end benchmark.

HCMA = Human Cell Morphology Atlas: ~2,546 microscopy images stitched from many
independent sources/labs. The agent is asked to do LABEL-FREE cell-state inference and
emit an AnnData carrying an embedding + a cluster assignment. Cell-type labels are NEVER
shown to the agent; a FIXED script (evaluate.py) owns scoring.

Pipeline shape (end to end, unlike representation_HPA whose crops are pre-segmented):
    raw image -> segment -> crop -> Cell_State_Analyzer_Tool (embed + cluster) -> AnnData
The agent runs the whole pipeline; the tool's execute() returns cell_state_analyzed.h5ad.

Two metric families, computed by evaluate.py against held-out cell-type labels:
  1. clustering vs cell type      -> ARI, NMI   (obs['leiden_*'] vs GT cell type)
  2. representation label probes   -> kNN acc, linear-probe acc, linear NLL, on the embedding

Two confound guards, BOTH enforced by evaluate.py (the point of the whole benchmark):
  (A) evaluate only cell types present in >= min_sources INDEPENDENT source images;
  (B) every probe uses GroupKFold grouped by image_id (an image is never split across
      train/test), so a probe cannot cheat via per-image technical signatures.

Data layout (discovered, never hard-coded):
  <data-root>/
    images/<image_id>.<ext>        # 2,546 raw multi-source images
    labels.parquet                 # [cell_id | image_id, cell_type]  -- HELD by evaluator only
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    # === Task framing (label-free; agent produces output only) ===
    task_prompt_file: str = "Prompt.txt"
    # The agent must output an AnnData with an embedding and a cluster column; it must
    # NOT receive or use cell-type labels at any point.

    # === Dataset discovery ===
    # Real HCMA = SHAPE-Morph/HCMA on HF: 1,205 images + HCMA_metadata.csv (latin-1),
    # 4 vendors (ATCC/ScienCell/Inhouse/LIVECell). Downloaded to gitignored data/HCMA/.
    data_root: str = "data/HCMA/Human_Cell_Morphology_Atlas_HCMA"
    image_glob: str = "images/*.jpg"
    labels_file: str = "HCMA_metadata.csv"      # read encoding="latin-1"; consumed by evaluate.py ONLY
    image_key: str = "image_id"                 # obs column = source image (CV group)
    cell_id_key: str = "cell_id"                # stable per-cell id used to join GT labels
    cell_type_key: str = "cell_type"            # fine label (127); see label_key for the locked choice

    # === LOCKED benchmark config: A = organ_category + cross-vendor batch correction ===
    # Stage-4 screening (results/stage4_hcma_screen.json) showed fine labels (cell_type/
    # cell_name) are single-vendor -> cross-vendor batch correction is only well-posed at
    # organ_category (14/21 categories span >= 2 vendors, 1,059 images). See README.
    label_key: str = "organ_category"           # the scored label (14 cross-vendor organ systems)
    batch_key: str = "source"                   # vendor -> the batch Harmony removes at stage 4
    min_vendors: int = 2                         # keep only organ categories spanning >= this many vendors
    configA_manifest: str = "results/hcma_configA_manifest.parquet"  # the 1,059-image subset

    # === End-to-end front half (segmentation + cropping) ===
    segment: bool = True                        # HCMA images are NOT pre-segmented
    segmenter_tool: str = "Cell_Segmenter_Tool" # SAM-based; produces per-cell crops
    min_crop_size: int = 15                     # drop sub-15px crops (matches seg client default)

    # === Cell_State_Analyzer_Tool hyperparams (embedding + clustering) ===
    zero_shot: bool = False                     # False = contrastive fine-tune; True = pretrained CLS
    augment_color: bool = True
    freeze_backbone: bool = False
    field_aware: bool = False                   # mask same-image crops from contrastive negatives
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    temperature: float = 0.1
    batch_size: Optional[int] = None            # None = tool auto-selects
    max_epochs: Optional[int] = None            # None = tool auto budget; set for fast screening
    cluster_resolution: float = 0.5             # -> obs key 'leiden_0.5' (the cluster the eval reads)
    n_neighbors: int = 15
    n_pcs: Optional[int] = None
    input_strategy: str = "full_crop"
    seed: int = 0

    # === Fixed evaluation (evaluate.py) ===
    min_sources: int = 2                        # guard (A): cell type needs >= this many images
    knn_k: int = 15                             # kNN probe neighbours
    n_splits: int = 5                           # GroupKFold-by-image folds (guard B)
    cluster_key: str = "leiden_0.5"             # obs cluster column the evaluator reads
    embedding_key: str = "X"                    # 'X' -> use adata.X (the tool's embedding)
    join_key: str = "cell_id"                   # join GT labels onto obs on this column

    # === Fast preset ===
    fast_max_epochs: int = 4
    fast_limit_per_class: int = 30

    # === Output ===
    output_root: str = "solver_cache/HCMA"      # where cell_state_analyzed.h5ad + scores land

    # === Metrics reported ===
    primary_metric: str = "linear_accuracy"     # image-grouped linear-probe accuracy on embedding
    clustering_metrics: tuple = ("ARI", "NMI")
    probe_metrics: tuple = ("knn_accuracy", "linear_accuracy", "linear_nll")

    # === Notes ===
    requires_gpu: bool = True                   # DINOv3 contrastive training is GPU-bound
    # Front-half (segment+crop) makes this the first FULLY end-to-end HCMA benchmark:
    # representation_HPA starts from pre-segmented crops; here the agent starts from raw images.


DEFAULT = BenchmarkConfig()
