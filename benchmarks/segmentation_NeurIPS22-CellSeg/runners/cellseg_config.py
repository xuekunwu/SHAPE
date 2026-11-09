"""
Hyperparameters for segmentation_NeurIPS22-CellSeg.

The seven-arm benchmark (run_cellseg_benchmark.py, runners/cellseg_arms.py) reads `arms`, `bands`, `n_per_band`, `seed`,
`llm_model` and `agent_max_steps`. The other fields are the defaults of the standalone SHAPE
runners, run_shape_neurips_cellseg.py (llm_model) and run_concurrent_benchmark.py (the rest).

Data layout:
  <data-root>/NeurIPS22-CellSeg/
    images/          # input images (.png/.tif/.tiff/.jpg/.jpeg/.bmp)
    labels/          # ground-truth instance masks (<stem>_label.tiff)

Set SHAPE_BENCH_DATA_ROOT or pass --data-root.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    # === Seven-arm benchmark (run_cellseg_benchmark.py) ===
    arms: tuple = ("cellpose_sam", "cellpose3", "microsam", "stardist", "gpt", "react", "shape")
    bands: tuple = ("hard", "medium", "easy")   # Cellpose-SAM-alone AP50: < 0.30 <= medium <= 0.60 <
    n_per_band: int = 20                        # images drawn per band (the pilot)
    seed: int = 0                               # the per-band draw
    llm_model: str = "gpt-5.6-luna"             # the gpt, react and shape arms
    agent_max_steps: int = 10                   # shape: solve() max_steps (react: SHAPE's BUDGET rounds)

    # === Standalone SHAPE runners (run_shape_neurips_cellseg.py, run_concurrent_benchmark.py) ===
    question_file: str = "results/Prompt.txt"      # relative to the benchmark directory
    question_fallback: str = "Segment and count every cell in each image."  # used if Prompt.txt missing
    enabled_tools: tuple = ("Cell_Segmenter_Tool",)  # segmenter only; no preprocess/crop/analyze
    max_steps: int = 8                          # max outer planner steps per image
    enable_verification: bool = True            # run the VLRR verify -> retune loop
    enable_llm_verifier: bool = True            # the vision-LLM judge inside VLRR
    verify_refine: bool = False                 # task-044 scale_mismatch evidence; default off

    # === Dataset ===
    data_root: Optional[str] = None             # $SHAPE_BENCH_DATA_ROOT/NeurIPS22-CellSeg
    image_subdir: str = "images"
    label_subdir: str = "labels"
    label_suffix: str = "_label"                # <stem><label_suffix><label_ext>
    label_ext: str = ".tiff"


DEFAULT = BenchmarkConfig()
