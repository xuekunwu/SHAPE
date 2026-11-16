"""
Canonical hyperparameter configuration for cropper_NeurIPS22-CellSeg.

Benchmarks single-cell crop generation on NeurIPS22-CellSeg images. The formal
runner is `run_benchmark.py` (top level); the legacy per-probe runners in `runners/`
read the defaults below.

Five formal arms:
  maskcrop      -- MaskCrop domain tool
  cellprofiler  -- native CellProfiler 4.2.8 SaveCroppedObjects
  gpt           -- gpt-5.6-luna alone
  react         -- gpt-5.6-luna + MaskCrop ReAct loop
  shape         -- SHAPE with gpt-5.6-luna + MaskCrop

Data: NeurIPS22-CellSeg root with images/ and labels/ subfolders.
GPU: not required.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for `shape.*`
from shape.tools.single_cell_cropper.tool import SEGMENTER_DEFAULT_MIN_SIZE  # noqa: E402


@dataclass
class BenchmarkConfig:
    """The fields the runners read (run_shape_crop.py)."""
    # === Solver path (run_shape_crop.py --solver) ===
    solver_question: str = (
        "Crop every individual cell from the segmentation mask, "
        "optimising crop geometry per cell."
    )
    solver_enabled_tools: tuple = ("Single_Cell_Cropper_Tool",)  # planned tool order under solve()
    solver_max_steps: int = 10                 # max planner steps under solve()
    llm_model: str = "gpt-5.6-luna"            # formal gpt/react/shape model

    # === Arm geometry (FAIR START: both arms use the same starting geometry) ===
    # task-038 redesign: SHAPE arm starts at the SAME naive margin as MaskCrop,
    # so the delta isolates exactly what the agent adds on top.
    default_min_area: int = SEGMENTER_DEFAULT_MIN_SIZE  # minimum cell area in pixels to crop: the tool's default,
                                                # the cell segmenter's min_size (2026-10-01; was 50)
    default_margin: int = 1                     # absolute margin in pixels (arm A raw fallback)
    default_maskcrop_margin_frac: float = 0.20  # shared naive starting margin (20% of equiv diam)

    # === Dataset ===
    label_suffix: str = "_label"               # GT mask filename: <stem><label_suffix><label_ext>
    label_ext: str = ".tiff"                   # GT instance-mask extension


DEFAULT = BenchmarkConfig()

# Non-pixel run artifacts (run_benchmark.py's lean/noimg export filters): every other suffix is treated
# as pixel data (crops, masks, overlays) and excluded from a lean copy.
PIXEL_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".zip", ".npy", ".npz"}
