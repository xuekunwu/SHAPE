"""Absolute image-quality metrics.

The preprocessing VLRR stage and `Image_Preprocessor_Tool` were removed from the
agentic loop, and with them `PreprocessingVerifier`. What survives here is the
pixel-statistics helper the crop verifier still reads
(`verifiers/crop.py` -> `absolute_quality_metrics`); the module keeps its path so
that import stays stable.
"""

from __future__ import annotations

import numpy as np

from .image_loading import _laplacian_var




def _block_illumination_cv(image: np.ndarray, grid_size: int = 8) -> float:
    """Robust low-frequency illumination variation from block medians."""
    image = np.asarray(image, dtype=np.float64)
    rows = np.array_split(image, min(grid_size, image.shape[0]), axis=0)
    block_medians = []
    for row in rows:
        for block in np.array_split(row, min(grid_size, image.shape[1]), axis=1):
            if block.size:
                block_medians.append(float(np.median(block)))
    values = np.asarray(block_medians, dtype=np.float64)
    return float(values.std() / (abs(values.mean()) + 1e-9))


def absolute_quality_metrics(image: np.ndarray, bit_range: float = 255.0) -> dict[str, float]:
    """Quantify whether one image is directly suitable for segmentation."""
    image = np.asarray(image, dtype=np.float64)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return {
            "finite_fraction": 0.0,
            "median_brightness": 0.0,
            "dynamic_range": 0.0,
            "illumination_CV": float("inf"),
            "saturation_fraction": 0.0,
            "zero_fraction": 1.0,
            "sharpness": 0.0,
        }
    p1, median, p99 = np.percentile(finite, [1, 50, 99])
    return {
        "finite_fraction": float(finite.size / image.size),
        "median_brightness": float(median / bit_range),
        "dynamic_range": float((p99 - p1) / bit_range),
        "illumination_CV": _block_illumination_cv(image),
        "saturation_fraction": float((finite >= 0.98 * bit_range).mean()),
        "zero_fraction": float((finite <= 0.02 * bit_range).mean()),
        "sharpness": _laplacian_var(image),
    }
