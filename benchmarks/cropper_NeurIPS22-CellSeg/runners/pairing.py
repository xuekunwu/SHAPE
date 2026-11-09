"""Image / GT-label pairing and the cropper defaults the drivers share.

`<data-dir>/images/<name>.<ext>` + `<data-dir>/labels/<name>_label.tiff` is the
NeurIPS22-CellSeg layout; `run_benchmark.py`, `run_shape_crop.py`, `run_cohort_crop.py`
and `run_calib_sweep.py` all read it through `_find_image_mask_pairs`. The pairing's
test is `test_cropper_arms.py::test_find_image_mask_pairs_matches_layout`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))  # repo root, for `shape.*`

from shape.tools.single_cell_cropper.tool import SEGMENTER_DEFAULT_MIN_SIZE  # noqa: E402

_IMAGE_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

# NeurIPS22-CellSeg is cell-level segmentation, so these match `tool.py::execute`'s own
# "cell_mask" auto-detect defaults: min_area = the cell segmenter's default min_size
# (`SEGMENTER_DEFAULT_MIN_SIZE`), margin = 1.
DEFAULT_MIN_AREA = SEGMENTER_DEFAULT_MIN_SIZE
DEFAULT_MARGIN = 1


def _find_image_mask_pairs(data_dir: Path, label_suffix: str = "_label",
                            label_ext: str = ".tiff") -> list[tuple[Path, Path]]:
    """<data-dir>/images/<name>.ext + <data-dir>/labels/<name>_label.tiff -> sorted
    (image_path, label_path) pairs, skipping any label with no matching image."""
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    images_by_stem = {p.stem: p for p in images_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in _IMAGE_EXTS}
    pairs = []
    for label_path in sorted(labels_dir.glob(f"*{label_suffix}{label_ext}")):
        stem = label_path.name[: -len(f"{label_suffix}{label_ext}")]
        image_path = images_by_stem.get(stem)
        if image_path is not None:
            pairs.append((image_path, label_path))
    return pairs
