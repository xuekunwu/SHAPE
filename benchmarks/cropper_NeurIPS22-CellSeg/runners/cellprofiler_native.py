#!/usr/bin/env python3
"""Run CellProfiler's native SaveCroppedObjects module for one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


IMAGE_NAME = "InputImage"
OBJECTS_NAME = "Cells"


def _read_source_image(path: Path) -> np.ndarray:
    """Read microscopy source images without treating every format as TIFF. `.npy` is the source image
    already decoded by the benchmark's environment (the CellProfiler image's tifffile has no `imagecodecs`,
    so it cannot decode e.g. LZW-compressed TIFFs)."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(str(path), allow_pickle=False)
    if suffix in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(str(path)))
    if suffix in {".png", ".bmp", ".jpg", ".jpeg"}:
        with Image.open(path) as image:
            return np.asarray(image).copy()
    raise ValueError(
        f"unsupported source image format {suffix!r} for {path}; "
        "expected TIFF, PNG, BMP, JPEG, or NPY"
    )


def _orient_source_image(image: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    image = np.squeeze(np.asarray(image))
    if image.ndim == 2:
        if image.shape != mask_shape:
            raise ValueError(f"image shape {image.shape} does not match mask shape {mask_shape}")
    elif image.ndim == 3:
        if image.shape[:2] == mask_shape:
            pass
        elif image.shape[1:] == mask_shape and image.shape[0] <= 4:
            image = np.moveaxis(image, 0, -1)
        else:
            raise ValueError(f"image shape {image.shape} does not match mask shape {mask_shape}")
    else:
        raise ValueError(f"expected a 2D or multichannel 2D image, got shape {image.shape}")
    return image


def _prepare_image(image: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    image = _orient_source_image(image, mask_shape)

    if np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
        image = image.astype(np.float32) / scale
    else:
        image = image.astype(np.float32)
        if not np.isfinite(image).all():
            raise ValueError("image contains non-finite values")
        low = float(image.min()) if image.size else 0.0
        high = float(image.max()) if image.size else 0.0
        if low < 0.0 or high > 1.0:
            image = (image - low) / (high - low) if high > low else np.zeros_like(image)
    return image


def _filter_labels(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, list[int]]:
    labels, counts = np.unique(mask.astype(np.int64), return_counts=True)
    keep = [int(label) for label, count in zip(labels, counts)
            if int(label) > 0 and int(count) >= int(min_area)]
    filtered = np.where(np.isin(mask, keep), mask, 0).astype(np.int32)
    return filtered, keep


def _bbox(mask: np.ndarray, label: int) -> list[int]:
    rows, cols = np.where(mask == label)
    if rows.size == 0:
        raise ValueError(f"label {label} is absent from the filtered mask")
    return [int(rows.min()), int(cols.min()), int(rows.max() + 1), int(cols.max() + 1)]


def run(image_path: Path, mask_path: Path, out_dir: Path, min_area: int) -> dict:
    import cellprofiler
    import cellprofiler_core
    import cellprofiler_core.image
    import cellprofiler_core.measurement
    import cellprofiler_core.object
    import cellprofiler_core.pipeline
    import cellprofiler_core.preferences
    import cellprofiler_core.workspace
    from cellprofiler.modules.savecroppedobjects import O_TIFF_16, SAVE_PER_OBJECT, SaveCroppedObjects

    mask = np.squeeze(tifffile.imread(str(mask_path))).astype(np.int64)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D instance mask, got shape {mask.shape}")
    image = _prepare_image(_read_source_image(image_path), mask.shape)
    labels, kept_labels = _filter_labels(mask, min_area)

    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    cellprofiler_core.preferences.set_headless()

    cp_image = cellprofiler_core.image.Image(image=image, dimensions=2)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(IMAGE_NAME, cp_image)

    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    objects.parent_image = cp_image
    object_set = cellprofiler_core.object.ObjectSet()
    object_set.add_objects(objects, OBJECTS_NAME)

    module = SaveCroppedObjects()
    module.objects_name.value = OBJECTS_NAME
    module.export_option.value = SAVE_PER_OBJECT
    module.use_filename.value = False
    module.nested_save.value = False
    module.file_format.value = O_TIFF_16
    module.image_name.value = IMAGE_NAME
    module.directory.value = "{}|{}".format(
        cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME, crop_dir.resolve())

    workspace = cellprofiler_core.workspace.Workspace(
        cellprofiler_core.pipeline.Pipeline(),
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    module.run(workspace)

    crops = []
    for label in kept_labels:
        bbox = _bbox(labels, label)
        crop_path = crop_dir / f"{OBJECTS_NAME}_{label}.tiff"
        if not crop_path.is_file():
            raise RuntimeError(f"SaveCroppedObjects did not create {crop_path}")
        crop_shape = tifffile.imread(str(crop_path)).shape
        expected_shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        if tuple(crop_shape[:2]) != expected_shape:
            raise RuntimeError(
                f"unexpected crop shape for label {label}: {crop_shape}, expected {expected_shape}")
        crops.append({
            "cell_label": label,
            "bbox": bbox,
            "source": "cellprofiler.SaveCroppedObjects",
            "crop_image_path": str(crop_path),
        })

    return {
        "schema": "shape.cropper.cellprofiler.v1",
        "backend": "CellProfiler",
        "module": "SaveCroppedObjects",
        "cellprofiler_version": getattr(cellprofiler, "__version__", "unknown"),
        "cellprofiler_core_version": getattr(cellprofiler_core, "__version__", "unknown"),
        "min_area": int(min_area),
        "crops": crops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--mask-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-area", type=int, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()

    payload = run(args.image_path, args.mask_path, args.out_dir, args.min_area)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "CELLPROFILER_NATIVE_OK "
        f"version={payload['cellprofiler_version']} crops={len(payload['crops'])}"
    )


if __name__ == "__main__":
    main()
