"""The managed solve() path's cropper inherits the delivering segmentation's object definition, as the
pipeline's cropper command does (`shape/agent/test/test_executor_crop_min_area.py`):
`integration.build_base_kwargs` passes the segmentation result's `parameters.min_size` for that image as the
cropper's `min_area`, which reaches the agentic crop loop's input record; an organoid mask gets 50; a
segmentation that records no min_size -> no `min_area` key (the tool's default applies)."""

from __future__ import annotations

import numpy as np

from shape.agent.verification.integration import build_base_kwargs
from shape.agent.verification.verifiers.crop import input_record


class _Memory:
    """The previous steps' results, as `memory.get_actions` returns them."""

    def __init__(self, *results: dict):
        self.results = results

    def get_actions(self, llm_safe: bool = False) -> list[dict]:
        return [{"result": result} for result in self.results]


def _mask(tmp_path, name: str) -> str:
    labeled = np.zeros((64, 64), dtype=np.int32)
    labeled[5:15, 5:15] = 1        # 100 px
    labeled[30:36, 30:36] = 2      # 36 px
    path = tmp_path / name
    np.save(path, labeled)
    return str(path)


def _crop_kwargs(result: dict, image_id: str | None = "a", *earlier: dict) -> dict:
    """The cropper's kwargs after `result` (and `earlier`): an image spec dict for a per_image result, the
    single-image path string for a flat one."""
    if image_id and "per_image" in result:
        image = {"path": "img.png", "image_id": image_id, "group": "default"}
    else:
        image = "img.png"
    return build_base_kwargs("Single_Cell_Cropper_Tool", image, _Memory(*earlier, result))


def test_the_segmentations_min_size_reaches_the_croppers_input_record(tmp_path):
    mask = _mask(tmp_path, "cell_mask_a.npy")
    kwargs = _crop_kwargs({"mask_path": mask, "parameters": {"min_size": 40}})           # single image
    assert kwargs["nuclei_mask"] == mask and kwargs["min_area"] == 40
    record = input_record(kwargs)
    assert record["min_area"] == 40 and record["n_objects"] == 2 and record["n_croppable"] == 1
    batch = {"per_image": [{"image_id": "a", "mask_path": mask, "parameters": {"min_size": 30}},
                           {"image_id": "b", "mask_path": mask, "parameters": {"min_size": 18.0}}]}
    assert _crop_kwargs(batch, "b")["min_area"] == 18                                  # the image's own result
    later = {"per_image": [{"image_id": "a", "mask_path": _mask(tmp_path, "cell_mask_a2.npy"),
                            "parameters": {"min_size": 22}}]}
    assert _crop_kwargs(later, "a", batch)["min_area"] == 22                           # the newest segmentation
    crops = {"per_image": [{"image_id": "a", "cell_crops_metadata_path": "meta.json",
                            "parameters": {"min_size": 99}}]}
    assert _crop_kwargs(crops, "a", batch)["min_area"] == 30          # the result that delivered the mask
    assert input_record(_crop_kwargs({"mask_path": mask}))["min_area"] == 15           # the tool's default


def test_no_min_area_when_the_segmentation_records_none(tmp_path):
    mask = _mask(tmp_path, "cell_mask_a.npy")
    for result in ({"mask_path": mask}, {"mask_path": mask, "parameters": {"min_size": None}},
                   {"mask_path": mask, "parameters": {"min_size": True}},
                   {"per_image": [{"image_id": "a", "mask_path": mask}]}):
        kwargs = _crop_kwargs(result)
        assert kwargs["nuclei_mask"] == mask and "min_area" not in kwargs, result
    assert "min_area" not in build_base_kwargs("Single_Cell_Cropper_Tool", "img.png", None)


def test_an_organoid_mask_gets_min_area_50(tmp_path):
    mask = _mask(tmp_path, "organoid_mask_a.npy")
    for result in ({"mask_path": mask, "organoid_mask_path": mask, "parameters": {"min_size": 15}},
                   {"organoid_mask_path": mask},
                   {"per_image": [{"image_id": "a", "mask_path": mask, "parameters": {"min_size": 15}}]}):
        kwargs = _crop_kwargs(result)
        assert kwargs["min_area"] == 50, result
    assert input_record(_crop_kwargs({"mask_path": mask}))["min_area"] == 50
