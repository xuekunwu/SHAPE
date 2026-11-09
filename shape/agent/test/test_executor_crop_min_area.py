"""The pipeline's cropper command inherits the delivering segmentation's object definition (2026-10-01):
`Executor.generate_tool_command` passes the segmentation result's `parameters.min_size` as the cropper's
`min_area`, and omits `min_area` when the result records none (the tool's default, the cell segmenter's
min_size, applies). Organoid masks are out of that change: they keep the command's previous `min_area=50`. The
margin stays 25."""

from __future__ import annotations

from shape.agent.executor import Executor


class _Memory:
    """The executor's view of the previous step: its result, as `memory.get_actions` returns it."""

    def __init__(self, result: dict):
        self.result = result

    def get_actions(self, llm_safe: bool = False) -> list[dict]:
        return [{"result": self.result}]


def _cropper_command(tmp_path, result: dict, image_id: str | None = None) -> str:
    executor = Executor("unused-model", query_cache_dir=str(tmp_path))
    return executor.generate_tool_command(
        "crop the cells", str(tmp_path / "img.tif"), "", "crop every cell", "Single_Cell_Cropper_Tool", {},
        memory=_Memory(result), image_id=image_id).command


def test_the_cropper_command_carries_the_segmentations_min_size(tmp_path):
    mask = str(tmp_path / "img_cell_mask.npy")
    command = _cropper_command(tmp_path, {"mask_path": mask, "parameters": {"min_size": 22}})
    assert "min_area=22, query_cache_dir" in command and "min_area=50" not in command
    assert "margin" not in command                                          # the tool's own margin rule
    batch = {"per_image": [{"image_id": "a", "mask_path": mask, "parameters": {"min_size": 30}},
                           {"image_id": "b", "mask_path": mask, "parameters": {"min_size": 18.0}}]}
    assert "min_area=18, query_cache_dir" in _cropper_command(tmp_path, batch, image_id="b")   # the image's own result


def test_the_cropper_command_omits_min_area_when_the_segmentation_records_none(tmp_path):
    mask = str(tmp_path / "img_cell_mask.npy")
    for result in ({"mask_path": mask}, {"mask_path": mask, "parameters": {"min_size": None}},
                   {"per_image": [{"image_id": "a", "mask_path": mask}]}):
        command = _cropper_command(tmp_path, result, image_id="a")
        assert "min_area" not in command and "margin" not in command, result


def test_an_organoid_mask_keeps_the_previous_min_area_50(tmp_path):
    """Organoid masks keep exactly the command's previous behaviour: `min_area=50`, even when the segmentation
    records a min_size (15, the organoid segmenter's default), single image and per image."""
    mask = str(tmp_path / "img_organoid_mask.npy")
    for result in ({"mask_path": mask, "parameters": {"min_size": 15}}, {"mask_path": mask},
                   {"per_image": [{"image_id": "a", "mask_path": mask, "parameters": {"min_size": 15}}]}):
        command = _cropper_command(tmp_path, result, image_id="a")
        assert "min_area=50, query_cache_dir" in command and "min_area=15" not in command, result
