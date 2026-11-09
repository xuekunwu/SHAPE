"""The Codex arm's crop calls take SHAPE's cell rules (user decision 2026-10-12): the same rule format, validated as
SHAPE validates it, and compiled per image by SHAPE's own compiler on the image's mask and, with a rules parent, on that
call's per-cell readouts."""
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks.full_CPJUMP1.harness import codex_arm as agent
from benchmarks.full_CPJUMP1.harness import pipeline
from benchmarks.full_CPJUMP1.harness.data import read_json, write_json

TIGHTEN = [{"key": "margin_mode", "value": "mask_bbox"}, {"key": "auto_tighten", "value": True}]


def test_crop_parameters_take_shapes_cell_rules():
    out = agent.parameters("crop", {"tight_buffer_px": 0,
                                    "cell_rules": [{"when": [{"readout": "area", "op": ">", "value": 100}],
                                                    "apply": TIGHTEN}]})
    assert out["tight_buffer_px"] == 0 and out["cell_rules"][0]["when"][0] == {"readout": "area", "op": ">",
                                                                                "value": "100"}
    call_readout = [{"when": [{"readout": "neighbor_contamination", "op": ">", "value": 0.2}], "apply": TIGHTEN}]
    with pytest.raises(ValueError, match="need rules_parent"):
        agent.parameters("crop", {"cell_rules": call_readout})
    assert agent.parameters("crop", {"cell_rules": call_readout, "rules_parent": "c2"})["rules_parent"] == "c2"
    with pytest.raises(ValueError, match="needs cell_rules"):
        agent.parameters("crop", {"rules_parent": "c2"})
    with pytest.raises(ValueError, match="cell_rules"):                    # the wrapper Codex once wrote
        agent.parameters("crop", {"cell_rules": {"rules": call_readout}})
    with pytest.raises(ValueError, match="unknown parameter"):             # crop only
        agent.parameters("segment", {"cell_rules": call_readout})
    text = agent._cell_rules_text()
    assert "rules_parent" in text and "neighbor_contamination" in text and "nn_gap_px" in text


@pytest.fixture
def image_and_mask(tmp_path):
    image = np.full((5, 96, 96), 100, np.uint16)
    labels = np.zeros((96, 96), np.uint32)
    labels[10:40, 10:40] = 1                                                # a big cell
    labels[60:70, 60:70] = 2                                                # a small one
    image[:, labels > 0] = 3000
    tifffile.imwrite(tmp_path / "img_0000.tif", image, photometric="minisblack")
    tifffile.imwrite(tmp_path / "cell_mask_img_0000.tif", labels)
    return tmp_path / "img_0000.tif", {"mask": str(tmp_path / "cell_mask_img_0000.tif"), "min_area": 20}


def test_rules_compile_by_shapes_compiler_on_the_input_measurements(image_and_mask):
    image, mask = image_and_mask
    rules = agent.parameters("crop", {"cell_rules": [{"when": [{"readout": "area", "op": ">", "value": 400}],
                                                      "apply": TIGHTEN}]})["cell_rules"]
    overrides, compiled = pipeline.compile_cell_rules(image, mask, "img_0000", {"margin_mode": "fraction"}, rules,
                                                      None)
    assert list(overrides.values()) == [{"margin_mode": "mask_bbox", "auto_tighten": True}]   # the big cell only
    assert next(iter(overrides)).endswith("img_0000_cell_0000") and "(1 cells)" in compiled[0]
    nothing = agent.parameters("crop", {"cell_rules": [{"when": [{"readout": "area", "op": ">", "value": 10 ** 6}],
                                                        "apply": TIGHTEN}]})["cell_rules"]
    with pytest.raises(ValueError, match="matches no cell"):
        pipeline.compile_cell_rules(image, mask, "img_0000", {}, nothing, None)


def test_rules_read_a_finished_calls_per_cell_readouts(image_and_mask, tmp_path):
    from benchmarks.full_HCMA.harness.shape_arm import CROP_GROUP
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool

    image, mask = image_and_mask
    parent = tmp_path / "c1"
    result = Single_Cell_Cropper_Tool().execute(
        original_image=str(image), nuclei_mask=mask["mask"], source_image_id="img_0000", group=CROP_GROUP,
        min_area=mask["min_area"], output_format="tif", query_cache_dir=str(parent / "tool"))
    write_json(parent / "result.json", pipeline._tool_dict(result))
    rules = agent.parameters("crop", {"rules_parent": "c1", "cell_rules": [
        {"when": [{"readout": "target_retention", "op": ">=", "value": 0.9}], "apply": TIGHTEN}]})["cell_rules"]
    overrides, compiled = pipeline.compile_cell_rules(image, mask, "img_0000", {}, rules, str(parent))
    metadata = Path(read_json(parent / "result.json")["cell_crops_metadata_path"])
    cells = [c["cell_id"] for c in read_json(metadata)["cell_metadata"]]
    assert sorted(overrides) == sorted(cells) and "(2 cells)" in compiled[0]
