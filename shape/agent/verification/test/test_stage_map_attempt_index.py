"""Regression: managed segmentation retries must preserve audit attempt ids.

The retry counter is injected in VLRR vocabulary as ``_vlrr_attempt_index`` and
must survive ``stage_map.map_tool_params`` into the organoid tool's public
``attempt_id`` kwarg. Direct tool tests alone do not cover that managed handoff.
"""

from __future__ import annotations

from shape.agent.verification.stage_map import default_params_for_tool, map_tool_params


def test_segmentation_attempt_index_maps_to_tool_attempt_id():
    out = map_tool_params(
        "Organoid_Segmenter_Tool",
        {"_vlrr_attempt_index": 3, "cellprob_threshold": -1.0},
        {"attempt_id", "cellprob_threshold"},
    )

    assert out == {"attempt_id": 3, "cellprob_threshold": -1.0}


def test_organoid_defaults_do_not_force_advanced_cellpose_kwargs():
    params = default_params_for_tool("Organoid_Segmenter_Tool")

    assert params["diameter"] is None   # the checkpoint's native scale
    assert params["flow_threshold"] == 0.4
    assert params["cellprob_threshold"] == 0.0
    assert params["min_size"] == 15
    assert "niter" not in params
    assert "tile_overlap" not in params
    assert "augment" not in params
    assert "resample" not in params
