"""Lossless-input verification for `Organoid_Segmenter_Tool.execute()` (this task):
the auto-applied brightness-adjust gate (`check_brightness_only`/`adjust_brightness_only`)
is removed, and the segmentation input path no longer forces a uint8 downcast
(`img_data.to_uint8(0)`) before Cellpose ever sees it. Co-located with the module
under test (AGENTS.md sec 6 P5). No real Cellpose weights are loaded --
`models.CellposeModel` is patched at its import site in `tool.py`, so `model_path`
only needs to be a non-empty string (never actually read as a file).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import tifffile
from PIL import Image

from shape.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool


class _RecordingStubModel:
    """Fake `CellposeModel` that records every array `.eval()` is called with."""

    def __init__(self, n_cells=2):
        self.calls = []
        self.eval_kwargs = []
        self._n_cells = n_cells

    def eval(self, imgs, **kwargs):
        arr = imgs[0]
        self.calls.append(arr)
        self.eval_kwargs.append(dict(kwargs))
        mask = np.zeros(arr.shape[:2], dtype=np.int32)
        if self._n_cells >= 1:
            mask[4:10, 4:10] = 1
        if self._n_cells > 1:
            mask[20:26, 20:26] = 2
        return [mask], None, None


class _SequencedStubModel(_RecordingStubModel):
    """Fake model that returns a planned object count for each eval call."""

    def __init__(self, n_cells_sequence):
        super().__init__(n_cells=0)
        self._sequence = list(n_cells_sequence)

    def eval(self, imgs, **kwargs):
        self._n_cells = self._sequence.pop(0) if self._sequence else 0
        return super().eval(imgs, **kwargs)


def _write_single_channel_uint16(path, size=32, seed=17):
    """A single-channel uint16 TIFF with a real, non-degenerate value range."""
    rng = np.random.default_rng(seed)
    ch = np.clip(15000 + rng.normal(0, 2000, (size, size)), 0, 65535).astype(np.uint16)
    tifffile.imwrite(str(path), ch, photometric="minisblack")
    return ch


def test_single_channel_uint16_values_preserved(tmp_path):
    """`img_data.to_segmentation_input(0)` must reach Cellpose unchanged (cast to
    float32, never renormalized/clipped) -- the pre-fix code fed
    `img_data.to_uint8(0)` here unconditionally, which would clip these ~15000
    values down to the uint8 [0, 255] range."""
    img_path = tmp_path / "organoid_uint16.tiff"
    source = _write_single_channel_uint16(img_path)

    stub = _RecordingStubModel(n_cells=2)
    with patch("shape.tools.organoid_segmenter.tool.models.CellposeModel", return_value=stub):
        tool = Organoid_Segmenter_Tool(model_path="dummy_model_path")
        result = tool.execute(image=str(img_path), query_cache_dir=str(tmp_path))

    assert "error" not in result, result
    assert result["status"] == "success"
    assert result["overlay_qc"]["passed"] is True
    assert len(stub.calls) >= 1
    received = stub.calls[0]
    assert received.ndim == 2
    assert received.shape == source.shape
    np.testing.assert_allclose(received, source.astype(np.float32))
    assert received.max() > 255, "value range was clipped/downcast toward uint8"


def test_dark_image_no_longer_auto_brightness_adjusted(tmp_path):
    """The removed `check_brightness_only`/`adjust_brightness_only` gate used to
    intercept a dark image (mean well below its 50 threshold) and rescale it with
    `cv2.convertScaleAbs` before Cellpose ever saw it. With that gate gone, this
    image must reach `model.eval` BYTE-IDENTICAL to the raw loaded array."""
    rng = np.random.default_rng(19)
    size = 32
    dark = np.clip(20 + rng.normal(0, 3, (size, size)), 0, 255).astype(np.uint8)
    assert dark.mean() < 50, "fixture must be darker than the old min_brightness=50 gate"
    img_path = tmp_path / "organoid_dark.png"
    Image.fromarray(dark, mode="L").save(img_path)

    stub = _RecordingStubModel(n_cells=2)
    with patch("shape.tools.organoid_segmenter.tool.models.CellposeModel", return_value=stub):
        tool = Organoid_Segmenter_Tool(model_path="dummy_model_path")
        result = tool.execute(image=str(img_path), query_cache_dir=str(tmp_path))

    assert "error" not in result, result
    assert result["status"] == "success"
    assert result["overlay_qc"]["passed"] is True
    assert len(stub.calls) >= 1
    np.testing.assert_allclose(stub.calls[0], dark.astype(np.float32))


def test_default_call_preserves_cellpose_defaults_for_advanced_kwargs(tmp_path):
    img_path = tmp_path / "organoid_uint16.tiff"
    _write_single_channel_uint16(img_path)

    stub = _RecordingStubModel(n_cells=2)
    with patch("shape.tools.organoid_segmenter.tool.models.CellposeModel", return_value=stub):
        tool = Organoid_Segmenter_Tool(model_path="dummy_model_path")
        result = tool.execute(image=str(img_path), query_cache_dir=str(tmp_path))

    assert "error" not in result, result
    kwargs = stub.eval_kwargs[-1]
    assert "niter" not in kwargs
    assert "tile_overlap" not in kwargs
    assert "augment" not in kwargs
    assert "resample" not in kwargs
    assert result["parameters"]["niter"] is None
    assert result["parameters"]["tile_overlap"] is None
    assert result["parameters"]["augment"] is None
    assert result["parameters"]["resample"] is None


def test_vlrr_cellpose_kwargs_and_attempt_artifacts_are_preserved(tmp_path):
    img_path = tmp_path / "organoid_uint16.tiff"
    _write_single_channel_uint16(img_path)

    stub = _RecordingStubModel(n_cells=2)
    with patch("shape.tools.organoid_segmenter.tool.models.CellposeModel", return_value=stub):
        tool = Organoid_Segmenter_Tool(model_path="dummy_model_path")
        result = tool.execute(
            image=str(img_path),
            query_cache_dir=str(tmp_path),
            niter=500,
            tile_overlap=0.25,
            augment=True,
            resample=False,
            _vlrr_attempt_index=3,
        )

    assert "error" not in result, result
    assert stub.eval_kwargs[-1]["niter"] == 500
    assert stub.eval_kwargs[-1]["tile_overlap"] == 0.25
    assert stub.eval_kwargs[-1]["augment"] is True
    assert stub.eval_kwargs[-1]["resample"] is False
    assert result["parameters"]["attempt_id"] == 3
    assert "_attempt_03" in result["overlay_path"]
    assert "_attempt_03" in result["mask_path"]
    assert result["latest_overlay_path"].endswith("organoid_overlay_organoid_uint16.png")
    assert result["latest_mask_path"].endswith("organoid_mask_organoid_uint16.npy")


def test_adaptive_rescue_tries_v4_sequence_after_two_zero_detections(tmp_path):
    img_path = tmp_path / "organoid_uint16.tiff"
    _write_single_channel_uint16(img_path)

    def fake_visual_qc(mask, overlay):
        count = len(np.unique(mask)) - 1
        return {
            "passed": count > 0,
            "overlay_is_black": False,
            "foreground_coverage": 0.1 if count > 0 else 0.0,
            "coverage_implausible": False,
        }

    stub = _SequencedStubModel([0, 0, 2])
    with patch("shape.tools.organoid_segmenter.tool.models.CellposeModel", return_value=stub), \
            patch("shape.tools.organoid_segmenter.tool.evaluate_segmentation_visual",
                  side_effect=fake_visual_qc):
        tool = Organoid_Segmenter_Tool(model_path="dummy_model_path")
        result = tool.execute(image=str(img_path), query_cache_dir=str(tmp_path))

    assert "error" not in result, result
    assert result["status"] == "success"
    assert len(stub.eval_kwargs) == 3
    assert stub.eval_kwargs[0]["diameter"] is None
    assert stub.eval_kwargs[0]["cellprob_threshold"] == 0
    assert stub.eval_kwargs[1]["diameter"] is None
    assert stub.eval_kwargs[1]["cellprob_threshold"] == -2
    assert stub.eval_kwargs[2]["diameter"] is None
    assert stub.eval_kwargs[2]["cellprob_threshold"] == -4
    assert result["adaptive_rescue"]["attempted"] is True
    assert len(result["adaptive_rescue"]["attempts"]) == 3
    assert "_attempt_03" in result["mask_path"]


def test_rescue_plan_stays_near_native_scale():
    # Cellpose resizes by 30 / diameter: no probe may shrink the image below half size.
    from shape.tools.organoid_segmenter.tool import _organoid_rescue_plan

    plan = _organoid_rescue_plan({"diameter": None, "cellprob_threshold": 0})
    assert [p["diameter"] for p in plan] == [None, None, 15, None, 60, None, None, None, None]


def test_adaptive_rescue_is_disabled_inside_vlrr_attempt(tmp_path):
    img_path = tmp_path / "organoid_uint16.tiff"
    _write_single_channel_uint16(img_path)

    stub = _SequencedStubModel([0, 2])
    with patch("shape.tools.organoid_segmenter.tool.models.CellposeModel", return_value=stub), \
            patch("shape.tools.organoid_segmenter.tool.evaluate_segmentation_visual",
                  return_value={
                      "passed": False,
                      "overlay_is_black": False,
                      "foreground_coverage": 0.0,
                      "coverage_implausible": False,
                  }):
        tool = Organoid_Segmenter_Tool(model_path="dummy_model_path")
        result = tool.execute(
            image=str(img_path),
            query_cache_dir=str(tmp_path),
            _vlrr_attempt_index=1,
        )

    assert result["status"] == "failed"
    assert len(stub.eval_kwargs) == 1
    assert "adaptive_rescue" not in result


def test_managed_agentic_call_preserves_native_pixels_and_avoids_nested_rescue(tmp_path):
    """Real tool + managed dispatch + verifier; only the learned model is stubbed."""
    from types import SimpleNamespace

    from shape.agent.verification.integration import verify_managed_tool
    from shape.agent.verification.stage_map import invoke_managed_tool

    img_path = tmp_path / "whole_organoid.tiff"
    source = _write_single_channel_uint16(img_path, size=128)

    class WholeOrganoidModel(_RecordingStubModel):
        def eval(self, imgs, **kwargs):
            self.calls.append(imgs[0])
            self.eval_kwargs.append(dict(kwargs))
            mask = np.zeros(imgs[0].shape[:2], dtype=np.int32)
            mask[20:80, 20:80] = 1
            return [mask], None, None

    model = WholeOrganoidModel()
    with patch("shape.tools.organoid_segmenter.tool.models.CellposeModel", return_value=model):
        tool = Organoid_Segmenter_Tool(model_path="dummy_model_path")
        executor = SimpleNamespace(
            enable_llm_verifier=False,
            run_managed_tool=lambda name, bk, params: invoke_managed_tool(tool, name, bk, params),
        )
        resolutions, result, status = verify_managed_tool(
            executor, "Organoid_Segmenter_Tool",
            {"image": str(img_path), "query_cache_dir": str(tmp_path)},
        )

    assert status == "satisfied" and resolutions[0].reason == "no_tuning_needed"
    assert result["organoid_count"] == 1 and result["parameters"]["diameter"] is None
    assert result["parameters"]["attempt_id"] == 1 and "adaptive_rescue" not in result
    assert result["deliverables"][0] == result["mask_tif_path"]   # the mask, not only its overlay
    assert len(model.calls) == 1 and model.eval_kwargs[0]["diameter"] is None
    np.testing.assert_array_equal(model.calls[0], source.astype(np.float32))
    assert "niter" not in model.eval_kwargs[0] and "augment" not in model.eval_kwargs[0]
