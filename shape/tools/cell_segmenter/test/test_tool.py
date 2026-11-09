"""Task 073 R2 fix (QC finding 2, FIX round 2): `Cell_Segmenter_Tool.execute()`'s
multi-channel recovery loop (`tool.py` ~445-475). Co-located with the module under test
(committed -- AGENTS.md sec 6 P5). No real Cellpose weights are loaded -- `models.CellposeModel`
is patched at its import site in `tool.py` with a lightweight stub whose `.eval()` is
sufficient to exercise the SELECTION logic this fix touches.

Also covers the Codex QC blocker on candidate 0b3c0ac: `_segment_channel` must refuse
(not silently feed) a non-finite (NaN/Inf) array to Cellpose; and, for the agentic VRR, the
call's reported seconds, concurrent calls of one image (distinct attempts, PIL figures in worker
threads) and the CARD lever Defaults (the tool at the planner defaults); and the large-image speed-up
(the per-mask B-R in one pass, the overlay as a colour lookup table, the colour-filtered mask saved
relabelled 1-N): the same outputs and readouts as before; and the mask before the colour filter, recorded
only when the filter removed masks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile
from PIL import Image
from skimage.segmentation import relabel_sequential

from shape.tools.cell_segmenter import color_class_filter as ccf
from shape.tools.cell_segmenter import tool as tool_module
from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool


class _CallLoggingStubModel:
    """Fake `CellposeModel`: fails visual QC on the PRIMARY (H, W, C) call (empty mask),
    then returns real cells on the SECOND `eval` call overall -- which, after the fix, is
    the first single-channel retry (channel 0, since it is the only channel with variance
    here). Distinguishes calls by ORDER, not pixel content, so it does not presuppose which
    channel the fix hands it -- that is exactly what the test observes."""

    def __init__(self):
        self.calls = []  # recorded (H, W[, C]) shapes, in call order

    def eval(self, imgs, **kwargs):
        img = imgs[0]
        self.calls.append(img.shape)
        mask = np.zeros(img.shape[:2], dtype=np.int32)
        if img.ndim == 2 and len(self.calls) == 2:
            mask[4:10, 4:10] = 1
            mask[20:26, 20:26] = 2
            mask[40:46, 40:46] = 3
        return [mask], None, None


def _write_signal_only_in_channel_zero(path, size=64, seed=5):
    """A 3-channel uint8 image whose only variance lives in channel 0 -- channels 1/2 are
    CONSTANT (the `ImageData.informative_channel_idx` degenerate case: 'a fluorescence image
    stored as RGB whose signal sits in [one channel], leaving the [others] all zeros').
    This fixture isolates the recovery loop: only channel 0 carries useful variance,
    while channels 1/2 are constant."""
    rng = np.random.default_rng(seed)
    ch0 = np.clip(128 + rng.normal(0, 15, (size, size)), 0, 255).astype(np.uint8)
    ch1 = np.full((size, size), 128, dtype=np.uint8)
    ch2 = np.full((size, size), 128, dtype=np.uint8)
    Image.fromarray(np.stack([ch0, ch1, ch2], axis=-1), mode="RGB").save(path)


def test_multichannel_recovery_gives_channel_zero_its_own_retry(tmp_path):
    """QC finding 2 (BLOCKER, FIX round 2): round 1 made the PRIMARY run the full (H,W,C)
    array (not any one channel), but the recovery loop's own `if c == primary_idx: continue`
    still assumed the primary run WAS `informative_channel_idx(0)` (here, channel 0) and
    skipped it -- so channel 0 was NEVER individually segmented. With channels 1/2 constant
    (zero variance, skipped by the loop's own empty-channel guard), the pre-fix loop tries
    NOTHING: the failed all-channel result stands, and `segmented_channel` is still (falsely)
    reported as 0 -- claiming a single-channel result that was never computed. Reproduces the
    independent QC review's exact repro."""
    img_path = tmp_path / "signal_ch0_only.png"
    _write_signal_only_in_channel_zero(img_path)

    stub = _CallLoggingStubModel()
    with patch("shape.tools.cell_segmenter.tool.models.CellposeModel", return_value=stub):
        tool = Cell_Segmenter_Tool()
        result = tool.execute(image=str(img_path), query_cache_dir=str(tmp_path))

    # Channel 0 actually reached `model.eval` alone (a (H, W) call, not just the primary
    # (H, W, 3) call) -- the coverage hole the fix closes.
    single_channel_calls = [c for c in stub.calls if len(c) == 2]
    assert len(single_channel_calls) >= 1, (
        f"channel 0 was never individually retried; calls={stub.calls}")

    # The recovery genuinely found channel 0's cells -- not the failed all-channel result.
    assert result["status"] == "success"
    assert result["cell_count"] == 3
    # True provenance: a single channel actually won, and it is reported as such.
    assert result["segmented_channel"] == 0


class _RecordingStubModel:
    """Fake `CellposeModel` that records every array `.eval()` is called with (not just
    its shape) -- used to prove Cellpose never receives a non-finite array (the guard
    test) and, separately, that an ordinary finite array reaches it unchanged (the
    guard-does-not-trigger control)."""

    def __init__(self, n_cells=2):
        self.calls = []
        self._n_cells = n_cells

    def eval(self, imgs, **kwargs):
        arr = imgs[0]
        self.calls.append(arr)
        mask = np.zeros(arr.shape[:2], dtype=np.int32)
        mask[4:10, 4:10] = 1
        if self._n_cells > 1:
            mask[20:26, 20:26] = 2
        return [mask], None, None


def _write_multichannel_float_tiff(path, size=32, non_finite_value=None, seed=3):
    """A 3-channel float32, channels-last TIFF where `img_data.data.astype(np.float32)`
    reaches `_segment_channel` directly. `non_finite_value`, when given, is written
    into ONE pixel of channel 0.
    Returns the source array (for asserting the fixture's own precondition)."""
    rng = np.random.default_rng(seed)
    ch0 = np.clip(120 + rng.normal(0, 5, (size, size)), 0, 255).astype(np.float32)
    ch1 = np.clip(120 + rng.normal(0, 5, (size, size)), 0, 255).astype(np.float32)
    ch2 = np.clip(120 + rng.normal(0, 5, (size, size)), 0, 255).astype(np.float32)
    img = np.stack([ch0, ch1, ch2], axis=-1)
    if non_finite_value is not None:
        img[0, 0, 0] = non_finite_value
    tifffile.imwrite(str(path), img, photometric="minisblack", planarconfig="contig")
    return img


@pytest.mark.parametrize("non_finite_value,label", [
    (np.nan, "nan"),
    (np.inf, "posinf"),
    (-np.inf, "neginf"),
])
def test_execute_refuses_non_finite_pixel_before_reaching_cellpose(tmp_path, non_finite_value, label):
    """Codex QC BLOCKER (candidate 0b3c0ac), human-authorized fix = loud refuse, not
    silent conversion. Pre-fix, `execute()` fed the multi-channel array straight to
    `self.model.eval` via `img = img_data.data.astype(np.float32)` with no finite
    check. Cellpose 4.0.4's own `normalize_img` silently COLLAPSES an entire channel
    to a single value when it contains even one non-finite pixel (verified directly:
    1 NaN in a 3-channel 16x16 float32 array -> that channel's normalized output has
    1 unique value while an untouched sibling channel keeps its full 256; no error,
    no warning) -- silently wrong segmentation, not a crash. The fix must stop
    Cellpose from EVER seeing the non-finite array, for a NaN, a +Inf, and a -Inf
    pixel alike."""
    img_path = tmp_path / f"clean_multichannel_{label}.tiff"
    source = _write_multichannel_float_tiff(img_path, non_finite_value=non_finite_value)
    assert not np.isfinite(source).all(), "fixture must genuinely contain a non-finite pixel"

    stub = _RecordingStubModel()
    with patch("shape.tools.cell_segmenter.tool.models.CellposeModel", return_value=stub):
        tool = Cell_Segmenter_Tool()
        result = tool.execute(image=str(img_path), query_cache_dir=str(tmp_path))

    # The load-bearing assertion: Cellpose must NEVER see the non-finite array.
    assert stub.calls == [], (
        f"model.eval was called {len(stub.calls)} time(s); the guard must refuse "
        "BEFORE any Cellpose call, not merely produce a finite input to it")
    assert "error" in result, f"expected a structured error result, got: {result}"
    assert "non-finite" in result["error"].lower()


def test_execute_finite_multichannel_image_still_segments_normally(tmp_path):
    """Guard-does-not-trigger control: an ordinary, fully-finite multi-channel image
    must still reach `model.eval` and segment normally."""
    img_path = tmp_path / "clean_multichannel_finite.tiff"
    source = _write_multichannel_float_tiff(img_path, non_finite_value=None)
    assert np.isfinite(source).all()

    stub = _RecordingStubModel(n_cells=2)
    with patch("shape.tools.cell_segmenter.tool.models.CellposeModel", return_value=stub):
        tool = Cell_Segmenter_Tool()
        result = tool.execute(image=str(img_path), query_cache_dir=str(tmp_path))

    assert len(stub.calls) >= 1
    assert np.isfinite(stub.calls[0]).all()
    assert "error" not in result
    assert result["status"] == "success"
    assert result["cell_count"] == 2
    assert result["seconds"] >= 0.0          # the call's run time, shown to the agentic loop per call


def test_concurrent_calls_of_one_image_get_distinct_attempts_and_pil_figures(tmp_path, monkeypatch):
    """The agentic loop runs a round's calls in worker threads: each call of the same image gets its own
    attempt number and files (the counter is under a lock), and no call in a worker thread draws a
    matplotlib figure (not thread-safe) -- the overlay and the mask display are PIL writes."""
    from concurrent.futures import ThreadPoolExecutor

    from shape.tools.cell_segmenter import tool as tool_module

    monkeypatch.delenv("SHAPE_FAST_FIGURES", raising=False)
    img_path = tmp_path / "cells.tiff"
    _write_multichannel_float_tiff(img_path)
    n = 6
    with patch("shape.tools.cell_segmenter.tool.models.CellposeModel", return_value=_RecordingStubModel(n_cells=2)), \
            patch.object(tool_module.VisualizationConfig, "create_professional_figure") as figure:
        tools = [Cell_Segmenter_Tool() for _ in range(n)]
        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(lambda t: t.execute(image=str(img_path), query_cache_dir=str(tmp_path),
                                                        image_id="cells"), tools))
    assert [r.get("status") for r in results] == ["success"] * n, [r.get("error") for r in results]
    assert sorted(r["attempt"] for r in results) == list(range(1, n + 1))
    overlays = {r["overlay_path"] for r in results}
    assert overlays == {str(Path(r["overlay_path"]).with_name(f"cell_overlay_cells_a{r['attempt']}.png"))
                        for r in results}
    assert len(overlays) == n and all(Path(p).exists() for p in overlays)
    assert len({r["mask_path"] for r in results}) == n
    figure.assert_not_called()


def test_the_agentic_loops_defaults_are_the_tool_at_the_planner_defaults():
    """The CARD lever Defaults (the agentic loop's configuration of a move from "input") are the
    planner's defaults (stage_map) and, for the other levers, execute()'s own."""
    import inspect

    from shape.agent.verification.stage_map import stage_by_name
    from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier

    sig = inspect.signature(Cell_Segmenter_Tool.execute).parameters
    planner = stage_by_name("segmentation").default_params
    for lever, value in SegmentationAgenticVerifier().defaults().items():
        assert value == planner.get(lever, sig[lever].default), lever


# ---- the large-image speed-up: the same outputs and readouts as before -------------------------------------
def _per_mask_br_scan(img_rgb, mask):
    """`color_class_filter._per_mask_br` before the speed-up: a full-image comparison and gather per mask."""
    R, B = img_rgb[..., 0], img_rgb[..., 2]
    ids = [int(i) for i in np.unique(mask) if i != 0]
    return {i: float(B[mask == i].mean() - R[mask == i].mean()) for i in ids}


def _label_fixtures(seed):
    """(RGB image, label mask) pairs: rectangles in random places (overlapping: masks of irregular shape), ids with
    gaps, several label dtypes and value ranges; a large mask (beyond numpy's pairwise-summation blocks) holding the
    first pixel in some; and a mask with no labels."""
    rng = np.random.default_rng(seed)
    for k in range(20):
        h, w = (int(x) for x in rng.integers(40, 200, 2))
        rgb = (rng.random((h, w, 3)) * (1.0, 255.0, 4095.0)[k % 3]).astype(np.float32)
        labels = np.zeros((h, w), np.int64)
        for i in range(1, int(rng.integers(2, 90))):
            r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
            dr, dc = (int(x) for x in rng.integers(1, 40, 2))
            labels[r:r + dr, c:c + dc] = i
        labels[rng.random(int(labels.max()) + 1)[labels] < 0.25] = 0          # ids with gaps
        if k % 4 == 0:
            labels[: 3 * h // 4, : 3 * w // 4] = labels.max() + 1             # a large mask at the first pixel
        yield rgb, labels.astype((np.int32, np.uint16, np.int64, np.uint32)[k % 4])
    yield np.ones((30, 40, 3), np.float32), np.zeros((30, 40), np.int32)


def test_per_mask_br_in_one_pass_gives_the_per_mask_scans_values_to_the_last_bit(monkeypatch):
    for rgb, labels in _label_fixtures(0):
        fast, scan = ccf._per_mask_br(rgb, labels), _per_mask_br_scan(rgb, labels)
        assert list(fast) == list(scan)                                       # the same ids, in the same order
        assert np.array(list(fast.values())).tobytes() == np.array(list(scan.values())).tobytes()
        stats = ccf.color_cluster_stats(rgb, labels)
        with monkeypatch.context() as m:
            m.setattr(ccf, "_per_mask_br", _per_mask_br_scan)
            assert ccf.color_cluster_stats(rgb, labels) == stats


def test_the_overlay_lookup_table_gives_cellposes_image_and_moves_the_random_state_as_cellpose():
    from cellpose import plot

    for k, (rgb, labels) in enumerate(_label_fixtures(1)):
        # the images the tool overlays (uint8 from normalize_for_overlay, 1- or 3-channel), and float ones
        img = (rgb[..., 0] if k % 2 else rgb) if k % 3 else rgb[..., 0] / max(1.0, float(rgb.max()))
        img = img.astype(np.uint8) if k % 3 == 1 else img
        np.random.seed(k)
        expected = plot.mask_overlay(img, labels)
        state = np.random.get_state()[1].copy()
        np.random.seed(k)
        overlay = tool_module._mask_overlay(img, labels)
        assert overlay.dtype == expected.dtype == np.uint8 and np.array_equal(overlay, expected), k
        assert np.array_equal(np.random.get_state()[1], state), k


class _LabelsModel:
    """Fake `CellposeModel`: the same label mask for every call."""

    def __init__(self, labels):
        self.labels = labels

    def eval(self, imgs, **kwargs):
        return [self.labels.copy()], None, None


def _smear(path, purple=(5, 17, 30)):
    """A stained-smear-like RGB image, 36 objects of 12 px in a 6 x 6 grid -- purple (bluer) at the ids `purple`,
    pink (redder) elsewhere -- and the label mask of all 36."""
    rgb = np.full((96, 96, 3), 235, np.uint8)
    labels = np.zeros((96, 96), np.int32)
    for k in range(36):
        r, c = divmod(k, 6)
        box = (slice(r * 16 + 2, r * 16 + 14), slice(c * 16 + 2, c * 16 + 14))
        labels[box] = k + 1
        rgb[box] = (120, 40, 170) if k + 1 in purple else (225, 120, 140)
    Image.fromarray(rgb).save(path)
    return labels


def test_the_large_image_speed_up_gives_the_same_outputs_and_readouts(tmp_path, monkeypatch):
    """The tool before the speed-up (Cellpose's overlay, the per-mask B-R scan, the filtered mask saved with its
    original ids) and after, from the same random state: the same overlay and mask-display pixels, the same masks
    (after the color/class filter relabelled 1-N, in their order), the same tool readouts and the same readouts of
    the agentic verifier."""
    from cellpose import plot

    from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier

    monkeypatch.setenv("SHAPE_FAST_FIGURES", "1")
    path = tmp_path / "smear.png"
    labels = _smear(path)
    readouts = ("status", "summary", "cell_count", "flow_error", "overlay_qc", "parameters", "segmented_channel",
                "attempt", "param_sig")

    def call(before, color_class_filter):
        with monkeypatch.context() as m:
            if before:
                m.setattr(tool_module, "_mask_overlay", plot.mask_overlay)
                m.setattr(ccf, "_per_mask_br", _per_mask_br_scan)
                m.setattr(tool_module, "relabel_sequential", lambda mask: (mask, None, None))
            with patch("shape.tools.cell_segmenter.tool.models.CellposeModel", return_value=_LabelsModel(labels)):
                np.random.seed(3)
                result = Cell_Segmenter_Tool().execute(
                    image=str(path), query_cache_dir=str(tmp_path / f"{before}_{color_class_filter}"),
                    image_id="smear", color_class_filter=color_class_filter)
            verifier = SegmentationAgenticVerifier()
            verifier._input_path = str(path)
            metrics = verifier.quantify(result, {"trial_index": 1, "image_path": str(path)})
            return result, metrics, verifier._colors[1]

    for color_class_filter in (None, "keep_high_br"):
        (old, old_metrics, old_colors), (new, new_metrics, new_colors) = (call(True, color_class_filter),
                                                                          call(False, color_class_filter))
        for old_png, new_png in zip(old["visual_outputs"], new["visual_outputs"]):    # the overlay, the mask display
            assert np.array_equal(np.asarray(Image.open(new_png)), np.asarray(Image.open(old_png)))
        old_mask, new_mask = np.load(old["mask_path"]), np.load(new["mask_path"])
        assert new_mask.dtype == old_mask.dtype and np.array_equal(new_mask, relabel_sequential(old_mask)[0])
        assert {k: new[k] for k in readouts} == {k: old[k] for k in readouts}
        assert new_metrics == old_metrics and new_colors == old_colors
    assert np.unique(old_mask).tolist() == [0, 5, 17, 30] and np.unique(new_mask).tolist() == [0, 1, 2, 3]
    assert new["cell_count"] == 3 and new_colors is None                 # the 3 kept masks: one color, no split


def test_the_mask_before_the_filter_is_recorded_only_when_the_filter_removed_masks(tmp_path, monkeypatch):
    """A call whose color/class filter removed masks also saves the mask as segmented (`prefilter_mask_path`,
    named outside the per-attempt readers' `cell_mask_*_a<N>.npy` pattern); every other output is the same as
    without it. No filter, or a filter with no split to act on (fewer than 4 masks), records nothing."""
    import fnmatch
    import os

    monkeypatch.setenv("SHAPE_FAST_FIGURES", "1")
    path = tmp_path / "smear.png"
    labels = _smear(path)

    def call(color_class_filter, mask=labels):
        out = tmp_path / f"run_{color_class_filter}_{int(mask.max())}"
        with patch("shape.tools.cell_segmenter.tool.models.CellposeModel", return_value=_LabelsModel(mask)):
            result = Cell_Segmenter_Tool().execute(image=str(path), query_cache_dir=str(out), image_id="smear",
                                                   color_class_filter=color_class_filter)
        return result, sorted(os.listdir(Path(result["mask_path"]).parent))

    plain, plain_files = call(None)
    filtered, filtered_files = call("keep_high_br")
    assert "prefilter_mask_path" not in plain and filtered["cell_count"] == 3
    before = filtered["prefilter_mask_path"]
    assert np.array_equal(np.load(before), labels) and Path(before).parent == Path(filtered["mask_path"]).parent
    assert set(filtered) - set(plain) == {"prefilter_mask_path"} and set(plain) <= set(filtered)
    assert set(filtered_files) - {Path(before).name} == set(plain_files)        # the same files, and this one
    assert not fnmatch.fnmatch(Path(before).name, "cell_mask_*_a*.npy")
    few = labels * np.isin(labels, (1, 2, 3))                          # 3 masks: no split, the filter keeps them all
    unsplit, _ = call("keep_high_br", few)
    assert unsplit["cell_count"] == 3 and "prefilter_mask_path" not in unsplit
