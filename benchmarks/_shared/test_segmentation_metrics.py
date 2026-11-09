"""Offline validation of `segmentation_metrics.py`'s pure metric engine (committed --
AGENTS.md sec 6 P5). No existing test covered this module before task 038 generalized
`compute_metrics`/`call_deltas`/`retune_deltas`/`total_gain` with a `metric_key` param
(default "ap50") so `crop_NeurIPS22-CellSeg/run_shape_crop.py` can reuse the SAME engine
for `crop_quality_score` -- this file locks in that the segmenter's own default path
(`metric_key="ap50"`, the implicit default before this change existed) is byte-identical,
and that a different `metric_key` produces the correctly-renamed keys/values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segmentation_metrics as sm  # noqa: E402


def _ap50_records():
    """Two images: one with a retune that helped, one with a wasted retune -- exercises
    tool_calls/retunes/positive/wasted/gain all at once."""
    return [
        {"image_id": "a", "baseline_ap50": 0.50, "call_ap50": [0.60, 0.75],
         "tokens": 100, "seconds": 10.0},
        {"image_id": "b", "baseline_ap50": 0.40, "call_ap50": [0.55, 0.551],
         "tokens": 50, "seconds": 5.0},
    ]


def test_compute_metrics_default_metric_key_is_ap50_byte_identical_keys():
    """The default call (no metric_key) must produce EXACTLY the same key names the
    segmenter's own callers (`run_shape_neurips_cellseg.py`, `_cmd_build`/`_cmd_compare`)
    already expect -- `total_ap50_gain`/`ap50_gain_per_tool_call`/etc, not
    `total_None_gain` or any other artifact of the generalization."""
    m = sm.compute_metrics(_ap50_records())
    expected_keys = {
        "n_images", "total_ap50_gain", "tool_calls_per_image", "retunes_per_image",
        "positive_tool_call_rate", "wasted_tool_call_rate", "token_cost_per_image",
        "time_per_image_s", "ap50_gain_per_tool_call", "ap50_gain_per_extra_call",
        "ap50_gain_per_1k_tokens", "ap50_gain_per_minute", "evaluable_retunes",
    }
    assert set(m) == expected_keys


def test_compute_metrics_default_metric_key_matches_explicit_ap50():
    """Calling without `metric_key` (positional-default) and calling with
    `metric_key="ap50"` explicitly must be IDENTICAL -- proves the new param is a true
    no-op default, not a behavior change for the segmenter's existing call sites."""
    records = _ap50_records()
    assert sm.compute_metrics(records) == sm.compute_metrics(records, metric_key="ap50")


def test_compute_metrics_ap50_values_hand_checked():
    """Hand-computed expectation on the fixture above, pinning the exact numbers (not just
    key names) so a future refactor cannot silently drift the arithmetic."""
    m = sm.compute_metrics(_ap50_records())
    # tool calls: 2 + 2 = 4 over 2 images
    assert m["tool_calls_per_image"] == pytest.approx(2.0)
    # retunes (calls after the first): 1 per image = 2 total
    assert m["retunes_per_image"] == pytest.approx(1.0)
    assert m["evaluable_retunes"] == 2
    # retune deltas: a: 0.75-0.60=0.15 (positive, >=eps) ; b: 0.551-0.55=0.001 (wasted, <eps)
    assert m["positive_tool_call_rate"] == pytest.approx(0.5)
    assert m["wasted_tool_call_rate"] == pytest.approx(0.5)
    # total gain: a: 0.75-0.50=0.25 ; b: 0.551-0.40=0.151 -> sum=0.401
    assert m["total_ap50_gain"] == pytest.approx(0.401)
    assert m["ap50_gain_per_tool_call"] == pytest.approx(0.401 / 4)
    # per EXTRA call: the gain over the retunes only (the first call is not extra): 0.401 / 2
    assert m["ap50_gain_per_extra_call"] == pytest.approx(0.401 / 2)
    # tokens: 100+50=150 over 2 images -> 75/image; gain_per_1k_tokens = 0.401*1000/150
    assert m["token_cost_per_image"] == pytest.approx(75.0)
    assert m["ap50_gain_per_1k_tokens"] == pytest.approx(0.401 * 1000.0 / 150.0)
    # seconds: 10+5=15 over 2 images -> 7.5/image; gain_per_minute = 0.401/(15/60)
    assert m["time_per_image_s"] == pytest.approx(7.5)
    assert m["ap50_gain_per_minute"] == pytest.approx(0.401 / (15.0 / 60.0))


def test_compute_metrics_generalizes_to_a_different_metric_key():
    """A non-AP50 metric_key (e.g. the cropper's `crop_quality_score`) produces the SAME
    metric SHAPE with correctly-renamed keys, reading `baseline_<key>`/`call_<key>` from
    the record instead of the hardcoded ap50 field names."""
    records = [
        {"image_id": "x", "baseline_crop_quality_score": 0.60,
         "call_crop_quality_score": [0.80], "tokens": 20, "seconds": 2.0},
        {"image_id": "y", "baseline_crop_quality_score": 0.55,
         "call_crop_quality_score": [0.70], "tokens": 30, "seconds": 3.0},
    ]
    m = sm.compute_metrics(records, metric_key="crop_quality_score")
    expected_keys = {
        "n_images", "total_crop_quality_score_gain", "tool_calls_per_image",
        "retunes_per_image", "positive_tool_call_rate", "wasted_tool_call_rate",
        "token_cost_per_image", "time_per_image_s", "crop_quality_score_gain_per_tool_call",
        "crop_quality_score_gain_per_extra_call", "crop_quality_score_gain_per_1k_tokens",
        "crop_quality_score_gain_per_minute", "evaluable_retunes",
    }
    assert set(m) == expected_keys
    # no ap50 keys leak in when a different metric_key is used
    assert not any("ap50" in k for k in m)
    # each image has exactly 1 call (no retunes) -> tool_calls_per_image=1, 0 evaluable retunes
    assert m["tool_calls_per_image"] == pytest.approx(1.0)
    assert m["evaluable_retunes"] == 0
    assert m["positive_tool_call_rate"] is None    # ratio(0, 0) -> None, no retunes to rate
    assert m["wasted_tool_call_rate"] is None
    # gain: x: 0.80-0.60=0.20 ; y: 0.70-0.55=0.15 -> sum=0.35
    assert m["total_crop_quality_score_gain"] == pytest.approx(0.35)
    assert m["crop_quality_score_gain_per_tool_call"] == pytest.approx(0.35 / 2)
    assert m["crop_quality_score_gain_per_extra_call"] is None   # no retunes: ratio(0.35, 0) -> None


def test_call_deltas_and_retune_deltas_respect_metric_key():
    record = {"baseline_x": 1.0, "call_x": [2.0, 2.5, 2.4]}
    assert sm.call_deltas(record, metric_key="x") == pytest.approx([1.0, 0.5, -0.1])
    assert sm.retune_deltas(record, metric_key="x") == pytest.approx([0.5, -0.1])
    # a metric_key the record has no fields for -> empty, not a crash
    assert sm.call_deltas(record, metric_key="ap50") == []
    assert sm.retune_deltas(record, metric_key="ap50") == []


def test_total_gain_respects_metric_key():
    record = {"baseline_y": 0.3, "call_y": [0.4, 0.6]}
    assert sm.total_gain(record, metric_key="y") == pytest.approx(0.3)
    assert sm.total_gain(record, metric_key="ap50") is None   # missing fields -> None


def test_compute_metrics_missing_tokens_or_seconds_yields_none_not_crash():
    records = [{"baseline_ap50": 0.5, "call_ap50": [0.6], "tokens": None, "seconds": None}]
    m = sm.compute_metrics(records)
    assert m["token_cost_per_image"] is None
    assert m["time_per_image_s"] is None
    assert m["ap50_gain_per_1k_tokens"] is None
    assert m["ap50_gain_per_minute"] is None
