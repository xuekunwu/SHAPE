"""Task 073, defect 3b (adversarial defect hunt on `image_preprocessor`):
`normalize_for_overlay` correctly computed `lo`/`hi` over the finite subset, but applied
them to the ORIGINAL `arr` (still holding any non-finite pixels) at the final clip/cast
line. `(NaN - lo) * ...` is NaN, `np.clip` does not remove NaN (NaN comparisons are always
False), and `.astype(np.uint8)` on NaN is UNDEFINED C behaviour (`RuntimeWarning: invalid
value encountered in cast` + a garbage byte). This is `gate_image`'s own production path
(`to_display_uint8` -> `normalize_for_overlay`), so its decisions depend on this function.

Co-located with `segmentation_qc.py` (AGENTS.md sec 6 P5).
"""

from __future__ import annotations

import warnings

import numpy as np

from shape.utils.segmentation_qc import normalize_for_overlay


def test_normalize_for_overlay_nan_pixel_is_defined_not_undefined_cast():
    """The NaN position must come back as a DEFINED uint8 value (the low end, matching
    what `np.clip` already does for a finite extremum), not an undefined cast result."""
    arr = np.full((8, 8), 500.0, dtype=np.float32)
    rng = np.random.default_rng(0)
    arr[:] = rng.uniform(0.0, 20000.0, size=(8, 8)).astype(np.float32)
    arr[3, 3] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = normalize_for_overlay(arr)

    assert out.dtype == np.uint8
    assert out[3, 3] == 0, "NaN must map to the low end (0), matching the clip convention"


def test_normalize_for_overlay_positive_and_negative_inf_map_to_high_and_low_end():
    rng = np.random.default_rng(1)
    arr = rng.uniform(0.0, 20000.0, size=(10, 10)).astype(np.float32)
    arr[1, 1] = np.inf
    arr[2, 2] = -np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = normalize_for_overlay(arr)

    assert out.dtype == np.uint8
    assert out[1, 1] == 255
    assert out[2, 2] == 0


def test_normalize_for_overlay_finite_input_byte_identical_to_pre_fix():
    """Constraint check (required by the task): the gate's decisions depend on
    `normalize_for_overlay`'s output, so a purely finite input must produce BIT-IDENTICAL
    uint8 output before and after the fix. Reference is the pre-073-fix formula applied
    directly (finite-only input means the fix's non-finite substitution is a no-op)."""
    rng = np.random.default_rng(42)
    arr = rng.uniform(0.0, 20000.0, size=(200, 200)).astype(np.float32)

    def _pre_fix_normalize_for_overlay(image):
        a = np.asarray(image, dtype=np.float32)
        finite = a[np.isfinite(a)]
        lo, hi = np.percentile(finite, (1.0, 99.0))
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        return np.clip((a - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

    expected = _pre_fix_normalize_for_overlay(arr)
    actual = normalize_for_overlay(arr)

    assert np.array_equal(actual, expected)


# ---- compute_flow_error: a metric it cannot measure is missing, never a perfect 0.0 ----------
def _fake_flow_error_fn(monkeypatch, fn):
    import shape.utils.segmentation_qc as qc
    monkeypatch.setattr(qc, "_get_flow_error_fn", lambda: fn)
    return qc


def _mask_and_flows():
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[2:5, 2:5] = 1
    return mask, [[None, np.zeros((2, 8, 8), dtype=np.float32), None]]


def test_compute_flow_error_failure_is_missing_not_zero(monkeypatch):
    def boom(mask, dP, device=None):
        raise RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")
    qc = _fake_flow_error_fn(monkeypatch, boom)
    mask, flows = _mask_and_flows()
    assert qc.compute_flow_error(mask, flows) is None
    assert qc.compute_flow_error(mask, [[None]]) is None              # no dP
    assert qc.compute_flow_error(np.zeros((8, 8), dtype=np.int32), flows) == 0.0   # no labels


def test_compute_flow_error_runs_on_the_models_device(monkeypatch):
    seen = {}

    def fn(mask, dP, device=None):
        seen["device"] = device
        return np.array([0.2, 0.4]), None
    qc = _fake_flow_error_fn(monkeypatch, fn)
    mask, flows = _mask_and_flows()
    assert abs(qc.compute_flow_error(mask, flows, device="cpu") - 0.3) < 1e-9
    assert seen["device"] == "cpu"
