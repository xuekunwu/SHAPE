"""Task 073, defect 3a (adversarial defect hunt on `image_preprocessor`): `ImageData.
create_merged_rgb`'s inner `normalize_channel` computed its 1st/99th percentiles over the
FULL channel array, so a single non-finite pixel made both percentiles NaN, silently routed
control into the "no variation" branch, and `NaN * 0` (still NaN) survived into the merged
RGB output -- which a downstream `(merged * 255).astype(np.uint8)` cast then hit with
undefined behaviour (`RuntimeWarning: invalid value encountered in cast`).

Co-located with `image_data.py` (AGENTS.md sec 6 P5), matching the convention used
elsewhere in this task (e.g. `shape/tools/cell_segmenter/test_tool.py`).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import tifffile

from shape.utils.image_data import ImageData, ImageOrientationError


def test_create_merged_rgb_nan_pixel_does_not_survive_into_output():
    """Master reproduction pattern (defect 3a): a channel with real finite variance PLUS
    one NaN pixel. Pre-fix, `np.percentile` over the full array returned NaN for both
    p1/p99, `p99 > p1` was False (NaN comparisons are always False), and `x * 0` == NaN at
    every position -- corrupting the WHOLE channel, not just the NaN pixel."""
    size = 32
    rng = np.random.default_rng(0)
    img = rng.uniform(0.0, 20000.0, size=(size, size, 1)).astype(np.float32)
    img[5, 5, 0] = np.nan
    img_data = ImageData(data=img, dtype=img.dtype, channel_names=["bright-field"])

    merged = img_data.create_merged_rgb()

    assert np.isfinite(merged).all(), "no NaN/Inf may survive into the merged RGB output"


def test_create_merged_rgb_all_non_finite_channel_is_zeros_not_nan():
    """A channel with NO finite pixel at all used to hit the SAME `x * 0` == NaN bug via a
    different route (percentile over an all-NaN array is NaN either way). Must produce
    real zeros, matching the "All zeros if no variation" comment's actual intent."""
    size = 16
    img = np.full((size, size, 1), np.nan, dtype=np.float32)
    img_data = ImageData(data=img, dtype=img.dtype, channel_names=["bright-field"])

    merged = img_data.create_merged_rgb()

    assert np.array_equal(merged, np.zeros_like(merged))


def test_create_merged_rgb_constant_finite_channel_is_zeros_not_nan():
    """A genuinely constant (zero-variance) FINITE channel must still take the "no
    variation" branch and produce zeros -- the fix must not accidentally require
    non-finite input to reach that branch."""
    size = 16
    img = np.full((size, size, 1), 500.0, dtype=np.float32)
    img_data = ImageData(data=img, dtype=img.dtype, channel_names=["bright-field"])

    merged = img_data.create_merged_rgb()

    assert np.array_equal(merged, np.zeros_like(merged))


def test_create_merged_rgb_finite_input_unaffected_by_fix():
    """Constraint check: a purely finite, real-variance image must produce the exact same
    result as the pre-fix percentile-normalization formula (computing p1/p99 over the
    finite subset of an all-finite array is identical to computing them over the full
    array)."""
    size = 24
    rng = np.random.default_rng(7)
    img = rng.uniform(0.0, 20000.0, size=(size, size, 2)).astype(np.float32)
    img_data = ImageData(data=img, dtype=img.dtype, channel_names=["bright-field", "GFP"])

    merged = img_data.create_merged_rgb()

    # Reference: pre-073-fix formula (percentile over the FULL, here all-finite, array).
    def _pre_fix_normalize_channel(x):
        x = x.astype(np.float32)
        p1, p99 = np.percentile(x, [1, 99])
        if p99 > p1:
            normalized = np.clip(x, p1, p99)
            normalized = (normalized - p1) / (p99 - p1 + 1e-8)
        else:
            normalized = x * 0
        return normalized

    expected = np.zeros((size, size, 3), dtype=np.float32)
    bf = _pre_fix_normalize_channel(img_data.get_channel(0))
    gfp = _pre_fix_normalize_channel(img_data.get_channel(1))
    expected[:, :, 0] = bf
    expected[:, :, 1] = bf
    expected[:, :, 2] = bf
    expected[:, :, 1] = expected[:, :, 1] + gfp
    expected = np.clip(expected, 0, 1)

    assert np.array_equal(merged, expected)


def test_create_merged_rgb_downstream_uint8_cast_raises_no_runtime_warning():
    """End-to-end guard matching the real call sites (`image_processor.py`,
    `image_context.py`): `(merged * 255).astype(np.uint8)` must not raise
    `RuntimeWarning: invalid value encountered in cast/multiply` on an image with
    non-finite pixels."""
    size = 20
    rng = np.random.default_rng(3)
    img = rng.uniform(0.0, 20000.0, size=(size, size, 3)).astype(np.float32)
    img[2:4, :, :] = np.nan
    img[10:12, :, :] = np.inf
    img_data = ImageData(data=img, dtype=img.dtype, channel_names=["bright-field", "GFP", "DAPI"])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        merged = img_data.create_merged_rgb()
        u8 = (merged * 255).astype(np.uint8)

    assert u8.dtype == np.uint8


# =================================================================================================
# Task 080 (post-merge image I/O integrity, successor to 073): `_from_array`/`from_path`'s
# channel/Z/T orientation. `from_path` now reads a TIFF's own definitive axes tag
# (`tifffile.TiffFile(path).series[0].axes`, e.g. 'CYX', 'ZCYX') and the matching pixel array
# TOGETHER via `_read_tiff_array_and_axes`, and `_normalize_by_axes` uses that tag INSTEAD of
# guessing orientation from shape -- removing the whole shape-ambiguity class for axes-tagged
# TIFFs. `_normalize_by_shape` (task 073's fix, unchanged in its own logic) remains the FALLBACK
# for metadata-less bare arrays / axes-less files, where a genuinely ambiguous shape is still a
# loud refusal, never a guess.
#
# Referee watchpoint / root cause found during this task: `skimage.io.imread`'s top-level
# wrapper applies an UNDOCUMENTED internal swap (moves a first-or-last axis of size 3/4 to the
# last position, "RGB-like") that is independent of a TIFF's own axes metadata -- verified
# empirically, a channel-first (3, 48, 48) array reads back (48, 48, 3) via `skimage.io.imread`
# while `tifffile`'s own reader keeps it (3, 48, 48). Combining `skimage.io.imread`'s pixel
# array with `tifffile`-read axes metadata would silently desync the two and double-transpose.
# `_read_tiff_array_and_axes` avoids this entirely by reading the pixel array and the axes tag
# from the SAME `tifffile` series object, bypassing `skimage.io.imread` for any TIFF with a
# readable axes tag.
# =================================================================================================

def _write_channel_first_tiff(path, channels):
    """A real multi-channel TIFF via `ImageData.save`'s own writer (channel-FIRST on disk,
    ImageJ hyperstack convention) -- exactly the tool's "own round-trip" D1a/D1b concern.

    `stacked` is already unambiguously (H, W, C) BY CONSTRUCTION (`np.stack(..., axis=-1)`),
    so no orientation needs to be inferred -- build the `ImageData` directly rather than
    routing it through `_from_array`'s metadata-less shape heuristic. Task 080 no-plausible-
    channel-axis fix: for a 20-channel `channels` list this stacks to (H, W, 20), and 20
    exceeds `_normalize_by_shape`'s `_CHANNEL_AXIS_MAX` (16) -- correctly refused as
    genuinely ambiguous input, but this fixture's shape was never ambiguous to begin with.
    """
    stacked = np.stack(channels, axis=-1)  # (H, W, C)
    ImageData(data=stacked, dtype=stacked.dtype).save(str(path), format="tiff")


def test_from_path_five_channel_channel_first_tiff_reads_num_channels_five(tmp_path):
    """D1a (BRIEF S3.1): a 5-channel channel-first TIFF must read back num_channels==5,
    emitted (H, W, 5) with pixels intact -- never num_channels==width (pre-fix `d78dcc1`: 48,
    the image width)."""
    h, w = 64, 48
    channels = [np.full((h, w), v, dtype=np.uint8) for v in (10, 20, 30, 40, 50)]
    path = tmp_path / "five.tiff"
    _write_channel_first_tiff(path, channels)

    img = ImageData.from_path(str(path))

    assert img.shape == (h, w, 5)
    assert img.num_channels == 5
    for idx, value in enumerate((10, 20, 30, 40, 50)):
        assert np.array_equal(img.get_channel(idx), np.full((h, w), value, dtype=np.uint8))


def test_from_path_twenty_channel_own_round_trip_reads_num_channels_twenty_with_pixels_intact(tmp_path):
    """D1b (BRIEF S3.2): the tool's OWN `ImageData.save()` -> `from_path()` round-trip for a
    TRUE 20-channel image MUST reload as num_channels==20 with pixels intact -- a tool that
    cannot read back its own output is broken, so this is HANDLED, never refused (pre-fix
    `d78dcc1`: silently 48, the image width; the reference WIP's `_CHANNEL_AXIS_MAX=16` cliff
    would also have mis-handled 20 channels had it been the only fix)."""
    h, w = 64, 48
    channels = [np.full((h, w), v, dtype=np.uint8) for v in range(20)]
    path = tmp_path / "twenty.tiff"
    _write_channel_first_tiff(path, channels)

    img = ImageData.from_path(str(path))

    assert img.shape == (h, w, 20)
    assert img.num_channels == 20
    for idx in range(20):
        assert np.array_equal(img.get_channel(idx), np.full((h, w), idx, dtype=np.uint8))


def test_from_path_zcyx_hyperstack_loud_refuses_naming_the_axis(tmp_path):
    """D1c (BRIEF S3.3): a real ZCYX hyperstack (Z=3 > 1) has a Z axis `ImageData`'s HWC-only
    model cannot honestly represent -- must loud-refuse, naming the axes, never silently
    collapse/drop Z (pre-fix `d78dcc1`: silent `num_channels==3`, Z silently dropped)."""
    z, c, h, w = 3, 5, 64, 48
    zcyx = np.zeros((z, c, h, w), dtype=np.uint8)
    path = tmp_path / "zcyx.tiff"
    tifffile.imwrite(str(path), zcyx, imagej=True, metadata={"axes": "ZCYX"})

    with pytest.raises(ImageOrientationError, match="[Mm]ulti-Z"):
        ImageData.from_path(str(path))


def test_from_path_tzcyx_hyperstack_loud_refuses(tmp_path):
    """D1c variant named in the BRIEF: a TZCYX hyperstack (both T and Z real, > 1) must also
    loud-refuse -- channel-count-only success is not acceptable for either axis."""
    t, z, c, h, w = 2, 3, 4, 32, 24
    tzcyx = np.zeros((t, z, c, h, w), dtype=np.uint8)
    path = tmp_path / "tzcyx.tiff"
    tifffile.imwrite(str(path), tzcyx, imagej=True, metadata={"axes": "TZCYX"})

    with pytest.raises(ImageOrientationError):
        ImageData.from_path(str(path))


def test_from_path_singleton_z_axis_is_dropped_not_refused(tmp_path):
    """Only a REAL (> 1) Z/T axis is refused -- a singleton (size-1) Z axis is not a genuine
    stack and must be silently squeezed out, reading normally."""
    channels = [np.full((40, 32), v, dtype=np.uint8) for v in (10, 20, 30)]
    zcyx = np.stack(channels, axis=0)[np.newaxis]  # (1, 3, 40, 32), axes 'ZCYX'
    assert zcyx.shape == (1, 3, 40, 32)
    path = tmp_path / "singleton_z.tiff"
    # No `imagej=True`: tifffile's ImageJ writer already collapses a size-1 Z axis out of its
    # OWN metadata on write (verified), so this OME-style write is what actually preserves a
    # literal singleton Z axis tag to exercise the drop-not-refuse code path. Explicit
    # `photometric="minisblack"` avoids tifffile's default 3-plane-looks-like-RGB guess.
    tifffile.imwrite(str(path), zcyx, metadata={"axes": "ZCYX"}, photometric="minisblack")

    img = ImageData.from_path(str(path))

    assert img.shape == (40, 32, 3)
    assert img.num_channels == 3
    for idx, value in enumerate((10, 20, 30)):
        assert np.array_equal(img.get_channel(idx), np.full((40, 32), value, dtype=np.uint8))


def test_from_path_unsupported_axes_string_loud_refuses(tmp_path):
    """An axes string this method doesn't recognize (not just Y/X/[C/S] plus singleton Z/T)
    is a loud, named refusal -- never a silent guess."""
    arr = np.zeros((5, 64, 48), dtype=np.uint8)
    path = tmp_path / "weird_axes.tiff"
    tifffile.imwrite(str(path), arr, metadata={"axes": "QYX"})

    with pytest.raises(ImageOrientationError, match="[Uu]nsupported"):
        ImageData.from_path(str(path))


def test_from_path_channel_last_tiff_axes_yxs_reads_correctly(tmp_path):
    """A channel-LAST TIFF (samples-per-pixel / 'S' axis, e.g. written by
    `channel_io.save_channels_last`) must read back correctly via the axes path too --
    regression control 6's "5-channel channel-LAST -> keep", exercised through `from_path`
    with real TIFF axes metadata ('YXS') rather than a bare array."""
    h, w = 40, 40
    channels = [np.full((h, w), v, dtype=np.uint8) for v in (10, 20, 30, 40, 50)]
    arr = np.stack(channels, axis=-1)  # (H, W, C) on disk too
    path = tmp_path / "channel_last.tiff"
    tifffile.imwrite(str(path), arr, photometric="minisblack", planarconfig="contig")
    with tifffile.TiffFile(str(path)) as tf:
        assert tf.series[0].axes == "YXS"

    img = ImageData.from_path(str(path))

    assert img.shape == (h, w, 5)
    for idx, value in enumerate((10, 20, 30, 40, 50)):
        assert np.array_equal(img.get_channel(idx), np.full((h, w), value, dtype=np.uint8))


def test_from_path_png_non_tiff_uses_shape_heuristic_unaffected(tmp_path):
    """A non-TIFF file (PNG) never has axes metadata read at all -- `from_path` must take the
    EXACT pre-080 skimage/PIL + shape-heuristic path, unaffected by this task's changes."""
    from PIL import Image as PILImage

    h, w = 32, 24
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    path = tmp_path / "rgb.png"
    PILImage.fromarray(rgb, mode="RGB").save(str(path))

    img = ImageData.from_path(str(path))

    assert img.shape == (h, w, 3)
    assert np.array_equal(img.data, rgb)


# ---- Regression / invariant controls (BRIEF S3.6): the metadata-less shape heuristic ----------
# These call `_from_array` directly (no `axes`), exactly like a bare in-memory array -- the
# fallback path this task's D1 fix must leave correct.

def test_from_array_five_channel_channel_first_is_transposed_correctly():
    """A metadata-less 5-channel channel-first array (no TIFF axes tag available) must still
    be transposed correctly by the shape-heuristic fallback -- num_channels==5, never 48."""
    h, w = 64, 48
    channels = [np.full((h, w), v, dtype=np.uint8) for v in (10, 20, 30, 40, 50)]
    chw = np.stack(channels, axis=0)  # genuine (C, H, W), shape (5, 64, 48)

    img_data = ImageData._from_array(chw)

    assert img_data.shape == (h, w, 5)
    assert img_data.num_channels == 5
    for idx, value in enumerate((10, 20, 30, 40, 50)):
        assert np.array_equal(img_data.get_channel(idx), np.full((h, w), value, dtype=np.uint8))


def test_from_array_five_channel_channel_last_is_kept_unchanged():
    """Regression control: a metadata-less 5-channel channel-LAST array (already (H, W, C))
    must stay untouched, not accidentally transposed by the >=4-channel fix."""
    h, w = 64, 48
    channels = [np.full((h, w), v, dtype=np.uint8) for v in (10, 20, 30, 40, 50)]
    hwc = np.stack(channels, axis=-1)  # (H, W, C), shape (64, 48, 5)

    img_data = ImageData._from_array(hwc)

    assert img_data.shape == (h, w, 5)
    assert np.array_equal(img_data.data, hwc)


def test_from_array_ambiguous_tied_axes_raises_error():
    """A metadata-less shape where the first and last axes are EQUAL and both a plausible
    channel count (e.g. (5, 64, 5)) cannot be resolved without axes metadata -- must raise,
    never silently guess an orientation."""
    arr = np.zeros((5, 64, 5), dtype=np.uint8)
    with pytest.raises(ImageOrientationError, match="[Aa]mbiguous"):
        ImageData._from_array(arr)


# ---- QC blocker (Codex diff-review, Master-reproduced): no-plausible-channel-axis leak -------
# Root cause: the pre-fix `_normalize_by_shape` computed `channel_first` explicitly, but its
# `else` branch kept the array as channel-last UNCONDITIONALLY -- even when the last axis was
# also an implausible channel count (> _CHANNEL_AXIS_MAX). A bare (20, 64, 48) array has h0=20
# (> cap, not confidently channel-first) AND h2=48 (> cap, not a plausible channel-last count
# either), so it was silently read as num_channels=48 (the width) instead of being refused.
# The fix makes the channel-last branch an explicit, checked condition (symmetric to
# channel-first) with its own loud refusal when neither axis qualifies.

def test_from_array_no_plausible_channel_axis_raises_error():
    """Master repro (CONFIRMED), exact QC repro: `ImageData._from_array(np.zeros((20, 64, 48),
    uint16))` must raise `ImageOrientationError`, never silently return num_channels=48 (the
    array's width, mis-taken for a channel-last axis count)."""
    arr = np.zeros((20, 64, 48), dtype=np.uint16)
    with pytest.raises(ImageOrientationError, match="[Cc]annot determine"):
        ImageData._from_array(arr)


def test_from_array_no_plausible_channel_axis_raises_error_reverse_shape():
    """The transposed shape (64, 48, 20) is symmetric: 20 exceeds `_CHANNEL_AXIS_MAX` (16) at
    EITHER end, so neither the shape-heuristic's `channel_first` nor `channel_last` condition
    is satisfied in either orientation -- a bare 20-channel array is genuinely ambiguous without
    axes metadata (contrast with `test_from_path_twenty_channel_own_round_trip_...`, which
    handles a true 20-channel TIFF via its CYX axes tag, not this shape heuristic)."""
    arr = np.zeros((64, 48, 20), dtype=np.uint16)
    with pytest.raises(ImageOrientationError, match="[Cc]annot determine"):
        ImageData._from_array(arr)


def test_from_path_metadataless_no_plausible_channel_axis_propagates(tmp_path, monkeypatch):
    """`execute()`-level propagation check #1: even when reached through the full `from_path`
    call chain (not just a direct `_from_array` unit call), the refusal must come straight out
    -- `from_path`'s own `try/except` only wraps its `skimage.io.imread`/PIL read fallback, not
    the `_from_array` call after it, so nothing in this module can swallow it into a silent
    low-channel result.

    A REAL .tiff file cannot exercise this branch: `tifffile` always assigns some axes guess
    (e.g. 'QYX' for a plain array with no explicit metadata -- verified empirically), so a real
    TIFF always takes `_normalize_by_axes`, never this shape fallback. `_read_tiff_array_and_axes`
    is monkeypatched to return `(array, None)` -- i.e. "TIFF pixels read, but no axes tag" --
    which is the genuine metadata-less condition this fallback exists for.
    """
    arr = np.zeros((20, 64, 48), dtype=np.uint16)
    monkeypatch.setattr(
        ImageData, "_read_tiff_array_and_axes",
        staticmethod(lambda path: (arr, None)))
    img_path = tmp_path / "metadataless.tiff"
    img_path.write_bytes(b"placeholder")  # content unused; the read is monkeypatched above

    with pytest.raises(ImageOrientationError, match="[Cc]annot determine"):
        ImageData.from_path(str(img_path))


@pytest.mark.parametrize("num_channels", [1, 2, 3])
def test_from_array_small_channel_first_still_transposes(num_channels):
    """Regression control: the pre-existing (1/2/3, H, W) -> (H, W, C) case, unaffected."""
    h, w = 64, 48
    channels = [np.full((h, w), (i + 1) * 10, dtype=np.uint8) for i in range(num_channels)]
    chw = np.stack(channels, axis=0)

    img_data = ImageData._from_array(chw)

    assert img_data.shape == (h, w, num_channels)


def test_from_array_finite_three_channel_channel_last_control_unchanged():
    """Required control: a finite, single-image, <=3-channel channel-LAST array must be read
    back completely unaffected -- same shape, same pixels, byte-identical to the input."""
    rng = np.random.default_rng(11)
    hwc = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)

    img_data = ImageData._from_array(hwc.copy())

    assert img_data.shape == (64, 48, 3)
    assert np.array_equal(img_data.data, hwc)


def test_from_array_tiny_channel_last_kept_not_misread_as_channel_first():
    """Required invariant named in the BRIEF: a tiny (8, 8, 3) channel-LAST array must stay
    kept (not transposed) -- 8 is not < 3, so it never satisfies the channel-first condition."""
    arr = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    img_data = ImageData._from_array(arr.copy())

    assert img_data.shape == (8, 8, 3)
    assert np.array_equal(img_data.data, arr)


def test_from_array_two_dimensional_grayscale_gets_channel_dim_added():
    """Regression control: 2-D grayscale unaffected -- gets a trailing channel dim added."""
    arr = np.arange(64 * 48, dtype=np.uint8).reshape(64, 48)

    img_data = ImageData._from_array(arr.copy())

    assert img_data.shape == (64, 48, 1)
    assert np.array_equal(img_data.data[:, :, 0], arr)


def test_from_array_metadataless_four_dimensional_is_loud_refused():
    """Task 080: the old `_from_array` 4D branch's `shape[3] <= 4` heuristic is REMOVED --
    its own inline comment claimed `io.imread` never returns ndim==4, which D1c disproves
    empirically (a ZCYX hyperstack does). It was untested and unreachable through any real
    caller (verified: every current caller either goes through the axes-aware TIFF path, or
    the legacy fallback in `tool.py` always pre-reduces to 2D before calling `_from_array`).
    A metadata-less bare 4D array now hits the pre-existing generic 'unsupported dimensions'
    refusal instead of silently guessing."""
    arr = np.zeros((3, 5, 64, 48), dtype=np.uint8)
    with pytest.raises(ValueError, match="[Uu]nsupported image dimensions"):
        ImageData._from_array(arr)


def test_from_array_metadataless_four_dimensional_refusal_is_image_orientation_error():
    """Round-2 adversarial hunt, defect A: the 'Unsupported image dimensions' refusal used
    to be a PLAIN ValueError -- indistinguishable, to `tool.py`'s
    `except ImageOrientationError: raise` guard, from any other failure, so it fell into the
    generic `except Exception` branch and was silently retried by the legacy shape-only
    fallback loader. It must be typed as ImageOrientationError (a ValueError subclass) so it
    PROPAGATES instead."""
    arr = np.zeros((3, 5, 64, 48), dtype=np.uint8)
    with pytest.raises(ImageOrientationError, match="[Uu]nsupported image dimensions"):
        ImageData._from_array(arr)


# =================================================================================================
# Round-2 adversarial hunt (post-staging), defect A: `_normalize_by_axes` collapsed an axes
# string carrying TWO real (>1) channel-type axes at once (both 'C' and 'S' present, e.g. a
# 'CYXS' TIFF with 2 real channels each RGB) into a 4-D array, which `_from_array`'s "Failed to
# normalize to 3D array" check then raised as a PLAIN ValueError -- swallowed by `tool.py`'s
# legacy fallback exactly like the "Unsupported image dimensions" case above. Fix: (1) that
# refusal is now typed ImageOrientationError; (2) `_normalize_by_axes` itself now detects two
# REAL channel-type axes and refuses BEFORE ever producing a non-3D array, naming both axes and
# their sizes; a SINGLETON channel-type axis alongside a real one (e.g. CYXS with C==1, S==3,
# ordinary RGB with an unused leading C axis) is squeezed out and handled correctly, mirroring
# the existing singleton-Z/T squeeze.
# =================================================================================================

def _write_cyxs_tiff(path, c, s, h=40, w=48, dtype=np.uint16):
    """A real TIFF whose `series[0].axes` reads back literally 'CYXS' -- shape (c, h, w, s).
    Each of the `c` channel-planes is filled with a distinct constant per `s`-sample so a
    HANDLED (c==1) case can assert pixel values survive intact."""
    arr = np.zeros((c, h, w, s), dtype=dtype)
    for ci in range(c):
        for si in range(s):
            arr[ci, :, :, si] = (ci * 10 + si + 1) * 1000
    tifffile.imwrite(str(path), arr, metadata={"axes": "CYXS"})
    with tifffile.TiffFile(str(path)) as tf:
        assert tf.series[0].axes == "CYXS", (
            f"test setup assumption broken: tifffile did not preserve axes='CYXS' "
            f"(got {tf.series[0].axes!r})")
    return arr


def test_from_path_cyxs_two_real_channel_axes_raises_image_orientation_error(tmp_path):
    """Master repro (CONFIRMED): a CYXS TIFF with C=2 real channels, each S=3 (RGB) --
    `ImageData.from_path` must raise `ImageOrientationError` (never a plain ValueError, which
    is swallowed by `tool.py`'s legacy fallback -- see the `execute()` test below) naming both
    axes, rather than silently collapsing to a 1-channel artifact."""
    path = tmp_path / "cyxs_2_3.tiff"
    _write_cyxs_tiff(path, c=2, s=3)

    with pytest.raises(ImageOrientationError, match="CYXS") as excinfo:
        ImageData.from_path(str(path))
    message = str(excinfo.value)
    assert "C=2" in message and "S=3" in message


def test_from_path_cyxs_singleton_channel_axis_is_squeezed_and_handled(tmp_path):
    """Acceptance criterion: CYXS with C==1, S==3 is ordinary RGB with an unused leading C
    axis -- must be HANDLED (num_channels==3, pixels intact), never refused. The singleton C
    axis is squeezed out exactly like a singleton Z/T axis."""
    h, w = 40, 48
    path = tmp_path / "cyxs_1_3.tiff"
    arr = _write_cyxs_tiff(path, c=1, s=3, h=h, w=w)

    img = ImageData.from_path(str(path))

    assert img.shape == (h, w, 3)
    assert img.num_channels == 3
    for si in range(3):
        expected_value = (0 * 10 + si + 1) * 1000
        assert np.array_equal(img.get_channel(si), np.full((h, w), expected_value, dtype=np.uint16))


def test_normalize_by_axes_two_real_channel_axes_direct_unit_check():
    """Direct unit test of `_normalize_by_axes` (no TIFF round-trip needed): two REAL (>1)
    channel-type axes must raise before ever producing a non-3D array."""
    arr = np.zeros((2, 10, 12, 3), dtype=np.uint8)
    with pytest.raises(ImageOrientationError, match="more than one"):
        ImageData._normalize_by_axes(arr, "CYXS")


def test_normalize_by_axes_singleton_plus_real_channel_axis_direct_unit_check():
    """Direct unit test: a singleton channel-type axis alongside a real one must be squeezed
    out and the real one kept as the sole channel axis -- both orders (C singleton before S,
    and S singleton before C) must work."""
    arr_c1 = np.zeros((1, 10, 12, 3), dtype=np.uint8)
    reduced_c1 = ImageData._normalize_by_axes(arr_c1, "CYXS")
    assert reduced_c1.shape == (10, 12, 3)

    arr_s1 = np.zeros((3, 10, 12, 1), dtype=np.uint8)
    reduced_s1 = ImageData._normalize_by_axes(arr_s1, "CYXS")
    assert reduced_s1.shape == (10, 12, 3)


# ---- a colour image is red / green / blue, not bright-field / GFP / DAPI ---------------------
def test_an_8bit_three_channel_image_is_named_and_merged_as_rgb(tmp_path):
    from PIL import Image as _Image
    from shape.utils.image_data import ImageData
    rgb = np.zeros((6, 6, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 200, 40, 160          # a Giemsa-like purple
    path = tmp_path / "smear.png"
    _Image.fromarray(rgb).save(path)
    img = ImageData.from_path(str(path))
    assert img.channel_names == ["red", "green", "blue"]
    merged = img.create_merged_rgb()
    assert np.allclose(merged, rgb / 255.0)                       # the image's own colours


def test_a_16bit_three_channel_stack_keeps_its_imaging_channel_names():
    from shape.utils.image_data import ImageData
    stack = np.random.default_rng(0).integers(0, 4000, size=(6, 6, 3)).astype(np.uint16)
    img = ImageData._from_array(stack)
    assert img.channel_names == ["bright-field", "GFP", "DAPI"]
