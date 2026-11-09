"""
Unified Image Data Representation

Core principle: Single-channel is a special case of multi-channel (C=1)
All images are represented as (H, W, C) format where C >= 1
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
import tifffile  # Still needed for saving multi-channel TIFF
from PIL import Image  # Still needed for saving single-channel images
from skimage import io  # Unified image loading (consistent with reference implementation)
import os

# Task 080: metadata-less fallback heuristic (`_normalize_by_shape`) largest channel count
# it will treat as plausibly "the channel axis". Real microscopy panels don't realistically
# exceed this; a small-image spatial axis (e.g. a genuine 8px-wide crop) can still
# coincidentally be this small, which is why the heuristic ALSO requires the candidate axis
# to be strictly smaller than both other axes before trusting it.
_CHANNEL_AXIS_MAX = 16

# A colour image's channels (`_infer_channel_names`): named so, and merged as true colour.
RGB_CHANNEL_NAMES = ['red', 'green', 'blue', 'alpha']
_RGB_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


class ImageOrientationError(ValueError):
    """A deliberate, informative refusal from `ImageData._from_array`/`from_path`: the
    source array's channel/Z/T orientation could not be read correctly (definitively from
    TIFF axes metadata, or unambiguously from shape) and was NOT silently guessed. Callers
    that fall back to a permissive legacy loader on a generic failure must catch this type
    separately and re-raise it, never swallow it into that fallback -- the whole point of
    this task is that these refusals must PROPAGATE, not be silently retried away."""
    pass


@dataclass
class ImageData:
    """
    Unified image data representation: single-channel is a special case of multi-channel (1 channel)
    
    Core principles:
    - All images are represented as (H, W, C) format, C >= 1
    - Single-channel images: C=1
    - Multi-channel images: C>1
    """
    data: np.ndarray  # Always in (H, W, C) format
    dtype: np.dtype
    channel_names: Optional[List[str]] = None  # e.g., ['bright-field', 'GFP', 'DAPI']
    source_path: Optional[str] = None
    
    @classmethod
    def from_path(cls, path: str, channel_names: Optional[List[str]] = None) -> 'ImageData':
        """
        Load image from file path, automatically detect and handle multi-channel

        Unified logic (consistent with reference implementation):
        1. For a TIFF with readable axes metadata (`series[0].axes`, e.g. 'CYX', 'ZCYX'):
           read the pixel array via `tifffile` directly (task 080) -- deliberately NOT
           `skimage.io.imread`, whose top-level wrapper applies an undocumented internal
           swap (moves a first-or-last axis of size 3/4 to the end, "RGB-like") that can
           desync the returned array's actual axis order from that same axes metadata
           (verified: a channel-first (3, 48, 48) array comes back (48, 48, 3) from
           `skimage.io.imread` while `tifffile`'s own reader keeps it (3, 48, 48)).
        2. Otherwise (non-TIFF, or a TIFF with no axes tag): load via skimage.io.imread,
           falling back to PIL on failure -- unchanged from the pre-080 behavior.
        3. Normalize to (H, W, C) format -- via the axes metadata when known, else the
           shape heuristic (fallback for metadata-less arrays / axes-less files)
        4. Detect channel count
        5. Infer channel names (if not provided)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        tiff_array, axes = cls._read_tiff_array_and_axes(path)
        if tiff_array is not None:
            # `_from_array` (below) is NOT wrapped in a try/except here, so an
            # `ImageOrientationError` it raises (ambiguous/unsupported orientation, a real
            # multi-Z/T stack) PROPAGATES rather than being swallowed into any fallback.
            return cls._from_array(tiff_array, channel_names=channel_names, source_path=path,
                                    axes=axes)

        try:
            img_array = io.imread(path)
        except Exception as e:
            print(f"Warning: skimage.io.imread failed for {path}, trying PIL: {e}")
            try:
                img = Image.open(path)
                img_array = np.array(img)
            except Exception as e2:
                raise ValueError(f"Failed to load image from {path}: {e2}") from e2

        return cls._from_array(img_array, channel_names=channel_names, source_path=path)

    @staticmethod
    def _read_tiff_array_and_axes(path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Read a TIFF's own pixel array AND its definitive axes tag (e.g. 'CYX', 'ZCYX')
        TOGETHER, from the SAME `tifffile` series object -- guaranteeing the array's actual
        axis order matches the returned axes string (see `from_path`'s docstring for why
        `skimage.io.imread` cannot give that guarantee). Returns `(None, None)` for a
        non-TIFF path, a TIFF whose first series has no axes tag, or any read failure --
        `from_path`'s skimage/PIL path is the fallback for those metadata-less cases."""
        lower = str(path).lower()
        if not (lower.endswith('.tif') or lower.endswith('.tiff')):
            return None, None
        try:
            with tifffile.TiffFile(path) as tf:
                axes = tf.series[0].axes
                if not axes:
                    return None, None
                array = tf.series[0].asarray()
            return array, axes
        except Exception:
            return None, None

    @classmethod
    def _from_array(cls, img_array: np.ndarray, channel_names: Optional[List[str]] = None,
                     source_path: Optional[str] = None, axes: Optional[str] = None) -> 'ImageData':
        """
        Create ImageData from numpy array, normalize to (H, W, C) format.

        `axes` (task 080): the source TIFF's own definitive axes tag (e.g. 'CYX', 'ZCYX'),
        when known -- read by `from_path` via `_read_tiff_array_and_axes` and used INSTEAD of
        guessing orientation from shape. Callers that construct an `ImageData` directly from
        a bare array (no `axes`) get the metadata-less shape heuristic, unchanged.
        """
        if axes:
            img_array = cls._normalize_by_axes(img_array, axes)
        else:
            img_array = cls._normalize_by_shape(img_array)

        # Ensure we have (H, W, C) format
        if img_array.ndim != 3:
            # Task 080 defect A: this used to be a PLAIN ValueError, which is
            # indistinguishable from any other failure to `tool.py`'s
            # `except ImageOrientationError: raise` guard -- it fell through to
            # `except Exception` and got silently retried by the legacy shape-only
            # fallback loader instead of propagating. Typed as ImageOrientationError so it
            # PROPAGATES, matching every other orientation refusal in this module.
            raise ImageOrientationError(
                f"Failed to normalize to 3D array, got shape: {img_array.shape}")

        h, w, c = img_array.shape

        # Infer channel names if not provided
        if channel_names is None:
            channel_names = cls._infer_channel_names(c, source_path, dtype=img_array.dtype)

        return cls(
            data=img_array,
            dtype=img_array.dtype,
            channel_names=channel_names,
            source_path=source_path
        )

    @classmethod
    def _normalize_by_shape(cls, img_array: np.ndarray) -> np.ndarray:
        """Metadata-less orientation heuristic -- the FALLBACK used only when no axes tag is
        available (a bare array, or a non-TIFF/axes-less file). Infers channel-first vs
        channel-last from the array shape; a genuinely ambiguous shape is a loud refusal
        rather than a silent guess (this task's standing contract: correct or loud-refuse,
        never silent-wrong).
        """
        if img_array.ndim == 2:
            # 2D grayscale: add channel dimension -> (H, W, 1)
            img_array = img_array[:, :, np.newaxis]
        elif img_array.ndim == 3:
            # 3D: determine if (H, W, C) or (C, H, W). The channel axis is whichever of the
            # FIRST or LAST axis is both a plausible channel count (<= _CHANNEL_AXIS_MAX) AND
            # strictly smaller than the OTHER two axes -- this keeps every previously-correct
            # case correct ((1/2/3, H, W) -> transpose; (H, W, C) -> keep; a tiny (8, 8, 3) ->
            # keep, since 8 is not < 3) and additionally handles channel-first counts >= 4
            # that the original `shape[0] in (1, 2, 3)` check missed. A shape where the first
            # and last axes are EQUAL and both a plausible channel count (e.g. (5, 64, 5)) is
            # a genuine tie that cannot be resolved from shape alone -- loud refuse rather
            # than silently guess. Symmetrically, a shape where NEITHER axis is a plausible
            # channel count (e.g. (20, 64, 48): first axis 20 > cap, last axis 48 > cap) used
            # to fall through to an unconditional "keep as channel-last" `else`, silently
            # mis-parsing (20, 64, 48) as 48 channels -- QC-flagged defect, fixed by making
            # the channel-last branch an explicit, checked condition with its own refusal.
            h0, h1, h2 = img_array.shape
            channel_first = h0 <= _CHANNEL_AXIS_MAX and h0 < h1 and h0 < h2
            channel_last = h2 <= _CHANNEL_AXIS_MAX and h2 < h0 and h2 < h1
            if h0 == h2 and h0 <= _CHANNEL_AXIS_MAX and h0 < h1:
                raise ImageOrientationError(
                    f"Ambiguous channel axis for array shape {img_array.shape}: the first "
                    f"axis and the last axis are equal ({h0}) and both a plausible channel "
                    f"count (<= {_CHANNEL_AXIS_MAX}), so channel-first vs channel-last cannot "
                    f"be determined from shape alone. Pass an unambiguous (H, W, C) array, a "
                    f"TIFF carrying axes metadata, or pre-transpose channel-first input "
                    f"before calling ImageData.from_path/_from_array.")
            elif channel_first:
                # First dimension is small and plausible -- (C, H, W), transpose to (H, W, C)
                img_array = np.transpose(img_array, (1, 2, 0))
            elif channel_last:
                # Last dimension is small and plausible -- already (H, W, C), keep as-is.
                pass
            else:
                raise ImageOrientationError(
                    f"Cannot determine channel axis for array shape {img_array.shape}: "
                    f"neither the first axis ({h0}) nor the last axis ({h2}) is a plausible "
                    f"channel count (<= {_CHANNEL_AXIS_MAX}) that is strictly smaller than "
                    f"the other two axes, so channel-first vs channel-last orientation cannot "
                    f"be inferred from shape alone. Pass a TIFF carrying axes metadata, or an "
                    f"unambiguous (H, W, C) array, before calling ImageData.from_path/"
                    f"_from_array.")
        else:
            # Task 080 defect A: typed as ImageOrientationError (was a plain ValueError) for
            # the same reason as `_from_array`'s "Failed to normalize to 3D array" refusal
            # above -- a bare ValueError is swallowed by `tool.py`'s legacy fallback instead
            # of propagating.
            raise ImageOrientationError(f"Unsupported image dimensions: {img_array.ndim}")
        return img_array

    @classmethod
    def _normalize_by_axes(cls, img_array: np.ndarray, axes: str) -> np.ndarray:
        """Reorder `img_array` to (H, W, C) using a TIFF's own definitive axes tag (e.g.
        'CYX', 'YXC', 'ZCYX', 'TZCYX', 'YXS') instead of guessing from shape. A real (>1) Z
        or T axis cannot be honestly represented by this class's HWC-only model -- loud
        refuse it, naming the axes, rather than silently collapsing/dropping it (the exact
        silent-wrong failure this task fixes). An axes string this method doesn't recognize
        is also a loud, named refusal, never a guess.
        """
        axes = axes.upper()
        if len(axes) != img_array.ndim:
            # Axes metadata inconsistent with the actual decoded array -- cannot be
            # trusted; fall back to the shape heuristic.
            return cls._normalize_by_shape(img_array)

        kept_positions = []
        for i, a in enumerate(axes):
            if a in ('Z', 'T'):
                if img_array.shape[i] > 1:
                    raise ImageOrientationError(
                        f"Cannot represent a multi-{a} TIFF (axes={axes!r}, shape="
                        f"{img_array.shape}) as a single (H, W, C) image -- ImageData's "
                        f"model has no Z/T axis, and honest Z/T handling is out of scope. "
                        f"Select a single {a} slice before loading, or use a Z/T-aware tool.")
                continue  # singleton Z/T -- not a real stack, drop it
            kept_positions.append(i)

        kept_axes = ''.join(axes[i] for i in kept_positions)
        if 'Y' not in kept_axes or 'X' not in kept_axes or set(kept_axes) - set('YXCS'):
            raise ImageOrientationError(
                f"Unsupported TIFF axes string {axes!r} for shape {img_array.shape} -- "
                f"ImageData only understands Y, X, and an optional channel axis (C or S), "
                f"plus singleton Z/T slices.")

        # Task 080 defect A: an axes string can carry TWO channel-type axes at once (both
        # 'C' and 'S' present, e.g. 'CYXS' from a multi-channel-RGB TIFF). A singleton one
        # (size 1) is ordinary -- e.g. CYXS with C==1, S==3 is plain RGB with an unused
        # leading C axis -- squeeze it out, mirroring the singleton-Z/T squeeze above. Two
        # REAL (>1) channel-type axes (e.g. C>=2 AND S>=2, "CYXS" with 2 real channels each
        # RGB) genuinely cannot be flattened into ImageData's single (H, W, C) channel axis
        # without guessing which one comes first -- loud refuse, naming both axes and their
        # sizes, rather than silently transposing to a 4-D array (which used to fall through
        # to the generic "Failed to normalize to 3D array" refusal below, undistinguishable
        # from any other failure and swallowed by `tool.py`'s legacy fallback).
        channel_type_positions = [i for i in kept_positions if axes[i] in ('C', 'S')]
        real_channel_positions = [i for i in channel_type_positions if img_array.shape[i] > 1]
        if len(real_channel_positions) > 1:
            sizes = ', '.join(f"{axes[i]}={img_array.shape[i]}" for i in real_channel_positions)
            raise ImageOrientationError(
                f"Unsupported TIFF axes {axes!r} for shape {img_array.shape}: more than one "
                f"real (>1) channel-type axis ({sizes}) -- ImageData's (H, W, C) model holds "
                f"only ONE channel axis, and flattening two real ones would require guessing "
                f"their relative order. Pre-reshape/select a single channel axis before "
                f"calling ImageData.from_path/_from_array.")
        if len(channel_type_positions) > 1:
            singleton_channel_positions = [i for i in channel_type_positions if img_array.shape[i] == 1]
            kept_positions = [i for i in kept_positions if i not in singleton_channel_positions]
            kept_axes = ''.join(axes[i] for i in kept_positions)

        reduced = img_array
        if len(kept_positions) != img_array.ndim:
            drop_positions = tuple(i for i in range(img_array.ndim) if i not in kept_positions)
            reduced = np.squeeze(img_array, axis=drop_positions)

        order = [kept_axes.index('Y'), kept_axes.index('X')]
        channel_positions = [i for i, a in enumerate(kept_axes) if a in ('C', 'S')]
        order.extend(channel_positions)
        reduced = np.transpose(reduced, order)

        if not channel_positions:
            reduced = reduced[:, :, np.newaxis]

        return reduced

    @staticmethod
    def _infer_channel_names(num_channels: int, source_path: Optional[str] = None,
                             dtype=None) -> List[str]:
        """
        Infer channel names based on channel count and file path.

        A 3- or 4-channel image stored as 8-bit, or in an RGB file format (png / jpg / bmp),
        is a colour image: its channels are red, green, blue (and alpha), not imaging
        channels -- a stained brightfield slide (H&E, Giemsa) must not be described as
        bright-field + GFP + DAPI, nor merged as such (`create_merged_rgb`).
        """
        ext = os.path.splitext(str(source_path or ""))[1].lower()
        is_colour = (np.dtype(dtype) == np.uint8 if dtype is not None else False) or ext in _RGB_EXTS
        if num_channels in (3, 4) and is_colour:
            return RGB_CHANNEL_NAMES[:num_channels]
        if num_channels == 1:
            return ['grayscale']
        elif num_channels == 2:
            # Common: bright-field + GFP
            return ['bright-field', 'GFP']
        elif num_channels == 3:
            # Common: bright-field + GFP + DAPI
            return ['bright-field', 'GFP', 'DAPI']
        elif num_channels == 4:
            return ['bright-field', 'GFP', 'DAPI', 'Channel 4']
        else:
            return [f'Channel {i+1}' for i in range(num_channels)]
    
    def get_channel(self, idx: int) -> np.ndarray:
        """Get single channel (returns 2D array)"""
        if idx < 0 or idx >= self.num_channels:
            raise IndexError(f"Channel index {idx} out of range [0, {self.num_channels})")
        return self.data[:, :, idx]
    
    def get_channels(self, indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """Get multiple channels"""
        if indices is None:
            indices = list(range(self.num_channels))
        return [self.get_channel(i) for i in indices]
    
    @property
    def num_channels(self) -> int:
        """Number of channels"""
        return self.data.shape[2]
    
    @property
    def height(self) -> int:
        """Image height"""
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        """Image width"""
        return self.data.shape[1]
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Image shape (H, W, C)"""
        return self.data.shape
    
    @property
    def is_multi_channel(self) -> bool:
        """Whether this is a multi-channel image (>1 channel)"""
        return self.num_channels > 1
    
    @property
    def is_single_channel(self) -> bool:
        """Whether this is a single-channel image (1 channel)"""
        return self.num_channels == 1
    
    def informative_channel_idx(self, default: int = 0) -> int:
        """Pick the channel to segment. Returns `default`, unless that channel is
        empty / near-constant (std ~ 0) while another channel carries signal -- e.g. a
        fluorescence image stored as RGB whose signal sits in green, leaving the red
        channel 0 all zeros. Feeding Cellpose that all-zero channel yields 0 cells, so
        fall back to the highest-variance channel. Single-channel images are unchanged.
        """
        if self.num_channels <= 1:
            return default
        stds = [float(np.std(self.get_channel(c))) for c in range(self.num_channels)]
        if stds[default] > 1e-6:
            return default
        best = int(np.argmax(stds))
        return best if stds[best] > 1e-6 else default

    def to_segmentation_input(self, channel_idx: int = 0) -> np.ndarray:
        """
        Convert to format needed by segmentation tools (single channel, float32).
        Uses `channel_idx`, but falls back to the informative channel if that one is
        empty (see informative_channel_idx).
        """
        channel = self.get_channel(self.informative_channel_idx(channel_idx))
        if channel.dtype != np.float32:
            channel = channel.astype(np.float32)
        return channel
    
    def to_uint8(self, channel_idx: Optional[int] = None) -> np.ndarray:
        """
        Convert to uint8 format for display
        If channel_idx is None, converts all channels
        """
        # Route through the canonical dtype-aware converter so display, overlay and
        # the SSL crop input all normalize the same way. It keeps integer values
        # (uint8 passthrough, uint16 fixed-scaled) and range-normalizes only float,
        # which a naive clip would black out ([0,1]) or saturate (12/16-bit-as-float).
        from shape.utils.segmentation_qc import to_display_uint8
        if channel_idx is not None:
            return to_display_uint8(self.get_channel(channel_idx))
        return to_display_uint8(self.data)
    
    def create_merged_rgb(self, channel_mapping: Optional[Dict[int, str]] = None) -> np.ndarray:
        """
        Create merged RGB view for visualization
        
        Unified logic:
        - Single-channel: grayscale -> RGB
        - Multi-channel: merge according to channel_mapping
        """
        def normalize_channel(x, method='percentile'):
            """
            Normalize channel to [0, 1] range
            
            Args:
                x: Channel data (numpy array)
                method: Normalization method
                    - 'percentile': Use 1st and 99th percentiles (robust to outliers, better for training)
                    - 'minmax': Use min/max (original method, better for display)
            """
            x = x.astype(np.float32)
            
            if method == 'percentile':
                # Use percentile normalization for better training stability
                # This reduces the impact of outliers and provides more consistent contrast
                # BLOCKER FIX (defect 3a, adversarial defect hunt): `np.percentile` over the
                # FULL array let one non-finite pixel make both p1/p99 NaN, so `p99 > p1`
                # was always False (NaN comparisons are always False) and control fell into
                # the "no variation" branch -- but `x * 0` is NaN, not 0, wherever `x` was
                # already NaN, so the NaN survived. Compute p1/p99 over the FINITE subset.
                finite = x[np.isfinite(x)]
                if finite.size == 0:
                    return np.zeros_like(x)
                p1, p99 = np.percentile(finite, [1, 99])
                if p99 > p1:
                    normalized = np.clip(x, p1, p99)
                    normalized = (normalized - p1) / (p99 - p1 + 1e-8)
                    # A NaN pixel is untouched by `np.clip` (NaN comparisons are always
                    # False, unlike +/-Inf which clip correctly bounds to p99/p1) and would
                    # otherwise survive into the merged RGB output and hit an undefined
                    # `.astype(np.uint8)` cast downstream. Map it to the low end, matching
                    # `normalize_for_overlay`'s convention for a non-finite source pixel.
                    normalized = np.nan_to_num(normalized, nan=0.0)
                else:
                    normalized = np.zeros_like(x)  # All zeros if no variation
                return normalized
            else:
                # Original min-max normalization (better for display)
                x_min = x.min()
                x_max = x.max()
                if x_max > x_min:
                    return (x - x_min) / (x_max - x_min + 1e-8)
                return x
        
        if self.is_single_channel:
            # Single-channel: grayscale -> RGB
            gray = normalize_channel(self.get_channel(0))
            merged = np.stack([gray, gray, gray], axis=2)
        elif channel_mapping is None and list(self.channel_names or [])[:3] == RGB_CHANNEL_NAMES[:3]:
            # A colour image (`_infer_channel_names`): its own colours. One scale for all three
            # channels (8-bit: /255; else a joint percentile stretch), so hues are kept.
            rgb = self.data[:, :, :3].astype(np.float32)
            if self.data.dtype == np.uint8:
                merged = rgb / 255.0
            else:
                merged = normalize_channel(rgb)
        else:
            # Multi-channel: merge channels
            merged = np.zeros((self.height, self.width, 3), dtype=np.float32)
            
            # Normalize all channels
            # Use percentile normalization for better training stability (robust to outliers)
            normalized_channels = []
            for c in range(self.num_channels):
                normalized_channels.append(normalize_channel(self.get_channel(c), method='percentile'))
            
            # Default mapping: bright-field -> gray, GFP -> green, DAPI -> blue
            if channel_mapping is None:
                channel_mapping = {}
                for i, name in enumerate(self.channel_names or []):
                    name_lower = name.lower()
                    if 'bright' in name_lower or 'field' in name_lower or i == 0:
                        channel_mapping[i] = 'gray'
                    elif 'gfp' in name_lower or i == 1:
                        channel_mapping[i] = 'green'
                    elif 'dapi' in name_lower or i == 2:
                        channel_mapping[i] = 'blue'
            
            # Apply mapping
            for ch_idx, color in channel_mapping.items():
                if ch_idx >= self.num_channels:
                    continue
                ch_norm = normalized_channels[ch_idx]
                
                if color == 'gray' or color == 'red':
                    merged[:, :, 0] = ch_norm  # Red
                    merged[:, :, 1] = ch_norm  # Green
                    merged[:, :, 2] = ch_norm  # Blue
                elif color == 'green':
                    merged[:, :, 1] = merged[:, :, 1] + ch_norm  # Add to green
                elif color == 'blue':
                    merged[:, :, 2] = merged[:, :, 2] + ch_norm  # Add to blue
            
            # Clip to [0, 1]
            merged = np.clip(merged, 0, 1)
        
        return merged
    
    def save(self, path: str, format: Optional[str] = None) -> str:
        """
        Save image to file
        For multi-channel, saves as TIFF to preserve channels
        For single-channel, saves in specified format (default: PNG)
        """
        if format is None:
            format = 'tiff' if self.is_multi_channel else 'png'
        
        if self.is_multi_channel and format.lower() in ('tif', 'tiff'):
            # Save as multi-channel TIFF
            # Convert (H, W, C) to (C, H, W) format for ImageJ compatibility
            # This ensures ImageJ correctly interprets channels (not as Z-stack)
            data_chw = np.moveaxis(self.data, -1, 0)  # Move channel dimension to first
            tifffile.imwrite(path, data_chw, imagej=True)
        else:
            # Save as single-channel or RGB image
            if self.is_single_channel:
                img_uint8 = self.to_uint8(0)
                Image.fromarray(img_uint8, mode='L').save(path)
            else:
                # Convert to RGB for display
                rgb = self.create_merged_rgb()
                rgb_uint8 = (rgb * 255).astype(np.uint8)
                Image.fromarray(rgb_uint8, mode='RGB').save(path)
        
        return path



