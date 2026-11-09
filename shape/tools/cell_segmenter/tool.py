from shape.tools.base import BaseTool
import colorsys
import numpy as np
import cv2
import os
import threading
import time
from PIL import Image
from uuid import uuid4
import hashlib
import random as _random
import matplotlib
# Pin the non-interactive Agg backend BEFORE importing pyplot: this tool only ever
# `savefig`s to disk (cell_overlay_*.png / cell_mask_viz_*.png via
# VisualizationConfig.save_professional_figure), never displays a window, so an
# auto-detected interactive backend is always wrong here. Newly surfaced (not caused) by
# this QC-fix round's own new co-located tests -- they are the first in this tool
# directory to reach `create_professional_figure` more than once in one pytest session,
# which on this Windows conda env intermittently makes matplotlib probe a broken Tk/Tcl
# install and raise TclError (~1/3 of combined runs; each test passes 100% in isolation).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from skimage.segmentation import relabel_sequential
from shape.utils.agent_io import VisualizationConfig
import traceback
import tifffile
from shape.utils.image_data import ImageData
from shape.utils.image_processor import ImageProcessor
from shape.utils.segmentation_qc import (
    compute_flow_error,
    evaluate_segmentation_visual,
    normalize_for_overlay,
)
from shape.utils.device import cellpose_precision_kwargs, resolve_torch_device


class _LazyCellpose:
    """A module-level stand-in for `cellpose.models` that defers the real import until used.
    Importing cellpose pulls torch + torchvision (~4s), which a worker in shared-server mode
    ($SHAPE_SAM_SERVER_URL) never needs -- it delegates every eval to the remote GPU and holds
    no local model. Deferring it lets a spawned remote-SAM segment worker cold-start ~4s faster
    (the import storm that stalled a large worker pool).

    `CellposeModel` is a real instance attribute (a lazy constructor), so `patch(
    "...tool.models.CellposeModel", stub)` sets/restores it SYMMETRICALLY under any test order
    (no __getattr__ hole that pytest-randomly's patch teardown could trip over), and the stub
    is read without ever importing cellpose."""
    def __init__(self):
        self.CellposeModel = self._construct_cellpose_model

    @staticmethod
    def _construct_cellpose_model(*args, **kwargs):
        from cellpose import models as _m
        return _m.CellposeModel(*args, **kwargs)

    def __getattr__(self, name):               # any OTHER cellpose.models attribute
        from cellpose import models as _m
        return getattr(_m, name)


models = _LazyCellpose()


def _save_display_png(array_uint8, path):
    """Cheap PNG write via PIL -- no matplotlib. ~15x faster than save_professional_figure
    on large images. Used in benchmark 'fast figures' mode ($SHAPE_FAST_FIGURES=1): the
    per-attempt overlay/mask PNGs are for the vision judge + human inspection, which need
    the pixels, not a 300dpi titled figure. Default (env unset) keeps the professional figure."""
    from PIL import Image
    Image.fromarray(np.asarray(array_uint8)).save(path)


# colorsys.hsv_to_rgb's six hue sectors: which of its terms (v, p, q, t) r, g and b are
_HSV_SECTORS = (("v", "t", "p"), ("q", "v", "p"), ("p", "v", "t"), ("p", "q", "v"), ("t", "p", "v"), ("v", "p", "q"))


def _mask_overlay(img, masks):
    """Cellpose's `plot.mask_overlay(img, masks)` (without `colors`) as a colour lookup table: the same image.

    Cellpose scans the whole image once per label and converts every pixel to RGB in Python
    (`utils.hsv_to_rgb`, an `np.vectorize` of `colorsys.hsv_to_rgb`). Here: the same grey value (V) and the
    same random hue per label, drawn as Cellpose draws it (one `np.random.permutation`, so the random state
    moves as there); each label's three colour factors from colorsys's own arithmetic (saturation 1 inside a
    mask, 0 outside); each channel = V x its label's factor, in float64 as colorsys computes it, then cast to
    the type `np.vectorize` gives that channel (set by colorsys's result on the first pixel)."""
    if img.ndim > 2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)
    value = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1).astype(np.float32)  # Cellpose's float32 V
    hues = np.linspace(0, 1, masks.max() + 1)[np.random.permutation(masks.max())]
    n = int(masks.max())
    hue = np.zeros(n + 1)
    hue[1:] = hues[:n].astype(np.float32)        # label k's hue, as Cellpose's float32 H holds it
    factors = np.ones((n + 1, 3))                # label 0 (saturation 0): colorsys returns (v, v, v)
    if n:
        s = 1.0
        sector = (hue[1:] * 6.0).astype(np.int64)
        f = hue[1:] * 6.0 - sector
        terms = {"v": np.ones(n), "p": np.full(n, 1.0 - s), "q": 1.0 - s * f, "t": 1.0 - s * (1.0 - f)}
        for k, names in enumerate(_HSV_SECTORS):
            rows = np.flatnonzero(sector % 6 == k)
            for c, name in enumerate(names):
                factors[rows + 1, c] = terms[name][rows]
    first = int(masks.flat[0])
    types = [np.asarray(x).dtype for x in
             colorsys.hsv_to_rgb(np.float32(hue[first]), np.float32(1.0 if first else 0.0), value.flat[0])]
    common = np.result_type(*types)             # np.stack of the three channels
    v = value.astype(np.float64)
    rgb = np.empty(value.shape + (3,), dtype=np.uint8)
    for c in range(3):
        channel = (v * factors[:, c][masks]).astype(types[c]).astype(common)
        rgb[..., c] = (channel * 255).astype(np.uint8)
    return rgb


def _remote_eval(url, img_array, eval_kwargs):
    """Delegate one Cellpose-SAM `eval` to a shared inference server (see sam_server.py).

    Sends {img, kwargs, with_flow_error} and returns (masks, flows, flow_error). The server
    computes the flow-error QC on its GPU (re-running cellpose dynamics, ~0.5s on GPU vs ~34s
    on this CPU client -- the dominant remote-mode cost), so the caller does NOT recompute it.
    flow_error is None if the server (an older one) did not return it, and the caller then
    recomputes locally, unchanged. Lets many concurrent workers share one GPU-resident model."""
    import pickle
    import urllib.request

    body = pickle.dumps({"img": img_array, "kwargs": eval_kwargs, "with_flow_error": True})
    req = urllib.request.Request(url.rstrip("/") + "/eval", data=body,
                                 headers={"Content-Type": "application/octet-stream"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = pickle.loads(resp.read())
    if isinstance(out, dict) and out.get("error"):
        raise RuntimeError(f"SAM server error: {out['error']}")
    return out["masks"], out["flows"], out.get("flow_error")


class Cell_Segmenter_Tool(BaseTool):
    """
    Cell segmentation tool for phase-contrast cell images using Cellpose CPSAM model.
    This tool extends the system to handle general cell types (not just nuclei) in phase-contrast microscopy.
    """

    # Attempt counter kept at CLASS level so it SURVIVES the per-retune tool re-instantiation
    # (each VLRR retune builds a fresh tool); on-disk tags then increment a1, a2, a3 ... across
    # attempts instead of every attempt overwriting a1.
    _attempt_counts: dict = {}
    _attempt_lock = threading.Lock()     # calls of one image may run in concurrent threads (wave mode)

    def __init__(self, model_path=None, model_name="cpsam"):
        super().__init__(
            tool_name="Cell_Segmenter_Tool",
            tool_description="Segments whole cells in phase-contrast microscopy images using Cellpose CPSAM model. Designed for general cell types, not just nuclei. Provides cell counting and visualization.",
            tool_version="1.0.0",
            input_types={
                "image": "str - Path to the input image (supports .tif, .png, .jpg formats).",
                "diameter": "float - Expected cell diameter in pixels (default: None, auto-detect).",
                "flow_threshold": "float - Flow threshold for cell detection (default: 0.4).",
                "cellprob_threshold": "float - Cell probability threshold (default: 0).",
                "model_type": "str - Model type to use: 'cpsam' for CPSAM model (default), or 'cyto' for general cellpose model."
            },
            output_type="dict - Contains segmentation results, cell count, and visualization paths.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/cells.tif")',
                    "description": "Segment cells in a phase-contrast microscopy image with default parameters."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/cells.tif", diameter=30, flow_threshold=0.5, model_type="cpsam")',
                    "description": "Segment cells with custom diameter and flow threshold using CPSAM model."
                }
            ],
            user_metadata={
                "limitation": "Requires GPU for optimal performance. May struggle with very dense cell populations or poor quality images. Model download required on first use. Designed for phase-contrast images of whole cells.",
                "best_practice": "Adjust diameter parameter based on your cell type and image resolution. For phase-contrast images, use 'cpsam' model type."
            },
            output_dir="output_visualizations"
        )

        self.model_name = model_name

        # Shared-server mode (checked FIRST, before ANY CUDA probe): when $SHAPE_SAM_SERVER_URL
        # is set, this process delegates ALL inference to the one GPU-resident Cellpose-SAM
        # server (sam_server.py) and needs NO GPU itself. Critically we must NOT call
        # resolve_torch_device()/torch.cuda.* here -- the first CUDA call allocates a ~600MB
        # CUDA primary context PER worker for nothing (16 workers x 612MB observed). Set the
        # device to CPU without touching CUDA. Default (env unset) path below is unchanged.
        # SHAPE_SAM_SERVER_URL may hold ONE url or a comma-separated list of server
        # processes; a list is load-balanced per eval (a single process plateaus at ~2x
        # serial on the GIL, so throughput scales with independent server PROCESSES).
        _raw_sam = os.environ.get("SHAPE_SAM_SERVER_URL")
        self._sam_server_urls = [u.strip() for u in _raw_sam.split(",") if u.strip()] if _raw_sam else []
        self._sam_server_url = self._sam_server_urls[0] if self._sam_server_urls else None
        if self._sam_server_url:
            self.device = torch.device("cpu")
            self.model = None
            self.model_type = "custom" if model_path else "cpsam"
            _shown = self._sam_server_url + (f" (+{len(self._sam_server_urls)-1} more, load-balanced)"
                                             if len(self._sam_server_urls) > 1 else "")
            print(f"Cell_Segmenter_Tool: remote SAM server mode -> {_shown} "
                  "(no local model, no CUDA context)")
            return

        # Enable GPU if available (fall back to CPU gracefully)
        self.device, device_warning = resolve_torch_device()
        print(f"Cell_Segmenter_Tool: Using device: {self.device}")
        if device_warning:
            print(f"Cell_Segmenter_Tool: {device_warning}")

        try:
            if model_path:
                self.model = models.CellposeModel(
                    device=self.device,
                    pretrained_model=model_path,
                    **cellpose_precision_kwargs(self.device),
                )
                self.model_type = "custom"
            else:
                # Cellpose manages the official CPSAM download and cache.
                self.model = models.CellposeModel(
                    device=self.device,
                    pretrained_model="cpsam",
                    **cellpose_precision_kwargs(self.device),
                )
                self.model_type = "cpsam"
                
            if self.model is None:
                raise ValueError("Failed to initialize CellposeModel")
            print(f"Cell_Segmenter_Tool: Model initialized successfully (type: {self.model_type})")
        except Exception as e:
            # Retry on CPU if GPU path fails (e.g., NVML issues)
            try:
                print(f"Cell_Segmenter_Tool: GPU init failed ({e}), retrying on CPU")
                self.device = torch.device('cpu')
                if model_path:
                    self.model = models.CellposeModel(
                        device=self.device,
                        pretrained_model=model_path,
                        **cellpose_precision_kwargs(self.device),
                    )
                    self.model_type = "custom"
                else:
                    self.model = models.CellposeModel(
                        device=self.device,
                        pretrained_model="cpsam",
                        **cellpose_precision_kwargs(self.device),
                    )
                    self.model_type = "cpsam"
                print("Cell_Segmenter_Tool: CPU model initialized successfully")
            except Exception as cpu_e:
                print(f"Cell_Segmenter_Tool: Error initializing model even on CPU: {cpu_e}")
                raise

    def _segment_channel(self, img_array, eval_kwargs):
        """Run Cellpose on one channel array and compute its QC.

        Returns (mask, flows, flow_error, overlay, overlay_qc). Used both for the
        primary segmentation and, when that segments poorly on a multi-channel image,
        to try the other channels and keep the best. This is the ONE call site in
        this file that invokes `self.model.eval`, so it is the single choke point
        for every caller: the primary run (multi-channel default, single-channel
        default, and the legacy-loader fallback all build `img` and reach
        `self.model.eval` only through this method via `execute()`'s primary call),
        and the per-channel recovery retry loop.

        Task 073 QC blocker fix (loud refuse, not silent conversion): refuse
        non-finite (NaN/Inf) input rather than handing it to Cellpose. Cellpose
        4.0.4's own `normalize_img` silently zeros an ENTIRE channel when it
        contains even one non-finite pixel (verified: a single NaN in one channel
        of a 3-channel array collapses that channel's normalized output to a
        single value, with no error/warning, while an untouched sibling channel
        keeps its full range) -- silently wrong segmentation, not a crash. The
        upstream image handling should preserve source NaN/Inf pixels rather than
        fabricating values, so this is the point that must reject them outright.
        """
        if not np.isfinite(img_array).all():
            non_finite = ~np.isfinite(img_array)
            count = int(non_finite.sum())
            if img_array.ndim >= 3:
                bad_channels = sorted(int(c) for c in np.flatnonzero(
                    non_finite.any(axis=tuple(range(img_array.ndim - 1)))))
            else:
                bad_channels = [0]
            raise ValueError(
                f"Cell_Segmenter_Tool: refusing to segment -- {count} non-finite "
                f"(NaN/Inf) pixel(s) found in channel(s) {bad_channels} of the "
                "array about to be handed to Cellpose. Non-finite input pixels "
                "are unsupported: Cellpose's normalization silently collapses an "
                "entire channel to a single value when it contains even one "
                "non-finite pixel, producing silently wrong segmentation. Supply "
                "finite data, or handle/mask non-finite pixels upstream, before "
                "calling this tool."
            )
        remote_flow_error = _UNSET = object()
        if getattr(self, "_sam_server_url", None):
            urls = getattr(self, "_sam_server_urls", None) or [self._sam_server_url]
            url = urls[0] if len(urls) == 1 else _random.choice(urls)   # load-balance across procs
            masks, flows, remote_flow_error = _remote_eval(url, img_array, eval_kwargs)
        else:
            masks, flows, _ = self.model.eval([img_array], **eval_kwargs)
        mask = masks[0]
        # flow-error QC re-runs cellpose dynamics (_extend_centers_gpu): ~0.5s on a GPU but ~34s
        # on a CPU. In remote mode the SERVER already computed it on its GPU, so use that value
        # and never recompute on this (CPU) node; recompute locally only for a local model, or if
        # an older server returned no flow_error. (device: a CPU fallback must not recompute on a
        # GPU it could not open.)
        if remote_flow_error is not _UNSET and remote_flow_error is not None:
            flow_error_val = remote_flow_error
        else:
            flow_error_val = compute_flow_error(mask, flows, device=getattr(self, "device", None))
        overlay = _mask_overlay(normalize_for_overlay(img_array), mask)
        overlay_qc = evaluate_segmentation_visual(mask, overlay)
        return mask, flows, flow_error_val, overlay, overlay_qc

    def _execute_native(self, image, diameter=30, flow_threshold=0.4, cellprob_threshold=0, min_size=15, niter=None, bsize=256, augment=False, resample=True, tile_overlap=0.1, model_type="cpsam", query_cache_dir=None, image_id=None, color_class_filter=None):
        """
        Execute cell segmentation on the input phase-contrast image.

        Args:
            image (str): Path to the input image
            diameter (float): Expected cell diameter in pixels
            flow_threshold (float): Flow threshold for cell detection
            cellprob_threshold (float): Cell probability threshold
            model_type (str): Model type to use ('cpsam' or 'cyto')
            query_cache_dir (str): Directory to save outputs
            image_id (str): Optional image identifier for consistent file naming

        Returns:
            dict: Segmentation results with cell count and visualization paths
        """
        started = time.time()
        try:
            if (not getattr(self, "_sam_server_url", None)
                    and model_type != self.model_type and model_type == "cpsam"):
                self.model = models.CellposeModel(
                    device=self.device,
                    pretrained_model="cpsam",
                    **cellpose_precision_kwargs(self.device),
                )
                self.model_type = "cpsam"
            
            # Normalize path for cross-platform compatibility
            image_path = os.path.normpath(image) if isinstance(image, str) else str(image)
            
            # Extract image identifier: use provided image_id, or extract from filename
            if image_id:
                image_identifier = image_id
            else:
                # Fallback: extract from filename
                filename = os.path.basename(image_path)
                image_identifier = os.path.splitext(filename)[0]
                # Sanitize: keep only alphanumeric and common separators, limit length
                image_identifier = "".join(c for c in image_identifier if c.isalnum() or c in ('-', '_'))[:50]
            
            # Check if file exists
            if not os.path.exists(image_path):
                return {
                    "error": f"Image file not found: {image_path}",
                    "summary": "Image file does not exist"
                }
            
            # Load image using unified abstraction layer
            try:
                img_data = ImageProcessor.load_image(image_path)
                print(f"Loaded image: {img_data.shape}, channels: {img_data.num_channels}")

                # Task 073 R2 fix (QC finding 2, FIX round): a multi-channel image feeds ALL
                # its channels to Cellpose-SAM BY DEFAULT -- Cellpose-SAM consumes up to 3
                # channels itself and truncates to its first 3 if more are given (BRIEF S0
                # finding 3). Convert to float32 format required by Cellpose either way.
                if img_data.is_multi_channel:
                    img = img_data.data.astype(np.float32)
                    print(f"Detected multi-channel image with {img_data.num_channels} channels "
                          f"({img_data.channel_names}). Feeding all channels to Cellpose-SAM "
                          "(it uses up to its first 3).")
                else:
                    img = img_data.to_segmentation_input(channel_idx=0)
            except Exception as load_error:
                # Fallback to legacy loading for backward compatibility
                print(f"Warning: Failed to load with ImageProcessor: {load_error}, trying legacy method")
                image_path_lower = image_path.lower()
                is_tiff = image_path_lower.endswith('.tif') or image_path_lower.endswith('.tiff')

                if is_tiff:
                    try:
                        img_full = tifffile.imread(image_path)
                        if img_full.ndim == 2:
                            img = img_full.astype(np.float32)
                        elif img_full.ndim == 3:
                            img = img_full[:, :, 0] if img_full.shape[2] <= 4 else img_full[0, :, :]
                            img = img.astype(np.float32)
                        elif img_full.ndim == 4:
                            img = img_full[0, :, :, 0] if img_full.shape[3] <= 4 else img_full[0, 0, :, :]
                            img = img.astype(np.float32)
                        else:
                            img = img_full.astype(np.float32)
                    except Exception as tiff_error:
                        print(f"Warning: Failed to load TIFF: {tiff_error}, trying cv2")
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = img.astype(np.float32)
                else:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = img.astype(np.float32)

                if img is None:
                    # Try alternative: use PIL to load and convert
                    try:
                        pil_img = Image.open(image_path)
                        if pil_img.mode != 'L':
                            pil_img = pil_img.convert('L')
                        img = np.array(pil_img, dtype=np.float32)
                        print(f"Loaded image using PIL: {image_path}")
                    except Exception as pil_error:
                        return {
                            "error": f"Failed to load image: {image_path}. All loading methods failed: {str(pil_error)}",
                            "summary": "Image loading failed with all methods"
                        }
            
            # No auto-applied preprocessing: the loaded image is handed directly to
            # Cellpose-SAM, preserving the source channels and values.
            img = img.astype(np.float32)
            
            # Handle diameter parameter: convert 'auto' string to None, ensure None or numeric
            if diameter is None or (isinstance(diameter, str) and diameter.lower() == 'auto'):
                diameter = None
            elif isinstance(diameter, str):
                # Try to convert string to float if it's a numeric string
                try:
                    diameter = float(diameter)
                except ValueError:
                    print(f"Warning: Invalid diameter value '{diameter}', using None for auto-detection")
                    diameter = None
            elif not isinstance(diameter, (int, float)):
                print(f"Warning: Unexpected diameter type {type(diameter)}, using None for auto-detection")
                diameter = None
            
            # Run segmentation
            # For phase-contrast images, use channels [0, 0] (grayscale)
            try:
                min_size_int = int(min_size) if min_size is not None else 15
            except (TypeError, ValueError):
                min_size_int = 15
            # niter: None -> Cellpose auto; bsize: SAM processing block size.
            try:
                niter_val = int(niter) if niter not in (None, "", "auto") else None
            except (TypeError, ValueError):
                niter_val = None
            try:
                bsize_val = int(bsize) if bsize else 256
            except (TypeError, ValueError):
                bsize_val = 256
            # second-tier recall knobs: augment = test-time augmentation (4x, recovers
            # faint cells); resample = run dynamics at full resolution; tile_overlap.
            augment_val = bool(augment) if not isinstance(augment, str) else augment.lower() == "true"
            resample_val = bool(resample) if not isinstance(resample, str) else resample.lower() != "false"
            try:
                tile_overlap_val = float(tile_overlap)
            except (TypeError, ValueError):
                tile_overlap_val = 0.1
            eval_kwargs = dict(
                diameter=diameter,
                channels=[0, 0],  # Grayscale image
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size_int,
                niter=niter_val,
                bsize=bsize_val,
                augment=augment_val,
                resample=resample_val,
                tile_overlap=tile_overlap_val,
            )
            # Primary segmentation. flow_error is the VLRR gate: high flow error means
            # Cellpose-SAM does not fit this image, so the verifier escalates rather
            # than retunes.
            mask, flows, flow_error_val, overlay, overlay_qc = self._segment_channel(img, eval_kwargs)

            # Multi-channel fluorescence: if the primary run segmented poorly (failed
            # visual QC -> 0 cells / black overlay), try EVERY individual channel and keep
            # whichever finds the most cells -- the signal may live in one channel that is
            # diluted/outweighed when all channels are fed to Cellpose together.
            #
            # Task 073 R2 fix (QC finding 2, FIX round 2): the primary run above is the FULL
            # (H,W,C) array (see the `img_data.is_multi_channel` branch above), not any one
            # channel -- so no channel has been individually segmented yet, and EVERY channel
            # (including `informative_channel_idx`) gets its own fair single-channel retry
            # here. Round 1's loop still special-cased `primary_idx` as "already tried" (true
            # only when the primary run WAS that one channel, pre-round-1) and skipped it,
            # silently disabling the informative-channel fallback for every multi-channel
            # input and mislabelling the failed all-channel result as if it came from that
            # channel (independent QC review, FIX round 2 F2). `segmented_channel` now stays
            # `None` unless a single channel's retry genuinely wins.
            segmented_channel = None
            if (not overlay_qc["passed"]) and ("img_data" in locals()) \
                    and img_data is not None and getattr(img_data, "num_channels", 1) > 1:
                best = (overlay_qc["cell_count"], mask, flows, flow_error_val,
                        overlay, overlay_qc, None)
                for c in range(img_data.num_channels):
                    ch = img_data.get_channel(c).astype(np.float32)
                    if float(np.std(ch)) < 1e-6:
                        continue  # empty channel -- nothing to segment
                    m, fl, fe, ov, oq = self._segment_channel(ch, eval_kwargs)
                    if oq["cell_count"] > best[0]:
                        best = (oq["cell_count"], m, fl, fe, ov, oq, c)
                cells, mask, flows, flow_error_val, overlay, overlay_qc, segmented_channel = best
                if segmented_channel is not None:
                    img = img_data.get_channel(segmented_channel).astype(np.float32)
                    print(f"  Multi-channel: primary (all-channel) result segmented poorly; "
                          f"using channel {segmented_channel} ({cells} cells)")

            # Verifier-invoked COLOR/CLASS FILTER (a retune action): drop a non-target color
            # population (e.g. RBCs in a stained smear) from the FINAL mask, then rebuild the
            # overlay + QC from the filtered mask so the vision judge + scoring see the filtered
            # result. `color_class_filter` in {keep_high_br, keep_low_br}; None/"none" is a no-op.
            # The VERIFIER decides the action (it sees the image); the tool only applies the
            # deterministic Otsu cut here (exec/tool has the pixels; the executor's LLM does not).
            color_filtered = False
            prefilter_mask = None      # the mask before the filter, kept only when the filter removed masks
            if color_class_filter and str(color_class_filter).lower() not in ("none", "false"):
                try:
                    from shape.tools.cell_segmenter import color_class_filter as _ccf
                    from skimage.io import imread as _rgb_imread
                    _rgb = _rgb_imread(image_path)   # RGB (NOT cv2/BGR) so B-R is correct
                    _filtered = _ccf.apply(_rgb, mask, str(color_class_filter))
                    if _filtered is not None and int(np.asarray(_filtered).max()) != 0:
                        if not np.array_equal(_filtered, mask):
                            prefilter_mask = mask
                        mask = _filtered
                        overlay = _mask_overlay(normalize_for_overlay(img), mask)
                        overlay_qc = evaluate_segmentation_visual(mask, overlay)
                        color_filtered = True
                        print(f"  color_class_filter={color_class_filter}: kept "
                              f"{len(np.unique(mask)) - 1} masks")
                except Exception as _cf_e:
                    print(f"  color_class_filter apply failed ({_cf_e}); keeping unfiltered mask")

            # Setup output directory using centralized configuration
            output_dir = VisualizationConfig.get_output_dir(query_cache_dir)

            # Per-attempt file tag: NUMBER each segmentation of this image (a1, a2, ...)
            # so the on-disk files say WHICH attempt produced them, instead of an opaque
            # param hash. Each re-segmentation thus writes its own files (independently
            # inspectable, in order); the verification record maps each attempt's params
            # to these paths for tracing. Keyed by (output_dir, image) so the baseline
            # (separate cache) and each image's VLRR retunes count from 1 independently.
            # The `image_id`/`image_identifier` result fields keep the plain id for
            # downstream matching -- only the on-disk filenames carry the tag.
            _sig_src = repr((
                round(float(diameter), 3) if isinstance(diameter, (int, float)) else diameter,
                round(float(cellprob_threshold), 3) if cellprob_threshold is not None else None,
                round(float(flow_threshold), 3) if flow_threshold is not None else None,
                int(min_size_int), niter_val, bool(augment_val), bool(resample_val),
                round(float(tile_overlap_val), 3),
            ))
            param_sig = hashlib.md5(_sig_src.encode()).hexdigest()[:8]
            attempt_key = (str(output_dir), str(image_identifier))
            counts = type(self)._attempt_counts          # mutate the shared CLASS-level counter
            with type(self)._attempt_lock:
                counts[attempt_key] = counts.get(attempt_key, 0) + 1
                attempt = counts[attempt_key]
            file_tag = f"{image_identifier}_a{attempt}"

            # Fast-figures mode ($SHAPE_FAST_FIGURES=1): render the per-attempt overlay/mask
            # PNGs with a cheap PIL write instead of a 300dpi matplotlib figure. On large
            # microscopy images the professional figure is ~1.8s each (2 per attempt) and is
            # CPU-bound -- the dominant per-image cost + concurrency amplifier at scale. The
            # vision judge reads overlay_path's PIXELS, so a plain PNG is equivalent for it.
            # matplotlib is not thread-safe: a call in a worker thread (wave mode) always uses it.
            fast_figures = (os.environ.get("SHAPE_FAST_FIGURES") == "1"
                            or threading.current_thread() is not threading.main_thread())

            # Save overlay visualization using the per-attempt tag (the vision judge reads this)
            output_path = os.path.join(output_dir, f"cell_overlay_{file_tag}.png")
            if fast_figures:
                _save_display_png(np.asarray(overlay).astype(np.uint8), output_path)
            else:
                fig, ax = VisualizationConfig.create_professional_figure(figsize=(12, 8))
                # Ensure overlay has same brightness as original image
                overlay_normalized = overlay.astype(np.float32) / 255.0
                ax.imshow(overlay_normalized, vmin=0, vmax=1)
                ax.axis('off')
                VisualizationConfig.apply_professional_styling(
                    ax, title=f"Cell Segmentation - {len(np.unique(mask))-1} cells detected"
                )
                VisualizationConfig.save_professional_figure(fig, output_path)
                plt.close(fig)
            
            # Count cells (unique mask values, excluding background 0)
            n_cells = len(np.unique(mask)) - 1 if mask is not None else 0
            
            # Save the mask as .npy (the label array downstream tools load). The .tif
            # copy is no longer written: the .npy holds the data and the overlay +
            # mask-viz PNGs below cover visual display, so the .tif was redundant.
            mask_npy_path = os.path.join(output_dir, f"cell_mask_{file_tag}.npy")
            if not os.path.isabs(mask_npy_path):
                if query_cache_dir and os.path.isabs(query_cache_dir):
                    mask_npy_path = os.path.join(query_cache_dir, "output_visualizations", f"cell_mask_{file_tag}.npy")
                else:
                    mask_npy_path = os.path.abspath(mask_npy_path)
            mask_npy_path = os.path.normpath(mask_npy_path)  # Normalize path separators

            # Cellpose masks are integer labels (0=background, 1-N for N cells). The color/class filter
            # keeps the original ids of the masks it keeps: the saved mask is relabelled 1-N in the same
            # order, so the per-mask loops of every reader scale with the kept masks (the overlay and the
            # mask display are drawn from the filtered mask itself: the same pixels as before).
            np.save(mask_npy_path, relabel_sequential(mask)[0] if color_filtered else mask)
            if not os.path.exists(mask_npy_path):
                raise FileNotFoundError(f"Failed to save mask NPY file: {mask_npy_path}")
            # When the filter removed masks, the mask before it too (as segmented): the agentic verifier measures a
            # filtered call's under-detection on it, the removed masks counting as covered, and the split the filter
            # made. Named outside the `cell_mask_*_a<N>.npy` pattern the per-attempt readers glob for.
            prefilter_npy_path = None
            if prefilter_mask is not None:
                prefilter_npy_path = os.path.join(os.path.dirname(mask_npy_path),
                                                  f"cell_prefilter_mask_{file_tag}.npy")
                np.save(prefilter_npy_path, prefilter_mask)

            # Also save a display-only mask visualization (not read by the judge)
            viz_mask_path = os.path.join(output_dir, f"cell_mask_viz_{file_tag}.png")
            if fast_figures:
                from skimage.color import label2rgb
                _save_display_png(
                    (label2rgb(mask, bg_label=0) * 255).astype(np.uint8), viz_mask_path)
            else:
                fig, ax = VisualizationConfig.create_professional_figure(figsize=(12, 8))
                ax.imshow(mask, cmap='tab20')
                ax.axis('off')
                VisualizationConfig.apply_professional_styling(
                    ax, title=f"Cell Masks - {n_cells} cells"
                )
                VisualizationConfig.save_professional_figure(fig, viz_mask_path)
                plt.close(fig)
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            result = {
                "summary": f"{n_cells} cells identified and segmented successfully.",
                "status": "success" if overlay_qc["passed"] else "failed",
                "cell_count": n_cells,
                "visual_outputs": [output_path, viz_mask_path],  # overlay + mask-viz PNGs
                "deliverables": [output_path],  # Add overlay to deliverables for better collection
                "mask_path": mask_npy_path,  # .npy format for morphological analysis (for downstream tools)
                "overlay_path": output_path,  # Explicit overlay_path key for collection
                "output_path": output_path,  # Explicit output_path key for collection
                "image_id": image_identifier,  # Add image_id for matching in downstream tools
                "image_identifier": image_identifier,  # Alias for compatibility
                "attempt": attempt,  # which segmentation attempt of this image (1, 2, ...)
                "segmented_channel": segmented_channel,  # set if a non-default channel won the multi-channel search
                "param_sig": param_sig,  # signature of the tuning params for this attempt
                "model_used": f"CellposeModel ({self.model_type})",
                "flow_error": flow_error_val,  # VLRR segmentation gate metric (mean Cellpose flow error)
                "overlay_qc": overlay_qc,
                "seconds": round(time.time() - started, 1),  # this call's run time (the agentic loop shows it)
                "parameters": {
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "min_size": min_size,
                    "niter": niter_val,
                    "bsize": bsize_val,
                    "augment": augment_val,
                    "resample": resample_val,
                    "tile_overlap": tile_overlap_val,
                    "model_type": self.model_type,
                    # record whether the color/class filter was applied this pass, so the
                    # verifier can recognize an already-class-filtered mask and accept it
                    # instead of re-litigating the removed non-target color class.
                    "color_class_filter": (str(color_class_filter)
                                           if color_class_filter and str(color_class_filter).lower()
                                           not in ("none", "false") else None),
                }
            }
            if prefilter_npy_path:
                result["prefilter_mask_path"] = prefilter_npy_path   # only when the color/class filter removed masks
            if not overlay_qc["passed"] and n_cells == 0 and getattr(self, "allow_empty_masks", False):
                result["status"] = "no_detection"
                result["empty_mask_policy"] = "record_no_detection_not_biological_absence"
                result["qc_warning"] = "Zero detections; original visual QC remains failed."
            elif (not overlay_qc["passed"] and not overlay_qc["overlay_is_black"]
                  and overlay_qc["foreground_coverage"] <= 0.0001 and getattr(self, "allow_empty_masks", False)):
                # A near-empty field: a few debris-sized detections below the visual QC's coverage floor. Under the
                # same opt-in policy as zero detections it is delivered as it is, the failed visual QC kept.
                result["status"] = "low_coverage_detection"
                result["empty_mask_policy"] = "record_low_coverage_detection_not_biological_absence"
                result["qc_warning"] = (f"{n_cells} detection(s) covering {overlay_qc['foreground_coverage']:.6f} of "
                                        "the image; original visual QC remains failed.")
            elif not overlay_qc["passed"]:
                result["error"] = (
                    "Segmentation output failed visual QC: "
                    f"cells={n_cells}, overlay_is_black={overlay_qc['overlay_is_black']}, "
                    f"foreground_coverage={overlay_qc['foreground_coverage']:.6f}"
                )
                result["summary"] = result["error"]
            return result
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            print(f"Cell_Segmenter_Tool: Error in cell segmentation: {error_msg}")
            print(f"Cell_Segmenter_Tool: Traceback: {error_traceback}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "error": f"Error during cell segmentation: {error_msg}",
                "summary": "Failed to process image",
                "error_details": {
                    "image_path": image_path if 'image_path' in locals() else str(image),
                    "diameter": diameter if 'diameter' in locals() else "unknown",
                    "flow_threshold": flow_threshold if 'flow_threshold' in locals() else "unknown",
                    "cellprob_threshold": cellprob_threshold if 'cellprob_threshold' in locals() else "unknown"
                }
            }

    def set_trial_cache(self, cache):
        """Opt in to durable native trial reuse; reasoning and delivery remain uncached."""
        from shape.tools.cell_segmenter.trial_cache import SegmentationTrialCache
        if cache is not None and not isinstance(cache, SegmentationTrialCache):
            raise TypeError('expected a SegmentationTrialCache or None')
        self._native_trial_cache = cache

    def execute(self, image, diameter=30, flow_threshold=0.4, cellprob_threshold=0, min_size=15, niter=None, bsize=256, augment=False, resample=True, tile_overlap=0.1, model_type="cpsam", query_cache_dir=None, image_id=None, color_class_filter=None):
        arguments = dict(image=image, diameter=diameter, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, min_size=min_size, niter=niter, bsize=bsize,
            augment=augment, resample=resample, tile_overlap=tile_overlap, model_type=model_type,
            query_cache_dir=query_cache_dir, image_id=image_id, color_class_filter=color_class_filter)
        cache = getattr(self, '_native_trial_cache', None)
        # Array-backed/anonymous calls retain their original behavior. The file
        # cache requires an explicit image ID to attest output-label identity.
        if cache is None or image_id is None or not isinstance(image, (str, os.PathLike)):
            return self._execute_native(**arguments)
        from pathlib import Path
        from shape.tools.cell_segmenter.trial_cache import effective_parameters
        parameters = effective_parameters(self._execute_native, arguments)
        def compute(folder):
            from contextlib import nullcontext
            seed = getattr(self, '_trial_cache_seed', None)
            if seed is not None:
                from shape.tools.cell_segmenter.cache_runtime import native_rng
                scope = native_rng(seed, self.device)
            else:
                scope = nullcontext()
            with scope:
                return self._execute_native(**{**arguments, 'query_cache_dir': str(folder)})
        return cache.get_or_compute(Path(image), str(image_id), parameters, compute,
            context=dict(caller_output_dir=str(query_cache_dir) if query_cache_dir is not None else None))

    execute.__doc__ = _execute_native.__doc__

    def get_metadata(self):
        """Returns the metadata for the Cell_Segmenter_Tool."""
        metadata = super().get_metadata()
        metadata["device"] = str(self.device)
        metadata["model_loaded"] = self.model is not None
        metadata["model_type"] = self.model_type if hasattr(self, 'model_type') else "unknown"
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd shape/tools/cell_segmenter
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Cell_Segmenter_Tool
    tool = Cell_Segmenter_Tool()
    tool.set_custom_output_dir("cell_outputs")

    # Get tool metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(metadata)

    # Construct the full path to the image using the script's directory
    relative_image_path = "../../../case studies/fibroblast.png"
    image_path = os.path.join(script_dir, relative_image_path)

    # Execute the tool
    try:
        execution = tool.execute(image=image_path)
        print("\nExecution Result:")
        print(f"Summary: {execution.get('summary', 'No summary')}")
        print(f"Cell Count: {execution.get('cell_count', 'Unknown')}")
        print(f"Visual Outputs: {execution.get('visual_outputs', [])}")
        if 'error' in execution:
            print(f"Error: {execution['error']}")
    except Exception as e:
        print(f"Execution failed: {e}")

    print("Done!")
