from shape.tools.base import BaseTool
from cellpose import models, plot
import numpy as np
import cv2
import os
import re
from PIL import Image
from uuid import uuid4
import matplotlib
# Pin the non-interactive Agg backend BEFORE importing pyplot -- same fix as
# `shape/tools/cell_segmenter/tool.py` (task 057 Group B): this tool only ever
# `savefig`s to disk (organoid_overlay_*.png via VisualizationConfig), never
# displays a window, so an auto-detected interactive backend is always wrong here.
# Surfaced by this task's new co-located tests -- the first in this tool directory
# to reach `create_professional_figure` more than once in one pytest session,
# which on a broken Tk/Tcl install raised TclError (verified: 1st call OK, 2nd
# call in the same session failed probing Tk).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from shape.utils.agent_io import VisualizationConfig
import tifffile
import tempfile
from skimage import io as skimage_io
from shape.utils.image_data import ImageData
from shape.utils.image_processor import ImageProcessor
from shape.utils.segmentation_qc import (
    compute_flow_error,
    evaluate_segmentation_visual,
    to_display_uint8,
)
from shape.utils.device import cellpose_precision_kwargs, resolve_torch_device


ZERO_RESCUE_FOREGROUND_COVERAGE_MAX = 1e-3


def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _adaptive_rescue_requested(value):
    if not _coerce_bool(value, True):
        return False
    env = os.getenv("SHAPE_ORGANOID_ADAPTIVE_RESCUE", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    if env in {"1", "true", "yes", "on"}:
        return True
    return True


def _zero_or_near_zero_failure(result):
    if not isinstance(result, dict):
        return False
    overlay_qc = result.get("overlay_qc")
    if isinstance(overlay_qc, dict):
        try:
            foreground_coverage = float(overlay_qc.get("foreground_coverage", 1.0))
        except (TypeError, ValueError):
            foreground_coverage = 1.0
        try:
            organoid_count = int(result.get("organoid_count", result.get("cell_count", -1)))
        except (TypeError, ValueError):
            organoid_count = -1
        return (
            result.get("status") == "failed"
            and overlay_qc.get("passed") is False
            and (
                organoid_count == 0
                or foreground_coverage <= ZERO_RESCUE_FOREGROUND_COVERAGE_MAX
            )
        )
    message = str(result.get("error") or "").lower()
    if "organoids=0" in message:
        return True
    match = re.search(r"foreground_coverage=([0-9.]+)", message)
    if match and result.get("status") == "failed":
        try:
            return float(match.group(1)) <= ZERO_RESCUE_FOREGROUND_COVERAGE_MAX
        except ValueError:
            return False
    return False


def _adaptive_attempt_summary(result, params, attempt_index):
    overlay_qc = result.get("overlay_qc") if isinstance(result, dict) else None
    foreground_coverage = None
    if isinstance(overlay_qc, dict):
        foreground_coverage = overlay_qc.get("foreground_coverage")
    return {
        "attempt": attempt_index,
        "status": result.get("status") if isinstance(result, dict) else None,
        "organoid_count": result.get("organoid_count") if isinstance(result, dict) else None,
        "foreground_coverage": foreground_coverage,
        "parameters": dict(params),
        "error": result.get("error") if isinstance(result, dict) else None,
    }


def _organoid_rescue_plan(base_params):
    base = dict(base_params)

    def merged(**overrides):
        out = dict(base)
        out.update(overrides)
        return out

    # Cellpose 4 resizes the image by 30 / diameter before inference; diameter=None keeps
    # the CO checkpoint's native training scale. Probe native first, then one enlarged
    # (diameter 15 = 2x) and one reduced (diameter 60 = 0.5x) scale.
    return [
        merged(diameter=None, cellprob_threshold=-2),
        merged(diameter=None, cellprob_threshold=-4),
        merged(diameter=15, cellprob_threshold=-3),
        merged(diameter=None, cellprob_threshold=-3),
        merged(diameter=60, cellprob_threshold=-3),
        merged(diameter=None, cellprob_threshold=-3, augment=True),
        merged(diameter=None, cellprob_threshold=-3, augment=True, niter=500),
        merged(diameter=None, cellprob_threshold=-3, augment=True, niter=500,
               tile_overlap=0.25),
        merged(diameter=None, cellprob_threshold=-5),
    ]


class Organoid_Segmenter_Tool(BaseTool):
    """
    Organoid segmentation tool using pretrained Cellpose models.
    This tool is designed for segmenting organoids in microscopy images.
    """
    def __init__(self, model_path=None):
        super().__init__(
            tool_name="Organoid_Segmenter_Tool",
            tool_description="Segments organoids in microscopy images using specialized organoid segmentation models. REQUIRES a custom trained organoid model - cyto/cyto2 models are not suitable for organoids.",
            tool_version="1.0.0",
            input_types={
                "image": "str - Path to the input image (supports .tif, .png, .jpg formats).",
                "diameter": "float or None - Cellpose rescale reference: the image is resized by 30 / diameter before inference. None (default) keeps the CO checkpoint's native training scale; this is not the organoid's pixel diameter.",
                "flow_threshold": "float - Flow threshold for organoid detection (default: 0.4).",
                "cellprob_threshold": "float - Cell probability threshold (default: 0).",
                "min_size": "int - Minimum segmented object size retained by Cellpose (default: 15).",
                "niter": "int - Maximum Cellpose dynamics iterations (default: 200).",
                "tile_overlap": "float - Cellpose tile overlap for inference (default: 0.1).",
                "augment": "bool - Enable Cellpose test-time augmentation (default: False).",
                "resample": "bool - Run Cellpose dynamics at full resolution (default: True).",
                "adaptive_rescue": "bool - If true, zero/near-zero visual-QC failures automatically try a bounded organoid rescue sequence.",
                "model_path": "str - Optional local path to the custom CO organoid model. Can also be set with SHAPE_ORGANOID_MODEL_PATH."
            },
            output_type="dict - Contains segmentation results, organoid count, and visualization paths.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/organoids.tif")',
                    "description": "Segment organoids using a local CO model supplied by model_path or SHAPE_ORGANOID_MODEL_PATH."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/organoids.tif", diameter=15, flow_threshold=0.5)',
                    "description": "Segment organoids with the image enlarged 2x (30 / 15) and a custom flow threshold."
                }
            ],
            user_metadata={
                "limitation": "Requires GPU for optimal performance and a local specialized CO organoid model.",
                "best_practice": "Set SHAPE_ORGANOID_MODEL_PATH or pass model_path. Use this tool only for organoid or spheroid images."
            },
            output_dir="output_visualizations"
        )
        
        # Enable GPU if available (fall back to CPU gracefully)
        self.device, device_warning = resolve_torch_device()
        print(f"Organoid_Segmenter_Tool: Using device: {self.device}")
        if device_warning:
            print(f"Organoid_Segmenter_Tool: {device_warning}")
        
        # Store model path for lazy loading (model will be loaded in execute method)
        # This avoids initialization failures due to CP3/CP4 compatibility issues
        self.model_path = model_path or os.getenv("SHAPE_ORGANOID_MODEL_PATH")
        self.model = None  # Will be loaded on first execute
        self.model_type = "organoid"
        self._model_loaded = False
        self._model_load_error = None
        
        if self.model_path is not None and not os.path.exists(self.model_path):
            print(f"Organoid_Segmenter_Tool: Warning - model path does not exist: {self.model_path}")

    def _execute_adaptive_rescue(
        self,
        *,
        image,
        diameter,
        flow_threshold,
        cellprob_threshold,
        min_size,
        niter,
        tile_overlap,
        augment,
        resample,
        model_path,
        query_cache_dir,
        image_id,
        adaptive_rescue_max_attempts,
        kwargs,
    ):
        base_params = {
            "diameter": diameter,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "min_size": min_size,
            "niter": niter,
            "tile_overlap": tile_overlap,
            "augment": augment,
            "resample": resample,
        }
        attempts = []
        result = self.execute(
            image=image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            niter=niter,
            tile_overlap=tile_overlap,
            augment=augment,
            resample=resample,
            model_path=model_path,
            query_cache_dir=query_cache_dir,
            image_id=image_id,
            adaptive_rescue=False,
            **kwargs,
        )
        attempts.append(_adaptive_attempt_summary(result, base_params, 1))
        if not _zero_or_near_zero_failure(result):
            if isinstance(result, dict):
                result["adaptive_rescue"] = {
                    "enabled": True,
                    "attempted": False,
                    "strategy": "fast_then_v4_zero_detection_rescue",
                    "attempts": attempts,
                }
            return result

        max_attempts = max(1, int(adaptive_rescue_max_attempts or 10))
        last_result = result
        for attempt_index, params in enumerate(
            _organoid_rescue_plan(base_params)[:max(0, max_attempts - 1)],
            start=2,
        ):
            last_result = self.execute(
                image=image,
                model_path=model_path,
                query_cache_dir=query_cache_dir,
                image_id=image_id,
                attempt_id=attempt_index,
                adaptive_rescue=False,
                **params,
                **kwargs,
            )
            attempts.append(_adaptive_attempt_summary(last_result, params, attempt_index))
            if not _zero_or_near_zero_failure(last_result):
                break

        if isinstance(last_result, dict):
            last_result["adaptive_rescue"] = {
                "enabled": True,
                "attempted": True,
                "strategy": "fast_then_v4_zero_detection_rescue",
                "trigger": "zero_or_near_zero_visual_qc",
                "attempts": attempts,
            }
        return last_result
    
    def _load_model(self, model_path=None):
        """Load the organoid model (lazy loading)."""
        if self._model_loaded and self.model is not None:
            return
        
        # Use provided model_path or fallback to stored one
        model_to_load = model_path or self.model_path
        
        if model_to_load is None:
            error_msg = (
                "Organoid segmentation requires the local CO model. "
                "Pass model_path or set SHAPE_ORGANOID_MODEL_PATH."
            )
            self._model_load_error = error_msg
            raise ValueError(error_msg)
        
        try:
            # Load custom organoid model from path
            self.model = models.CellposeModel(
                device=self.device,
                pretrained_model=model_to_load,
                **cellpose_precision_kwargs(self.device),
            )
            self.model_path = model_to_load
            
            if self.model is None:
                raise ValueError("Failed to initialize CellposeModel")
            print(f"Organoid_Segmenter_Tool: Organoid model loaded successfully from {model_to_load}")
            self._model_loaded = True
            self._model_load_error = None
        except Exception as e:
            # Check if it's a CP3 compatibility issue
            error_str = str(e).lower()
            if "cp3" in error_str or "cp4" in error_str or "not compatible" in error_str:
                # Retry on CPU (sometimes helps)
                try:
                    print(f"Organoid_Segmenter_Tool: GPU load failed (possible CP3 compatibility issue: {e}), retrying on CPU")
                    self.device = torch.device('cpu')
                    self.model = models.CellposeModel(
                        device=self.device,
                        pretrained_model=model_to_load,
                        **cellpose_precision_kwargs(self.device),
                    )
                    self.model_path = model_to_load
                    if self.model is None:
                        raise ValueError("Failed to initialize CellposeModel on CPU")
                    print("Organoid_Segmenter_Tool: CPU model loaded successfully")
                    self._model_loaded = True
                    self._model_load_error = None
                except Exception as cpu_e:
                    error_msg = (
                        f"Organoid_Segmenter_Tool: Model loading failed on both GPU and CPU: {cpu_e}\n"
                        f"ERROR: The model appears to be a Cellpose 3 (CP3) model, but Cellpose 4 is installed.\n"
                        f"Cellpose 4 does not support CP3 models. Solutions:\n"
                        f"1. Convert your model to CP4 format\n"
                        f"2. Use Cellpose 3.x instead of Cellpose 4.x (pip install 'cellpose<2.2')\n"
                        f"3. Train a new model using Cellpose 4\n"
                        f"Model path: {model_to_load}"
                    )
                    self._model_load_error = error_msg
                    raise ValueError(error_msg)
            else:
                # Retry on CPU for other errors
                try:
                    print(f"Organoid_Segmenter_Tool: GPU load failed ({e}), retrying on CPU")
                    self.device = torch.device('cpu')
                    self.model = models.CellposeModel(
                        device=self.device,
                        pretrained_model=model_to_load,
                        **cellpose_precision_kwargs(self.device),
                    )
                    self.model_path = model_to_load
                    if self.model is None:
                        raise ValueError("Failed to initialize CellposeModel on CPU")
                    print("Organoid_Segmenter_Tool: CPU model loaded successfully")
                    self._model_loaded = True
                    self._model_load_error = None
                except Exception as cpu_e:
                    error_msg = (
                        f"Organoid_Segmenter_Tool: Error loading organoid model on both GPU and CPU: {cpu_e}\n"
                        f"Please ensure the model file is valid and compatible with Cellpose.\n"
                        f"Model path: {model_to_load}"
                    )
                    self._model_load_error = error_msg
                    raise ValueError(error_msg)

    def _execute_native(
        self,
        image,
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0,
        min_size=15,
        niter=None,
        tile_overlap=None,
        augment=None,
        resample=None,
        model_path=None,
        query_cache_dir=None,
        image_id=None,
        attempt_id=None,
        adaptive_rescue=True,
        adaptive_rescue_max_attempts=10,
        **kwargs,
    ):
        """
        Execute organoid segmentation on the input image.
        
        Args:
            image (str): Path to the input image
            diameter (float or None): Cellpose rescale reference; the image is resized by
                30 / diameter. None keeps the checkpoint's native training scale. It is not
                the organoid's pixel diameter: larger values shrink the image.
            flow_threshold (float): Flow threshold for organoid detection
            cellprob_threshold (float): Cell probability threshold
            min_size (int): Minimum mask size kept by Cellpose.
            niter (int): Optional maximum dynamics iterations for Cellpose.
            tile_overlap (float): Optional tile overlap for Cellpose inference.
            augment (bool): Optional Cellpose test-time augmentation override.
            resample (bool): Optional Cellpose full-resolution dynamics override.
            adaptive_rescue (bool): Automatically try a bounded organoid rescue
                sequence after zero/near-zero visual-QC failures.
            model_path (str): Optional path to a custom organoid model. If not provided, uses the model loaded during initialization.
            query_cache_dir (str): Directory to save outputs
            image_id (str): Optional image identifier for consistent file naming
            
        Returns:
            dict: Segmentation results with organoid count and visualization paths
        """
        if (
            _adaptive_rescue_requested(adaptive_rescue)
            and attempt_id is None
            and kwargs.get("_vlrr_attempt_index") is None
        ):
            return self._execute_adaptive_rescue(
                image=image,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size,
                niter=niter,
                tile_overlap=tile_overlap,
                augment=augment,
                resample=resample,
                model_path=model_path,
                query_cache_dir=query_cache_dir,
                image_id=image_id,
                adaptive_rescue_max_attempts=adaptive_rescue_max_attempts,
                kwargs=kwargs,
            )
        try:
            # Load model if not already loaded (lazy loading)
            if model_path and model_path != self.model_path:
                # New model path provided, reset and load it
                self.model = None
                self._model_loaded = False
                self.model_path = model_path
            
            if not self._model_loaded or self.model is None:
                try:
                    self._load_model(model_path)
                except ValueError as ve:
                    # Model loading failed, return error
                    return {
                        "error": str(ve),
                        "summary": "Failed to load organoid segmentation model",
                        "error_details": {
                            "model_path": model_path or self.model_path,
                            "error_type": "Model loading failed - possible CP3/CP4 compatibility issue"
                        }
                    }
            
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

                # Segmentation input: full bit-depth, NO downcast (this task).
                # `img_data.to_uint8(0)` used to be the ONLY array built here and fed
                # straight to Cellpose -- a uint8 downcast baked in unconditionally,
                # before any preprocessing gate. `seg_input` now carries channel 0's
                # raw values as float32; `phase_contrast_img` (uint8) is built ONLY
                # for the overlay visualization further below and is never fed to
                # Cellpose. Organoid segmentation stays single-channel BY DESIGN (see
                # class docstring) -- unrelated to this lossless-input fix.
                seg_input = img_data.to_segmentation_input(channel_idx=0)
                phase_contrast_img = img_data.to_uint8(0)
                is_multi_channel = img_data.is_multi_channel

                if is_multi_channel:
                    print(f"Detected multi-channel image with {img_data.num_channels} channels. Using first channel ({img_data.channel_names[0] if img_data.channel_names else 'phase contrast'}) for segmentation.")
            except Exception as load_error:
                # Fallback to legacy loading for backward compatibility. Hardened
                # (this task): the selected channel/plane keeps its NATIVE dtype/range
                # here (no forced uint8 downcast baked in before Cellpose ever sees
                # it) -- `phase_contrast_img` (uint8, overlay-only) is derived from it
                # separately below via the canonical percentile converter, not a fixed
                # /65535 scale. Prefers skimage.io.imread's full-array read over
                # cv2.imread(IMREAD_GRAYSCALE); the latter is now a LOUD, documented
                # last resort, not a silent default.
                print(f"Warning: Failed to load with ImageProcessor: {load_error}, trying legacy method")
                image_path_lower = image_path.lower()
                is_tiff = image_path_lower.endswith('.tif') or image_path_lower.endswith('.tiff')
                is_multi_channel = False
                raw_img = None

                if is_tiff:
                    try:
                        raw_img = tifffile.imread(image_path)
                    except Exception as tiff_error:
                        print(f"Warning: Failed to load TIFF via tifffile: {tiff_error}, trying skimage")
                        try:
                            raw_img = skimage_io.imread(image_path)
                        except Exception as skimage_error:
                            print(f"Warning: skimage.io.imread also failed: {skimage_error}, "
                                  "trying cv2 grayscale -- LOUD LAST RESORT, collapses channels")
                            raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                else:
                    try:
                        raw_img = skimage_io.imread(image_path)
                    except Exception as skimage_error:
                        print(f"Warning: skimage.io.imread failed: {skimage_error}, "
                              "trying cv2 grayscale -- LOUD LAST RESORT, collapses channels")
                        raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if raw_img is None:
                    return {
                        "error": f"Cannot read image: {image_path}",
                        "summary": "Image loading failed"
                    }

                # Select the channel/plane to segment -- organoid segmentation is
                # single-channel BY DESIGN (unrelated to the lossless-input fix); the
                # native dtype/range of that channel is kept (no downcast here).
                if raw_img.ndim == 2:
                    seg_input = raw_img
                elif raw_img.ndim == 3:
                    if raw_img.shape[2] <= 4:
                        seg_input = raw_img[:, :, 0]
                        is_multi_channel = raw_img.shape[2] > 1
                    else:
                        seg_input = raw_img[0, :, :]
                        is_multi_channel = raw_img.shape[0] > 1
                elif raw_img.ndim == 4:
                    # A genuine Z/T stack: this tool has no Z/T model, so a single
                    # plane must be picked -- unavoidable, but LOUD, not silent.
                    print(f"Warning: legacy loader collapsing a 4D (Z/T) stack "
                          f"{raw_img.shape} to its first plane/channel -- this tool "
                          "has no Z/T model.")
                    seg_input = raw_img[0, :, :, 0] if raw_img.shape[3] <= 4 else raw_img[0, 0, :, :]
                    is_multi_channel = True
                else:
                    seg_input = raw_img

                # Overlay/display view only (uint8, canonical percentile converter) --
                # built from the same seg_input, never fed to Cellpose.
                phase_contrast_img = to_display_uint8(seg_input)

            # Convert to float32 for Cellpose segmentation, full range preserved (no
            # renormalization/clipping). Ensure image is 2D (grayscale) for organoid
            # segmentation.
            if seg_input.ndim == 3:
                # If somehow we got a 3D image, take first channel
                print(f"Warning: seg_input has 3 dimensions {seg_input.shape}, using first channel")
                seg_input = seg_input[:, :, 0] if seg_input.shape[2] <= seg_input.shape[0] else seg_input[0, :, :]

            if seg_input.ndim != 2:
                return {
                    "error": f"Invalid image dimensions: {seg_input.shape}. Expected 2D grayscale image.",
                    "summary": "Image format error"
                }

            img = seg_input.astype(np.float32)
            
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
            # For organoids, we use grayscale images [0, 0]
            # img should be 2D at this point (already validated above)
            if len(img.shape) == 2:
                channels = [0, 0]  # Grayscale
            else:
                # This should not happen, but handle gracefully
                print(f"Warning: Unexpected image shape {img.shape}, forcing to grayscale")
                if len(img.shape) == 3:
                    img = img[:, :, 0] if img.shape[2] <= img.shape[0] else img[0, :, :]
                channels = [0, 0]
            
            # Run segmentation
            try:
                min_size_int = int(min_size) if min_size is not None else 15
            except (TypeError, ValueError):
                min_size_int = 15
            eval_kwargs = {
                "diameter": diameter,
                "channels": channels,
                "flow_threshold": flow_threshold,
                "cellprob_threshold": cellprob_threshold,
                "min_size": min_size_int,
            }
            niter_int = None
            tile_overlap_float = None
            augment_bool = None
            resample_bool = None
            if niter is not None:
                try:
                    niter_int = int(niter)
                    eval_kwargs["niter"] = niter_int
                except (TypeError, ValueError):
                    print(f"Warning: Invalid niter value '{niter}', using Cellpose default")
            if tile_overlap is not None:
                try:
                    tile_overlap_float = float(tile_overlap)
                    eval_kwargs["tile_overlap"] = tile_overlap_float
                except (TypeError, ValueError):
                    print(f"Warning: Invalid tile_overlap value '{tile_overlap}', using Cellpose default")
            if augment is not None:
                augment_bool = _coerce_bool(augment, False)
                eval_kwargs["augment"] = augment_bool
            if resample is not None:
                resample_bool = _coerce_bool(resample, True)
                eval_kwargs["resample"] = resample_bool

            if os.environ.get("ORGANOID_REQUIRE_GPU") == "1" and not str(self.device).startswith("cuda"):
                raise RuntimeError("GPU required; the CO model fell back to CPU")
            if not np.isfinite(img).all():
                raise ValueError("Organoid_Segmenter_Tool refuses non-finite (NaN/Inf) input pixels")
            # Cellpose scales by 30 / diameter. Bound the allocation before the native call;
            # a small diameter on a large image otherwise creates tens of millions of pixels.
            from shape.tools.organoid_segmenter.runtime_limits import check_native_size
            check_native_size(img.shape, diameter, augment=augment_bool, niter=niter_int)
            masks, flows, styles = self.model.eval([img], **eval_kwargs)

            mask = masks[0]
            # VLRR segmentation gate: mean Cellpose flow error (see segmentation_qc)
            flow_error_val = compute_flow_error(mask, flows, device=getattr(self, "device", None))
            
            # Setup output directory using centralized configuration
            output_dir = VisualizationConfig.get_output_dir(query_cache_dir)
            
            # Use phase contrast image for overlay (ensure it's uint8)
            phase_contrast_for_overlay = phase_contrast_img.copy()
            if phase_contrast_for_overlay.dtype != np.uint8:
                phase_contrast_for_overlay = np.clip(phase_contrast_for_overlay, 0, 255).astype(np.uint8)
            
            # Create overlay on phase contrast image
            overlay = plot.mask_overlay(phase_contrast_for_overlay, mask)
            overlay_qc = evaluate_segmentation_visual(mask, overlay)
            
            # Count organoids (unique mask values, excluding background 0)
            n_organoids = len(np.unique(mask)) - 1 if mask is not None else 0
            
            attempt_index = attempt_id or kwargs.get("_vlrr_attempt_index")
            try:
                attempt_suffix = (
                    f"_attempt_{int(attempt_index):02d}"
                    if attempt_index is not None else ""
                )
            except (TypeError, ValueError):
                attempt_suffix = f"_attempt_{str(attempt_index)}"

            # Save overlay visualization with professional styling using image identifier.
            # VLRR retries get attempt-specific files so the benchmark can audit every
            # parameter probe instead of only the final overwritten artifact.
            overlay_path = os.path.join(
                output_dir, f"organoid_overlay_{image_identifier}{attempt_suffix}.png"
            )
            latest_overlay_path = os.path.join(
                output_dir, f"organoid_overlay_{image_identifier}.png"
            )
            fig, ax = VisualizationConfig.create_professional_figure(figsize=(12, 8))
            # Ensure overlay has same brightness as original phase contrast image
            overlay_normalized = overlay.astype(np.float32) / 255.0
            ax.imshow(overlay_normalized, vmin=0, vmax=1)
            ax.axis('off')
            VisualizationConfig.apply_professional_styling(
                ax, title=f"Organoid Segmentation - {n_organoids} organoids detected"
            )
            VisualizationConfig.save_professional_figure(fig, overlay_path)
            if latest_overlay_path != overlay_path:
                VisualizationConfig.save_professional_figure(fig, latest_overlay_path)
            plt.close(fig)
            
            # Save mask in two formats:
            # 1. .tif format for visual display in Visual Outputs (16-bit TIFF preserves all label values)
            # 2. .npy format for morphological analysis in Downloadable Files (original array data)
            mask_tif_path = os.path.join(
                output_dir, f"organoid_mask_{image_identifier}{attempt_suffix}.tif"
            )
            mask_npy_path = os.path.join(
                output_dir, f"organoid_mask_{image_identifier}{attempt_suffix}.npy"
            )
            latest_mask_tif_path = os.path.join(
                output_dir, f"organoid_mask_{image_identifier}.tif"
            )
            latest_mask_npy_path = os.path.join(
                output_dir, f"organoid_mask_{image_identifier}.npy"
            )
            
            # Save .tif format for visual display (16-bit preserves all label values, supports up to 65535 organoids)
            mask_uint16 = mask.astype(np.uint16)
            tifffile.imwrite(mask_tif_path, mask_uint16)
            if latest_mask_tif_path != mask_tif_path:
                tifffile.imwrite(latest_mask_tif_path, mask_uint16)
            
            # Save .npy format for morphological analysis (preserves original array data)
            # Cellpose masks are integer labels (0=background, 1-N for N organoids)
            np.save(mask_npy_path, mask)
            if latest_mask_npy_path != mask_npy_path:
                np.save(latest_mask_npy_path, mask)
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Build visual outputs list - overlay and .tif mask (for display)
            # .npy mask will be automatically added to Downloadable Files by app.py
            visual_outputs_list = [overlay_path, mask_tif_path]
            
            result = {
                "summary": f"{n_organoids} organoids identified and segmented successfully.",
                "status": "success" if overlay_qc["passed"] else "failed",
                "organoid_count": n_organoids,
                "cell_count": n_organoids,  # For compatibility with other tools that expect cell_count
                "visual_outputs": visual_outputs_list,
                # The instance mask is the deliverable; the overlay is its visual check.
                "deliverables": [mask_tif_path, overlay_path],
                "model_used": f"CellposeModel (organoid model from {self.model_path})",
                "flow_error": flow_error_val,  # VLRR segmentation gate metric (mean Cellpose flow error)
                "parameters": {
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "min_size": min_size,
                    "niter": niter_int,
                    "tile_overlap": tile_overlap_float,
                    "augment": augment_bool,
                    "resample": resample_bool,
                    "attempt_id": attempt_index,
                    "model_path": self.model_path
                },
                "mask_path": mask_npy_path,  # .npy format for morphological analysis (for compatibility with downstream tools)
                "mask_tif_path": mask_tif_path,  # .tif format for visual display
                "organoid_mask_path": mask_npy_path,  # Explicit name for organoid masks (.npy format)
                "overlay_path": overlay_path,  # Segmentation overlay path
                "latest_mask_path": latest_mask_npy_path,
                "latest_mask_tif_path": latest_mask_tif_path,
                "latest_overlay_path": latest_overlay_path,
                "overlay_qc": overlay_qc,
                "image_id": image_identifier,  # Add image_id for matching
                "image_identifier": image_identifier  # Alias for compatibility
            }

            if not overlay_qc["passed"]:
                result["error"] = (
                    "Segmentation output failed visual QC: "
                    f"organoids={n_organoids}, "
                    f"overlay_is_black={overlay_qc['overlay_is_black']}, "
                    f"foreground_coverage={overlay_qc['foreground_coverage']:.6f}"
                )
                result["summary"] = result["error"]
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"Organoid_Segmenter_Tool: Error: {error_msg}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "error": f"Error during organoid segmentation: {error_msg}",
                "summary": f"Failed to process image: {error_msg}"
            }


    def enable_trial_cache(self, root, *, seed=0):
        """Reuse verified native trials; every hit still enters the managed VRR."""
        from pathlib import Path
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("native cache seed must be a uint32 integer")
        options = (str(Path(root).resolve()), seed)
        if options != getattr(self, "_trial_cache_options", None):
            self._native_trial_cache = None
            self._trial_cache_model = None
        self._trial_cache_options = options

    def disable_trial_cache(self):
        self._trial_cache_options = None
        self._native_trial_cache = None
        self._trial_cache_model = None

    def set_trial_cache(self, cache):
        from shape.tools.cell_segmenter.trial_cache import SegmentationTrialCache
        if cache is not None and not isinstance(cache, SegmentationTrialCache):
            raise TypeError("expected a SegmentationTrialCache or None")
        self._native_trial_cache = cache

    def execute(
        self, image, diameter=None, flow_threshold=0.4, cellprob_threshold=0,
        min_size=15, niter=None, tile_overlap=None, augment=None, resample=None,
        model_path=None, query_cache_dir=None, image_id=None, attempt_id=None,
        adaptive_rescue=True, adaptive_rescue_max_attempts=10, **kwargs,
    ):
        """Run one managed native trial, or the original direct-call rescue."""
        from pathlib import Path
        arguments = dict(image=image, diameter=diameter, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, min_size=min_size, niter=niter,
            tile_overlap=tile_overlap, augment=augment, resample=resample,
            model_path=model_path, query_cache_dir=query_cache_dir, image_id=image_id,
            attempt_id=attempt_id, adaptive_rescue=adaptive_rescue,
            adaptive_rescue_max_attempts=adaptive_rescue_max_attempts, **kwargs)
        options = getattr(self, "_trial_cache_options", None)
        index = attempt_id if attempt_id is not None else kwargs.get("_vlrr_attempt_index")
        # Only bounded managed calls have one model evaluation. Unknown kwargs
        # and direct adaptive-rescue calls retain the original uncached behavior.
        if (options is None or image_id is None or index is None
                or set(kwargs) - {"_vlrr_attempt_index"}
                or not isinstance(image, (str, os.PathLike))):
            return self._execute_native(**arguments)
        if model_path and model_path != self.model_path:
            self.model, self._model_loaded, self.model_path = None, False, model_path
        if not self._model_loaded or self.model is None:
            self._load_model(model_path)
        parameters = {key: arguments[key] for key in (
            "diameter", "flow_threshold", "cellprob_threshold", "min_size",
            "niter", "tile_overlap", "augment", "resample")}
        cache = getattr(self, "_native_trial_cache", None)
        if cache is None or self.model is not getattr(self, "_trial_cache_model", None):
            from shape.tools.cell_segmenter.cache_runtime import configure
            configure(self, options[0], seed=options[1], required_parameters=parameters)
            cache = self._native_trial_cache
            self._trial_cache_model = self.model

        def compute(folder):
            from shape.tools.cell_segmenter.cache_runtime import native_rng
            with native_rng(self._trial_cache_seed, self.device):
                return self._execute_native(**{**arguments, "query_cache_dir": str(folder)})

        return cache.get_or_compute(Path(image), str(image_id), parameters, compute,
            context={"caller_output_dir": str(query_cache_dir), "managed_attempt_id": index})


    def get_metadata(self):
        """Returns the metadata for the Organoid_Segmenter_Tool."""
        metadata = super().get_metadata()
        metadata["device"] = str(self.device)
        metadata["model_loaded"] = self._model_loaded if hasattr(self, '_model_loaded') else False
        metadata["model_type"] = self.model_type if hasattr(self, 'model_type') else "unknown"
        metadata["model_path"] = self.model_path if hasattr(self, 'model_path') else None
        if hasattr(self, '_model_load_error') and self._model_load_error:
            metadata["model_load_error"] = self._model_load_error
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd shape/tools/organoid_segmenter
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Organoid_Segmenter_Tool
    tool = Organoid_Segmenter_Tool()
    tool.set_custom_output_dir("organoid_outputs")

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
        print(f"Organoid Count: {execution.get('organoid_count', 'Unknown')}")
        print(f"Visual Outputs: {execution.get('visual_outputs', [])}")
        if 'error' in execution:
            print(f"Error: {execution['error']}")
    except Exception as e:
        print(f"Execution failed: {e}")

    print("Done!")
