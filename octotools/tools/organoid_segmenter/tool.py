from octotools.tools.base import BaseTool
from cellpose import models, plot
import numpy as np
import cv2
import os
from PIL import Image
from uuid import uuid4
import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
from octotools.models.utils import VisualizationConfig
import traceback
import tifffile

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
                "diameter": "float - Expected organoid diameter in pixels (default: None, auto-detect).",
                "flow_threshold": "float - Flow threshold for organoid detection (default: 0.4).",
                "cellprob_threshold": "float - Cell probability threshold (default: 0).",
                "model_path": "str - Optional path to a custom organoid model. If not provided, downloads from Hugging Face."
            },
            output_type="dict - Contains segmentation results, organoid count, and visualization paths.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/organoids.tif")',
                    "description": "Segment organoids in a microscopy image with default parameters. Requires organoid model to be uploaded to Hugging Face."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/organoids.tif", diameter=100, flow_threshold=0.5)',
                    "description": "Segment organoids with custom diameter and flow threshold."
                }
            ],
            user_metadata={
                "limitation": "Requires GPU for optimal performance. REQUIRES a specialized organoid segmentation model - cyto/cyto2 models will NOT work for organoids. Model must be uploaded to Hugging Face or provided as model_path.",
                "best_practice": "Use with Image_Preprocessor_Tool for better results on low-quality images. Adjust diameter parameter based on your organoid size and image resolution. For organoids, typically use larger diameter values (50-200 pixels). Upload your trained organoid model to Hugging Face following MODEL_UPLOAD_GUIDE.md."
            },
            output_dir="output_visualizations"
        )
        
        # Enable GPU if available (fall back to CPU gracefully)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Organoid_Segmenter_Tool: Using device: {self.device}")
        
        # Store model path for lazy loading (model will be loaded in execute method)
        # This avoids initialization failures due to CP3/CP4 compatibility issues
        self.model_path = model_path
        self.model = None  # Will be loaded on first execute
        self.model_type = "organoid"
        self._model_loaded = False
        self._model_load_error = None
        
        # If model_path is provided, validate it exists but don't load yet
        if model_path is not None:
            if not os.path.exists(model_path):
                print(f"Organoid_Segmenter_Tool: Warning - provided model_path does not exist: {model_path}")
        else:
            # Try to download model path (but don't load yet)
            try:
                self.model_path = hf_hub_download(
                    repo_id="5xuekun/organoid-segmenter-model",
                    filename="cpsam_CO_10x_260106",
                    token=os.getenv("HUGGINGFACE_TOKEN")
                )
                print(f"Organoid_Segmenter_Tool: Custom organoid model path determined: {self.model_path}")
            except Exception as e:
                print(f"Organoid_Segmenter_Tool: Could not determine model path at initialization: {e}")
                print(f"Organoid_Segmenter_Tool: Model will be downloaded when first used (in execute method)")
    
    def _load_model(self, model_path=None):
        """Load the organoid model (lazy loading)."""
        if self._model_loaded and self.model is not None:
            return
        
        # Use provided model_path or fallback to stored one
        model_to_load = model_path or self.model_path
        
        # If still no path, try to download
        if model_to_load is None:
            try:
                model_to_load = hf_hub_download(
                    repo_id="5xuekun/organoid-segmenter-model",
                    filename="cpsam_CO_10x_260106",
                    token=os.getenv("HUGGINGFACE_TOKEN")
                )
                self.model_path = model_to_load
                print(f"Organoid_Segmenter_Tool: Downloaded model to {model_to_load}")
            except Exception as e:
                error_msg = (
                    f"Organoid_Segmenter_Tool: Failed to download organoid model: {e}\n"
                    f"ERROR: Organoid segmentation REQUIRES a specialized organoid model.\n"
                    f"Standard Cellpose models (cyto/cyto2) are NOT suitable for organoids.\n"
                    f"Please upload your trained organoid model to Hugging Face following MODEL_UPLOAD_GUIDE.md,\n"
                    f"or provide a model_path parameter with the path to your organoid model."
                )
                self._model_load_error = error_msg
                raise ValueError(error_msg)
        
        try:
            # Load custom organoid model from path
            self.model = models.CellposeModel(
                gpu=torch.cuda.is_available(), 
                pretrained_model=model_to_load
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
                        gpu=False,
                        pretrained_model=model_to_load
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
                        gpu=False,
                        pretrained_model=model_to_load
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

    def execute(self, image, diameter=100, flow_threshold=0.4, cellprob_threshold=0, model_path=None, query_cache_dir=None, image_id=None):
        """
        Execute organoid segmentation on the input image.
        
        Args:
            image (str): Path to the input image
            diameter (float): Expected organoid diameter in pixels (typically larger than cells: 50-200)
            flow_threshold (float): Flow threshold for organoid detection
            cellprob_threshold (float): Cell probability threshold
            model_path (str): Optional path to a custom organoid model. If not provided, uses the model loaded during initialization.
            query_cache_dir (str): Directory to save outputs
            image_id (str): Optional image identifier for consistent file naming
            
        Returns:
            dict: Segmentation results with organoid count and visualization paths
        """
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
            
            # Load and preprocess image
            # Check if it's a multi-channel TIFF file
            image_path_lower = image_path.lower()
            is_tiff = image_path_lower.endswith('.tif') or image_path_lower.endswith('.tiff')
            is_multi_channel = False
            phase_contrast_img = None  # Store phase contrast image for overlay
            
            if is_tiff:
                # Try loading with tifffile first to handle multi-channel TIFF
                try:
                    img_full = tifffile.imread(image_path)
                    # Check if multi-channel (shape: (H, W, C) or (C, H, W) or (Z, H, W, C))
                    if img_full.ndim == 2:
                        # Single channel 2D image - use directly
                        is_multi_channel = False
                        print(f"Detected single-channel 2D TIFF. Using directly for segmentation.")
                        phase_contrast_img = img_full
                    elif img_full.ndim == 3:
                        # 3D: could be (H, W, C) or (C, H, W) or (H, W, 1) single channel
                        if img_full.shape[2] > 1 and img_full.shape[2] <= 4:  # Multi-channel in last dimension (H, W, C)
                            is_multi_channel = True
                            print(f"Detected multi-channel TIFF with {img_full.shape[2]} channels. Using first channel (phase contrast) for segmentation.")
                            phase_contrast_img = img_full[:, :, 0]  # Extract first channel (phase contrast)
                        elif img_full.shape[0] > 1 and img_full.shape[0] <= 4:  # Multi-channel in first dimension (C, H, W)
                            is_multi_channel = True
                            print(f"Detected multi-channel TIFF with {img_full.shape[0]} channels. Using first channel (phase contrast) for segmentation.")
                            phase_contrast_img = img_full[0, :, :]  # Extract first channel (phase contrast)
                        else:
                            # Single channel 3D (H, W, 1) or ambiguous - squeeze to 2D
                            is_multi_channel = False
                            print(f"Detected single-channel 3D TIFF. Squeezing to 2D for segmentation.")
                            phase_contrast_img = np.squeeze(img_full)
                            if phase_contrast_img.ndim != 2:
                                # If still not 2D, use first slice
                                phase_contrast_img = img_full[:, :, 0] if img_full.shape[2] < img_full.shape[0] else img_full[0, :, :]
                    elif img_full.ndim == 4:
                        # 4D: could be (Z, H, W, C) or (C, Z, H, W)
                        is_multi_channel = True
                        print(f"Detected 4D multi-channel TIFF. Using first channel of first slice (phase contrast) for segmentation.")
                        if img_full.shape[3] > 1 and img_full.shape[3] <= 4:  # (Z, H, W, C)
                            phase_contrast_img = img_full[0, :, :, 0]
                        else:  # (C, Z, H, W)
                            phase_contrast_img = img_full[0, 0, :, :]
                    else:
                        # Unexpected dimensions - try to use as-is
                        is_multi_channel = False
                        print(f"Warning: Unexpected TIFF dimensions {img_full.shape}. Attempting to use directly.")
                        phase_contrast_img = img_full
                    
                    # Normalize phase contrast image to uint8 if needed
                    if phase_contrast_img.dtype == np.uint16:
                        phase_contrast_img = (phase_contrast_img / 65535.0 * 255).astype(np.uint8)
                    elif phase_contrast_img.dtype != np.uint8:
                        phase_contrast_img = np.clip(phase_contrast_img, 0, 255).astype(np.uint8)
                    
                    img = phase_contrast_img.astype(np.float32)
                except Exception as tiff_error:
                    print(f"Warning: Failed to load TIFF with tifffile: {tiff_error}, trying cv2")
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    phase_contrast_img = img.copy() if img is not None else None
            else:
                # For non-TIFF files, use cv2
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                phase_contrast_img = img.copy() if img is not None else None
            
            if img is None:
                # Try alternative: use PIL to load and convert
                try:
                    pil_img = Image.open(image_path)
                    if pil_img.mode != 'L':
                        pil_img = pil_img.convert('L')
                    img = np.array(pil_img)
                    phase_contrast_img = img.copy()
                    print(f"Loaded image using PIL: {image_path}")
                except Exception as pil_error:
                    return {
                        "error": f"Failed to load image: {image_path}. All loading methods failed: {str(pil_error)}",
                        "summary": "Image loading failed with all methods"
                    }
            
            if phase_contrast_img is None:
                phase_contrast_img = img.copy()
            
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
            # For organoids, we typically use grayscale images [0, 0]
            # But also support RGB if provided
            if len(img.shape) == 2:
                channels = [0, 0]  # Grayscale
            elif len(img.shape) == 3 and img.shape[2] == 3:
                channels = [1, 2]  # RGB - use green and red channels
            else:
                channels = [0, 0]  # Default to grayscale
            
            masks, flows, styles = self.model.eval(
                [img],
                diameter=diameter,
                channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold
            )
            
            mask = masks[0]
            
            # Setup output directory using centralized configuration
            output_dir = VisualizationConfig.get_output_dir(query_cache_dir)
            
            # Save phase contrast image before segmentation for display
            phase_contrast_for_display = phase_contrast_img.copy()
            if phase_contrast_for_display.dtype != np.uint8:
                phase_contrast_for_display = np.clip(phase_contrast_for_display, 0, 255).astype(np.uint8)
            
            phase_contrast_path = os.path.join(output_dir, f"phase_contrast_{image_identifier}.png")
            try:
                phase_contrast_pil = Image.fromarray(phase_contrast_for_display, mode='L')
                phase_contrast_pil.save(phase_contrast_path)
                print(f"Saved phase contrast image to: {phase_contrast_path}")
            except Exception as e:
                print(f"Warning: Failed to save phase contrast image: {e}")
                phase_contrast_path = None
            
            # Use phase contrast image for overlay (ensure it's uint8)
            phase_contrast_for_overlay = phase_contrast_img.copy()
            if phase_contrast_for_overlay.dtype != np.uint8:
                phase_contrast_for_overlay = np.clip(phase_contrast_for_overlay, 0, 255).astype(np.uint8)
            
            # Create overlay on phase contrast image
            overlay = plot.mask_overlay(phase_contrast_for_overlay, mask)
            
            # Save overlay visualization with professional styling using image identifier
            output_path = os.path.join(output_dir, f"organoid_overlay_{image_identifier}.png")
            fig, ax = VisualizationConfig.create_professional_figure(figsize=(12, 8))
            # Ensure overlay has same brightness as original phase contrast image
            overlay_normalized = overlay.astype(np.float32) / 255.0
            ax.imshow(overlay_normalized, vmin=0, vmax=1)
            ax.axis('off')
            VisualizationConfig.apply_professional_styling(
                ax, title=f"Organoid Segmentation - {len(np.unique(mask))-1} organoids detected"
            )
            VisualizationConfig.save_professional_figure(fig, output_path)
            plt.close(fig)
            
            # Count organoids (unique mask values, excluding background 0)
            n_organoids = len(np.unique(mask)) - 1 if mask is not None else 0
            
            # Save mask as separate visualization with professional styling using image identifier
            # Use .tif format with 16-bit depth to preserve all label values (supports up to 65535 organoids)
            mask_path = os.path.join(output_dir, f"organoid_mask_{image_identifier}.tif")
            
            # Save the original mask array as 16-bit TIFF to preserve all label values
            # Cellpose masks are integer labels (0=background, 1-N for N organoids), which can exceed 255
            mask_uint16 = mask.astype(np.uint16)
            tifffile.imwrite(mask_path, mask_uint16)
            
            # Also save a visualization version for display with professional styling
            viz_mask_path = os.path.join(output_dir, f"organoid_mask_viz_{image_identifier}.png")
            fig, ax = VisualizationConfig.create_professional_figure(figsize=(12, 8))
            ax.imshow(mask, cmap='tab20')
            ax.axis('off')
            VisualizationConfig.apply_professional_styling(
                ax, title=f"Organoid Masks - {n_organoids} organoids"
            )
            VisualizationConfig.save_professional_figure(fig, viz_mask_path)
            plt.close(fig)
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Build visual outputs list - include phase contrast image first, then overlay and mask
            visual_outputs_list = []
            if phase_contrast_path and os.path.exists(phase_contrast_path):
                visual_outputs_list.append(phase_contrast_path)
            visual_outputs_list.extend([output_path, mask_path])
            
            return {
                "summary": f"{n_organoids} organoids identified and segmented successfully.",
                "organoid_count": n_organoids,
                "cell_count": n_organoids,  # For compatibility with other tools that expect cell_count
                "visual_outputs": visual_outputs_list,
                "model_used": f"CellposeModel (organoid model from {self.model_path})",
                "parameters": {
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "model_path": self.model_path
                }
            }
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            print(f"Organoid_Segmenter_Tool: Error in organoid segmentation: {error_msg}")
            print(f"Organoid_Segmenter_Tool: Traceback: {error_traceback}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "error": f"Error during organoid segmentation: {error_msg}",
                "summary": "Failed to process image",
                "error_details": {
                    "image_path": image_path if 'image_path' in locals() else str(image),
                    "diameter": diameter if 'diameter' in locals() else "unknown",
                    "flow_threshold": flow_threshold if 'flow_threshold' in locals() else "unknown",
                    "cellprob_threshold": cellprob_threshold if 'cellprob_threshold' in locals() else "unknown"
                }
            }

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
    
    cd octotools/tools/organoid_segmenter
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
    relative_image_path = "../../../examples/fibroblast.png"
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
