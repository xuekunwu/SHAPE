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

class Cell_Segmenter_Tool(BaseTool):
    """
    Cell segmentation tool for phase-contrast cell images using Cellpose CPSAM model.
    This tool extends the system to handle general cell types (not just nuclei) in phase-contrast microscopy.
    """
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
                "best_practice": "Use with Image_Preprocessor_Tool for better results on low-quality images. Adjust diameter parameter based on your cell type and image resolution. For phase-contrast images, use 'cpsam' model type."
            },
            output_dir="output_visualizations"
        )
        
        # Enable GPU if available (fall back to CPU gracefully)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Cell_Segmenter_Tool: Using device: {self.device}")
        
        self.model_name = model_name
        
        # Download model from Hugging Face Hub
        if model_path is None:
            try:
                # For CPSAM model - you'll need to update this with your actual repo ID after uploading
                model_path = hf_hub_download(
                    repo_id="5xuekun/cell-segmenter-cpsam-model",  # UPDATE THIS after uploading your model
                    filename="cpsam",  # Model filename is 'cpsam'
                    token=os.getenv("HUGGINGFACE_TOKEN")
                )
                print(f"Cell_Segmenter_Tool: CPSAM model downloaded to {model_path}")
            except Exception as e:
                print(f"Cell_Segmenter_Tool: Failed to download CPSAM model: {e}")
                # Fallback to default Cellpose cyto model for phase-contrast
                print(f"Cell_Segmenter_Tool: Falling back to Cellpose 'cyto' model for phase-contrast images")
                model_path = None
        
        try:
            if model_path:
                # Load custom CPSAM model
                self.model = models.CellposeModel(
                    gpu=torch.cuda.is_available(), 
                    pretrained_model=model_path
                )
                self.model_type = "cpsam"
            else:
                # Use default Cellpose cyto model (good for phase-contrast whole cells)
                self.model = models.CellposeModel(
                    gpu=torch.cuda.is_available(),
                    model_type='cyto'  # 'cyto' is trained for whole cell segmentation in phase-contrast
                )
                self.model_type = "cyto"
                
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
                        gpu=False,
                        pretrained_model=model_path
                    )
                    self.model_type = "cpsam"
                else:
                    self.model = models.CellposeModel(
                        gpu=False,
                        model_type='cyto'
                    )
                    self.model_type = "cyto"
                print("Cell_Segmenter_Tool: CPU model initialized successfully")
            except Exception as cpu_e:
                print(f"Cell_Segmenter_Tool: Error initializing model even on CPU: {cpu_e}")
                raise

    def execute(self, image, diameter=30, flow_threshold=0.4, cellprob_threshold=0, model_type="cpsam", query_cache_dir=None, image_id=None):
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
        try:
            # If model_type is specified and different from current, reload model
            if model_type != self.model_type and model_type == "cpsam":
                # Try to reload CPSAM model
                try:
                    model_path = hf_hub_download(
                        repo_id="5xuekun/cell-segmenter-cpsam-model",  # Should match __init__ repo_id
                        filename="cpsam",  # Model filename is 'cpsam'
                        token=os.getenv("HUGGINGFACE_TOKEN")
                    )
                    self.model = models.CellposeModel(
                        gpu=torch.cuda.is_available(),
                        pretrained_model=model_path
                    )
                    self.model_type = "cpsam"
                    print(f"Cell_Segmenter_Tool: Switched to CPSAM model")
                except Exception as e:
                    print(f"Cell_Segmenter_Tool: Could not load CPSAM model: {e}, using current model")
            
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
            
            if is_tiff:
                # Try loading with tifffile first to handle multi-channel TIFF
                try:
                    img_full = tifffile.imread(image_path)
                    # Check if multi-channel (shape: (H, W, C) or (C, H, W) or (Z, H, W, C))
                    if img_full.ndim == 3:
                        # Check if last dimension is channels (typical: H, W, C)
                        if img_full.shape[2] <= 4:  # Likely channels in last dimension
                            print(f"Detected multi-channel TIFF with {img_full.shape[2]} channels. Using first channel (bright-field) for segmentation.")
                            img = img_full[:, :, 0]  # Extract first channel (bright-field)
                        elif img_full.shape[0] <= 4:  # Likely channels in first dimension (C, H, W)
                            print(f"Detected multi-channel TIFF with {img_full.shape[0]} channels. Using first channel (bright-field) for segmentation.")
                            img = img_full[0, :, :]  # Extract first channel (bright-field)
                        else:
                            # Assume 2D + depth, use first slice
                            img = img_full[:, :, 0] if img_full.shape[2] < img_full.shape[0] else img_full[0, :, :]
                    elif img_full.ndim == 4:
                        # 4D: could be (Z, H, W, C) or (C, Z, H, W)
                        print(f"Detected 4D TIFF. Using first channel of first slice for segmentation.")
                        if img_full.shape[3] <= 4:  # (Z, H, W, C)
                            img = img_full[0, :, :, 0]
                        else:  # (C, Z, H, W)
                            img = img_full[0, 0, :, :]
                    else:
                        # 2D grayscale
                        img = img_full
                except Exception as tiff_error:
                    print(f"Warning: Failed to load TIFF with tifffile: {tiff_error}, trying cv2")
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                # For non-TIFF files, use cv2
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                # Try alternative: use PIL to load and convert
                try:
                    pil_img = Image.open(image_path)
                    if pil_img.mode != 'L':
                        pil_img = pil_img.convert('L')
                    img = np.array(pil_img)
                    print(f"Loaded image using PIL: {image_path}")
                except Exception as pil_error:
                    return {
                        "error": f"Failed to load image: {image_path}. All loading methods failed: {str(pil_error)}",
                        "summary": "Image loading failed with all methods"
                    }
            
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
            masks, flows, styles = self.model.eval(
                [img],
                diameter=diameter,
                channels=[0, 0],  # Grayscale image
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold
            )
            
            mask = masks[0]
            overlay = plot.mask_overlay(img, mask)
            
            # Setup output directory using centralized configuration
            output_dir = VisualizationConfig.get_output_dir(query_cache_dir)
            
            # Save overlay visualization with professional styling using image identifier
            output_path = os.path.join(output_dir, f"cell_overlay_{image_identifier}.png")
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
            
            # Save mask as separate visualization with professional styling using image identifier
            # Use .tif format with 16-bit depth to preserve all label values (supports up to 65535 cells)
            mask_path = os.path.join(output_dir, f"cell_mask_{image_identifier}.tif")
            
            # Save the original mask array as 16-bit TIFF to preserve all label values
            # Cellpose masks are integer labels (0=background, 1-N for N cells), which can exceed 255
            mask_uint16 = mask.astype(np.uint16)
            tifffile.imwrite(mask_path, mask_uint16)
            
            # Also save a visualization version for display with professional styling
            viz_mask_path = os.path.join(output_dir, f"cell_mask_viz_{image_identifier}.png")
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
                
            return {
                "summary": f"{n_cells} cells identified and segmented successfully.",
                "cell_count": n_cells,
                "visual_outputs": [output_path, mask_path],
                "image_id": image_identifier,  # Add image_id for matching in downstream tools
                "model_used": f"CellposeModel ({self.model_type})",
                "parameters": {
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "model_type": self.model_type
                }
            }
            
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
    
    cd octotools/tools/cell_segmenter
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
    relative_image_path = "../../../examples/fibroblast.png"
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
