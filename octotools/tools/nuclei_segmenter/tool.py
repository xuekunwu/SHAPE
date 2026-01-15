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
import tempfile
from octotools.models.image_data import ImageData
from octotools.utils.image_processor import ImageProcessor

def check_image_quality(img, min_brightness=50, max_brightness=200, max_cv_threshold=0.3):
    """
    Check image quality: brightness and illumination uniformity.
    
    Args:
        img: Input image as numpy array (uint8, 0-255)
        min_brightness: Minimum acceptable mean brightness (default: 50)
        max_brightness: Maximum acceptable mean brightness (default: 200)
        max_cv_threshold: Maximum acceptable coefficient of variation for illumination (default: 0.3)
        
    Returns:
        tuple: (needs_preprocessing: bool, reason: str, stats: dict)
    """
    if img is None or img.size == 0:
        return True, "Empty image", {}
    
    # Ensure image is uint8
    if img.dtype != np.uint8:
        if img.dtype == np.uint16:
            img = (img / 65535.0 * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Calculate mean brightness
    mean_brightness = np.mean(img)
    
    # Calculate illumination uniformity using coefficient of variation
    # Divide image into blocks and calculate local brightness
    h, w = img.shape[:2]
    block_size = min(64, h // 4, w // 4)  # Adaptive block size
    if block_size < 8:
        block_size = 8
    
    local_brightness = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:min(i+block_size, h), j:min(j+block_size, w)]
            if block.size > 0:
                local_brightness.append(np.mean(block))
    
    if len(local_brightness) > 1:
        local_brightness = np.array(local_brightness)
        cv_illumination = np.std(local_brightness) / (np.mean(local_brightness) + 1e-6)  # Coefficient of variation
    else:
        cv_illumination = 0.0
    
    # Determine if preprocessing is needed
    needs_preprocessing = False
    reasons = []
    
    if mean_brightness < min_brightness:
        needs_preprocessing = True
        reasons.append(f"too dark (brightness: {mean_brightness:.1f} < {min_brightness})")
    elif mean_brightness > max_brightness:
        needs_preprocessing = True
        reasons.append(f"too bright (brightness: {mean_brightness:.1f} > {max_brightness})")
    
    if cv_illumination > max_cv_threshold:
        needs_preprocessing = True
        reasons.append(f"uneven illumination (CV: {cv_illumination:.3f} > {max_cv_threshold})")
    
    reason = "; ".join(reasons) if reasons else "good quality"
    
    stats = {
        "mean_brightness": float(mean_brightness),
        "cv_illumination": float(cv_illumination),
        "needs_preprocessing": needs_preprocessing
    }
    
    return needs_preprocessing, reason, stats

class Nuclei_Segmenter_Tool(BaseTool):
    def __init__(self, model_path=None):
        super().__init__(
            tool_name="Nuclei_Segmenter_Tool",
            tool_description="Segments nuclei in microscopy images using Cellpose model. Provides cell counting and visualization.",
            tool_version="1.0.0",
            input_types={
                "image": "str - Path to the input image (supports .tif, .png, .jpg formats).",
                "diameter": "float - Expected cell diameter in pixels (default: None, auto-detect).",
                "flow_threshold": "float - Flow threshold for cell detection (default: 0.4).",
                "cellprob_threshold": "float - Cell probability threshold (default: 0)."
            },
            output_type="dict - Contains segmentation results, cell count, and visualization paths.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/cells.tif")',
                    "description": "Segment nuclei in a microscopy image with default parameters."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/cells.tif", diameter=30, flow_threshold=0.5)',
                    "description": "Segment nuclei with custom diameter and flow threshold parameters."
                }
            ],
            user_metadata={
                "limitation": "Requires GPU for optimal performance. May struggle with very dense cell populations or poor quality images. Model download required on first use.",
                "best_practice": "Use with Image_Preprocessor_Tool for better results on low-quality images. Adjust diameter parameter based on your cell type and image resolution."
            },
            output_dir="output_visualizations"  # Set default output directory to output_visualizations
        )
        
        # Enable GPU if available (fall back to CPU gracefully)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Nuclei_Segmenter_Tool: Using device: {self.device}")
        
        # Download model from Hugging Face Hub
        if model_path is None:
            try:
                model_path = hf_hub_download(
                    repo_id="5xuekun/nuclei-segmenter-model",
                    filename="cpsam_lr_1e-04",
                    token=os.getenv("HUGGINGFACE_TOKEN")
                )
                print(f"Nuclei_Segmenter_Tool: Model downloaded to {model_path}")
            except Exception as e:
                print(f"Nuclei_Segmenter_Tool: Failed to download model: {e}")
                # Fallback to default Cellpose model
                model_path = None
        
        try:
            self.model = models.CellposeModel(
                gpu=torch.cuda.is_available(), 
                pretrained_model=model_path
            )
            if self.model is None:
                raise ValueError("Failed to initialize CellposeModel")
            print(f"Nuclei_Segmenter_Tool: Model initialized successfully")
        except Exception as e:
            # Retry on CPU if GPU path fails (e.g., NVML issues)
            try:
                print(f"Nuclei_Segmenter_Tool: GPU init failed ({e}), retrying on CPU")
                self.device = torch.device('cpu')
                self.model = models.CellposeModel(
                    gpu=False,
                    pretrained_model=model_path
                )
                print("Nuclei_Segmenter_Tool: CPU model initialized successfully")
            except Exception as cpu_e:
                print(f"Nuclei_Segmenter_Tool: Error initializing model even on CPU: {cpu_e}")
                raise

    def execute(self, image, diameter=25, flow_threshold=0.6, cellprob_threshold=0, query_cache_dir=None, image_id=None):
        """
        Execute nuclei segmentation on the input image.
        
        Args:
            image (str): Path to the input image
            diameter (float): Expected cell diameter in pixels
            flow_threshold (float): Flow threshold for cell detection
            cellprob_threshold (float): Cell probability threshold
            query_cache_dir (str): Directory to save outputs
            image_id (str): Optional image identifier for consistent file naming
            
        Returns:
            dict: Segmentation results with cell count and visualization paths
        """
        try:
            # Normalize path for cross-platform compatibility
            image_path = os.path.normpath(image) if isinstance(image, str) else str(image)
            
            # Extract image identifier: use provided image_id, or extract from filename
            if image_id:
                image_identifier = image_id
            else:
                # Fallback: extract from filename (e.g., "a23b168beeda4d69b4e75fab887e349e.jpg" -> "a23b168beeda4d69b4e75fab887e349e")
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
                
                # Extract first channel for segmentation (bright-field/phase contrast)
                # Convert to float32 format required by Cellpose
                img = img_data.to_segmentation_input(channel_idx=0)
                
                if img_data.is_multi_channel:
                    print(f"Detected multi-channel image with {img_data.num_channels} channels. Using first channel ({img_data.channel_names[0] if img_data.channel_names else 'channel 0'}) for segmentation.")
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
            
            # Ensure img is uint8 for quality check
            img_uint8 = img.copy()
            if img_uint8.dtype != np.uint8:
                if img_uint8.dtype == np.uint16:
                    img_uint8 = (img_uint8 / 65535.0 * 255).astype(np.uint8)
                else:
                    img_uint8 = np.clip(img_uint8, 0, 255).astype(np.uint8)
            
            # Check image quality and apply preprocessing if needed
            needs_preprocessing, quality_reason, quality_stats = check_image_quality(img_uint8)
            if needs_preprocessing:
                print(f"⚠️ Image quality check: {quality_reason}")
                print(f"   Brightness: {quality_stats['mean_brightness']:.1f}, Illumination CV: {quality_stats['cv_illumination']:.3f}")
                print(f"   Auto-applying Image_Preprocessor_Tool for better segmentation results...")
                
                # Import and use Image_Preprocessor_Tool
                from octotools.tools.image_preprocessor.tool import Image_Preprocessor_Tool
                preprocessor = Image_Preprocessor_Tool()
                
                # Save image temporarily for preprocessing
                temp_dir = tempfile.mkdtemp() if query_cache_dir is None else os.path.join(query_cache_dir, "temp_preprocess")
                os.makedirs(temp_dir, exist_ok=True)
                temp_img_path = os.path.join(temp_dir, f"image_{image_identifier}.png")
                Image.fromarray(img_uint8, mode='L').save(temp_img_path)
                
                try:
                    # Apply preprocessing
                    preprocess_result = preprocessor.execute(
                        image=temp_img_path,
                        target_brightness=120,
                        gaussian_kernel_size=151,
                        output_format='png',
                        save_intermediate=False,
                        image_id=f"{image_identifier}_preprocessed",
                        query_cache_dir=query_cache_dir
                    )
                    
                    # Load preprocessed image
                    if isinstance(preprocess_result, dict) and "processed_image_path" in preprocess_result:
                        preprocessed_path = preprocess_result["processed_image_path"]
                        if os.path.exists(preprocessed_path):
                            preprocessed_img = cv2.imread(preprocessed_path, cv2.IMREAD_GRAYSCALE)
                            if preprocessed_img is not None:
                                img = preprocessed_img
                                print(f"✅ Successfully applied preprocessing. New brightness: {np.mean(img):.1f}")
                            else:
                                print(f"⚠️ Failed to load preprocessed image, using original")
                        else:
                            print(f"⚠️ Preprocessed image not found, using original")
                    else:
                        print(f"⚠️ Preprocessing returned unexpected result, using original")
                except Exception as preprocess_error:
                    print(f"⚠️ Error during automatic preprocessing: {preprocess_error}")
                    print(f"   Continuing with original image...")
                finally:
                    # Clean up temp file
                    try:
                        if os.path.exists(temp_img_path):
                            os.remove(temp_img_path)
                    except:
                        pass
            else:
                print(f"✅ Image quality check passed: {quality_reason}")
                print(f"   Brightness: {quality_stats['mean_brightness']:.1f}, Illumination CV: {quality_stats['cv_illumination']:.3f}")
            
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
            output_path = os.path.join(output_dir, f"nuclei_overlay_{image_identifier}.png")
            fig, ax = VisualizationConfig.create_professional_figure(figsize=(12, 8))
            # Ensure overlay has same brightness as original image
            overlay_normalized = overlay.astype(np.float32) / 255.0
            ax.imshow(overlay_normalized, vmin=0, vmax=1)
            ax.axis('off')
            VisualizationConfig.apply_professional_styling(
                ax, title=f"Nuclei Segmentation - {len(np.unique(mask))-1} cells detected"
            )
            VisualizationConfig.save_professional_figure(fig, output_path)
            plt.close(fig)
            
            # Count nuclei (unique mask values, excluding background 0)
            n_nuclei = len(np.unique(mask)) - 1 if mask is not None else 0
            
            # Save mask as separate visualization with professional styling using image identifier
            # Use .tif format with 16-bit depth to preserve all label values (supports up to 65535 nuclei)
            mask_path = os.path.join(output_dir, f"nuclei_mask_{image_identifier}.tif")
            
            # Save the original mask array as 16-bit TIFF to preserve all label values
            # Cellpose masks are integer labels (0=background, 1-N for N nuclei), which can exceed 255
            mask_uint16 = mask.astype(np.uint16)
            tifffile.imwrite(mask_path, mask_uint16)
            
            # Also save a visualization version for display with professional styling
            viz_mask_path = os.path.join(output_dir, f"nuclei_mask_viz_{image_identifier}.png")
            fig, ax = VisualizationConfig.create_professional_figure(figsize=(12, 8))
            ax.imshow(mask, cmap='tab20')
            ax.axis('off')
            VisualizationConfig.apply_professional_styling(
                ax, title=f"Cell Masks - {n_nuclei} cells"
            )
            VisualizationConfig.save_professional_figure(fig, viz_mask_path)
            plt.close(fig)
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return {
                "summary": f"{n_nuclei} cells identified and segmented successfully.",
                "cell_count": n_nuclei,
                "visual_outputs": [output_path, mask_path],
                "deliverables": [output_path],  # Add overlay to deliverables for better collection
                "mask_path": mask_path,  # Explicitly include mask_path for downstream tools
                "overlay_path": output_path,  # Explicit overlay_path key for collection
                "output_path": output_path,  # Explicit output_path key for collection
                "image_id": image_identifier,  # Add image_id for matching in downstream tools
                "image_identifier": image_identifier,  # Alias for compatibility
                "model_used": f"CellposeModel ({self.model.pretrained_model})",
                "parameters": {
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold
                }
            }
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            print(f"Nuclei_Segmenter_Tool: Error in nuclei segmentation: {error_msg}")
            print(f"Nuclei_Segmenter_Tool: Traceback: {error_traceback}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "error": f"Error during nuclei segmentation: {error_msg}",
                "summary": "Failed to process image",
                "error_details": {
                    "image_path": image_path if 'image_path' in locals() else str(image),
                    "diameter": diameter if 'diameter' in locals() else "unknown",
                    "flow_threshold": flow_threshold if 'flow_threshold' in locals() else "unknown",
                    "cellprob_threshold": cellprob_threshold if 'cellprob_threshold' in locals() else "unknown"
                }
            }

    def get_metadata(self):
        """Returns the metadata for the Nuclei_Segmenter_Tool."""
        metadata = super().get_metadata()
        metadata["device"] = str(self.device)
        metadata["model_loaded"] = self.model is not None
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/nuclei_segmenter
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Nuclei_Segmenter_Tool
    tool = Nuclei_Segmenter_Tool()
    tool.set_custom_output_dir("nuclei_outputs")

    # Get tool metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(metadata)

    # Construct the full path to the image using the script's directory
    relative_image_path = "examples/fibroblast.png"
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
