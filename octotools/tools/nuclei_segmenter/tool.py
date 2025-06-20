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
            }
        )
        
        # Enable GPU if available
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
            print(f"Nuclei_Segmenter_Tool: Error initializing model: {e}")
            raise

    def execute(self, image, diameter=25, flow_threshold=0.6, cellprob_threshold=0, query_cache_dir=None):
        """
        Execute nuclei segmentation on the input image.
        
        Args:
            image (str): Path to the input image
            diameter (float): Expected cell diameter in pixels
            flow_threshold (float): Flow threshold for cell detection
            cellprob_threshold (float): Cell probability threshold
            query_cache_dir (str): Directory to save outputs
            
        Returns:
            dict: Segmentation results with cell count and visualization paths
        """
        try:
            # Load and preprocess image
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {
                    "error": f"Failed to load image: {image}",
                    "summary": "Image loading failed"
                }
            
            img = img.astype(np.float32)
            
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
            
            # Setup output directory
            if query_cache_dir is None:
                query_cache_dir = "solver_cache/temp"
            tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
            os.makedirs(tool_cache_dir, exist_ok=True)
            
            # Save overlay visualization
            output_path = os.path.join(tool_cache_dir, f"nuclei_overlay_{uuid4().hex[:8]}.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(overlay)
            plt.axis('off')
            plt.title(f"Nuclei Segmentation - {len(np.unique(mask))-1} cells detected")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            
            # Count nuclei (unique mask values, excluding background 0)
            n_nuclei = len(np.unique(mask)) - 1 if mask is not None else 0
            
            # Save mask as separate visualization
            mask_path = os.path.join(tool_cache_dir, f"nuclei_mask_{uuid4().hex[:8]}.png")
            
            # Save the original mask array (not matplotlib visualization)
            # This ensures Single_Cell_Cropper_Tool can properly process it
            cv2.imwrite(mask_path, mask.astype(np.uint8))
            
            # Also save a visualization version for display
            viz_mask_path = os.path.join(tool_cache_dir, f"nuclei_mask_viz_{uuid4().hex[:8]}.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(mask, cmap='tab20')
            plt.axis('off')
            plt.title(f"Cell Masks - {n_nuclei} cells")
            plt.savefig(viz_mask_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return {
                "summary": f"{n_nuclei} cells identified and segmented successfully.",
                "cell_count": n_nuclei,
                "visual_outputs": [output_path, mask_path, viz_mask_path],
                "model_used": f"CellposeModel ({self.model.pretrained_model})",
                "parameters": {
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold
                }
            }
            
        except Exception as e:
            print(f"Nuclei_Segmenter_Tool: Error in nuclei segmentation: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "error": f"Error during nuclei segmentation: {str(e)}",
                "summary": "Failed to process image"
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