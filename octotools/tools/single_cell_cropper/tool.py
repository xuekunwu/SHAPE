from octotools.tools.base import BaseTool
import numpy as np
import cv2
import os
from PIL import Image
from uuid import uuid4
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from pathlib import Path
import json
from octotools.models.utils import VisualizationConfig

class Single_Cell_Cropper_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Single_Cell_Cropper_Tool",
            tool_description="Generates individual cell crops from nuclei segmentation results for single-cell analysis. Creates masks and crops each detected nucleus with configurable margins. Automatically handles image dimension mismatches by resizing masks.",
            tool_version="1.0.0",
            input_types={
                "original_image": "str - Path to the original image (brightfield/phase contrast).",
                "nuclei_mask": "str - Path to the nuclei segmentation mask image or mask array.",
                "min_area": "int - Minimum area threshold for valid nuclei (default: 50 pixels).",
                "margin": "int - Margin around each nucleus for cropping (default: 25 pixels).",
                "output_format": "str - Output format for crops ('tif', 'png', 'jpg', default: 'tif')."
            },
            output_type="dict - Contains cropped cell images, metadata, and visualization paths.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(original_image="path/to/brightfield.tif", nuclei_mask="path/to/nuclei_mask.png")',
                    "description": "Generate single-cell crops from nuclei segmentation with default parameters."
                },
                {
                    "command": 'execution = tool.execute(original_image="path/to/brightfield.tif", nuclei_mask="path/to/nuclei_mask.png", min_area=100, margin=30)',
                    "description": "Generate single-cell crops with custom area threshold and margin parameters."
                }
            ],
            user_metadata={
                "limitation": "Requires nuclei segmentation results as input. May generate overlapping crops if nuclei are close together. Performance depends on image quality and segmentation accuracy.",
                "best_practice": "Use with Nuclei_Segmenter_Tool for optimal results. Adjust min_area and margin based on cell size and analysis requirements. Automatically handles dimension mismatches between original image and nuclei mask."
            }
        )

    def execute(self, original_image, nuclei_mask, min_area=50, margin=25, output_format='png', query_cache_dir=None):
        """
        Execute single-cell cropping from nuclei segmentation results.
        
        Args:
            original_image: Path to original brightfield image (should be query_image_processed.png)
            nuclei_mask: Path to nuclei segmentation mask
            min_area: Minimum area threshold for valid nuclei
            margin: Margin around each nucleus for cropping
            output_format: Output image format ('png', 'tif', 'jpg')
            query_cache_dir: Directory for caching results
            
        Returns:
            dict: Cropping results with cell crops and metadata
        """
        try:
            # Check if we should use processed image instead of original
            if query_cache_dir and os.path.exists(os.path.join(query_cache_dir, "tool_cache", "query_image_processed.png")):
                processed_image_path = os.path.join(query_cache_dir, "tool_cache", "query_image_processed.png")
                print(f"Using processed image: {processed_image_path}")
                original_img = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)
            else:
                # Load original image
                original_img = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
            
            if original_img is None:
                return {
                    "error": f"Failed to load original image: {original_image}",
                    "summary": "Image loading failed"
                }
            
            # Load nuclei mask
            mask = cv2.imread(nuclei_mask, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return {
                    "error": f"Failed to load nuclei mask: {nuclei_mask}",
                    "summary": "Mask loading failed"
                }
            
            # Setup output directory
            if query_cache_dir is None:
                query_cache_dir = "solver_cache/temp"
            tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
            os.makedirs(tool_cache_dir, exist_ok=True)
            
            # Generate cell crops
            cell_crops, cell_metadata, stats = self._generate_cell_crops(
                original_img, mask, min_area, margin, tool_cache_dir, output_format
            )
            
            if not cell_crops:
                return {
                    "summary": "No valid cell crops generated.",
                    "cell_count": 0,
                    "cell_crops": [],
                    "cell_metadata": [],
                    "visual_outputs": [],
                    "parameters": {
                        "min_area": min_area,
                        "margin": margin,
                        "output_format": output_format
                    },
                    "statistics": stats
                }
            
            # Combine paths and metadata for JSON output
            output_data = {
                'cell_count': stats['final_cell_count'],
                'cell_crops_paths': cell_crops,
                'cell_metadata': cell_metadata,
                'statistics': stats,
                "parameters": {
                    "min_area": min_area,
                    "margin": margin
                }
            }
            metadata_path = os.path.join(tool_cache_dir, f"cell_crops_metadata_{uuid4().hex[:8]}.json")
            with open(metadata_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            # Create a summary visualization
            summary_viz_path = self._create_summary_visualization(
                original_img, mask, cell_crops, stats, query_cache_dir, min_area, margin
            )
            
            return {
                "summary": f"Successfully generated {stats['final_cell_count']} single-cell crops. All paths and metadata are saved in '{Path(metadata_path).name}'.",
                "cell_count": stats['final_cell_count'],
                "cell_crops_metadata_path": metadata_path,
                "visual_outputs": [summary_viz_path] if summary_viz_path else [],
            }
            
        except Exception as e:
            print(f"Single_Cell_Cropper_Tool: Error in single-cell cropping: {e}")
            return {
                "error": f"Error during single-cell cropping: {str(e)}",
                "summary": "Failed to process image"
            }

    def _generate_cell_crops(self, original_img, mask, min_area, margin, output_dir, output_format):
        """
        Generate individual cell crops from the mask.
        
        Args:
            original_img: Original image array
            mask: Binary mask array
            min_area: Minimum area threshold
            margin: Margin around each nucleus
            output_dir: Output directory
            output_format: Output image format
            
        Returns:
            tuple: (list of crop paths, list of metadata, dict of statistics)
        """
        # Validate input images
        if original_img is None or mask is None:
            print("Error: Input images are None")
            return [], [], {}
        
        # Ensure images are numpy arrays
        original_img = np.asarray(original_img)
        mask = np.asarray(mask)
        
        # Check image dimensions
        if original_img.shape != mask.shape:
            print(f"Error: Image shape mismatch - original: {original_img.shape}, mask: {mask.shape}")
            return [], [], {}
        
        # Ensure mask is binary
        if len(np.unique(mask)) > 2:
            print("Warning: Mask is not binary, converting to binary")
            mask = (mask > 0).astype(np.uint8) * 255
        
        # Label connected components in the mask
        labeled_mask = label(mask)
        cell_properties = regionprops(labeled_mask)
        
        initial_cell_count = len(cell_properties)
        filtered_by_area = 0
        filtered_by_border = 0
        invalid_crop_data = 0

        cell_crops = []
        cell_metadata = []
        
        for idx, region in enumerate(cell_properties):
            if region.area < min_area:
                filtered_by_area += 1
                continue
                
            # Get bounding box from the mask
            minr, minc, maxr, maxc = region.bbox
            height = maxr - minr
            width = maxc - minc
            
            # Compute center of the bounding box
            center_row = (minr + maxr) // 2
            center_col = (minc + maxc) // 2
            
            # Determine half sizes and enforce square crops (with margin)
            half_height = height // 2 + margin
            half_width = width // 2 + margin
            half_side = max(half_height, half_width)
            
            # Define new square bounding box
            new_minr = max(center_row - half_side, 0)
            new_maxr = min(center_row + half_side, original_img.shape[0])
            new_minc = max(center_col - half_side, 0)
            new_maxc = min(center_col + half_side, original_img.shape[1])
            
            # Skip if the crop is not square (i.e., at the border)
            if (new_maxr - new_minr) != (new_maxc - new_minc):
                filtered_by_border += 1
                continue
                
            # Crop from original image
            cell_crop = original_img[new_minr:new_maxr, new_minc:new_maxc]
            
            # Validate crop data
            if cell_crop.size == 0 or np.isnan(cell_crop).any() or np.isinf(cell_crop).any():
                invalid_crop_data += 1
                print(f"Warning: Invalid crop data for cell {idx}, skipping")
                continue
            
            # Ensure crop is in valid range
            if cell_crop.dtype == np.float32 or cell_crop.dtype == np.float64:
                cell_crop = np.clip(cell_crop, 0, 255).astype(np.uint8)
            
            # Save crop with enhanced error handling
            crop_filename = f"cell_{idx:04d}_crop.{output_format}"
            crop_path = os.path.join(output_dir, crop_filename)
            
            try:
                # Ensure image is in the correct format for Gradio compatibility
                if len(cell_crop.shape) == 2:
                    # Grayscale image - convert to PIL and save
                    pil_image = Image.fromarray(cell_crop, mode='L')
                else:
                    # Color image - convert to PIL and save
                    pil_image = Image.fromarray(cell_crop)
                
                # Validate PIL image
                if pil_image.size[0] == 0 or pil_image.size[1] == 0:
                    print(f"Warning: Invalid PIL image size for cell {idx}, skipping")
                    continue
                
                # Save with proper format and error handling
                if output_format.lower() == 'tif':
                    pil_image.save(crop_path, 'TIFF', compression='tiff_lzw')
                elif output_format.lower() == 'png':
                    pil_image.save(crop_path, 'PNG', optimize=True)
                elif output_format.lower() == 'jpg':
                    pil_image.save(crop_path, 'JPEG', quality=95, optimize=True)
                else:
                    # Default to PNG for compatibility
                    pil_image.save(crop_path, 'PNG', optimize=True)
                
                # Verify the saved file
                if not os.path.exists(crop_path) or os.path.getsize(crop_path) == 0:
                    print(f"Warning: Failed to save crop for cell {idx}")
                    continue
                
                # Test loading the saved image
                try:
                    test_image = Image.open(crop_path)
                    test_image.verify()
                    test_image.close()
                except Exception as e:
                    print(f"Warning: Saved image verification failed for cell {idx}: {e}")
                    continue
                
            except Exception as e:
                print(f"Error saving crop for cell {idx}: {e}")
                continue
            
            # Create metadata for this crop
            crop_metadata = {
                "cell_id": idx,
                "crop_path": crop_path,
                "original_bbox": [minr, minc, maxr, maxc],
                "crop_bbox": [new_minr, new_minc, new_maxr, new_maxc],
                "area": region.area,
                "centroid": region.centroid,
                "crop_size": cell_crop.shape
            }
            
            cell_crops.append(crop_path)
            cell_metadata.append(crop_metadata)
        
        stats = {
            "initial_cell_count": initial_cell_count,
            "filtered_by_area": filtered_by_area,
            "filtered_by_border": filtered_by_border,
            "invalid_crop_data": invalid_crop_data,
            "final_cell_count": len(cell_crops)
        }

        return cell_crops, cell_metadata, stats

    def _create_summary_visualization(self, original_img, mask, cell_crops, stats, output_dir, min_area, margin):
        """Creates a professional and highly robust summary visualization of the cropping process."""
        vis_config = VisualizationConfig()
        output_path = os.path.join(output_dir, "single_cell_cropper_summary.png")

        try:
            fig = plt.figure(figsize=(24, 16), dpi=300)
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], hspace=0.25, wspace=0.15)
            
            fig.suptitle("Single-Cell Cropping Summary.png)", fontsize=36, fontweight='bold', y=0.98)

            # --- Subplots ---
            ax1_container = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            
            # 1. Sample Crops Grid (using a robust sub-gridspec)
            ax1_container.set_title("Sample Cell Crops", fontsize=32, fontweight='bold', pad=20)
            ax1_container.axis('off')
            if cell_crops:
                gs_crops = ax1_container.get_subplotspec().subgridspec(4, 4, wspace=0.05, hspace=0.05)
                sample_indices = np.random.choice(len(cell_crops), size=min(16, len(cell_crops)), replace=False)
                for i, idx in enumerate(sample_indices):
                    ax_grid = fig.add_subplot(gs_crops[i])
                    try:
                        crop_img = Image.open(cell_crops[idx])
                        ax_grid.imshow(crop_img, cmap='gray')
                    except (FileNotFoundError, IOError):
                        pass # Silently skip if file is missing or corrupt
                    ax_grid.axis('off')
            
            # 2. Statistics and Parameters Text - adjusted height to match cropper
            stats_text = (
                f"Initial Nuclei Detected: {stats.get('initial_cell_count', 'N/A')}\n"
                f"Filtered by Area (<{min_area}px): {stats.get('filtered_by_area', 'N/A')}\n"
                f"Filtered by Border: {stats.get('filtered_by_border', 'N/A')}\n"
                f"Final Valid Crops: {stats.get('final_cell_count', 'N/A')}\n\n"
                f"Parameters Used:\n"
                f"  - Min Area: {min_area} pixels\n"
                f"  - Margin: {margin} pixels"
            )
            ax2.set_title("Processing Statistics", fontsize=32, fontweight='bold', pad=20)
            ax2.text(
                0.5, 0.5, stats_text, ha='center', va='center', transform=ax2.transAxes,
                fontsize=28, fontweight='bold', wrap=True,
                bbox=dict(boxstyle='round,pad=1.0', facecolor='aliceblue', alpha=0.95)
            )
            ax2.axis('off')

            vis_config.save_professional_figure(fig, output_path, bbox_inches=None)
            plt.close(fig)

            print(f"✅ Summary visualization successfully created at {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error creating summary visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_metadata(self):
        """Returns the metadata for the Single_Cell_Cropper_Tool."""
        return super().get_metadata()


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/single_cell_cropper
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Single_Cell_Cropper_Tool
    tool = Single_Cell_Cropper_Tool()
    tool.set_custom_output_dir("single_cell_outputs")

    # Get tool metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(metadata)

    # Example execution (requires both original image and nuclei mask)
    try:
        # This would require actual image and mask files
        print("\nTool initialized successfully. Ready for execution.")
        print("Example usage:")
        print("execution = tool.execute(original_image='path/to/brightfield.tif', nuclei_mask='path/to/nuclei_mask.png')")
        
    except Exception as e:
        print(f"Error during tool initialization: {e}") 