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
            original_image: Path to original brightfield image
            nuclei_mask: Path to nuclei segmentation mask
            min_area: Minimum area threshold for valid nuclei
            margin: Margin around each nucleus for cropping
            output_format: Output image format ('png', 'tif', 'jpg')
            query_cache_dir: Directory for caching results
            
        Returns:
            dict: Cropping results with cell crops and metadata
        """
        try:
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
            
            # Save metadata
            metadata_path = os.path.join(tool_cache_dir, f"cell_crops_metadata_{uuid4().hex[:8]}.json")
            with open(metadata_path, 'w') as f:
                json.dump(cell_metadata, f, indent=2)
            
            # Create a summary visualization instead of individual crop display
            summary_viz_path = self._create_summary_visualization(
                cell_crops, cell_metadata, stats, tool_cache_dir, min_area, margin
            )
            
            return {
                "summary": f"Successfully generated {len(cell_crops)} single-cell crops for downstream analysis.",
                "cell_count": len(cell_crops),
                "cell_crops": cell_crops,
                "cell_metadata": cell_metadata,
                "visual_outputs": [summary_viz_path, metadata_path] if summary_viz_path else [metadata_path],
                "parameters": {
                    "min_area": min_area,
                    "margin": margin,
                    "output_format": output_format
                },
                "statistics": stats
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

    def _create_summary_visualization(self, cell_crops, cell_metadata, stats, output_dir, min_area, margin):
        """
        Create a compact and well-structured summary visualization.
        """
        try:
            if not cell_crops:
                return None

            # --- Layout Setup ---
            fig = plt.figure(figsize=(20, 24))
            gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 2, 2], hspace=0.3)
            
            # Top part: Summary Text and Histogram
            gs_top = gs[0].subgridspec(1, 2, wspace=0.2)
            ax_text = fig.add_subplot(gs_top[0])
            ax_hist = fig.add_subplot(gs_top[1])

            # Bottom part: Title and Crop Grid
            gs_bottom = gs[1:].subgridspec(5, 1, height_ratios=[0.2, 1, 1, 1, 1], hspace=0)
            ax_crop_title = fig.add_subplot(gs_bottom[0])
            
            # --- Content Generation ---
            
            # Top Left: Summary Text (Corrected newlines)
            ax_text.axis('off')
            summary_text = (
                f"Single-Cell Cropping Summary\n"
                f"---------------------------------\n"
                f"Initial detected objects: {stats['initial_cell_count']}\n"
                f"Filtered (area < {min_area} px): {stats['filtered_by_area']}\n"
                f"Filtered (border objects): {stats['filtered_by_border']}\n"
                f"Filtered (invalid data): {stats['invalid_crop_data']}\n"
                f"---------------------------------\n"
                f"Final valid cells: {stats['final_cell_count']}\n\n"
                f"Crop Parameters:\n"
                f" - Min Area Threshold: {min_area} pixels\n"
                f" - Margin: {margin} pixels"
            )
            ax_text.text(0, 0.95, summary_text, transform=ax_text.transAxes,
                         fontsize=16, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9))

            # Top Right: Cell Area Distribution
            cell_areas = [md.get('area', 0) for md in cell_metadata]
            ax_hist.hist(cell_areas, bins=20, color='skyblue', edgecolor='black')
            ax_hist.set_title(f"Cell Area Distribution (Total: {stats['final_cell_count']} cells)", fontsize=18)
            ax_hist.set_xlabel("Cell Area (pixels)", fontsize=14)
            ax_hist.set_ylabel("Number of Cells", fontsize=14)
            ax_hist.tick_params(axis='both', which='major', labelsize=12)
            ax_hist.grid(True, linestyle='--', alpha=0.6)

            # Crop Grid Title
            ax_crop_title.text(0.5, 0.7, f"Randomly Selected Single-Cell Crops (40 of {stats['final_cell_count']})",
                               ha='center', va='center', fontsize=18, style='italic', color='dimgray')
            ax_crop_title.axis('off')
            
            # 4x10 Crop Grid with 3x vertical spacing
            sample_size = min(40, len(cell_crops))
            if sample_size > 0:
                indices = np.random.choice(len(cell_crops), sample_size, replace=False)
                
                # Using hspace for vertical spacing and wspace for horizontal
                gs_crops = gs_bottom[1:].subgridspec(4, 10, hspace=0.6, wspace=0.2)

                for i in range(sample_size):
                    ax_crop = fig.add_subplot(gs_crops[i // 10, i % 10])
                    crop_path = cell_crops[indices[i]]
                    try:
                        img = Image.open(crop_path)
                        ax_crop.imshow(img, cmap='gray')
                    except FileNotFoundError:
                        ax_crop.text(0.5, 0.5, 'Not Found', ha='center', va='center')
                    
                    ax_crop.set_title(Path(crop_path).name, fontsize=10, y=-0.2)
                    ax_crop.axis('off')

            fig.suptitle("Single-Cell Cropper Analysis Report", fontsize=24, weight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.96], h_pad=3.0)

            # Save visualization
            viz_path = os.path.join(output_dir, f"cropping_summary_{uuid4().hex[:8]}.png")
            plt.savefig(viz_path, bbox_inches='tight', dpi=150, format='png')
            plt.close(fig)
            return viz_path

        except Exception as e:
            print(f"Error creating summary visualization: {e}")
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