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

    def execute(self, original_image, nuclei_mask, min_area=50, margin=25, output_format='tif', query_cache_dir=None):
        """
        Execute single-cell cropping from nuclei segmentation results.
        
        Args:
            original_image (str): Path to the original brightfield/phase contrast image
            nuclei_mask (str): Path to nuclei segmentation mask or mask array
            min_area (int): Minimum area threshold for valid nuclei
            margin (int): Margin around each nucleus for cropping
            output_format (str): Output format for crops
            query_cache_dir (str): Directory to save outputs
            
        Returns:
            dict: Cropping results with individual cell images and metadata
        """
        try:
            # Setup output directory
            if query_cache_dir is None:
                query_cache_dir = "solver_cache/temp"
            tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
            os.makedirs(tool_cache_dir, exist_ok=True)
            
            # Load original image
            if isinstance(original_image, str):
                original_img = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
                if original_img is None:
                    return {
                        "error": f"Failed to load original image: {original_image}",
                        "summary": "Original image loading failed"
                    }
            else:
                original_img = np.array(original_image)
                if len(original_img.shape) == 3:
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            
            # Load nuclei mask
            if isinstance(nuclei_mask, str):
                if nuclei_mask.endswith('.png') or nuclei_mask.endswith('.jpg') or nuclei_mask.endswith('.tif'):
                    # Load mask image
                    mask_img = cv2.imread(nuclei_mask, cv2.IMREAD_GRAYSCALE)
                    if mask_img is None:
                        return {
                            "error": f"Failed to load nuclei mask: {nuclei_mask}",
                            "summary": "Nuclei mask loading failed"
                        }
                    # Convert to binary mask (assuming non-zero values are nuclei)
                    mask = (mask_img > 0).astype(np.uint8) * 255
                else:
                    return {
                        "error": f"Unsupported mask format: {nuclei_mask}",
                        "summary": "Mask format not supported"
                    }
            else:
                # Assume nuclei_mask is already a numpy array
                mask = np.array(nuclei_mask)
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
            
            # Ensure mask is binary
            if len(np.unique(mask)) > 2:
                # If mask has multiple values, convert to binary
                mask = (mask > 0).astype(np.uint8) * 255
            
            # Check image dimensions match
            if original_img.shape != mask.shape:
                print(f"Image dimensions mismatch: original {original_img.shape} vs mask {mask.shape}")
                print("Resizing mask to match original image dimensions...")
                
                # Resize mask to match original image dimensions
                mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Ensure mask is still binary after resizing
                mask = (mask > 0).astype(np.uint8) * 255
                
                print(f"Mask resized to: {mask.shape}")
            
            # Generate individual cell crops
            cell_crops, cell_metadata = self._generate_cell_crops(
                original_img, mask, min_area, margin, tool_cache_dir, output_format
            )
            
            # Create visualization of all crops
            visualization_path = self._create_crops_visualization(
                cell_crops, cell_metadata, tool_cache_dir
            )
            
            # Save metadata
            metadata_path = os.path.join(tool_cache_dir, f"cell_crops_metadata_{uuid4().hex[:8]}.json")
            with open(metadata_path, 'w') as f:
                json.dump(cell_metadata, f, indent=2)
            
            return {
                "summary": f"Successfully generated {len(cell_crops)} single-cell crops.",
                "cell_count": len(cell_crops),
                "cell_crops": cell_crops,
                "cell_metadata": cell_metadata,
                "visual_outputs": [visualization_path, metadata_path],
                "parameters": {
                    "min_area": min_area,
                    "margin": margin,
                    "output_format": output_format
                }
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
            tuple: (list of crop paths, list of metadata)
        """
        # Validate input images
        if original_img is None or mask is None:
            print("Error: Input images are None")
            return [], []
        
        # Ensure images are numpy arrays
        original_img = np.asarray(original_img)
        mask = np.asarray(mask)
        
        # Check image dimensions
        if original_img.shape != mask.shape:
            print(f"Error: Image shape mismatch - original: {original_img.shape}, mask: {mask.shape}")
            return [], []
        
        # Ensure mask is binary
        if len(np.unique(mask)) > 2:
            print("Warning: Mask is not binary, converting to binary")
            mask = (mask > 0).astype(np.uint8) * 255
        
        # Label connected components in the mask
        labeled_mask = label(mask)
        cell_properties = regionprops(labeled_mask)
        
        cell_crops = []
        cell_metadata = []
        
        for idx, region in enumerate(cell_properties):
            if region.area < min_area:
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
            
            # Skip if the crop is not square
            if (new_maxr - new_minr) != (new_maxc - new_minc):
                continue
                
            # Crop from original image
            cell_crop = original_img[new_minr:new_maxr, new_minc:new_maxc]
            
            # Validate crop data
            if cell_crop.size == 0 or np.isnan(cell_crop).any() or np.isinf(cell_crop).any():
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
        
        return cell_crops, cell_metadata

    def _create_crops_visualization(self, cell_crops, cell_metadata, output_dir):
        """
        Create a visualization showing all generated cell crops.
        
        Args:
            cell_crops: List of crop file paths
            cell_metadata: List of crop metadata
            output_dir: Output directory
            
        Returns:
            str: Path to visualization image
        """
        if not cell_crops:
            return None
        
        try:
            # Validate crop files exist
            valid_crops = []
            valid_metadata = []
            for crop_path, metadata in zip(cell_crops, cell_metadata):
                if os.path.exists(crop_path) and os.path.getsize(crop_path) > 0:
                    valid_crops.append(crop_path)
                    valid_metadata.append(metadata)
                else:
                    print(f"Warning: Crop file not found or empty: {crop_path}")
            
            if not valid_crops:
                print("No valid crop files found for visualization")
                return None
            
            # Calculate grid layout
            n_crops = len(valid_crops)
            cols = min(5, n_crops)  # Max 5 columns
            rows = (n_crops + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, (crop_path, metadata) in enumerate(zip(valid_crops, valid_metadata)):
                row = idx // cols
                col = idx % cols
                
                try:
                    # Load crop image with error handling
                    crop_img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
                    if crop_img is None:
                        print(f"Warning: Failed to load crop image: {crop_path}")
                        continue
                    
                    # Validate image data
                    if crop_img.size == 0 or np.isnan(crop_img).any():
                        print(f"Warning: Invalid image data in crop: {crop_path}")
                        continue
                    
                    # Display crop
                    axes[row, col].imshow(crop_img, cmap='gray')
                    axes[row, col].set_title(f"Cell {metadata['cell_id']}\nArea: {metadata['area']}")
                    axes[row, col].axis('off')
                    
                except Exception as e:
                    print(f"Error processing crop {crop_path}: {e}")
                    continue
            
            # Hide empty subplots
            for idx in range(n_crops, rows * cols):
                row = idx // cols
                col = idx % cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # Save visualization with enhanced error handling
            viz_path = os.path.join(output_dir, f"cell_crops_overview_{uuid4().hex[:8]}.png")
            
            try:
                # Save with multiple fallback options
                save_success = False
                
                # Try high quality first
                try:
                    plt.savefig(viz_path, bbox_inches='tight', pad_inches=0.1, dpi=150, format='png', 
                               facecolor='white', edgecolor='none')
                    save_success = True
                except Exception as e:
                    print(f"Warning: High quality save failed: {e}")
                
                # Fallback to lower quality
                if not save_success:
                    try:
                        plt.savefig(viz_path, bbox_inches='tight', pad_inches=0.1, dpi=100, format='png',
                                   facecolor='white', edgecolor='none')
                        save_success = True
                    except Exception as e:
                        print(f"Warning: Lower quality save failed: {e}")
                
                # Final fallback
                if not save_success:
                    try:
                        plt.savefig(viz_path, bbox_inches='tight', pad_inches=0.1, dpi=72, format='png',
                                   facecolor='white', edgecolor='none')
                        save_success = True
                    except Exception as e:
                        print(f"Error: All save attempts failed: {e}")
                        return None
                
                plt.close()
                
                # Verify the saved file
                if not os.path.exists(viz_path) or os.path.getsize(viz_path) == 0:
                    print("Warning: Visualization file not saved properly")
                    return None
                
                # Test loading the saved image
                try:
                    test_image = Image.open(viz_path)
                    test_image.verify()
                    test_image.close()
                    print(f"Successfully saved visualization: {viz_path}")
                except Exception as e:
                    print(f"Warning: Visualization verification failed: {e}")
                    return None
                
                return viz_path
                
            except Exception as e:
                print(f"Error in visualization saving: {e}")
                plt.close()
                return None
                
        except Exception as e:
            print(f"Error creating visualization: {e}")
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