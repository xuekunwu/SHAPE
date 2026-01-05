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
from octotools.models.task_state import CellCrop
import tifffile

class Single_Cell_Cropper_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Single_Cell_Cropper_Tool",
            tool_description="Generates individual cell/object crops from segmentation masks (nuclei, cells, or organoids) for single-cell analysis. Creates crops for each detected object with configurable margins. Supports masks from Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, or Organoid_Segmenter_Tool. Automatically handles image dimension mismatches by resizing masks.",
            tool_version="1.0.0",
            input_types={
                "original_image": "str - Path to the original image (brightfield/phase contrast).",
                "nuclei_mask": "str - Path to the segmentation mask image. Accepts nuclei_mask, cell_mask, or organoid_mask from segmentation tools.",
                "source_image_id": "str - Optional source image ID for cell tracking (default: extracted from image path).",
                "group": "str - Optional group/condition label for multi-group analysis (default: 'default').",
                "min_area": "int - Minimum area threshold for valid objects (auto-detected from mask type: cell_mask=50, nuclei_mask=50, organoid_mask=200).",
                "margin": "int - Margin around each object for cropping (auto-detected from mask type: cell_mask=1, nuclei_mask=25, organoid_mask=50).",
                "output_format": "str - Output format for crops ('tif', 'png', 'jpg', default: 'png')."
            },
            output_type="dict - Contains cropped cell images, metadata, and visualization paths.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(original_image="path/to/image.tif", nuclei_mask="path/to/nuclei_mask.png")',
                    "description": "Generate single-cell crops from nuclei segmentation mask with default parameters."
                },
                {
                    "command": 'execution = tool.execute(original_image="path/to/image.tif", nuclei_mask="path/to/cell_mask.png", min_area=100, margin=30)',
                    "description": "Generate single-cell crops from cell segmentation mask with custom area threshold and margin parameters."
                },
                {
                    "command": 'execution = tool.execute(original_image="path/to/image.tif", nuclei_mask="path/to/organoid_mask.png")',
                    "description": "Generate organoid crops from organoid segmentation mask."
                }
            ],
            user_metadata={
                "limitation": "Requires segmentation mask results as input (from Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, or Organoid_Segmenter_Tool). May generate overlapping crops if objects are close together. Performance depends on image quality and segmentation accuracy.",
                "best_practice": "Use with any segmentation tool (Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, or Organoid_Segmenter_Tool) for optimal results. Adjust min_area and margin based on object size and analysis requirements. For organoids, use larger min_area and margin values. Automatically handles dimension mismatches between original image and mask."
            }
        )

    def execute(self, original_image, nuclei_mask, source_image_id=None, group="default", min_area=None, margin=None, output_format='png', query_cache_dir=None):
        """
        Execute single-cell/object cropping from segmentation masks.
        
        This is a Stage 2 (cell-level) tool. It operates on segmentation results (masks)
        from Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, or Organoid_Segmenter_Tool,
        and produces CellCrop objects as the atomic data units for downstream analysis.
        
        Args:
            original_image: Path to original image (brightfield/phase contrast)
            nuclei_mask: Path to segmentation mask (accepts nuclei_mask, cell_mask, or organoid_mask)
            source_image_id: Optional source image ID for tracking (default: extracted from path)
            group: Optional group/condition label (default: 'default')
            min_area: Minimum area threshold for valid objects (default: auto-detected based on mask type)
            margin: Margin around each object for cropping (default: auto-detected based on mask type)
            output_format: Output image format ('png', 'tif', 'jpg')
            query_cache_dir: Directory for caching results
            
        Returns:
            dict: Cropping results with cell/object crops, CellCrop objects, and metadata
        """
        try:
            # Auto-detect mask type from filename and set default parameters if not provided
            mask_filename_lower = os.path.basename(nuclei_mask).lower()
            auto_detected = False
            
            if min_area is None or margin is None:
                if "cell_mask" in mask_filename_lower:
                    # Cell mask: min_area=50, margin=1
                    auto_min_area = 50
                    auto_margin = 1
                    auto_detected = True
                    print(f"Auto-detected cell_mask - using min_area={auto_min_area}, margin={auto_margin}")
                elif "organoid_mask" in mask_filename_lower:
                    # Organoid mask: min_area=200, margin=50
                    auto_min_area = 200
                    auto_margin = 50
                    auto_detected = True
                    print(f"Auto-detected organoid_mask - using min_area={auto_min_area}, margin={auto_margin}")
                elif "nuclei_mask" in mask_filename_lower:
                    # Nuclei mask: min_area=50, margin=25
                    auto_min_area = 50
                    auto_margin = 25
                    auto_detected = True
                    print(f"Auto-detected nuclei_mask - using min_area={auto_min_area}, margin={auto_margin}")
                else:
                    # Default to nuclei_mask parameters if cannot detect
                    auto_min_area = 50
                    auto_margin = 25
                    print(f"Could not detect mask type from filename, using default (nuclei_mask) parameters: min_area={auto_min_area}, margin={auto_margin}")
                
                # Use auto-detected values if user didn't specify
                if min_area is None:
                    min_area = auto_min_area
                if margin is None:
                    margin = auto_margin
            
            # Check if we should use processed image instead of original
            if query_cache_dir and os.path.exists(os.path.join(query_cache_dir, "tool_cache", "query_image_processed.png")):
                processed_image_path = os.path.join(query_cache_dir, "tool_cache", "query_image_processed.png")
                print(f"Using processed image: {processed_image_path}")
                original_img = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)
            else:
                # Load original image
                original_img = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
            
            # Setup output directory early, so we can save metadata even on errors
            if query_cache_dir is None:
                query_cache_dir = "solver_cache/temp"
            tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
            os.makedirs(tool_cache_dir, exist_ok=True)
            
            # Helper function to save error metadata
            def save_error_metadata(error_type, error_message):
                """Save metadata file even when errors occur."""
                try:
                    error_metadata = {
                        'cell_count': 0,
                        'cell_crops_paths': [],
                        'cell_metadata': [],
                        'cell_crop_objects': [],
                        'statistics': {},
                        "parameters": {
                            "min_area": min_area,
                            "margin": margin
                        },
                        "execution_status": "error",
                        "error_type": error_type,
                        "error_message": error_message
                    }
                    metadata_path = os.path.join(tool_cache_dir, f"cell_crops_metadata_error_{uuid4().hex[:8]}.json")
                    print(f"Single_Cell_Cropper_Tool: Saving error metadata to: {metadata_path}")
                    with open(metadata_path, 'w') as f:
                        json.dump(error_metadata, f, indent=2)
                    print(f"Single_Cell_Cropper_Tool: ✅ Error metadata file saved: {metadata_path}")
                    return metadata_path
                except Exception as save_error:
                    print(f"Single_Cell_Cropper_Tool: ❌ Failed to save error metadata: {save_error}")
                    return None
            
            if original_img is None:
                metadata_path = save_error_metadata("image_load_failed", f"Failed to load original image: {original_image}")
                return {
                    "error": f"Failed to load original image: {original_image}",
                    "summary": "Image loading failed",
                    "execution_status": "error",
                    "cell_crops_metadata_path": metadata_path
                }
            
            # Load segmentation mask (supports nuclei_mask, cell_mask, or organoid_mask)
            # Prioritize tifffile for .tif files to preserve 16-bit label values
            mask_path_lower = nuclei_mask.lower()
            if mask_path_lower.endswith('.tif') or mask_path_lower.endswith('.tiff'):
                try:
                    mask = tifffile.imread(nuclei_mask)
                    # Ensure mask is 2D and convert to appropriate dtype
                    if len(mask.shape) > 2:
                        mask = mask.squeeze()
                    # Preserve uint16 for label masks (>255 labels), convert others to uint16 for consistency
                    if mask.dtype not in [np.uint8, np.uint16]:
                        mask = mask.astype(np.uint16)
                    elif mask.dtype == np.uint8:
                        # Convert uint8 to uint16 to preserve all values (backward compatibility)
                        mask = mask.astype(np.uint16)
                except Exception as e:
                    print(f"Warning: Failed to load mask with tifffile: {e}, trying cv2")
                    mask = cv2.imread(nuclei_mask, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = mask.astype(np.uint16)  # Convert to uint16 for consistency
            else:
                # For PNG/other formats, use cv2 (may lose labels >255 in old masks)
                # Convert to uint16 for consistency with label mask handling
                mask = cv2.imread(nuclei_mask, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = mask.astype(np.uint16)  # Convert to uint16 to handle any labels up to 65535
            
            if mask is None:
                metadata_path = save_error_metadata("mask_load_failed", f"Failed to load segmentation mask: {nuclei_mask}")
                return {
                    "error": f"Failed to load segmentation mask: {nuclei_mask}",
                    "summary": "Mask loading failed. Please provide a valid mask from Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, or Organoid_Segmenter_Tool.",
                    "execution_status": "error",
                    "cell_crops_metadata_path": metadata_path
                }
            
            # Extract source_image_id from path if not provided
            if source_image_id is None:
                source_image_id = Path(original_image).stem
            
            # Generate cell crops and CellCrop objects
            cell_crops, cell_metadata, cell_crop_objects, stats = self._generate_cell_crops(
                original_img, mask, original_image, source_image_id, group, min_area, margin, tool_cache_dir, output_format
            )
            
            # Always save metadata file, even if no crops were generated
            # This ensures Cell_State_Analyzer_Tool knows Single_Cell_Cropper_Tool has executed
            # Combine paths and metadata for JSON output (backward compatibility)
            output_data = {
                'cell_count': stats.get('final_cell_count', 0) if cell_crops else 0,
                'cell_crops_paths': cell_crops if cell_crops else [],
                'cell_metadata': cell_metadata if cell_crops else [],
                'cell_crop_objects': [cell.to_dict() for cell in cell_crop_objects] if cell_crop_objects else [],  # Serialize CellCrop objects
                'statistics': stats if cell_crops else {},
                "parameters": {
                    "min_area": min_area,
                    "margin": margin
                },
                "execution_status": "success" if cell_crops else "no_crops_generated"
            }
            metadata_path = os.path.join(tool_cache_dir, f"cell_crops_metadata_{uuid4().hex[:8]}.json")
            print(f"Single_Cell_Cropper_Tool: Saving metadata to: {metadata_path}")
            print(f"Single_Cell_Cropper_Tool: query_cache_dir={query_cache_dir}, tool_cache_dir={tool_cache_dir}")
            print(f"Single_Cell_Cropper_Tool: cell_count={output_data['cell_count']}, execution_status={output_data['execution_status']}")
            with open(metadata_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            # Verify file was saved
            if os.path.exists(metadata_path):
                print(f"Single_Cell_Cropper_Tool: ✅ Metadata file saved successfully: {metadata_path}")
            else:
                print(f"Single_Cell_Cropper_Tool: ❌ ERROR: Metadata file was not saved: {metadata_path}")
            
            if not cell_crops:
                return {
                    "summary": "No valid cell crops generated. Metadata file saved for tracking.",
                    "cell_count": 0,
                    "cell_crops": [],
                    "cell_metadata": [],
                    "cell_crop_objects": [],  # Empty list of CellCrop objects
                    "visual_outputs": [],
                    "cell_crops_metadata_path": metadata_path,  # Include metadata path even when no crops
                    "parameters": {
                        "min_area": min_area,
                        "margin": margin,
                        "output_format": output_format
                    },
                    "statistics": stats,
                    "execution_status": "no_crops_generated"
                }

            # Create a summary visualization
            summary_viz_path = self._create_summary_visualization(
                original_img, mask, cell_crops, stats, query_cache_dir, min_area, margin
            )
            
            # Build detailed summary with filtering information
            summary_parts = [f"Successfully generated {stats['final_cell_count']} single-cell crops."]
            
            # Add filtering statistics if available
            total_detected = stats.get('total_objects', stats.get('initial_cell_count', 0))
            filtered_by_area = stats.get('filtered_by_area', 0)
            filtered_by_border = stats.get('filtered_by_border', 0)
            invalid_crop_data = stats.get('invalid_crop_data', 0)
            total_filtered = filtered_by_area + filtered_by_border + invalid_crop_data
            
            if total_detected > 0:
                if total_filtered > 0:
                    filter_details = []
                    if filtered_by_area > 0:
                        filter_details.append(f"{filtered_by_area} by min_area={min_area}px")
                    if filtered_by_border > 0:
                        filter_details.append(f"{filtered_by_border} by border constraints")
                    if invalid_crop_data > 0:
                        filter_details.append(f"{invalid_crop_data} invalid crops")
                    
                    filter_summary = ", ".join(filter_details)
                    summary_parts.append(f"Detected {total_detected} objects in mask, filtered out {total_filtered} ({filter_summary}).")
                    summary_parts.append(f"({total_detected} detected → {stats['final_cell_count']} valid crops)")
                else:
                    summary_parts.append(f"All {total_detected} detected objects were successfully cropped.")
            
            summary_parts.append(f"All paths and metadata are saved in '{Path(metadata_path).name}'.")
            
            return {
                "summary": " ".join(summary_parts),
                "cell_count": stats['final_cell_count'],
                "total_detected_objects": total_detected,
                "filtered_by_area": filtered_by_area,
                "filtered_by_border": filtered_by_border,
                "invalid_crop_data": invalid_crop_data,
                "cell_crops_metadata_path": metadata_path,
                "cell_crop_objects": cell_crop_objects,  # Return CellCrop objects for Stage 2 tools
                "visual_outputs": [summary_viz_path] if summary_viz_path else [],
            }
            
        except Exception as e:
            print(f"Single_Cell_Cropper_Tool: Error in single-cell cropping: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to save metadata file even on error, so Cell_State_Analyzer_Tool knows the tool executed
            try:
                if query_cache_dir is None:
                    query_cache_dir = "solver_cache/temp"
                tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
                os.makedirs(tool_cache_dir, exist_ok=True)
                
                error_metadata = {
                    'cell_count': 0,
                    'cell_crops_paths': [],
                    'cell_metadata': [],
                    'cell_crop_objects': [],
                    'statistics': {},
                    "parameters": {
                        "min_area": min_area,
                        "margin": margin
                    },
                    "execution_status": "error",
                    "error_message": str(e)
                }
                metadata_path = os.path.join(tool_cache_dir, f"cell_crops_metadata_error_{uuid4().hex[:8]}.json")
                print(f"Single_Cell_Cropper_Tool: Saving error metadata to: {metadata_path}")
                with open(metadata_path, 'w') as f:
                    json.dump(error_metadata, f, indent=2)
                print(f"Single_Cell_Cropper_Tool: ✅ Error metadata file saved: {metadata_path}")
            except Exception as save_error:
                print(f"Single_Cell_Cropper_Tool: ❌ Failed to save error metadata: {save_error}")
            
            return {
                "error": f"Error during single-cell cropping: {str(e)}",
                "summary": "Failed to process image",
                "execution_status": "error"
            }

    def _generate_cell_crops(self, original_img, mask, original_image_path, source_image_id, group, min_area, margin, output_dir, output_format):
        """
        Generate individual cell crops from the mask and create CellCrop objects.
        
        This method produces CellCrop objects as atomic data units for Stage 2 analysis.
        
        Args:
            original_img: Original image array
            mask: Binary mask array
            original_image_path: Path to the original source image
            source_image_id: Source image ID for tracking
            group: Group/condition label
            min_area: Minimum area threshold
            margin: Margin around each nucleus
            output_dir: Output directory
            output_format: Output image format
            
        Returns:
            tuple: (list of crop paths, list of metadata dicts, list of CellCrop objects, dict of statistics)
        """
        # Validate input images
        if original_img is None or mask is None:
            print("Error: Input images are None")
            return [], [], [], {}
        
        # Ensure images are numpy arrays
        original_img = np.asarray(original_img)
        mask = np.asarray(mask)
        
        # Check image dimensions
        if original_img.shape != mask.shape:
            print(f"Error: Image shape mismatch - original: {original_img.shape}, mask: {mask.shape}")
            return [], [], [], {}
        
        # Handle mask format: Cellpose generates label masks (0=background, 1-N for N cells)
        # We should preserve the original labels, not convert to binary
        # Check if mask is already a label mask (multiple unique values) or binary
        unique_values = np.unique(mask)
        num_unique = len(unique_values)
        
        if num_unique > 2:
            # Label mask from Cellpose: use directly with regionprops
            # Each unique value (except 0) represents a distinct cell/object
            print(f"Detected label mask with {num_unique - 1} unique objects (excluding background)")
            # Use the mask directly - regionprops can work with label masks
            labeled_mask = mask
        else:
            # Binary mask: need to label connected components
            print("Detected binary mask, labeling connected components")
            labeled_mask = label(mask)
        
        cell_properties = regionprops(labeled_mask)
        
        initial_cell_count = len(cell_properties)
        filtered_by_area = 0
        filtered_by_border = 0
        invalid_crop_data = 0

        cell_crops = []
        cell_metadata = []
        cell_crop_objects = []  # List of CellCrop objects
        
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
            
            # Create cell ID in format: {group}_{image_name}_{crop_id}
            # This allows tracing back to group and image_name from cell_id
            crop_id = f"cell_{idx:04d}"
            # Use source_image_id as image_name (extracted from path if not provided)
            image_name = source_image_id if source_image_id else Path(original_image_path).stem
            # Create cell_id in format: group_image_name_crop_id (e.g., "control_image1_cell_0001")
            cell_id = f"{group}_{image_name}_{crop_id}"
            
            # Create metadata for this crop (backward compatibility)
            crop_metadata = {
                "cell_id": cell_id,  # Use full cell_id string instead of idx
                "crop_path": crop_path,
                "original_bbox": [minr, minc, maxr, maxc],
                "crop_bbox": [new_minr, new_minc, new_maxr, new_maxc],
                "area": region.area,
                "centroid": list(region.centroid),  # Convert tuple to list for JSON serialization
                "crop_size": cell_crop.shape,
                "group": group,  # Include group in metadata
                "image_name": image_name  # Include image_name in metadata
            }
            
            # Create CellCrop object (atomic data unit for Stage 2)
            # Extract individual cell mask
            # regionprops iterates in label order, where labels start at 1
            # We need to get the label value for this region
            region_label = labeled_mask[int(region.centroid[0]), int(region.centroid[1])]
            cell_mask = np.zeros_like(mask)
            cell_mask[labeled_mask == region_label] = 255
            cell_mask_crop = cell_mask[new_minr:new_maxr, new_minc:new_maxc]
            
            # Save cell mask if needed
            mask_filename = f"cell_{idx:04d}_mask.{output_format}"
            mask_path = os.path.join(output_dir, mask_filename)
            try:
                if len(cell_mask_crop.shape) == 2:
                    mask_pil = Image.fromarray(cell_mask_crop, mode='L')
                else:
                    mask_pil = Image.fromarray(cell_mask_crop)
                
                if output_format.lower() == 'tif':
                    mask_pil.save(mask_path, 'TIFF', compression='tiff_lzw')
                elif output_format.lower() == 'png':
                    mask_pil.save(mask_path, 'PNG', optimize=True)
                elif output_format.lower() == 'jpg':
                    mask_pil.save(mask_path, 'JPEG', quality=95, optimize=True)
                else:
                    mask_pil.save(mask_path, 'PNG', optimize=True)
            except Exception as e:
                print(f"Warning: Failed to save mask for cell {idx}: {e}")
                mask_path = None
            
            cell_crop_obj = CellCrop(
                cell_id=cell_id,
                crop_id=crop_id,
                source_image_id=source_image_id,
                source_image_path=original_image_path,
                group=group,
                crop_path=crop_path,
                mask_path=mask_path,
                bbox=[new_minr, new_minc, new_maxr, new_maxc],
                centroid=list(region.centroid),
                area=int(region.area),
                metadata={
                    "original_bbox": [int(minr), int(minc), int(maxr), int(maxc)],
                    "crop_size": list(cell_crop.shape),
                    "min_area": min_area,
                    "margin": margin
                }
            )
            
            cell_crops.append(crop_path)
            cell_metadata.append(crop_metadata)
            cell_crop_objects.append(cell_crop_obj)
        
        stats = {
            "total_objects": initial_cell_count,  # Total objects detected in mask
            "initial_cell_count": initial_cell_count,
            "filtered_by_area": filtered_by_area,
            "filtered_by_border": filtered_by_border,
            "invalid_crop_data": invalid_crop_data,
            "final_cell_count": len(cell_crops)
        }

        return cell_crops, cell_metadata, cell_crop_objects, stats

    def _create_summary_visualization(self, original_img, mask, cell_crops, stats, output_dir, min_area, margin):
        """Creates a professional and highly robust summary visualization of the cropping process."""
        vis_config = VisualizationConfig()
        output_path = os.path.join(output_dir, "single_cell_cropper_summary.png")

        try:
            fig = plt.figure(figsize=(24, 16), dpi=300)
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], hspace=0.25, wspace=0.15)
            
            fig.suptitle("Single-Cell Cropping Summary", fontsize=36, fontweight='bold', y=0.98)

            # --- Subplots ---
            ax1_container = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            
            # 1. Sample Crops Grid (show 5 random crops)
            ax1_container.set_title("Sample Cell Crops (5 random)", fontsize=32, fontweight='bold', pad=20)
            ax1_container.axis('off')
            if cell_crops:
                num_samples = min(5, len(cell_crops))
                gs_crops = ax1_container.get_subplotspec().subgridspec(1, num_samples, wspace=0.05, hspace=0.05)
                sample_indices = np.random.choice(len(cell_crops), size=num_samples, replace=False)
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
        print("execution = tool.execute(original_image='path/to/image.tif', nuclei_mask='path/to/mask.png')")
        print("Note: nuclei_mask parameter accepts any segmentation mask: nuclei_mask, cell_mask, or organoid_mask")
        
    except Exception as e:
        print(f"Error during tool initialization: {e}") 