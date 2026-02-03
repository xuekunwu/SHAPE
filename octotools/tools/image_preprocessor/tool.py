# Image Preprocessing Tool
# Provides global illumination correction and brightness adjustment for microscopy images

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import tifffile

from octotools.tools.base import BaseTool
from octotools.models.utils import VisualizationConfig
from octotools.models.image_data import ImageData
from octotools.utils.image_processor import ImageProcessor


class Image_Preprocessor_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Image_Preprocessor_Tool",
            tool_description="A tool for preprocessing microscopy images with global illumination correction and brightness adjustment.",
            tool_version="1.0.0",
            input_types={
                "image": "str - The path to the input image file.",
                "groups": "dict | list | str | None - Optional grouping for each image.",
                "target_brightness": "int - Target brightness level (0-255, default: 120).",
                "gaussian_kernel_size": "int - Size of Gaussian kernel for illumination correction (default: 151).",
                "output_format": "str - Output format: Always use 'png' for Visual Outputs display compatibility (default: 'png', will be forced to 'png' regardless of input).",
                "save_intermediate": "bool - Whether to save intermediate processing steps (default: False).",
                "image_id": "str - Optional image identifier for consistent file naming and tracking.",
                "skip_illumination_correction": "bool - If True, skip illumination correction and only adjust brightness. Use for organoid images (default: False)."
            },
            output_type="dict - A dictionary containing processed image paths and processing statistics.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/image.png", target_brightness=120)',
                    "description": "Apply global illumination correction and normalize brightness to 120."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/image.png", target_brightness=150, gaussian_kernel_size=101)',
                    "description": "Apply illumination correction with custom kernel size and normalize to brightness 150."
                }
            ],
            user_metadata={
                "limitation": "This tool works best with grayscale microscopy images. Color images will be converted to grayscale. The tool may not work well with images that have very low contrast or extremely uneven illumination.",
                "recommended_usage": "Use for phase contrast, brightfield, or fluorescence microscopy images that need illumination correction and brightness normalization."
            },
            output_dir="output_visualizations"  # Set default output directory to output_visualizations
        )

    def global_illumination_correction(self, image, kernel_size=151):
        """
        Apply Gaussian-based illumination correction to an image.
        
        Args:
            image: Input grayscale image as numpy array
            kernel_size: Size of Gaussian kernel (should be odd number)
            
        Returns:
            Corrected image as numpy array
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Apply Gaussian blur to estimate illumination pattern
        illumination = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Correct illumination by dividing image by illumination pattern
        corrected_image = cv2.divide(image.astype(np.float32), illumination.astype(np.float32) + 1) * 128
        
        # Normalize to 0-255 range
        corrected_image_normalized = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return corrected_image_normalized

    def _load_image_unified(self, img_path: str) -> ImageData:
        """
        Unified image loading using ImageData abstraction
        Maintains backward compatibility by returning ImageData object
        """
        try:
            return ImageProcessor.load_image(img_path)
        except Exception as e:
            print(f"Warning: Failed to load image with ImageData: {e}, falling back to legacy method")
            # Fallback to legacy loading for backward compatibility
            raise
    
    def adjust_brightness(self, image, target_brightness=120):
        """
        Adjust image brightness to target level.
        
        Args:
            image: Input image as numpy array
            target_brightness: Target brightness level (0-255)
            
        Returns:
            Brightness-adjusted image as numpy array
        """
        current_brightness = np.mean(image)
        adjustment_factor = target_brightness / current_brightness
        
        # Apply brightness adjustment
        adjusted_image = cv2.convertScaleAbs(image, alpha=adjustment_factor, beta=0)
        
        return adjusted_image

    def create_multi_channel_visualization(self, channels, channel_names, normalized_channels, filename, vis_config, group=None, image_identifier=None, query_cache_dir=None):
        """
        Create a multi-channel visualization showing individual channels and merged view.
        
        Args:
            channels: List of original channel images (numpy arrays)
            channel_names: List of channel names (e.g., ['bright-field', 'GFP'])
            normalized_channels: List of normalized channel images
            filename: Original filename for title
            vis_config: An instance of VisualizationConfig
            group: Optional group name
            image_identifier: Optional image identifier for file naming
            query_cache_dir: Optional directory for caching results
            
        Returns:
            Path to saved visualization
        """
        # Use centralized output directory from the vis_config instance
        output_dir = vis_config.get_output_dir(query_cache_dir)
        if group:
            output_dir = os.path.join(output_dir, group)
            os.makedirs(output_dir, exist_ok=True)
        
        num_channels = len(channels)
        
        # Create figure with appropriate layout
        # For 2 channels: 1 row, 3 columns (BF, GFP, Merged)
        # For more channels: adjust layout
        if num_channels == 2:
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            axs = axs.flatten()
        else:
            # For more channels, create a grid
            cols = min(4, num_channels + 1)  # channels + merged view
            rows = (num_channels + cols - 1) // cols
            fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
            if rows == 1:
                axs = axs.flatten()
            else:
                axs = axs.flatten()
        
        # Display individual channels
        for idx, (channel, name, norm_channel) in enumerate(zip(channels, channel_names, normalized_channels)):
            if idx < len(axs):
                ax = axs[idx]
                
                # Choose colormap based on channel type
                if 'gfp' in name.lower() or 'fluorescence' in name.lower():
                    ax.imshow(norm_channel, cmap='Greens')
                    ax.set_title(f"{name} (fluorescence)", fontsize=12, fontweight='bold')
                else:
                    ax.imshow(norm_channel, cmap='gray')
                    ax.set_title(f"{name} (grayscale)", fontsize=12, fontweight='bold')
                ax.axis('off')
        
        # Create merged RGB visualization (for 2-channel images)
        if num_channels == 2:
            # Bright-field mapped to grayscale (R=G=B)
            # GFP mapped to green channel
            merged = np.zeros((*normalized_channels[0].shape, 3), dtype=np.float32)
            bf_n = normalized_channels[0]
            gfp_n = normalized_channels[1] if len(normalized_channels) > 1 else bf_n
            
            merged[..., 0] = bf_n          # Red
            merged[..., 1] = bf_n + gfp_n  # Green (BF + GFP for visibility)
            merged[..., 2] = bf_n          # Blue
            merged = np.clip(merged, 0, 1)
            
            # Display merged view
            if len(axs) > num_channels:
                ax = axs[num_channels]
                ax.imshow(merged)
                ax.set_title("Merged (BF + GFP)", fontsize=12, fontweight='bold')
                ax.axis('off')
        
        # Add overall title
        fig.suptitle(f"Multi-Channel Image: {os.path.basename(filename)}", 
                     fontsize=vis_config.PROFESSIONAL_STYLE['axes.titlesize'], 
                     fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save with professional settings
        group_suffix = f"_{group}" if group else ""
        if image_identifier:
            plot_name = f"multi_channel_{image_identifier}{group_suffix}.png"
        else:
            plot_name = f"multi_channel_{os.path.splitext(os.path.basename(filename))[0]}{group_suffix}.png"
        plot_path = os.path.join(output_dir, plot_name)
        vis_config.save_professional_figure(fig, plot_path)
        plt.close(fig)
        
        return plot_path

    def create_comparison_plot(self, original, corrected, normalized, filename, vis_config, group=None, image_identifier=None, query_cache_dir=None):
        """
        Create a comparison plot of original, corrected, and normalized images with professional styling.
        
        Args:
            original: Original image
            corrected: Illumination-corrected image
            normalized: Brightness-normalized image
            filename: Original filename for title
            vis_config: An instance of VisualizationConfig
            group: Optional group name
            image_identifier: Optional image identifier for file naming
            query_cache_dir: Optional directory for caching results
            
        Returns:
            Path to saved comparison plot
        """
        # Use centralized output directory from the vis_config instance
        output_dir = vis_config.get_output_dir(query_cache_dir)
        if group:
            output_dir = os.path.join(output_dir, group)
            os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with professional styling
        fig, axs = vis_config.create_professional_figure(figsize=(18, 6), ncols=3)
        
        # Plot original image
        axs[0].imshow(original, cmap='gray')
        vis_config.apply_professional_styling(axs[0], title="Original Image")
        axs[0].axis('off')
        
        # Plot corrected image
        axs[1].imshow(corrected, cmap='gray')
        vis_config.apply_professional_styling(axs[1], title="Illumination Corrected")
        axs[1].axis('off')
        
        # Plot normalized image
        axs[2].imshow(normalized, cmap='gray')
        vis_config.apply_professional_styling(axs[2], title="Brightness Normalized")
        axs[2].axis('off')
        
        # Add overall title with professional styling
        fig.suptitle(f"Image Preprocessing: {os.path.basename(filename)}", 
                     fontsize=vis_config.PROFESSIONAL_STYLE['axes.titlesize'], 
                     fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save with professional settings using image_identifier if provided
        group_suffix = f"_{group}" if group else ""
        if image_identifier:
            plot_name = f"comparison_{image_identifier}{group_suffix}.png"
        else:
            plot_name = f"comparison_{os.path.splitext(os.path.basename(filename))[0]}{group_suffix}.png"
        plot_path = os.path.join(output_dir, plot_name)
        vis_config.save_professional_figure(fig, plot_path)
        plt.close(fig)
        
        return plot_path

    def _resolve_groups(self, images, groups):
        """
        Resolve group assignment for each image.
        """
        if not isinstance(images, list):
            images = [images]
        if groups is None:
            if len(images) > 1:
                raise ValueError("Groups must be provided when processing multiple images.")
            return {img: "default" for img in images}
        if isinstance(groups, str):
            return {img: groups for img in images}
        if isinstance(groups, list):
            if len(groups) != len(images):
                raise ValueError("Length of groups list must match images list.")
            return dict(zip(images, groups))

        if isinstance(groups, dict):
            mapping = {}
            for img in images:
                # Try multiple matching strategies
                matched = False
                # Strategy 1: Direct path match
                if img in groups:
                    mapping[img] = groups[img]
                    matched = True
                # Strategy 2: Basename match
                elif not matched:
                    key = os.path.basename(img)
                    if key in groups:
                        mapping[img] = groups[key]
                        matched = True
                # Strategy 3: Try matching by image_id if available (extract from path)
                if not matched:
                    # Try to extract image identifier from path and match
                    path_parts = os.path.splitext(os.path.basename(img))[0]
                    for key in groups.keys():
                        if path_parts in str(key) or str(key) in path_parts:
                            mapping[img] = groups[key]
                            matched = True
                            break
                # Strategy 4: If still not matched, try to extract group from path structure
                if not matched:
                    # Try to extract group from path (e.g., /path/to/Control/image.jpg -> Control)
                    path_parts = img.split(os.sep)
                    for part in reversed(path_parts):
                        if part in groups.values():
                            # Find the key for this group value
                            for k, v in groups.items():
                                if v == part:
                                    mapping[img] = part
                                    matched = True
                                    break
                            if matched:
                                break
                # Fallback: use default if still not matched
                if not matched:
                    mapping[img] = "default"
            return mapping
        raise ValueError("Unsupported groups format.")

    def execute(self, image, target_brightness=120, gaussian_kernel_size=151, output_format='png', save_intermediate=False, groups=None, image_id=None, query_cache_dir=None, skip_illumination_correction=False):
        """
        Execute the image preprocessing pipeline.
        
        Args:
            image: Path to input image
            target_brightness: Target brightness level (0-255)
            gaussian_kernel_size: Size of Gaussian kernel for illumination correction
            output_format: Output image format (forced to 'png' for Visual Outputs compatibility)
            save_intermediate: Whether to save intermediate processing steps
            groups: Optional grouping for images
            image_id: Optional image identifier for consistent file naming and tracking
            query_cache_dir: Optional directory for caching results
            skip_illumination_correction: If True, skip illumination correction (only adjust brightness). 
                                        Use for organoid images (default: False)
            
        Returns:
            Dictionary containing processing results and file paths
        """
        # Force output_format to 'png' for visual outputs compatibility
        output_format = 'png'
        
        # Ensure output directory exists
        images = image if isinstance(image, list) else [image]
        group_map = self._resolve_groups(images, groups)
        os.makedirs(self.output_dir, exist_ok=True)
        all_results = []
        for img_path in images:
            group = group_map[img_path]
            # Normalize path separators for cross-platform compatibility
            img_path = os.path.normpath(img_path) if isinstance(img_path, str) else str(img_path)
            # Check if file exists
            if not os.path.exists(img_path):
                raise ValueError(f"Image file not found: {img_path}")
            
            # Load image using unified abstraction layer
            try:
                img_data = ImageProcessor.load_image(img_path)
                print(f"Loaded image: {img_data.shape}, channels: {img_data.num_channels}, dtype: {img_data.dtype}")
            except Exception as load_error:
                # Fallback to legacy loading for backward compatibility
                print(f"Warning: Failed to load with ImageProcessor: {load_error}, trying legacy method")
                image_path_lower = img_path.lower()
                is_tiff = image_path_lower.endswith('.tif') or image_path_lower.endswith('.tiff')
                if is_tiff:
                    try:
                        img_full = tifffile.imread(img_path)
                        if img_full.ndim == 2:
                            original_image = img_full
                        elif img_full.ndim == 3:
                            original_image = img_full[:, :, 0] if img_full.shape[2] <= 4 else img_full[0, :, :]
                        else:
                            original_image = img_full[0, :, :, 0] if img_full.ndim == 4 else img_full[0, 0, :, :]
                        if original_image.dtype != np.uint8:
                            if original_image.dtype == np.uint16:
                                original_image = (original_image / 65535.0 * 255).astype(np.uint8)
                            else:
                                original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    except Exception:
                        original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if original_image is None:
                    raise ValueError(f"Cannot read image: {img_path}. Please check if the file is a valid image format.")
                
                # Convert legacy format to ImageData for unified processing
                img_data = ImageData._from_array(original_image)
            
            # Extract channels using unified abstraction
            # Use first channel for standard preprocessing (backward compatibility)
            original_image = img_data.to_uint8(0)  # Get first channel as uint8
            
            # Prepare multi-channel data for later processing
            multi_channel_images = None
            channel_names = img_data.channel_names
            
            if img_data.is_multi_channel:
                # Extract all channels as uint8 arrays for processing
                multi_channel_images = []
                for ch_idx in range(img_data.num_channels):
                    channel_uint8 = img_data.to_uint8(ch_idx)
                    multi_channel_images.append(channel_uint8)
                print(f"Detected multi-channel image with {img_data.num_channels} channels: {channel_names}")
            
            filename = os.path.basename(img_path)
            # Extract image identifier: use provided image_id, or use original filename (without extension)
            if image_id:
                image_identifier = image_id
            else:
                # Use the original filename without extension to preserve the original image name
                # e.g., "A1_02_1_1_Phase Contrast_001.png" -> "A1_02_1_1_Phase Contrast_001"
                image_identifier = os.path.splitext(filename)[0]
            
            base_name = image_identifier
            
            # For organoid images: skip illumination correction, only adjust brightness
            if skip_illumination_correction:
                print(f"⚠️ Skipping illumination correction (organoid mode). Only adjusting brightness...")
                corrected_image = original_image  # Use original image without illumination correction
                normalized_image = self.adjust_brightness(corrected_image, target_brightness)
            else:
                # Standard preprocessing: illumination correction + brightness adjustment
                corrected_image = self.global_illumination_correction(original_image, gaussian_kernel_size)
                normalized_image = self.adjust_brightness(corrected_image, target_brightness)

            group_dir = os.path.join(self.output_dir, group)
            os.makedirs(group_dir, exist_ok=True)

            final_output_path = os.path.join(group_dir, f"{base_name}_{group}_processed.{output_format}")
            # Normalize path for cross-platform compatibility
            final_output_path = os.path.normpath(final_output_path)
            
            # Save the processed image
            try:
                Image.fromarray(normalized_image).save(final_output_path)
                # Verify file was saved
                if not os.path.exists(final_output_path):
                    raise ValueError(f"File was not created: {final_output_path}")
                file_size = os.path.getsize(final_output_path)
                if file_size == 0:
                    raise ValueError(f"File is empty: {final_output_path}")
                print(f"Successfully saved processed image to: {final_output_path} (size: {file_size} bytes)")
            except Exception as save_error:
                error_msg = f"Failed to save processed image to {final_output_path}: {str(save_error)}"
                print(f"Error: {error_msg}")
                raise ValueError(error_msg)
            
            # Handle multi-channel extraction: save each channel separately
            channel_outputs = []
            channel_mapping = {}
            
            if img_data.is_multi_channel and multi_channel_images is not None and len(multi_channel_images) > 1:
                print(f"Extracting and saving {len(multi_channel_images)} channels separately...")
                
                # Initialize list for normalized channels for visualization
                normalized_channels_for_vis = []
                
                # Normalize function for visualization (independent normalization per channel)
                def normalize_channel(x):
                    """Normalize channel to [0, 1] range for visualization"""
                    x = x.astype(np.float32)
                    x_min = x.min()
                    x_max = x.max()
                    if x_max > x_min:
                        return (x - x_min) / (x_max - x_min + 1e-8)
                    return x
                
                for idx, channel_img in enumerate(multi_channel_images):
                    channel_name = channel_names[idx] if channel_names and idx < len(channel_names) else f"channel_{idx+1}"
                    
                    # For bright-field: apply minimal/careful preprocessing
                    # For other channels (GFP, etc.): only normalize, no heavy preprocessing
                    is_brightfield = 'bright' in channel_name.lower() or 'field' in channel_name.lower() or idx == 0
                    
                    if is_brightfield:
                        # For organoid images: skip illumination correction, only adjust brightness
                        if skip_illumination_correction:
                            print(f"⚠️ Skipping illumination correction for {channel_name} channel (organoid mode). Only adjusting brightness...")
                            channel_corrected = channel_img  # Use original channel without illumination correction
                            channel_normalized = self.adjust_brightness(channel_corrected, target_brightness)
                        else:
                            # Apply careful preprocessing to bright-field only
                            # Use smaller kernel size for more conservative correction
                            careful_kernel_size = min(gaussian_kernel_size, 101)  # Cap at 101 for bright-field
                            channel_corrected = self.global_illumination_correction(channel_img, careful_kernel_size)
                            # Use more conservative brightness adjustment for bright-field
                            channel_normalized = self.adjust_brightness(channel_corrected, target_brightness)
                        print(f"Applied preprocessing to {channel_name} channel")
                    else:
                        # For GFP and other channels: only normalize, preserve original characteristics
                        # Just normalize to uint8 for saving, but keep original for visualization
                        if channel_img.dtype == np.uint16:
                            channel_normalized = (channel_img / 65535.0 * 255).astype(np.uint8)
                        else:
                            channel_normalized = np.clip(channel_img, 0, 255).astype(np.uint8)
                        print(f"Preserved original characteristics for {channel_name} channel (no heavy preprocessing)")
                    
                    # Normalize for visualization (independent normalization)
                    channel_for_vis = normalize_channel(channel_img.astype(np.float32))
                    normalized_channels_for_vis.append(channel_for_vis)
                    
                    # Generate channel-specific filename
                    channel_name_safe = channel_name.replace(" ", "_").replace("-", "_").lower()
                    channel_output_path = os.path.join(group_dir, f"{base_name}_{channel_name_safe}.{output_format}")
                    channel_output_path = os.path.normpath(channel_output_path)
                    
                    # Save channel as grayscale PNG
                    try:
                        Image.fromarray(channel_normalized, mode='L').save(channel_output_path)
                        if os.path.exists(channel_output_path):
                            channel_outputs.append(channel_output_path)
                            channel_mapping[channel_name] = {
                                "channel_index": idx,
                                "file_path": channel_output_path,
                                "brightness": round(float(np.mean(channel_normalized)), 2),
                                "preprocessing_applied": is_brightfield
                            }
                            print(f"Saved {channel_name} channel (index {idx}) to: {channel_output_path}")
                    except Exception as channel_save_error:
                        print(f"Warning: Failed to save {channel_name} channel: {channel_save_error}")
                
                # Create multi-channel visualization using unified ImageProcessor
                vis_config = VisualizationConfig()
                multi_channel_plot_path = os.path.join(group_dir, f"multi_channel_{base_name}.png")
                multi_channel_plot_path = os.path.normpath(multi_channel_plot_path)
                
                # Use original img_data for visualization (preserves original channel data)
                # The visualization will show original channels, not preprocessed ones
                multi_channel_plot = ImageProcessor.create_multi_channel_visualization(
                    img_data,
                    multi_channel_plot_path,
                    vis_config,
                    group=group,
                    image_identifier=image_identifier
                )
                
                if multi_channel_plot:
                    channel_outputs.append(multi_channel_plot)
                    print(f"Created multi-channel visualization: {multi_channel_plot}")

            stats = {
                "original_brightness": round(float(np.mean(original_image)), 2),
                "corrected_brightness": round(float(np.mean(corrected_image)), 2),
                "final_brightness": round(float(np.mean(normalized_image)), 2),
                "target_brightness": target_brightness,
                "brightness_delta": round(
                    float(np.mean(normalized_image) - np.mean(original_image)), 2
                ),
            }

            vis_config = VisualizationConfig()
            comparison_plot = self.create_comparison_plot(
                original_image,
                corrected_image,
                normalized_image,
                filename,
                vis_config,
                group=group,
                image_identifier=image_identifier,
                query_cache_dir=query_cache_dir,
            )

            # Build result in FBagent_250627 compatible format (flat structure)
            visual_outputs_list = [
                p for p in [comparison_plot, final_output_path if output_format in ("png", "jpg", "jpeg") else None] if p is not None
            ]
            # Only add multi_channel_ visualization to visual_outputs (not individual channel images)
            # Individual channel images are saved but not displayed to avoid cluttering Visual Outputs
            if channel_outputs:
                # Filter to only include multi_channel_ visualization
                multi_channel_vis = [p for p in channel_outputs if "multi_channel_" in os.path.basename(p).lower()]
                if multi_channel_vis:
                    visual_outputs_list.extend(multi_channel_vis)
            
            result = {
                "original_image_path": img_path,
                "processed_image_path": final_output_path,
                "processing_statistics": stats,
                "parameters_used": {
                    "target_brightness": target_brightness,
                    "gaussian_kernel_size": gaussian_kernel_size,
                    "output_format": output_format,
                },
                "visual_outputs": visual_outputs_list,
                "image_id": image_identifier,  # Add image_id for matching in downstream tools
                "image_identifier": image_identifier,  # Alias for compatibility
            }
            
            # Add multi-channel information if available
            if multi_channel_images is not None and len(multi_channel_images) > 1:
                result["multi_channel"] = {
                    "num_channels": len(multi_channel_images),
                    "channel_mapping": channel_mapping,
                    "channel_outputs": channel_outputs
                }
                print(f"Multi-channel extraction complete: {len(multi_channel_images)} channels saved separately.")
            all_results.append(result)
        
        # Return single result if only one image, otherwise return per_image structure
        # This ensures each image has its own processed_image_path for downstream tools
        if len(all_results) == 1:
            return all_results[0]
        else:
            # For multiple images, return per_image structure for proper tracking
            return {"per_image": all_results}

    def get_metadata(self):
        """Return tool metadata."""
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    # Test the tool
    tool = Image_Preprocessor_Tool()
    tool.set_custom_output_dir("test_output")
    
    # Test with a sample image
    test_image_path = "examples/fibroblast.png"
    if os.path.exists(test_image_path):
        result = tool.execute(
            image=test_image_path,
            target_brightness=120,
            gaussian_kernel_size=151,
            output_format='png',
            save_intermediate=True
        )
        print("Processing Results:")
        print(f"Original brightness: {result['processing_statistics']['original_brightness']}")
        print(f"Final brightness: {result['processing_statistics']['final_brightness']}")
        print(f"Processed image saved to: {result['processed_image_path']}")
    else:
        print(f"Test image not found: {test_image_path}") 
