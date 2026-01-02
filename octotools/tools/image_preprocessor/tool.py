# Image Preprocessing Tool
# Provides global illumination correction and brightness adjustment for microscopy images

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

from octotools.tools.base import BaseTool
from octotools.models.utils import VisualizationConfig


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
                "save_intermediate": "bool - Whether to save intermediate processing steps (default: False)."
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

    def create_comparison_plot(self, original, corrected, normalized, filename, vis_config, group=None):
        """
        Create a comparison plot of original, corrected, and normalized images with professional styling.
        
        Args:
            original: Original image
            corrected: Illumination-corrected image
            normalized: Brightness-normalized image
            filename: Original filename for title
            vis_config: An instance of VisualizationConfig
            
        Returns:
            Path to saved comparison plot
        """
        # Use centralized output directory from the vis_config instance
        output_dir = vis_config.get_output_dir()
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
        
        # Save with professional settings
        group_suffix = f"_{group}" if group else ""
        plot_path = os.path.join(
            output_dir,
            f"comparison_{os.path.splitext(os.path.basename(filename))[0]}{group_suffix}.png",
        )
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
                key = os.path.basename(img)
                if img in groups:
                    mapping[img] = groups[img]
                elif key in groups:
                    mapping[img] = groups[key]
                else:
                    raise ValueError(f"Missing group for image: {img}")
            return mapping
        raise ValueError("Unsupported groups format.")

    def execute(self, image, target_brightness=120, gaussian_kernel_size=151, output_format='png', save_intermediate=False, groups=None):
        """
        Execute the image preprocessing pipeline.
        
        Args:
            image: Path to input image
            target_brightness: Target brightness level (0-255)
            gaussian_kernel_size: Size of Gaussian kernel for illumination correction
            output_format: Output image format (forced to 'png' for Visual Outputs compatibility)
            save_intermediate: Whether to save intermediate processing steps
            
        Returns:
            Dictionary containing processing results and file paths
        """
        # Force output_format to 'png' for Visual Outputs compatibility
        # PNG format is required for proper display in Gradio Gallery
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
            original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                raise ValueError(f"Cannot read image: {img_path}. Please check if the file is a valid image format.")
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
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
            )

            # Build result in FBagent_250627 compatible format (flat structure)
            result = {
                "original_image_path": img_path,
                "processed_image_path": final_output_path,
                "processing_statistics": stats,
                "parameters_used": {
                    "target_brightness": target_brightness,
                    "gaussian_kernel_size": gaussian_kernel_size,
                    "output_format": output_format,
                },
                "visual_outputs": [
                    p for p in [comparison_plot, final_output_path if output_format in ("png", "jpg", "jpeg") else None] if p is not None
                ],
            }
            all_results.append(result)
        
        # Return single result if only one image, otherwise return first result (for backward compatibility)
        # Note: Multiple images support can be added later if needed
        if len(all_results) == 1:
            return all_results[0]
        else:
            # For multiple images, return the first one (maintain backward compatibility)
            # Future: could return aggregated result
            return all_results[0]

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
