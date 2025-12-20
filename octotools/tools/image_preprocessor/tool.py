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
                "target_brightness": "int - Target brightness level (0-255, default: 120).",
                "gaussian_kernel_size": "int - Size of Gaussian kernel for illumination correction (default: 151).",
                "output_format": "str - Output format: 'png', 'jpg', or 'tiff' (default: 'png').",
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
        # Registry metadata for capability-driven planning
        self.capabilities = ["image_preprocess"]
        self.domain = "fibroblast"

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

    def create_comparison_plot(self, original, corrected, normalized, filename, vis_config):
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
        plot_path = os.path.join(output_dir, f"comparison_{os.path.splitext(os.path.basename(filename))[0]}.png")
        vis_config.save_professional_figure(fig, plot_path)
        plt.close(fig)
        
        return plot_path

    def execute(self, image, target_brightness=120, gaussian_kernel_size=151, output_format='png', save_intermediate=False):
        """
        Execute the image preprocessing pipeline.
        
        Args:
            image: Path to input image
            target_brightness: Target brightness level (0-255)
            gaussian_kernel_size: Size of Gaussian kernel for illumination correction
            output_format: Output image format
            save_intermediate: Whether to save intermediate processing steps
            
        Returns:
            Dictionary containing processing results and file paths
        """
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Read the input image
            if isinstance(image, str):
                # Read as grayscale
                original_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                if original_image is None:
                    raise ValueError(f"Cannot read image: {image}")
                filename = os.path.basename(image)
            else:
                # Handle PIL Image or other formats
                if hasattr(image, 'convert'):
                    # Convert PIL Image to numpy array
                    original_image = np.array(image.convert('L'))
                else:
                    raise ValueError("Unsupported image format")
                filename = "processed_image"
            
            # Step 1: Global illumination correction
            corrected_image = self.global_illumination_correction(original_image, gaussian_kernel_size)
            
            # Step 2: Brightness adjustment
            normalized_image = self.adjust_brightness(corrected_image, target_brightness)
            
            # Calculate processing statistics
            original_brightness = np.mean(original_image)
            corrected_brightness = np.mean(corrected_image)
            final_brightness = np.mean(normalized_image)
            
            # Save processed images
            base_name = os.path.splitext(filename)[0]
            
            # Save final processed image using PIL for better format support
            final_output_path = os.path.join(self.output_dir, f"{base_name}_processed.{output_format}")
            final_pil_image = Image.fromarray(normalized_image)
            if output_format.lower() == 'png':
                final_pil_image.save(final_output_path, 'PNG', optimize=True)
            elif output_format.lower() == 'jpg':
                final_pil_image.save(final_output_path, 'JPEG', quality=95)
            elif output_format.lower() == 'tiff':
                final_pil_image.save(final_output_path, 'TIFF', compression='tiff_lzw')
            else:
                final_pil_image.save(final_output_path, 'PNG', optimize=True)
            
            # Save intermediate images if requested
            intermediate_paths = {}
            if save_intermediate:
                corrected_path = os.path.join(self.output_dir, f"{base_name}_corrected.{output_format}")
                corrected_pil_image = Image.fromarray(corrected_image)
                if output_format.lower() == 'png':
                    corrected_pil_image.save(corrected_path, 'PNG', optimize=True)
                elif output_format.lower() == 'jpg':
                    corrected_pil_image.save(corrected_path, 'JPEG', quality=95)
                elif output_format.lower() == 'tiff':
                    corrected_pil_image.save(corrected_path, 'TIFF', compression='tiff_lzw')
                else:
                    corrected_pil_image.save(corrected_path, 'PNG', optimize=True)
                intermediate_paths["corrected"] = corrected_path
            
            # Prepare results
            results = {
                "original_image_path": image if isinstance(image, str) else "input_image",
                "processed_image_path": final_output_path,
                "processing_statistics": {
                    "original_brightness": round(original_brightness, 2),
                    "corrected_brightness": round(corrected_brightness, 2),
                    "final_brightness": round(final_brightness, 2),
                    "target_brightness": target_brightness,
                    "brightness_improvement": round(final_brightness - original_brightness, 2)
                },
                "parameters_used": {
                    "target_brightness": target_brightness,
                    "gaussian_kernel_size": gaussian_kernel_size,
                    "output_format": output_format
                }
            }
            
            # Add intermediate paths if saved
            if save_intermediate:
                results["intermediate_paths"] = intermediate_paths
            
            # Create an instance of the visualization config
            vis_config = VisualizationConfig()
            
            # Create the comparison plot for visualization
            comparison_plot_path = self.create_comparison_plot(
                original_image, corrected_image, normalized_image, filename, vis_config
            )
            
            # Add visual outputs for Gradio
            visual_outputs = [comparison_plot_path, final_output_path]
            if save_intermediate:
                visual_outputs.append(intermediate_paths["corrected"])
            results["visual_outputs"] = visual_outputs
            
            return results
            
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

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
