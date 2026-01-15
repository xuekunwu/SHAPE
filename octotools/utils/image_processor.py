"""
Unified Image Processing Utility

Provides unified image loading, processing, and visualization utilities
All tools should use this class for consistent image handling
"""

import os
import numpy as np
from typing import Optional, List, Dict, Tuple
from PIL import Image
import matplotlib.pyplot as plt

from octotools.models.image_data import ImageData
from octotools.models.utils import VisualizationConfig


class ImageProcessor:
    """
    Unified image processing utility class
    All tools should use this class for consistent image handling
    """
    
    @staticmethod
    def load_image(path: str, channel_names: Optional[List[str]] = None) -> ImageData:
        """
        Unified image loading interface
        
        Uses skimage.io.imread for consistent image loading (handles TIFF, PNG, JPEG, etc.)
        Consistent with reference implementation for multi-channel image processing.
        
        Args:
            path: Path to image file
            channel_names: Optional list of channel names (e.g., ['bright-field', 'GFP'])
        
        Returns:
            ImageData object with normalized (H, W, C) format
        """
        return ImageData.from_path(path, channel_names=channel_names)
    
    @staticmethod
    def normalize_for_display(img_data: ImageData, channel_idx: int = 0) -> np.ndarray:
        """
        Normalize single channel for display (uint8)
        
        Args:
            img_data: ImageData object
            channel_idx: Channel index to normalize (default: 0)
        
        Returns:
            Normalized uint8 array (2D)
        """
        return img_data.to_uint8(channel_idx)
    
    @staticmethod
    def create_multi_channel_visualization(
        img_data: ImageData,
        output_path: str,
        vis_config: VisualizationConfig,
        group: Optional[str] = None,
        image_identifier: Optional[str] = None
    ) -> Optional[str]:
        """
        Create multi-channel visualization (unified logic)
        
        Shows individual channels and merged RGB view
        
        Args:
            img_data: ImageData object
            output_path: Path to save visualization
            vis_config: VisualizationConfig instance
            group: Optional group name
            image_identifier: Optional image identifier
        
        Returns:
            Path to saved visualization, or None if failed
        """
        try:
            num_channels = img_data.num_channels
            
            # Create figure with appropriate layout
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
            for ch_idx in range(num_channels):
                if ch_idx < len(axs):
                    ax = axs[ch_idx]
                    channel = img_data.get_channel(ch_idx)
                    channel_uint8 = img_data.to_uint8(ch_idx)
                    
                    # Determine colormap based on channel name
                    channel_name = img_data.channel_names[ch_idx] if img_data.channel_names else f'Channel {ch_idx+1}'
                    name_lower = channel_name.lower()
                    
                    if 'gfp' in name_lower or 'green' in name_lower:
                        cmap = 'Greens'
                    elif 'dapi' in name_lower or 'blue' in name_lower:
                        cmap = 'Blues'
                    elif 'bright' in name_lower or 'field' in name_lower or 'phase' in name_lower:
                        cmap = 'gray'
                    else:
                        cmap = 'gray'
                    
                    ax.imshow(channel_uint8, cmap=cmap)
                    ax.set_title(f"{channel_name}", fontsize=10)
                    ax.axis('off')
            
            # Display merged RGB view
            merged_idx = num_channels
            if merged_idx < len(axs):
                ax = axs[merged_idx]
                merged_rgb = img_data.create_merged_rgb()
                merged_rgb_uint8 = (merged_rgb * 255).astype(np.uint8)
                ax.imshow(merged_rgb_uint8)
                ax.set_title("Merged RGB", fontsize=10)
                ax.axis('off')
            
            # Hide unused subplots
            for idx in range(merged_idx + 1, len(axs)):
                axs[idx].axis('off')
            
            # Set title
            filename = os.path.basename(img_data.source_path) if img_data.source_path else "Image"
            title = f"Multi-channel Visualization: {filename}"
            if group:
                title += f" (Group: {group})"
            fig.suptitle(title, fontsize=12, y=0.98)
            
            # Apply professional styling
            plt.tight_layout()
            
            # Save using VisualizationConfig
            vis_config.save_professional_figure(fig, output_path)
            plt.close(fig)
            
            return output_path if os.path.exists(output_path) else None
            
        except Exception as e:
            print(f"Warning: Failed to create multi-channel visualization: {e}")
            return None
    
    @staticmethod
    def create_merged_rgb_for_display(img_data: ImageData) -> np.ndarray:
        """
        Create merged RGB image for display (returns uint8 array)
        
        Args:
            img_data: ImageData object
        
        Returns:
            Merged RGB image as uint8 array (H, W, 3)
        """
        merged = img_data.create_merged_rgb()
        return (merged * 255).astype(np.uint8)
    
    @staticmethod
    def extract_channel_for_segmentation(img_data: ImageData, channel_idx: int = 0) -> np.ndarray:
        """
        Extract channel for segmentation (float32 format)
        
        Args:
            img_data: ImageData object
            channel_idx: Channel index to extract (default: 0, typically bright-field/phase contrast)
        
        Returns:
            Single channel as float32 array (2D)
        """
        return img_data.to_segmentation_input(channel_idx)
    
    @staticmethod
    def save_multi_channel_crop(img_data: ImageData, path: str) -> str:
        """
        Save multi-channel crop preserving all channels
        
        Args:
            img_data: ImageData object (crop)
            path: Path to save
            
        Returns:
            Path to saved file
        """
        return img_data.save(path, format='tiff' if img_data.is_multi_channel else 'png')
    
    @staticmethod
    def save_image(img_data: ImageData, path: str, format: Optional[str] = None) -> str:
        """
        Save image to file with automatic format detection
        
        Args:
            img_data: ImageData object
            path: Path to save
            format: Optional format override ('tiff', 'png', etc.)
            
        Returns:
            Path to saved file
        """
        return img_data.save(path, format=format)
    
    @staticmethod
    def copy_image_preserving_format(source_path: str, dest_path: str) -> str:
        """
        Copy image file preserving multi-channel format
        
        Args:
            source_path: Source image path
            dest_path: Destination image path
            
        Returns:
            Destination path
        """
        import shutil
        import tifffile
        
        # Check if source is multi-channel TIFF
        try:
            img_data = ImageProcessor.load_image(source_path)
            if img_data.is_multi_channel:
                # Use ImageData.save to preserve format
                img_data.save(dest_path, format='tiff')
            else:
                # Simple copy for single-channel
                shutil.copy(source_path, dest_path)
        except Exception as e:
            # Fallback to simple copy
            print(f"Warning: Failed to preserve format during copy: {e}, using simple copy")
            shutil.copy(source_path, dest_path)
        
        return dest_path
    
    @staticmethod
    def is_multi_channel_image(path: str) -> bool:
        """
        Check if image is multi-channel without full loading
        
        Args:
            path: Image file path
            
        Returns:
            True if multi-channel, False otherwise
        """
        try:
            img_data = ImageProcessor.load_image(path)
            return img_data.is_multi_channel
        except Exception:
            return False
    
    @staticmethod
    def load_for_display(path: str) -> Optional[Image.Image]:
        """
        Load image for display (returns PIL Image)
        Handles multi-channel by creating merged RGB view
        
        Args:
            path: Image file path
            
        Returns:
            PIL Image for display, or None if failed
        """
        try:
            img_data = ImageProcessor.load_image(path)
            
            if img_data.is_multi_channel:
                # Create merged RGB for multi-channel
                merged_rgb = ImageProcessor.create_merged_rgb_for_display(img_data)
                return Image.fromarray(merged_rgb, mode='RGB')
            else:
                # Single channel: convert to grayscale PIL Image
                channel_uint8 = img_data.to_uint8(0)
                return Image.fromarray(channel_uint8, mode='L')
        except Exception as e:
            print(f"Warning: Failed to load image for display: {e}")
            # Fallback to PIL
            try:
                return Image.open(path)
            except Exception:
                return None

