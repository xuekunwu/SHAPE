"""
Unified Image Data Representation

Core principle: Single-channel is a special case of multi-channel (C=1)
All images are represented as (H, W, C) format where C >= 1
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
import tifffile
from PIL import Image
import os


@dataclass
class ImageData:
    """
    Unified image data representation: single-channel is a special case of multi-channel (1 channel)
    
    Core principles:
    - All images are represented as (H, W, C) format, C >= 1
    - Single-channel images: C=1
    - Multi-channel images: C>1
    """
    data: np.ndarray  # Always in (H, W, C) format
    dtype: np.dtype
    channel_names: Optional[List[str]] = None  # e.g., ['bright-field', 'GFP', 'DAPI']
    source_path: Optional[str] = None
    
    @classmethod
    def from_path(cls, path: str, channel_names: Optional[List[str]] = None) -> 'ImageData':
        """
        Load image from file path, automatically detect and handle multi-channel
        
        Unified logic:
        1. Load image (TIFF/PIL)
        2. Normalize to (H, W, C) format
        3. Detect channel count
        4. Infer channel names (if not provided)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        
        path_lower = path.lower()
        is_tiff = path_lower.endswith('.tif') or path_lower.endswith('.tiff')
        
        if is_tiff:
            # Load with tifffile to handle multi-channel TIFF
            try:
                img_full = tifffile.imread(path)
                return cls._from_array(img_full, channel_names=channel_names, source_path=path)
            except Exception as e:
                # Fallback to PIL if tifffile fails
                print(f"Warning: tifffile failed for {path}, trying PIL: {e}")
                img = Image.open(path)
                img_array = np.array(img)
                return cls._from_array(img_array, channel_names=channel_names, source_path=path)
        else:
            # Load with PIL for other formats
            img = Image.open(path)
            img_array = np.array(img)
            return cls._from_array(img_array, channel_names=channel_names, source_path=path)
    
    @classmethod
    def _from_array(cls, img_array: np.ndarray, channel_names: Optional[List[str]] = None, source_path: Optional[str] = None) -> 'ImageData':
        """
        Create ImageData from numpy array, normalize to (H, W, C) format
        """
        # Normalize to (H, W, C) format
        if img_array.ndim == 2:
            # 2D grayscale: add channel dimension -> (H, W, 1)
            img_array = img_array[:, :, np.newaxis]
        elif img_array.ndim == 3:
            # 3D: determine if (H, W, C) or (C, H, W)
            h, w, c = img_array.shape
            if c > 1 and c <= 4:
                # Likely (H, W, C) - already in correct format
                pass
            elif h <= 4 and w > h:
                # Likely (C, H, W) - transpose to (H, W, C)
                img_array = np.transpose(img_array, (1, 2, 0))
            else:
                # Ambiguous: assume (H, W, C) if last dim is small, else (C, H, W)
                if c <= 4:
                    # Assume (H, W, C)
                    pass
                else:
                    # Assume (C, H, W)
                    img_array = np.transpose(img_array, (1, 2, 0))
        elif img_array.ndim == 4:
            # 4D: could be (Z, H, W, C) or (C, Z, H, W)
            # Use first slice for 4D images
            if img_array.shape[3] <= 4:
                # (Z, H, W, C) - take first Z slice
                img_array = img_array[0, :, :, :]
            else:
                # (C, Z, H, W) - take first Z slice and transpose
                img_array = np.transpose(img_array[:, 0, :, :], (1, 2, 0))
        else:
            raise ValueError(f"Unsupported image dimensions: {img_array.ndim}")
        
        # Ensure we have (H, W, C) format
        if img_array.ndim != 3:
            raise ValueError(f"Failed to normalize to 3D array, got shape: {img_array.shape}")
        
        h, w, c = img_array.shape
        
        # Infer channel names if not provided
        if channel_names is None:
            channel_names = cls._infer_channel_names(c, source_path)
        
        return cls(
            data=img_array,
            dtype=img_array.dtype,
            channel_names=channel_names,
            source_path=source_path
        )
    
    @staticmethod
    def _infer_channel_names(num_channels: int, source_path: Optional[str] = None) -> List[str]:
        """
        Infer channel names based on channel count and file path
        """
        if num_channels == 1:
            return ['grayscale']
        elif num_channels == 2:
            # Common: bright-field + GFP
            return ['bright-field', 'GFP']
        elif num_channels == 3:
            # Common: bright-field + GFP + DAPI
            return ['bright-field', 'GFP', 'DAPI']
        elif num_channels == 4:
            return ['bright-field', 'GFP', 'DAPI', 'Channel 4']
        else:
            return [f'Channel {i+1}' for i in range(num_channels)]
    
    def get_channel(self, idx: int) -> np.ndarray:
        """Get single channel (returns 2D array)"""
        if idx < 0 or idx >= self.num_channels:
            raise IndexError(f"Channel index {idx} out of range [0, {self.num_channels})")
        return self.data[:, :, idx]
    
    def get_channels(self, indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """Get multiple channels"""
        if indices is None:
            indices = list(range(self.num_channels))
        return [self.get_channel(i) for i in indices]
    
    @property
    def num_channels(self) -> int:
        """Number of channels"""
        return self.data.shape[2]
    
    @property
    def height(self) -> int:
        """Image height"""
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        """Image width"""
        return self.data.shape[1]
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Image shape (H, W, C)"""
        return self.data.shape
    
    @property
    def is_multi_channel(self) -> bool:
        """Whether this is a multi-channel image (>1 channel)"""
        return self.num_channels > 1
    
    @property
    def is_single_channel(self) -> bool:
        """Whether this is a single-channel image (1 channel)"""
        return self.num_channels == 1
    
    def to_segmentation_input(self, channel_idx: int = 0) -> np.ndarray:
        """
        Convert to format needed by segmentation tools (single channel, float32)
        Default uses first channel
        """
        channel = self.get_channel(channel_idx)
        if channel.dtype != np.float32:
            channel = channel.astype(np.float32)
        return channel
    
    def to_uint8(self, channel_idx: Optional[int] = None) -> np.ndarray:
        """
        Convert to uint8 format for display
        If channel_idx is None, converts all channels
        """
        if channel_idx is not None:
            channel = self.get_channel(channel_idx)
            if channel.dtype == np.uint16:
                return (channel / 65535.0 * 255).astype(np.uint8)
            elif channel.dtype != np.uint8:
                return np.clip(channel, 0, 255).astype(np.uint8)
            return channel
        else:
            # Convert all channels
            result = self.data.copy()
            if result.dtype == np.uint16:
                result = (result / 65535.0 * 255).astype(np.uint8)
            elif result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)
            return result
    
    def create_merged_rgb(self, channel_mapping: Optional[Dict[int, str]] = None) -> np.ndarray:
        """
        Create merged RGB view for visualization
        
        Unified logic:
        - Single-channel: grayscale -> RGB
        - Multi-channel: merge according to channel_mapping
        """
        def normalize_channel(x):
            """Normalize channel to [0, 1] range"""
            x = x.astype(np.float32)
            x_min = x.min()
            x_max = x.max()
            if x_max > x_min:
                return (x - x_min) / (x_max - x_min + 1e-8)
            return x
        
        if self.is_single_channel:
            # Single-channel: grayscale -> RGB
            gray = normalize_channel(self.get_channel(0))
            merged = np.stack([gray, gray, gray], axis=2)
        else:
            # Multi-channel: merge channels
            merged = np.zeros((self.height, self.width, 3), dtype=np.float32)
            
            # Normalize all channels
            normalized_channels = []
            for c in range(self.num_channels):
                normalized_channels.append(normalize_channel(self.get_channel(c)))
            
            # Default mapping: bright-field -> gray, GFP -> green, DAPI -> blue
            if channel_mapping is None:
                channel_mapping = {}
                for i, name in enumerate(self.channel_names or []):
                    name_lower = name.lower()
                    if 'bright' in name_lower or 'field' in name_lower or i == 0:
                        channel_mapping[i] = 'gray'
                    elif 'gfp' in name_lower or i == 1:
                        channel_mapping[i] = 'green'
                    elif 'dapi' in name_lower or i == 2:
                        channel_mapping[i] = 'blue'
            
            # Apply mapping
            for ch_idx, color in channel_mapping.items():
                if ch_idx >= self.num_channels:
                    continue
                ch_norm = normalized_channels[ch_idx]
                
                if color == 'gray' or color == 'red':
                    merged[:, :, 0] = ch_norm  # Red
                    merged[:, :, 1] = ch_norm  # Green
                    merged[:, :, 2] = ch_norm  # Blue
                elif color == 'green':
                    merged[:, :, 1] = merged[:, :, 1] + ch_norm  # Add to green
                elif color == 'blue':
                    merged[:, :, 2] = merged[:, :, 2] + ch_norm  # Add to blue
            
            # Clip to [0, 1]
            merged = np.clip(merged, 0, 1)
        
        return merged
    
    def save(self, path: str, format: Optional[str] = None) -> str:
        """
        Save image to file
        For multi-channel, saves as TIFF to preserve channels
        For single-channel, saves in specified format (default: PNG)
        """
        if format is None:
            format = 'tiff' if self.is_multi_channel else 'png'
        
        if self.is_multi_channel and format.lower() in ('tif', 'tiff'):
            # Save as multi-channel TIFF
            tifffile.imwrite(path, self.data)
        else:
            # Save as single-channel or RGB image
            if self.is_single_channel:
                img_uint8 = self.to_uint8(0)
                Image.fromarray(img_uint8, mode='L').save(path)
            else:
                # Convert to RGB for display
                rgb = self.create_merged_rgb()
                rgb_uint8 = (rgb * 255).astype(np.uint8)
                Image.fromarray(rgb_uint8, mode='RGB').save(path)
        
        return path

