"""
Basic tests for ImageData and ImageProcessor
"""

import numpy as np
import os
import tempfile
from octotools.models.image_data import ImageData
from octotools.utils.image_processor import ImageProcessor


def test_image_data_from_array():
    """Test ImageData creation from numpy array"""
    # Test single-channel (2D)
    img_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    img_data = ImageData._from_array(img_2d)
    assert img_data.shape == (100, 100, 1), f"Expected (100, 100, 1), got {img_data.shape}"
    assert img_data.is_single_channel, "Should be single channel"
    
    # Test multi-channel (H, W, C)
    img_3d = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_data = ImageData._from_array(img_3d)
    assert img_data.shape == (100, 100, 3), f"Expected (100, 100, 3), got {img_data.shape}"
    assert img_data.is_multi_channel, "Should be multi-channel"
    
    # Test (C, H, W) format
    img_chw = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)
    img_data = ImageData._from_array(img_chw)
    assert img_data.shape == (100, 100, 3), f"Expected (100, 100, 3) after transpose, got {img_data.shape}"
    
    print("✓ ImageData from array tests passed")


def test_image_data_channels():
    """Test channel access methods"""
    img_3d = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
    img_data = ImageData._from_array(img_3d)
    
    # Test get_channel
    ch0 = img_data.get_channel(0)
    assert ch0.shape == (50, 50), f"Expected (50, 50), got {ch0.shape}"
    
    # Test get_channels
    channels = img_data.get_channels([0, 1])
    assert len(channels) == 2, f"Expected 2 channels, got {len(channels)}"
    
    print("✓ Channel access tests passed")


def test_image_data_properties():
    """Test ImageData properties"""
    img_3d = np.random.randint(0, 255, (100, 200, 2), dtype=np.uint8)
    img_data = ImageData._from_array(img_3d)
    
    assert img_data.height == 100, f"Expected height 100, got {img_data.height}"
    assert img_data.width == 200, f"Expected width 200, got {img_data.width}"
    assert img_data.num_channels == 2, f"Expected 2 channels, got {img_data.num_channels}"
    
    print("✓ ImageData properties tests passed")


def test_merged_rgb():
    """Test merged RGB creation"""
    # Single channel
    img_2d = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    img_data = ImageData._from_array(img_2d)
    merged = img_data.create_merged_rgb()
    assert merged.shape == (50, 50, 3), f"Expected (50, 50, 3), got {merged.shape}"
    
    # Multi-channel
    img_3d = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
    img_data = ImageData._from_array(img_3d, channel_names=['bright-field', 'GFP'])
    merged = img_data.create_merged_rgb()
    assert merged.shape == (50, 50, 3), f"Expected (50, 50, 3), got {merged.shape}"
    
    print("✓ Merged RGB tests passed")


if __name__ == "__main__":
    print("Running ImageData tests...")
    test_image_data_from_array()
    test_image_data_channels()
    test_image_data_properties()
    test_merged_rgb()
    print("\n✅ All tests passed!")

