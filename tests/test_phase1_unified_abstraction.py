"""
Test Phase 1: Unified Image Abstraction Layer
Tests for ImageData and ImageProcessor classes
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import tempfile
from PIL import Image
import tifffile

from octotools.models.image_data import ImageData
from octotools.utils.image_processor import ImageProcessor


def test_image_data_from_array_2d():
    """Test ImageData creation from 2D array (single channel)"""
    print("\n=== Test 1: ImageData from 2D array (single channel) ===")
    img_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    img_data = ImageData._from_array(img_2d)
    
    assert img_data.shape == (100, 100, 1), f"Expected (100, 100, 1), got {img_data.shape}"
    assert img_data.is_single_channel, "Should be single channel"
    assert img_data.num_channels == 1, f"Expected 1 channel, got {img_data.num_channels}"
    assert img_data.height == 100, f"Expected height 100, got {img_data.height}"
    assert img_data.width == 100, f"Expected width 100, got {img_data.width}"
    
    print("[OK] 2D array correctly converted to (H, W, 1)")
    print(f"  Shape: {img_data.shape}")
    print(f"  Is single channel: {img_data.is_single_channel}")


def test_image_data_from_array_3d_hwc():
    """Test ImageData creation from 3D array (H, W, C) format"""
    print("\n=== Test 2: ImageData from 3D array (H, W, C) ===")
    img_3d = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_data = ImageData._from_array(img_3d)
    
    assert img_data.shape == (100, 100, 3), f"Expected (100, 100, 3), got {img_data.shape}"
    assert img_data.is_multi_channel, "Should be multi-channel"
    assert img_data.num_channels == 3, f"Expected 3 channels, got {img_data.num_channels}"
    
    print("[OK] 3D array (H, W, C) correctly handled")
    print(f"  Shape: {img_data.shape}")
    print(f"  Is multi-channel: {img_data.is_multi_channel}")


def test_image_data_from_array_3d_chw():
    """Test ImageData creation from 3D array (C, H, W) format"""
    print("\n=== Test 3: ImageData from 3D array (C, H, W) ===")
    img_chw = np.random.randint(0, 255, (2, 100, 100), dtype=np.uint8)
    img_data = ImageData._from_array(img_chw)
    
    assert img_data.shape == (100, 100, 2), f"Expected (100, 100, 2) after transpose, got {img_data.shape}"
    assert img_data.is_multi_channel, "Should be multi-channel"
    
    print("[OK] 3D array (C, H, W) correctly transposed to (H, W, C)")
    print(f"  Shape: {img_data.shape}")


def test_channel_access():
    """Test channel access methods"""
    print("\n=== Test 4: Channel access methods ===")
    img_3d = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
    img_data = ImageData._from_array(img_3d, channel_names=['bright-field', 'GFP'])
    
    # Test get_channel
    ch0 = img_data.get_channel(0)
    assert ch0.shape == (50, 50), f"Expected (50, 50), got {ch0.shape}"
    
    # Test get_channels
    channels = img_data.get_channels([0, 1])
    assert len(channels) == 2, f"Expected 2 channels, got {len(channels)}"
    
    # Test channel names
    assert img_data.channel_names == ['bright-field', 'GFP'], \
        f"Expected ['bright-field', 'GFP'], got {img_data.channel_names}"
    
    print("[OK] Channel access methods work correctly")
    print(f"  Channel 0 shape: {ch0.shape}")
    print(f"  Channel names: {img_data.channel_names}")


def test_to_segmentation_input():
    """Test conversion to segmentation input format"""
    print("\n=== Test 5: Conversion to segmentation input ===")
    img_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    img_data = ImageData._from_array(img_2d)
    
    seg_input = img_data.to_segmentation_input()
    assert seg_input.dtype == np.float32, f"Expected float32, got {seg_input.dtype}"
    assert seg_input.shape == (100, 100), f"Expected (100, 100), got {seg_input.shape}"
    
    print("[OK] Correctly converted to segmentation input format")
    print(f"  Dtype: {seg_input.dtype}")
    print(f"  Shape: {seg_input.shape}")


def test_merged_rgb_single_channel():
    """Test merged RGB creation for single channel"""
    print("\n=== Test 6: Merged RGB (single channel) ===")
    img_2d = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    img_data = ImageData._from_array(img_2d)
    
    merged = img_data.create_merged_rgb()
    assert merged.shape == (50, 50, 3), f"Expected (50, 50, 3), got {merged.shape}"
    assert merged.dtype == np.float32, f"Expected float32, got {merged.dtype}"
    assert 0 <= merged.min() and merged.max() <= 1, "Values should be in [0, 1] range"
    
    print("[OK] Single channel correctly converted to RGB")
    print(f"  Shape: {merged.shape}")
    print(f"  Value range: [{merged.min():.3f}, {merged.max():.3f}]")


def test_merged_rgb_multi_channel():
    """Test merged RGB creation for multi-channel"""
    print("\n=== Test 7: Merged RGB (multi-channel) ===")
    img_3d = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
    img_data = ImageData._from_array(img_3d, channel_names=['bright-field', 'GFP'])
    
    merged = img_data.create_merged_rgb()
    assert merged.shape == (50, 50, 3), f"Expected (50, 50, 3), got {merged.shape}"
    assert merged.dtype == np.float32, f"Expected float32, got {merged.dtype}"
    
    print("[OK] Multi-channel correctly merged to RGB")
    print(f"  Shape: {merged.shape}")
    print(f"  Value range: [{merged.min():.3f}, {merged.max():.3f}]")


def test_save_image():
    """Test image saving"""
    print("\n=== Test 8: Image saving ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test single channel save
        img_2d = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        img_data = ImageData._from_array(img_2d)
        png_path = os.path.join(tmpdir, "test_single.png")
        saved_path = img_data.save(png_path)
        assert os.path.exists(saved_path), f"File not saved: {saved_path}"
        
        # Test multi-channel save (should save as TIFF)
        img_3d = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
        img_data_multi = ImageData._from_array(img_3d)
        tiff_path = os.path.join(tmpdir, "test_multi.tiff")
        saved_path_multi = img_data_multi.save(tiff_path)
        assert os.path.exists(saved_path_multi), f"File not saved: {saved_path_multi}"
        
        # Verify multi-channel TIFF can be loaded
        loaded = tifffile.imread(saved_path_multi)
        assert loaded.shape == (50, 50, 2) or loaded.shape == (2, 50, 50), \
            f"Unexpected shape after loading: {loaded.shape}"
        
        print("[OK] Image saving works correctly")
        print(f"  Single channel saved: {saved_path}")
        print(f"  Multi-channel saved: {saved_path_multi}")


def test_image_processor_load():
    """Test ImageProcessor.load_image"""
    print("\n=== Test 9: ImageProcessor.load_image ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        # Single channel PNG
        img_2d = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        png_path = os.path.join(tmpdir, "test.png")
        Image.fromarray(img_2d, mode='L').save(png_path)
        
        # Load with ImageProcessor
        img_data = ImageProcessor.load_image(png_path)
        assert img_data.shape == (50, 50, 1), f"Expected (50, 50, 1), got {img_data.shape}"
        
        print("[OK] ImageProcessor.load_image works correctly")
        print(f"  Loaded shape: {img_data.shape}")


def test_channel_name_inference():
    """Test channel name inference"""
    print("\n=== Test 10: Channel name inference ===")
    
    # Test 1 channel
    img_1ch = np.random.randint(0, 255, (50, 50, 1), dtype=np.uint8)
    img_data = ImageData._from_array(img_1ch)
    assert img_data.channel_names == ['grayscale'], \
        f"Expected ['grayscale'], got {img_data.channel_names}"
    
    # Test 2 channels
    img_2ch = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
    img_data = ImageData._from_array(img_2ch)
    assert img_data.channel_names == ['bright-field', 'GFP'], \
        f"Expected ['bright-field', 'GFP'], got {img_data.channel_names}"
    
    # Test 3 channels
    img_3ch = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    img_data = ImageData._from_array(img_3ch)
    assert img_data.channel_names == ['bright-field', 'GFP', 'DAPI'], \
        f"Expected ['bright-field', 'GFP', 'DAPI'], got {img_data.channel_names}"
    
    print("[OK] Channel name inference works correctly")
    print(f"  1 channel: {ImageData._from_array(np.random.randint(0, 255, (10, 10, 1), dtype=np.uint8)).channel_names}")
    print(f"  2 channels: {ImageData._from_array(np.random.randint(0, 255, (10, 10, 2), dtype=np.uint8)).channel_names}")
    print(f"  3 channels: {ImageData._from_array(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)).channel_names}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Phase 1: Unified Image Abstraction Layer - Test Suite")
    print("=" * 60)
    
    tests = [
        test_image_data_from_array_2d,
        test_image_data_from_array_3d_hwc,
        test_image_data_from_array_3d_chw,
        test_channel_access,
        test_to_segmentation_input,
        test_merged_rgb_single_channel,
        test_merged_rgb_multi_channel,
        test_save_image,
        test_image_processor_load,
        test_channel_name_inference,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] Test failed: {test_func.__name__}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n[SUCCESS] All tests passed! Phase 1 implementation is working correctly.")
    else:
        print(f"\n[ERROR] {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

