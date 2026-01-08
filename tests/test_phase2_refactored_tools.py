"""
Test Phase 2: Refactored Tools with Unified Abstraction Layer
Tests for tools that have been refactored to use ImageData and ImageProcessor
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


def test_image_processor_load_single_channel():
    """Test ImageProcessor.load_image for single-channel image"""
    print("\n=== Test 1: ImageProcessor.load_image (single-channel) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test single-channel PNG
        img_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        png_path = os.path.join(tmpdir, "test_single.png")
        Image.fromarray(img_2d, mode='L').save(png_path)
        
        # Load with ImageProcessor
        img_data = ImageProcessor.load_image(png_path)
        
        assert img_data.shape == (100, 100, 1), f"Expected (100, 100, 1), got {img_data.shape}"
        assert img_data.is_single_channel, "Should be single channel"
        
        print("[OK] Single-channel image loaded correctly")
        print(f"  Shape: {img_data.shape}")


def test_image_processor_load_multi_channel():
    """Test ImageProcessor.load_image for multi-channel TIFF"""
    print("\n=== Test 2: ImageProcessor.load_image (multi-channel) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test multi-channel TIFF
        img_3d = np.random.randint(0, 255, (100, 100, 2), dtype=np.uint8)
        tiff_path = os.path.join(tmpdir, "test_multi.tiff")
        tifffile.imwrite(tiff_path, img_3d)
        
        # Load with ImageProcessor
        img_data = ImageProcessor.load_image(tiff_path)
        
        assert img_data.shape == (100, 100, 2), f"Expected (100, 100, 2), got {img_data.shape}"
        assert img_data.is_multi_channel, "Should be multi-channel"
        assert img_data.channel_names == ['bright-field', 'GFP'], \
            f"Expected ['bright-field', 'GFP'], got {img_data.channel_names}"
        
        print("[OK] Multi-channel TIFF loaded correctly")
        print(f"  Shape: {img_data.shape}")
        print(f"  Channel names: {img_data.channel_names}")


def test_to_segmentation_input():
    """Test to_segmentation_input method"""
    print("\n=== Test 3: to_segmentation_input ===")
    img_2d = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    img_data = ImageData._from_array(img_2d)
    
    seg_input = img_data.to_segmentation_input()
    assert seg_input.dtype == np.float32, f"Expected float32, got {seg_input.dtype}"
    assert seg_input.shape == (50, 50), f"Expected (50, 50), got {seg_input.shape}"
    
    print("[OK] to_segmentation_input works correctly")
    print(f"  Dtype: {seg_input.dtype}")
    print(f"  Shape: {seg_input.shape}")


def test_to_uint8():
    """Test to_uint8 method"""
    print("\n=== Test 4: to_uint8 ===")
    # Test with uint16
    img_uint16 = np.random.randint(0, 65535, (50, 50), dtype=np.uint16)
    img_data = ImageData._from_array(img_uint16)
    
    uint8_channel = img_data.to_uint8(0)
    assert uint8_channel.dtype == np.uint8, f"Expected uint8, got {uint8_channel.dtype}"
    assert uint8_channel.max() <= 255, "Values should be <= 255"
    
    print("[OK] to_uint8 works correctly")
    print(f"  Dtype: {uint8_channel.dtype}")
    print(f"  Value range: [{uint8_channel.min()}, {uint8_channel.max()}]")


def test_create_merged_rgb_for_display():
    """Test ImageProcessor.create_merged_rgb_for_display"""
    print("\n=== Test 5: create_merged_rgb_for_display ===")
    # Single channel
    img_2d = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    img_data = ImageData._from_array(img_2d)
    merged = ImageProcessor.create_merged_rgb_for_display(img_data)
    assert merged.shape == (50, 50, 3), f"Expected (50, 50, 3), got {merged.shape}"
    assert merged.dtype == np.uint8, f"Expected uint8, got {merged.dtype}"
    
    # Multi-channel
    img_3d = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
    img_data_multi = ImageData._from_array(img_3d, channel_names=['bright-field', 'GFP'])
    merged_multi = ImageProcessor.create_merged_rgb_for_display(img_data_multi)
    assert merged_multi.shape == (50, 50, 3), f"Expected (50, 50, 3), got {merged_multi.shape}"
    
    print("[OK] create_merged_rgb_for_display works correctly")
    print(f"  Single channel merged shape: {merged.shape}")
    print(f"  Multi-channel merged shape: {merged_multi.shape}")


def test_extract_channel_for_segmentation():
    """Test ImageProcessor.extract_channel_for_segmentation"""
    print("\n=== Test 6: extract_channel_for_segmentation ===")
    img_3d = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
    img_data = ImageData._from_array(img_3d)
    
    seg_channel = ImageProcessor.extract_channel_for_segmentation(img_data, channel_idx=0)
    assert seg_channel.dtype == np.float32, f"Expected float32, got {seg_channel.dtype}"
    assert seg_channel.shape == (50, 50), f"Expected (50, 50), got {seg_channel.shape}"
    
    print("[OK] extract_channel_for_segmentation works correctly")
    print(f"  Dtype: {seg_channel.dtype}")
    print(f"  Shape: {seg_channel.shape}")


def test_save_multi_channel_crop():
    """Test ImageProcessor.save_multi_channel_crop"""
    print("\n=== Test 7: save_multi_channel_crop ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multi-channel crop
        crop_3d = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
        img_data = ImageData._from_array(crop_3d)
        
        crop_path = os.path.join(tmpdir, "test_crop.tiff")
        saved_path = ImageProcessor.save_multi_channel_crop(img_data, crop_path)
        
        assert os.path.exists(saved_path), f"File not saved: {saved_path}"
        
        # Verify can be loaded back
        loaded = tifffile.imread(saved_path)
        assert loaded.shape == (50, 50, 2) or loaded.shape == (2, 50, 50), \
            f"Unexpected shape after loading: {loaded.shape}"
        
        print("[OK] save_multi_channel_crop works correctly")
        print(f"  Saved path: {saved_path}")
        print(f"  Loaded shape: {loaded.shape}")


def test_image_data_from_path():
    """Test ImageData.from_path with actual files"""
    print("\n=== Test 8: ImageData.from_path ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test PNG
        img_2d = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        png_path = os.path.join(tmpdir, "test.png")
        Image.fromarray(img_2d, mode='L').save(png_path)
        
        img_data = ImageData.from_path(png_path)
        assert img_data.shape == (50, 50, 1), f"Expected (50, 50, 1), got {img_data.shape}"
        
        # Test TIFF
        img_3d = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
        tiff_path = os.path.join(tmpdir, "test.tiff")
        tifffile.imwrite(tiff_path, img_3d)
        
        img_data_tiff = ImageData.from_path(tiff_path)
        assert img_data_tiff.shape == (50, 50, 2), f"Expected (50, 50, 2), got {img_data_tiff.shape}"
        
        print("[OK] ImageData.from_path works correctly")
        print(f"  PNG shape: {img_data.shape}")
        print(f"  TIFF shape: {img_data_tiff.shape}")


def test_channel_access_unified():
    """Test unified channel access across different image types"""
    print("\n=== Test 9: Unified channel access ===")
    
    # Single channel
    img_1ch = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    img_data_1ch = ImageData._from_array(img_1ch)
    ch0 = img_data_1ch.get_channel(0)
    assert ch0.shape == (50, 50), f"Expected (50, 50), got {ch0.shape}"
    
    # Multi-channel
    img_2ch = np.random.randint(0, 255, (50, 50, 2), dtype=np.uint8)
    img_data_2ch = ImageData._from_array(img_2ch)
    ch0_multi = img_data_2ch.get_channel(0)
    ch1_multi = img_data_2ch.get_channel(1)
    assert ch0_multi.shape == (50, 50), f"Expected (50, 50), got {ch0_multi.shape}"
    assert ch1_multi.shape == (50, 50), f"Expected (50, 50), got {ch1_multi.shape}"
    
    print("[OK] Unified channel access works correctly")
    print(f"  Single channel: {ch0.shape}")
    print(f"  Multi-channel ch0: {ch0_multi.shape}, ch1: {ch1_multi.shape}")


def test_backward_compatibility_format():
    """Test that ImageData maintains backward compatibility with different formats"""
    print("\n=== Test 10: Backward compatibility (format handling) ===")
    
    # Test (C, H, W) format
    img_chw = np.random.randint(0, 255, (2, 50, 50), dtype=np.uint8)
    img_data = ImageData._from_array(img_chw)
    assert img_data.shape == (50, 50, 2), f"Expected (50, 50, 2) after transpose, got {img_data.shape}"
    
    # Test 4D format
    img_4d = np.random.randint(0, 255, (1, 50, 50, 2), dtype=np.uint8)  # (Z, H, W, C)
    img_data_4d = ImageData._from_array(img_4d)
    assert img_data_4d.shape == (50, 50, 2), f"Expected (50, 50, 2) from 4D, got {img_data_4d.shape}"
    
    print("[OK] Backward compatibility maintained")
    print(f"  (C, H, W) -> (H, W, C): {img_data.shape}")
    print(f"  4D -> (H, W, C): {img_data_4d.shape}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Phase 2: Refactored Tools - Test Suite")
    print("=" * 60)
    
    tests = [
        test_image_processor_load_single_channel,
        test_image_processor_load_multi_channel,
        test_to_segmentation_input,
        test_to_uint8,
        test_create_merged_rgb_for_display,
        test_extract_channel_for_segmentation,
        test_save_multi_channel_crop,
        test_image_data_from_path,
        test_channel_access_unified,
        test_backward_compatibility_format,
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
        print("\n[SUCCESS] All tests passed! Phase 2 refactoring is working correctly.")
    else:
        print(f"\n[ERROR] {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

