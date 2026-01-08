"""
Integration Test: Verify refactored tools can be imported and initialized
Tests that tools can be instantiated and basic methods work
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_tool_imports():
    """Test that all refactored tools can be imported"""
    print("\n=== Test 1: Tool Imports ===")
    
    try:
        from octotools.tools.image_preprocessor.tool import Image_Preprocessor_Tool
        print("[OK] Image_Preprocessor_Tool imported")
    except Exception as e:
        print(f"[FAIL] Image_Preprocessor_Tool import failed: {e}")
        return False
    
    try:
        from octotools.tools.cell_segmenter.tool import Cell_Segmenter_Tool
        print("[OK] Cell_Segmenter_Tool imported")
    except Exception as e:
        print(f"[FAIL] Cell_Segmenter_Tool import failed: {e}")
        return False
    
    try:
        from octotools.tools.nuclei_segmenter.tool import Nuclei_Segmenter_Tool
        print("[OK] Nuclei_Segmenter_Tool imported")
    except Exception as e:
        print(f"[FAIL] Nuclei_Segmenter_Tool import failed: {e}")
        return False
    
    try:
        from octotools.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool
        print("[OK] Organoid_Segmenter_Tool imported")
    except Exception as e:
        print(f"[FAIL] Organoid_Segmenter_Tool import failed: {e}")
        return False
    
    try:
        from octotools.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool
        print("[OK] Single_Cell_Cropper_Tool imported")
    except Exception as e:
        print(f"[FAIL] Single_Cell_Cropper_Tool import failed: {e}")
        return False
    
    try:
        from octotools.tools.analysis_visualizer.tool import Analysis_Visualizer_Tool
        print("[OK] Analysis_Visualizer_Tool imported")
    except Exception as e:
        print(f"[FAIL] Analysis_Visualizer_Tool import failed: {e}")
        return False
    
    return True


def test_unified_abstraction_imports():
    """Test that unified abstraction classes can be imported"""
    print("\n=== Test 2: Unified Abstraction Imports ===")
    
    try:
        from octotools.models.image_data import ImageData
        print("[OK] ImageData imported")
    except Exception as e:
        print(f"[FAIL] ImageData import failed: {e}")
        return False
    
    try:
        from octotools.utils.image_processor import ImageProcessor
        print("[OK] ImageProcessor imported")
    except Exception as e:
        print(f"[FAIL] ImageProcessor import failed: {e}")
        return False
    
    return True


def test_tool_initialization():
    """Test that tools can be initialized (without GPU/model download)"""
    print("\n=== Test 3: Tool Initialization ===")
    
    try:
        from octotools.tools.image_preprocessor.tool import Image_Preprocessor_Tool
        tool = Image_Preprocessor_Tool()
        print("[OK] Image_Preprocessor_Tool initialized")
    except Exception as e:
        print(f"[FAIL] Image_Preprocessor_Tool initialization failed: {e}")
        return False
    
    try:
        from octotools.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool
        tool = Single_Cell_Cropper_Tool()
        print("[OK] Single_Cell_Cropper_Tool initialized")
    except Exception as e:
        print(f"[FAIL] Single_Cell_Cropper_Tool initialization failed: {e}")
        return False
    
    try:
        from octotools.tools.analysis_visualizer.tool import Analysis_Visualizer_Tool
        tool = Analysis_Visualizer_Tool()
        print("[OK] Analysis_Visualizer_Tool initialized")
    except Exception as e:
        print(f"[FAIL] Analysis_Visualizer_Tool initialization failed: {e}")
        return False
    
    # Note: Segmenter tools require model download, so we skip full initialization
    # but verify they can be imported
    print("[OK] Segmenter tools can be imported (initialization requires models)")
    
    return True


def test_unified_abstraction_usage():
    """Test that unified abstraction can be used in tools"""
    print("\n=== Test 4: Unified Abstraction Usage in Tools ===")
    
    try:
        from octotools.utils.image_processor import ImageProcessor
        from octotools.models.image_data import ImageData
        import numpy as np
        import tempfile
        from PIL import Image
        
        # Create test image
        with tempfile.TemporaryDirectory() as tmpdir:
            img_2d = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
            png_path = os.path.join(tmpdir, "test.png")
            Image.fromarray(img_2d, mode='L').save(png_path)
            
            # Test ImageProcessor.load_image (used by tools)
            img_data = ImageProcessor.load_image(png_path)
            assert img_data.shape == (50, 50, 1), f"Expected (50, 50, 1), got {img_data.shape}"
            
            # Test methods used by tools
            seg_input = ImageProcessor.extract_channel_for_segmentation(img_data, 0)
            assert seg_input.dtype == np.float32, f"Expected float32, got {seg_input.dtype}"
            
            uint8_channel = ImageProcessor.normalize_for_display(img_data, 0)
            assert uint8_channel.dtype == np.uint8, f"Expected uint8, got {uint8_channel.dtype}"
            
            print("[OK] Unified abstraction methods work correctly")
            print(f"  Image shape: {img_data.shape}")
            print(f"  Segmentation input dtype: {seg_input.dtype}")
            print(f"  Display channel dtype: {uint8_channel.dtype}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Unified abstraction usage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that tools maintain backward compatibility"""
    print("\n=== Test 5: Backward Compatibility ===")
    
    try:
        from octotools.tools.image_preprocessor.tool import Image_Preprocessor_Tool
        tool = Image_Preprocessor_Tool()
        
        # Check that execute method signature is compatible
        import inspect
        sig = inspect.signature(tool.execute)
        params = list(sig.parameters.keys())
        
        # Should have query_cache_dir parameter (added in refactoring)
        assert 'query_cache_dir' in params, "Missing query_cache_dir parameter"
        assert 'image' in params, "Missing image parameter"
        
        print("[OK] Tool method signatures maintain backward compatibility")
        print(f"  Execute parameters: {params}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("Phase 2: Tool Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Tool Imports", test_tool_imports),
        ("Unified Abstraction Imports", test_unified_abstraction_imports),
        ("Tool Initialization", test_tool_initialization),
        ("Unified Abstraction Usage", test_unified_abstraction_usage),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Integration Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n[SUCCESS] All integration tests passed!")
        print("Phase 2 refactoring is working correctly and tools are ready to use.")
    else:
        print(f"\n[ERROR] {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

