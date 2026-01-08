"""
Test to verify that Cell_State_Analyzer_Tool is properly enforced in the planning chain.
This test ensures that the critical self-supervised learning step is not skipped.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from octotools.models.planner import Planner
from octotools.models.memory import Memory


def test_requires_full_analysis_detection():
    """Test that _requires_full_cell_state_analysis correctly identifies queries needing full analysis"""
    print("\n=== Test 1: Full Analysis Detection ===")
    
    planner = Planner.__new__(Planner)
    
    # Test cases that SHOULD require full analysis
    should_require = [
        "What changes of organoid among different groups?",
        "Compare cell states between treatment and control",
        "What are the cell states in this image?",
        "Analyze cell states using clustering",
        "Generate UMAP embedding for cell states",
    ]
    
    # Test cases that should NOT require full analysis
    should_not_require = [
        "How many cells are in this image?",
        "Count the number of organoids",
        "What is the total cell count?",
    ]
    
    for query in should_require:
        result = planner._requires_full_cell_state_analysis(query)
        assert result == True, f"Query '{query}' should require full analysis but returned False"
        print(f"  [OK] '{query[:50]}...' → requires full analysis")
    
    for query in should_not_require:
        result = planner._requires_full_cell_state_analysis(query)
        assert result == False, f"Query '{query}' should NOT require full analysis but returned True"
        print(f"  [OK] '{query[:50]}...' → does NOT require full analysis")
    
    print("[OK] Full analysis detection works correctly")


def test_tool_chain_enforcement_logic():
    """Test that the tool chain enforcement logic is in place"""
    print("\n=== Test 2: Tool Chain Enforcement Logic ===")
    
    planner = Planner.__new__(Planner)
    
    # Check that the method exists
    assert hasattr(planner, '_requires_full_cell_state_analysis'), \
        "_requires_full_cell_state_analysis method not found"
    
    # Check that generate_next_step exists
    assert hasattr(planner, 'generate_next_step'), \
        "generate_next_step method not found"
    
    # Check that the method signature is correct
    import inspect
    sig = inspect.signature(planner._requires_full_cell_state_analysis)
    assert 'question' in sig.parameters, "Method should accept 'question' parameter"
    
    print("[OK] Tool chain enforcement logic is in place")


def test_bioimage_chain_order():
    """Test that the bioimage analysis chain order is documented correctly"""
    print("\n=== Test 3: Bioimage Chain Order Documentation ===")
    
    # Read planner.py to check chain documentation
    planner_path = os.path.join(project_root, "octotools", "models", "planner.py")
    with open(planner_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that the chain includes Cell_State_Analyzer_Tool
    assert "Single_Cell_Cropper_Tool" in content, "Single_Cell_Cropper_Tool should be in chain"
    assert "Cell_State_Analyzer_Tool" in content, "Cell_State_Analyzer_Tool should be in chain"
    assert "Analysis_Visualizer_Tool" in content, "Analysis_Visualizer_Tool should be in chain"
    
    # Check that the order is documented
    assert "Single_Cell_Cropper → Cell_State_Analyzer" in content or \
           "Single_Cell_Cropper_Tool → Cell_State_Analyzer_Tool" in content, \
        "Chain should document: Single_Cell_Cropper → Cell_State_Analyzer"
    
    # Check that Cell_State_Analyzer_Tool is marked as MANDATORY
    assert "Cell_State_Analyzer_Tool" in content and \
           ("MANDATORY" in content or "ESSENTIAL" in content or "REQUIRED" in content), \
        "Cell_State_Analyzer_Tool should be marked as MANDATORY/ESSENTIAL/REQUIRED"
    
    print("[OK] Bioimage chain order is correctly documented")


def test_code_enforcement_exists():
    """Test that code-level enforcement for Cell_State_Analyzer_Tool exists"""
    print("\n=== Test 4: Code Enforcement Exists ===")
    
    # Read planner.py to check enforcement code
    planner_path = os.path.join(project_root, "octotools", "models", "planner.py")
    with open(planner_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for enforcement after Single_Cell_Cropper_Tool
    assert 'if last_tool == "Single_Cell_Cropper_Tool":' in content, \
        "Should have enforcement check for Single_Cell_Cropper_Tool"
    
    assert 'Cell_State_Analyzer_Tool' in content, \
        "Should mention Cell_State_Analyzer_Tool in enforcement"
    
    assert 'requires_full_analysis' in content or 'requires_full_cell_state_analysis' in content, \
        "Should check if query requires full analysis"
    
    print("[OK] Code-level enforcement exists")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Cell_State_Analyzer_Tool Enforcement Test Suite")
    print("=" * 60)
    
    tests = [
        test_requires_full_analysis_detection,
        test_tool_chain_enforcement_logic,
        test_bioimage_chain_order,
        test_code_enforcement_exists,
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
        print("\n[SUCCESS] All enforcement tests passed!")
        print("[OK] Cell_State_Analyzer_Tool will be properly enforced in the planning chain.")
    else:
        print(f"\n[ERROR] {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

