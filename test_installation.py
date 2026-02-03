"""
Test script to verify SHAPE installation.

Run this script after installation to verify that all dependencies
and components are working correctly.
"""

import sys
import os


def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from shape import solve, construct_solver, get_available_tools
        print("[OK] shape module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import shape: {e}")
        return False
    
    try:
        from shape.agent.planner import Planner
        from shape.agent.executor import Executor
        from shape.agent.memory import Memory
        print("[OK] shape.agent modules imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import shape.agent modules: {e}")
        return False
    
    try:
        from shape.models.initializer import Initializer
        from shape.models.planner import Planner as ShapePlanner
        from shape.models.executor import Executor as ShapeExecutor
        print("[OK] shape modules imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import shape modules: {e}")
        return False
    
    return True


def test_tool_discovery():
    """Test that tools can be discovered."""
    print("\nTesting tool discovery...")
    
    try:
        from shape import get_available_tools
        tools = get_available_tools()
        print(f"[OK] Discovered {len(tools)} tools")
        if len(tools) > 0:
            print(f"  Sample tools: {tools[:5]}")
        return True
    except Exception as e:
        print(f"[FAIL] Tool discovery failed: {e}")
        return False


def test_solver_construction():
    """Test that solver can be constructed (without API key)."""
    print("\nTesting solver construction...")
    
    try:
        from shape import construct_solver
        
        # This should fail gracefully without API key
        try:
            solver = construct_solver(
                llm_engine_name="gpt-4o",
                api_key="test-key"
            )
            print("[OK] Solver construction successful")
            print(f"  Available tools: {len(solver['available_tools'])}")
            return True
        except ValueError as e:
            if "API key" in str(e):
                print("[OK] Solver construction works (API key validation working)")
                return True
            else:
                print(f"[FAIL] Unexpected error: {e}")
                return False
    except Exception as e:
        print(f"[FAIL] Solver construction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """Test that key dependencies are available."""
    print("\nTesting dependencies...")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("PIL", "Pillow"),
        ("openai", "OpenAI"),
        ("skimage", "scikit-image"),
        ("cv2", "OpenCV"),
        ("scanpy", "Scanpy"),
    ]
    
    all_ok = True
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"[OK] {display_name} available")
        except ImportError:
            print(f"[FAIL] {display_name} not found")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("="*60)
    print("SHAPE Installation Verification")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("Tool Discovery", test_tool_discovery()))
    results.append(("Solver Construction", test_solver_construction()))
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n[OK] All tests passed! SHAPE is ready to use.")
        print("\nNote: To use SHAPE, set your OPENAI_API_KEY environment variable:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

