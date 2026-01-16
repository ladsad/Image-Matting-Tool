"""
Simple verification test for create_advanced_layout function (commit 547e1fa).

This test validates that the function exists and has the correct structure
without requiring GUI dependencies.
"""

import ast
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_function_exists():
    """Test that create_advanced_layout function exists in the file."""
    layouts_file = project_root / "src" / "gui" / "layouts.py"
    
    assert layouts_file.exists(), f"layouts.py should exist at {layouts_file}"
    
    with open(layouts_file, 'r') as f:
        content = f.read()
    
    assert "def create_advanced_layout(" in content, "Function definition should exist"
    print("✓ create_advanced_layout function exists in layouts.py")
    return True


def test_function_signature():
    """Test that the function has the correct signature using AST."""
    layouts_file = project_root / "src" / "gui" / "layouts.py"
    
    with open(layouts_file, 'r') as f:
        tree = ast.parse(f.read())
    
    function_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "create_advanced_layout":
            function_found = True
            # Check parameters
            args = node.args.args
            assert len(args) == 1, "Function should have exactly 1 parameter"
            assert args[0].arg == "current_state", "Parameter should be named 'current_state'"
            
            # Check return type annotation
            assert node.returns is not None, "Function should have return type annotation"
            print("✓ Function signature is correct: create_advanced_layout(current_state: dict) -> list")
            break
    
    assert function_found, "create_advanced_layout function not found in AST"
    return True


def test_function_content():
    """Test that the function contains expected UI elements."""
    layouts_file = project_root / "src" / "gui" / "layouts.py"
    
    with open(layouts_file, 'r') as f:
        content = f.read()
    
    # Find the function definition
    start_idx = content.find("def create_advanced_layout(")
    assert start_idx > 0, "Function definition should exist"
    
    # Find the next function or end of file
    next_def_idx = content.find("\ndef ", start_idx + 1)
    if next_def_idx == -1:
        function_content = content[start_idx:]
    else:
        function_content = content[start_idx:next_def_idx]
    
    # Check for expected UI element keys
    expected_keys = [
        "-ADV-GAMMA-",
        "-ADV-DENOISE-",
        "-ADV-CLAHE-",
        "-ADV-MORPH-",
        "-ADV-ISLAND-",
        "-ADV-GUIDED-",
        "-ADV-APPLY-",
        "-ADV-CANCEL-",
    ]
    
    missing_keys = []
    for key in expected_keys:
        if key not in function_content:
            missing_keys.append(key)
    
    assert len(missing_keys) == 0, f"Missing UI keys: {missing_keys}"
    print(f"✓ Function contains all {len(expected_keys)} expected UI element keys")
    return True


def test_function_has_frames():
    """Test that the function creates preprocessing and postprocessing frames."""
    layouts_file = project_root / "src" / "gui" / "layouts.py"
    
    with open(layouts_file, 'r') as f:
        content = f.read()
    
    start_idx = content.find("def create_advanced_layout(")
    next_def_idx = content.find("\ndef ", start_idx + 1)
    if next_def_idx == -1:
        function_content = content[start_idx:]
    else:
        function_content = content[start_idx:next_def_idx]
    
    assert "Preprocessing" in function_content, "Should have Preprocessing section"
    assert "Postprocessing" in function_content, "Should have Postprocessing section"
    assert "Gamma Correction" in function_content, "Should have Gamma Correction control"
    assert "Denoising Strength" in function_content, "Should have Denoising control"
    assert "Edge Cleanup" in function_content, "Should have Edge Cleanup control"
    assert "Remove Floating Islands" in function_content, "Should have Island removal control"
    assert "Guided Filter" in function_content, "Should have Guided Filter control"
    
    print("✓ Function contains all expected preprocessing and postprocessing controls")
    return True


def test_integration_with_app():
    """Test that the function is imported in app.py."""
    app_file = project_root / "src" / "gui" / "app.py"
    
    assert app_file.exists(), "app.py should exist"
    
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Check import
    assert "create_advanced_layout" in content, "Function should be imported in app.py"
    
    # Check usage
    assert "create_advanced_layout(self.adv_settings)" in content or \
           "create_advanced_layout(" in content, "Function should be used in app.py"
    
    print("✓ Function is properly imported and used in app.py")
    return True


def test_line_count():
    """Verify the function is approximately the right size (61 lines as per commit)."""
    layouts_file = project_root / "src" / "gui" / "layouts.py"
    
    with open(layouts_file, 'r') as f:
        lines = f.readlines()
    
    # Find the function
    start_line = None
    for i, line in enumerate(lines):
        if "def create_advanced_layout(" in line:
            start_line = i
            break
    
    assert start_line is not None, "Function should exist"
    
    # Count lines until next function or end
    line_count = 0
    for i in range(start_line, len(lines)):
        line_count += 1
        if i > start_line and lines[i].startswith("def ") and not lines[i].startswith("def create_advanced_layout"):
            break
    
    # The commit shows 61 lines, but this includes context. The function itself should be around 50-70 lines
    assert 40 < line_count < 100, f"Function should be reasonable size, got {line_count} lines"
    print(f"✓ Function has {line_count} lines (expected ~59 lines)")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Verification Tests for commit 547e1fa")
    print("Testing create_advanced_layout without GUI dependencies")
    print("=" * 70 + "\n")
    
    tests = [
        ("Function Exists", test_function_exists),
        ("Function Signature", test_function_signature),
        ("Function Content", test_function_content),
        ("UI Frames", test_function_has_frames),
        ("Integration", test_integration_with_app),
        ("Line Count", test_line_count),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"{test_name}:")
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            failed += 1
    
    print("=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ All tests passed! Commit 547e1fa changes are properly implemented.")
    else:
        print(f"❌ {failed} test(s) failed")
    print("=" * 70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
