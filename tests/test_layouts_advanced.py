"""
Test for create_advanced_layout function added in commit 547e1fa.

This test verifies that the create_advanced_layout function is properly
defined and returns a valid layout structure.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_create_advanced_layout_exists():
    """Test that create_advanced_layout function can be imported."""
    # Direct import to avoid app dependencies
    import importlib.util
    layouts_path = project_root / "src" / "gui" / "layouts.py"
    spec = importlib.util.spec_from_file_location("layouts", layouts_path)
    layouts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layouts)
    
    assert hasattr(layouts, "create_advanced_layout"), "layouts module should have create_advanced_layout"
    assert callable(layouts.create_advanced_layout), "create_advanced_layout should be callable"
    print("✓ create_advanced_layout function exists and is callable")


def test_create_advanced_layout_signature():
    """Test that create_advanced_layout has correct signature."""
    import importlib.util
    import inspect
    
    layouts_path = project_root / "src" / "gui" / "layouts.py"
    spec = importlib.util.spec_from_file_location("layouts", layouts_path)
    layouts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layouts)
    
    sig = inspect.signature(layouts.create_advanced_layout)
    params = list(sig.parameters.keys())
    
    assert "current_state" in params, "Function should have 'current_state' parameter"
    assert len(params) == 1, "Function should have exactly one parameter"
    print("✓ Function signature is correct")


def test_create_advanced_layout_returns_list():
    """Test that create_advanced_layout returns a list (layout)."""
    import importlib.util
    
    layouts_path = project_root / "src" / "gui" / "layouts.py"
    spec = importlib.util.spec_from_file_location("layouts", layouts_path)
    layouts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layouts)
    
    test_state = {
        "gamma": 1.0,
        "denoise": 0.0,
        "clahe": False,
        "morph": 0,
        "island": False,
        "guided": False,
    }
    
    result = layouts.create_advanced_layout(test_state)
    
    assert isinstance(result, list), "Function should return a list"
    assert len(result) > 0, "Layout should not be empty"
    print(f"✓ Function returns a valid layout with {len(result)} rows")


def test_create_advanced_layout_with_various_states():
    """Test that create_advanced_layout works with different state values."""
    import importlib.util
    
    layouts_path = project_root / "src" / "gui" / "layouts.py"
    spec = importlib.util.spec_from_file_location("layouts", layouts_path)
    layouts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layouts)
    
    test_cases = [
        # Default state
        {
            "gamma": 1.0,
            "denoise": 0.0,
            "clahe": False,
            "morph": 0,
            "island": False,
            "guided": False,
        },
        # Modified state
        {
            "gamma": 1.5,
            "denoise": 3.0,
            "clahe": True,
            "morph": 2,
            "island": True,
            "guided": True,
        },
        # Edge case values
        {
            "gamma": 0.5,
            "denoise": 10.0,
            "clahe": False,
            "morph": 5,
            "island": False,
            "guided": False,
        },
    ]
    
    for i, test_state in enumerate(test_cases, 1):
        result = layouts.create_advanced_layout(test_state)
        assert isinstance(result, list), f"Test case {i} should return a list"
        assert len(result) > 0, f"Test case {i} layout should not be empty"
    
    print(f"✓ Function works correctly with {len(test_cases)} different states")


def test_create_advanced_layout_has_required_keys():
    """Test that the layout contains expected UI element keys."""
    import importlib.util
    
    layouts_path = project_root / "src" / "gui" / "layouts.py"
    spec = importlib.util.spec_from_file_location("layouts", layouts_path)
    layouts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layouts)
    
    test_state = {
        "gamma": 1.0,
        "denoise": 0.0,
        "clahe": False,
        "morph": 0,
        "island": False,
        "guided": False,
    }
    
    layout = layouts.create_advanced_layout(test_state)
    
    # Convert layout to string to check for keys
    layout_str = str(layout)
    
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
    
    for key in expected_keys:
        assert key in layout_str, f"Layout should contain key: {key}"
    
    print(f"✓ Layout contains all {len(expected_keys)} expected UI element keys")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Testing create_advanced_layout function (commit 547e1fa)")
    print("=" * 70 + "\n")
    
    tests = [
        ("Import Test", test_create_advanced_layout_exists),
        ("Signature Test", test_create_advanced_layout_signature),
        ("Return Type Test", test_create_advanced_layout_returns_list),
        ("State Variations Test", test_create_advanced_layout_with_various_states),
        ("UI Elements Test", test_create_advanced_layout_has_required_keys),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
