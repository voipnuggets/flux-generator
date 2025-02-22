#!/usr/bin/env python3

import pytest
import sys
import os
import subprocess
from pathlib import Path
import importlib.util

def check_test_file(file_path):
    """Check if a test file can be imported without errors."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except ImportError as e:
        print(f"Skipping {file_path.name}: {str(e)}")
        return False
    except Exception as e:
        print(f"Error checking {file_path.name}: {str(e)}")
        return False

def run_shell_tests():
    """Run the shell script tests."""
    test_script = Path(__file__).parent / "test_run_script.sh"
    if not test_script.exists():
        print("Error: test_run_script.sh not found")
        return False
    
    # Make the script executable
    os.chmod(test_script, 0o755)
    
    # Run the tests
    print("\nRunning shell script tests...")
    result = subprocess.run([str(test_script)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def run_python_tests():
    """Run the Python tests with pytest and generate coverage report."""
    test_path = Path(__file__).parent
    test_files = []
    
    # Include all test files
    all_tests = [
        "test_cli.py",           # CLI functionality tests
        "test_connectivity.py",   # Network connectivity tests
        "test_ui.py",            # Basic UI structure tests
        "test_api.py",           # API endpoint tests
        "test_generation.py",     # Image generation tests
        "test_run_script.sh"     # Shell script tests
    ]
    
    print("\nChecking test files:")
    for test_file in all_tests:
        file_path = test_path / test_file
        if file_path.exists():
            if test_file.endswith('.sh'):
                test_files.append(str(file_path))
                print(f"  ✓ {test_file}")
            elif check_test_file(file_path):
                test_files.append(str(file_path))
                print(f"  ✓ {test_file}")
            else:
                print(f"  ✗ {test_file} (import error)")
        else:
            print(f"  ✗ {test_file} (not found)")
    
    if not test_files:
        print("\nError: No test files found to run")
        return 1
    
    args = [
        *[f for f in test_files if f.endswith('.py')],  # Only include Python files for pytest
        "-v",
        "--cov=flux_app",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_report"
    ]
    
    print(f"\nRunning {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {os.path.basename(test_file)}")
    print()
    
    return pytest.main(args)

if __name__ == "__main__":
    # Run shell tests first
    shell_tests_passed = run_shell_tests()
    
    # Run Python tests
    python_tests_result = run_python_tests()
    
    # Exit with failure if either test suite failed
    sys.exit(0 if shell_tests_passed and python_tests_result == 0 else 1) 