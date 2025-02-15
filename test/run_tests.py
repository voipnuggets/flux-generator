#!/usr/bin/env python3

import pytest
import sys
from pathlib import Path

def run_tests():
    """Run all tests with pytest and generate coverage report."""
    test_path = Path(__file__).parent
    args = [
        str(test_path),
        "-v",
        "--cov=flux_app",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_report"
    ]
    return pytest.main(args)

if __name__ == "__main__":
    sys.exit(run_tests()) 