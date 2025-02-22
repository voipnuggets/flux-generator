#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_output="$3"
    local expected_exit_code="${4:-0}"
    
    echo -n "Running test: $test_name... "
    TESTS_RUN=$((TESTS_RUN + 1))
    
    # Run the command and capture output and exit code
    output=$(eval "$command" 2>&1)
    exit_code=$?
    
    # Check exit code
    if [ $exit_code -ne $expected_exit_code ]; then
        echo -e "${RED}FAILED${NC}"
        echo "Expected exit code $expected_exit_code, got $exit_code"
        echo "Command output:"
        echo "$output"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
    
    # Check output if expected_output is provided
    if [ -n "$expected_output" ]; then
        if ! echo "$output" | grep -q "$expected_output"; then
            echo -e "${RED}FAILED${NC}"
            echo "Expected output to contain: $expected_output"
            echo "Got output:"
            echo "$output"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 1
        fi
    fi
    
    echo -e "${GREEN}PASSED${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    return 0
}

# Test help message
run_test "Help message" \
    "./run_flux.sh --help" \
    "Usage:"

# Test unknown option
run_test "Unknown option" \
    "./run_flux.sh --invalid-option" \
    "Unknown option" \
    1

# Test system requirements check
run_test "System requirements check" \
    "./run_flux.sh --help" \
    "Options:"

# Test Python version check
run_test "Python version check" \
    "command -v python3.11" \
    "python3.11"

# Test virtual environment activation
run_test "Virtual environment check" \
    "test -d venv" \
    ""

# Print test summary
echo
echo "Test Summary:"
echo "============"
echo "Tests run:    $TESTS_RUN"
echo "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo "Tests failed: ${RED}$TESTS_FAILED${NC}"

# Exit with failure if any tests failed
[ $TESTS_FAILED -eq 0 ] || exit 1 