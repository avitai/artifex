#!/bin/bash
# Smart Test Runner for Artifex
# ================================
#
# PURPOSE:
#   Intelligently runs pytest with automatic GPU detection and parallel execution
#   optimization. Determines the best execution strategy based on test type and
#   available hardware.
#
# USAGE:
#   ./scripts/smart_test_runner.sh [pytest-arguments]
#
# EXAMPLES:
#   ./scripts/smart_test_runner.sh tests/                    # All tests
#   ./scripts/smart_test_runner.sh tests/ -k test_gpu        # GPU tests only
#   ./scripts/smart_test_runner.sh tests/ -m "not gpu"      # CPU tests only
#   ./scripts/smart_test_runner.sh tests/ --parallel         # Force parallel
#   ./scripts/smart_test_runner.sh tests/ --sequential       # Force sequential
#
# FEATURES:
#   - Automatic GPU detection using nvidia-smi
#   - Smart parallelization: CPU tests run in parallel, GPU tests run sequentially
#   - Detects GPU test markers in test files
#   - Environment variable loading from .env
#   - Colored output for better readability
#
# EXECUTION STRATEGIES:
#   - GPU tests on GPU hardware: Sequential (prevents CUDA conflicts)
#   - GPU tests on CPU hardware: Skip GPU-marked tests with warning
#   - CPU-only tests: Parallel execution using pytest-xdist
#   - Mixed tests: Automatic detection and appropriate strategy
#
# EXIT CODES:
#   0 - Tests passed
#   1 - Tests failed or error in execution
#
# DEPENDENCIES:
#   - pytest
#   - pytest-xdist (for parallel execution)
#   - nvidia-smi (for GPU detection, optional)
#
# ENVIRONMENT VARIABLES:
#   Loads from .env file if present
#
# Author: Artifex Team
# License: MIT

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Artifex Smart Test Runner${NC}"
echo "=================================="

# Function to detect GPU availability
detect_gpu() {
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        return 0  # GPU available
    else
        return 1  # No GPU
    fi
}

# Function to check if tests contain GPU markers
has_gpu_tests() {
    local test_path="$1"
    if grep -r "@pytest.mark.gpu\|@pytest.mark.cuda" "$test_path" > /dev/null 2>&1; then
        return 0  # Has GPU tests
    else
        return 1  # No GPU tests
    fi
}

# Function to run tests with appropriate parallelization
run_tests() {
    local test_args="$*"
    local gpu_available=false
    local has_gpu_markers=false

    # Detect environment
    if detect_gpu; then
        gpu_available=true
        echo -e "${GREEN}✅ GPU detected${NC}"
    else
        echo -e "${YELLOW}⚠️  No GPU detected - CPU only mode${NC}"
    fi

    # Check for GPU test markers in test arguments
    if [[ "$test_args" == *"gpu"* ]] || [[ "$test_args" == *"cuda"* ]]; then
        has_gpu_markers=true
    fi

    # Check if test paths contain GPU tests
    for arg in $test_args; do
        if [[ -d "$arg" ]] || [[ -f "$arg" ]]; then
            if has_gpu_tests "$arg"; then
                has_gpu_markers=true
                break
            fi
        fi
    done

    # Determine execution strategy
    if [[ "$has_gpu_markers" == true ]] && [[ "$gpu_available" == true ]]; then
        echo -e "${BLUE}🎮 Running GPU tests (sequential to avoid conflicts)${NC}"
        echo "   Strategy: Sequential execution for GPU safety"
        source .env 2>/dev/null || true
        uv run pytest "$test_args" --tb=short -v
    elif [[ "$has_gpu_markers" == true ]] && [[ "$gpu_available" == false ]]; then
        echo -e "${YELLOW}⚠️  GPU tests requested but no GPU available${NC}"
        echo "   Strategy: Running GPU tests on CPU (may fail)"
        uv run pytest "$test_args" --tb=short -v -m "not gpu"
    else
        echo -e "${GREEN}🚀 Running CPU tests (parallel execution)${NC}"
        echo "   Strategy: Parallel execution for speed"
        uv run pytest "$test_args" --tb=short -v -n auto
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [pytest-arguments]"
    echo ""
    echo "Examples:"
    echo "  $0 tests/                              # All tests"
    echo "  $0 tests/ -k test_gpu                  # GPU tests only"
    echo "  $0 tests/ -m \"not gpu\"                # CPU tests only"
    echo "  $0 tests/ --parallel                   # Force parallel (dangerous for GPU)"
    echo "  $0 tests/ --sequential                 # Force sequential"
    echo ""
    echo "Test Categories:"
    echo "  • GPU tests: Run sequentially to avoid conflicts"
    echo "  • CPU tests: Run in parallel for speed"
    echo "  • Mixed: Automatically determined"
}

# Parse special arguments
FORCE_PARALLEL=false
FORCE_SEQUENTIAL=false
ARGS=()

for arg in "$@"; do
    case $arg in
        --parallel)
            FORCE_PARALLEL=true
            ;;
        --sequential)
            FORCE_SEQUENTIAL=true
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

# Handle forced execution modes
if [[ "$FORCE_PARALLEL" == true ]]; then
    echo -e "${YELLOW}⚠️  Forced parallel execution (may cause GPU conflicts)${NC}"
    uv run pytest "${ARGS[@]}" --tb=short -v -n auto
elif [[ "$FORCE_SEQUENTIAL" == true ]]; then
    echo -e "${BLUE}🔄 Forced sequential execution${NC}"
    source .env 2>/dev/null || true
    uv run pytest "${ARGS[@]}" --tb=short -v
else
    # Smart execution
    if [[ ${#ARGS[@]} -eq 0 ]]; then
        # Default: run all tests
        run_tests "tests/"
    else
        run_tests "${ARGS[@]}"
    fi
fi

echo ""
echo -e "${GREEN}✅ Test execution completed${NC}"
