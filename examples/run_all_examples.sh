#!/bin/bash
# Run all Artifex framework examples
# This script automatically discovers and runs all Python examples in the examples directory

set -e  # Exit on first error

echo "=========================================="
echo "Running All Artifex Framework Examples"
echo "=========================================="
echo ""

# Create output directory
mkdir -p examples_output

# Counter for tracking
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

# Arrays to store results
declare -a FAILED_EXAMPLES
declare -a SUCCESS_EXAMPLES
declare -a SKIPPED_EXAMPLES

# Function to run an example
run_example() {
    local path=$1
    local relative_path=${path#examples/}  # Remove 'examples/' prefix for display

    TOTAL=$((TOTAL + 1))
    echo "[$TOTAL] Running: $relative_path"

    # Check if file should be skipped
    if grep -q "SKIP_IN_RUN_ALL" "$path" 2>/dev/null; then
        echo "    ⏭️  SKIPPED (marked with SKIP_IN_RUN_ALL)"
        SKIPPED=$((SKIPPED + 1))
        SKIPPED_EXAMPLES+=("$relative_path")
        echo ""
        return
    fi

    # Check for known problematic examples that should be skipped
    case "$relative_path" in
        *"loss_examples.py")
            echo "    ⏭️  SKIPPED (known issues)"
            SKIPPED=$((SKIPPED + 1))
            SKIPPED_EXAMPLES+=("$relative_path")
            echo ""
            return
            ;;
        *"multi_beta_vae_benchmark_demo.py")
            echo "    ⏭️  SKIPPED (requires dataset download)"
            SKIPPED=$((SKIPPED + 1))
            SKIPPED_EXAMPLES+=("$relative_path")
            echo ""
            return
            ;;
    esac

    # Run the example with timeout
    if timeout 60 python "$path" > /tmp/example_output.log 2>&1; then
        echo "    ✅ SUCCESS"
        SUCCESS=$((SUCCESS + 1))
        SUCCESS_EXAMPLES+=("$relative_path")
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "    ⏱️  TIMEOUT (exceeded 60 seconds)"
            FAILED=$((FAILED + 1))
            FAILED_EXAMPLES+=("$relative_path (timeout)")
        else
            echo "    ❌ FAILED"
            FAILED=$((FAILED + 1))
            FAILED_EXAMPLES+=("$relative_path")
            echo "    Error output (last 20 lines):"
            tail -20 /tmp/example_output.log | sed 's/^/        /'
        fi
    fi
    echo ""
}

# Function to display section header
show_section() {
    local section=$1
    echo ""
    echo "=== $section ==="
    echo ""
}

echo "Discovering Python examples in examples/ directory..."
echo ""

# Find all Python files in examples directory, excluding __pycache__ and test files
mapfile -t EXAMPLE_FILES < <(find examples -name "*.py" -type f \
    ! -path "*/\__pycache__/*" \
    ! -path "*/.pytest_cache/*" \
    ! -name "*test*.py" \
    ! -name "*Test*.py" \
    ! -name "__init__.py" \
    | sort)

echo "Found ${#EXAMPLE_FILES[@]} Python examples"
echo ""

if [ ${#EXAMPLE_FILES[@]} -eq 0 ]; then
    echo "❌ No Python examples found in examples/ directory"
    exit 1
fi

echo "Starting example tests..."
echo "=========================================="

# Group examples by directory for better organization
CURRENT_DIR=""

for example in "${EXAMPLE_FILES[@]}"; do
    # Extract directory path
    DIR=$(dirname "$example")
    DIR=${DIR#examples/}  # Remove 'examples/' prefix

    # Show section header when entering new directory
    if [ "$DIR" != "$CURRENT_DIR" ]; then
        show_section "$DIR"
        CURRENT_DIR="$DIR"
    fi

    run_example "$example"
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "📊 Statistics:"
echo "  Total examples found: $TOTAL"
echo "  ✅ Successful: $SUCCESS"
echo "  ❌ Failed: $FAILED"
echo "  ⏭️  Skipped: $SKIPPED"
echo ""

# Show successful examples if any
if [ ${#SUCCESS_EXAMPLES[@]} -gt 0 ]; then
    echo "✅ Successful examples:"
    for example in "${SUCCESS_EXAMPLES[@]}"; do
        echo "    • $example"
    done
    echo ""
fi

# Show skipped examples if any
if [ ${#SKIPPED_EXAMPLES[@]} -gt 0 ]; then
    echo "⏭️  Skipped examples:"
    for example in "${SKIPPED_EXAMPLES[@]}"; do
        echo "    • $example"
    done
    echo ""
fi

# Show failed examples if any
if [ ${#FAILED_EXAMPLES[@]} -gt 0 ]; then
    echo "❌ Failed examples:"
    for example in "${FAILED_EXAMPLES[@]}"; do
        echo "    • $example"
    done
    echo ""
fi

# Final status
if [ $FAILED -eq 0 ]; then
    if [ $SKIPPED -eq 0 ]; then
        echo "🎉 All examples completed successfully!"
    else
        echo "🎉 All non-skipped examples completed successfully!"
    fi
    exit 0
else
    echo "⚠️  Some examples failed. Please check the output above."
    echo ""
    echo "💡 Tips for debugging:"
    echo "  • Run individual failed examples directly to see full output"
    echo "  • Check if required dependencies are installed"
    echo "  • Ensure GPU/CPU configuration matches example requirements"
    exit 1
fi
