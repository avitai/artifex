#!/usr/bin/env bash
# Run the reviewed root examples smoke subset.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

readonly TIMEOUT_SECONDS="${ARTIFEX_EXAMPLE_TIMEOUT_SECONDS:-120}"
readonly LOG_FILE="/tmp/artifex_example_output.log"
readonly -a SUPPORTED_EXAMPLES=(
  "examples/generative_models/framework_features_demo.py"
  "examples/generative_models/image/vae/vae_mnist.py"
  "examples/generative_models/geometric/simple_point_cloud_example.py"
  "examples/generative_models/energy/simple_ebm_example.py"
)

declare -a FAILED_EXAMPLES=()
declare -a PASSED_EXAMPLES=()

run_with_timeout() {
  local path=$1

  if command -v timeout >/dev/null 2>&1; then
    timeout "$TIMEOUT_SECONDS" uv run python "$path"
    return
  fi

  uv run python "$path"
}

run_example() {
  local path=$1
  local index=$2

  echo "[$index/${#SUPPORTED_EXAMPLES[@]}] Running: $path"

  if run_with_timeout "$path" >"$LOG_FILE" 2>&1; then
    echo "    PASS"
    PASSED_EXAMPLES+=("$path")
    return
  fi

  local exit_code=$?
  FAILED_EXAMPLES+=("$path")
  if [[ $exit_code -eq 124 ]]; then
    echo "    FAIL (timeout after ${TIMEOUT_SECONDS}s)"
  else
    echo "    FAIL"
  fi
  tail -20 "$LOG_FILE" | sed 's/^/        /'
}

echo "=========================================="
echo "Running reviewed Artifex example smoke subset"
echo "=========================================="
echo "Repository root: $REPO_ROOT"
echo "Examples covered: ${#SUPPORTED_EXAMPLES[@]}"
echo ""

for index in "${!SUPPORTED_EXAMPLES[@]}"; do
  run_example "${SUPPORTED_EXAMPLES[$index]}" "$((index + 1))"
  echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Passed: ${#PASSED_EXAMPLES[@]}"
echo "Failed: ${#FAILED_EXAMPLES[@]}"

if [[ ${#PASSED_EXAMPLES[@]} -gt 0 ]]; then
  echo ""
  echo "Passed examples:"
  for path in "${PASSED_EXAMPLES[@]}"; do
    echo "  - $path"
  done
fi

if [[ ${#FAILED_EXAMPLES[@]} -gt 0 ]]; then
  echo ""
  echo "Failed examples:"
  for path in "${FAILED_EXAMPLES[@]}"; do
    echo "  - $path"
  done
  exit 1
fi

echo ""
echo "All curated smoke examples passed."
