#!/usr/bin/env python
"""Helper script to run BlackJAX tests.

This script enables and runs all BlackJAX tests by setting the appropriate
environment variable before invoking pytest.
"""

import os
import subprocess
import sys


def main():
    """Run BlackJAX tests with appropriate environment variables set."""
    print("Running BlackJAX tests (including tests that are normally skipped)...")

    # Set environment variable to enable BlackJAX tests
    env = os.environ.copy()
    env["ENABLE_BLACKJAX_TESTS"] = "1"

    # Build command with any args passed to this script
    cmd = [sys.executable, "-m", "pytest"]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    else:
        # Default to running just the BlackJAX tests
        test_path = "tests/artifex/generative_models/core/sampling/"
        cmd.append(f"{test_path}test_blackjax_samplers.py")

    cmd.extend(["-v"])  # Add verbosity

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
