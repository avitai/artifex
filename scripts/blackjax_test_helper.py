#!/usr/bin/env python3
"""Helper script to enable or disable BlackJAX tests.

This script provides a simple way to enable or disable BlackJAX tests
by setting the appropriate environment variables.
"""

import argparse
import os
import subprocess
import sys


def set_env_var(name, value):
    """Set an environment variable for the current session."""
    if os.name == "nt":  # Windows
        os.system(f"set {name}={value}")
        print(f"Set {name}={value} for this session only")
        print("To make permanent, use System Properties > Environment Variables")
    else:  # Unix-like
        shell = os.environ.get("SHELL", "/bin/bash").split("/")[-1]
        if shell in ["bash", "zsh", "sh"]:
            rc_file = f"~/.{shell}rc"
            print(f"To make permanent, add the following to {rc_file}:")
            print(f"export {name}={value}")

        os.environ[name] = value
        print(f"Set {name}={value} for this session only")


def show_status():
    """Show the current status of BlackJAX tests."""
    enable_var = os.environ.get("ENABLE_BLACKJAX_TESTS", "")
    skip_var = os.environ.get("SKIP_BLACKJAX_TESTS", "1")

    if enable_var != "":
        status = "ENABLED"
        active_var = f"ENABLE_BLACKJAX_TESTS={enable_var}"
    elif skip_var == "":
        status = "ENABLED"
        active_var = "SKIP_BLACKJAX_TESTS=''"
    else:
        status = "DISABLED"
        active_var = f"SKIP_BLACKJAX_TESTS={skip_var}"

    print(f"BlackJAX tests are currently: {status}")
    print(f"Active environment variable: {active_var}")


def run_tests(args):
    """Run the tests with the current settings."""
    command = ["./scripts/run_tests.sh"]

    if args.all:
        command.append("--all")
    elif args.only_blackjax:
        command.append("--only-blackjax")
    else:
        command.append("--fast")

    if args.verbose:
        command.append("-v")

    # Handle parallel execution (default is now parallel in run_tests.sh)
    if args.no_parallel:
        command.append("--no-parallel")
    elif args.jobs:
        command.append("-j")
        command.append(args.jobs)

    if args.coverage:
        command.append("-c")

    if args.no_progress:
        command.append("--no-progress")

    print(f"Running: {' '.join(command)}")
    subprocess.run(command)


def main():
    """Parse arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description="Enable or disable BlackJAX tests",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Command groups
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--enable", action="store_true", help="Enable BlackJAX tests")
    action_group.add_argument("--disable", action="store_true", help="Disable BlackJAX tests")
    action_group.add_argument("--status", action="store_true", help="Show current status")

    # Test run options
    parser.add_argument("--run", action="store_true", help="Run tests after changing settings")
    parser.add_argument(
        "--all", action="store_true", help="Run all tests (implied when using --enable)"
    )
    parser.add_argument("--only-blackjax", action="store_true", help="Run only BlackJAX tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Run tests in verbose mode")

    # Parallel execution options
    parallel_group = parser.add_argument_group("parallel execution")
    parallel_group.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution (parallel is default)",
    )
    parallel_group.add_argument(
        "-j", "--jobs", type=str, help="Number of parallel jobs (default: auto)"
    )

    # Other options
    parser.add_argument("-c", "--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar (use dots instead)"
    )

    args = parser.parse_args()

    # Show status by default if no action specified
    if not (args.enable or args.disable or args.status or args.run):
        args.status = True

    # Handle actions
    if args.enable:
        # Set appropriate environment variables
        set_env_var("ENABLE_BLACKJAX_TESTS", "1")
        os.environ.pop("SKIP_BLACKJAX_TESTS", None)

        print("BlackJAX tests are now ENABLED")

        # When enabling, run all tests including BlackJAX by default
        if args.run and not args.only_blackjax:
            args.all = True

    elif args.disable:
        # Set appropriate environment variables
        set_env_var("SKIP_BLACKJAX_TESTS", "1")
        os.environ.pop("ENABLE_BLACKJAX_TESTS", None)

        print("BlackJAX tests are now DISABLED")

    elif args.status:
        show_status()

    # Run tests if requested
    if args.run:
        run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
