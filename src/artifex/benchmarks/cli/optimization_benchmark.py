"""Retained status helper for the deprecated optimization benchmark CLI."""

from __future__ import annotations

import argparse


UNSUPPORTED_OPTIMIZATION_BENCHMARK_MESSAGE = (
    "There is no supported public optimization benchmark CLI. Use the retained Python runtime "
    "directly for real evaluation, or keep the old optimization benchmark flow as an explicit "
    "local demo only."
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the retained status helper."""
    return argparse.ArgumentParser(
        description=(
            "Retained status helper for the old optimization benchmark demo. Public CLI "
            "execution is not shipped."
        )
    )


def main() -> None:
    """Exit with the retained benchmark CLI status message."""
    parser = create_parser()
    parser.parse_args()
    parser.error(UNSUPPORTED_OPTIMIZATION_BENCHMARK_MESSAGE)


if __name__ == "__main__":
    main()
