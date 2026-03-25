#!/usr/bin/env python3
"""Repo-local GPU diagnostics that reflect the current runtime device surface."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--detailed",
        action="store_true",
        help="Show a detailed summary of the active runtime devices",
    )
    group.add_argument(
        "--test",
        action="store_true",
        help="Run the full runtime device diagnostic suite",
    )
    group.add_argument(
        "--test-critical",
        action="store_true",
        help="Run only the critical runtime device diagnostics",
    )
    return parser.parse_args(argv)


def _format_quick_status() -> str:
    """Render the quick status view."""
    from artifex.generative_models.core.device_manager import get_device_manager

    manager = get_device_manager()
    capabilities = manager.capabilities
    lines = [
        "Artifex GPU Status",
        "==================",
        f"Backend: {capabilities.default_backend or 'unavailable'}",
        f"Device type: {capabilities.device_type.value}",
        f"Visible devices: {manager.device_count}",
        f"Visible GPUs: {manager.gpu_count}",
    ]

    if capabilities.error:
        lines.append(f"Runtime error: {capabilities.error}")
    elif capabilities.visible_devices:
        lines.append("Devices:")
        for device in capabilities.visible_devices:
            lines.append(f"  - {device}")

    return "\n".join(lines)


def _format_detailed_status() -> str:
    """Render the detailed status view."""
    from artifex.generative_models.core.device_manager import get_device_manager

    manager = get_device_manager()
    capabilities = manager.capabilities
    info = manager.get_device_info()

    lines = [
        "Artifex GPU Status",
        "==================",
        f"Backend: {capabilities.default_backend or 'unavailable'}",
        f"Device type: {capabilities.device_type.value}",
        f"Visible devices: {manager.device_count}",
        f"Visible GPUs: {manager.gpu_count}",
        f"Supports mixed precision: {capabilities.supports_mixed_precision}",
        f"Supports distributed execution: {capabilities.supports_distributed}",
        f"Default device: {info['default_device'] or 'none'}",
    ]

    if capabilities.total_memory_mb is not None:
        lines.append(f"GPU memory (MB): {capabilities.total_memory_mb}")
    if capabilities.cuda_version:
        lines.append(f"CUDA version: {capabilities.cuda_version}")
    if capabilities.compute_capability:
        lines.append(f"Compute capability: {capabilities.compute_capability}")
    if capabilities.driver_version:
        lines.append(f"Driver version: {capabilities.driver_version}")

    if capabilities.error:
        lines.append(f"Runtime error: {capabilities.error}")
    elif capabilities.visible_devices:
        lines.append("Devices:")
        for device in capabilities.visible_devices:
            lines.append(f"  - {device}")

    return "\n".join(lines)


def _run_diagnostics(*, critical_only: bool) -> int:
    """Run the runtime diagnostics and return a process exit code."""
    from artifex.generative_models.core.device_testing import (
        print_test_results,
        run_device_tests,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    suite = run_device_tests(critical_only=critical_only)
    print_test_results(suite)
    return 0 if suite.is_healthy else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GPU diagnostics CLI."""
    args = parse_args(argv)

    if args.test:
        return _run_diagnostics(critical_only=False)
    if args.test_critical:
        return _run_diagnostics(critical_only=True)

    output = _format_detailed_status() if args.detailed else _format_quick_status()
    sys.stdout.write(f"{output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
