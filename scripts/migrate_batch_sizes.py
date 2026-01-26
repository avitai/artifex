#!/usr/bin/env python3
"""Migrate batch sizes to hardware-optimal values."""

from pathlib import Path

import yaml

from artifex.generative_models.core.performance import HardwareDetector


def update_batch_sizes():
    """Update all config files with optimal batch sizes."""
    detector = HardwareDetector()
    optimal_size = detector.get_critical_batch_size()

    config_dir = Path("src/artifex/configs/defaults/training")

    for config_file in config_dir.glob("*.yaml"):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Update batch sizes
        if "batch_size" in config:
            old_size = config["batch_size"]
            config["batch_size"] = optimal_size
            print(f"Updated {config_file.name}: {old_size} -> {optimal_size}")

        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


if __name__ == "__main__":
    update_batch_sizes()
