#!/usr/bin/env python3
"""
Geometric Demo Utilities

This script provides utilities for running and configuring the geometric benchmark demo.
It includes preset configurations, troubleshooting tools, and performance analysis.

Usage:
    # Quick demo (fast, minimal resources)
    python demo_utils.py --preset quick

    # Full demo (comprehensive, longer training)
    python demo_utils.py --preset full

    # Custom configuration
    python demo_utils.py --config custom_config.json

    # Check system requirements
    python demo_utils.py --check-system

    # Troubleshoot issues
    python demo_utils.py --troubleshoot
"""

import argparse
import json
import os
import sys
from pathlib import Path

import psutil


def get_preset_configs():
    """Get predefined configuration presets."""
    return {
        "quick": {
            "description": "Quick demo - minimal resources, fast execution",
            "workdir": "./examples_output/geometric_demo_quick",
            "dataset": {
                "data_path": "./data/shapenet_quick",
                "num_points": 512,
                "synsets": ["02691156"],  # Just airplanes
                "normalize": True,
                "data_source": "synthetic",  # Use synthetic for speed
                "models_per_synset": 10,
                "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            },
            "model": {
                "embed_dim": 64,  # Smaller model
                "num_points": 512,
                "num_layers": 2,  # Fewer layers
                "num_heads": 4,
                "dropout": 0.1,
            },
            "training": {
                "batch_size": 4,  # Small batch
                "num_epochs": 10,  # Quick training
                "log_freq": 2,
                "eval_freq": 5,
                "save_freq": 10,
                "optimizer": {
                    "optimizer_type": "adam",
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-6,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "eps": 1e-8,
                },
                "scheduler": {
                    "scheduler_type": "constant",
                    "warmup_steps": 0,
                    "warmup_ratio": 0.0,
                    "min_lr_ratio": 1.0,
                },
            },
        },
        "full": {
            "description": "Full demo - comprehensive training with real data",
            "workdir": "./examples_output/geometric_demo_full",
            "dataset": {
                "data_path": "./data/shapenet",
                "num_points": 2048,  # Full resolution
                "synsets": ["02691156", "02958343", "03001627"],  # Multiple categories
                "normalize": True,
                "data_source": "auto",  # Try real data first
                "models_per_synset": 50,
                "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            },
            "model": {
                "embed_dim": 256,  # Larger model
                "num_points": 2048,
                "num_layers": 6,  # Deeper model
                "num_heads": 8,
                "dropout": 0.1,
            },
            "training": {
                "batch_size": 8,
                "num_epochs": 100,  # Long training
                "log_freq": 10,
                "eval_freq": 20,
                "save_freq": 25,
                "optimizer": {
                    "optimizer_type": "adamw",
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-5,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "eps": 1e-8,
                },
                "scheduler": {
                    "scheduler_type": "cosine",
                    "warmup_steps": 200,
                    "warmup_ratio": 0.1,
                    "min_lr_ratio": 0.01,
                },
            },
        },
        "debug": {
            "description": "Debug mode - minimal setup for testing",
            "workdir": "./examples_output/geometric_demo_debug",
            "dataset": {
                "data_path": "./data/shapenet_debug",
                "num_points": 128,  # Very small
                "synsets": ["02691156"],
                "normalize": True,
                "data_source": "synthetic",
                "models_per_synset": 5,  # Minimal data
                "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            },
            "model": {
                "embed_dim": 32,  # Tiny model
                "num_points": 128,
                "num_layers": 1,
                "num_heads": 2,
                "dropout": 0.0,  # No dropout for debugging
            },
            "training": {
                "batch_size": 2,
                "num_epochs": 3,  # Very short
                "log_freq": 1,
                "eval_freq": 2,
                "save_freq": 5,
                "optimizer": {
                    "optimizer_type": "adam",
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "eps": 1e-8,
                },
                "scheduler": {
                    "scheduler_type": "constant",
                    "warmup_steps": 0,
                    "warmup_ratio": 0.0,
                    "min_lr_ratio": 1.0,
                },
            },
        },
    }


def check_system_requirements():
    """Check system requirements for running the demo."""
    print("🔍 System Requirements Check")
    print("=" * 40)

    requirements_met = True

    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 9):
        print("   ❌ Python 3.9+ required")
        requirements_met = False
    else:
        print("   ✅ Python version OK")

    # Check JAX installation
    try:
        import jax

        print(f"JAX version: {jax.__version__}")
        print(f"JAX backend: {jax.default_backend()}")

        # Test JAX functionality
        test_array = jax.numpy.array([1, 2, 3])
        jax.numpy.sum(test_array)
        print("   ✅ JAX working")

        # Check devices
        devices = jax.devices()
        print(f"   Available devices: {[str(d) for d in devices]}")

    except ImportError:
        print("   ❌ JAX not installed")
        requirements_met = False
    except Exception as e:
        print(f"   ❌ JAX error: {e}")
        requirements_met = False

    # Check memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    print(f"Available memory: {memory_gb:.1f} GB")
    if memory_gb < 4:
        print("   ⚠️  Low memory - consider using 'quick' preset")
    else:
        print("   ✅ Memory OK")

    # Check disk space
    disk = psutil.disk_usage(".")
    disk_gb = disk.free / (1024**3)
    print(f"Available disk space: {disk_gb:.1f} GB")
    if disk_gb < 1:
        print("   ❌ Insufficient disk space")
        requirements_met = False
    else:
        print("   ✅ Disk space OK")

    # Check required packages
    required_packages = ["flax", "optax", "matplotlib", "numpy", "trimesh", "huggingface_hub"]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
            requirements_met = False

    print("\n📋 Summary:")
    if requirements_met:
        print("✅ All requirements met! Ready to run the demo.")
        return True
    else:
        print("❌ Some requirements not met.")
        if missing_packages:
            print("\nInstall missing packages:")
            print(f"pip install {' '.join(missing_packages)}")
        return False


def estimate_resources(config):
    """Estimate resource requirements for a given configuration."""
    print("📊 Resource Estimation")
    print("=" * 30)

    # Model size estimation
    embed_dim = config["model"]["embed_dim"]
    num_layers = config["model"]["num_layers"]
    config["model"]["num_heads"]
    num_points = config["model"]["num_points"]

    # Rough parameter count estimation
    # Transformer blocks + embeddings + projections
    params_per_layer = embed_dim * embed_dim * 4  # Self-attention + MLP
    total_params = (
        params_per_layer * num_layers  # Transformer layers
        + num_points * embed_dim  # Position embeddings
        + embed_dim * 3  # Output projection
    )

    memory_mb = total_params * 4 / (1024**2)  # Float32

    print(f"Model Parameters: ~{total_params:,}")
    print(f"Model Memory: ~{memory_mb:.1f} MB")

    # Training estimation
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    models_per_synset = config["dataset"]["models_per_synset"]
    num_synsets = len(config["dataset"]["synsets"])

    total_samples = models_per_synset * num_synsets
    steps_per_epoch = total_samples // batch_size
    total_steps = steps_per_epoch * num_epochs

    print(f"\nTraining Steps: {total_steps:,}")
    print(f"Estimated time: {total_steps * 0.1:.1f} seconds (rough)")

    # Memory estimation
    batch_memory_mb = batch_size * num_points * 3 * 4 / (1024**2)  # Point clouds
    total_memory_mb = memory_mb + batch_memory_mb * 2  # Model + batch + gradients

    print(f"Training Memory: ~{total_memory_mb:.1f} MB")

    if total_memory_mb > 2000:  # 2GB
        print("   ⚠️  High memory usage - consider reducing batch size or model size")

    return {
        "total_params": total_params,
        "memory_mb": total_memory_mb,
        "total_steps": total_steps,
        "estimated_time": total_steps * 0.1,
    }


def troubleshoot_common_issues():
    """Provide troubleshooting guidance for common issues."""
    print("🔧 Troubleshooting Guide")
    print("=" * 30)

    issues = [
        {
            "issue": "Out of Memory Error",
            "solutions": [
                "Reduce batch_size in config",
                "Reduce model embed_dim or num_layers",
                "Reduce num_points in dataset",
                "Use 'quick' preset for minimal memory usage",
            ],
        },
        {
            "issue": "JAX/GPU Issues",
            "solutions": [
                "Install JAX with GPU support: pip install jax[cuda12]",
                "Check CUDA compatibility",
                "Set XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "Use CPU-only with XLA_FLAGS=--xla_force_host_platform_device_count=1",
            ],
        },
        {
            "issue": "Dataset Download Fails",
            "solutions": [
                "Check internet connection",
                "Use data_source: 'synthetic' for offline usage",
                "Manually create test data with debug preset",
                "Check disk space for downloads",
                "Try: python demo_utils.py --preset quick (uses synthetic data)",
            ],
        },
        {
            "issue": "No valid 3D models found in data directory",
            "solutions": [
                "Use data_source: 'synthetic' in config",
                "Run with --preset quick for guaranteed synthetic data",
                "Check if trimesh is installed: pip install trimesh",
                "Verify write permissions to data directory",
                "Delete existing data directory and retry",
            ],
        },
        {
            "issue": "Training Too Slow",
            "solutions": [
                "Use 'quick' preset",
                "Reduce num_epochs",
                "Increase batch_size (if memory allows)",
                "Use smaller model (reduce embed_dim/num_layers)",
            ],
        },
        {
            "issue": "Configuration/Import Errors",
            "solutions": [
                "Use preset configurations: python demo_utils.py --preset quick",
                "Check import paths and module availability",
                "Verify all dependencies are installed",
                "Use debug preset for minimal setup testing",
            ],
        },
        {
            "issue": "Dependencies Missing",
            "solutions": [
                "Install with: pip install flax optax matplotlib trimesh huggingface_hub",
                "Use virtual environment: python -m venv artifex_env",
                "Check Python version (3.9+ required)",
                "Update pip: pip install --upgrade pip",
            ],
        },
    ]

    for i, issue_info in enumerate(issues, 1):
        print(f"\n{i}. {issue_info['issue']}:")
        for solution in issue_info["solutions"]:
            print(f"   • {solution}")

    print("\n💡 Quick Fixes:")
    print("   • For fastest demo: python demo_utils.py --preset quick")
    print("   • For debugging: python demo_utils.py --preset debug")
    print("   • Check system: python demo_utils.py --check-system")
    print("   • Config errors: Demo now uses direct config creation (no pydantic)")


def save_config(config, filename):
    """Save configuration to file."""
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {filename}")


def load_config(filename):
    """Load configuration from file."""
    with open(filename, "r") as f:
        return json.load(f)


def run_demo_with_config(config):
    """Run the demo with the given configuration."""
    print("🚀 Running demo with configuration...")
    print(f"   Preset: {config.get('description', 'Custom')}")

    # Import and run the main demo
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        # Create a temporary config file
        config_file = "temp_demo_config.json"
        save_config(config, config_file)

        # Run the demo (simplified version)
        print("   Starting demo execution...")

        # Estimate resources first
        estimate_resources(config)

        print("\n   To run the full demo, execute:")
        print("   python examples/geometric_benchmark_demo.py")
        print("   (Make sure to update the demo with your config)")

        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)

    except Exception as e:
        print(f"   ❌ Error running demo: {e}")
        print("   Try running with --troubleshoot for help")


def main():
    """Main function for demo utilities."""
    parser = argparse.ArgumentParser(description="Geometric Demo Utilities")
    parser.add_argument(
        "--preset", choices=["quick", "full", "debug"], help="Use predefined configuration preset"
    )
    parser.add_argument("--config", type=str, help="Load custom configuration from file")
    parser.add_argument("--save-preset", type=str, help="Save preset to file")
    parser.add_argument("--check-system", action="store_true", help="Check system requirements")
    parser.add_argument("--troubleshoot", action="store_true", help="Show troubleshooting guide")
    parser.add_argument("--estimate", action="store_true", help="Estimate resource requirements")

    args = parser.parse_args()

    if args.check_system:
        check_system_requirements()
        return

    if args.troubleshoot:
        troubleshoot_common_issues()
        return

    presets = get_preset_configs()

    # Determine configuration to use
    if args.preset:
        config = presets[args.preset]
        print(f"Using preset: {args.preset}")
        print(f"Description: {config['description']}")
    elif args.config:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        # Default to quick preset
        config = presets["quick"]
        print("No preset specified, using 'quick' preset")

    # Save preset if requested
    if args.save_preset:
        save_config(config, args.save_preset)
        return

    # Estimate resources if requested
    if args.estimate:
        estimate_resources(config)
        return

    # Run the demo
    run_demo_with_config(config)


if __name__ == "__main__":
    main()
