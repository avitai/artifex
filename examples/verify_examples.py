#!/usr/bin/env python3
"""Maintainer smoke checks for the root examples contract.

This utility reads the live root examples README, verifies that documented
`uv run python examples/...` commands resolve to real files, and then runs a
small set of maintained API smoke checks that back the root example surface.
It is not a full-catalog example runner.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = Path(__file__).resolve().with_name("README.md")
UV_COMMAND_PATTERN = re.compile(r"uv run python (?P<path>examples/[\w./-]+\.py)")
BARE_PYTHON_PATTERN = re.compile(r"(?m)^python examples/[\w./-]+\.py$")
EXAMPLE_FAILURES = (
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
)


def show(message: str) -> None:
    """Emit verification output through logging."""
    logger.info(message)


class SmokeTester:
    """Simple runner for root examples contract smoke checks."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.results: list[tuple[str, bool, str]] = []

    def test(self, name: str, test_func: Callable[[], None]) -> None:
        """Run a single smoke check."""
        show(f"Testing {name}...")
        try:
            test_func()
            show("  PASSED")
            self.passed += 1
            self.results.append((name, True, ""))
        except EXAMPLE_FAILURES as error:
            show(f"  FAILED: {error}")
            self.failed += 1
            self.results.append((name, False, str(error)))

    def print_summary(self) -> bool:
        """Print the final smoke-check summary."""
        total = self.passed + self.failed
        show("")
        show("=" * 50)
        show("Root Examples Contract Smoke Summary")
        show("=" * 50)
        show(f"Total checks: {total}")
        show(f"Passed: {self.passed}")
        show(f"Failed: {self.failed}")
        if total > 0:
            show(f"Success rate: {(self.passed / total) * 100:.1f}%")

        if self.failed > 0:
            show("")
            show("Failed checks:")
            for name, passed, error in self.results:
                if not passed:
                    show(f"  - {name}: {error}")

        return self.failed == 0


def _read_root_examples_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _readme_commands() -> list[str]:
    contents = _read_root_examples_readme()
    return UV_COMMAND_PATTERN.findall(contents)


def test_root_examples_readme_contract() -> None:
    """Check that the root README advertises real source-checkout commands."""
    contents = _read_root_examples_readme()
    commands = _readme_commands()

    assert "source ./activate.sh" in contents
    assert "uv run python" in contents
    assert "JAX_PLATFORMS" not in contents
    assert not BARE_PYTHON_PATTERN.search(contents)
    assert commands, "README should document at least one uv-backed example command"

    for relative_path in commands:
        path = PROJECT_ROOT / relative_path
        assert path.exists(), f"Documented example path is missing: {relative_path}"


def test_hardware_detection() -> None:
    """Test hardware detection example."""
    from artifex.generative_models.core.performance import HardwareDetector, PerformanceEstimator

    detector = HardwareDetector()
    estimator = PerformanceEstimator()

    specs = detector.detect_hardware()
    valid_platforms = ["cpu", "gpu", "tpu"]
    assert specs.platform in valid_platforms, f"Invalid platform: {specs.platform}"
    assert specs.device_count > 0, "Device count should be positive"

    flops = estimator.estimate_flops_linear(batch_size=32, input_size=784, output_size=128)
    assert flops > 0, "FLOPs should be positive"

    memory_usage = detector.estimate_memory_usage(
        batch_size=32, sequence_length=512, hidden_size=768, num_layers=12
    )
    assert memory_usage > 0, "Memory usage should be positive"


def test_ddpm_model() -> None:
    """Test DDPM model creation."""
    from artifex.generative_models.core.configuration import (
        DDPMConfig,
        NoiseScheduleConfig,
        UNetBackboneConfig,
    )
    from artifex.generative_models.factory import create_model

    key = jax.random.key(0)
    key, params_key, dropout_key = jax.random.split(key, 3)
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

    backbone = UNetBackboneConfig(
        name="unet_backbone",
        hidden_dims=(64, 128, 256),
        activation="gelu",
        in_channels=1,
        out_channels=1,
    )
    schedule = NoiseScheduleConfig(name="noise_schedule", num_timesteps=50)
    config = DDPMConfig(
        name="simple_ddpm",
        backbone=backbone,
        noise_schedule=schedule,
        input_shape=(28, 28, 1),
    )

    model = create_model(config, rngs=rngs)
    assert hasattr(model, "generate"), "Model should have generate method"


def test_point_cloud_model() -> None:
    """Test point cloud model example."""
    from artifex.generative_models.core.configuration import (
        PointCloudConfig,
        PointCloudNetworkConfig,
    )
    from artifex.generative_models.models.geometric import PointCloudModel

    key = jax.random.key(0)
    rngs = nnx.Rngs(params=key)

    network = PointCloudNetworkConfig(
        name="network",
        hidden_dims=(128, 256),
        activation="gelu",
        embed_dim=128,
        num_heads=8,
        num_layers=3,
        dropout_rate=0.1,
    )
    config = PointCloudConfig(
        name="point_cloud",
        network=network,
        num_points=1024,
        dropout_rate=0.1,
    )

    model = PointCloudModel(config, rngs=rngs)
    assert hasattr(model, "generate"), "Model should have generate method"


def test_ebm_model() -> None:
    """Test EBM model creation."""
    from artifex.generative_models.models.energy import create_mnist_ebm

    rngs = nnx.Rngs(0)
    model = create_mnist_ebm(rngs=rngs)
    assert hasattr(model, "generate"), "Model should have generate method"
    assert hasattr(model, "energy"), "Model should have energy function"


def test_ebm_configuration() -> None:
    """Test EBM configuration system."""
    from artifex.generative_models.core.configuration import (
        EBMConfig,
        EnergyNetworkConfig,
        MCMCConfig,
        SampleBufferConfig,
    )
    from artifex.generative_models.factory import create_model

    energy_network = EnergyNetworkConfig(
        name="energy_net",
        hidden_dims=(32, 64, 128),
        activation="relu",
        network_type="mlp",
    )
    mcmc = MCMCConfig(name="mcmc", n_steps=60, step_size=0.01)
    sample_buffer = SampleBufferConfig(name="buffer", capacity=10000)

    config = EBMConfig(
        name="my_ebm",
        input_dim=784,
        energy_network=energy_network,
        mcmc=mcmc,
        sample_buffer=sample_buffer,
    )

    rngs = nnx.Rngs(0)
    model = create_model(config, rngs=rngs)
    assert hasattr(model, "generate"), "Model should have generate method"


def test_production_optimization() -> None:
    """Test production optimization components."""
    from artifex.generative_models.core.performance import HardwareDetector
    from artifex.generative_models.inference.optimization.production import ProductionOptimizer

    detector = HardwareDetector()
    specs = detector.detect_hardware()
    assert specs is not None, "Hardware specs should be available"

    optimizer = ProductionOptimizer()
    assert optimizer is not None, "ProductionOptimizer should be created"


def test_sharding_strategies() -> None:
    """Test sharding strategy imports."""
    from artifex.generative_models.scaling.sharding import (
        DataParallelStrategy,
        MultiDimensionalStrategy,
        ParallelismConfig,
        ShardingConfig,
        TensorParallelStrategy,
    )

    data_strategy = DataParallelStrategy("data", 0)
    tensor_strategy = TensorParallelStrategy("model", 1)
    strategies = {"data": data_strategy, "tensor": tensor_strategy}
    sharding_config = ShardingConfig(data_parallel_size=2, tensor_parallel_size=2)
    config = ParallelismConfig.from_sharding_config(sharding_config)
    multi_strategy = MultiDimensionalStrategy(strategies, config)

    assert data_strategy is not None
    assert tensor_strategy is not None
    assert multi_strategy is not None


def test_protein_extensions() -> None:
    """Test protein extensions."""
    from artifex.configs import (
        ProteinExtensionConfig,
        ProteinExtensionsConfig,
        ProteinMixinConfig,
    )
    from artifex.generative_models.extensions.protein import create_protein_extensions

    rngs = nnx.Rngs(params=jax.random.key(42))
    protein_config = ProteinExtensionsConfig(
        name="verify_examples_protein_extensions",
        bond_length=ProteinExtensionConfig(
            name="bond_length",
            weight=1.0,
            bond_length_weight=1.0,
        ),
        bond_angle=ProteinExtensionConfig(
            name="bond_angle",
            weight=0.5,
            bond_angle_weight=0.5,
        ),
        mixin=ProteinMixinConfig(
            name="protein_mixin",
            embedding_dim=16,
            num_aa_types=21,
        ),
    )

    extensions = create_protein_extensions(protein_config, rngs=rngs)
    assert extensions is not None
    assert len(extensions) > 0


def test_memory_usage() -> None:
    """Test basic model memory usage."""
    from artifex.generative_models.core.configuration import (
        PointCloudConfig,
        PointCloudNetworkConfig,
    )
    from artifex.generative_models.models.geometric import PointCloudModel

    rngs = nnx.Rngs(0)
    network = PointCloudNetworkConfig(
        name="network",
        hidden_dims=(32, 64),
        activation="gelu",
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.0,
    )
    config = PointCloudConfig(
        name="test_memory",
        network=network,
        num_points=64,
        dropout_rate=0.0,
    )
    model = PointCloudModel(config, rngs=rngs)

    dummy_input = jnp.ones((2, 64, 3))
    output = model(dummy_input, deterministic=True)

    assert output is not None
    assert "positions" in output


def main() -> int:
    """Run the retained root examples contract smoke checks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    show("Verifying root examples contract")
    show("=" * 50)

    tester = SmokeTester()
    tester.test("Root README Contract", test_root_examples_readme_contract)
    tester.test("Hardware Detection", test_hardware_detection)
    tester.test("DDPM Model Creation", test_ddpm_model)
    tester.test("Point Cloud Model", test_point_cloud_model)
    tester.test("EBM Model Creation", test_ebm_model)
    tester.test("EBM Configuration", test_ebm_configuration)
    tester.test("Production Optimization", test_production_optimization)
    tester.test("Sharding Strategies", test_sharding_strategies)
    tester.test("Protein Extensions", test_protein_extensions)
    tester.test("Memory Usage", test_memory_usage)

    success = tester.print_summary()
    if success:
        show("")
        show("Root README contract checks and maintained API smoke tests passed.")
        return 0

    show("")
    show("Root examples contract smoke checks need attention.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
