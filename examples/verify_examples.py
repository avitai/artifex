#!/usr/bin/env python3
"""
Comprehensive verification script for README examples.

This script tests all the key code examples from README files to ensure
they work correctly after the scaling architecture implementation.
"""

import os
import sys

import jax
import jax.numpy as jnp
from flax import nnx


# Force CPU-only mode before any JAX imports
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "false"


class ExampleTester:
    """Test runner for README examples."""

    def __init__(self):
        """Initialize the README example verifier.

        Args:
            root_dir: Root directory to search for README files
        """
        self.passed = 0
        self.failed = 0
        self.results: list[tuple[str, bool, str]] = []

    def test(self, name: str, test_func):
        """Run a single test."""
        print(f"Testing {name}...", end=" ")
        try:
            test_func()
            print("✅ PASSED")
            self.passed += 1
            self.results.append((name, True, ""))
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            self.failed += 1
            self.results.append((name, False, str(e)))

    def print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'=' * 50}")
        print("README Examples Test Summary")
        print(f"{'=' * 50}")
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        if total > 0:
            print(f"Success rate: {(self.passed / total) * 100:.1f}%")

        if self.failed > 0:
            print("\nFailed Tests:")
            for name, passed, error in self.results:
                if not passed:
                    print(f"  - {name}: {error}")

        return self.failed == 0


def test_hardware_detection():
    """Test hardware detection example."""
    # Force JAX to use CPU explicitly
    jax.config.update("jax_platforms", "cpu")

    from artifex.generative_models.core.performance import HardwareDetector, PerformanceEstimator

    detector = HardwareDetector()
    estimator = PerformanceEstimator()

    # Test hardware detection
    specs = detector.detect_hardware()
    valid_platforms = ["cpu", "gpu", "tpu"]
    assert specs.platform in valid_platforms, f"Invalid platform: {specs.platform}"
    assert specs.device_count > 0, "Device count should be positive"

    # Test performance estimation
    flops = estimator.estimate_flops_linear(batch_size=32, input_size=784, output_size=128)
    assert flops > 0, "FLOPs should be positive"

    # Test memory estimation
    memory_usage = detector.estimate_memory_usage(
        batch_size=32, sequence_length=512, hidden_size=768, num_layers=12
    )
    assert memory_usage > 0, "Memory usage should be positive"


def test_ddpm_model():
    """Test DDPM model creation."""
    from artifex.generative_models.core.configuration import (
        DDPMConfig,
        NoiseScheduleConfig,
        UNetBackboneConfig,
    )
    from artifex.generative_models.factory import create_model

    # Create RNG
    key = jax.random.key(0)
    key, params_key, dropout_key = jax.random.split(key, 3)
    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

    # Create nested configurations for DDPM
    backbone = UNetBackboneConfig(
        name="unet_backbone",
        hidden_dims=(64, 128, 256),
        activation="gelu",
        in_channels=1,
        out_channels=1,
    )
    schedule = NoiseScheduleConfig(
        name="noise_schedule",
        num_timesteps=50,
    )
    config = DDPMConfig(
        name="simple_ddpm",
        backbone=backbone,
        noise_schedule=schedule,
        input_shape=(28, 28, 1),
    )

    # Create model
    model = create_model(config, rngs=rngs)
    assert hasattr(model, "generate"), "Model should have generate method"


def test_point_cloud_model():
    """Test point cloud model example."""
    from artifex.generative_models.core.configuration import (
        PointCloudConfig,
        PointCloudNetworkConfig,
    )
    from artifex.generative_models.models.geometric import PointCloudModel

    # Create RNG
    key = jax.random.key(0)
    rngs = nnx.Rngs(params=key)

    # Create configuration with frozen dataclass
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

    # Create model
    model = PointCloudModel(config, rngs=rngs)
    assert hasattr(model, "generate"), "Model should have generate method"


def test_ebm_model():
    """Test EBM model creation."""
    from artifex.generative_models.models.energy import create_mnist_ebm

    # Create RNG
    rngs = nnx.Rngs(0)

    # Create model
    model = create_mnist_ebm(rngs=rngs)
    assert hasattr(model, "generate"), "Model should have generate method"
    assert hasattr(model, "energy"), "Model should have energy function"


def test_ebm_configuration():
    """Test EBM configuration system."""
    from artifex.generative_models.core.configuration import (
        EBMConfig,
        EnergyNetworkConfig,
        MCMCConfig,
        SampleBufferConfig,
    )
    from artifex.generative_models.factory import create_model

    # Create nested configurations for EBM
    energy_network = EnergyNetworkConfig(
        name="energy_net",
        hidden_dims=(32, 64, 128),
        activation="relu",
        network_type="mlp",
    )
    mcmc = MCMCConfig(
        name="mcmc",
        n_steps=60,
        step_size=0.01,
    )
    sample_buffer = SampleBufferConfig(
        name="buffer",
        capacity=10000,
    )

    # Create EBM configuration
    config = EBMConfig(
        name="my_ebm",
        input_dim=784,  # 28 * 28 for MNIST
        energy_network=energy_network,
        mcmc=mcmc,
        sample_buffer=sample_buffer,
    )

    # Create model
    rngs = nnx.Rngs(0)
    model = create_model(config, rngs=rngs)
    assert hasattr(model, "generate"), "Model should have generate method"


def test_production_optimization():
    """Test production optimization components."""
    from artifex.generative_models.core.performance import HardwareDetector
    from artifex.generative_models.inference.optimization.production import ProductionOptimizer

    # Test hardware detection
    detector = HardwareDetector()
    specs = detector.detect_hardware()
    assert specs is not None, "Hardware specs should be available"

    # Test production optimizer creation
    optimizer = ProductionOptimizer()
    assert optimizer is not None, "ProductionOptimizer should be created"


def test_sharding_strategies():
    """Test sharding strategy imports."""
    from artifex.generative_models.scaling.sharding import (
        DataParallelStrategy,
        MultiDimensionalStrategy,
        ParallelismConfig,
        ShardingConfig,
        TensorParallelStrategy,
    )

    # Test strategy creation
    data_strategy = DataParallelStrategy("data", 0)
    tensor_strategy = TensorParallelStrategy("model", 1)

    # Test MultiDimensionalStrategy with proper initialization
    strategies = {"data": data_strategy, "tensor": tensor_strategy}

    # Create ShardingConfig first, then ParallelismConfig
    sharding_config = ShardingConfig(data_parallel_size=2, tensor_parallel_size=2)
    config = ParallelismConfig.from_sharding_config(sharding_config)
    multi_strategy = MultiDimensionalStrategy(strategies, config)

    assert data_strategy is not None, "DataParallelStrategy created"
    assert tensor_strategy is not None, "TensorParallelStrategy created"
    assert multi_strategy is not None, "MultiDimensionalStrategy created"


def test_model_adapters():
    """Test model adapter functionality."""
    from artifex.generative_models.core.adapters import create_transformer_adapter
    from artifex.generative_models.core.configuration import (
        PointCloudConfig,
        PointCloudNetworkConfig,
    )
    from artifex.generative_models.core.performance import HardwareDetector

    # Create dummy model for testing
    from artifex.generative_models.models.geometric import PointCloudModel

    rngs = nnx.Rngs(0)
    network = PointCloudNetworkConfig(
        name="network",
        hidden_dims=(64, 128),
        activation="gelu",
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
    )
    config = PointCloudConfig(
        name="test_adapter",
        network=network,
        num_points=512,
        dropout_rate=0.1,
    )
    model = PointCloudModel(config, rngs=rngs)

    # Get hardware specs
    detector = HardwareDetector()
    hardware_specs = detector.detect_hardware()

    # Test adapter creation using factory function
    adapter = create_transformer_adapter(
        model=model, hardware_specs=hardware_specs, num_layers=2, hidden_size=64, num_heads=4
    )
    assert adapter is not None, "TransformerAdapter should be created"

    # Test scaling recommendations
    specs = adapter.get_model_specs()
    assert specs is not None, "Model specs should be available"

    performance = adapter.get_performance_characteristics()
    assert performance is not None, "Performance characteristics available"

    optimal_batch = adapter.get_optimal_batch_size()
    assert optimal_batch > 0, "Optimal batch size should be positive"


def test_protein_extensions():
    """Test protein extensions."""
    from artifex.generative_models.extensions.protein import create_protein_extensions

    # Create RNG
    rngs = nnx.Rngs(params=jax.random.key(42))

    # Create protein extensions
    protein_config = {
        "use_backbone_constraints": True,
        "bond_length_weight": 1.0,
        "bond_angle_weight": 0.5,
        "use_protein_mixin": True,
    }

    extensions = create_protein_extensions(protein_config, rngs=rngs)
    assert extensions is not None, "Protein extensions should be created"
    assert len(extensions) > 0, "Extensions dictionary should not be empty"


def test_memory_usage():
    """Test basic model memory usage."""
    from artifex.generative_models.core.configuration import (
        PointCloudConfig,
        PointCloudNetworkConfig,
    )
    from artifex.generative_models.models.geometric import PointCloudModel

    # Create small model to test memory
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

    # Test forward pass (note: no rngs parameter in __call__)
    dummy_input = jnp.ones((2, 64, 3))  # batch_size=2, num_points=64, 3D
    output = model(dummy_input, deterministic=True)

    assert output is not None, "Model should produce output"
    assert "positions" in output, "Output should contain positions"


def main():
    """Run all README example tests."""
    print("🔍 Verifying README Examples")
    print("=" * 50)

    tester = ExampleTester()

    # Test all examples
    tester.test("Hardware Detection", test_hardware_detection)
    tester.test("DDPM Model Creation", test_ddpm_model)
    tester.test("Point Cloud Model", test_point_cloud_model)
    tester.test("EBM Model Creation", test_ebm_model)
    tester.test("EBM Configuration", test_ebm_configuration)
    tester.test("Production Optimization", test_production_optimization)
    tester.test("Sharding Strategies", test_sharding_strategies)
    tester.test("Model Adapters", test_model_adapters)
    tester.test("Protein Extensions", test_protein_extensions)
    tester.test("Memory Usage", test_memory_usage)

    # Print summary
    success = tester.print_summary()

    if success:
        print("\n🎉 All README examples are working correctly!")
        return 0
    else:
        print("\n⚠️  Some README examples need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
