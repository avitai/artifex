"""Test suite for scaling factory functions and easy scaling utilities.

This module tests the factory functions that make scaling easy for end users,
including hardware-aware configuration, automatic parallelism selection,
and production-ready scaling utilities.
"""

from unittest.mock import Mock, patch

import pytest


# Import the components we'll implement
try:
    from artifex.generative_models.factories.scaling import (
        create_scaled_diffusion,
        create_scaled_model,
        create_scaled_transformer,
        get_optimal_parallelism_strategy,
        ModelScalingSpecs,
        ScalingConfig,
        ScalingConfigBuilder,
    )
    from artifex.generative_models.scaling.sharding import ParallelismConfig, ShardingConfig
except ImportError:
    # These will be implemented after tests are written
    pytest.skip("Scaling factory functions not yet implemented", allow_module_level=True)


class TestScalingConfigBuilder:
    """Test the ScalingConfigBuilder for easy configuration creation."""

    def test_builder_initialization(self):
        """Test ScalingConfigBuilder can be initialized."""
        builder = ScalingConfigBuilder()

        assert builder is not None
        assert hasattr(builder, "build")
        assert hasattr(builder, "with_hardware_detection")
        assert hasattr(builder, "with_parallelism_strategy")

    def test_hardware_detection_integration(self):
        """Test builder can integrate with hardware detection."""
        builder = ScalingConfigBuilder()

        # Test with automatic hardware detection
        config = builder.with_hardware_detection().build()

        assert isinstance(config, ScalingConfig)
        assert hasattr(config, "hardware_specs")
        assert hasattr(config, "parallelism_config")
        assert hasattr(config, "performance_config")

    def test_custom_parallelism_strategy(self):
        """Test builder can accept custom parallelism strategies."""
        builder = ScalingConfigBuilder()

        # Test with custom strategy
        config = builder.with_parallelism_strategy(
            data_parallel=True, tensor_parallel=2, pipeline_parallel=1
        ).build()

        assert config.parallelism_config is not None
        assert config.parallelism_config.sharding_config.data_parallel_size >= 1
        assert config.parallelism_config.sharding_config.tensor_parallel_size == 2

    def test_model_specific_configuration(self):
        """Test builder can create model-specific configurations."""
        builder = ScalingConfigBuilder()

        # Test transformer-specific config
        transformer_config = builder.for_transformer(
            num_layers=24, hidden_size=1024, num_heads=16
        ).build()

        assert transformer_config.model_specs is not None
        assert transformer_config.model_specs.num_layers == 24
        assert transformer_config.model_specs.hidden_size == 1024

    def test_automatic_optimization(self):
        """Test builder can automatically optimize configuration."""
        builder = ScalingConfigBuilder()

        # Test automatic optimization
        config = builder.with_automatic_optimization(
            target_throughput="1000 samples/sec", memory_budget="8GB"
        ).build()

        assert config.optimization_targets is not None
        assert config.optimization_targets.throughput == "1000 samples/sec"
        assert config.optimization_targets.memory == "8GB"


class TestModelFactories:
    """Test hardware-aware model factory functions."""

    def test_create_scaled_transformer(self):
        """Test create_scaled_transformer factory function."""
        # Test with minimal configuration
        model, config = create_scaled_transformer(
            vocab_size=1000, hidden_size=512, num_layers=6, num_heads=8
        )

        assert model is not None
        assert config is not None
        assert isinstance(config, ScalingConfig)
        assert config.model_specs.hidden_size == 512
        assert config.model_specs.num_layers == 6

    def test_create_scaled_transformer_with_hardware(self):
        """Test transformer factory with automatic hardware optimization."""
        # Mock hardware detection
        with patch("artifex.generative_models.core.performance.HardwareDetector") as mock_detector:
            mock_hw = Mock()
            mock_hw.memory_gb = 16
            mock_hw.compute_capability = "7.5"
            mock_hw.device_type = "gpu"
            mock_detector.return_value.detect_hardware.return_value = mock_hw

            model, config = create_scaled_transformer(
                vocab_size=1000, hidden_size=1024, num_layers=12, num_heads=16, auto_optimize=True
            )

            assert model is not None
            assert config.hardware_specs is not None
            assert config.parallelism_config is not None

    def test_create_scaled_diffusion(self):
        """Test create_scaled_diffusion factory function."""
        model, config = create_scaled_diffusion(
            image_size=256, channels=3, model_channels=128, num_res_blocks=2
        )

        assert model is not None
        assert config is not None
        assert isinstance(config, ScalingConfig)
        assert config.model_specs.image_size == 256
        assert config.model_specs.channels == 3

    def test_create_scaled_model_generic(self):
        """Test generic create_scaled_model function."""
        model_config = {"model_type": "transformer", "vocab_size": 1000, "hidden_size": 512}

        model, config = create_scaled_model(model_config)

        assert model is not None
        assert config is not None
        assert config.model_specs.model_type == "transformer"

    def test_factory_with_device_count(self):
        """Test factory functions adapt to device count."""
        # Test with specific device count
        model, config = create_scaled_transformer(
            vocab_size=1000, hidden_size=512, num_layers=6, num_heads=8, device_count=4
        )

        # Should configure parallelism for 4 devices
        assert config.parallelism_config is not None
        total_devices = (
            config.parallelism_config.sharding_config.data_parallel_size
            * config.parallelism_config.sharding_config.tensor_parallel_size
            * config.parallelism_config.sharding_config.pipeline_parallel_size
        )
        assert total_devices <= 4


class TestAutomaticParallelismSelection:
    """Test automatic parallelism strategy selection."""

    def test_get_optimal_parallelism_strategy(self):
        """Test automatic parallelism strategy selection."""
        # Mock model specifications
        model_specs = ModelScalingSpecs(
            num_parameters=1_000_000_000,  # 1B parameters
            memory_requirement_gb=4.0,
            compute_intensity="high",
        )

        strategy = get_optimal_parallelism_strategy(
            model_specs=model_specs, device_count=8, memory_per_device_gb=16
        )

        assert isinstance(strategy, ParallelismConfig)
        assert strategy.sharding_config.data_parallel_size >= 1
        assert strategy.sharding_config.tensor_parallel_size >= 1
        assert strategy.sharding_config.pipeline_parallel_size >= 1

    def test_strategy_selection_for_large_model(self):
        """Test strategy selection for large models requiring sharding."""
        model_specs = ModelScalingSpecs(
            num_parameters=100_000_000_000,  # 100B parameters
            memory_requirement_gb=200.0,
            compute_intensity="very_high",
        )

        strategy = get_optimal_parallelism_strategy(
            model_specs=model_specs, device_count=32, memory_per_device_gb=80
        )

        # Large model should use multiple parallelism dimensions
        assert (
            strategy.sharding_config.tensor_parallel_size > 1
            or strategy.sharding_config.pipeline_parallel_size > 1
        )
        assert strategy.sharding_config.fsdp_enabled is True

    def test_strategy_selection_for_small_model(self):
        """Test strategy selection for small models."""
        model_specs = ModelScalingSpecs(
            num_parameters=10_000_000,  # 10M parameters
            memory_requirement_gb=0.1,
            compute_intensity="low",
        )

        strategy = get_optimal_parallelism_strategy(
            model_specs=model_specs, device_count=4, memory_per_device_gb=16
        )

        # Small model should primarily use data parallelism
        assert strategy.sharding_config.data_parallel_size >= 2
        assert strategy.sharding_config.tensor_parallel_size == 1
        assert strategy.sharding_config.pipeline_parallel_size == 1

    def test_memory_constrained_selection(self):
        """Test strategy selection under memory constraints."""
        model_specs = ModelScalingSpecs(
            num_parameters=50_000_000_000,  # 50B parameters
            memory_requirement_gb=100.0,
            compute_intensity="high",
        )

        strategy = get_optimal_parallelism_strategy(
            model_specs=model_specs,
            device_count=16,
            memory_per_device_gb=24,  # Limited memory
        )

        # Should enable FSDP and use tensor/pipeline parallelism
        assert strategy.sharding_config.fsdp_enabled is True
        assert (
            strategy.sharding_config.tensor_parallel_size > 1
            or strategy.sharding_config.pipeline_parallel_size > 1
        )


class TestProductionScalingUtilities:
    """Test production-ready scaling utilities."""

    def test_estimate_training_time(self):
        """Test training time estimation."""
        from artifex.generative_models.factories.scaling import estimate_training_time

        config = ScalingConfig(
            model_specs=ModelScalingSpecs(num_parameters=1_000_000_000, memory_requirement_gb=4.0),
            parallelism_config=ParallelismConfig.from_sharding_config(
                ShardingConfig(data_parallel_size=4, tensor_parallel_size=2)
            ),
        )

        time_estimate = estimate_training_time(
            config=config, dataset_size=1_000_000, batch_size=32, num_epochs=10
        )

        assert isinstance(time_estimate, dict)
        assert "total_hours" in time_estimate
        assert "samples_per_second" in time_estimate
        assert time_estimate["total_hours"] > 0

    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        from artifex.generative_models.factories.scaling import estimate_memory_usage

        config = ScalingConfig(
            model_specs=ModelScalingSpecs(num_parameters=1_000_000_000, memory_requirement_gb=4.0),
            parallelism_config=ParallelismConfig.from_sharding_config(
                ShardingConfig(data_parallel_size=1, tensor_parallel_size=1)
            ),
        )

        memory_estimate = estimate_memory_usage(config=config, batch_size=16, sequence_length=512)

        assert isinstance(memory_estimate, dict)
        assert "model_memory_gb" in memory_estimate
        assert "activation_memory_gb" in memory_estimate
        assert "total_memory_gb" in memory_estimate

    def test_cost_estimation(self):
        """Test cloud cost estimation."""
        from artifex.generative_models.factories.scaling import estimate_cloud_cost

        config = ScalingConfig(
            parallelism_config=ParallelismConfig.from_sharding_config(
                ShardingConfig(data_parallel_size=4, tensor_parallel_size=2)
            )
        )

        cost_estimate = estimate_cloud_cost(
            config=config, training_hours=100, cloud_provider="gcp", instance_type="a100-40gb"
        )

        assert isinstance(cost_estimate, dict)
        assert "total_cost_usd" in cost_estimate
        assert "cost_per_hour" in cost_estimate
        assert cost_estimate["total_cost_usd"] > 0


class TestIntegrationScaling:
    """Test integration between all scaling components."""

    def test_end_to_end_transformer_scaling(self):
        """Test complete end-to-end transformer scaling workflow."""
        # Build configuration
        builder = ScalingConfigBuilder()
        config = (
            builder.for_transformer(num_layers=12, hidden_size=768, num_heads=12)
            .with_hardware_detection()
            .with_automatic_optimization(target_throughput="500 samples/sec")
            .build()
        )

        # Create scaled model with all required parameters
        model, final_config = create_scaled_transformer(
            vocab_size=50000, hidden_size=768, num_layers=12, num_heads=12, config=config
        )

        assert model is not None
        assert final_config is not None
        assert final_config.hardware_specs is not None
        assert final_config.parallelism_config is not None

    def test_multi_node_scaling_configuration(self):
        """Test configuration for multi-node scaling."""
        config = (
            ScalingConfigBuilder()
            .for_transformer(num_layers=12, hidden_size=768, num_heads=12)
            .with_parallelism_strategy(data_parallel=True, tensor_parallel=2, pipeline_parallel=2)
            .build()
        )

        # Simulate multi-node configuration
        assert config.parallelism_config is not None
        total_devices = (
            config.parallelism_config.sharding_config.data_parallel_size
            * config.parallelism_config.sharding_config.tensor_parallel_size
            * config.parallelism_config.sharding_config.pipeline_parallel_size
        )
        # Should configure parallelism across available devices
        assert total_devices >= 1

    def test_inference_optimization_mode(self):
        """Test scaling configuration optimized for inference."""
        config = (
            ScalingConfigBuilder()
            .for_transformer(num_layers=12, hidden_size=768, num_heads=12)
            .with_parallelism_strategy(pipeline_parallel=2)
            .build()
        )

        assert config.parallelism_config is not None
        # Inference should prefer pipeline parallelism
        assert config.parallelism_config.sharding_config.pipeline_parallel_size >= 1

    def test_configuration_validation(self):
        """Test that invalid configurations are caught."""
        builder = ScalingConfigBuilder()

        with pytest.raises(ValueError, match="Invalid device count"):
            builder.for_transformer(num_layers=12, hidden_size=768).with_device_count(-1).build()

        with pytest.raises(ValueError, match="Memory budget too low"):
            builder.with_automatic_optimization(
                memory_budget="100MB"  # Too low for any model
            ).build()
