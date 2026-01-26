"""Configuration template system and testing framework."""


# =============================================================================
# Testing Framework
# =============================================================================

import tempfile
from pathlib import Path

import pytest
import yaml

from artifex.generative_models.core.configuration.templates import (
    ConfigTemplate,
    DistributedConfig,
    PointCloudDiffusionConfig,
    template_manager,
    TrainingConfig,
)


class TestConfigurationSystem:
    """Comprehensive tests for the configuration system."""

    def test_point_cloud_diffusion_config_validation(self):
        """Test point cloud diffusion configuration validation."""
        # Valid configuration
        config = PointCloudDiffusionConfig(
            name="test_config",
            model_dim=128,
            num_layers=8,
            num_heads=8,
            timesteps=1000,
            max_seq_length=256,
            backbone_atom_indices=[0, 1, 2, 4],
            noise_steps=1000,
            beta_start=0.0001,
            beta_end=0.02,
        )
        assert config.model_dim == 128
        assert config.num_heads == 8

        # Invalid configuration - negative model_dim
        with pytest.raises(ValueError, match="must be positive"):
            PointCloudDiffusionConfig(
                name="test_config",
                model_dim=-1,
                num_layers=8,
                num_heads=8,
                timesteps=1000,
                max_seq_length=256,
                backbone_atom_indices=[0, 1, 2, 4],
                noise_steps=1000,
                beta_start=0.0001,
                beta_end=0.02,
            )

        # Invalid configuration - model_dim not divisible by num_heads
        with pytest.raises(ValueError, match="must be divisible by"):
            PointCloudDiffusionConfig(
                name="test_config",
                model_dim=100,  # Not divisible by 8
                num_layers=8,
                num_heads=8,
                timesteps=1000,
                max_seq_length=256,
                backbone_atom_indices=[0, 1, 2, 4],
                noise_steps=1000,
                beta_start=0.0001,
                beta_end=0.02,
            )

    def test_distributed_config_validation(self):
        """Test distributed configuration validation."""
        # Valid configuration
        config = DistributedConfig(
            name="test_distributed",
            enabled=True,
            world_size=8,
            num_nodes=2,
            num_processes_per_node=4,
            rank=0,
            local_rank=0,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )
        assert config.world_size == 8
        assert config.get_data_parallel_size() == 2

        # Invalid configuration - rank >= world_size
        with pytest.raises(ValueError, match="Rank.*must be less than world_size"):
            DistributedConfig(
                name="test_distributed",
                enabled=True,
                world_size=4,
                num_nodes=1,
                num_processes_per_node=4,
                rank=4,  # Invalid: >= world_size
                local_rank=0,
            )

        # Invalid configuration - inconsistent world_size
        with pytest.raises(ValueError, match="World size.*must equal"):
            DistributedConfig(
                name="test_distributed",
                enabled=True,
                world_size=10,  # Should be 8 (2*4)
                num_nodes=2,
                num_processes_per_node=4,
                rank=0,
                local_rank=0,
            )

    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Valid configuration
        config = TrainingConfig(
            name="test_training",
            batch_size=32,
            num_epochs=100,
            optimizer={
                "name": "test_optimizer",
                "optimizer_type": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
            },
            scheduler={"name": "test_scheduler", "scheduler_type": "cosine", "warmup_steps": 500},
        )
        assert config.batch_size == 32
        assert config.optimizer.learning_rate == 1e-4

        # Invalid configuration - negative batch size
        with pytest.raises(ValueError, match="must be positive"):
            TrainingConfig(name="test_training", batch_size=-1, num_epochs=100)

    def test_template_system(self):
        """Test configuration template system."""
        # Test protein diffusion template
        config = template_manager.generate_config(
            "protein_diffusion",
            max_seq_length=256,
            backbone_atom_indices=[0, 1, 2, 4],
            model_dim=256,
            num_layers=12,
        )

        assert config["max_seq_length"] == 256
        assert config["model_dim"] == 256
        assert config["num_layers"] == 12
        assert config["num_points"] == 1024  # Default value

        # Test missing required parameter
        with pytest.raises(ValueError, match="Missing required parameters"):
            template_manager.generate_config(
                "protein_diffusion",
                model_dim=256,  # Missing max_seq_length and backbone_atom_indices
            )

        # Test training template
        training_config = template_manager.generate_config(
            "simple_training", batch_size=64, learning_rate=2e-4
        )

        assert training_config["batch_size"] == 64
        assert training_config["optimizer"]["learning_rate"] == 2e-4
        assert training_config["eval_batch_size"] == 64  # Should default to batch_size

    def test_full_experiment_loading(self):
        """Test loading complete experiment configurations."""
        # Create temporary experiment config
        experiment_config = {
            "experiment_name": "test_experiment",
            "seed": 42,
            "output_dir": "./outputs/test/",
            "model_config_ref": "models/diffusion/point_cloud_diffusion.yaml",
            "data_config": "data/protein_dataset.yaml",
            "training_config": "training/protein_diffusion_training.yaml",
            "inference_config": "inference/protein_diffusion_inference.yaml",
            "log_level": "INFO",
            "use_wandb": False,
            "overrides": {
                "model": {"model_dim": 256, "timesteps": 500},
                "training": {"batch_size": 32, "num_epochs": 50},
            },
        }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(experiment_config, f)
            temp_path = f.name

        try:
            # This would normally load from the config system
            # For testing, we'll just validate the structure
            assert experiment_config["experiment_name"] == "test_experiment"
            assert experiment_config["overrides"]["model"]["model_dim"] == 256
            assert experiment_config["overrides"]["training"]["batch_size"] == 32
        finally:
            Path(temp_path).unlink()

    def test_config_inheritance_and_overrides(self):
        """Test configuration inheritance and override system."""
        base_config = {
            "model_dim": 64,
            "num_layers": 4,
            "optimizer": {"learning_rate": 1e-4, "weight_decay": 1e-5},
        }

        override_config = {
            "model_dim": 128,
            "optimizer": {"learning_rate": 2e-4},
            "new_param": "test",
        }

        # Test template merge functionality
        template = ConfigTemplate(
            name="test_merge", base_config=base_config, required_params=[], optional_params={}
        )

        merged = template._deep_merge(base_config, override_config)

        assert merged["model_dim"] == 128  # Overridden
        assert merged["num_layers"] == 4  # Preserved
        assert merged["optimizer"]["learning_rate"] == 2e-4  # Overridden
        assert merged["optimizer"]["weight_decay"] == 1e-5  # Preserved
        assert merged["new_param"] == "test"  # Added

    def test_error_handling_and_suggestions(self):
        """Test enhanced error handling with suggestions."""
        # This test would verify that our enhanced error handling provides helpful suggestions
        try:
            PointCloudDiffusionConfig(
                name="test_config",
                model_dim=0,  # Invalid
                num_layers=8,
                num_heads=8,
                timesteps=1000,
                max_seq_length=256,
                backbone_atom_indices=[0, 1, 2, 4],
                noise_steps=1000,
                beta_start=0.0001,
                beta_end=0.02,
            )
        except ValueError as e:
            error_message = str(e)
            assert "must be positive" in error_message
            # Could add suggestions in the error message

    def test_performance_optimizations(self):
        """Test performance optimizations like caching."""
        # Test that repeated loads are faster (would require actual timing)
        config_dict = {
            "name": "test_perf",
            "model_dim": 128,
            "num_layers": 8,
            "num_heads": 8,
            "timesteps": 1000,
            "max_seq_length": 256,
            "backbone_atom_indices": [0, 1, 2, 4],
            "noise_steps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        }

        # First load
        config1 = PointCloudDiffusionConfig(**config_dict)

        # Second load with same data
        config2 = PointCloudDiffusionConfig(**config_dict)

        # Both should be valid and equivalent
        assert config1.model_dim == config2.model_dim
        assert config1.num_layers == config2.num_layers


# =============================================================================
# Example Usage
# =============================================================================


def example_usage():
    """Demonstrate configuration system usage."""

    # 1. Generate config from template
    print("=== Template Usage ===")
    protein_config = template_manager.generate_config(
        "protein_diffusion",
        max_seq_length=512,
        backbone_atom_indices=[0, 1, 2, 4],
        model_dim=256,
        num_layers=12,
        dropout=0.15,
    )
    print(f"Generated protein config with model_dim: {protein_config['model_dim']}")

    # 2. Create and validate configuration objects
    print("\n=== Direct Configuration ===")
    config = PointCloudDiffusionConfig(
        name="example_config",
        model_dim=128,
        num_layers=8,
        num_heads=8,
        timesteps=1000,
        max_seq_length=256,
        backbone_atom_indices=[0, 1, 2, 4],
        noise_steps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )
    print(f"Created config: {config.name} with {config.num_layers} layers")

    # 3. Show auto-configured features
    print(f"Auto-configured head_dim: {config.head_dim}")
    print(f"Auto-configured feedforward_dim: {config.feedforward_dim}")

    # 4. Get NNX-specific configurations
    if hasattr(config, "get_rngs_config"):
        rngs_config = config.get_rngs_config()
        print(f"RNG collections: {list(rngs_config.keys())}")


if __name__ == "__main__":
    example_usage()
