"""Tests for Trainer class with unified configuration system.

Following TDD principles - these tests are written FIRST before implementation.
"""

import optax
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    VAEConfig,
)
from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.factory import create_model
from artifex.generative_models.training.trainer import Trainer


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def model_config():
    """Valid VAEConfig for creating a model."""
    encoder = EncoderConfig(
        name="test_encoder",
        input_shape=(28, 28, 1),
        latent_dim=32,
        hidden_dims=(256, 128),
        activation="gelu",
    )
    decoder = DecoderConfig(
        name="test_decoder",
        latent_dim=32,
        output_shape=(28, 28, 1),
        hidden_dims=(128, 256),
        activation="gelu",
    )
    return VAEConfig(
        name="test_vae",
        encoder=encoder,
        decoder=decoder,
        encoder_type="dense",
        kl_weight=1.0,
    )


@pytest.fixture
def model(model_config, rngs):
    """Create a test model."""
    return create_model(config=model_config, rngs=rngs)


@pytest.fixture
def valid_training_config():
    """Valid TrainingConfig for testing."""
    optimizer_config = OptimizerConfig(
        name="adam_optimizer",
        optimizer_type="adam",
        learning_rate=1e-3,
    )

    scheduler_config = SchedulerConfig(
        name="cosine_scheduler",
        scheduler_type="cosine",
        warmup_steps=100,
    )

    return TrainingConfig(
        name="test_training",
        batch_size=32,
        num_epochs=10,
        optimizer=optimizer_config,
        gradient_clip_norm=1.0,
        scheduler=scheduler_config,
    )


@pytest.fixture
def optimizer():
    """Create a simple optimizer."""
    return optax.adam(1e-3)


class TestTrainerUnifiedConfig:
    """Test Trainer class with unified configuration requirements."""

    def test_trainer_requires_training_configuration(self, model, optimizer):
        """Test that Trainer raises TypeError for non-TrainingConfig."""
        # Dict config should raise TypeError
        with pytest.raises(TypeError, match="training_config must be a TrainingConfig"):
            Trainer(
                model=model,
                optimizer=optimizer,
                training_config={"learning_rate": 1e-3, "batch_size": 32},
            )

        # Any other type should raise TypeError
        with pytest.raises(TypeError, match="training_config must be a TrainingConfig"):
            Trainer(
                model=model,
                optimizer=optimizer,
                training_config="invalid",
            )

        # None should raise TypeError (required parameter)
        with pytest.raises(TypeError, match="training_config is required"):
            Trainer(
                model=model,
                optimizer=optimizer,
                training_config=None,
            )

    def test_trainer_accepts_training_configuration(self, model, optimizer, valid_training_config):
        """Test that Trainer works correctly with TrainingConfig."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            training_config=valid_training_config,
        )

        # Trainer should be created successfully
        assert trainer is not None
        assert trainer.training_config == valid_training_config
        assert trainer.model is model
        assert trainer.optimizer is optimizer

    def test_trainer_training_config_attributes_accessible(
        self, model, optimizer, valid_training_config
    ):
        """Test that TrainingConfig attributes are accessible in trainer."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            training_config=valid_training_config,
        )

        # Should be able to access training config attributes
        assert trainer.training_config.optimizer.learning_rate == 1e-3
        assert trainer.training_config.batch_size == 32
        assert trainer.training_config.num_epochs == 10
        assert trainer.training_config.optimizer.optimizer_type == "adam"

    def test_trainer_with_all_parameters(self, model, optimizer, valid_training_config):
        """Test Trainer with all parameters including typed config."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            training_config=valid_training_config,
            train_data_loader=lambda: None,  # Mock data loader
            val_data_loader=lambda: None,  # Mock data loader
            workdir="/tmp/test",
            checkpoint_dir="/tmp/test/checkpoints",
            save_interval=500,
        )

        assert trainer.training_config == valid_training_config
        assert trainer.workdir == "/tmp/test"
        assert trainer.checkpoint_dir == "/tmp/test/checkpoints"
        assert trainer.save_interval == 500

    def test_legacy_training_config_rejected(self, model, optimizer):
        """Test that legacy training config classes are rejected."""

        # Mock a legacy training config class
        class LegacyTrainingConfig:
            def __init__(self):
                self.learning_rate = 1e-3
                self.batch_size = 32
                self.num_epochs = 10

        legacy_config = LegacyTrainingConfig()

        with pytest.raises(TypeError, match="training_config must be a TrainingConfig"):
            Trainer(
                model=model,
                optimizer=optimizer,
                training_config=legacy_config,
            )

    def test_training_configuration_validation(self, model, optimizer):
        """Test that invalid TrainingConfig raises appropriate errors."""
        # Test with minimal valid config
        minimal_optimizer = OptimizerConfig(
            name="minimal_optimizer",
            optimizer_type="adam",
            learning_rate=1e-3,
        )

        minimal_config = TrainingConfig(
            name="minimal",
            optimizer=minimal_optimizer,
            batch_size=32,
            num_epochs=1,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            training_config=minimal_config,
        )
        assert trainer.training_config == minimal_config

    def test_trainer_methods_with_typed_config(self, model, optimizer, valid_training_config):
        """Test that trainer methods can access typed config properly."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            training_config=valid_training_config,
        )

        # Test that we can access config in trainer methods
        # This tests the integration, not the actual training
        assert hasattr(trainer, "training_config")
        assert isinstance(trainer.training_config, TrainingConfig)

        # The trainer should be able to use config for training parameters
        if hasattr(trainer, "get_learning_rate"):
            lr = trainer.get_learning_rate()
            assert lr == valid_training_config.learning_rate
