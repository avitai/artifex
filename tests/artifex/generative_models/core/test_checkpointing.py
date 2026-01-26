"""Tests for checkpointing utilities."""

import shutil
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    setup_checkpoint_manager,
)
from tests.utils.test_models import SimpleModel


# SimpleModel is now imported from tests.utils.test_models


@pytest.fixture
def key():
    """Fixture for JAX random key."""
    return jax.random.key(0)


@pytest.fixture
def temp_dir():
    """Fixture for temporary directory."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def model(key):
    """Fixture for simple model."""
    params_key, _ = jax.random.split(key)
    return SimpleModel(rngs=nnx.Rngs(params=params_key))


class TestCheckpointing:
    """Test cases for checkpointing utilities."""

    def test_setup_checkpoint_manager(self, temp_dir):
        """Test setup_checkpoint_manager function."""
        manager, path = setup_checkpoint_manager(temp_dir)

        # Check if manager is created
        assert manager is not None

        # Check if directory is created
        assert Path(path).exists()

        # Check if path is absolute
        assert Path(path).is_absolute()

    def test_save_and_load_checkpoint(self, temp_dir, model, key):
        """Test save_checkpoint and load_checkpoint functions."""
        # Setup checkpoint manager
        manager, _ = setup_checkpoint_manager(temp_dir)

        # Save model at step 0
        save_checkpoint(manager, model, 0)

        # Check if checkpoint exists
        assert 0 in manager.all_steps()

        # Create a different model
        new_key = jax.random.key(1)  # Different seed
        new_model = SimpleModel(rngs=nnx.Rngs(params=new_key))

        # Verify the models are different
        x = jnp.ones((1, 10))
        out1 = model(x)
        out2 = new_model(x)
        assert not jnp.allclose(out1, out2)

        # Load checkpoint into new model
        loaded_model, loaded_step = load_checkpoint(manager, new_model)

        # Check loaded step
        assert loaded_step == 0

        # Verify the loaded model outputs match the original model
        out3 = loaded_model(x)
        assert jnp.allclose(out1, out3)

    def test_load_latest_checkpoint(self, temp_dir, model, key):
        """Test loading the latest checkpoint."""
        # Setup checkpoint manager
        manager, _ = setup_checkpoint_manager(temp_dir)

        # Save model at multiple steps
        save_checkpoint(manager, model, 0)

        # Update model (simulate training)
        x = jnp.ones((1, 10))
        # Forward pass to ensure variables are initialized
        _ = model(x)

        # Create new parameters for step 10
        w1 = model.dense1.kernel.value * 2
        model.dense1.kernel.value = w1

        # Save at step 10
        save_checkpoint(manager, model, 10)

        # Create a new model instance
        new_model = SimpleModel(rngs=nnx.Rngs(params=jax.random.key(1)))

        # Load latest checkpoint (should be step 10)
        loaded_model, loaded_step = load_checkpoint(manager, new_model)

        # Check loaded step
        assert loaded_step == 10

        # Verify the loaded model has the updated parameters
        assert jnp.allclose(loaded_model.dense1.kernel.value, w1)

    def test_load_specific_checkpoint(self, temp_dir, model, key):
        """Test loading a specific checkpoint step."""
        # Setup checkpoint manager
        manager, _ = setup_checkpoint_manager(temp_dir)

        # Initial model state
        x = jnp.ones((1, 10))
        y_initial = model(x)

        # Save model at step 0
        save_checkpoint(manager, model, 0)

        # Update model (simulate training)
        w1 = model.dense1.kernel.value * 2
        model.dense1.kernel.value = w1
        y_updated = model(x)

        # Save at step 10
        save_checkpoint(manager, model, 10)

        # Create a new model instance
        new_model = SimpleModel(rngs=nnx.Rngs(params=jax.random.key(1)))

        # Load specific checkpoint (step 0)
        loaded_model, loaded_step = load_checkpoint(manager, new_model, step=0)

        # Check loaded step
        assert loaded_step == 0

        # Verify the loaded model outputs match the initial model
        y_loaded = loaded_model(x)
        assert jnp.allclose(y_loaded, y_initial)
        assert not jnp.allclose(y_loaded, y_updated)

    def test_load_nonexistent_checkpoint(self, temp_dir, model):
        """Test loading a checkpoint that doesn't exist."""
        # Setup checkpoint manager
        manager, _ = setup_checkpoint_manager(temp_dir)

        # Try to load a checkpoint that doesn't exist
        loaded_model, loaded_step = load_checkpoint(manager, model)

        # Check that nothing was loaded
        assert loaded_model is None
        assert loaded_step is None

    def test_raw_state_loading(self, temp_dir, model, key):
        """Test loading checkpoint as raw state dictionary."""
        # Setup checkpoint manager
        manager, _ = setup_checkpoint_manager(temp_dir)

        # Save model
        save_checkpoint(manager, model, 0)

        # Load checkpoint as raw state
        state_dict, loaded_step = load_checkpoint(manager, target_model_template=None)

        # Check loaded step
        assert loaded_step == 0

        # Check that state_dict contains model parameters
        assert "dense1" in state_dict
        assert "dense2" in state_dict


class TestOptimizerCheckpointing:
    """Test cases for optimizer checkpointing utilities."""

    def test_save_checkpoint_with_optimizer_exists(self):
        """save_checkpoint_with_optimizer should be importable."""
        from artifex.generative_models.core.checkpointing import (
            save_checkpoint_with_optimizer,
        )

        assert save_checkpoint_with_optimizer is not None

    def test_load_checkpoint_with_optimizer_exists(self):
        """load_checkpoint_with_optimizer should be importable."""
        from artifex.generative_models.core.checkpointing import (
            load_checkpoint_with_optimizer,
        )

        assert load_checkpoint_with_optimizer is not None

    def test_save_and_load_checkpoint_with_optimizer(self, temp_dir, model):
        """Test saving and loading both model and optimizer state."""
        import optax

        from artifex.generative_models.core.checkpointing import (
            load_checkpoint_with_optimizer,
            save_checkpoint_with_optimizer,
            setup_checkpoint_manager,
        )

        # Setup checkpoint manager and optimizer
        manager, _ = setup_checkpoint_manager(temp_dir)
        optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

        # Initialize optimizer by doing a dummy update
        x = jnp.ones((1, 10))

        def loss_fn(model):
            return jnp.mean(model(x) ** 2)

        grads = nnx.grad(loss_fn)(model)
        optimizer.update(model, grads)

        # Get the original model output and optimizer step count
        original_output = model(x)

        # Save checkpoint with optimizer
        save_checkpoint_with_optimizer(manager, model, optimizer, step=1)

        # Create new model and optimizer with different initialization
        new_model = SimpleModel(rngs=nnx.Rngs(params=jax.random.key(42)))
        new_optimizer = nnx.Optimizer(new_model, optax.adam(1e-4), wrt=nnx.Param)

        # Verify different initialization
        new_output_before = new_model(x)
        assert not jnp.allclose(original_output, new_output_before)

        # Load checkpoint
        loaded_model, loaded_optimizer, loaded_step = load_checkpoint_with_optimizer(
            manager,
            new_model,
            new_optimizer,
            step=1,
        )

        # Verify step
        assert loaded_step == 1

        # Verify model outputs match
        loaded_output = loaded_model(x)
        assert jnp.allclose(original_output, loaded_output)

    def test_load_latest_optimizer_checkpoint(self, temp_dir, model):
        """Test loading the latest optimizer checkpoint when step is None."""
        import optax

        from artifex.generative_models.core.checkpointing import (
            load_checkpoint_with_optimizer,
            save_checkpoint_with_optimizer,
            setup_checkpoint_manager,
        )

        manager, _ = setup_checkpoint_manager(temp_dir)
        optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

        # Save at step 5
        save_checkpoint_with_optimizer(manager, model, optimizer, step=5)

        # Update model and save at step 10
        jnp.ones((1, 10))
        model.dense1.kernel.value = model.dense1.kernel.value * 2
        save_checkpoint_with_optimizer(manager, model, optimizer, step=10)

        # Create new model and optimizer
        new_model = SimpleModel(rngs=nnx.Rngs(params=jax.random.key(42)))
        new_optimizer = nnx.Optimizer(new_model, optax.adam(1e-4), wrt=nnx.Param)

        # Load latest checkpoint (step=None)
        loaded_model, loaded_optimizer, loaded_step = load_checkpoint_with_optimizer(
            manager,
            new_model,
            new_optimizer,
            step=None,
        )

        # Should load step 10
        assert loaded_step == 10

    def test_load_nonexistent_optimizer_checkpoint(self, temp_dir, model):
        """Test loading when no checkpoint exists."""
        import optax

        from artifex.generative_models.core.checkpointing import (
            load_checkpoint_with_optimizer,
            setup_checkpoint_manager,
        )

        manager, _ = setup_checkpoint_manager(temp_dir)
        optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

        # No checkpoint saved
        result = load_checkpoint_with_optimizer(manager, model, optimizer, step=None)

        assert result == (None, None, None)


class TestCheckpointValidation:
    """Test cases for checkpoint validation utilities."""

    def test_validate_checkpoint_exists(self):
        """validate_checkpoint should be importable."""
        from artifex.generative_models.core.checkpointing import validate_checkpoint

        assert validate_checkpoint is not None

    def test_validate_checkpoint_success(self, temp_dir, model):
        """Test validating a correct checkpoint."""
        from artifex.generative_models.core.checkpointing import (
            save_checkpoint,
            setup_checkpoint_manager,
            validate_checkpoint,
        )

        manager, _ = setup_checkpoint_manager(temp_dir)

        # Save checkpoint
        save_checkpoint(manager, model, step=1)

        # Validation data
        validation_data = jnp.ones((2, 10))

        # Validate checkpoint
        is_valid = validate_checkpoint(
            manager,
            model,
            step=1,
            validation_data=validation_data,
        )

        assert is_valid is True

    def test_validate_checkpoint_no_checkpoint(self, temp_dir, model):
        """Test validation when checkpoint doesn't exist."""
        from artifex.generative_models.core.checkpointing import (
            setup_checkpoint_manager,
            validate_checkpoint,
        )

        manager, _ = setup_checkpoint_manager(temp_dir)

        validation_data = jnp.ones((2, 10))

        # Validate non-existent checkpoint
        is_valid = validate_checkpoint(
            manager,
            model,
            step=999,
            validation_data=validation_data,
        )

        assert is_valid is False


class TestCorruptionRecovery:
    """Test cases for checkpoint corruption recovery."""

    def test_recover_from_corruption_exists(self):
        """recover_from_corruption should be importable."""
        from artifex.generative_models.core.checkpointing import recover_from_corruption

        assert recover_from_corruption is not None

    def test_recover_from_corruption_loads_latest(self, temp_dir, model):
        """Test recovering loads the latest valid checkpoint."""
        from artifex.generative_models.core.checkpointing import (
            recover_from_corruption,
            save_checkpoint,
            setup_checkpoint_manager,
        )

        manager, _ = setup_checkpoint_manager(temp_dir)

        # Save multiple checkpoints
        x = jnp.ones((1, 10))
        model(x)
        save_checkpoint(manager, model, step=5)

        # Update and save at step 10
        model.dense1.kernel.value = model.dense1.kernel.value * 2
        output_step_10 = model(x)
        save_checkpoint(manager, model, step=10)

        # Create new model for recovery
        new_model = SimpleModel(rngs=nnx.Rngs(params=jax.random.key(42)))

        # Recover
        recovered_model, recovered_step = recover_from_corruption(
            temp_dir,
            new_model,
        )

        # Should recover the latest checkpoint (step 10)
        assert recovered_step == 10
        assert jnp.allclose(recovered_model(x), output_step_10)

    def test_recover_from_corruption_no_checkpoints(self, temp_dir, model):
        """Test recovery when no checkpoints exist."""
        from artifex.generative_models.core.checkpointing import recover_from_corruption

        # No checkpoints saved
        recovered_model, recovered_step = recover_from_corruption(
            temp_dir,
            model,
        )

        assert recovered_model is None
        assert recovered_step is None

    def test_recover_from_corruption_with_model_factory(self, temp_dir, model):
        """Test recovery using a model factory function."""
        from artifex.generative_models.core.checkpointing import (
            recover_from_corruption,
            save_checkpoint,
            setup_checkpoint_manager,
        )

        manager, _ = setup_checkpoint_manager(temp_dir)

        # Save checkpoint
        x = jnp.ones((1, 10))
        original_output = model(x)
        save_checkpoint(manager, model, step=5)

        # Factory function to create fresh model
        def model_factory():
            return SimpleModel(rngs=nnx.Rngs(params=jax.random.key(99)))

        # Recover using factory
        recovered_model, recovered_step = recover_from_corruption(
            temp_dir,
            model_factory(),
        )

        assert recovered_step == 5
        assert jnp.allclose(recovered_model(x), original_output)
