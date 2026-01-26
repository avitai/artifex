"""TDD tests for Autoregressive model trainer.

These tests define the expected behavior for autoregressive model training including:
- Teacher forcing training
- Scheduled sampling for reducing exposure bias
- Label smoothing for regularization
- Causal masking utilities
- Text generation with various sampling strategies

References:
    - Teacher Forcing: Williams & Zipser, 1989
    - Scheduled Sampling: https://arxiv.org/abs/1506.03099
    - Label Smoothing: https://arxiv.org/abs/1512.00567
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleAutoregressiveModel(nnx.Module):
    """Simple autoregressive model for testing trainer functionality.

    A minimal transformer-like model that returns logits for next token prediction.
    """

    def __init__(self, vocab_size: int = 100, hidden_size: int = 32, *, rngs: nnx.Rngs):
        super().__init__()
        self.embed = nnx.Embed(num_embeddings=vocab_size, features=hidden_size, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.output = nnx.Linear(hidden_size, vocab_size, rngs=rngs)
        self.vocab_size = vocab_size

    def __call__(self, input_ids: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """Forward pass returning logits.

        Args:
            input_ids: Token IDs, shape (batch, seq_length).
            mask: Optional attention mask (unused in simple model).

        Returns:
            Logits of shape (batch, seq_length, vocab_size).
        """
        del mask  # Unused in simple model
        x = self.embed(input_ids)
        x = nnx.relu(self.linear(x))
        logits = self.output(x)
        return logits


@pytest.fixture
def simple_autoregressive_model() -> SimpleAutoregressiveModel:
    """Create a simple autoregressive model for testing."""
    return SimpleAutoregressiveModel(vocab_size=100, hidden_size=32, rngs=nnx.Rngs(0))


@pytest.fixture
def sample_batch() -> dict[str, jax.Array]:
    """Create a sample batch of token IDs for testing."""
    # Random token IDs in valid range
    return {"input_ids": jax.random.randint(jax.random.key(0), (4, 16), 0, 100)}


@pytest.fixture
def sample_key() -> jax.Array:
    """Create a sample PRNG key for testing."""
    return jax.random.key(42)


# =============================================================================
# AutoregressiveTrainingConfig Tests
# =============================================================================


class TestAutoregressiveTrainingConfig:
    """Tests for AutoregressiveTrainingConfig dataclass."""

    def test_config_exists(self) -> None:
        """AutoregressiveTrainingConfig should be importable."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainingConfig,
        )

        assert AutoregressiveTrainingConfig is not None

    def test_config_default_values(self) -> None:
        """AutoregressiveTrainingConfig should have sensible defaults."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        assert config.use_teacher_forcing is True
        assert config.scheduled_sampling == "none"
        assert config.sampling_warmup_steps == 0
        assert config.sampling_decay_steps == 10000
        assert config.min_teacher_forcing_prob == 0.0
        assert config.label_smoothing == 0.0
        assert config.use_causal_mask is True
        assert config.pad_token_id is None
        assert config.ignore_index == -100

    def test_config_custom_values(self) -> None:
        """AutoregressiveTrainingConfig should accept custom values."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig(
            use_teacher_forcing=True,
            scheduled_sampling="linear",
            sampling_warmup_steps=1000,
            sampling_decay_steps=5000,
            min_teacher_forcing_prob=0.1,
            label_smoothing=0.1,
            use_causal_mask=True,
            pad_token_id=0,
            ignore_index=-100,
        )
        assert config.scheduled_sampling == "linear"
        assert config.sampling_warmup_steps == 1000
        assert config.sampling_decay_steps == 5000
        assert config.min_teacher_forcing_prob == 0.1
        assert config.label_smoothing == 0.1
        assert config.pad_token_id == 0

    def test_config_all_sampling_schedules(self) -> None:
        """AutoregressiveTrainingConfig should support all sampling schedules."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainingConfig,
        )

        for schedule in ["none", "linear", "exponential", "inverse_sigmoid"]:
            config = AutoregressiveTrainingConfig(scheduled_sampling=schedule)
            assert config.scheduled_sampling == schedule


# =============================================================================
# Causal Mask Tests
# =============================================================================


class TestCausalMask:
    """Tests for causal masking utilities."""

    def test_create_causal_mask_shape(self) -> None:
        """create_causal_mask should return correct shape."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            create_causal_mask,
        )

        mask = create_causal_mask(8)
        assert mask.shape == (8, 8)

    def test_create_causal_mask_lower_triangular(self) -> None:
        """create_causal_mask should be lower triangular."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            create_causal_mask,
        )

        mask = create_causal_mask(4)

        # Check it's lower triangular (True on and below diagonal)
        expected = jnp.array(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        assert jnp.array_equal(mask, expected)

    def test_create_padding_mask(self) -> None:
        """create_padding_mask should mask padding tokens."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            create_padding_mask,
        )

        tokens = jnp.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])  # 0 is padding
        mask = create_padding_mask(tokens, pad_token_id=0)

        expected = jnp.array(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ]
        )
        assert jnp.array_equal(mask, expected)

    def test_create_combined_mask(self) -> None:
        """create_combined_mask should combine causal and padding masks."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            create_combined_mask,
        )

        tokens = jnp.array([[1, 2, 3, 0]])  # 0 is padding
        mask = create_combined_mask(tokens, pad_token_id=0)

        # Shape should be (batch, 1, seq, seq) for broadcasting
        assert mask.shape == (1, 1, 4, 4)


# =============================================================================
# Scheduled Sampling Tests
# =============================================================================


class TestScheduledSampling:
    """Tests for scheduled sampling schedules."""

    def test_no_sampling_returns_one(self) -> None:
        """No scheduled sampling should always return 1.0."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig(scheduled_sampling="none")
        trainer = AutoregressiveTrainer(config)

        assert trainer.get_teacher_forcing_prob(0) == 1.0
        assert trainer.get_teacher_forcing_prob(10000) == 1.0
        assert trainer.get_teacher_forcing_prob(100000) == 1.0

    def test_linear_sampling_warmup(self) -> None:
        """Linear sampling should respect warmup period."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig(
            scheduled_sampling="linear",
            sampling_warmup_steps=100,
            sampling_decay_steps=1000,
        )
        trainer = AutoregressiveTrainer(config)

        # During warmup, should return 1.0
        assert trainer.get_teacher_forcing_prob(0) == 1.0
        assert trainer.get_teacher_forcing_prob(50) == 1.0
        assert trainer.get_teacher_forcing_prob(99) == 1.0

    def test_linear_sampling_decay(self) -> None:
        """Linear sampling should decay linearly after warmup."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig(
            scheduled_sampling="linear",
            sampling_warmup_steps=0,
            sampling_decay_steps=100,
            min_teacher_forcing_prob=0.0,
        )
        trainer = AutoregressiveTrainer(config)

        # Should decay linearly
        assert trainer.get_teacher_forcing_prob(0) == pytest.approx(1.0)
        assert trainer.get_teacher_forcing_prob(50) == pytest.approx(0.5)
        assert trainer.get_teacher_forcing_prob(100) == pytest.approx(0.0)

    def test_exponential_sampling_decay(self) -> None:
        """Exponential sampling should decay exponentially."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig(
            scheduled_sampling="exponential",
            sampling_warmup_steps=0,
            sampling_decay_steps=100,
            min_teacher_forcing_prob=0.01,
        )
        trainer = AutoregressiveTrainer(config)

        # Should decay exponentially
        prob_start = trainer.get_teacher_forcing_prob(0)
        prob_mid = trainer.get_teacher_forcing_prob(50)
        prob_end = trainer.get_teacher_forcing_prob(100)

        assert prob_start > prob_mid > prob_end
        assert prob_end >= config.min_teacher_forcing_prob

    def test_inverse_sigmoid_sampling(self) -> None:
        """Inverse sigmoid sampling should have smooth S-shaped decay."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig(
            scheduled_sampling="inverse_sigmoid",
            sampling_warmup_steps=0,
            sampling_decay_steps=100,
            min_teacher_forcing_prob=0.0,
        )
        trainer = AutoregressiveTrainer(config)

        # Should decay smoothly
        prob_start = trainer.get_teacher_forcing_prob(0)
        prob_end = trainer.get_teacher_forcing_prob(100)

        assert prob_start > prob_end

    def test_min_teacher_forcing_prob_respected(self) -> None:
        """Minimum teacher forcing probability should be respected."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig(
            scheduled_sampling="linear",
            sampling_warmup_steps=0,
            sampling_decay_steps=100,
            min_teacher_forcing_prob=0.2,
        )
        trainer = AutoregressiveTrainer(config)

        # Even after decay, should not go below min
        prob = trainer.get_teacher_forcing_prob(200)
        assert prob >= 0.2


# =============================================================================
# Loss Computation Tests
# =============================================================================


class TestAutoregressiveLossComputation:
    """Tests for autoregressive loss computation."""

    def test_compute_loss_shape(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_batch: dict,
    ) -> None:
        """compute_loss should return scalar loss."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        loss, _ = trainer.compute_loss(simple_autoregressive_model, sample_batch, step=0)

        assert loss.shape == ()
        assert isinstance(loss, jax.Array)

    def test_compute_loss_metrics(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_batch: dict,
    ) -> None:
        """compute_loss should return expected metrics."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        _, metrics = trainer.compute_loss(simple_autoregressive_model, sample_batch, step=0)

        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "accuracy" in metrics
        assert "teacher_forcing_prob" in metrics
        assert "num_tokens" in metrics

    def test_perplexity_is_exp_loss(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_batch: dict,
    ) -> None:
        """Perplexity should be exp(loss)."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        loss, metrics = trainer.compute_loss(simple_autoregressive_model, sample_batch, step=0)

        expected_ppl = jnp.exp(loss)
        assert metrics["perplexity"] == pytest.approx(float(expected_ppl), rel=1e-5)

    def test_label_smoothing_reduces_loss_variance(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_batch: dict,
    ) -> None:
        """Label smoothing should produce different loss than without."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        # Without smoothing
        config_no_smooth = AutoregressiveTrainingConfig(label_smoothing=0.0)
        trainer_no_smooth = AutoregressiveTrainer(config_no_smooth)

        # With smoothing
        config_smooth = AutoregressiveTrainingConfig(label_smoothing=0.1)
        trainer_smooth = AutoregressiveTrainer(config_smooth)

        loss_no_smooth, _ = trainer_no_smooth.compute_loss(
            simple_autoregressive_model, sample_batch, step=0
        )
        loss_smooth, _ = trainer_smooth.compute_loss(
            simple_autoregressive_model, sample_batch, step=0
        )

        # Losses should be different
        assert not jnp.allclose(loss_no_smooth, loss_smooth)


# =============================================================================
# Training Step Tests
# =============================================================================


class TestAutoregressiveTrainStep:
    """Tests for autoregressive model training step."""

    def test_train_step_updates_model(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_batch: dict,
    ) -> None:
        """train_step should update model parameters."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        optimizer = nnx.Optimizer(simple_autoregressive_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = AutoregressiveTrainer(config)

        # Get initial params
        initial_params = nnx.state(simple_autoregressive_model, nnx.Param)
        initial_kernel = initial_params["linear"]["kernel"].value.copy()

        # Run train step
        trainer.train_step(simple_autoregressive_model, optimizer, sample_batch, step=0)

        # Get updated params
        updated_params = nnx.state(simple_autoregressive_model, nnx.Param)
        updated_kernel = updated_params["linear"]["kernel"].value

        # Params should have changed
        assert not jnp.allclose(initial_kernel, updated_kernel)

    def test_train_step_returns_metrics(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_batch: dict,
    ) -> None:
        """train_step should return loss and metrics."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        optimizer = nnx.Optimizer(simple_autoregressive_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = AutoregressiveTrainer(config)

        loss, metrics = trainer.train_step(
            simple_autoregressive_model, optimizer, sample_batch, step=0
        )

        assert isinstance(loss, jax.Array)
        assert "perplexity" in metrics
        assert "accuracy" in metrics


# =============================================================================
# Generation Tests
# =============================================================================


class TestAutoregressiveGeneration:
    """Tests for text generation from autoregressive models."""

    def test_generate_shape(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_key: jax.Array,
    ) -> None:
        """generate should return correct shape."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        prompt = jnp.array([[1, 2, 3], [4, 5, 6]])  # batch=2, prompt_len=3
        generated = trainer.generate(
            simple_autoregressive_model,
            prompt=prompt,
            max_length=10,
            key=sample_key,
        )

        assert generated.shape == (2, 10)

    def test_generate_preserves_prompt(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_key: jax.Array,
    ) -> None:
        """generate should preserve the prompt at the beginning."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        prompt = jnp.array([[1, 2, 3]])
        generated = trainer.generate(
            simple_autoregressive_model,
            prompt=prompt,
            max_length=8,
            key=sample_key,
        )

        # First 3 tokens should be the prompt
        assert jnp.array_equal(generated[0, :3], prompt[0])

    def test_generate_with_temperature(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_key: jax.Array,
    ) -> None:
        """generate should accept temperature parameter."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        prompt = jnp.array([[1, 2, 3]])

        # Different temperatures should produce results without error
        gen_low = trainer.generate(
            simple_autoregressive_model,
            prompt=prompt,
            max_length=8,
            key=sample_key,
            temperature=0.5,
        )
        gen_high = trainer.generate(
            simple_autoregressive_model,
            prompt=prompt,
            max_length=8,
            key=sample_key,
            temperature=2.0,
        )

        assert gen_low.shape == (1, 8)
        assert gen_high.shape == (1, 8)

    def test_generate_with_top_k(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_key: jax.Array,
    ) -> None:
        """generate should support top-k sampling."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        prompt = jnp.array([[1, 2, 3]])
        generated = trainer.generate(
            simple_autoregressive_model,
            prompt=prompt,
            max_length=8,
            key=sample_key,
            top_k=10,
        )

        assert generated.shape == (1, 8)

    def test_generate_with_top_p(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
        sample_key: jax.Array,
    ) -> None:
        """generate should support nucleus (top-p) sampling."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        prompt = jnp.array([[1, 2, 3]])
        generated = trainer.generate(
            simple_autoregressive_model,
            prompt=prompt,
            max_length=8,
            key=sample_key,
            top_p=0.9,
        )

        assert generated.shape == (1, 8)


# =============================================================================
# DRY Integration Tests
# =============================================================================


class TestAutoregressiveLossFunctionIntegration:
    """Tests for using Autoregressive loss with base Trainer."""

    def test_create_loss_fn_for_base_trainer(
        self,
        simple_autoregressive_model: SimpleAutoregressiveModel,
    ) -> None:
        """AutoregressiveTrainer should provide loss_fn compatible with base Trainer."""
        from artifex.generative_models.training.trainers.autoregressive_trainer import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig()
        trainer = AutoregressiveTrainer(config)

        # Should be able to create a loss function for the base Trainer
        loss_fn = trainer.create_loss_fn(step=100)

        # Loss function should have correct signature: (model, batch, rng) -> (loss, metrics)
        batch = {"input_ids": jax.random.randint(jax.random.key(0), (4, 16), 0, 100)}
        rng = jax.random.key(42)
        loss, metrics = loss_fn(simple_autoregressive_model, batch, rng)

        assert isinstance(loss, jax.Array)
        assert "perplexity" in metrics


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestAutoregressiveTrainerExports:
    """Tests for Autoregressive trainer exports."""

    def test_exports_from_trainers_init(self) -> None:
        """Autoregressive trainer classes should be exported from trainers __init__."""
        from artifex.generative_models.training.trainers import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
            create_causal_mask,
            create_combined_mask,
            create_padding_mask,
        )

        assert AutoregressiveTrainer is not None
        assert AutoregressiveTrainingConfig is not None
        assert create_causal_mask is not None
        assert create_padding_mask is not None
        assert create_combined_mask is not None
