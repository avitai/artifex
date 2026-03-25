"""TDD tests for RL training components.

Tests for REINFORCE, PPO, GRPO, and DPO trainers following test-driven development.
Tests are written first to define expected behavior before implementation.

State-of-the-art techniques incorporated:
- GRPO (Group Relative Policy Optimization): DeepSeek's critic-free RL algorithm
  that normalizes rewards within groups, eliminating the need for a value network.
- SimPO-style reference-free DPO: Eliminates reference model requirement.
- Modern variance reduction baselines for REINFORCE.
- PPO with GAE (Generalized Advantage Estimation).

References:
- GRPO: https://arxiv.org/abs/2402.03300 (DeepSeek-R1)
- DPO: https://arxiv.org/abs/2305.18290
- SimPO: https://github.com/princeton-nlp/SimPO
- PPO: https://arxiv.org/abs/1707.06347
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


VOCAB_SIZE = 32
SEQ_LEN = 6
PROMPT_LEN = 2


def make_token_sequences(
    batch_size: int,
    *,
    seq_len: int = SEQ_LEN,
    offset: int = 0,
) -> jax.Array:
    """Create deterministic token sequences for autoregressive rollout tests."""
    tokens = jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape(batch_size, seq_len)
    return (tokens + offset) % VOCAB_SIZE


def make_sequence_mask(
    batch_size: int,
    *,
    seq_len: int = SEQ_LEN,
    prompt_len: int = PROMPT_LEN,
) -> jax.Array:
    """Create a prompt/response split mask with 1s over response tokens."""
    return jnp.concatenate(
        [
            jnp.zeros((batch_size, prompt_len), dtype=jnp.float32),
            jnp.ones((batch_size, seq_len - prompt_len), dtype=jnp.float32),
        ],
        axis=1,
    )


def make_sequence_rollout_batch(
    batch_size: int,
    *,
    seq_len: int = SEQ_LEN,
    prompt_len: int = PROMPT_LEN,
    sequence_offset: int = 0,
    old_log_probs: jax.Array | None = None,
    token_rewards: jax.Array | None = None,
    returns: jax.Array | None = None,
    advantages: jax.Array | None = None,
    dones: jax.Array | None = None,
    sequence_rewards: jax.Array | None = None,
):
    """Construct a typed sequence rollout batch for policy-gradient trainers."""
    from artifex.generative_models.training.rl import (
        GeneratedBatch,
        GeneratedSequenceBatch,
        SequenceRolloutBatch,
    )

    sequences = make_token_sequences(batch_size, seq_len=seq_len, offset=sequence_offset)
    response_mask = make_sequence_mask(batch_size, seq_len=seq_len, prompt_len=prompt_len)
    prompt_mask = 1.0 - response_mask

    return SequenceRolloutBatch(
        sequence_batch=GeneratedSequenceBatch(
            generation=GeneratedBatch(
                outputs=sequences,
                rewards=sequence_rewards,
            ),
            prompt_mask=prompt_mask,
            response_mask=response_mask,
        ),
        old_log_probs=old_log_probs,
        token_rewards=token_rewards,
        returns=returns,
        advantages=advantages,
        dones=dones,
    )


def make_group_rollout_batch(
    batch_size: int,
    *,
    group_size: int,
    seq_len: int = SEQ_LEN,
    prompt_len: int = PROMPT_LEN,
    old_log_probs: jax.Array | None = None,
    token_rewards: jax.Array | None = None,
    returns: jax.Array | None = None,
    advantages: jax.Array | None = None,
    dones: jax.Array | None = None,
    sequence_rewards: jax.Array | None = None,
):
    """Construct a grouped typed rollout batch for GRPO."""
    from artifex.generative_models.training.rl import GroupRolloutBatch

    rollout = make_sequence_rollout_batch(
        batch_size,
        seq_len=seq_len,
        prompt_len=prompt_len,
        old_log_probs=old_log_probs,
        token_rewards=token_rewards,
        returns=returns,
        advantages=advantages,
        dones=dones,
        sequence_rewards=sequence_rewards,
    )
    return GroupRolloutBatch(rollout=rollout, group_size=group_size)


def assert_finite_nnx_grads(grads: nnx.Module) -> None:
    """Assert that an NNX gradient tree is finite and non-empty."""
    grad_state = nnx.state(grads)
    grad_leaves = jax.tree_util.tree_leaves(grad_state)

    assert grad_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)
    assert any(not jnp.allclose(leaf, 0.0) for leaf in grad_leaves)


class SimpleSequencePolicy(nnx.Module):
    """Simple autoregressive policy network for sequence RL tests."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.embedding = nnx.Embed(vocab_size, hidden_dim, rngs=rngs)
        self.dense = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.output = nnx.Linear(hidden_dim, vocab_size, rngs=rngs)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """Forward pass returning per-token logits."""
        embeddings = self.embedding(tokens)
        hidden = nnx.relu(self.dense(embeddings))
        return self.output(hidden)


class SimpleSequenceActorCritic(nnx.Module):
    """Autoregressive actor-critic network with a per-token value head."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.embedding = nnx.Embed(vocab_size, hidden_dim, rngs=rngs)
        self.backbone = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.actor = nnx.Linear(hidden_dim, vocab_size, rngs=rngs)
        self.critic = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, tokens: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass returning autoregressive logits and per-token values."""
        embeddings = self.embedding(tokens)
        features = nnx.relu(self.backbone(embeddings))
        action_logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return action_logits, value


class SimplePreferenceModel(nnx.Module):
    """Simple model for DPO testing (language model style)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.embedding = nnx.Embed(vocab_size, hidden_dim, rngs=rngs)
        self.dense = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.output = nnx.Linear(hidden_dim, vocab_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass returning per-token logits over vocabulary.

        Returns shape (batch_size, seq_len, vocab_size) for autoregressive
        log-probability computation.
        """
        emb = self.embedding(x)
        h = nnx.relu(self.dense(emb))
        return self.output(h)


class DictOutputPreferenceModel(nnx.Module):
    """Preference model variant returning a dict with ``logits``."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.embedding = nnx.Embed(vocab_size, hidden_dim, rngs=rngs)
        self.dense = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.output = nnx.Linear(hidden_dim, vocab_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> dict[str, jax.Array]:
        """Forward pass returning an autoregressive-style output dict."""
        emb = self.embedding(x)
        h = nnx.relu(self.dense(emb))
        return {"logits": self.output(h)}


class StrictMaskPreferenceModel(nnx.Module):
    """Preference model that requires a mask kwarg during forward passes."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.embedding = nnx.Embed(vocab_size, hidden_dim, rngs=rngs)
        self.dense = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.output = nnx.Linear(hidden_dim, vocab_size, rngs=rngs)

    def __call__(self, x: jax.Array, *, mask: jax.Array) -> dict[str, jax.Array]:
        """Forward pass requiring mask-aware kwargs from the adapter."""
        emb = self.embedding(x)
        h = nnx.relu(self.dense(emb))
        logits = self.output(h)
        return {"logits": logits + mask[:, :, None].astype(logits.dtype) * 0.0}


@pytest.fixture
def policy_model() -> SimpleSequencePolicy:
    """Create a simple autoregressive policy model for testing."""
    rngs = nnx.Rngs(0)
    return SimpleSequencePolicy(vocab_size=VOCAB_SIZE, hidden_dim=16, rngs=rngs)


@pytest.fixture
def actor_critic_model() -> SimpleSequenceActorCritic:
    """Create a simple autoregressive actor-critic model for testing."""
    rngs = nnx.Rngs(0)
    return SimpleSequenceActorCritic(vocab_size=VOCAB_SIZE, hidden_dim=16, rngs=rngs)


@pytest.fixture
def preference_model() -> SimplePreferenceModel:
    """Create a simple preference model for DPO testing."""
    rngs = nnx.Rngs(0)
    return SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=rngs)


@pytest.fixture
def dict_preference_model() -> DictOutputPreferenceModel:
    """Create a dict-output preference model for sequence adapter tests."""
    rngs = nnx.Rngs(2)
    return DictOutputPreferenceModel(vocab_size=100, hidden_dim=32, rngs=rngs)


# =============================================================================
# REINFORCE Configuration Tests
# =============================================================================


class TestREINFORCEConfig:
    """Tests for REINFORCE configuration."""

    def test_reinforce_config_exists(self) -> None:
        """REINFORCEConfig class should exist."""
        from artifex.generative_models.training.rl import REINFORCEConfig

        assert REINFORCEConfig is not None

    def test_reinforce_config_default_values(self) -> None:
        """REINFORCEConfig should have sensible defaults."""
        from artifex.generative_models.training.rl import REINFORCEConfig

        config = REINFORCEConfig()
        assert hasattr(config, "gamma")
        assert hasattr(config, "normalize_returns")
        assert hasattr(config, "entropy_coeff")

    def test_reinforce_config_gamma_default(self) -> None:
        """Default gamma should be 0.99."""
        from artifex.generative_models.training.rl import REINFORCEConfig

        config = REINFORCEConfig()
        assert config.gamma == 0.99

    def test_reinforce_config_normalize_returns_default(self) -> None:
        """Default normalize_returns should be True."""
        from artifex.generative_models.training.rl import REINFORCEConfig

        config = REINFORCEConfig()
        assert config.normalize_returns is True

    def test_reinforce_config_entropy_coeff_default(self) -> None:
        """Default entropy_coeff should be 0.01."""
        from artifex.generative_models.training.rl import REINFORCEConfig

        config = REINFORCEConfig()
        assert config.entropy_coeff == 0.01

    def test_reinforce_config_custom_values(self) -> None:
        """REINFORCEConfig should accept custom values."""
        from artifex.generative_models.training.rl import REINFORCEConfig

        config = REINFORCEConfig(
            gamma=0.95,
            normalize_returns=False,
            entropy_coeff=0.1,
        )
        assert config.gamma == 0.95
        assert config.normalize_returns is False
        assert config.entropy_coeff == 0.1


# =============================================================================
# REINFORCE Trainer Tests
# =============================================================================


class TestREINFORCETrainer:
    """Tests for REINFORCE trainer."""

    def test_reinforce_trainer_exists(self) -> None:
        """REINFORCETrainer class should exist."""
        from artifex.generative_models.training.rl import REINFORCETrainer

        assert REINFORCETrainer is not None

    def test_reinforce_trainer_initialization(self, policy_model: SimpleSequencePolicy) -> None:
        """REINFORCETrainer should initialize with model and optimizer."""
        from artifex.generative_models.training.rl import (
            REINFORCEConfig,
            REINFORCETrainer,
        )

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = REINFORCEConfig()
        trainer = REINFORCETrainer(policy_model, optimizer, config)

        assert trainer.model is policy_model
        assert trainer.optimizer is optimizer
        assert trainer.config is config

    def test_reinforce_trainer_default_config(self, policy_model: SimpleSequencePolicy) -> None:
        """REINFORCETrainer should use default config if not provided."""
        from artifex.generative_models.training.rl import (
            REINFORCEConfig,
            REINFORCETrainer,
        )

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = REINFORCETrainer(policy_model, optimizer)

        assert isinstance(trainer.config, REINFORCEConfig)

    def test_reinforce_compute_returns(self, policy_model: SimpleSequencePolicy) -> None:
        """REINFORCETrainer should compute discounted returns correctly."""
        from artifex.generative_models.training.rl import (
            REINFORCEConfig,
            REINFORCETrainer,
        )

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = REINFORCEConfig(gamma=0.99)
        trainer = REINFORCETrainer(policy_model, optimizer, config)

        rewards = jnp.array([1.0, 1.0, 1.0])
        returns = trainer.compute_returns(rewards)

        # Expected: [1 + 0.99 + 0.99^2, 1 + 0.99, 1]
        expected = jnp.array([1 + 0.99 + 0.99**2, 1 + 0.99, 1.0])
        assert jnp.allclose(returns, expected, atol=1e-5)

    def test_reinforce_compute_returns_with_zeros(self, policy_model: SimpleSequencePolicy) -> None:
        """REINFORCETrainer should handle zero rewards."""
        from artifex.generative_models.training.rl import (
            REINFORCEConfig,
            REINFORCETrainer,
        )

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = REINFORCEConfig(gamma=0.99)
        trainer = REINFORCETrainer(policy_model, optimizer, config)

        rewards = jnp.array([0.0, 0.0, 1.0])
        returns = trainer.compute_returns(rewards)

        expected = jnp.array([0.99**2, 0.99, 1.0])
        assert jnp.allclose(returns, expected, atol=1e-5)

    def test_reinforce_normalize_returns(self, policy_model: SimpleSequencePolicy) -> None:
        """REINFORCETrainer should normalize returns when configured."""
        from artifex.generative_models.training.rl import (
            REINFORCEConfig,
            REINFORCETrainer,
        )

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = REINFORCEConfig(normalize_returns=True)
        trainer = REINFORCETrainer(policy_model, optimizer, config)

        returns = jnp.array([1.0, 2.0, 3.0])
        normalized = trainer.normalize_returns(returns)

        # Should have mean ~0 and std ~1
        assert jnp.abs(jnp.mean(normalized)) < 0.1
        assert jnp.abs(jnp.std(normalized) - 1.0) < 0.1

    def test_reinforce_policy_loss(self, policy_model: SimpleSequencePolicy) -> None:
        """REINFORCETrainer should compute policy loss from typed sequence rollouts."""
        from artifex.generative_models.training.rl import (
            REINFORCEConfig,
            REINFORCETrainer,
        )

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = REINFORCEConfig()
        trainer = REINFORCETrainer(policy_model, optimizer, config)

        batch = make_sequence_rollout_batch(
            4,
            returns=jnp.array(
                [
                    [1.0, 0.5, 0.25, 0.1, 0.0],
                    [0.9, 0.45, 0.2, 0.05, 0.0],
                    [0.8, 0.4, 0.15, 0.05, 0.0],
                    [0.7, 0.35, 0.1, 0.05, 0.0],
                ],
                dtype=jnp.float32,
            ),
        )

        loss, metrics = trainer.compute_loss(batch)

        assert jnp.isfinite(loss)
        assert "policy_loss" in metrics
        assert "entropy" in metrics

    def test_reinforce_train_step(self, policy_model: SimpleSequencePolicy) -> None:
        """REINFORCETrainer should perform a training step on sequence rollouts."""
        from artifex.generative_models.training.rl import (
            REINFORCEConfig,
            REINFORCETrainer,
        )

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = REINFORCEConfig()
        trainer = REINFORCETrainer(policy_model, optimizer, config)

        batch = make_sequence_rollout_batch(
            4,
            token_rewards=jnp.array(
                [
                    [0.0, 0.0, 1.0, 0.5, 0.0],
                    [0.0, 0.0, 0.8, 0.4, 0.0],
                    [0.0, 0.0, 0.6, 0.3, 0.0],
                    [0.0, 0.0, 0.4, 0.2, 0.0],
                ],
                dtype=jnp.float32,
            ),
        )

        loss, metrics = trainer.train_step(batch)

        assert jnp.isfinite(loss)
        assert "policy_loss" in metrics


# =============================================================================
# PPO Configuration Tests
# =============================================================================


class TestPPOConfig:
    """Tests for PPO configuration."""

    def test_ppo_config_exists(self) -> None:
        """PPOConfig class should exist."""
        from artifex.generative_models.training.rl import PPOConfig

        assert PPOConfig is not None

    def test_ppo_config_default_values(self) -> None:
        """PPOConfig should have sensible defaults."""
        from artifex.generative_models.training.rl import PPOConfig

        config = PPOConfig()
        assert hasattr(config, "gamma")
        assert hasattr(config, "gae_lambda")
        assert hasattr(config, "clip_param")
        assert hasattr(config, "vf_coeff")
        assert hasattr(config, "entropy_coeff")
        assert hasattr(config, "max_grad_norm")

    def test_ppo_config_gamma_default(self) -> None:
        """Default gamma should be 0.99."""
        from artifex.generative_models.training.rl import PPOConfig

        config = PPOConfig()
        assert config.gamma == 0.99

    def test_ppo_config_gae_lambda_default(self) -> None:
        """Default gae_lambda should be 0.95."""
        from artifex.generative_models.training.rl import PPOConfig

        config = PPOConfig()
        assert config.gae_lambda == 0.95

    def test_ppo_config_clip_param_default(self) -> None:
        """Default clip_param should be 0.2."""
        from artifex.generative_models.training.rl import PPOConfig

        config = PPOConfig()
        assert config.clip_param == 0.2

    def test_ppo_config_vf_coeff_default(self) -> None:
        """Default vf_coeff should be 0.5."""
        from artifex.generative_models.training.rl import PPOConfig

        config = PPOConfig()
        assert config.vf_coeff == 0.5

    def test_ppo_config_entropy_coeff_default(self) -> None:
        """Default entropy_coeff should be 0.01."""
        from artifex.generative_models.training.rl import PPOConfig

        config = PPOConfig()
        assert config.entropy_coeff == 0.01

    def test_ppo_config_max_grad_norm_default(self) -> None:
        """Default max_grad_norm should be 0.5."""
        from artifex.generative_models.training.rl import PPOConfig

        config = PPOConfig()
        assert config.max_grad_norm == 0.5


# =============================================================================
# PPO Trainer Tests
# =============================================================================


class TestPPOTrainer:
    """Tests for PPO trainer."""

    def test_ppo_trainer_exists(self) -> None:
        """PPOTrainer class should exist."""
        from artifex.generative_models.training.rl import PPOTrainer

        assert PPOTrainer is not None

    def test_ppo_trainer_initialization(
        self, actor_critic_model: SimpleSequenceActorCritic
    ) -> None:
        """PPOTrainer should initialize with model and optimizer."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        config = PPOConfig()
        trainer = PPOTrainer(actor_critic_model, optimizer, config)

        assert trainer.model is actor_critic_model
        assert trainer.optimizer is optimizer
        assert trainer.config is config

    def test_ppo_trainer_default_config(
        self, actor_critic_model: SimpleSequenceActorCritic
    ) -> None:
        """PPOTrainer should use default config if not provided."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = PPOTrainer(actor_critic_model, optimizer)

        assert isinstance(trainer.config, PPOConfig)

    def test_ppo_compute_gae(self, actor_critic_model: SimpleSequenceActorCritic) -> None:
        """PPOTrainer should compute GAE advantages correctly."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        config = PPOConfig(gamma=0.99, gae_lambda=0.95)
        trainer = PPOTrainer(actor_critic_model, optimizer, config)

        rewards = jnp.array([1.0, 1.0, 1.0, 1.0])
        values = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0])  # Including next value
        dones = jnp.array([False, False, False, True])

        advantages = trainer.compute_gae(rewards, values, dones)

        assert advantages.shape == (4,)
        assert jnp.all(jnp.isfinite(advantages))

    def test_ppo_clipped_surrogate_loss(
        self, actor_critic_model: SimpleSequenceActorCritic
    ) -> None:
        """PPOTrainer should compute clipped surrogate loss."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        config = PPOConfig(clip_param=0.2)
        trainer = PPOTrainer(actor_critic_model, optimizer, config)

        # Create fake data
        log_probs = jnp.array([-0.5, -1.0, -1.5, -2.0])
        old_log_probs = jnp.array([-0.6, -1.1, -1.4, -1.9])
        advantages = jnp.array([1.0, -0.5, 0.5, -1.0])

        loss = trainer.compute_clipped_loss(log_probs, old_log_probs, advantages)

        assert jnp.isfinite(loss)

    def test_ppo_value_loss(self, actor_critic_model: SimpleSequenceActorCritic) -> None:
        """PPOTrainer should compute value function loss."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        config = PPOConfig()
        trainer = PPOTrainer(actor_critic_model, optimizer, config)

        values = jnp.array([0.5, 1.0, 1.5, 2.0])
        returns = jnp.array([0.6, 0.9, 1.6, 2.1])

        loss = trainer.compute_value_loss(values, returns)

        assert jnp.isfinite(loss)
        assert loss > 0  # MSE loss should be positive

    def test_ppo_entropy_bonus(self, actor_critic_model: SimpleSequenceActorCritic) -> None:
        """PPOTrainer should compute entropy bonus."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        config = PPOConfig()
        trainer = PPOTrainer(actor_critic_model, optimizer, config)

        # Uniform distribution has high entropy
        log_probs = jnp.log(jnp.ones((4, 4)) / 4)

        entropy = trainer.compute_entropy(log_probs)

        assert jnp.isfinite(entropy)
        assert entropy > 0  # Entropy should be positive

    def test_ppo_train_step(self, actor_critic_model: SimpleSequenceActorCritic) -> None:
        """PPOTrainer should perform a training step on typed sequence rollouts."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        config = PPOConfig()
        trainer = PPOTrainer(actor_critic_model, optimizer, config)

        batch = make_sequence_rollout_batch(
            4,
            old_log_probs=jnp.full((4, SEQ_LEN - 1), -0.75, dtype=jnp.float32),
            returns=jnp.full((4, SEQ_LEN - 1), 0.5, dtype=jnp.float32),
            advantages=jnp.array(
                [
                    [0.0, 0.0, 0.8, 0.4, 0.1],
                    [0.0, 0.0, 0.7, 0.35, 0.1],
                    [0.0, 0.0, 0.6, 0.3, 0.1],
                    [0.0, 0.0, 0.5, 0.25, 0.1],
                ],
                dtype=jnp.float32,
            ),
        )

        loss, metrics = trainer.train_step(batch)

        assert jnp.isfinite(loss)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_ppo_train_step_respects_max_grad_norm(self) -> None:
        """Smaller clipping thresholds should produce smaller parameter updates."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        clipped_model = SimpleSequenceActorCritic(
            vocab_size=VOCAB_SIZE, hidden_dim=16, rngs=nnx.Rngs(0)
        )
        unclipped_model = SimpleSequenceActorCritic(
            vocab_size=VOCAB_SIZE,
            hidden_dim=16,
            rngs=nnx.Rngs(0),
        )

        clipped_optimizer = nnx.Optimizer(clipped_model, optax.adam(1e-3), wrt=nnx.Param)
        unclipped_optimizer = nnx.Optimizer(unclipped_model, optax.adam(1e-3), wrt=nnx.Param)

        clipped_trainer = PPOTrainer(
            clipped_model,
            clipped_optimizer,
            PPOConfig(max_grad_norm=1e-4),
        )
        unclipped_trainer = PPOTrainer(
            unclipped_model,
            unclipped_optimizer,
            PPOConfig(max_grad_norm=1e6),
        )

        batch = make_sequence_rollout_batch(
            8,
            old_log_probs=jnp.full((8, SEQ_LEN - 1), -10.0, dtype=jnp.float32),
            returns=jnp.full((8, SEQ_LEN - 1), 1_000.0, dtype=jnp.float32),
            advantages=jnp.full((8, SEQ_LEN - 1), 1_000.0, dtype=jnp.float32),
        )

        clipped_initial = jax.tree.map(
            lambda x: x.copy() if hasattr(x, "copy") else x,
            nnx.state(clipped_model, nnx.Param),
        )
        unclipped_initial = jax.tree.map(
            lambda x: x.copy() if hasattr(x, "copy") else x,
            nnx.state(unclipped_model, nnx.Param),
        )

        clipped_trainer.train_step(batch)
        unclipped_trainer.train_step(batch)

        clipped_final = nnx.state(clipped_model, nnx.Param)
        unclipped_final = nnx.state(unclipped_model, nnx.Param)

        def total_update_norm(initial_state, final_state) -> float:
            total = 0.0
            for initial_leaf, final_leaf in zip(
                jax.tree.leaves(initial_state), jax.tree.leaves(final_state)
            ):
                total += float(jnp.linalg.norm(final_leaf - initial_leaf))
            return total

        clipped_update = total_update_norm(clipped_initial, clipped_final)
        unclipped_update = total_update_norm(unclipped_initial, unclipped_final)

        assert clipped_update < unclipped_update


# =============================================================================
# DPO Configuration Tests
# =============================================================================


class TestDPOConfig:
    """Tests for DPO configuration."""

    def test_dpo_config_exists(self) -> None:
        """DPOConfig class should exist."""
        from artifex.generative_models.training.rl import DPOConfig

        assert DPOConfig is not None

    def test_dpo_config_default_values(self) -> None:
        """DPOConfig should have sensible defaults."""
        from artifex.generative_models.training.rl import DPOConfig

        config = DPOConfig()
        assert hasattr(config, "beta")
        assert hasattr(config, "label_smoothing")
        assert hasattr(config, "reference_free")

    def test_dpo_config_beta_default(self) -> None:
        """Default beta should be 0.1."""
        from artifex.generative_models.training.rl import DPOConfig

        config = DPOConfig()
        assert config.beta == 0.1

    def test_dpo_config_label_smoothing_default(self) -> None:
        """Default label_smoothing should be 0.0."""
        from artifex.generative_models.training.rl import DPOConfig

        config = DPOConfig()
        assert config.label_smoothing == 0.0

    def test_dpo_config_reference_free_default(self) -> None:
        """Default reference_free should be False."""
        from artifex.generative_models.training.rl import DPOConfig

        config = DPOConfig()
        assert config.reference_free is False

    def test_dpo_config_custom_values(self) -> None:
        """DPOConfig should accept custom values."""
        from artifex.generative_models.training.rl import DPOConfig

        config = DPOConfig(
            beta=0.5,
            label_smoothing=0.1,
            reference_free=True,
        )
        assert config.beta == 0.5
        assert config.label_smoothing == 0.1
        assert config.reference_free is True


# =============================================================================
# DPO Trainer Tests
# =============================================================================


class TestDPOTrainer:
    """Tests for DPO trainer."""

    def test_dpo_trainer_exists(self) -> None:
        """DPOTrainer class should exist."""
        from artifex.generative_models.training.rl import DPOTrainer

        assert DPOTrainer is not None

    def test_dpo_trainer_initialization(self, preference_model: SimplePreferenceModel) -> None:
        """DPOTrainer should initialize with model, reference, and optimizer."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        # Create a copy for reference model
        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))

        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        config = DPOConfig()
        trainer = DPOTrainer(preference_model, ref_model, optimizer, config)

        assert trainer.model is preference_model
        assert trainer.reference_model is ref_model
        assert trainer.optimizer is optimizer
        assert trainer.config is config

    def test_dpo_trainer_default_config(self, preference_model: SimplePreferenceModel) -> None:
        """DPOTrainer should use default config if not provided."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        trainer = DPOTrainer(preference_model, ref_model, optimizer)

        assert isinstance(trainer.config, DPOConfig)

    def test_dpo_compute_log_probs(self, preference_model: SimplePreferenceModel) -> None:
        """DPOTrainer should compute log probabilities for sequences."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        config = DPOConfig()
        trainer = DPOTrainer(preference_model, ref_model, optimizer, config)

        # Create fake sequence data
        sequences = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        log_probs = trainer.compute_log_probs(preference_model, sequences)

        assert log_probs.shape == (2,)  # One log prob per sequence
        assert jnp.all(jnp.isfinite(log_probs))

    def test_sequence_policy_adapter_scores_array_output_models(
        self,
        preference_model: SimplePreferenceModel,
    ) -> None:
        """SequencePolicyAdapter should score models that return logits arrays."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            SequencePolicyAdapter,
        )

        batch = GeneratedSequenceBatch(
            generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
        )

        log_probs = SequencePolicyAdapter(preference_model).sequence_log_probs(batch)

        assert log_probs.shape == (2,)
        assert jnp.all(jnp.isfinite(log_probs))

    def test_sequence_policy_adapter_scores_dict_output_models(
        self,
        dict_preference_model: DictOutputPreferenceModel,
    ) -> None:
        """SequencePolicyAdapter should score models that return output dicts."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            SequencePolicyAdapter,
        )

        batch = GeneratedSequenceBatch(
            generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])),
            response_mask=jnp.array([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=jnp.float32),
        )

        log_probs = SequencePolicyAdapter(dict_preference_model).sequence_log_probs(batch)

        assert log_probs.shape == (2,)
        assert jnp.all(jnp.isfinite(log_probs))

    def test_sequence_policy_adapter_supports_model_kwargs_factory(self) -> None:
        """SequencePolicyAdapter should support custom model kwargs factories."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            SequencePolicyAdapter,
        )

        model = StrictMaskPreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(3))
        batch = GeneratedSequenceBatch(
            generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3, 4]])),
            prompt_mask=jnp.array([[1, 1, 0, 0]], dtype=jnp.float32),
            response_mask=jnp.array([[0, 0, 1, 1]], dtype=jnp.float32),
        )

        adapter = SequencePolicyAdapter(
            model,
            model_kwargs_factory=lambda scored_batch: {
                "mask": (scored_batch.prompt_mask + scored_batch.response_mask).astype(jnp.float32)
            },
        )

        log_probs = adapter.sequence_log_probs(batch)

        assert log_probs.shape == (1,)
        assert jnp.all(jnp.isfinite(log_probs))

    def test_dpo_compute_log_probs_supports_loss_masks(
        self, preference_model: SimplePreferenceModel
    ) -> None:
        """DPO scoring should support masks for padding and prompt exclusion."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        trainer = DPOTrainer(preference_model, ref_model, optimizer, DPOConfig())

        trimmed = jnp.array([[1, 2, 3, 4]])
        padded = jnp.array([[1, 2, 3, 4, 0, 0]])
        padded_mask = jnp.array([[0, 1, 1, 1, 0, 0]], dtype=jnp.float32)

        trimmed_score = trainer.compute_log_probs(preference_model, trimmed)
        padded_score = trainer.compute_log_probs(preference_model, padded, padded_mask)

        assert jnp.allclose(trimmed_score, padded_score, atol=1e-6)

    def test_dpo_compute_log_ratios(self, preference_model: SimplePreferenceModel) -> None:
        """DPOTrainer should compute log ratios between policy and reference."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        config = DPOConfig()
        trainer = DPOTrainer(preference_model, ref_model, optimizer, config)

        sequences = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        log_ratios = trainer.compute_log_ratios(sequences)

        assert log_ratios.shape == (2,)
        assert jnp.all(jnp.isfinite(log_ratios))

    def test_dpo_loss(self, preference_model: SimplePreferenceModel) -> None:
        """DPOTrainer should compute DPO loss correctly."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        config = DPOConfig(beta=0.1)
        trainer = DPOTrainer(preference_model, ref_model, optimizer, config)

        # Create preference pair batch
        batch = {
            "chosen": jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "rejected": jnp.array([[1, 2, 9, 10], [5, 6, 11, 12]]),
        }

        loss, metrics = trainer.compute_loss(batch)

        assert jnp.isfinite(loss)
        assert "dpo_loss" in metrics
        assert "reward_accuracy" in metrics

    def test_dpo_loss_accepts_typed_preference_batch(
        self,
        preference_model: SimplePreferenceModel,
    ) -> None:
        """DPOTrainer should accept typed preference batches."""
        from artifex.generative_models.training.rl import (
            DPOConfig,
            DPOTrainer,
            GeneratedBatch,
            GeneratedSequenceBatch,
            PreferenceBatch,
        )

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        trainer = DPOTrainer(preference_model, ref_model, optimizer, DPOConfig(beta=0.1))

        batch = PreferenceBatch(
            chosen=GeneratedSequenceBatch(
                generation=GeneratedBatch(outputs=jnp.array([[10, 11, 12, 13, 0, 0]])),
                response_mask=jnp.array([[0, 0, 1, 1, 0, 0]], dtype=jnp.float32),
            ),
            rejected=GeneratedSequenceBatch(
                generation=GeneratedBatch(outputs=jnp.array([[10, 11, 14, 15, 0, 0]])),
                response_mask=jnp.array([[0, 0, 1, 1, 0, 0]], dtype=jnp.float32),
            ),
        )

        loss, metrics = trainer.compute_loss(batch)

        assert jnp.isfinite(loss)
        assert jnp.isfinite(metrics["margin"])

    def test_dpo_loss_supports_response_masks(
        self, preference_model: SimplePreferenceModel
    ) -> None:
        """DPO loss should accept response-only masks for prompt-conditioned batches."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        trainer = DPOTrainer(preference_model, ref_model, optimizer, DPOConfig(beta=0.1))

        batch = {
            "chosen": jnp.array([[10, 11, 12, 13, 0, 0]]),
            "rejected": jnp.array([[10, 11, 14, 15, 0, 0]]),
            "chosen_loss_mask": jnp.array([[0, 0, 1, 1, 0, 0]], dtype=jnp.float32),
            "rejected_loss_mask": jnp.array([[0, 0, 1, 1, 0, 0]], dtype=jnp.float32),
        }

        loss, metrics = trainer.compute_loss(batch)

        assert jnp.isfinite(loss)
        assert jnp.isfinite(metrics["margin"])

    def test_dpo_loss_with_label_smoothing(self, preference_model: SimplePreferenceModel) -> None:
        """DPOTrainer should apply label smoothing when configured."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        config = DPOConfig(beta=0.1, label_smoothing=0.1)
        trainer = DPOTrainer(preference_model, ref_model, optimizer, config)

        batch = {
            "chosen": jnp.array([[1, 2, 3, 4]]),
            "rejected": jnp.array([[1, 2, 9, 10]]),
        }

        loss, metrics = trainer.compute_loss(batch)

        assert jnp.isfinite(loss)

    def test_dpo_train_step(self, preference_model: SimplePreferenceModel) -> None:
        """DPOTrainer should perform a training step."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        config = DPOConfig()
        trainer = DPOTrainer(preference_model, ref_model, optimizer, config)

        batch = {
            "chosen": jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "rejected": jnp.array([[1, 2, 9, 10], [5, 6, 11, 12]]),
        }

        loss, metrics = trainer.train_step(batch)

        assert jnp.isfinite(loss)
        assert "dpo_loss" in metrics

    def test_dpo_train_step_accepts_typed_preference_batch(
        self,
        preference_model: SimplePreferenceModel,
    ) -> None:
        """DPOTrainer should update on typed preference batches."""
        from artifex.generative_models.training.rl import (
            DPOConfig,
            DPOTrainer,
            GeneratedBatch,
            GeneratedSequenceBatch,
            PreferenceBatch,
        )

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        trainer = DPOTrainer(preference_model, ref_model, optimizer, DPOConfig())

        batch = PreferenceBatch(
            chosen=GeneratedSequenceBatch(
                generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])),
                response_mask=jnp.array(
                    [[0, 1, 1, 1], [0, 1, 1, 1]],
                    dtype=jnp.float32,
                ),
            ),
            rejected=GeneratedSequenceBatch(
                generation=GeneratedBatch(outputs=jnp.array([[1, 2, 9, 10], [5, 6, 11, 12]])),
                response_mask=jnp.array(
                    [[0, 1, 1, 1], [0, 1, 1, 1]],
                    dtype=jnp.float32,
                ),
            ),
        )

        loss, metrics = trainer.train_step(batch)

        assert jnp.isfinite(loss)
        assert "dpo_loss" in metrics

    def test_dpo_reference_free_mode(self, preference_model: SimplePreferenceModel) -> None:
        """DPOTrainer should support reference-free mode."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        config = DPOConfig(reference_free=True)

        # In reference-free mode, no reference model is needed
        trainer = DPOTrainer(preference_model, None, optimizer, config)

        assert trainer.reference_model is None

        batch = {
            "chosen": jnp.array([[1, 2, 3, 4]]),
            "rejected": jnp.array([[1, 2, 9, 10]]),
        }

        loss, metrics = trainer.compute_loss(batch)
        assert jnp.isfinite(loss)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestRLUtilities:
    """Tests for RL utility functions."""

    def test_compute_discounted_returns(self) -> None:
        """Should compute discounted returns correctly."""
        from artifex.generative_models.training.rl import compute_discounted_returns

        rewards = jnp.array([1.0, 1.0, 1.0])
        gamma = 0.99

        returns = compute_discounted_returns(rewards, gamma)

        expected = jnp.array([1 + 0.99 + 0.99**2, 1 + 0.99, 1.0])
        assert jnp.allclose(returns, expected, atol=1e-5)

    def test_compute_gae_advantages(self) -> None:
        """Should compute GAE advantages correctly."""
        from artifex.generative_models.training.rl import compute_gae_advantages

        rewards = jnp.array([1.0, 1.0, 1.0, 1.0])
        values = jnp.array([0.5, 0.5, 0.5, 0.5, 0.0])
        dones = jnp.array([False, False, False, True])
        gamma = 0.99
        gae_lambda = 0.95

        advantages = compute_gae_advantages(rewards, values, dones, gamma, gae_lambda)

        assert advantages.shape == (4,)
        assert jnp.all(jnp.isfinite(advantages))

    def test_normalize_advantages(self) -> None:
        """Should normalize advantages to zero mean, unit variance."""
        from artifex.generative_models.training.rl import normalize_advantages

        advantages = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        normalized = normalize_advantages(advantages)

        assert jnp.abs(jnp.mean(normalized)) < 1e-5
        assert jnp.abs(jnp.std(normalized) - 1.0) < 0.1

    def test_compute_policy_entropy(self) -> None:
        """Should compute policy entropy from log probabilities."""
        from artifex.generative_models.training.rl import compute_policy_entropy

        # Uniform distribution
        log_probs = jnp.log(jnp.ones((4, 4)) / 4)

        entropy = compute_policy_entropy(log_probs)

        # Entropy of uniform distribution over 4 actions = log(4)
        expected_entropy = jnp.log(4.0)
        assert jnp.abs(entropy - expected_entropy) < 0.1


# =============================================================================
# Module Export Tests
# =============================================================================


class TestRLModuleExports:
    """Tests for RL module exports."""

    def test_reinforce_exports(self) -> None:
        """REINFORCE components should be exported from rl module."""
        from artifex.generative_models.training.rl import (
            REINFORCEConfig,
            REINFORCETrainer,
        )

        assert REINFORCEConfig is not None
        assert REINFORCETrainer is not None

    def test_ppo_exports(self) -> None:
        """PPO components should be exported from rl module."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        assert PPOConfig is not None
        assert PPOTrainer is not None

    def test_dpo_exports(self) -> None:
        """DPO components should be exported from rl module."""
        from artifex.generative_models.training.rl import (
            DPOConfig,
            DPOTrainer,
            SequencePolicyAdapter,
        )

        assert DPOConfig is not None
        assert DPOTrainer is not None
        assert SequencePolicyAdapter is not None

    def test_utility_exports(self) -> None:
        """Utility functions should be exported from rl module."""
        from artifex.generative_models.training.rl import (
            compute_clipped_surrogate_loss,
            compute_discounted_returns,
            compute_gae_advantages,
            compute_kl_divergence,
            compute_policy_entropy,
            normalize_advantages,
        )

        assert compute_discounted_returns is not None
        assert compute_gae_advantages is not None
        assert normalize_advantages is not None
        assert compute_policy_entropy is not None
        assert compute_kl_divergence is not None
        assert compute_clipped_surrogate_loss is not None

    def test_main_training_module_exports(self) -> None:
        """RL components should be exported from main training module."""
        from artifex.generative_models.training import (
            ClippedReward,
            CompositeReward,
            compute_clipped_surrogate_loss,
            compute_discounted_returns,
            compute_gae_advantages,
            compute_kl_divergence,
            compute_policy_entropy,
            ConstantReward,
            DPOConfig,
            DPOTrainer,
            GRPOConfig,
            GRPOTrainer,
            normalize_advantages,
            PPOConfig,
            PPOTrainer,
            REINFORCEConfig,
            REINFORCETrainer,
            RewardFunction,
            ScaledReward,
            SequencePolicyAdapter,
            ThresholdReward,
        )

        assert REINFORCEConfig is not None
        assert REINFORCETrainer is not None
        assert PPOConfig is not None
        assert PPOTrainer is not None
        assert GRPOConfig is not None
        assert GRPOTrainer is not None
        assert DPOConfig is not None
        assert DPOTrainer is not None
        assert SequencePolicyAdapter is not None
        assert RewardFunction is not None
        assert ConstantReward is not None
        assert CompositeReward is not None
        assert ThresholdReward is not None
        assert ScaledReward is not None
        assert ClippedReward is not None
        assert compute_discounted_returns is not None
        assert compute_gae_advantages is not None
        assert normalize_advantages is not None
        assert compute_policy_entropy is not None
        assert compute_kl_divergence is not None
        assert compute_clipped_surrogate_loss is not None


# =============================================================================
# GRPO Configuration Tests
# =============================================================================


class TestGRPOConfig:
    """Tests for GRPO configuration."""

    def test_grpo_config_exists(self) -> None:
        """GRPOConfig class should exist."""
        from artifex.generative_models.training.rl import GRPOConfig

        assert GRPOConfig is not None

    def test_grpo_config_default_values(self) -> None:
        """GRPOConfig should have sensible defaults."""
        from artifex.generative_models.training.rl import GRPOConfig

        config = GRPOConfig()
        assert hasattr(config, "num_generations")
        assert hasattr(config, "clip_param")
        assert hasattr(config, "beta")
        assert hasattr(config, "entropy_coeff")

    def test_grpo_config_num_generations_default(self) -> None:
        """Default num_generations should be 4."""
        from artifex.generative_models.training.rl import GRPOConfig

        config = GRPOConfig()
        assert config.num_generations == 4

    def test_grpo_config_clip_param_default(self) -> None:
        """Default clip_param should be 0.2."""
        from artifex.generative_models.training.rl import GRPOConfig

        config = GRPOConfig()
        assert config.clip_param == 0.2

    def test_grpo_config_beta_default(self) -> None:
        """Default beta (KL penalty) should be 0.01."""
        from artifex.generative_models.training.rl import GRPOConfig

        config = GRPOConfig()
        assert config.beta == 0.01

    def test_grpo_config_custom_values(self) -> None:
        """GRPOConfig should accept custom values."""
        from artifex.generative_models.training.rl import GRPOConfig

        config = GRPOConfig(
            num_generations=8,
            clip_param=0.1,
            beta=0.05,
            entropy_coeff=0.02,
        )
        assert config.num_generations == 8
        assert config.clip_param == 0.1
        assert config.beta == 0.05
        assert config.entropy_coeff == 0.02


# =============================================================================
# GRPO Trainer Tests
# =============================================================================


class TestGRPOTrainer:
    """Tests for GRPO trainer."""

    def test_grpo_trainer_exists(self) -> None:
        """GRPOTrainer class should exist."""
        from artifex.generative_models.training.rl import GRPOTrainer

        assert GRPOTrainer is not None

    def test_grpo_trainer_initialization(self, policy_model: SimpleSequencePolicy) -> None:
        """GRPOTrainer should initialize with model and optimizer."""
        from artifex.generative_models.training.rl import GRPOConfig, GRPOTrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = GRPOConfig()
        trainer = GRPOTrainer(policy_model, optimizer, config)

        assert trainer.model is policy_model
        assert trainer.optimizer is optimizer
        assert trainer.config is config

    def test_grpo_trainer_default_config(self, policy_model: SimpleSequencePolicy) -> None:
        """GRPOTrainer should use default config if not provided."""
        from artifex.generative_models.training.rl import GRPOConfig, GRPOTrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = GRPOTrainer(policy_model, optimizer)

        assert isinstance(trainer.config, GRPOConfig)

    def test_grpo_normalize_group_rewards(self, policy_model: SimpleSequencePolicy) -> None:
        """GRPOTrainer should normalize rewards within groups."""
        from artifex.generative_models.training.rl import GRPOConfig, GRPOTrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = GRPOConfig(num_generations=4)
        trainer = GRPOTrainer(policy_model, optimizer, config)

        # 2 prompts, 4 generations each = 8 total
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        normalized = trainer.normalize_group_rewards(rewards, group_size=4)

        # Each group should have zero mean
        group1 = normalized[:4]
        group2 = normalized[4:]
        assert jnp.abs(jnp.mean(group1)) < 1e-5
        assert jnp.abs(jnp.mean(group2)) < 1e-5

    def test_grpo_with_reference_model(self, policy_model: SimpleSequencePolicy) -> None:
        """GRPOTrainer should support optional reference model for KL penalty."""
        from artifex.generative_models.training.rl import GRPOConfig, GRPOTrainer

        ref_model = SimpleSequencePolicy(vocab_size=VOCAB_SIZE, hidden_dim=16, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = GRPOConfig(beta=0.1)
        trainer = GRPOTrainer(policy_model, optimizer, config, reference_model=ref_model)

        assert trainer.reference_model is ref_model

    def test_grpo_compute_loss(self, policy_model: SimpleSequencePolicy) -> None:
        """GRPOTrainer should compute loss from grouped typed sequence rollouts."""
        from artifex.generative_models.training.rl import GRPOConfig, GRPOTrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = GRPOConfig(num_generations=2)
        trainer = GRPOTrainer(policy_model, optimizer, config)

        batch = make_group_rollout_batch(
            4,
            group_size=2,
            old_log_probs=jnp.full((4, SEQ_LEN - 1), -0.5, dtype=jnp.float32),
            sequence_rewards=jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        )

        loss, metrics = trainer.compute_loss(batch)

        assert jnp.isfinite(loss)
        assert "policy_loss" in metrics
        assert "entropy" in metrics
        assert "advantages_mean" in metrics

    def test_grpo_train_step(self, policy_model: SimpleSequencePolicy) -> None:
        """GRPOTrainer should perform a training step on grouped sequence rollouts."""
        from artifex.generative_models.training.rl import GRPOConfig, GRPOTrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = GRPOConfig(num_generations=2)
        trainer = GRPOTrainer(policy_model, optimizer, config)

        batch = make_group_rollout_batch(
            4,
            group_size=2,
            old_log_probs=jnp.full((4, SEQ_LEN - 1), -0.5, dtype=jnp.float32),
            sequence_rewards=jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        )

        loss, metrics = trainer.train_step(batch)

        assert jnp.isfinite(loss)
        assert "policy_loss" in metrics


# =============================================================================
# Reward Function Tests
# =============================================================================


class TestRewardFunctions:
    """Tests for reward function implementations."""

    def test_constant_reward(self) -> None:
        """ConstantReward should return fixed value for all samples."""
        from artifex.generative_models.training.rl import ConstantReward

        reward_fn = ConstantReward(value=5.0)
        samples = jnp.ones((4, 10))

        rewards = reward_fn(samples)

        assert rewards.shape == (4,)
        assert jnp.allclose(rewards, jnp.array([5.0, 5.0, 5.0, 5.0]))

    def test_composite_reward(self) -> None:
        """CompositeReward should combine multiple reward functions."""
        from artifex.generative_models.training.rl import CompositeReward, ConstantReward

        fn1 = ConstantReward(value=1.0)
        fn2 = ConstantReward(value=2.0)
        composite = CompositeReward([fn1, fn2], weights=[0.5, 0.5])

        samples = jnp.ones((3, 10))
        rewards = composite(samples)

        assert rewards.shape == (3,)
        # 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        assert jnp.allclose(rewards, jnp.array([1.5, 1.5, 1.5]))

    def test_composite_reward_default_weights(self) -> None:
        """CompositeReward should use uniform weights by default."""
        from artifex.generative_models.training.rl import CompositeReward, ConstantReward

        fn1 = ConstantReward(value=2.0)
        fn2 = ConstantReward(value=4.0)
        composite = CompositeReward([fn1, fn2])

        samples = jnp.ones((2, 10))
        rewards = composite(samples)

        # Uniform: 0.5 * 2.0 + 0.5 * 4.0 = 3.0
        assert jnp.allclose(rewards, jnp.array([3.0, 3.0]))

    def test_threshold_reward_above(self) -> None:
        """ThresholdReward should reward when metric exceeds threshold."""
        from artifex.generative_models.training.rl import ConstantReward, ThresholdReward

        metric_fn = ConstantReward(value=10.0)
        threshold_reward = ThresholdReward(metric_fn, threshold=5.0, above=True)

        samples = jnp.ones((3, 10))
        rewards = threshold_reward(samples)

        # 10.0 > 5.0, so all should be 1.0
        assert jnp.allclose(rewards, jnp.array([1.0, 1.0, 1.0]))

    def test_threshold_reward_below(self) -> None:
        """ThresholdReward should reward when metric is below threshold."""
        from artifex.generative_models.training.rl import ConstantReward, ThresholdReward

        metric_fn = ConstantReward(value=3.0)
        threshold_reward = ThresholdReward(metric_fn, threshold=5.0, above=False)

        samples = jnp.ones((3, 10))
        rewards = threshold_reward(samples)

        # 3.0 < 5.0, so all should be 1.0
        assert jnp.allclose(rewards, jnp.array([1.0, 1.0, 1.0]))

    def test_scaled_reward(self) -> None:
        """ScaledReward should apply linear transformation."""
        from artifex.generative_models.training.rl import ConstantReward, ScaledReward

        base_fn = ConstantReward(value=2.0)
        scaled = ScaledReward(base_fn, scale=3.0, offset=1.0)

        samples = jnp.ones((2, 10))
        rewards = scaled(samples)

        # 3.0 * 2.0 + 1.0 = 7.0
        assert jnp.allclose(rewards, jnp.array([7.0, 7.0]))

    def test_clipped_reward(self) -> None:
        """ClippedReward should clip rewards to specified range."""
        from artifex.generative_models.training.rl import ClippedReward, ConstantReward

        base_fn = ConstantReward(value=10.0)
        clipped = ClippedReward(base_fn, min_value=-1.0, max_value=1.0)

        samples = jnp.ones((2, 10))
        rewards = clipped(samples)

        # 10.0 clipped to [-1, 1] = 1.0
        assert jnp.allclose(rewards, jnp.array([1.0, 1.0]))

    def test_reward_function_exports(self) -> None:
        """All reward functions should be exported from rl module."""
        from artifex.generative_models.training.rl import (
            ClippedReward,
            CompositeReward,
            ConstantReward,
            RewardFunction,
            ScaledReward,
            ThresholdReward,
        )

        assert RewardFunction is not None
        assert ConstantReward is not None
        assert CompositeReward is not None
        assert ThresholdReward is not None
        assert ScaledReward is not None
        assert ClippedReward is not None


# =============================================================================
# Transform Compatibility Tests
# =============================================================================


class TestRLTransformCompatibility:
    """Tests for the public JAX and NNX transform boundaries in RL training."""

    def test_sequence_policy_adapter_action_log_probs_support_jax_jit(
        self,
        policy_model: SimpleSequencePolicy,
    ) -> None:
        """SequencePolicyAdapter should expose a JAX-jittable rollout scorer."""
        from artifex.generative_models.training.rl import SequencePolicyAdapter

        adapter = SequencePolicyAdapter(policy_model)
        batch = make_sequence_rollout_batch(4)

        compiled = jax.jit(lambda rollout: adapter.action_log_probs(rollout))
        action_log_probs = compiled(batch)

        assert action_log_probs.shape == (4, SEQ_LEN - 1)
        assert jnp.all(jnp.isfinite(action_log_probs))

    def test_reinforce_compute_loss_supports_jax_grad_over_returns(
        self,
        policy_model: SimpleSequencePolicy,
    ) -> None:
        """REINFORCE loss should remain differentiable for JAX-traced rollout tensors."""
        from artifex.generative_models.training.rl import REINFORCEConfig, REINFORCETrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = REINFORCETrainer(
            policy_model,
            optimizer,
            REINFORCEConfig(normalize_returns=False),
        )
        old_log_probs = jnp.full((4, SEQ_LEN - 1), -0.5, dtype=jnp.float32)

        def scalar_loss(returns: jax.Array) -> jax.Array:
            batch = make_sequence_rollout_batch(
                4,
                old_log_probs=old_log_probs,
                returns=returns,
            )
            loss, _ = trainer.compute_loss(batch)
            return loss

        returns = jnp.ones((4, SEQ_LEN - 1), dtype=jnp.float32)
        grad_returns = jax.grad(scalar_loss)(returns)

        assert grad_returns.shape == returns.shape
        assert jnp.all(jnp.isfinite(grad_returns))

    def test_reinforce_create_loss_fn_is_nnx_jittable(
        self,
        policy_model: SimpleSequencePolicy,
    ) -> None:
        """REINFORCE should expose an NNX-jittable public loss boundary."""
        from artifex.generative_models.training.rl import REINFORCEConfig, REINFORCETrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = REINFORCETrainer(policy_model, optimizer, REINFORCEConfig())
        loss_fn = trainer.create_loss_fn()
        batch = make_sequence_rollout_batch(
            4,
            token_rewards=jnp.full((4, SEQ_LEN - 1), 0.5, dtype=jnp.float32),
        )

        compiled_loss = nnx.jit(loss_fn)
        loss, metrics = compiled_loss(policy_model, batch)

        assert jnp.isfinite(loss)
        assert jnp.isfinite(metrics["policy_loss"])
        assert jnp.isfinite(metrics["entropy"])

    def test_ppo_create_loss_fn_supports_nnx_grad(
        self,
        actor_critic_model: SimpleSequenceActorCritic,
    ) -> None:
        """PPO loss should remain differentiable with respect to model parameters."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = PPOTrainer(actor_critic_model, optimizer, PPOConfig())
        loss_fn = trainer.create_loss_fn()
        batch = make_sequence_rollout_batch(
            4,
            old_log_probs=jnp.full((4, SEQ_LEN - 1), -0.75, dtype=jnp.float32),
            returns=jnp.full((4, SEQ_LEN - 1), 0.5, dtype=jnp.float32),
            advantages=jnp.full((4, SEQ_LEN - 1), 1.0, dtype=jnp.float32),
        )

        def scalar_loss(model: nnx.Module) -> jax.Array:
            loss, _ = loss_fn(model, batch)
            return loss

        grads = nnx.grad(scalar_loss)(actor_critic_model)

        assert_finite_nnx_grads(grads)

    def test_grpo_create_loss_fn_supports_nnx_value_and_grad(
        self,
        policy_model: SimpleSequencePolicy,
    ) -> None:
        """GRPO should expose a value-and-grad compatible public loss boundary."""
        from artifex.generative_models.training.rl import GRPOConfig, GRPOTrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = GRPOTrainer(policy_model, optimizer, GRPOConfig(num_generations=2))
        loss_fn = trainer.create_loss_fn()
        batch = make_group_rollout_batch(
            4,
            group_size=2,
            old_log_probs=jnp.full((4, SEQ_LEN - 1), -0.25, dtype=jnp.float32),
            sequence_rewards=jnp.array([0.0, 1.0, 0.5, 1.5], dtype=jnp.float32),
        )

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy_model, batch)

        assert jnp.isfinite(loss)
        assert jnp.isfinite(metrics["policy_loss"])
        assert jnp.isfinite(metrics["entropy"])
        assert_finite_nnx_grads(grads)

    def test_dpo_create_loss_fn_is_nnx_jittable(
        self,
        preference_model: SimplePreferenceModel,
    ) -> None:
        """DPO should expose an NNX-jittable public loss boundary."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        trainer = DPOTrainer(preference_model, ref_model, optimizer, DPOConfig(beta=0.1))
        loss_fn = trainer.create_loss_fn()
        batch = {
            "chosen": jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32),
            "rejected": jnp.array([[1, 2, 2, 4], [5, 6, 6, 8]], dtype=jnp.int32),
        }

        compiled_loss = nnx.jit(loss_fn)
        loss, metrics = compiled_loss(preference_model, batch)

        assert jnp.isfinite(loss)
        assert jnp.isfinite(metrics["dpo_loss"])
        assert jnp.isfinite(metrics["reward_accuracy"])


# =============================================================================
# Integration Tests - Multi-Step Training
# =============================================================================


class TestMultiStepTrainingIntegration:
    """Integration tests for multi-step training scenarios."""

    def test_reinforce_multiple_training_steps(self, policy_model: SimpleSequencePolicy) -> None:
        """REINFORCE should support multiple consecutive training steps."""
        from artifex.generative_models.training.rl import REINFORCEConfig, REINFORCETrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = REINFORCEConfig(gamma=0.99, entropy_coeff=0.01)
        trainer = REINFORCETrainer(policy_model, optimizer, config)

        # Run multiple training steps
        losses = []
        for step in range(5):
            token_rewards = jax.random.uniform(jax.random.key(step + 100), (8, SEQ_LEN - 1))
            rollout = make_sequence_rollout_batch(
                8, sequence_offset=step, token_rewards=token_rewards
            )
            loss, metrics = trainer.train_step(rollout)
            losses.append(float(loss))

        # All losses should be finite
        assert all(jnp.isfinite(l) for l in losses)
        # Losses should vary (not stuck)
        assert len(set(losses)) > 1

    def test_ppo_multiple_training_steps(
        self, actor_critic_model: SimpleSequenceActorCritic
    ) -> None:
        """PPO should support multiple consecutive training steps."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        config = PPOConfig(gamma=0.99, gae_lambda=0.95, clip_param=0.2)
        trainer = PPOTrainer(actor_critic_model, optimizer, config)

        losses = []
        for step in range(5):
            batch = make_sequence_rollout_batch(
                8,
                sequence_offset=step,
                old_log_probs=jax.random.uniform(jax.random.key(step), (8, SEQ_LEN - 1)) * -2,
                returns=jax.random.uniform(jax.random.key(step + 50), (8, SEQ_LEN - 1)),
                advantages=jax.random.normal(jax.random.key(step + 100), (8, SEQ_LEN - 1)),
            )
            loss, metrics = trainer.train_step(batch)
            losses.append(float(loss))

        assert all(jnp.isfinite(l) for l in losses)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics

    def test_grpo_multiple_training_steps(self, policy_model: SimpleSequencePolicy) -> None:
        """GRPO should support multiple consecutive training steps."""
        from artifex.generative_models.training.rl import GRPOConfig, GRPOTrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = GRPOConfig(num_generations=2, clip_param=0.2)
        trainer = GRPOTrainer(policy_model, optimizer, config)

        losses = []
        for step in range(5):
            batch = make_group_rollout_batch(
                4,
                group_size=2,
                seq_len=SEQ_LEN,
                old_log_probs=jax.random.uniform(jax.random.key(step), (4, SEQ_LEN - 1)) * -2,
                sequence_rewards=jax.random.uniform(jax.random.key(step + 50), (4,)),
            )
            loss, metrics = trainer.train_step(batch)
            losses.append(float(loss))

        assert all(jnp.isfinite(l) for l in losses)

    def test_dpo_multiple_training_steps(self, preference_model: SimplePreferenceModel) -> None:
        """DPO should support multiple consecutive training steps."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-4), wrt=nnx.Param)
        config = DPOConfig(beta=0.1)
        trainer = DPOTrainer(preference_model, ref_model, optimizer, config)

        losses = []
        for step in range(5):
            batch = {
                "chosen": jax.random.randint(jax.random.key(step), (4, 8), 0, 100),
                "rejected": jax.random.randint(jax.random.key(step + 50), (4, 8), 0, 100),
            }
            loss, metrics = trainer.train_step(batch)
            losses.append(float(loss))

        assert all(jnp.isfinite(l) for l in losses)

    def test_model_parameters_update(self, policy_model: SimpleSequencePolicy) -> None:
        """Training should update model parameters."""
        from artifex.generative_models.training.rl import REINFORCEConfig, REINFORCETrainer

        # Store initial parameters
        initial_params = jax.tree.map(lambda x: x.copy(), nnx.state(policy_model))

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-2), wrt=nnx.Param)
        trainer = REINFORCETrainer(policy_model, optimizer, REINFORCEConfig())

        # Train for several steps
        for step in range(10):
            rollout = make_sequence_rollout_batch(
                8,
                sequence_offset=step,
                token_rewards=jnp.ones((8, SEQ_LEN - 1), dtype=jnp.float32),
            )
            trainer.train_step(rollout)

        # Get final parameters
        final_params = nnx.state(policy_model)

        # Parameters should have changed
        def params_different(init, final):
            leaves_init = jax.tree.leaves(init)
            leaves_final = jax.tree.leaves(final)
            for i, f in zip(leaves_init, leaves_final):
                if not jnp.allclose(i, f):
                    return True
            return False

        assert params_different(initial_params, final_params)


# =============================================================================
# Integration Tests - Gradient Accumulation
# =============================================================================


class TestGradientAccumulationIntegration:
    """Integration tests for RL trainers with gradient accumulation."""

    def test_reinforce_with_gradient_accumulation(self, policy_model: SimpleSequencePolicy) -> None:
        """REINFORCE should work with gradient accumulation."""
        from artifex.generative_models.training import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )
        from artifex.generative_models.training.rl import REINFORCEConfig, REINFORCETrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = REINFORCETrainer(policy_model, optimizer, REINFORCEConfig())

        accumulator = GradientAccumulator(
            GradientAccumulatorConfig(accumulation_steps=4, normalize_gradients=True)
        )

        # Simulate gradient accumulation workflow
        accumulated_losses = []
        for micro_step in range(8):  # 2 full accumulation cycles
            trajectory = make_sequence_rollout_batch(
                4,
                sequence_offset=micro_step,
                token_rewards=jax.random.uniform(
                    jax.random.key(micro_step),
                    (4, SEQ_LEN - 1),
                ),
            )

            # Compute loss and metrics (without updating)
            loss, metrics = trainer.train_step(trajectory)
            accumulated_losses.append(float(loss))

            if accumulator.should_update(micro_step):
                # Would apply accumulated gradients here in real workflow
                pass

        assert len(accumulated_losses) == 8
        assert all(jnp.isfinite(l) for l in accumulated_losses)

    def test_ppo_with_gradient_accumulation(
        self, actor_critic_model: SimpleSequenceActorCritic
    ) -> None:
        """PPO should work with gradient accumulation."""
        from artifex.generative_models.training import (
            GradientAccumulator,
            GradientAccumulatorConfig,
        )
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = PPOTrainer(actor_critic_model, optimizer, PPOConfig())

        GradientAccumulator(
            GradientAccumulatorConfig(accumulation_steps=2, normalize_gradients=True)
        )

        metrics_history = []
        for micro_step in range(4):
            batch = make_sequence_rollout_batch(
                4,
                sequence_offset=micro_step,
                old_log_probs=jax.random.uniform(
                    jax.random.key(micro_step),
                    (4, SEQ_LEN - 1),
                )
                * -2,
                returns=jax.random.uniform(jax.random.key(micro_step + 50), (4, SEQ_LEN - 1)),
                advantages=jax.random.normal(
                    jax.random.key(micro_step + 100),
                    (4, SEQ_LEN - 1),
                ),
            )

            loss, metrics = trainer.train_step(batch)
            metrics_history.append(metrics)

        # Check all metrics were tracked
        assert len(metrics_history) == 4
        assert all("policy_loss" in m for m in metrics_history)
        assert all("value_loss" in m for m in metrics_history)


# =============================================================================
# Integration Tests - Callbacks
# =============================================================================


class TestCallbackIntegration:
    """Integration tests for RL trainers with callbacks."""

    def test_training_with_early_stopping_check(self, policy_model: SimpleSequencePolicy) -> None:
        """RL trainers should produce metrics compatible with early stopping."""
        from artifex.generative_models.training.callbacks import (
            EarlyStopping,
            EarlyStoppingConfig,
        )
        from artifex.generative_models.training.rl import REINFORCEConfig, REINFORCETrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = REINFORCETrainer(policy_model, optimizer, REINFORCEConfig())

        EarlyStopping(
            EarlyStoppingConfig(
                monitor="policy_loss",
                patience=3,
                mode="min",
                min_delta=0.001,
            )
        )

        # Simulate epochs with early stopping
        for epoch in range(10):
            epoch_losses = []
            for step in range(5):
                trajectory = make_sequence_rollout_batch(
                    4,
                    sequence_offset=epoch * 5 + step,
                    token_rewards=jax.random.uniform(
                        jax.random.key(epoch * 5 + step + 100),
                        (4, SEQ_LEN - 1),
                    ),
                )
                loss, metrics = trainer.train_step(trajectory)
                epoch_losses.append(float(metrics["policy_loss"]))

            # Simulate epoch end with mean loss
            epoch_metrics = {"policy_loss": sum(epoch_losses) / len(epoch_losses)}

            # This would be called with a TrainerLike object in real usage
            # early_stopping.on_epoch_end(trainer_like, epoch, epoch_metrics)

            # Verify metrics are compatible format
            assert "policy_loss" in epoch_metrics
            assert jnp.isfinite(epoch_metrics["policy_loss"])

    def test_training_metrics_for_logging(
        self, actor_critic_model: SimpleSequenceActorCritic
    ) -> None:
        """PPO should produce complete metrics suitable for logging."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = PPOTrainer(actor_critic_model, optimizer, PPOConfig())

        batch = make_sequence_rollout_batch(
            8,
            old_log_probs=jax.random.uniform(jax.random.key(0), (8, SEQ_LEN - 1)) * -2,
            returns=jax.random.uniform(jax.random.key(50), (8, SEQ_LEN - 1)),
            advantages=jax.random.normal(jax.random.key(100), (8, SEQ_LEN - 1)),
        )

        loss, metrics = trainer.train_step(batch)

        # Verify all expected metrics for logging
        expected_metrics = ["policy_loss", "value_loss", "entropy"]
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert jnp.isfinite(metrics[metric]), f"Non-finite metric: {metric}"

    def test_grpo_metrics_for_monitoring(self, policy_model: SimpleSequencePolicy) -> None:
        """GRPO should produce metrics suitable for monitoring."""
        from artifex.generative_models.training.rl import GRPOConfig, GRPOTrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = GRPOTrainer(policy_model, optimizer, GRPOConfig(num_generations=2))

        batch = make_group_rollout_batch(
            4,
            group_size=2,
            old_log_probs=jax.random.uniform(jax.random.key(0), (4, SEQ_LEN - 1)) * -2,
            sequence_rewards=jax.random.uniform(jax.random.key(50), (4,)),
        )

        loss, metrics = trainer.train_step(batch)

        # GRPO should have these metrics
        assert "policy_loss" in metrics
        assert "entropy" in metrics
        assert "advantages_mean" in metrics
        assert all(jnp.isfinite(v) for v in metrics.values())


# =============================================================================
# Integration Tests - Cross-Component
# =============================================================================


class TestCrossComponentIntegration:
    """Integration tests for interactions between RL components."""

    def test_reward_functions_with_grpo(self, policy_model: SimpleSequencePolicy) -> None:
        """Reward functions should integrate with GRPO trainer."""
        from artifex.generative_models.training.rl import (
            CompositeReward,
            ConstantReward,
            GRPOConfig,
            GRPOTrainer,
            ScaledReward,
        )

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = GRPOTrainer(policy_model, optimizer, GRPOConfig(num_generations=2))

        # Create composite reward function
        reward_fn = CompositeReward(
            [
                ConstantReward(value=1.0),
                ScaledReward(ConstantReward(value=0.5), scale=2.0, offset=0.0),
            ],
            weights=[0.5, 0.5],
        )

        # Generate fake samples
        samples = jax.random.normal(jax.random.key(0), (4, 10))
        rewards = reward_fn(samples)

        # Use rewards in training
        batch = make_group_rollout_batch(
            4,
            group_size=2,
            old_log_probs=jax.random.uniform(jax.random.key(2), (4, SEQ_LEN - 1)) * -2,
            sequence_rewards=rewards,
        )

        loss, metrics = trainer.train_step(batch)
        assert jnp.isfinite(loss)

    def test_utility_functions_in_training_loop(self) -> None:
        """Utility functions should work together in training loop."""
        from artifex.generative_models.training.rl import (
            compute_discounted_returns,
            compute_gae_advantages,
            normalize_advantages,
        )

        # Simulate trajectory
        rewards = jax.random.uniform(jax.random.key(0), (10,))
        values = jax.random.uniform(jax.random.key(1), (11,))
        dones = jnp.zeros(10, dtype=bool).at[9].set(True)

        # Compute returns
        returns = compute_discounted_returns(rewards, gamma=0.99)
        assert returns.shape == (10,)

        # Compute GAE advantages
        advantages = compute_gae_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        assert advantages.shape == (10,)

        # Normalize advantages
        normalized = normalize_advantages(advantages)
        assert jnp.abs(jnp.mean(normalized)) < 0.1
        assert jnp.abs(jnp.std(normalized) - 1.0) < 0.2

    def test_different_trainers_same_model_architecture(self) -> None:
        """Different RL trainers should work with compatible model architectures."""
        from artifex.generative_models.training.rl import (
            GRPOConfig,
            GRPOTrainer,
            REINFORCEConfig,
            REINFORCETrainer,
        )

        # Create models
        rngs = nnx.Rngs(0)
        model1 = SimpleSequencePolicy(vocab_size=VOCAB_SIZE, hidden_dim=16, rngs=rngs)
        rngs2 = nnx.Rngs(1)
        model2 = SimpleSequencePolicy(vocab_size=VOCAB_SIZE, hidden_dim=16, rngs=rngs2)

        # Train with REINFORCE
        opt1 = nnx.Optimizer(model1, optax.adam(1e-3), wrt=nnx.Param)
        reinforce_trainer = REINFORCETrainer(model1, opt1, REINFORCEConfig())

        trajectory = make_sequence_rollout_batch(
            4,
            token_rewards=jnp.ones((4, SEQ_LEN - 1), dtype=jnp.float32),
        )
        loss1, _ = reinforce_trainer.train_step(trajectory)

        # Train with GRPO
        opt2 = nnx.Optimizer(model2, optax.adam(1e-3), wrt=nnx.Param)
        grpo_trainer = GRPOTrainer(model2, opt2, GRPOConfig(num_generations=2))

        batch = make_group_rollout_batch(
            4,
            group_size=2,
            old_log_probs=jnp.full((4, SEQ_LEN - 1), -0.5, dtype=jnp.float32),
            sequence_rewards=jnp.ones((4,), dtype=jnp.float32),
        )
        loss2, _ = grpo_trainer.train_step(batch)

        # Both should produce valid losses
        assert jnp.isfinite(loss1)
        assert jnp.isfinite(loss2)


# =============================================================================
# Integration Tests - End-to-End Workflows
# =============================================================================


class TestEndToEndWorkflows:
    """End-to-end workflow integration tests."""

    def test_complete_reinforce_training_loop(self, policy_model: SimpleSequencePolicy) -> None:
        """Complete REINFORCE training workflow from start to finish."""
        from artifex.generative_models.training.rl import REINFORCEConfig, REINFORCETrainer

        # Setup
        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        config = REINFORCEConfig(gamma=0.99, normalize_returns=True, entropy_coeff=0.01)
        trainer = REINFORCETrainer(policy_model, optimizer, config)

        # Training loop
        num_epochs = 3
        steps_per_epoch = 5
        all_metrics = []

        for epoch in range(num_epochs):
            epoch_metrics = []
            for step in range(steps_per_epoch):
                trajectory = make_sequence_rollout_batch(
                    8,
                    sequence_offset=epoch * steps_per_epoch + step,
                    token_rewards=jax.random.uniform(
                        jax.random.key(step + 1000),
                        (8, SEQ_LEN - 1),
                    ),
                )

                # Train step
                loss, metrics = trainer.train_step(trajectory)
                epoch_metrics.append(metrics)

            all_metrics.append(epoch_metrics)

        # Verify training completed
        assert len(all_metrics) == num_epochs
        assert all(len(em) == steps_per_epoch for em in all_metrics)

    def test_complete_ppo_training_loop(
        self, actor_critic_model: SimpleSequenceActorCritic
    ) -> None:
        """Complete PPO training workflow with GAE computation."""
        from artifex.generative_models.training.rl import PPOConfig, PPOTrainer

        # Setup
        optimizer = nnx.Optimizer(actor_critic_model, optax.adam(3e-4), wrt=nnx.Param)
        config = PPOConfig(
            gamma=0.99, gae_lambda=0.95, clip_param=0.2, vf_coeff=0.5, entropy_coeff=0.01
        )
        trainer = PPOTrainer(actor_critic_model, optimizer, config)

        # Collect trajectory and train
        for iteration in range(3):
            sequences = make_token_sequences(16, offset=iteration)
            token_rewards = jax.random.uniform(
                jax.random.key(iteration + 200),
                (16, SEQ_LEN - 1),
            )

            # Get values from model
            logits, values = actor_critic_model(sequences)

            # Compute log probs
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            old_log_probs = jnp.take_along_axis(
                log_probs[:, :-1, :],
                sequences[:, 1:, None],
                axis=-1,
            ).squeeze(-1)

            # Compute returns and advantages using GAE
            dones = jnp.zeros((16, SEQ_LEN - 1), dtype=bool).at[:, -1].set(True)
            values_with_next = jnp.concatenate(
                [values[:, :-1], jnp.zeros((16, 1), dtype=values.dtype)],
                axis=1,
            )
            advantages = trainer.compute_gae(token_rewards, values_with_next, dones)
            returns = advantages + values[:, :-1]

            # Train
            batch = make_sequence_rollout_batch(
                16,
                sequence_offset=iteration,
                old_log_probs=old_log_probs,
                returns=returns,
                advantages=advantages,
            )
            loss, metrics = trainer.train_step(batch)

            assert jnp.isfinite(loss)
            assert all(jnp.isfinite(v) for v in metrics.values())

    def test_complete_grpo_training_loop(self, policy_model: SimpleSequencePolicy) -> None:
        """Complete GRPO training workflow with group normalization."""
        from artifex.generative_models.training.rl import (
            ConstantReward,
            GRPOConfig,
            GRPOTrainer,
            ScaledReward,
        )

        # Setup
        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-4), wrt=nnx.Param)
        config = GRPOConfig(num_generations=4, clip_param=0.2, beta=0.01)
        trainer = GRPOTrainer(policy_model, optimizer, config)

        # Create reward function
        ScaledReward(ConstantReward(value=1.0), scale=1.0, offset=0.0)

        # Training loop
        num_prompts = 4
        num_generations = config.num_generations
        batch_size = num_prompts * num_generations

        for iteration in range(3):
            sequences = make_token_sequences(batch_size, offset=iteration)
            logits = policy_model(sequences)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            old_log_probs = jnp.take_along_axis(
                log_probs[:, :-1, :],
                sequences[:, 1:, None],
                axis=-1,
            ).squeeze(-1)
            rewards = jax.random.uniform(jax.random.key(iteration + 1000), (batch_size,))
            batch = make_group_rollout_batch(
                batch_size,
                group_size=num_generations,
                old_log_probs=old_log_probs,
                sequence_rewards=rewards,
            )
            loss, metrics = trainer.train_step(batch)

            assert jnp.isfinite(loss)
            assert "policy_loss" in metrics
            assert "advantages_mean" in metrics

    def test_complete_dpo_preference_learning(
        self, preference_model: SimplePreferenceModel
    ) -> None:
        """Complete DPO training workflow with preference pairs."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        # Setup with reference model
        ref_model = SimplePreferenceModel(vocab_size=100, hidden_dim=32, rngs=nnx.Rngs(999))
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-5), wrt=nnx.Param)
        config = DPOConfig(beta=0.1, label_smoothing=0.0)
        trainer = DPOTrainer(preference_model, ref_model, optimizer, config)

        # Training loop with preference pairs
        for iteration in range(5):
            # Generate preference pairs
            key = jax.random.key(iteration)
            chosen = jax.random.randint(key, (8, 16), 0, 100)
            rejected = jax.random.randint(jax.random.key(iteration + 500), (8, 16), 0, 100)

            batch = {"chosen": chosen, "rejected": rejected}
            loss, metrics = trainer.train_step(batch)

            assert jnp.isfinite(loss)
            assert "dpo_loss" in metrics
            assert "reward_accuracy" in metrics

            # Reward accuracy should be between 0 and 1
            assert 0 <= float(metrics["reward_accuracy"]) <= 1

    def test_simpo_reference_free_workflow(self, preference_model: SimplePreferenceModel) -> None:
        """Complete SimPO (reference-free DPO) training workflow."""
        from artifex.generative_models.training.rl import DPOConfig, DPOTrainer

        # Setup without reference model
        optimizer = nnx.Optimizer(preference_model, optax.adam(1e-5), wrt=nnx.Param)
        config = DPOConfig(beta=0.1, reference_free=True)
        trainer = DPOTrainer(preference_model, None, optimizer, config)

        # Training loop
        for iteration in range(3):
            key = jax.random.key(iteration)
            chosen = jax.random.randint(key, (4, 12), 0, 100)
            rejected = jax.random.randint(jax.random.key(iteration + 100), (4, 12), 0, 100)

            batch = {"chosen": chosen, "rejected": rejected}
            loss, metrics = trainer.train_step(batch)

            assert jnp.isfinite(loss)
            assert trainer.reference_model is None  # Verify no reference model


# =============================================================================
# Integration Tests - Dynamic Loss Scaling with RL
# =============================================================================


class TestDynamicLossScalingIntegration:
    """Integration tests for RL trainers with dynamic loss scaling."""

    def test_reinforce_with_loss_scaling(self, policy_model: SimpleSequencePolicy) -> None:
        """REINFORCE losses should be compatible with loss scaling."""
        from artifex.generative_models.training import (
            DynamicLossScaler,
            DynamicLossScalerConfig,
        )
        from artifex.generative_models.training.rl import REINFORCEConfig, REINFORCETrainer

        optimizer = nnx.Optimizer(policy_model, optax.adam(1e-3), wrt=nnx.Param)
        trainer = REINFORCETrainer(policy_model, optimizer, REINFORCEConfig())

        scaler = DynamicLossScaler(
            DynamicLossScalerConfig(initial_scale=2**15, growth_interval=100)
        )

        # Training loop with loss scaling
        for step in range(5):
            trajectory = make_sequence_rollout_batch(
                4,
                sequence_offset=step,
                token_rewards=jax.random.uniform(
                    jax.random.key(step + 100),
                    (4, SEQ_LEN - 1),
                ),
            )

            loss, metrics = trainer.train_step(trajectory)

            # Scale and unscale loss (simulating mixed precision)
            scaled_loss = scaler.scale_loss(loss)
            assert jnp.isfinite(scaled_loss)

            # Check for overflow
            overflow = scaler.check_overflow(jnp.array([scaled_loss]))
            scaler.update_scale(overflow)

            # Scale should remain reasonable
            assert scaler.scale >= scaler.config.min_scale
            assert scaler.scale <= scaler.config.max_scale


# =============================================================================
# Integration Tests - Reward Function Composition
# =============================================================================


class TestRewardFunctionCompositionIntegration:
    """Integration tests for complex reward function compositions."""

    def test_nested_composite_rewards(self) -> None:
        """Nested composite reward functions should work correctly."""
        from artifex.generative_models.training.rl import (
            ClippedReward,
            CompositeReward,
            ConstantReward,
            ScaledReward,
        )

        # Create nested reward structure
        base1 = ConstantReward(value=1.0)
        base2 = ConstantReward(value=2.0)
        scaled = ScaledReward(base1, scale=3.0, offset=0.5)  # 3.5
        clipped = ClippedReward(base2, min_value=0.0, max_value=1.5)  # 1.5

        inner_composite = CompositeReward([scaled, clipped], weights=[0.5, 0.5])
        # Inner: 0.5 * 3.5 + 0.5 * 1.5 = 2.5

        outer = CompositeReward([inner_composite, ConstantReward(value=0.5)], weights=[0.8, 0.2])
        # Outer: 0.8 * 2.5 + 0.2 * 0.5 = 2.1

        samples = jnp.ones((3, 10))
        rewards = outer(samples)

        assert rewards.shape == (3,)
        assert jnp.allclose(rewards, jnp.array([2.1, 2.1, 2.1]), atol=0.01)

    def test_reward_function_with_conditions(self) -> None:
        """Reward functions should handle optional conditions parameter."""
        from artifex.generative_models.training.rl import ConstantReward

        reward_fn = ConstantReward(value=1.0)
        samples = jnp.ones((4, 10))
        conditions = jnp.ones((4, 5))

        # Should work with conditions
        rewards = reward_fn(samples, conditions=conditions)
        assert rewards.shape == (4,)

        # Should work without conditions
        rewards_no_cond = reward_fn(samples)
        assert rewards_no_cond.shape == (4,)

        # Results should be the same for constant reward
        assert jnp.allclose(rewards, rewards_no_cond)
