"""Autoregressive model trainer with teacher forcing and scheduled sampling.

Provides specialized training utilities for autoregressive generative models
including language models, image transformers, and sequence-to-sequence models.

Features:
- Teacher forcing with optional scheduled sampling
- Multiple sampling schedules (linear, exponential, inverse sigmoid)
- Label smoothing for cross-entropy loss
- Causal masking utilities
- Temperature-controlled generation

References:
    - Teacher Forcing: Williams & Zipser, 1989
    - Scheduled Sampling: https://arxiv.org/abs/1506.03099
    - Label Smoothing: https://arxiv.org/abs/1512.00567
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(slots=True)
class AutoregressiveTrainingConfig:
    """Configuration for autoregressive model training.

    Attributes:
        use_teacher_forcing: Whether to use teacher forcing during training.
        scheduled_sampling: Type of sampling schedule for transitioning away
            from teacher forcing. Only used if use_teacher_forcing=True.
            - "none": Always use teacher forcing
            - "linear": Linear decay of teacher forcing probability
            - "exponential": Exponential decay
            - "inverse_sigmoid": Inverse sigmoid decay (smoother transition)
        sampling_warmup_steps: Steps before scheduled sampling starts.
        sampling_decay_steps: Steps over which sampling schedule decays.
        min_teacher_forcing_prob: Minimum teacher forcing probability.
        label_smoothing: Label smoothing factor for cross-entropy (0 = none).
        use_causal_mask: Whether to apply causal masking.
        pad_token_id: Token ID for padding (excluded from loss).
        ignore_index: Index to ignore in loss computation (default: -100).
    """

    use_teacher_forcing: bool = True
    scheduled_sampling: Literal["none", "linear", "exponential", "inverse_sigmoid"] = "none"
    sampling_warmup_steps: int = 0
    sampling_decay_steps: int = 10000
    min_teacher_forcing_prob: float = 0.0
    label_smoothing: float = 0.0
    use_causal_mask: bool = True
    pad_token_id: int | None = None
    ignore_index: int = -100


def create_causal_mask(seq_length: int) -> jax.Array:
    """Create a causal attention mask.

    Args:
        seq_length: Length of the sequence.

    Returns:
        Boolean mask of shape (seq_length, seq_length) where True indicates
        positions that should be attended to (lower triangular).
    """
    # Lower triangular mask: position i can attend to positions <= i
    return jnp.tril(jnp.ones((seq_length, seq_length), dtype=jnp.bool_))


def create_padding_mask(
    tokens: jax.Array,
    pad_token_id: int,
) -> jax.Array:
    """Create padding mask.

    Args:
        tokens: Token IDs of shape (batch, seq_length).
        pad_token_id: ID of padding token.

    Returns:
        Boolean mask of shape (batch, seq_length) where True indicates
        valid (non-padding) positions.
    """
    return tokens != pad_token_id


def create_combined_mask(
    tokens: jax.Array,
    pad_token_id: int | None = None,
) -> jax.Array:
    """Create combined causal and padding mask.

    Args:
        tokens: Token IDs of shape (batch, seq_length).
        pad_token_id: ID of padding token (None = no padding mask).

    Returns:
        Attention mask of shape (batch, 1, seq_length, seq_length).
    """
    seq_length = tokens.shape[1]

    # Causal mask: (seq_length, seq_length)
    causal_mask = create_causal_mask(seq_length)

    if pad_token_id is not None:
        # Padding mask: (batch, seq_length)
        padding_mask = create_padding_mask(tokens, pad_token_id)
        # Expand for broadcasting: (batch, 1, 1, seq_length)
        padding_mask = padding_mask[:, None, None, :]
        # Combine: (batch, 1, seq_length, seq_length)
        combined_mask = causal_mask[None, None, :, :] & padding_mask
    else:
        # Just causal mask: (1, 1, seq_length, seq_length)
        combined_mask = causal_mask[None, None, :, :]

    return combined_mask


class AutoregressiveTrainer:
    """Autoregressive model trainer with teacher forcing and scheduled sampling.

    This trainer provides a JIT-compatible interface for sequence-to-sequence
    training with support for:
    - Teacher forcing (using ground truth as input)
    - Scheduled sampling (gradually using model predictions)
    - Label smoothing for regularization
    - Causal masking for autoregressive generation

    The train_step method takes model and optimizer as explicit arguments,
    allowing it to be wrapped with nnx.jit for performance.

    The standard training objective is next-token prediction:
        L = -sum_t log P(x_{t+1} | x_{1:t})

    Example (non-JIT):
        ```python
        from artifex.generative_models.training.trainers import (
            AutoregressiveTrainer,
            AutoregressiveTrainingConfig,
        )

        config = AutoregressiveTrainingConfig(
            use_teacher_forcing=True,
            scheduled_sampling="linear",
            label_smoothing=0.1,
        )
        trainer = AutoregressiveTrainer(config)

        # Create model and optimizer separately
        model = TransformerLM(config, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-4))

        # Training loop
        for step, batch in enumerate(data):
            rng, step_rng = jax.random.split(rng)
            loss, metrics = trainer.train_step(
                model, optimizer, batch, step=step, key=step_rng
            )
        ```

    Example (JIT-compiled):
        ```python
        trainer = AutoregressiveTrainer(config)
        jit_step = nnx.jit(trainer.train_step)

        for step, batch in enumerate(data):
            rng, step_rng = jax.random.split(rng)
            loss, metrics = jit_step(model, optimizer, batch, step=step, key=step_rng)
        ```

    Note:
        The model should accept (input_tokens, mask) and return logits
        of shape (batch, seq_length, vocab_size).
    """

    __slots__ = ("config",)

    def __init__(
        self,
        config: AutoregressiveTrainingConfig | None = None,
    ) -> None:
        """Initialize autoregressive trainer.

        Args:
            config: Training configuration.
        """
        self.config = config or AutoregressiveTrainingConfig()

    def get_teacher_forcing_prob(self, step: int) -> float:
        """Compute teacher forcing probability for scheduled sampling.

        Args:
            step: Current training step.

        Returns:
            Probability of using teacher forcing (0 to 1).
        """
        if self.config.scheduled_sampling == "none":
            return 1.0

        # Apply warmup
        if step < self.config.sampling_warmup_steps:
            return 1.0

        # Compute progress through decay
        decay_step = step - self.config.sampling_warmup_steps
        progress = min(1.0, decay_step / max(1, self.config.sampling_decay_steps))

        if self.config.scheduled_sampling == "linear":
            # Linear decay from 1 to min_prob
            prob = 1.0 - progress * (1.0 - self.config.min_teacher_forcing_prob)

        elif self.config.scheduled_sampling == "exponential":
            # Exponential decay: p = k^progress where k = min_prob
            k = max(self.config.min_teacher_forcing_prob, 1e-6)
            prob = k**progress

        elif self.config.scheduled_sampling == "inverse_sigmoid":
            # Inverse sigmoid: smoother transition
            # p = k / (k + exp(progress/k)) where k controls steepness
            k = 0.5
            sigmoid_val = float(k / (k + jnp.exp(progress / k)))
            min_prob = self.config.min_teacher_forcing_prob
            prob = sigmoid_val * (1.0 - min_prob) + min_prob

        else:
            prob = 1.0

        return max(self.config.min_teacher_forcing_prob, prob)

    def apply_label_smoothing(
        self,
        log_probs: jax.Array,
        targets: jax.Array,
    ) -> jax.Array:
        """Apply label smoothing to cross-entropy loss.

        Label smoothing redistributes some probability mass from the target
        class to all other classes, acting as a regularizer.

        Args:
            log_probs: Log probabilities, shape (batch, seq, vocab_size).
            targets: Target token IDs, shape (batch, seq).

        Returns:
            Smoothed cross-entropy loss per position, shape (batch, seq).
        """
        eps = self.config.label_smoothing

        if eps == 0:
            # Standard cross-entropy: -log P(target)
            return -jnp.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)

        # Label smoothing: (1-eps)*target + eps/vocab_size for all classes
        # Loss = -(1-eps)*log P(target) - (eps/vocab)*sum(log P(k))
        target_log_prob = jnp.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)
        mean_log_prob = jnp.mean(log_probs, axis=-1)

        return -(1.0 - eps) * target_log_prob - eps * mean_log_prob

    def compute_loss(
        self,
        model: nnx.Module,
        batch: dict[str, Any],
        step: int = 0,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute autoregressive training loss.

        Standard next-token prediction with optional scheduled sampling.

        Args:
            model: Autoregressive model.
            batch: Batch dict with "input_ids" and optionally "labels".
            step: Current training step (for scheduled sampling).
            key: PRNG key (required for scheduled sampling, optional otherwise).

        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Extract inputs and targets
        input_ids = batch.get("input_ids", batch.get("tokens", batch.get("data")))
        if input_ids is None:
            msg = "Batch must contain 'input_ids', 'tokens', or 'data'"
            raise KeyError(msg)

        # Labels: either provided or shifted input_ids
        labels = batch.get("labels")
        if labels is None:
            # Standard autoregressive: predict next token
            # Input: x[:-1], Target: x[1:]
            labels = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]

        # Create attention mask
        if self.config.use_causal_mask:
            mask = create_combined_mask(input_ids, self.config.pad_token_id)
        else:
            mask = None

        # Teacher forcing vs scheduled sampling
        tf_prob = self.get_teacher_forcing_prob(step)
        use_scheduled_sampling = tf_prob < 1.0 and key is not None

        if use_scheduled_sampling and key is not None:
            # Scheduled sampling: mix teacher forcing with model predictions
            logits = self._forward_with_sampling(model, input_ids, tf_prob, mask, key)
        else:
            # Pure teacher forcing: use ground truth inputs
            logits = model(input_ids, mask=mask) if mask is not None else model(input_ids)

        # Compute loss
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # Cross-entropy with optional label smoothing
        loss_per_position = self.apply_label_smoothing(log_probs, labels)

        # Create mask for valid positions (not padding, not ignore_index)
        valid_mask = labels != self.config.ignore_index
        if self.config.pad_token_id is not None:
            valid_mask = valid_mask & (labels != self.config.pad_token_id)

        # Masked mean
        num_valid = jnp.sum(valid_mask)
        loss = jnp.sum(loss_per_position * valid_mask) / jnp.maximum(num_valid, 1.0)

        # Compute perplexity
        perplexity = jnp.exp(loss)

        # Accuracy
        predictions = jnp.argmax(logits, axis=-1)
        correct = (predictions == labels) & valid_mask
        accuracy = jnp.sum(correct) / jnp.maximum(num_valid, 1.0)

        metrics = {
            "loss": loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "teacher_forcing_prob": tf_prob,
            "num_tokens": num_valid,
        }

        return loss, metrics

    def _forward_with_sampling(
        self,
        model: nnx.Module,
        input_ids: jax.Array,
        tf_prob: float,
        mask: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        """Forward pass with scheduled sampling.

        At each position, randomly choose between using the ground truth
        token (teacher forcing) or the model's prediction.

        Args:
            model: Autoregressive model.
            input_ids: Input token IDs (batch, seq).
            tf_prob: Probability of teacher forcing.
            mask: Attention mask.
            key: PRNG key for sampling.

        Returns:
            Logits of shape (batch, seq, vocab_size).
        """
        batch_size, seq_length = input_ids.shape[:2]

        # Generate random values for deciding teacher forcing
        use_teacher = jax.random.uniform(key, (batch_size, seq_length)) < tf_prob

        # Forward pass with teacher forcing
        if mask is not None:
            logits = model(input_ids, mask=mask)
        else:
            logits = model(input_ids)

        # For scheduled sampling, we would ideally run step by step
        # For efficiency, we use a simplified version where we randomly
        # replace some input positions with predictions from previous step
        # This is an approximation that works well in practice

        # Get predictions from logits
        predictions = jnp.argmax(logits, axis=-1)

        # Mix: use teacher (ground truth shifted) or model prediction
        # For next-token prediction, the "input" at position t should be
        # either labels[t-1] (teacher) or predictions[t-1] (model)
        mixed_inputs = jnp.where(use_teacher, input_ids, predictions)

        # Always re-run with mixed inputs (JIT-compatible: no Python if on traced values)
        if mask is not None:
            logits = model(mixed_inputs, mask=mask)
        else:
            logits = model(mixed_inputs)

        return logits

    def train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, Any],
        step: int = 0,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Execute a single training step.

        This method can be wrapped with nnx.jit for performance:
            jit_step = nnx.jit(trainer.train_step)
            loss, metrics = jit_step(model, optimizer, batch, step=step, key=key)

        Args:
            model: Autoregressive model to train.
            optimizer: NNX optimizer for parameter updates.
            batch: Batch dictionary with token IDs.
            step: Current training step (for scheduled sampling).
            key: PRNG key (required for scheduled sampling).

        Returns:
            Tuple of (loss, metrics_dict).
        """

        def loss_fn(m: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
            return self.compute_loss(m, batch, step, key)

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)

        return loss, metrics

    def create_loss_fn(
        self,
        step: int = 0,
    ) -> Callable[[nnx.Module, dict[str, Any], jax.Array], tuple[jax.Array, dict[str, Any]]]:
        """Create loss function compatible with base Trainer.

        Args:
            step: Training step for scheduled sampling schedule.

        Returns:
            Function with signature: (model, batch, rng) -> (loss, metrics)
        """

        def loss_fn(
            model: nnx.Module,
            batch: dict[str, Any],
            rng: jax.Array,
        ) -> tuple[jax.Array, dict[str, Any]]:
            return self.compute_loss(model, batch, step, rng)

        return loss_fn

    def generate(
        self,
        model: nnx.Module,
        prompt: jax.Array,
        max_length: int,
        key: jax.Array,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> jax.Array:
        """Generate tokens autoregressively.

        Args:
            model: Autoregressive model for generation.
            prompt: Initial token IDs, shape (batch, prompt_length).
            max_length: Maximum sequence length to generate.
            key: PRNG key for sampling.
            temperature: Sampling temperature (higher = more random).
            top_k: Keep only top-k tokens for sampling (None = all).
            top_p: Nucleus sampling threshold (None = disabled).
            eos_token_id: Stop generation at this token.

        Returns:
            Generated token IDs, shape (batch, total_length).
        """
        batch_size, prompt_length = prompt.shape

        # Pre-allocate fixed-size output buffer (avoids shape changes per step)
        # Fill with pad token or 0
        pad_value = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        tokens = jnp.full((batch_size, max_length), pad_value, dtype=prompt.dtype)
        tokens = tokens.at[:, :prompt_length].set(prompt)

        # Track which sequences are still generating
        use_causal = self.config.use_causal_mask
        pad_token_id = self.config.pad_token_id

        for i in range(max_length - prompt_length):
            step_key = jax.random.fold_in(key, i)
            current_len = prompt_length + i

            # Use only the valid prefix for the forward pass
            current_tokens = tokens[:, :current_len]

            # Forward pass
            if use_causal:
                mask = create_combined_mask(current_tokens, pad_token_id)
                logits = model(current_tokens, mask=mask)
            else:
                logits = model(current_tokens)

            # Get logits for the last position
            next_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_values, _ = jax.lax.top_k(next_logits, top_k)
                threshold = top_k_values[:, -1:]
                next_logits = jnp.where(
                    next_logits >= threshold,
                    next_logits,
                    jnp.full_like(next_logits, -1e10),
                )

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits = jnp.sort(next_logits, axis=-1)[:, ::-1]
                sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
                cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)

                cutoff_idx = jnp.sum(cumsum_probs <= top_p, axis=-1, keepdims=True)
                cutoff_logit = jnp.take_along_axis(sorted_logits, cutoff_idx, axis=-1)

                next_logits = jnp.where(
                    next_logits >= cutoff_logit,
                    next_logits,
                    jnp.full_like(next_logits, -1e10),
                )

            # Sample next token
            probs = jax.nn.softmax(next_logits, axis=-1)
            next_token = jax.random.categorical(step_key, jnp.log(probs + 1e-10))

            # Write to pre-allocated buffer (no shape change)
            tokens = tokens.at[:, current_len].set(next_token)

            # Early stopping on EOS (Python-level check, safe outside JIT)
            if eos_token_id is not None and jnp.all(next_token == eos_token_id):
                break

        return tokens
