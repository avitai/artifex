"""Comprehensive tests for CLIP Text Encoder.

This module tests the CLIP-like text encoder used in Stable Diffusion,
following Test-Driven Development (TDD) principles.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.models.diffusion.clip_text_encoder import CLIPTextEncoder


class TestCLIPTextEncoderInitialization:
    """Test CLIP text encoder initialization."""

    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        rngs = nnx.Rngs(0)

        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=6,
            num_heads=8,
            rngs=rngs,
        )

        assert encoder is not None
        assert hasattr(encoder, "token_embedding")
        assert hasattr(encoder, "positional_embedding")
        assert hasattr(encoder, "transformer")
        assert hasattr(encoder, "ln_final")

    def test_initialization_clip_config(self):
        """Test initialization with CLIP-standard configuration."""
        rngs = nnx.Rngs(42)

        encoder = CLIPTextEncoder(
            vocab_size=49408,  # CLIP vocab size
            max_length=77,  # CLIP sequence length
            embedding_dim=768,  # CLIP text width
            num_layers=12,  # CLIP layers
            num_heads=12,  # CLIP heads
            rngs=rngs,
        )

        assert encoder is not None

    def test_initialization_minimal_config(self):
        """Test initialization with minimal configuration."""
        rngs = nnx.Rngs(123)

        encoder = CLIPTextEncoder(
            vocab_size=100,
            max_length=10,
            embedding_dim=64,
            num_layers=2,
            num_heads=2,
            rngs=rngs,
        )

        assert encoder is not None

    def test_initialization_different_seeds(self):
        """Test that different seeds produce different initializations."""
        rngs1 = nnx.Rngs(0)
        rngs2 = nnx.Rngs(1)

        encoder1 = CLIPTextEncoder(
            vocab_size=1000, max_length=77, embedding_dim=512, num_layers=4, num_heads=8, rngs=rngs1
        )
        encoder2 = CLIPTextEncoder(
            vocab_size=1000, max_length=77, embedding_dim=512, num_layers=4, num_heads=8, rngs=rngs2
        )

        # Get token embedding weights
        token_emb1 = encoder1.token_embedding.embedding.value
        token_emb2 = encoder2.token_embedding.embedding.value

        assert not jnp.allclose(token_emb1, token_emb2)


class TestCLIPTextEncoderForwardPass:
    """Test CLIP text encoder forward pass."""

    @pytest.fixture
    def encoder(self):
        """Create a test encoder."""
        rngs = nnx.Rngs(0)
        return CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=4,
            num_heads=8,
            rngs=rngs,
        )

    def test_forward_pass_basic(self, encoder):
        """Test basic forward pass."""
        batch_size = 2
        seq_len = 77

        # Create random token IDs
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)

        # Forward pass
        embeddings = encoder(token_ids)

        # Check output shape
        assert embeddings.shape == (batch_size, seq_len, 512)
        assert embeddings.dtype == jnp.float32

    def test_forward_pass_different_batch_sizes(self, encoder):
        """Test forward pass with different batch sizes."""
        seq_len = 77

        for batch_size in [1, 2, 4, 8]:
            token_ids = jax.random.randint(
                jax.random.key(batch_size), (batch_size, seq_len), 0, 1000
            )
            embeddings = encoder(token_ids)
            assert embeddings.shape == (batch_size, seq_len, 512)

    def test_forward_pass_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        rngs = nnx.Rngs(0)
        max_length = 100

        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=max_length,
            embedding_dim=512,
            num_layers=4,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2

        for seq_len in [10, 25, 50, 77, 100]:
            token_ids = jax.random.randint(jax.random.key(seq_len), (batch_size, seq_len), 0, 1000)
            embeddings = encoder(token_ids)
            assert embeddings.shape == (batch_size, seq_len, 512)

    def test_forward_pass_with_padding(self, encoder):
        """Test forward pass with padded sequences."""
        batch_size = 2
        seq_len = 77

        # Create token IDs with padding (0 is typically padding token)
        token_ids = jnp.array([[1, 2, 3, 4, 0, 0, 0] + [0] * 70, [5, 6, 7, 0, 0, 0, 0] + [0] * 70])

        embeddings = encoder(token_ids)
        assert embeddings.shape == (batch_size, seq_len, 512)

    def test_forward_pass_with_attention_mask(self, encoder):
        """Test forward pass with attention mask."""
        batch_size = 2
        seq_len = 77

        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = jnp.ones((batch_size, seq_len))
        attention_mask = attention_mask.at[0, 50:].set(0)  # Mask second half of first sequence
        attention_mask = attention_mask.at[1, 30:].set(0)  # Mask last part of second sequence

        embeddings = encoder(token_ids, attention_mask=attention_mask)

        assert embeddings.shape == (batch_size, seq_len, 512)

        # Check that masked positions have different values than unmasked
        # (they should be zeroed out or have reduced magnitude)
        unmasked_norm_0 = jnp.linalg.norm(embeddings[0, :50], axis=-1).mean()
        masked_norm_0 = jnp.linalg.norm(embeddings[0, 50:], axis=-1).mean()

        # Masked positions should have smaller magnitude
        assert masked_norm_0 < unmasked_norm_0

    def test_forward_pass_deterministic(self, encoder):
        """Test that forward pass is deterministic."""
        batch_size = 2
        seq_len = 77
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)

        # Multiple forward passes with same input
        embeddings1 = encoder(token_ids)
        embeddings2 = encoder(token_ids)
        embeddings3 = encoder(token_ids)

        # All outputs should be identical
        assert jnp.allclose(embeddings1, embeddings2)
        assert jnp.allclose(embeddings2, embeddings3)

    def test_forward_pass_different_inputs(self, encoder):
        """Test that different inputs produce different outputs."""
        batch_size = 2
        seq_len = 77

        token_ids1 = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)
        token_ids2 = jax.random.randint(jax.random.key(1), (batch_size, seq_len), 0, 1000)

        embeddings1 = encoder(token_ids1)
        embeddings2 = encoder(token_ids2)

        # Different inputs should produce different outputs
        assert not jnp.allclose(embeddings1, embeddings2)


class TestCLIPTextEncoderOutputProperties:
    """Test output properties of CLIP text encoder."""

    @pytest.fixture
    def encoder(self):
        """Create a test encoder."""
        rngs = nnx.Rngs(0)
        return CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=4,
            num_heads=8,
            rngs=rngs,
        )

    def test_output_value_range(self, encoder):
        """Test that output values are in reasonable range."""
        batch_size = 2
        seq_len = 77
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)

        embeddings = encoder(token_ids)

        # Check that values are not too large or too small
        assert jnp.abs(embeddings).max() < 100.0
        assert jnp.abs(embeddings).mean() < 10.0

    def test_output_not_nan_or_inf(self, encoder):
        """Test that output doesn't contain NaN or Inf."""
        batch_size = 2
        seq_len = 77
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)

        embeddings = encoder(token_ids)

        assert not jnp.isnan(embeddings).any()
        assert not jnp.isinf(embeddings).any()

    def test_output_distribution(self, encoder):
        """Test that output has reasonable distribution."""
        batch_size = 8
        seq_len = 77
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)

        embeddings = encoder(token_ids)

        # Check mean and std are reasonable
        mean = embeddings.mean()
        std = embeddings.std()

        assert jnp.abs(mean) < 1.0  # Mean should be close to zero
        assert 0.1 < std < 10.0  # Std should be reasonable


class TestCLIPTextEncoderEdgeCases:
    """Test edge cases for CLIP text encoder."""

    def test_single_token_sequence(self):
        """Test with sequence length of 1."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=1,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        token_ids = jnp.array([[42]])
        embeddings = encoder(token_ids)

        assert embeddings.shape == (1, 1, 512)

    def test_max_length_sequence(self):
        """Test with maximum sequence length."""
        rngs = nnx.Rngs(0)
        max_length = 200

        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=max_length,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        token_ids = jax.random.randint(jax.random.key(0), (1, max_length), 0, 1000)
        embeddings = encoder(token_ids)

        assert embeddings.shape == (1, max_length, 512)

    def test_all_padding_tokens(self):
        """Test with all padding tokens."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        # All zeros (padding)
        token_ids = jnp.zeros((2, 77), dtype=jnp.int32)
        embeddings = encoder(token_ids)

        assert embeddings.shape == (2, 77, 512)
        assert not jnp.isnan(embeddings).any()

    def test_repeated_tokens(self):
        """Test with repeated tokens."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        # All same token
        token_ids = jnp.full((2, 77), fill_value=42, dtype=jnp.int32)
        embeddings = encoder(token_ids)

        assert embeddings.shape == (2, 77, 512)
        # Different positions should still have different embeddings due to positional encoding
        assert not jnp.allclose(embeddings[:, 0, :], embeddings[:, 1, :])


class TestCLIPTextEncoderGradients:
    """Test gradient computation for CLIP text encoder."""

    def test_gradients_computable(self):
        """Test that gradients can be computed."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        seq_len = 77
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)

        def loss_fn(model):
            embeddings = model(token_ids)
            return jnp.mean(embeddings**2)

        # Compute gradients
        loss, grads = nnx.value_and_grad(loss_fn)(encoder)

        assert isinstance(loss, jax.Array)
        assert grads is not None

    def test_gradients_not_nan(self):
        """Test that gradients are not NaN."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        seq_len = 77
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)

        def loss_fn(model):
            embeddings = model(token_ids)
            return jnp.mean(embeddings**2)

        loss, grads = nnx.value_and_grad(loss_fn)(encoder)

        # Check gradients - use jax.tree.leaves to get all gradient arrays
        grad_leaves = jax.tree.leaves(grads)

        # Check that gradients were computed
        assert len(grad_leaves) > 0, "No gradients computed"

        # Check all gradient values for NaN
        for i, grad_value in enumerate(grad_leaves):
            if isinstance(grad_value, jax.Array):
                assert not jnp.isnan(grad_value).any(), f"NaN gradient in leaf {i}"


class TestCLIPTextEncoderTrainEvalModes:
    """Test train/eval mode switching."""

    def test_eval_mode_default(self):
        """Test that encoder is in eval mode by default."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        # Should work without issues in eval mode
        token_ids = jax.random.randint(jax.random.key(0), (2, 77), 0, 1000)
        embeddings = encoder(token_ids)

        assert embeddings.shape == (2, 77, 512)

    def test_train_mode(self):
        """Test encoder in train mode."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        # Switch to train mode
        encoder.train()

        token_ids = jax.random.randint(jax.random.key(0), (2, 77), 0, 1000)
        embeddings = encoder(token_ids)

        assert embeddings.shape == (2, 77, 512)

    def test_mode_switching(self):
        """Test switching between train and eval modes."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        token_ids = jax.random.randint(jax.random.key(0), (2, 77), 0, 1000)

        # Eval mode
        encoder.eval()
        emb_eval1 = encoder(token_ids)

        # Train mode
        encoder.train()
        encoder(token_ids)

        # Back to eval mode
        encoder.eval()
        emb_eval2 = encoder(token_ids)

        # Eval mode outputs should be identical
        assert jnp.allclose(emb_eval1, emb_eval2)


class TestCLIPTextEncoderJITCompatibility:
    """Test JIT compilation compatibility."""

    def test_jit_forward_pass(self):
        """Test that forward pass can be JIT compiled."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=4,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        seq_len = 77
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)

        # JIT compile the forward pass
        @nnx.jit
        def forward(model, inputs):
            return model(inputs)

        # First call (compilation)
        embeddings1 = forward(encoder, token_ids)
        assert embeddings1.shape == (batch_size, seq_len, 512)

        # Second call (should use cached compilation)
        embeddings2 = forward(encoder, token_ids)
        assert jnp.allclose(embeddings1, embeddings2)

    def test_jit_with_different_batch_sizes(self):
        """Test JIT with different batch sizes (should recompile)."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        @nnx.jit
        def forward(model, inputs):
            return model(inputs)

        # Different batch sizes
        for batch_size in [1, 2, 4]:
            token_ids = jax.random.randint(jax.random.key(batch_size), (batch_size, 77), 0, 1000)
            embeddings = forward(encoder, token_ids)
            assert embeddings.shape == (batch_size, 77, 512)

    def test_jit_with_mask(self):
        """Test JIT compilation with attention mask."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        seq_len = 77

        @nnx.jit
        def forward_with_mask(model, inputs, mask):
            return model(inputs, attention_mask=mask)

        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)
        attention_mask = jnp.ones((batch_size, seq_len))

        embeddings = forward_with_mask(encoder, token_ids, attention_mask)
        assert embeddings.shape == (batch_size, seq_len, 512)

    def test_jit_gradient_computation(self):
        """Test that gradient computation can be JIT compiled."""
        rngs = nnx.Rngs(0)
        encoder = CLIPTextEncoder(
            vocab_size=1000,
            max_length=77,
            embedding_dim=512,
            num_layers=2,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        seq_len = 77
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 1000)

        @nnx.jit
        def compute_loss_and_grads(model, inputs):
            def loss_fn(m):
                embeddings = m(inputs)
                return jnp.mean(embeddings**2)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            return loss, grads

        # Compile and run
        loss, grads = compute_loss_and_grads(encoder, token_ids)

        assert isinstance(loss, jax.Array)
        assert loss.shape == ()
        assert grads is not None

        # Verify gradients are valid
        grad_leaves = jax.tree.leaves(grads)
        assert len(grad_leaves) > 0
        for grad_value in grad_leaves:
            if isinstance(grad_value, jax.Array):
                assert not jnp.isnan(grad_value).any()
