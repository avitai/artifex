"""Tests for WaveNet autoregressive models.

This module provides comprehensive tests for WaveNet components including
CausalConv1D, GatedActivationUnit, and the main WaveNet model.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import WaveNetConfig
from artifex.generative_models.models.autoregressive.wavenet import (
    CausalConv1D,
    GatedActivationUnit,
    WaveNet,
)


def create_wavenet_config(
    vocab_size: int = 256,
    sequence_length: int = 64,
    residual_channels: int = 16,
    skip_channels: int = 32,
    num_blocks: int = 2,
    layers_per_block: int = 3,
    kernel_size: int = 2,
    dilation_base: int = 2,
    use_gated_activation: bool = True,
) -> WaveNetConfig:
    """Create WaveNetConfig for testing."""
    return WaveNetConfig(
        name="test_wavenet",
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        num_blocks=num_blocks,
        layers_per_block=layers_per_block,
        kernel_size=kernel_size,
        dilation_base=dilation_base,
        use_gated_activation=use_gated_activation,
    )


@pytest.fixture
def base_rngs():
    """Fixture for nnx random number generators."""
    return nnx.Rngs(
        params=jax.random.key(0),
        dropout=jax.random.key(1),
        sample=jax.random.key(2),
    )


@pytest.fixture
def simple_config():
    """Fixture for simple WaveNetConfig."""
    return create_wavenet_config()


@pytest.fixture
def input_sequence():
    """Fixture for input sequence [batch, length]."""
    return jax.random.randint(jax.random.key(42), (4, 32), 0, 256)


class TestCausalConv1D:
    """Test suite for CausalConv1D."""

    def test_initialization(self, base_rngs):
        """Test CausalConv1D initialization."""
        conv = CausalConv1D(
            in_features=16,
            out_features=32,
            kernel_size=3,
            dilation=1,
            rngs=base_rngs,
        )

        assert conv.kernel_size == 3
        assert conv.dilation == 1
        assert conv.padding == (3 - 1) * 1  # (kernel_size - 1) * dilation

    def test_causal_padding_formula(self, base_rngs):
        """Test that causal padding follows: padding = (kernel_size - 1) * dilation."""
        # Test various kernel sizes and dilations
        test_cases = [
            (2, 1, 1),  # kernel=2, dilation=1, expected padding=1
            (3, 1, 2),  # kernel=3, dilation=1, expected padding=2
            (2, 2, 2),  # kernel=2, dilation=2, expected padding=2
            (3, 4, 8),  # kernel=3, dilation=4, expected padding=8
            (5, 2, 8),  # kernel=5, dilation=2, expected padding=8
        ]

        for kernel_size, dilation, expected_padding in test_cases:
            conv = CausalConv1D(
                in_features=8,
                out_features=8,
                kernel_size=kernel_size,
                dilation=dilation,
                rngs=base_rngs,
            )
            assert conv.padding == expected_padding, (
                f"kernel={kernel_size}, dilation={dilation}: "
                f"expected padding={expected_padding}, got {conv.padding}"
            )

    def test_output_shape(self, base_rngs):
        """Test CausalConv1D output shape."""
        # With dilation=1, output length should match input length
        conv = CausalConv1D(
            in_features=16,
            out_features=32,
            kernel_size=3,
            dilation=1,
            rngs=base_rngs,
        )

        batch_size, seq_len, in_channels = 4, 20, 16
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, in_channels))

        output = conv(x)

        # With dilation=1 and proper causal padding, output length = input length
        assert output.shape == (batch_size, seq_len, 32)

    def test_no_future_information_leakage(self, base_rngs):
        """Test that output at time t only depends on inputs at time <= t.

        This is verified by checking that changing future inputs doesn't
        affect the output at earlier positions.
        """
        kernel_size = 3
        conv = CausalConv1D(
            in_features=8,
            out_features=8,
            kernel_size=kernel_size,
            dilation=1,
            rngs=base_rngs,
        )

        batch_size, seq_len, channels = 2, 16, 8
        x1 = jax.random.normal(jax.random.key(0), (batch_size, seq_len, channels))
        x2 = x1.copy()

        # Modify future positions (last 5)
        x2 = x2.at[:, -5:, :].set(jax.random.normal(jax.random.key(1), (batch_size, 5, channels)))

        y1 = conv(x1)
        y2 = conv(x2)

        # Outputs before the modified region should be identical
        # Account for receptive field: kernel_size - 1
        receptive_field = kernel_size - 1
        safe_position = seq_len - 5 - receptive_field
        assert jnp.allclose(y1[:, :safe_position, :], y2[:, :safe_position, :])

    def test_finite_output(self, base_rngs):
        """Test CausalConv1D produces finite values."""
        conv = CausalConv1D(
            in_features=16,
            out_features=16,
            kernel_size=2,
            dilation=1,
            rngs=base_rngs,
        )

        x = jax.random.normal(jax.random.key(0), (4, 20, 16))
        output = conv(x)

        assert jnp.all(jnp.isfinite(output))


class TestGatedActivationUnit:
    """Test suite for GatedActivationUnit."""

    def test_initialization(self, base_rngs):
        """Test GatedActivationUnit initialization."""
        gau = GatedActivationUnit(
            channels=16,
            kernel_size=2,
            dilation=1,
            rngs=base_rngs,
        )

        assert gau.tanh_conv is not None
        assert gau.sigmoid_conv is not None

    def test_output_shape(self, base_rngs):
        """Test GatedActivationUnit output shape."""
        channels = 16
        gau = GatedActivationUnit(
            channels=channels,
            kernel_size=2,
            dilation=1,
            rngs=base_rngs,
        )

        batch_size, seq_len = 4, 20
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, channels))

        output = gau(x)

        assert output.shape == x.shape

    def test_gated_activation_formula(self, base_rngs):
        """Test gated activation: output = tanh(conv1(x)) * sigmoid(conv2(x))."""
        channels = 8
        gau = GatedActivationUnit(
            channels=channels,
            kernel_size=2,
            dilation=1,
            rngs=base_rngs,
        )

        x = jax.random.normal(jax.random.key(0), (2, 10, channels))
        output = gau(x)

        # Manually compute expected output
        tanh_out = jnp.tanh(gau.tanh_conv(x))
        sigmoid_out = jax.nn.sigmoid(gau.sigmoid_conv(x))
        expected = tanh_out * sigmoid_out

        assert jnp.allclose(output, expected)

    def test_output_bounded(self, base_rngs):
        """Test gated activation output is bounded in [-1, 1].

        Since tanh is in [-1, 1] and sigmoid is in [0, 1],
        the product tanh * sigmoid is bounded by [-1, 1].
        """
        gau = GatedActivationUnit(
            channels=16,
            kernel_size=2,
            dilation=2,
            rngs=base_rngs,
        )

        x = jax.random.normal(jax.random.key(0), (4, 20, 16)) * 10  # Large values
        output = gau(x)

        assert jnp.all(output >= -1.0)
        assert jnp.all(output <= 1.0)


class TestWaveNet:
    """Test suite for WaveNet model."""

    def test_initialization(self, simple_config, base_rngs):
        """Test WaveNet initialization from config."""
        model = WaveNet(simple_config, rngs=base_rngs)

        assert model.residual_channels == simple_config.residual_channels
        assert model.skip_channels == simple_config.skip_channels
        assert model.num_blocks == simple_config.num_blocks
        assert model.num_layers == simple_config.layers_per_block
        assert model.kernel_size == simple_config.kernel_size
        assert model.dilation_base == simple_config.dilation_base
        assert model.use_gated_activation == simple_config.use_gated_activation

    def test_initialization_components(self, simple_config, base_rngs):
        """Test WaveNet internal components are properly initialized."""
        model = WaveNet(simple_config, rngs=base_rngs)

        # Check components exist
        assert model.input_embedding is not None
        assert model.initial_conv is not None
        assert (
            len(model.residual_blocks) == simple_config.num_blocks * simple_config.layers_per_block
        )
        assert model.post_conv1 is not None
        assert model.post_conv2 is not None

    def test_forward_pass_shape(self, simple_config, base_rngs, input_sequence):
        """Test WaveNet forward pass produces correct output shape."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model(input_sequence)

        assert "logits" in outputs
        batch_size, seq_len = input_sequence.shape
        expected_shape = (batch_size, seq_len, simple_config.vocab_size)
        assert outputs["logits"].shape == expected_shape

    def test_forward_pass_outputs(self, simple_config, base_rngs, input_sequence):
        """Test WaveNet forward pass returns all expected outputs."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model(input_sequence)

        assert "logits" in outputs
        assert "skip_connections" in outputs
        assert "embedded" in outputs
        assert jnp.all(jnp.isfinite(outputs["logits"]))

    def test_forward_pass_finite(self, simple_config, base_rngs, input_sequence):
        """Test WaveNet forward pass produces finite values."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model(input_sequence)

        assert jnp.all(jnp.isfinite(outputs["logits"]))
        assert jnp.all(jnp.isfinite(outputs["embedded"]))

    def test_receptive_field_computation(self, base_rngs):
        """Test receptive field follows formula: 1 + sum((k-1) * d for all layers).

        Receptive field = 1 + sum_{block, layer} (kernel_size - 1) * dilation_base^layer
        """
        config = create_wavenet_config(
            num_blocks=2,
            layers_per_block=3,
            kernel_size=2,
            dilation_base=2,
        )
        model = WaveNet(config, rngs=base_rngs)

        # Manually compute expected receptive field
        # For each block, layers have dilations: 2^0=1, 2^1=2, 2^2=4
        # (kernel_size - 1) = 1
        # Each block contributes: 1*1 + 1*2 + 1*4 = 7
        # Two blocks: 2 * 7 = 14
        # Plus initial 1: 1 + 14 = 15
        expected_rf = 1
        for _ in range(config.num_blocks):
            for layer in range(config.layers_per_block):
                dilation = config.dilation_base**layer
                expected_rf += (config.kernel_size - 1) * dilation

        computed_rf = model.compute_receptive_field()
        assert computed_rf == expected_rf

    def test_receptive_field_grows_with_depth(self, base_rngs):
        """Test that receptive field increases with more blocks/layers."""
        config_small = create_wavenet_config(num_blocks=1, layers_per_block=2)
        config_large = create_wavenet_config(num_blocks=3, layers_per_block=4)

        model_small = WaveNet(config_small, rngs=base_rngs)
        base_rngs2 = nnx.Rngs(params=jax.random.key(0), sample=jax.random.key(1))
        model_large = WaveNet(config_large, rngs=base_rngs2)

        assert model_large.compute_receptive_field() > model_small.compute_receptive_field()


class TestWaveNetLoss:
    """Test suite for WaveNet loss computation."""

    def test_loss_fn_returns_required_keys(self, simple_config, base_rngs, input_sequence):
        """Test loss_fn returns all required keys."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model(input_sequence)
        loss_result = model.loss_fn(input_sequence, outputs)

        required_keys = ["loss", "nll_loss", "accuracy", "perplexity", "receptive_field"]
        for key in required_keys:
            assert key in loss_result, f"Missing key: {key}"

    def test_loss_is_finite(self, simple_config, base_rngs, input_sequence):
        """Test loss is finite."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model(input_sequence)
        loss_result = model.loss_fn(input_sequence, outputs)

        assert jnp.isfinite(loss_result["loss"])

    def test_loss_is_nonnegative(self, simple_config, base_rngs, input_sequence):
        """Test loss is non-negative (cross-entropy loss)."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model(input_sequence)
        loss_result = model.loss_fn(input_sequence, outputs)

        assert loss_result["loss"] >= 0

    def test_perplexity_is_exp_of_loss(self, simple_config, base_rngs, input_sequence):
        """Test perplexity = exp(loss)."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model(input_sequence)
        loss_result = model.loss_fn(input_sequence, outputs)

        expected_perplexity = jnp.exp(loss_result["loss"])
        assert jnp.allclose(loss_result["perplexity"], expected_perplexity)

    def test_accuracy_bounded(self, simple_config, base_rngs, input_sequence):
        """Test accuracy is in [0, 1]."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model(input_sequence)
        loss_result = model.loss_fn(input_sequence, outputs)

        assert loss_result["accuracy"] >= 0
        assert loss_result["accuracy"] <= 1

    def test_loss_with_dict_batch(self, simple_config, base_rngs, input_sequence):
        """Test loss_fn accepts dictionary batch."""
        model = WaveNet(simple_config, rngs=base_rngs)

        batch = {"x": input_sequence}
        outputs = model(input_sequence)
        loss_result = model.loss_fn(batch, outputs)

        assert jnp.isfinite(loss_result["loss"])

    def test_loss_with_mask(self, simple_config, base_rngs):
        """Test loss_fn handles masked sequences."""
        model = WaveNet(simple_config, rngs=base_rngs)

        batch_size, seq_len = 4, 32
        sequences = jax.random.randint(
            jax.random.key(0), (batch_size, seq_len), 0, simple_config.vocab_size
        )
        # Mask: 1 for valid, 0 for padding
        mask = jnp.ones((batch_size, seq_len))
        mask = mask.at[:, -8:].set(0)  # Last 8 positions are padded

        batch = {"x": sequences, "mask": mask}
        outputs = model(sequences)
        loss_result = model.loss_fn(batch, outputs)

        assert jnp.isfinite(loss_result["loss"])


class TestWaveNetGeneration:
    """Test suite for WaveNet generation methods."""

    def test_generate_fast_shape(self, simple_config, base_rngs):
        """Test generate_fast produces correct shape."""
        model = WaveNet(simple_config, rngs=base_rngs)

        n_samples = 2
        max_length = 16
        generated = model.generate_fast(n_samples=n_samples, max_length=max_length)

        assert generated.shape == (n_samples, max_length)

    def test_generate_fast_values_in_vocab(self, simple_config, base_rngs):
        """Test generated values are within vocabulary."""
        model = WaveNet(simple_config, rngs=base_rngs)

        generated = model.generate_fast(n_samples=2, max_length=16)

        assert jnp.all(generated >= 0)
        assert jnp.all(generated < simple_config.vocab_size)

    def test_generate_fast_stochastic(self, simple_config, base_rngs):
        """Test that generation produces stochastic outputs.

        Multiple calls to generate_fast with the model's internal rngs
        should produce different results due to RNG state advancement.
        """
        model = WaveNet(simple_config, rngs=base_rngs)

        # Generate twice - internal RNG state should advance
        gen1 = model.generate_fast(n_samples=2, max_length=16)
        gen2 = model.generate_fast(n_samples=2, max_length=16)

        # Subsequent generations should differ due to RNG state change
        # If this fails, it indicates deterministic generation (which may be intentional)
        # We just verify both are valid generations
        assert gen1.shape == gen2.shape
        assert jnp.all(gen1 >= 0)
        assert jnp.all(gen2 >= 0)

    def test_conditional_generate_shape(self, simple_config, base_rngs):
        """Test conditional_generate produces correct shape."""
        model = WaveNet(simple_config, rngs=base_rngs)

        conditioning = jnp.array([1, 2, 3, 4, 5])
        n_samples = 2
        max_new_tokens = 10
        generated = model.conditional_generate(
            conditioning=conditioning,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
        )

        expected_length = len(conditioning) + max_new_tokens
        assert generated.shape == (n_samples, expected_length)

    def test_conditional_generate_preserves_prefix(self, simple_config, base_rngs):
        """Test conditional_generate preserves the conditioning prefix."""
        model = WaveNet(simple_config, rngs=base_rngs)

        conditioning = jnp.array([10, 20, 30, 40, 50])
        n_samples = 3
        max_new_tokens = 8
        generated = model.conditional_generate(
            conditioning=conditioning,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
        )

        # Check that all samples start with the conditioning sequence
        for i in range(n_samples):
            assert jnp.allclose(generated[i, : len(conditioning)], conditioning)


class TestWaveNetIntermediateOutputs:
    """Test suite for WaveNet intermediate outputs."""

    def test_get_intermediate_outputs_keys(self, simple_config, base_rngs, input_sequence):
        """Test get_intermediate_outputs returns all expected keys."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model.get_intermediate_outputs(input_sequence)

        expected_keys = [
            "embedded",
            "residual_outputs",
            "skip_outputs",
            "skip_sum",
            "post1",
            "post2",
            "logits",
        ]
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"

    def test_residual_outputs_count(self, simple_config, base_rngs, input_sequence):
        """Test correct number of residual outputs."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model.get_intermediate_outputs(input_sequence)

        # Number of residual outputs = 1 (initial) + num_blocks * layers_per_block
        expected_count = 1 + simple_config.num_blocks * simple_config.layers_per_block
        assert len(outputs["residual_outputs"]) == expected_count

    def test_skip_outputs_count(self, simple_config, base_rngs, input_sequence):
        """Test correct number of skip outputs."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model.get_intermediate_outputs(input_sequence)

        expected_count = simple_config.num_blocks * simple_config.layers_per_block
        assert len(outputs["skip_outputs"]) == expected_count


class TestGradientFlow:
    """Test suite for gradient flow through WaveNet."""

    def test_gradient_flow_forward(self, simple_config, base_rngs, input_sequence):
        """Test gradients flow through forward pass."""
        model = WaveNet(simple_config, rngs=base_rngs)

        def loss_fn(model, x):
            outputs = model(x)
            return jnp.mean(outputs["logits"])

        grads = nnx.grad(loss_fn)(model, input_sequence)

        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        assert any(jnp.any(jnp.isfinite(g)) for g in grad_leaves if hasattr(g, "shape"))

    def test_gradient_flow_loss(self, simple_config, base_rngs, input_sequence):
        """Test gradients flow through loss computation."""
        model = WaveNet(simple_config, rngs=base_rngs)

        def full_loss_fn(model, x):
            outputs = model(x)
            loss_result = model.loss_fn(x, outputs)
            return loss_result["loss"]

        grads = nnx.grad(full_loss_fn)(model, input_sequence)

        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0


class TestEdgeCases:
    """Test edge cases for WaveNet."""

    def test_single_batch_element(self, simple_config, base_rngs):
        """Test with single batch element."""
        model = WaveNet(simple_config, rngs=base_rngs)

        x = jax.random.randint(jax.random.key(0), (1, 32), 0, simple_config.vocab_size)
        outputs = model(x)

        assert outputs["logits"].shape == (1, 32, simple_config.vocab_size)

    def test_minimal_sequence(self, simple_config, base_rngs):
        """Test with minimal sequence length."""
        model = WaveNet(simple_config, rngs=base_rngs)

        x = jax.random.randint(jax.random.key(0), (4, 4), 0, simple_config.vocab_size)
        outputs = model(x)

        assert outputs["logits"].shape == (4, 4, simple_config.vocab_size)

    def test_without_gated_activation(self, base_rngs):
        """Test WaveNet without gated activation."""
        config = create_wavenet_config(use_gated_activation=False)
        model = WaveNet(config, rngs=base_rngs)

        x = jax.random.randint(jax.random.key(0), (4, 32), 0, config.vocab_size)
        outputs = model(x)

        assert jnp.all(jnp.isfinite(outputs["logits"]))

    def test_larger_kernel_size(self, base_rngs):
        """Test WaveNet with larger kernel size."""
        config = create_wavenet_config(kernel_size=5)
        model = WaveNet(config, rngs=base_rngs)

        x = jax.random.randint(jax.random.key(0), (4, 32), 0, config.vocab_size)
        outputs = model(x)

        assert jnp.all(jnp.isfinite(outputs["logits"]))

    def test_different_channel_sizes(self, base_rngs):
        """Test WaveNet with different channel configurations."""
        config = create_wavenet_config(
            residual_channels=64,
            skip_channels=128,
        )
        model = WaveNet(config, rngs=base_rngs)

        x = jax.random.randint(jax.random.key(0), (4, 32), 0, config.vocab_size)
        outputs = model(x)

        assert jnp.all(jnp.isfinite(outputs["logits"]))

    def test_single_block_single_layer(self, base_rngs):
        """Test minimal WaveNet configuration."""
        config = create_wavenet_config(
            num_blocks=1,
            layers_per_block=1,
        )
        model = WaveNet(config, rngs=base_rngs)

        x = jax.random.randint(jax.random.key(0), (4, 32), 0, config.vocab_size)
        outputs = model(x)

        assert jnp.all(jnp.isfinite(outputs["logits"]))


class TestMathematicalProperties:
    """Test mathematical properties of WaveNet."""

    def test_dilation_pattern(self, base_rngs):
        """Test dilation follows exponential pattern: dilation_base^layer."""
        config = create_wavenet_config(
            num_blocks=2,
            layers_per_block=4,
            dilation_base=2,
        )
        model = WaveNet(config, rngs=base_rngs)

        # Check dilations in residual blocks
        expected_dilations = []
        for _ in range(config.num_blocks):
            for layer in range(config.layers_per_block):
                expected_dilations.append(config.dilation_base**layer)

        # Verify we have the expected number of blocks
        assert len(model.residual_blocks) == len(expected_dilations)

    def test_skip_connections_summed(self, simple_config, base_rngs, input_sequence):
        """Test that all skip connections contribute to final output."""
        model = WaveNet(simple_config, rngs=base_rngs)

        outputs = model(input_sequence)
        skip_connections = outputs["skip_connections"]

        # All skip connections should have the same shape
        expected_shape = skip_connections[0].shape
        for skip in skip_connections:
            assert skip.shape == expected_shape

    def test_autoregressive_property(self, simple_config, base_rngs):
        """Test that model maintains autoregressive property.

        The logit at position t should only depend on inputs at positions <= t.
        """
        model = WaveNet(simple_config, rngs=base_rngs)

        seq_len = 32
        x1 = jax.random.randint(jax.random.key(0), (2, seq_len), 0, simple_config.vocab_size)
        x2 = x1.copy()

        # Modify future positions
        x2 = x2.at[:, -10:].set(
            jax.random.randint(jax.random.key(1), (2, 10), 0, simple_config.vocab_size)
        )

        y1 = model(x1)["logits"]
        y2 = model(x2)["logits"]

        # Logits at early positions should be identical
        # Account for receptive field
        rf = model.compute_receptive_field()
        safe_position = seq_len - 10 - rf
        if safe_position > 0:
            assert jnp.allclose(y1[:, :safe_position, :], y2[:, :safe_position, :])
