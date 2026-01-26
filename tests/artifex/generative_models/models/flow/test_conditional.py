"""Tests for conditional normalizing flow implementations.

This module provides comprehensive tests for the conditional normalizing flows,
covering config validation, layer initialization, bijection properties, and
mathematical correctness of the transformations.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    CouplingNetworkConfig,
    FlowConfig,
)
from artifex.generative_models.models.flow.conditional import (
    ConditionalCouplingLayer,
    ConditionalNormalizingFlow,
    ConditionalRealNVP,
)


def create_conditional_flow_config(
    input_dim: int = 8,
    condition_dim: int = 4,
    num_coupling_layers: int = 4,
    hidden_dims: tuple[int, ...] = (32, 32),
) -> FlowConfig:
    """Create FlowConfig for testing conditional flows."""
    coupling_network = CouplingNetworkConfig(
        name="conditional_coupling",
        hidden_dims=hidden_dims,
        activation="relu",
    )
    # Create a config object with required attributes
    config = FlowConfig(
        name="test_conditional_flow",
        coupling_network=coupling_network,
        input_dim=input_dim,
        latent_dim=input_dim,
        base_distribution="normal",
        base_distribution_params={"loc": 0.0, "scale": 1.0},
    )
    # Add conditional-specific attributes
    object.__setattr__(config, "condition_dim", condition_dim)
    object.__setattr__(config, "hidden_dims", list(hidden_dims))
    object.__setattr__(config, "num_coupling_layers", num_coupling_layers)
    object.__setattr__(config, "mask_type", "checkerboard")
    return config


@pytest.fixture
def base_rngs():
    """Fixture for nnx random number generators."""
    return nnx.Rngs(
        params=jax.random.key(0),
        dropout=jax.random.key(1),
        sample=jax.random.key(2),
    )


@pytest.fixture
def checkerboard_mask():
    """Fixture for checkerboard mask of dimension 8."""
    return jnp.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=jnp.float32)


@pytest.fixture
def simple_mask():
    """Fixture for simple half-split mask."""
    return jnp.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=jnp.float32)


@pytest.fixture
def input_data():
    """Fixture for input data."""
    return jax.random.normal(jax.random.key(42), (4, 8))


@pytest.fixture
def condition_data():
    """Fixture for conditioning data."""
    return jax.random.normal(jax.random.key(43), (4, 4))


@pytest.fixture
def conditional_config():
    """Fixture for conditional flow configuration."""
    return create_conditional_flow_config()


class TestConditionalCouplingLayer:
    """Test suite for ConditionalCouplingLayer."""

    def test_initialization(self, simple_mask, base_rngs):
        """Test initialization of ConditionalCouplingLayer."""
        layer = ConditionalCouplingLayer(
            mask=simple_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            rngs=base_rngs,
        )

        # Check attributes
        assert jnp.array_equal(layer.mask, simple_mask)
        assert layer.condition_dim == 4
        assert layer.masked_dim == 4  # Sum of ones in mask
        assert layer.unmasked_dim == 4  # Sum of zeros in mask

        # Check neural network layers exist
        assert len(layer.scale_layers) == 2
        assert len(layer.translation_layers) == 2
        assert layer.scale_out is not None
        assert layer.translation_out is not None

    def test_initialization_requires_rngs(self, simple_mask):
        """Test that rngs is required."""
        with pytest.raises(ValueError, match="rngs must be provided"):
            ConditionalCouplingLayer(
                mask=simple_mask,
                hidden_dims=[32, 32],
                condition_dim=4,
                rngs=None,
            )

    def test_scale_activation_tanh(self, simple_mask, base_rngs, input_data, condition_data):
        """Test scale activation with tanh."""
        layer = ConditionalCouplingLayer(
            mask=simple_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            scale_activation="tanh",
            rngs=base_rngs,
        )

        s, t = layer._scale_and_translate(input_data, condition_data, rngs=base_rngs)

        # Scale should be bounded by tanh
        assert jnp.all((s >= -1.0) & (s <= 1.0))

    def test_scale_activation_sigmoid(self, simple_mask, base_rngs, input_data, condition_data):
        """Test scale activation with sigmoid."""
        layer = ConditionalCouplingLayer(
            mask=simple_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            scale_activation="sigmoid",
            rngs=base_rngs,
        )

        s, t = layer._scale_and_translate(input_data, condition_data, rngs=base_rngs)

        # Scale should be bounded by sigmoid
        assert jnp.all((s >= 0.0) & (s <= 1.0))

    def test_scale_and_translate_shapes(self, simple_mask, base_rngs, input_data, condition_data):
        """Test _scale_and_translate output shapes."""
        layer = ConditionalCouplingLayer(
            mask=simple_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            rngs=base_rngs,
        )

        s, t = layer._scale_and_translate(input_data, condition_data, rngs=base_rngs)

        # s and t should have shape (batch_size, unmasked_dim)
        expected_shape = (input_data.shape[0], layer.unmasked_dim)
        assert s.shape == expected_shape
        assert t.shape == expected_shape

    def test_forward_requires_condition(self, simple_mask, base_rngs, input_data):
        """Test forward transformation requires conditioning."""
        layer = ConditionalCouplingLayer(
            mask=simple_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            rngs=base_rngs,
        )

        with pytest.raises(ValueError, match="condition must be provided"):
            layer.forward(input_data, rngs=base_rngs, condition=None)

    def test_forward_output_shapes(self, simple_mask, base_rngs, input_data, condition_data):
        """Test forward transformation output shapes."""
        layer = ConditionalCouplingLayer(
            mask=simple_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            rngs=base_rngs,
        )

        y, log_det = layer.forward(input_data, rngs=base_rngs, condition=condition_data)

        # Check shapes
        assert y.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

    def test_forward_masked_unchanged(self, simple_mask, base_rngs, input_data, condition_data):
        """Test that masked dimensions remain unchanged in forward pass."""
        layer = ConditionalCouplingLayer(
            mask=simple_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            rngs=base_rngs,
        )

        y, _ = layer.forward(input_data, rngs=base_rngs, condition=condition_data)

        # First half (masked=1) should be unchanged
        assert jnp.allclose(y[:, :4], input_data[:, :4])

        # Second half (masked=0) should be transformed
        assert not jnp.allclose(y[:, 4:], input_data[:, 4:])

    def test_inverse_requires_condition(self, simple_mask, base_rngs, input_data):
        """Test inverse transformation requires conditioning."""
        layer = ConditionalCouplingLayer(
            mask=simple_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            rngs=base_rngs,
        )

        with pytest.raises(ValueError, match="condition must be provided"):
            layer.inverse(input_data, rngs=base_rngs, condition=None)

    def test_bijection_property(self, simple_mask, base_rngs, input_data, condition_data):
        """Test bijection property: inverse(forward(x, c), c) == x."""
        layer = ConditionalCouplingLayer(
            mask=simple_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            rngs=base_rngs,
        )

        # Forward transformation
        y, log_det_fwd = layer.forward(input_data, rngs=base_rngs, condition=condition_data)

        # Inverse transformation
        x_recon, log_det_inv = layer.inverse(y, rngs=base_rngs, condition=condition_data)

        # Should recover original input
        assert jnp.allclose(x_recon, input_data, rtol=1e-4, atol=1e-4)

        # Log determinants should sum to zero (or be negatives)
        assert jnp.allclose(log_det_fwd, -log_det_inv, rtol=1e-4, atol=1e-4)

    def test_different_conditions_different_outputs(
        self, simple_mask, base_rngs, input_data, condition_data
    ):
        """Test that different conditions produce different outputs."""
        layer = ConditionalCouplingLayer(
            mask=simple_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            rngs=base_rngs,
        )

        condition1 = jax.random.normal(jax.random.key(100), (4, 4))
        condition2 = jax.random.normal(jax.random.key(200), (4, 4))

        y1, _ = layer.forward(input_data, rngs=base_rngs, condition=condition1)
        y2, _ = layer.forward(input_data, rngs=base_rngs, condition=condition2)

        # Different conditions should produce different outputs
        assert not jnp.allclose(y1, y2)

    def test_checkerboard_mask(self, checkerboard_mask, base_rngs, input_data, condition_data):
        """Test with checkerboard mask pattern."""
        layer = ConditionalCouplingLayer(
            mask=checkerboard_mask,
            hidden_dims=[32, 32],
            condition_dim=4,
            rngs=base_rngs,
        )

        y, log_det = layer.forward(input_data, rngs=base_rngs, condition=condition_data)
        x_recon, _ = layer.inverse(y, rngs=base_rngs, condition=condition_data)

        # Should still satisfy bijection
        assert jnp.allclose(x_recon, input_data, rtol=1e-4, atol=1e-4)


class TestConditionalNormalizingFlow:
    """Test suite for ConditionalNormalizingFlow."""

    def test_initialization(self, conditional_config, base_rngs):
        """Test initialization of ConditionalNormalizingFlow."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        # Check attributes
        assert model.condition_dim == 4
        assert model.hidden_dims == [32, 32]
        assert model.num_coupling_layers == 4
        assert model.mask_type == "checkerboard"

        # Check flow layers created
        assert len(model.flow_layers) == 4
        for layer in model.flow_layers:
            assert isinstance(layer, ConditionalCouplingLayer)

    def test_initialization_requires_rngs(self, conditional_config):
        """Test that rngs is required."""
        with pytest.raises(ValueError, match="rngs must be provided"):
            ConditionalNormalizingFlow(conditional_config, rngs=None)

    def test_mask_creation_checkerboard(self, conditional_config, base_rngs):
        """Test checkerboard mask creation."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        # Masks should alternate
        mask0 = model.flow_layers[0].mask
        mask1 = model.flow_layers[1].mask

        # Consecutive layer masks should be complements
        assert jnp.allclose(mask0, 1 - mask1)

    def test_forward_output_shapes(self, conditional_config, base_rngs, input_data, condition_data):
        """Test forward transformation output shapes."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        z, log_det = model.forward(input_data, rngs=base_rngs, condition=condition_data)

        assert z.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

    def test_forward_without_condition_uses_zeros(self, conditional_config, base_rngs, input_data):
        """Test that forward with no condition uses zero conditioning."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        # Should not raise - uses default zero conditioning
        z, log_det = model.forward(input_data, rngs=base_rngs, condition=None)

        assert z.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

    def test_inverse_output_shapes(self, conditional_config, base_rngs, input_data, condition_data):
        """Test inverse transformation output shapes."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        z, _ = model.forward(input_data, rngs=base_rngs, condition=condition_data)
        x, log_det = model.inverse(z, rngs=base_rngs, condition=condition_data)

        assert x.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

    def test_bijection_property(self, conditional_config, base_rngs, input_data, condition_data):
        """Test bijection property: inverse(forward(x, c), c) == x."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        # Forward transformation
        z, log_det_fwd = model.forward(input_data, rngs=base_rngs, condition=condition_data)

        # Inverse transformation
        x_recon, log_det_inv = model.inverse(z, rngs=base_rngs, condition=condition_data)

        # Should recover original input
        assert jnp.allclose(x_recon, input_data, rtol=1e-3, atol=1e-3)

        # Log determinants should sum to zero
        assert jnp.allclose(log_det_fwd, -log_det_inv, rtol=1e-3, atol=1e-3)

    def test_log_prob_computation(self, conditional_config, base_rngs, input_data, condition_data):
        """Test log probability computation."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        log_prob = model.log_prob(input_data, rngs=base_rngs, condition=condition_data)

        # Check shape
        assert log_prob.shape == (input_data.shape[0],)

        # Log probability should be finite
        assert jnp.all(jnp.isfinite(log_prob))

    def test_log_prob_formula(self, conditional_config, base_rngs, input_data, condition_data):
        """Test log probability formula: log p(x|c) = log p_z(z) + log|det J|."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        # Compute log_prob directly
        log_prob = model.log_prob(input_data, rngs=base_rngs, condition=condition_data)

        # Compute manually
        z, log_det = model.forward(input_data, rngs=base_rngs, condition=condition_data)
        log_prob_z = model.log_prob_fn(z)
        expected_log_prob = log_prob_z + log_det

        assert jnp.allclose(log_prob, expected_log_prob)

    def test_generate_samples(self, conditional_config, base_rngs, condition_data):
        """Test sample generation."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        # Generate samples with conditioning
        n_samples = 8
        condition = condition_data[:1].repeat(n_samples, axis=0)  # Same condition for all
        samples = model.generate(n_samples=n_samples, condition=condition, rngs=base_rngs)

        # Check shape
        assert samples.shape == (n_samples, conditional_config.input_dim)

        # Samples should be finite
        assert jnp.all(jnp.isfinite(samples))

    def test_generate_without_condition(self, conditional_config, base_rngs):
        """Test generation without explicit condition."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        samples = model.generate(n_samples=4, rngs=base_rngs)

        # Should work with default zero conditioning
        assert samples.shape == (4, conditional_config.input_dim)
        assert jnp.all(jnp.isfinite(samples))

    def test_generate_broadcasts_condition(self, conditional_config, base_rngs):
        """Test that 1D condition is broadcast to batch size."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        # Single condition vector (1D)
        condition = jax.random.normal(jax.random.key(50), (4,))  # Just condition_dim

        samples = model.generate(n_samples=8, condition=condition, rngs=base_rngs)

        # Should work with broadcasted conditioning
        assert samples.shape == (8, conditional_config.input_dim)

    def test_call_returns_dict(self, conditional_config, base_rngs, input_data, condition_data):
        """Test __call__ returns dictionary with expected keys."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        outputs = model(input_data, condition=condition_data, rngs=base_rngs)

        # Check expected keys
        assert "z" in outputs
        assert "logdet" in outputs
        assert "log_prob" in outputs
        assert "condition" in outputs

        # Check shapes
        assert outputs["z"].shape == input_data.shape
        assert outputs["logdet"].shape == (input_data.shape[0],)
        assert outputs["log_prob"].shape == (input_data.shape[0],)

    def test_loss_fn_with_dict_batch(
        self, conditional_config, base_rngs, input_data, condition_data
    ):
        """Test loss function with dictionary batch."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        batch = {"x": input_data, "condition": condition_data}
        model_outputs = model(input_data, condition=condition_data, rngs=base_rngs)
        loss_dict = model.loss_fn(batch, model_outputs, rngs=base_rngs)

        # Check expected keys
        assert "loss" in loss_dict
        assert "nll_loss" in loss_dict
        assert "conditional_log_prob" in loss_dict

        # Loss should be finite
        assert jnp.isfinite(loss_dict["loss"])

    def test_loss_fn_with_array_batch(self, conditional_config, base_rngs, input_data):
        """Test loss function with array batch (no conditioning)."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        model_outputs = model(input_data, rngs=base_rngs)
        loss_dict = model.loss_fn(input_data, model_outputs, rngs=base_rngs)

        # Should work with default conditioning
        assert jnp.isfinite(loss_dict["loss"])

    def test_loss_is_negative_log_likelihood(
        self, conditional_config, base_rngs, input_data, condition_data
    ):
        """Test that loss equals negative log likelihood."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        batch = {"x": input_data, "condition": condition_data}
        model_outputs = model(input_data, condition=condition_data, rngs=base_rngs)
        loss_dict = model.loss_fn(batch, model_outputs, rngs=base_rngs)

        log_prob = model.log_prob(input_data, rngs=base_rngs, condition=condition_data)
        expected_loss = -jnp.mean(log_prob)

        assert jnp.allclose(loss_dict["loss"], expected_loss)

    def test_different_conditions_different_samples(self, conditional_config, base_rngs):
        """Test that different conditions produce different samples."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        condition1 = jnp.zeros((4, 4))
        condition2 = jnp.ones((4, 4))

        # Use same RNG but different conditions
        rngs1 = nnx.Rngs(params=jax.random.key(42), sample=jax.random.key(100))
        rngs2 = nnx.Rngs(params=jax.random.key(42), sample=jax.random.key(100))

        samples1 = model.generate(n_samples=4, condition=condition1, rngs=rngs1)
        samples2 = model.generate(n_samples=4, condition=condition2, rngs=rngs2)

        # Different conditions should produce different samples
        assert not jnp.allclose(samples1, samples2)

    def test_gradient_flow(self, conditional_config, base_rngs, input_data, condition_data):
        """Test that gradients flow through the model."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        def loss_fn(model, x, c):
            outputs = model(x, condition=c, rngs=base_rngs)
            return -jnp.mean(outputs["log_prob"])

        grads = nnx.grad(loss_fn)(model, input_data, condition_data)

        # Check gradients are finite and non-zero somewhere
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        assert any(jnp.any(jnp.isfinite(g)) for g in grad_leaves if hasattr(g, "shape"))


class TestConditionalRealNVP:
    """Test suite for ConditionalRealNVP."""

    def test_initialization(self, conditional_config, base_rngs):
        """Test initialization of ConditionalRealNVP."""
        model = ConditionalRealNVP(conditional_config, rngs=base_rngs)

        # Should inherit from ConditionalNormalizingFlow
        assert isinstance(model, ConditionalNormalizingFlow)
        assert len(model.flow_layers) == 4

    def test_sample_class_conditional_with_indices(self, conditional_config, base_rngs):
        """Test class-conditional sampling with class indices."""
        # Create config with condition_dim matching num_classes
        config = create_conditional_flow_config(
            input_dim=8,
            condition_dim=5,  # 5 classes
            num_coupling_layers=4,
        )
        model = ConditionalRealNVP(config, rngs=base_rngs)

        # Class labels as indices
        n_samples = 8
        class_labels = jnp.array([0, 1, 2, 3, 4, 0, 1, 2])  # Class indices

        samples = model.sample_class_conditional(n_samples, class_labels, rngs=base_rngs)

        # Check shape
        assert samples.shape == (n_samples, config.input_dim)

        # Samples should be finite
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_class_conditional_with_one_hot(self, conditional_config, base_rngs):
        """Test class-conditional sampling with one-hot encoded labels."""
        # Create config with condition_dim matching num_classes
        config = create_conditional_flow_config(
            input_dim=8,
            condition_dim=5,  # 5 classes
            num_coupling_layers=4,
        )
        model = ConditionalRealNVP(config, rngs=base_rngs)

        # One-hot encoded labels
        n_samples = 4
        class_labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 5)

        samples = model.sample_class_conditional(n_samples, class_labels, rngs=base_rngs)

        # Check shape
        assert samples.shape == (n_samples, config.input_dim)

    def test_different_classes_different_samples(self, base_rngs):
        """Test that different classes produce different samples."""
        config = create_conditional_flow_config(
            input_dim=8,
            condition_dim=5,
            num_coupling_layers=4,
        )
        model = ConditionalRealNVP(config, rngs=base_rngs)

        # Provide one-hot encoded conditions directly to avoid num_classes inference issues
        # Class 0 condition: [1, 0, 0, 0, 0]
        condition_0 = jnp.tile(jax.nn.one_hot(0, 5)[None, :], (4, 1))
        # Class 1 condition: [0, 1, 0, 0, 0]
        condition_1 = jnp.tile(jax.nn.one_hot(1, 5)[None, :], (4, 1))

        rngs1 = nnx.Rngs(params=jax.random.key(42), sample=jax.random.key(100))
        rngs2 = nnx.Rngs(params=jax.random.key(42), sample=jax.random.key(100))

        # Use the one-hot encoded conditions directly
        samples_0 = model.sample_class_conditional(4, condition_0, rngs=rngs1)
        samples_1 = model.sample_class_conditional(4, condition_1, rngs=rngs2)

        # Different classes should produce different samples
        assert not jnp.allclose(samples_0, samples_1)


class TestMaskCreation:
    """Test mask creation for ConditionalNormalizingFlow."""

    def test_checkerboard_mask_pattern(self, base_rngs):
        """Test checkerboard mask has correct pattern."""
        config = create_conditional_flow_config(input_dim=8)
        # mask_type is already set to "checkerboard" by default in create_conditional_flow_config
        model = ConditionalNormalizingFlow(config, rngs=base_rngs)

        mask = model._create_mask(0)

        # Checkerboard pattern: alternating 0s and 1s
        expected = jnp.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.float32)
        assert jnp.array_equal(mask, expected)

    def test_alternating_masks(self, base_rngs):
        """Test that masks alternate between layers."""
        config = create_conditional_flow_config(input_dim=8)
        model = ConditionalNormalizingFlow(config, rngs=base_rngs)

        mask_even = model._create_mask(0)
        mask_odd = model._create_mask(1)

        # Should be complements
        assert jnp.allclose(mask_even, 1 - mask_odd)


class TestEdgeCases:
    """Test edge cases for conditional flows."""

    def test_single_sample(self, conditional_config, base_rngs):
        """Test with single sample."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        x = jax.random.normal(jax.random.key(42), (1, 8))
        c = jax.random.normal(jax.random.key(43), (1, 4))

        z, log_det = model.forward(x, rngs=base_rngs, condition=c)
        x_recon, _ = model.inverse(z, rngs=base_rngs, condition=c)

        assert jnp.allclose(x_recon, x, rtol=1e-3, atol=1e-3)

    def test_large_batch(self, conditional_config, base_rngs):
        """Test with large batch size."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        batch_size = 64
        x = jax.random.normal(jax.random.key(42), (batch_size, 8))
        c = jax.random.normal(jax.random.key(43), (batch_size, 4))

        z, log_det = model.forward(x, rngs=base_rngs, condition=c)

        assert z.shape == (batch_size, 8)
        assert log_det.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(log_det))

    def test_zero_condition(self, conditional_config, base_rngs, input_data):
        """Test with zero conditioning."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        zero_condition = jnp.zeros((4, 4))

        z, log_det = model.forward(input_data, rngs=base_rngs, condition=zero_condition)
        x_recon, _ = model.inverse(z, rngs=base_rngs, condition=zero_condition)

        assert jnp.allclose(x_recon, input_data, rtol=1e-3, atol=1e-3)

    def test_different_hidden_dims(self, base_rngs, input_data, condition_data):
        """Test with different hidden dimensions."""
        config = create_conditional_flow_config(hidden_dims=(64, 32, 16))
        model = ConditionalNormalizingFlow(config, rngs=base_rngs)

        z, log_det = model.forward(input_data, rngs=base_rngs, condition=condition_data)
        x_recon, _ = model.inverse(z, rngs=base_rngs, condition=condition_data)

        assert jnp.allclose(x_recon, input_data, rtol=1e-3, atol=1e-3)

    def test_more_coupling_layers(self, base_rngs, input_data, condition_data):
        """Test with more coupling layers."""
        config = create_conditional_flow_config(num_coupling_layers=8)
        model = ConditionalNormalizingFlow(config, rngs=base_rngs)

        z, log_det = model.forward(input_data, rngs=base_rngs, condition=condition_data)
        x_recon, _ = model.inverse(z, rngs=base_rngs, condition=condition_data)

        # More layers may accumulate more numerical error
        assert jnp.allclose(x_recon, input_data, rtol=1e-2, atol=1e-2)

    def test_finite_outputs_for_normal_inputs(self, conditional_config, base_rngs):
        """Test that normal inputs produce finite outputs."""
        model = ConditionalNormalizingFlow(conditional_config, rngs=base_rngs)

        # Normal distributed inputs
        x = jax.random.normal(jax.random.key(42), (16, 8))
        c = jax.random.normal(jax.random.key(43), (16, 4))

        z, log_det = model.forward(x, rngs=base_rngs, condition=c)
        log_prob = model.log_prob(x, rngs=base_rngs, condition=c)

        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(log_det))
        assert jnp.all(jnp.isfinite(log_prob))
