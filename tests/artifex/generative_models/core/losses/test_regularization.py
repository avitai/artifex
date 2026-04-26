"""Tests for regularization loss functions."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.losses.regularization import (
    DropoutRegularization,
    exclude_bias_predicate,
    exclude_norm_predicate,
    gradient_penalty,
    l1_regularization,
    l2_regularization,
    only_conv_predicate,
    orthogonal_regularization,
    spectral_norm_regularization,
    SpectralNormRegularization,
    total_variation_loss,
)


@pytest.fixture
def key():
    """Fixture for JAX random key."""
    return jax.random.key(0)


@pytest.fixture
def params():
    """Fixture for model parameters."""
    key = jax.random.key(0)
    return {
        "layer1": {
            "kernel": jax.random.normal(key, (10, 5)),
            "bias": jax.random.normal(jax.random.fold_in(key, 1), (5,)),
        },
        "layer2": {
            "kernel": jax.random.normal(jax.random.fold_in(key, 2), (5, 3)),
            "bias": jax.random.normal(jax.random.fold_in(key, 3), (3,)),
        },
    }


@pytest.fixture
def linear_model_fn():
    """Fixture for a simple linear model function."""

    def model(x, params):
        h = x @ params["layer1"]["kernel"] + params["layer1"]["bias"]
        h = nnx.relu(h)
        y = h @ params["layer2"]["kernel"] + params["layer2"]["bias"]
        return y

    return model


class TestL1Regularization:
    """Test cases for L1 regularization."""

    def test_l1_scalar(self, params):
        """Test L1 regularization with scalar factor."""
        # Compute L1 regularization loss
        loss = l1_regularization(params, 0.1)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Test with different weight
        loss2 = l1_regularization(params, 0.2)
        # Double the weight, double the loss
        assert jnp.isclose(loss2, 2 * loss)

    def test_l1_predicate(self, params):
        """Test L1 regularization with a predicate function."""

        # Only regularize kernel weights, not biases
        def kernel_only(name, param):
            return "kernel" in name

        # Compute L1 regularization loss
        loss = l1_regularization(params, 0.1, predicate=kernel_only)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Calculate manually to verify
        manual_sum = 0.0
        for layer in params.values():
            manual_sum += jnp.sum(jnp.abs(layer["kernel"]))
        expected_loss = 0.1 * manual_sum

        assert jnp.isclose(loss, expected_loss)

    def test_l1_supports_array_list_and_mean_reduction(self):
        """L1 regularization should cover all supported parameter container paths."""
        array = jnp.array([1.0, -2.0, 3.0])
        params = [array, -array]

        assert jnp.isclose(l1_regularization(array, scale=0.5), 3.0)
        assert jnp.isclose(l1_regularization(params, scale=0.5), 6.0)
        assert jnp.isclose(l1_regularization(params, scale=1.0, reduction="mean"), 2.0)


class TestL2Regularization:
    """Test cases for L2 regularization (weight decay)."""

    def test_l2_scalar(self, params):
        """Test L2 regularization with scalar factor."""
        # Compute L2 regularization loss
        loss = l2_regularization(params, 0.1)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Manual calculation for verification
        manual_sum = 0.0
        for layer in params.values():
            for param in layer.values():
                manual_sum += jnp.sum(param**2)
        expected_loss = 0.1 * manual_sum

        assert jnp.isclose(loss, expected_loss)

        # Test with different weight
        loss2 = l2_regularization(params, 0.2)
        # Double the weight, double the loss
        assert jnp.isclose(loss2, 2 * loss)

    def test_l2_predicate(self, params):
        """Test L2 regularization with predicate functions."""

        # Set different weights for different parameter groups
        def kernel_only(name, param):
            return "kernel" in name

        # Compute weight decay loss
        loss = l2_regularization(params, 0.1, predicate=kernel_only)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Manual calculation for verification
        manual_sum = 0.0
        for layer_name, layer in params.items():
            manual_sum += jnp.sum(layer["kernel"] ** 2)
        expected_loss = 0.1 * manual_sum

        assert jnp.isclose(loss, expected_loss)

    def test_l2_supports_array_list_and_no_reduction_alias(self):
        """L2 regularization should cover array, list, and none-reduction paths."""
        array = jnp.array([1.0, -2.0, 3.0])
        params = [array, array + 1.0]

        assert jnp.isclose(l2_regularization(array, scale=0.5), 7.0)
        assert jnp.isclose(l2_regularization(params, scale=1.0, reduction="mean"), 35.0 / 6.0)
        assert jnp.isclose(l2_regularization(array, scale=0.5, reduction="none"), 7.0)


class TestGradientPenalty:
    """Test cases for gradient penalty regularization."""

    def test_gradient_penalty_basic(self, key, linear_model_fn, params):
        """Test basic gradient penalty calculation."""
        # Generate random input
        x = jax.random.normal(key, (4, 10))

        # Create fake real and generated samples
        x_real = x
        x_fake = x + 0.1

        def discriminator(x):
            return linear_model_fn(x, params)

        # Compute gradient penalty
        gp = gradient_penalty(x_real, x_fake, discriminator, key=key)

        # Loss should be scalar and finite
        assert gp.shape == ()
        assert jnp.isfinite(gp)

        # Gradient penalty should be non-negative
        assert gp >= 0.0

    def test_gradient_penalty_lambda(self, key, linear_model_fn, params):
        """Test gradient penalty with different lambda values."""
        # Generate random input
        x = jax.random.normal(key, (4, 10))

        # Create fake real and generated samples
        x_real = x
        x_fake = x + 0.1

        def discriminator(x):
            return linear_model_fn(x, params)

        # Compute gradient penalty with default lambda
        gp1 = gradient_penalty(x_real, x_fake, discriminator, key=key)

        # Compute with different lambda
        gp2 = gradient_penalty(x_real, x_fake, discriminator, lambda_gp=20.0, key=key)

        # Second penalty should be twice the first
        assert jnp.isclose(gp2, 2.0 * gp1)

    def test_gradient_penalty_uniform_interpolation_and_weighted_none_reduction(self):
        """Uniform interpolation should not require a key and should expose per-sample values."""
        real = jnp.array([[0.0, 1.0], [1.0, 2.0]])
        fake = real + 0.5

        def discriminator(x):
            return jnp.sum(x, axis=1)

        result = gradient_penalty(
            real,
            fake,
            discriminator,
            lambda_gp=2.0,
            reduction="none",
            weights=jnp.array([1.0, 0.5]),
            interpolation_mode="uniform",
        )

        assert result.shape == (2,)
        assert jnp.isfinite(result).all()

    def test_gradient_penalty_rejects_missing_key_and_unknown_mode(self):
        """Gradient penalty should fail clearly for unsupported interpolation settings."""
        samples = jnp.ones((2, 2))

        with pytest.raises(ValueError, match="Random key required"):
            gradient_penalty(samples, samples, lambda x: jnp.sum(x, axis=1))
        with pytest.raises(ValueError, match="Unknown interpolation_mode"):
            gradient_penalty(
                samples,
                samples,
                lambda x: jnp.sum(x, axis=1),
                interpolation_mode="unsupported",
            )


class TestOrthogonalRegularization:
    """Test cases for orthogonal regularization."""

    def test_orthogonal_regularization_basic(self, params):
        """Test basic orthogonal regularization."""
        # Compute orthogonal regularization for a single weight matrix
        loss = orthogonal_regularization(params["layer1"]["kernel"])

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Orthogonal regularization should be non-negative
        assert loss >= 0.0

        # Create almost orthogonal parameters (orthogonal matrix)
        q, _ = jnp.linalg.qr(params["layer1"]["kernel"])

        # Regularization for orthogonal matrices should be close to zero
        ortho_loss = orthogonal_regularization(q)
        assert ortho_loss < loss

    def test_orthogonal_regularization_scale(self, params):
        """Test orthogonal regularization with different scale values."""
        # Compute regularization with default scale
        loss1 = orthogonal_regularization(params["layer1"]["kernel"])

        # Compute with different scale
        loss2 = orthogonal_regularization(params["layer1"]["kernel"], scale=2.0)

        # Second loss should be twice the first
        assert jnp.isclose(loss2, 2.0 * loss1)


class TestSpectralRegularization:
    """Test cases for spectral regularization."""

    def test_spectral_regularization_basic(self, params):
        """Test basic spectral regularization."""
        # Compute spectral regularization for a single weight matrix
        loss, u_vector = spectral_norm_regularization(params["layer1"]["kernel"])

        # Loss should be scalar and finite
        assert loss.shape == () or loss.shape == (1,)
        assert jnp.isfinite(loss)

        # Create parameters with small spectral norm (scaled down)
        small_params = params["layer1"]["kernel"] * 0.1

        # Regularization for scaled-down matrices should be smaller
        small_loss, small_u = spectral_norm_regularization(small_params)
        assert small_loss < loss

    def test_spectral_regularization_scale(self, params):
        """Test spectral regularization with different scale values."""
        # Compute regularization with default scale
        loss1, u1 = spectral_norm_regularization(params["layer1"]["kernel"])

        # Compute with different scale
        loss2, u2 = spectral_norm_regularization(params["layer1"]["kernel"], scale=2.0)

        # Second loss should be twice the first
        assert jnp.isclose(loss2, 2.0 * loss1)

    def test_stateful_spectral_regularization_reuses_u_state(self, params):
        """The NNX spectral regularizer should initialize and update a named state."""
        regularizer = SpectralNormRegularization(n_power_iterations=2)
        weight = params["layer1"]["kernel"]

        first = regularizer(weight, weight_name="layer1/kernel")
        second = regularizer(weight, weight_name="layer1/kernel")

        assert first.shape == ()
        assert second.shape == ()
        assert "layer1/kernel" in regularizer._u_states
        assert regularizer._u_states["layer1/kernel"][...].shape == (weight.shape[0], 1)


class TestTotalVariationLoss:
    """Test cases for total variation regularization."""

    def test_total_variation_loss_supports_l1_l2_and_grayscale_inputs(self):
        """TV loss should support both image ranks and norm types."""
        images = jnp.array(
            [
                [
                    [[0.0], [1.0]],
                    [[2.0], [3.0]],
                ]
            ]
        )
        grayscale = images[..., 0]

        assert jnp.isclose(total_variation_loss(images, norm_type="l1"), 6.0)
        assert jnp.isclose(total_variation_loss(grayscale, norm_type="l2"), 10.0)

    def test_total_variation_loss_supports_weights_and_rejects_unknown_norm(self):
        """TV loss should apply weights and validate norm names."""
        images = jnp.stack([jnp.zeros((2, 2, 1)), jnp.ones((2, 2, 1))])
        weights = jnp.array([1.0, 0.5])

        result = total_variation_loss(images, reduction="none", weights=weights)

        assert result.shape == (2,)
        with pytest.raises(ValueError, match="Unknown norm_type"):
            total_variation_loss(images, norm_type="linf")


class TestDropoutRegularization:
    """Test cases for the dropout regularization module."""

    def test_dropout_regularization_returns_zero_in_all_current_paths(self, key):
        """The retained dropout regularizer should not add an auxiliary loss."""
        activations = jnp.ones((2, 3))

        assert DropoutRegularization(rate=0.0)(activations, training=True, key=key) == 0.0
        assert DropoutRegularization(rate=0.5)(activations, training=False, key=key) == 0.0
        assert DropoutRegularization(rate=0.5)(activations, training=True, key=key) == 0.0


class TestRegularizationPredicates:
    """Test cases for common regularization predicates."""

    def test_common_predicates_match_parameter_names_and_shapes(self):
        """Common predicates should encode bias, convolution, and norm exclusions."""
        matrix = jnp.ones((3, 3))
        conv_kernel = jnp.ones((3, 3, 2, 4))

        assert exclude_bias_predicate("encoder/kernel", matrix) is True
        assert exclude_bias_predicate("encoder/bias", matrix) is False
        assert only_conv_predicate("conv/kernel", conv_kernel) is True
        assert only_conv_predicate("dense/kernel", matrix) is False
        assert exclude_norm_predicate("block/kernel", matrix) is True
        assert exclude_norm_predicate("block/layer_norm/scale", matrix) is False


class TestRegularizationJAXTransformCompatibility:
    """JIT and differentiation checks for regularization paths used in training losses."""

    @pytest.mark.parametrize(
        ("name", "loss_fn"),
        [
            ("l1", lambda weight: l1_regularization(weight)),
            ("l2", lambda weight: l2_regularization(weight)),
            ("orthogonal", lambda weight: orthogonal_regularization(weight)),
            ("spectral_norm", lambda weight: spectral_norm_regularization(weight)[0]),
            ("total_variation", lambda weight: total_variation_loss(weight.reshape(1, 2, 2, 1))),
        ],
    )
    def test_functional_regularizers_are_jittable_and_differentiable(self, name, loss_fn):
        """Functional regularizers should compile and expose finite input gradients."""
        weight = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        compiled_value = jax.jit(loss_fn)(weight)
        gradients = jax.grad(loss_fn)(weight)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value), name
        assert jnp.isfinite(gradients).all(), name

    def test_gradient_penalty_is_jittable_and_differentiable(self):
        """Gradient penalty should compile and expose finite sample gradients."""
        real = jnp.array([[0.0, 1.0], [1.0, 2.0]])
        fake = real + 0.5

        def discriminator(samples):
            return jnp.sum(samples * samples, axis=1)

        def loss_fn(real_samples):
            return gradient_penalty(
                real_samples,
                fake,
                discriminator,
                interpolation_mode="uniform",
            )

        compiled_value = jax.jit(loss_fn)(real)
        gradients = jax.grad(loss_fn)(real)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value)
        assert jnp.isfinite(gradients).all()

    def test_dropout_regularization_module_is_jittable_and_differentiable(self):
        """The retained dropout regularization module should compile with zero gradients."""
        regularizer = DropoutRegularization(rate=0.5)
        activations = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        def loss_fn(values):
            return regularizer(values, training=True)

        compiled_value = jax.jit(loss_fn)(activations)
        gradients = jax.grad(loss_fn)(activations)

        assert compiled_value.shape == ()
        assert compiled_value == 0.0
        assert jnp.allclose(gradients, jnp.zeros_like(activations))

    def test_stateful_spectral_regularization_module_is_nnx_jittable(self):
        """The stateful spectral regularizer should compile through the NNX transform path."""
        regularizer = SpectralNormRegularization(n_power_iterations=1)
        weight = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        regularizer(weight, weight_name="weight")

        compiled_call = nnx.jit(lambda module, value: module(value, weight_name="weight"))
        compiled_value = compiled_call(regularizer, weight)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value)
