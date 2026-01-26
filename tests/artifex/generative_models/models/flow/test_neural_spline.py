"""Tests for Neural Spline Flow implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    CouplingNetworkConfig,
    NeuralSplineConfig,
)
from artifex.generative_models.models.flow.neural_spline import (
    NeuralSplineFlow,
    RationalQuadraticSplineTransform,
    SplineCouplingLayer,
)


@pytest.fixture
def rngs():
    """Create random number generators for tests."""
    return nnx.Rngs(42)


class TestRationalQuadraticSplineTransform:
    """Test rational quadratic spline transformation."""

    def test_init(self, rngs):
        """Test spline transform initialization."""
        transform = RationalQuadraticSplineTransform(
            num_bins=8,
            tail_bound=3.0,
            rngs=rngs,
        )
        assert transform.num_bins == 8
        assert transform.tail_bound == 3.0

    def test_spline_forward_identity(self, rngs):
        """Test spline forward transformation gives identity for simple case."""
        transform = RationalQuadraticSplineTransform(
            num_bins=4,
            tail_bound=1.0,
            rngs=rngs,
        )

        batch_size = 2
        dim = 3
        x = jnp.array([[0.5, -0.5, 0.0], [0.2, -0.8, 0.3]])

        # Create dummy spline parameters (uniform bins)
        widths = jnp.ones((batch_size, dim, 4)) / 4 * 2.0  # Uniform widths
        heights = jnp.ones((batch_size, dim, 4)) / 4 * 2.0  # Uniform heights
        derivatives = jnp.ones((batch_size, dim, 5))  # Uniform derivatives

        y, log_det = transform.apply_spline(x, widths, heights, derivatives, inverse=False)

        # For our simplified implementation, should return identity within domain
        assert y.shape == x.shape
        assert log_det.shape == (batch_size,)  # Should be summed across dimensions
        # Since our implementation is identity for now, should be equal
        assert jnp.allclose(y, x)

    def test_spline_inverse_identity(self, rngs):
        """Test spline inverse transformation."""
        transform = RationalQuadraticSplineTransform(
            num_bins=4,
            tail_bound=1.0,
            rngs=rngs,
        )

        batch_size = 2
        dim = 3
        y = jnp.array([[0.5, -0.5, 0.0], [0.2, -0.8, 0.3]])

        # Create dummy spline parameters
        widths = jnp.ones((batch_size, dim, 4)) / 4 * 2.0
        heights = jnp.ones((batch_size, dim, 4)) / 4 * 2.0
        derivatives = jnp.ones((batch_size, dim, 5))

        x, log_det = transform.apply_spline(y, widths, heights, derivatives, inverse=True)

        assert x.shape == y.shape
        assert log_det.shape == (batch_size,)  # Should be summed across dimensions
        # Since our implementation is identity for now, should be equal
        assert jnp.allclose(x, y)

    def test_constraint_parameters(self, rngs):
        """Test parameter constraining works correctly."""
        transform = RationalQuadraticSplineTransform(
            num_bins=4,
            tail_bound=2.0,
            rngs=rngs,
        )

        batch_size = 2
        dim = 3

        # Create unconstrained parameters
        unnorm_widths = jax.random.normal(rngs.params(), (batch_size, dim, 4))
        unnorm_heights = jax.random.normal(rngs.params(), (batch_size, dim, 4))
        unnorm_derivatives = jax.random.normal(rngs.params(), (batch_size, dim, 5))

        widths, heights, derivatives = transform._constrain_parameters(
            unnorm_widths, unnorm_heights, unnorm_derivatives
        )

        # Check shapes
        assert widths.shape == (batch_size, dim, 4)
        assert heights.shape == (batch_size, dim, 4)
        assert derivatives.shape == (batch_size, dim, 5)

        # Check constraints
        # Widths should sum to 2*tail_bound per dimension
        width_sums = jnp.sum(widths, axis=-1)
        expected_sum = 2 * transform.tail_bound
        assert jnp.allclose(width_sums, expected_sum, atol=1e-5)

        # Heights should sum to 2*tail_bound per dimension
        height_sums = jnp.sum(heights, axis=-1)
        assert jnp.allclose(height_sums, expected_sum, atol=1e-5)

        # All values should be positive
        assert jnp.all(widths > 0)
        assert jnp.all(heights > 0)
        assert jnp.all(derivatives > 0)

    def test_compute_knots(self, rngs):
        """Test knot computation from widths and heights."""
        transform = RationalQuadraticSplineTransform(
            num_bins=4,
            tail_bound=3.0,
            rngs=rngs,
        )

        batch_size = 2
        dim = 3
        widths = jnp.ones((batch_size, dim, 4)) * 1.5
        heights = jnp.ones((batch_size, dim, 4)) * 1.5

        knots_x, knots_y = transform._compute_knots(widths, heights)

        # Check shapes - should have num_bins + 1 knots
        assert knots_x.shape == (batch_size, dim, 5)  # num_bins + 1
        assert knots_y.shape == (batch_size, dim, 5)

        # Check boundary knots
        assert jnp.allclose(knots_x[..., 0], -3.0)  # -tail_bound
        assert jnp.allclose(knots_y[..., 0], -3.0)
        assert jnp.allclose(knots_x[..., -1], 3.0)  # tail_bound
        assert jnp.allclose(knots_y[..., -1], 3.0)

    def test_spline_outside_domain(self, rngs):
        """Test spline behavior outside domain."""
        transform = RationalQuadraticSplineTransform(
            num_bins=4,
            tail_bound=1.0,
            rngs=rngs,
        )

        batch_size = 2
        dim = 3
        # Test points outside domain
        x = jnp.array([[2.0, -2.0, 1.5], [3.0, -3.0, 2.0]])  # Outside [-1, 1]

        # Create dummy parameters
        widths = jnp.ones((batch_size, dim, 4)) / 4 * 2.0
        heights = jnp.ones((batch_size, dim, 4)) / 4 * 2.0
        derivatives = jnp.ones((batch_size, dim, 5))

        y, log_det = transform.apply_spline(x, widths, heights, derivatives, inverse=False)

        # Outside domain should be identity transformation
        assert jnp.allclose(y, x)
        assert jnp.allclose(log_det, 0.0)

    def test_spline_invertibility(self, rngs):
        """Test that forward and inverse transformations are consistent."""
        transform = RationalQuadraticSplineTransform(
            num_bins=4,
            tail_bound=1.0,
            rngs=rngs,
        )

        batch_size = 2
        dim = 3
        x = jnp.array([[0.5, -0.5, 0.0], [0.2, -0.8, 0.3]])

        # Create dummy parameters
        widths = jnp.ones((batch_size, dim, 4)) / 4 * 2.0
        heights = jnp.ones((batch_size, dim, 4)) / 4 * 2.0
        derivatives = jnp.ones((batch_size, dim, 5))

        # Forward then inverse
        y, log_det_fwd = transform.apply_spline(x, widths, heights, derivatives, inverse=False)
        x_reconstructed, log_det_inv = transform.apply_spline(
            y, widths, heights, derivatives, inverse=True
        )

        # Should reconstruct original input
        assert jnp.allclose(x, x_reconstructed, atol=1e-5)
        # Log determinants should be negatives (for our identity case, both are 0)
        assert jnp.allclose(log_det_fwd, -log_det_inv, atol=1e-5)


class TestSplineCouplingLayer:
    """Test coupling layer with spline transformation."""

    def test_init(self, rngs):
        """Test coupling layer initialization."""
        mask = jnp.array([1, 0, 1, 0])  # Alternating mask
        layer = SplineCouplingLayer(
            mask=mask,
            hidden_dims=[16, 16],
            num_bins=4,
            rngs=rngs,
        )

        assert jnp.array_equal(layer.mask, mask)
        assert layer.num_bins == 4
        assert layer.masked_dim == 2  # Two 1s in mask
        assert layer.unmasked_dim == 2  # Two 0s in mask

    def test_forward_pass_shape(self, rngs):
        """Test coupling layer forward pass shapes."""
        mask = jnp.array([1, 0, 1, 0])
        layer = SplineCouplingLayer(
            mask=mask,
            hidden_dims=[16, 16],
            num_bins=4,
            rngs=rngs,
        )

        batch_size = 3
        x = jnp.ones((batch_size, 4))

        y, log_det = layer.forward(x, rngs=rngs)

        assert y.shape == (batch_size, 4)
        assert log_det.shape == (batch_size,)

    def test_coupling_layer_invertibility(self, rngs):
        """Test coupling layer invertibility."""
        mask = jnp.array([1, 0, 1, 0])
        layer = SplineCouplingLayer(
            mask=mask,
            hidden_dims=[16, 16],
            num_bins=4,
            rngs=rngs,
        )

        batch_size = 3
        x = jax.random.normal(rngs.params(), (batch_size, 4))

        # Forward then inverse
        y, log_det_fwd = layer.forward(x, rngs=rngs)
        x_reconstructed, log_det_inv = layer.inverse(y, rngs=rngs)

        # Check reconstruction
        assert jnp.allclose(x, x_reconstructed, atol=1e-4)
        # Log determinants should be negatives
        assert jnp.allclose(log_det_fwd, -log_det_inv, atol=1e-4)

    def test_coupling_preserves_conditioning_dims(self, rngs):
        """Test that coupling layer preserves conditioning dimensions."""
        mask = jnp.array([1, 0, 1, 0])
        layer = SplineCouplingLayer(
            mask=mask,
            hidden_dims=[16, 16],
            num_bins=4,
            rngs=rngs,
        )

        batch_size = 3
        x = jax.random.normal(rngs.params(), (batch_size, 4))

        y, _ = layer.forward(x, rngs=rngs)

        # Conditioning dimensions (mask == 1) should be unchanged
        conditioning_mask = mask == 1
        assert jnp.allclose(x[:, conditioning_mask], y[:, conditioning_mask])


class TestNeuralSplineFlow:
    """Test Neural Spline Flow model."""

    def test_init(self, rngs):
        """Test NSF initialization."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(16, 16),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_layers=2,
            num_bins=4,
            base_distribution="normal",
        )

        nsf = NeuralSplineFlow(config, rngs=rngs)

        assert nsf.input_dim == 4
        assert nsf.num_layers == 2
        assert len(nsf.coupling_layers) == 2

    def test_init_requires_rngs(self):
        """Test that NSF initialization requires rngs."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(16, 16),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_layers=2,
        )

        with pytest.raises(ValueError, match="rngs must be provided"):
            NeuralSplineFlow(config, rngs=None)

    def test_forward_pass(self, rngs):
        """Test NSF forward pass."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(16, 16),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_layers=2,
            num_bins=4,
        )

        nsf = NeuralSplineFlow(config, rngs=rngs)

        batch_size = 3
        x = jax.random.normal(rngs.params(), (batch_size, 4))

        z, log_det = nsf.forward(x)

        assert z.shape == (batch_size, 4)
        assert log_det.shape == (batch_size,)
        assert jnp.isfinite(z).all()
        assert jnp.isfinite(log_det).all()

    def test_inverse_pass(self, rngs):
        """Test NSF inverse pass."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(16, 16),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_layers=2,
            num_bins=4,
        )

        nsf = NeuralSplineFlow(config, rngs=rngs)

        batch_size = 3
        z = jax.random.normal(rngs.params(), (batch_size, 4))

        x, log_det = nsf.inverse(z)

        assert x.shape == (batch_size, 4)
        assert log_det.shape == (batch_size,)
        assert jnp.isfinite(x).all()
        assert jnp.isfinite(log_det).all()

    def test_nsf_invertibility(self, rngs):
        """Test NSF forward/inverse consistency."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(16, 16),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_layers=2,
            num_bins=4,
        )

        nsf = NeuralSplineFlow(config, rngs=rngs)

        batch_size = 3
        x = jax.random.normal(rngs.params(), (batch_size, 4))

        # Forward then inverse
        z, log_det_fwd = nsf.forward(x)
        x_reconstructed, log_det_inv = nsf.inverse(z)

        # Check reconstruction
        assert jnp.allclose(x, x_reconstructed, atol=1e-3)
        # Log determinants should be negatives
        assert jnp.allclose(log_det_fwd, -log_det_inv, atol=1e-3)

    def test_nsf_log_prob(self, rngs):
        """Test NSF log probability computation."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(16, 16),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_layers=2,
            num_bins=4,
            base_distribution="normal",
        )

        nsf = NeuralSplineFlow(config, rngs=rngs)

        batch_size = 3
        x = jax.random.normal(rngs.params(), (batch_size, 4))

        log_prob = nsf.log_prob(x)

        assert log_prob.shape == (batch_size,)
        assert jnp.isfinite(log_prob).all()

    def test_nsf_sample(self, rngs):
        """Test NSF sampling."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(16, 16),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_layers=2,
            num_bins=4,
            base_distribution="normal",
        )

        nsf = NeuralSplineFlow(config, rngs=rngs)

        n_samples = 5
        samples = nsf.sample(n_samples, rngs=rngs)

        assert samples.shape == (n_samples, 4)
        assert jnp.isfinite(samples).all()

    def test_nsf_generate(self, rngs):
        """Test NSF generate method."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(16, 16),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_layers=2,
            num_bins=4,
            base_distribution="normal",
        )

        nsf = NeuralSplineFlow(config, rngs=rngs)

        n_samples = 5
        samples = nsf.generate(n_samples, rngs=rngs)

        assert samples.shape == (n_samples, 4)
        assert jnp.isfinite(samples).all()

    def test_nsf_loss_fn(self, rngs):
        """Test NSF loss function."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(16, 16),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_layers=2,
            num_bins=4,
            base_distribution="normal",
        )

        nsf = NeuralSplineFlow(config, rngs=rngs)

        batch_size = 3
        x = jax.random.normal(rngs.params(), (batch_size, 4))
        batch = {"x": x}

        # Get model outputs
        z, log_det = nsf.forward(x)
        model_outputs = {"z": z, "log_det": log_det}

        loss = nsf.loss_fn(batch, model_outputs)

        assert isinstance(loss, (float, jax.Array))
        assert jnp.isfinite(loss)

    def test_nsf_different_hidden_dims(self, rngs):
        """Test NSF with different hidden dimensions."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(32, 64, 32),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=6,
            latent_dim=6,
            num_layers=3,
            num_bins=8,
        )

        nsf = NeuralSplineFlow(config, rngs=rngs)

        batch_size = 2
        x = jax.random.normal(rngs.params(), (batch_size, 6))

        z, log_det = nsf.forward(x)
        x_reconstructed, _ = nsf.inverse(z)

        assert z.shape == (batch_size, 6)
        assert log_det.shape == (batch_size,)
        assert jnp.allclose(x, x_reconstructed, atol=1e-3)

    def test_nsf_gradient_computation(self, rngs):
        """Test gradient computation for NSF."""
        coupling_network = CouplingNetworkConfig(
            name="nsf_coupling",
            hidden_dims=(16, 16),
            activation="relu",
        )
        config = NeuralSplineConfig(
            name="test_neural_spline",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_layers=2,
            num_bins=4,
        )

        nsf = NeuralSplineFlow(config, rngs=rngs)

        batch_size = 3
        x = jax.random.normal(rngs.params(), (batch_size, 4))
        batch = {"x": x}

        def loss_fn(model):
            z, log_det = model.forward(x)
            model_outputs = {"z": z, "log_det": log_det}
            return model.loss_fn(batch, model_outputs)

        # Compute gradients
        loss, grads = nnx.value_and_grad(loss_fn)(nsf)

        assert jnp.isfinite(loss)
        # Check that gradients exist and are finite
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        assert all(jnp.isfinite(leaf).all() for leaf in grad_leaves)


class TestNeuralSplineFlowIntegration:
    """Integration tests for Neural Spline Flow."""

    def test_nsf_with_different_configurations(self, rngs):
        """Test NSF with various configurations."""
        configs = [
            NeuralSplineConfig(
                name="test_neural_spline_1",
                coupling_network=CouplingNetworkConfig(
                    name="coupling1",
                    hidden_dims=(8,),
                    activation="relu",
                ),
                input_dim=2,
                latent_dim=2,
                num_layers=1,
                num_bins=4,
            ),
            NeuralSplineConfig(
                name="test_neural_spline_2",
                coupling_network=CouplingNetworkConfig(
                    name="coupling2",
                    hidden_dims=(32, 32),
                    activation="relu",
                ),
                input_dim=8,
                latent_dim=8,
                num_layers=4,
                num_bins=8,
            ),
            NeuralSplineConfig(
                name="test_neural_spline_3",
                coupling_network=CouplingNetworkConfig(
                    name="coupling3",
                    hidden_dims=(16, 32, 16),
                    activation="relu",
                ),
                input_dim=3,
                latent_dim=3,
                num_layers=2,
                num_bins=6,
            ),
        ]

        for config in configs:
            nsf = NeuralSplineFlow(config, rngs=rngs)

            batch_size = 2
            input_dim = config.input_dim
            x = jax.random.normal(rngs.params(), (batch_size, input_dim))

            # Test forward pass
            z, log_det_fwd = nsf.forward(x, rngs=rngs)
            assert z.shape == x.shape
            assert log_det_fwd.shape == (batch_size,)

            # Test inverse pass
            x_reconstructed, log_det_inv = nsf.inverse(z, rngs=rngs)
            assert x_reconstructed.shape == x.shape
            assert log_det_inv.shape == (batch_size,)

            # Test invertibility
            assert jnp.allclose(x, x_reconstructed, atol=1e-3)
            assert jnp.allclose(log_det_fwd, -log_det_inv, atol=1e-3)
