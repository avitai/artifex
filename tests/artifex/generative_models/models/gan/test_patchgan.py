"""Tests for PatchGAN discriminator implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    MultiScalePatchGANConfig,
    PatchGANDiscriminatorConfig,
)
from artifex.generative_models.models.gan.patchgan import (
    MultiScalePatchGANDiscriminator,
    PatchGANDiscriminator,
)


class TestPatchGANDiscriminator:
    """Test cases for PatchGAN discriminator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def input_shape(self):
        """Input shape fixture for RGB images."""
        return (3, 64, 64)  # C, H, W

    @pytest.fixture
    def patchgan_config(self, input_shape):
        """PatchGAN discriminator config fixture."""
        return PatchGANDiscriminatorConfig(
            name="test_patchgan",
            input_shape=input_shape,
            hidden_dims=(64, 128, 256),
            activation="leaky_relu",
            num_filters=64,
            num_layers=3,
            batch_norm=True,
            dropout_rate=0.1,
        )

    @pytest.fixture
    def patchgan_discriminator(self, rng, patchgan_config):
        """PatchGAN discriminator fixture."""
        return PatchGANDiscriminator(
            config=patchgan_config,
            rngs=nnx.Rngs(rng),
        )

    def test_patchgan_initialization(self, patchgan_discriminator, input_shape):
        """Test PatchGAN discriminator initialization."""
        assert patchgan_discriminator.input_shape == input_shape
        assert patchgan_discriminator.patchgan_num_filters == 64
        assert patchgan_discriminator.patchgan_num_layers == 3
        assert patchgan_discriminator.batch_norm is True
        assert patchgan_discriminator.dropout_rate == 0.1

        # Check that layers are created
        assert hasattr(patchgan_discriminator, "initial_conv")
        assert hasattr(patchgan_discriminator, "patchgan_conv_layers")
        assert hasattr(patchgan_discriminator, "final_conv")
        assert len(patchgan_discriminator.patchgan_conv_layers) == 3
        assert len(patchgan_discriminator.patchgan_batch_norm_layers) == 3

    def test_patchgan_forward_pass(self, patchgan_discriminator, rng, input_shape):
        """Test PatchGAN discriminator forward pass."""
        batch_size = 4
        # Create input in channel-first format (B, C, H, W)
        input_images = jax.random.normal(rng, (batch_size, *input_shape))

        # Forward pass
        patchgan_discriminator.train()
        features = patchgan_discriminator(input_images)

        # Check that we get the expected number of features
        # Should have initial_conv + num_layers conv_layers + final_conv = 1 + 3 + 1 = 5 features
        assert len(features) == 5

        # Check output shape - should be patch-level predictions
        final_output = features[-1]
        assert final_output.ndim == 4  # (B, H, W, 1)
        assert final_output.shape[0] == batch_size
        assert final_output.shape[-1] == 1  # Single channel output

        # Output spatial dimensions should be reduced from input
        assert final_output.shape[1] < input_shape[1]
        assert final_output.shape[2] < input_shape[2]

    def test_patchgan_training_vs_inference(self, patchgan_discriminator, rng, input_shape):
        """Test PatchGAN discriminator training vs inference mode."""
        batch_size = 2
        input_images = jax.random.normal(rng, (batch_size, *input_shape))

        # Get outputs in training and inference modes
        patchgan_discriminator.train()
        features_train = patchgan_discriminator(input_images)
        patchgan_discriminator.eval()
        features_inference = patchgan_discriminator(input_images)

        # Should have same number of features
        assert len(features_train) == len(features_inference)

        # Shapes should be identical
        for f_train, f_inference in zip(features_train, features_inference):
            assert f_train.shape == f_inference.shape

    def test_patchgan_different_input_sizes(self, rng):
        """Test PatchGAN discriminator with different input sizes."""
        test_shapes = [(3, 32, 32), (3, 128, 128), (1, 64, 64)]

        for input_shape in test_shapes:
            config = PatchGANDiscriminatorConfig(
                name=f"test_patchgan_{input_shape[1]}",
                input_shape=input_shape,
                hidden_dims=(32, 64),
                activation="leaky_relu",
                num_filters=32,
                num_layers=2,
            )
            discriminator = PatchGANDiscriminator(
                config=config,
                rngs=nnx.Rngs(rng),
            )

            batch_size = 2
            input_images = jax.random.normal(rng, (batch_size, *input_shape))
            discriminator.eval()
            features = discriminator(input_images)

            # Should always get features
            assert len(features) > 0
            final_output = features[-1]
            assert final_output.shape[0] == batch_size
            assert final_output.shape[-1] == 1

    def test_patchgan_no_batch_norm(self, rng, input_shape):
        """Test PatchGAN discriminator without batch normalization."""
        config = PatchGANDiscriminatorConfig(
            name="test_patchgan_no_bn",
            input_shape=input_shape,
            hidden_dims=(64, 128, 256),
            activation="leaky_relu",
            num_filters=64,
            num_layers=3,
            batch_norm=False,
        )
        discriminator = PatchGANDiscriminator(
            config=config,
            rngs=nnx.Rngs(rng),
        )

        # Should have no batch norm layers
        assert len(discriminator.patchgan_batch_norm_layers) == 0

        batch_size = 2
        input_images = jax.random.normal(rng, (batch_size, *input_shape))
        discriminator.train()
        features = discriminator(input_images)

        # Should still work
        assert len(features) > 0

    def test_patchgan_different_num_layers(self, rng, input_shape):
        """Test PatchGAN discriminator with different number of layers."""
        for num_layers in [1, 2, 4]:
            config = PatchGANDiscriminatorConfig(
                name=f"test_patchgan_{num_layers}layers",
                input_shape=input_shape,
                hidden_dims=(32,) * num_layers,
                activation="leaky_relu",
                num_filters=32,
                num_layers=num_layers,
            )
            discriminator = PatchGANDiscriminator(
                config=config,
                rngs=nnx.Rngs(rng),
            )

            assert len(discriminator.patchgan_conv_layers) == num_layers

            batch_size = 2
            input_images = jax.random.normal(rng, (batch_size, *input_shape))
            discriminator.eval()
            features = discriminator(input_images)

            # Should get initial_conv + num_layers + final_conv features
            expected_features = num_layers + 2
            assert len(features) == expected_features

    def test_patchgan_gradient_flow(self, patchgan_discriminator, rng, input_shape):
        """Test gradient flow through PatchGAN discriminator."""
        batch_size = 2
        input_images = jax.random.normal(rng, (batch_size, *input_shape))

        def loss_fn(model, x):
            """Simple loss function for gradient testing."""
            features = model(x)
            output = features[-1]
            return jnp.mean(output**2)

        # Test gradient computation using NNX
        @nnx.jit
        def compute_loss(model, x):
            return loss_fn(model, x)

        loss = compute_loss(patchgan_discriminator, input_images)
        assert jnp.isfinite(loss)
        assert loss >= 0.0

        # Test gradients
        grad_fn = nnx.grad(loss_fn, argnums=0)
        grads = grad_fn(patchgan_discriminator, input_images)
        assert grads is not None

    def test_patchgan_config_type_validation(self, rng, input_shape):
        """Test that PatchGAN raises TypeError for wrong config type."""
        from artifex.generative_models.core.configuration.network_configs import (
            DiscriminatorConfig,
        )

        # Create a base DiscriminatorConfig (not PatchGANDiscriminatorConfig)
        wrong_config = DiscriminatorConfig(
            name="wrong_config",
            input_shape=input_shape,
            hidden_dims=(64, 128),
            activation="leaky_relu",
        )

        with pytest.raises(TypeError, match="must be PatchGANDiscriminatorConfig"):
            PatchGANDiscriminator(config=wrong_config, rngs=nnx.Rngs(rng))

    def test_patchgan_rngs_required(self, patchgan_config):
        """Test that PatchGAN raises ValueError when rngs is None."""
        with pytest.raises(ValueError, match="rngs must be provided"):
            PatchGANDiscriminator(config=patchgan_config, rngs=None)


class TestMultiScalePatchGANDiscriminator:
    """Test cases for multi-scale PatchGAN discriminator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def input_shape(self):
        """Input shape fixture."""
        return (3, 256, 256)  # Large enough for multi-scale with 3 discriminators and 3 layers

    @pytest.fixture
    def base_disc_config(self, input_shape):
        """Base PatchGAN discriminator config fixture."""
        return PatchGANDiscriminatorConfig(
            name="test_base_disc",
            input_shape=input_shape,
            hidden_dims=(64, 128, 256),
            activation="leaky_relu",
            num_filters=64,
            num_layers=3,
            batch_norm=True,
        )

    @pytest.fixture
    def multiscale_config(self, base_disc_config):
        """Multi-scale PatchGAN config fixture."""
        return MultiScalePatchGANConfig(
            discriminator=base_disc_config,
            num_discriminators=3,
        )

    @pytest.fixture
    def multiscale_discriminator(self, rng, multiscale_config):
        """Multi-scale PatchGAN discriminator fixture."""
        return MultiScalePatchGANDiscriminator(
            config=multiscale_config,
            rngs=nnx.Rngs(rng),
        )

    def test_multiscale_initialization(self, multiscale_discriminator, input_shape):
        """Test multi-scale discriminator initialization."""
        assert multiscale_discriminator.input_shape == input_shape
        assert multiscale_discriminator.num_discriminators == 3
        assert len(multiscale_discriminator.discriminators) == 3
        assert len(multiscale_discriminator.num_layers_per_disc) == 3

        # All discriminators should have the same number of layers
        for layers in multiscale_discriminator.num_layers_per_disc:
            assert layers == 3

    def test_multiscale_forward_pass(self, multiscale_discriminator, rng, input_shape):
        """Test multi-scale discriminator forward pass."""
        batch_size = 2
        input_images = jax.random.normal(rng, (batch_size, *input_shape))

        multiscale_discriminator.train()
        outputs, all_features = multiscale_discriminator(input_images)

        # Should have outputs from all discriminators
        assert len(outputs) == 3
        assert len(all_features) == 3

        # Check output shapes
        for output in outputs:
            assert output.ndim == 4  # (B, H, W, 1)
            assert output.shape[0] == batch_size
            assert output.shape[-1] == 1

        # Check features
        for features in all_features:
            assert len(features) > 0  # Should have intermediate features

    def test_multiscale_downsampling(self, multiscale_discriminator, rng, input_shape):
        """Test image downsampling in multi-scale discriminator."""
        batch_size = 1
        # Create input in JAX format (B, H, W, C)
        input_images = jax.random.normal(
            rng, (batch_size, input_shape[1], input_shape[2], input_shape[0])
        )

        # Test downsampling
        downsampled_2x = multiscale_discriminator.downsample_image(input_images, 2)
        downsampled_4x = multiscale_discriminator.downsample_image(input_images, 4)

        # Check shapes
        assert downsampled_2x.shape[1] == input_shape[1] // 2
        assert downsampled_2x.shape[2] == input_shape[2] // 2
        assert downsampled_4x.shape[1] == input_shape[1] // 4
        assert downsampled_4x.shape[2] == input_shape[2] // 4

        # No downsampling with factor 1
        no_downsample = multiscale_discriminator.downsample_image(input_images, 1)
        assert jnp.array_equal(no_downsample, input_images)

    def test_multiscale_different_layer_configs(self, rng, input_shape, base_disc_config):
        """Test multi-scale discriminator with different layer configurations."""
        # Test with list of different layer numbers
        layer_configs = (1, 2, 3)  # Use smaller layer counts to avoid downsampling issues
        config = MultiScalePatchGANConfig(
            discriminator=base_disc_config,
            num_discriminators=3,
            num_layers_per_disc=layer_configs,
        )
        discriminator = MultiScalePatchGANDiscriminator(
            config=config,
            rngs=nnx.Rngs(rng),
        )

        assert discriminator.num_layers_per_disc == list(layer_configs)

        batch_size = 2
        input_images = jax.random.normal(rng, (batch_size, *input_shape))
        discriminator.eval()
        outputs, all_features = discriminator(input_images)

        assert len(outputs) == 3
        assert len(all_features) == 3

    def test_multiscale_minimum_size_validation(self, rng):
        """Test multi-scale discriminator minimum size validation."""
        # This should fail - too many layers for small input
        small_input_shape = (3, 16, 16)

        disc_config = PatchGANDiscriminatorConfig(
            name="test_small_disc",
            input_shape=small_input_shape,
            hidden_dims=(64, 128, 256, 512, 1024),
            activation="leaky_relu",
            num_filters=64,
            num_layers=5,  # Too many layers
        )

        config = MultiScalePatchGANConfig(
            discriminator=disc_config,
            num_discriminators=3,
            minimum_size=8,
        )

        with pytest.raises(ValueError, match="would downsample to size"):
            MultiScalePatchGANDiscriminator(
                config=config,
                rngs=nnx.Rngs(rng),
            )

    def test_multiscale_invalid_layer_config(self, rng, input_shape, base_disc_config):
        """Test multi-scale discriminator with invalid layer configuration."""
        # Mismatched length - should fail at config validation
        with pytest.raises(ValueError, match="must match num_discriminators"):
            MultiScalePatchGANConfig(
                discriminator=base_disc_config,
                num_discriminators=3,
                num_layers_per_disc=(2, 3),  # Wrong length
            )

    def test_multiscale_training_vs_inference(self, multiscale_discriminator, rng, input_shape):
        """Test multi-scale discriminator training vs inference mode."""
        batch_size = 2
        input_images = jax.random.normal(rng, (batch_size, *input_shape))

        # Get outputs in both modes
        multiscale_discriminator.train()
        outputs_train, features_train = multiscale_discriminator(input_images)
        multiscale_discriminator.eval()
        outputs_inference, features_inference = multiscale_discriminator(input_images)

        # Should have same structure
        assert len(outputs_train) == len(outputs_inference)
        assert len(features_train) == len(features_inference)

        # Shapes should be identical
        for out_train, out_inference in zip(outputs_train, outputs_inference):
            assert out_train.shape == out_inference.shape

    def test_multiscale_config_type_validation(self, rng, base_disc_config):
        """Test that MultiScalePatchGAN raises TypeError for wrong config type."""
        with pytest.raises(TypeError, match="must be MultiScalePatchGANConfig"):
            MultiScalePatchGANDiscriminator(config=base_disc_config, rngs=nnx.Rngs(rng))


class TestPatchGANIntegration:
    """Integration tests for PatchGAN discriminators."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(123)

    def test_patchgan_with_different_activations(self, rng):
        """Test PatchGAN with different activation functions."""
        input_shape = (3, 64, 64)
        activations = ["leaky_relu", "relu", "silu"]

        for activation in activations:
            config = PatchGANDiscriminatorConfig(
                name=f"test_patchgan_{activation}",
                input_shape=input_shape,
                hidden_dims=(64, 128, 256),
                num_filters=64,
                num_layers=3,
                activation=activation,
            )
            discriminator = PatchGANDiscriminator(
                config=config,
                rngs=nnx.Rngs(rng),
            )

            batch_size = 2
            input_images = jax.random.normal(rng, (batch_size, *input_shape))
            discriminator.eval()
            features = discriminator(input_images)

            assert len(features) > 0
            assert features[-1].shape[0] == batch_size

    def test_patchgan_feature_matching_capability(self, rng):
        """Test PatchGAN's capability for feature matching loss."""
        input_shape = (3, 64, 64)
        config = PatchGANDiscriminatorConfig(
            name="test_patchgan_feature_matching",
            input_shape=input_shape,
            hidden_dims=(64, 128, 256),
            activation="leaky_relu",
            num_filters=64,
            num_layers=3,
        )
        discriminator = PatchGANDiscriminator(
            config=config,
            rngs=nnx.Rngs(rng),
        )

        batch_size = 2
        real_images = jax.random.normal(rng, (batch_size, *input_shape))
        fake_images = jax.random.normal(jax.random.split(rng)[0], (batch_size, *input_shape))

        # Get features for both real and fake images
        discriminator.train()
        real_features = discriminator(real_images)
        fake_features = discriminator(fake_images)

        # Should have same number of features
        assert len(real_features) == len(fake_features)

        # Can compute feature matching loss
        feature_loss = 0.0
        for rf, ff in zip(real_features[:-1], fake_features[:-1]):  # Exclude final output
            feature_loss += jnp.mean(jnp.abs(rf - ff))

        assert jnp.isfinite(feature_loss)
        assert feature_loss >= 0.0

    def test_patchgan_discriminator_combined_loss(self, rng):
        """Test combined adversarial and feature matching loss."""
        input_shape = (3, 64, 64)
        config = PatchGANDiscriminatorConfig(
            name="test_patchgan_combined_loss",
            input_shape=input_shape,
            hidden_dims=(64, 128, 256),
            activation="leaky_relu",
            num_filters=64,
            num_layers=3,
        )
        discriminator = PatchGANDiscriminator(
            config=config,
            rngs=nnx.Rngs(rng),
        )

        batch_size = 2
        real_images = jax.random.normal(rng, (batch_size, *input_shape))
        fake_images = jax.random.normal(jax.random.split(rng)[0], (batch_size, *input_shape))

        # Get discriminator outputs
        discriminator.train()
        real_features = discriminator(real_images)
        fake_features = discriminator(fake_images)

        real_scores = real_features[-1]
        fake_scores = fake_features[-1]

        # Adversarial loss (simple version)
        real_loss = jnp.mean((real_scores - 1) ** 2)  # LSGAN style
        fake_loss = jnp.mean(fake_scores**2)
        adversarial_loss = real_loss + fake_loss

        # Feature matching loss
        feature_loss = 0.0
        for rf, ff in zip(real_features[:-1], fake_features[:-1]):
            feature_loss += jnp.mean(jnp.abs(rf - ff))

        # Combined loss
        total_loss = adversarial_loss + 10.0 * feature_loss

        assert jnp.isfinite(total_loss)
        assert total_loss >= 0.0

    def test_multiscale_vs_single_scale_consistency(self, rng):
        """Test that multi-scale discriminator with one discriminator behaves like single-scale."""
        input_shape = (3, 64, 64)

        # Single-scale discriminator
        single_config = PatchGANDiscriminatorConfig(
            name="test_single_scale",
            input_shape=input_shape,
            hidden_dims=(64, 128, 256),
            activation="leaky_relu",
            num_filters=64,
            num_layers=3,
        )
        single_disc = PatchGANDiscriminator(
            config=single_config,
            rngs=nnx.Rngs(rng),
        )

        # Multi-scale with only one discriminator
        multi_config = MultiScalePatchGANConfig(
            discriminator=PatchGANDiscriminatorConfig(
                name="test_multi_scale",
                input_shape=input_shape,
                hidden_dims=(64, 128, 256),
                activation="leaky_relu",
                num_filters=64,
                num_layers=3,
            ),
            num_discriminators=1,
        )
        multi_disc = MultiScalePatchGANDiscriminator(
            config=multi_config,
            rngs=nnx.Rngs(jax.random.split(rng)[0]),
        )

        batch_size = 2
        input_images = jax.random.normal(rng, (batch_size, *input_shape))

        # Get outputs
        single_disc.eval()
        single_features = single_disc(input_images)
        multi_disc.eval()
        multi_outputs, multi_features = multi_disc(input_images)

        # Multi-scale should have one output and one feature list
        assert len(multi_outputs) == 1
        assert len(multi_features) == 1

        # Output shapes should be similar (may differ due to different parameter initialization)
        assert single_features[-1].shape == multi_outputs[0].shape
