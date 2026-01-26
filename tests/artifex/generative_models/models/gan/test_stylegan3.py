"""Tests for StyleGAN3 Generator and Discriminator.

Tests cover:
- StyleGAN3Generator with config
- StyleGAN3Discriminator with config
- Forward pass functionality
- JIT compatibility
- Sampling functionality
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.configuration.network_configs import (
    StyleGAN3DiscriminatorConfig,
    StyleGAN3GeneratorConfig,
)
from artifex.generative_models.models.gan.stylegan3 import (
    MappingNetwork,
    StyleGAN3Discriminator,
    StyleGAN3Generator,
    StyleModulatedConv,
    SynthesisBlock,
    SynthesisNetwork,
)


class TestStyleGAN3GeneratorConfig:
    """Tests for StyleGAN3GeneratorConfig."""

    def test_config_creation_defaults(self):
        """Test creating config with default values."""
        config = StyleGAN3GeneratorConfig(
            name="test_generator",
            latent_dim=512,
            hidden_dims=(512,),
            output_shape=(64, 64, 3),
            activation="leaky_relu",
        )
        assert config.style_dim == 512
        assert config.mapping_layers == 8
        assert config.img_resolution == 256
        assert config.img_channels == 3

    def test_config_creation_custom(self):
        """Test creating config with custom values."""
        config = StyleGAN3GeneratorConfig(
            name="test_generator",
            latent_dim=256,
            hidden_dims=(256,),
            output_shape=(32, 32, 3),
            activation="leaky_relu",
            style_dim=256,
            mapping_layers=4,
            img_resolution=32,
            img_channels=1,
        )
        assert config.latent_dim == 256
        assert config.style_dim == 256
        assert config.mapping_layers == 4
        assert config.img_resolution == 32
        assert config.img_channels == 1

    def test_config_invalid_resolution(self):
        """Test that invalid resolution raises error."""
        with pytest.raises(ValueError, match="power of 2"):
            StyleGAN3GeneratorConfig(
                name="test_generator",
                latent_dim=512,
                hidden_dims=(512,),
                output_shape=(30, 30, 3),
                activation="leaky_relu",
                img_resolution=30,  # Not a power of 2
            )

    def test_config_resolution_too_small(self):
        """Test that resolution < 4 raises error."""
        with pytest.raises(ValueError, match="at least 4"):
            StyleGAN3GeneratorConfig(
                name="test_generator",
                latent_dim=512,
                hidden_dims=(512,),
                output_shape=(2, 2, 3),
                activation="leaky_relu",
                img_resolution=2,
            )


class TestStyleGAN3DiscriminatorConfig:
    """Tests for StyleGAN3DiscriminatorConfig."""

    def test_config_creation_defaults(self):
        """Test creating config with default values."""
        config = StyleGAN3DiscriminatorConfig(
            name="test_discriminator",
            input_shape=(64, 64, 3),
            hidden_dims=(64, 128),
            activation="leaky_relu",
        )
        assert config.img_resolution == 256
        assert config.img_channels == 3
        assert config.base_channels == 64
        assert config.max_channels == 512

    def test_config_creation_custom(self):
        """Test creating config with custom values."""
        config = StyleGAN3DiscriminatorConfig(
            name="test_discriminator",
            input_shape=(32, 32, 1),
            hidden_dims=(32, 64),
            activation="leaky_relu",
            img_resolution=32,
            img_channels=1,
            base_channels=32,
            max_channels=256,
        )
        assert config.img_resolution == 32
        assert config.img_channels == 1
        assert config.base_channels == 32
        assert config.max_channels == 256

    def test_config_invalid_resolution(self):
        """Test that invalid resolution raises error."""
        with pytest.raises(ValueError, match="power of 2"):
            StyleGAN3DiscriminatorConfig(
                name="test_discriminator",
                input_shape=(30, 30, 3),
                hidden_dims=(64,),
                activation="leaky_relu",
                img_resolution=30,
            )


class TestMappingNetwork:
    """Tests for MappingNetwork component."""

    @pytest.fixture
    def rng(self):
        return jax.random.key(42)

    def test_mapping_network_forward(self, rng):
        """Test MappingNetwork forward pass."""
        mapping = MappingNetwork(
            latent_dim=64,
            style_dim=64,
            num_layers=2,
            num_ws=4,
            rngs=nnx.Rngs(rng),
        )

        z = jax.random.normal(rng, (2, 64))
        w = mapping(z)

        assert w.shape == (2, 4, 64)

    def test_mapping_network_truncation(self, rng):
        """Test MappingNetwork with truncation."""
        mapping = MappingNetwork(
            latent_dim=64,
            style_dim=64,
            num_layers=2,
            num_ws=4,
            rngs=nnx.Rngs(rng),
        )

        z = jax.random.normal(rng, (2, 64))
        w = mapping(z, truncation_psi=0.5, truncation_cutoff=2)

        assert w.shape == (2, 4, 64)


class TestStyleModulatedConv:
    """Tests for StyleModulatedConv component."""

    @pytest.fixture
    def rng(self):
        return jax.random.key(42)

    def test_style_modulated_conv_forward(self, rng):
        """Test StyleModulatedConv forward pass."""
        conv = StyleModulatedConv(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            style_dim=64,
            rngs=nnx.Rngs(rng),
        )

        x = jax.random.normal(rng, (2, 8, 8, 32))
        style = jax.random.normal(rng, (2, 64))
        out = conv(x, style)

        assert out.shape == (2, 8, 8, 64)


class TestSynthesisBlock:
    """Tests for SynthesisBlock component."""

    @pytest.fixture
    def rng(self):
        return jax.random.key(42)

    def test_synthesis_block_no_upsample(self, rng):
        """Test SynthesisBlock without upsampling."""
        block = SynthesisBlock(
            in_channels=64,
            out_channels=64,
            style_dim=64,
            upsample=False,
            rngs=nnx.Rngs(rng),
        )

        x = jax.random.normal(rng, (2, 8, 8, 64))
        style1 = jax.random.normal(rng, (2, 64))
        style2 = jax.random.normal(rng, (2, 64))
        out = block(x, style1, style2)

        assert out.shape == (2, 8, 8, 64)

    def test_synthesis_block_with_upsample(self, rng):
        """Test SynthesisBlock with upsampling."""
        block = SynthesisBlock(
            in_channels=64,
            out_channels=32,
            style_dim=64,
            upsample=True,
            rngs=nnx.Rngs(rng),
        )

        x = jax.random.normal(rng, (2, 8, 8, 64))
        style1 = jax.random.normal(rng, (2, 64))
        style2 = jax.random.normal(rng, (2, 64))
        out = block(x, style1, style2)

        assert out.shape == (2, 16, 16, 32)


class TestSynthesisNetwork:
    """Tests for SynthesisNetwork component."""

    @pytest.fixture
    def rng(self):
        return jax.random.key(42)

    def test_synthesis_network_forward(self, rng):
        """Test SynthesisNetwork forward pass."""
        synthesis = SynthesisNetwork(
            style_dim=64,
            img_resolution=16,
            img_channels=3,
            rngs=nnx.Rngs(rng),
        )

        # num_ws = 2 * log2(16/4) = 2 * 2 = 4
        w = jax.random.normal(rng, (2, 4, 64))
        out = synthesis(w)

        assert out.shape == (2, 16, 16, 3)
        # Output should be in [-1, 1] due to tanh
        assert jnp.all(out >= -1) and jnp.all(out <= 1)


class TestStyleGAN3Generator:
    """Tests for StyleGAN3Generator."""

    @pytest.fixture
    def rng(self):
        return jax.random.key(42)

    @pytest.fixture
    def generator_config(self):
        return StyleGAN3GeneratorConfig(
            name="test_generator",
            latent_dim=64,
            hidden_dims=(64,),
            output_shape=(16, 16, 3),
            activation="leaky_relu",
            style_dim=64,
            mapping_layers=2,
            img_resolution=16,
            img_channels=3,
        )

    @pytest.fixture
    def generator(self, rng, generator_config):
        return StyleGAN3Generator(config=generator_config, rngs=nnx.Rngs(rng))

    def test_generator_init_stores_config(self, generator, generator_config):
        """Test generator stores config."""
        assert generator.config == generator_config
        assert generator.latent_dim == generator_config.latent_dim
        assert generator.style_dim == generator_config.style_dim
        assert generator.img_resolution == generator_config.img_resolution
        assert generator.img_channels == generator_config.img_channels

    def test_generator_init_invalid_config(self, rng):
        """Test generator raises error for invalid config type."""
        with pytest.raises(TypeError, match="StyleGAN3GeneratorConfig"):
            StyleGAN3Generator(config="invalid", rngs=nnx.Rngs(rng))

    def test_generator_forward(self, rng, generator):
        """Test generator forward pass."""
        z = jax.random.normal(rng, (2, 64))
        out = generator(z)

        assert out.shape == (2, 16, 16, 3)
        # Output should be in [-1, 1] due to tanh
        assert jnp.all(out >= -1) and jnp.all(out <= 1)

    def test_generator_forward_with_truncation(self, rng, generator):
        """Test generator forward pass with truncation."""
        z = jax.random.normal(rng, (2, 64))
        out = generator(z, truncation_psi=0.7)

        assert out.shape == (2, 16, 16, 3)

    def test_generator_sample(self, rng, generator):
        """Test generator sample method."""
        out = generator.sample(num_samples=3, rngs=nnx.Rngs(rng))

        assert out.shape == (3, 16, 16, 3)

    def test_generator_jit_forward(self, rng, generator):
        """Test JIT compilation of generator forward pass."""

        @nnx.jit
        def jit_forward(model, z):
            return model(z)

        z = jax.random.normal(rng, (2, 64))
        out = jit_forward(generator, z)

        assert out.shape == (2, 16, 16, 3)

    def test_generator_jit_sample(self, generator):
        """Test JIT compilation of generator sample method."""

        @nnx.jit
        def jit_sample(model, rngs):
            return model.sample(num_samples=2, rngs=rngs)

        rngs = nnx.Rngs(jax.random.key(123))
        out = jit_sample(generator, rngs)

        assert out.shape == (2, 16, 16, 3)


class TestStyleGAN3Discriminator:
    """Tests for StyleGAN3Discriminator."""

    @pytest.fixture
    def rng(self):
        return jax.random.key(42)

    @pytest.fixture
    def discriminator_config(self):
        return StyleGAN3DiscriminatorConfig(
            name="test_discriminator",
            input_shape=(16, 16, 3),
            hidden_dims=(64,),
            activation="leaky_relu",
            img_resolution=16,
            img_channels=3,
            base_channels=32,
            max_channels=128,
        )

    @pytest.fixture
    def discriminator(self, rng, discriminator_config):
        return StyleGAN3Discriminator(config=discriminator_config, rngs=nnx.Rngs(rng))

    def test_discriminator_init_stores_config(self, discriminator, discriminator_config):
        """Test discriminator stores config."""
        assert discriminator.config == discriminator_config
        assert discriminator.img_resolution == discriminator_config.img_resolution
        assert discriminator.img_channels == discriminator_config.img_channels

    def test_discriminator_init_invalid_config(self, rng):
        """Test discriminator raises error for invalid config type."""
        with pytest.raises(TypeError, match="StyleGAN3DiscriminatorConfig"):
            StyleGAN3Discriminator(config="invalid", rngs=nnx.Rngs(rng))

    def test_discriminator_forward(self, rng, discriminator):
        """Test discriminator forward pass."""
        x = jax.random.normal(rng, (2, 16, 16, 3))
        out = discriminator(x)

        assert out.shape == (2, 1)

    def test_discriminator_jit_forward(self, rng, discriminator):
        """Test JIT compilation of discriminator forward pass."""

        @nnx.jit
        def jit_forward(model, x):
            return model(x)

        x = jax.random.normal(rng, (2, 16, 16, 3))
        out = jit_forward(discriminator, x)

        assert out.shape == (2, 1)


class TestStyleGAN3Integration:
    """Integration tests for StyleGAN3 components."""

    @pytest.fixture
    def rng(self):
        return jax.random.key(42)

    def test_generator_discriminator_pipeline(self, rng):
        """Test generator and discriminator work together."""
        gen_config = StyleGAN3GeneratorConfig(
            name="test_generator",
            latent_dim=64,
            hidden_dims=(64,),
            output_shape=(16, 16, 3),
            activation="leaky_relu",
            style_dim=64,
            mapping_layers=2,
            img_resolution=16,
            img_channels=3,
        )

        disc_config = StyleGAN3DiscriminatorConfig(
            name="test_discriminator",
            input_shape=(16, 16, 3),
            hidden_dims=(64,),
            activation="leaky_relu",
            img_resolution=16,
            img_channels=3,
            base_channels=32,
            max_channels=128,
        )

        generator = StyleGAN3Generator(config=gen_config, rngs=nnx.Rngs(rng))
        discriminator = StyleGAN3Discriminator(config=disc_config, rngs=nnx.Rngs(rng))

        # Generate fake images
        z = jax.random.normal(rng, (2, 64))
        fake_images = generator(z)

        # Discriminate fake images
        disc_output = discriminator(fake_images)

        assert disc_output.shape == (2, 1)

    def test_jit_full_pipeline(self, rng):
        """Test JIT compilation of full pipeline."""
        gen_config = StyleGAN3GeneratorConfig(
            name="test_generator",
            latent_dim=64,
            hidden_dims=(64,),
            output_shape=(16, 16, 3),
            activation="leaky_relu",
            style_dim=64,
            mapping_layers=2,
            img_resolution=16,
            img_channels=3,
        )

        disc_config = StyleGAN3DiscriminatorConfig(
            name="test_discriminator",
            input_shape=(16, 16, 3),
            hidden_dims=(64,),
            activation="leaky_relu",
            img_resolution=16,
            img_channels=3,
            base_channels=32,
            max_channels=128,
        )

        generator = StyleGAN3Generator(config=gen_config, rngs=nnx.Rngs(rng))
        discriminator = StyleGAN3Discriminator(config=disc_config, rngs=nnx.Rngs(rng))

        @nnx.jit
        def generate_and_discriminate(gen, disc, z):
            fake = gen(z)
            return disc(fake)

        z = jax.random.normal(rng, (2, 64))
        out = generate_and_discriminate(generator, discriminator, z)

        assert out.shape == (2, 1)
