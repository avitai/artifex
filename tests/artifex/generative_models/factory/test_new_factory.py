"""Tests for the centralized factory system with dataclass configs."""

import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
    CouplingNetworkConfig,
    DCGANConfig,
    DecoderConfig,
    EncoderConfig,
    RealNVPConfig,
    VAEConfig,
)
from artifex.generative_models.core.configuration.diffusion_config import (
    DDPMConfig,
    NoiseScheduleConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.factory import create_model
from artifex.generative_models.factory.core import ModelFactory
from artifex.generative_models.factory.registry import (
    BuilderNotFoundError,
    DuplicateBuilderError,
    ModelTypeRegistry,
)


@pytest.fixture
def rngs():
    """Create random number generators."""
    return nnx.Rngs(params=0, dropout=1, sample=2)


@pytest.fixture
def vae_config():
    """Create VAE configuration using dataclass config."""
    encoder = EncoderConfig(
        name="encoder",
        input_shape=(28, 28, 1),
        latent_dim=64,
        hidden_dims=(256, 128),
        activation="relu",
    )
    decoder = DecoderConfig(
        name="decoder",
        latent_dim=64,
        output_shape=(28, 28, 1),
        hidden_dims=(128, 256),
        activation="relu",
    )
    return VAEConfig(
        name="test_vae",
        encoder=encoder,
        decoder=decoder,
        encoder_type="dense",
        kl_weight=1.0,
    )


@pytest.fixture
def gan_config():
    """Create GAN configuration using dataclass config."""
    generator = ConvGeneratorConfig(
        name="generator",
        latent_dim=100,
        output_shape=(1, 28, 28),
        hidden_dims=(256, 128, 64),
        activation="relu",
    )
    discriminator = ConvDiscriminatorConfig(
        name="discriminator",
        input_shape=(1, 28, 28),
        hidden_dims=(64, 128, 256),
        activation="leaky_relu",
    )
    return DCGANConfig(
        name="test_gan",
        generator=generator,
        discriminator=discriminator,
    )


@pytest.fixture
def diffusion_config():
    """Create diffusion configuration using dataclass config."""
    backbone = UNetBackboneConfig(
        name="backbone",
        hidden_dims=(128, 256, 128),
        activation="gelu",
        in_channels=1,
        out_channels=1,
    )
    schedule = NoiseScheduleConfig(
        name="schedule",
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )
    return DDPMConfig(
        name="test_diffusion",
        backbone=backbone,
        noise_schedule=schedule,
        input_shape=(28, 28, 1),
    )


class TestModelFactory:
    """Test the centralized model factory with dataclass configs."""

    def test_create_vae(self, vae_config, rngs):
        """Test creating a VAE model."""
        model = create_model(vae_config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "encode")
        assert hasattr(model, "decode")
        assert hasattr(model, "latent_dim")
        assert model.latent_dim == 64

    def test_create_gan(self, gan_config, rngs):
        """Test creating a GAN model."""
        model = create_model(gan_config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "generator")
        assert hasattr(model, "discriminator")

    def test_create_diffusion(self, diffusion_config, rngs):
        """Test creating a diffusion model."""
        model = create_model(diffusion_config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "q_sample")
        assert hasattr(model, "p_sample")

    def test_create_with_modality(self, vae_config, rngs):
        """Test creating a model with modality adaptation."""
        # This will apply the modality adapter if registered
        try:
            model = create_model(vae_config, modality="protein", rngs=rngs)
            assert model is not None
        except ValueError as e:
            if "Failed to apply modality" not in str(e):
                raise

    def test_invalid_config_type(self, rngs):
        """Test error handling for invalid config types."""
        with pytest.raises(TypeError, match="Expected dataclass config.*got dict"):
            create_model({"latent_dim": 32}, rngs=rngs)

    def test_none_config(self, rngs):
        """Test error handling for None config."""
        with pytest.raises(TypeError, match="config cannot be None"):
            create_model(None, rngs=rngs)

    def test_factory_instance(self):
        """Test factory instance management."""
        factory1 = ModelFactory()
        factory2 = ModelFactory()

        # Each factory should have its own registry
        assert factory1.registry is not factory2.registry

        # Should have default builders registered
        builders = factory1.registry.list_builders()
        assert "vae" in builders
        assert "gan" in builders
        assert "diffusion" in builders
        assert "flow" in builders
        assert "ebm" in builders
        assert "autoregressive" in builders
        assert "geometric" in builders


class TestModelTypeRegistry:
    """Test the model type registry."""

    def test_register_builder(self):
        """Test registering a builder."""
        registry = ModelTypeRegistry()

        class MockBuilder:
            def build(self, config, *, rngs, **kwargs):
                return {"type": "mock"}

        builder = MockBuilder()
        registry.register("mock", builder)

        assert registry.get_builder("mock") == builder
        assert "mock" in registry.list_builders()

    def test_duplicate_builder_error(self):
        """Test error on duplicate registration."""
        registry = ModelTypeRegistry()

        class MockBuilder:
            def build(self, config, *, rngs, **kwargs):
                return {"type": "mock"}

        builder = MockBuilder()
        registry.register("mock", builder)

        with pytest.raises(DuplicateBuilderError):
            registry.register("mock", builder)

    def test_builder_not_found_error(self):
        """Test error when builder not found."""
        registry = ModelTypeRegistry()

        with pytest.raises(BuilderNotFoundError):
            registry.get_builder("nonexistent")

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = ModelTypeRegistry()

        class MockBuilder:
            def build(self, config, *, rngs, **kwargs):
                return {"type": "mock"}

        registry.register("mock", MockBuilder())
        assert len(registry.list_builders()) == 1

        registry.clear()
        assert len(registry.list_builders()) == 0


class TestFactoryIntegration:
    """Integration tests for the factory system."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing."""
        return nnx.Rngs(params=0, dropout=1, sample=2)

    def test_all_model_types(self, rngs):
        """Test creating all supported model types."""
        # VAE
        encoder = EncoderConfig(
            name="encoder",
            input_shape=(28, 28, 1),
            latent_dim=32,
            hidden_dims=(128,),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=32,
            output_shape=(28, 28, 1),
            hidden_dims=(128,),
            activation="relu",
        )
        vae_config = VAEConfig(
            name="test_vae",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )
        vae = create_model(vae_config, rngs=rngs)
        assert vae is not None
        assert hasattr(vae, "encode")
        assert hasattr(vae, "decode")

        # GAN
        generator = ConvGeneratorConfig(
            name="generator",
            latent_dim=100,
            output_shape=(1, 28, 28),
            hidden_dims=(128,),
            activation="relu",
        )
        discriminator = ConvDiscriminatorConfig(
            name="discriminator",
            input_shape=(1, 28, 28),
            hidden_dims=(128,),
            activation="leaky_relu",
        )
        gan_config = DCGANConfig(
            name="test_gan",
            generator=generator,
            discriminator=discriminator,
        )
        gan = create_model(gan_config, rngs=rngs)
        assert gan is not None
        assert hasattr(gan, "generator")
        assert hasattr(gan, "discriminator")

        # Flow
        coupling_network = CouplingNetworkConfig(
            name="coupling",
            hidden_dims=(128,),
            activation="relu",
            network_type="mlp",
        )
        flow_config = RealNVPConfig(
            name="test_flow",
            coupling_network=coupling_network,
            input_dim=2,
            num_coupling_layers=2,
        )
        flow = create_model(flow_config, rngs=rngs)
        assert flow is not None
        assert hasattr(flow, "forward")
        assert hasattr(flow, "inverse")

        # EBM
        energy_network = EnergyNetworkConfig(
            name="energy_net",
            network_type="cnn",
            hidden_dims=(128,),
            activation="silu",
        )
        mcmc = MCMCConfig(name="mcmc")
        sample_buffer = SampleBufferConfig(name="buffer")
        ebm_config = DeepEBMConfig(
            name="test_ebm",
            input_shape=(28, 28, 1),
            energy_network=energy_network,
            mcmc=mcmc,
            sample_buffer=sample_buffer,
        )
        ebm = create_model(ebm_config, rngs=rngs)
        assert ebm is not None
        assert hasattr(ebm, "energy")

    def test_model_type_extraction(self):
        """Test that factory extracts model type from config type."""
        factory = ModelFactory()

        # Create configs of different types
        encoder = EncoderConfig(
            name="encoder",
            input_shape=(28, 28, 1),
            latent_dim=32,
            hidden_dims=(64,),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=32,
            output_shape=(28, 28, 1),
            hidden_dims=(64,),
            activation="relu",
        )
        vae_config = VAEConfig(
            name="test",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

        generator = ConvGeneratorConfig(
            name="generator",
            latent_dim=32,
            output_shape=(1, 28, 28),
            hidden_dims=(64,),
            activation="relu",
        )
        discriminator = ConvDiscriminatorConfig(
            name="discriminator",
            input_shape=(1, 28, 28),
            hidden_dims=(64,),
            activation="leaky_relu",
        )
        gan_config = DCGANConfig(
            name="test",
            generator=generator,
            discriminator=discriminator,
        )

        # Factory should extract model type from config type
        assert factory._extract_model_type(vae_config) == "vae"
        assert factory._extract_model_type(gan_config) == "gan"
