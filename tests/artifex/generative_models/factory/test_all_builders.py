"""Test that all builders are properly registered and work with dataclass configs."""

import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
    CouplingNetworkConfig,
    DCGANConfig,
    DecoderConfig,
    EncoderConfig,
    PointCloudConfig,
    PointCloudNetworkConfig,
    RealNVPConfig,
    TransformerConfig,
    TransformerNetworkConfig,
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
from artifex.generative_models.factory.core import create_model, ModelFactory


class TestAllBuilders:
    """Test all builders are registered and functional with dataclass configs."""

    @pytest.fixture
    def factory(self):
        """Create factory instance."""
        return ModelFactory()

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing with all required streams."""
        return nnx.Rngs(params=0, dropout=1, sample=2)

    def test_all_builders_registered(self, factory):
        """Test that all expected builders are registered."""
        expected_builders = [
            "vae",
            "gan",
            "diffusion",
            "flow",
            "ebm",
            "autoregressive",
            "geometric",
        ]

        for builder_type in expected_builders:
            builder = factory.registry.get_builder(builder_type)
            assert builder is not None

    def test_vae_builder_works(self, factory, rngs):
        """Test VAE builder creates models with VAEConfig."""
        encoder = EncoderConfig(
            name="encoder",
            input_shape=(28, 28, 1),
            latent_dim=64,
            hidden_dims=(512, 256),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=64,
            output_shape=(28, 28, 1),
            hidden_dims=(256, 512),
            activation="relu",
        )
        config = VAEConfig(
            name="test_vae",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

        model = factory.create(config, rngs=rngs)
        assert model is not None
        assert hasattr(model, "encode")
        assert hasattr(model, "decode")

    def test_gan_builder_works(self, factory, rngs):
        """Test GAN builder creates models with DCGANConfig."""
        generator = ConvGeneratorConfig(
            name="generator",
            latent_dim=100,
            output_shape=(3, 32, 32),
            hidden_dims=(128, 256, 512, 256, 128),
            activation="relu",
        )
        discriminator = ConvDiscriminatorConfig(
            name="discriminator",
            input_shape=(3, 32, 32),
            hidden_dims=(128, 256, 512, 256, 128),
            activation="leaky_relu",
        )
        config = DCGANConfig(
            name="test_gan",
            generator=generator,
            discriminator=discriminator,
        )

        model = factory.create(config, rngs=rngs)
        assert model is not None
        assert hasattr(model, "generator")
        assert hasattr(model, "discriminator")

    def test_diffusion_builder_works(self, factory, rngs):
        """Test Diffusion builder creates models with DDPMConfig."""
        backbone = UNetBackboneConfig(
            name="test_backbone",
            hidden_dims=(64, 128, 256),
            activation="silu",
            in_channels=3,
            out_channels=3,
        )
        schedule = NoiseScheduleConfig(name="test_schedule")
        config = DDPMConfig(
            name="test_diffusion",
            backbone=backbone,
            noise_schedule=schedule,
            input_shape=(32, 32, 3),
        )

        model = factory.create(config, rngs=rngs)
        assert model is not None
        assert hasattr(model, "q_sample")
        assert hasattr(model, "p_sample")

    def test_flow_builder_works(self, factory, rngs):
        """Test Flow builder creates models with RealNVPConfig."""
        coupling_network = CouplingNetworkConfig(
            name="coupling",
            hidden_dims=(256, 256),
            activation="relu",
            network_type="mlp",
        )
        config = RealNVPConfig(
            name="test_flow",
            coupling_network=coupling_network,
            input_dim=784,
            num_coupling_layers=4,
        )

        model = factory.create(config, rngs=rngs)
        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "inverse")

    def test_ebm_builder_works(self, factory, rngs):
        """Test EBM builder creates models with DeepEBMConfig."""
        energy_network = EnergyNetworkConfig(
            name="energy_net",
            network_type="cnn",
            hidden_dims=(64, 128, 256),
            activation="silu",
        )
        mcmc = MCMCConfig(name="mcmc")
        sample_buffer = SampleBufferConfig(name="buffer")
        config = DeepEBMConfig(
            name="test_ebm",
            input_shape=(32, 32, 3),
            energy_network=energy_network,
            mcmc=mcmc,
            sample_buffer=sample_buffer,
        )

        model = factory.create(config, rngs=rngs)
        assert model is not None
        assert hasattr(model, "energy")

    def test_autoregressive_builder_works(self, factory, rngs):
        """Test Autoregressive builder creates models with TransformerConfig."""
        network = TransformerNetworkConfig(
            name="network",
            hidden_dims=(512,),
            activation="gelu",
            embed_dim=512,
            num_heads=8,
            mlp_ratio=4.0,
        )
        config = TransformerConfig(
            name="test_transformer",
            vocab_size=50257,
            sequence_length=1024,
            num_layers=4,
            network=network,
        )

        model = factory.create(config, rngs=rngs)
        assert model is not None
        assert hasattr(model, "generate")

    def test_geometric_builder_works(self, factory, rngs):
        """Test Geometric builder creates models with PointCloudConfig."""
        network = PointCloudNetworkConfig(
            name="network",
            hidden_dims=(64, 128, 256),
            activation="relu",
        )
        config = PointCloudConfig(
            name="test_pointcloud",
            num_points=1024,
            point_dim=3,
            network=network,
            global_features_dim=128,
        )

        model = factory.create(config, rngs=rngs)
        assert model is not None


class TestFactoryRejectsInvalidConfigs:
    """Test that factory properly rejects invalid config types."""

    @pytest.fixture
    def rngs(self):
        """Create test RNGs with all required streams."""
        return nnx.Rngs(params=0, dropout=1, sample=2)

    def test_rejects_dict_config(self, rngs):
        """Test that dict configs are rejected."""
        with pytest.raises(TypeError, match="Expected dataclass config.*got dict"):
            create_model({"latent_dim": 64}, rngs=rngs)

    def test_rejects_none_config(self, rngs):
        """Test that None config is rejected."""
        with pytest.raises(TypeError, match="config cannot be None"):
            create_model(None, rngs=rngs)

    def test_rejects_string_config(self, rngs):
        """Test that string config is rejected."""
        with pytest.raises(TypeError, match="Expected dataclass config.*got str"):
            create_model("invalid", rngs=rngs)

    def test_rejects_legacy_class_config(self, rngs):
        """Test that legacy class-based configs are rejected."""

        class LegacyConfig:
            def __init__(self):
                self.latent_dim = 32

        with pytest.raises(TypeError, match="Expected dataclass config"):
            create_model(LegacyConfig(), rngs=rngs)


class TestFactoryWithDifferentModelTypes:
    """Test factory creates correct model types based on config type."""

    @pytest.fixture
    def rngs(self):
        """Create test RNGs with all required streams."""
        return nnx.Rngs(params=0, dropout=1, sample=2)

    def test_vae_config_creates_vae(self, rngs):
        """Test VAEConfig creates VAE model."""
        encoder = EncoderConfig(
            name="encoder",
            input_shape=(28, 28, 1),
            latent_dim=64,
            hidden_dims=(256, 128),
            activation="gelu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=64,
            output_shape=(28, 28, 1),
            hidden_dims=(128, 256),
            activation="gelu",
        )
        config = VAEConfig(
            name="vae_model",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

        vae = create_model(config, rngs=rngs)
        assert hasattr(vae, "encode")
        assert hasattr(vae, "decode")
        assert hasattr(vae, "sample")

    def test_ddpm_config_creates_diffusion(self, rngs):
        """Test DDPMConfig creates diffusion model."""
        backbone = UNetBackboneConfig(
            name="test_backbone",
            hidden_dims=(64, 128, 256, 512),
            activation="silu",
            in_channels=3,
            out_channels=3,
        )
        schedule = NoiseScheduleConfig(name="test_schedule")
        config = DDPMConfig(
            name="diffusion_model",
            backbone=backbone,
            noise_schedule=schedule,
            input_shape=(32, 32, 3),
        )

        diffusion = create_model(config, rngs=rngs)
        assert hasattr(diffusion, "q_sample")
        assert hasattr(diffusion, "p_sample")

    def test_ebm_config_creates_ebm(self, rngs):
        """Test DeepEBMConfig creates EBM model."""
        energy_network = EnergyNetworkConfig(
            name="energy_net",
            network_type="cnn",
            hidden_dims=(64, 128, 256),
            activation="silu",
        )
        mcmc = MCMCConfig(name="mcmc")
        sample_buffer = SampleBufferConfig(name="buffer")
        config = DeepEBMConfig(
            name="ebm_model",
            input_shape=(28, 28, 1),
            energy_network=energy_network,
            mcmc=mcmc,
            sample_buffer=sample_buffer,
        )

        ebm = create_model(config, rngs=rngs)
        assert hasattr(ebm, "energy")
