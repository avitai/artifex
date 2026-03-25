"""Runtime contracts for factory-ready modality adaptation."""

import jax
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DecoderConfig,
    EncoderConfig,
    VAEConfig,
)
from artifex.generative_models.factory import create_model
from artifex.generative_models.modalities.registry import list_modalities


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Create RNGs for model creation."""
    return nnx.Rngs(
        params=jax.random.PRNGKey(0),
        dropout=jax.random.PRNGKey(1),
        sample=jax.random.PRNGKey(2),
    )


@pytest.fixture
def vae_config() -> VAEConfig:
    """Create a minimal typed VAE config for modality adaptation tests."""
    encoder = EncoderConfig(
        name="encoder",
        input_shape=(16, 16, 3),
        latent_dim=8,
        hidden_dims=(32, 16),
        activation="relu",
    )
    decoder = DecoderConfig(
        name="decoder",
        latent_dim=8,
        output_shape=(16, 16, 3),
        hidden_dims=(16, 32),
        activation="relu",
    )
    return VAEConfig(
        name="modality_runtime_contract",
        encoder=encoder,
        decoder=decoder,
        encoder_type="dense",
    )


def test_default_registry_only_exposes_factory_ready_modalities() -> None:
    """The public default registry should list only factory-usable modalities."""
    assert set(list_modalities()) == {"image", "molecular", "protein"}


@pytest.mark.parametrize("modality", ["image", "molecular", "protein"])
def test_factory_supports_all_retained_registered_modalities(
    modality: str,
    vae_config: VAEConfig,
    rngs: nnx.Rngs,
) -> None:
    """Every retained registered modality should work through the real factory path."""
    model = create_model(vae_config, modality=modality, rngs=rngs)
    assert model is not None


@pytest.mark.parametrize("modality", ["audio", "text", "tabular", "timeseries"])
def test_factory_rejects_modalities_removed_from_default_registry(
    modality: str,
    vae_config: VAEConfig,
    rngs: nnx.Rngs,
) -> None:
    """Unsupported default modalities should fail at the registry boundary."""
    with pytest.raises(ValueError, match="Unknown modality"):
        create_model(vae_config, modality=modality, rngs=rngs)
