"""Test zoo loading with actual config files.

Note: The ModelZoo class currently uses ModelConfig (legacy pattern).
Tests that depend on factory integration (create_model) are skipped
until the zoo is refactored to use specific dataclass configs.
"""

import pytest

from artifex.generative_models.zoo import ModelZoo


@pytest.mark.skip(
    reason="Zoo config loading depends on ModelConfig which needs refactoring "
    "to use specific dataclass configs."
)
def test_zoo_loads_configs():
    """Test that the global zoo loads configuration files."""
    test_zoo = ModelZoo()
    configs = test_zoo.list_configs()

    assert "vae_mnist" in configs
    assert "gan_cifar10" in configs
    assert "diffusion_celeba" in configs
    assert "vae_protein" in configs


@pytest.mark.skip(reason="Zoo config loading depends on ModelConfig which needs refactoring.")
def test_zoo_list_by_category():
    """Test filtering configs by category."""
    test_zoo = ModelZoo()

    vision_configs = test_zoo.list_configs(category="vision")
    assert "vae_mnist" in vision_configs
    assert "gan_cifar10" in vision_configs
    assert "diffusion_celeba" in vision_configs
    assert "vae_protein" not in vision_configs

    protein_configs = test_zoo.list_configs(category="protein")
    assert "vae_protein" in protein_configs
    assert "vae_mnist" not in protein_configs


@pytest.mark.skip(reason="Zoo config loading depends on ModelConfig which needs refactoring.")
def test_zoo_get_config():
    """Test getting a specific configuration."""
    test_zoo = ModelZoo()

    config = test_zoo.get_config("vae_mnist")
    assert config.name == "vae_mnist"
    assert config.model_class == "artifex.generative_models.models.vae.VAE"
    assert config.input_dim == (28, 28, 1)
    assert config.output_dim == 64


@pytest.mark.skip(
    reason="ModelZoo.create_model uses ModelConfig which is not supported by factory. "
    "Zoo needs refactoring to use specific dataclass configs."
)
def test_zoo_create_model():
    """Test creating a model from zoo."""
    pass


@pytest.mark.skip(
    reason="ModelZoo.create_model uses ModelConfig which is not supported by factory. "
    "Zoo needs refactoring to use specific dataclass configs."
)
def test_zoo_create_with_overrides():
    """Test creating model with configuration overrides."""
    pass
