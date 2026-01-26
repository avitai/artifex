"""Model zoo for pre-configured models."""

import dataclasses
from pathlib import Path
from typing import Any

from flax import nnx

from artifex.generative_models.core.configuration import ModelConfig
from artifex.generative_models.factory import create_model


class ModelZoo:
    """Registry for pre-configured model configurations.

    NOTE: The zoo now uses frozen dataclass configs (ModelConfig).
    Configs are loaded from YAML files and converted to dataclass instances.
    """

    def __init__(self):
        """Initialize the model zoo."""
        self._configs: dict[str, ModelConfig] = {}
        self._zoo_dir = Path(__file__).parent / "configs"
        self._load_zoo_configs()

    def _load_zoo_configs(self):
        """Load all zoo configurations from YAML files."""
        if not self._zoo_dir.exists():
            return

        # Import yaml for loading configs
        try:
            import yaml
        except ImportError:
            # If yaml not available, skip loading
            return

        # Load configurations from all subdirectories
        for subdir in self._zoo_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("_"):
                for config_file in subdir.glob("*.yaml"):
                    try:
                        with open(config_file, "r") as f:
                            config_data = yaml.safe_load(f)

                        # Create ModelConfig from dict
                        config = ModelConfig(**config_data)
                        self._configs[config.name] = config
                    except (yaml.YAMLError, ValueError, TypeError, KeyError, AttributeError):
                        # Skip invalid configs - only catch specific expected errors
                        # yaml.YAMLError: invalid YAML syntax
                        # ValueError/TypeError: invalid config data
                        # KeyError/AttributeError: missing required fields
                        continue

    def get_config(self, name: str) -> ModelConfig:
        """Get a pre-configured model configuration.

        Args:
            name: Name of the configuration

        Returns:
            Model configuration

        Raises:
            KeyError: If configuration not found
        """
        if name not in self._configs:
            raise KeyError(f"Configuration '{name}' not found in zoo")
        return self._configs[name]

    def list_configs(self, category: str | None = None) -> list[str]:
        """List all available configurations.

        Args:
            category: Optional category filter (e.g., "vision", "protein")

        Returns:
            List of configuration names
        """
        if not category:
            return list(self._configs.keys())

        # Filter by category (based on tags in metadata)
        filtered = []
        for name, config in self._configs.items():
            # Check if config has tags in metadata
            tags = []
            if config.metadata and "tags" in config.metadata:
                tags = config.metadata["tags"]
            elif hasattr(config, "tags"):
                tags = config.tags

            if category in tags:
                filtered.append(name)

        return filtered

    def create_model(
        self, name: str, *, rngs: nnx.Rngs, modality: str | None = None, **overrides
    ) -> Any:
        """Create a model from zoo configuration.

        Args:
            name: Name of the configuration
            rngs: Random number generators
            modality: Optional modality adaptation
            **overrides: Configuration overrides

        Returns:
            Created model instance
        """
        # Get base configuration
        config = self.get_config(name)

        # Apply overrides if provided
        if overrides:
            config = self._apply_overrides(config, overrides)

        # Create model using factory
        return create_model(config, modality=modality, rngs=rngs)

    def _apply_overrides(self, config: ModelConfig, overrides: dict[str, Any]) -> ModelConfig:
        """Apply overrides to configuration.

        Args:
            config: Base configuration (frozen dataclass)
            overrides: Override values

        Returns:
            New configuration with overrides applied
        """
        # Convert dataclass to dict for manipulation
        config_dict = dataclasses.asdict(config)

        # Deep merge overrides
        for key, value in overrides.items():
            if key in config_dict:
                if isinstance(config_dict[key], dict) and isinstance(value, dict):
                    # Merge dictionaries
                    config_dict[key].update(value)
                else:
                    # Replace value
                    config_dict[key] = value
            else:
                # Add to metadata if not a top-level field
                if "metadata" not in config_dict:
                    config_dict["metadata"] = {}
                config_dict["metadata"][key] = value

        # Create new configuration (frozen dataclass requires new instance)
        return ModelConfig(**config_dict)

    def register_config(self, config: ModelConfig) -> None:
        """Register a configuration in the zoo.

        Args:
            config: Configuration to register
        """
        self._configs[config.name] = config

    def get_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about a configuration.

        Args:
            name: Name of the configuration

        Returns:
            Dictionary with configuration details
        """
        config = self.get_config(name)

        info = {
            "name": config.name,
            "description": config.description,
            "model_class": config.model_class,
            "version": config.version,
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "hidden_dims": config.hidden_dims,
            "metadata": config.metadata,
        }

        # Add tags if present
        if hasattr(config, "tags"):
            info["tags"] = config.tags
        elif config.metadata and "tags" in config.metadata:
            info["tags"] = config.metadata["tags"]
        else:
            info["tags"] = []

        return info


# Global zoo instance
zoo = ModelZoo()

# Export main functions
__all__ = ["ModelZoo", "zoo"]
