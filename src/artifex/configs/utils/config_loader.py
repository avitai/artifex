"""Utilities for loading and processing configuration files."""

import os
from pathlib import Path
from typing import Any, Type

import yaml
from pydantic import ValidationError

from artifex.configs.schema import (
    BaseConfig,
    DataConfig,
    ExperimentConfig,
    InferenceConfig,
    TrainingConfig,
)
from artifex.configs.utils.error_handling import (
    ConfigNotFoundError,
    ConfigValidationError,
    safe_load_config,
)


# Define paths
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "defaults"
EXPERIMENT_CONFIG_DIR = Path(__file__).parent.parent / "experiments"


def get_config_path(config_name: str, config_type: str | None = None) -> Path:
    """
    Get the full path to a configuration file.

    Args:
        config_name: Name of the configuration file or relative path
        config_type: Optional type of configuration (models, data, etc.)

    Returns:
        Path to the configuration file

    Raises:
        ConfigNotFoundError: If the configuration file cannot be found
    """
    search_paths = []

    if os.path.isabs(config_name):
        path = Path(config_name)
        if path.exists():
            return path
        search_paths.append(str(path))
    else:
        # Try with file extension
        if config_name.endswith(".yaml") or config_name.endswith(".yml"):
            if config_type:
                # Look in the specific type directory
                path = DEFAULT_CONFIG_DIR / config_type / config_name
                search_paths.append(str(path))
                if path.exists():
                    return path

            # Look in defaults directory
            if config_type:
                path = DEFAULT_CONFIG_DIR / config_type / config_name
            else:
                path = DEFAULT_CONFIG_DIR / config_name
            search_paths.append(str(path))
            if path.exists():
                return path

            # Try as path relative to experiments directory
            path = EXPERIMENT_CONFIG_DIR / config_name
            search_paths.append(str(path))
            if path.exists():
                return path

        # Try without file extension
        if config_type:
            path = DEFAULT_CONFIG_DIR / config_type / f"{config_name}.yaml"
            search_paths.append(str(path))
            if path.exists():
                return path

        # Look in defaults directory
        path = DEFAULT_CONFIG_DIR / f"{config_name}.yaml"
        search_paths.append(str(path))
        if path.exists():
            return path

        # Try as path relative to experiments directory
        path = EXPERIMENT_CONFIG_DIR / f"{config_name}.yaml"
        search_paths.append(str(path))
        if path.exists():
            return path

    raise ConfigNotFoundError(config_name, search_paths)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file with error handling.

    Args:
        config_path: Path to the configuration file

    Returns:
        dictionary containing the configuration

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid
    """

    def _load_yaml(path: str | Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    return safe_load_config(_load_yaml, config_path)


def create_config_from_yaml(config_path: str | Path, config_class: Type[BaseConfig]) -> BaseConfig:
    """
    Create a configuration object from a YAML file with error handling.

    Args:
        config_path: Path to the configuration file
        config_class: Configuration class to instantiate

    Returns:
        Configuration object

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid
    """
    try:
        config_dict = load_yaml_config(config_path)
        return config_class(**config_dict)
    except ValidationError as e:
        # We don't use config_type here, but keep for clarity
        raise ConfigValidationError(config_path, e) from e


def get_model_config(config_name: str, model_type: str = "diffusion") -> dict[str, Any]:
    """
    Get a model configuration with error handling.

    Note: Model-specific config classes have been moved to the unified configuration system.
    This function now returns a dictionary that can be used to create a ModelConfiguration.

    Args:
        config_name: Name of the model configuration file
        model_type: Type of model (diffusion, autoregressive, etc.)

    Returns:
        Model configuration dictionary

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid
    """
    try:
        config_path = get_config_path(config_name, f"models/{model_type}")
        config_dict = load_yaml_config(config_path)

        # Return the raw config dict
        # Users should create ModelConfiguration from artifex.generative_models.core.configuration
        return config_dict
    except Exception as e:
        # Use the config name as the path if actual path isn't available
        path = config_name
        try:
            path = get_config_path(config_name, f"models/{model_type}")
        except Exception:
            pass
        raise ConfigValidationError(path, e) from e


def get_data_config(config_name: str) -> DataConfig:
    """
    Get a data configuration with error handling.

    Args:
        config_name: Name of the data configuration file

    Returns:
        Data configuration object

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid
    """
    try:
        config_path = get_config_path(config_name, "data")
        config_dict = load_yaml_config(config_path)
        return DataConfig(**config_dict)
    except ValidationError as e:
        # Use the config name as the path if actual path isn't available
        path = config_name
        try:
            path = get_config_path(config_name, "data")
        except Exception:
            pass
        raise ConfigValidationError(path, e) from e


def get_training_config(config_name: str) -> TrainingConfig:
    """
    Get a training configuration with error handling.

    Args:
        config_name: Name of the training configuration file

    Returns:
        Training configuration object

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid
    """
    try:
        config_path = get_config_path(config_name, "training")
        config_dict = load_yaml_config(config_path)
        return TrainingConfig(**config_dict)
    except ValidationError as e:
        # Use the config name as the path if actual path isn't available
        path = config_name
        try:
            path = get_config_path(config_name, "training")
        except Exception:
            pass
        raise ConfigValidationError(path, e) from e


def get_inference_config(config_name: str) -> InferenceConfig:
    """
    Get an inference configuration with error handling.

    Args:
        config_name: Name of the inference configuration file

    Returns:
        Inference configuration object

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid
    """
    try:
        config_path = get_config_path(config_name, "inference")
        config_dict = load_yaml_config(config_path)
        return InferenceConfig(**config_dict)
    except ValidationError as e:
        # Use the config name as the path if actual path isn't available
        path = config_name
        try:
            path = get_config_path(config_name, "inference")
        except Exception:
            pass
        raise ConfigValidationError(path, e) from e


def load_experiment_config(experiment_name: str, base_path: str | None = None) -> dict[str, Any]:
    """Load experiment configuration from a file.

    This function loads an experiment configuration from a YAML file based on the
    experiment name and validates it against the expected schema.

    Args:
        experiment_name: Name of the experiment
        base_path: Base path to look for experiment configs

    Returns:
        The loaded and validated experiment configuration

    Raises:
        ConfigNotFoundError: If the experiment config file is not found
        ConfigLoadError: If there's an error loading the config
        ConfigValidationError: If the config fails validation
    """
    # Load the experiment configuration file
    try:
        experiment_path = get_config_path(experiment_name, "experiments")
        experiment_dict = load_yaml_config(experiment_path)
        experiment_config = ExperimentConfig(**experiment_dict)

        # Load individual configurations with descriptive error context
        configs = {}

        # Load model config
        try:
            # Attempt to load the config file
            config_path = get_config_path(experiment_config.model_config_ref, "models")
            _ = load_yaml_config(config_path)  # Validate config file exists
            configs["model"] = get_model_config(experiment_config.model_config_ref)
        except Exception as e:
            ref_context = f" (referenced from experiment '{experiment_name}')"
            raise type(e)(str(e) + ref_context) from e

        # Load data config
        try:
            # Attempt to load the config file
            config_path = get_config_path(experiment_config.data_config, "data")
            _ = load_yaml_config(config_path)  # Validate config file exists
            configs["data"] = get_data_config(experiment_config.data_config)
        except Exception as e:
            ref_context = f" (referenced from experiment '{experiment_name}')"
            raise type(e)(str(e) + ref_context) from e

        # Load training config
        try:
            # Attempt to load the config file
            config_path = get_config_path(experiment_config.training_config, "training")
            _ = load_yaml_config(config_path)  # Validate config file exists
            configs["training"] = get_training_config(experiment_config.training_config)
        except Exception as e:
            ref_context = f" (referenced from experiment '{experiment_name}')"
            raise type(e)(str(e) + ref_context) from e

        # Load inference config if specified
        configs["inference"] = None
        if experiment_config.inference_config:
            try:
                # Attempt to load the config file
                config_path = get_config_path(experiment_config.inference_config, "inference")
                _ = load_yaml_config(config_path)  # Validate config file exists
                configs["inference"] = get_inference_config(experiment_config.inference_config)
            except Exception as e:
                ref_context = f" (referenced from experiment '{experiment_name}')"
                raise type(e)(str(e) + ref_context) from e

        # Apply overrides from experiment configuration
        overrides = experiment_dict.get("overrides", {})

        if "model" in overrides and configs["model"]:
            configs["model"].update(overrides["model"])

        if "data" in overrides and configs["data"]:
            configs["data"].update(overrides["data"])

        if "training" in overrides and configs["training"]:
            configs["training"].update(overrides["training"])

        if configs["inference"] and "inference" in overrides:
            configs["inference"].update(overrides["inference"])

        # Add experiment config to the returned dictionary
        configs["experiment"] = experiment_config

        return configs
    except ValidationError as e:
        # Use the config name as the path if actual path isn't available
        path = experiment_name
        try:
            path = get_config_path(experiment_name, "experiments")
        except Exception:
            pass
        raise ConfigValidationError(path, e) from e
