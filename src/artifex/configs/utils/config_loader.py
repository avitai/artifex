"""Utilities for loading and processing configuration files."""

from pathlib import Path
from typing import Any, TypeVar

import yaml

from artifex.configs.utils.error_handling import (
    ConfigNotFoundError,
    safe_load_config,
)
from artifex.generative_models.core.configuration import (
    DataConfig,
    DiffusionInferenceConfig,
    ExperimentTemplateConfig,
    InferenceConfig,
    ProteinDiffusionInferenceConfig,
    ProteinExtensionsConfig,
    TrainingConfig,
)
from artifex.generative_models.core.configuration.base_dataclass import ConfigDocument


# Define paths
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "defaults"
EXPERIMENT_CONFIG_DIR = Path(__file__).parent.parent / "experiments"
_DIFFUSION_INFERENCE_FIELDS = frozenset(
    {
        "sampler",
        "timesteps",
        "temperature",
        "sample_with_classifier_guidance",
        "guidance_scale",
        "save_intermediate_steps",
        "intermediate_step_interval",
        "seed",
    }
)
_PROTEIN_INFERENCE_FIELDS = frozenset(
    {
        "target_seq_length",
        "backbone_atom_indices",
        "calculate_metrics",
        "visualize_structures",
        "save_as_pdb",
    }
)
TConfigDocument = TypeVar("TConfigDocument", bound=ConfigDocument)


def get_config_path(config_name: str, config_type: str | None = None) -> Path:
    """Get the full path to a configuration file.

    Args:
        config_name: Name of the configuration file or relative path
        config_type: Optional type of configuration (models, data, etc.)

    Returns:
        Path to the configuration file

    Raises:
        ConfigNotFoundError: If the configuration file cannot be found
    """
    search_paths: list[str] = []

    if Path(config_name).is_absolute():
        path = Path(config_name)
        if path.exists():
            return path
        search_paths.append(str(path))
    else:
        if config_name.endswith(".yaml") or config_name.endswith(".yml"):
            if config_type:
                path = DEFAULT_CONFIG_DIR / config_type / config_name
                search_paths.append(str(path))
                if path.exists():
                    return path

            if config_type:
                path = DEFAULT_CONFIG_DIR / config_type / config_name
            else:
                path = DEFAULT_CONFIG_DIR / config_name
            search_paths.append(str(path))
            if path.exists():
                return path

            path = EXPERIMENT_CONFIG_DIR / config_name
            search_paths.append(str(path))
            if path.exists():
                return path

        if config_type:
            path = DEFAULT_CONFIG_DIR / config_type / f"{config_name}.yaml"
            search_paths.append(str(path))
            if path.exists():
                return path

        path = DEFAULT_CONFIG_DIR / f"{config_name}.yaml"
        search_paths.append(str(path))
        if path.exists():
            return path

        path = EXPERIMENT_CONFIG_DIR / f"{config_name}.yaml"
        search_paths.append(str(path))
        if path.exists():
            return path

    raise ConfigNotFoundError(config_name, search_paths)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file with error handling.

    Args:
        config_path: Path to the configuration file

    Returns:
        dictionary containing the configuration

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid
    """

    def _load_yaml(path: str | Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if payload is None:
            raise ValueError("configuration file is empty")
        if not isinstance(payload, dict):
            raise ValueError(
                f"configuration file must contain a YAML mapping, got {type(payload).__name__}"
            )
        return payload

    return safe_load_config(_load_yaml, config_path)


def _load_typed_config_document(
    config_path: str | Path,
    config_dict: dict[str, Any],
    config_class: type[TConfigDocument],
) -> TConfigDocument:
    """Materialize a typed config document behind the canonical loader boundary."""

    def _build_typed_config(path: str | Path) -> TConfigDocument:
        del path
        return config_class.from_dict(config_dict)

    return safe_load_config(_build_typed_config, config_path)


def get_data_config(config_name: str) -> DataConfig:
    """Get a data configuration with error handling.

    Args:
        config_name: Name of the data configuration file

    Returns:
        Data configuration object

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid
    """
    config_path = get_config_path(config_name, "data")
    config_dict = load_yaml_config(config_path)
    return _load_typed_config_document(config_path, config_dict, DataConfig)


def get_training_config(config_name: str) -> TrainingConfig:
    """Get a training configuration with error handling.

    Args:
        config_name: Name of the training configuration file

    Returns:
        Training configuration object

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid
    """
    config_path = get_config_path(config_name, "training")
    config_dict = load_yaml_config(config_path)
    return _load_typed_config_document(config_path, config_dict, TrainingConfig)


def get_inference_config(config_name: str) -> InferenceConfig:
    """Get an inference configuration with error handling.

    Args:
        config_name: Name of the inference configuration file

    Returns:
        Inference configuration object

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid
    """
    config_path = get_config_path(config_name, "inference")
    config_dict = load_yaml_config(config_path)
    config_class = _select_inference_config_class(config_dict)
    return _load_typed_config_document(config_path, config_dict, config_class)


def get_protein_extensions_config(config_name: str = "protein") -> ProteinExtensionsConfig:
    """Get a typed protein extension bundle from the shipped extension defaults.

    Args:
        config_name: Name of the protein extension bundle under ``defaults/extensions``.

    Returns:
        Typed protein extension bundle.

    Raises:
        ConfigError: If the configuration cannot be loaded or is invalid.
    """
    config_path = get_config_path(config_name, "extensions")
    config_dict = load_yaml_config(config_path)
    return _load_typed_config_document(config_path, config_dict, ProteinExtensionsConfig)


def load_experiment_config(experiment_name: str) -> ExperimentTemplateConfig:
    """Load a retained experiment template as a typed reference config.

    The shipped experiment YAMLs under ``src/artifex/configs/experiments`` are
    reference templates. They point at other config files and carry optional
    override mappings; they are not direct ``ExperimentConfig`` payloads.

    Args:
        experiment_name: Name of the experiment template

    Returns:
        Typed experiment template config
    """
    experiment_path = get_config_path(experiment_name, "experiments")
    experiment_dict = load_yaml_config(experiment_path)
    return _load_typed_config_document(
        experiment_path,
        experiment_dict,
        ExperimentTemplateConfig,
    )


def _select_inference_config_class(config_dict: dict[str, Any]) -> type[InferenceConfig]:
    """Choose the narrowest supported inference config class for a raw payload."""
    payload_keys = frozenset(config_dict)

    if payload_keys & _PROTEIN_INFERENCE_FIELDS:
        return ProteinDiffusionInferenceConfig
    if payload_keys & _DIFFUSION_INFERENCE_FIELDS:
        return DiffusionInferenceConfig
    return InferenceConfig
