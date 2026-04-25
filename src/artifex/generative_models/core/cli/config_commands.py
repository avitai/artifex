"""Reusable CLI functions for configuration management."""

import dataclasses
import enum
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import yaml  # pyright: ignore[reportMissingModuleSource]
from dacite import DaciteError

from artifex.configs import (
    DataConfig,
    DistributedConfig,
    ExperimentConfig,
    ExperimentTemplateConfig,
    get_config_path,
    get_data_config,
    get_inference_config,
    get_protein_extensions_config,
    get_training_config,
    HyperparamSearchConfig,
    load_experiment_config,
    ProteinExtensionsConfig,
    template_manager,
    TrainingConfig,
)
from artifex.configs.utils.config_loader import load_yaml_config
from artifex.configs.utils.error_handling import (
    ConfigError,
    ConfigValidationError,
)
from artifex.generative_models.core.configuration.base_dataclass import ConfigDocument
from artifex.generative_models.core.configuration.management.versioning import (
    ConfigVersion,
    ConfigVersionRegistry,
)
from artifex.generative_models.core.configuration.model_creation import (
    materialize_model_creation_config,
)


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
_TRAINING_HINT_FIELDS = frozenset({"optimizer", "scheduler", "batch_size", "num_epochs"})
_DATA_HINT_FIELDS = frozenset({"dataset_name", "data_dir", "split"})
_DISTRIBUTED_HINT_FIELDS = frozenset(
    {
        "backend",
        "world_size",
        "rank",
        "local_rank",
        "num_nodes",
        "num_processes_per_node",
    }
)
_HYPERPARAM_HINT_FIELDS = frozenset({"search_type", "search_space", "num_trials"})
_PROTEIN_EXTENSION_HINT_FIELDS = frozenset({"bond_length", "bond_angle", "backbone", "mixin"})
_EXPERIMENT_TEMPLATE_HINT_FIELDS = frozenset(
    {"experiment_name", "model_config", "data_config", "training_config"}
)
_EXPERIMENT_CONFIG_HINT_FIELDS = frozenset({"model_cfg", "training_cfg", "data_cfg"})
_OUTPUT_FORMATS = frozenset({"yaml", "json"})


def parse_template_param(param: str) -> tuple[str, Any]:
    """Parse a flat key=value template parameter."""
    if "=" not in param:
        raise ValueError(f"Invalid parameter format: {param}. Expected key=value")

    key, raw_value = param.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Parameter key cannot be empty")
    if "." in key:
        raise ValueError("nested dotted keys are not supported; pass flat template params only")

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value

    return key, value


def _parse_template_params(params: list[str]) -> dict[str, Any]:
    """Parse repeated template params into a typed mapping."""
    parsed_params: dict[str, Any] = {}
    for param in params:
        key, value = parse_template_param(param)
        parsed_params[key] = value
    return parsed_params


def _normalize_for_output(value: Any) -> Any:
    """Normalize config values for YAML/JSON output."""
    if isinstance(value, ConfigDocument):
        return _normalize_for_output(value.to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize_for_output(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: _normalize_for_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_for_output(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_for_output(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    return value


def _serialize_config_data(config_data: Any, output_format: str) -> str:
    """Serialize typed config data to YAML or JSON."""
    if output_format not in _OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")

    normalized = _normalize_for_output(config_data)
    if output_format == "yaml":
        return yaml.safe_dump(normalized, default_flow_style=False, sort_keys=False)
    return json.dumps(normalized, indent=2)


def _write_serialized_config(config_data: Any, output_path: str | Path, output_format: str) -> None:
    """Write serialized config data to disk."""
    output = _serialize_config_data(config_data, output_format)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")


def _typed_document_summary(document: ConfigDocument) -> list[str]:
    """Build a stable human-readable summary for a typed config document."""
    summary_lines = [f"Typed document: {type(document).__name__}", "", "Configuration summary:"]
    for field in dataclasses.fields(document):
        value = getattr(document, field.name)
        if field.name == "overrides":
            populated_sections = [
                section_name
                for section_name, section_value in dataclasses.asdict(value).items()
                if section_value
            ]
            summary = populated_sections or ["none"]
            summary_lines.append(f"  overrides: {', '.join(summary)}")
            continue
        summary_lines.append(f"  {field.name}: {type(value).__name__}")
    return summary_lines


def _raise_validation_error(config_ref: str | Path, error: Exception) -> NoReturn:
    """Raise the canonical typed-config validation error."""
    raise ConfigValidationError(config_ref, error) from error


def _load_named_typed_document(
    config_ref: str | Path,
    config_type: str,
    config_class: type[ConfigDocument],
) -> ConfigDocument:
    """Load a typed document from the named config tree."""
    config_path = get_config_path(str(config_ref), config_type)
    config_dict = load_yaml_config(config_path)
    try:
        return config_class.from_dict(config_dict)
    except (DaciteError, TypeError, ValueError) as error:
        _raise_validation_error(config_path, error)


def _load_supported_typed_document(config_ref: str | Path) -> ConfigDocument:
    """Load only the retained typed config document surface."""
    config_ref_str = str(config_ref)

    for loader in (
        load_experiment_config,
        get_training_config,
        get_data_config,
        get_inference_config,
        get_protein_extensions_config,
    ):
        try:
            return loader(config_ref_str)
        except ConfigError:
            continue

    for config_type, config_class in (
        ("distributed", DistributedConfig),
        ("hyperparam", HyperparamSearchConfig),
    ):
        try:
            return _load_named_typed_document(config_ref_str, config_type, config_class)
        except ConfigError:
            continue

    raw_config = load_yaml_config(config_ref_str)
    raw_keys = frozenset(raw_config)

    try:
        if raw_keys & _EXPERIMENT_TEMPLATE_HINT_FIELDS:
            return ExperimentTemplateConfig.from_dict(raw_config)
        if raw_keys & _EXPERIMENT_CONFIG_HINT_FIELDS:
            return ExperimentConfig.from_dict(raw_config)
        if raw_keys & _TRAINING_HINT_FIELDS:
            return TrainingConfig.from_dict(raw_config)
        if raw_keys & _DATA_HINT_FIELDS:
            return DataConfig.from_dict(raw_config)
        if raw_keys & (
            _DIFFUSION_INFERENCE_FIELDS | _PROTEIN_INFERENCE_FIELDS | {"checkpoint_path"}
        ):
            return get_inference_config(config_ref_str)
        if raw_keys & _DISTRIBUTED_HINT_FIELDS:
            return DistributedConfig.from_dict(raw_config)
        if raw_keys & _HYPERPARAM_HINT_FIELDS:
            return HyperparamSearchConfig.from_dict(raw_config)
        if raw_keys & _PROTEIN_EXTENSION_HINT_FIELDS:
            return ProteinExtensionsConfig.from_dict(raw_config)
        return materialize_model_creation_config(raw_config)
    except ConfigError:
        raise
    except (DaciteError, TypeError, ValueError) as error:
        if raw_keys & (
            _EXPERIMENT_TEMPLATE_HINT_FIELDS
            | _EXPERIMENT_CONFIG_HINT_FIELDS
            | _TRAINING_HINT_FIELDS
            | _DATA_HINT_FIELDS
            | _DIFFUSION_INFERENCE_FIELDS
            | _PROTEIN_INFERENCE_FIELDS
            | _DISTRIBUTED_HINT_FIELDS
            | _HYPERPARAM_HINT_FIELDS
            | _PROTEIN_EXTENSION_HINT_FIELDS
            | {"checkpoint_path"}
        ):
            _raise_validation_error(config_ref_str, error)

    _raise_validation_error(
        config_ref_str,
        ValueError("expected a supported typed config document, not an arbitrary YAML mapping"),
    )


def create_config(args) -> int:
    """Create a new configuration from a template."""
    try:
        template_params = _parse_template_params(getattr(args, "param", []))
        template = template_manager.generate_config(args.template, **template_params)
        _write_serialized_config(template, args.output, args.format)
        print(f"Created configuration: {args.output}")
        return 0
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Error creating or writing to output file: {e}", file=sys.stderr)
        return 1
    except (ValueError, TypeError, yaml.YAMLError) as e:
        print(f"Error creating configuration data: {e}", file=sys.stderr)
        return 1


def validate_config_file(args) -> int:
    """Validate a configuration file."""
    try:
        document = _load_supported_typed_document(args.config_file)
        print(f"Configuration is valid: {args.config_file}")
        for line in _typed_document_summary(document):
            print(line)
        return 0
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1


def show_config(args) -> int:
    """Show a configuration in a readable format."""
    try:
        document = _load_supported_typed_document(args.config_file)
        output = _serialize_config_data(document, args.format)
        print(output)
        return 0
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1


def diff_config(args) -> int:
    """Show differences between two configurations."""
    try:
        # Load configurations
        config1 = load_yaml_config(args.config1)
        config2 = load_yaml_config(args.config2)

        # Find differences
        diff = _dict_diff(config1, config2)

        if not diff:
            print("Configurations are identical")
            return 0

        # Print differences
        print("Differences found:")
        for path, (val1, val2) in diff.items():
            print(f"  {path}:")
            print(f"    - {val1}")
            print(f"    + {val2}")

        return 0
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1


def _dict_diff(d1: dict, d2: dict, path: str = "") -> dict[str, tuple]:
    """Find differences between two dictionaries."""
    diff = {}

    # Keys in d1 but not in d2
    for k in set(d1.keys()) - set(d2.keys()):
        new_path = f"{path}.{k}" if path else k
        diff[new_path] = (d1[k], None)

    # Keys in d2 but not in d1
    for k in set(d2.keys()) - set(d1.keys()):
        new_path = f"{path}.{k}" if path else k
        diff[new_path] = (None, d2[k])

    # Keys in both but different values
    for k in set(d1.keys()) & set(d2.keys()):
        new_path = f"{path}.{k}" if path else k

        if isinstance(d1[k], dict) and isinstance(d2[k], dict):
            # Recursively diff nested dictionaries
            nested_diff = _dict_diff(d1[k], d2[k], new_path)
            diff.update(nested_diff)
        elif d1[k] != d2[k]:
            diff[new_path] = (d1[k], d2[k])

    return diff


def version_config(args) -> int:
    """Version a configuration file."""
    try:
        # Load configuration
        config = load_yaml_config(args.config_file)

        # Create registry
        registry = ConfigVersionRegistry(args.registry)

        # Register configuration
        version = registry.register(config, description=args.description)

        print(f"Registered configuration version: {version.version}")
        print(f"  Hash: {version.hash}")
        print(f"  Timestamp: {version.timestamp.isoformat()}")
        if args.description:
            print(f"  Description: {args.description}")

        return 0
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Error accessing configuration file or registry: {e}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError) as e:
        print(f"Error versioning configuration: {e}", file=sys.stderr)
        return 1


def list_configs(args) -> int:
    """List versioned configurations."""
    try:
        # Create registry
        registry = get_config_registry(args.registry)

        # List versions
        versions = registry.list_versions()

        if not versions:
            print("No versioned configurations found")
            return 0

        # Apply limit
        if args.limit > 0:
            versions = versions[: args.limit]

        print(f"Found {len(versions)} versioned configurations:")
        for version in versions:
            # Handle both ConfigVersion objects and dictionaries
            if isinstance(version, ConfigVersion):
                version_id = version.version
                timestamp = version.timestamp.isoformat()
                description = version.description
            else:
                # dictionary format
                version_id = version.get("version_id") or version.get("version")
                timestamp = version.get("datetime") or version.get("timestamp")
                description = version.get("description")

            print(f"  {version_id} - {timestamp}")
            if description:
                print(f"    {description}")

        return 0
    except OSError as e:
        print(f"Error accessing registry directory: {e}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as e:
        print(f"Error listing configuration versions: {e}", file=sys.stderr)
        return 1


def get_config(args) -> int:
    """Get a specific versioned configuration."""
    try:
        # Create registry
        registry = get_config_registry(args.registry)

        # Get version_or_hash from args (handle different attribute names)
        version_or_hash = getattr(args, "version_or_hash", getattr(args, "version", None))
        if version_or_hash is None:
            print("Error: version_or_hash or version argument required", file=sys.stderr)
            return 1

        # Try to find by version first, then by hash
        try:
            config_version = registry.get_by_version(version_or_hash)
        except ValueError:
            try:
                config_version = registry.get_by_hash(version_or_hash)
            except ValueError:
                print(f"No configuration found with version or hash: {version_or_hash}")
                return 1

        # Extract config data
        if isinstance(config_version, ConfigVersion):
            config_data = config_version.config
        else:
            config_data = config_version.get("config", config_version)

        output = _serialize_config_data(config_data, args.format)

        # Write to output file or stdout
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Wrote configuration to {args.output}")
        else:
            print(output)

        return 0
    except OSError as e:
        print(f"Error accessing registry or output file: {e}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError) as e:
        print(f"Error retrieving configuration version: {e}", file=sys.stderr)
        return 1


def get_config_registry(registry_path="./config_registry"):
    """Get configuration registry (alias for ConfigVersionRegistry)."""
    return ConfigVersionRegistry(registry_path)
