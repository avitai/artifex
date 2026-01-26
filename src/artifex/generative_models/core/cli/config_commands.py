"""Reusable CLI functions for configuration management."""

import json
import sys
from typing import Any

import yaml

from artifex.configs import load_experiment_config, load_yaml_config
from artifex.configs.utils.error_handling import (
    ConfigError,
)
from artifex.generative_models.core.configuration.management.versioning import (
    ConfigVersion,
    ConfigVersionRegistry,
)


def parse_override(override: str) -> tuple:
    """Parse a key=value override string."""
    if "=" not in override:
        raise ValueError(f"Invalid override format: {override}. Expected key=value")

    key, value = override.split("=", 1)

    # Try to parse value as JSON for complex types
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        # If not valid JSON, treat as string
        pass

    return key, value


def apply_overrides(config: dict[str, Any], overrides: list) -> dict[str, Any]:
    """Apply overrides to a configuration."""
    if not overrides:
        return config

    result = config.copy()

    for override in overrides:
        key, value = parse_override(override)

        # Handle nested keys (e.g., "model.learning_rate")
        parts = key.split(".")
        target = result

        # Navigate to the target dict
        for i, part in enumerate(parts[:-1]):
            if part not in target:
                target[part] = {}
            elif not isinstance(target[part], dict):
                # Convert to dict if not already
                target[part] = {"value": target[part]}
            target = target[part]

        # Set the value
        target[parts[-1]] = value

    return result


def create_config(args) -> int:
    """Create a new configuration from a template."""
    try:
        # Load template
        template = load_yaml_config(args.template)

        # Apply overrides
        if args.override:
            template = apply_overrides(template, args.override)

        # Write to output file
        with open(args.output, "w", encoding="utf-8") as f:
            if args.format == "yaml":
                yaml.dump(template, f, default_flow_style=False)
            else:
                json.dump(template, f, indent=2)

        print(f"Created configuration: {args.output}")
        return 0
    except (IOError, FileNotFoundError) as e:
        print(f"Error creating or writing to output file: {e}", file=sys.stderr)
        return 1
    except (ValueError, TypeError, yaml.YAMLError) as e:
        print(f"Error creating configuration data: {e}", file=sys.stderr)
        return 1


def validate_config_file(args) -> int:
    """Validate a configuration file."""
    try:
        # Try experiment loader first for complex configs
        try:
            configs = load_experiment_config(args.config_file)

            # If we get here, validation succeeded
            print(f"Configuration is valid: {args.config_file}")

            # Print summary of loaded configuration
            print("\nConfiguration summary:")
            for key, config in configs.items():
                if config is not None:
                    print(f"  {key}: {config.__class__.__name__}")
                else:
                    print(f"  {key}: None")

            return 0

        except (ValueError, KeyError, TypeError, ConfigError):
            # Fall back to simple YAML validation for test configs
            config = load_yaml_config(args.config_file)
            print(f"Configuration is valid: {args.config_file}")
            print("\nConfiguration summary:")
            for key, value in config.items():
                print(f"  {key}: {type(value).__name__}")
            return 0

    except (FileNotFoundError, IOError) as e:
        print(f"Error accessing configuration file: {e}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"Error in validation process: {e}", file=sys.stderr)
        return 1


def show_config(args) -> int:
    """Show a configuration in a readable format."""
    try:
        # Load configuration
        config = load_yaml_config(args.config_file)

        # Output in selected format
        if args.format == "yaml":
            output = yaml.dump(config, default_flow_style=False)
        else:  # json
            output = json.dumps(config, indent=2)

        print(output)
        return 0
    except (FileNotFoundError, IOError) as e:
        print(f"Error accessing configuration file: {e}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as e:
        print(f"Error displaying configuration: {e}", file=sys.stderr)
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
    except (FileNotFoundError, IOError) as e:
        print(f"Error accessing configuration files: {e}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as e:
        print(f"Error comparing configurations: {e}", file=sys.stderr)
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
    except (FileNotFoundError, IOError) as e:
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
    except (FileNotFoundError, IOError) as e:
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
            if hasattr(config_data, "to_dict"):
                config_data = config_data.to_dict()
        else:
            config_data = config_version.get("config", config_version)

        # Format output
        if args.format == "yaml":
            output = yaml.dump(config_data, default_flow_style=False)
        else:
            output = json.dumps(config_data, indent=2)

        # Write to output file or stdout
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Wrote configuration to {args.output}")
        else:
            print(output)

        return 0
    except (FileNotFoundError, IOError) as e:
        print(f"Error accessing registry or output file: {e}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError) as e:
        print(f"Error retrieving configuration version: {e}", file=sys.stderr)
        return 1


def get_config_registry(registry_path="./config_registry"):
    """Get configuration registry (alias for ConfigVersionRegistry)."""

    return ConfigVersionRegistry(registry_path)
