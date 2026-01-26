"""Base frozen dataclass configuration for Artifex.

This module replaces the Pydantic-based configuration system with
frozen dataclasses, which are:
- JAX-native (no metaclasses, fully immutable)
- JIT-safe (frozen=True + tuples)
- Simpler (Python stdlib, no magic)
- Proven in production JAX codebases

Key Design Decisions:
1. All configs are frozen dataclasses (immutable)
2. All sequence fields use tuples (not lists)
3. Validation happens in __post_init__ (fail-fast)
4. dacite handles dict → dataclass conversion with type checking
"""

import dataclasses
from pathlib import Path
from typing import Any

import yaml
from dacite import Config as DaciteConfig, from_dict as dacite_from_dict


def _path_type_hook(value: Any) -> Path:
    """Convert string to Path for dacite."""
    if isinstance(value, str):
        return Path(value)
    return value


@dataclasses.dataclass(frozen=True)
class BaseConfig:
    """Base configuration for all configs.

    Replaces Pydantic BaseConfiguration with frozen dataclass.

    This provides:
    - Immutable configuration (frozen=True)
    - Type-safe with dataclasses
    - Automatic dict conversion with dacite
    - YAML serialization/deserialization
    - Validation in __post_init__

    All configs in Artifex inherit from this class.

    Attributes:
        name: Unique name for this configuration
        description: Human-readable description
        tags: Tuple of tags for categorization (immutable!)
        metadata: Non-functional metadata for experiment tracking
    """

    # Core fields - only 'name' is required
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()  # Tuple, not list! (immutable)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate base configuration.

        This runs automatically after __init__.
        Raises ValueError if validation fails (fail-fast).
        """
        # Validate name is non-empty
        name = self.name.strip() if isinstance(self.name, str) else self.name
        if not name:
            raise ValueError("name must be non-empty")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BaseConfig":
        """Create config from dict using dacite.

        This handles automatic type conversion, including:
        - list → tuple conversion
        - Nested dataclass creation
        - Type checking

        Args:
            config_dict: Dictionary with config data

        Returns:
            Instance of this config class

        Raises:
            dacite exceptions if data is invalid
        """
        return dacite_from_dict(
            data_class=cls,
            data=config_dict,
            config=DaciteConfig(
                strict=True,  # No extra fields allowed
                check_types=True,  # Type checking enabled
                cast=[tuple],  # Auto-cast lists to tuples
                type_hooks={Path: _path_type_hook},  # Auto-convert strings to Path
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "BaseConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Instance of this config class

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file.

        Creates parent directories if needed.
        Converts tuples to lists for YAML compatibility.

        Args:
            path: Path to save YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and prepare for YAML
        data = self.to_dict()
        data = self._prepare_for_yaml(data)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _prepare_for_yaml(self, obj: Any) -> Any:
        """Prepare object for YAML serialization.

        YAML doesn't have tuples, so convert them to lists.
        Handle other special types as needed.

        Args:
            obj: Object to prepare

        Returns:
            YAML-safe object
        """
        if isinstance(obj, tuple):
            # Convert tuple to list for YAML
            return [self._prepare_for_yaml(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._prepare_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_yaml(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
