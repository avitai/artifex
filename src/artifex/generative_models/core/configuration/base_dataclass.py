"""Typed frozen dataclass configuration foundations for Artifex.

This module defines the shared typed-document and runtime-config foundations
used across Artifex. It is built around frozen dataclasses, which are:
- JAX-native (no metaclasses, fully immutable)
- JIT-safe (frozen=True + tuples)
- Simpler (Python stdlib, no magic)
- Proven in production JAX codebases

Key Design Decisions:
1. All config documents are frozen dataclasses (immutable)
2. All sequence fields use tuples (not lists)
3. Validation happens in __post_init__ (fail-fast)
4. dacite handles dict → dataclass conversion with type checking
"""

import dataclasses
import enum
from pathlib import Path
from typing import Any, TypeVar

import yaml
from dacite import Config as DaciteConfig, from_dict as dacite_from_dict


def _path_type_hook(value: Any) -> Path:
    """Convert string to Path for dacite."""
    if isinstance(value, str):
        return Path(value)
    return value


TConfigDocument = TypeVar("TConfigDocument", bound="ConfigDocument")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ConfigDocument:
    """Serialization and YAML helpers for typed config documents.

    This base provides the typed-document mechanics shared by both named
    runtime configs and retained reference-template documents.
    """

    def __post_init__(self) -> None:
        """Support cooperative dataclass validation in subclasses."""

    @classmethod
    def from_dict(cls: type[TConfigDocument], config_dict: dict[str, Any]) -> TConfigDocument:
        """Create a typed config document from a dictionary using dacite."""
        return dacite_from_dict(
            data_class=cls,
            data=config_dict,
            config=DaciteConfig(
                strict=True,
                check_types=True,
                cast=[tuple, enum.Enum],
                type_hooks={Path: _path_type_hook},
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert this typed config document to a dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls: type[TConfigDocument], path: Path | str) -> TConfigDocument:
        """Load a typed config document from YAML."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_yaml(self, path: Path | str) -> None:
        """Save a typed config document to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        data = self._prepare_for_yaml(data)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _prepare_for_yaml(self, obj: Any) -> Any:
        """Prepare an object for YAML serialization."""
        if isinstance(obj, tuple):
            return [self._prepare_for_yaml(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._prepare_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_yaml(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BaseConfig(ConfigDocument):
    """Base configuration for named runtime configs.

    This provides:
    - Immutable configuration (frozen=True)
    - Common runtime metadata fields
    - Type-safe conversion inherited from ConfigDocument
    - Validation in __post_init__

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
