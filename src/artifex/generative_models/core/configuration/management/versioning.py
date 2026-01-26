"""Configuration versioning and management for artifex.generative_models.core."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from artifex.generative_models.core.configuration import BaseConfig


# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    def default(self, o: Any) -> Any:
        """Convert datetime objects to ISO format strings.

        Args:
            o: Object to convert

        Returns:
            ISO format string for datetime objects, or default conversion for other types
        """
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def compute_config_hash(config: dict[str, Any] | BaseConfig) -> str:
    """
    Compute a deterministic hash for a configuration.

    Args:
        config: Configuration dictionary or object

    Returns:
        Hex digest of the configuration hash
    """
    if isinstance(config, BaseConfig):
        config_dict = config.to_dict()
    else:
        config_dict = config

    # Remove non-deterministic fields like timestamps
    clean_dict = _clean_config_for_hash(config_dict)

    # Convert to a deterministic string representation
    config_str = json.dumps(clean_dict, sort_keys=True)

    # Compute hash
    hash_obj = hashlib.sha256(config_str.encode())
    # Return first 8 characters for tests to pass
    return hash_obj.hexdigest()[:8]


def _clean_config_for_hash(config: dict[str, Any]) -> dict[str, Any]:
    """
    Remove non-deterministic fields from configuration for hashing.

    Args:
        config: Configuration dictionary

    Returns:
        Clean configuration dictionary
    """
    # Create a copy to avoid modifying the original
    config = config.copy()

    # list of keys to remove (non-deterministic or irrelevant for
    # reproducibility)
    keys_to_remove = [
        "timestamp",
        "run_id",
        "version_id",
        "config_hash",
        "output_dir",
        "checkpoint_path",
        "log_dir",
        "created_at",
        "path",
    ]

    # Remove keys at the top level
    for key in keys_to_remove:
        if key in config:
            del config[key]

    # Recursively clean nested dictionaries
    for key, value in list(config.items()):
        if isinstance(value, dict):
            config[key] = _clean_config_for_hash(value)

    return config


class ConfigVersion:
    """Configuration version information."""

    def __init__(
        self,
        config: dict[str, Any] | BaseConfig,
        description: str | None = None,
        version: str | None = None,
        timestamp: datetime | None = None,
    ):
        """Initialize with configuration and optional description.

        Args:
            config: Configuration dictionary or object
            description: Optional description of this configuration version
            version: Optional version string (generated if not provided)
            timestamp: Optional timestamp (current time if not provided)
        """
        self.config = config
        self.description = description or ""
        self.timestamp = timestamp or datetime.now()
        self.config_hash = compute_config_hash(config)

        # Generate version ID: YYYYMMDD-hash
        if version:
            self.version = version
        else:
            date_format = "%Y%m%d"
            date_str = self.timestamp.strftime(date_format)
            self.version = f"{date_str}-{self.config_hash}"

    def __str__(self):
        """Return version string."""
        return self.version

    @property
    def hash(self) -> str:
        """Alias for config_hash to make tests pass."""
        return self.config_hash

    @property
    def version_id(self) -> str:
        """Alias for version to make tests pass."""
        return self.version

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        config_dict = self.config.to_dict() if isinstance(self.config, BaseConfig) else self.config

        return {
            "version": self.version,
            # Add version_id for test compatibility
            "version_id": self.version,
            # Convert to float timestamp
            "timestamp": self.timestamp.timestamp(),
            # for JSON
            "datetime": self.timestamp.isoformat(),
            # Use 'hash' key for tests
            "hash": self.config_hash,
            "description": self.description,
            "config": config_dict,
        }

    def save(self, config_dir: str | Path) -> Path:
        """
        Save versioned configuration to file.

        Args:
            config_dir: Directory to save configuration

        Returns:
            Path to saved configuration file
        """
        config_dir = Path(config_dir)
        versions_dir = config_dir / "versions"
        os.makedirs(versions_dir, exist_ok=True)

        # Create filename with version_id
        filename = f"{self.version_id}.json"
        file_path = versions_dir / filename

        # Save as YAML
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, cls=DateTimeEncoder)

        return file_path


class ConfigVersionRegistry:
    """Registry for tracking configuration versions."""

    def __init__(self, registry_dir: str | Path):
        """
        Initialize a configuration version registry.

        Args:
            registry_dir: Directory to store versioned configurations
        """
        self.registry_dir = Path(registry_dir)
        os.makedirs(self.registry_dir, exist_ok=True)

        # Create versions directory required by tests
        versions_dir = self.registry_dir / "versions"
        os.makedirs(versions_dir, exist_ok=True)

        # Index file
        self.index_file = self.registry_dir / "index.json"

        # Create empty index file if it doesn't exist
        if not self.index_file.exists():
            with open(self.index_file, "w", encoding="utf-8") as f:
                f.write("{}")

        # Initialize or load index
        self.index = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load configuration version index."""
        if not self.index_file.exists():
            return {}

        with open(self.index_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f) or {}
            except json.JSONDecodeError:
                return {}

    def _save_index(self) -> None:
        """Save configuration version index."""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2, cls=DateTimeEncoder)

    def register(
        self,
        config: dict[str, Any] | BaseConfig,
        description: str | None = None,
    ) -> ConfigVersion:
        """
        Register a configuration version.

        Args:
            config: Configuration dictionary or object
            description: Optional description of this configuration version

        Returns:
            Versioned configuration
        """
        # Create versioned configuration
        config_version = ConfigVersion(config, description=description)

        # Add to index
        self.index[config_version.version] = {
            "hash": config_version.config_hash,
            "timestamp": config_version.timestamp,
            "datetime": config_version.timestamp.isoformat(),
            "description": config_version.description,
        }

        # Save config file
        config_version.save(self.registry_dir)

        # Update index
        self._save_index()

        return config_version

    def get_by_version(self, version: str) -> ConfigVersion:
        """
        Get a configuration by version.

        Args:
            version: Version string

        Returns:
            ConfigVersion object

        Raises:
            ValueError: If version not found
        """
        if version not in self.index:
            raise ValueError(f"Version {version} not found in registry")

        # Get data from index
        info = self.index[version]

        # Load from file
        file_path = self.registry_dir / "versions" / f"{version}.json"
        if not file_path.exists():
            raise ValueError(f"Config file for version {version} not found")

        with open(file_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # Create ConfigVersion object
        return ConfigVersion(
            config=config_data["config"],
            version=version,
            timestamp=info["timestamp"],
            description=info["description"],
        )

    def get_by_hash(self, config_hash: str) -> ConfigVersion:
        """
        Get a configuration by hash.

        Args:
            config_hash: Configuration hash (can be a prefix)

        Returns:
            ConfigVersion object

        Raises:
            ValueError: If no configuration with matching hash is found
        """
        # Reload the index to catch any external changes (needed for tests)
        self.index = self._load_index()

        # Find versions with matching hash
        matching_versions: list[str] = []
        for version, info in self.index.items():
            if info["hash"].startswith(config_hash):
                matching_versions.append(version)

        if not matching_versions:
            raise ValueError(f"No configuration found with hash prefix {config_hash}")

        # Check for ambiguous hash prefixes
        if len(matching_versions) > 1:
            matching_hashes = [self.index[v]["hash"] for v in matching_versions]
            raise ValueError(
                f"ambiguous hash prefix '{config_hash}'. "
                f"Matches multiple versions: {matching_hashes}"
            )

        # Get the matching version
        return self.get_by_version(matching_versions[0])

    def list_versions(self) -> list[ConfigVersion]:
        """
        List all tracked configuration versions.

        Returns:
            list of ConfigVersion objects
        """
        result: list[ConfigVersion] = []
        for version in self.index.keys():
            try:
                config_version = self.get_by_version(version)
                result.append(config_version)
            except (ValueError, FileNotFoundError):
                # Skip versions with missing files
                pass

        # Sort by timestamp (newest first)
        result.sort(key=lambda x: x.timestamp, reverse=True)

        return result
