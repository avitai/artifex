from __future__ import annotations

import json
from datetime import datetime

import pytest

from artifex.generative_models.core.configuration.management.versioning import (
    compute_config_hash,
    ConfigVersion,
    ConfigVersionRegistry,
    DateTimeEncoder,
)


def test_compute_config_hash_ignores_nondeterministic_fields_without_mutating_input() -> None:
    """Configuration hashes should ignore runtime-only fields and preserve caller data."""
    config = {
        "name": "model",
        "timestamp": "first",
        "output_dir": "/tmp/a",
        "nested": {"run_id": "abc", "value": 3},
    }
    equivalent = {
        "name": "model",
        "timestamp": "second",
        "output_dir": "/tmp/b",
        "nested": {"run_id": "def", "value": 3},
    }

    assert compute_config_hash(config) == compute_config_hash(equivalent)
    assert config["timestamp"] == "first"
    assert config["nested"]["run_id"] == "abc"


def test_config_version_generates_stable_metadata_and_saves_json(tmp_path) -> None:
    """ConfigVersion should expose aliases, serializable timestamps, and saved payloads."""
    timestamp = datetime(2026, 4, 26, 9, 30, 0)
    version = ConfigVersion(
        {"name": "vae", "latent_dim": 8},
        description="baseline",
        timestamp=timestamp,
    )

    assert str(version).startswith("20260426-")
    assert version.hash == version.config_hash
    assert version.version_id == version.version

    saved_path = version.save(tmp_path)
    saved_payload = json.loads(saved_path.read_text())

    assert saved_path == tmp_path / "versions" / f"{version.version}.json"
    assert saved_payload["version_id"] == version.version
    assert saved_payload["datetime"] == timestamp.isoformat()
    assert saved_payload["description"] == "baseline"
    assert saved_payload["config"] == {"name": "vae", "latent_dim": 8}


def test_datetime_encoder_delegates_non_datetime_values() -> None:
    """The custom JSON encoder should only special-case datetimes."""
    encoded = json.dumps({"created": datetime(2026, 4, 26, 9, 30)}, cls=DateTimeEncoder)

    assert "2026-04-26T09:30:00" in encoded


def test_registry_round_trips_versions_after_reloading_index(tmp_path) -> None:
    """Registry lookups should preserve usable datetime metadata after JSON reload."""
    registry = ConfigVersionRegistry(tmp_path)
    registered = registry.register({"name": "flow", "depth": 4}, description="candidate")

    reloaded = ConfigVersionRegistry(tmp_path)
    by_version = reloaded.get_by_version(registered.version)
    by_hash = reloaded.get_by_hash(registered.hash[:6])

    assert by_version.version == registered.version
    assert by_hash.version == registered.version
    assert isinstance(by_version.timestamp, datetime)
    assert by_version.to_dict()["datetime"] == registered.timestamp.isoformat()


def test_registry_reports_missing_file_and_missing_hash(tmp_path) -> None:
    """Registry lookups should fail clearly for missing versions and hash prefixes."""
    registry = ConfigVersionRegistry(tmp_path)

    with pytest.raises(ValueError, match="not found in registry"):
        registry.get_by_version("missing")
    with pytest.raises(ValueError, match="No configuration found"):
        registry.get_by_hash("abcdef")


def test_registry_reports_ambiguous_hash_prefix(tmp_path) -> None:
    """Hash-prefix lookups should reject ambiguous matches."""
    registry = ConfigVersionRegistry(tmp_path)
    registry.index = {
        "v1": {"hash": "abc11111", "timestamp": datetime(2026, 4, 26), "description": "one"},
        "v2": {"hash": "abc22222", "timestamp": datetime(2026, 4, 27), "description": "two"},
    }
    registry._save_index()

    reloaded = ConfigVersionRegistry(tmp_path)
    with pytest.raises(ValueError, match="ambiguous hash prefix"):
        reloaded.get_by_hash("abc")
