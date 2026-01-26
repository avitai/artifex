"""Basic sanity check tests for configuration system."""

import pytest
from pydantic import BaseModel, Field, field_validator


class SimpleConfig(BaseModel):
    """Simple configuration class for basic testing."""

    name: str = Field("default_name", description="Configuration name")
    version: str = Field("1.0.0", description="Configuration version")
    enabled: bool = Field(True, description="Whether the configuration is enabled")
    value: int = Field(0, description="A numeric value for testing")

    @field_validator("value")
    @classmethod
    def validate_value(cls, v):
        """Validate that value is non-negative."""
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        """Validate that version follows semantic versioning."""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError("Version must follow semantic versioning (x.y.z)")
        try:
            # Check that all parts are integers
            for part in parts:
                int(part)
        except ValueError as err:
            raise ValueError("Version parts must be integers") from err
        return v

    class Config:
        """Pydantic config."""

        extra = "forbid"


def test_simple_config_defaults():
    """Test default values for simple configuration."""
    config = SimpleConfig()
    assert config.name == "default_name"
    assert config.version == "1.0.0"
    assert config.enabled is True
    assert config.value == 0


def test_simple_config_custom_values():
    """Test that custom values are properly set."""
    config = SimpleConfig(
        name="test_config",
        version="2.0.0",
        enabled=False,
        value=42,
    )
    assert config.name == "test_config"
    assert config.version == "2.0.0"
    assert config.enabled is False
    assert config.value == 42


def test_simple_config_validation():
    """Test validation rules for simple configuration."""
    with pytest.raises(ValueError, match="Value must be non-negative"):
        SimpleConfig(value=-1)

    # Valid values should not raise exceptions
    SimpleConfig(value=0)  # Min valid value
    SimpleConfig(value=100)  # Regular valid value


def test_simple_config_version_validation():
    """Test version validation in SimpleConfig."""
    # Valid versions
    config1 = SimpleConfig(version="1.0.0")
    config2 = SimpleConfig(version="2.1.3")
    assert config1.version == "1.0.0"
    assert config2.version == "2.1.3"

    # Invalid versions
    with pytest.raises(ValueError, match="Version must follow semantic"):
        SimpleConfig(version="1.0")

    with pytest.raises(ValueError, match="Version must follow semantic"):
        SimpleConfig(version="1.0.0.0")

    with pytest.raises(ValueError, match="Version parts must be integers"):
        SimpleConfig(version="1.a.0")


def test_simple_config_extra_forbidden():
    """Test that extra fields are not allowed."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SimpleConfig(extra_field="not allowed")
