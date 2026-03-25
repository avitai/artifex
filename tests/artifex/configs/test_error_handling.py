"""Tests for configuration error handling utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from artifex.configs.utils.error_handling import (
    ConfigError,
    ConfigLoadError,
    ConfigNotFoundError,
    ConfigValidationError,
    safe_load_config,
)


class TestConfigError:
    """Test base ConfigError class."""

    def test_basic_message(self) -> None:
        """Test error with just a message."""
        err = ConfigError("something went wrong")
        assert "something went wrong" in str(err)
        assert err.config_path is None
        assert err.field is None

    def test_with_config_path(self) -> None:
        """Test error includes file name."""
        err = ConfigError("bad config", config_path="/tmp/model.yaml")
        assert "model.yaml" in str(err)
        assert err.config_path == "/tmp/model.yaml"

    def test_with_field(self) -> None:
        """Test error includes field name."""
        err = ConfigError("invalid value", field="learning_rate")
        assert "learning_rate" in str(err)

    def test_with_context(self) -> None:
        """Test error includes context dict."""
        err = ConfigError("error", context={"Expected": "float", "Got": "str"})
        msg = str(err)
        assert "Expected: float" in msg
        assert "Got: str" in msg

    def test_inheritance(self) -> None:
        """Test that ConfigError is an Exception."""
        assert issubclass(ConfigError, Exception)


class TestConfigNotFoundError:
    """Test ConfigNotFoundError."""

    def test_basic(self) -> None:
        """Test not found error with config name."""
        err = ConfigNotFoundError("model.yaml")
        assert "model.yaml" in str(err)
        assert "could not be found" in str(err)

    def test_with_search_paths(self) -> None:
        """Test not found error includes search paths."""
        paths = ["/a/model.yaml", "/b/model.yaml"]
        err = ConfigNotFoundError("model.yaml", paths)
        msg = str(err)
        assert "/a/model.yaml" in msg
        assert "/b/model.yaml" in msg

    def test_inheritance(self) -> None:
        """Test that it's a ConfigError subclass."""
        assert issubclass(ConfigNotFoundError, ConfigError)


class TestConfigLoadError:
    """Test ConfigLoadError."""

    def test_wraps_original_error(self) -> None:
        """Test that original error info is preserved."""
        original = ValueError("invalid yaml")
        err = ConfigLoadError("/tmp/bad.yaml", original)
        msg = str(err)
        assert "invalid yaml" in msg
        assert "ValueError" in msg
        assert "bad.yaml" in msg

    def test_inheritance(self) -> None:
        """Test that it's a ConfigError subclass."""
        assert issubclass(ConfigLoadError, ConfigError)


class TestConfigValidationError:
    """Test ConfigValidationError."""

    def test_wraps_validation_error(self) -> None:
        """Test that validation error info is preserved."""
        original = ValueError("lr must be positive")
        err = ConfigValidationError("/tmp/config.yaml", original)
        msg = str(err)
        assert "lr must be positive" in msg
        assert "config.yaml" in msg

    def test_inheritance(self) -> None:
        """Test that it's a ConfigError subclass."""
        assert issubclass(ConfigValidationError, ConfigError)


class TestSafeLoadConfig:
    """Test safe_load_config wrapper."""

    def test_successful_load(self) -> None:
        """Test that successful loads pass through."""
        result = safe_load_config(lambda p: {"key": "value"}, "dummy.yaml")
        assert result == {"key": "value"}

    def test_file_not_found(self) -> None:
        """Test FileNotFoundError is wrapped."""

        def bad_load(p: str | Path) -> None:
            raise FileNotFoundError("no such file")

        with pytest.raises(ConfigNotFoundError):
            safe_load_config(bad_load, "missing.yaml")

    def test_value_error(self) -> None:
        """Test ValueError is wrapped as validation error."""

        def bad_load(p: str | Path) -> None:
            raise ValueError("invalid yaml")

        with pytest.raises(ConfigValidationError):
            safe_load_config(bad_load, "bad.yaml")

    def test_generic_error(self) -> None:
        """Test generic exceptions are wrapped as load error."""

        def bad_load(p: str | Path) -> None:
            raise RuntimeError("io failure")

        with pytest.raises(ConfigLoadError):
            safe_load_config(bad_load, "broken.yaml")

    def test_existing_config_error_passthrough(self) -> None:
        """Existing config errors should survive without being rewrapped."""
        original_error = ConfigValidationError("broken.yaml", ValueError("bad payload"))

        def bad_load(p: str | Path) -> None:
            raise original_error

        with pytest.raises(ConfigValidationError) as exc_info:
            safe_load_config(bad_load, "broken.yaml")

        assert exc_info.value is original_error
