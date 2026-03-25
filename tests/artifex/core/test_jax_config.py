import os

import jax
import pytest

from artifex.generative_models import jax_config


class TestJAXConfiguration:
    @pytest.fixture(autouse=True)
    def _clear_backend_override(self, monkeypatch):
        """Keep tests independent from caller shell backend forcing."""
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    def test_compilation_cache_enabled(self):
        """Test that compilation cache is properly configured."""
        jax_config.configure_jax()

        # Check cache is enabled
        assert jax.config.jax_compilation_cache_dir is not None
        assert os.path.exists(jax.config.jax_compilation_cache_dir)

    def test_mixed_precision_configuration(self):
        """Test mixed precision settings."""
        jax_config.configure_jax(precision="bfloat16")
        assert jax.config.jax_default_matmul_precision == "bfloat16"

    def test_default_xla_flags_left_unset(self, monkeypatch):
        """Test the default configuration does not force XLA flags."""
        monkeypatch.delenv("XLA_FLAGS", raising=False)
        jax_config.configure_jax()
        xla_flags = os.environ.get("XLA_FLAGS", "")
        assert xla_flags == ""

    def test_deterministic_xla_flag_set(self, monkeypatch):
        """Test deterministic mode opts into the explicit GPU flag."""
        monkeypatch.delenv("XLA_FLAGS", raising=False)
        jax_config.configure_jax(deterministic=True)
        xla_flags = os.environ.get("XLA_FLAGS", "")
        assert "--xla_gpu_deterministic_ops=true" in xla_flags
