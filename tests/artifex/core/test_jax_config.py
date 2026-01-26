import os

import jax

from artifex.generative_models import jax_config


class TestJAXConfiguration:
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

    def test_xla_flags_set(self):
        """Test XLA optimization flags are set.

        Note: Many XLA flags were removed in JAX 0.7.x. Only test for
        flags that are still valid.
        """
        jax_config.configure_jax()
        xla_flags = os.environ.get("XLA_FLAGS", "")
        # In JAX 0.7.x, only strict_conv_algorithm_picker is set
        assert "xla_gpu_strict_conv_algorithm_picker" in xla_flags
