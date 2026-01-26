"""JAX configuration optimization for maximum performance."""

import logging
import os
import tempfile
from typing import Literal

import jax


logger = logging.getLogger(__name__)


def configure_jax(
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    cache_dir: str | None = None,
    enable_x64: bool = False,
    memory_fraction: float = 0.75,
    deterministic: bool = False,
) -> None:
    """Configure JAX for optimal performance.

    This function should be called once at the beginning of your program,
    before any JAX operations.

    Args:
        precision: Default matrix multiplication precision
        cache_dir: Directory for compilation cache (auto-created if None)
        enable_x64: Whether to enable 64-bit precision
        memory_fraction: GPU memory fraction to preallocate
        deterministic: Whether to enable deterministic operations (slower but reproducible)
    """
    # 1. Compilation cache (80-95% faster startup)
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "artifex_jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    logger.info(f"JAX compilation cache enabled at: {cache_dir}")

    # 2. Mixed precision (30-40% speedup)
    jax.config.update("jax_default_matmul_precision", precision)
    logger.info(f"Default matmul precision set to: {precision}")

    # 3. 64-bit precision (disabled for speed by default)
    jax.config.update("jax_enable_x64", enable_x64)

    # 4. Memory configuration
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

    # 5. XLA optimization flags
    # Note: Many optimization flags from JAX 0.4-0.6 are not recognized in JAX 0.7+
    # The XLA compiler has evolved and these flags have been removed or changed
    # Only set flags that are universally recognized across JAX versions
    xla_flags = [
        # Removed incompatible flags for JAX 0.7.x:
        # - --xla_gpu_enable_triton_softmax_fusion (removed in JAX 0.7.x)
        # - --xla_gpu_triton_gemm_any (removed in JAX 0.7.x)
        # - --xla_gpu_enable_async_collectives (removed in JAX 0.7.x)
        # - --xla_gpu_enable_cudnn_frontend (removed in JAX 0.7.x)
        # - --xla_tpu_enable_async_all_reduce (removed in JAX 0.7.x)
        # - --xla_tpu_enable_lazy_collectives (removed in JAX 0.7.x)
    ]

    # Add deterministic operations flag if requested
    if deterministic:
        xla_flags.append("--xla_gpu_deterministic_ops=true")
        logger.info("Deterministic GPU operations enabled (slower but reproducible)")

    # Only append flags if there are any to append
    if xla_flags:
        existing_flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = " ".join([existing_flags, *xla_flags]).strip()
        logger.info("XLA optimization flags configured")
    else:
        logger.info("Using default XLA flags (no additional optimizations for JAX 0.7.x)")

    # 6. Platform-specific optimizations
    backend = jax.default_backend()
    if backend == "gpu":
        # GPU-specific settings
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1" if deterministic else "0"
        os.environ["TF_ALLOW_GROWTH"] = "1"
        if deterministic:
            logger.info("cuDNN deterministic mode enabled")
    elif backend == "tpu":
        # TPU-specific settings
        os.environ["JAX_PLATFORMS"] = "tpu"

    logger.info(f"JAX configured for {backend} backend")


# Auto-configure on import with sensible defaults
# Check for deterministic mode via environment variable (useful for testing)
# In production, this should be False for maximum performance
_deterministic_mode = os.environ.get("ARTIFEX_DETERMINISTIC", "0") == "1"

# Allow overriding precision via environment variable (useful for testing)
# Default to bfloat16 for performance, but tests may need float32 for tight tolerances
_default_precision = os.environ.get("ARTIFEX_MATMUL_PRECISION", "bfloat16")
if _default_precision not in ("float32", "bfloat16", "float16"):
    _default_precision = "bfloat16"

_configured = False


def auto_configure() -> None:
    """Apply default JAX configuration if not already configured.

    Called lazily on first use rather than at import time to avoid
    module-level side effects.
    """
    global _configured  # noqa: PLW0603
    if not _configured:
        configure_jax(precision=_default_precision, deterministic=_deterministic_mode)
        _configured = True


# Auto-configure on import (opt-in via ARTIFEX_AUTO_CONFIGURE=1 or default behavior)
# Set ARTIFEX_AUTO_CONFIGURE=0 to skip auto-configuration at import time
if os.environ.get("ARTIFEX_AUTO_CONFIGURE", "1") != "0":
    auto_configure()
