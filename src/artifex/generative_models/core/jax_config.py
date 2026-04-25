"""JAX configuration optimization for maximum performance."""

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import cast, Literal


logger = logging.getLogger(__name__)

MatmulPrecision = Literal["float32", "bfloat16", "float16"]


@contextlib.contextmanager
def _suppress_xla_bridge_probe_logs():
    """Silence noisy backend-probe logs when JAX falls back to CPU."""
    bridge_logger = logging.getLogger("jax._src.xla_bridge")
    previous_disabled = bridge_logger.disabled
    bridge_logger.disabled = True
    try:
        yield
    finally:
        bridge_logger.disabled = previous_disabled


def _get_jax():
    """Import JAX lazily so importing this module stays side-effect free."""
    import jax

    return jax


def configure_jax(
    precision: MatmulPrecision = "bfloat16",
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
    jax = _get_jax()

    # 1. Compilation cache (80-95% faster startup)
    if cache_dir is None:
        cache_dir = str(Path(tempfile.gettempdir()) / "artifex_jax_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
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
    # Most legacy optimization flags from older JAX releases are no longer
    # stable or necessary. Leave XLA_FLAGS untouched by default and only add
    # explicit flags for opt-in behaviors such as deterministic execution.
    xla_flags = [
        # Intentionally empty by default.
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
        logger.info("Using default XLA flags (no additional XLA flags configured)")

    # 6. Platform-specific optimizations
    with _suppress_xla_bridge_probe_logs():
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
_env_precision = os.environ.get("ARTIFEX_MATMUL_PRECISION", "bfloat16")
if _env_precision not in ("float32", "bfloat16", "float16"):
    _env_precision = "bfloat16"
_default_precision: MatmulPrecision = cast(MatmulPrecision, _env_precision)

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


# Auto-configure on import only when explicitly requested.
if os.environ.get("ARTIFEX_AUTO_CONFIGURE", "0") == "1":
    auto_configure()
