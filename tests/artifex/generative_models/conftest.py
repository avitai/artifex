"""Common fixtures for artifex generative models tests."""

from pathlib import Path

import jax
import pytest


@pytest.fixture
def rng_key() -> jax.Array:
    """Create a fixed RNG key for tests."""
    return jax.random.key(42)


@pytest.fixture
def rng_keys() -> dict[str, jax.Array]:
    """Create a dictionary of RNG keys for different purposes."""
    main_key = jax.random.key(42)
    keys = jax.random.split(main_key, 5)
    return {
        "init": keys[0],
        "params": keys[1],
        "dropout": keys[2],
        "sampling": keys[3],
        "data": keys[4],
    }


@pytest.fixture
def batch_size() -> int:
    """Return a standard batch size for tests."""
    return 4


@pytest.fixture
def seq_length() -> int:
    """Return a standard sequence length for tests."""
    return 16


@pytest.fixture
def embed_dim() -> int:
    """Return a standard embedding dimension for tests."""
    return 32


@pytest.fixture
def hidden_dim() -> int:
    """Return a standard hidden dimension size for tests."""
    return 64


@pytest.fixture
def num_heads() -> int:
    """Return a standard number of attention heads for tests."""
    return 4


@pytest.fixture
def image_dims() -> tuple[int, int, int]:
    """Return standard height, width, channels for image tests."""
    return (32, 32, 3)


@pytest.fixture
def height() -> int:
    """Return a standard image height for tests."""
    return 32


@pytest.fixture
def width() -> int:
    """Return a standard image width for tests."""
    return 32


@pytest.fixture
def channels() -> int:
    """Return a standard number of image channels for tests."""
    return 3


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for file operations."""
    test_dir = tmp_path / "test_output"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir
