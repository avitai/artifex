"""Configuration and fixtures for end-to-end tests."""

import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture(scope="session")
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace for E2E tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        yield workspace


@pytest.fixture(scope="session")
def e2e_config() -> dict[str, Any]:
    """Configuration for E2E tests."""
    return {
        "batch_size": 4,
        "num_epochs": 2,
        "learning_rate": 1e-3,
        "image_size": (32, 32, 3),
        "latent_dim": 16,
        "hidden_dims": [32, 64],
        "num_samples": 8,
    }


@pytest.fixture
def sample_dataset() -> dict[str, jax.Array]:
    """Create a small sample dataset for E2E testing."""
    # Ensure JAX uses CPU
    os.environ["JAX_PLATFORMS"] = "cpu"

    # Create synthetic data
    batch_size = 16
    image_shape = (32, 32, 3)

    data = {
        "images": jnp.ones((batch_size, *image_shape)),
        "labels": jnp.arange(batch_size) % 4,  # 4 classes
    }
    return data


@pytest.fixture
def model_save_path(temp_workspace: Path) -> Path:
    """Path for saving models during E2E tests."""
    save_path = temp_workspace / "models"
    save_path.mkdir(exist_ok=True)
    return save_path


@pytest.fixture
def results_path(temp_workspace: Path) -> Path:
    """Path for saving test results."""
    results_path = temp_workspace / "results"
    results_path.mkdir(exist_ok=True)
    return results_path
