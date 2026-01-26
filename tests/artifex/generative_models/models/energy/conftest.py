"""Fixtures for energy model tests."""

import pytest


# Import JAX dependencies with fallback for tests
try:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

    # Create dummy objects for environments without JAX
    from typing import Any

    import numpy as np

    class DummyJAX:
        """Mock JAX implementation for testing energy models."""

        class random:
            """Mock JAX random module."""

            @staticmethod
            def key(seed):
                """Create a mock random key from a seed."""
                return np.array([seed], dtype=np.uint32)

            @staticmethod
            def split(key, num=2):
                """Split a mock random key into multiple keys."""
                return [np.array([k], dtype=np.uint32) for k in range(key[0], key[0] + num)]

            @staticmethod
            def normal(key, shape):
                """Generate mock normal random values."""
                np.random.seed(key[0])
                return np.random.normal(size=shape)

            @staticmethod
            def uniform(key, shape, minval=0, maxval=1):
                """Generate mock uniform random values."""
                np.random.seed(key[0])
                return np.random.uniform(minval, maxval, size=shape)

    mock_jax: Any = DummyJAX()

    class DummyJNP:
        """Mock JAX NumPy implementation for testing."""

        @staticmethod
        def array(x):
            """Convert to a mock JAX array."""
            return np.array(x)

        @staticmethod
        def ones(shape):
            """Create an array of ones."""
            return np.ones(shape)

        @staticmethod
        def zeros(shape):
            """Create an array of zeros."""
            return np.zeros(shape)

        @staticmethod
        def float32(x):
            """Convert to float32."""
            return np.float32(x)

    mock_jnp: Any = DummyJNP()

    class DummyNNX:
        """Mock NNX implementation for testing energy models."""

        class Rngs:
            """Mock NNX Rngs class."""

            def __init__(self, seed=0):
                """Initialize with a seed value.

                Args:
                    seed: Random seed for initialization
                """
                self.seed = seed
                self.keys = {"params": np.array([seed], dtype=np.uint32)}

    mock_nnx: Any = DummyNNX()


# Skip tests that require JAX if it's not available
jax_required = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


@pytest.fixture
def energy_rng_key():
    """Create a fixed RNG key for energy model tests."""
    if not JAX_AVAILABLE:
        return 0
    return jax.random.key(12345)


@pytest.fixture
def energy_rngs():
    """Create RNG objects for energy model initialization."""
    if not JAX_AVAILABLE:
        return nnx.Rngs(12345)
    return nnx.Rngs(12345)


@pytest.fixture
def energy_rng_keys():
    """Create multiple RNG keys for different purposes."""
    if not JAX_AVAILABLE:
        return {
            "init": 0,
            "params": 1,
            "sampling": 2,
            "mcmc": 3,
            "noise": 4,
        }

    main_key = jax.random.key(12345)
    keys = jax.random.split(main_key, 5)
    return {
        "init": keys[0],
        "params": keys[1],
        "sampling": keys[2],
        "mcmc": keys[3],
        "noise": keys[4],
    }


@pytest.fixture
def energy_test_batch_size():
    """Standard batch size for energy model tests."""
    return 8


@pytest.fixture
def energy_test_input_dim():
    """Standard input dimension for MLP energy functions."""
    return 16


@pytest.fixture
def energy_test_hidden_dims():
    """Standard hidden dimensions for energy functions."""
    return [32, 32]


@pytest.fixture
def energy_test_image_shape():
    """Standard image shape for CNN energy functions."""
    return (16, 16, 1)  # Small images for fast testing


@pytest.fixture
def energy_test_mlp_data(energy_test_batch_size, energy_test_input_dim):
    """Generate test data for MLP energy functions."""
    if not JAX_AVAILABLE:
        return [[0.5] * energy_test_input_dim] * energy_test_batch_size

    key = jax.random.key(42)
    return jax.random.normal(key, (energy_test_batch_size, energy_test_input_dim))


@pytest.fixture
def energy_test_image_data(energy_test_batch_size, energy_test_image_shape):
    """Generate test data for CNN energy functions."""
    if not JAX_AVAILABLE:
        return [[[[0.5]]]] * energy_test_batch_size

    key = jax.random.key(42)
    shape = (energy_test_batch_size, *energy_test_image_shape)
    return jax.random.normal(key, shape)


@pytest.fixture
def energy_mcmc_config():
    """Standard MCMC configuration for testing."""
    return {
        "n_steps": 10,  # Small number for fast testing
        "step_size": 0.01,
        "noise_scale": 0.005,
        "clip_range": (-1.0, 1.0),
        "grad_clip": 0.03,
    }


@pytest.fixture
def energy_buffer_config():
    """Standard sample buffer configuration for testing."""
    return {
        "capacity": 64,  # Small buffer for testing
        "reinit_prob": 0.1,
    }


@pytest.fixture
def energy_loss_config():
    """Standard loss configuration for testing."""
    return {
        "alpha": 0.01,  # Regularization strength
    }


@pytest.fixture(params=["mlp", "cnn"])
def energy_function_type(request):
    """Parameterized fixture for different energy function types."""
    return request.param


@pytest.fixture(params=[1, 2, 4])
def energy_test_batch_sizes(request):
    """Parameterized fixture for different batch sizes."""
    return request.param


@pytest.fixture
def energy_tolerance():
    """Numerical tolerance for energy model tests."""
    return 1e-4


@pytest.fixture
def energy_small_tolerance():
    """Smaller numerical tolerance for precise tests."""
    return 1e-6


# Synthetic data generators for different scenarios
@pytest.fixture
def energy_synthetic_data_2d():
    """Generate 2D synthetic data for energy model testing."""
    if not JAX_AVAILABLE:
        return [[0.5, 0.5]] * 16

    key = jax.random.key(999)
    # Create mixture of 2D Gaussians
    n_points = 16
    centers = jnp.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])

    data_points = []
    for i in range(n_points):
        center_idx = i % len(centers)
        center = centers[center_idx]
        key, subkey = jax.random.split(key)
        point = center + 0.2 * jax.random.normal(subkey, (2,))
        data_points.append(point)

    return jnp.stack(data_points)


@pytest.fixture
def energy_synthetic_images():
    """Generate synthetic images for CNN energy model testing."""
    if not JAX_AVAILABLE:
        return [[[[0.5]]]] * 8

    key = jax.random.key(888)
    batch_size = 8
    height, width, channels = 8, 8, 1

    # Create simple patterns
    images = []
    for i in range(batch_size):
        key, subkey = jax.random.split(key)
        # Create checkerboard-like patterns with noise
        image = jnp.zeros((height, width, channels))
        for h in range(height):
            for w in range(width):
                if (h + w) % 2 == i % 2:
                    image = image.at[h, w, 0].set(1.0)

        # Add noise
        noise = 0.1 * jax.random.normal(subkey, (height, width, channels))
        image = image + noise
        images.append(image)

    return jnp.stack(images)


@pytest.fixture
def energy_training_batch():
    """Create a standard training batch for energy models."""
    if not JAX_AVAILABLE:
        return {"data": [[0.5] * 16] * 4}

    key = jax.random.key(777)
    batch_size = 4
    data_dim = 16

    data = jax.random.normal(key, (batch_size, data_dim))

    return {
        "data": data,
        "batch_size": batch_size,
    }
