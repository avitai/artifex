"""Test script for NNX compatibility across refactored components."""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.distributions import (
    Bernoulli,
    Categorical,
    Mixture,
    MixtureOfGaussians,
)
from artifex.generative_models.core.layers.positional import (
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from artifex.generative_models.core.layers.resnet import (
    BottleneckBlock,
    ResNetBlock,
)


def test_resnet_blocks():
    """Test ResNetBlock and BottleneckBlock with NNX Rngs."""
    print("\n----- Testing ResNet Blocks -----")

    # Create test input
    key = jax.random.PRNGKey(0)
    input_data = jax.random.normal(key, (2, 32, 32, 16))

    # Initialize with nnx.Rngs
    key, params_key = jax.random.split(key)
    rngs = nnx.Rngs(params=params_key)

    # Test ResNetBlock
    print("Testing ResNetBlock...")
    resnet_block = ResNetBlock(in_features=16, features=16, kernel_size=3, stride=1, rngs=rngs)

    try:
        output = resnet_block(input_data, deterministic=True, rngs=rngs)
        print(f"  Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"  Failed: {e}")

    # Test BottleneckBlock
    print("Testing BottleneckBlock...")
    bottleneck_block = BottleneckBlock(
        in_features=16,
        out_features=16,
        bottleneck_expansion_ratio=4,
        kernel_size=3,
        stride=1,
        rngs=rngs,
    )

    try:
        output = bottleneck_block(input_data, deterministic=True, rngs=rngs)
        print(f"  Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"  Failed: {e}")


def test_positional_encodings():
    """Test positional encoding classes with NNX Rngs."""
    print("\n----- Testing Positional Encodings -----")

    # Create test input
    key = jax.random.PRNGKey(0)
    input_data = jax.random.normal(key, (2, 10, 64))

    # Initialize with nnx.Rngs
    key, params_key = jax.random.split(key)
    rngs = nnx.Rngs(params=params_key)

    # Test SinusoidalPositionalEncoding
    print("Testing SinusoidalPositionalEncoding...")
    sin_pe = SinusoidalPositionalEncoding(dim=64, max_len=20, rngs=rngs)

    try:
        output = sin_pe(input_data, rngs=rngs)
        print(f"  Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"  Failed: {e}")

    # Test LearnedPositionalEncoding
    print("Testing LearnedPositionalEncoding...")
    learned_pe = LearnedPositionalEncoding(dim=64, max_len=20, rngs=rngs)

    try:
        output = learned_pe(input_data, rngs=rngs)
        print(f"  Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"  Failed: {e}")

    # Test RotaryPositionalEncoding
    print("Testing RotaryPositionalEncoding...")
    # Make sure dimensions are even for rotary encoding
    even_input = jax.random.normal(key, (2, 10, 64))
    rotary_pe = RotaryPositionalEncoding(dim=64, max_len=20, rngs=rngs)

    try:
        output = rotary_pe(even_input, rngs=rngs)
        print(f"  Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"  Failed: {e}")


def test_distributions():
    """Test distribution classes with NNX Rngs."""
    print("\n----- Testing Distributions -----")

    # Initialize with nnx.Rngs
    key = jax.random.PRNGKey(0)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Test Bernoulli
    print("Testing Bernoulli...")
    bern_dist = Bernoulli(probs=jnp.array(0.7), rngs=rngs)

    try:
        samples = bern_dist.sample(sample_shape=(10,), rngs=rngs)
        log_prob = bern_dist.log_prob(jnp.array([0, 1]))
        print(f"  Success! Sample: {samples.shape}, Log prob: {log_prob.shape}")
    except Exception as e:
        print(f"  Failed: {e}")

    # Test Categorical
    print("Testing Categorical...")
    cat_dist = Categorical(probs=jnp.array([0.2, 0.3, 0.5]), rngs=rngs)

    try:
        samples = cat_dist.sample(sample_shape=(10,), rngs=rngs)
        log_prob = cat_dist.log_prob(jnp.array([0, 1, 2]))
        print(f"  Success! Sample: {samples.shape}, Log prob: {log_prob.shape}")
    except Exception as e:
        print(f"  Failed: {e}")

    # Test MixtureOfGaussians
    print("Testing MixtureOfGaussians...")
    mog_dist = MixtureOfGaussians(
        locs=jnp.array([[0.0, 0.0], [5.0, 5.0]]),
        scales=jnp.array([[1.0, 1.0], [0.5, 0.5]]),
        weights=jnp.array([0.3, 0.7]),
        rngs=rngs,
    )

    try:
        samples = mog_dist.sample(sample_shape=(10,), rngs=rngs)
        log_prob = mog_dist.log_prob(jnp.array([[0.0, 0.0], [1.0, 1.0]]))
        print(f"  Success! Sample: {samples.shape}, Log prob: {log_prob.shape}")
    except Exception as e:
        print(f"  Failed: {e}")

    # Test Mixture with component distributions
    print("Testing Mixture...")
    # Create component distributions
    bernoulli1 = Bernoulli(probs=jnp.array(0.3), rngs=rngs)
    bernoulli2 = Bernoulli(probs=jnp.array(0.7), rngs=rngs)

    # Create the mixture distribution
    mixture_dist = Mixture(
        components=[bernoulli1, bernoulli2], weights=jnp.array([0.4, 0.6]), rngs=rngs
    )

    try:
        samples = mixture_dist.sample(sample_shape=(10,), rngs=rngs)
        log_prob = mixture_dist.log_prob(jnp.array([0, 1]))
        print(f"  Success! Sample: {samples.shape}, Log prob: {log_prob.shape}")
    except Exception as e:
        print(f"  Failed: {e}")


if __name__ == "__main__":
    test_resnet_blocks()
    test_positional_encodings()
    test_distributions()
    print("\nAll tests completed!")
