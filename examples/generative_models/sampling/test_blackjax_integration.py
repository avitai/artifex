"""Example script demonstrating BlackJAX integration with Artifex distributions.

This script shows how to use the BlackJAX samplers with various Artifex
distributions for MCMC sampling.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from artifex.generative_models.core.distributions import (
    Mixture,
    Normal,
)
from artifex.generative_models.core.sampling.blackjax_samplers import (
    hmc_sampling,
    mala_sampling,
    nuts_sampling,
)


def plot_samples(samples, title="MCMC Samples", filename=None, true_params=None):
    """Plot MCMC samples.

    Args:
        samples: Array of samples with shape [n_samples, d]
        title: Plot title
        filename: If provided, save plot to this file
        true_params: Optional tuple of (mean, scale) to plot true distribution
    """
    plt.figure(figsize=(10, 6))

    # For 1D data, plot histogram
    if samples.shape[1] == 1:
        plt.hist(samples, bins=50, density=True, alpha=0.7)

        # Plot true density if provided
        if true_params:
            true_mean, true_scale = true_params
            x = jnp.linspace(jnp.min(samples) - 1, jnp.max(samples) + 1, 1000)
            density = jnp.exp(-0.5 * ((x - true_mean) / true_scale) ** 2) / (
                true_scale * jnp.sqrt(2 * jnp.pi)
            )
            plt.plot(x, density, "r-", linewidth=2)

        plt.xlabel("Value")
        plt.ylabel("Density")

    # For 2D data, plot scatter
    elif samples.shape[1] == 2:
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)

        # Plot true mean if provided
        if true_params:
            true_mean, _ = true_params
            if hasattr(true_mean, "__len__") and len(true_mean) == 2:
                plt.plot(true_mean[0], true_mean[1], "r*", markersize=15, label="True Mean")
                plt.legend()

        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")

    plt.title(title)
    plt.grid(True)

    if filename:
        import os

        output_dir = "examples_output"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        print(f"Plot saved as '{filepath}'")

    # Print summary statistics
    mean = jnp.mean(samples, axis=0)
    std = jnp.std(samples, axis=0)

    print(f"\nSample Statistics for {title}:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")


def example_normal_hmc():
    """Sample from a normal distribution using HMC."""
    print("\n===== Example: Normal Distribution with HMC =====")

    # Create distribution
    true_mean = jnp.array([3.0, -2.0])
    true_scale = jnp.array([1.5, 0.8])
    normal_dist = Normal(loc=true_mean, scale=true_scale)

    # Initial position
    init_position = jnp.zeros(2)

    # Set sampling parameters
    key = jax.random.key(0)
    n_samples = 1000
    n_burnin = 500

    # Sample using HMC
    samples = hmc_sampling(
        normal_dist,  # Directly pass the distribution
        init_position,
        key,
        n_samples=n_samples,
        n_burnin=n_burnin,
        step_size=0.1,
        num_integration_steps=10,
    )

    # Plot results
    plot_samples(
        samples,
        "Normal Distribution - HMC Sampling",
        "normal_hmc_samples.png",
        (true_mean, true_scale),
    )


def example_mixture_mala():
    """Sample from a mixture of Gaussians using MALA."""
    print("\n===== Example: Mixture of Gaussians with MALA =====")

    # Create a mixture of two Gaussians
    weights = jnp.array([0.7, 0.3])
    means = jnp.array([[0.0, 0.0], [5.0, 5.0]])
    scales = jnp.array([[1.0, 1.0], [0.5, 0.5]])

    components = [Normal(loc=means[0], scale=scales[0]), Normal(loc=means[1], scale=scales[1])]

    mixture = Mixture(components, weights)

    # Initial position
    init_position = jnp.zeros(2)

    # Set sampling parameters
    key = jax.random.key(1)
    n_samples = 1000
    n_burnin = 500

    # Sample using MALA
    samples = mala_sampling(
        mixture,  # Directly pass the mixture distribution
        init_position,
        key,
        n_samples=n_samples,
        n_burnin=n_burnin,
        step_size=0.05,
    )

    # Calculate the true mean of the mixture
    true_mean = jnp.sum(weights[:, None] * means, axis=0)

    # Plot results
    plot_samples(
        samples,
        "Mixture of Gaussians - MALA Sampling",
        "mixture_mala_samples.png",
        (true_mean, None),
    )


def example_univariate_normal_nuts():
    """Sample from a univariate normal distribution using NUTS."""
    print("\n===== Example: Univariate Normal with NUTS =====")
    print("Note: This may fail if memory is limited")

    # Create univariate normal distribution
    true_mean = jnp.array([2.0])
    true_scale = jnp.array([1.0])
    normal_dist = Normal(loc=true_mean, scale=true_scale)

    # Initial position
    init_position = jnp.zeros(1)

    # Set sampling parameters
    key = jax.random.key(2)
    n_samples = 500  # Fewer samples for NUTS
    n_burnin = 200

    try:
        # Sample using NUTS with minimal parameters
        samples = nuts_sampling(
            normal_dist,  # Directly pass the distribution
            init_position,
            key,
            n_samples=n_samples,
            n_burnin=n_burnin,
            step_size=0.1,
            max_depth=5,  # Limit tree depth for memory
        )

        # Plot results
        plot_samples(
            samples,
            "Univariate Normal - NUTS Sampling",
            "univariate_nuts_samples.png",
            (true_mean, true_scale),
        )
    except (RuntimeError, MemoryError) as e:
        print(f"NUTS sampling failed due to: {e}")
        print("This is likely due to memory constraints.")
        print("Try using HMC or MALA instead, or reduce the number of samples.")


if __name__ == "__main__":
    print("Testing BlackJAX integration with Artifex distributions...")

    # Run examples
    example_normal_hmc()
    example_mixture_mala()
    example_univariate_normal_nuts()

    print("\nAll examples complete!")
