# %% [markdown]
"""
# BlackJAX Integration Examples with Artifex Distributions

This example demonstrates advanced integration patterns between BlackJAX samplers and
Artifex's distribution framework.

## Learning Objectives

After completing this example, you will:

- [ ] Understand how to use BlackJAX sampler classes with Artifex distributions
- [ ] Learn to use both class-based and functional sampling APIs
- [ ] Apply samplers to Artifex distributions (Normal, Mixture)
- [ ] Compare class-based vs functional sampling approaches
- [ ] Handle memory constraints in NUTS sampling
- [ ] Sample from mixture distributions using MCMC

## Prerequisites

- Understanding of MCMC sampling concepts
- Familiarity with Artifex distributions module
- Basic knowledge of HMC, MALA, and NUTS algorithms
- Artifex core sampling and distributions modules

## Integration Approaches

Artifex supports two ways to integrate with BlackJAX:

### 1. Direct BlackJAX API (This Example)
Use BlackJAX's native API directly with Artifex distributions:
- Create log probability function from Artifex distribution
- Use `blackjax.hmc()`, `blackjax.mala()`, `blackjax.nuts()` directly
- Full control over sampling parameters and state management
- Useful when you need maximum flexibility

### 2. Artifex Functional API
Use Artifex's convenience functions like `hmc_sampling()`, `mala_sampling()`:
- Single function call for complete sampling workflow
- Automatic burn-in and state management
- Simplified interface for common use cases
- Recommended for most applications

## Example Overview

This example includes:

1. **Normal Distribution with Direct BlackJAX HMC**: Using BlackJAX HMC API directly
2. **Normal Distribution with hmc_sampling**: Using Artifex's functional API
3. **Normal Distribution with Direct BlackJAX MALA**: Using BlackJAX MALA API directly
4. **Univariate Normal with Direct BlackJAX NUTS**: NUTS on 1D distribution (memory-aware)
5. **Multimodal Distribution Comparison**: Teaching example comparing MALA vs NUTS
   - 5a: Mixture with MALA (demonstrates local sampler limitations)
   - 5b: Mixture with NUTS (demonstrates Hamiltonian dynamics success)

## Author

Artifex Team

## License

MIT License
"""

# %%
"""Import required libraries."""

import time

import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from artifex.generative_models.core.distributions import Mixture, Normal
from artifex.generative_models.core.sampling.blackjax_samplers import (
    hmc_sampling,
    mala_sampling,
    nuts_sampling,
)


# %% [markdown]
"""
## Helper Function: Plotting Samples

We define a helper function to visualize MCMC samples for 1D and 2D distributions.
"""


# %%
def plot_samples(samples, title="MCMC Samples", filename=None, true_params=None):
    """Plot MCMC samples.

    Args:
        samples: Array of samples with shape [n_samples, d]
        title: Plot title
        filename: If provided, save plot to this file
        true_params: Optional tuple of (mean, scale) for true distribution
    """
    plt.figure(figsize=(10, 6))

    # For 1D data, plot histogram
    if samples.shape[1] == 1:
        plt.hist(samples, bins=50, density=True, alpha=0.7)

        # Plot true density if provided
        if true_params:
            true_mean, true_scale = true_params
            if true_mean is not None and true_scale is not None:
                x = jnp.linspace(jnp.min(samples) - 1, jnp.max(samples) + 1, 1000)
                density = jnp.exp(-0.5 * ((x - true_mean) / true_scale) ** 2) / (
                    true_scale * jnp.sqrt(2 * jnp.pi)
                )
                plt.plot(x, density, "r-", linewidth=2, label="True Density")
                plt.legend()

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
    sample_mean = jnp.mean(samples, axis=0)
    sample_std = jnp.std(samples, axis=0)

    print(f"\n{title} - Statistics:")
    print("=" * 70)

    # Print true parameters if provided
    if true_params:
        true_mean, true_scale = true_params
        if true_mean is None and true_scale is None:
            # No true parameters - just print sample statistics
            print(f"Sample Mean: {sample_mean}")
            print(f"Sample Std:  {sample_std}")
        elif true_scale is not None:
            # Format as table
            print(f"{'Statistic':<15} {'True Value':<30} {'Sample Value':<30}")
            print("-" * 70)

            # Handle scalar vs vector
            if hasattr(true_mean, "__len__"):
                # Vector format (handles all arrays including 1D)
                mean_true_str = "[" + ", ".join(f"{float(x):>7.4f}" for x in true_mean) + "]"
                mean_sample_str = "[" + ", ".join(f"{float(x):>7.4f}" for x in sample_mean) + "]"
                print(f"{'Mean':<15} {mean_true_str:<30} {mean_sample_str:<30}")

                std_true_str = "[" + ", ".join(f"{float(x):>7.4f}" for x in true_scale) + "]"
                std_sample_str = "[" + ", ".join(f"{float(x):>7.4f}" for x in sample_std) + "]"
                print(f"{'Std':<15} {std_true_str:<30} {std_sample_str:<30}")
            else:
                # Scalar format (true scalars only)
                mean_true = f"{float(true_mean):>8.4f}"
                mean_sample = f"{float(sample_mean):>8.4f}"
                print(f"{'Mean':<15} {mean_true:<30} {mean_sample:<30}")
                std_true = f"{float(true_scale):>8.4f}"
                std_sample = f"{float(sample_std):>8.4f}"
                print(f"{'Std':<15} {std_true:<30} {std_sample:<30}")
        else:
            # Only mean provided (for mixture)
            print(f"{'Statistic':<15} {'True Value':<30} {'Sample Value':<30}")
            print("-" * 70)
            if hasattr(true_mean, "__len__"):
                mean_true_str = "[" + ", ".join(f"{float(x):>7.4f}" for x in true_mean) + "]"
                mean_sample_str = "[" + ", ".join(f"{float(x):>7.4f}" for x in sample_mean) + "]"
                print(f"{'Mean':<15} {mean_true_str:<30} {mean_sample_str:<30}")

                std_sample_str = "[" + ", ".join(f"{float(x):>7.4f}" for x in sample_std) + "]"
                print(f"{'Std':<15} {'N/A':<30} {std_sample_str:<30}")
            else:
                print(
                    f"{'Mean':<15} {float(true_mean):>8.4f}              {float(sample_mean):>8.4f}"
                )
                print(f"{'Std':<15} {'N/A':<30} {float(sample_std):>8.4f}")
    else:
        print(f"Sample Mean: {sample_mean}")
        print(f"Sample Std:  {sample_std}")

    print("=" * 70)


# %% [markdown]
"""
## Example 1: Normal Distribution with Direct BlackJAX HMC

This example demonstrates using BlackJAX's HMC sampler directly with a Artifex
Normal distribution. This shows the low-level BlackJAX API for maximum control.

Key points:
- Create log probability function from Artifex distribution
- Initialize BlackJAX HMC sampler with proper parameters
- Manually manage sampler state and run burn-in
- Use JAX random keys correctly for each step
"""


# %%
def example_normal_hmc():
    """Sample from a normal distribution using BlackJAX HMC directly."""
    print()
    print("===== Example: Normal Distribution with Direct BlackJAX HMC =====")

    # Create distribution
    true_mean = jnp.array([3.0, -2.0])
    true_scale = jnp.array([1.5, 0.8])
    normal_dist = Normal(loc=true_mean, scale=true_scale)

    # Create log probability function that returns scalar
    def logdensity_fn(x):
        log_prob = normal_dist.log_prob(x)
        # Ensure scalar output by summing if needed
        if hasattr(log_prob, "shape") and len(log_prob.shape) > 0:
            return jnp.sum(log_prob)
        return log_prob

    # Initial position and parameters
    init_position = jnp.zeros(2)
    step_size = 0.1
    inverse_mass_matrix = jnp.ones(2)
    num_integration_steps = 10

    # Create HMC sampler
    hmc = blackjax.hmc(
        logdensity_fn,
        step_size,
        inverse_mass_matrix,
        num_integration_steps,
    )

    # Initialize sampler state
    state = hmc.init(init_position)

    # JIT compile the step function for speed
    step_fn = jax.jit(hmc.step)

    # Warm-up JIT compilation (don't count this in timing)
    key = jax.random.key(0)
    warmup_key = jax.random.fold_in(key, 999999)
    _, _ = step_fn(warmup_key, state)

    # Number of samples to collect
    n_samples = 10000
    n_burnin = 2000

    # Collect samples
    samples = jnp.zeros((n_samples, 2))

    # Run burn-in with progress bar
    print("Running burn-in...")
    start_time = time.time()
    for i in tqdm(range(n_burnin), desc="Burn-in", ncols=80):
        key = jax.random.fold_in(key, i)
        state, _ = step_fn(key, state)
    burnin_time = time.time() - start_time

    # Collect samples with progress bar
    print("Collecting samples...")
    start_time = time.time()
    for i in tqdm(range(n_samples), desc="Sampling", ncols=80):
        key = jax.random.fold_in(key, n_burnin + i)
        state, _ = step_fn(key, state)
        samples = samples.at[i].set(state.position)
    sampling_time = time.time() - start_time

    # Print timing info
    total_time = burnin_time + sampling_time
    samples_per_sec = n_samples / sampling_time
    print(f"\nTiming: {total_time:.2f}s total ({samples_per_sec:.1f} samples/sec)")

    # Plot results
    plot_samples(
        samples,
        "Normal Distribution - Direct BlackJAX HMC",
        "normal_blackjax_hmc.png",
        (true_mean, true_scale),
    )


# %% [markdown]
"""
## Example 2: Normal Distribution with hmc_sampling (Functional)

This example demonstrates using the `hmc_sampling()` function, which provides a
simplified interface for the same task. This is the recommended approach for most
use cases.

Key points:
- Single function call replaces manual state management
- Automatic burn-in handling
- Direct integration with Artifex distributions
- Cleaner, more concise code
"""


# %%
def example_normal_hmc_function():
    """Sample from a normal distribution using hmc_sampling function."""
    print()
    print("===== Example: Normal Distribution with hmc_sampling =====")

    # Create distribution
    true_mean = jnp.array([3.0, -2.0])
    true_scale = jnp.array([1.5, 0.8])
    normal_dist = Normal(loc=true_mean, scale=true_scale)

    # Initial position
    init_position = jnp.zeros(2)

    # Set sampling parameters
    key = jax.random.key(1)
    n_samples = 10000
    n_burnin = 5000

    # Warm-up run to compile JIT (don't count in timing)
    print("Warming up JIT compilation...")
    _ = hmc_sampling(
        normal_dist,
        init_position,
        jax.random.key(999999),
        n_samples=10,
        n_burnin=5,
        step_size=0.1,
        num_integration_steps=10,
    )

    print(f"Sampling {n_samples} samples with {n_burnin} burn-in...")
    start_time = time.time()

    # Call the functional API (already JIT-compiled from warm-up)
    samples = hmc_sampling(
        normal_dist,
        init_position,
        key,
        n_samples=n_samples,
        n_burnin=n_burnin,
        step_size=0.1,
        num_integration_steps=10,
    )

    total_time = time.time() - start_time
    samples_per_sec = n_samples / total_time
    print(f"Timing: {total_time:.2f}s total ({samples_per_sec:.1f} samples/sec)")

    # Plot results
    plot_samples(
        samples,
        "Normal Distribution - hmc_sampling Function",
        "normal_hmc_function.png",
        (true_mean, true_scale),
    )


# %% [markdown]
"""
## Example 3: Normal Distribution with Direct BlackJAX MALA

This example demonstrates using BlackJAX's MALA sampler directly. MALA is often
faster per iteration than HMC but may require more iterations for the same effective
sample size.

MALA uses gradient information through Langevin dynamics, making it efficient for
smooth distributions with well-behaved gradients.
"""


# %%
def example_normal_mala():
    """Sample from a normal distribution using BlackJAX MALA directly."""
    print()
    print("===== Example: Normal Distribution with Direct BlackJAX MALA =====")

    # Create distribution
    true_mean = jnp.array([3.0, -2.0])
    true_scale = jnp.array([1.5, 0.8])
    normal_dist = Normal(loc=true_mean, scale=true_scale)

    # Create log probability function that returns scalar
    def logdensity_fn(x):
        log_prob = normal_dist.log_prob(x)
        if hasattr(log_prob, "shape") and len(log_prob.shape) > 0:
            return jnp.sum(log_prob)
        return log_prob

    # Initial position and parameters
    init_position = jnp.zeros(2)
    step_size = 0.1

    # Create MALA sampler
    mala = blackjax.mala(logdensity_fn, step_size)

    # Initialize sampler state
    state = mala.init(init_position)

    # JIT compile the step function for speed
    step_fn = jax.jit(mala.step)

    # Warm-up JIT compilation (don't count this in timing)
    key = jax.random.key(2)
    warmup_key = jax.random.fold_in(key, 999999)
    _, _ = step_fn(warmup_key, state)

    # Number of samples to collect
    n_samples = 10000
    n_burnin = 2000

    # Collect samples
    samples = jnp.zeros((n_samples, 2))

    # Run burn-in with progress bar
    print("Running burn-in...")
    start_time = time.time()
    for i in tqdm(range(n_burnin), desc="Burn-in", ncols=80):
        key = jax.random.fold_in(key, i)
        state, _ = step_fn(key, state)
    burnin_time = time.time() - start_time

    # Collect samples with progress bar
    print("Collecting samples...")
    start_time = time.time()
    for i in tqdm(range(n_samples), desc="Sampling", ncols=80):
        key = jax.random.fold_in(key, n_burnin + i)
        state, _ = step_fn(key, state)
        samples = samples.at[i].set(state.position)
    sampling_time = time.time() - start_time

    # Print timing info
    total_time = burnin_time + sampling_time
    samples_per_sec = n_samples / sampling_time
    print(f"\nTiming: {total_time:.2f}s total ({samples_per_sec:.1f} samples/sec)")

    # Plot results
    plot_samples(
        samples,
        "Normal Distribution - Direct BlackJAX MALA",
        "normal_blackjax_mala.png",
        (true_mean, true_scale),
    )


# %% [markdown]
"""
## Example 4: Univariate Normal with Direct BlackJAX NUTS

This example demonstrates NUTS sampling on a univariate (1D) normal distribution.
NUTS automatically tunes the HMC trajectory length, but can be memory-intensive.

Note: We use a 1D distribution and fewer samples to avoid memory issues. NUTS stores
trajectory information which scales with dimensionality and number of doublings.

The example includes error handling to gracefully handle potential memory constraints.
"""


# %%
def example_univariate_normal_nuts():
    """Sample from a univariate normal distribution using BlackJAX NUTS directly."""
    print()
    print("===== Example: Univariate Normal with Direct BlackJAX NUTS =====")
    print("Note: This may fail if memory is limited")

    # Create univariate normal distribution
    true_mean = jnp.array([2.0])
    true_scale = jnp.array([1.0])
    normal_dist = Normal(loc=true_mean, scale=true_scale)

    try:
        # Create log probability function that returns scalar
        def logdensity_fn(x):
            log_prob = normal_dist.log_prob(x)
            if hasattr(log_prob, "shape") and len(log_prob.shape) > 0:
                return jnp.sum(log_prob)
            return log_prob

        # Initial position and parameters
        init_position = jnp.zeros(1)
        step_size = 0.1
        inverse_mass_matrix = jnp.ones(1)

        # Create NUTS sampler
        nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)

        # Initialize sampler state
        state = nuts.init(init_position)

        # JIT compile the step function for speed
        step_fn = jax.jit(nuts.step)

        # Warm-up JIT compilation (don't count this in timing)
        key = jax.random.key(3)
        warmup_key = jax.random.fold_in(key, 999999)
        _, _ = step_fn(warmup_key, state)

        # Number of samples to collect
        n_samples = 5000  # Fewer samples for NUTS due to memory
        n_burnin = 1000

        # Collect samples
        samples = jnp.zeros((n_samples, 1))

        # Run burn-in with progress bar
        print("Running burn-in...")
        start_time = time.time()
        for i in tqdm(range(n_burnin), desc="Burn-in", ncols=80):
            key = jax.random.fold_in(key, i)
            state, _ = step_fn(key, state)
        burnin_time = time.time() - start_time

        # Collect samples with progress bar
        print("Collecting samples...")
        start_time = time.time()
        for i in tqdm(range(n_samples), desc="Sampling", ncols=80):
            key = jax.random.fold_in(key, n_burnin + i)
            state, _ = step_fn(key, state)
            samples = samples.at[i].set(state.position)
        sampling_time = time.time() - start_time

        # Print timing info
        total_time = burnin_time + sampling_time
        samples_per_sec = n_samples / sampling_time
        print(f"\nTiming: {total_time:.2f}s total ({samples_per_sec:.1f} samples/sec)")

        # Plot results
        plot_samples(
            samples,
            "Univariate Normal - Direct BlackJAX NUTS",
            "univariate_blackjax_nuts.png",
            (true_mean, true_scale),
        )
    except (RuntimeError, MemoryError) as e:
        print(f"NUTS sampling failed due to: {e}")
        print("This is likely due to memory constraints.")
        print("Try using HMC or MALA instead, or reduce parameters.")


# %% [markdown]
"""
## Example 5: Mixture of Gaussians with mala_sampling

This example demonstrates sampling from a mixture distribution using the functional API.
Mixture distributions are multimodal and can be challenging for MCMC samplers.

We create a mixture of two Gaussians with different weights, means, and scales, then
use MALA to sample from the mixture distribution. The Artifex Mixture class handles
the log probability calculation for us.
"""


# %%
def example_mixture_mala_function():
    """Sample from a mixture of Gaussians using mala_sampling function.

    This example demonstrates MALA's limitations with multimodal distributions.
    MALA is a local sampler that takes small gradient-guided steps, making it
    difficult to jump between distant modes separated by low-probability regions.
    """
    print()
    print("===== Example 5a: Mixture of Gaussians with MALA (Demonstrating Limitations) =====")

    # Use 1D mixture with widely separated modes
    weights = jnp.array([0.6, 0.4])
    means = jnp.array([[-2.0], [8.0]])  # 10 units apart - very distant
    scales = jnp.array([[0.8], [0.8]])

    components = [Normal(loc=means[0], scale=scales[0]), Normal(loc=means[1], scale=scales[1])]

    mixture = Mixture(components, weights)

    # Initial position
    init_position = jnp.array([-2.0])  # Start at first mode

    # Set sampling parameters
    key = jax.random.key(4)
    n_samples = 10000
    n_burnin = 5000

    # Warm-up run to compile JIT (don't count in timing)
    print("Warming up JIT compilation...")
    _ = mala_sampling(
        mixture,
        init_position,
        jax.random.key(999999),
        n_samples=10,
        n_burnin=5,
        step_size=0.05,
    )

    print(f"Sampling {n_samples} samples with {n_burnin} burn-in...")
    start_time = time.time()

    # Call the functional API (already JIT-compiled from warm-up)
    samples = mala_sampling(
        mixture,  # Directly pass the mixture distribution
        init_position,
        key,
        n_samples=n_samples,
        n_burnin=n_burnin,
        step_size=0.05,
    )

    total_time = time.time() - start_time
    samples_per_sec = n_samples / total_time
    print(f"Timing: {total_time:.2f}s total ({samples_per_sec:.1f} samples/sec)")

    # Print mixture information
    sep_dist = abs(float(means[1][0] - means[0][0]))
    m1, s1 = float(means[0][0]), float(scales[0][0])
    m2, s2 = float(means[1][0]), float(scales[1][0])
    print("\nMixture Distribution Info (1D):")
    print(f"  Component 1 (weight={weights[0]:.1f}): mean={m1:.1f}, scale={s1:.1f}")
    print(f"  Component 2 (weight={weights[1]:.1f}): mean={m2:.1f}, scale={s2:.1f}")
    print(f"  Separation: {sep_dist:.1f} units (widely separated)")
    print(f"  Step size: {0.05}")
    print("\n⚠️  Educational Note:")
    print("  MALA is a LOCAL sampler - it takes small gradient-guided steps.")
    print(f"  With step_size=0.05 and modes {sep_dist:.0f} units apart, MALA will")
    print(f"  get stuck in one mode (starting at {m1:.1f}).")
    print("  This demonstrates why algorithm choice matters for multimodal distributions!")
    print("  See Example 5b: NUTS handles MODERATE separation better.\n")

    # Calculate sample statistics
    sample_mean = jnp.mean(samples, axis=0)
    sample_std = jnp.std(samples, axis=0)

    # Check mode occupancy (1D)
    dist_to_mode1 = jnp.abs(samples[:, 0] - means[0][0])
    dist_to_mode2 = jnp.abs(samples[:, 0] - means[1][0])
    in_mode1 = (dist_to_mode1 < dist_to_mode2).sum()
    in_mode2 = (dist_to_mode2 <= dist_to_mode1).sum()
    pct_mode1 = 100 * in_mode1 / n_samples
    pct_mode2 = 100 * in_mode2 / n_samples

    print("\nSample Statistics:")
    print(f"  Sample mean: {float(sample_mean[0]):>7.4f}")
    print(f"  Sample std:  {float(sample_std[0]):>7.4f}")
    w1_pct = weights[0] * 100
    w2_pct = weights[1] * 100
    print(f"\nMode occupancy (should be [{w1_pct:.0f}%, {w2_pct:.0f}%]):")
    print(f"  Mode 1 at {m1:.1f}: {pct_mode1:.1f}% of samples")
    print(f"  Mode 2 at {m2:.1f}: {pct_mode2:.1f}% of samples")
    if pct_mode1 > 95 or pct_mode2 > 95:
        stuck_mode = 1 if pct_mode1 > 95 else 2
        print(f"  ✗ MALA stuck in mode {stuck_mode} - too distant for local sampler")
    else:
        print("  ⚠ Partial exploration - better than expected for MALA!")

    # Plot results (no true_mean for mixture)
    plot_samples(
        samples,
        "Example 5a: 1D Mixture with MALA (Wide Separation)",
        "mixture_mala_limitation.png",
        (None, None),  # Don't show "true" statistics for mixture
    )


# %% [markdown]
"""
## Example 5b: Mixture of Gaussians with NUTS (Better but Not Perfect)

Now let's demonstrate NUTS on a multimodal distribution with moderately-separated modes.
NUTS uses Hamiltonian dynamics for better exploration than MALA, but still faces
challenges with very distant modes due to energy conservation constraints.

This example uses **closer modes** (5.7 units apart) vs Example 5a (14.1 units),
showing that NUTS can handle moderate multimodality better than local samplers.
"""


# %%
def example_mixture_nuts_function():
    """Sample from a mixture of Gaussians using nuts_sampling function.

    This example demonstrates NUTS's improved (but not perfect) performance on
    moderately-separated multimodal distributions. NUTS uses Hamiltonian dynamics
    for better exploration than MALA, but energy constraints still limit mode-switching
    when modes are very far apart. For extreme multimodality, parallel tempering or
    SMC methods are needed.
    """
    print()
    print("===== Example 5b: Mixture with NUTS (Better Exploration, Closer Modes) =====")

    # Use 1D mixture for clearer demonstration
    # Note: Even NUTS struggles with very distant modes due to energy constraints
    weights = jnp.array([0.6, 0.4])
    means = jnp.array([[-2.0], [3.0]])  # 5 units apart in 1D
    scales = jnp.array([[0.8], [0.8]])

    components = [Normal(loc=means[0], scale=scales[0]), Normal(loc=means[1], scale=scales[1])]

    mixture = Mixture(components, weights)

    # Initial position
    init_position = jnp.array([-2.0])  # Start at first mode

    # Set sampling parameters
    key = jax.random.key(5)
    n_samples = 10000
    n_burnin = 5000

    # Warm-up run to compile JIT (don't count in timing)
    print("Warming up JIT compilation...")
    _ = nuts_sampling(
        mixture,
        init_position,
        jax.random.key(999999),
        n_samples=10,
        n_burnin=5,
        step_size=0.5,  # Initial step size; NUTS adapts this
    )

    print(f"Sampling {n_samples} samples with {n_burnin} burn-in...")
    print("(NUTS automatically adapts step size and trajectory length)")
    start_time = time.time()

    # Call the functional API (already JIT-compiled from warm-up)
    samples = nuts_sampling(
        mixture,  # Directly pass the mixture distribution
        init_position,
        key,
        n_samples=n_samples,
        n_burnin=n_burnin,
        step_size=0.5,  # Initial step size; NUTS adapts during burn-in
    )

    total_time = time.time() - start_time
    samples_per_sec = n_samples / total_time
    print(f"Timing: {total_time:.2f}s total ({samples_per_sec:.1f} samples/sec)")

    # Print mixture information
    sep_dist = abs(float(means[1][0] - means[0][0]))
    m1, s1 = float(means[0][0]), float(scales[0][0])
    m2, s2 = float(means[1][0]), float(scales[1][0])
    print("\nMixture Distribution Info (1D for clarity):")
    print(f"  Component 1 (weight={weights[0]:.1f}): mean={m1:.1f}, scale={s1:.1f}")
    print(f"  Component 2 (weight={weights[1]:.1f}): mean={m2:.1f}, scale={s2:.1f}")
    print(f"  Separation: {sep_dist:.1f} units (vs 14.1 in Example 5a)")
    print("\n✓ Educational Note:")
    print("  NUTS uses HAMILTONIAN DYNAMICS - momentum allows better exploration than MALA.")
    print("  With MODERATELY-separated modes (5 units), NUTS can transition between modes.")
    print("  However, even NUTS struggles with VERY distant modes due to energy constraints:")
    print("  Maximum potential energy increase is bounded by initial kinetic energy.")
    print("  For extreme multimodality: use parallel tempering, SMC, or tempered transitions.")
    print("  Compare with Example 5a: NUTS explores moderate separation better than MALA.\n")

    # Calculate sample statistics
    sample_mean = jnp.mean(samples, axis=0)
    sample_std = jnp.std(samples, axis=0)

    # Check mode occupancy (1D)
    dist_to_mode1 = jnp.abs(samples[:, 0] - means[0][0])
    dist_to_mode2 = jnp.abs(samples[:, 0] - means[1][0])
    # Assign each sample to nearest mode
    in_mode1 = (dist_to_mode1 < dist_to_mode2).sum()
    in_mode2 = (dist_to_mode2 <= dist_to_mode1).sum()
    pct_mode1 = 100 * in_mode1 / n_samples
    pct_mode2 = 100 * in_mode2 / n_samples

    print("\nSample Statistics:")
    print(f"  Sample mean: {float(sample_mean[0]):>7.4f}")
    print(f"  Sample std:  {float(sample_std[0]):>7.4f}")
    w1_pct = weights[0] * 100
    w2_pct = weights[1] * 100
    print(f"\nMode occupancy (should match weights [{w1_pct:.0f}%, {w2_pct:.0f}%]):")
    print(f"  Mode 1 at {m1:.1f}: {pct_mode1:.1f}% of samples")
    print(f"  Mode 2 at {m2:.1f}: {pct_mode2:.1f}% of samples")
    expected_pct1 = weights[0] * 100
    expected_pct2 = weights[1] * 100
    if abs(pct_mode1 - expected_pct1) < 15 and abs(pct_mode2 - expected_pct2) < 15:
        print("  ✓ NUTS successfully explored both modes!")
    elif pct_mode1 > 95 or pct_mode2 > 95:
        print("  ✗ NUTS got stuck in one mode - modes too far apart for energy constraints")
    else:
        print("  ⚠ Partial exploration - NUTS found both modes but proportions are off")

    # Plot results (no true_mean for mixture)
    plot_samples(
        samples,
        "Example 5b: 1D Mixture with NUTS (Moderate Separation)",
        "mixture_nuts_moderate.png",
        (None, None),  # Don't show "true" statistics for mixture
    )


# %% [markdown]
"""
## Running All Examples

Now let's run all integration examples to see both class-based and functional APIs
in action, including the multimodal distribution comparison.
"""


# %%
if __name__ == "__main__":
    print("===== Testing BlackJAX integration with Artifex distributions =====")

    # Run examples with BlackJAX Direct API
    example_normal_hmc()
    example_normal_mala()
    example_univariate_normal_nuts()

    # Run examples with Artifex Functional API
    example_normal_hmc_function()

    # Run multimodal distribution comparison
    example_mixture_mala_function()  # Example 5a: MALA struggles
    example_mixture_nuts_function()  # Example 5b: NUTS succeeds

    print()
    print("All examples complete!")

# %% [markdown]
"""
## Key Takeaways

After running this example, you should understand:

1. **Integration Approaches**:
   - Direct BlackJAX API: Maximum flexibility and control
   - Artifex Functional API: Simplified interface, recommended for most use cases

2. **Distribution Integration**: Artifex distributions work seamlessly with BlackJAX
   through their `log_prob` methods - just wrap them to return scalars

3. **Key BlackJAX Pattern**:
   - Create sampler: `sampler = blackjax.hmc(logdensity_fn, step_size, ...)`
   - Initialize state: `state = sampler.init(position)`
   - Step with random key: `state, info = sampler.step(key, state)`

4. **Memory Management**: NUTS requires careful memory management due to trajectory
   storage. Use lower dimensions and fewer samples when memory is limited

5. **Mixture Sampling**: Artifex's Mixture class enables easy MCMC sampling from
   multimodal distributions

6. **Random Key Management**: Always use `jax.random.fold_in()` to generate unique
   keys for each sampling step

## Experiments to Try

1. **Compare APIs**: Time class-based vs functional approaches for the same task
2. **Higher dimensions**: Try 5D or 10D normal distributions to see scalability
3. **Complex mixtures**: Create mixtures with 3+ components and varying weights
4. **Custom distributions**: Create custom distributions by defining log_prob functions
5. **Convergence diagnostics**: Implement R-hat and effective sample size calculations
6. **Adaptive tuning**: Experiment with different step sizes and integration steps

## Next Steps

- Review `blackjax_example.py` for simpler introductory examples
- Explore `blackjax_sampling_examples.py` for more sampling algorithms
- Read Artifex distributions documentation for available distributions
- Check BlackJAX documentation for additional sampler options
- Learn about advanced diagnostics for MCMC convergence assessment
"""
