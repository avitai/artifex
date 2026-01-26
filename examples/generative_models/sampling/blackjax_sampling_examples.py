# %% [markdown]
"""
# BlackJAX Sampling Examples for Generative Models

This example demonstrates comprehensive usage of BlackJAX samplers integrated with Artifex's
generative modeling framework.

## Learning Objectives

After completing this example, you will:

- [ ] Understand different MCMC sampling algorithms (HMC, MALA, NUTS)
- [ ] Learn to use Artifex's BlackJAX integration API
- [ ] Compare Artifex's sampler wrappers with direct BlackJAX usage
- [ ] Apply MCMC sampling to mixture distributions
- [ ] Visualize and interpret sampling results
- [ ] Handle memory constraints in NUTS sampling

## Prerequisites

- Understanding of MCMC sampling fundamentals
- Familiarity with probability distributions
- Basic knowledge of Hamiltonian Monte Carlo
- Artifex core distributions and sampling modules

## MCMC Algorithms Overview

This example demonstrates three state-of-the-art MCMC algorithms:

### HMC (Hamiltonian Monte Carlo)
- Uses gradient information to propose efficient moves
- Simulates Hamiltonian dynamics for exploration
- Requires tuning of step size and number of integration steps
- Excellent for smooth, continuous distributions

### MALA (Metropolis-Adjusted Langevin Algorithm)
- Gradient-based Metropolis method
- Uses Langevin dynamics for proposals
- Single step per iteration (faster than HMC)
- Good for smooth posteriors with strong gradients

### NUTS (No-U-Turn Sampler)
- Automatically tunes HMC trajectory length
- No manual tuning of integration steps needed
- Adaptive step size selection
- State-of-the-art for Bayesian inference

## Example Overview

This example includes:

1. **Artifex HMC**: Using Artifex's HMC wrapper on mixture distribution
2. **Artifex MALA**: Using Artifex's MALA wrapper on mixture distribution
3. **Artifex NUTS**: Using Artifex's NUTS wrapper on standard normal
4. **Direct BlackJAX**: Using BlackJAX HMC API directly for comparison

## Author

Artifex Team

## License

MIT License
"""

# %%
"""Import required libraries."""

from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from artifex.generative_models.core.distributions import Normal
from artifex.generative_models.core.sampling.blackjax_samplers import (
    hmc_sampling,
    mala_sampling,
    nuts_sampling,
)


# %%
"""Set up output directory and random seed."""

EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples_output"

# Set a random seed for reproducibility
key = jax.random.key(0)

# %% [markdown]
"""
## Helper Functions

We define helper functions to create target distributions and visualize sampling results.
"""


# %%
def create_mixture_logprob():
    """Create a simple 2D mixture of Gaussians log probability function.

    This function creates a mixture of two Gaussian distributions located
    at [3,3] and [-3,-3] with unit standard deviations.

    Returns:
        A function that computes the log probability of a point.
    """
    # Define two Gaussian components for the mixture
    mean1 = jnp.array([3.0, 3.0])
    mean2 = jnp.array([-3.0, -3.0])

    # Create log probability function
    def log_prob_fn(x):
        # Create two normal distributions
        dist1 = Normal(loc=mean1, scale=jnp.array([1.0, 1.0]))
        dist2 = Normal(loc=mean2, scale=jnp.array([1.0, 1.0]))

        # Sum the log probabilities for each component
        log_prob1 = jnp.sum(dist1.log_prob(x))
        log_prob2 = jnp.sum(dist2.log_prob(x))

        # Mixture log probability
        return jnp.logaddexp(log_prob1, log_prob2) - jnp.log(2.0)

    return log_prob_fn


# %%
def create_normal_logprob():
    """Create a simple 2D normal distribution.

    This function creates a standard normal distribution centered at origin.

    Returns:
        A function that computes the log probability of a point.
    """
    # Just a single normal distribution
    mean = jnp.array([0.0, 0.0])
    scale = jnp.array([1.0, 1.0])

    # Create log probability function
    def log_prob_fn(x):
        # Create a normal distribution
        dist = Normal(loc=mean, scale=scale)
        # Sum the log probabilities (ensure scalar output)
        return jnp.sum(dist.log_prob(x))

    return log_prob_fn


# %%
def plot_samples(samples, title="MCMC Samples", filename="mcmc_samples.png"):
    """Plot the sampling results.

    Args:
        samples: Array of samples with shape [n_samples, 2]
        title: Plot title
        filename: Filename to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    # Ensure output directory exists
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(EXAMPLES_DIR / filename)
    print(f"Plot saved as '{EXAMPLES_DIR / filename}'")

    # Print summary statistics
    mean = jnp.mean(samples, axis=0)
    std = jnp.std(samples, axis=0)

    print()
    print("Sample Statistics:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")


# %% [markdown]
"""
## Example 1: Artifex HMC Sampling

This example demonstrates using Artifex's HMC wrapper to sample from a bimodal mixture
of Gaussians. HMC uses Hamiltonian dynamics to propose moves, making it efficient for
exploring smooth, continuous distributions.

The mixture has two modes at [3,3] and [-3,-3], which tests the sampler's ability to
explore multiple modes.
"""


# %%
def example_hmc_artifex():
    """Example using artifex's HMC implementation."""
    print()
    print("===== Example: Artifex HMC Sampling =====")

    # Create log probability function
    mixture_logprob = create_mixture_logprob()

    # Initial state
    init_state = jnp.zeros(2)

    # Set parameters
    n_samples = 1000
    n_burnin = 500

    print(f"Running {n_samples} samples with {n_burnin} burn-in steps...")

    # Run HMC sampling
    key_hmc = jax.random.fold_in(key, 0)
    hmc_samples = hmc_sampling(
        mixture_logprob,
        init_state,
        key_hmc,
        n_samples=n_samples,
        n_burnin=n_burnin,
        step_size=0.1,
        num_integration_steps=10,
    )
    plot_samples(hmc_samples, "Artifex HMC Sampling", "hmc_artifex_samples.png")


# %% [markdown]
"""
## Example 2: Artifex MALA Sampling

MALA (Metropolis-Adjusted Langevin Algorithm) is a gradient-based sampler that uses
Langevin dynamics for proposals. It's faster than HMC per iteration but may require
more iterations to achieve the same effective sample size.

This example uses the same bimodal mixture to compare MALA's performance with HMC.
"""


# %%
def example_mala_artifex():
    """Example using artifex's MALA implementation."""
    print()
    print("===== Example: Artifex MALA Sampling =====")

    # Create log probability function
    mixture_logprob = create_mixture_logprob()

    # Initial state
    init_state = jnp.zeros(2)

    # Set parameters
    n_samples = 1000
    n_burnin = 500

    print(f"Running {n_samples} samples with {n_burnin} burn-in steps...")

    # Run MALA sampling
    key_mala = jax.random.fold_in(key, 1)
    mala_samples = mala_sampling(
        mixture_logprob,
        init_state,
        key_mala,
        n_samples=n_samples,
        n_burnin=n_burnin,
        step_size=0.05,
    )
    plot_samples(mala_samples, "Artifex MALA Sampling", "mala_artifex_samples.png")


# %% [markdown]
"""
## Example 3: Artifex NUTS Sampling

NUTS (No-U-Turn Sampler) automatically tunes the HMC trajectory length, eliminating the
need to manually set the number of integration steps. This makes it particularly useful
for complex, high-dimensional posteriors.

Note: This example uses a simpler standard normal distribution to avoid memory issues
that can occur with NUTS on mixture distributions. NUTS stores trajectory information
which can be memory-intensive.
"""


# %%
def example_nuts_artifex():
    """Example using artifex's NUTS implementation.

    Note: This example uses a simpler distribution to avoid memory issues.
    """
    print()
    print("===== Example: Artifex NUTS Sampling =====")

    # Create log probability function (using simpler distribution)
    simple_logprob = create_normal_logprob()

    # Initial state
    init_state = jnp.zeros(2)

    # Set parameters (use fewer samples to reduce memory)
    n_samples = 500
    n_burnin = 200

    print(f"Running {n_samples} samples with {n_burnin} burn-in steps...")

    # Run NUTS sampling with simple parameters
    key_nuts = jax.random.fold_in(key, 2)
    try:
        nuts_samples = nuts_sampling(
            simple_logprob,
            init_state,
            key_nuts,
            n_samples=n_samples,
            n_burnin=n_burnin,
            step_size=0.8,
            max_num_doublings=5,  # Lower value to reduce memory usage
        )
        plot_samples(nuts_samples, "Artifex NUTS Sampling", "nuts_artifex_samples.png")
    except Exception as e:
        print(f"NUTS sampling failed with error: {e}")
        print("This may be due to memory constraints on your system.")
        print("Try reducing n_samples, n_burnin, or max_num_doublings.")


# %% [markdown]
"""
## Example 4: Direct BlackJAX HMC

This example demonstrates using BlackJAX's HMC API directly, without Artifex's wrapper.
This is useful when you need more fine-grained control over the sampling process or want
to implement custom sampling logic.

Comparing this with Artifex's HMC wrapper shows how Artifex simplifies the API while
maintaining flexibility.
"""


# %%
def example_direct_blackjax_hmc():
    """Example using BlackJAX's HMC implementation directly."""
    print()
    print("===== Example: Direct BlackJAX HMC =====")

    # Create log probability function
    mixture_logprob = create_mixture_logprob()

    # Initial state
    init_state = jnp.zeros(2)

    # Set parameters
    n_samples = 1000
    n_burnin = 500
    step_size = 0.1
    num_integration_steps = 10

    print(f"Running {n_samples} samples with {n_burnin} burn-in steps...")

    # Initialize the HMC algorithm
    inverse_mass_matrix = jnp.eye(2)  # Identity matrix for 2D problem
    hmc = blackjax.hmc(
        mixture_logprob,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        num_integration_steps=num_integration_steps,
    )

    # Initialize the sampling state
    key_hmc = jax.random.fold_in(key, 3)
    initial_state = hmc.init(init_state)

    # Run sampling
    @nnx.jit
    def one_step(state, key):
        state, _ = hmc.step(key, state)
        return state, state

    # Burn-in
    key_hmc, subkey = jax.random.split(key_hmc)
    state = initial_state
    for _ in range(n_burnin):
        key_hmc, subkey = jax.random.split(key_hmc)
        state, _ = one_step(state, subkey)

    # Collect samples
    key_hmc, subkey = jax.random.split(key_hmc)
    state, samples = jax.lax.scan(one_step, state, jax.random.split(subkey, n_samples))
    samples = samples.position

    plot_samples(samples, "Direct BlackJAX HMC", "direct_blackjax_hmc_samples.png")


# %% [markdown]
"""
## Running All Examples

Now let's run all four examples to compare different sampling approaches.
"""


# %%
def run_all_examples():
    """Run all sampling examples."""
    print("Running BlackJAX sampling examples...")
    print()

    example_hmc_artifex()
    example_mala_artifex()
    example_nuts_artifex()
    example_direct_blackjax_hmc()

    print()
    print("All examples completed!")


# %%
if __name__ == "__main__":
    run_all_examples()

# %% [markdown]
"""
## Key Takeaways

After running this example, you should understand:

1. **Algorithm Selection**:
   - HMC: Best for smooth distributions, requires tuning
   - MALA: Faster per iteration, gradient-based
   - NUTS: Automatic tuning, excellent for complex posteriors

2. **Artifex Integration**: Artifex provides simple wrappers around BlackJAX that
   handle common use cases while maintaining access to advanced features

3. **Memory Considerations**: NUTS can be memory-intensive due to trajectory storage.
   Use `max_num_doublings` to control memory usage

4. **Direct API Access**: When needed, you can use BlackJAX directly for maximum control

## Experiments to Try

1. **Compare convergence**: Plot autocorrelation for each sampler to assess mixing
2. **Tune hyperparameters**: Experiment with step sizes and integration steps
3. **Higher dimensions**: Extend to 10D or 20D mixtures to see scalability
4. **Different targets**: Try non-Gaussian distributions (e.g., Rosenbrock, funnel)
5. **Acceptance rates**: Track and compare acceptance rates across samplers
6. **Effective sample size**: Compute ESS to measure sampling efficiency

## Next Steps

- Explore `blackjax_integration_examples.py` for advanced integration patterns
- Check out `blackjax_example.py` for simpler introductory examples
- Read BlackJAX documentation for more sampler options (e.g., SGLD, SGHMC)
- Learn about diagnostics (R-hat, ESS) for assessing convergence
"""
