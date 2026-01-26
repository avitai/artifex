# %% [markdown]
"""
# BlackJAX Integration Example

This example demonstrates how to use BlackJAX samplers with Artifex's distribution framework.

## Learning Objectives

After completing this example, you will:

- [ ] Understand how to use BlackJAX samplers (HMC, NUTS, MALA) with Artifex
- [ ] Learn to sample from multimodal distributions using different MCMC methods
- [ ] Implement Bayesian regression using NUTS sampling
- [ ] Compare different sampling algorithms for the same problem
- [ ] Visualize and interpret MCMC sampling results

## Prerequisites

- Understanding of MCMC sampling concepts
- Basic knowledge of Bayesian inference
- Familiarity with probability distributions
- Artifex core sampling module

## What is BlackJAX?

BlackJAX is a library of samplers for JAX that provides state-of-the-art MCMC algorithms.
Artifex integrates BlackJAX to offer advanced sampling capabilities including:

- **HMC (Hamiltonian Monte Carlo)**: Uses gradient information for efficient sampling
- **NUTS (No-U-Turn Sampler)**: Automatically tunes HMC step size and trajectory length
- **MALA (Metropolis-Adjusted Langevin Algorithm)**: Gradient-based Metropolis method

These algorithms are particularly useful for:
- Sampling from complex, high-dimensional distributions
- Bayesian inference and parameter estimation
- Comparing sampling efficiency across methods

## Example Overview

This example includes two demonstrations:

1. **Multimodal Distribution Sampling**: Compare different samplers on a bimodal Gaussian mixture
2. **Bayesian Regression**: Use NUTS to infer parameters in a linear regression model

## Author

Artifex Team

## License

MIT License
"""

# %%
"""Import required libraries."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from artifex.generative_models.core.sampling import (
    hmc_sampling,
    mala_sampling,
    mcmc_sampling,
    nuts_sampling,
)


# %% [markdown]
"""
## Multimodal Distribution Example

This example demonstrates sampling from a bimodal distribution using different MCMC methods.
We'll compare the performance of:

- Metropolis-Hastings (our basic implementation)
- HMC (BlackJAX)
- NUTS (BlackJAX)
- MALA (BlackJAX)

The target distribution is a mixture of two Gaussians centered at x=-2 and x=+2.
"""


# %%
def multimodal_distribution_example():
    """Example with a multimodal distribution."""
    print("Running multimodal distribution example...")

    # Define a multi-modal log probability function (mixture of two Gaussians)
    def log_prob_fn(x):
        # Create bimodal distribution (mixture of Gaussians)
        log_prob1 = -0.5 * ((x - 2.0) ** 2) / 0.5
        log_prob2 = -0.5 * ((x + 2.0) ** 2) / 0.5
        return jnp.logaddexp(log_prob1, log_prob2)

    # Initial state and key
    init_state = jnp.array(0.0)
    key = jax.random.key(0)

    # Sample using different methods
    n_samples = 2000

    # Regular Metropolis-Hastings (our implementation)
    mh_samples = mcmc_sampling(
        log_prob_fn=log_prob_fn,
        init_state=init_state,
        key=key,
        n_samples=n_samples,
        n_burnin=500,
        step_size=0.5,
    )

    # HMC sampling (BlackJAX)
    hmc_samples = hmc_sampling(
        log_prob_fn=log_prob_fn,
        init_state=init_state,
        key=key,
        n_samples=n_samples,
        n_burnin=500,
        step_size=0.1,
        num_integration_steps=10,
    )

    # NUTS sampling (BlackJAX)
    nuts_samples = nuts_sampling(
        log_prob_fn=log_prob_fn,
        init_state=init_state,
        key=key,
        n_samples=n_samples,
        n_burnin=500,
    )

    # MALA sampling (BlackJAX)
    mala_samples = mala_sampling(
        log_prob_fn=log_prob_fn,
        init_state=init_state,
        key=key,
        n_samples=n_samples,
        n_burnin=500,
        step_size=0.1,
    )

    # Plot histograms of samples
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.hist(mh_samples, bins=50, alpha=0.7, density=True)
    plt.title("Metropolis-Hastings")
    plt.xlabel("x")
    plt.ylabel("Density")

    plt.subplot(2, 2, 2)
    plt.hist(hmc_samples, bins=50, alpha=0.7, density=True)
    plt.title("Hamiltonian Monte Carlo (BlackJAX)")
    plt.xlabel("x")
    plt.ylabel("Density")

    plt.subplot(2, 2, 3)
    plt.hist(nuts_samples, bins=50, alpha=0.7, density=True)
    plt.title("NUTS (BlackJAX)")
    plt.xlabel("x")
    plt.ylabel("Density")

    plt.subplot(2, 2, 4)
    plt.hist(mala_samples, bins=50, alpha=0.7, density=True)
    plt.title("MALA (BlackJAX)")
    plt.xlabel("x")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig("blackjax_multimodal_comparison.png")
    plt.close()

    print("Multimodal sampling complete. Results saved to blackjax_multimodal_comparison.png")


# %% [markdown]
"""
## Bayesian Regression Example

This example demonstrates using NUTS to perform Bayesian linear regression.
We'll:

1. Generate synthetic data with known parameters
2. Define a Bayesian model with priors on coefficients and noise
3. Use NUTS to sample from the posterior distribution
4. Visualize the posterior distributions and compare to true values

NUTS is particularly well-suited for Bayesian regression because it:
- Efficiently explores high-dimensional parameter spaces
- Automatically adapts step size and trajectory length
- Provides robust sampling without manual tuning
"""


# %%
def bayesian_regression_example():
    """Example with Bayesian regression."""
    print("Running Bayesian regression example...")

    # Generate synthetic data
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    n_samples = 100
    n_features = 5

    # True parameters
    true_beta = jnp.array([0.5, -0.3, 0.8, -0.2, 0.4])
    noise_scale = 0.1

    # Generate features
    X = jax.random.normal(key, (n_samples, n_features))

    # Generate target with noise
    key, subkey = jax.random.split(key)
    y = X @ true_beta + noise_scale * jax.random.normal(subkey, (n_samples,))

    # Define log probability function for Bayesian regression
    def log_prob_fn(params):
        # Unpack parameters
        beta = params["beta"]
        log_sigma = params["log_sigma"]
        sigma = jnp.exp(log_sigma)

        # Prior
        prior_beta = jnp.sum(jax.scipy.stats.norm.logpdf(beta, loc=0.0, scale=1.0))
        prior_sigma = jax.scipy.stats.norm.logpdf(log_sigma, loc=-2.0, scale=1.0)

        # Likelihood
        y_pred = X @ beta
        likelihood = jnp.sum(jax.scipy.stats.norm.logpdf(y, loc=y_pred, scale=sigma))

        return prior_beta + prior_sigma + likelihood

    # Initial state
    init_state = {
        "beta": jnp.zeros(n_features),
        "log_sigma": jnp.array(0.0),
    }

    # Sample using NUTS (the best choice for this problem)
    key = jax.random.key(1)  # New key for sampling
    nuts_samples = nuts_sampling(
        log_prob_fn=log_prob_fn,
        init_state=init_state,
        key=key,
        n_samples=2000,
        n_burnin=1000,
    )

    # Extract samples
    beta_samples = nuts_samples["beta"]
    sigma_samples = jnp.exp(nuts_samples["log_sigma"])

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot coefficient distributions
    plt.subplot(1, 2, 1)
    for i in range(n_features):
        beta_label = f"$\\beta_{i}$ (true: {true_beta[i]:.2f})"
        plt.hist(beta_samples[:, i], bins=30, alpha=0.6, label=beta_label)
    plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    plt.title("Posterior Distributions of Coefficients")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    # Plot sigma distribution
    plt.subplot(1, 2, 2)
    plt.hist(sigma_samples, bins=30, alpha=0.6)
    sigma_label = f"True $\\sigma$: {noise_scale:.2f}"
    plt.axvline(x=noise_scale, color="r", linestyle="--", label=sigma_label)
    plt.title("Posterior Distribution of Noise Scale ($\\sigma$)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig("blackjax_regression_example.png")
    plt.close()

    print("Bayesian regression complete. Results saved to blackjax_regression_example.png")


# %% [markdown]
"""
## Running the Examples

Now let's run both examples to see the different sampling methods in action.
"""


# %%
def main():
    """Run the examples."""
    multimodal_distribution_example()
    print()
    print("-" * 50)
    print()
    bayesian_regression_example()


# %%
if __name__ == "__main__":
    main()

# %% [markdown]
"""
## Key Takeaways

After running this example, you should understand:

1. **BlackJAX Integration**: Artifex provides seamless integration with BlackJAX samplers
2. **Sampler Selection**: Different samplers have different strengths:
   - NUTS: Best for complex posteriors, automatic tuning
   - HMC: Good balance of efficiency and control
   - MALA: Gradient-based Metropolis for smooth distributions
   - MH: Simple baseline, useful for comparison
3. **Bayesian Inference**: NUTS excels at Bayesian parameter estimation
4. **Multimodal Distributions**: All samplers can handle multimodal targets, but with
   varying efficiency

## Experiments to Try

1. **Change the distribution**: Modify the bimodal distribution to have three modes or
   varying widths
2. **Tune hyperparameters**: Experiment with different step sizes, integration steps,
   and burn-in periods
3. **Compare convergence**: Track acceptance rates and effective sample sizes across
   methods
4. **Higher dimensions**: Extend the Bayesian regression to more features
   (e.g., 20-50 dimensions)
5. **Different priors**: Try different prior distributions on the regression
   coefficients
6. **Visualize traces**: Plot MCMC traces to check for convergence and mixing

## Next Steps

- Explore `blackjax_sampling_examples.py` for more sampling algorithms
- Learn about advanced diagnostics in `blackjax_integration_examples.py`
- Check out the full BlackJAX documentation for more sampler options
"""
