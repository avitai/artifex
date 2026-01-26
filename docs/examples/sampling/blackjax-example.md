# BlackJAX Integration Example

![Level: Intermediate](https://img.shields.io/badge/Level-Intermediate-yellow)
![Runtime: 2-5 min](https://img.shields.io/badge/Runtime-2--5%20min-blue)
![Format: Dual](https://img.shields.io/badge/Format-Dual%20(.py%20%2B%20.ipynb)-green)

## Overview

This example demonstrates how to use BlackJAX samplers with Artifex's distribution
framework, comparing different MCMC algorithms on both multimodal distributions and
Bayesian regression tasks.

## Files

- Python script: [`examples/generative_models/sampling/blackjax_example.py`](https://github.com/avitai/artifex/examples/generative_models/sampling/blackjax_example.py)
- Jupyter notebook: [`examples/generative_models/sampling/blackjax_example.ipynb`](https://github.com/avitai/artifex/examples/generative_models/sampling/blackjax_example.ipynb)

## Quick Start

=== "Python Script"
    ```bash
    # Run the complete example
    python examples/generative_models/sampling/blackjax_example.py
    ```

=== "Jupyter Notebook"
    ```bash
    # Launch Jupyter and open the notebook
    jupyter notebook examples/generative_models/sampling/blackjax_example.ipynb
    ```

## Learning Objectives

After completing this example, you will:

- [x] Understand how to use BlackJAX samplers (HMC, NUTS, MALA) with Artifex
- [x] Learn to sample from multimodal distributions using different MCMC methods
- [x] Implement Bayesian regression using NUTS sampling
- [x] Compare different sampling algorithms for the same problem
- [x] Visualize and interpret MCMC sampling results

## Prerequisites

- Understanding of MCMC sampling concepts
- Basic knowledge of Bayesian inference
- Familiarity with probability distributions
- Artifex core sampling module

## What is BlackJAX?

[BlackJAX](https://blackjax-devs.github.io/blackjax/) is a library of samplers for JAX that provides
state-of-the-art MCMC algorithms. Artifex integrates BlackJAX to offer advanced sampling capabilities.

### Supported Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **HMC** | Hamiltonian Monte Carlo | Smooth, continuous distributions |
| **NUTS** | No-U-Turn Sampler | Complex posteriors, automatic tuning |
| **MALA** | Metropolis-Adjusted Langevin | Gradient-based sampling |

## Theory

### Hamiltonian Monte Carlo (HMC)

HMC uses Hamiltonian dynamics to propose efficient moves in the parameter space:

$$
H(q, p) = U(q) + K(p)
$$

where $U(q) = -\log p(q)$ is the potential energy and $K(p) = \frac{1}{2}p^T M^{-1} p$ is the kinetic energy.

The algorithm simulates Hamiltonian dynamics using the leapfrog integrator:

$$
\begin{align}
p_{i+\frac{1}{2}} &= p_i - \frac{\epsilon}{2} \nabla U(q_i) \\
q_{i+1} &= q_i + \epsilon M^{-1} p_{i+\frac{1}{2}} \\
p_{i+1} &= p_{i+\frac{1}{2}} - \frac{\epsilon}{2} \nabla U(q_{i+1})
\end{align}
$$

### No-U-Turn Sampler (NUTS)

NUTS automatically tunes the HMC trajectory length by building a tree of states until
the trajectory makes a "U-turn". This eliminates the need to manually set the number
of integration steps.

### MALA (Metropolis-Adjusted Langevin Algorithm)

MALA uses Langevin dynamics for proposals:

$$
\theta' = \theta + \frac{\epsilon^2}{2} \nabla \log p(\theta) + \epsilon \eta
$$

where $\eta \sim \mathcal{N}(0, I)$ is Gaussian noise.

## Code Walkthrough

### Example 1: Multimodal Distribution Sampling

The first example compares four sampling methods on a bimodal Gaussian mixture:

```python
# Define a bimodal log probability function
def log_prob_fn(x):
    log_prob1 = -0.5 * ((x - 2.0) ** 2) / 0.5
    log_prob2 = -0.5 * ((x + 2.0) ** 2) / 0.5
    return jnp.logaddexp(log_prob1, log_prob2)

# Sample using Metropolis-Hastings
mh_samples = mcmc_sampling(
    log_prob_fn=log_prob_fn,
    init_state=init_state,
    key=key,
    n_samples=2000,
    n_burnin=500,
    step_size=0.5,
)

# Sample using HMC (BlackJAX)
hmc_samples = hmc_sampling(
    log_prob_fn=log_prob_fn,
    init_state=init_state,
    key=key,
    n_samples=2000,
    n_burnin=500,
    step_size=0.1,
    num_integration_steps=10,
)
```

**Key Points:**

- The bimodal distribution has two modes at $x = -2$ and $x = 2$
- HMC uses gradient information for more efficient exploration
- NUTS automatically tunes trajectory length
- MALA balances speed and efficiency

### Example 2: Bayesian Linear Regression

The second example demonstrates Bayesian parameter estimation:

```python
# Define Bayesian regression model
def log_prob_fn(params):
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

# Sample using NUTS
nuts_samples = nuts_sampling(
    log_prob_fn=log_prob_fn,
    init_state=init_state,
    key=key,
    n_samples=2000,
    n_burnin=1000,
)
```

**Key Points:**

- NUTS is ideal for Bayesian inference with multiple parameters
- The model includes priors on coefficients and noise scale
- Posterior distributions recover true parameter values
- No manual tuning of trajectory length needed

## Expected Output

### Multimodal Distribution

The example generates comparison plots showing samples from each method:

- **Metropolis-Hastings**: Baseline performance
- **HMC**: Efficient exploration with gradient information
- **NUTS**: Similar to HMC but with automatic tuning
- **MALA**: Fast per-iteration sampling

You should see all methods successfully sample from both modes of the distribution.

### Bayesian Regression

The regression example produces:

1. **Coefficient posteriors**: Distributions for each $\beta_i$ parameter
2. **Noise posterior**: Distribution for noise scale $\sigma$
3. **Comparison with truth**: True values marked on plots

Expected results:

- Posterior means close to true parameter values
- Reasonable posterior uncertainty
- Successful convergence after burn-in

## Performance Considerations

### Computational Cost

| Method | Time per Sample | Effective Sample Size (ESS) | Overall Efficiency |
|--------|----------------|----------------------------|-------------------|
| MH | Low | Low | Baseline |
| HMC | Medium | High | Good |
| NUTS | High | Very High | Excellent |
| MALA | Low-Medium | Medium | Good |

### Memory Usage

- **HMC/MALA**: Moderate memory usage
- **NUTS**: Higher memory usage due to trajectory storage
- For memory-constrained systems, reduce `max_num_doublings` in NUTS

### Tuning Guidelines

**HMC:**

- `step_size`: Start with 0.1, adjust based on acceptance rate (target: 0.6-0.8)
- `num_integration_steps`: Start with 10, increase for complex distributions

**MALA:**

- `step_size`: Start with 0.05, adjust based on acceptance rate (target: 0.5-0.7)

**NUTS:**

- `step_size`: Usually auto-tuned, but can be set manually
- `max_num_doublings`: Controls trajectory length and memory usage (default: 10)

## Experiments to Try

1. **Change the distribution**: Modify the bimodal distribution to have three modes
   or varying widths

2. **Tune hyperparameters**: Experiment with different step sizes and integration
   steps, observing effects on acceptance rate and mixing

3. **Compare convergence**: Plot traces and autocorrelation to assess convergence
   and mixing for each method

4. **Higher dimensions**: Extend Bayesian regression to 20-50 features to see
   how samplers scale

5. **Different priors**: Try informative vs uninformative priors on regression
   coefficients

6. **Visualize traces**: Add MCMC trace plots to check convergence

## Troubleshooting

### Low Acceptance Rate

**Symptom**: Acceptance rate below 0.5 for HMC/MALA

**Solution**:

- Decrease `step_size` for HMC/MALA
- Check gradient computation (should not have NaNs)
- Verify log probability function is correct

### Poor Mixing

**Symptom**: Samples stay in one mode of multimodal distribution

**Solution**:

- Increase burn-in period
- Try different initialization points
- Consider tempering or parallel chains

### NUTS Memory Errors

**Symptom**: Out of memory errors with NUTS

**Solution**:

```python
# Reduce max_num_doublings
nuts_samples = nuts_sampling(
    log_prob_fn=log_prob_fn,
    init_state=init_state,
    key=key,
    n_samples=1000,  # Reduce number of samples
    n_burnin=500,
    max_num_doublings=5,  # Lower from default 10
)
```

### Divergent Transitions (NUTS)

**Symptom**: Warning about divergent transitions

**Solution**:

- Decrease `step_size`
- Reparameterize the model (e.g., use non-centered parameterization)
- Check for prior-likelihood conflicts

## Next Steps

### Related Examples

<div class="grid cards" markdown>

- **BlackJAX Sampling Examples**

    Explore more sampling algorithms and advanced usage patterns

    [blackjax-sampling-examples.md](blackjax-sampling-examples.md)

- **BlackJAX Integration Examples**

    Learn advanced integration with Artifex distributions

    [blackjax-integration-examples.md](blackjax-integration-examples.md)

</div>

### Further Learning

- [BlackJAX Documentation](https://blackjax-devs.github.io/blackjax/)
- [MCMC Diagnostics](https://mc-stan.org/docs/reference-manual/mcmc-diagnostics.html)
- [HMC Tutorial](https://arxiv.org/abs/1701.02434)
- Artifex Sampling Module Documentation

## Additional Resources

### Papers

1. Neal, R. M. (2011). "MCMC using Hamiltonian dynamics"
2. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler"
3. Roberts, G. O., & Tweedie, R. L. (1996). "Exponential convergence of Langevin
   distributions and their discrete approximations"

### Code References

- **Distribution creation**: `artifex.generative_models.core.distributions`
- **Sampling functions**: `artifex.generative_models.core.sampling`
- **BlackJAX integration**: `artifex.generative_models.core.sampling.blackjax_samplers`

## Support

If you encounter issues:

1. Check that BlackJAX is installed: `pip install blackjax`
2. Verify JAX GPU/CPU setup is correct
3. Review error messages for parameter constraints
4. Consult Artifex documentation or open an issue

---

**Tags:** #mcmc #blackjax #hmc #nuts #mala #bayesian #sampling

**Difficulty:** Intermediate

**Estimated Time:** 15-20 minutes
