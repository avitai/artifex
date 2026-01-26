# BlackJAX Sampling Examples

![Level: Advanced](https://img.shields.io/badge/Level-Advanced-red)
![Runtime: 5-10 min](https://img.shields.io/badge/Runtime-5--10%20min-blue)
![Format: Dual](https://img.shields.io/badge/Format-Dual%20(.py%20%2B%20.ipynb)-green)

## Overview

This example provides a comprehensive exploration of BlackJAX samplers integrated with Artifex's distribution framework. It compares four different approaches to MCMC sampling: Artifex's HMC wrapper, Artifex's MALA wrapper, Artifex's NUTS wrapper, and direct BlackJAX API usage.

## Files

- Python script: [`examples/generative_models/sampling/blackjax_sampling_examples.py`](https://github.com/avitai/artifex/examples/generative_models/sampling/blackjax_sampling_examples.py)
- Jupyter notebook: [`examples/generative_models/sampling/blackjax_sampling_examples.ipynb`](https://github.com/avitai/artifex/examples/generative_models/sampling/blackjax_sampling_examples.ipynb)

## Quick Start

=== "Python Script"
    ```bash
    # Run the complete example
    python examples/generative_models/sampling/blackjax_sampling_examples.py
    ```

=== "Jupyter Notebook"
    ```bash
    # Launch Jupyter and open the notebook
    jupyter notebook examples/generative_models/sampling/blackjax_sampling_examples.ipynb
    ```

## Learning Objectives

After completing this example, you will:

- [x] Understand different MCMC sampling algorithms (HMC, MALA, NUTS)
- [x] Learn to use Artifex's BlackJAX integration API
- [x] Compare Artifex's sampler wrappers with direct BlackJAX usage
- [x] Apply MCMC sampling to mixture distributions
- [x] Visualize and interpret sampling results
- [x] Handle memory constraints in NUTS sampling

## Prerequisites

- Understanding of MCMC sampling fundamentals
- Familiarity with probability distributions
- Basic knowledge of Hamiltonian Monte Carlo
- Completion of [BlackJAX Integration Example](blackjax-example.md)
- Artifex core sampling module

## MCMC Algorithms Overview

This example demonstrates three state-of-the-art MCMC algorithms from the BlackJAX library.

### Hamiltonian Monte Carlo (HMC)

HMC uses gradient information to propose efficient moves in parameter space by simulating Hamiltonian dynamics.

**Key Characteristics:**

- Uses gradient information for exploration
- Requires tuning of step size and integration steps
- Excellent for smooth, continuous distributions
- Higher computational cost per iteration than MH

**Mathematical Formulation:**

The Hamiltonian system is defined as:
$$
H(q, p) = U(q) + K(p)
$$

where:

- $U(q) = -\log p(q)$ is the potential energy
- $K(p) = \frac{1}{2}p^T M^{-1} p$ is the kinetic energy
- $M$ is the mass matrix

The leapfrog integrator updates positions and momenta:

$$
\begin{align}
p_{i+\frac{1}{2}} &= p_i - \frac{\epsilon}{2} \nabla U(q_i) \\
q_{i+1} &= q_i + \epsilon M^{-1} p_{i+\frac{1}{2}} \\
p_{i+1} &= p_{i+\frac{1}{2}} - \frac{\epsilon}{2} \nabla U(q_{i+1})
\end{align}
$$

### Metropolis-Adjusted Langevin Algorithm (MALA)

MALA is a gradient-based Metropolis method that uses Langevin dynamics for proposals.

**Key Characteristics:**

- Single step per iteration (faster than HMC)
- Gradient-based proposals for efficiency
- Good for smooth posteriors with strong gradients
- Lower acceptance rate than HMC typically

**Mathematical Formulation:**

The proposal distribution is:

$$
\theta' = \theta + \frac{\epsilon^2}{2} \nabla \log p(\theta) + \epsilon \eta
$$

where $\eta \sim \mathcal{N}(0, I)$ is Gaussian noise.

The acceptance probability follows the Metropolis-Hastings rule:

$$
\alpha(\theta' | \theta) = \min\left(1, \frac{p(\theta') q(\theta | \theta')}{p(\theta) q(\theta' | \theta)}\right)
$$

### No-U-Turn Sampler (NUTS)

NUTS automatically tunes the HMC trajectory length by building a tree of states until the trajectory makes a "U-turn".

**Key Characteristics:**

- No manual tuning of integration steps needed
- Adaptive step size selection
- State-of-the-art for Bayesian inference
- Higher memory usage due to trajectory storage
- Excellent for complex, high-dimensional posteriors

**Algorithm Overview:**

NUTS builds a balanced binary tree of trajectory states by recursively doubling until:

1. The trajectory makes a U-turn (forward/backward directions oppose)
2. Maximum tree depth is reached (`max_num_doublings`)

The U-turn criterion is:

$$
(\theta^+ - \theta^-) \cdot p^- < 0 \quad \text{or} \quad (\theta^+ - \theta^-) \cdot p^+ < 0
$$

where $\theta^+, p^+$ are the forward endpoint and $\theta^-, p^-$ are the backward endpoint.

## Code Walkthrough

### Example 1: Artifex HMC Sampling

This example uses Artifex's HMC wrapper to sample from a bimodal mixture of Gaussians:

```python
# Create a 2D mixture of Gaussians
def create_mixture_logprob():
    mean1 = jnp.array([3.0, 3.0])
    mean2 = jnp.array([-3.0, -3.0])

    def log_prob_fn(x):
        dist1 = Normal(loc=mean1, scale=jnp.array([1.0, 1.0]))
        dist2 = Normal(loc=mean2, scale=jnp.array([1.0, 1.0]))

        log_prob1 = jnp.sum(dist1.log_prob(x))
        log_prob2 = jnp.sum(dist2.log_prob(x))

        # Equal mixture weights
        return jnp.logaddexp(log_prob1, log_prob2) - jnp.log(2.0)

    return log_prob_fn

# Sample using Artifex's HMC wrapper
mixture_logprob = create_mixture_logprob()
init_state = jnp.zeros(2)

hmc_samples = hmc_sampling(
    mixture_logprob,
    init_state,
    key,
    n_samples=1000,
    n_burnin=500,
    step_size=0.1,
    num_integration_steps=10,
)
```

**Key Points:**

- The mixture has two well-separated modes at [3, 3] and [-3, -3]
- HMC explores both modes efficiently using gradient information
- Artifex wrapper handles initialization and sampling loop
- Returns array of samples with shape `[n_samples, 2]`

### Example 2: Artifex MALA Sampling

This example demonstrates MALA on the same bimodal distribution:

```python
mala_samples = mala_sampling(
    mixture_logprob,
    init_state,
    key,
    n_samples=1000,
    n_burnin=500,
    step_size=0.05,  # Smaller step size than HMC
)
```

**Key Points:**

- MALA uses smaller step sizes than HMC (typically 0.05 vs 0.1)
- Single Langevin step per iteration makes it faster per sample
- May need more samples to achieve same effective sample size as HMC
- Good for problems where gradient evaluation is cheap

### Example 3: Artifex NUTS Sampling

NUTS automatically tunes trajectory length, eliminating manual tuning:

```python
# Use simpler distribution to avoid memory issues
simple_logprob = create_normal_logprob()

nuts_samples = nuts_sampling(
    simple_logprob,
    init_state,
    key,
    n_samples=500,  # Fewer samples due to memory
    n_burnin=200,
    step_size=0.8,
    max_num_doublings=5,  # Control memory usage
)
```

**Key Points:**

- NUTS is memory-intensive due to trajectory tree storage
- Use `max_num_doublings` to control memory usage (default: 10)
- Excellent for complex posteriors where tuning is difficult
- This example uses a simpler distribution to demonstrate the API

### Example 4: Direct BlackJAX HMC

This example shows how to use BlackJAX's API directly without Artifex wrappers:

```python
import blackjax

# Initialize the HMC algorithm
inverse_mass_matrix = jnp.eye(2)
hmc = blackjax.hmc(
    mixture_logprob,
    step_size=0.1,
    inverse_mass_matrix=inverse_mass_matrix,
    num_integration_steps=10,
)

# Initialize sampling state
initial_state = hmc.init(init_state)

# Define one step function
@nnx.jit
def one_step(state, key):
    state, _ = hmc.step(key, state)
    return state, state

# Burn-in phase
state = initial_state
for _ in range(n_burnin):
    key, subkey = jax.random.split(key)
    state, _ = one_step(state, subkey)

# Collect samples
key, subkey = jax.random.split(key)
state, samples = jax.lax.scan(
    one_step,
    state,
    jax.random.split(subkey, n_samples)
)
samples = samples.position
```

**Key Points:**

- Direct API provides fine-grained control over sampling
- Must manually manage state and random keys
- Use `jax.lax.scan` for efficient sample collection
- JIT compilation improves performance significantly
- Useful when implementing custom sampling logic

## Expected Output

### Sample Plots

Each example generates a scatter plot showing the samples in 2D space:

- **HMC samples**: Should show clear exploration of both modes
- **MALA samples**: Similar coverage but potentially more concentrated
- **NUTS samples**: For the normal distribution, centered at origin
- **Direct API samples**: Should match Artifex HMC results

### Statistics

Each example prints sample statistics:

```
Sample Statistics:
Mean: [ 0.1234 -0.2345]
Std: [2.9876  2.8765]
```

For the bimodal mixture, expect:

- Mean near [0, 0] (average of two modes)
- Large standard deviation (reflecting mode separation)

## Performance Comparison

### Computational Cost

| Method | Time per Sample | ESS per Sample | Memory Usage | Tuning Required |
|--------|----------------|----------------|--------------|-----------------|
| HMC | Medium | High | Low | Yes (step size, steps) |
| MALA | Low | Medium | Low | Yes (step size) |
| NUTS | High | Very High | High | Minimal (auto-tuning) |
| Direct API | Medium | High | Low | Yes (same as HMC) |

### When to Use Each Method

**Use HMC when:**

- You have smooth, continuous target distributions
- You can afford moderate computational cost
- You want efficient exploration with gradients

**Use MALA when:**

- Gradient evaluation is cheap
- You need many samples quickly
- Target has strong gradients

**Use NUTS when:**

- You have complex, high-dimensional posteriors
- You can afford higher memory usage
- You want to avoid manual tuning
- You need robust inference

**Use Direct API when:**

- You need custom sampling logic
- You want fine-grained control
- You're implementing advanced algorithms
- Artifex wrappers don't fit your use case

## Tuning Guidelines

### HMC Tuning

**Step Size (`step_size`):**

- Start with 0.1
- Target acceptance rate: 0.6-0.8
- Too large: low acceptance rate
- Too small: slow mixing

**Integration Steps (`num_integration_steps`):**

- Start with 10
- Increase for better exploration
- Higher values increase cost per sample

### MALA Tuning

**Step Size (`step_size`):**

- Start with 0.05 (smaller than HMC)
- Target acceptance rate: 0.5-0.7
- Adjust based on acceptance diagnostics

### NUTS Tuning

**Step Size (`step_size`):**

- Often auto-tuned during warmup
- Can set manually if needed
- Usually between 0.1-1.0

**Max Doublings (`max_num_doublings`):**

- Controls trajectory length and memory
- Default: 10 (max trajectory length = 2^10 = 1024)
- Reduce if encountering memory errors
- Values 5-7 often sufficient

## Experiments to Try

1. **Compare mixing**: Plot trace plots and autocorrelation for each sampler to assess mixing quality

2. **Tune hyperparameters**: Systematically vary step sizes and integration steps, tracking acceptance rates and ESS

3. **Higher dimensions**: Extend the mixture to 10D or 20D to see how samplers scale

4. **Different targets**: Try non-Gaussian distributions like:
   - Rosenbrock's banana-shaped distribution
   - Neal's funnel distribution
   - Mixture of many components

5. **Effective sample size**: Compute ESS using `arviz` or similar tools to measure sampling efficiency

6. **Warmup strategies**: Experiment with different warmup lengths and adaptive schemes

7. **Parallel chains**: Run multiple chains and assess convergence using R-hat

## Troubleshooting

### Low Acceptance Rate

**Symptom**: Acceptance rate below 0.5

**Solution**:

- Reduce `step_size` by factor of 2
- Check gradient computation (no NaNs)
- Verify log probability is correct
- Try simpler test distribution first

### Poor Mixing

**Symptom**: Samples stuck in one mode of multimodal distribution

**Solution**:

- Increase burn-in period (try 2x-5x current)
- Try different initialization points
- Consider parallel tempering for multimodal targets
- Increase `num_integration_steps` for HMC

### NUTS Memory Errors

**Symptom**: Out of memory errors with NUTS

**Solution**:

```python
# Reduce memory usage
nuts_samples = nuts_sampling(
    log_prob_fn,
    init_state,
    key,
    n_samples=500,  # Reduce sample count
    n_burnin=200,
    max_num_doublings=5,  # Lower from default 10
)
```

### Divergent Transitions (NUTS)

**Symptom**: Warning about divergent transitions

**Solution**:

- Decrease `step_size` (try 0.5x current)
- Reparameterize the model (e.g., non-centered parameterization)
- Check for prior-likelihood conflicts
- Increase warmup period

### Slow Performance

**Symptom**: Sampling taking too long

**Solution**:

- Ensure JIT compilation is used (`@nnx.jit` or `@jax.jit`)
- Check if GPU is available and being used
- Use Direct API with `jax.lax.scan` for efficient loops
- Reduce sample count for testing

## Next Steps

### Related Examples

<div class="grid cards" markdown>

- **BlackJAX Integration Example**

    Learn the basics of BlackJAX with Artifex

    [blackjax-example.md](blackjax-example.md)

- **BlackJAX Integration Examples**

    Advanced integration patterns and production use cases

    [blackjax-integration-examples.md](blackjax-integration-examples.md)

</div>

### Further Learning

- [BlackJAX Documentation](https://blackjax-devs.github.io/blackjax/)
- [BlackJAX Sampling Book](https://blackjax-devs.github.io/sampling-book/)
- [MCMC Diagnostics Guide](https://mc-stan.org/docs/reference-manual/mcmc-diagnostics.html)
- [HMC Tutorial by Betancourt](https://arxiv.org/abs/1701.02434)
- [NUTS Paper](https://arxiv.org/abs/1111.4246)
- Artifex Sampling Module Documentation

## Additional Resources

### Papers

1. **HMC**: Neal, R. M. (2011). "MCMC using Hamiltonian dynamics". *Handbook of Markov Chain Monte Carlo*.

2. **NUTS**: Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo". *Journal of Machine Learning Research*.

3. **MALA**: Roberts, G. O., & Tweedie, R. L. (1996). "Exponential convergence of Langevin distributions and their discrete approximations". *Bernoulli*.

4. **Convergence Diagnostics**: Vehtari, A., et al. (2021). "Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC". *Bayesian Analysis*.

### Code References

- **Distribution creation**: `artifex.generative_models.core.distributions`
- **Sampling functions**: `artifex.generative_models.core.sampling`
- **BlackJAX wrappers**: `artifex.generative_models.core.sampling.blackjax_samplers`
- **Direct BlackJAX API**: `blackjax.hmc`, `blackjax.nuts`, `blackjax.mala`

### Diagnostic Tools

- **ArviZ**: Python package for MCMC diagnostics and visualization
- **PyStan**: Stan interface with excellent diagnostics
- **PyMC**: Bayesian modeling with built-in diagnostics

## Support

If you encounter issues:

1. Check that BlackJAX is installed: `pip install blackjax`
2. Verify JAX GPU/CPU setup is correct
3. Review error messages for parameter constraints
4. Check BlackJAX documentation for API changes
5. Consult Artifex documentation or open an issue

---

**Tags:** #mcmc #blackjax #hmc #nuts #mala #sampling #comparison #advanced

**Difficulty:** Advanced

**Estimated Time:** 20-30 minutes
