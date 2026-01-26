# BlackJAX Integration Examples

![Level: Advanced](https://img.shields.io/badge/Level-Advanced-red)
![Runtime: 5-10 min](https://img.shields.io/badge/Runtime-5--10%20min-blue)
![Format: Dual](https://img.shields.io/badge/Format-Dual%20(.py%20%2B%20.ipynb)-green)

## Overview

This example demonstrates advanced integration patterns between BlackJAX samplers and Artifex's distribution framework. It compares two approaches: using BlackJAX's API directly for maximum control and visual feedback, versus using Artifex's functional API for simplicity and maximum performance.

## Files

- Python script: [`examples/generative_models/sampling/blackjax_integration_examples.py`](https://github.com/avitai/artifex/examples/generative_models/sampling/blackjax_integration_examples.py)
- Jupyter notebook: [`examples/generative_models/sampling/blackjax_integration_examples.ipynb`](https://github.com/avitai/artifex/examples/generative_models/sampling/blackjax_integration_examples.ipynb)

## Quick Start

=== "Python Script"
    ```bash
    # Run the complete example
    python examples/generative_models/sampling/blackjax_integration_examples.py
    ```

=== "Jupyter Notebook"
    ```bash
    # Launch Jupyter and open the notebook
    jupyter notebook examples/generative_models/sampling/blackjax_integration_examples.ipynb
    ```

## Learning Objectives

After completing this example, you will:

- [x] Understand how to use BlackJAX sampler classes with Artifex distributions
- [x] Learn to use both class-based and functional sampling APIs
- [x] Apply samplers to Artifex distributions (Normal, Mixture)
- [x] Compare class-based vs functional sampling approaches
- [x] Handle memory constraints in NUTS sampling
- [x] Sample from mixture distributions using MCMC
- [x] Understand the trade-offs between direct API (progress bars) and functional API (speed)

## Prerequisites

- Understanding of MCMC sampling concepts
- Familiarity with Artifex distributions module
- Basic knowledge of HMC, MALA, and NUTS algorithms
- Completion of [BlackJAX Integration Example](blackjax-example.md)
- Artifex core sampling and distributions modules

## Integration Approaches

Artifex supports two ways to integrate with BlackJAX, each with distinct advantages:

### 1. Direct BlackJAX API

**Use BlackJAX's native API directly with Artifex distributions:**

**Advantages:**

- Full control over sampling parameters and state management
- Progress bars with `tqdm` for visual feedback
- Per-iteration monitoring and debugging
- Custom sampling logic and diagnostics

**Disadvantages:**

- More verbose code
- Requires manual state management
- Must handle random key splitting manually

**When to use:**

- Interactive development and exploration
- When you need visual feedback on long-running samples
- Debugging and monitoring sampling behavior
- Implementing custom sampling algorithms

### 2. Artifex Functional API

**Use Artifex's convenience functions like `hmc_sampling()`, `mala_sampling()`:**

**Advantages:**

- Single function call for complete sampling workflow
- Automatic burn-in and state management
- Fully JIT-compiled for maximum performance
- Simplified interface for common use cases
- Cleanest, most concise code

**Disadvantages:**

- No progress bars (due to JIT compilation)
- Less fine-grained control
- Limited customization options

**When to use:**

- Production code requiring maximum performance
- Batch processing and automated workflows
- When simplicity is more important than monitoring
- Recommended for most applications

## Example Overview

This example includes six demonstrations:

1. **Normal Distribution with Direct BlackJAX HMC**: Full control with progress bars
2. **Normal Distribution with hmc_sampling**: Simple, fast functional API
3. **Normal Distribution with Direct BlackJAX MALA**: MALA sampler with monitoring
4. **Univariate Normal with Direct BlackJAX NUTS**: Memory-aware NUTS implementation
5. **Multimodal Distribution Comparison**: Teaching example comparing samplers
   - **5a: Mixture with MALA** (Wide Separation): Demonstrates local sampler limitations
   - **5b: Mixture with NUTS** (Moderate Separation): Shows Hamiltonian dynamics advantage

## Code Walkthrough

### Example 1: Direct BlackJAX HMC with Progress Bars

This example demonstrates the direct API approach with full visual feedback:

```python
import blackjax
from artifex.generative_models.core.distributions import Normal
from tqdm import tqdm

# Create distribution
true_mean = jnp.array([3.0, -2.0])
true_scale = jnp.array([1.5, 0.8])
normal_dist = Normal(loc=true_mean, scale=true_scale)

# Set up HMC sampler
inverse_mass_matrix = jnp.eye(2)
hmc = blackjax.hmc(
    normal_dist.log_prob,
    step_size=0.1,
    inverse_mass_matrix=inverse_mass_matrix,
    num_integration_steps=10,
)

# Initialize
init_position = jnp.zeros(2)
state = hmc.init(init_position)
step_fn = jax.jit(hmc.step)  # JIT compile step function

# Burn-in with progress bar
print("Running burn-in...")
for i in tqdm(range(n_burnin), desc="Burn-in", ncols=80):
    key = jax.random.fold_in(key, i)
    state, _ = step_fn(key, state)

# Sampling with progress bar
samples = jnp.zeros((n_samples, 2))
print("Sampling...")
for i in tqdm(range(n_samples), desc="Sampling", ncols=80):
    key = jax.random.fold_in(key, n_burnin + i)
    state, _ = step_fn(key, state)
    samples = samples.at[i].set(state.position)
```

**Key Points:**

- JIT-compile the step function for performance: `step_fn = jax.jit(hmc.step)`
- Use `tqdm` for visual feedback during long-running operations
- Manual state management provides full control
- Use `jax.random.fold_in()` for deterministic key splitting

### Example 2: Functional API for Maximum Performance

This example shows the simplified functional API:

```python
from artifex.generative_models.core.sampling.blackjax_samplers import hmc_sampling

# Create distribution (same as above)
normal_dist = Normal(loc=true_mean, scale=true_scale)

# Single function call - fully JIT-compiled
samples = hmc_sampling(
    normal_dist,
    init_position,
    key,
    n_samples=1000,
    n_burnin=500,
    step_size=0.1,
    num_integration_steps=10,
)
```

**Key Points:**

- Single function call replaces ~20 lines of code
- Automatically JIT-compiled using `jax.lax.scan` internally
- No progress bars, but maximum performance
- Automatic state management and burn-in
- Recommended for production and batch processing

**Performance Comparison:**

- Direct API with progress bars: ~2-3s for 1000 samples (with tqdm overhead)
- Functional API: ~1-2s for 1000 samples (fully optimized)
- Both use JIT compilation, but functional API has less Python overhead

### Example 3: Direct BlackJAX MALA

MALA demonstrates faster per-iteration sampling:

```python
# Create MALA sampler
mala = blackjax.mala(normal_dist.log_prob, step_size=0.05)

# Initialize and run (similar pattern to HMC)
state = mala.init(init_position)
step_fn = jax.jit(mala.step)

# Burn-in and sampling with progress bars
# (same structure as HMC example)
```

**Key Points:**

- MALA uses smaller step sizes than HMC (typically 0.05 vs 0.1)
- Faster per-iteration, but may need more samples for same ESS
- Good for problems where gradient evaluation is cheap

### Example 4: NUTS with Memory Awareness

NUTS requires special attention to memory constraints:

```python
# Use 1D distribution to reduce memory
true_mean = jnp.array([2.0])
true_scale = jnp.array([1.0])
normal_dist_1d = Normal(loc=true_mean, scale=true_scale)

# Create NUTS sampler with memory constraints
inverse_mass_matrix = jnp.array([1.0])
nuts = blackjax.nuts(
    normal_dist_1d.log_prob,
    step_size=0.8,
    inverse_mass_matrix=inverse_mass_matrix,
    max_depth=5,  # Lower to reduce memory usage
)
```

**Key Points:**

- NUTS stores trajectory information, requiring more memory
- Use `max_depth` parameter to control memory usage (default: 10)
- Start with lower-dimensional problems for testing
- Reduce `max_depth` if encountering memory errors
- For production, use smaller `n_samples` or increase system memory

### Example 5: Multimodal Distribution Comparison

This teaching example demonstrates how different MCMC samplers handle multimodal distributions,
comparing local samplers (MALA) with Hamiltonian samplers (NUTS).

#### Example 5a: MALA on Widely-Separated Mixture

Demonstrates MALA's limitation with distant modes:

```python
from artifex.generative_models.core.distributions import Mixture, Normal
from artifex.generative_models.core.sampling.blackjax_samplers import mala_sampling

# Create 1D mixture with modes 10 units apart
weights = jnp.array([0.6, 0.4])
means = jnp.array([[-2.0], [8.0]])  # Widely separated
scales = jnp.array([[0.8], [0.8]])

components = [Normal(loc=means[0], scale=scales[0]),
              Normal(loc=means[1], scale=scales[1])]
mixture = Mixture(components, weights)

# Sample with MALA
samples = mala_sampling(
    mixture,
    init_position=jnp.array([-2.0]),  # Start at first mode
    key=key,
    n_samples=10000,
    n_burnin=5000,
    step_size=0.05,
)
```

**Observation**: MALA gets stuck at the starting mode due to small gradient-guided steps.
With `step_size=0.05` and modes 10 units apart, the sampler cannot efficiently jump between modes.

**Key Teaching Points**:

- MALA is a **local sampler** - takes small gradient-guided steps
- Struggles with modes separated by low-probability regions
- Step size trade-off: small = slow mixing, large = poor acceptance
- Demonstrates importance of algorithm selection for problem structure

#### Example 5b: NUTS on Moderately-Separated Mixture

Shows NUTS's improved exploration:

```python
from artifex.generative_models.core.sampling.blackjax_samplers import nuts_sampling

# Create 1D mixture with modes 5 units apart (more moderate)
weights = jnp.array([0.6, 0.4])
means = jnp.array([[-2.0], [3.0]])  # Moderately separated
scales = jnp.array([[0.8], [0.8]])

components = [Normal(loc=means[0], scale=scales[0]),
              Normal(loc=means[1], scale=scales[1])]
mixture = Mixture(components, weights)

# Sample with NUTS
samples = nuts_sampling(
    mixture,
    init_position=jnp.array([-2.0]),
    key=key,
    n_samples=10000,
    n_burnin=5000,
    step_size=0.5,  # NUTS adapts this
)
```

**Observation**: NUTS successfully explores both modes, achieving ~53%/47% occupancy
(close to target 60%/40%). Hamiltonian dynamics enable long-range exploration.

**Key Teaching Points**:

- NUTS uses **Hamiltonian dynamics** - momentum enables distant exploration
- Automatically adapts step size and trajectory length (no-U-turn criterion)
- Handles moderate multimodality better than local samplers
- Still faces challenges with very distant modes (energy conservation constraints)
- For extreme multimodality: need parallel tempering, SMC, or tempered transitions

**Comparison Summary**:

| Aspect | MALA (5a) | NUTS (5b) |
|--------|-----------|-----------|
| Separation | 10 units (wide) | 5 units (moderate) |
| Result | Stuck in one mode | Both modes explored |
| Mechanism | Gradient-guided steps | Hamiltonian dynamics |
| Best for | Unimodal/log-concave | Moderate multimodality |

**Research Foundation**:

This comparison is supported by extensive MCMC research:

1. **Roberts & Tweedie (1996)** - "Exponential convergence of Langevin diffusions and their discrete approximations"
   - Established MALA's limitations with multimodal distributions
   - Showed MALA struggles with modes separated by low-probability regions

2. **Neal (2011)** - "MCMC Using Hamiltonian Dynamics" (Handbook of MCMC)
   - Comprehensive treatment of HMC advantages for exploration
   - Explains energy conservation constraints limiting extreme mode-switching

3. **Hoffman & Gelman (2014)** - "The No-U-Turn Sampler" (JMLR 15:1593-1623)
   - Introduced NUTS as adaptive HMC
   - Demonstrated superior performance on complex posteriors
   - Note: Still faces challenges with very distant modes

4. **Betancourt (2017)** - "A Conceptual Introduction to Hamiltonian Monte Carlo"
   - Explains why HMC/NUTS struggle with strongly multimodal distributions
   - Maximum potential energy increase bounded by initial kinetic energy
   - Recommends tempering for extreme multimodality

## Expected Output

### Sample Plots

Examples generate visualizations showing sampling behavior:

- **Normal distributions** (Examples 1-4): Scatter plots (2D) or histograms (1D) centered at true parameters
- **Mixture 5a (MALA)**: Histogram shows samples stuck near -2.0, very few near 8.0
- **Mixture 5b (NUTS)**: Histogram shows clear bimodal structure with both modes explored

### Statistics Tables

Examples print comparison tables showing true vs sampled statistics:

```
Statistic       True Value                     Sample Value
----------------------------------------------------------------------
Mean            [ 3.0000, -2.0000]             [ 3.0123, -1.9987]
Std             [ 1.5000,  0.8000]             [ 1.4987,  0.8012]

Timing: 1.23s total (812.3 samples/sec)
```

### Timing Information

- **Direct API**: Includes separate burn-in and sampling times
- **Functional API**: Reports total time (including JIT compilation on first call)
- **Samples/sec**: Measures sampling throughput (excluding burn-in)

## Performance Considerations

### API Comparison

| Aspect | Direct API | Functional API |
|--------|-----------|----------------|
| **Speed** | Fast (JIT-compiled steps) | Fastest (fully JIT-compiled) |
| **Progress Bars** | ✅ Yes | ❌ No (JIT limitation) |
| **Code Complexity** | Medium (~30-40 lines) | Low (~5-10 lines) |
| **Flexibility** | High (full control) | Medium (common parameters) |
| **Memory Efficiency** | Good | Excellent (optimized scan) |
| **Best For** | Development, debugging | Production, batch jobs |

### Memory Usage

**HMC/MALA:**

- Memory scales with problem dimension and sample count
- Minimal overhead beyond sample storage

**NUTS:**

- Stores trajectory tree: memory = $O(2^{\text{max\_depth}} \times \text{dimension})$
- Reduce `max_depth` from default 10 to 5-7 for memory-constrained systems
- Use smaller dimensions for testing (1D-5D)

### Tuning Recommendations

**For Direct API:**

- JIT-compile step function: `step_fn = jax.jit(sampler.step)`
- Use `jax.random.fold_in()` for deterministic key generation
- Add progress bars for long-running samples (burn-in > 1000 or n_samples > 5000)

**For Functional API:**

- No additional tuning needed - already optimized
- First call includes JIT compilation time (~1-5s)
- Subsequent calls with same parameters are instant

## Troubleshooting

### Slow Sampling (Direct API)

**Symptom**: Direct API examples taking too long

**Solution**:

```python
# Always JIT-compile the step function
step_fn = jax.jit(hmc.step)  # DO THIS

# Not this:
state, _ = hmc.step(key, state)  # Too slow!
```

### No Progress Bars (Functional API)

**Symptom**: Functional API appears to hang with no feedback

**Solution**:

- This is expected behavior - functional API is fully JIT-compiled
- First call takes longer (JIT compilation)
- No progress bars due to JIT compilation
- If you need progress bars, use the Direct API approach

### NUTS Memory Errors

**Symptom**: Out of memory when using NUTS

**Solution**:

```python
# Reduce max_depth
nuts = blackjax.nuts(
    log_prob_fn,
    step_size=0.8,
    inverse_mass_matrix=mass_matrix,
    max_depth=5,  # Lower from default 10
)

# Or reduce problem dimension for testing
# Or use fewer samples
n_samples = 500  # Instead of 2000
```

### Poor Mixing on Mixture

**Symptom**: Samples stuck in one mode of mixture distribution

**Solution**:

```python
# Increase burn-in significantly
n_burnin = 2000  # Or more

# Try different initialization
init_position = jnp.array([3.0, 3.0])  # Start near one mode

# Use longer sampling
n_samples = 5000

# Consider HMC instead of MALA for better mode exploration
```

## Design Patterns

### When to Use Each Approach

**Use Direct API when:**

- Developing and debugging new sampling strategies
- Need visual feedback on long-running operations
- Implementing custom diagnostics or monitoring
- Interactive exploration in Jupyter notebooks
- Learning and understanding MCMC behavior

**Use Functional API when:**

- Running production inference pipelines
- Batch processing many sampling tasks
- Maximizing performance is critical
- Code simplicity is valued
- You're confident in the sampling parameters

### Hybrid Approach

For best of both worlds, use this pattern:

```python
# Development: Use Direct API with progress bars
if __name__ == "__main__":
    # Interactive development with monitoring
    samples = sample_with_progress_bars(...)

# Production: Switch to functional API
def production_inference(data):
    # Fast, JIT-compiled sampling
    return hmc_sampling(...)
```

## Experiments to Try

1. **Compare timing**: Measure Direct API vs Functional API performance on same problem

2. **Memory profiling**: Test NUTS with different `max_depth` values and monitor memory usage

3. **Multimodal exploration**: Visualize how different samplers explore the mixture distribution

4. **Scaling experiment**: Test both APIs with increasing problem dimensions (2D, 10D, 50D, 100D)

5. **Thinning effects**: Experiment with `thinning` parameter in functional API to reduce autocorrelation

6. **Acceptance rates**: Track acceptance rates in Direct API examples to optimize step sizes

## Next Steps

### Related Examples

<div class="grid cards" markdown>

- **BlackJAX Integration Example**

    Start with the basics of BlackJAX integration

    [blackjax-example.md](blackjax-example.md)

- **BlackJAX Sampling Examples**

    Compare different sampling algorithms

    [blackjax-sampling-examples.md](blackjax-sampling-examples.md)

</div>

### Further Learning

- [BlackJAX Documentation](https://blackjax-devs.github.io/blackjax/)
- [BlackJAX Sampling Book](https://blackjax-devs.github.io/sampling-book/)
- [JAX Documentation on Scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)
- Artifex Distributions Module Documentation
- Artifex Sampling Module Documentation

## Additional Resources

### Papers

1. **HMC Performance**: Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo"

2. **NUTS Algorithm**: Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler"

3. **JAX for MCMC**: Lao, J., et al. (2020). "tfp.mcmc: Modern Markov Chain Monte Carlo Tools Built for Modern Hardware"

### Code References

- **Distribution classes**: `artifex.generative_models.core.distributions`
- **Functional samplers**: `artifex.generative_models.core.sampling.blackjax_samplers`
- **Direct BlackJAX API**: `blackjax.hmc`, `blackjax.nuts`, `blackjax.mala`
- **JAX primitives**: `jax.lax.scan`, `jax.lax.fori_loop`

## Support

If you encounter issues:

1. Check that BlackJAX is installed: `pip install blackjax`
2. Verify JAX GPU/CPU setup is correct
3. For memory errors, reduce `max_depth` in NUTS or problem dimension
4. For slow performance, ensure step functions are JIT-compiled
5. Check progress bar behavior matches API expectations
6. Consult Artifex documentation or open an issue

---

**Tags:** #mcmc #blackjax #hmc #nuts #mala #integration #advanced #performance

**Difficulty:** Advanced

**Estimated Time:** 25-35 minutes
