# Energy-Based Models (EBMs) Explained

<div class="grid cards" markdown>

- :material-lightning-bolt:{ .lg .middle } **Energy Functions**

    ---

    Learn distributions by assigning low energy to data and high energy to non-data, capturing complex dependencies

- :material-chart-scatter-plot:{ .lg .middle } **Flexible Modeling**

    ---

    No architectural constraints—any neural network can define the energy landscape

- :material-check-decagram:{ .lg .middle } **Unified Framework**

    ---

    Simultaneously perform classification, generation, and out-of-distribution detection

- :material-sync:{ .lg .middle } **MCMC Sampling**

    ---

    Generate samples by traversing the energy landscape using Markov Chain Monte Carlo methods

</div>

---

!!! tip "New here?"
    For a one-page map of how EBMs fit next to VAEs, GANs, Diffusion, Flows, and Autoregressive models, start with [**Generative Models — A Unified View**](generative-models-unified.md). This page is the deep-dive on energy-based models specifically.

## Overview

Energy-Based Models (EBMs), formalised in the influential tutorial by [LeCun, Chopra, Hadsell, Ranzato & Huang (2006)](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf) and put on a deep-learning footing by [Du & Mordatch (2019)](https://arxiv.org/abs/1903.08689), are a class of **deep generative models** that learn data distributions by assigning scalar energy values to every possible configuration. Unlike models that explicitly parameterize probability distributions, EBMs define probabilities implicitly through an energy function: **lower energy corresponds to higher probability**. The 2024–2026 wave of *energy-flow* methods (**Energy Matching**, **Equilibrium Matching**, **Energy-Based Diffusion Language Models**) has put EBMs back at the centre of generative-modelling research after a quiet 2020–2023 — see [Recent Advances (2024–2026)](#recent-advances-20242026-energy-flow-methods-and-implicit-ebms) below.

**What makes EBMs special?**

EBMs solve generative modeling from a fundamentally different perspective. Rather than learning to directly generate samples (GANs) or compress data through bottlenecks (VAEs), EBMs learn an **energy landscape** where:

- **Data regions have low energy** - the model assigns favorable (low) energy to realistic configurations
- **Non-data regions have high energy** - implausible configurations receive unfavorable (high) energy
- **Sampling traverses the landscape** - MCMC methods move from high to low energy regions
- **No architectural constraints** - any neural network architecture can define the energy function

### The Intuition: Probability Through Energy

Think of EBMs like a physical landscape with hills and valleys:

1. **The Energy Function** defines the terrain—each point (data configuration) has an elevation (energy value). Data points sit in deep valleys (low energy), while meaningless noise occupies high peaks (high energy).

2. **Training Shapes the Landscape** - The model learns to carve valleys where training data exists and raise peaks everywhere else, creating a terrain that naturally guides samples toward realistic outputs.

3. **Sampling Rolls Downhill** - Like a ball released on a hillside, MCMC sampling iteratively moves toward lower energy, eventually settling in valleys that correspond to realistic data.

4. **Probability is Energy-Based** - The probability of any configuration depends on its energy through the Boltzmann distribution: $p(x) \propto e^{-E(x)}$. Lower energy means exponentially higher probability.

The critical insight: by learning to distinguish data from non-data through energy assignment, EBMs capture complex distributions without explicit density modeling or generation mechanisms.

---

## Mathematical Foundation

### The Boltzmann Distribution

EBMs define probability distributions through the **Boltzmann distribution**:

$$
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}
$$

where:

- $E_\theta(x)$ is the **energy function** parameterized by neural network with weights $\theta$
- $Z_\theta = \int e^{-E_\theta(x')} dx'$ is the **partition function** (normalizing constant)
- Lower energy $E_\theta(x)$ corresponds to higher probability $p_\theta(x)$

```mermaid
graph TD
    X["Data x"] --> E["Energy Function<br/>E_θ(x)"]
    E --> NegE["-E_θ(x)"]
    NegE --> Exp["exp(-E_θ(x))"]
    Exp --> Norm["÷ Z_θ"]
    Z["Partition Function<br/>Z_θ = ∫ exp(-E_θ(x')) dx'"] --> Norm
    Norm --> P["Probability p_θ(x)"]

    style X fill:#e1f5ff
    style E fill:#fff3cd
    style P fill:#c8e6c9
    style Z fill:#ffccbc
```

**Why the exponential?** The exponential function ensures probabilities are positive and creates sharp distinctions between energy levels—a small energy difference produces a large probability ratio.

### The Intractable Partition Function

The partition function $Z_\theta$ poses the **fundamental challenge** in EBM training:

$$
Z_\theta = \int e^{-E_\theta(x)} dx
$$

This integral sums over **all possible configurations** of $x$, which becomes intractable for high-dimensional data:

- **Images (256×256×3)**: $Z_\theta$ requires integration over $196{,}608$-dimensional space
- **Text (512 tokens, vocab 50k)**: $Z_\theta$ requires summing over $(50{,}000)^{512} \approx 10^{2414}$ configurations

The intractability of $Z_\theta$ makes **maximum likelihood training** directly infeasible and necessitates alternative training methods.

### Energy Functions and Neural Networks

The energy function $E_\theta(x)$ can be implemented using **any neural network architecture**:

**Multi-Layer Perceptron (MLP) Energy:**

$$
E_\theta(x) = \text{MLP}_\theta(x) \in \mathbb{R}
$$

```python
class MLPEnergyFunction(nnx.Module):
    def __init__(self, hidden_dims, input_dim, *, rngs):
        self.layers = []
        for dim in hidden_dims:
            self.layers.append(nnx.Linear(input_dim, dim, rngs=rngs))
            input_dim = dim
        self.output = nnx.Linear(input_dim, 1, rngs=rngs)

    def __call__(self, x):
        for layer in self.layers:
            x = nnx.relu(layer(x))
        return self.output(x).squeeze(-1)  # Scalar energy
```

**Convolutional Energy (for images):**

$$
E_\theta(x) = \text{CNN}_\theta(x) \in \mathbb{R}
$$

```python
class CNNEnergyFunction(nnx.Module):
    def __init__(self, hidden_dims, input_channels, *, rngs):
        self.conv_layers = []
        for out_channels in hidden_dims:
            self.conv_layers.append(
                nnx.Conv(input_channels, out_channels,
                        kernel_size=(3, 3), rngs=rngs)
            )
            input_channels = out_channels
        self.global_pool = lambda x: jnp.mean(x, axis=(1, 2))
        self.output = nnx.Linear(hidden_dims[-1], 1, rngs=rngs)

    def __call__(self, x):
        for conv in self.conv_layers:
            x = nnx.relu(conv(x))
        x = self.global_pool(x)
        return self.output(x).squeeze(-1)
```

**Key properties:**

- **Scalar output** - energy function outputs single real number per input
- **Unrestricted architecture** - no invertibility, dimension matching, or structural requirements
- **Learned landscape** - network parameters define the energy surface topology

---

## Training Energy-Based Models

### Maximum Likelihood: The Ideal but Intractable Approach

The maximum likelihood objective seeks to maximize the log-probability of observed data:

$$
\max_\theta \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]
$$

Substituting the Boltzmann distribution:

$$
\log p_\theta(x) = -E_\theta(x) - \log Z_\theta
$$

Taking gradients with respect to $\theta$:

$$
\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) + \nabla_\theta \log Z_\theta
$$

The first term $-\nabla_\theta E_\theta(x)$ is tractable—just backpropagation through the energy function evaluated at data $x$.

The second term is problematic:

$$
\nabla_\theta \log Z_\theta = \frac{1}{Z_\theta} \nabla_\theta Z_\theta = \frac{1}{Z_\theta} \int \nabla_\theta e^{-E_\theta(x')} dx'
$$

$$
= -\int \frac{e^{-E_\theta(x')}}{Z_\theta} \nabla_\theta E_\theta(x') dx' = -\mathbb{E}_{x' \sim p_\theta}[\nabla_\theta E_\theta(x')]
$$

This requires **sampling from the model distribution** $p_\theta(x)$, which is exactly what we're trying to learn! This chicken-and-egg problem necessitates approximate training methods.

### Contrastive Divergence: Practical Approximation

**Contrastive Divergence (CD)** ([Hinton, 2002 — Neural Computation](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf)) approximates the intractable expectation using short MCMC chains:

**Algorithm**:

1. **Positive phase**: Sample data $x^+ \sim p_{\text{data}}$, compute $\nabla_\theta E_\theta(x^+)$.
2. **Negative phase**: Initialise $x^- = x^+$ (or from a persistent buffer), run $k$ MCMC steps to obtain a model sample $x^-$, compute $\nabla_\theta E_\theta(x^-)$.
3. **Gradient of the negative log-likelihood**:

    $$
    \nabla_\theta \mathcal{L}_{\text{CD}} \;\approx\; \nabla_\theta E_\theta(x^+) \;-\; \nabla_\theta E_\theta(x^-).
    $$

    Equivalently, the SGD parameter update is $\theta \leftarrow \theta - \alpha\, \nabla_\theta \mathcal{L}_{\text{CD}}$, which **lowers** the energy on data and **raises** it on model samples.

```mermaid
graph TD
    Data["Data x⁺<br/>(real sample)"] --> PosGrad["Positive Gradient<br/>-∇E(x⁺)"]
    Data --> Init["Initialize<br/>x⁻ = x⁺"]
    Init --> MCMC["k MCMC Steps<br/>(typically k=1-60)"]
    MCMC --> Neg["Model Sample<br/>x⁻"]
    Neg --> NegGrad["Negative Gradient<br/>+∇E(x⁻)"]
    PosGrad --> Update["Parameter Update"]
    NegGrad --> Update

    style Data fill:#c8e6c9
    style Neg fill:#ffccbc
    style Update fill:#e1f5ff
```

**Why it works:**

- **Positive gradient** pushes down energy at data points
- **Negative gradient** pushes up energy at model samples
- **Short chains** approximate the full expectation with $k \ll \infty$ steps
- **Persistent chains** (PCD) reuse samples across iterations for better mixing

!!!warning "CD-k Limitations"
    CD with very short chains ($k=1$) can be unstable. Persistent Contrastive Divergence (PCD) maintains a persistent pool of samples across training iterations, providing better gradient estimates.

### Persistent Contrastive Divergence (PCD)

**PCD** ([Tieleman, 2008 — ICML](https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf)) maintains a persistent sample buffer across training iterations:

**Algorithm**:

1. **Initialize buffer**: $\mathcal{B} = \{x_1, x_2, \ldots, x_M\}$ with random samples
2. **Each training iteration**:
    - Sample data batch $\{x^+_i\}$
    - Sample buffer batch $\{x^-_j\} \sim \mathcal{B}$
    - Run $k$ MCMC steps from each $x^-_j$ to get updated $\tilde{x}^-_j$
    - Update buffer: $\mathcal{B} \leftarrow \mathcal{B} \cup \{\tilde{x}^-_j\}$
    - Compute gradient: $\nabla_\theta \mathcal{L} \approx -\frac{1}{N}\sum_i \nabla E(x^+_i) + \frac{1}{N}\sum_j \nabla E(\tilde{x}^-_j)$
    - Update parameters

```python
def persistent_contrastive_divergence(
    energy_fn, real_samples, sample_buffer, rng_key,
    n_mcmc_steps=60, step_size=0.01, noise_scale=0.005
):
    """Persistent Contrastive Divergence training step."""
    batch_size = real_samples.shape[0]
    sample_shape = real_samples.shape[1:]

    # Sample initial points from buffer
    init_samples = sample_buffer.sample_initial(batch_size, rng_key, sample_shape)

    # Run MCMC chain
    final_samples = langevin_dynamics(
        energy_fn, init_samples, n_mcmc_steps, step_size, noise_scale, rng_key
    )

    # Update buffer with new samples
    sample_buffer.push(final_samples)

    return init_samples, final_samples
```

**Advantages over CD**:

- **Better mixing**: Chains continue across iterations
- **More accurate gradients**: Samples closer to true model distribution
- **Stable training**: Reduces oscillations and mode collapse

### Langevin Dynamics: MCMC Sampling

**Langevin dynamics** ([Welling & Teh, 2011 — ICML](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)) provides the MCMC sampler for EBMs. The continuous-time SDE has the well-calibrated discretisation

$$
x_{t+1} = x_t - \frac{\epsilon}{2} \nabla_x E_\theta(x_t) + \sqrt{\epsilon} \, \xi_t, \qquad \xi_t \sim \mathcal{N}(0, I),
$$

which is **unbiased** for $p_\theta$ in the limit $\epsilon \to 0$. In practice, modern deep-EBM training (Du & Mordatch 2019 onward) instead uses an **SGLD-style decoupled** step where the gradient step size $\eta$ and the noise scale $\sigma$ are tuned independently:

$$
x_{t+1} = x_t - \eta\, \nabla_x E_\theta(x_t) + \sigma\, \xi_t,
$$

typically with $\eta \approx 10$–$100\times$ larger than the well-calibrated $\epsilon/2$ would suggest, plus a per-step clip to a valid data range (e.g. $[-1, 1]$ for images). This **breaks the formal Langevin sampling guarantee** but empirically produces much sharper samples within a feasible step budget; the implementation below uses this convention.

**Two components:**

1. **Gradient descent** $-\eta \nabla_x E_\theta(x_t)$ moves toward lower energy.
2. **Brownian motion** $\sigma \xi_t$ explores the space and prevents the chain from collapsing onto a single mode.

```python
def langevin_dynamics(
    energy_fn, init_samples, n_steps, step_size, noise_scale, rng_key
):
    """Sample from EBM using Langevin dynamics."""
    x = init_samples

    for step in range(n_steps):
        # Compute energy gradient
        energy_grad = jax.grad(lambda x: jnp.sum(energy_fn(x)))(x)

        # Gradient descent step
        x = x - (step_size / 2) * energy_grad

        # Add noise
        rng_key, noise_key = jax.random.split(rng_key)
        noise = jax.random.normal(noise_key, x.shape) * noise_scale
        x = x + noise

        # Clip to valid range
        x = jnp.clip(x, -1.0, 1.0)

    return x
```

**Convergence**: As $n_{\text{steps}} \to \infty$ and $\epsilon \to 0$, Langevin dynamics samples from $p_\theta(x)$ exactly. In practice, finite steps and step sizes introduce bias.

### Score Matching: Avoiding Sampling

**Score Matching** ([Hyvärinen, 2005 — JMLR](https://jmlr.org/papers/v6/hyvarinen05a.html)) bypasses MCMC sampling by matching the **score function** (gradient of log-density):

$$
\psi_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)
$$

**Objective** - Minimize the Fisher divergence:

$$
\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_{\text{data}}}[\|\nabla_x E_\theta(x) + \nabla_x \log p_{\text{data}}(x)\|^2]
$$

Since $\nabla_x \log p_{\text{data}}(x)$ is unknown, Hyvärinen showed this is equivalent to:

$$
\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_{\text{data}}}[-\text{tr}(\nabla_x^2 E_\theta(x)) + \frac{1}{2}\|\nabla_x E_\theta(x)\|^2]
$$

**Advantages**:

- **No sampling required** during training
- **No partition function** computation needed
- **Fast training** compared to CD/PCD

**Disadvantages**:

- **Second derivatives** $\nabla_x^2 E_\theta(x)$ in the Hyvärinen objective are expensive to compute (sliced score matching ([Song et al., 2019](https://arxiv.org/abs/1905.07088)) and DSM both side-step this).
- **Generation still requires MCMC** at test time (or a learned sampler).
- The vanilla Hyvärinen objective scales poorly with dimension because the trace term has high variance; **denoising / sliced score matching** completely fix this and are the basis of modern score-based diffusion ([Song & Ermon, 2019 — NCSN](https://arxiv.org/abs/1907.05600); [Song et al., 2021 — score-SDE](https://arxiv.org/abs/2011.13456)).

**Denoising Score Matching** ([Vincent, 2011 — Neural Computation](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)): A practical variant adds noise to data:

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{p(\tilde{x}|x)}[\|\nabla_{\tilde{x}} E_\theta(\tilde{x}) + \frac{x - \tilde{x}}{\sigma^2}\|^2]
$$

where $\tilde{x} = x + \sigma \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$.

### Noise Contrastive Estimation (NCE)

**NCE** ([Gutmann & Hyvärinen, 2010 — AISTATS](http://proceedings.mlr.press/v9/gutmann10a.html)) frames training as **binary classification**:

**Setup**: Given data samples $\{x_i\}$ and noise samples $\{\tilde{x}_j\}$ from distribution $p_n$, train a classifier to distinguish them.

**Decision rule**: sample $x$ comes from data if $\frac{p_\theta(x)}{p_n(x)} > 1$. Substituting $\log p_\theta(x) = -E_\theta(x) - c$ where $c = \log Z_\theta$ is treated as a **learnable scalar** (a key NCE trick that turns $Z$ from a doomed integral into one extra parameter):

$$
h_\theta(x) = \sigma(-E_\theta(x) - c - \log p_n(x)).
$$

**NCE objective**:

$$
\mathcal{L}_{\text{NCE}} = -\mathbb{E}_{x \sim p_{\text{data}}}[\log h_\theta(x)] - \mathbb{E}_{\tilde{x} \sim p_n}[\log(1 - h_\theta(\tilde{x}))]
$$

**Advantages**:

- **No partition function** in gradient
- **Simple implementation** like standard classification
- **Flexible noise distribution** $p_n$ (typically Gaussian)

**When to use**:

- High-dimensional problems where MCMC is slow
- When generation quality is secondary to density estimation
- Contrastive learning scenarios

---

## Joint Energy-Based Models (JEM)

**Joint Energy-Based Models** ([Grathwohl, Wang, Jacobsen, Duvenaud, Norouzi & Swersky, 2020 — ICLR](https://arxiv.org/abs/1912.03263)) unify generative and discriminative modeling in a single framework.

### The Key Insight

A standard classifier $p(y|x)$ can be reinterpreted as an energy model:

$$
p_\theta(y|x) = \frac{e^{f_\theta[x](y)}}{\sum_{y'} e^{f_\theta[x](y')}}
$$

where $f_\theta(x) \in \mathbb{R}^C$ are logits.

**Define energy as**:

$$
E_\theta(x, y) = -f_\theta[x](y)
$$

Then the **joint distribution** becomes:

$$
p_\theta(x, y) = \frac{e^{f_\theta[x](y)}}{Z_\theta}, \quad Z_\theta = \sum_{y'} \int e^{f_\theta[x'](y')} dx'
$$

And the **marginal** data distribution:

$$
p_\theta(x) = \frac{\sum_y e^{f_\theta[x](y)}}{Z_\theta}
$$

```mermaid
graph TD
    X["Input x"] --> F["Classifier f_θ(x)<br/>(C logits)"]
    F --> Cond["Conditional p(y|x)<br/>(softmax)"]
    F --> LogSumExp["log-sum-exp"]
    LogSumExp --> Energy["E_θ(x) = -logsumexp(f_θ(x))"]
    Energy --> Joint["Joint p(x,y)<br/>∝ exp(f_θ(x)[y])"]
    Joint --> Marg["Marginal p(x)<br/>∝ exp(-E_θ(x))"]

    style F fill:#fff3cd
    style Cond fill:#e1f5ff
    style Marg fill:#c8e6c9
```

### Training JEM

**Hybrid objective** combining classification and generation:

$$
\mathcal{L}_{\text{JEM}} = \underbrace{\mathcal{L}_{\text{class}}(f_\theta)}_{\text{discriminative}} + \lambda \underbrace{\mathcal{L}_{\text{gen}}(p_\theta)}_{\text{generative}}
$$

where:

- $\mathcal{L}_{\text{class}} = -\mathbb{E}_{(x, y) \sim p_{\text{data}}}[\log p_\theta(y|x)]$ (cross-entropy)
- $\mathcal{L}_{\text{gen}} = -\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]$ (MLE via SGLD)

**Training algorithm**:

1. Sample batch $\{(x_i, y_i)\}$ from labeled data
2. **Classification update**: Compute cross-entropy loss, backprop
3. **Generative update**:
   - Generate samples via SGLD
   - Compute CD gradient
   - Update parameters
4. Alternate or combine both losses

**Capabilities of JEM**:

<div class="grid cards" markdown>

- :material-brain:{ .lg .middle } **Classification**

    ---

    Achieves competitive accuracy on CIFAR-10, SVHN, ImageNet

- :material-image-multiple:{ .lg .middle } **Generation**

    ---

    Realistic samples — historically below GAN quality on CIFAR-10 (FID 38), reaching FID ≈ 9–11 with stabilised training ([Zhang et al., 2023](https://arxiv.org/abs/2303.04187)).

- :material-shield-alert:{ .lg .middle } **Out-of-Distribution Detection**

    ---

    Uses $\log p(x)$ as an OOD score; achieves AUROC > 0.90 on standard CIFAR-10-vs-SVHN benchmarks.

- :material-shield-check:{ .lg .middle } **Adversarial Robustness**

    ---

    More robust to adversarial examples than standard classifiers

- :material-label-variant:{ .lg .middle } **Calibration**

    ---

    Better uncertainty estimates than standard softmax

- :material-chart-scatter-plot:{ .lg .middle } **Hybrid Discriminative-Generative**

    ---

    Leverages both labeled and unlabeled data

</div>

**Challenges**:

- **Training instability**: SGLD can fail to produce quality samples
- **Computational cost**: Requires MCMC sampling during training
- **Hyperparameter sensitivity**: Step size, noise scale, buffer size critical

Recent work has substantially stabilised JEM — see [Recent Advances → Foundational Refinements](#foundational-refinements-20212023) for *Stabilized JEM* ([Zhang et al., 2023](https://arxiv.org/abs/2303.04187)), which delivers the first stable JEM training on ImageNet-scale data via better initialisation, curriculum learning, and adaptive SGLD.

---

## Architecture Design

### Energy Function Design Principles

**Key considerations** when designing energy functions:

1. **Scalar output**: $E_\theta: \mathcal{X} \to \mathbb{R}$ must map inputs to single energy value
2. **Sufficient capacity**: Deep networks capture complex dependencies
3. **Residual connections**: Enable deep architectures (10+ layers)
4. **Normalization layers**: GroupNorm or LayerNorm (BatchNorm can interfere with MCMC)
5. **Stable optimization**: Regularization, gradient clipping, and conservative step sizes keep training tractable

### MLP Energy Function (Tabular Data)

```python
class MLPEnergyFunction(nnx.Module):
    def __init__(
        self, hidden_dims, input_dim, activation=nnx.gelu,
        dropout_rate=0.0, *, rngs
    ):
        super().__init__(rngs=rngs)
        self.layers = []

        for dim in hidden_dims:
            self.layers.append(nnx.Linear(input_dim, dim, rngs=rngs))
            if dropout_rate > 0:
                self.layers.append(nnx.Dropout(dropout_rate, rngs=rngs))
            input_dim = dim

        self.output_layer = nnx.Linear(input_dim, 1, rngs=rngs)
        self.activation = activation

    def __call__(self, x, *, deterministic=True):
        for layer in self.layers:
            if isinstance(layer, nnx.Linear):
                x = self.activation(layer(x))
            elif isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
        return self.output_layer(x).squeeze(-1)
```

**Use cases**: Tabular data, low-to-moderate dimensional problems (< 1000 dimensions)

### CNN Energy Function (Images)

```python
class CNNEnergyFunction(nnx.Module):
    def __init__(
        self, hidden_dims, input_channels=3,
        activation=nnx.silu, *, rngs
    ):
        super().__init__(rngs=rngs)
        self.conv_blocks = []

        in_channels = input_channels
        for out_channels in hidden_dims:
            self.conv_blocks.append(EnergyBlock(
                in_channels, out_channels, activation=activation, rngs=rngs
            ))
            in_channels = out_channels

        self.global_pool = lambda x: jnp.mean(x, axis=(1, 2))
        self.fc_layers = [
            nnx.Linear(hidden_dims[-1], hidden_dims[-1]//2, rngs=rngs),
            nnx.Linear(hidden_dims[-1]//2, 1, rngs=rngs),
        ]
        self.activation = activation

    def __call__(self, x, *, deterministic=True):
        for block in self.conv_blocks:
            x = block(x)
        x = self.global_pool(x)
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:
                x = self.activation(x)
        return x.squeeze(-1)

class EnergyBlock(nnx.Module):
    """Residual block with GroupNorm for energy function."""
    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=2,
        use_residual=False, activation=nnx.silu, *, rngs
    ):
        super().__init__()
        self.conv = nnx.Conv(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride), padding="SAME", rngs=rngs
        )
        self.norm = nnx.GroupNorm(
            min(32, out_channels), out_channels, rngs=rngs
        )
        self.activation = activation
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)

    def __call__(self, x):
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        if self.use_residual:
            x = x + residual
        return x
```

**Use cases**: Image data (MNIST, CIFAR-10, CelebA), spatial data

### Deep Energy Functions with Residual Blocks

For complex distributions, deeper architectures with residual blocks and explicit normalization improve stability:

```python
class DeepCNNEnergyFunction(nnx.Module):
    def __init__(
        self, hidden_dims, input_channels=3,
        use_residual=True, *, rngs
    ):
        super().__init__(rngs=rngs)
        # Build deeper architecture (8-12 layers)
        # Use residual connections where dimensions match
        # Normalize intermediate activations for stable MCMC training
        ...
```

**Benefits**:

- **Group normalization**: Keeps intermediate activations well-scaled during sampling
- **Residual connections**: Enable training of 10+ layer networks
- **Better sample quality**: Captures finer details in distribution

---

## Training Dynamics and Best Practices

### Sample Buffer Management

**Persistent sample buffers** are critical for stable EBM training:

**Buffer initialization**:

- Random noise (Gaussian or uniform)
- Real data with added noise
- Pre-trained model samples

**Buffer operations**:

```python
class SampleBuffer:
    def __init__(self, capacity=8192, reinit_prob=0.05):
        self.capacity = capacity
        self.reinit_prob = reinit_prob
        self.buffer = []

    def sample_initial(self, batch_size, rng_key, sample_shape):
        """Sample starting points for MCMC."""
        if len(self.buffer) < batch_size:
            # Not enough samples - return random noise
            return jax.random.normal(rng_key, (batch_size, *sample_shape))

        # Sample from buffer
        indices = jax.random.choice(rng_key, len(self.buffer), (batch_size,))
        samples = jnp.array([self.buffer[i] for i in indices])

        # Reinitialize some samples with noise
        reinit_mask = jax.random.uniform(rng_key, (batch_size,)) < self.reinit_prob
        noise = jax.random.normal(rng_key, samples.shape)
        samples = jnp.where(reinit_mask[:, None, None, None], noise, samples)

        return samples

    def push(self, samples):
        """Add new samples to buffer."""
        for sample in samples:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)  # Remove oldest
            self.buffer.append(sample)
```

**Reinitialiation probability**: Setting $p_{\text{reinit}} = 0.05$ helps escape local minima.

### Hyperparameter Guidelines

**MCMC sampling**:

| Parameter | Value Range | Typical | Notes |
|-----------|-------------|---------|-------|
| **MCMC steps** | 20-200 | 60 | More steps = better samples, slower |
| **Step size** | 0.001-0.05 | 0.01 | Too large: instability, too small: slow |
| **Noise scale** | 0.001-0.01 | 0.005 | Exploration vs exploitation |

**Training**:

| Parameter | Value Range | Typical | Notes |
|-----------|-------------|---------|-------|
| **Learning rate** | 1e-5 to 1e-3 | 1e-4 | Lower than standard supervised |
| **Batch size** | 32-256 | 128 | Larger batches stabilize |
| **Buffer capacity** | 2048-16384 | 8192 | More = better mixing |
| **Alpha (regularization)** | 0.001-0.1 | 0.01 | Prevents energy collapse |

**Data preprocessing**:

- **Normalize** images to $[-1, 1]$ or $[0, 1]$
- **Add noise** to real data during training (noise scale ~0.005)
- **Clip samples** after MCMC to valid range

### Common Training Issues

<div class="grid cards" markdown>

- :material-alert-circle: **Mode Collapse**

    ---

    **Symptom**: Generated samples lack diversity, all similar

    **Solutions**: Increase buffer size, use reinitialize probability, longer MCMC chains, reduce learning rate

- :material-alert-circle: **Energy Explosion**

    ---

    **Symptom**: Energy values grow unbounded, NaN losses

    **Solutions**: Add regularization term $\alpha \mathbb{E}[E(x)^2]$, gradient clipping, smaller learning rate, and stronger residual/group-normalized architectures

- :material-alert-circle: **Poor Sample Quality**

    ---

    **Symptom**: Samples look like noise or blurry averages

    **Solutions**: More MCMC steps (100+), better step size tuning, deeper energy function, larger model capacity

- :material-alert-circle: **Training Instability**

    ---

    **Symptom**: Oscillating losses, sudden divergence

    **Solutions**: Persistent buffer, lower learning rate, gradient clipping, and careful residual/group-normalized architecture scaling

</div>

### Monitoring and Diagnostics

**Essential metrics**:

1. **Energy values**: Monitor $\mathbb{E}[E(x_{\text{data}})]$ and $\mathbb{E}[E(x_{\text{gen}})]$
   - Should satisfy $E(x_{\text{data}}) < E(x_{\text{gen}})$
   - Gap should be positive and stable

2. **MCMC diagnostics**:
   - **Acceptance rate**: Track how many proposed moves are accepted
   - **Energy trajectory**: Plot energy over MCMC steps (should decrease)

3. **Sample quality**: Visual inspection, FID/IS scores

4. **Gradient norms**: Should be stable, not exploding

```python
def training_step_with_diagnostics(model, batch, rngs):
    # Forward pass
    real_data = batch['x']
    real_energy = model.energy(real_data).mean()

    # Generate samples
    fake_data = generate_samples_via_mcmc(model, batch_size, rngs)
    fake_energy = model.energy(fake_data).mean()

    # Compute loss
    loss = real_energy - fake_energy + alpha * (real_energy**2).mean()

    # Diagnostics
    energy_gap = fake_energy - real_energy

    print(f"Real Energy: {real_energy:.3f}, "
          f"Fake Energy: {fake_energy:.3f}, "
          f"Gap: {energy_gap:.3f}, "
          f"Loss: {loss:.3f}")

    return loss
```

---

## Comparing EBMs with Other Generative Models

### EBMs vs GANs: Stable Training vs Sharp Samples

| Aspect | Energy-Based Models | GANs |
|--------|---------------------|------|
| **Training stability** | Stable with PCD / DRL / Energy Matching; classical CD with short chains is brittle | Adversarial; needs spectral norm + R1 |
| **Sample quality** | Pre-2024 below GANs; modern EqM hits **FID 1.90 on ImageNet 256²** ([Wang et al., 2025](https://arxiv.org/abs/2510.02300)), surpassing GAN-class | Strong; modern R3GAN / GigaGAN remain competitive |
| **Sampling speed** | Classical: 50–200 MCMC steps. EqM: a few adaptive-optimiser steps | 1 forward pass |
| **Mode coverage** | Excellent (explicit density) | Prone to mode collapse |
| **Likelihood** | Unnormalised; exact only in small / special structures, otherwise estimated or bounded because $Z$ is intractable | None (implicit) |
| **Architecture** | Any neural network producing a scalar | Generator-discriminator pair |
| **Strong use cases** | Density estimation, OOD detection, scientific applications, hybrid generative-discriminative tasks | Real-time image synthesis, latent editing |

**When to use EBMs over GANs**:

- Unnormalised scoring or density-ratio estimates needed
- Out-of-distribution detection
- Training stability priority
- Avoiding mode collapse essential

### EBMs vs VAEs: Flexibility vs Efficiency

| Aspect | Energy-Based Models | VAEs |
|--------|---------------------|------|
| **Architecture** | Unrestricted (any scalar-output network) | Encoder-decoder with bottleneck |
| **Likelihood** | Unnormalised; exact only in small / special structures, otherwise estimated or bounded because $Z$ is intractable | Tractable lower bound (ELBO) |
| **Sample quality** | Modern EqM matches diffusion (FID 1.90 on ImageNet 256²) | Blurry under L2 / Gaussian decoder; sharp once paired with adversarial / VQ losses (the SD-VAE recipe) |
| **Sampling speed** | Classical: MCMC (slow). Modern (EqM): a few gradient-descent steps | Single forward pass |
| **Training** | CD / PCD / DRL / Energy Matching | Stable (reparameterisation trick) |
| **Latent space** | No explicit latent (the energy *is* the model) | Structured, interpretable, supports interpolation |
| **Role in modern stacks** | Scoring, OOD detection, joint discriminative-generative | The *codec* feeding diffusion / flow priors |

**When to use EBMs over VAEs**:

- No need for learned latent representations
- Flexible unnormalised density more important than speed
- Want architectural flexibility
- Avoid reconstruction-based blur

### EBMs vs Diffusion Models: Principled Energy vs Iterative Denoising

| Aspect | Energy-Based Models | Diffusion Models |
|--------|---------------------|------------------|
| **Conceptual framework** | Stationary energy landscape; samples are fixed points | Time-dependent score field; samples are end of a denoising trajectory |
| **Training** | Classical CD/PCD: brittle. Modern Energy Matching / EqM / DRL: stable, simulation-free | Stable (MSE denoising) |
| **Sampling** | Classical: 50–200 MCMC steps. Modern (EqM): adaptive gradient descent, often 5–50 steps | 20–1000 steps un-distilled; 1–4 steps distilled |
| **Sample quality** | Modern EqM: FID 1.90 on ImageNet 256², surpassing several diffusion baselines | Leading across many image / video / 3D settings |
| **Likelihood** | Unnormalised; $Z$ is usually intractable | ELBO / probability-flow ODE estimates |
| **Architecture** | Any scalar-output network | DiT / U-Net / MMDiT (flexible but conventionally chosen) |
| **2026 status** | Energy Matching (NeurIPS 2025), EqM (2025), EDLM (ICLR 2025), ARM↔EBM bijection (2026) — actively competitive again | Mature; flow-matching DiTs in production (SD3, FLUX.1) |

!!! note "Three formal connections"
    1. **Score-based diffusion as time-indexed EBMs**: when $s_\theta(x, t)$ is a true score field, it can be written as $s_\theta(x, t) = -\nabla_x E_\theta(x, t)$ for an energy defined up to an additive constant — the score-SDE perspective ([Song et al., 2021](https://arxiv.org/abs/2011.13456)). An unconstrained vector field is not automatically conservative.
    2. **Diffusion Recovery Likelihood** ([Gao et al., 2021](https://arxiv.org/abs/2012.08125)) explicitly parameterises a *sequence* of EBMs at progressively noisier scales, recovering each from the next.
    3. **Energy Matching / Equilibrium Matching** invert the relationship: train a *time-invariant* energy whose gradient behaves like a denoising / flow-matching velocity field. The two paradigms share mathematical structure without becoming identical in likelihood, sampler, or modelling constraints.

---

## Recent Advances (2024–2026): Energy-Flow Methods and Implicit EBMs

Between 2014 and 2023, EBM research focused on *making MCMC work* (Improved CD, PCD variants, Diffusion Recovery Likelihood). Between 2024 and 2026, the field has pivoted toward **simulation-free energy training** that borrows the regression objectives of flow matching and the architectural freedom of diffusion. This section organises the headline contributions, grounded in the recent comprehensive survey [*Hitchhiker's Guide on the Relation of EBMs with Other Generative Models*](https://arxiv.org/abs/2406.13661) (updated May 2025) and the continuously-updated [Awesome-EBM](https://github.com/yataobian/awesome-ebm) reading list.

### Foundational Refinements (2021–2023)

Before discussing the 2024–2026 wave, three earlier pieces set the table:

- **Improved CD** ([Du, Li, Tenenbaum & Mordatch, 2021 — ICML](https://arxiv.org/abs/2012.01316)) — adds a KL-divergence term to address gradient bias, plus data augmentation, multi-scale processing, and reservoir-sampled buffers. Achieves Inception Score 8.30 on CIFAR-10.
- **Diffusion Recovery Likelihood (DRL)** ([Gao, Song, Poole, Wu & Kingma, 2021 — ICLR](https://arxiv.org/abs/2012.08125)) — trains a *sequence* of EBMs at progressively noisier scales, each one a tractable conditional that recovers the previous scale; bridges EBMs with diffusion.
- **Cooperative Diffusion Recovery Likelihood (CDRL)** ([Zhu et al., 2024 — ICLR](https://arxiv.org/abs/2309.05153)) — pairs each scale's EBM with an explicit initialiser model, dramatically improving CIFAR-10 / ImageNet generation while retaining tractable training.
- **Persistently Trained Diffusion-Assisted EBMs** ([Yu et al., 2023](https://arxiv.org/abs/2304.10707)) — combines persistent training with diffusion-assisted sampling for long-run stability, post-training generation, and superior OOD detection.
- **Stabilized JEM** ([Zhang et al., 2023](https://arxiv.org/abs/2303.04187)) — addresses JEM training instability via better initialisation, curriculum learning, and adaptive SGLD; first stable JEM training on ImageNet-scale data.

### Energy Matching: Unifying Flows and EBMs

**Energy Matching** ([Balcerak et al., 2025 — NeurIPS](https://arxiv.org/abs/2504.10612)) is the key conceptual unification of the 2024–2026 wave:

- Far from the data manifold, samples follow **irrotational, optimal-transport** paths from noise to data — like flow matching.
- Near the data manifold, an **entropic energy** term takes over and guides the system into a Boltzmann equilibrium — like a classical EBM.
- A *single scalar potential* parameterises both regimes; **no time conditioning, no auxiliary generators, no second network**.

The result substantially outperforms prior EBMs on CIFAR-10 and ImageNet generation while retaining simulation-free training away from the data manifold. The framework also supports interaction-energy terms for diverse-mode exploration, demonstrated on protein generation.

### Equilibrium Matching: Implicit Energy via Gradient Descent

**Equilibrium Matching (EqM)** ([Wang et al., 2025](https://arxiv.org/abs/2510.02300)) discards the time-conditional dynamics of diffusion / flow matching entirely:

- Learn a **time-invariant gradient field** $\nabla E_\theta(x)$ such that the data manifold is a set of equilibrium (zero-gradient) points.
- At inference, samples are produced by **gradient descent on the learned energy landscape** with *adjustable step sizes, adaptive optimisers, and adaptive compute* — replacing fixed-horizon ODE solvers with general optimisation.
- Achieves **FID 1.90 on ImageNet 256×256**, surpassing diffusion / flow models at matched compute.

EqM also handles partially-noised denoising, OOD detection, and image composition under a single unified framework — recovering the classical EBM "one model, many tasks" promise on modern benchmarks.

### Energy-Based Diffusion Language Models (EDLM)

**EDLM** ([NVIDIA, 2025 — ICLR](https://arxiv.org/abs/2410.21357)) attacks discrete-text diffusion's approximation gap by attaching a **full-sequence energy model** at each diffusion step:

- The standard discrete-diffusion factorisation $p(x_0 \mid x_t) \approx \prod_i p(x_0^{(i)} \mid x_t)$ breaks down for long sequences.
- EDLM replaces the per-token approximation with a **sequence-level energy** $E_\theta(x_0, x_t)$ trained to score full reconstructions, then samples via NCE-style contrastive training plus importance-weighted decoding.
- Closes a substantial chunk of the perplexity gap to autoregressive language models on PTB, OWT, and LM1B.

This places EBMs at the centre of the discrete-diffusion-LM line of work alongside LLaDA and Mercury — see the [Diffusion explainer](diffusion-explained.md#diffusion-language-models-20242026).

### ARMs are Secretly EBMs (2026)

**Autoregressive Language Models are Secretly Energy-Based Models** ([Wang et al., 2026](https://arxiv.org/abs/2512.15605)) establishes an explicit *bijection* between autoregressive models and EBMs in function space:

- The mapping corresponds to a special case of the **soft Bellman equation** in maximum-entropy reinforcement learning.
- Implications: post-training alignment (RLHF, DPO) becomes natural energy-shaping; ARMs gain principled lookahead through their EBM dual.

A companion 2026 paper ([*A Theoretical Lens for RL-Tuned LLMs via EBMs*](https://arxiv.org/abs/2512.18730)) develops the practical consequences for RLHF analysis.

### Maximum-Entropy IRL for Diffusion Models

**MaxEnt IRL for Diffusion** ([Anonymous, 2024 — NeurIPS](https://neurips.cc/virtual/2024/poster/94930)) interprets diffusion sample-quality fine-tuning as **inverse reinforcement learning with an EBM-shaped reward**:

- Especially helpful when the diffusion sampler uses few steps, where standard losses become harder to optimise.
- Connects EBMs back to the post-training-of-diffusion thread (DPO, DDPO, SiD).

### Practical Recipes for Modern EBMs

Reading across the 2024–2026 papers above, the consistent practical recipe is:

1. **Pre-train a flow-matching or diffusion model** on the same data as a strong initialiser.
2. **Add a scalar-potential head** (Energy Matching / EqM) or a **sequence-level energy** (EDLM) on top.
3. **Train with simulation-free regression** for the bulk of the data manifold and **short MCMC / gradient-descent steps** near it.
4. **Sample with adaptive optimisers** (Adam-like updates on $x$) rather than fixed ODE / SDE schedules — this is the EqM contribution and pairs cleanly with the production fast-inference stack.

This recipe sidesteps the 2014–2020 era's combination of long-run MCMC instability and short-MCMC gradient bias — *the* historical reason EBMs stayed niche.

---

## Evaluation Metrics

EBMs are evaluated on **density-estimation**, **sample-quality**, **out-of-distribution-detection**, and **representation-quality** axes:

### Density-Estimation Metrics

The partition function $Z_\theta$ makes *exact* likelihood intractable, so EBMs use *bounds and ratios*:

- **Annealed Importance Sampling (AIS)** — gold-standard estimator of $\log Z_\theta$ for small-to-moderate dimensions; reported as **bits per dimension** on images.
- **Reverse AIS** — provides upper-bound estimates; pair with AIS to bracket the true NLL.
- **NCE log-Z estimate** — when training with NCE, the learned $c = \log Z$ parameter is itself an estimate.
- **Energy-gap reporting** — $\mathbb{E}[E_\theta(x_\mathrm{data})] - \mathbb{E}[E_\theta(x_\mathrm{sample})]$; should be *positive and stable* during healthy training even when $Z$ is unknown.

### Sample-Quality Metrics

EBM sampling is unconditional by default; the standard image-generation metrics apply:

- **FID** ([Heusel et al., 2017](https://arxiv.org/abs/1706.08500)), **Inception Score**, **CMMD** ([Jayasumana et al., 2024](https://arxiv.org/abs/2401.09603)).
- Reference numbers from the [Recent Advances](#recent-advances-20242026-energy-flow-methods-and-implicit-ebms) section: **EqM achieves FID 1.90 on ImageNet 256²** ([Wang et al., 2025](https://arxiv.org/abs/2510.02300)); **CDRL** matches or surpasses diffusion baselines on CIFAR-10.
- For language, **EDLM** ([NVIDIA, 2025](https://arxiv.org/abs/2410.21357)) reports **perplexity** competitive with similar-scale autoregressive LMs.

### Out-of-Distribution Detection

EBMs *naturally* support OOD detection via the energy score, which has been a historic strength:

- **AUROC, AUPR** on standard CIFAR-10 vs SVHN / Textures / LSUN benchmarks. JEM and successors typically achieve **AUROC > 0.90**.
- **Likelihood-ratio scores** ([Ren et al., 2019](https://arxiv.org/abs/1906.02845)) — divide by the energy of a "background" model to fix the well-known counter-intuitive higher-likelihood-on-OOD problem (cf. [Nalisnick et al., 2019](https://arxiv.org/abs/1810.09136)).

### Representation Quality (JEM, conditional EBMs)

- **Linear probing** on top of the JEM classifier head.
- **Adversarial robustness** under PGD / AutoAttack — JEM and its successors are typically more robust than equivalent vanilla classifiers.
- **Calibration** (ECE — Expected Calibration Error) — JEM-class models calibrate better than softmax baselines.

---

## Production Considerations

### Inference Cost

EBMs have historically had bad reputations at inference because of MCMC. The 2024–2026 landscape splits into three regimes:

| Sampling regime | NFEs | 1024² latency on H100 |
| --- | --- | --- |
| Classical Langevin / SGLD | 50–200 | ~5–20 s (uncompetitive) |
| Diffusion-Recovery-Likelihood ladder | 50–200 | ~5–20 s |
| Equilibrium Matching gradient descent + Adam | 5–50 | ~200 ms – 2 s |
| Mean-Flow-style 1-step distilled EBM | 1 | ~40 ms |

Equilibrium Matching's **adaptive-optimiser sampling** is the key practical contribution: trade the rigidity of fixed ODE / SDE schedules for variable per-step costs and add early-termination criteria.

### Quantisation and Deployment

- **FP16 / BF16** universally safe for the energy network.
- **INT8 PTQ** is risky — small errors in the energy gradient compound over MCMC steps. Quantise *only* the forward pass and keep gradients in FP16.
- **Distillation** of EBMs into a single-step student is an active 2025–2026 research area (see [Equilibrium Matching's appendix on adaptive-NFE sampling](https://arxiv.org/abs/2510.02300)).

### Common Pitfalls in Production

- **Long-run sample collapse** — a classic EBM bug: samples slowly drift to a single low-energy mode after thousands of MCMC steps. Use a persistent buffer with periodic re-init (5 % per step) or switch to EqM-style adaptive sampling.
- **Energy explosion during fine-tuning** — add a $\lambda \mathbb{E}[E^2]$ regulariser ($\lambda \approx 0.01$) to keep the energy bounded.
- **OOD detection drift** — the AUROC on a held-out OOD set should be tracked as a *training metric*, not just a final-evaluation one.

For the broader unified picture and how EBMs fit alongside diffusion / flow / VAE / GAN / AR systems in 2026, see [Generative Models — A Unified View](generative-models-unified.md).

---

## Summary and Key Takeaways

Energy-Based Models provide a principled framework for learning probability distributions through energy functions. While training challenges have historically limited their adoption, recent advances in contrastive divergence and hybrid methods are revitalizing the field.

### Core Principles

<div class="grid cards" markdown>

- :material-lightning-bolt: **Energy Landscape**

    ---

    Model assigns scalar energy to configurations—low energy = high probability

- :material-chart-bell-curve: **Boltzmann Distribution**

    ---

    $p(x) \propto e^{-E(x)}$ connects energy to probability via exponential

- :material-sync: **MCMC Sampling**

    ---

    Generate samples by traversing energy landscape with Langevin dynamics

- :material-brain: **Flexible Architecture**

    ---

    Any neural network can define energy function—no structural constraints

</div>

### Training Methods

| Method | Sampling During Training | Pros | Cons |
|--------|-------------------------|------|------|
| **Contrastive Divergence** | Short MCMC chains (k=1-60) | Practical, widely used | Biased gradient estimates |
| **Persistent CD** | Persistent buffer + MCMC | Better gradients, stable | Requires careful buffer management |
| **Score Matching** | None | No sampling needed | Expensive second derivatives |
| **NCE** | Noise samples only | Simple, no MCMC | Requires good noise distribution |

### When to Use EBMs

**Best suited for**:

- **Density estimation** where exact likelihood matters
- **Out-of-distribution detection** leveraging $p(x)$
- **Anomaly detection** in high-stakes applications
- **Hybrid tasks** combining generation and classification (JEM)
- **Structured prediction** with complex dependencies

**Avoid when**:

- Real-time generation required (use GANs or fast flows)
- Training data limited (VAEs/diffusion more sample-efficient)
- Computational resources constrained at training time

### Future Directions (2026 and Beyond)

- **Energy-flow unification** — Energy Matching ([Balcerak et al., 2025](https://arxiv.org/abs/2504.10612)) and Equilibrium Matching ([Wang et al., 2025](https://arxiv.org/abs/2510.02300)) have shown that simulation-free EBM training can match diffusion FID; expect the next wave of large-scale generators to ship with energy heads.
- **EBM dual of LLMs** — the bijection between ARMs and EBMs ([Wang et al., 2026](https://arxiv.org/abs/2512.15605)) suggests that post-training alignment (RLHF, DPO) is most cleanly understood as energy shaping; new RLHF methods built on this duality are emerging.
- **Energy-based discrete diffusion for text** — EDLM ([NVIDIA, 2025](https://arxiv.org/abs/2410.21357)) closes the perplexity gap with autoregression on long sequences; expect this to extend to multimodal LLMs.
- **Unified one-model-many-tasks** — EqM's single landscape solves generation, OOD detection, denoising, and composition; reviving the classical EBM ambition.
- **Scientific applications** — protein design (Energy Matching), molecular generation, materials discovery, and physics-informed simulation continue to be where EBMs *naturally* dominate diffusion / flow alternatives.

---

## Next Steps

<div class="grid cards" markdown>

- :material-book-open-variant:{ .lg .middle } **[EBM User Guide](../models/ebm-guide.md)**

    ---

    Practical usage guide with implementation examples and training workflows

- :material-code-braces:{ .lg .middle } **[EBM API Reference](../../models/ebm.md)**

    ---

    Complete API documentation for EBM, DeepEBM, and JEM classes

- :material-school:{ .lg .middle } **[EBM Tutorial](../../examples/energy/simple-ebm.md)**

    ---

    Step-by-step hands-on tutorial: train an EBM on MNIST from scratch

- :material-flask:{ .lg .middle } **[Advanced Examples](../../examples/index.md)**

    ---

    Explore Joint Energy-Based Models, hybrid training, and applications

</div>

---

## Further Reading

### Seminal Papers (Must Read)

:material-file-document: **LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006).** "A Tutorial on Energy-Based Learning"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [Technical Report](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Foundational tutorial establishing the energy-based learning framework

:material-file-document: **Hinton, G. E. (2002).** "Training Products of Experts by Minimizing Contrastive Divergence"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [Neural Computation 14(8)](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Introduced Contrastive Divergence, making EBM training practical

:material-file-document: **Tieleman, T. (2008).** "Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [ICML 2008](https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Persistent Contrastive Divergence for improved training

:material-file-document: **Hyvärinen, A. (2005).** "Estimation of Non-Normalized Statistical Models by Score Matching"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [JMLR 6](https://jmlr.org/papers/v6/hyvarinen05a.html)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Score matching method avoiding partition function

:material-file-document: **Gutmann, M., & Hyvärinen, A. (2010).** "Noise-contrastive estimation: A new estimation principle for unnormalized statistical models"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [AISTATS 2010](http://proceedings.mlr.press/v9/gutmann10a.html)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: NCE training via binary classification

### Modern EBMs and Applications

:material-file-document: **Du, Y., & Mordatch, I. (2019).** "Implicit Generation and Modeling with Energy Based Models"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:1903.08689](https://arxiv.org/abs/1903.08689) | [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/378a063b8fdb1db941e34f4bde584c7d-Abstract.html)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Modern EBMs for image generation with improved training

:material-file-document: **Grathwohl, W., Wang, K.-C., Jacobsen, J.-H., Duvenaud, D., Norouzi, M., & Swersky, K. (2020).** "Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:1912.03263](https://arxiv.org/abs/1912.03263) | [ICLR 2020](https://openreview.net/forum?id=HpI7BkCuHr)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Joint Energy-Based Models unifying classification and generation

:material-file-document: **Du, Y., Li, S., Tenenbaum, J., & Mordatch, I. (2020).** "Improved Contrastive Divergence Training of Energy Based Models"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2012.01316](https://arxiv.org/abs/2012.01316) | [ICML 2021](http://proceedings.mlr.press/v139/du21b.html)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Addresses gradient bias in CD with KL regularization

:material-file-document: **Gao, R., Song, Y., Poole, B., Wu, Y. N., & Kingma, D. P. (2021).** "Learning Energy-Based Models by Diffusion Recovery Likelihood"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2012.08125](https://arxiv.org/abs/2012.08125) | [ICLR 2021](https://openreview.net/forum?id=v_1Soh8QUNc)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Connects EBMs with diffusion models via recovery likelihood

### Recent Advances (2023–2026)

:material-file-document: **Wu, D., Wang, L., & Hong, P. (2023).** "Training Energy-Based Models with Diffusion Contrastive Divergences"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2307.01668](https://arxiv.org/abs/2307.01668)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Replaces Langevin with general diffusion processes for the negative phase.

:material-file-document: **Zhu, Y., et al. (2024 — ICLR).** "Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood" (CDRL)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2309.05153](https://arxiv.org/abs/2309.05153)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Per-scale EBMs paired with explicit initialiser models; major CIFAR-10 / ImageNet improvements.

:material-file-document: **Yu, X., et al. (2023).** "Persistently Trained, Diffusion-Assisted Energy-Based Models"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2304.10707](https://arxiv.org/abs/2304.10707)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Combines persistent training with diffusion-assisted sampling; long-run stability + post-training generation.

:material-file-document: **Zhang, Y., et al. (2023).** "Stabilized Training of Joint Energy-based Models and their Practical Applications"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2303.04187](https://arxiv.org/abs/2303.04187)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: First stable JEM training on ImageNet-scale data.

:material-file-document: **(2024 — NeurIPS).** "Maximum Entropy Inverse Reinforcement Learning of Diffusion Models with Energy-Based Models"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [neurips.cc/virtual/2024/poster/94930](https://neurips.cc/virtual/2024/poster/94930)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Re-frames diffusion sample-quality fine-tuning as MaxEnt IRL with an EBM-shaped reward.

:material-file-document: **Balcerak, M., et al. (2025 — NeurIPS).** "Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2504.10612](https://arxiv.org/abs/2504.10612) | [github.com/m1balcerak/EnergyMatching](https://github.com/m1balcerak/EnergyMatching)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Single scalar potential — OT flow far from data, Boltzmann equilibrium near data; substantially outperforms prior EBMs.

:material-file-document: **Wang, R., et al. (2025).** "Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models" (EqM)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2510.02300](https://arxiv.org/abs/2510.02300) | [project page](https://raywang4.github.io/equilibrium_matching/)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Time-invariant gradient field; gradient-descent sampling; FID 1.90 on ImageNet 256² beats diffusion / flow.

:material-file-document: **(NVIDIA, 2025 — ICLR).** "Energy-Based Diffusion Language Models for Text Generation" (EDLM)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2410.21357](https://arxiv.org/abs/2410.21357)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Sequence-level energy at each diffusion step closes the AR-LLM perplexity gap on PTB / OWT / LM1B.

:material-file-document: **Wang, J., et al. (2026).** "Autoregressive Language Models are Secretly Energy-Based Models: Insights into the Lookahead Capabilities of Next-Token Prediction"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2512.15605](https://arxiv.org/abs/2512.15605)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Explicit bijection between ARMs and EBMs corresponding to the soft Bellman equation in MaxEnt RL.

:material-file-document: **(2026).** "A Theoretical Lens for RL-Tuned Language Models via Energy-Based Models"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2512.18730](https://arxiv.org/abs/2512.18730)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Uses the ARM↔EBM duality to analyse RLHF and DPO theoretically.

### Surveys

:material-file-document: **(2024–2025).** "Hitchhiker's Guide on the Relation of Energy-Based Models with Other Generative Models, Sampling and Statistical Physics: A Comprehensive Review"<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [arXiv:2406.13661](https://arxiv.org/abs/2406.13661)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Recent comprehensive review (updated May 2025) connecting EBMs with GANs, VAEs, flows, MCMC, and statistical physics.

:material-github: **Awesome-EBM**<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [github.com/yataobian/awesome-ebm](https://github.com/yataobian/awesome-ebm)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Continuously-updated curated list of energy / entropy-based-model papers.

:material-web: **Energy-Based-Model.github.io — Du & Mordatch's continuously-updated EBM portal**<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [energy-based-model.github.io](https://energy-based-model.github.io/)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Project list and code links.

### Tutorial Resources

:material-web: **NYU Deep Learning Course: Energy-Based Models**<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [atcold.github.io/NYU-DLSP20/en/week07/07-1](https://atcold.github.io/NYU-DLSP20/en/week07/07-1/)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Complete lectures by Yann LeCun and Alfredo Canziani

:material-web: **UvA Deep Learning Tutorial 8: Deep Energy-Based Generative Models**<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [uvadlc-notebooks.readthedocs.io](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Hands-on Colab notebooks with implementations

:material-web: **Stanford CS236: Deep Generative Models (EBM Lecture)**<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [Lecture Slides](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture11.pdf)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Academic course materials covering EBM theory

:material-github: **Energy-Based Models GitHub Repositories**<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-link: [github.com/yilundu/improved_contrastive_divergence](https://github.com/yilundu/improved_contrastive_divergence)<br>
&nbsp;&nbsp;&nbsp;&nbsp;:material-lightbulb-outline: Official implementation of Improved CD

---

**Ready to explore Energy-Based Models?** Start with the [EBM User Guide](../models/ebm-guide.md) for practical implementations, check the [API Reference](../../api/models/ebm.md) for complete documentation, or run the [Simple EBM Example](../../examples/energy/simple-ebm.md) for a retained working walkthrough.
