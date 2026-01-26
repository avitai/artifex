# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: py:percent,ipynb
#     notebook_metadata_filter: all,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.0
# ---

# %% [markdown]
# # [Example Title]
#
# **Duration:** [X minutes] | **Level:** [Beginner/Intermediate/Advanced]
# | **GPU Required:** [Yes/No]
#
# ## 🎯 Learning Objectives
#
# By the end of this notebook, you will:
# 1. Understand [concept 1]
# 2. Be able to [skill 1]
# 3. Know how to [application 1]
# 4. Gain intuition about [principle 1]
#
# ## 📋 Table of Contents
#
# 1. [Introduction](#introduction)
# 2. [Setup](#setup)
# 3. [Core Concepts](#concepts)
# 4. [Implementation](#implementation)
# 5. [Experiments](#experiments)
# 6. [Exercises](#exercises)
# 7. [Summary](#summary)
#
# ## ℹ️ Prerequisites
#
# - Basic Python knowledge
# - Understanding of [specific concepts]
# - Artifex installed (see setup instructions below)
# - [Any optional background knowledge]
#
# ---

# %%
# Cell 2: Environment Setup and Verification
"""
This cell checks that all required packages are installed and working correctly.
Run this first to catch any environment issues early.
"""

import sys
from pathlib import Path


# Check Python version
assert sys.version_info >= (3, 9), "Python 3.9+ required"
print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")

# Import and verify JAX
try:
    import jax
    import jax.numpy as jnp

    print(f"✅ JAX version: {jax.__version__}")
    print(f"✅ JAX backend: {jax.default_backend()}")
    print(f"✅ Available devices: {jax.device_count()}")
except ImportError as e:
    print(f"❌ JAX import failed: {e}")
    print("Please run: source activate.sh")

# Import Flax NNX
try:
    from flax import nnx

    print("✅ Flax NNX imported successfully")
except ImportError as e:
    print(f"❌ Flax import failed: {e}")

# Import Artifex
try:
    from artifex.generative_models.core.device_manager import DeviceManager

    print("✅ Artifex imported successfully")
except ImportError as e:
    print(f"❌ Artifex import failed: {e}")
    print("Make sure Artifex is installed: pip install -e .")

print("\n🎉 Environment setup complete!")

# %% [markdown]
# <a id="introduction"></a>
# ## 1. Introduction
#
# [3-5 paragraphs explaining the example in depth. Cover:]
#
# **What problem does this solve?**
#
# [Explain the problem context, real-world applications, and why this matters.]
#
# **Why is this approach interesting?**
#
# [Discuss the key innovations, advantages over alternatives, and theoretical foundations.]
#
# **Where is this used in practice?**
#
# [Provide examples of practical applications, research use cases, and industry adoption.]
#
# ### 📚 Background
#
# [Educational content with mathematical foundations if appropriate. For example:]
#
# The key mathematical principle behind this approach is:
#
# $$
# \mathcal{L} = \mathbb{E}_{x \sim p(x)} [\text{loss}(x)]
# $$
#
# [Explain each component of the equation]
#
# ### 🔑 Key Concepts
#
# - **Concept 1**: [Explanation with intuition, not just definition]
# - **Concept 2**: [How it relates to Concept 1]
# - **Concept 3**: [Common pitfalls and how to avoid them]
#
# ---

# %% [markdown]
# <a id="setup"></a>
# ## 2. Setup
#
# Let's set up our environment and define configuration parameters.

# %%
# Cell: Import all required dependencies
"""
📖 WHAT THIS CELL DOES:
Imports all libraries and modules needed for the example.

🎓 KEY CONCEPTS:
- JAX provides automatic differentiation and GPU acceleration
- Flax NNX is the neural network framework we use
- Artifex provides high-level abstractions for generative models
"""

# Standard library
from typing import Tuple

# JAX and numerical computing
import jax
import jax.numpy as jnp

# Visualization
import matplotlib.pyplot as plt
from flax import nnx
from IPython.display import display, Markdown

# Artifex core
from artifex.generative_models.core.device_manager import DeviceManager


# Progress tracking

print("✅ All imports successful!")

# %%
# Cell: Configuration parameters
"""
📖 WHAT THIS CELL DOES:
Defines all hyperparameters and configuration for the example.

💡 TRY EXPERIMENTING:
- Change LATENT_DIM to see how it affects model capacity
- Adjust LEARNING_RATE and observe training dynamics
- Try different HIDDEN_DIMS architectures
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Model architecture
INPUT_DIM = (28, 28, 1)  # Example: MNIST images
HIDDEN_DIMS = [256, 128, 64]
LATENT_DIM = 32

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

# Output settings
OUTPUT_DIR = Path("outputs/example_notebook")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(
    Markdown(f"""
### Configuration Summary

- **Input dimensions**: {INPUT_DIM}
- **Latent dimension**: {LATENT_DIM}
- **Hidden layers**: {HIDDEN_DIMS}
- **Batch size**: {BATCH_SIZE}
- **Learning rate**: {LEARNING_RATE}
- **Training epochs**: {NUM_EPOCHS}
""")
)

# %% [markdown]
# <a id="concepts"></a>
# ## 3. Core Concepts
#
# Before diving into implementation, let's understand the key concepts in depth.
#
# ### Concept 1: [Name]
#
# [Educational explanation with diagrams or equations if helpful]
#
# **Why this matters:**
# [Explain the significance and practical implications]
#
# **Common pitfalls:**
# - [Pitfall 1 and how to avoid it]
# - [Pitfall 2 and how to avoid it]
#
# ### Concept 2: [Name]
#
# [Build on Concept 1, showing connections]
#
# **Implementation details:**
# - [Detail 1]
# - [Detail 2]
#
# ---

# %% [markdown]
# <a id="implementation"></a>
# ## 4. Implementation
#
# Now let's implement the model step by step.

# %%
# Cell: Initialize environment
"""
📖 WHAT THIS CELL DOES:
Sets up the random number generator and device manager.

🎓 KEY CONCEPTS:
- RNG streams ensure reproducibility
- DeviceManager handles GPU/CPU automatically
- JAX uses functional random number generation
"""

# Initialize RNG
rngs = nnx.Rngs(RANDOM_SEED)

# Setup device manager
device_manager = DeviceManager()
device = device_manager.get_device()

display(
    Markdown(f"""
### Environment Status

- **Device**: {device}
- **Backend**: {jax.default_backend()}
- **Device count**: {jax.device_count()}
- **Random seed**: {RANDOM_SEED}
""")
)

# %%
# Cell: Create sample data
"""
📖 WHAT THIS CELL DOES:
Generates synthetic data for demonstration.

💡 TRY EXPERIMENTING:
- Change num_samples to see effect on training
- Visualize the data distribution
"""

num_samples = 1000

# Generate random data
if "sample" in rngs:
    key = rngs.sample()
else:
    key = jax.random.key(0)

data = jax.random.uniform(key, shape=(num_samples, *INPUT_DIM))

print(f"✅ Generated {num_samples} samples")
print(f"   Shape: {data.shape}")
print(f"   Range: [{data.min():.3f}, {data.max():.3f}]")

# Visualize a few samples
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(data[i, :, :, 0], cmap="gray")
    ax.axis("off")
    ax.set_title(f"Sample {i + 1}")
plt.tight_layout()
plt.show()

# %%
# Cell: Define model architecture
"""
📖 WHAT THIS CELL DOES:
Defines the model class following Artifex/Flax NNX patterns.

🎓 KEY CONCEPTS:
- Always call super().__init__() first
- Use rngs parameter for initialization
- No rngs in __call__ for standard modules
- Use nnx activation functions
"""


class ExampleModel(nnx.Module):
    """Example model demonstrating Artifex patterns."""

    def __init__(
        self,
        input_dim: Tuple[int, ...],
        hidden_dims: list[int],
        output_dim: int,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize the example model.

        Args:
            input_dim: Shape of input data (e.g., (28, 28, 1) for MNIST)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            rngs: Random number generators for initialization
            dtype: Data type for model parameters
        """
        # ALWAYS call super().__init__()
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        flat_input = int(jnp.prod(jnp.array(input_dim)))
        layers = []
        prev_dim = flat_input

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs, dtype=dtype))
            prev_dim = hidden_dim

        layers.append(nnx.Linear(prev_dim, output_dim, rngs=rngs, dtype=dtype))

        self.layers = layers

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Forward pass through the model.

        Args:
            x: Input data array
            deterministic: Whether to run in deterministic mode (for dropout, etc.)

        Returns:
            Output array after forward pass
        """
        # Flatten input
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Forward pass
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)  # Use nnx.relu, not jax.nn.relu

        x = self.layers[-1](x)
        return x


# Create model
model = ExampleModel(
    input_dim=INPUT_DIM,
    hidden_dims=HIDDEN_DIMS,
    output_dim=LATENT_DIM,
    rngs=rngs,
)

print(f"✅ Model created with {LATENT_DIM}-dimensional output")

# %% [markdown]
# ### Understanding the Model Architecture
#
# The model we just defined has the following structure:
#
# ```
# Input → Flatten → Hidden Layers → Output
# ```
#
# **Key design decisions:**
#
# 1. **Input flattening**: We flatten the input to work with dense layers
# 2. **ReLU activations**: Non-linearity between layers for learning complex patterns
# 3. **No activation on output**: Allows unbounded outputs
#
# ---

# %% [markdown]
# <a id="experiments"></a>
# ## 5. Experiments
#
# Let's run some experiments to understand model behavior.

# %%
# Cell: Test forward pass
"""
📖 WHAT THIS CELL DOES:
Tests that the model can process a batch of data.

💡 TRY EXPERIMENTING:
- Try different batch sizes
- Observe output statistics
"""

# Get a small batch
test_batch = data[:BATCH_SIZE]

# Forward pass
output = model(test_batch)

print("✅ Forward pass successful!")
print(f"   Input shape: {test_batch.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
print(f"   Output mean: {output.mean():.3f}")
print(f"   Output std: {output.std():.3f}")

# %% [markdown]
# <a id="exercises"></a>
# ## 6. Exercises
#
# Now it's your turn to experiment!

# %% [markdown]
# ### 🏋️ Exercise 1: Modify Architecture
#
# **Difficulty:** ⭐⭐☆☆☆
#
# **Goal:** Create a model with a different architecture and compare outputs.
#
# **Hints:**
# - Try adding more layers or changing layer sizes
# - Compare output statistics with the original model
# - Think about how architecture affects capacity
#
# **Your code here:**

# %%
# Exercise 1: Your solution

# TODO: Create a model with different HIDDEN_DIMS
# TODO: Run forward pass and compare with original


# %% [markdown]
# <details>
# <summary>💡 Click to see solution</summary>
#
# ```python
# # Exercise 1 Solution
# new_hidden_dims = [512, 256, 128, 64]
#
# model_deeper = ExampleModel(
#     input_dim=INPUT_DIM,
#     hidden_dims=new_hidden_dims,
#     output_dim=LATENT_DIM,
#     rngs=nnx.Rngs(RANDOM_SEED),
# )
#
# output_deeper = model_deeper(test_batch)
# print(f"Deeper model output shape: {output_deeper.shape}")
# print(f"Deeper model output range: [{output_deeper.min():.3f}, {output_deeper.max():.3f}]")
# ```
# </details>

# %% [markdown]
# ### 🏋️ Exercise 2: [Another Exercise]
#
# **Difficulty:** ⭐⭐⭐☆☆
#
# **Goal:** [Describe the goal]
#
# **Hints:**
# - [Hint 1]
# - [Hint 2]
#
# **Your code here:**

# %%
# Exercise 2: Your solution


# %% [markdown]
# <a id="summary"></a>
# ## 7. Summary and Next Steps
#
# Congratulations on completing this example!
#
# ### 🎓 What You Learned
#
# - ✅ [Key learning 1 with brief explanation]
# - ✅ [Key learning 2 with brief explanation]
# - ✅ [Key learning 3 with brief explanation]
# - ✅ [Practical skill gained]
#
# ### 🔬 Further Experiments
#
# Try these modifications to deepen your understanding:
#
# 1. **Experiment 1**: Change [parameter X] and observe [effect Y]
#    - Expected outcome: [Description]
#    - Why this matters: [Explanation]
#
# 2. **Experiment 2**: Implement [extension Z]
#    - Hints: [Where to start]
#    - Resources: [Links to relevant docs]
#
# 3. **Experiment 3**: Compare with [alternative approach W]
#    - What to compare: [Metrics or behaviors]
#    - Expected insights: [What you'll learn]
#
# ### 📚 Additional Resources
#
# **Artifex Documentation:**
# - [Core concepts](link)
# - [API reference](link)
# - [Best practices](link)
#
# **Research Papers:**
# - [Paper 1 title] - [Brief description and link]
# - [Paper 2 title] - [Brief description and link]
#
# **Related Examples:**
# - `[example1].ipynb` - For learning about [topic]
# - `[example2].ipynb` - For advanced [technique]
# - `[example3].ipynb` - For [application domain]
#
# ### 🐛 Troubleshooting
#
# **Problem:** [Common issue 1]
# - **Symptoms:** [What you'll see]
# - **Solution:** [How to fix it]
# - **Prevention:** [How to avoid it]
#
# **Problem:** [Common issue 2]
# - **Symptoms:** [What you'll see]
# - **Solution:** [How to fix it]
# - **Prevention:** [How to avoid it]
#
# ### 💬 Feedback
#
# Found a bug or have suggestions for improving this example? Please open an issue on our [GitHub repository](https://github.com/avitai/artifex)!
#
# ---
#
# **Author:** Artifex Team
# **Last Updated:** [YYYY-MM-DD]
# **License:** MIT
#
# ---
#
# ### 🎉 Thank you for completing this example!
#
# Continue your learning journey with more Artifex examples and tutorials.
