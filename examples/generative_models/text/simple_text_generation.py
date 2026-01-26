# %% [markdown]
r"""
# Simple Text Generation with Character-Level Models

This example demonstrates text generation using character-level language modeling
with JAX/Flax NNX. Learn how to build a simple recurrent text generator that
processes sequences one character at a time.

## Learning Objectives

- [ ] Understand character-level text generation fundamentals
- [ ] Implement embedding layers for character representations
- [ ] Build recurrent-style networks with sequential processing
- [ ] Apply temperature-based sampling for generation diversity
- [ ] Handle variable-length sequence generation

## Prerequisites

- Basic understanding of neural networks
- Familiarity with sequence modeling concepts
- Knowledge of JAX/Flax NNX module patterns
- Understanding of text tokenization basics

## Key Concepts

### Character-Level Language Modeling

Character-level models process text one character at a time, making them simple
yet effective for learning text patterns:

$$P(x_t | x_{<t}) = \\text{softmax}(f_\\theta(x_{<t}))$$

where $x_t$ is the character at position $t$ and $f_\\theta$ is the neural network.

### Temperature Sampling

Temperature controls generation randomness:

$$P_T(x_t) = \\frac{\\exp(z_t / T)}{\\sum_i \\exp(z_i / T)}$$

- Low temperature ($T < 1$): More deterministic, conservative
- High temperature ($T > 1$): More random, creative
"""

# %%
#!/usr/bin/env python
"""Simple text generation example using the Artifex framework.

This example demonstrates basic text generation using character-level
language modeling with JAX/Flax.

Source Code Dependencies:
    - flax.nnx: Neural network modules (Embed, Linear, Sequential)
    - jax.numpy: Array operations
    - jax.random: Sampling operations
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


# %% [markdown]
"""
## Model Architecture

The `SimpleTextGenerator` implements a character-level language model with:

1. **Embedding Layer**: Maps character IDs to dense vectors
2. **Recurrent Network**: Processes sequences position-by-position
3. **Output Projection**: Maps hidden states to vocabulary logits

This architecture demonstrates fundamental sequence processing patterns
while remaining simple enough to understand and modify.
"""


# %%
class SimpleTextGenerator(nnx.Module):
    """Simple character-level text generator."""

    def __init__(
        self,
        vocab_size: int = 128,  # ASCII characters
        embed_dim: int = 64,
        hidden_dim: int = 128,
        seq_length: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the text generator.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            seq_length: Maximum sequence length
            rngs: Random number generators
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length

        # Embedding layer
        self.embedding = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)

        # Simple RNN-like network (using dense layers for simplicity)
        self.rnn = nnx.Sequential(
            nnx.Linear(embed_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, vocab_size, rngs=rngs),
        )

        # Output projection
        self.output_layer = nnx.Linear(vocab_size, vocab_size, rngs=rngs)

    def __call__(self, input_ids):
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            Logits for next token prediction [batch, seq_len, vocab_size]
        """
        # Embed input tokens
        x = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

        # Process through RNN-like network
        batch_size, seq_len, _ = x.shape
        outputs = []

        for i in range(seq_len):
            # Process each position
            h = self.rnn(x[:, i, :])  # [batch, vocab_size]
            outputs.append(h)

        # Stack outputs
        logits = jnp.stack(outputs, axis=1)  # [batch, seq_len, vocab_size]

        # Apply output layer
        logits = self.output_layer(logits)

        return logits

    def generate(
        self, prompt: str = "", max_length: int = 100, temperature: float = 1.0, *, rngs: nnx.Rngs
    ):
        """Generate text from a prompt.

        Args:
            prompt: Starting text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            rngs: Random number generators

        Returns:
            Generated text string
        """
        # Convert prompt to token IDs (simple ASCII encoding)
        if prompt:
            input_ids = jnp.array([ord(c) % self.vocab_size for c in prompt])
        else:
            # Start with a random character
            key = rngs.sample()
            input_ids = jax.random.randint(key, (1,), 0, self.vocab_size)

        generated = list(input_ids)

        for _ in range(max_length - len(generated)):
            # Prepare input (pad or truncate to seq_length)
            current_seq = jnp.array(generated[-self.seq_length :])
            if len(current_seq) < self.seq_length:
                # Pad with zeros
                padding = jnp.zeros(self.seq_length - len(current_seq), dtype=jnp.int32)
                current_seq = jnp.concatenate([padding, current_seq])

            # Get predictions
            logits = self(current_seq[None, :])  # Add batch dimension

            # Sample next token
            next_logits = logits[0, -1, :] / temperature
            key = rngs.sample()
            next_token = jax.random.categorical(key, next_logits)

            generated.append(int(next_token))

            # Stop if we generate a newline (optional)
            if next_token == ord("\n"):
                break

        # Convert back to text
        text = "".join([chr(t % 128) for t in generated])
        return text


# %% [markdown]
"""
## Training Data Creation

For demonstration purposes, create simple repetitive text patterns
that allow the model to learn basic character relationships quickly.
"""


# %%
def create_training_data():
    """Create simple training data for demonstration."""
    # Simple repetitive patterns for easy learning
    patterns = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a text generation example.",
        "JAX and Flax make neural networks easy.",
        "Machine learning with Python is fun.",
        "Generative models can create text.",
    ]

    # Repeat patterns to create more data
    text = " ".join(patterns * 10)

    # Convert to token IDs (simple ASCII encoding)
    token_ids = jnp.array([ord(c) % 128 for c in text])

    return text, token_ids


# %% [markdown]
"""
## Text Generation Demonstration

This section demonstrates the text generation capabilities including:

- Model initialization with RNG handling
- Forward pass testing with batched sequences
- Temperature-based sampling with various prompts
- Batch processing for multiple sequences

Each generation uses different temperature values to show how this
parameter affects output diversity.
"""


# %%
def demonstrate_text_generation():
    """Demonstrate text generation capabilities."""
    print("=" * 60)
    print("Simple Text Generation Example")
    print("=" * 60)

    # Set random seed
    seed = 42
    key = jax.random.key(seed)
    rngs = nnx.Rngs(params=key, sample=key)

    # Create text generator
    print("\nCreating text generator...")
    generator = SimpleTextGenerator(
        vocab_size=128, embed_dim=64, hidden_dim=128, seq_length=32, rngs=rngs
    )

    print(f"Vocabulary size: {generator.vocab_size}")
    print(f"Embedding dimension: {generator.embed_dim}")
    print(f"Hidden dimension: {generator.hidden_dim}")
    print(f"Sequence length: {generator.seq_length}")

    # Create training data
    print("\nCreating training data...")
    text_data, token_ids = create_training_data()
    print(f"Training text length: {len(text_data)} characters")
    print(f"Sample text: '{text_data[:50]}...'")

    # Test forward pass
    print("\nTesting forward pass...")
    # Create a batch of sequences
    seq_len = 16
    batch_size = 4
    test_input = token_ids[: batch_size * seq_len].reshape(batch_size, seq_len)
    logits = generator(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Generate text with different temperatures
    print()
    print("=" * 40)
    print("Text Generation Examples")
    print("=" * 40)

    prompts = ["The ", "Hello ", "Machine ", ""]
    temperatures = [0.5, 0.8, 1.0, 1.5]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        for temp in temperatures:
            # Create new RNG for each generation
            gen_key = jax.random.key(np.random.randint(0, 10000))
            gen_rngs = nnx.Rngs(sample=gen_key)

            generated = generator.generate(
                prompt=prompt, max_length=50, temperature=temp, rngs=gen_rngs
            )

            # Clean up output for display
            generated_clean = generated.replace("\n", " ").replace("\r", " ")
            if len(generated_clean) > 60:
                generated_clean = generated_clean[:60] + "..."

            print(f"  Temp {temp:.1f}: {generated_clean}")

    # Demonstrate batch generation
    print()
    print("=" * 40)
    print("Batch Processing Example")
    print("=" * 40)

    # Process multiple sequences
    batch_prompts = ["The ", "Hello ", "JAX "]
    batch_ids = []

    for prompt in batch_prompts:
        ids = [ord(c) % 128 for c in prompt]
        # Pad to seq_length
        if len(ids) < seq_len:
            ids = [0] * (seq_len - len(ids)) + ids
        else:
            ids = ids[-seq_len:]
        batch_ids.append(ids)

    batch_input = jnp.array(batch_ids)
    batch_logits = generator(batch_input)

    print(f"\nBatch input shape: {batch_input.shape}")
    print(f"Batch output shape: {batch_logits.shape}")

    # Get predictions for next character
    next_char_logits = batch_logits[:, -1, :]
    next_chars = jnp.argmax(next_char_logits, axis=-1)

    print("\nNext character predictions:")
    for i, (prompt, next_id) in enumerate(zip(batch_prompts, next_chars)):
        next_char = chr(int(next_id) % 128)
        print(f"  '{prompt}' -> '{next_char}'")

    print("\nSimple text generation example completed successfully!")


# %% [markdown]
"""
## Summary and Key Takeaways

This example demonstrated fundamental text generation concepts:

### What We Learned

1. **Character-Level Modeling**: Process text one character at a time
2. **Embedding Layers**: Map discrete tokens to continuous representations
3. **Sequential Processing**: Handle variable-length sequences
4. **Temperature Sampling**: Control generation diversity
5. **Batch Processing**: Efficient multi-sequence handling

### Experiments to Try

1. **Architecture Modifications**:
   - Increase embedding and hidden dimensions
   - Add more RNN layers or use LSTM/GRU
   - Implement attention mechanisms

2. **Training Improvements**:
   - Use real training data from text corpora
   - Implement proper training loop with optimization
   - Add regularization (dropout, weight decay)

3. **Generation Techniques**:
   - Try different sampling strategies (top-k, nucleus)
   - Implement beam search for better quality
   - Add length penalties and repetition controls

4. **Advanced Features**:
   - Switch from character to word/subword tokenization
   - Implement conditional generation
   - Add style control mechanisms

### Next Steps

Explore related examples:
- **Transformer Models**: Modern attention-based architectures
- **Text Compression**: Information-theoretic approaches
- **Sequence-to-Sequence**: Translation and summarization tasks
"""


# %%
def main():
    """Run the text generation example."""
    demonstrate_text_generation()


# %%
if __name__ == "__main__":
    main()
