# Simple Text Generation with Character-Level Models

<div class="example-badges">
<span class="badge badge-beginner">Beginner</span>
<span class="badge badge-runtime-5min">Runtime: ~5min</span>
<span class="badge badge-format-dual">📓 Dual Format</span>
</div>

## Files

- **Python Script**: [`simple_text_generation.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/text/simple_text_generation.py)
- **Jupyter Notebook**: [`simple_text_generation.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/text/simple_text_generation.ipynb)

## Quick Start

```bash
# Run the Python script
uv run python examples/generative_models/text/simple_text_generation.py

# Or open the Jupyter notebook
jupyter lab examples/generative_models/text/simple_text_generation.ipynb
```

## Overview

This example demonstrates fundamental text generation using character-level language modeling. Learn how to build a simple recurrent text generator that processes sequences one character at a time, implementing the basic building blocks of more sophisticated language models.

### Learning Objectives

After completing this example, you will understand:

- [ ] Character-level text generation fundamentals
- [ ] How to implement embedding layers for character representations
- [ ] Building recurrent-style networks with sequential processing
- [ ] Temperature-based sampling for generation diversity
- [ ] Handling variable-length sequence generation

### Prerequisites

- Basic understanding of neural networks
- Familiarity with sequence modeling concepts
- Knowledge of JAX/Flax NNX module patterns
- Understanding of text tokenization basics

## Theory

### Character-Level Language Modeling

Character-level language models process text one character at a time, making them simpler but often less powerful than word-level or subword models. However, they offer several advantages:

1. **Small Vocabulary**: Only need to represent ~128 ASCII characters
2. **No Unknown Tokens**: Can process any text without special handling
3. **Morphology Learning**: Can learn word formation patterns
4. **Simplicity**: Easy to implement and understand

The model learns the conditional probability distribution:

$$P(x_t | x_{<t}) = \text{softmax}(f_\theta(x_{<t}))$$

where $x_t$ is the character at position $t$ and $f_\theta$ is the neural network that encodes the context $x_{<t}$.

### Model Architecture

The `SimpleTextGenerator` consists of three main components:

1. **Embedding Layer**: Maps discrete character IDs to continuous vector representations
   - Input: Character IDs $\in \{0, ..., 127\}$
   - Output: Dense vectors $\in \mathbb{R}^{d}$

2. **Recurrent Network**: Processes sequences position-by-position
   - Uses a simple feedforward architecture for demonstration
   - In practice, LSTM/GRU layers would be more effective
   - Maintains hidden states across sequence positions

3. **Output Projection**: Maps hidden states to vocabulary logits
   - Produces probability distribution over next characters
   - Uses softmax activation for normalization

### Temperature Sampling

Temperature is a hyperparameter that controls generation diversity:

$$P_T(x_t) = \frac{\exp(z_t / T)}{\sum_i \exp(z_i / T)}$$

where $z_t$ are the logits and $T$ is the temperature.

- **Low temperature** ($T < 1$): Sharpens distribution, more deterministic
- **Temperature = 1**: Standard softmax distribution
- **High temperature** ($T > 1$): Flattens distribution, more random

## Code Walkthrough

### 1. Model Definition

The `SimpleTextGenerator` class implements the core architecture:

```python
class SimpleTextGenerator(nnx.Module):
    """Simple character-level text generator."""

    def __init__(
        self,
        vocab_size: int = 128,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        seq_length: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        # Embedding layer maps character IDs to vectors
        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=embed_dim,
            rngs=rngs
        )

        # Simple RNN-like network
        self.rnn = nnx.Sequential(
            nnx.Linear(embed_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, vocab_size, rngs=rngs),
        )

        # Output projection
        self.output_layer = nnx.Linear(
            vocab_size, vocab_size, rngs=rngs
        )
```

Key implementation details:

- Uses `nnx.Embed` for efficient character embedding lookup
- `nnx.Sequential` chains layers for compact definition
- ReLU activations introduce non-linearity
- All layers require `rngs` parameter for initialization

### 2. Forward Pass

The forward pass processes input sequences:

```python
def __call__(self, input_ids):
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
```

This implementation:

- Processes each position independently (simplified RNN)
- Accumulates outputs across sequence length
- Returns logits for next-character prediction at each position

### 3. Text Generation

The generation method implements autoregressive sampling:

```python
def generate(
    self,
    prompt: str = "",
    max_length: int = 100,
    temperature: float = 1.0,
    *,
    rngs: nnx.Rngs
):
    # Convert prompt to token IDs
    if prompt:
        input_ids = jnp.array([ord(c) % self.vocab_size for c in prompt])
    else:
        # Start with random character
        key = rngs.sample()
        input_ids = jax.random.randint(key, (1,), 0, self.vocab_size)

    generated = list(input_ids)

    for _ in range(max_length - len(generated)):
        # Prepare input (pad or truncate to seq_length)
        current_seq = jnp.array(generated[-self.seq_length:])
        if len(current_seq) < self.seq_length:
            padding = jnp.zeros(
                self.seq_length - len(current_seq),
                dtype=jnp.int32
            )
            current_seq = jnp.concatenate([padding, current_seq])

        # Get predictions
        logits = self(current_seq[None, :])

        # Sample next token with temperature
        next_logits = logits[0, -1, :] / temperature
        key = rngs.sample()
        next_token = jax.random.categorical(key, next_logits)

        generated.append(int(next_token))

    # Convert back to text
    text = "".join([chr(t % 128) for t in generated])
    return text
```

Key features:

- Handles variable-length prompts with padding
- Uses sliding window of last `seq_length` characters
- Temperature scaling before sampling
- Autoregressive generation (feeds outputs back as inputs)

### 4. Training Data Creation

Creates simple patterns for demonstration:

```python
def create_training_data():
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
```

### 5. Demonstration

The main demonstration shows:

1. **Model Creation**: Initialize with RNG handling
2. **Forward Pass Testing**: Batch processing of sequences
3. **Temperature Sampling**: Generate with different temperatures
4. **Batch Processing**: Handle multiple prompts efficiently

## Experiments to Try

### 1. Architecture Modifications

**Increase Model Capacity**:

```python
generator = SimpleTextGenerator(
    vocab_size=128,
    embed_dim=128,      # Increased from 64
    hidden_dim=256,     # Increased from 128
    seq_length=64,      # Increased from 32
    rngs=rngs
)
```

**Add More Layers**:

- Stack multiple RNN layers
- Implement LSTM or GRU cells
- Add residual connections

### 2. Training Improvements

**Implement Proper Training**:

```python
# Add optimizer (wrt=nnx.Param required in NNX 0.11.0+)
optimizer = nnx.Optimizer(generator, optax.adam(1e-3), wrt=nnx.Param)

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        def loss_fn(generator):
            # Forward pass
            logits = generator(batch['input_ids'])

            # Compute loss
            return optax.softmax_cross_entropy_with_integer_labels(
                logits[:, :-1, :],
                batch['target_ids'][:, 1:]
            ).mean()

        # Compute gradients and update
        loss, grads = nnx.value_and_grad(loss_fn)(generator)
        optimizer.update(generator, grads)
```

**Data Augmentation**:

- Use real text corpora (Wikipedia, books)
- Implement dynamic sequence length
- Add noise for robustness

### 3. Generation Techniques

**Top-k Sampling**:

```python
def top_k_sample(logits, k=5):
    top_k_logits, top_k_indices = jax.lax.top_k(logits, k)
    probs = jax.nn.softmax(top_k_logits)
    sampled_idx = jax.random.categorical(key, jnp.log(probs))
    return top_k_indices[sampled_idx]
```

**Nucleus (Top-p) Sampling**:

```python
def nucleus_sample(logits, p=0.9):
    probs = jax.nn.softmax(logits)
    sorted_probs = jnp.sort(probs)[::-1]
    cumsum = jnp.cumsum(sorted_probs)
    cutoff = sorted_probs[jnp.searchsorted(cumsum, p)]
    # Sample from probs >= cutoff
```

**Beam Search**:

- Maintain top-k hypotheses
- Score based on cumulative probability
- Return highest-scoring sequence

### 4. Advanced Features

**Conditional Generation**:

```python
class ConditionalTextGenerator(nnx.Module):
    def __init__(self, num_conditions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_embed = nnx.Embed(
            num_embeddings=num_conditions,
            features=self.hidden_dim,
            rngs=kwargs['rngs']
        )

    def __call__(self, input_ids, condition):
        x = self.embedding(input_ids)
        c = self.condition_embed(condition)
        # Combine input and condition
        ...
```

**Byte-Pair Encoding**:

- Use subword tokenization for better vocabulary
- Implement BPE/WordPiece tokenizer
- Balance vocabulary size and coverage

## Next Steps

<div class="grid cards" markdown>

- :material-transformer: **Transformer Models**

    ---

    Explore modern attention-based architectures that revolutionized NLP

    [:octicons-arrow-right-24: Transformer Examples](../basic/transformer-text.md)

- :material-zip-box: **Text Compression**

    ---

    Learn information-theoretic approaches to text modeling

    [:octicons-arrow-right-24: Compression Examples](../advanced/text-compression.md)

- :material-translate: **Sequence-to-Sequence**

    ---

    Build models for translation, summarization, and other seq2seq tasks

    [:octicons-arrow-right-24: Seq2Seq Examples](../advanced/seq2seq.md)

- :material-image-text: **Multimodal Models**

    ---

    Combine text with images for richer representations

    [:octicons-arrow-right-24: Multimodal Examples](../advanced/multimodal.md)

</div>

## Troubleshooting

### Common Issues

**Out of Memory**:

- Reduce `seq_length` or `batch_size`
- Use gradient accumulation
- Enable mixed precision training

**Poor Generation Quality**:

- Increase training data size
- Add more model capacity
- Tune temperature and sampling parameters
- Implement better training procedures

**Slow Generation**:

- Use JIT compilation: `@jax.jit`
- Implement caching for KV pairs
- Reduce sequence length
- Use beam search pruning

### Performance Tips

1. **JIT Compilation**:

```python
@jax.jit
def generate_step(model, input_seq, key):
    logits = model(input_seq)
    return jax.random.categorical(key, logits[0, -1, :])
```

2. **Batched Generation**:

- Process multiple prompts in parallel
- Use vectorized operations
- Minimize Python loops

3. **Caching**:

- Cache intermediate hidden states
- Reuse computation from previous steps
- Implement incremental decoding

## Additional Resources

### Documentation

- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/nnx/index.html)
- [JAX Random Numbers](https://jax.readthedocs.io/en/latest/jax.random.html)
- [Language Modeling Tutorial](https://flax.readthedocs.io/en/latest/guides/text_classification.html)

### Research Papers

- [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615) - Kim et al., 2015
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) - Graves, 2013
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Holtzman et al., 2019

### Related Examples

- **EBM Text Modeling**: Energy-based approaches to language modeling
- **Autoregressive Transformers**: Self-attention for sequence modeling
- **VAE for Text**: Variational autoencoders for text generation

---

**Author**: Artifex Team
**Last Updated**: 2025-10-22
**Difficulty**: Beginner
**Time to Complete**: ~30 minutes
