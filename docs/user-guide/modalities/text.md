# Text Modality Guide

This guide covers working with text data in Artifex, including tokenization, vocabulary management, text datasets, and best practices for text-based generative models.

## Overview

Artifex's text modality provides a unified interface for processing text data, handling tokenization, vocabulary management, and sequence processing for generative models.

<div class="grid cards" markdown>

- :material-alphabetical-variant:{ .lg .middle } **Tokenization**

    ---

    Built-in tokenization with special token handling (BOS, EOS, PAD, UNK)

- :material-book-alphabet:{ .lg .middle } **Vocabulary Management**

    ---

    Configurable vocabulary size and token mapping

- :material-form-textbox:{ .lg .middle } **Sequence Handling**

    ---

    Padding, truncation, and sequence length management

- :material-database-outline:{ .lg .middle } **Synthetic Datasets**

    ---

    Ready-to-use synthetic text datasets for testing

- :material-shuffle-variant:{ .lg .middle } **Text Augmentation**

    ---

    Token masking, replacement, and sequence augmentation

- :material-speedometer:{ .lg .middle } **JAX-Native**

    ---

    Full JAX compatibility with efficient batch processing

</div>

## Text Configuration

### Basic Configuration

```python
from artifex.generative_models.core.configuration import ModalityConfiguration
from artifex.generative_models.modalities import TextModality
from flax import nnx

# Initialize RNG
rngs = nnx.Rngs(0)

# Configure text modality
text_config = ModalityConfiguration(
    name="text",
    modality_type="text",
    metadata={
        "text_params": {
            "vocab_size": 10000,
            "max_length": 512,
            "pad_token_id": 0,
            "unk_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "case_sensitive": False
        }
    }
)

# Create modality
text_modality = TextModality(config=text_config, rngs=rngs)

# Access configuration
print(f"Vocab size: {text_modality.vocab_size}")  # 10000
print(f"Max length: {text_modality.max_length}")  # 512
```

### Special Tokens

Artifex uses standard special tokens for sequence processing:

| Token | ID | Purpose |
|-------|----|---------|
| **PAD** | 0 | Padding token for variable-length sequences |
| **UNK** | 1 | Unknown token for out-of-vocabulary words |
| **BOS** | 2 | Beginning-of-sequence marker |
| **EOS** | 3 | End-of-sequence marker |

```python
# Special token configuration
text_config = ModalityConfiguration(
    name="text",
    modality_type="text",
    metadata={
        "text_params": {
            "vocab_size": 50000,
            "max_length": 1024,
            "pad_token_id": 0,    # Padding
            "unk_token_id": 1,    # Unknown
            "bos_token_id": 2,    # Beginning
            "eos_token_id": 3,    # End
            "case_sensitive": True  # Preserve case
        }
    }
)
```

## Text Datasets

### Synthetic Text Datasets

Artifex provides several synthetic text dataset types:

#### Random Sentences

```python
from artifex.generative_models.modalities.text.datasets import SyntheticTextDataset

# Create dataset with random sentences
random_text = SyntheticTextDataset(
    config=text_config,
    dataset_size=5000,
    pattern_type="random_sentences",
    split="train",
    rngs=rngs
)

# Get sample
sample = next(iter(random_text))
print(sample["text"])  # "the cat runs quickly"
print(sample["text_tokens"].shape)  # (512,) - padded to max_length

# Get batch
batch = random_text.get_batch(batch_size=32)
print(batch["text_tokens"].shape)  # (32, 512)
print(len(batch["texts"]))  # 32 - list of strings
```

**Generated patterns:**

- Simple subject-verb-adverb sentences
- Random selection from vocabulary
- Natural-looking structure

#### Repeated Phrases

```python
# Dataset with repeated phrases
repeated_text = SyntheticTextDataset(
    config=text_config,
    dataset_size=5000,
    pattern_type="repeated_phrases",
    split="train",
    rngs=rngs
)

# Example output: "hello world hello world hello world"
sample = next(iter(repeated_text))
print(sample["text"])
```

**Useful for:**

- Testing sequence models
- Pattern recognition
- Repetition detection

#### Numerical Sequences

```python
# Dataset with numerical sequences
sequences = SyntheticTextDataset(
    config=text_config,
    dataset_size=5000,
    pattern_type="sequences",
    split="train",
    rngs=rngs
)

# Example output: "0 1 2 3 4"
sample = next(iter(sequences))
print(sample["text"])
```

**Useful for:**

- Sequence learning tasks
- Arithmetic operations
- Ordering and counting

#### Palindromes

```python
# Dataset with palindromic patterns
palindromes = SyntheticTextDataset(
    config=text_config,
    dataset_size=5000,
    pattern_type="palindromes",
    split="train",
    rngs=rngs
)

# Example output: "racecar is a palindrome racecar"
sample = next(iter(palindromes))
print(sample["text"])
```

**Useful for:**

- Reversibility testing
- Symmetry detection
- Pattern recognition

### Simple Text Datasets

For custom text data:

```python
from artifex.generative_models.modalities.text.datasets import SimpleTextDataset

# Your text data
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing enables text understanding",
    "Transformers revolutionized NLP with attention mechanisms"
]

# Create dataset
text_dataset = SimpleTextDataset(
    config=text_config,
    texts=texts,
    split="train",
    rngs=rngs
)

# Iterate over samples
for sample in text_dataset:
    print(f"Text: {sample['text']}")
    print(f"Tokens: {sample['text_tokens'].shape}")
    print(f"Index: {sample['index']}")
    break

# Get batch
batch = text_dataset.get_batch(batch_size=3)
print(batch["text_tokens"].shape)  # (3, 512)
print(batch["texts"])  # List of 3 strings
```

### Factory Function

```python
from artifex.generative_models.modalities.text.datasets import create_text_dataset

# Create synthetic dataset
dataset = create_text_dataset(
    config=text_config,
    dataset_type="synthetic",
    split="train",
    pattern_type="random_sentences",
    dataset_size=10000,
    rngs=rngs
)

# Create simple dataset
custom_dataset = create_text_dataset(
    config=text_config,
    dataset_type="simple",
    split="train",
    texts=["text 1", "text 2", ...],
    rngs=rngs
)
```

## Tokenization

### Basic Tokenization

Artifex's text datasets use simple hash-based tokenization:

```python
def tokenize_text(text: str, config) -> jax.Array:
    """Tokenize text to token IDs.

    Args:
        text: Input text string
        config: Text configuration

    Returns:
        Token sequence (max_length,)
    """
    # Get parameters from config
    text_params = config.metadata.get("text_params", {})
    vocab_size = text_params.get("vocab_size", 10000)
    max_length = text_params.get("max_length", 512)
    pad_token_id = text_params.get("pad_token_id", 0)
    bos_token_id = text_params.get("bos_token_id", 2)
    eos_token_id = text_params.get("eos_token_id", 3)
    case_sensitive = text_params.get("case_sensitive", False)

    # Normalize case
    if not case_sensitive:
        text = text.lower()

    # Split into words
    words = text.strip().split()

    # Convert to tokens
    tokens = [bos_token_id]  # Add BOS

    for word in words:
        # Simple hash-based token ID
        token_id = hash(word) % (vocab_size - 4) + 4
        tokens.append(token_id)

    tokens.append(eos_token_id)  # Add EOS

    # Pad or truncate
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens.extend([pad_token_id] * (max_length - len(tokens)))

    return jnp.array(tokens, dtype=jnp.int32)

# Usage
text = "Hello world, this is a test"
tokens = tokenize_text(text, text_config)
print(tokens.shape)  # (512,)
print(tokens[:10])  # [2, 5234, 8761, 1234, 9876, 4321, 6543, 3, 0, 0]
```

### Detokenization

```python
def detokenize_tokens(tokens: jax.Array, config) -> str:
    """Convert tokens back to text.

    Args:
        tokens: Token sequence
        config: Text configuration

    Returns:
        Detokenized text string
    """
    text_params = config.metadata.get("text_params", {})
    pad_token_id = text_params.get("pad_token_id", 0)
    bos_token_id = text_params.get("bos_token_id", 2)
    eos_token_id = text_params.get("eos_token_id", 3)

    # Convert to list
    token_list = tokens.tolist()

    # Remove special tokens
    filtered_tokens = []
    for token in token_list:
        if token in [pad_token_id, bos_token_id, eos_token_id]:
            if token == eos_token_id:
                break  # Stop at EOS
            continue
        filtered_tokens.append(token)

    # Convert back to words (placeholder - in practice use vocabulary)
    words = [f"token_{token}" for token in filtered_tokens]

    return " ".join(words)

# Usage
text = "Hello world"
tokens = tokenize_text(text, text_config)
recovered = detokenize_tokens(tokens, text_config)
print(recovered)
```

### Custom Tokenizers

For production use, integrate real tokenizers:

```python
class CustomTextDataset(BaseDataset):
    """Dataset with custom tokenizer."""

    def __init__(
        self,
        config: ModalityConfiguration,
        texts: list[str],
        tokenizer,  # Your tokenizer (e.g., from HuggingFace)
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, split, rngs=rngs)
        self.texts = texts
        self.tokenizer = tokenizer

        # Tokenize all texts
        self.tokens = self._tokenize_all()

    def _tokenize_all(self):
        """Tokenize all texts using custom tokenizer."""
        tokens = []
        for text in self.texts:
            # Use your tokenizer
            # encoded = self.tokenizer.encode(text)
            # For demo:
            encoded = jnp.array([2, 100, 200, 300, 3])  # BOS ... EOS
            tokens.append(encoded)
        return tokens

    def __len__(self) -> int:
        return len(self.texts)

    def __iter__(self):
        for i, (text, tokens) in enumerate(zip(self.texts, self.tokens)):
            yield {
                "text": text,
                "text_tokens": tokens,
                "index": jnp.array(i)
            }

    def get_batch(self, batch_size: int):
        key = self.rngs.sample() if "sample" in self.rngs else jax.random.key(0)
        indices = jax.random.randint(key, (batch_size,), 0, len(self))

        batch_tokens = [self.tokens[int(idx)] for idx in indices]
        batch_texts = [self.texts[int(idx)] for idx in indices]

        return {
            "text_tokens": jnp.stack(batch_tokens),
            "texts": batch_texts,
            "indices": indices
        }
```

## Text Preprocessing

### Padding and Truncation

```python
import jax.numpy as jnp

def pad_sequence(tokens: jax.Array, max_length: int, pad_token_id: int = 0):
    """Pad token sequence to max_length.

    Args:
        tokens: Token sequence
        max_length: Target length
        pad_token_id: Padding token ID

    Returns:
        Padded sequence
    """
    current_length = len(tokens)

    if current_length >= max_length:
        return tokens[:max_length]

    padding = jnp.full((max_length - current_length,), pad_token_id, dtype=tokens.dtype)
    return jnp.concatenate([tokens, padding])

def truncate_sequence(tokens: jax.Array, max_length: int, eos_token_id: int = 3):
    """Truncate sequence and add EOS token.

    Args:
        tokens: Token sequence
        max_length: Maximum length
        eos_token_id: EOS token ID

    Returns:
        Truncated sequence with EOS
    """
    if len(tokens) <= max_length:
        return tokens

    # Truncate and add EOS
    truncated = tokens[:max_length-1]
    return jnp.concatenate([truncated, jnp.array([eos_token_id])])

# Usage
tokens = jnp.array([2, 100, 200, 300, 400, 3])  # BOS ... EOS

# Pad to 10
padded = pad_sequence(tokens, max_length=10, pad_token_id=0)
print(padded)  # [2, 100, 200, 300, 400, 3, 0, 0, 0, 0]

# Truncate to 5
truncated = truncate_sequence(tokens, max_length=5, eos_token_id=3)
print(truncated)  # [2, 100, 200, 300, 3]
```

### Batch Padding

```python
def pad_batch(token_sequences: list[jax.Array], pad_token_id: int = 0):
    """Pad batch of sequences to same length.

    Args:
        token_sequences: List of token sequences
        pad_token_id: Padding token ID

    Returns:
        Padded batch (batch_size, max_length)
    """
    # Find maximum length
    max_length = max(len(seq) for seq in token_sequences)

    # Pad all sequences
    padded = []
    for seq in token_sequences:
        padded_seq = pad_sequence(seq, max_length, pad_token_id)
        padded.append(padded_seq)

    return jnp.stack(padded)

# Usage
sequences = [
    jnp.array([2, 100, 200, 3]),
    jnp.array([2, 300, 400, 500, 600, 3]),
    jnp.array([2, 700, 3])
]

batch = pad_batch(sequences, pad_token_id=0)
print(batch.shape)  # (3, 6) - padded to longest sequence
print(batch)
# [[  2 100 200   3   0   0]
#  [  2 300 400 500 600   3]
#  [  2 700   3   0   0   0]]
```

### Attention Masks

```python
def create_attention_mask(tokens: jax.Array, pad_token_id: int = 0):
    """Create attention mask for padded sequences.

    Args:
        tokens: Token sequence with padding
        pad_token_id: Padding token ID

    Returns:
        Attention mask (1 for real tokens, 0 for padding)
    """
    return (tokens != pad_token_id).astype(jnp.int32)

def create_causal_mask(seq_length: int):
    """Create causal mask for autoregressive generation.

    Args:
        seq_length: Sequence length

    Returns:
        Causal mask (seq_length, seq_length)
    """
    mask = jnp.tril(jnp.ones((seq_length, seq_length)))
    return mask

# Usage
tokens = jnp.array([2, 100, 200, 300, 3, 0, 0, 0])

# Padding mask
pad_mask = create_attention_mask(tokens, pad_token_id=0)
print(pad_mask)  # [1 1 1 1 1 0 0 0]

# Causal mask for generation
causal_mask = create_causal_mask(8)
print(causal_mask)
# [[1 0 0 0 0 0 0 0]
#  [1 1 0 0 0 0 0 0]
#  [1 1 1 0 0 0 0 0]
#  ...
#  [1 1 1 1 1 1 1 1]]
```

## Positional Embeddings

Artifex provides multiple positional encoding methods for transformer-based models.

### Learned Position Embeddings

The default approach using learnable position embeddings:

```python
from artifex.generative_models.extensions.nlp.embeddings import TextEmbeddings
from artifex.generative_models.core.configuration import ExtensionConfig
from flax import nnx

rngs = nnx.Rngs(0)

# Configure embeddings
config = ExtensionConfig(
    weight=1.0,
    enabled=True,
    extensions={
        "embeddings": {
            "embedding_dim": 512,
            "vocab_size": 50000,
            "max_position_embeddings": 1024,
            "dropout_rate": 0.1,
            "use_position_embeddings": True
        }
    }
)

# Create embedding module
embeddings = TextEmbeddings(config=config, rngs=rngs)

# Embed tokens with learned positions
tokens = jnp.array([[2, 100, 200, 300, 3]])  # [batch, seq_len]
embedded = embeddings.embed(tokens, deterministic=True)
print(embedded.shape)  # (1, 5, 512)
```

### Rotary Position Embeddings (RoPE)

RoPE is the state-of-the-art positional encoding used in modern LLMs like Llama 2. It encodes position through rotation of embedding vectors:

```python
# Embed with RoPE (Rotary Position Embeddings)
embedded_rope = embeddings.embed_with_rope(
    tokens,
    deterministic=True,
    base=10000.0  # RoPE base frequency
)
print(embedded_rope.shape)  # (1, 5, 512)

# Apply RoPE to existing embeddings
raw_embeddings = embeddings.get_token_embeddings(tokens[0])
rotated = embeddings.apply_rope_embeddings(raw_embeddings[None], base=10000.0)
```

**Key benefits of RoPE:**

- Enables relative position attention patterns
- Better length generalization
- No learned parameters for positions
- Used in Llama 2, PaLM, and other modern LLMs

### Sinusoidal Position Embeddings

Fixed positional encodings from the original Transformer paper "Attention is All You Need":

```python
# Embed with sinusoidal positions
embedded_sin = embeddings.embed_with_sinusoidal_positions(
    tokens,
    deterministic=True,
    base=10000.0
)
print(embedded_sin.shape)  # (1, 5, 512)

# Get raw sinusoidal encodings
sin_encodings = embeddings.get_sinusoidal_embeddings(
    seq_len=100,
    dim=512,
    base=10000.0
)
print(sin_encodings.shape)  # (100, 512)
```

**Formula:**

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Standalone RoPE Functions

For custom implementations, use the standalone utility functions:

```python
from artifex.generative_models.extensions.nlp.embeddings import (
    precompute_rope_freqs,
    apply_rope,
    create_sinusoidal_positions
)

# Precompute RoPE frequencies
freqs_sin, freqs_cos = precompute_rope_freqs(
    dim=64,          # Must be even
    max_seq_len=512,
    base=10000.0
)

# Apply to query/key tensors in attention
q = jnp.ones((2, 8, 128, 64))  # [batch, heads, seq, dim]
k = jnp.ones((2, 8, 128, 64))

q_rotated = apply_rope(q, freqs_sin, freqs_cos)
k_rotated = apply_rope(k, freqs_sin, freqs_cos)

# Create standalone sinusoidal positions
positions = create_sinusoidal_positions(
    max_seq_len=1024,
    dim=512
)
```

## Text Augmentation

### Token Masking

```python
import jax
import jax.numpy as jnp

def mask_tokens(tokens: jax.Array, key, mask_prob: float = 0.15, mask_token_id: int = 1):
    """Randomly mask tokens (BERT-style).

    Args:
        tokens: Token sequence
        key: Random key
        mask_prob: Probability of masking
        mask_token_id: Token ID for masked positions

    Returns:
        Masked tokens, original tokens
    """
    # Create mask (don't mask special tokens)
    special_tokens = jnp.array([0, 1, 2, 3])  # PAD, UNK, BOS, EOS
    is_special = jnp.isin(tokens, special_tokens)

    # Random mask
    mask = jax.random.bernoulli(key, mask_prob, tokens.shape)
    mask = mask & (~is_special)  # Don't mask special tokens

    # Apply mask
    masked_tokens = jnp.where(mask, mask_token_id, tokens)

    return masked_tokens, tokens

# Usage
tokens = jnp.array([2, 100, 200, 300, 400, 3, 0, 0])
key = jax.random.key(0)

masked, original = mask_tokens(tokens, key, mask_prob=0.15)
print("Original:", original)
print("Masked:  ", masked)
```

### Token Replacement

```python
def replace_tokens(
    tokens: jax.Array,
    key,
    replace_prob: float = 0.1,
    vocab_size: int = 10000
):
    """Randomly replace tokens with random tokens.

    Args:
        tokens: Token sequence
        key: Random key
        replace_prob: Probability of replacement
        vocab_size: Vocabulary size

    Returns:
        Augmented tokens
    """
    keys = jax.random.split(key, 2)

    # Create replacement mask (don't replace special tokens)
    special_tokens = jnp.array([0, 1, 2, 3])
    is_special = jnp.isin(tokens, special_tokens)

    mask = jax.random.bernoulli(keys[0], replace_prob, tokens.shape)
    mask = mask & (~is_special)

    # Generate random tokens (from vocab, excluding special tokens)
    random_tokens = jax.random.randint(keys[1], tokens.shape, 4, vocab_size)

    # Apply replacement
    augmented = jnp.where(mask, random_tokens, tokens)

    return augmented

# Usage
tokens = jnp.array([2, 100, 200, 300, 400, 3, 0, 0])
key = jax.random.key(0)

augmented = replace_tokens(tokens, key, replace_prob=0.1, vocab_size=10000)
print("Original:  ", tokens)
print("Augmented: ", augmented)
```

### Sequence Shuffling

```python
def shuffle_tokens(
    tokens: jax.Array,
    key,
    shuffle_prob: float = 0.1
):
    """Randomly shuffle tokens within a window.

    Args:
        tokens: Token sequence
        key: Random key
        shuffle_prob: Probability of shuffling each position

    Returns:
        Shuffled tokens
    """
    # Don't shuffle special tokens
    special_tokens = jnp.array([0, 1, 2, 3])
    is_special = jnp.isin(tokens, special_tokens)

    # For simplicity, shuffle entire sequence
    should_shuffle = jax.random.bernoulli(key, shuffle_prob)

    def do_shuffle(t):
        # Extract non-special tokens
        non_special_mask = ~is_special
        non_special_tokens = t[non_special_mask]

        # Shuffle
        shuffled_key = jax.random.key(0)
        shuffled = jax.random.permutation(shuffled_key, non_special_tokens)

        # Put back
        result = t.copy()
        result = jnp.where(non_special_mask, shuffled, t)
        return result

    result = jax.lax.cond(
        should_shuffle,
        do_shuffle,
        lambda t: t,
        tokens
    )

    return result

# Usage
tokens = jnp.array([2, 100, 200, 300, 400, 3, 0, 0])
key = jax.random.key(0)

shuffled = shuffle_tokens(tokens, key, shuffle_prob=0.5)
print("Original: ", tokens)
print("Shuffled: ", shuffled)
```

### Complete Augmentation Pipeline

```python
@jax.jit
def augment_text(tokens: jax.Array, key, vocab_size: int = 10000):
    """Apply comprehensive text augmentation.

    Args:
        tokens: Token sequence
        key: Random key
        vocab_size: Vocabulary size

    Returns:
        Augmented tokens
    """
    keys = jax.random.split(key, 3)

    # Token masking (15%)
    tokens, _ = mask_tokens(tokens, keys[0], mask_prob=0.15)

    # Token replacement (5%)
    tokens = replace_tokens(tokens, keys[1], replace_prob=0.05, vocab_size=vocab_size)

    # Note: Shuffling typically not used with masking
    # tokens = shuffle_tokens(tokens, keys[2], shuffle_prob=0.05)

    return tokens

# Batch augmentation
def augment_text_batch(token_batch: jax.Array, key, vocab_size: int = 10000):
    """Augment batch of text sequences.

    Args:
        token_batch: Batch of token sequences (N, max_length)
        key: Random key
        vocab_size: Vocabulary size

    Returns:
        Augmented batch
    """
    batch_size = token_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    # Vectorize over batch
    augmented = jax.vmap(lambda t, k: augment_text(t, k, vocab_size))(
        token_batch, keys
    )

    return augmented

# Usage in training
key = jax.random.key(0)
for batch in data_loader:
    key, subkey = jax.random.split(key)
    augmented_tokens = augment_text_batch(
        batch["text_tokens"],
        subkey,
        vocab_size=10000
    )
    # Use augmented_tokens for training
```

## Vocabulary Statistics

### Computing Statistics

```python
def compute_vocab_stats(dataset):
    """Compute vocabulary statistics for dataset.

    Args:
        dataset: Text dataset

    Returns:
        Dictionary of statistics
    """
    all_tokens = set()
    sequence_lengths = []
    token_frequencies = {}

    for sample in dataset:
        tokens = sample["text_tokens"]

        # Collect unique tokens
        all_tokens.update(tokens.tolist())

        # Sequence length (excluding padding)
        pad_token_id = 0
        length = jnp.sum(tokens != pad_token_id)
        sequence_lengths.append(int(length))

        # Token frequencies
        for token in tokens:
            token = int(token)
            if token != pad_token_id:
                token_frequencies[token] = token_frequencies.get(token, 0) + 1

    return {
        "unique_tokens": len(all_tokens),
        "vocab_coverage": len(all_tokens) / dataset.vocab_size,
        "avg_sequence_length": jnp.mean(jnp.array(sequence_lengths)),
        "max_sequence_length": max(sequence_lengths),
        "min_sequence_length": min(sequence_lengths),
        "total_tokens": sum(token_frequencies.values()),
        "most_common": sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
    }

# Usage
stats = compute_vocab_stats(text_dataset)
print(f"Unique tokens: {stats['unique_tokens']}")
print(f"Vocab coverage: {stats['vocab_coverage']:.2%}")
print(f"Avg sequence length: {stats['avg_sequence_length']:.1f}")
print(f"Most common tokens: {stats['most_common']}")
```

## Complete Examples

### Example 1: Text Generation Dataset

```python
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.core.configuration import ModalityConfiguration
from artifex.generative_models.modalities.text.datasets import SyntheticTextDataset

# Setup
rngs = nnx.Rngs(0)

# Configure
text_config = ModalityConfiguration(
    name="text",
    modality_type="text",
    metadata={
        "text_params": {
            "vocab_size": 50000,
            "max_length": 256,
            "pad_token_id": 0,
            "unk_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "case_sensitive": False
        }
    }
)

# Create datasets
train_dataset = SyntheticTextDataset(
    config=text_config,
    dataset_size=100000,
    pattern_type="random_sentences",
    split="train",
    rngs=rngs
)

val_dataset = SyntheticTextDataset(
    config=text_config,
    dataset_size=10000,
    pattern_type="random_sentences",
    split="val",
    rngs=rngs
)

# Training loop
batch_size = 64
num_epochs = 10
key = jax.random.key(42)

for epoch in range(num_epochs):
    num_batches = len(train_dataset) // batch_size

    for i in range(num_batches):
        # Get batch
        batch = train_dataset.get_batch(batch_size)
        tokens = batch["text_tokens"]

        # Apply augmentation during training
        key, subkey = jax.random.split(key)
        augmented_tokens = augment_text_batch(tokens, subkey, vocab_size=50000)

        # Training step
        # loss = train_step(model, augmented_tokens)

    # Validation (no augmentation)
    val_batches = len(val_dataset) // batch_size
    for i in range(val_batches):
        val_batch = val_dataset.get_batch(batch_size)
        # val_loss = validate_step(model, val_batch["text_tokens"])

    print(f"Epoch {epoch + 1}/{num_epochs} complete")
```

### Example 2: Custom Text Dataset

```python
from typing import Iterator
from artifex.generative_models.modalities.base import BaseDataset

class CustomTextDataset(BaseDataset):
    """Custom text dataset from file."""

    def __init__(
        self,
        config: ModalityConfiguration,
        text_file: str,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, split, rngs=rngs)
        self.text_file = text_file

        # Get text parameters
        text_params = config.metadata.get("text_params", {})
        self.vocab_size = text_params.get("vocab_size", 10000)
        self.max_length = text_params.get("max_length", 512)

        # Load texts
        self.texts = self._load_texts()
        self.tokens = self._tokenize_all()

    def _load_texts(self):
        """Load texts from file."""
        texts = []
        # In practice: with open(self.text_file) as f: ...
        # For demo:
        texts = [
            "Sample text 1",
            "Sample text 2",
            "Sample text 3"
        ]
        return texts

    def _tokenize_all(self):
        """Tokenize all texts."""
        tokens = []
        for text in self.texts:
            # Use tokenization function
            token_seq = tokenize_text(text, self.config)
            tokens.append(token_seq)
        return tokens

    def __len__(self) -> int:
        return len(self.texts)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        for i, (text, tokens) in enumerate(zip(self.texts, self.tokens)):
            yield {
                "text": text,
                "text_tokens": tokens,
                "index": jnp.array(i)
            }

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        key = self.rngs.sample() if "sample" in self.rngs else jax.random.key(0)
        indices = jax.random.randint(key, (batch_size,), 0, len(self))

        batch_tokens = [self.tokens[int(idx)] for idx in indices]
        batch_texts = [self.texts[int(idx)] for idx in indices]

        return {
            "text_tokens": jnp.stack(batch_tokens),
            "texts": batch_texts,
            "indices": indices
        }

# Usage
custom_dataset = CustomTextDataset(
    config=text_config,
    text_file="data/texts.txt",
    rngs=rngs
)
```

## Best Practices

### DO

!!! tip "Tokenization"
    - Use consistent tokenization across train/val/test splits
    - Handle special tokens properly (BOS, EOS, PAD, UNK)
    - Choose appropriate vocabulary size for your task
    - Preserve case if semantically important
    - Validate tokenized sequences
    - Cache tokenized data when possible

!!! tip "Sequence Handling"
    - Pad sequences to consistent length for batching
    - Use attention masks to handle padding
    - Truncate long sequences appropriately
    - Add BOS/EOS tokens for generation tasks
    - Handle variable-length sequences efficiently

!!! tip "Augmentation"
    - Apply augmentation only during training
    - Don't mask special tokens
    - Balance augmentation strength
    - Use JIT compilation for speed
    - Validate augmented sequences

### DON'T

!!! danger "Common Mistakes"
    - Mix different tokenization schemes
    - Forget to add special tokens
    - Ignore padding in loss computation
    - Apply augmentation during validation
    - Use case-sensitive when not needed
    - Exceed vocabulary size with token IDs

!!! danger "Performance Issues"
    - Tokenize on-the-fly during training
    - Use Python loops for token processing
    - Load entire corpus into memory
    - Recompute masks every forward pass

!!! danger "Data Quality"
    - Skip sequence validation
    - Mix different sequence lengths without padding
    - Use inconsistent special token IDs
    - Ignore out-of-vocabulary tokens

## Summary

This guide covered:

- **Text configuration** - Vocabulary, sequence length, special tokens
- **Text datasets** - Synthetic and custom text datasets
- **Tokenization** - Token mapping, padding, truncation
- **Preprocessing** - Attention masks, batch padding
- **Positional embeddings** - Learned, RoPE, and sinusoidal encoding methods
- **Augmentation** - Token masking, replacement, shuffling
- **Vocabulary stats** - Computing coverage and frequency
- **Complete examples** - Training pipelines and custom datasets
- **Best practices** - DOs and DON'Ts for text data

## Next Steps

<div class="grid cards" markdown>

- :material-volume-high:{ .lg .middle } **[Audio Modality Guide](audio.md)**

    ---

    Learn about audio processing, spectrograms, and audio augmentation

- :material-layers-triple:{ .lg .middle } **[Multi-modal Guide](multimodal.md)**

    ---

    Working with multiple modalities and aligned multi-modal datasets

- :material-image:{ .lg .middle } **[Image Modality Guide](image.md)**

    ---

    Deep dive into image datasets, preprocessing, and augmentation

- :material-api:{ .lg .middle } **[Data API Reference](../../api/data/loaders.md)**

    ---

    Complete API documentation for all dataset classes and functions

</div>
