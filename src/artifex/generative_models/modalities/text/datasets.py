"""Text datasets backed by datarax MemorySource.

Provides pure data generation functions and factory functions that wrap
generated data in datarax MemorySource for pipeline integration.
"""

from typing import Any

import jax
import jax.numpy as jnp
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx


# ---------------------------------------------------------------------------
# Tokenization (standalone function)
# ---------------------------------------------------------------------------


def simple_tokenize(
    text: str,
    *,
    vocab_size: int = 10000,
    max_length: int = 512,
    pad_token_id: int = 0,
    bos_token_id: int = 2,
    eos_token_id: int = 3,
    case_sensitive: bool = False,
) -> jnp.ndarray:
    """Simple hash-based tokenization.

    Args:
        text: Input text string.
        vocab_size: Size of the vocabulary.
        max_length: Maximum sequence length.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
        case_sensitive: Whether tokenization is case-sensitive.

    Returns:
        Token IDs as JAX array of shape (max_length,).
    """
    if not case_sensitive:
        text = text.lower()

    words = text.strip().split()
    tokens: list[int] = [bos_token_id]

    for word in words:
        token_id = hash(word) % (vocab_size - 4) + 4
        tokens.append(token_id)

    tokens.append(eos_token_id)

    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens.extend([pad_token_id] * (max_length - len(tokens)))

    return jnp.array(tokens, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Text generation (pure functions)
# ---------------------------------------------------------------------------


def _generate_text(idx: int, pattern_type: str) -> str:
    """Generate text deterministically based on index.

    Args:
        idx: Sample index used as seed.
        pattern_type: Type of text pattern.

    Returns:
        Generated text string.
    """
    if pattern_type == "random_sentences":
        return _generate_random_sentence(idx)
    elif pattern_type == "repeated_phrases":
        return _generate_repeated_phrase(idx)
    elif pattern_type == "sequences":
        return _generate_sequence(idx)
    elif pattern_type == "palindromes":
        return _generate_palindrome(idx)
    else:
        return f"sample text {idx}"


def _generate_random_sentence(seed: int) -> str:
    """Generate a random sentence."""
    key = jax.random.key(seed)
    subjects = ["the cat", "a dog", "the bird", "a fish", "the robot"]
    verbs = ["runs", "jumps", "flies", "swims", "thinks"]
    adverbs = ["quickly", "slowly", "gracefully", "loudly", "quietly"]

    subj_idx = int(jax.random.randint(key, (), 0, len(subjects)))
    key1, key2 = jax.random.split(key)
    verb_idx = int(jax.random.randint(key1, (), 0, len(verbs)))
    adv_idx = int(jax.random.randint(key2, (), 0, len(adverbs)))

    return f"{subjects[subj_idx]} {verbs[verb_idx]} {adverbs[adv_idx]}"


def _generate_repeated_phrase(seed: int) -> str:
    """Generate text with repeated phrases."""
    phrases = [
        "hello world",
        "machine learning",
        "neural networks",
        "deep learning",
    ]
    phrase = phrases[seed % len(phrases)]
    repeats = (seed % 3) + 1
    return " ".join([phrase] * repeats)


def _generate_sequence(seed: int) -> str:
    """Generate numerical sequences as text."""
    start = seed % 10
    length = (seed % 5) + 3
    return " ".join(str(start + i) for i in range(length))


def _generate_palindrome(seed: int) -> str:
    """Generate palindromic text."""
    words = ["racecar", "level", "noon", "civic", "radar"]
    word = words[seed % len(words)]
    return f"{word} is a palindrome {word}"


# ---------------------------------------------------------------------------
# Data generation (pure functions)
# ---------------------------------------------------------------------------


def generate_synthetic_text_data(
    num_samples: int,
    *,
    vocab_size: int = 10000,
    max_length: int = 512,
    pattern_type: str = "random_sentences",
    pad_token_id: int = 0,
    bos_token_id: int = 2,
    eos_token_id: int = 3,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """Generate synthetic text data with token sequences.

    Args:
        num_samples: Number of text samples.
        vocab_size: Size of the vocabulary.
        max_length: Maximum sequence length.
        pattern_type: Text generation pattern
            ('random_sentences', 'repeated_phrases', 'sequences', 'palindromes').
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
        case_sensitive: Whether tokenization is case-sensitive.

    Returns:
        Dictionary with 'text_tokens' array of shape (num_samples, max_length)
        and 'index' array of shape (num_samples,).
    """
    texts = [_generate_text(i, pattern_type) for i in range(num_samples)]
    tokens = [
        simple_tokenize(
            text,
            vocab_size=vocab_size,
            max_length=max_length,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            case_sensitive=case_sensitive,
        )
        for text in texts
    ]

    return {
        "text_tokens": jnp.stack(tokens),
        "index": jnp.arange(num_samples, dtype=jnp.int32),
    }


def generate_text_from_strings(
    texts: list[str],
    *,
    vocab_size: int = 10000,
    max_length: int = 512,
    pad_token_id: int = 0,
    bos_token_id: int = 2,
    eos_token_id: int = 3,
    case_sensitive: bool = False,
) -> dict[str, jnp.ndarray]:
    """Generate token data from a list of text strings.

    Args:
        texts: List of text strings.
        vocab_size: Size of the vocabulary.
        max_length: Maximum sequence length.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
        case_sensitive: Whether tokenization is case-sensitive.

    Returns:
        Dictionary with 'text_tokens' and 'index' arrays.
    """
    tokens = [
        simple_tokenize(
            text,
            vocab_size=vocab_size,
            max_length=max_length,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            case_sensitive=case_sensitive,
        )
        for text in texts
    ]

    return {
        "text_tokens": jnp.stack(tokens),
        "index": jnp.arange(len(texts), dtype=jnp.int32),
    }


# ---------------------------------------------------------------------------
# Factory functions — return MemorySource instances
# ---------------------------------------------------------------------------


def create_text_dataset(
    dataset_type: str = "synthetic",
    *,
    rngs: nnx.Rngs,
    shuffle: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a text dataset as a MemorySource.

    Args:
        dataset_type: Type of dataset ('synthetic', 'simple').
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.
        **kwargs: Additional generation parameters.
            For 'synthetic': dataset_size, vocab_size, max_length, pattern_type, etc.
            For 'simple': texts (list[str]), vocab_size, max_length, etc.

    Returns:
        MemorySource backed by generated text data.

    Raises:
        ValueError: If dataset_type is unknown.
    """
    if dataset_type == "synthetic":
        num_samples = kwargs.pop("dataset_size", kwargs.pop("num_samples", 1000))
        data = generate_synthetic_text_data(num_samples, **kwargs)
    elif dataset_type == "simple":
        texts = kwargs.pop("texts", ["hello world", "machine learning", "deep learning"])
        data = generate_text_from_strings(texts, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    source_config = MemorySourceConfig(shuffle=shuffle)
    return MemorySource(source_config, data, rngs=rngs)
