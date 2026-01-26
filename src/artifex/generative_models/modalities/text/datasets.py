"""Text dataset handling for the text modality.

This module provides utilities for loading and processing text datasets,
including synthetic data generation for testing and development.

Note: Dataset classes don't inherit from nnx.Module because they're data
containers, not neural network modules.
"""

from abc import ABC
from typing import Iterator

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import ModalityConfig


class TextDataset(ABC):
    """Base class for text datasets."""

    def __init__(
        self,
        config: ModalityConfig,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize text dataset.

        Args:
            config: Text modality configuration (ModalityConfig)
            split: Dataset split ('train', 'val', 'test')
            rngs: Random number generators
        """
        self.config = config
        self.split = split
        self.rngs = rngs

        # Extract text-specific parameters from config metadata
        text_params = (
            self.config.metadata.get("text_params", {}) if hasattr(self.config, "metadata") else {}
        )
        self.vocab_size = text_params.get("vocab_size", 10000)
        self.max_length = text_params.get("max_length", 512)
        self.pad_token_id = text_params.get("pad_token_id", 0)
        self.unk_token_id = text_params.get("unk_token_id", 1)
        self.bos_token_id = text_params.get("bos_token_id", 2)
        self.eos_token_id = text_params.get("eos_token_id", 3)
        self.case_sensitive = text_params.get("case_sensitive", False)

    def __len__(self) -> int:
        """Return dataset size."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over dataset samples."""
        raise NotImplementedError

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Batch dictionary with 'text_tokens' and potentially 'labels'
        """
        raise NotImplementedError


class SyntheticTextDataset(TextDataset):
    """Synthetic text dataset for testing and development."""

    def __init__(
        self,
        config: ModalityConfig,
        dataset_size: int = 1000,
        pattern_type: str = "random_sentences",
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize synthetic text dataset.

        Args:
            config: Text modality configuration (ModalityConfig)
            dataset_size: Number of synthetic samples
            pattern_type: Type of pattern to generate
                ('random_sentences', 'repeated_phrases', 'sequences', 'palindromes')
            split: Dataset split
            rngs: Random number generators
        """
        super().__init__(config, split, rngs=rngs)
        self.dataset_size = dataset_size
        self.pattern_type = pattern_type
        self._generate_dataset()

    def _generate_dataset(self):
        """Generate synthetic text data."""
        self._texts = []
        self._tokens = []

        # Generate texts based on pattern type
        for i in range(self.dataset_size):
            if self.pattern_type == "random_sentences":
                text = self._generate_random_sentence(i)
            elif self.pattern_type == "repeated_phrases":
                text = self._generate_repeated_phrase(i)
            elif self.pattern_type == "sequences":
                text = self._generate_sequence(i)
            elif self.pattern_type == "palindromes":
                text = self._generate_palindrome(i)
            else:
                text = f"sample text {i}"

            self._texts.append(text)
            # Convert to tokens using simple tokenization
            tokens = self._simple_tokenize(text)
            self._tokens.append(tokens)

    def _generate_random_sentence(self, seed: int) -> str:
        """Generate a random sentence.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Generated sentence
        """
        key = jax.random.key(seed)

        # Simple word vocabulary
        subjects = ["the cat", "a dog", "the bird", "a fish", "the robot"]
        verbs = ["runs", "jumps", "flies", "swims", "thinks"]
        objects = ["quickly", "slowly", "gracefully", "loudly", "quietly"]

        # Random selection
        subj_idx = jax.random.randint(key, (), 0, len(subjects))
        verb_idx = jax.random.randint(jax.random.split(key)[0], (), 0, len(verbs))
        obj_idx = jax.random.randint(jax.random.split(key)[1], (), 0, len(objects))

        subject = subjects[int(subj_idx)]
        verb = verbs[int(verb_idx)]
        obj = objects[int(obj_idx)]

        return f"{subject} {verb} {obj}"

    def _generate_repeated_phrase(self, seed: int) -> str:
        """Generate text with repeated phrases.

        Args:
            seed: Random seed

        Returns:
            Text with repeated phrases
        """
        phrases = ["hello world", "machine learning", "neural networks", "deep learning"]
        phrase = phrases[seed % len(phrases)]
        repeats = (seed % 3) + 1
        return " ".join([phrase] * repeats)

    def _generate_sequence(self, seed: int) -> str:
        """Generate numerical sequences as text.

        Args:
            seed: Random seed

        Returns:
            Sequence as text
        """
        start = seed % 10
        length = (seed % 5) + 3
        sequence = [str(start + i) for i in range(length)]
        return " ".join(sequence)

    def _generate_palindrome(self, seed: int) -> str:
        """Generate palindromic text.

        Args:
            seed: Random seed

        Returns:
            Palindromic text
        """
        words = ["racecar", "level", "noon", "civic", "radar"]
        word = words[seed % len(words)]
        return f"{word} is a palindrome {word}"

    def _simple_tokenize(self, text: str) -> jax.Array:
        """Simple tokenization for synthetic data.

        Args:
            text: Input text

        Returns:
            Token sequence as JAX array
        """
        if not self.case_sensitive:
            text = text.lower()

        words = text.strip().split()
        tokens = []

        # Add BOS token
        tokens.append(self.bos_token_id)

        # Convert words to token IDs
        for word in words:
            # Simple hash-based assignment
            token_id = hash(word) % (self.vocab_size - 4) + 4
            tokens.append(token_id)

        # Add EOS token
        tokens.append(self.eos_token_id)

        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens.extend([self.pad_token_id] * (self.max_length - len(tokens)))

        return jnp.array(tokens, dtype=jnp.int32)

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over dataset samples."""
        for i in range(self.dataset_size):
            yield {
                "text_tokens": self._tokens[i],
                "text": self._texts[i],
                "index": jnp.array(i, dtype=jnp.int32),
            }

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Batch dictionary with text data
        """
        key = self.rngs.sample()

        # Random sampling with replacement
        indices = jax.random.randint(key, (batch_size,), 0, self.dataset_size)

        batch_tokens = []
        batch_texts = []

        for idx in indices:
            batch_tokens.append(self._tokens[int(idx)])
            batch_texts.append(self._texts[int(idx)])

        return {
            "text_tokens": jnp.stack(batch_tokens),
            "texts": batch_texts,  # Keep as strings for reference
            "indices": indices,
        }

    def get_sample_text(self, index: int) -> str:
        """Get sample text by index.

        Args:
            index: Sample index

        Returns:
            Text sample
        """
        if index < 0 or index >= self.dataset_size:
            raise IndexError(f"Index {index} out of range [0, {self.dataset_size})")
        return self._texts[index]

    def get_vocab_stats(self) -> dict[str, int]:
        """Get vocabulary statistics.

        Returns:
            Dictionary with vocabulary statistics
        """
        all_tokens: set[int] = set()
        for tokens in self._tokens:
            all_tokens.update(tokens.tolist())

        return {
            "unique_tokens": len(all_tokens),
            "vocab_coverage": len(all_tokens) / self.vocab_size,
            "total_sequences": self.dataset_size,
            "max_length": self.max_length,
        }


class SimpleTextDataset(TextDataset):
    """Simple text dataset from list of strings."""

    def __init__(
        self,
        config: ModalityConfig,
        texts: list[str],
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize simple text dataset.

        Args:
            config: Text modality configuration (ModalityConfig)
            texts: List of text strings
            split: Dataset split
            rngs: Random number generators
        """
        super().__init__(config, split, rngs=rngs)
        self.texts = texts
        self._preprocess_texts()

    def _preprocess_texts(self):
        """Preprocess texts into token sequences."""
        self._tokens = []
        for text in self.texts:
            tokens = self._simple_tokenize(text)
            self._tokens.append(tokens)

    def _simple_tokenize(self, text: str) -> jax.Array:
        """Simple tokenization.

        Args:
            text: Input text

        Returns:
            Token sequence
        """
        if not self.case_sensitive:
            text = text.lower()

        words = text.strip().split()
        tokens = [self.bos_token_id]

        for word in words:
            token_id = hash(word) % (self.vocab_size - 4) + 4
            tokens.append(token_id)

        tokens.append(self.eos_token_id)

        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens.extend([self.pad_token_id] * (self.max_length - len(tokens)))

        return jnp.array(tokens, dtype=jnp.int32)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over dataset samples."""
        for i, text in enumerate(self.texts):
            yield {
                "text_tokens": self._tokens[i],
                "text": text,
                "index": jnp.array(i, dtype=jnp.int32),
            }

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Batch dictionary
        """
        key = self.rngs.sample()

        indices = jax.random.randint(key, (batch_size,), 0, len(self.texts))

        batch_tokens = jnp.stack([self._tokens[int(idx)] for idx in indices])
        batch_texts = [self.texts[int(idx)] for idx in indices]

        return {
            "text_tokens": batch_tokens,
            "texts": batch_texts,
            "indices": indices,
        }


def create_text_dataset(
    config: ModalityConfig,
    dataset_type: str = "synthetic",
    split: str = "train",
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> TextDataset:
    """Factory function to create text datasets.

    Args:
        config: Text modality configuration
        dataset_type: Type of dataset ('synthetic', 'simple')
        split: Dataset split
        rngs: Random number generators
        **kwargs: Additional arguments for specific dataset types

    Returns:
        Text dataset instance
    """
    if dataset_type == "synthetic":
        return SyntheticTextDataset(
            config=config,
            split=split,
            rngs=rngs,
            **kwargs,
        )
    elif dataset_type == "simple":
        texts = kwargs.get("texts", ["hello world", "machine learning", "deep learning"])
        return SimpleTextDataset(
            config=config,
            texts=texts,
            split=split,
            rngs=rngs,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
