"""Base classes and protocols for text modality.

This module defines the core interfaces and base classes for text generation,
following the established modality patterns in the framework.
"""

from enum import Enum
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import BaseConfig, ModalityConfig


class TextRepresentation(Enum):
    """Text representation formats supported by the modality."""

    WORD_LEVEL = "word_level"
    SUBWORD = "subword"
    CHARACTER = "character"
    BYTE_LEVEL = "byte_level"


class TokenizationStrategy(Enum):
    """Tokenization strategies for text processing."""

    SIMPLE = "simple"  # Basic whitespace tokenization
    BPE = "bpe"  # Byte-pair encoding
    WORDPIECE = "wordpiece"  # WordPiece tokenization
    SENTENCEPIECE = "sentencepiece"  # SentencePiece tokenization


def create_default_text_modality_config() -> ModalityConfig:
    """Create default text modality configuration.

    Returns:
        ModalityConfig with text-specific parameters
    """
    return ModalityConfig(
        name="text_modality",
        modality_name="text",
        supported_models=["transformer", "vae", "diffusion"],
        default_metrics=["perplexity", "bleu", "rouge"],
        preprocessing_steps=[
            {"type": "tokenize", "strategy": "simple"},
            {"type": "pad", "max_length": 512},
        ],
        metadata={
            "text_params": {
                "representation": "word_level",
                "vocab_size": 10000,
                "max_length": 512,
                "min_length": 1,
                "tokenization_strategy": "simple",
                "pad_token_id": 0,
                "unk_token_id": 1,
                "bos_token_id": 2,
                "eos_token_id": 3,
                "case_sensitive": False,
                "handle_oov": "unk",
            }
        },
    )


class TextGenerationProtocol(Protocol):
    """Protocol for text generation models."""

    def generate_text(
        self,
        n_samples: int = 1,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate text samples.

        Args:
            n_samples: Number of text samples to generate
            max_length: Maximum length override (uses config default if None)
            temperature: Sampling temperature for generation
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            rngs: Random number generators
            **kwargs: Additional generation parameters

        Returns:
            Generated token sequences of shape (n_samples, max_length)
        """
        ...

    def compute_likelihood(self, tokens: jax.Array) -> jax.Array:
        """Compute likelihood of token sequences.

        Args:
            tokens: Token sequences to evaluate

        Returns:
            Log-likelihood values
        """
        ...

    def compute_perplexity(self, tokens: jax.Array) -> jax.Array:
        """Compute perplexity of token sequences.

        Args:
            tokens: Token sequences to evaluate

        Returns:
            Perplexity values
        """
        ...


class TextModality(nnx.Module):
    """Text modality for generative models.

    This class provides text-specific functionality for generative models,
    including tokenization, sequence processing, and evaluation metrics.
    """

    def __init__(
        self,
        config: ModalityConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize text modality.

        Args:
            config: Configuration for text processing (ModalityConfig)
            rngs: Random number generators
        """
        super().__init__()
        self.config = config or create_default_text_modality_config()
        self.name = "text"

        # Extract text-specific parameters
        text_params = self.config.metadata.get("text_params", {})
        self.vocab_size = text_params.get("vocab_size", 10000)
        self.max_length = text_params.get("max_length", 512)
        self.min_length = text_params.get("min_length", 1)
        self.pad_token_id = text_params.get("pad_token_id", 0)
        self.unk_token_id = text_params.get("unk_token_id", 1)
        self.bos_token_id = text_params.get("bos_token_id", 2)
        self.eos_token_id = text_params.get("eos_token_id", 3)
        self.case_sensitive = text_params.get("case_sensitive", False)
        self.handle_oov = text_params.get("handle_oov", "unk")

    def get_extensions(self, config: BaseConfig) -> dict[str, Any]:
        """Get text-specific extensions.

        Args:
            config: Extension configuration (BaseConfig)

        Returns:
            Dictionary of extension configurations
        """
        # Config validation is now handled by dataclass __post_init__
        extensions = {}

        # Extract extension settings from metadata
        ext_metadata = config.metadata if hasattr(config, "metadata") else {}

        # Position encoding extension
        if ext_metadata.get("use_position_encoding", True):
            extensions["position_encoding"] = {
                "type": ext_metadata.get("position_encoding_type", "sinusoidal"),
                "max_length": ext_metadata.get("max_length", self.max_length),
                "embedding_dim": ext_metadata.get("embedding_dim", 512),
            }

        # Attention monitoring extension
        if ext_metadata.get("use_attention_monitoring", False):
            extensions["attention_monitoring"] = {
                "track_head_importance": ext_metadata.get("track_head_importance", True),
                "visualize_attention": ext_metadata.get("visualize_attention", False),
            }

        # Tokenization extension
        if ext_metadata.get("use_custom_tokenizer", False):
            extensions["tokenization"] = {
                "strategy": ext_metadata.get("tokenization_strategy", "simple"),
                "vocab_size": ext_metadata.get("vocab_size", self.vocab_size),
                "case_sensitive": ext_metadata.get("case_sensitive", self.case_sensitive),
            }

        return extensions

    def preprocess_text(self, text: str | list[str]) -> jax.Array:
        """Preprocess text into token sequences.

        Args:
            text: Input text (single string or list of strings)

        Returns:
            Token sequences as JAX array
        """
        if isinstance(text, str):
            text = [text]

        # Simple tokenization for now (can be extended with real tokenizers)
        tokenized = []
        for t in text:
            tokens = self._simple_tokenize(t)
            tokenized.append(tokens)

        # Pad sequences to max_length
        max_len = self.max_length
        padded = []
        for tokens in tokenized:
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [self.pad_token_id] * (max_len - len(tokens))
            padded.append(tokens)

        return jnp.array(padded, dtype=jnp.int32)

    def _simple_tokenize(self, text: str) -> list[int]:
        """Simple whitespace tokenization.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if not self.case_sensitive:
            text = text.lower()

        # Split on whitespace and convert to token IDs
        words = text.strip().split()
        tokens = []

        for word in words:
            # Simple hash-based token assignment (replace with real tokenizer)
            token_id = hash(word) % (self.vocab_size - 4) + 4
            tokens.append(token_id)

        # Add BOS and EOS tokens
        tokens = [self.bos_token_id, *tokens, self.eos_token_id]
        return tokens

    def postprocess_tokens(self, tokens: jax.Array) -> list[str]:
        """Convert token sequences back to text.

        Args:
            tokens: Token sequences to convert

        Returns:
            List of text strings
        """
        texts = []
        for token_seq in tokens:
            # Simple reverse mapping (replace with real detokenizer)
            words = []
            for token_id in token_seq:
                if token_id == self.pad_token_id:
                    continue
                elif token_id == self.bos_token_id:
                    continue
                elif token_id == self.eos_token_id:
                    break
                else:
                    # Simple mapping back to word (placeholder)
                    words.append(f"token_{int(token_id)}")

            texts.append(" ".join(words))

        return texts

    def process(self, data: jax.Array, **kwargs) -> jax.Array:
        """Process text data for multi-modal fusion.

        Args:
            data: Text token sequences with shape (sequence_length,) or (batch, sequence_length)
            **kwargs: Additional processing arguments

        Returns:
            Processed text features as flattened array
        """
        # Ensure we have a batch dimension
        if data.ndim == 1:
            data = data[jnp.newaxis, ...]

        batch_size, seq_len = data.shape

        # Simple processing: flatten the sequence and pad/truncate to consistent size
        # This creates a basic feature representation for multi-modal fusion
        target_size = 50  # Fixed size for multi-modal compatibility

        processed_batch = []
        for i in range(batch_size):
            sequence = data[i]

            # Convert to float for processing
            features = sequence.astype(jnp.float32)

            # Pad or truncate to target size
            if seq_len > target_size:
                features = features[:target_size]
            elif seq_len < target_size:
                # Pad with zeros
                padding = jnp.zeros(target_size - seq_len)
                features = jnp.concatenate([features, padding])

            processed_batch.append(features)

        result = jnp.stack(processed_batch)

        # If batch size is 1, return without batch dimension for compatibility
        if batch_size == 1:
            return result[0]

        return result


def create_text_modality(
    config: ModalityConfig | None = None,
    *,
    rngs: nnx.Rngs,
) -> TextModality:
    """Factory function to create text modality.

    Args:
        config: Configuration for text modality
        rngs: Random number generators

    Returns:
        Initialized text modality instance
    """
    return TextModality(config=config, rngs=rngs)
