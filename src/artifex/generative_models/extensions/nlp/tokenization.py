"""Advanced tokenization for text generation tasks.

This module provides JAX-compatible tokenization utilities for text processing
and generation tasks.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.base import ModelExtension


class AdvancedTokenization(ModelExtension):
    """Advanced tokenization for text generation tasks."""

    def __init__(
        self,
        config: ExtensionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize tokenization module.

        Args:
            config: Extension configuration with tokenization parameters:
                - weight: Weight for the extension (default: 1.0)
                - enabled: Whether the extension is enabled (default: True)
                - extensions.tokenization.vocab_size: Size of vocabulary
                - extensions.tokenization.max_length: Maximum sequence length
                - extensions.tokenization.special_tokens: Dictionary of special tokens
            rngs: Random number generator keys
        """
        # Handle configuration
        if not isinstance(config, ExtensionConfig):
            raise TypeError(f"config must be ExtensionConfig, got {type(config).__name__}")

        super().__init__(config, rngs=rngs)

        # Get tokenization parameters from extensions field
        tokenization_params = getattr(config, "extensions", {}).get("tokenization", {})

        self.vocab_size = tokenization_params.get("vocab_size", 50000)
        self.max_length = tokenization_params.get("max_length", 512)
        self.rngs = rngs

        # Default special tokens
        default_special_tokens = {
            "pad": "<PAD>",
            "unk": "<UNK>",
            "bos": "<BOS>",
            "eos": "<EOS>",
            "mask": "<MASK>",
        }
        self.special_tokens = {
            **default_special_tokens,
            **tokenization_params.get("special_tokens", {}),
        }

        # Initialize vocabulary
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build vocabulary with special tokens."""
        # For simplicity, create a mock vocabulary
        # In practice, this would load from a file or be trained
        self.token_to_id = {}
        self.id_to_token = {}

        # Add special tokens first
        current_id = 0
        for token in self.special_tokens.values():
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Add regular vocabulary (mock for demonstration)
        for i in range(current_id, self.vocab_size):
            token = f"token_{i}"
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def tokenize(self, text: str) -> jax.Array:
        """Tokenize text to integer sequences.

        Args:
            text: Input text string

        Returns:
            Token sequence as integer array
        """
        # Simple word-level tokenization for demonstration
        # In practice, would use BPE, SentencePiece, etc.
        words = text.lower().split()

        tokens = []
        for word in words:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                tokens.append(self.token_to_id[self.special_tokens["unk"]])

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            pad_id = self.token_to_id[self.special_tokens["pad"]]
            tokens.extend([pad_id] * (self.max_length - len(tokens)))

        return jnp.array(tokens)

    def detokenize(self, tokens: jax.Array) -> str:
        """Convert token sequences back to text.

        Args:
            tokens: Token sequence as integer array

        Returns:
            Reconstructed text string
        """
        words = []
        pad_id = self.token_to_id[self.special_tokens["pad"]]

        for token_id in tokens:
            token_id = int(token_id)
            if token_id == pad_id:
                break  # Stop at padding

            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in self.special_tokens.values():
                    words.append(token)

        return " ".join(words)

    def encode_batch(self, texts: list[str]) -> jax.Array:
        """Encode a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            Batch of token sequences [batch_size, max_length]
        """
        encoded = []
        for text in texts:
            encoded.append(self.tokenize(text))

        return jnp.stack(encoded)

    def decode_batch(self, token_batch: jax.Array) -> list[str]:
        """Decode a batch of token sequences.

        Args:
            token_batch: Batch of token sequences [batch_size, seq_length]

        Returns:
            List of decoded text strings
        """
        decoded = []
        for tokens in token_batch:
            decoded.append(self.detokenize(tokens))

        return decoded

    def create_attention_mask(self, tokens: jax.Array) -> jax.Array:
        """Create attention mask for token sequence.

        Args:
            tokens: Token sequence [seq_length] or [batch_size, seq_length]

        Returns:
            Attention mask (1 for real tokens, 0 for padding)
        """
        pad_id = self.token_to_id[self.special_tokens["pad"]]
        mask = (tokens != pad_id).astype(jnp.float32)
        return mask

    def add_special_tokens(
        self, tokens: jax.Array, add_bos: bool = True, add_eos: bool = True
    ) -> jax.Array:
        """Add special tokens to sequence.

        Args:
            tokens: Token sequence
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token

        Returns:
            Modified token sequence
        """
        result = tokens

        if add_bos:
            bos_id = self.token_to_id[self.special_tokens["bos"]]
            result = jnp.concatenate([jnp.array([bos_id]), result])

        if add_eos:
            eos_id = self.token_to_id[self.special_tokens["eos"]]
            # Find first padding token to insert EOS before it
            self.token_to_id[self.special_tokens["pad"]]

            # Simple approach: add EOS at the end
            result = jnp.concatenate([result, jnp.array([eos_id])])

        # Ensure we don't exceed max_length
        if len(result) > self.max_length:
            result = result[: self.max_length]

        return result

    def compute_token_frequencies(self, token_sequences: jax.Array) -> dict[int, int]:
        """Compute token frequency statistics.

        Args:
            token_sequences: Batch of token sequences [batch_size, seq_length]

        Returns:
            Dictionary mapping token IDs to frequencies
        """
        # Flatten all sequences
        all_tokens = token_sequences.flatten()

        # Count frequencies
        frequencies = {}
        for token_id in all_tokens:
            token_id = int(token_id)
            frequencies[token_id] = frequencies.get(token_id, 0) + 1

        return frequencies

    def apply_masking(
        self, tokens: jax.Array, mask_probability: float = 0.15
    ) -> tuple[jax.Array, jax.Array]:
        """Apply random masking for masked language modeling.

        Args:
            tokens: Token sequence [seq_length] or [batch_size, seq_length]
            mask_probability: Probability of masking each token

        Returns:
            Tuple of (masked_tokens, mask_positions)
        """
        if tokens.ndim == 1:
            return self._apply_masking_single(tokens, mask_probability)
        else:
            # Apply to each sequence in batch
            masked_batch = []
            mask_positions_batch = []

            for sequence in tokens:
                masked_seq, mask_pos = self._apply_masking_single(sequence, mask_probability)
                masked_batch.append(masked_seq)
                mask_positions_batch.append(mask_pos)

            return jnp.stack(masked_batch), jnp.stack(mask_positions_batch)

    def _apply_masking_single(
        self, tokens: jax.Array, mask_probability: float
    ) -> tuple[jax.Array, jax.Array]:
        """Apply masking to a single sequence."""
        mask_id = self.token_to_id[self.special_tokens["mask"]]
        pad_id = self.token_to_id[self.special_tokens["pad"]]

        # Don't mask special tokens
        maskable = tokens != pad_id
        for special_id in self.token_to_id.values():
            maskable = maskable & (tokens != special_id)

        # Random masking
        random_mask = jax.random.uniform(self.rngs.mask(), tokens.shape) < mask_probability

        # Combine conditions
        final_mask = maskable & random_mask

        # Apply masking
        masked_tokens = jnp.where(final_mask, mask_id, tokens)

        return masked_tokens, final_mask.astype(jnp.int32)

    def create_position_ids(self, tokens: jax.Array) -> jax.Array:
        """Create position IDs for token sequences.

        Args:
            tokens: Token sequence [seq_length] or [batch_size, seq_length]

        Returns:
            Position IDs array
        """
        if tokens.ndim == 1:
            return jnp.arange(len(tokens))
        else:
            batch_size, seq_length = tokens.shape
            position_ids = jnp.tile(jnp.arange(seq_length), (batch_size, 1))
            return position_ids

    def truncate_sequences(
        self,
        tokens_a: jax.Array,
        tokens_b: jax.Array | None = None,
        max_total_length: int | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Truncate sequences to fit within max length.

        Args:
            tokens_a: First token sequence
            tokens_b: Optional second token sequence
            max_total_length: Maximum combined length

        Returns:
            Tuple of truncated sequences
        """
        if max_total_length is None:
            max_total_length = self.max_length

        if tokens_b is None:
            # Single sequence truncation
            if len(tokens_a) > max_total_length:
                tokens_a = tokens_a[:max_total_length]
            return tokens_a, None
        else:
            # Dual sequence truncation
            total_length = len(tokens_a) + len(tokens_b)

            if total_length <= max_total_length:
                return tokens_a, tokens_b

            # Truncate the longer sequence first
            excess = total_length - max_total_length

            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[: len(tokens_a) - excess]
            else:
                tokens_b = tokens_b[: len(tokens_b) - excess]

            return tokens_a, tokens_b

    def get_vocabulary_info(self) -> dict[str, int | dict]:
        """Get vocabulary information.

        Returns:
            Dictionary with vocabulary statistics
        """
        return {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "special_tokens": self.special_tokens,
            "special_token_ids": {
                token: self.token_to_id[token] for token in self.special_tokens.values()
            },
        }

    def __call__(self, inputs: Any, model_outputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Process model inputs/outputs.

        Args:
            inputs: Original inputs to the model.
            model_outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            dictionary of extension outputs including tokenization features.
        """
        if not self.enabled:
            return {"extension_type": "advanced_tokenization"}

        results = {"extension_type": "advanced_tokenization"}

        # Process text inputs if available
        if isinstance(inputs, dict) and "text" in inputs:
            text = inputs["text"]
            if isinstance(text, str):
                # Tokenize the input text
                tokens = self.tokenize(text)
                results["input_tokens"] = tokens
                results["input_position_ids"] = self.create_position_ids(tokens)

                # Apply masking if requested
                if kwargs.get("apply_masking", False):
                    mask_prob = kwargs.get("mask_probability", 0.15)
                    masked_tokens, mask_positions = self.apply_masking(tokens, mask_prob)
                    results["masked_tokens"] = masked_tokens
                    results["mask_positions"] = mask_positions

        # Process model outputs if they contain tokens
        if isinstance(model_outputs, dict):
            if "tokens" in model_outputs:
                output_tokens = model_outputs["tokens"]
                # Compute token frequencies
                results["token_frequencies"] = self.compute_token_frequencies(output_tokens)

                # Detokenize if requested
                if kwargs.get("detokenize", False):
                    if output_tokens.ndim == 1:
                        results["decoded_text"] = self.detokenize(output_tokens)
                    else:
                        # Batch detokenization
                        decoded_texts = []
                        for seq in output_tokens:
                            decoded_texts.append(self.detokenize(seq))
                        results["decoded_texts"] = decoded_texts

            elif "generated_tokens" in model_outputs:
                # Handle generated tokens
                gen_tokens = model_outputs["generated_tokens"]
                if gen_tokens.ndim == 1:
                    results["generated_text"] = self.detokenize(gen_tokens)
                else:
                    # Batch detokenization
                    generated_texts = []
                    for seq in gen_tokens:
                        generated_texts.append(self.detokenize(seq))
                    results["generated_texts"] = generated_texts

        # Add vocabulary information
        results["vocabulary_info"] = self.get_vocabulary_info()

        return results
