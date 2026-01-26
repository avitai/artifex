"""Text representation processing for the text modality.

This module provides utilities for text representation processing,
including tokenization, position encoding, and sequence processing.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import ModalityConfig


class TextProcessor(nnx.Module):
    """Base text processor for sequence handling."""

    def __init__(
        self,
        config: ModalityConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize text processor.

        Args:
            config: Text modality configuration (ModalityConfig)
            rngs: Random number generators
        """
        super().__init__()
        self.config = config
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
        self.handle_oov = text_params.get("handle_oov", "unk")

    def process_sequences(
        self,
        sequences: jax.Array,
        mask: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Process text sequences.

        Args:
            sequences: Token sequences [batch_size, seq_len]
            mask: Optional attention mask

        Returns:
            Processed sequences with metadata
        """
        batch_size, seq_len = sequences.shape

        # Create attention mask if not provided
        if mask is None:
            mask = sequences != self.pad_token_id

        # Compute sequence lengths
        seq_lengths = jnp.sum(mask, axis=1)

        return {
            "sequences": sequences,
            "mask": mask,
            "lengths": seq_lengths,
            "batch_size": jnp.array(batch_size),
            "seq_len": jnp.array(seq_len),
        }

    def create_causal_mask(self, seq_len: int) -> jax.Array:
        """Create causal attention mask.

        Args:
            seq_len: Sequence length

        Returns:
            Causal mask [seq_len, seq_len]
        """
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def truncate_sequences(
        self,
        sequences: jax.Array,
        max_length: int | None = None,
    ) -> jax.Array:
        """Truncate sequences to maximum length.

        Args:
            sequences: Input sequences
            max_length: Maximum length (uses config default if None)

        Returns:
            Truncated sequences
        """
        max_len = max_length or self.max_length
        return sequences[:, :max_len]

    def pad_sequences(
        self,
        sequences: list[jax.Array],
        max_length: int | None = None,
    ) -> jax.Array:
        """Pad sequences to same length.

        Args:
            sequences: List of variable-length sequences
            max_length: Target length (uses config default if None)

        Returns:
            Padded sequences
        """
        max_len = max_length or self.max_length
        padded = []

        for seq in sequences:
            if len(seq) > max_len:
                padded_seq = seq[:max_len]
            else:
                pad_length = max_len - len(seq)
                pad_tokens = jnp.full((pad_length,), self.pad_token_id)
                padded_seq = jnp.concatenate([seq, pad_tokens])
            padded.append(padded_seq)

        return jnp.stack(padded)


class TokenizationProcessor(nnx.Module):
    """Tokenization processor for text sequences."""

    def __init__(
        self,
        config: ModalityConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize tokenization processor.

        Args:
            config: Text modality configuration (ModalityConfig)
            rngs: Random number generators
        """
        super().__init__()
        self.config = config
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
        self.handle_oov = text_params.get("handle_oov", "unk")

        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary for tokenization."""
        # Simple vocabulary for demonstration
        # In practice, this would be loaded from a file or trained
        self.vocab = {
            "<PAD>": self.pad_token_id,
            "<UNK>": self.unk_token_id,
            "<BOS>": self.bos_token_id,
            "<EOS>": self.eos_token_id,
        }

        # Add common words
        common_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "must",
            "hello",
            "world",
            "machine",
            "learning",
            "deep",
            "neural",
            "network",
            "model",
            "data",
            "train",
            "test",
            "validation",
        ]

        for i, word in enumerate(common_words):
            if len(self.vocab) < self.vocab_size:
                self.vocab[word] = len(self.vocab)

        # Fill remaining slots with placeholder tokens
        while len(self.vocab) < self.vocab_size:
            token = f"token_{len(self.vocab)}"
            self.vocab[token] = len(self.vocab)

        # Create reverse mapping
        self.idx_to_token = {v: k for k, v in self.vocab.items()}

    def tokenize_text(self, text: str) -> jax.Array:
        """Tokenize text into token IDs.

        Args:
            text: Input text

        Returns:
            Token ID sequence
        """
        if not self.case_sensitive:
            text = text.lower()

        words = text.strip().split()
        tokens = [self.bos_token_id]

        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                if self.handle_oov == "unk":
                    tokens.append(self.unk_token_id)
                elif self.handle_oov == "skip":
                    continue
                else:  # error
                    raise ValueError(f"Unknown word: {word}")

        tokens.append(self.eos_token_id)

        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens.extend([self.pad_token_id] * (self.max_length - len(tokens)))

        return jnp.array(tokens, dtype=jnp.int32)

    def detokenize_sequence(self, tokens: jax.Array) -> str:
        """Convert token sequence back to text.

        Args:
            tokens: Token ID sequence

        Returns:
            Reconstructed text
        """
        words = []
        for token_id in tokens:
            if token_id == self.pad_token_id:
                continue
            elif token_id == self.bos_token_id:
                continue
            elif token_id == self.eos_token_id:
                break
            elif int(token_id) in self.idx_to_token:
                words.append(self.idx_to_token[int(token_id)])
            else:
                words.append("<UNK>")

        return " ".join(words)

    def encode_batch(self, texts: list[str]) -> jax.Array:
        """Encode batch of texts.

        Args:
            texts: List of text strings

        Returns:
            Batch of token sequences
        """
        return jnp.stack([self.tokenize_text(text) for text in texts])

    def decode_batch(self, token_batch: jax.Array) -> list[str]:
        """Decode batch of token sequences.

        Args:
            token_batch: Batch of token sequences

        Returns:
            List of decoded texts
        """
        return [self.detokenize_sequence(seq) for seq in token_batch]

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_special_tokens(self) -> dict[str, jax.Array]:
        """Get special token IDs."""
        return {
            "pad": jnp.array(self.pad_token_id),
            "unk": jnp.array(self.unk_token_id),
            "bos": jnp.array(self.bos_token_id),
            "eos": jnp.array(self.eos_token_id),
        }


class PositionEncodingProcessor(nnx.Module):
    """Position encoding processor for text sequences."""

    def __init__(
        self,
        config: ModalityConfig,
        embedding_dim: int = 512,
        encoding_type: str = "sinusoidal",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize position encoding processor.

        Args:
            config: Text modality configuration (ModalityConfig)
            embedding_dim: Embedding dimension
            encoding_type: Type of position encoding ('sinusoidal', 'learned')
            rngs: Random number generators
        """
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.encoding_type = encoding_type

        # Extract text-specific parameters from config metadata
        text_params = (
            self.config.metadata.get("text_params", {}) if hasattr(self.config, "metadata") else {}
        )
        self.max_length = text_params.get("max_length", 512)

        if encoding_type == "sinusoidal":
            self.pos_encoding = self._create_sinusoidal_encoding()
        elif encoding_type == "learned":
            self.pos_embedding = nnx.Embed(
                num_embeddings=self.max_length,
                features=embedding_dim,
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

    def _create_sinusoidal_encoding(self) -> jax.Array:
        """Create sinusoidal position encoding.

        Returns:
            Position encoding matrix [max_length, embedding_dim]
        """
        pos = jnp.arange(self.max_length)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.embedding_dim, 2) * -(jnp.log(10000.0) / self.embedding_dim)
        )

        pe = jnp.zeros((self.max_length, self.embedding_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))

        if self.embedding_dim > 1:
            pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term[: self.embedding_dim // 2]))

        return pe

    def apply_position_encoding(
        self,
        embeddings: jax.Array,
        positions: jax.Array | None = None,
    ) -> jax.Array:
        """Apply position encoding to embeddings.

        Args:
            embeddings: Token embeddings [batch_size, seq_len, embedding_dim]
            positions: Position indices (uses sequential if None)

        Returns:
            Position-encoded embeddings
        """
        batch_size, seq_len, _ = embeddings.shape

        if positions is None:
            positions = jnp.arange(seq_len)

        if self.encoding_type == "sinusoidal":
            pos_enc = self.pos_encoding[positions]
            return embeddings + pos_enc
        else:  # learned
            pos_enc = self.pos_embedding(positions)
            return embeddings + pos_enc

    def get_position_embeddings(
        self,
        positions: jax.Array,
    ) -> jax.Array:
        """Get position embeddings for given positions.

        Args:
            positions: Position indices

        Returns:
            Position embeddings
        """
        if self.encoding_type == "sinusoidal":
            return self.pos_encoding[positions]
        else:  # learned
            return self.pos_embedding(positions)


class SequenceAugmentationProcessor(nnx.Module):
    """Sequence augmentation processor for text data."""

    def __init__(
        self,
        config: ModalityConfig,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize sequence augmentation processor.

        Args:
            config: Text modality configuration (ModalityConfig)
            dropout_rate: Token dropout rate
            rngs: Random number generators
        """
        super().__init__()
        self.config = config
        self.dropout_rate = dropout_rate
        self.rngs = rngs

        # Extract text-specific parameters from config metadata
        text_params = (
            self.config.metadata.get("text_params", {}) if hasattr(self.config, "metadata") else {}
        )
        self.vocab_size = text_params.get("vocab_size", 10000)
        self.pad_token_id = text_params.get("pad_token_id", 0)
        self.unk_token_id = text_params.get("unk_token_id", 1)
        self.bos_token_id = text_params.get("bos_token_id", 2)
        self.eos_token_id = text_params.get("eos_token_id", 3)

        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None  # type: ignore[assignment]

    def apply_token_dropout(
        self,
        sequences: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Apply token dropout to sequences.

        Args:
            sequences: Token sequences
            deterministic: Whether to apply dropout

        Returns:
            Sequences with token dropout applied
        """
        if self.dropout is None or deterministic:
            return sequences

        # Create mask for non-special tokens
        special_tokens = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        mask = jnp.ones_like(sequences, dtype=bool)
        for token_id in special_tokens:
            mask = mask & (sequences != token_id)

        # Apply dropout only to non-special tokens
        if "dropout" in self.rngs:
            key = self.rngs.dropout()
        else:
            key = jax.random.key(0)

        dropout_mask = jax.random.bernoulli(key, 1 - self.dropout_rate, sequences.shape)
        dropout_mask = dropout_mask | ~mask

        # Replace dropped tokens with UNK
        return jnp.where(
            dropout_mask,
            sequences,
            self.unk_token_id,
        )

    def apply_random_substitution(
        self,
        sequences: jax.Array,
        substitution_rate: float = 0.15,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Apply random token substitution.

        Args:
            sequences: Token sequences
            substitution_rate: Rate of token substitution
            deterministic: Whether to apply substitution

        Returns:
            Sequences with random substitutions
        """
        if deterministic or substitution_rate <= 0:
            return sequences

        key = self.rngs.sample()

        # Create substitution mask
        subst_mask = jax.random.bernoulli(key, substitution_rate, sequences.shape)

        # Don't substitute special tokens
        special_tokens = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        for token_id in special_tokens:
            subst_mask = subst_mask & (sequences != token_id)

        # Generate random tokens
        random_tokens = jax.random.randint(
            jax.random.split(key)[0],
            sequences.shape,
            4,  # Start after special tokens
            self.vocab_size,
        )

        return jnp.where(subst_mask, random_tokens, sequences)

    def apply_sequence_shuffle(
        self,
        sequences: jax.Array,
        window_size: int = 3,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Apply local sequence shuffling within windows.

        Args:
            sequences: Token sequences
            window_size: Size of shuffle windows
            deterministic: Whether to apply shuffling

        Returns:
            Sequences with local shuffling
        """
        if deterministic or window_size <= 1:
            return sequences

        batch_size, seq_len = sequences.shape
        shuffled = sequences.copy()

        key = self.rngs.sample()

        for i in range(batch_size):
            seq = shuffled[i]

            # Find valid positions (exclude special tokens at start/end)
            start_pos = 1  # Skip BOS
            end_pos = seq_len - 1  # Skip EOS/padding

            # Find actual end position
            for j in range(seq_len):
                if seq[j] == self.eos_token_id:
                    end_pos = j
                    break

            # Apply windowed shuffling
            for pos in range(start_pos, end_pos - window_size + 1, window_size):
                window_end = min(pos + window_size, end_pos)
                window = seq[pos:window_end]

                # Shuffle within window
                perm = jax.random.permutation(jax.random.split(key, batch_size)[i], len(window))
                shuffled = shuffled.at[i, pos:window_end].set(window[perm])

        return shuffled


def create_text_processor(
    config: ModalityConfig,
    processor_type: str = "basic",
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> TextProcessor:
    """Factory function to create text processors.

    Args:
        config: Text modality configuration
        processor_type: Type of processor ('basic', 'tokenization', 'position', 'augmentation')
        rngs: Random number generators
        **kwargs: Additional arguments for specific processors

    Returns:
        Text processor instance
    """
    if processor_type == "basic":
        return TextProcessor(config=config, rngs=rngs)
    elif processor_type == "tokenization":
        return TokenizationProcessor(config=config, rngs=rngs)
    elif processor_type == "position":
        embedding_dim = kwargs.get("embedding_dim", 512)
        encoding_type = kwargs.get("encoding_type", "sinusoidal")
        return PositionEncodingProcessor(
            config=config,
            embedding_dim=embedding_dim,
            encoding_type=encoding_type,
            rngs=rngs,
        )
    elif processor_type == "augmentation":
        dropout_rate = kwargs.get("dropout_rate", 0.1)
        return SequenceAugmentationProcessor(
            config=config,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")
