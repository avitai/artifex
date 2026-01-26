"""Text evaluation metrics for the text modality.

This module provides comprehensive evaluation metrics for text generation,
including BLEU, ROUGE, perplexity, and other text quality measures.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import ModalityConfig


@dataclass
class TextMetrics:
    """Container for text evaluation metrics."""

    bleu_1: float
    bleu_2: float
    bleu_4: float
    rouge_l: float
    perplexity: float
    distinct_1: float
    distinct_2: float
    avg_length: float
    vocab_coverage: float
    repetition_rate: float


class TextEvaluationSuite(nnx.Module):
    """Comprehensive text evaluation suite."""

    def __init__(
        self,
        config: ModalityConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize text evaluation suite.

        Args:
            config: Text modality configuration
            rngs: Random number generators
        """
        super().__init__()
        self.config = config
        self.rngs = rngs

        # Extract text parameters for easier access
        self.text_params = config.metadata.get("text_params", {})

    def compute_bleu_score(
        self,
        generated_tokens: jax.Array,
        reference_tokens: jax.Array,
        n: int = 4,
    ) -> jax.Array:
        """Compute BLEU score.

        Args:
            generated_tokens: Generated token sequences [batch_size, seq_len]
            reference_tokens: Reference token sequences [batch_size, seq_len]
            n: N-gram order

        Returns:
            BLEU scores for each sample
        """
        batch_size = generated_tokens.shape[0]
        bleu_scores = []

        for i in range(batch_size):
            gen_seq = generated_tokens[i]
            ref_seq = reference_tokens[i]

            # Remove padding tokens
            gen_seq = self._remove_padding(gen_seq)
            ref_seq = self._remove_padding(ref_seq)

            # Compute n-gram overlap
            bleu = self._compute_ngram_overlap(gen_seq, ref_seq, n)
            bleu_scores.append(bleu)

        return jnp.array(bleu_scores)

    def compute_rouge_l(
        self,
        generated_tokens: jax.Array,
        reference_tokens: jax.Array,
    ) -> jax.Array:
        """Compute ROUGE-L score based on longest common subsequence.

        Args:
            generated_tokens: Generated token sequences
            reference_tokens: Reference token sequences

        Returns:
            ROUGE-L scores
        """
        batch_size = generated_tokens.shape[0]
        rouge_scores = []

        for i in range(batch_size):
            gen_seq = self._remove_padding(generated_tokens[i])
            ref_seq = self._remove_padding(reference_tokens[i])

            # Compute LCS
            lcs_length = self._compute_lcs_length(gen_seq, ref_seq)

            # ROUGE-L = LCS / max(len(gen), len(ref))
            gen_len = len(gen_seq)
            ref_len = len(ref_seq)

            if gen_len == 0 and ref_len == 0:
                rouge_l = 1.0
            elif gen_len == 0 or ref_len == 0:
                rouge_l = 0.0
            else:
                rouge_l = lcs_length / max(gen_len, ref_len)

            rouge_scores.append(rouge_l)

        return jnp.array(rouge_scores)

    def compute_perplexity(
        self,
        tokens: jax.Array,
        log_probs: jax.Array,
    ) -> jax.Array:
        """Compute perplexity from log probabilities.

        Args:
            tokens: Token sequences [batch_size, seq_len]
            log_probs: Log probabilities [batch_size, seq_len, vocab_size]

        Returns:
            Perplexity values
        """
        batch_size, seq_len = tokens.shape
        perplexities = []

        for i in range(batch_size):
            token_seq = tokens[i]
            log_prob_seq = log_probs[i]

            # Remove padding and compute log likelihood
            valid_mask = token_seq != self.text_params.get("pad_token_id", 0)
            valid_tokens = token_seq[valid_mask]
            valid_log_probs = log_prob_seq[valid_mask]

            if len(valid_tokens) == 0:
                perplexities.append(float("inf"))
                continue

            # Get log probabilities for actual tokens
            token_log_probs = []
            for j, token in enumerate(valid_tokens):
                token_log_probs.append(valid_log_probs[j, int(token)])

            avg_log_prob = jnp.mean(jnp.array(token_log_probs))
            perplexity = jnp.exp(-avg_log_prob)
            perplexities.append(float(perplexity))

        return jnp.array(perplexities)

    def compute_distinct_ngrams(
        self,
        tokens: jax.Array,
        n: int = 1,
    ) -> jax.Array:
        """Compute distinct n-gram ratios.

        Args:
            tokens: Token sequences
            n: N-gram order

        Returns:
            Distinct n-gram ratios
        """
        batch_size = tokens.shape[0]
        distinct_ratios = []

        for i in range(batch_size):
            token_seq = self._remove_padding(tokens[i])

            if len(token_seq) < n:
                distinct_ratios.append(0.0)
                continue

            # Extract n-grams
            ngrams = []
            for j in range(len(token_seq) - n + 1):
                ngram = tuple(token_seq[j : j + n].tolist())
                ngrams.append(ngram)

            # Compute distinct ratio
            if len(ngrams) == 0:
                distinct_ratio = 0.0
            else:
                distinct_ratio = len(set(ngrams)) / len(ngrams)

            distinct_ratios.append(distinct_ratio)

        return jnp.array(distinct_ratios)

    def compute_repetition_rate(self, tokens: jax.Array) -> jax.Array:
        """Compute repetition rate in generated sequences.

        Args:
            tokens: Token sequences

        Returns:
            Repetition rates
        """
        batch_size = tokens.shape[0]
        repetition_rates = []

        for i in range(batch_size):
            token_seq = self._remove_padding(tokens[i])

            if len(token_seq) < 2:
                repetition_rates.append(0.0)
                continue

            # Count repeated adjacent tokens
            repetitions = 0
            for j in range(len(token_seq) - 1):
                if token_seq[j] == token_seq[j + 1]:
                    repetitions += 1

            repetition_rate = repetitions / (len(token_seq) - 1)
            repetition_rates.append(repetition_rate)

        return jnp.array(repetition_rates)

    def compute_vocabulary_coverage(self, tokens: jax.Array) -> float:
        """Compute vocabulary coverage.

        Args:
            tokens: Token sequences

        Returns:
            Vocabulary coverage ratio
        """
        # Flatten and get unique tokens
        flat_tokens = tokens.flatten()
        valid_tokens = flat_tokens[flat_tokens != self.text_params.get("pad_token_id", 0)]
        unique_tokens = jnp.unique(valid_tokens)

        coverage = len(unique_tokens) / self.text_params.get("vocab_size", 1000)
        return float(coverage)

    def _remove_padding(self, tokens: jax.Array) -> jax.Array:
        """Remove padding tokens from sequence.

        Args:
            tokens: Token sequence

        Returns:
            Sequence without padding
        """
        # Handle scalar tokens (0-dimensional arrays)
        if tokens.ndim == 0:
            # If it's a scalar and it's a padding token, return empty array
            if tokens == self.text_params.get("pad_token_id", 0):
                return jnp.array([], dtype=tokens.dtype)
            else:
                return jnp.array([tokens])

        # Find first padding token
        pad_mask = tokens == self.text_params.get("pad_token_id", 0)
        if not jnp.any(pad_mask):
            return tokens

        first_pad = int(jnp.argmax(pad_mask))
        return tokens[:first_pad]

    def _compute_ngram_overlap(
        self,
        gen_seq: jax.Array,
        ref_seq: jax.Array,
        n: int,
    ) -> float:
        """Compute n-gram overlap between sequences.

        Args:
            gen_seq: Generated sequence
            ref_seq: Reference sequence
            n: N-gram order

        Returns:
            N-gram overlap score
        """
        if len(gen_seq) < n or len(ref_seq) < n:
            return 0.0

        # Extract n-grams
        gen_ngrams = set()
        ref_ngrams = set()

        for i in range(len(gen_seq) - n + 1):
            ngram = tuple(gen_seq[i : i + n].tolist())
            gen_ngrams.add(ngram)

        for i in range(len(ref_seq) - n + 1):
            ngram = tuple(ref_seq[i : i + n].tolist())
            ref_ngrams.add(ngram)

        # Compute overlap
        if len(gen_ngrams) == 0:
            return 0.0

        overlap = len(gen_ngrams.intersection(ref_ngrams))
        precision = overlap / len(gen_ngrams)
        return precision

    def _compute_lcs_length(
        self,
        seq1: jax.Array,
        seq2: jax.Array,
    ) -> int:
        """Compute longest common subsequence length.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            LCS length
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def evaluate_batch(
        self,
        generated_tokens: jax.Array,
        reference_tokens: jax.Array | None = None,
        log_probs: jax.Array | None = None,
    ) -> TextMetrics:
        """Evaluate a batch of generated text.

        Args:
            generated_tokens: Generated token sequences
            reference_tokens: Reference token sequences (optional)
            log_probs: Log probabilities for perplexity (optional)

        Returns:
            Text evaluation metrics
        """
        # Compute intrinsic metrics
        distinct_1 = jnp.mean(self.compute_distinct_ngrams(generated_tokens, n=1))
        distinct_2 = jnp.mean(self.compute_distinct_ngrams(generated_tokens, n=2))
        repetition_rate = jnp.mean(self.compute_repetition_rate(generated_tokens))
        vocab_coverage = self.compute_vocabulary_coverage(generated_tokens)

        # Compute average length
        lengths = []
        for i in range(generated_tokens.shape[0]):
            seq = self._remove_padding(generated_tokens[i])
            lengths.append(len(seq))
        avg_length = float(jnp.mean(jnp.array(lengths)))

        # Compute reference-based metrics if available
        bleu_1 = bleu_2 = bleu_4 = rouge_l = 0.0
        if reference_tokens is not None:
            bleu_1 = float(
                jnp.mean(self.compute_bleu_score(generated_tokens, reference_tokens, n=1))
            )
            bleu_2 = float(
                jnp.mean(self.compute_bleu_score(generated_tokens, reference_tokens, n=2))
            )
            bleu_4 = float(
                jnp.mean(self.compute_bleu_score(generated_tokens, reference_tokens, n=4))
            )
            rouge_l = float(jnp.mean(self.compute_rouge_l(generated_tokens, reference_tokens)))

        # Compute perplexity if log_probs available
        perplexity = 0.0
        if log_probs is not None:
            perplexity = float(jnp.mean(self.compute_perplexity(generated_tokens, log_probs)))

        return TextMetrics(
            bleu_1=bleu_1,
            bleu_2=bleu_2,
            bleu_4=bleu_4,
            rouge_l=rouge_l,
            perplexity=perplexity,
            distinct_1=float(distinct_1),
            distinct_2=float(distinct_2),
            avg_length=avg_length,
            vocab_coverage=vocab_coverage,
            repetition_rate=float(repetition_rate),
        )


def compute_text_metrics(
    generated_tokens: jax.Array,
    config: ModalityConfig,
    reference_tokens: jax.Array | None = None,
    log_probs: jax.Array | None = None,
    *,
    rngs: nnx.Rngs,
) -> TextMetrics:
    """Compute comprehensive text metrics.

    Args:
        generated_tokens: Generated token sequences
        config: Text modality configuration
        reference_tokens: Reference sequences for comparison
        log_probs: Log probabilities for perplexity computation
        rngs: Random number generators

    Returns:
        Computed text metrics
    """
    evaluator = TextEvaluationSuite(config=config, rngs=rngs)
    return evaluator.evaluate_batch(
        generated_tokens=generated_tokens,
        reference_tokens=reference_tokens,
        log_probs=log_probs,
    )


def compute_semantic_similarity(
    generated_tokens: jax.Array,
    reference_tokens: jax.Array,
    config: ModalityConfig,
) -> jax.Array:
    """Compute semantic similarity between sequences.

    Args:
        generated_tokens: Generated sequences
        reference_tokens: Reference sequences
        config: Text modality configuration

    Returns:
        Semantic similarity scores
    """
    # Placeholder for semantic similarity computation
    # In a real implementation, this would use embeddings
    batch_size = generated_tokens.shape[0]
    similarities = []

    for i in range(batch_size):
        gen_seq = generated_tokens[i]
        ref_seq = reference_tokens[i]

        # Simple token overlap similarity
        gen_tokens = set(gen_seq[gen_seq != config.pad_token_id].tolist())
        ref_tokens = set(ref_seq[ref_seq != config.pad_token_id].tolist())

        if len(gen_tokens) == 0 and len(ref_tokens) == 0:
            similarity = 1.0
        elif len(gen_tokens) == 0 or len(ref_tokens) == 0:
            similarity = 0.0
        else:
            intersection = len(gen_tokens.intersection(ref_tokens))
            union = len(gen_tokens.union(ref_tokens))
            similarity = intersection / union

        similarities.append(similarity)

    return jnp.array(similarities)


def compute_diversity_metrics(
    tokens_batch: jax.Array,
    config: ModalityConfig,
) -> dict[str, float]:
    """Compute diversity metrics for a batch of sequences.

    Args:
        tokens_batch: Batch of token sequences
        config: Text modality configuration

    Returns:
        Dictionary of diversity metrics
    """
    batch_size = tokens_batch.shape[0]

    # Extract text parameters from metadata
    text_params = config.metadata.get("text_params", {})
    pad_token_id = text_params.get("pad_token_id", 0)

    # Collect all sequences without padding
    all_sequences = []
    for i in range(batch_size):
        seq = tokens_batch[i]
        valid_seq = seq[seq != pad_token_id]
        all_sequences.append(valid_seq.tolist())

    # Compute inter-sequence diversity
    unique_sequences = set(tuple(seq) for seq in all_sequences)
    sequence_diversity = len(unique_sequences) / batch_size

    # Compute vocabulary diversity
    all_tokens: set[int] = set()
    for seq in all_sequences:
        all_tokens.update(seq)
    vocab_diversity = len(all_tokens) / config.vocab_size

    # Compute average sequence length
    avg_length = sum(len(seq) for seq in all_sequences) / batch_size

    return {
        "sequence_diversity": sequence_diversity,
        "vocab_diversity": vocab_diversity,
        "avg_length": avg_length,
        "unique_sequences": len(unique_sequences),
        "total_sequences": batch_size,
    }
