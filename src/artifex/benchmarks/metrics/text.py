"""Text-specific metrics for generative model evaluation."""

import re
from collections import Counter

import flax.nnx as nnx
import jax.numpy as jnp

from artifex.generative_models.core.configuration import EvaluationConfig

from .core import MetricBase


class BLEUMetric(MetricBase):
    """BLEU score metric for text generation quality assessment."""

    def __init__(self, *, config: EvaluationConfig, rngs: nnx.Rngs):
        """Initialize BLEU metric.

        Args:
            config: Evaluation configuration (must be EvaluationConfig)
            rngs: NNX Rngs for stochastic operations
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        # BLEU parameters from config
        bleu_params = config.metric_params.get("bleu", {})
        self.max_n = bleu_params.get("max_n", 4)
        self.smooth = bleu_params.get("smooth", True)
        self.weights = bleu_params.get("weights", [0.25, 0.25, 0.25, 0.25])

    def validate_inputs(self, real_data, generated_data) -> bool:
        """Validate input data for BLEU computation."""
        if not isinstance(real_data, list) or not isinstance(generated_data, list):
            return False
        if len(real_data) == 0 or len(generated_data) == 0:
            return False
        if len(real_data) != len(generated_data):
            return False
        # Check that all elements are strings
        if not all(isinstance(text, str) for text in real_data):
            return False
        if not all(isinstance(text, str) for text in generated_data):
            return False
        return True

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute BLEU score between reference and generated text."""
        total_score = 0.0

        for ref, gen in zip(real_data, generated_data):
            # Tokenize texts
            ref_tokens = self._tokenize(ref)
            gen_tokens = self._tokenize(gen)

            # Compute BLEU score for this pair
            score = self._compute_bleu_score(ref_tokens, gen_tokens)
            total_score += score

        bleu_score: float = total_score / len(real_data)

        return {"bleu_score": float(bleu_score)}

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        return re.findall(r"\w+", text.lower())

    def _compute_bleu_score(self, reference: list[str], generated: list[str]) -> float:
        """Compute BLEU score for a single text pair."""
        if len(generated) == 0:
            return 0.0

        # Compute n-gram precisions
        precisions: list[float] = []

        for n in range(1, self.max_n + 1):
            ref_ngrams = self._get_ngrams(reference, n)
            gen_ngrams = self._get_ngrams(generated, n)

            if len(gen_ngrams) == 0:
                precisions.append(0.0)
                continue

            # Count matches
            matches = 0
            for ngram in gen_ngrams:
                if ngram in ref_ngrams:
                    matches += min(gen_ngrams[ngram], ref_ngrams[ngram])

            precision = matches / sum(gen_ngrams.values())
            precisions.append(precision)

        # Geometric mean of precisions
        if 0.0 in precisions:
            return 0.0

        geometric_mean = jnp.exp(jnp.mean(jnp.log(jnp.array(precisions))))

        # Brevity penalty
        ref_len = len(reference)
        gen_len = len(generated)

        if gen_len > ref_len:
            bp = 1.0
        else:
            bp = float(jnp.exp(1 - ref_len / gen_len)) if gen_len > 0 else 0.0

        return float(bp * geometric_mean)

    def _get_ngrams(self, tokens: list[str], n: int) -> Counter:
        """Extract n-grams from tokens."""
        ngrams: Counter[tuple[str, ...]] = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams[ngram] += 1
        return ngrams


class ROUGEMetric(MetricBase):
    """ROUGE score metric for text summarization and generation."""

    def __init__(self, *, config: EvaluationConfig, rngs: nnx.Rngs):
        """Initialize ROUGE metric.

        Args:
            config: Evaluation configuration (must be EvaluationConfig)
            rngs: NNX Rngs for stochastic operations
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with minimal config to satisfy MetricBase requirements
        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        # ROUGE parameters from config
        rouge_params = config.metric_params.get("rouge", {})
        self.rouge_types = rouge_params.get("rouge_types", ["rouge1", "rouge2", "rougeL"])
        self.use_stemmer = rouge_params.get("use_stemmer", True)

    def validate_inputs(self, real_data, generated_data) -> bool:
        """Validate input data for ROUGE computation."""
        if not isinstance(real_data, list) or not isinstance(generated_data, list):
            return False
        if len(real_data) == 0 or len(generated_data) == 0:
            return False
        if len(real_data) != len(generated_data):
            return False
        # Check that all elements are strings
        if not all(isinstance(text, str) for text in real_data):
            return False
        if not all(isinstance(text, str) for text in generated_data):
            return False
        return True

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute ROUGE scores between reference and generated text."""
        results: dict[str, float] = {}

        for rouge_type in self.rouge_types:
            scores = []

            for ref, gen in zip(real_data, generated_data):
                score = self._compute_rouge_score(ref, gen, rouge_type)
                scores.append(score)

            results[rouge_type] = float(jnp.mean(jnp.array(scores)))

        return results

    def _compute_rouge_score(self, reference: str, generated: str, rouge_type: str) -> float:
        """Compute specific ROUGE score for a text pair."""
        ref_tokens = self._tokenize(reference)
        gen_tokens = self._tokenize(generated)

        if rouge_type == "rouge-1":
            return self._rouge_n(ref_tokens, gen_tokens, 1)
        elif rouge_type == "rouge-2":
            return self._rouge_n(ref_tokens, gen_tokens, 2)
        elif rouge_type == "rouge-l":
            return self._rouge_l(ref_tokens, gen_tokens)
        else:
            return 0.0

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        return re.findall(r"\w+", text.lower())

    def _rouge_n(self, reference: list[str], generated: list[str], n: int) -> float:
        """Compute ROUGE-N score."""
        ref_ngrams = set(self._get_ngrams_list(reference, n))
        gen_ngrams = set(self._get_ngrams_list(generated, n))

        if len(ref_ngrams) == 0:
            return 0.0

        overlap = len(ref_ngrams.intersection(gen_ngrams))
        return overlap / len(ref_ngrams)

    def _rouge_l(self, reference: list[str], generated: list[str]) -> float:
        """Compute ROUGE-L score using longest common subsequence."""
        lcs_length = self._lcs_length(reference, generated)

        if len(reference) == 0 or len(generated) == 0:
            return 0.0

        precision = lcs_length / len(generated) if len(generated) > 0 else 0.0
        recall = lcs_length / len(reference) if len(reference) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _get_ngrams_list(self, tokens: list[str], n: int) -> list[tuple]:
        """Extract n-grams as list of tuples."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))
        return ngrams

    def _lcs_length(self, seq1: list[str], seq2: list[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]


class PerplexityMetric(MetricBase):
    """Perplexity metric for language model evaluation."""

    def __init__(self, *, config: EvaluationConfig, rngs: nnx.Rngs):
        """Initialize perplexity metric.

        Args:
            config: Evaluation configuration (must be EvaluationConfig)
            rngs: NNX Rngs for stochastic operations
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        # Perplexity parameters from config
        perplexity_params = config.metric_params.get("perplexity", {})
        self.model_name = perplexity_params.get("model_name", "mock")
        self.use_mock = perplexity_params.get("use_mock", True)

    def validate_inputs(self, real_data, generated_data) -> bool:
        """Validate input data for perplexity computation."""
        if not isinstance(generated_data, list):
            return False
        if len(generated_data) == 0:
            return False
        # Check that all elements are strings
        if not all(isinstance(text, str) for text in generated_data):
            return False
        return True

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute perplexity of generated text."""
        total_log_prob = 0.0
        total_tokens = 0

        for text in generated_data:
            tokens = self._tokenize(text)
            if len(tokens) == 0:
                continue

            # Mock language model probability computation
            log_prob = self._compute_log_probability(tokens)
            total_log_prob += log_prob
            total_tokens += len(tokens)

        # Compute perplexity using centralized function
        from artifex.generative_models.core.evaluation.metrics.information import (
            compute_perplexity,
        )

        perplexity = compute_perplexity(total_log_prob, total_tokens)

        return {"perplexity": perplexity}

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        return re.findall(r"\w+", text.lower())

    def _compute_log_probability(self, tokens: list[str]) -> float:
        """Mock computation of log probability for tokens."""
        # In real implementation, this would use a language model
        # Mock with simple uniform probability
        vocab_size = 10000  # Assumed vocabulary size
        uniform_prob = 1.0 / vocab_size
        log_prob = len(tokens) * jnp.log(uniform_prob)
        return float(log_prob)


class DiversityMetric(MetricBase):
    """Diversity metric for text generation evaluation."""

    def __init__(self, *, config: EvaluationConfig, rngs: nnx.Rngs):
        """Initialize diversity metric.

        Args:
            config: Evaluation configuration (must be EvaluationConfig)
            rngs: NNX Rngs for stochastic operations
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        # Diversity parameters from config
        diversity_params = config.metric_params.get("diversity", {})
        self.n_gram_sizes = diversity_params.get("n_gram_sizes", [1, 2, 3])
        self.measure_self_bleu = diversity_params.get("measure_self_bleu", False)

    def validate_inputs(self, real_data, generated_data) -> bool:
        """Validate input data for diversity computation."""
        if not isinstance(generated_data, list):
            return False
        if len(generated_data) < 2:
            return False  # Need at least 2 samples for diversity
        # Check that all elements are strings
        if not all(isinstance(text, str) for text in generated_data):
            return False
        return True

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute diversity metrics for generated text."""
        results: dict[str, float] = {}

        # Compute n-gram diversity
        for n in self.n_gram_sizes:
            diversity = self._compute_ngram_diversity(generated_data, n)
            results[f"diversity_{n}gram"] = diversity

        # Overall diversity score (average)
        overall_diversity = jnp.mean(
            jnp.array([results[f"diversity_{n}gram"] for n in self.n_gram_sizes])
        )
        results["diversity_score"] = float(overall_diversity)

        # Self-BLEU if requested
        if self.measure_self_bleu:
            self_bleu = self._compute_self_bleu(generated_data)
            results["self_bleu"] = self_bleu

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        return re.findall(r"\w+", text.lower())

    def _compute_ngram_diversity(self, texts: list[str], n: int) -> float:
        """Compute n-gram diversity across texts."""
        all_ngrams = set()
        total_ngrams = 0

        for text in texts:
            tokens = self._tokenize(text)
            ngrams = self._get_ngrams_list(tokens, n)
            all_ngrams.update(ngrams)
            total_ngrams += len(ngrams)

        if total_ngrams == 0:
            return 0.0

        # Diversity = unique ngrams / total ngrams
        diversity = len(all_ngrams) / total_ngrams
        return diversity

    def _get_ngrams_list(self, tokens: list[str], n: int) -> list[tuple]:
        """Extract n-grams as list of tuples."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))
        return ngrams

    def _compute_self_bleu(self, texts: list[str]) -> float:
        """Compute average BLEU score between generated texts."""
        # Mock implementation for now
        # In real implementation, would compute BLEU between each text and others
        return 0.5  # Placeholder value


# Factory functions for convenient metric creation
def create_bleu_metric(
    *,
    rngs: nnx.Rngs,
    max_n: int = 4,
    smooth: bool = True,
    weights: list[float] | None = None,
    batch_size: int = 32,
    config_name: str = "bleu_metric",
) -> BLEUMetric:
    """Create BLEU metric with typed configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        max_n: Maximum n-gram size
        smooth: Whether to use smoothing
        weights: Weights for n-gram precisions
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured BLEUMetric instance
    """
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]

    config = EvaluationConfig(
        name=config_name,
        metrics=["bleu"],
        metric_params={
            "bleu": {
                "max_n": max_n,
                "smooth": smooth,
                "weights": weights,
                "higher_is_better": True,
            }
        },
        eval_batch_size=batch_size,
    )

    return BLEUMetric(config=config, rngs=rngs)


def create_rouge_metric(
    *,
    rngs: nnx.Rngs,
    rouge_types: list[str] | None = None,
    use_stemmer: bool = True,
    batch_size: int = 32,
    config_name: str = "rouge_metric",
) -> ROUGEMetric:
    """Create ROUGE metric with typed configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        rouge_types: List of ROUGE types to compute
        use_stemmer: Whether to use stemming
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured ROUGEMetric instance
    """
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    config = EvaluationConfig(
        name=config_name,
        metrics=["rouge"],
        metric_params={
            "rouge": {
                "rouge_types": rouge_types,
                "use_stemmer": use_stemmer,
                "higher_is_better": True,
            }
        },
        eval_batch_size=batch_size,
    )

    return ROUGEMetric(config=config, rngs=rngs)


def create_perplexity_metric(
    *,
    rngs: nnx.Rngs,
    model_name: str = "mock",
    use_mock: bool = True,
    batch_size: int = 8,
    config_name: str = "perplexity_metric",
) -> PerplexityMetric:
    """Create Perplexity metric with typed configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        model_name: Name of the language model
        use_mock: Whether to use mock implementation
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured PerplexityMetric instance
    """
    config = EvaluationConfig(
        name=config_name,
        metrics=["perplexity"],
        metric_params={
            "perplexity": {
                "model_name": model_name,
                "use_mock": use_mock,
                "higher_is_better": False,
            }
        },
        eval_batch_size=batch_size,
    )

    return PerplexityMetric(config=config, rngs=rngs)


def create_diversity_metric(
    *,
    rngs: nnx.Rngs,
    n_gram_sizes: list[int] | None = None,
    measure_self_bleu: bool = False,
    batch_size: int = 32,
    config_name: str = "diversity_metric",
) -> DiversityMetric:
    """Create Diversity metric with typed configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        n_gram_sizes: List of n-gram sizes to evaluate
        measure_self_bleu: Whether to compute self-BLEU
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured DiversityMetric instance
    """
    if n_gram_sizes is None:
        n_gram_sizes = [1, 2, 3]

    config = EvaluationConfig(
        name=config_name,
        metrics=["diversity"],
        metric_params={
            "diversity": {
                "n_gram_sizes": n_gram_sizes,
                "measure_self_bleu": measure_self_bleu,
                "higher_is_better": True,
            }
        },
        eval_batch_size=batch_size,
    )

    return DiversityMetric(config=config, rngs=rngs)
