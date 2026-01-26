"""Multi-modal evaluation metrics and suite.

This module provides evaluation metrics for multi-modal models, including
cross-modal alignment, consistency, and quality metrics.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class MultiModalEvaluationSuite(nnx.Module):
    """Evaluation suite for multi-modal models."""

    def __init__(
        self,
        modalities: list[str],
        metrics: list[str] | None = None,
        alignment_method: str = "cosine",
        alignment_dim: int = 256,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize multi-modal evaluation suite.

        Args:
            modalities: List of modality names
            metrics: List of metric names to compute
            alignment_method: Method for computing alignment
            alignment_dim: Common dimension for alignment computation
            rngs: Random number generators
        """
        super().__init__()
        self.modalities = modalities
        self.alignment_method = alignment_method
        self.alignment_dim = alignment_dim

        if metrics is None:
            metrics = ["alignment", "consistency", "quality"]
        self.metrics = metrics

        # Initialize alignment projectors for each modality using nnx.Dict
        if rngs is not None:
            self._init_alignment_projectors(rngs)
        else:
            self.alignment_projectors = nnx.Dict({})

        # Initialize modality-specific evaluators
        self._init_modality_evaluators(rngs)

    def _init_alignment_projectors(self, rngs: nnx.Rngs):
        """Initialize alignment projectors for each modality.

        Args:
            rngs: Random number generators
        """
        # Build projectors dict first, then wrap in nnx.Dict
        projectors: dict[str, nnx.Module] = {}

        # Define input dimensions for each modality
        modality_dims = {
            "image": 3072,  # 32x32x3 flattened
            "text": 50,  # Text embedding dimension
            "audio": 256,  # Audio feature dimension
        }

        for modality in self.modalities:
            input_dim = modality_dims.get(modality, 256)

            # Create projector to map to common alignment dimension
            projectors[modality] = nnx.Sequential(
                nnx.Linear(
                    in_features=input_dim,
                    out_features=512,
                    rngs=rngs,
                ),
                nnx.relu,
                nnx.Linear(
                    in_features=512,
                    out_features=self.alignment_dim,
                    rngs=rngs,
                ),
            )

        # Store as nnx.Dict for proper parameter tracking
        self.alignment_projectors = nnx.Dict(projectors)

    def _init_modality_evaluators(self, rngs: nnx.Rngs | None):
        """Initialize evaluators for individual modalities.

        Args:
            rngs: Random number generators
        """
        # Build evaluators dict first, then wrap in nnx.Dict
        evaluators: dict[str, nnx.Module] = {}

        # Create evaluators with appropriate configs
        for modality in self.modalities:
            if modality == "image":
                from artifex.generative_models.core.configuration import ModalityConfig
                from artifex.generative_models.modalities.image.evaluation import (
                    ImageEvaluationSuite,
                )

                config = ModalityConfig(
                    name="image_modality",
                    modality_name="image",
                    supported_models=["vae", "gan", "diffusion"],
                    default_metrics=["fid", "is", "lpips"],
                    preprocessing_steps=[],
                )
                evaluators[modality] = ImageEvaluationSuite(config=config, rngs=rngs or nnx.Rngs())
            elif modality == "text":
                from artifex.generative_models.core.configuration import ModalityConfig
                from artifex.generative_models.modalities.text.evaluation import (
                    TextEvaluationSuite,
                )

                config = ModalityConfig(
                    name="text_modality",
                    modality_name="text",
                    supported_models=["transformer", "vae"],
                    default_metrics=["perplexity", "bleu", "rouge"],
                    preprocessing_steps=[],
                    metadata={
                        "text_params": {
                            "vocab_size": 10000,
                            "max_length": 128,
                            "pad_token_id": 0,
                        }
                    },
                )
                evaluators[modality] = TextEvaluationSuite(config=config, rngs=rngs or nnx.Rngs())
            elif modality == "audio":
                from artifex.generative_models.modalities.audio.base import (
                    AudioModalityConfig,
                )
                from artifex.generative_models.modalities.audio.evaluation import (
                    AudioEvaluationSuite,
                )

                config = AudioModalityConfig()
                evaluators[modality] = AudioEvaluationSuite(config=config, rngs=rngs or nnx.Rngs())

        # Store as nnx.Dict for proper parameter tracking
        self.modality_evaluators = nnx.Dict(evaluators)

    def evaluate(
        self,
        generated: dict[str, jax.Array],
        reference: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate multi-modal generation quality.

        Args:
            generated: Generated samples for each modality
            reference: Reference samples for each modality
            **kwargs: Additional arguments

        Returns:
            Dictionary of evaluation metrics
        """
        results = {}

        # Compute modality-specific metrics
        if "quality" in self.metrics:
            quality_scores = self._compute_quality_metrics(generated, reference)
            results["quality_scores"] = quality_scores

        # Compute cross-modal alignment
        if "alignment" in self.metrics:
            alignment_score = self._compute_alignment_score(generated)
            results["alignment_score"] = alignment_score

        # Compute multi-modal consistency
        if "consistency" in self.metrics:
            consistency_score = self._compute_consistency_score(generated, reference)
            results["consistency_score"] = consistency_score

        # Compute overall score
        overall_score = self._compute_overall_score(results)
        results["overall_score"] = overall_score

        return results

    def _compute_quality_metrics(
        self,
        generated: dict[str, jax.Array],
        reference: dict[str, jax.Array],
    ) -> dict[str, float]:
        """Compute quality metrics for each modality.

        Args:
            generated: Generated samples
            reference: Reference samples

        Returns:
            Quality scores for each modality
        """
        quality_scores = {}

        # First handle modalities that have evaluators - iterate over nnx.Dict
        for modality, evaluator in self.modality_evaluators.items():
            if modality in generated and modality in reference:
                # Different modalities have different evaluation methods
                if modality == "image" and hasattr(evaluator, "evaluate"):
                    scores = evaluator.evaluate(
                        generated[modality],
                        reference[modality],
                    )
                    quality_scores[f"{modality}_quality"] = scores.get("overall_score", 0.0)
                elif modality == "text" and hasattr(evaluator, "evaluate_batch"):
                    evaluator.evaluate_batch(
                        generated[modality],
                        reference[modality],
                    )
                    # Extract a simple score from TextMetrics
                    quality_scores[f"{modality}_quality"] = 1.0  # Placeholder
                elif modality == "audio" and hasattr(evaluator, "evaluate"):
                    scores = evaluator.evaluate(
                        generated[modality],
                        reference[modality],
                    )
                    quality_scores[f"{modality}_quality"] = scores.get("overall_score", 0.0)

        # Handle remaining modalities with fallback MSE
        evaluator_keys = set(self.modality_evaluators.keys())
        for modality in self.modalities:
            if modality not in evaluator_keys:
                if modality in generated and modality in reference:
                    # Fallback to MSE
                    mse = jnp.mean((generated[modality] - reference[modality]) ** 2)
                    quality_scores[f"{modality}_quality"] = float(1.0 / (1.0 + mse))

        return quality_scores

    def _compute_alignment_score(self, samples: dict[str, jax.Array]) -> float:
        """Compute cross-modal alignment score.

        Args:
            samples: Samples from each modality

        Returns:
            Alignment score
        """
        if len(samples) < 2:
            return 1.0

        # Compute pairwise alignment scores only for known modalities
        alignment_scores = []
        # Filter to only modalities we know about
        modality_list = [m for m in samples.keys() if m in self.modalities]

        for i in range(len(modality_list)):
            for j in range(i + 1, len(modality_list)):
                mod1, mod2 = modality_list[i], modality_list[j]

                if mod1 in samples and mod2 in samples:
                    score = self._compute_pairwise_alignment(
                        samples[mod1],
                        samples[mod2],
                        mod1,
                        mod2,
                    )
                    alignment_scores.append(score)

        if alignment_scores:
            return float(jnp.mean(jnp.array(alignment_scores)))
        else:
            return 0.0

    def _compute_pairwise_alignment(
        self,
        features1: jax.Array,
        features2: jax.Array,
        modality1: str,
        modality2: str,
    ) -> float:
        """Compute alignment between two modalities.

        Args:
            features1: Features from first modality
            features2: Features from second modality
            modality1: Name of first modality
            modality2: Name of second modality

        Returns:
            Pairwise alignment score
        """
        # Handle both batched and unbatched inputs
        # For unbatched inputs, add batch dimension
        if isinstance(features1, (int, float)):
            # Handle scalar case
            features1 = jnp.array([[features1]])
        elif features1.ndim == 1:
            # 1D array - add batch dimension
            features1 = features1[jnp.newaxis, :]
        elif features1.ndim == 2:
            # Could be (batch, features) or (height, width) for grayscale
            # Check if it looks like an image shape
            if features1.shape[0] > 10 and features1.shape[1] > 10:
                # Likely an image, add batch dimension
                features1 = features1[jnp.newaxis, ...]
        elif features1.ndim == 3:
            # Could be (batch, height, width) or (height, width, channels)
            # Check last dimension - if it's small (1-4), likely channels
            if features1.shape[-1] <= 4:
                # (height, width, channels) - add batch dimension
                features1 = features1[jnp.newaxis, ...]

        # Same for features2
        if isinstance(features2, (int, float)):
            features2 = jnp.array([[features2]])
        elif features2.ndim == 1:
            features2 = features2[jnp.newaxis, :]
        elif features2.ndim == 2:
            if features2.shape[0] > 10 and features2.shape[1] > 10:
                features2 = features2[jnp.newaxis, ...]
        elif features2.ndim == 3:
            if features2.shape[-1] <= 4:
                features2 = features2[jnp.newaxis, ...]

        # Now flatten to (batch, features)
        feat1 = features1.reshape(features1.shape[0], -1)
        feat2 = features2.reshape(features2.shape[0], -1)

        # Ensure same number of samples
        min_samples = min(feat1.shape[0], feat2.shape[0])
        feat1 = feat1[:min_samples]
        feat2 = feat2[:min_samples]

        # Project to common alignment dimension if projectors are available
        # Get projector keys using .keys() to avoid nnx.Dict membership issues
        projector_keys = set(self.alignment_projectors.keys())
        if projector_keys:
            if modality1 in projector_keys:
                feat1 = self.alignment_projectors[modality1](feat1)
            else:
                # If no projector, create a simple linear projection to alignment dim
                current_dim = feat1.shape[-1]
                if current_dim != self.alignment_dim:
                    # Simple projection using random matrix (not trainable, just for compatibility)
                    key = jax.random.key(hash(modality1) % 2**32)
                    W = jax.random.normal(key, (current_dim, self.alignment_dim)) * 0.02
                    feat1 = feat1 @ W

            if modality2 in projector_keys:
                feat2 = self.alignment_projectors[modality2](feat2)
            else:
                # If no projector, create a simple linear projection to alignment dim
                current_dim = feat2.shape[-1]
                if current_dim != self.alignment_dim:
                    key = jax.random.key(hash(modality2) % 2**32)
                    W = jax.random.normal(key, (current_dim, self.alignment_dim)) * 0.02
                    feat2 = feat2 @ W

        if self.alignment_method == "cosine":
            # Cosine similarity
            feat1_norm = feat1 / (jnp.linalg.norm(feat1, axis=1, keepdims=True) + 1e-8)
            feat2_norm = feat2 / (jnp.linalg.norm(feat2, axis=1, keepdims=True) + 1e-8)

            # Average cosine similarity
            similarities = jnp.sum(feat1_norm * feat2_norm, axis=1)
            score = jnp.mean(similarities)
        else:
            # Euclidean distance-based score
            distances = jnp.linalg.norm(feat1 - feat2, axis=1)
            score = 1.0 / (1.0 + jnp.mean(distances))

        return float(score)

    def _compute_consistency_score(
        self,
        generated: dict[str, jax.Array],
        reference: dict[str, jax.Array],
    ) -> float:
        """Compute multi-modal consistency score.

        Args:
            generated: Generated samples
            reference: Reference samples

        Returns:
            Consistency score
        """
        # Compute how consistent the generated modalities are with each other
        # compared to the reference

        gen_alignment = self._compute_alignment_score(generated)
        ref_alignment = self._compute_alignment_score(reference)

        # Consistency is high if generated alignment is similar to reference
        consistency = 1.0 - abs(gen_alignment - ref_alignment)

        return float(consistency)

    def _compute_overall_score(self, results: dict[str, float | dict]) -> float:
        """Compute overall multi-modal score.

        Args:
            results: Individual metric results

        Returns:
            Overall score
        """
        scores = []

        # Add quality scores
        if "quality_scores" in results:
            quality_scores = results["quality_scores"]
            if isinstance(quality_scores, dict):
                scores.extend(quality_scores.values())

        # Add alignment score
        if "alignment_score" in results:
            scores.append(results["alignment_score"])

        # Add consistency score
        if "consistency_score" in results:
            scores.append(results["consistency_score"])

        if scores:
            return float(jnp.mean(jnp.array(scores)))
        else:
            return 0.0


def compute_multi_modal_metrics(
    generated: dict[str, jax.Array],
    reference: dict[str, jax.Array],
    modalities: list[str],
    metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute multi-modal metrics.

    Args:
        generated: Generated samples for each modality
        reference: Reference samples for each modality
        modalities: List of modality names
        metric_names: Specific metrics to compute

    Returns:
        Dictionary of computed metrics
    """
    results = {}

    if metric_names is None:
        metric_names = ["mse", "alignment"]

    # Compute modality-specific metrics
    if "mse" in metric_names:
        for modality in modalities:
            if modality in generated and modality in reference:
                mse = jnp.mean((generated[modality] - reference[modality]) ** 2)
                results[f"{modality}_mse"] = float(mse)

    # Compute cross-modal alignment
    if "alignment" in metric_names:
        alignment_scores = []

        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]

                if mod1 in generated and mod2 in generated:
                    # Simple alignment based on correlation
                    feat1 = generated[mod1].reshape(-1)
                    feat2 = generated[mod2].reshape(-1)

                    # Ensure same size
                    min_size = min(len(feat1), len(feat2))
                    feat1 = feat1[:min_size]
                    feat2 = feat2[:min_size]

                    # Compute correlation
                    corr = jnp.corrcoef(feat1, feat2)[0, 1]
                    alignment_scores.append(corr)

        if alignment_scores:
            results["cross_modal_alignment"] = float(jnp.mean(jnp.array(alignment_scores)))

    return results


def multi_modal_consistency_loss(
    representations: dict[str, jax.Array],
    target_similarity: float = 1.0,
) -> jax.Array:
    """Compute multi-modal consistency loss.

    This loss encourages different modalities to have consistent representations.

    Args:
        representations: Dictionary of modality representations
        target_similarity: Target similarity between modalities

    Returns:
        Consistency loss value
    """
    if len(representations) < 2:
        return jnp.array(0.0)

    losses = []
    modalities = list(representations.keys())

    # Compute pairwise consistency losses
    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            feat1 = representations[modalities[i]]
            feat2 = representations[modalities[j]]

            # Normalize features
            feat1_norm = feat1 / (jnp.linalg.norm(feat1) + 1e-8)
            feat2_norm = feat2 / (jnp.linalg.norm(feat2) + 1e-8)

            # Compute cosine similarity
            similarity = jnp.sum(feat1_norm * feat2_norm)

            # Loss is distance from target similarity
            loss = (similarity - target_similarity) ** 2
            losses.append(loss)

    return jnp.mean(jnp.array(losses))


class CrossModalContrastiveLoss(nnx.Module):
    """Cross-modal contrastive loss for alignment."""

    def __init__(
        self,
        temperature: float = 0.07,
        similarity_metric: str = "cosine",
    ):
        """Initialize cross-modal contrastive loss.

        Args:
            temperature: Temperature parameter for scaling
            similarity_metric: Similarity metric to use
        """
        super().__init__()
        self.temperature = temperature
        self.similarity_metric = similarity_metric

    def __call__(
        self,
        features1: jax.Array,
        features2: jax.Array,
        labels: jax.Array | None = None,
    ) -> jax.Array:
        """Compute cross-modal contrastive loss.

        Args:
            features1: Features from first modality [batch_size, feature_dim]
            features2: Features from second modality [batch_size, feature_dim]
            labels: Optional pairing labels

        Returns:
            Contrastive loss value
        """
        batch_size = features1.shape[0]

        # Normalize features
        features1_norm = features1 / (jnp.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
        features2_norm = features2 / (jnp.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)

        # Compute similarity matrix
        if self.similarity_metric == "cosine":
            similarity = jnp.matmul(features1_norm, features2_norm.T) / self.temperature
        else:
            # Euclidean distance-based similarity
            dist_matrix = jnp.sum(
                (features1_norm[:, None, :] - features2_norm[None, :, :]) ** 2,
                axis=2,
            )
            similarity = -dist_matrix / self.temperature

        # If no labels provided, assume diagonal pairs are positive
        if labels is None:
            labels = jnp.eye(batch_size)

        # Cross-entropy loss
        log_probs = jax.nn.log_softmax(similarity, axis=1)
        loss1 = -jnp.sum(labels * log_probs) / batch_size

        # Symmetric loss
        log_probs2 = jax.nn.log_softmax(similarity.T, axis=1)
        loss2 = -jnp.sum(labels.T * log_probs2) / batch_size

        return (loss1 + loss2) / 2
