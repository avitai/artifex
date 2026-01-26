"""Disentanglement metrics for generative models.

This module provides metrics for evaluating disentanglement in generative models,
particularly for VAE models with controllable generation capabilities.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from artifex.benchmarks.metrics.core import MetricBase
from artifex.generative_models.core.configuration import EvaluationConfig


class MutualInformationGapMetric(MetricBase):
    """Mutual Information Gap (MIG) metric for disentanglement evaluation.

    MIG measures the normalized gap in mutual information between the top two
    latent dimensions that are most informative about a specific ground truth factor.
    Higher MIG values indicate better disentanglement.

    Reference:
        Chen et al. "Isolating Sources of Disentanglement in Variational Autoencoders"
        https://arxiv.org/abs/1802.04942
    """

    def __init__(self, *, config: EvaluationConfig, rngs: nnx.Rngs):
        """Initialize MIG metric.

        Args:
            config: Evaluation configuration (must be EvaluationConfig)
            rngs: NNX Rngs for stochastic operations
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size
        self.metric_name = "mig_score"

    def compute(
        self, real_data: jax.Array, generated_data: jax.Array, **kwargs
    ) -> dict[str, float]:
        """Compute the MIG score.

        Args:
            real_data: Ground truth factors [batch_size, num_factors]
            generated_data: Latent vectors from the model [batch_size, latent_dim]
            **kwargs: Additional parameters including:
                - latent_representations: Alternative way to pass latent codes
                - ground_truth_factors: Alternative way to pass factors

        Returns:
            Dictionary with MIG score
        """
        # Handle both direct parameters and kwargs for backward compatibility
        if "latent_representations" in kwargs and "ground_truth_factors" in kwargs:
            latent_representations = kwargs["latent_representations"]
            ground_truth_factors = kwargs["ground_truth_factors"]
        else:
            # Use the standard protocol: real_data = factors, generated_data = latents
            ground_truth_factors = real_data
            latent_representations = generated_data
        # Convert to numpy for computation
        latents = jnp.array(latent_representations)
        factors = jnp.array(ground_truth_factors)

        # Normalize latent representations
        latents = self._normalize_representations(latents)

        # Calculate mutual information for each latent-factor pair
        latent_dim = latents.shape[1]
        num_factors = factors.shape[1]
        mi_matrix = jnp.zeros((num_factors, latent_dim))

        for f_idx in range(num_factors):
            for l_idx in range(latent_dim):
                mi_matrix = mi_matrix.at[f_idx, l_idx].set(
                    self._compute_mutual_information(latents[:, l_idx], factors[:, f_idx])
                )

        # For each factor, find the gap between top two latent dimensions
        sorted_mi = jnp.sort(mi_matrix, axis=1)[:, ::-1]  # Sort in descending order
        gaps = sorted_mi[:, 0] - sorted_mi[:, 1]

        # Normalize by entropy of each factor
        factor_entropies = jnp.array(
            [self._compute_entropy(factors[:, f_idx]) for f_idx in range(num_factors)]
        )
        normalized_gaps = gaps / (factor_entropies + 1e-8)

        # MIG is the mean of normalized gaps
        mig_score = float(jnp.mean(normalized_gaps))

        return {"mig_score": mig_score}

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> bool:
        """Validate input data compatibility.

        Args:
            real_data: Ground truth factors
            generated_data: Latent representations

        Returns:
            True if inputs are valid
        """
        # Check that both inputs are arrays
        if not isinstance(real_data, jax.Array) or not isinstance(generated_data, jax.Array):
            return False

        # Check that both are 2D arrays
        if len(real_data.shape) != 2 or len(generated_data.shape) != 2:
            return False

        # Check that batch sizes match
        if real_data.shape[0] != generated_data.shape[0]:
            return False

        return True

    def _normalize_representations(self, latents: jax.Array) -> jax.Array:
        """Normalize latent representations to have zero mean and unit variance.

        Args:
            latents: Latent vectors [batch_size, latent_dim]

        Returns:
            Normalized latent vectors
        """
        scaler = StandardScaler()
        return scaler.fit_transform(latents)

    def _compute_mutual_information(self, x: jax.Array, y: jax.Array) -> float:
        """Compute mutual information between two variables.

        Args:
            x: First variable
            y: Second variable

        Returns:
            Mutual information value
        """
        # Discretize continuous variables if needed
        if jnp.issubdtype(x.dtype, jnp.floating):
            x = self._discretize(x)
        if jnp.issubdtype(y.dtype, jnp.floating):
            y = self._discretize(y)

        # Compute mutual information using entropy
        h_x = self._compute_entropy(x)
        h_y = self._compute_entropy(y)
        h_xy = self._compute_joint_entropy(x, y)

        mi = h_x + h_y - h_xy
        return max(0.0, mi)  # Ensure non-negative

    def _discretize(self, x: jax.Array, num_bins: int = 20) -> jax.Array:
        """Discretize continuous variable into bins.

        Args:
            x: Continuous variable
            num_bins: Number of bins

        Returns:
            Discretized variable
        """
        return jnp.digitize(x, jnp.linspace(jnp.min(x), jnp.max(x), num_bins))

    def _compute_entropy(self, x: jax.Array) -> float:
        """Compute entropy of a discrete variable.

        Args:
            x: Discrete variable

        Returns:
            Entropy value
        """
        _, counts = jnp.unique(x, return_counts=True)
        probs = counts / len(x)
        return float(-jnp.sum(probs * jnp.log(probs + 1e-10)))

    def _compute_joint_entropy(self, x: jax.Array, y: jax.Array) -> float:
        """Compute joint entropy of two discrete variables.

        Args:
            x: First discrete variable
            y: Second discrete variable

        Returns:
            Joint entropy value
        """
        # Create joint variable
        joint = jnp.stack([x, y], axis=1)

        # Count unique combinations
        _, counts = jnp.unique(joint, return_counts=True, axis=0)
        probs = counts / len(joint)

        return float(-jnp.sum(probs * jnp.log(probs + 1e-10)))


class SeparationMetric(MetricBase):
    """Separated Attribute Predictability (SAP) metric for disentanglement.

    SAP measures how well each ground truth factor can be predicted from
    individual latent dimensions. Higher SAP scores indicate better separation
    of factors in the latent space.

    Reference:
        Kumar et al. "Variational Inference of Disentangled Latent Concepts
         from Unlabeled Observations"
        https://arxiv.org/abs/1711.00848
    """

    def __init__(self, *, config: EvaluationConfig, rngs: nnx.Rngs):
        """Initialize SAP metric.

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
        self.metric_name = "sap_score"

    def compute(
        self, real_data: jax.Array, generated_data: jax.Array, **kwargs
    ) -> dict[str, float]:
        """Compute the SAP score.

        Args:
            real_data: Ground truth factors [batch_size, num_factors]
            generated_data: Latent vectors from the model [batch_size, latent_dim]
            **kwargs: Additional parameters including:
                - latent_representations: Alternative way to pass latent codes
                - ground_truth_factors: Alternative way to pass factors

        Returns:
            Dictionary with SAP score
        """
        # Handle both direct parameters and kwargs for backward compatibility
        if "latent_representations" in kwargs and "ground_truth_factors" in kwargs:
            latent_representations = kwargs["latent_representations"]
            ground_truth_factors = kwargs["ground_truth_factors"]
        else:
            # Use the standard protocol: real_data = factors, generated_data = latents
            ground_truth_factors = real_data
            latent_representations = generated_data
        # Convert to numpy for computation
        latents = jnp.array(latent_representations)
        factors = jnp.array(ground_truth_factors)

        # Normalize latent representations
        latents = self._normalize_representations(latents)

        # Calculate predictability scores for each latent-factor pair
        latent_dim = latents.shape[1]
        num_factors = factors.shape[1]
        score_matrix = jnp.zeros((num_factors, latent_dim))

        for f_idx in range(num_factors):
            for l_idx in range(latent_dim):
                # Use single latent dimension to predict factor
                score_matrix = score_matrix.at[f_idx, l_idx].set(
                    self._compute_predictability(latents[:, l_idx : l_idx + 1], factors[:, f_idx])
                )

        # For each factor, find the gap between top two latent dimensions
        sorted_scores = jnp.sort(score_matrix, axis=1)[:, ::-1]  # Sort in descending order
        gaps = sorted_scores[:, 0] - sorted_scores[:, 1]

        # SAP is the mean of these gaps
        sap_score = float(jnp.mean(gaps))

        return {"sap_score": sap_score}

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> bool:
        """Validate input data compatibility.

        Args:
            real_data: Ground truth factors
            generated_data: Latent representations

        Returns:
            True if inputs are valid
        """
        # Check that both inputs are arrays
        if not isinstance(real_data, jax.Array) or not isinstance(generated_data, jax.Array):
            return False

        # Check that both are 2D arrays
        if len(real_data.shape) != 2 or len(generated_data.shape) != 2:
            return False

        # Check that batch sizes match
        if real_data.shape[0] != generated_data.shape[0]:
            return False

        return True

    def _normalize_representations(self, latents: jax.Array) -> jax.Array:
        """Normalize latent representations to have zero mean and unit variance.

        Args:
            latents: Latent vectors [batch_size, latent_dim]

        Returns:
            Normalized latent vectors
        """
        scaler = StandardScaler()
        return scaler.fit_transform(latents)

    def _compute_predictability(self, latent: jax.Array, factor: jax.Array) -> float:
        """Compute how well a factor can be predicted from a latent dimension.

        Args:
            latent: Single latent dimension [batch_size, 1]
            factor: Ground truth factor [batch_size]

        Returns:
            Predictability score (R² for continuous factors, accuracy for binary)
        """
        # Check if factor is binary
        is_binary = jnp.all(jnp.logical_or(factor == 0, factor == 1))

        try:
            if is_binary:
                # Use logistic regression for binary factors
                model = LogisticRegression(solver="lbfgs", max_iter=1000)
                model.fit(latent, factor)
                predictions = model.predict(latent)
                score = jnp.mean(predictions == factor)
            else:
                # Use linear regression for continuous factors
                # Simple linear regression using numpy
                X = jnp.hstack([latent, jnp.ones((latent.shape[0], 1))])
                beta = jnp.linalg.lstsq(X, factor, rcond=None)[0]
                predictions = X @ beta

                # R² score
                score = r2_score(factor, predictions)
        except Exception:
            # Fallback if model fitting fails
            score = 0.0

        return float(max(0.0, score))  # Ensure non-negative


class DisentanglementMetric(MetricBase):
    """Disentanglement, Completeness, Informativeness (DCI) metric.

    DCI evaluates three aspects of disentanglement:
    1. Disentanglement: Each latent dimension captures at most one factor
    2. Completeness: Each factor is captured by at most one latent dimension
    3. Informativeness: How well factors can be predicted from latent dimensions

    Reference:
        Eastwood & Williams, "A Framework for the Quantitative Evaluation
         of Disentangled Representations"
        https://arxiv.org/abs/1806.07547
    """

    def __init__(self, *, config: EvaluationConfig, rngs: nnx.Rngs):
        """Initialize DCI metric.

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
        self.metric_name = "dci_score"

        # DCI parameters from config
        dci_params = config.metric_params.get("dci", {})
        self.weights = dci_params.get(
            "weights", {"disentanglement": 0.4, "completeness": 0.4, "informativeness": 0.2}
        )

    def compute(
        self, real_data: jax.Array, generated_data: jax.Array, **kwargs
    ) -> dict[str, float]:
        """Compute the DCI metrics.

        Args:
            real_data: Ground truth factors [batch_size, num_factors]
            generated_data: Latent vectors from the model [batch_size, latent_dim]
            **kwargs: Additional parameters including:
                - latent_representations: Alternative way to pass latent codes
                - ground_truth_factors: Alternative way to pass factors
                - weights: Weights for combining DCI components

        Returns:
            Dictionary with DCI metrics
        """
        # Handle both direct parameters and kwargs for backward compatibility
        if "latent_representations" in kwargs and "ground_truth_factors" in kwargs:
            latent_representations = kwargs["latent_representations"]
            ground_truth_factors = kwargs["ground_truth_factors"]
        else:
            # Use the standard protocol: real_data = factors, generated_data = latents
            ground_truth_factors = real_data
            latent_representations = generated_data

        # Convert to numpy for computation
        latents = jnp.array(latent_representations)
        factors = jnp.array(ground_truth_factors)

        # Normalize latent representations
        latents = self._normalize_representations(latents)

        # Train a predictor for each factor
        importance_matrix = self._compute_importance_matrix(latents, factors)

        # Calculate DCI metrics
        disentanglement = self._compute_disentanglement(importance_matrix)
        completeness = self._compute_completeness(importance_matrix)
        informativeness = self._compute_informativeness(latents, factors)

        # Combined DCI score (weighted average)
        # Use weights from configuration
        dci_score = (
            self.weights["disentanglement"] * disentanglement
            + self.weights["completeness"] * completeness
            + self.weights["informativeness"] * informativeness
        )

        return {
            "dci_score": float(dci_score),
            "disentanglement": float(disentanglement),
            "completeness": float(completeness),
            "informativeness": float(informativeness),
        }

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> bool:
        """Validate input data compatibility.

        Args:
            real_data: Ground truth factors
            generated_data: Latent representations

        Returns:
            True if inputs are valid
        """
        # Check that both inputs are arrays
        if not isinstance(real_data, jax.Array) or not isinstance(generated_data, jax.Array):
            return False

        # Check that both are 2D arrays
        if len(real_data.shape) != 2 or len(generated_data.shape) != 2:
            return False

        # Check that batch sizes match
        if real_data.shape[0] != generated_data.shape[0]:
            return False

        return True

    def _normalize_representations(self, latents: jax.Array) -> jax.Array:
        """Normalize latent representations to have zero mean and unit variance.

        Args:
            latents: Latent vectors [batch_size, latent_dim]

        Returns:
            Normalized latent vectors
        """
        scaler = StandardScaler()
        return scaler.fit_transform(latents)

    def _compute_importance_matrix(self, latents: jax.Array, factors: jax.Array) -> jax.Array:
        """Compute importance matrix using random forest feature importance.

        Args:
            latents: Latent vectors [batch_size, latent_dim]
            factors: Ground truth factors [batch_size, num_factors]

        Returns:
            Importance matrix [num_factors, latent_dim]
        """
        latent_dim = latents.shape[1]
        num_factors = factors.shape[1]
        importance_matrix = jnp.zeros((num_factors, latent_dim))

        for f_idx in range(num_factors):
            try:
                # Train random forest to predict factor from latents
                rf = RandomForestClassifier(n_estimators=50, max_depth=5)

                # Convert factor to discrete if continuous
                factor = factors[:, f_idx]
                if not jnp.all(jnp.logical_or(factor == 0, factor == 1)):
                    factor = self._discretize(factor)

                rf.fit(latents, factor)

                # Get feature importances
                importance_matrix = importance_matrix.at[f_idx].set(rf.feature_importances_)
            except Exception:
                # Fallback if model fitting fails
                importance_matrix = importance_matrix.at[f_idx].set(
                    jnp.ones(latent_dim) / latent_dim
                )

        # Normalize importance per factor
        row_sums = jnp.sum(importance_matrix, axis=1, keepdims=True)
        importance_matrix = importance_matrix / (row_sums + 1e-10)

        return importance_matrix

    def _compute_disentanglement(self, importance_matrix: jax.Array) -> float:
        """Compute disentanglement score from importance matrix.

        Args:
            importance_matrix: Importance matrix [num_factors, latent_dim]

        Returns:
            Disentanglement score
        """
        # Transpose to get [latent_dim, num_factors]
        importance_per_latent = importance_matrix.T

        # Compute entropy for each latent dimension
        latent_dim = importance_per_latent.shape[0]
        disentanglement_scores = jnp.zeros(latent_dim)

        for l_idx in range(latent_dim):
            # Compute entropy of importance distribution
            p = importance_per_latent[l_idx]
            entropy = -jnp.sum(p * jnp.log(p + 1e-10))
            max_entropy = jnp.log(len(p))

            # Higher value means more concentrated importance (better disentanglement)
            if max_entropy > 0:
                disentanglement_scores = disentanglement_scores.at[l_idx].set(
                    1.0 - entropy / max_entropy
                )
            else:
                disentanglement_scores = disentanglement_scores.at[l_idx].set(1.0)

        # Weight by relative importance of each latent
        latent_importance = importance_per_latent.sum(axis=1)
        latent_importance = latent_importance / latent_importance.sum()

        return float(jnp.sum(latent_importance * disentanglement_scores))

    def _compute_completeness(self, importance_matrix: jax.Array) -> float:
        """Compute completeness score from importance matrix.

        Args:
            importance_matrix: Importance matrix [num_factors, latent_dim]

        Returns:
            Completeness score
        """
        # Compute entropy for each factor
        num_factors = importance_matrix.shape[0]
        completeness_scores = jnp.zeros(num_factors)

        for f_idx in range(num_factors):
            # Compute entropy of importance distribution
            p = importance_matrix[f_idx]
            entropy = -jnp.sum(p * jnp.log(p + 1e-10))
            max_entropy = jnp.log(len(p))

            # Normalized entropy (1 - entropy/max_entropy)
            # Higher value means more concentrated importance (better completeness)
            if max_entropy > 0:
                completeness_scores = completeness_scores.at[f_idx].set(1.0 - entropy / max_entropy)
            else:
                completeness_scores = completeness_scores.at[f_idx].set(1.0)

        # All factors weighted equally
        return float(jnp.mean(completeness_scores))

    def _compute_informativeness(self, latents: jax.Array, factors: jax.Array) -> float:
        """Compute informativeness score.

        Args:
            latents: Latent vectors [batch_size, latent_dim]
            factors: Ground truth factors [batch_size, num_factors]

        Returns:
            Informativeness score
        """
        num_factors = factors.shape[1]
        prediction_scores = jnp.zeros(num_factors)

        for f_idx in range(num_factors):
            try:
                # Train random forest to predict factor from latents
                rf = RandomForestClassifier(n_estimators=50, max_depth=5)

                # Convert factor to discrete if continuous
                factor = factors[:, f_idx]
                if not jnp.all(jnp.logical_or(factor == 0, factor == 1)):
                    factor = self._discretize(factor)

                # Split data for evaluation
                n_samples = len(latents)
                train_size = int(0.8 * n_samples)

                X_train, X_test = latents[:train_size], latents[train_size:]
                y_train, y_test = factor[:train_size], factor[train_size:]

                rf.fit(X_train, y_train)
                prediction_scores = prediction_scores.at[f_idx].set(rf.score(X_test, y_test))
            except Exception:
                # Fallback if model fitting fails
                prediction_scores = prediction_scores.at[f_idx].set(0.0)

        return float(jnp.mean(prediction_scores))

    def _discretize(self, x: jax.Array, num_bins: int = 10) -> jax.Array:
        """Discretize continuous variable into bins.

        Args:
            x: Continuous variable
            num_bins: Number of bins

        Returns:
            Discretized variable
        """
        return jnp.digitize(x, jnp.linspace(jnp.min(x), jnp.max(x), num_bins))


# Factory functions for convenient metric creation
def create_mig_metric(
    *,
    rngs: nnx.Rngs,
    batch_size: int = 32,
    config_name: str = "mig_metric",
) -> MutualInformationGapMetric:
    """Create MIG metric with typed configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured MutualInformationGapMetric instance
    """
    config = EvaluationConfig(
        name=config_name,
        metrics=["mig"],
        metric_params={
            "mig": {
                "higher_is_better": True,
            }
        },
        eval_batch_size=batch_size,
    )

    return MutualInformationGapMetric(config=config, rngs=rngs)


def create_sap_metric(
    *,
    rngs: nnx.Rngs,
    batch_size: int = 32,
    config_name: str = "sap_metric",
) -> SeparationMetric:
    """Create SAP metric with typed configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured SeparationMetric instance
    """
    config = EvaluationConfig(
        name=config_name,
        metrics=["sap"],
        metric_params={
            "sap": {
                "higher_is_better": True,
            }
        },
        eval_batch_size=batch_size,
    )

    return SeparationMetric(config=config, rngs=rngs)


def create_dci_metric(
    *,
    rngs: nnx.Rngs,
    disentanglement_weight: float = 0.4,
    completeness_weight: float = 0.4,
    informativeness_weight: float = 0.2,
    batch_size: int = 32,
    config_name: str = "dci_metric",
) -> DisentanglementMetric:
    """Create DCI metric with typed configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        disentanglement_weight: Weight for disentanglement score
        completeness_weight: Weight for completeness score
        informativeness_weight: Weight for informativeness score
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured DisentanglementMetric instance
    """
    config = EvaluationConfig(
        name=config_name,
        metrics=["dci"],
        metric_params={
            "dci": {
                "higher_is_better": True,
                "weights": {
                    "disentanglement": disentanglement_weight,
                    "completeness": completeness_weight,
                    "informativeness": informativeness_weight,
                },
            }
        },
        eval_batch_size=batch_size,
    )

    return DisentanglementMetric(config=config, rngs=rngs)
