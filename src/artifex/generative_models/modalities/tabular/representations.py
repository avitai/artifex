"""Tabular data representation and processing modules."""

import jax
import jax.numpy as jnp
from flax import nnx

from .base import TabularModalityConfig


class NumericalProcessor(nnx.Module):
    """Processor for numerical features."""

    def __init__(
        self,
        features: list[str],
        normalization_type: str = "standard",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize numerical processor.

        Args:
            features: List of numerical feature names
            normalization_type: Type of normalization
            rngs: Random number generators (unused but kept for consistency)
        """
        super().__init__()
        self.features = features
        self.normalization_type = normalization_type
        self.num_features = len(features)

        # Statistics will be computed during fit
        self._fitted = False
        self.mean = None
        self.std = None
        self.min_vals = None
        self.max_vals = None

    def fit(self, data: dict[str, jax.Array]) -> None:
        """Fit normalization statistics to data.

        Args:
            data: Dictionary mapping feature names to arrays
        """
        feature_arrays = [data[feature] for feature in self.features]
        if not feature_arrays:
            self._fitted = True
            return

        # Stack all numerical features
        X = jnp.stack(feature_arrays, axis=-1)  # [batch_size, num_features]

        if self.normalization_type == "standard":
            self.mean = nnx.data(jnp.mean(X, axis=0))
            self.std = nnx.data(jnp.std(X, axis=0) + 1e-8)  # Add epsilon for stability

        elif self.normalization_type == "minmax":
            self.min_vals = nnx.data(jnp.min(X, axis=0))
            max_vals_temp = jnp.max(X, axis=0)
            # Handle constant features
            max_vals_final = jnp.where(
                max_vals_temp == self.min_vals,
                self.min_vals + 1.0,
                max_vals_temp,
            )
            self.max_vals = nnx.data(max_vals_final)

        elif self.normalization_type == "robust":
            # Use median and IQR for robust scaling
            self.median = nnx.data(jnp.median(X, axis=0))
            q75 = jnp.percentile(X, 75, axis=0)
            q25 = jnp.percentile(X, 25, axis=0)
            self.iqr = nnx.data(q75 - q25 + 1e-8)  # Add epsilon for stability

        self._fitted = True

    def transform(self, data: dict[str, jax.Array]) -> jax.Array:
        """Transform numerical features.

        Args:
            data: Dictionary mapping feature names to arrays

        Returns:
            Normalized numerical features [batch_size, num_features]
        """
        if not self._fitted:
            raise RuntimeError("Processor must be fitted before transform")

        feature_arrays = [data[feature] for feature in self.features]
        if not feature_arrays:
            # Return empty array with correct batch size
            batch_size = next(iter(data.values())).shape[0]
            return jnp.zeros((batch_size, 0))

        X = jnp.stack(feature_arrays, axis=-1)

        if self.normalization_type == "standard":
            return (X - self.mean) / self.std

        elif self.normalization_type == "minmax":
            return (X - self.min_vals) / (self.max_vals - self.min_vals)

        elif self.normalization_type == "robust":
            return (X - self.median) / self.iqr

        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

    def inverse_transform(self, normalized_data: jax.Array) -> jax.Array:
        """Inverse transform normalized features back to original scale.

        Args:
            normalized_data: Normalized features [batch_size, num_features]

        Returns:
            Original scale features
        """
        if not self._fitted:
            raise RuntimeError("Processor must be fitted before inverse transform")

        if self.normalization_type == "standard":
            return normalized_data * self.std + self.mean

        elif self.normalization_type == "minmax":
            return normalized_data * (self.max_vals - self.min_vals) + self.min_vals

        elif self.normalization_type == "robust":
            return normalized_data * self.iqr + self.median

        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")


class CategoricalEncoder(nnx.Module):
    """Encoder for categorical features using one-hot encoding."""

    def __init__(
        self,
        features: list[str],
        vocab_sizes: dict[str, int],
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize categorical encoder.

        Args:
            features: List of categorical feature names
            vocab_sizes: Vocabulary sizes for each feature
            rngs: Random number generators (unused but kept for consistency)
        """
        super().__init__()
        self.features = features
        self.vocab_sizes = vocab_sizes
        self.total_dim = sum(vocab_sizes[f] for f in features)

    def encode(self, data: dict[str, jax.Array]) -> jax.Array:
        """Encode categorical features to one-hot.

        Args:
            data: Dictionary mapping feature names to integer arrays

        Returns:
            One-hot encoded features [batch_size, total_categorical_dim]
        """
        if not self.features:
            # Return empty array with correct batch size
            batch_size = next(iter(data.values())).shape[0]
            return jnp.zeros((batch_size, 0))

        encoded_features = []
        for feature in self.features:
            feature_data = data[feature]
            vocab_size = self.vocab_sizes[feature]

            # One-hot encode
            one_hot = jax.nn.one_hot(feature_data, vocab_size)
            encoded_features.append(one_hot)

        return jnp.concatenate(encoded_features, axis=-1)

    def decode(self, encoded_data: jax.Array) -> dict[str, jax.Array]:
        """Decode one-hot encoded features back to categorical indices.

        Args:
            encoded_data: One-hot encoded features [batch_size, total_categorical_dim]

        Returns:
            Dictionary mapping feature names to categorical indices
        """
        if not self.features:
            return {}

        decoded_features = {}
        start_idx = 0

        for feature in self.features:
            vocab_size = self.vocab_sizes[feature]
            end_idx = start_idx + vocab_size

            # Extract one-hot for this feature and convert to indices
            feature_one_hot = encoded_data[:, start_idx:end_idx]
            feature_indices = jnp.argmax(feature_one_hot, axis=-1)
            decoded_features[feature] = feature_indices

            start_idx = end_idx

        return decoded_features


class TabularProcessor(nnx.Module):
    """Main processor for tabular data combining all feature types."""

    def __init__(
        self,
        config: TabularModalityConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize tabular processor.

        Args:
            config: Tabular modality configuration
            rngs: Random number generators
        """
        super().__init__()
        self.config = config

        # Initialize feature processors
        self.numerical_processor = NumericalProcessor(
            features=config.numerical_features,
            normalization_type=config.normalization_type,
            rngs=rngs,
        )

        self.categorical_encoder = CategoricalEncoder(
            features=config.categorical_features,
            vocab_sizes=config.categorical_vocab_sizes,
            rngs=rngs,
        )

        # Binary and ordinal features are handled directly
        self.binary_features = config.binary_features
        self.ordinal_features = config.ordinal_features
        self.ordinal_orders = config.ordinal_orders

    def fit(self, data: dict[str, jax.Array]) -> None:
        """Fit processors to data.

        Args:
            data: Dictionary mapping feature names to arrays
        """
        self.numerical_processor.fit(data)

    def encode(self, data: dict[str, jax.Array]) -> jax.Array:
        """Encode all features to a single vector.

        Args:
            data: Dictionary mapping feature names to arrays

        Returns:
            Encoded feature vector [batch_size, total_encoding_dim]
        """
        encoded_parts = []

        # Process numerical features
        if self.config.numerical_features:
            numerical_encoded = self.numerical_processor.transform(data)
            encoded_parts.append(numerical_encoded)

        # Process categorical features
        if self.config.categorical_features:
            categorical_encoded = self.categorical_encoder.encode(data)
            encoded_parts.append(categorical_encoded)

        # Process ordinal features
        if self.ordinal_features:
            ordinal_parts = []
            for feature in self.ordinal_features:
                feature_data = data[feature]
                # Convert to normalized ordinal values [0, 1]
                order_size = len(self.ordinal_orders[feature])
                normalized = feature_data.astype(jnp.float32) / (order_size - 1)
                ordinal_parts.append(normalized[:, None])  # Add feature dimension
            ordinal_encoded = jnp.concatenate(ordinal_parts, axis=-1)
            encoded_parts.append(ordinal_encoded)

        # Process binary features
        if self.binary_features:
            binary_parts = []
            for feature in self.binary_features:
                feature_data = data[feature].astype(jnp.float32)
                binary_parts.append(feature_data[:, None])  # Add feature dimension
            binary_encoded = jnp.concatenate(binary_parts, axis=-1)
            encoded_parts.append(binary_encoded)

        if not encoded_parts:
            # Return empty array with correct batch size
            batch_size = next(iter(data.values())).shape[0]
            return jnp.zeros((batch_size, 0))

        return jnp.concatenate(encoded_parts, axis=-1)

    def decode(self, encoded_data: jax.Array) -> dict[str, jax.Array]:
        """Decode encoded features back to original format.

        Args:
            encoded_data: Encoded feature vector [batch_size, total_encoding_dim]

        Returns:
            Dictionary mapping feature names to decoded arrays
        """
        decoded_data = {}
        start_idx = 0

        # Decode numerical features
        if self.config.numerical_features:
            num_numerical = len(self.config.numerical_features)
            end_idx = start_idx + num_numerical
            numerical_encoded = encoded_data[:, start_idx:end_idx]
            numerical_decoded = self.numerical_processor.inverse_transform(numerical_encoded)

            for i, feature in enumerate(self.config.numerical_features):
                decoded_data[feature] = numerical_decoded[:, i]

            start_idx = end_idx

        # Decode categorical features
        if self.config.categorical_features:
            categorical_dim = sum(
                self.config.categorical_vocab_sizes[f] for f in self.config.categorical_features
            )
            end_idx = start_idx + categorical_dim
            categorical_encoded = encoded_data[:, start_idx:end_idx]
            categorical_decoded = self.categorical_encoder.decode(categorical_encoded)
            decoded_data.update(categorical_decoded)
            start_idx = end_idx

        # Decode ordinal features
        if self.ordinal_features:
            for feature in self.ordinal_features:
                end_idx = start_idx + 1
                normalized_value = encoded_data[:, start_idx:end_idx].squeeze(-1)
                # Convert back to ordinal indices
                order_size = len(self.ordinal_orders[feature])
                ordinal_value = jnp.round(normalized_value * (order_size - 1)).astype(jnp.int32)
                # Clip to valid range
                ordinal_value = jnp.clip(ordinal_value, 0, order_size - 1)
                decoded_data[feature] = ordinal_value
                start_idx = end_idx

        # Decode binary features
        if self.binary_features:
            for feature in self.binary_features:
                end_idx = start_idx + 1
                binary_value = encoded_data[:, start_idx:end_idx].squeeze(-1)
                # Convert to binary (threshold at 0.5)
                binary_value = (binary_value > 0.5).astype(jnp.int32)
                decoded_data[feature] = binary_value
                start_idx = end_idx

        return decoded_data

    def get_encoding_dimensions(self) -> dict[str, tuple[int, int]]:
        """Get start and end indices for each feature type in encoded vector.

        Returns:
            Dictionary mapping feature types to (start_idx, end_idx) tuples
        """
        dimensions = {}
        start_idx = 0

        if self.config.numerical_features:
            num_numerical = len(self.config.numerical_features)
            dimensions["numerical"] = (start_idx, start_idx + num_numerical)
            start_idx += num_numerical

        if self.config.categorical_features:
            categorical_dim = sum(
                self.config.categorical_vocab_sizes[f] for f in self.config.categorical_features
            )
            dimensions["categorical"] = (start_idx, start_idx + categorical_dim)
            start_idx += categorical_dim

        if self.ordinal_features:
            num_ordinal = len(self.ordinal_features)
            dimensions["ordinal"] = (start_idx, start_idx + num_ordinal)
            start_idx += num_ordinal

        if self.binary_features:
            num_binary = len(self.binary_features)
            dimensions["binary"] = (start_idx, start_idx + num_binary)
            start_idx += num_binary

        return dimensions
