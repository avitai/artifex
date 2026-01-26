"""Protein mixin extensions for generative models.

This module implements protein-specific mixin extensions that add protein
functionality to models such as amino acid type handling.
"""

from typing import Any

import jax
from flax import nnx

from artifex.generative_models.core.configuration import ProteinMixinConfig
from artifex.generative_models.extensions.base import ModelExtension


class ProteinMixinExtension(ModelExtension):
    """Extension for adding protein-specific features to models.

    This extension adds capabilities such as amino acid type encoding
    and protein-specific metadata handling.
    """

    def __init__(
        self,
        config: ProteinMixinConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the protein mixin extension.

        Args:
            config: Protein mixin configuration with:
                - embedding_dim: Dimension for amino acid type embeddings
                - use_one_hot: Whether to use one-hot encoding for amino acids
                - num_aa_types: Number of amino acid types
            rngs: Random number generator keys.
        """
        if not isinstance(config, ProteinMixinConfig):
            raise TypeError(f"config must be ProteinMixinConfig, got {type(config).__name__}")

        super().__init__(config, rngs=rngs)

        # Get protein parameters directly from config
        self.embedding_dim = config.embedding_dim
        self.use_one_hot = config.use_one_hot
        self.num_aa_types = config.num_aa_types

        # Initialize embedding matrix if not using one-hot
        if not self.use_one_hot:
            # Create trainable embedding matrix
            self.aa_embedding = nnx.Param(
                jax.random.normal(jax.random.PRNGKey(0), (self.num_aa_types, self.embedding_dim))
                * 0.1
            )

    def __call__(self, inputs: Any, model_outputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Process model inputs/outputs.

        Args:
            inputs: Original inputs to the model.
            model_outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            dictionary with protein-specific features.
        """
        # Extract amino acid types from inputs
        aatype = inputs.get("aatype", None)

        result: dict[str, Any] = {"extension_type": "protein_mixin"}

        # If aatype is available, create embeddings
        if aatype is not None:
            # Create amino acid type encodings
            aa_encoding = self._encode_aatype(aatype)
            result["aa_encoding"] = aa_encoding

        return result

    def _encode_aatype(self, aatype: jax.Array) -> jax.Array:
        """Create encoding for amino acid types.

        Args:
            aatype: Amino acid type indices with shape [batch_size, num_residues]

        Returns:
            Amino acid type encoding with shape [batch_size, num_residues, embedding_dim]
        """
        if self.use_one_hot:
            # One-hot encoding using JAX
            encoding = jax.nn.one_hot(aatype, self.num_aa_types)
            return encoding
        else:
            # Learned embedding
            encoding = self.aa_embedding[aatype]
            return encoding
