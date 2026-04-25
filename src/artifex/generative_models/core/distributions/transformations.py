"""Distribution transformations.

This module provides transformations for distributions that enable operations
like shifting, scaling, and more complex bijective transformations.
"""

import distrax
import jax
from flax import nnx

from .base import Distribution


class AffineTransform(Distribution):
    """Affine transformation of a distribution.

    This transformation applies the mapping
      y = scale * x + shift to a base distribution.
    """

    def __init__(
        self,
        base_distribution: Distribution,
        shift: jax.Array | None = None,
        scale: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize an affine transformed distribution.

        Args:
            base_distribution: The base distribution to transform
            shift: Shift parameter (loc). If None, no shift is applied.
            scale: Scale parameter. If None, no scaling is applied.
            rngs: Random number generators for initialization and sampling.
        """
        super().__init__(rngs=rngs)
        self.base_distribution = base_distribution

        self.shift = shift if shift is not None else 0.0
        self.scale = scale if scale is not None else 1.0
        self._dist = nnx.data(True)

    def _distribution(self) -> distrax.Transformed:
        """Construct the current affine-transformed Distrax distribution."""
        if self._dist is None:
            raise ValueError("Distribution not initialized.")
        bijector = distrax.ScalarAffine(
            shift=self._materialize_array(self.shift),
            scale=self._materialize_array(self.scale),
        )
        return distrax.Transformed(
            distribution=self.base_distribution._distribution(),
            bijector=bijector,
        )


class TransformedDistribution(Distribution):
    """Generic transformed distribution with a custom bijector.

    This class allows applying arbitrary bijectors to distributions.
    """

    def __init__(
        self,
        base_distribution: Distribution,
        bijector: distrax.Bijector,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize a transformed distribution.

        Args:
            base_distribution: The base distribution to transform
            bijector: The bijector to apply to the distribution
            rngs: Random number generators for initialization and sampling.
        """
        super().__init__(rngs=rngs)
        self.base_distribution = base_distribution
        self.bijector = nnx.data(bijector)
        self._dist = nnx.data(True)

    def _distribution(self) -> distrax.Transformed:
        """Construct the current Distrax transformed distribution."""
        if self._dist is None:
            raise ValueError("Distribution not initialized.")
        return distrax.Transformed(
            distribution=self.base_distribution._distribution(),
            bijector=self._materialize_leaf(self.bijector),
        )

    def entropy(self) -> jax.Array:
        """Compute the entropy of the distribution.

        Returns:
            Entropy of the distribution
        """
        # Entropy of a transformed distribution requires the Jacobian
        # determinant. This is only provided by some bijectors, so the shared
        # base implementation is the best available default.
        return super().entropy()
