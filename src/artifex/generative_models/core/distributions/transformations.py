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

        # Handle default values
        self.shift = shift if shift is not None else 0.0
        self.scale = scale if scale is not None else 1.0

        # Create distrax bijector and wrap with nnx.data for NNX compatibility
        # (bijectors contain JAX arrays internally)
        bijector = distrax.ScalarAffine(shift=self.shift, scale=self.scale)
        self.bijector = nnx.data(bijector)

        # Create transformed distribution and wrap with nnx.data for NNX compatibility
        self._dist = nnx.data(
            distrax.Transformed(distribution=self.base_distribution._dist, bijector=bijector)
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
        # Wrap bijector with nnx.data for NNX compatibility (bijectors contain JAX arrays)
        self.bijector = nnx.data(bijector)

        # Create transformed distribution and wrap with nnx.data for NNX compatibility
        self._dist = nnx.data(
            distrax.Transformed(distribution=self.base_distribution._dist, bijector=bijector)
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
