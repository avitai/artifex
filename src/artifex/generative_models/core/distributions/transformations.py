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

    def __call__(
        self, x: jax.Array | None = None, *, rngs: nnx.Rngs | None = None
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Compute log probability of x or sample if x is None.

        Args:
            x: Input tensor. If None, returns a sample.
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            If x is provided, returns log probability of x.
            If x is None, returns a sample from the distribution.
        """
        if x is None:
            return self.sample(sample_shape=(), rngs=rngs)
        return self.log_prob(x)

    def sample(self, sample_shape: tuple = (), *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Sample from the transformed distribution.

        Args:
            sample_shape: Shape of the samples
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            Samples from the distribution
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized.")

        # Handle RNG keys - JIT-compatible approach
        if rngs is None:
            rngs = self._rngs

        if rngs is None:
            raise ValueError("rngs must be provided for sampling")

        sample_key = rngs.sample()

        return self._dist.sample(seed=sample_key, sample_shape=sample_shape)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Compute log probability of x.

        Args:
            x: Input tensor

        Returns:
            Log probability of x
        """
        return super().log_prob(x)

    def entropy(self) -> jax.Array:
        """Compute the entropy of the distribution.

        Returns:
            Entropy of the distribution
        """
        return super().entropy()


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

    def __call__(
        self, x: jax.Array | None = None, *, rngs: nnx.Rngs | None = None
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Compute log probability of x or sample if x is None.

        Args:
            x: Input tensor. If None, returns a sample.
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            If x is provided, returns log probability of x.
            If x is None, returns a sample from the distribution.
        """
        if x is None:
            return self.sample(sample_shape=(), rngs=rngs)
        return self.log_prob(x)

    def sample(self, sample_shape: tuple = (), *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Sample from the transformed distribution.

        Args:
            sample_shape: Shape of the samples
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            Samples from the distribution
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized.")

        # Handle RNG keys - JIT-compatible approach
        if rngs is None:
            rngs = self._rngs

        if rngs is None:
            raise ValueError("rngs must be provided for sampling")

        sample_key = rngs.sample()

        return self._dist.sample(seed=sample_key, sample_shape=sample_shape)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Compute log probability of x.

        Args:
            x: Input tensor

        Returns:
            Log probability of x
        """
        return super().log_prob(x)

    def entropy(self) -> jax.Array:
        """Compute the entropy of the distribution.

        Returns:
            Entropy of the distribution
        """
        # Note: Entropy of a transformed distribution requires the Jacobian
        # determinant. This is only provided by some bijectors, so we use
        # the base implementation.
        return super().entropy()
