"""Continuous probability distributions backed by Distrax."""

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from .base import Distribution


class Normal(Distribution):
    """Normal (Gaussian) distribution.

    This class wraps distrax's Normal distribution and
    integrates it with Flax's nnx framework.
    """

    def __init__(
        self,
        loc: jax.Array | None = None,
        scale: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        trainable_loc: bool = True,
        trainable_scale: bool = True,
    ):
        """Initialize a Normal distribution with optimized variable types.

        Args:
            loc: Mean of the distribution. If None,
                will be initialized as a parameter.
            scale: Standard deviation of the distribution.
                If None, will be initialized as a parameter.
            rngs: Random number generators for initialization and sampling.
            trainable_loc: Whether location parameter should be trainable
            trainable_scale: Whether scale parameter should be trainable
        """
        super().__init__(rngs=rngs)

        # Initialize parameters with appropriate variable types
        if loc is not None:
            if trainable_loc:
                self.loc = nnx.Param(loc)
            else:
                self.loc = nnx.Cache(loc)  # Static parameter for better JIT performance
        else:
            self.loc = nnx.Param(jnp.zeros(()))  # Default to trainable

        if scale is not None:
            if trainable_scale:
                self.scale = nnx.Param(scale)
            else:
                self.scale = nnx.Cache(scale)  # Static parameter for better JIT performance
        else:
            self.scale = nnx.Param(jnp.ones(()))  # Default to trainable

        # Store trainability flags for optimization
        self._trainable_loc = nnx.Cache(trainable_loc)
        self._trainable_scale = nnx.Cache(trainable_scale)

    def _distribution(self) -> distrax.Normal:
        """Construct the current Distrax Normal from NNX parameters."""
        return distrax.Normal(
            loc=self._materialize_array(self.loc),
            scale=self._materialize_array(self.scale),
        )

    def kl_divergence(self, other: Distribution) -> jax.Array:
        """Compute KL divergence between this Normal and another.

        Args:
            other: Another distribution

        Returns:
            KL divergence
        """
        if not isinstance(other, Normal):
            return super().kl_divergence(other)

        # For two Normal distributions, we can compute KL divergence analytically
        self_loc = self._materialize_array(self.loc)
        self_scale = self._materialize_array(self.scale)
        other_loc = self._materialize_array(other.loc)
        other_scale = self._materialize_array(other.scale)
        return 0.5 * (
            jnp.log(other_scale**2 / self_scale**2)
            + (self_scale**2 + (self_loc - other_loc) ** 2) / other_scale**2
            - 1
        )


class Beta(Distribution):
    """Beta distribution.

    This class wraps distrax's Beta distribution and
    integrates it with Flax's nnx framework.
    """

    def __init__(
        self,
        concentration0: jax.Array | None = None,
        concentration1: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize a Beta distribution.

        Args:
            concentration0: First concentration parameter (alpha).
                If None, will be initialized as a parameter.
            concentration1: Second concentration parameter (beta).
                If None, will be initialized as a parameter.
            rngs: Random number generators for initialization and sampling.
        """
        super().__init__(rngs=rngs)

        # Initialize parameters
        self.concentration0 = nnx.Param(
            concentration0 if concentration0 is not None else jnp.ones(())
        )
        self.concentration1 = nnx.Param(
            concentration1 if concentration1 is not None else jnp.ones(())
        )

    def _distribution(self) -> distrax.Beta:
        """Construct the current Distrax Beta from NNX parameters."""
        return distrax.Beta(
            alpha=self._materialize_array(self.concentration0),
            beta=self._materialize_array(self.concentration1),
        )

    def kl_divergence(self, other: Distribution) -> jax.Array:
        """Compute KL divergence between this Beta and another.

        Args:
            other: Another distribution

        Returns:
            KL divergence
        """
        return super().kl_divergence(other)

    def mean(self) -> jax.Array:
        """Compute the mean of the distribution."""
        return jnp.asarray(self._distribution().mean())

    def variance(self) -> jax.Array:
        """Compute the variance of the distribution."""
        return jnp.asarray(self._distribution().variance())

    def mode(self) -> jax.Array:
        """Compute the mode of the distribution."""
        return jnp.asarray(self._distribution().mode())
