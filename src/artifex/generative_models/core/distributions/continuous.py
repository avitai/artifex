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

        # Create the underlying distribution and wrap with nnx.data for NNX compatibility
        self._dist = nnx.data(distrax.Normal(loc=self.loc, scale=self.scale))

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
        """Sample from the normal distribution.

        Args:
            sample_shape: Shape of the samples
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            Samples from the distribution

        Raises:
            ValueError: If the distribution is not initialized.
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
        return 0.5 * (
            jnp.log(other.scale**2 / self.scale**2)
            + (self.scale**2 + (self.loc - other.loc) ** 2) / other.scale**2
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
        self.concentration0 = (
            concentration0 if concentration0 is not None else nnx.Param(jnp.ones(()))
        )
        self.concentration1 = (
            concentration1 if concentration1 is not None else nnx.Param(jnp.ones(()))
        )

        # Create the underlying distribution and wrap with nnx.data for NNX compatibility
        self._dist = nnx.data(distrax.Beta(alpha=self.concentration0, beta=self.concentration1))

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
        """Sample from the beta distribution.

        Args:
            sample_shape: Shape of the samples
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            Samples from the distribution

        Raises:
            ValueError: If the distribution is not initialized.
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
        if self._dist is None:
            raise ValueError("Distribution not initialized.")
        return self._dist.mean()

    def variance(self) -> jax.Array:
        """Compute the variance of the distribution."""
        if self._dist is None:
            raise ValueError("Distribution not initialized.")
        return self._dist.variance()

    def mode(self) -> jax.Array:
        """Compute the mode of the distribution."""
        if self._dist is None:
            raise ValueError("Distribution not initialized.")
        return self._dist.mode()
