import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from .base import Distribution


class Bernoulli(Distribution):
    """Bernoulli distribution.

    This class wraps distrax's Bernoulli distribution and
    integrates it with Flax's nnx framework.
    """

    def __init__(
        self,
        probs: jax.Array | None = None,
        logits: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize a Bernoulli distribution.

        Args:
            probs: Probability of success.
                If None, will be initialized as a parameter.
            logits: Logits of success probability.
                If None, will be initialized as a parameter.
            rngs: Random number generators for initialization and sampling.
        """
        super().__init__(rngs=rngs)

        if probs is not None and logits is not None:
            raise ValueError("Cannot specify both probs and logits.")

        # Initialize parameters
        if probs is None and logits is None:
            # Initialize with logits = 0 (probs = 0.5)
            self.logits = nnx.Param(jnp.zeros(()))
            self._dist = nnx.data(distrax.Bernoulli(logits=self.logits))
        elif probs is not None:
            self.probs = probs
            self._dist = nnx.data(distrax.Bernoulli(probs=self.probs))
        else:
            self.logits = nnx.Param(logits) if not isinstance(logits, nnx.Param) else logits
            self._dist = nnx.data(distrax.Bernoulli(logits=self.logits))

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
        """Sample from the Bernoulli distribution.

        Args:
            sample_shape: Shape of the samples.
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            Samples from the distribution.

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

    def kl_divergence(self, other: "Bernoulli") -> jax.Array:
        """Compute KL divergence between this Bernoulli and another.

        Args:
            other: Another Bernoulli distribution

        Returns:
            KL divergence
        """
        return super().kl_divergence(other)


class Categorical(Distribution):
    """Categorical distribution.

    This class wraps distrax's Categorical distribution and
    integrates it with Flax's nnx framework.
    """

    def __init__(
        self,
        probs: jax.Array | None = None,
        logits: jax.Array | None = None,
        num_classes: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize a Categorical distribution.

        Args:
            probs: Probability of each category.
                If None, will be initialized as a parameter.
            logits: Logits of category probabilities.
                If None, will be initialized as a parameter.
            num_classes: Number of categories.
                Only used when both probs and logits are None.
            rngs: Random number generators for initialization and sampling.
        """
        super().__init__(rngs=rngs)
        self.num_classes = num_classes

        if probs is not None and logits is not None:
            raise ValueError("Cannot specify both probs and logits.")

        if probs is None and logits is None and num_classes is None:
            raise ValueError("Must specify either probs, logits, or num_classes.")

        if num_classes is not None and (probs is not None or logits is not None):
            if probs is not None and probs.shape[-1] != num_classes:
                raise ValueError("probs.shape[-1] must match num_classes.")
            if logits is not None and logits.shape[-1] != num_classes:
                raise ValueError("logits.shape[-1] must match num_classes.")

        # Initialize parameters
        if probs is None and logits is None:
            # Initialize with uniform logits (equal probabilities)
            self.logits = nnx.Param(jnp.zeros((self.num_classes,)))
            self._dist = nnx.data(distrax.Categorical(logits=self.logits))
        elif probs is not None:
            self.probs = nnx.Param(probs) if not isinstance(probs, nnx.Param) else probs
            self._dist = nnx.data(distrax.Categorical(probs=self.probs))
        else:
            self.logits = nnx.Param(logits) if not isinstance(logits, nnx.Param) else logits
            self._dist = nnx.data(distrax.Categorical(logits=self.logits))

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
        """Sample from the categorical distribution.

        Args:
            sample_shape: Shape of the samples.
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            Samples from the distribution.

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

    def kl_divergence(self, other: "Categorical") -> jax.Array:
        """Compute KL divergence between this Categorical and another.

        Args:
            other: Another Categorical distribution

        Returns:
            KL divergence
        """
        return super().kl_divergence(other)

    def mode(self) -> jax.Array:
        """Compute the mode of the distribution.

        Returns:
            Mode of the distribution
        """
        # Get the index of highest probability
        if hasattr(self, "probs"):
            return jnp.argmax(self.probs)
        elif hasattr(self, "logits"):
            return jnp.argmax(self.logits)
        else:
            raise ValueError("Cannot compute mode: no probs or logits available")


class OneHotCategorical(Distribution):
    """OneHot Categorical distribution.

    This class wraps a Categorical distribution and
    returns samples as one-hot vectors.
    It integrates with Flax's nnx framework.
    """

    def __init__(
        self,
        probs: jax.Array | None = None,
        logits: jax.Array | None = None,
        num_classes: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize a OneHotCategorical distribution.

        Args:
            probs: Probability of each category.
                If None, will be initialized as a parameter.
            logits: Logits of category probabilities.
                If None, will be initialized as a parameter.
            num_classes: Number of categories.
                Only used when both probs and logits are None.
            rngs: Random number generators for initialization and sampling.
        """
        super().__init__(rngs=rngs)
        self.num_classes = num_classes

        # Create underlying categorical distribution
        self.categorical = Categorical(
            probs=probs, logits=logits, num_classes=num_classes, rngs=rngs
        )

        # Get the number of classes from the categorical distribution
        if hasattr(self.categorical, "num_classes") and self.categorical.num_classes is not None:
            self.num_classes = self.categorical.num_classes
        elif probs is not None:
            self.num_classes = probs.shape[-1]
        elif logits is not None:
            self.num_classes = logits.shape[-1]

        # Ensure num_classes is not None
        if self.num_classes is None:
            raise ValueError("Failed to determine number of classes")

        # Note: We don't set self._dist here because OneHotCategorical delegates
        # all operations to self.categorical, which has its own _dist already wrapped

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
            If x is None, returns a one-hot encoded sample.
        """
        if x is None:
            return self.sample(sample_shape=(), rngs=rngs)
        return self.log_prob(x)

    def sample(self, sample_shape: tuple = (), *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Sample from the OneHotCategorical distribution.

        Args:
            sample_shape: Shape of the samples.
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            One-hot encoded samples from the distribution.

        Raises:
            ValueError: If the distribution is not initialized.
        """
        # Get categorical indices
        indices = self.categorical.sample(sample_shape=sample_shape, rngs=rngs)

        # Create a one-hot representation
        num_classes = self.num_classes if self.num_classes is not None else indices.max() + 1
        return jax.nn.one_hot(indices, num_classes)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Compute log probability of x.

        Args:
            x: Input tensor as one-hot vectors

        Returns:
            Log probability of x
        """
        # Convert one-hot to indices
        num_classes = self.num_classes if self.num_classes is not None else x.shape[-1]
        if x.ndim > 1 and x.shape[-1] == num_classes:
            indices = jnp.argmax(x, axis=-1)
            return self.categorical.log_prob(indices)
        return self.categorical.log_prob(x)

    def entropy(self) -> jax.Array:
        """Compute the entropy of the distribution.

        Returns:
            Entropy of the distribution
        """
        return self.categorical.entropy()

    def kl_divergence(self, other: Distribution) -> jax.Array:
        """Compute KL divergence between this OneHotCategorical and another.

        Args:
            other: Another distribution

        Returns:
            KL divergence
        """
        if isinstance(other, OneHotCategorical):
            return self.categorical.kl_divergence(other.categorical)
        return super().kl_divergence(other)

    def mode(self) -> jax.Array:
        """Compute the mode of the distribution as a one-hot vector.

        Returns:
            Mode of the distribution as a one-hot vector
        """
        # Get the categorical mode (index of highest probability)
        if hasattr(self.categorical, "probs"):
            mode_index = jnp.argmax(self.categorical.probs, axis=-1)
        elif hasattr(self.categorical, "logits"):
            mode_index = jnp.argmax(self.categorical.logits, axis=-1)
        else:
            raise ValueError("Cannot compute mode: no probs or logits available")

        num_classes = self.num_classes if self.num_classes is not None else mode_index.max() + 1
        return jax.nn.one_hot(mode_index, num_classes)
