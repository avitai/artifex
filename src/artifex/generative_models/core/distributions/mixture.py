"""Mixture distributions.

This module implements mixture distributions that combine multiple distributions
with discrete probability weights.
"""

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from .base import Distribution


class MixtureOfGaussians(Distribution):
    """Mixture of Gaussian distributions.

    This class wraps distrax's MixtureSameFamily distribution with Normal
    components and integrates it with Flax's nnx framework.
    """

    def __init__(
        self,
        locs: jax.Array,
        scales: jax.Array,
        weights: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize a Mixture of Gaussians distribution.

        Args:
            locs: Means of the component distributions.
                Shape [..., num_components, event_size]
            scales: Standard deviations of the component distributions.
                Shape [..., num_components, event_size]
            weights: Mixture weights for the components. If None, uniform weights
                are used. Shape [..., num_components]
            rngs: Random number generators for initialization and sampling.
        """
        super().__init__(rngs=rngs)

        # Check and set parameters
        self.locs = locs
        self.scales = scales

        # Infer number of components from locs
        self.num_components = locs.shape[-2]

        # Set weights (uniform if not provided)
        if weights is None:
            self.weights = jnp.ones(locs.shape[:-1]) / self.num_components
        else:
            # Normalize weights to sum to 1
            self.weights = weights / jnp.sum(weights, axis=-1, keepdims=True)

        # Create component distributions
        components_dist = distrax.Independent(
            distrax.Normal(
                loc=self.locs,
                scale=self.scales,
            ),
            reinterpreted_batch_ndims=1,  # Event dimension
        )

        # Create the mixture distribution and wrap with nnx.data for NNX compatibility
        self._dist = nnx.data(
            distrax.MixtureSameFamily(
                mixture_distribution=distrax.Categorical(probs=self.weights),
                components_distribution=components_dist,
            )
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
        """Sample from the mixture of gaussians.

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


class Mixture(Distribution):
    """General mixture distribution combining arbitrary component distributions.

    This class allows combining different types of distributions with discrete
    probability weights.
    """

    def __init__(
        self,
        components: list[Distribution],
        weights: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize a mixture distribution.

        Args:
            components: list of component distributions
            weights: Mixture weights for the components. If None, uniform weights
                are used. Shape [num_components]
            rngs: Random number generators for initialization and sampling.
        """
        super().__init__(rngs=rngs)
        # Wrap components list with nnx.List for NNX compatibility
        self.components = nnx.List(components)
        self.num_components = len(components)

        # Set weights (uniform if not provided)
        if weights is None:
            self.weights = jnp.ones(self.num_components) / self.num_components
        else:
            # Normalize weights to sum to 1
            self.weights = weights / jnp.sum(weights)

        # Create the categorical distribution for component selection and wrap with nnx.data
        self._cat_dist = nnx.data(distrax.Categorical(probs=self.weights))

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

    def _get_sample_key(self, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Extract a JAX random key from the input rngs.

        Args:
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            A JAX random key for sampling.
        """
        # Handle RNG case - JIT-compatible approach
        if rngs is None:
            rngs = self._rngs

        if rngs is None:
            raise ValueError("rngs must be provided for sampling")

        sample_key = rngs.sample()

        return sample_key

    def sample(self, sample_shape: tuple = (), *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Sample from the mixture distribution using vectorized operations.

        Args:
            sample_shape: Shape of the samples
            rngs: Random number generators for sampling.
                If None, uses the distribution's internal RNGs.

        Returns:
            Samples from the distribution
        """
        # Convert sample_shape to a tuple if it's not already
        if not isinstance(sample_shape, tuple):
            sample_shape = (sample_shape,)

        # Get the sample key
        sample_key = self._get_sample_key(rngs)

        # Split keys for component selection and sampling
        key_select, key_sample = jax.random.split(sample_key)

        # Split keys for each component to ensure independent sampling
        num_components = len(self.components)
        component_keys = jax.random.split(key_sample, num_components)

        # Select components
        component_indices = self._cat_dist.sample(seed=key_select, sample_shape=sample_shape)

        # Sample from all components (can't use vmap with heterogeneous objects)
        all_samples = []
        for i, (component_key, component) in enumerate(zip(component_keys, self.components)):
            component_rngs = nnx.Rngs(sample=component_key)
            sample = component.sample(sample_shape=sample_shape, rngs=component_rngs)
            all_samples.append(sample)

        # Stack samples efficiently
        all_samples = self._efficient_stack(all_samples, axis=0)

        # Select samples based on component indices using advanced indexing
        # all_samples shape: (num_components, *sample_shape, *event_shape)
        # component_indices shape: (*sample_shape,)

        # Create indices for advanced indexing
        batch_indices = jnp.arange(jnp.prod(jnp.array(sample_shape))).reshape(sample_shape)
        flat_component_indices = component_indices.flatten().astype(jnp.int32)
        flat_batch_indices = batch_indices.flatten().astype(jnp.int32)

        # Reshape all_samples for indexing: (num_components, batch_size, *event_shape)
        original_shape = all_samples.shape
        batch_size = int(jnp.prod(jnp.array(sample_shape)))
        event_shape = original_shape[2:]  # Skip (num_components, *sample_shape)

        reshaped_samples = all_samples.reshape((num_components, batch_size, *event_shape))

        # Select samples using advanced indexing
        selected_samples = reshaped_samples[flat_component_indices, flat_batch_indices]

        # Reshape back to original sample shape
        final_shape = sample_shape + event_shape
        return selected_samples.reshape(final_shape)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Compute log probability of x using vectorized operations.

        Args:
            x: Input tensor

        Returns:
            Log probability of x
        """

        # Compute log probabilities for all components
        component_log_probs = []
        for component in self.components:
            log_prob = component.log_prob(x)
            component_log_probs.append(log_prob)

        # Stack log probs efficiently
        stacked_log_probs = self._efficient_stack(component_log_probs, axis=-1)

        # Add log weights and use logsumexp for numerical stability
        log_weights = jnp.log(self.weights)
        weighted_log_probs = stacked_log_probs + log_weights
        return jax.scipy.special.logsumexp(weighted_log_probs, axis=-1)

    def entropy(self) -> jax.Array:
        """Compute the entropy of the distribution using vectorized operations.

        Note: For mixture distributions, exact entropy calculation is often
        intractable. This implementation returns an approximation.

        Returns:
            Approximate entropy of the distribution
        """

        # Compute component entropies
        component_entropies = []
        for component in self.components:
            try:
                entropy = component.entropy()
                component_entropies.append(entropy)
            except (NotImplementedError, ValueError):
                # If component doesn't support entropy, return zero
                component_entropies.append(jnp.array(0.0))

        # Stack entropies
        component_entropies = jnp.stack(component_entropies)

        # Check if any component returned zero (unsupported entropy)
        if jnp.any(component_entropies == 0.0):
            return jnp.array(0.0)  # Return dummy value if any component fails

        # Weight the component entropies
        weighted_entropies = jnp.sum(self.weights * component_entropies)

        # Add entropy of the categorical distribution over components
        # Use numerical stability: avoid log(0) by adding small epsilon
        eps = jnp.finfo(self.weights.dtype).eps
        safe_weights = jnp.maximum(self.weights, eps)
        mixture_entropy = -jnp.sum(self.weights * jnp.log(safe_weights))

        return weighted_entropies + mixture_entropy
