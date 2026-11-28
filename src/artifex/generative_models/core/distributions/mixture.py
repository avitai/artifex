"""Mixture distributions.

This module implements mixture distributions that combine multiple distributions
with discrete probability weights.
"""

import distrax
import jax
import jax.numpy as jnp
from flax import nnx
from substrax.rng import SAMPLE_STREAM, stream_key

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
        rng_collection: str = SAMPLE_STREAM,
    ):
        """Initialize a Mixture of Gaussians distribution.

        Args:
            locs: Means of the component distributions.
                Shape [..., num_components, event_size]
            scales: Standard deviations of the component distributions.
                Shape [..., num_components, event_size]
            weights: Mixture weights for the components. If None, uniform weights
                are used. Shape [..., num_components]
            rng_collection: The ``nnx.Rngs`` stream ``sample`` draws its key from.
        """
        super().__init__(rng_collection=rng_collection)

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
        rng_collection: str = SAMPLE_STREAM,
    ):
        """Initialize a mixture distribution.

        Args:
            components: list of component distributions
            weights: Mixture weights for the components. If None, uniform weights
                are used. Shape [num_components]
            rng_collection: The ``nnx.Rngs`` stream ``sample`` draws its key from.
        """
        super().__init__(rng_collection=rng_collection)
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

    def sample(self, sample_shape: tuple = (), *, rngs: nnx.Rngs) -> jax.Array:
        """Sample by choosing a component per draw and taking that component's sample.

        One key of this mixture's stream is split into a key for the choice and one per
        component, so no key is used twice; every component is drawn and the chosen draws are
        gathered. This is numpyro's ``MixtureGeneral`` (``sample_with_intermediates`` and
        ``component_sample``). Each component draws from an ``nnx.Rngs`` holding its key as
        ``default``, which serves whatever stream the component names.

        The choice is a discrete draw, so a sample has no gradient in ``weights`` (numpyro marks
        the same with ``has_rsample = False``); gradients reach the components' parameters.

        Args:
            sample_shape: Leading shape of the samples.
            rngs: Holds the ``rng_collection`` stream, or a ``default`` one.

        Returns:
            Samples of shape ``sample_shape`` plus the components' event shape.
        """
        if not isinstance(sample_shape, tuple):
            sample_shape = (sample_shape,)

        # The first half of the split is the choice's key, as before this method took its
        # key at call time, so a given key still gives the same draws.
        key_choice, key_components = jax.random.split(stream_key(rngs, self.rng_collection))
        component_keys = jax.random.split(key_components, self.num_components)
        choice = self._cat_dist.sample(seed=key_choice, sample_shape=sample_shape)

        # The components may be of different classes, so they are drawn one by one and stacked
        # on a new axis after the sample axes: (*sample_shape, num_components, *event_shape).
        mixture_axis = len(sample_shape)
        samples = jnp.stack(
            [
                component.sample(sample_shape=sample_shape, rngs=nnx.Rngs(component_key))
                for component_key, component in zip(component_keys, self.components, strict=True)
            ],
            axis=mixture_axis,
        )
        event_ndim = samples.ndim - mixture_axis - 1
        indices = choice.reshape(choice.shape + (1,) * (event_ndim + 1))
        selected = jnp.take_along_axis(samples, indices, axis=mixture_axis)
        return jnp.squeeze(selected, axis=mixture_axis)

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
