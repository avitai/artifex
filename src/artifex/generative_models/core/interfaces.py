"""Interfaces module.

This module provides common interfaces and abstract classes
that are used across the generative models package.
It helps avoid circular imports between modules.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class Distribution(nnx.Module):
    """Base class for all probability distributions.

    This class provides a common interface for all distributions and integrates
    with Flax's nnx framework for parameter management.
    """

    def __init__(self, *, rngs: nnx.Rngs | None = None):
        """Initialize the distribution.

        Args:
            rngs: Random number generators for stochastic operations.
        """
        super().__init__()
        self._dist = None
        self._rngs = rngs if rngs is not None else nnx.Rngs(0)

        # Caching infrastructure for expensive computations (JIT-compatible)
        # Use static flags to control caching behavior
        self._enable_caching = True
        self._entropy_cache = nnx.Cache(None)
        self._kl_cache = nnx.Cache({})  # Cache KL divergences with other distributions
        self._param_hash_cache = nnx.Cache(
            None
        )  # Hash of current parameters for cache invalidation

    def __call__(
        self, x: jax.Array | None = None, *, rngs: nnx.Rngs | None = None
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Compute log probability of x or sample if x is None.

        Args:
            x: Input tensor. If None, returns a sample.
            rngs: Random number generators. If None, uses the distribution's internal RNGs.

        Returns:
            If x is provided, returns log probability of x.
            If x is None, returns a sample from the distribution.
        """
        if x is None:
            return self.sample(sample_shape=(), rngs=rngs)
        return self.log_prob(x)

    def sample(self, sample_shape: tuple = (), *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Sample from the distribution.

        Args:
            sample_shape: Shape of the samples.
            rngs: Random number generators. If None, uses the distribution's internal RNGs.

        Returns:
            Samples from the distribution.

        Raises:
            ValueError: If the distribution is not initialized.
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized.")

        # Handle RNG key
        rngs = rngs if rngs is not None else self._rngs

        if rngs is None:
            raise ValueError("rngs must be provided for sampling")

        sample_key = rngs.sample()

        return self._dist.sample(seed=sample_key, sample_shape=sample_shape)

    def _get_param_hash(self) -> int:
        """Compute a hash of current parameters for cache invalidation.

        Returns:
            Hash of current parameters
        """
        if self._dist is None:
            return hash(None)

        # Get all parameter values and compute hash
        try:
            # For distrax distributions, get the parameters
            params = []
            if hasattr(self._dist, "concentration"):
                params.append(self._dist.concentration)
            if hasattr(self._dist, "rate"):
                params.append(self._dist.rate)
            if hasattr(self._dist, "loc"):
                params.append(self._dist.loc)
            if hasattr(self._dist, "scale"):
                params.append(self._dist.scale)
            if hasattr(self._dist, "logits"):
                params.append(self._dist.logits)
            if hasattr(self._dist, "probs"):
                params.append(self._dist.probs)
            if hasattr(self._dist, "concentration0"):
                params.append(self._dist.concentration0)
            if hasattr(self._dist, "concentration1"):
                params.append(self._dist.concentration1)

            # Convert to hashable representation
            param_values = []
            for param in params:
                if hasattr(param, "value"):
                    param_values.append(param.value.tobytes())
                else:
                    param_values.append(param.tobytes())

            return hash(tuple(param_values))
        except (AttributeError, TypeError):
            # Fallback: use object id (less efficient but safe)
            return id(self._dist)

    def _should_use_cache(self) -> bool:
        """Check if caching should be used (JIT-compatible)."""
        # Simple boolean check - JIT-friendly
        return self._enable_caching and self._dist is not None

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Compute log probability of x with numerical stability checks.

        Args:
            x: Input tensor.

        Returns:
            Log probability of x.

        Raises:
            ValueError: If the distribution is not initialized or parameters are invalid.
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized.")

        # Validate input
        self._check_finite(x, "input")

        # Validate parameters
        self._validate_parameters()

        # Compute log probability
        log_prob_value = self._dist.log_prob(x)

        # Check output for numerical issues
        self._check_finite(log_prob_value, "log probability")

        return log_prob_value

    def entropy(self, *, use_cache: bool = True) -> jax.Array:
        """Compute the entropy of the distribution with optional caching.

        Args:
            use_cache: Whether to use caching (set to False for JIT contexts)

        Returns:
            Entropy of the distribution.

        Raises:
            ValueError: If the distribution is not initialized.
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized.")

        # Simple functional approach - compute directly in JIT contexts
        if not use_cache or not self._should_use_cache():
            return self._dist.entropy()

        # Check if parameters changed and invalidate cache if needed
        current_param_hash = self._get_param_hash()
        if self._param_hash_cache.value != current_param_hash:
            # Parameters changed, invalidate cache
            self._entropy_cache.value = None
            self._kl_cache.value = {}
            self._param_hash_cache.value = current_param_hash

        # Non-JIT path with caching
        if self._entropy_cache.value is not None:
            return self._entropy_cache.value

        # Compute and cache
        entropy_value = self._dist.entropy()
        self._entropy_cache.value = entropy_value
        return entropy_value

    def kl_divergence(self, other: "Distribution", *, use_cache: bool = True) -> jax.Array:
        """Compute KL divergence between this distribution and another with optional caching.

        Args:
            other: Another distribution.
            use_cache: Whether to use caching (set to False for JIT contexts)

        Returns:
            KL divergence.

        Raises:
            ValueError: If either distribution is not initialized.
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized.")
        if other._dist is None:
            raise ValueError("Other distribution not initialized.")

        # Simple functional approach - compute directly in JIT contexts
        if not use_cache or not self._should_use_cache():
            return self._dist.kl_divergence(other._dist)

        # Check if parameters changed and invalidate cache if needed
        current_param_hash = self._get_param_hash()
        if self._param_hash_cache.value != current_param_hash:
            # Parameters changed, invalidate cache
            self._entropy_cache.value = None
            self._kl_cache.value = {}
            self._param_hash_cache.value = current_param_hash

        # Non-JIT path with caching
        other_hash = other._get_param_hash()
        cache_key = f"{type(other).__name__}_{other_hash}"

        # Return cached value if available
        if cache_key in self._kl_cache.value:
            return self._kl_cache.value[cache_key]

        # Compute and cache KL divergence
        kl_value = self._dist.kl_divergence(other._dist)

        # Update cache (create new dict to maintain immutability for JAX)
        new_cache = dict(self._kl_cache.value)
        new_cache[cache_key] = kl_value
        self._kl_cache.value = new_cache

        return kl_value

    @staticmethod
    def _safe_log(x: jax.Array, eps: float | None = None) -> jax.Array:
        """Compute log with numerical stability.

        Args:
            x: Input array
            eps: Small value to add for stability. If None, uses machine epsilon.

        Returns:
            Numerically stable log(x)
        """
        if eps is None:
            eps = jnp.finfo(x.dtype).eps
        return jnp.log(jnp.maximum(x, eps))

    @staticmethod
    def _safe_div(
        numerator: jax.Array, denominator: jax.Array, eps: float | None = None
    ) -> jax.Array:
        """Compute division with numerical stability.

        Args:
            numerator: Numerator array
            denominator: Denominator array
            eps: Small value to add to denominator for stability. If None, uses machine epsilon.

        Returns:
            Numerically stable numerator / denominator
        """
        if eps is None:
            eps = jnp.finfo(denominator.dtype).eps
        return numerator / jnp.maximum(denominator, eps)

    @staticmethod
    def _check_finite(x: jax.Array, name: str = "value") -> jax.Array:
        """Check if array contains finite values (JIT-compatible version).

        Args:
            x: Input array
            name: Name of the value for error reporting

        Returns:
            Input array (unchanged)

        Note:
            This is a JIT-compatible version that doesn't raise exceptions.
            For debugging, use the non-JIT version outside of compiled functions.
        """
        # JIT-compatible version: just return the input
        # In JIT context, we can't raise exceptions based on traced values
        return x

    def _check_finite_debug(self, x: jax.Array, name: str = "value") -> jax.Array:
        """Non-JIT version for debugging that can raise exceptions.

        Args:
            x: Input array
            name: Name of the value for error reporting

        Returns:
            Input array (unchanged)

        Raises:
            ValueError: If array contains non-finite values
        """
        if not jnp.all(jnp.isfinite(x)):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
        return x

    def _validate_parameters(self) -> None:
        """Validate distribution parameters for numerical stability."""
        if self._dist is None:
            return

        # Check common parameters for finite values
        try:
            if hasattr(self._dist, "loc"):
                self._check_finite(self._dist.loc, "location parameter")
            if hasattr(self._dist, "scale"):
                scale = self._dist.scale
                self._check_finite(scale, "scale parameter")
                # JIT-compatible: no conditional raising based on traced values
            if hasattr(self._dist, "concentration"):
                conc = self._dist.concentration
                self._check_finite(conc, "concentration parameter")
                # JIT-compatible: no conditional raising based on traced values
            if hasattr(self._dist, "concentration0"):
                conc0 = self._dist.concentration0
                self._check_finite(conc0, "concentration0 parameter")
                # JIT-compatible: no conditional raising based on traced values
            if hasattr(self._dist, "concentration1"):
                conc1 = self._dist.concentration1
                self._check_finite(conc1, "concentration1 parameter")
                # JIT-compatible: no conditional raising based on traced values
            if hasattr(self._dist, "probs"):
                probs = self._dist.probs
                self._check_finite(probs, "probability parameter")
                # JIT-compatible: no conditional raising based on traced values
        except AttributeError:
            # Parameter doesn't exist for this distribution type
            pass

    def _get_rng_key(self, rngs: nnx.Rngs | None = None, key_name: str = "sample") -> jax.Array:
        """Get RNG key with improved handling and fallbacks.

        Args:
            rngs: Random number generators
            key_name: Name of the key to retrieve

        Returns:
            RNG key for the specified operation
        """
        # Use provided RNGs or fall back to internal ones
        if rngs is None:
            rngs = self._rngs

        # Try to get the requested key
        if rngs is not None:
            if key_name in rngs:
                return rngs[key_name]()  # Use method call for JIT compatibility
            elif "default" in rngs:
                return rngs.default()
            elif hasattr(rngs, key_name):
                return getattr(rngs, key_name)()

        # Fallback to a deterministic key (for reproducibility in tests)
        return jax.random.key(0)

    def _split_rng_key(self, key: jax.Array, num_splits: int) -> list[jax.Array]:
        """Split RNG key for parallel operations.

        Args:
            key: Base RNG key
            num_splits: Number of keys to generate

        Returns:
            List of split RNG keys
        """
        if num_splits <= 1:
            return [key]
        return jax.random.split(key, num_splits)

    def _create_parallel_rngs(
        self, base_rngs: nnx.Rngs | None, num_parallel: int
    ) -> list[nnx.Rngs]:
        """Create multiple RNG objects for parallel operations.

        Args:
            base_rngs: Base RNG object
            num_parallel: Number of parallel RNG objects needed

        Returns:
            List of RNG objects for parallel operations
        """
        base_key = self._get_rng_key(base_rngs, "sample")
        split_keys = self._split_rng_key(base_key, num_parallel)

        return [nnx.Rngs(sample=key, default=key) for key in split_keys]

    @staticmethod
    def _efficient_stack(arrays: list[jax.Array], axis: int = 0) -> jax.Array:
        """Memory-efficient stacking of arrays.

        Args:
            arrays: List of arrays to stack
            axis: Axis along which to stack

        Returns:
            Stacked array with minimal memory overhead
        """
        if len(arrays) == 1:
            return jnp.expand_dims(arrays[0], axis=axis)

        # Use concatenate with pre-expanded dims for better memory efficiency
        expanded_arrays = [jnp.expand_dims(arr, axis=axis) for arr in arrays]
        return jnp.concatenate(expanded_arrays, axis=axis)

    @staticmethod
    def _memory_efficient_vmap(func, in_axes=0, out_axes=0, chunk_size: int | None = None):
        """Memory-efficient vectorized mapping with optional chunking.

        Args:
            func: Function to vectorize
            in_axes: Input axes for vmap
            out_axes: Output axes for vmap
            chunk_size: If provided, process in chunks to reduce memory usage

        Returns:
            Vectorized function with optional chunking
        """
        if chunk_size is None:
            return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)

        def chunked_vmap(*args, **kwargs):
            # Determine the size of the first dimension
            first_arg = args[0] if args else next(iter(kwargs.values()))
            if isinstance(in_axes, int):
                total_size = first_arg.shape[in_axes]
            else:
                # Handle tuple of axes
                axis = in_axes[0] if isinstance(in_axes, (tuple, list)) else in_axes
                total_size = first_arg.shape[axis]

            if total_size <= chunk_size:
                # No need to chunk
                return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)(*args, **kwargs)

            # Process in chunks
            results = []
            for i in range(0, total_size, chunk_size):
                end_idx = min(i + chunk_size, total_size)

                # Slice inputs for this chunk
                chunk_args = []
                for arg, axis in zip(
                    args, in_axes if isinstance(in_axes, (tuple, list)) else [in_axes] * len(args)
                ):
                    if axis is None:
                        chunk_args.append(arg)
                    else:
                        chunk_args.append(jax.lax.dynamic_slice_in_dim(arg, i, end_idx - i, axis))

                # Apply vmap to chunk
                chunk_result = jax.vmap(func, in_axes=in_axes, out_axes=out_axes)(*chunk_args)
                results.append(chunk_result)

            # Concatenate results
            if isinstance(out_axes, int):
                return jnp.concatenate(results, axis=out_axes)
            else:
                # Handle multiple outputs
                return tuple(
                    jnp.concatenate([r[i] for r in results], axis=axis)
                    for i, axis in enumerate(out_axes)
                )

        return chunked_vmap
