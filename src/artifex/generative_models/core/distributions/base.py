"""Shared concrete base class for probability distributions.

This module owns the reusable runtime behavior for Artifex distributions.
It is intentionally a concrete distribution foundation, not an abstract
interface layer.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from artifex.generative_models.core.rng import extract_rng_key


class Distribution(nnx.Module):
    """Base class for all probability distributions."""

    def __init__(self, *, rngs: nnx.Rngs | None = None):
        super().__init__()
        self._dist = None
        self._rngs = rngs

        self._enable_caching = True
        self._entropy_cache = nnx.Cache(None)
        self._kl_cache = nnx.Cache({})
        self._param_hash_cache = nnx.Cache(None)

    def __call__(
        self, x: jax.Array | None = None, *, rngs: nnx.Rngs | None = None
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        if x is None:
            return self.sample(sample_shape=(), rngs=rngs)
        return self.log_prob(x)

    def sample(self, sample_shape: tuple = (), *, rngs: nnx.Rngs | None = None) -> jax.Array:
        if self._dist is None:
            raise ValueError("Distribution not initialized.")

        sample_key = self._get_rng_key(rngs)
        return self._dist.sample(seed=sample_key, sample_shape=sample_shape)

    def _get_param_hash(self) -> int:
        if self._dist is None:
            return hash(None)

        try:
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

            param_values = []
            for param in params:
                raw_param = param.get_value() if hasattr(param, "get_value") else param
                param_values.append(jnp.asarray(raw_param).tobytes())

            return hash(tuple(param_values))
        except (AttributeError, TypeError):
            return id(self._dist)

    def _should_use_cache(self) -> bool:
        return self._enable_caching and self._dist is not None

    def log_prob(self, x: jax.Array) -> jax.Array:
        if self._dist is None:
            raise ValueError("Distribution not initialized.")

        self._check_finite(x, "input")
        self._validate_parameters()
        log_prob_value = self._dist.log_prob(x)
        self._check_finite(log_prob_value, "log probability")
        return log_prob_value

    def entropy(self, *, use_cache: bool = True) -> jax.Array:
        if self._dist is None:
            raise ValueError("Distribution not initialized.")

        if not use_cache or not self._should_use_cache():
            return self._dist.entropy()

        current_param_hash = self._get_param_hash()
        cached_param_hash = self._param_hash_cache.get_value()
        if cached_param_hash != current_param_hash:
            self._entropy_cache.set_value(None)
            self._kl_cache.set_value({})
            self._param_hash_cache.set_value(current_param_hash)

        cached_entropy = self._entropy_cache.get_value()
        if cached_entropy is not None:
            return cached_entropy

        entropy_value = self._dist.entropy()
        self._entropy_cache.set_value(entropy_value)
        return entropy_value

    def kl_divergence(self, other: "Distribution", *, use_cache: bool = True) -> jax.Array:
        if self._dist is None:
            raise ValueError("Distribution not initialized.")
        if other._dist is None:
            raise ValueError("Other distribution not initialized.")

        if not use_cache or not self._should_use_cache():
            return self._dist.kl_divergence(other._dist)

        current_param_hash = self._get_param_hash()
        cached_param_hash = self._param_hash_cache.get_value()
        if cached_param_hash != current_param_hash:
            self._entropy_cache.set_value(None)
            self._kl_cache.set_value({})
            self._param_hash_cache.set_value(current_param_hash)

        other_hash = other._get_param_hash()
        cache_key = f"{type(other).__name__}_{other_hash}"
        kl_cache = self._kl_cache.get_value()
        if cache_key in kl_cache:
            return kl_cache[cache_key]

        kl_value = self._dist.kl_divergence(other._dist)
        new_cache = dict(kl_cache)
        new_cache[cache_key] = kl_value
        self._kl_cache.set_value(new_cache)
        return kl_value

    @staticmethod
    def _safe_log(x: jax.Array, eps: float | None = None) -> jax.Array:
        if eps is None:
            eps = jnp.finfo(x.dtype).eps
        return jnp.log(jnp.maximum(x, eps))

    @staticmethod
    def _safe_div(
        numerator: jax.Array, denominator: jax.Array, eps: float | None = None
    ) -> jax.Array:
        if eps is None:
            eps = jnp.finfo(denominator.dtype).eps
        return numerator / jnp.maximum(denominator, eps)

    @staticmethod
    def _materialize_leaf(value):
        if hasattr(value, "get_value"):
            return value.get_value()
        return value

    @staticmethod
    def _is_concrete_value(value) -> bool:
        if isinstance(value, jax.core.Tracer):
            return False
        try:
            return bool(jax.core.is_concrete(value))
        except TypeError:
            return True

    @classmethod
    def _check_finite(cls, x: jax.Array, name: str = "value") -> jax.Array:
        leaves = jax.tree_util.tree_leaves(x)
        for leaf in leaves:
            materialized = cls._materialize_leaf(leaf)
            if not cls._is_concrete_value(materialized):
                continue
            array = jnp.asarray(materialized)
            if not cls._is_concrete_value(array):
                continue
            concrete_array = np.asarray(jax.device_get(array))
            if not np.isfinite(concrete_array).all():
                raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
        return x

    @classmethod
    def _check_finite_debug(cls, x: jax.Array, name: str = "value") -> jax.Array:
        return cls._check_finite(x, name)

    def _validate_parameters(self) -> None:
        if self._dist is None:
            return

        try:
            if hasattr(self._dist, "loc"):
                self._check_finite(self._dist.loc, "location parameter")
            if hasattr(self._dist, "scale"):
                self._check_finite(self._dist.scale, "scale parameter")
            if hasattr(self._dist, "concentration"):
                self._check_finite(self._dist.concentration, "concentration parameter")
            if hasattr(self._dist, "concentration0"):
                self._check_finite(self._dist.concentration0, "concentration0 parameter")
            if hasattr(self._dist, "concentration1"):
                self._check_finite(self._dist.concentration1, "concentration1 parameter")
            if hasattr(self._dist, "probs"):
                self._check_finite(self._dist.probs, "probability parameter")
        except AttributeError:
            pass

    def _get_rng_key(self, rngs: nnx.Rngs | None = None, key_name: str = "sample") -> jax.Array:
        source_rngs = rngs if rngs is not None else self._rngs
        if source_rngs is None:
            raise ValueError("rngs must be provided for sampling")
        return extract_rng_key(
            source_rngs,
            streams=(key_name, "default"),
            context="distribution sampling",
        )

    def _split_rng_key(self, key: jax.Array, num_splits: int) -> list[jax.Array]:
        if num_splits <= 1:
            return [key]
        return jax.random.split(key, num_splits)

    def _create_parallel_rngs(
        self, base_rngs: nnx.Rngs | None, num_parallel: int
    ) -> list[nnx.Rngs]:
        base_key = self._get_rng_key(base_rngs, "sample")
        split_keys = self._split_rng_key(base_key, num_parallel)
        return [nnx.Rngs(sample=key, default=key) for key in split_keys]

    @staticmethod
    def _efficient_stack(arrays: list[jax.Array], axis: int = 0) -> jax.Array:
        if len(arrays) == 1:
            return jnp.expand_dims(arrays[0], axis=axis)

        expanded_arrays = [jnp.expand_dims(arr, axis=axis) for arr in arrays]
        return jnp.concatenate(expanded_arrays, axis=axis)

    @staticmethod
    def _memory_efficient_vmap(func, in_axes=0, out_axes=0, chunk_size: int | None = None):
        if chunk_size is None:
            return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)

        def chunked_vmap(*args, **kwargs):
            first_arg = args[0] if args else next(iter(kwargs.values()))
            if isinstance(in_axes, int):
                total_size = first_arg.shape[in_axes]
            else:
                axis = in_axes[0] if isinstance(in_axes, (tuple, list)) else in_axes
                total_size = first_arg.shape[axis]

            if total_size <= chunk_size:
                return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)(*args, **kwargs)

            results = []
            for i in range(0, total_size, chunk_size):
                end_idx = min(i + chunk_size, total_size)
                chunk_args = []
                for arg, axis in zip(
                    args, in_axes if isinstance(in_axes, (tuple, list)) else [in_axes] * len(args)
                ):
                    if axis is None:
                        chunk_args.append(arg)
                    else:
                        chunk_args.append(jax.lax.dynamic_slice_in_dim(arg, i, end_idx - i, axis))

                chunk_result = jax.vmap(func, in_axes=in_axes, out_axes=out_axes)(*chunk_args)
                results.append(chunk_result)

            if isinstance(out_axes, int):
                return jnp.concatenate(results, axis=out_axes)

            return tuple(
                jnp.concatenate([result[i] for result in results], axis=axis)
                for i, axis in enumerate(out_axes)
            )

        return chunked_vmap


__all__ = ["Distribution"]
