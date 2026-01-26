"""Shared utility functions for neural network layers.

Provides reusable normalization and parameter processing utilities
used across ResNet, Clifford, and other layer implementations.
"""

from collections.abc import Sequence

import jax
from flax import nnx


def normalize_size_param(
    size_param: int | Sequence[int],
    spatial_ndim: int,
    param_name: str,
) -> tuple[int, ...]:
    """Normalize kernel_size/stride/dilation to an N-dimensional tuple.

    Args:
        size_param: Integer or sequence of integers.
        spatial_ndim: Number of spatial dimensions.
        param_name: Name of parameter for error messages.

    Returns:
        Tuple of integers with length ``spatial_ndim``.

    Raises:
        ValueError: If size_param has the wrong type or length.
    """
    if isinstance(size_param, int):
        return (size_param,) * spatial_ndim
    if isinstance(size_param, Sequence) and len(size_param) == spatial_ndim:
        return tuple(int(s) for s in size_param)
    raise ValueError(f"{param_name} must be an int or a sequence of {spatial_ndim} ints.")


def create_norm_layer(
    norm_type: str,
    num_features: int,
    *,
    group_norm_num_groups: int = 32,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create a single normalization layer.

    Args:
        norm_type: Type of normalization (``'batch'``, ``'layer'``, or ``'group'``).
        num_features: Number of features for the normalization layer.
        group_norm_num_groups: Number of groups for GroupNorm.
        rngs: Random number generators.

    Returns:
        Normalization layer instance.

    Raises:
        ValueError: If norm_type is unknown or features are incompatible with group norm.
    """
    if norm_type == "batch":
        return nnx.BatchNorm(num_features=num_features, use_running_average=True, rngs=rngs)
    if norm_type == "layer":
        return nnx.LayerNorm(num_features=num_features, rngs=rngs)
    if norm_type == "group":
        if num_features % group_norm_num_groups != 0:
            raise ValueError(
                f"Features ({num_features}) must be divisible by "
                f"group_norm_num_groups ({group_norm_num_groups})"
            )
        return nnx.GroupNorm(num_groups=group_norm_num_groups, num_features=num_features, rngs=rngs)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def apply_norm(
    x: jax.Array,
    norm_layer: nnx.Module | None,
    norm_type: str,
    *,
    deterministic: bool,
) -> jax.Array:
    """Apply a normalization layer with proper dispatch for batch vs. layer/group norm.

    Args:
        x: Input tensor.
        norm_layer: Normalization layer instance, or ``None`` to skip.
        norm_type: Type of normalization (``'batch'``, ``'layer'``, or ``'group'``).
        deterministic: If ``True``, use running averages for batch norm.

    Returns:
        Normalized tensor, or input unchanged if ``norm_layer`` is ``None``.
    """
    if norm_layer is None:
        return x
    if norm_type == "batch":
        return norm_layer(x, use_running_average=deterministic)
    return norm_layer(x)
