"""CliffordConv1d/2d/3d — Clifford algebra convolution layers.

Ported from ``microsoft/cliffordlayers`` (MIT license).
Uses ``lax.conv_general_dilated`` under the hood via ``_functional.clifford_convnd``.
"""

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.layers._utils import normalize_size_param
from artifex.generative_models.core.layers.clifford._functional import clifford_convnd
from artifex.generative_models.core.layers.clifford.kernels import (
    get_1d_clifford_kernel,
    get_2d_clifford_kernel,
    get_2d_clifford_rotation_kernel,
    get_3d_clifford_kernel,
)


class _CliffordConvNd(nnx.Module):
    """Base class for all Clifford convolution modules.

    Subclasses set ``spatial_ndim`` and delegate to this base.

    Input shape:  ``(B, *D, C_in, n_blades)``
    Output shape: ``(B, *D', C_out, n_blades)``
    """

    def __init__(
        self,
        metric: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int],
        padding: str | int | Sequence[int],
        dilation: int | Sequence[int],
        groups: int,
        use_bias: bool,
        rotation: bool,
        spatial_ndim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.metric = jnp.asarray(metric, dtype=jnp.float32)
        self.dim = len(metric)
        self.n_blades = 2**self.dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.rotation = rotation
        self.spatial_ndim = spatial_ndim

        # Normalize spatial params
        self.kernel_size = normalize_size_param(kernel_size, spatial_ndim, "kernel_size")
        self.strides = normalize_size_param(stride, spatial_ndim, "stride")
        self.dilation = normalize_size_param(dilation, spatial_ndim, "dilation")

        # Padding
        if isinstance(padding, str):
            self.padding = padding.upper()
        else:
            if isinstance(padding, int):
                p = (padding,) * spatial_ndim
            else:
                p = tuple(padding)
            self.padding = [(pi, pi) for pi in p]

        # Rotation checks
        if rotation and self.dim != 2:
            raise ValueError("Rotation kernel only available for 2D Clifford algebras.")

        # Kernel dispatch
        if self.dim == 1:
            self._get_kernel = get_1d_clifford_kernel
        elif self.dim == 2 and rotation:
            self._get_kernel = get_2d_clifford_rotation_kernel
        elif self.dim == 2:
            self._get_kernel = get_2d_clifford_kernel
        elif self.dim == 3:
            self._get_kernel = get_3d_clifford_kernel
        else:
            raise ValueError(f"CliffordConv supports dim 1-3, got {self.dim}.")

        # Weight initialization
        fan_in = in_channels * self.n_blades // groups
        for k in self.kernel_size:
            fan_in *= k
        bound = 1.0 / math.sqrt(fan_in)
        key = rngs.params()

        w_shape = (self.n_blades, out_channels, in_channels // groups, *self.kernel_size)
        self.weight = nnx.Param(jax.random.uniform(key, w_shape, minval=-bound, maxval=bound))

        # Bias
        if use_bias:
            key_b = rngs.params()
            self.bias = nnx.Param(
                jax.random.uniform(
                    key_b, (self.n_blades, out_channels), minval=-bound, maxval=bound
                )
            )
        else:
            self.bias = None

        # Rotation extras
        if rotation:
            key_s = rngs.params()
            single_shape = (out_channels, in_channels // groups, *self.kernel_size)
            self.scale_param = nnx.Param(
                jax.random.uniform(key_s, single_shape, minval=-bound, maxval=bound)
            )
            self.zero_kernel = jnp.zeros(single_shape)

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Forward pass.

        Args:
            x: Input ``(B, *D, C_in, n_blades)``.
            deterministic: Unused, kept for API consistency.

        Returns:
            Output ``(B, *D', C_out, n_blades)``.
        """
        n_blades = x.shape[-1]
        if n_blades != self.n_blades:
            raise ValueError(f"Input has {n_blades} blades, expected {self.n_blades}.")

        # Build weight input for kernel function
        if self.rotation:
            w = jnp.concatenate(
                [
                    self.weight.value,
                    self.scale_param.value[None],
                    self.zero_kernel[None],
                ],
                axis=0,
            )
        else:
            w = self.weight.value

        _, kernel = self._get_kernel(w, self.metric)

        bias_flat = self.bias.value.reshape(-1) if self.bias is not None else None

        return clifford_convnd(
            x,
            self.n_blades,
            kernel,
            bias_flat,
            strides=self.strides,
            padding=self.padding,
            dilation=self.dilation,
        )


class CliffordConv1d(_CliffordConvNd):
    """1D Clifford convolution for Cl(1) algebras.

    Input:  ``(B, L, C_in, 2)``
    Output: ``(B, L', C_out, 2)``
    """

    def __init__(
        self,
        metric: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str | int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize CliffordConv1d."""
        super().__init__(
            metric,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            rotation=False,
            spatial_ndim=1,
            rngs=rngs,
        )
        if self.dim != 1:
            raise ValueError(f"CliffordConv1d requires 1D metric, got dim={self.dim}.")


class CliffordConv2d(_CliffordConvNd):
    """2D Clifford convolution for Cl(2) algebras.

    Input:  ``(B, H, W, C_in, 4)``
    Output: ``(B, H', W', C_out, 4)``
    """

    def __init__(
        self,
        metric: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: str | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        rotation: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize CliffordConv2d."""
        super().__init__(
            metric,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            rotation=rotation,
            spatial_ndim=2,
            rngs=rngs,
        )
        if self.dim != 2:
            raise ValueError(f"CliffordConv2d requires 2D metric, got dim={self.dim}.")


class CliffordConv3d(_CliffordConvNd):
    """3D Clifford convolution for Cl(3) algebras.

    Input:  ``(B, D, H, W, C_in, 8)``
    Output: ``(B, D', H', W', C_out, 8)``
    """

    def __init__(
        self,
        metric: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: str | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize CliffordConv3d."""
        super().__init__(
            metric,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            rotation=False,
            spatial_ndim=3,
            rngs=rngs,
        )
        if self.dim != 3:
            raise ValueError(f"CliffordConv3d requires 3D metric, got dim={self.dim}.")
