"""CliffordSpectralConv2d/3d — Fourier-space Clifford convolutions.

Ported from ``microsoft/cliffordlayers`` (MIT license).
Uses ``jnp.fft`` for the forward/inverse transforms and dual-complex
pairing for the Clifford Fourier transform.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.layers.clifford._functional import batchmul2d, batchmul3d
from artifex.generative_models.core.layers.clifford.kernels import (
    get_2d_clifford_kernel,
    get_3d_clifford_kernel,
)


class CliffordSpectralConv2d(nnx.Module):
    """2D Clifford spectral convolution for Cl(2) algebras.

    Performs Clifford Fourier transform via dual complex pairs,
    applies geometric product in Fourier space, and inverse-transforms back.

    Input:  ``(B, H, W, C_in, 4)``
    Output: ``(B, H, W, C_out, 4)``

    Args:
        metric: 2D metric signature, e.g. ``(1, 1)``.
        in_channels: Input channel count.
        out_channels: Output channel count.
        modes1: Number of retained Fourier modes in dim 1.
        modes2: Number of retained Fourier modes in dim 2.
        multiply: If ``False``, skip kernel multiplication (mode truncation only).
        rngs: Flax NNX RNGs.
    """

    def __init__(
        self,
        metric: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        multiply: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize CliffordSpectralConv2d."""
        super().__init__()
        self.metric = jnp.asarray(metric, dtype=jnp.float32)
        if len(metric) != 2:
            raise ValueError("CliffordSpectralConv2d requires a 2D metric.")
        self.n_blades = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.multiply = multiply

        if multiply:
            scale = 1.0 / (in_channels * out_channels)
            key = rngs.params()
            self.weights = nnx.Param(
                scale
                * jax.random.uniform(
                    key,
                    (4, out_channels, in_channels, modes1 * 2, modes2 * 2),
                )
            )

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Forward pass.

        Args:
            x: Input ``(B, H, W, C_in, 4)``.
            deterministic: Unused.

        Returns:
            Output ``(B, H, W, C_out, 4)``.
        """
        # x: (B, H, W, C, I=4)
        batch = x.shape[0]
        h, w = x.shape[1], x.shape[2]
        channels_in = x.shape[3]
        n_blades = x.shape[4]

        # Dual complex pairs: blade indices (0,3) and (1,2)
        dual_1 = x[..., 0] + 1j * x[..., 3]  # (B, H, W, C)
        dual_2 = x[..., 1] + 1j * x[..., 2]  # (B, H, W, C)

        # FFT over spatial dims
        dual_1_ft = jnp.fft.fft2(dual_1, axes=(1, 2))
        dual_2_ft = jnp.fft.fft2(dual_2, axes=(1, 2))

        # Reassemble into real multivector in Fourier space
        # Layout: (B, C, H, W) per blade -> stack along channel dim
        # We need (B, I*C, H, W) for batchmul
        mv_ft = jnp.concatenate(
            [
                dual_1_ft.real.transpose(0, 3, 1, 2),  # blade 0
                dual_2_ft.real.transpose(0, 3, 1, 2),  # blade 1
                dual_2_ft.imag.transpose(0, 3, 1, 2),  # blade 2
                dual_1_ft.imag.transpose(0, 3, 1, 2),  # blade 3
            ],
            axis=1,
        )  # (B, 4*C, H, W)

        m1, m2 = self.modes1, self.modes2

        # Concatenate pos/neg frequency modes
        input_mul = jnp.concatenate(
            [
                jnp.concatenate([mv_ft[:, :, :m1, :m2], mv_ft[:, :, :m1, -m2:]], axis=-1),
                jnp.concatenate([mv_ft[:, :, -m1:, :m2], mv_ft[:, :, -m1:, -m2:]], axis=-1),
            ],
            axis=-2,
        )  # (B, 4*C, 2*m1, 2*m2)

        if self.multiply:
            _, kernel = get_2d_clifford_kernel(self.weights.value, self.metric)
            output_mul = batchmul2d(input_mul, kernel)
        else:
            output_mul = input_mul

        # Fill output modes
        out_ch = self.out_channels if self.multiply else channels_in
        out_ft = jnp.zeros((batch, n_blades * out_ch, h, w), dtype=mv_ft.dtype)
        out_ft = out_ft.at[:, :, :m1, :m2].set(output_mul[:, :, :m1, :m2])
        out_ft = out_ft.at[:, :, -m1:, :m2].set(output_mul[:, :, -m1:, :m2])
        out_ft = out_ft.at[:, :, :m1, -m2:].set(output_mul[:, :, :m1, -m2:])
        out_ft = out_ft.at[:, :, -m1:, -m2:].set(output_mul[:, :, -m1:, -m2:])

        # Reshape: (B, I*C, H, W) -> (B, I, C, H, W) -> (B, C, H, W, I)
        out_ft = out_ft.reshape(batch, n_blades, -1, h, w)
        out_ft = out_ft.transpose(0, 2, 3, 4, 1)  # (B, C, H, W, I)

        # Inverse FFT via dual pairs
        out_d1 = out_ft[..., 0] + 1j * out_ft[..., 3]
        out_d2 = out_ft[..., 1] + 1j * out_ft[..., 2]
        d1_ifft = jnp.fft.ifft2(out_d1, axes=(2, 3))
        d2_ifft = jnp.fft.ifft2(out_d2, axes=(2, 3))

        # Reconstruct multivector: (B, C, H, W, 4) -> (B, H, W, C, 4)
        output = jnp.stack(
            [
                d1_ifft.real,
                d2_ifft.real,
                d2_ifft.imag,
                d1_ifft.imag,
            ],
            axis=-1,
        )
        output = output.transpose(0, 2, 3, 1, 4)  # (B, H, W, C, I)
        return output


class CliffordSpectralConv3d(nnx.Module):
    """3D Clifford spectral convolution for Cl(3) algebras.

    Input:  ``(B, D, H, W, C_in, 8)``
    Output: ``(B, D, H, W, C_out, 8)``

    Args:
        metric: 3D metric signature, e.g. ``(1, 1, 1)``.
        in_channels: Input channel count.
        out_channels: Output channel count.
        modes1: Retained modes in dim 1.
        modes2: Retained modes in dim 2.
        modes3: Retained modes in dim 3.
        multiply: If ``False``, skip kernel multiplication.
        rngs: Flax NNX RNGs.
    """

    def __init__(
        self,
        metric: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        multiply: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize CliffordSpectralConv3d."""
        super().__init__()
        self.metric = jnp.asarray(metric, dtype=jnp.float32)
        if len(metric) != 3:
            raise ValueError("CliffordSpectralConv3d requires a 3D metric.")
        self.n_blades = 8
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.multiply = multiply

        if multiply:
            scale = 1.0 / (in_channels * out_channels)
            key = rngs.params()
            self.weights = nnx.Param(
                scale
                * jax.random.uniform(
                    key,
                    (8, out_channels, in_channels, modes1 * 2, modes2 * 2, modes3 * 2),
                )
            )

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Forward pass.

        Args:
            x: Input ``(B, D, H, W, C_in, 8)``.
            deterministic: Unused.

        Returns:
            Output ``(B, D, H, W, C_out, 8)``.
        """
        batch = x.shape[0]
        d, h, w = x.shape[1], x.shape[2], x.shape[3]
        channels_in = x.shape[4]
        n_blades = x.shape[5]

        # 4 dual pairs: (0,7), (1,6), (2,5), (3,4)
        dual_1 = x[..., 0] + 1j * x[..., 7]  # (B, D, H, W, C)
        dual_2 = x[..., 1] + 1j * x[..., 6]
        dual_3 = x[..., 2] + 1j * x[..., 5]
        dual_4 = x[..., 3] + 1j * x[..., 4]

        # FFT over spatial dims (1, 2, 3)
        dual_1_ft = jnp.fft.fftn(dual_1, axes=(1, 2, 3))
        dual_2_ft = jnp.fft.fftn(dual_2, axes=(1, 2, 3))
        dual_3_ft = jnp.fft.fftn(dual_3, axes=(1, 2, 3))
        dual_4_ft = jnp.fft.fftn(dual_4, axes=(1, 2, 3))

        # Reassemble: (B, C, D, H, W) per blade -> (B, 8*C, D, H, W)
        def _to_bchw(t):
            return t.transpose(0, 4, 1, 2, 3)

        mv_ft = jnp.concatenate(
            [
                _to_bchw(dual_1_ft.real),
                _to_bchw(dual_2_ft.real),
                _to_bchw(dual_3_ft.real),
                _to_bchw(dual_4_ft.real),
                _to_bchw(dual_4_ft.imag),
                _to_bchw(dual_3_ft.imag),
                _to_bchw(dual_2_ft.imag),
                _to_bchw(dual_1_ft.imag),
            ],
            axis=1,
        )

        m1, m2, m3 = self.modes1, self.modes2, self.modes3

        # Concatenate pos/neg frequency modes
        def _cat_modes(t):
            return jnp.concatenate(
                [
                    jnp.concatenate(
                        [
                            jnp.concatenate(
                                [t[:, :, :m1, :m2, :m3], t[:, :, :m1, :m2, -m3:]], axis=-1
                            ),
                            jnp.concatenate(
                                [t[:, :, :m1, -m2:, :m3], t[:, :, :m1, -m2:, -m3:]], axis=-1
                            ),
                        ],
                        axis=-2,
                    ),
                    jnp.concatenate(
                        [
                            jnp.concatenate(
                                [t[:, :, -m1:, :m2, :m3], t[:, :, -m1:, :m2, -m3:]], axis=-1
                            ),
                            jnp.concatenate(
                                [t[:, :, -m1:, -m2:, :m3], t[:, :, -m1:, -m2:, -m3:]], axis=-1
                            ),
                        ],
                        axis=-2,
                    ),
                ],
                axis=-3,
            )

        input_mul = _cat_modes(mv_ft)  # (B, 8*C, 2*m1, 2*m2, 2*m3)

        if self.multiply:
            _, kernel = get_3d_clifford_kernel(self.weights.value, self.metric)
            output_mul = batchmul3d(input_mul, kernel)
        else:
            output_mul = input_mul

        # Fill output modes
        out_ch = self.out_channels if self.multiply else channels_in
        out_ft = jnp.zeros((batch, n_blades * out_ch, d, h, w), dtype=mv_ft.dtype)

        # Scatter back the 8 octants
        def _set_octant(arr, s1, s2, s3, src_s1, src_s2, src_s3):
            return arr.at[:, :, s1, s2, s3].set(output_mul[:, :, src_s1, src_s2, src_s3])

        s_p1, s_n1 = slice(None, m1), slice(-m1, None)
        s_p2, s_n2 = slice(None, m2), slice(-m2, None)
        s_p3, s_n3 = slice(None, m3), slice(-m3, None)
        src_p1, src_n1 = slice(None, m1), slice(-m1, None)
        src_p2, src_n2 = slice(None, m2), slice(-m2, None)
        src_p3, src_n3 = slice(None, m3), slice(-m3, None)

        out_ft = out_ft.at[:, :, s_p1, s_p2, s_p3].set(output_mul[:, :, src_p1, src_p2, src_p3])
        out_ft = out_ft.at[:, :, s_p1, s_p2, s_n3].set(output_mul[:, :, src_p1, src_p2, src_n3])
        out_ft = out_ft.at[:, :, s_p1, s_n2, s_p3].set(output_mul[:, :, src_p1, src_n2, src_p3])
        out_ft = out_ft.at[:, :, s_p1, s_n2, s_n3].set(output_mul[:, :, src_p1, src_n2, src_n3])
        out_ft = out_ft.at[:, :, s_n1, s_p2, s_p3].set(output_mul[:, :, src_n1, src_p2, src_p3])
        out_ft = out_ft.at[:, :, s_n1, s_p2, s_n3].set(output_mul[:, :, src_n1, src_p2, src_n3])
        out_ft = out_ft.at[:, :, s_n1, s_n2, s_p3].set(output_mul[:, :, src_n1, src_n2, src_p3])
        out_ft = out_ft.at[:, :, s_n1, s_n2, s_n3].set(output_mul[:, :, src_n1, src_n2, src_n3])

        # Reshape: (B, 8*C, D, H, W) -> (B, 8, C, D, H, W) -> (B, C, D, H, W, 8)
        out_ft = out_ft.reshape(batch, n_blades, -1, d, h, w)
        out_ft = out_ft.transpose(0, 2, 3, 4, 5, 1)  # (B, C, D, H, W, I)

        # Inverse FFT via dual pairs
        out_d1 = out_ft[..., 0] + 1j * out_ft[..., 7]
        out_d2 = out_ft[..., 1] + 1j * out_ft[..., 6]
        out_d3 = out_ft[..., 2] + 1j * out_ft[..., 5]
        out_d4 = out_ft[..., 3] + 1j * out_ft[..., 4]
        d1_ifft = jnp.fft.ifftn(out_d1, axes=(2, 3, 4))
        d2_ifft = jnp.fft.ifftn(out_d2, axes=(2, 3, 4))
        d3_ifft = jnp.fft.ifftn(out_d3, axes=(2, 3, 4))
        d4_ifft = jnp.fft.ifftn(out_d4, axes=(2, 3, 4))

        # Reconstruct: (B, C, D, H, W, 8) -> (B, D, H, W, C, 8)
        output = jnp.stack(
            [
                d1_ifft.real,
                d2_ifft.real,
                d3_ifft.real,
                d4_ifft.real,
                d4_ifft.imag,
                d3_ifft.imag,
                d2_ifft.imag,
                d1_ifft.imag,
            ],
            axis=-1,
        )
        output = output.transpose(0, 2, 3, 4, 1, 5)  # (B, D, H, W, C, I)
        return output
