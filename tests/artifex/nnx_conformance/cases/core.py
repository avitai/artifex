"""Cases for ``artifex.generative_models.core``: its layers and its base networks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.testing.nnx_conformance import Case

from tests.artifex.nnx_conformance.cases.inputs import BATCH, normal


_CORE = "artifex.generative_models.core"
_LAYERS = f"{_CORE}.layers"


def _egnn_inputs(key: jax.Array) -> tuple[jax.Array, ...]:
    nodes, features, edge = 5, 8, 4
    k1, k2, k3 = jax.random.split(key, 3)
    adjacency = jnp.broadcast_to(jnp.ones((nodes, nodes)) - jnp.eye(nodes), (BATCH, nodes, nodes))
    return (
        jax.random.normal(k1, (BATCH, nodes, features)),
        jax.random.normal(k2, (BATCH, nodes, 3)),
        adjacency,
        jax.random.normal(k3, (BATCH, nodes, nodes, edge)),
    )


def _decoder_inputs(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (BATCH, 6, 16)), jax.random.normal(k2, (BATCH, 7, 16))


def layer_cases() -> dict[str, Case]:
    """Return the cases for ``core.layers``, by the qualified name of the class each builds."""

    from artifex.generative_models.core.layers import (  # noqa: PLC0415
        clifford,
        egnn,
        flash_attention,
        positional,
        residual,
        resnet,
        transformers,
    )
    from artifex.generative_models.core.layers.kan import (  # noqa: PLC0415
        basis_layers,
        conv,
        grids,
        spline,
    )

    return {
        f"{_LAYERS}.clifford.linear.CliffordLinear": Case(
            build=lambda r: clifford.CliffordLinear(
                metric=(1, 1), in_channels=3, out_channels=5, rngs=r
            ),
            inputs=normal(3, 4),
        ),
        f"{_LAYERS}.clifford.conv.CliffordConv1d": Case(
            build=lambda r: clifford.CliffordConv1d(
                metric=(1,), in_channels=3, out_channels=5, kernel_size=3, padding="SAME", rngs=r
            ),
            inputs=normal(8, 3, 2),
        ),
        f"{_LAYERS}.clifford.conv.CliffordConv2d": Case(
            build=lambda r: clifford.CliffordConv2d(
                metric=(1, 1), in_channels=3, out_channels=5, kernel_size=3, padding="SAME", rngs=r
            ),
            inputs=normal(6, 6, 3, 4),
        ),
        f"{_LAYERS}.clifford.conv.CliffordConv3d": Case(
            build=lambda r: clifford.CliffordConv3d(
                metric=(1, 1, 1),
                in_channels=2,
                out_channels=3,
                kernel_size=3,
                padding="SAME",
                rngs=r,
            ),
            inputs=normal(4, 4, 4, 2, 8),
        ),
        f"{_LAYERS}.clifford.spectral.CliffordSpectralConv2d": Case(
            build=lambda r: clifford.CliffordSpectralConv2d(
                metric=(1, 1), in_channels=3, out_channels=5, modes1=2, modes2=2, rngs=r
            ),
            inputs=normal(8, 8, 3, 4),
        ),
        f"{_LAYERS}.clifford.spectral.CliffordSpectralConv3d": Case(
            build=lambda r: clifford.CliffordSpectralConv3d(
                metric=(1, 1, 1),
                in_channels=2,
                out_channels=3,
                modes1=2,
                modes2=2,
                modes3=2,
                rngs=r,
            ),
            inputs=normal(4, 4, 4, 2, 8),
        ),
        f"{_LAYERS}.clifford.norm.CliffordBatchNorm": Case(
            build=lambda r: clifford.CliffordBatchNorm(metric=(1, 1), channels=3, rngs=r),
            inputs=normal(6, 3, 4),
            has_batch_stats=True,
        ),
        f"{_LAYERS}.clifford.norm.CliffordGroupNorm": Case(
            build=lambda r: clifford.CliffordGroupNorm(
                metric=(1, 1), num_groups=2, channels=4, rngs=r
            ),
            inputs=normal(5, 5, 4, 4),
        ),
        f"{_LAYERS}.clifford.norm.MultiVectorActivation": Case(
            build=lambda r: clifford.MultiVectorActivation(
                channels=3, n_blades=4, input_blades=(0, 1, 2, 3), rngs=r
            ),
            inputs=normal(3, 4),
        ),
        f"{_LAYERS}.egnn.EGNNLayer": Case(
            build=lambda r: egnn.EGNNLayer(edge_dim=4, hidden_dim=8, dropout_rate=0.5, rngs=r),
            inputs=_egnn_inputs,
            has_dropout=True,
        ),
        f"{_LAYERS}.egnn.EGNNBlock": Case(
            build=lambda r: egnn.EGNNBlock(8, 6, 16, dropout_rate=0.5, rngs=r),
            inputs=normal(5, 8),
            has_dropout=True,
        ),
        f"{_LAYERS}.flash_attention.FlashMultiHeadAttention": Case(
            build=lambda r: flash_attention.FlashMultiHeadAttention(
                num_heads=2, in_features=16, dropout_rate=0.5, decode=False, rngs=r
            ),
            inputs=normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.positional.SinusoidalPositionalEncoding": Case(
            build=lambda r: positional.SinusoidalPositionalEncoding(
                dim=16, max_len=12, dropout_rate=0.5, rngs=r
            ),
            inputs=normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.positional.LearnedPositionalEncoding": Case(
            build=lambda r: positional.LearnedPositionalEncoding(
                dim=16, max_len=12, dropout_rate=0.5, rngs=r
            ),
            inputs=normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.positional.RotaryPositionalEncoding": Case(
            build=lambda r: positional.RotaryPositionalEncoding(
                dim=16, max_len=12, dropout_rate=0.5, rngs=r
            ),
            inputs=normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.residual.Conv1DResidualBlock": Case(
            build=lambda r: residual.Conv1DResidualBlock(residual_channels=4, dilation=2, rngs=r),
            inputs=normal(10, 4),
        ),
        f"{_LAYERS}.residual.Conv2DResidualBlock": Case(
            build=lambda r: residual.Conv2DResidualBlock(channels=4, activation=nnx.silu, rngs=r),
            inputs=normal(6, 6, 4),
        ),
        f"{_LAYERS}.resnet.ResNetBlock": Case(
            build=lambda r: resnet.ResNetBlock(
                in_features=4, features=6, activation=nnx.silu, rngs=r
            ),
            inputs=normal(6, 6, 4),
            has_batch_stats=True,
            is_rowwise=False,
        ),
        f"{_LAYERS}.resnet.BottleneckBlock": Case(
            build=lambda r: resnet.BottleneckBlock(
                in_features=8, out_features=8, activation=nnx.silu, rngs=r
            ),
            inputs=normal(6, 6, 8),
            has_batch_stats=True,
            is_rowwise=False,
        ),
        f"{_LAYERS}.transformers.FeedForwardNetwork": Case(
            build=lambda r: transformers.FeedForwardNetwork(
                in_features=16, hidden_features=32, dropout_rate=0.5, rngs=r
            ),
            inputs=normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.transformers.TransformerEncoderBlock": Case(
            build=lambda r: transformers.TransformerEncoderBlock(
                hidden_dim=16, num_heads=2, dropout_rate=0.5, attention_dropout_rate=0.5, rngs=r
            ),
            inputs=normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.transformers.TransformerEncoder": Case(
            build=lambda r: transformers.TransformerEncoder(
                num_layers=2,
                hidden_dim=16,
                num_heads=2,
                dropout_rate=0.5,
                attention_dropout_rate=0.5,
                rngs=r,
            ),
            inputs=normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.transformers.TransformerDecoderBlock": Case(
            build=lambda r: transformers.TransformerDecoderBlock(
                hidden_dim=16, num_heads=2, dropout_rate=0.5, attention_dropout_rate=0.5, rngs=r
            ),
            inputs=_decoder_inputs,
            has_dropout=True,
        ),
        f"{_LAYERS}.transformers.TransformerDecoder": Case(
            build=lambda r: transformers.TransformerDecoder(
                num_layers=2,
                hidden_dim=16,
                num_heads=2,
                dropout_rate=0.5,
                attention_dropout_rate=0.5,
                rngs=r,
            ),
            inputs=_decoder_inputs,
            has_dropout=True,
        ),
        f"{_LAYERS}.kan.spline.DenseKANLayer": Case(
            build=lambda r: spline.DenseKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=normal(3),
        ),
        f"{_LAYERS}.kan.spline.EfficientKANLayer": Case(
            build=lambda r: spline.EfficientKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=normal(3),
        ),
        f"{_LAYERS}.kan.basis_layers.ChebyshevKANLayer": Case(
            build=lambda r: basis_layers.ChebyshevKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=normal(3),
        ),
        f"{_LAYERS}.kan.basis_layers.FourierKANLayer": Case(
            build=lambda r: basis_layers.FourierKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=normal(3),
        ),
        f"{_LAYERS}.kan.basis_layers.LegendreKANLayer": Case(
            build=lambda r: basis_layers.LegendreKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=normal(3),
        ),
        f"{_LAYERS}.kan.basis_layers.RBFKANLayer": Case(
            build=lambda r: basis_layers.RBFKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=normal(3),
        ),
        f"{_LAYERS}.kan.basis_layers.SineKANLayer": Case(
            build=lambda r: basis_layers.SineKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=normal(3),
        ),
        f"{_LAYERS}.kan.conv.ConvKANLayer": Case(
            build=lambda r: conv.ConvKANLayer(in_channels=2, out_channels=3, rngs=r),
            inputs=normal(5, 5, 2),
        ),
        f"{_LAYERS}.kan.grids.BSplineBasis": Case(
            build=lambda r: grids.BSplineBasis(k=3),
            inputs=lambda key: (jax.random.uniform(key, (BATCH, 3), minval=-1.0, maxval=1.0),),
            call=lambda module, x: module(
                x, grids.EfficientKANGrid(n_nodes=3, k=3, grid_intervals=3).knots[...]
            ),
            is_differentiable=False,
        ),
    }


def base_cases() -> dict[str, Case]:
    """Return the cases for ``core.base``'s networks, by qualified class name."""

    from artifex.generative_models.core import base  # noqa: PLC0415

    return {
        f"{_CORE}.base.MLP": Case(
            build=lambda r: base.MLP([8, 6], in_features=4, dropout_rate=0.5, rngs=r),
            inputs=normal(4),
            has_dropout=True,
        ),
        f"{_CORE}.base.CNN": Case(
            build=lambda r: base.CNN([4, 6], in_features=3, activation=nnx.silu, rngs=r),
            inputs=normal(8, 8, 3),
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_CORE}.base.GenerativeModule": "base class; each subclass's case runs its behaviour",
    f"{_CORE}.base.GenerativeModel": "base class; each subclass's case runs its behaviour",
    f"{_CORE}.base._DepthwiseSeparableConv": "private part of CNN, run through CNN's case",
    f"{_LAYERS}.clifford.conv._CliffordConvNd": "private base of the 1-D, 2-D and 3-D convolutions",
    f"{_LAYERS}.positional.PositionalEncoding": "base class whose __call__ raises NotImplementedError",
    f"{_LAYERS}.kan.grids.DenseKANGrid": "holds a KAN layer's knots; run through DenseKANLayer",
    f"{_LAYERS}.kan.grids.EfficientKANGrid": "holds a KAN layer's knots; run through EfficientKANLayer",
    f"{_LAYERS}.kan.grids.RBFKANGrid": "holds a KAN layer's centres; run through RBFKANLayer",
}
