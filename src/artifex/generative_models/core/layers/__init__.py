"""Backbone layers.

The building blocks generative models are assembled from: attention, positional
encodings, residual and ResNet blocks, transformers, and equivariant graph layers.
The package also holds backbone families that no artifex model uses yet and that
sibling libraries build on: the Kolmogorov-Arnold Network layers (``kan``) and the
Clifford-algebra layers (``clifford``). A layer belongs here when it can serve as
the backbone of a generative model, whichever library consumes it today.
"""

# Causal utilities for autoregressive models
# Flash attention
from artifex.generative_models.core.layers.attention_backend import AttentionBackend
from artifex.generative_models.core.layers.causal import (
    apply_causal_mask,
    create_attention_mask,
    create_causal_mask,
    shift_right,
)

# Clifford algebra layers
from artifex.generative_models.core.layers.clifford import (
    BasisBladeOrder,
    CliffordAlgebra,
    CliffordBatchNorm,
    CliffordConv1d,
    CliffordConv2d,
    CliffordConv3d,
    CliffordGroupNorm,
    CliffordLinear,
    CliffordSpectralConv2d,
    CliffordSpectralConv3d,
    get_1d_clifford_kernel,
    get_2d_clifford_kernel,
    get_2d_clifford_rotation_kernel,
    get_3d_clifford_kernel,
    MultiVectorActivation,
)
from artifex.generative_models.core.layers.egnn import EGNNBlock, EGNNLayer
from artifex.generative_models.core.layers.flash_attention import (
    flash_dot_product_attention,
    FlashMultiHeadAttention,
)

# KAN layers
from artifex.generative_models.core.layers.kan import (
    BSplineBasis,
    ChebyshevKANLayer,
    ConvKANLayer,
    create_kan_layer,
    DenseKANGrid,
    DenseKANLayer,
    EfficientKANGrid,
    EfficientKANLayer,
    FourierKANLayer,
    KANConfig,
    LegendreKANLayer,
    RBFKANGrid,
    RBFKANLayer,
    SineKANLayer,
)

# Positional encodings
from artifex.generative_models.core.layers.positional import (
    LearnedPositionalEncoding,
    PositionalEncoding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)

# Residual blocks
from artifex.generative_models.core.layers.residual import (
    BaseResidualBlock,
    Conv1DResidualBlock,
    Conv2DResidualBlock,
    create_residual_block,
    ResidualBlock,
    WaveNetResidualBlock,
)

# ResNet blocks
from artifex.generative_models.core.layers.resnet import BottleneckBlock, ResNetBlock

# Transformer blocks
from artifex.generative_models.core.layers.transformers import (
    TransformerDecoder,
    TransformerDecoderBlock,
    TransformerEncoder,
    TransformerEncoderBlock,
)


__all__ = [
    # Causal utilities
    "create_causal_mask",
    "shift_right",
    "create_attention_mask",
    "apply_causal_mask",
    # Positional encodings
    "PositionalEncoding",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "RotaryPositionalEncoding",
    # Residual blocks
    "BaseResidualBlock",
    "ResidualBlock",
    "Conv1DResidualBlock",
    "Conv2DResidualBlock",
    "WaveNetResidualBlock",
    "create_residual_block",
    # ResNet blocks
    "ResNetBlock",
    "BottleneckBlock",
    # Transformer blocks
    "TransformerEncoderBlock",
    "TransformerDecoderBlock",
    "TransformerEncoder",
    "TransformerDecoder",
    # Flash attention
    "FlashMultiHeadAttention",
    "flash_dot_product_attention",
    "AttentionBackend",
    # EGNN layers
    "EGNNBlock",
    "EGNNLayer",
    # KAN layers
    "KANConfig",
    "BSplineBasis",
    "DenseKANGrid",
    "EfficientKANGrid",
    "RBFKANGrid",
    "DenseKANLayer",
    "EfficientKANLayer",
    "ChebyshevKANLayer",
    "FourierKANLayer",
    "LegendreKANLayer",
    "RBFKANLayer",
    "SineKANLayer",
    "ConvKANLayer",
    "create_kan_layer",
    # Clifford algebra layers
    "BasisBladeOrder",
    "CliffordAlgebra",
    "CliffordLinear",
    "CliffordConv1d",
    "CliffordConv2d",
    "CliffordConv3d",
    "CliffordSpectralConv2d",
    "CliffordSpectralConv3d",
    "CliffordBatchNorm",
    "CliffordGroupNorm",
    "MultiVectorActivation",
    "get_1d_clifford_kernel",
    "get_2d_clifford_kernel",
    "get_2d_clifford_rotation_kernel",
    "get_3d_clifford_kernel",
]
