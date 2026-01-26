"""Clifford algebra layers for geometric deep learning.

Ported from ``microsoft/cliffordlayers`` (MIT license).
Provides convolution, linear, spectral, normalization, and activation
layers that operate on multivector-valued fields in Clifford algebras
Cl(1), Cl(2), and Cl(3).
"""

from artifex.generative_models.core.layers.clifford.algebra import (
    BasisBladeOrder,
    CliffordAlgebra,
)
from artifex.generative_models.core.layers.clifford.conv import (
    CliffordConv1d,
    CliffordConv2d,
    CliffordConv3d,
)
from artifex.generative_models.core.layers.clifford.kernels import (
    get_1d_clifford_kernel,
    get_2d_clifford_kernel,
    get_2d_clifford_rotation_kernel,
    get_3d_clifford_kernel,
)
from artifex.generative_models.core.layers.clifford.linear import CliffordLinear
from artifex.generative_models.core.layers.clifford.norm import (
    CliffordBatchNorm,
    CliffordGroupNorm,
    MultiVectorActivation,
)
from artifex.generative_models.core.layers.clifford.spectral import (
    CliffordSpectralConv2d,
    CliffordSpectralConv3d,
)


__all__ = [
    # Algebra
    "BasisBladeOrder",
    "CliffordAlgebra",
    # Kernels
    "get_1d_clifford_kernel",
    "get_2d_clifford_kernel",
    "get_2d_clifford_rotation_kernel",
    "get_3d_clifford_kernel",
    # Linear
    "CliffordLinear",
    # Convolution
    "CliffordConv1d",
    "CliffordConv2d",
    "CliffordConv3d",
    # Spectral
    "CliffordSpectralConv2d",
    "CliffordSpectralConv3d",
    # Normalization & Activation
    "CliffordBatchNorm",
    "CliffordGroupNorm",
    "MultiVectorActivation",
]
