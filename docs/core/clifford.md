# Clifford Algebra Layers

**Module:** `generative_models.core.layers.clifford`

**Source:** `generative_models/core/layers/clifford/`

## Overview

Clifford layers operate on multivector-valued fields: every channel carries the
`2^n` blade components of a Clifford algebra `Cl(p, q)` over `n = 1, 2, 3` basis
vectors, and each layer applies the geometric product rather than treating the
components as independent scalars. They are the backbone of neural operators for
PDEs whose fields are naturally multivectors (velocity and vorticity, complex
and quaternionic signals). No artifex model uses them yet; they stay in the
layers package under the backbone rule, and opifex builds its Clifford Fourier
neural operators on the convolution and spectral layers.

The layers are ported from `microsoft/cliffordlayers` (MIT) to Flax NNX. Inputs
are channels-last with the blade axis innermost: a 2D field is
`(batch, height, width, channels, n_blades)`.

Based on:

- Brandstetter, van den Berg, Welling, Gupta, *Clifford Neural Layers for PDE
  Modeling* (ICLR 2023): <https://arxiv.org/abs/2209.04934>

## The algebra

`CliffordAlgebra` is a plain object, not a module: `CliffordAlgebra(metric)`. The `metric` is the
diagonal of the quadratic form, one entry per basis vector, so `(1, 1)` is
`Cl(2, 0)` (the complex numbers as a subalgebra) and `(1, 1, 1)` is `Cl(3, 0)`
(the quaternions). It precomputes the Cayley table of the geometric product and
the `BasisBladeOrder` (short-lex blade indexing with a grade per blade) that
every layer below uses. Only dimensions 1, 2 and 3 are supported; anything else
raises `ValueError`.

The four kernel builders turn a stack of per-blade weights into the block
matrix that implements the geometric product for a given dimension:
`get_1d_clifford_kernel`, `get_2d_clifford_kernel`,
`get_2d_clifford_rotation_kernel` (the rotational variant for `Cl(2, 0)`) and
`get_3d_clifford_kernel`.

## Layers

Every layer takes the `metric` first, then channel counts, and `rngs`
keyword-only.

| Layer | Input | What it does |
| --- | --- | --- |
| `CliffordLinear` | `(B, C_in, n_blades)` | Linear map in Clifford space through the kernel construction |
| `CliffordConv1d` | `(B, L, C_in, 2)` | Convolution over multivector fields in `Cl(1)` |
| `CliffordConv2d` | `(B, H, W, C_in, 4)` | Convolution in `Cl(2)`, with the rotation kernel available |
| `CliffordConv3d` | `(B, D, H, W, C_in, 8)` | Convolution in `Cl(3)` |
| `CliffordSpectralConv2d` | `(B, H, W, C_in, 4)` | Clifford Fourier transform through dual complex pairs, geometric product on the retained `modes1 × modes2`, inverse transform |
| `CliffordSpectralConv3d` | `(B, D, H, W, C_in, 8)` | The 3D counterpart with `modes1`, `modes2`, `modes3` |
| `CliffordBatchNorm` | `(B, *D, C, n_blades)` | Whitening across blades with running statistics |
| `CliffordGroupNorm` | `(B, *D, C, n_blades)` | The same whitening over `num_groups` channel groups |
| `MultiVectorActivation` | `(B, *D, C, n_blades)` | A gate computed from selected blades (`kernel_blades`) applied to the field, with `linear`, `sum` or `mean` aggregation |

The convolutions accept `kernel_size`, `stride`, `padding`, `dilation`,
`groups` and `use_bias` with the usual meanings. The spectral layers take
`multiply=False` to skip the kernel multiplication and only truncate modes.

```python
from flax import nnx

from artifex.generative_models.core.layers.clifford import (
    CliffordConv2d,
    CliffordSpectralConv2d,
)

rngs = nnx.Rngs(0)
conv = CliffordConv2d(metric=(1, 1), in_channels=8, out_channels=16, kernel_size=3, rngs=rngs)
spectral = CliffordSpectralConv2d(
    metric=(1, 1), in_channels=16, out_channels=16, modes1=12, modes2=12, rngs=rngs
)
y = spectral(conv(x))  # x: (batch, height, width, 8, 4)
```

## Classes

### CliffordAlgebra, BasisBladeOrder

```python
class CliffordAlgebra(metric: Sequence[int] | jax.Array)
class BasisBladeOrder(n_vectors: int)
```

### CliffordLinear

```python
class CliffordLinear(metric, in_channels, out_channels, use_bias=True, *, rngs)
```

### CliffordConv1d, CliffordConv2d, CliffordConv3d

```python
class CliffordConv2d(metric, in_channels, out_channels, kernel_size=3, stride=1,
                     padding=0, dilation=1, groups=1, use_bias=True,
                     rotation=False, *, rngs)
```

`rotation=True` selects `get_2d_clifford_rotation_kernel`; it is only defined
for the 2D layer.

```python
```

### CliffordSpectralConv2d, CliffordSpectralConv3d

```python
class CliffordSpectralConv2d(metric, in_channels, out_channels, modes1, modes2,
                             multiply=True, *, rngs)
```

### CliffordBatchNorm, CliffordGroupNorm, MultiVectorActivation

```python
class CliffordBatchNorm(metric, channels, epsilon=1e-5, momentum=0.1,
                        use_affine=True, use_running_stats=True, *, rngs)
class CliffordGroupNorm(metric, num_groups, channels, epsilon=1e-5, use_affine=True, *, rngs)
class MultiVectorActivation(channels, n_blades, input_blades, kernel_blades=None,
                            aggregation="linear", *, rngs)
```

## Functions

### get_1d_clifford_kernel, get_2d_clifford_kernel, get_2d_clifford_rotation_kernel, get_3d_clifford_kernel

Kernel builders that assemble the geometric-product block matrix from per-blade
weights for the given dimension; the layers call them, and a custom layer can
reuse them.
