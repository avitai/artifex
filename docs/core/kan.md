# Kolmogorov-Arnold Network Layers

**Module:** `generative_models.core.layers.kan`

**Source:** `generative_models/core/layers/kan/`

## Overview

Kolmogorov-Arnold Networks (KANs) replace the fixed activation on each node of a
dense layer with a learnable univariate function on each edge. Every layer here
takes an input of shape `(batch, n_in)` and returns `(batch, n_out)`, so it is a
drop-in replacement for `nnx.Linear` in a backbone. The family is a function
approximator, not a generative model: no artifex model uses it yet, and it stays
in the layers package under the backbone rule (U-KAN uses KAN layers as the
noise predictor of a diffusion U-Net). Today's consumer is opifex, which builds
physics-informed KANs on `create_kan_layer`.

The implementation is adapted from jaxKAN (MIT) with the artifex conventions:
`nnx.Module` base classes, a keyword-only `rngs`, and `deterministic` on
`__call__`. `tests/.../test_kan_reference.py` checks the spline and Chebyshev
layers against jaxKAN numerically.

Based on:

- Liu et al., *KAN: Kolmogorov-Arnold Networks*: <https://arxiv.org/abs/2404.19756>
- Rigas et al., *Adaptive Training of Grid-Dependent Physics-Informed KANs*
  (the jaxKAN paper): <https://arxiv.org/abs/2407.17611>
- Li et al., *U-KAN Makes Strong Backbone for Medical Image Segmentation and
  Generation* (the diffusion precedent): <https://arxiv.org/abs/2406.02918>

## Choosing a layer

| Layer | Edge function | Grid | Extra arguments |
| --- | --- | --- | --- |
| `DenseKANLayer` | B-spline with a knot vector per edge | `DenseKANGrid` | `k`, `grid_intervals`, `grid_range`, `grid_e` |
| `EfficientKANLayer` | B-spline with one knot vector per input | `EfficientKANGrid` | as above |
| `ChebyshevKANLayer` | Chebyshev polynomials up to degree `D` | none | `D`, `flavor` (`default`, `modified`, `exact`) |
| `LegendreKANLayer` | Legendre polynomials up to degree `D` | none | `D`, `flavor` |
| `FourierKANLayer` | Fourier series up to degree `D` | none | `D` |
| `RBFKANLayer` | Radial basis functions on learnable centers | `RBFKANGrid` | `kernel`, `grid_intervals`, `grid_range` |
| `SineKANLayer` | Sines with learnable frequency and phase | none | `D` |
| `ConvKANLayer` | An `EfficientKANLayer` applied to every extracted patch | `EfficientKANGrid` | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `spatial_ndim` |

Every dense-style layer takes `n_in`, `n_out`, `residual` (an activation added
to the edge function, `nnx.silu` for the spline layers and `None` for the basis
layers), `external_weights`, `init_scheme`, `add_bias` and `rngs`.
`BSplineBasis` is the evaluator the two spline layers share; `KANConfig` is a
frozen record of the same hyperparameters for code that builds several layers
from one setting.

```python
from flax import nnx

from artifex.generative_models.core.layers.kan import EfficientKANLayer

layer = EfficientKANLayer(n_in=2, n_out=5, k=3, grid_intervals=5, rngs=nnx.Rngs(0))
y = layer(x)  # (batch, 5)
```

`create_kan_layer(layer_type, **kwargs)` builds a layer from one of the names
`dense`, `efficient`, `chebyshev`, `fourier`, `legendre`, `rbf`, `sine`, `conv`
and raises `ValueError` for anything else.

## Grid refinement

The spline layers and the RBF layer keep their knots or centers in an
`nnx.Variable` (state, not a trained parameter), so a grid can be refined
during training without rebuilding the module. `update_grid(x, new_intervals)`
rebuilds the grid with `new_intervals` intervals from the activations `x`
(`grid_e` mixes a uniform grid with a sample-dependent one) and refits the
coefficients by least squares so the edge functions are preserved. The basis
layers without a grid implement the same method, so a training loop can call
it positionally on every layer of a network.

## Classes

### DenseKANLayer, EfficientKANLayer

```python
class DenseKANLayer(nnx.Module)
class EfficientKANLayer(nnx.Module)
```

Both expose `basis(x)` (the B-spline features), `update_grid(x, new_intervals)` and
`__call__(x, *, deterministic=False)`. The dense variant stores an independent
knot vector for every edge; the efficient variant shares one per input feature,
which is the jaxKAN `SplineLayer` layout.

### ChebyshevKANLayer, LegendreKANLayer, FourierKANLayer, RBFKANLayer, SineKANLayer

Fixed-basis layers: the edge function is a linear combination of `D` basis
functions with learnable coefficients. `RBFKANLayer` keeps its centers in an
`RBFKANGrid` and moves them with `update_grid`.

### ConvKANLayer

Extracts local patches with `jax.lax.conv_general_dilated_patches` and applies
one `EfficientKANLayer` at every spatial position, so the kernel is a KAN rather
than a matrix. Works for `spatial_ndim` of 1, 2 or 3 on channels-last inputs with
the usual `kernel_size`, `stride` and `padding` conventions.

### Grids and basis

`DenseKANGrid`, `EfficientKANGrid` and `RBFKANGrid` own the knot or center
positions; `BSplineBasis(k)` evaluates B-splines of order `k` on a knot vector.

### KANConfig

A frozen dataclass with `k`, `grid_intervals`, `grid_range`, `grid_e`,
`degree`, `residual`, `external_weights`, `add_bias` and `init_scheme`;
validation runs in `__post_init__`.

## Functions

### create_kan_layer

```python
def create_kan_layer(layer_type: str, **kwargs) -> nnx.Module
```
