# Core Base Classes

The live base layer is intentionally small. Use it for shared NNX-friendly
building blocks, not for a fake universal training API.

## Public Imports

```python
from flax import nnx
from artifex.generative_models.core.base import GenerativeModel, GenerativeModule
```

## GenerativeModule

`GenerativeModule` is the low-level base for Artifex components.

- constructor: `GenerativeModule(*, rngs: nnx.Rngs, precision: jax.lax.Precision | None = None)`
- stores RNG state on the module instance at initialization time
- keeps precision handling local to the module owner

## GenerativeModel

`GenerativeModel` is the shared user-facing base for concrete model families.
The interface is deliberately narrow:

- `GenerativeModel.__call__(x, *args, **kwargs)`
- `GenerativeModel.generate(n_samples=1, **kwargs)`
- `GenerativeModel.loss_fn(batch, model_outputs, **kwargs)` for single-objective families only

Shared models use **stored module RNG state** instead of call-time RNG plumbing.
Switch modes with `model.train()` and `model.eval()`; this shared layer does not
teach a generic `training=` keyword.

## Objective Ownership

Artifex does not force one fake objective API across incompatible families.

- `single-objective` families commonly implement `loss_fn(...)`
- `multi-objective` families such as GANs keep family-local helpers like
  `generator_objective(...)` and `discriminator_objective(...)`

Optional helpers such as `encode(...)`, `decode(...)`, `log_prob(...)`,
`reconstruct(...)`, and `interpolate(...)` remain family-owned surfaces instead
of shared protocol requirements.

## Minimal Example

```python
import jax
from flax import nnx
from artifex.generative_models.core.base import GenerativeModule


class CustomLayer(GenerativeModule):
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        super().__init__(rngs=rngs)
        self.dense = nnx.Linear(64, features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dense(x)
```
