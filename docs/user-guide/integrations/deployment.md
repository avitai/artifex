# Model Deployment

Artifex does not currently ship a generic deployment framework, service
scaffold, or deployment CLI. The retained deployment boundary is low-level and
family-owned: save state with `artifex.generative_models.core.checkpointing`,
rebuild the concrete model template from its typed config, and optionally use
`ProductionOptimizer` before wiring the model into your own application
service.

## What Exists Today

- low-level checkpoint persistence through `setup_checkpoint_manager(...)`,
  `save_checkpoint(...)`, and `load_checkpoint(...)`
- family-owned model loading and generation as described in
  [../inference/overview.md](../inference/overview.md)
- experimental compiled-pipeline measurement through `OptimizationTarget` and
  `ProductionOptimizer`

## Export Checkpoint State

```python
import json
from pathlib import Path

from artifex.generative_models.core.checkpointing import (
    save_checkpoint,
    setup_checkpoint_manager,
)

export_dir = Path("./deployments/vae-v1")
checkpoint_manager, checkpoint_path = setup_checkpoint_manager(export_dir)
save_checkpoint(checkpoint_manager, model, step=final_step)

(export_dir / "metadata.json").write_text(
    json.dumps({"family": "vae", "step": final_step}, indent=2),
    encoding="utf-8",
)
```

## Restore The Concrete Model Template

Use the same family-owned template you trained, then restore Orbax state into
that template.

```python
from flax import nnx

from artifex.generative_models.core.checkpointing import (
    load_checkpoint,
    setup_checkpoint_manager,
)
from artifex.generative_models.core.configuration import (
    DecoderConfig,
    EncoderConfig,
    VAEConfig,
)
from artifex.generative_models.models.vae import VAE

latent_dim = 32
vae_config = VAEConfig(
    name="mnist-vae",
    encoder=EncoderConfig(
        input_shape=(28, 28, 1),
        latent_dim=latent_dim,
        hidden_dims=(256, 128),
        activation="relu",
    ),
    decoder=DecoderConfig(
        latent_dim=latent_dim,
        output_shape=(28, 28, 1),
        hidden_dims=(128, 256),
        activation="relu",
        output_activation="sigmoid",
    ),
    encoder_type="dense",
)

model_template = VAE(vae_config, rngs=nnx.Rngs(0))
checkpoint_manager, _ = setup_checkpoint_manager("./deployments/vae-v1")
restored_model, step = load_checkpoint(checkpoint_manager, model_template)
```

## Experimental Production Optimization

```python
import jax.numpy as jnp

from artifex.generative_models.inference.optimization.production import (
    OptimizationTarget,
    ProductionOptimizer,
)

optimizer = ProductionOptimizer()
result = optimizer.optimize_for_production(
    model=restored_model,
    optimization_target=OptimizationTarget(latency_ms=50.0),
    sample_inputs=(jnp.zeros((1, 28, 28, 1)),),
)

assert result.optimization_techniques == ["jit_compilation"]
```

## Boundary

Bring your own HTTP server, queue, or worker orchestration around this
family-owned loading and inference surface. Artifex currently stops at
checkpointing plus the experimental optimizer; the application service layer
and surrounding application framework are owned by your deployment stack.
