# Weights & Biases Integration

Artifex currently supports Weights & Biases through `WandbLoggerCallback` in
the training callback layer and the package-local `WandbLogger` utility for
direct logging. It does not ship a broader sweep, artifact, or model-registry
framework on top of W&B.

## Supported Owners

- `WandbLoggerCallback`
- `WandbLoggerConfig`
- `CallbackList`
- `Trainer.train(...)`
- `WandbLogger`

## Wire The Built-In Callback

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    WandbLoggerCallback,
    WandbLoggerConfig,
)
from artifex.generative_models.training.trainer import Trainer

callbacks = CallbackList(
    [
        WandbLoggerCallback(
            WandbLoggerConfig(
                project="my-project",
                name="experiment-1",
                tags=["vae", "baseline"],
                config={"learning_rate": 1e-3},
            )
        )
    ]
)

trainer = Trainer(
    model=model,
    training_config=training_config,
    loss_fn=loss_fn,
    callbacks=callbacks,
)

trainer.train(
    train_data=train_data,
    num_epochs=10,
    batch_size=32,
    val_data=val_data,
)
```

## Direct Logging Outside Trainer

```python
from artifex.generative_models.utils.logging import WandbLogger

logger = WandbLogger(
    name="experiment-1",
    project="my-project",
    config={"learning_rate": 1e-3},
)
logger.log_scalars({"train/loss": 0.12, "val/loss": 0.15}, step=10)
logger.close()
```

## Boundary

Keep sweeps, artifact pipelines, and registry workflows in your W&B or
application-layer code unless a real Artifex owner is added for them. The live
runtime currently owns callback-based metric logging plus the package-local
logger, not a full shared integrations framework.
