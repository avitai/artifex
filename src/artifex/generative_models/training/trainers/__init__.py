"""Model-specific trainers for different generative model architectures.

This module provides specialized trainers for various generative model types:
- VAE: Variational Autoencoders with KL annealing and beta-VAE support
- GAN: Generative Adversarial Networks with multiple loss types
- Diffusion: Diffusion models with SOTA training techniques
- Flow: Flow matching models with configurable time sampling
- Energy: Energy-based models with Contrastive Divergence and Langevin MCMC
- Autoregressive: Sequence models with teacher forcing and scheduled sampling

Trainer-owned objective closures provide the integration boundary to the shared
training infrastructure when that boundary is useful for callbacks,
checkpointing, and logging.

Example:
    ```python
    from artifex.generative_models.training.trainers import (
        DiffusionTrainer,
        DiffusionTrainingConfig,
    )
    from artifex.generative_models.training import Trainer
    from artifex.generative_models.training.callbacks import (
        CallbackList,
        CheckpointConfig,
        ModelCheckpoint,
    )

    # Create diffusion-specific trainer
    diff_trainer = DiffusionTrainer(
        noise_schedule,
        DiffusionTrainingConfig(prediction_type="v_prediction"),
    )

    # Integrate the explicit trainer-owned objective with the shared Trainer
    base_trainer = Trainer(
        model=model,
        training_config=training_config,
        loss_fn=diff_trainer.create_loss_fn(),
        callbacks=CallbackList(
            [ModelCheckpoint(CheckpointConfig(dirpath="checkpoints", monitor="val_loss"))]
        ),
    )
    ```
"""

from artifex.generative_models.training.trainers.autoregressive_trainer import (
    AutoregressiveTrainer,
    AutoregressiveTrainingConfig,
    create_causal_mask,
    create_combined_mask,
    create_padding_mask,
)
from artifex.generative_models.training.trainers.diffusion_trainer import (
    DiffusionTrainer,
    DiffusionTrainingConfig,
)
from artifex.generative_models.training.trainers.energy_trainer import (
    EnergyTrainer,
    EnergyTrainingConfig,
    ReplayBuffer,
)
from artifex.generative_models.training.trainers.flow_trainer import (
    FlowTrainer,
    FlowTrainingConfig,
)
from artifex.generative_models.training.trainers.gan_trainer import (
    GANTrainer,
    GANTrainingConfig,
)
from artifex.generative_models.training.trainers.vae_trainer import (
    VAETrainer,
    VAETrainingConfig,
)


__all__ = [
    # VAE
    "VAETrainer",
    "VAETrainingConfig",
    # GAN
    "GANTrainer",
    "GANTrainingConfig",
    # Diffusion
    "DiffusionTrainer",
    "DiffusionTrainingConfig",
    # Flow
    "FlowTrainer",
    "FlowTrainingConfig",
    # Energy
    "EnergyTrainer",
    "EnergyTrainingConfig",
    "ReplayBuffer",
    # Autoregressive
    "AutoregressiveTrainer",
    "AutoregressiveTrainingConfig",
    "create_causal_mask",
    "create_padding_mask",
    "create_combined_mask",
]
