"""Model-specific trainers for different generative model architectures.

This module provides specialized trainers for various generative model types:
- VAE: Variational Autoencoders with KL annealing and beta-VAE support
- GAN: Generative Adversarial Networks with multiple loss types
- Diffusion: Diffusion models with SOTA training techniques
- Flow: Flow matching models with CFM and OT-CFM support
- Energy: Energy-based models with Contrastive Divergence and MCMC
- Autoregressive: Sequence models with teacher forcing and scheduled sampling

Each trainer provides a `create_loss_fn()` method for DRY integration with
the base Trainer, enabling callbacks, checkpointing, and logging support.

Example:
    ```python
    from artifex.generative_models.training.trainers import (
        DiffusionTrainer,
        DiffusionTrainingConfig,
    )
    from artifex.generative_models.training import Trainer

    # Create diffusion-specific trainer
    diff_trainer = DiffusionTrainer(model, optimizer, noise_schedule, config)

    # Integrate with base Trainer
    base_trainer = Trainer(
        model=model,
        training_config=training_config,
        loss_fn=diff_trainer.create_loss_fn(),
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
