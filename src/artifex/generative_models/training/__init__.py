"""Training module for generative models."""

from artifex.generative_models.training.gradient_accumulation import (
    DynamicLossScaler,
    DynamicLossScalerConfig,
    GradientAccumulator,
    GradientAccumulatorConfig,
)
from artifex.generative_models.training.loops import (
    train_epoch_staged,
    train_epoch_streaming,
)
from artifex.generative_models.training.optimizers import create_optimizer
from artifex.generative_models.training.rl import (
    DPOConfig,
    DPOTrainer,
    GRPOConfig,
    GRPOTrainer,
    PPOConfig,
    PPOTrainer,
    REINFORCEConfig,
    REINFORCETrainer,
)
from artifex.generative_models.training.schedulers import create_scheduler
from artifex.generative_models.training.trainer import Trainer


__all__ = [
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    # High-performance training loops
    "train_epoch_staged",
    "train_epoch_streaming",
    # Gradient accumulation and loss scaling
    "GradientAccumulator",
    "GradientAccumulatorConfig",
    "DynamicLossScaler",
    "DynamicLossScalerConfig",
    # RL Training
    "REINFORCEConfig",
    "REINFORCETrainer",
    "PPOConfig",
    "PPOTrainer",
    "GRPOConfig",
    "GRPOTrainer",
    "DPOConfig",
    "DPOTrainer",
]
