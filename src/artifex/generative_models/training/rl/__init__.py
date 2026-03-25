"""Reinforcement learning components for generative models.

This module provides RL algorithms for training generative models:
- REINFORCE: Basic policy gradient with variance reduction
- PPO: Proximal Policy Optimization with GAE
- GRPO: Group Relative Policy Optimization (critic-free, from DeepSeek)
- DPO: Direct Preference Optimization with SimPO support

Example:
    >>> from artifex.generative_models.training.rl import (
    ...     REINFORCEConfig,
    ...     REINFORCETrainer,
    ...     PPOConfig,
    ...     PPOTrainer,
    ... )
    >>> config = PPOConfig(clip_param=0.2, gae_lambda=0.95)
"""

from artifex.generative_models.training.rl.adapters import (
    SequencePolicyAdapter,
    SequenceValueHeadAdapter,
)
from artifex.generative_models.training.rl.backends import (
    LocalSequenceGenerationBackend,
    LocalSequenceRolloutBackend,
)
from artifex.generative_models.training.rl.configs import (
    DPOConfig,
    GRPOConfig,
    PPOConfig,
    REINFORCEConfig,
)
from artifex.generative_models.training.rl.dpo import DPOTrainer
from artifex.generative_models.training.rl.grpo import GRPOTrainer
from artifex.generative_models.training.rl.ppo import PPOTrainer
from artifex.generative_models.training.rl.protocols import (
    IterativeGenerationBackend,
    IterativePolicyAdapter,
    LogProbScorer,
    PolicyAdapter,
    RewardModel,
    SequenceGenerationBackend,
    SequenceRolloutBackend,
    SequenceRolloutPolicyAdapter,
    SequenceRolloutValueAdapter,
    ValueAdapter,
)
from artifex.generative_models.training.rl.reinforce import REINFORCETrainer
from artifex.generative_models.training.rl.rewards import (
    ClippedReward,
    CompositeReward,
    ConstantReward,
    RewardFunction,
    ScaledReward,
    ThresholdReward,
)
from artifex.generative_models.training.rl.types import (
    GeneratedBatch,
    GeneratedSequenceBatch,
    GenerationRequest,
    GroupRolloutBatch,
    IterativeGenerationBatch,
    IterativeGenerationRequest,
    PreferenceBatch,
    SequenceGenerationRequest,
    SequenceRolloutBatch,
    TrajectoryBatch,
)
from artifex.generative_models.training.rl.utils import (
    compute_clipped_surrogate_loss,
    compute_discounted_returns,
    compute_gae_advantages,
    compute_kl_divergence,
    compute_masked_clipped_surrogate_loss,
    compute_masked_kl_divergence,
    compute_masked_policy_entropy,
    compute_policy_entropy,
    masked_mean,
    masked_normalize,
    normalize_advantages,
)


__all__ = [
    # Configurations
    "REINFORCEConfig",
    "PPOConfig",
    "GRPOConfig",
    "DPOConfig",
    # Trainers
    "REINFORCETrainer",
    "PPOTrainer",
    "GRPOTrainer",
    "DPOTrainer",
    "SequencePolicyAdapter",
    "SequenceValueHeadAdapter",
    "LocalSequenceGenerationBackend",
    "LocalSequenceRolloutBackend",
    # Reward functions
    "RewardFunction",
    "ConstantReward",
    "CompositeReward",
    "ThresholdReward",
    "ScaledReward",
    "ClippedReward",
    # Typed RL contracts
    "TrajectoryBatch",
    "GenerationRequest",
    "GeneratedBatch",
    "GeneratedSequenceBatch",
    "SequenceRolloutBatch",
    "PreferenceBatch",
    "GroupRolloutBatch",
    "SequenceGenerationRequest",
    "IterativeGenerationRequest",
    "IterativeGenerationBatch",
    "PolicyAdapter",
    "LogProbScorer",
    "SequenceGenerationBackend",
    "IterativeGenerationBackend",
    "IterativePolicyAdapter",
    "SequenceRolloutBackend",
    "SequenceRolloutPolicyAdapter",
    "SequenceRolloutValueAdapter",
    "ValueAdapter",
    "RewardModel",
    # Utility functions
    "compute_discounted_returns",
    "compute_gae_advantages",
    "normalize_advantages",
    "masked_mean",
    "masked_normalize",
    "compute_policy_entropy",
    "compute_masked_policy_entropy",
    "compute_kl_divergence",
    "compute_masked_kl_divergence",
    "compute_clipped_surrogate_loss",
    "compute_masked_clipped_surrogate_loss",
]
