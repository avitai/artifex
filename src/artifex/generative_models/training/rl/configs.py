"""Configuration classes for RL training algorithms.

This module provides dataclass configurations for reinforcement learning trainers:
- REINFORCEConfig: Basic policy gradient with variance reduction
- PPOConfig: Proximal Policy Optimization with GAE
- GRPOConfig: Group Relative Policy Optimization (critic-free, DeepSeek-style)
- DPOConfig: Direct Preference Optimization with SimPO support
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class REINFORCEConfig:
    """Configuration for REINFORCE policy gradient algorithm.

    Attributes:
        gamma: Discount factor for computing returns. Default 0.99.
        normalize_returns: Whether to normalize returns for variance reduction.
        entropy_coeff: Coefficient for entropy bonus to encourage exploration.
    """

    gamma: float = 0.99
    normalize_returns: bool = True
    entropy_coeff: float = 0.01


@dataclass(slots=True)
class PPOConfig:
    """Configuration for Proximal Policy Optimization.

    Implements PPO with clipped surrogate objective and GAE.

    Attributes:
        gamma: Discount factor for computing returns. Default 0.99.
        gae_lambda: Lambda for Generalized Advantage Estimation. Default 0.95.
        clip_param: Clipping parameter for surrogate objective. Default 0.2.
        vf_coeff: Coefficient for value function loss. Default 0.5.
        entropy_coeff: Coefficient for entropy bonus. Default 0.01.
        max_grad_norm: Maximum gradient norm for clipping. Default 0.5.
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    vf_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5


@dataclass(slots=True)
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization.

    GRPO is a critic-free RL algorithm from DeepSeek-R1 that:
    - Generates multiple completions per prompt (num_generations)
    - Normalizes advantages within each group
    - Uses PPO-style clipping
    - Saves ~50% memory by eliminating the value network

    Attributes:
        num_generations: Number of completions to generate per prompt. Default 4.
        clip_param: Clipping parameter for surrogate objective. Default 0.2.
        beta: KL penalty coefficient for regularization. Default 0.01.
        entropy_coeff: Coefficient for entropy bonus. Default 0.01.
        gamma: Discount factor (used if computing returns). Default 0.99.
    """

    num_generations: int = 4
    clip_param: float = 0.2
    beta: float = 0.01
    entropy_coeff: float = 0.01
    gamma: float = 0.99


@dataclass(slots=True)
class DPOConfig:
    """Configuration for Direct Preference Optimization.

    DPO enables preference learning without an explicit reward model.
    SimPO mode (reference_free=True) eliminates the need for a reference model.

    Attributes:
        beta: Reward scaling parameter. Higher values = stronger preference.
            Default 0.1.
        label_smoothing: Label smoothing for preference loss. Default 0.0.
        reference_free: Whether to use SimPO-style reference-free training.
            When True, no reference model is needed. Default False.
    """

    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False
