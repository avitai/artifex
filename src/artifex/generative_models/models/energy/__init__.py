"""Energy-based models module.

This module provides implementations of energy-based models (EBMs) using Flax NNX.
EBMs learn data distributions by modeling an energy function E(x) where p(x) ∝ exp(-E(x)).
"""

from artifex.generative_models.models.energy.base import (
    CNNEnergyFunction,
    EnergyBasedModel,
    EnergyFunction,
    MLPEnergyFunction,
)
from artifex.generative_models.models.energy.ebm import (
    create_cifar_ebm,
    create_mnist_ebm,
    create_simple_ebm,
    DeepCNNEnergyFunction,
    DeepEBM,
    EBM,
    EnergyBlock,
)
from artifex.generative_models.models.energy.mcmc import (
    improved_langevin_dynamics,
    langevin_dynamics,
    langevin_dynamics_with_trajectory,
    persistent_contrastive_divergence,
    SampleBuffer,
)


__all__ = [
    # Base classes
    "EnergyFunction",
    "EnergyBasedModel",
    "MLPEnergyFunction",
    "CNNEnergyFunction",
    # Main EBM implementations
    "EBM",
    "DeepEBM",
    "DeepCNNEnergyFunction",
    "EnergyBlock",
    # MCMC utilities
    "SampleBuffer",
    "langevin_dynamics",
    "langevin_dynamics_with_trajectory",
    "improved_langevin_dynamics",
    "persistent_contrastive_divergence",
    # Factory functions
    "create_mnist_ebm",
    "create_cifar_ebm",
    "create_simple_ebm",
]
