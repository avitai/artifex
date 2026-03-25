"""Energy function factory for Energy-Based Models.

Creates energy functions from configuration objects, keeping the
configuration module free of concrete model imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

from artifex.generative_models.core.base import get_activation_function
from artifex.generative_models.core.configuration.energy_config import (
    EnergyNetworkConfig,
    VALID_NETWORK_TYPES,
)


if TYPE_CHECKING:
    from artifex.generative_models.models.energy.base import EnergyFunction


def create_energy_function(
    config: EnergyNetworkConfig,
    *,
    input_dim: int | None = None,
    input_channels: int | None = None,
    rngs: nnx.Rngs,
) -> EnergyFunction:
    """Create an energy function from configuration.

    This factory function creates the appropriate energy function type
    (MLPEnergyFunction or CNNEnergyFunction) based on the network_type
    in the configuration.

    Args:
        config: EnergyNetworkConfig with network_type discriminator
        input_dim: Input dimension for MLP (required if network_type="mlp")
        input_channels: Input channels for CNN (required if network_type="cnn")
        rngs: Random number generators for initialization

    Returns:
        Initialized energy function (MLPEnergyFunction or CNNEnergyFunction)

    Raises:
        ValueError: If required parameters are missing for the network type
        ValueError: If network_type is not supported
    """
    from artifex.generative_models.models.energy.base import (
        CNNEnergyFunction,
        MLPEnergyFunction,
    )

    activation = get_activation_function(config.activation)

    match config.network_type:
        case "mlp":
            if input_dim is None:
                raise ValueError("input_dim is required for MLP energy function")

            return MLPEnergyFunction(
                hidden_dims=list(config.hidden_dims),
                input_dim=input_dim,
                activation=activation,
                use_bias=config.use_bias,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )

        case "cnn":
            if input_channels is None:
                raise ValueError("input_channels is required for CNN energy function")

            return CNNEnergyFunction(
                hidden_dims=list(config.hidden_dims),
                input_channels=input_channels,
                activation=activation,
                use_bias=config.use_bias,
                rngs=rngs,
            )

        case _:
            raise ValueError(
                f"Unsupported network_type: {config.network_type}. "
                f"Expected one of: {VALID_NETWORK_TYPES}"
            )
