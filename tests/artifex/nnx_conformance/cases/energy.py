"""Cases for the energy-based family, ``artifex.generative_models.models.energy``."""

from __future__ import annotations

from flax import nnx
from substrax.testing.nnx_conformance import Case

from tests.artifex.nnx_conformance.cases.inputs import normal


_ENERGY = "artifex.generative_models.models.energy"
_FEATURES, _CHANNELS = 5, 2
_IMAGE = (8, 8, _CHANNELS)


def _network(network_type: str, **fields: object) -> object:
    from artifex.generative_models.core.configuration.energy_config import (  # noqa: PLC0415
        EnergyNetworkConfig,
    )

    return EnergyNetworkConfig(
        name="energy", network_type=network_type, activation="silu", **fields
    )


def _sampling() -> dict[str, object]:
    from artifex.generative_models.core.configuration.energy_config import (  # noqa: PLC0415
        MCMCConfig,
        SampleBufferConfig,
    )

    return {
        "mcmc": MCMCConfig(name="mcmc"),
        "sample_buffer": SampleBufferConfig(name="buffer", capacity=16),
    }


def cases() -> dict[str, Case]:
    """Return the energy-based cases, by the qualified name of the class each builds."""

    from artifex.generative_models.core.configuration.energy_config import (  # noqa: PLC0415
        DeepEBMConfig,
        EBMConfig,
    )
    from artifex.generative_models.models.energy import base, ebm  # noqa: PLC0415

    def mlp_energy(r: nnx.Rngs) -> nnx.Module:
        return base.MLPEnergyFunction(
            [8, 6], input_dim=_FEATURES, activation=nnx.silu, dropout_rate=0.5, rngs=r
        )

    ebm_config = EBMConfig(
        name="ebm",
        input_dim=_FEATURES,
        energy_network=_network("mlp", hidden_dims=(8, 6), dropout_rate=0.5),
        **_sampling(),
    )
    deep_config = DeepEBMConfig(
        name="deep_ebm",
        input_shape=_IMAGE,
        energy_network=_network("cnn", hidden_dims=(8, 8)),
        **_sampling(),
    )
    return {
        f"{_ENERGY}.base.MLPEnergyFunction": Case(
            build=mlp_energy,
            inputs=normal(_FEATURES),
            has_dropout=True,
        ),
        f"{_ENERGY}.base.CNNEnergyFunction": Case(
            build=lambda r: base.CNNEnergyFunction([4, 6], input_channels=_CHANNELS, rngs=r),
            inputs=normal(*_IMAGE),
        ),
        f"{_ENERGY}.base.EnergyBasedModel": Case(
            build=lambda r: base.EnergyBasedModel(mlp_energy(r), rngs=r),
            inputs=normal(_FEATURES),
            has_dropout=True,
        ),
        f"{_ENERGY}.ebm.EnergyBlock": Case(
            build=lambda r: ebm.EnergyBlock(_CHANNELS, 4, rngs=r),
            inputs=normal(*_IMAGE),
        ),
        f"{_ENERGY}.ebm.DeepCNNEnergyFunction": Case(
            build=lambda r: ebm.DeepCNNEnergyFunction([8, 8], input_channels=_CHANNELS, rngs=r),
            inputs=normal(*_IMAGE),
        ),
        f"{_ENERGY}.ebm.EBM": Case(
            build=lambda r: ebm.EBM(ebm_config, rngs=r),
            inputs=normal(_FEATURES),
            has_dropout=True,
        ),
        f"{_ENERGY}.ebm.DeepEBM": Case(
            build=lambda r: ebm.DeepEBM(deep_config, rngs=r),
            inputs=normal(*_IMAGE),
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_ENERGY}.base.EnergyFunction": (
        "base class whose __call__ raises NotImplementedError; each subclass's case runs it"
    ),
}
