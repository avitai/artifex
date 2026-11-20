"""Cases for the normalizing-flow family, ``artifex.generative_models.models.flow``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import BATCH, normal


_FLOW = "artifex.generative_models.models.flow"
_FLOW_DIM, _CONDITION_DIM, _ATOMS = 4, 3, 6
_GLOW_IMAGE = (4, 4, 4)


def _both_ways(module: nnx.Module, x: jax.Array) -> PyTree:
    """A flow's two directions: density (forward) and sampling (inverse), each with its log-det."""
    return module.forward(x), module.inverse(x)


def _conditioned_both_ways(module: nnx.Module, x: jax.Array, condition: jax.Array) -> PyTree:
    return module.forward(x, condition=condition), module.inverse(x, condition=condition)


def _model_both_ways(module: nnx.Module, x: jax.Array) -> PyTree:
    return module(x), module.inverse(x)


def _coupling_network() -> object:
    from artifex.generative_models.core.configuration.flow_config import (  # noqa: PLC0415
        CouplingNetworkConfig,
    )

    return CouplingNetworkConfig(name="coupling", hidden_dims=(8, 8), activation="silu")


def _flow_config(config_class: type, **fields: object) -> object:
    return config_class(
        name="flow",
        coupling_network=_coupling_network(),
        input_dim=_FLOW_DIM,
        latent_dim=_FLOW_DIM,
        **fields,
    )


def _mask() -> jax.Array:
    return jnp.array([1.0, 0.0, 1.0, 0.0])


def _features_and_condition(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return (
        jax.random.normal(k1, (BATCH, _FLOW_DIM)),
        jax.random.normal(k2, (BATCH, _CONDITION_DIM)),
    )


def _molecule(key: jax.Array, *, features: int | None = None) -> tuple[jax.Array, ...]:
    """Coordinates, then atom features or atom types, then a mask whose last two atoms are padding."""
    k1, k2 = jax.random.split(key)
    coordinates = jax.random.normal(k1, (BATCH, _ATOMS, 3))
    if features is None:
        second = jax.random.randint(k2, (BATCH, _ATOMS), 0, 5)
    else:
        second = jax.random.normal(k2, (BATCH, _ATOMS, features))
    mask = jnp.broadcast_to(jnp.arange(_ATOMS) < _ATOMS - 2, (BATCH, _ATOMS))
    return coordinates, second, mask


def _spline_inputs(key: jax.Array) -> tuple[jax.Array, ...]:
    bins = 4
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return (
        jax.random.uniform(k1, (BATCH, 2), minval=-2.0, maxval=2.0),
        jax.random.normal(k2, (BATCH, 2, bins)),
        jax.random.normal(k3, (BATCH, 2, bins)),
        jax.random.normal(k4, (BATCH, 2, bins + 1)),
    )


def cases() -> dict[str, Case]:
    """Return the flow cases, by the qualified name of the class each builds."""

    from artifex.generative_models.core.configuration.flow_config import (  # noqa: PLC0415
        ConditionalFlowConfig,
        GlowConfig,
        IAFConfig,
        MAFConfig,
        NeuralSplineConfig,
        RealNVPConfig,
    )
    from artifex.generative_models.models.flow import (  # noqa: PLC0415
        conditional,
        glow,
        iaf,
        made,
        maf,
        neural_spline,
        real_nvp,
        se3_molecular,
    )

    spline = neural_spline
    se3 = se3_molecular
    channels = _GLOW_IMAGE[-1]
    return {
        f"{_FLOW}.real_nvp.CouplingLayer": Case(
            build=lambda r: real_nvp.CouplingLayer(_mask(), [8, 8], rngs=r),
            inputs=normal(_FLOW_DIM),
            call=_both_ways,
        ),
        f"{_FLOW}.real_nvp.RealNVP": Case(
            build=lambda r: real_nvp.RealNVP(
                _flow_config(RealNVPConfig, num_coupling_layers=2), rngs=r
            ),
            inputs=normal(_FLOW_DIM),
            call=_model_both_ways,
        ),
        f"{_FLOW}.conditional.ConditionalCouplingLayer": Case(
            build=lambda r: conditional.ConditionalCouplingLayer(
                _mask(), [8, 8], _CONDITION_DIM, rngs=r
            ),
            inputs=_features_and_condition,
            call=_conditioned_both_ways,
        ),
        f"{_FLOW}.conditional.ConditionalNormalizingFlow": Case(
            build=lambda r: conditional.ConditionalNormalizingFlow(
                _flow_config(
                    ConditionalFlowConfig, condition_dim=_CONDITION_DIM, num_coupling_layers=2
                ),
                rngs=r,
            ),
            inputs=_features_and_condition,
            call=_conditioned_both_ways,
        ),
        f"{_FLOW}.conditional.ConditionalRealNVP": Case(
            build=lambda r: conditional.ConditionalRealNVP(
                _flow_config(
                    ConditionalFlowConfig, condition_dim=_CONDITION_DIM, num_coupling_layers=2
                ),
                rngs=r,
            ),
            inputs=_features_and_condition,
            call=_conditioned_both_ways,
        ),
        f"{_FLOW}.glow.ActNormLayer": Case(
            build=lambda r: glow.ActNormLayer(channels, rngs=r),
            inputs=normal(*_GLOW_IMAGE),
            call=_both_ways,
        ),
        f"{_FLOW}.glow.InvertibleConv1x1": Case(
            build=lambda r: glow.InvertibleConv1x1(channels, rngs=r),
            inputs=normal(*_GLOW_IMAGE),
            call=_both_ways,
        ),
        f"{_FLOW}.glow.AffineCouplingLayer": Case(
            build=lambda r: glow.AffineCouplingLayer(channels, [8], rngs=r),
            inputs=normal(*_GLOW_IMAGE),
            call=_both_ways,
        ),
        f"{_FLOW}.glow.GlowBlock": Case(
            build=lambda r: glow.GlowBlock(channels, [8], rngs=r),
            inputs=normal(*_GLOW_IMAGE),
            call=_both_ways,
        ),
        f"{_FLOW}.glow.Glow": Case(
            build=lambda r: glow.Glow(
                GlowConfig(
                    name="glow",
                    coupling_network=_coupling_network(),
                    input_dim=int(jnp.prod(jnp.array(_GLOW_IMAGE))),
                    latent_dim=int(jnp.prod(jnp.array(_GLOW_IMAGE))),
                    image_shape=_GLOW_IMAGE,
                    blocks_per_scale=2,
                ),
                rngs=r,
            ),
            inputs=normal(*_GLOW_IMAGE),
            call=_model_both_ways,
        ),
        f"{_FLOW}.made.MADE": Case(
            build=lambda r: made.MADE(_FLOW_DIM, [8, 8], rngs=r, activation="tanh"),
            inputs=normal(_FLOW_DIM),
        ),
        f"{_FLOW}.maf.MAFLayer": Case(
            build=lambda r: maf.MAFLayer(_FLOW_DIM, [8, 8], rngs=r),
            inputs=normal(_FLOW_DIM),
            call=_both_ways,
        ),
        f"{_FLOW}.maf.MAF": Case(
            build=lambda r: maf.MAF(_flow_config(MAFConfig, num_layers=2), rngs=r),
            inputs=normal(_FLOW_DIM),
            call=_model_both_ways,
        ),
        f"{_FLOW}.iaf.IAFLayer": Case(
            build=lambda r: iaf.IAFLayer(_FLOW_DIM, [8, 8], rngs=r),
            inputs=normal(_FLOW_DIM),
            call=_both_ways,
        ),
        f"{_FLOW}.iaf.IAF": Case(
            build=lambda r: iaf.IAF(_flow_config(IAFConfig, num_layers=2), rngs=r),
            inputs=normal(_FLOW_DIM),
            call=_model_both_ways,
        ),
        f"{_FLOW}.neural_spline.RationalQuadraticSplineTransform": Case(
            build=lambda r: spline.RationalQuadraticSplineTransform(num_bins=4, rngs=r),
            inputs=_spline_inputs,
            # apply_spline takes constrained parameters, as SplineCouplingLayer passes them.
            call=lambda module, x, widths, heights, derivatives: (
                module.apply_spline(x, *module._constrain_parameters(widths, heights, derivatives)),
                module.apply_spline(
                    x, *module._constrain_parameters(widths, heights, derivatives), inverse=True
                ),
            ),
            is_differentiable=False,
        ),
        f"{_FLOW}.neural_spline.SplineCouplingLayer": Case(
            build=lambda r: spline.SplineCouplingLayer(_mask(), [8, 8], num_bins=4, rngs=r),
            inputs=normal(_FLOW_DIM),
            call=_both_ways,
        ),
        f"{_FLOW}.neural_spline.NeuralSplineFlow": Case(
            build=lambda r: spline.NeuralSplineFlow(
                _flow_config(NeuralSplineConfig, num_layers=2, num_bins=4), rngs=r
            ),
            inputs=normal(_FLOW_DIM),
            call=_model_both_ways,
        ),
        f"{_FLOW}.se3_molecular.SE3EquivariantLayer": Case(
            build=lambda r: se3.SE3EquivariantLayer(8, 8, _ATOMS, rngs=r),
            inputs=lambda key: _molecule(key, features=8),
        ),
        f"{_FLOW}.se3_molecular.SE3CouplingLayer": Case(
            build=lambda r: se3.SE3CouplingLayer(8, _ATOMS, 5, rngs=r),
            inputs=_molecule,
            call=lambda module, coordinates, types, mask: (
                module(coordinates, types, mask),
                module(coordinates, types, mask, reverse=True),
            ),
        ),
        f"{_FLOW}.se3_molecular.SE3MolecularFlow": Case(
            build=lambda r: se3.SE3MolecularFlow(
                hidden_dim=8, num_layers=1, num_coupling_layers=2, max_atoms=_ATOMS, rngs=r
            ),
            inputs=_molecule,
            call=lambda module, coordinates, types, mask: module.log_prob(coordinates, types, mask),
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_FLOW}.base.FlowLayer": "base class whose forward and inverse raise NotImplementedError",
    f"{_FLOW}.base.NormalizingFlow": (
        "base class holding an empty layer list; each flow model's case runs its forward,"
        " inverse and log-density"
    ),
}
