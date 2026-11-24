"""Cases for the core noise schedules and loss modules.

``artifex.generative_models.core.noise_schedule`` and ``core.losses`` hold modules with no
``nnx.Param`` of their own: a schedule's tables are fixed by its configuration, and a loss
compares its inputs. Their cases check state, transforms and modes, and say where an output is a
batch reduction rather than one value per row.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import BATCH, normal


_CORE = "artifex.generative_models.core"
_STEPS, _FEATURES, _VERTICES = 50, 5, 4
_FACES = ((0, 1, 2), (0, 2, 3), (1, 2, 3))


def _signal_noise_and_steps(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2, k3 = jax.random.split(key, 3)
    return (
        jax.random.normal(k1, (BATCH, _FEATURES)),
        jax.random.normal(k2, (BATCH, _FEATURES)),
        jax.random.randint(k3, (BATCH,), 0, _STEPS),
    )


def _forward_and_posterior(
    module: nnx.Module, x_start: jax.Array, noise: jax.Array, t: jax.Array
) -> PyTree:
    """Noise a clean signal to step t, recover it, and take the reverse-step posterior."""
    x_t = module.q_sample(x_start, t, noise)
    recovered = module.predict_start_from_noise(x_t, t, noise)
    return x_t, recovered, module.q_posterior_mean_variance(recovered, x_t, t)


def _meshes(key: jax.Array) -> tuple[jax.Array, ...]:
    """Predicted and target vertices and normals, one mesh per row."""
    keys = jax.random.split(key, 4)
    return tuple(jax.random.normal(k, (BATCH, _VERTICES, 3)) for k in keys)


def _call_mesh_loss(
    module: nnx.Module,
    pred_vertices: jax.Array,
    pred_normals: jax.Array,
    target_vertices: jax.Array,
    target_normals: jax.Array,
) -> PyTree:
    faces = jnp.asarray(_FACES)
    return module((pred_vertices, faces, pred_normals), (target_vertices, faces, target_normals))


class _Features(nnx.Module):
    """A two-layer convolutional feature extractor for ``PerceptualLoss``, standing in for VGG."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.first = nnx.Conv(3, 4, kernel_size=(3, 3), rngs=rngs)
        self.second = nnx.Conv(4, 4, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, images: jax.Array) -> dict[str, jax.Array]:
        first = nnx.silu(self.first(images))
        return {"first": first, "second": nnx.silu(self.second(first))}


def cases() -> dict[str, Case]:
    """Return the schedule and loss cases, by the qualified name of the class each builds."""

    from artifex.generative_models.core import noise_schedule  # noqa: PLC0415
    from artifex.generative_models.core.configuration.diffusion_config import (  # noqa: PLC0415
        NoiseScheduleConfig,
    )
    from artifex.generative_models.core.losses import (  # noqa: PLC0415
        geometric,
        perceptual,
        regularization,
    )

    config = NoiseScheduleConfig(name="schedule", num_timesteps=_STEPS)
    schedules = {
        "LinearNoiseSchedule": noise_schedule.LinearNoiseSchedule,
        "CosineNoiseSchedule": noise_schedule.CosineNoiseSchedule,
        "QuadraticNoiseSchedule": noise_schedule.QuadraticNoiseSchedule,
        "SqrtNoiseSchedule": noise_schedule.SqrtNoiseSchedule,
    }
    image = (6, 6, 3)
    return {
        **{
            f"{_CORE}.noise_schedule.{name}": Case(
                build=lambda r, schedule=schedule: schedule(config),
                inputs=_signal_noise_and_steps,
                call=_forward_and_posterior,
                is_differentiable=False,
            )
            for name, schedule in schedules.items()
        },
        f"{_CORE}.losses.geometric.MeshLoss": Case(
            build=lambda r: geometric.MeshLoss(laplacian_weight=0.1),
            inputs=_meshes,
            call=_call_mesh_loss,
            is_differentiable=False,
            # A mean over the batch.
            is_rowwise=False,
        ),
        f"{_CORE}.losses.perceptual.PerceptualLoss": Case(
            build=lambda r: perceptual.PerceptualLoss(
                _Features(rngs=r), content_weight=1.0, style_weight=0.5
            ),
            inputs=lambda key: (
                jax.random.normal(key, (BATCH, *image)),
                jax.random.normal(jax.random.fold_in(key, 1), (BATCH, *image)),
            ),
            # A mean over the batch.
            is_rowwise=False,
        ),
        f"{_CORE}.losses.regularization.SpectralNormRegularization": Case(
            build=lambda r: regularization.SpectralNormRegularization(n_power_iterations=2),
            # The input is one weight matrix, whose rows are not a batch.
            inputs=normal(_FEATURES),
            is_differentiable=False,
            is_rowwise=False,
        ),
        f"{_CORE}.losses.regularization.DropoutRegularization": Case(
            build=lambda r: regularization.DropoutRegularization(rate=0.5),
            inputs=normal(_FEATURES),
            is_differentiable=False,
            # A single penalty for the batch.
            is_rowwise=False,
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_CORE}.protocols.metrics.MetricBase": (
        "base class whose compute, which __call__ runs, raises NotImplementedError; each"
        " metric's case runs it"
    ),
}
