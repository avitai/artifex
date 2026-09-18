"""A checkpoint written by artifex 0.1.10 (substrax's format 2) restores and upgrades.

The fixture is generated, not committed: ``scripts/make_format2_trainer_fixture.py`` writes
it with the releases that produced it. Run, in isolation::

    uv run --no-project --with-requirements scripts/format2_fixture_requirements.txt \\
        python scripts/make_format2_trainer_fixture.py tests/artifex/fixtures/format2
"""

from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore, upgrade_checkpoints

from artifex.generative_models.core.configuration import OptimizerConfig, TrainingConfig
from artifex.generative_models.training import TRAINER_FORMAT2
from artifex.generative_models.training.trainer import Trainer


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "format2" / "trainer"
FIXTURE_STEP = 7
GENERATE = (
    "uv run --no-project --with-requirements scripts/format2_fixture_requirements.txt "
    "python scripts/make_format2_trainer_fixture.py tests/artifex/fixtures/format2"
)

if not FIXTURE_ROOT.is_dir():
    raise RuntimeError(
        f"the format-2 trainer checkpoint is missing under {FIXTURE_ROOT}: {GENERATE}"
    )


def objective(model, batch, rng, step):
    """The objective the fixture's trainer used: a squared error."""
    del rng, step
    loss = jnp.mean(jnp.square(model(batch["x"]) - batch["y"]))
    return loss, {"loss": loss}


def build_trainer(checkpoint_dir: Path) -> Trainer:
    """The fixture's trainer, built the way ``make_format2_trainer_fixture.py`` built it."""
    return Trainer(
        nnx.Linear(2, 1, rngs=nnx.Rngs(3)),
        TrainingConfig(
            name="fixture",
            optimizer=OptimizerConfig(name="adam", optimizer_type="adam", learning_rate=1e-3),
            checkpoint_dir=checkpoint_dir,
        ),
        loss_fn=objective,
    )


def test_the_layout_names_the_optimizer_item():
    items = TRAINER_FORMAT2.items_of({"model": "m", "opt_state": "o", "rng": "r", "extensions": {}})

    assert items == {"model": "m", "optimizer": "o", "rng": "r", "extensions": {}}
    assert TRAINER_FORMAT2.template_of(items) == {
        "model": "m",
        "opt_state": "o",
        "rng": "r",
        "extensions": {},
    }


def test_load_checkpoint_reads_the_old_root():
    trainer = build_trainer(FIXTURE_ROOT)
    before = trainer.model.kernel[...].copy()

    trainer.load_checkpoint(step=FIXTURE_STEP)

    assert trainer.step == FIXTURE_STEP
    assert not jnp.array_equal(trainer.model.kernel[...], before)
    # The restored optimizer state carries on: one more step runs on it.
    trainer.train_step({"x": jnp.ones((2, 2)), "y": jnp.ones((2, 1))})
    assert trainer.step == FIXTURE_STEP + 1


def test_upgrade_writes_a_format_3_root(tmp_path):
    steps = upgrade_checkpoints(FIXTURE_ROOT, tmp_path / "v3", legacy_layout=TRAINER_FORMAT2)

    assert steps == [FIXTURE_STEP]
    with OrbaxCheckpointStore(tmp_path / "v3") as store:
        metadata = store.read_metadata(FIXTURE_STEP)
    assert set(metadata.items) == {"model", "optimizer", "rng", "extensions"}
    trainer = build_trainer(tmp_path / "v3")
    trainer.load_checkpoint()
    assert trainer.step == FIXTURE_STEP
