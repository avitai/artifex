r"""Write the format-2 trainer checkpoint the migration test restores.

artifex 0.1.10 saved ``Trainer.checkpoint_state()`` (``model``, ``opt_state``, ``rng`` and
``extensions``) as the one payload of substrax's format 2. artifex now writes format 3 and reads
such a root through ``TRAINER_FORMAT2``; this script writes one with the releases that
produced it, so the test reads a real one. Run it in isolation, never from the project venv::

    uv run --no-project --with "avitai-artifex==0.1.10" --with "substrax==0.1.9" \
        python scripts/make_format2_trainer_fixture.py tests/artifex/fixtures/format2

0.1.10 is the last release that wrote format 2 with the optimizer state tree the trainer
holds today: 0.1.9 moved the optimizer onto substrax's transformation, so a checkpoint from
0.1.8 or earlier carries a different tree and restores into no current trainer.

The fixture is a few kilobytes: a two-by-one linear model under adam after seven steps.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import OptimizerConfig, TrainingConfig
from artifex.generative_models.training.trainer import Trainer


STEPS = 7


def objective(model, batch, rng, step):
    """A squared error; the test that reads the fixture uses the same one."""
    del rng, step
    loss = jnp.mean(jnp.square(model(batch["x"]) - batch["y"]))
    return loss, {"loss": loss}


def main(argv: list[str] | None = None) -> int:
    """Write the fixture under the root named on the command line."""
    parser = argparse.ArgumentParser(
        description="Write the format-2 trainer checkpoint the migration test restores."
    )
    parser.add_argument("root", type=Path, help="Directory the fixture is written under")
    root = parser.parse_args(argv).root / "trainer"
    if root.exists():
        shutil.rmtree(root)

    trainer = Trainer(
        nnx.Linear(2, 1, rngs=nnx.Rngs(3)),
        TrainingConfig(
            name="fixture",
            optimizer=OptimizerConfig(name="adam", optimizer_type="adam", learning_rate=1e-3),
        ),
        loss_fn=objective,
        checkpoint_dir=str(root),
    )
    batch = {"x": jnp.ones((2, 2)), "y": jnp.ones((2, 1))}
    for _ in range(STEPS):
        trainer.train_step(batch)
    path = trainer.save_checkpoint()
    sys.stdout.write(f"trainer: step {trainer.step} at {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
