"""Contracts for the shared training surface."""

from pathlib import Path

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    OptimizerConfig,
    TrainingConfig,
)
from artifex.generative_models.training.trainer import Trainer


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _SimpleTrainableModule(nnx.Module):
    """Minimal module for generic Trainer contract tests."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.linear = nnx.Linear(4, 4, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        name="contract-training",
        batch_size=4,
        num_epochs=1,
        optimizer=OptimizerConfig(
            name="adam",
            optimizer_type="adam",
            learning_rate=1e-3,
        ),
    )


def _explicit_loss_fn(model, batch, rng, step):
    del model, batch, rng, step
    loss = jnp.array(0.0)
    return loss, {"loss": loss}


def test_generic_trainer_requires_explicit_loss_fn() -> None:
    """The generic Trainer should execute an explicit objective only."""
    model = _SimpleTrainableModule(rngs=nnx.Rngs(0))

    with pytest.raises(TypeError, match="loss_fn"):
        Trainer(
            model=model,
            training_config=_training_config(),
        )

    trainer = Trainer(
        model=model,
        training_config=_training_config(),
        loss_fn=_explicit_loss_fn,
    )
    assert trainer.loss_fn is _explicit_loss_fn


def test_generic_trainer_passes_dynamic_step_to_loss_fn() -> None:
    """The generic Trainer should provide the current training step to the objective."""
    model = _SimpleTrainableModule(rngs=nnx.Rngs(0))
    observed_steps: list[int] = []

    def loss_fn(model, batch, rng, step):
        del model, batch, rng
        observed_steps.append(int(step))
        loss = jnp.array(0.0)
        return loss, {"loss": loss}

    trainer = Trainer(
        model=model,
        training_config=_training_config(),
        loss_fn=loss_fn,
    )

    batch = {"x": jnp.zeros((4, 4), dtype=jnp.float32)}
    trainer.train_step(batch)
    trainer.train_step(batch)

    assert observed_steps == [0, 1]


def test_generic_trainer_docs_require_explicit_loss_fn_in_examples() -> None:
    """Generic Trainer docs should show explicit objectives in construction examples."""
    docs_to_check = [
        PROJECT_ROOT / "docs" / "api" / "training" / "trainer.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "overview.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "training-guide.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "logging.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "profiling.md",
        PROJECT_ROOT / "docs" / "examples" / "framework" / "framework-features-demo.md",
        PROJECT_ROOT / "docs" / "community" / "faq.md",
        PROJECT_ROOT / "docs" / "models" / "index.md",
        PROJECT_ROOT / "docs" / "user-guide" / "advanced" / "architectures.md",
        PROJECT_ROOT / "docs" / "user-guide" / "advanced" / "distributed.md",
        PROJECT_ROOT / "docs" / "user-guide" / "integrations" / "tensorboard.md",
        PROJECT_ROOT / "docs" / "user-guide" / "integrations" / "wandb.md",
    ]

    constructor_tokens = (
        "trainer = Trainer(",
        "trainer = CustomTrainer(",
        "distributed_trainer = DistributedTrainer(",
        "trainer = TensorBoardTrainer(",
        "trainer = ArtifexWandBTrainer(",
    )

    for path in docs_to_check:
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if any(token in line for token in constructor_tokens):
                window = "\n".join(lines[index : index + 12])
                assert "loss_fn=" in window, (
                    f"{path} should pass an explicit loss_fn when constructing a generic trainer"
                )


def test_trainer_api_reference_matches_the_real_trainer_surface() -> None:
    """The main Trainer API doc should not teach removed or invented methods."""
    path = PROJECT_ROOT / "docs" / "api" / "training" / "trainer.md"
    text = path.read_text(encoding="utf-8")

    forbidden_tokens = (
        "TrainingState",
        "def create(",
        "def generate_samples(",
        "### generate_samples",
        "### create",
    )
    required_tokens = (
        "loss_fn(model, batch, rng, step)",
        "### save_checkpoint",
        "### load_checkpoint",
        "### train_step",
        "### validate_step",
        "### train_epoch",
        "### train",
        "### evaluate",
    )

    for token in forbidden_tokens:
        assert token not in text, f"{path} still contains stale Trainer API text: {token}"

    for token in required_tokens:
        assert token in text, f"{path} should document the real Trainer surface: {token}"


def test_loop_docs_describe_step_aware_objectives_without_universal_trainer_claims() -> None:
    """Loop docs should teach the explicit step-aware objective contract only."""
    docs_to_check = [
        PROJECT_ROOT
        / "src"
        / "artifex"
        / "generative_models"
        / "training"
        / "loops"
        / "__init__.py",
        PROJECT_ROOT / "src" / "artifex" / "generative_models" / "training" / "loops" / "staged.py",
        PROJECT_ROOT
        / "src"
        / "artifex"
        / "generative_models"
        / "training"
        / "loops"
        / "streaming.py",
        PROJECT_ROOT / "docs" / "getting-started" / "quickstart.md",
    ]

    forbidden_tokens = (
        "ANY trainer",
        "all 6 trainers",
        "All existing trainers implement create_loss_fn()",
        "Compatible with all existing trainers",
        "works with ANY trainer",
        "create_loss_fn(step=",
    )
    required_tokens = (
        "(model, batch, rng, step)",
        "step-aware",
    )

    for path in docs_to_check:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, (
                f"{path} still contains stale training-loop contract text: {token}"
            )
        assert any(token in text for token in required_tokens), (
            f"{path} should describe the explicit step-aware loop objective contract"
        )


def test_training_docs_do_not_keep_stale_duplicate_trainer_reference_page() -> None:
    """The stale duplicate training/trainer.md page should stay deleted."""
    assert not (PROJECT_ROOT / "docs" / "training" / "trainer.md").exists()


def test_checkpoint_docs_match_the_orbax_callback_contract() -> None:
    """Checkpoint docs should only describe the supported Orbax-backed surface."""
    docs_to_check = [
        PROJECT_ROOT / "docs" / "training" / "checkpoint.md",
        PROJECT_ROOT / "docs" / "training" / "base.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "training-guide.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "logging.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "profiling.md",
    ]

    forbidden_tokens = (
        "filename=",
        "save_last",
        "save_weights_only",
        "best_checkpoint_path",
        "saved_checkpoints",
    )
    required_tokens = (
        "best_checkpoint_step",
        "Orbax",
    )

    for path in docs_to_check:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, (
                f"{path} still contains stale checkpoint/callback API text: {token}"
            )
        if path.name == "checkpoint.md":
            assert any(token in text for token in required_tokens), (
                f"{path} should describe the step-based Orbax checkpoint contract"
            )
        if path.name == "base.md":
            assert "append(" not in text, (
                f"{path} should document CallbackList.add(...) instead of append(...)"
            )


def test_callback_docs_use_trainer_train_with_callbacklist_instead_of_fit() -> None:
    """Callback-facing docs should use the real Trainer constructor and train flow."""
    docs_to_check = [
        PROJECT_ROOT / "docs" / "training" / "checkpoint.md",
        PROJECT_ROOT / "docs" / "training" / "early_stopping.md",
        PROJECT_ROOT / "docs" / "training" / "logging.md",
        PROJECT_ROOT / "docs" / "training" / "profiling.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "logging.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "profiling.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "training-guide.md",
    ]

    for path in docs_to_check:
        text = path.read_text(encoding="utf-8")
        assert "trainer.fit(" not in text, f"{path} should use Trainer.train(...), not fit(...)"
        assert "CallbackList" in text, (
            f"{path} should show callback registration through CallbackList"
        )
        assert "trainer.train(" in text, (
            f"{path} should show the callback-aware Trainer.train(...) path"
        )


def test_family_trainer_docs_use_runtime_constructor_shapes() -> None:
    """Family-trainer docs should not pretend trainers own model/optimizer construction."""
    docs_to_check = [
        PROJECT_ROOT / "docs" / "training" / "index.md",
        PROJECT_ROOT / "docs" / "training" / "vae_trainer.md",
        PROJECT_ROOT / "docs" / "training" / "diffusion_trainer.md",
        PROJECT_ROOT / "docs" / "training" / "flow_trainer.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "logging.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "advanced-features.md",
        PROJECT_ROOT
        / "src"
        / "artifex"
        / "generative_models"
        / "training"
        / "trainers"
        / "__init__.py",
    ]

    forbidden_tokens = (
        "VAETrainer(model, optimizer",
        "DiffusionTrainer(model, optimizer",
        "FlowTrainer(model, optimizer",
        "EnergyTrainer(model, optimizer",
        "AutoregressiveTrainer(model, optimizer",
    )

    for path in docs_to_check:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, (
                f"{path} still contains stale family-trainer construction text: {token}"
            )


def test_training_overview_does_not_reintroduce_removed_runtime_types() -> None:
    """Shared training docs should not teach removed TrainingState or trainer generation APIs."""
    docs_to_check = [
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "overview.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "training-guide.md",
    ]

    forbidden_tokens = (
        "from artifex.generative_models.training.trainer import TrainingState",
        "state = TrainingState.create(",
        "trainer.generate_samples(",
    )

    for path in docs_to_check:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, (
                f"{path} still contains removed shared training API text: {token}"
            )


def test_training_overview_logging_examples_use_concrete_logger_contracts() -> None:
    """Training overview docs should use concrete loggers and explicit metric objects."""
    path = PROJECT_ROOT / "docs" / "user-guide" / "training" / "overview.md"
    text = path.read_text(encoding="utf-8")

    forbidden_tokens = (
        "logger = Logger(",
        "MetricsLogger(log_dir=",
        "get_default_metrics(",
    )
    required_tokens = (
        "FileLogger(",
        "MetricsLogger(",
        "logger=logger",
        "metrics={",
    )

    for token in forbidden_tokens:
        assert token not in text, f"{path} still teaches dead logging helper surface: {token}"

    for token in required_tokens:
        assert token in text, f"{path} should use the retained logging contract: {token}"


def test_training_docs_and_module_refs_drop_dead_callback_and_family_knobs() -> None:
    """Training docs should only advertise callback and trainer knobs the runtime consumes."""
    per_file_contracts = {
        PROJECT_ROOT / "docs" / "training" / "logging.md": {
            "forbidden": ("log_graph", "refresh_rate"),
            "required": ("TensorBoardLoggerConfig", "ProgressBarConfig", "show_metrics"),
        },
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "logging.md": {
            "forbidden": ("log_graph", "refresh_rate"),
            "required": ("TensorBoardLoggerConfig", "ProgressBarConfig", "show_metrics"),
        },
        PROJECT_ROOT / "docs" / "training" / "profiling.md": {
            "forbidden": ("trace_memory", "trace_python"),
            "required": ("ProfilingConfig", "start_step", "end_step"),
        },
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "profiling.md": {
            "forbidden": ("trace_memory", "trace_python"),
            "required": ("ProfilingConfig", "start_step", "end_step"),
        },
        PROJECT_ROOT / "docs" / "training" / "flow_trainer.md": {
            "forbidden": (
                "flow_type",
                "sigma_min",
                "use_ot",
                "ot_regularization",
                "ot_cfm",
                "rectified_flow",
            ),
            "required": ("FlowTrainingConfig", "time_sampling", "logit_normal_loc"),
        },
        PROJECT_ROOT / "docs" / "training" / "gan_trainer.md": {
            "forbidden": (
                "n_critic",
                "generator=",
                "discriminator=",
                "g_optimizer=",
                "d_optimizer=",
            ),
            "required": ("GANTrainingConfig", "GANTrainer(config)", "discriminator_step"),
        },
        PROJECT_ROOT / "docs" / "training" / "energy_trainer.md": {
            "forbidden": ("mcmc_sampler",),
            "required": ("EnergyTrainingConfig", "training_method", "Langevin"),
        },
        PROJECT_ROOT / "docs" / "training" / "index.md": {
            "forbidden": ("flow_type=", "ot_cfm", "rectified_flow"),
            "required": ("FlowTrainingConfig(time_sampling=",),
        },
        PROJECT_ROOT
        / "src"
        / "artifex"
        / "generative_models"
        / "training"
        / "trainers"
        / "__init__.py": {
            "forbidden": ("OT-CFM", "Rectified Flow"),
            "required": ("Flow: Flow matching models with configurable time sampling",),
        },
    }

    for path, contract in per_file_contracts.items():
        text = path.read_text(encoding="utf-8")
        for token in contract["forbidden"]:
            assert token not in text, f"{path} still advertises dead training surface text: {token}"
        for token in contract["required"]:
            assert token in text, f"{path} should document the retained training surface: {token}"
