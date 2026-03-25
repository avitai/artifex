from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "training"

RETAINED_RUNTIME_PAGES: dict[str, dict[str, object]] = {
    "base.md": {
        "modules": ["artifex.generative_models.training.callbacks.base"],
        "sources": ["src/artifex/generative_models/training/callbacks/base.py"],
    },
    "checkpoint.md": {
        "modules": ["artifex.generative_models.training.callbacks.checkpoint"],
        "sources": ["src/artifex/generative_models/training/callbacks/checkpoint.py"],
    },
    "data_parallel.md": {
        "modules": ["artifex.generative_models.training.distributed.data_parallel"],
        "sources": ["src/artifex/generative_models/training/distributed/data_parallel.py"],
    },
    "device_placement.md": {
        "modules": ["artifex.generative_models.training.distributed.device_placement"],
        "sources": ["src/artifex/generative_models/training/distributed/device_placement.py"],
    },
    "diffusion_trainer.md": {
        "modules": ["artifex.generative_models.training.trainers.diffusion_trainer"],
        "sources": ["src/artifex/generative_models/training/trainers/diffusion_trainer.py"],
    },
    "distributed_metrics.md": {
        "modules": ["artifex.generative_models.training.distributed.metrics"],
        "sources": ["src/artifex/generative_models/training/distributed/metrics.py"],
    },
    "dpo.md": {
        "modules": ["artifex.generative_models.training.rl.dpo"],
        "sources": ["src/artifex/generative_models/training/rl/dpo.py"],
    },
    "early_stopping.md": {
        "modules": ["artifex.generative_models.training.callbacks.early_stopping"],
        "sources": ["src/artifex/generative_models/training/callbacks/early_stopping.py"],
    },
    "energy_trainer.md": {
        "modules": ["artifex.generative_models.training.trainers.energy_trainer"],
        "sources": ["src/artifex/generative_models/training/trainers/energy_trainer.py"],
    },
    "factory.md": {
        "modules": [
            "artifex.generative_models.training.optimizers.factory",
            "artifex.generative_models.training.schedulers.factory",
        ],
        "sources": [
            "src/artifex/generative_models/training/optimizers/factory.py",
            "src/artifex/generative_models/training/schedulers/factory.py",
        ],
    },
    "flow_trainer.md": {
        "modules": ["artifex.generative_models.training.trainers.flow_trainer"],
        "sources": ["src/artifex/generative_models/training/trainers/flow_trainer.py"],
    },
    "gan_trainer.md": {
        "modules": ["artifex.generative_models.training.trainers.gan_trainer"],
        "sources": ["src/artifex/generative_models/training/trainers/gan_trainer.py"],
    },
    "gradient_accumulation.md": {
        "modules": ["artifex.generative_models.training.gradient_accumulation"],
        "sources": ["src/artifex/generative_models/training/gradient_accumulation.py"],
    },
    "grpo.md": {
        "modules": ["artifex.generative_models.training.rl.grpo"],
        "sources": ["src/artifex/generative_models/training/rl/grpo.py"],
    },
    "logging.md": {
        "modules": ["artifex.generative_models.training.callbacks.logging"],
        "sources": ["src/artifex/generative_models/training/callbacks/logging.py"],
    },
    "mesh.md": {
        "modules": ["artifex.generative_models.training.distributed.mesh"],
        "sources": ["src/artifex/generative_models/training/distributed/mesh.py"],
    },
    "ppo.md": {
        "modules": ["artifex.generative_models.training.rl.ppo"],
        "sources": ["src/artifex/generative_models/training/rl/ppo.py"],
    },
    "profiling.md": {
        "modules": ["artifex.generative_models.training.callbacks.profiling"],
        "sources": ["src/artifex/generative_models/training/callbacks/profiling.py"],
    },
    "reinforce.md": {
        "modules": ["artifex.generative_models.training.rl.reinforce"],
        "sources": ["src/artifex/generative_models/training/rl/reinforce.py"],
    },
    "autoregressive_trainer.md": {
        "modules": ["artifex.generative_models.training.trainers.autoregressive_trainer"],
        "sources": ["src/artifex/generative_models/training/trainers/autoregressive_trainer.py"],
    },
    "utils.md": {
        "modules": ["artifex.generative_models.training.utils"],
        "sources": ["src/artifex/generative_models/training/utils.py"],
    },
    "vae_trainer.md": {
        "modules": ["artifex.generative_models.training.trainers.vae_trainer"],
        "sources": ["src/artifex/generative_models/training/trainers/vae_trainer.py"],
    },
}

COMING_SOON_PAGES = {
    "adafactor.md": {
        "planned_module": "artifex.generative_models.training.optimizers.adafactor",
        "current_owner": "artifex.generative_models.training.optimizers.factory",
    },
    "adamw.md": {
        "planned_module": "artifex.generative_models.training.optimizers.adamw",
        "current_owner": "artifex.generative_models.training.optimizers.factory",
    },
    "cosine.md": {
        "planned_module": "artifex.generative_models.training.schedulers.cosine",
        "current_owner": "artifex.generative_models.training.schedulers.factory",
    },
    "exponential.md": {
        "planned_module": "artifex.generative_models.training.schedulers.exponential",
        "current_owner": "artifex.generative_models.training.schedulers.factory",
    },
    "linear.md": {
        "planned_module": "artifex.generative_models.training.schedulers.linear",
        "current_owner": "artifex.generative_models.training.schedulers.factory",
    },
    "lion.md": {
        "planned_module": "artifex.generative_models.training.optimizers.lion",
        "current_owner": "artifex.generative_models.training.optimizers.factory",
    },
    "mixed_precision.md": {
        "planned_module": "artifex.generative_models.training.mixed_precision",
        "current_owner": "artifex.generative_models.training.gradient_accumulation",
    },
    "model_parallel.md": {
        "planned_module": "artifex.generative_models.training.distributed.model_parallel",
        "current_owner": "artifex.generative_models.training.distributed.mesh",
    },
    "optax_wrappers.md": {
        "planned_module": "artifex.generative_models.training.optimizers.optax_wrappers",
        "current_owner": "artifex.generative_models.training.optimizers.factory",
    },
    "scheduler.md": {
        "planned_module": "artifex.generative_models.training.schedulers.scheduler",
        "current_owner": "artifex.generative_models.training.schedulers.factory",
    },
    "tracking.md": {
        "planned_module": "artifex.generative_models.training.tracking",
        "current_owner": "artifex.generative_models.training.callbacks.logging",
    },
    "visualization.md": {
        "planned_module": "artifex.generative_models.training.callbacks.visualization",
        "current_owner": "artifex.generative_models.training.callbacks.profiling",
    },
}


def test_training_reference_pages_are_runtime_backed_or_coming_soon() -> None:
    """Each training page should be either live or clearly marked as coming soon."""
    actual_pages = {path.name for path in DOCS_ROOT.glob("*.md") if path.name != "index.md"}
    expected_pages = set(RETAINED_RUNTIME_PAGES) | set(COMING_SOON_PAGES)

    assert actual_pages == expected_pages

    for page_name, expected in RETAINED_RUNTIME_PAGES.items():
        contents = (DOCS_ROOT / page_name).read_text(encoding="utf-8")
        assert "**Status:** `Supported runtime training surface`" in contents

        for module in expected["modules"]:
            assert module in contents

        for source in expected["sources"]:
            assert source in contents

    for page_name, expected in COMING_SOON_PAGES.items():
        contents = (DOCS_ROOT / page_name).read_text(encoding="utf-8")
        assert "**Status:** `Coming soon`" in contents
        assert expected["planned_module"] in contents
        assert expected["current_owner"] in contents
        assert "not shipped as a standalone runtime module" in contents
        assert "See [Training Systems](index.md) for the current shared training docs." in contents


def test_training_index_and_example_docs_use_live_training_surface() -> None:
    """Connected training docs should teach the retained shared runtime only."""
    index_docs = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    factory_docs = (DOCS_ROOT / "factory.md").read_text(encoding="utf-8")

    combined_docs = index_docs + factory_docs

    for banned in [
        "DataParallelTrainer",
        "ModelParallelTrainer",
        "DeviceMesh(",
        "artifex.generative_models.training.distributed.model_parallel",
        "artifex.generative_models.training.mixed_precision",
        "artifex.generative_models.training.tracking",
        "artifex.generative_models.training.callbacks.visualization",
        "artifex.generative_models.training.optimizers.optax_wrappers",
        "create_optimizer(\n    optimizer_type=",
        "create_scheduler(\n    scheduler_type=",
        "tracking={",
    ]:
        assert banned not in combined_docs

    for required in [
        "DeviceMeshManager",
        "DataParallel",
        "DevicePlacement",
        "GradientAccumulator",
        "DynamicLossScaler",
        "OptimizerConfig",
        "SchedulerConfig",
        "create_optimizer",
        "create_scheduler",
        "`artifex.generative_models.training` keeps the shared owner set narrow",
        "## Current Training Pages",
        "## Coming Soon",
    ]:
        assert required in combined_docs


def test_training_factory_examples_use_typed_configs() -> None:
    """Factory examples should pass typed config objects, not kwargs-style helpers."""
    docs_to_check = [
        DOCS_ROOT / "index.md",
        DOCS_ROOT / "factory.md",
    ]

    for path in docs_to_check:
        lines = path.read_text(encoding="utf-8").splitlines()

        for index, line in enumerate(lines):
            if "create_optimizer(" in line:
                window = "\n".join(lines[index : index + 12])
                assert "OptimizerConfig(" in window, (
                    f"{path} should pass an OptimizerConfig into create_optimizer(...)"
                )
            if "create_scheduler(" in line:
                window = "\n".join(lines[index : index + 12])
                assert "SchedulerConfig(" in window, (
                    f"{path} should pass a SchedulerConfig into create_scheduler(...)"
                )


def test_training_nav_and_index_split_supported_from_coming_soon() -> None:
    """The training catalog should separate current runtime pages from roadmap-only pages."""
    index_contents = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    mkdocs_contents = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    training_reference_block = mkdocs_contents.split("      - Training Reference:\n", 1)[1].split(
        "      - Tutorials:\n", 1
    )[0]
    current_block = training_reference_block.split("        - Current Training Pages:\n", 1)[
        1
    ].split("        - Coming Soon:\n", 1)[0]
    coming_soon_block = training_reference_block.split("        - Coming Soon:\n", 1)[1]

    assert "**Status:** `Supported runtime training reference`" in index_contents
    assert "## Current Training Pages" in index_contents
    assert "## Coming Soon" in index_contents

    assert "training/index.md" in current_block
    for page_name in RETAINED_RUNTIME_PAGES:
        assert f"training/{page_name}" in current_block

    for page_name in COMING_SOON_PAGES:
        assert f"training/{page_name}" not in current_block
        assert f"training/{page_name}" in coming_soon_block
