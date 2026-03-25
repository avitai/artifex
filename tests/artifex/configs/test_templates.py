"""Tests for the public configuration template surface."""

from __future__ import annotations

import pytest

from artifex.configs import (
    ConfigTemplateManager,
    DISTRIBUTED_TEMPLATE,
    SIMPLE_TRAINING_TEMPLATE,
    template_manager,
)
from artifex.generative_models.core.configuration import DistributedConfig, TrainingConfig
from artifex.generative_models.core.configuration.management import templates as template_module
from artifex.generative_models.core.configuration.management.templates import ConfigTemplate


def test_artifex_configs_reexports_canonical_template_surface() -> None:
    """The public config package should expose the canonical template objects."""
    assert ConfigTemplateManager is template_module.ConfigTemplateManager
    assert SIMPLE_TRAINING_TEMPLATE is template_module.SIMPLE_TRAINING_TEMPLATE
    assert DISTRIBUTED_TEMPLATE is template_module.DISTRIBUTED_TEMPLATE
    assert template_manager is template_module.template_manager
    assert ConfigTemplate.__module__ == (
        "artifex.generative_models.core.configuration.management.templates"
    )


def test_template_manager_generates_training_configs_via_public_surface() -> None:
    """Template generation should work through the supported artifex.configs API."""
    generated = template_manager.generate_config(
        "simple_training",
        batch_size=64,
        learning_rate=2e-4,
    )

    assert isinstance(generated, TrainingConfig)
    assert generated.batch_size == 64
    assert generated.optimizer.learning_rate == pytest.approx(2e-4)
    assert generated.scheduler is not None
    assert generated.scheduler.warmup_steps == 500
    assert generated.log_frequency == 100
    assert generated.save_frequency == 1000


def test_template_manager_lists_supported_templates() -> None:
    """The public template manager should report the built-in templates."""
    assert template_manager.list_templates() == [
        "simple_training",
        "distributed_training",
    ]


def test_legacy_model_template_is_not_exposed() -> None:
    """Broken legacy model templates should not remain in the supported API."""
    with pytest.raises(ValueError, match="protein_diffusion"):
        template_manager.get_template("protein_diffusion")


def test_template_manager_routes_nested_optimizer_and_scheduler_params() -> None:
    """Flat overrides should be routed into the nested config objects they target."""
    generated = template_manager.generate_config(
        "simple_training",
        batch_size=32,
        learning_rate=1e-3,
        optimizer_type="sgd",
        momentum=0.8,
        scheduler_type="linear",
        total_steps=5000,
    )

    assert isinstance(generated, TrainingConfig)
    assert generated.optimizer.optimizer_type == "sgd"
    assert generated.optimizer.momentum == pytest.approx(0.8)
    assert generated.scheduler is not None
    assert generated.scheduler.scheduler_type == "linear"
    assert generated.scheduler.total_steps == 5000


def test_template_manager_generates_distributed_configs_via_public_surface() -> None:
    """Supported built-ins should materialize typed frozen configs."""
    generated = template_manager.generate_config(
        "distributed_training",
        world_size=8,
        num_nodes=2,
        num_processes_per_node=4,
    )

    assert isinstance(generated, DistributedConfig)
    assert generated.world_size == 8
    assert generated.num_nodes == 2
    assert generated.num_processes_per_node == 4
