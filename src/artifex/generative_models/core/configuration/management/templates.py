"""Configuration template system for typed Artifex configs."""

import copy
from typing import Any

from artifex.generative_models.core.configuration.base_dataclass import ConfigDocument
from artifex.generative_models.core.configuration.distributed_config import (
    DistributedConfig,
)
from artifex.generative_models.core.configuration.training_config import TrainingConfig


_TRAINING_OPTIMIZER_PARAMS = (
    "optimizer_type",
    "learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "eps",
    "momentum",
    "nesterov",
    "initial_accumulator_value",
    "gradient_clip_value",
)

_TRAINING_SCHEDULER_PARAMS = (
    "scheduler_type",
    "warmup_steps",
    "min_lr_ratio",
    "cycle_length",
    "decay_rate",
    "decay_steps",
    "total_steps",
    "step_size",
    "gamma",
    "milestones",
)


class ConfigTemplate:
    """Template for generating configurations with validation."""

    def __init__(
        self,
        name: str,
        base_config: dict[str, Any],
        required_params: list[str],
        optional_params: dict[str, Any] | None = None,
        config_class: type[ConfigDocument] | None = None,
    ):
        """Initialize configuration template."""
        self.name = name
        self.base_config = base_config
        self.required_params = required_params
        self.optional_params = optional_params or {}
        self.config_class = config_class

    def generate(self, **params: Any) -> ConfigDocument | dict[str, Any]:
        """Generate configuration from the template."""
        missing = set(self.required_params) - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        config = copy.deepcopy(self.base_config)

        for key, default_value in self.optional_params.items():
            if key not in params:
                params[key] = default_value

        config = self._deep_merge(config, params)

        if self.config_class:
            try:
                return self.config_class.from_dict(config)
            except Exception as e:
                raise ValueError(f"Configuration validation failed: {e}") from e

        return config

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


class TrainingConfigTemplate(ConfigTemplate):
    """Special template for training configs that handles parameter placement."""

    def generate(self, **params: Any) -> TrainingConfig:
        """Generate training config with special parameter handling."""
        params = params.copy()

        # Validate required parameters first
        missing = set(self.required_params) - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        # Handle special parameters that need to go in nested configs
        optimizer_params = {}
        scheduler_params = {}

        # Extract optimizer parameters
        for param in _TRAINING_OPTIMIZER_PARAMS:
            if param in params:
                optimizer_params[param] = params.pop(param)

        # Extract scheduler parameters
        for param in _TRAINING_SCHEDULER_PARAMS:
            if param in params:
                scheduler_params[param] = params.pop(param)

        # Start with base config
        config = copy.deepcopy(self.base_config)

        # Apply optional defaults for missing params
        for key, default_value in self.optional_params.items():
            if key not in params:
                params[key] = default_value

        # Deep merge parameters into config
        config = self._deep_merge(config, params)

        # Apply optimizer parameters
        if optimizer_params:
            if "optimizer" not in config:
                config["optimizer"] = {}
            config["optimizer"].update(optimizer_params)

        # Apply scheduler parameters
        if scheduler_params:
            if "scheduler" not in config:
                config["scheduler"] = {}
            config["scheduler"].update(scheduler_params)

        # Validate if config class provided
        if self.config_class:
            try:
                validated_config = self.config_class.from_dict(config)
                if not isinstance(validated_config, TrainingConfig):
                    raise TypeError("training template must materialize TrainingConfig")
                return validated_config
            except Exception as e:
                raise ValueError(f"Template config validation failed: {e}") from e

        return config


SIMPLE_TRAINING_TEMPLATE = TrainingConfigTemplate(
    name="simple_training",
    base_config={
        "name": "simple_training",
        "description": "Simple training configuration",
        "num_epochs": 100,
        "optimizer": {
            "name": "default_optimizer",
            "optimizer_type": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        "scheduler": {
            "name": "default_scheduler",
            "scheduler_type": "cosine",
            "warmup_steps": 500,
            "min_lr_ratio": 0.001,
        },
        "log_frequency": 100,
        "save_frequency": 1000,
        "max_checkpoints": 5,
        "gradient_clip_norm": 1.0,
    },
    required_params=["batch_size", "learning_rate"],
    optional_params={},
    config_class=TrainingConfig,
)

DISTRIBUTED_TEMPLATE = ConfigTemplate(
    name="distributed_training",
    base_config={
        "name": "distributed_training",
        "description": "Distributed training configuration",
        "enabled": True,
        "backend": "nccl",
        "master_addr": "localhost",
        "master_port": 29500,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "broadcast_buffers": True,
        "mixed_precision": "fp16",
    },
    required_params=["world_size", "num_nodes", "num_processes_per_node"],
    optional_params={
        "rank": 0,
        "local_rank": 0,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    },
    config_class=DistributedConfig,
)


class ConfigTemplateManager:
    """Manager for configuration templates."""

    def __init__(self) -> None:
        """Initialize template manager."""
        self.templates = {
            "simple_training": SIMPLE_TRAINING_TEMPLATE,
            "distributed_training": DISTRIBUTED_TEMPLATE,
        }

    def register_template(self, template: ConfigTemplate) -> None:
        """Register a new template."""
        self.templates[template.name] = template

    def get_template(self, name: str) -> ConfigTemplate:
        """Get template by name."""
        if name not in self.templates:
            raise ValueError(
                f"Template '{name}' not found. Available: {list(self.templates.keys())}"
            )
        return self.templates[name]

    def generate_config(self, template_name: str, **params: Any) -> ConfigDocument | dict[str, Any]:
        """Generate configuration from template."""
        template = self.get_template(template_name)
        return template.generate(**params)

    def list_templates(self) -> list[str]:
        """List available template names."""
        return list(self.templates.keys())


# Global template manager instance
template_manager = ConfigTemplateManager()
