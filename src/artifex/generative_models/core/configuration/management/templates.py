"""Configuration template system for artifex.generative_models.core."""

from typing import Any

from artifex.configs import DistributedConfig, TrainingConfig
from artifex.configs.utils.error_handling import ConfigValidationError
from artifex.generative_models.core.configuration import ModelConfig
from artifex.generative_models.core.protocols.configuration import ConfigTemplate


# Predefined templates
PROTEIN_DIFFUSION_TEMPLATE = ConfigTemplate(
    name="protein_diffusion",
    base_config={
        "name": "protein_diffusion",
        "description": "Protein diffusion model",
        "num_layers": 8,
        "model_dim": 128,
        "dropout": 0.1,
        "noise_steps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "linear",
    },
    required_params=["max_seq_length", "backbone_atom_indices"],
    optional_params={
        "hidden_dim": 256,
    },
    config_class=ModelConfig,
)


class TrainingConfigTemplate(ConfigTemplate):
    """Special template for training configs that handles parameter placement."""

    def generate(self, **params) -> dict[str, Any]:
        """Generate training config with special parameter handling."""
        # Validate required parameters first
        missing = set(self.required_params) - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        # Handle special parameters that need to go in nested configs
        optimizer_params = {}
        scheduler_params = {}

        # Extract optimizer parameters
        for param in ["learning_rate", "weight_decay", "beta1", "beta2", "eps"]:
            if param in params:
                optimizer_params[param] = params.pop(param)

        # Extract scheduler parameters
        for param in ["warmup_steps", "warmup_ratio", "min_lr_ratio"]:
            if param in params:
                scheduler_params[param] = params.pop(param)

        # Handle eval_batch_size defaulting to batch_size
        if "eval_batch_size" not in params and "batch_size" in params:
            params["eval_batch_size"] = params["batch_size"]

        # Start with base config
        config = self.base_config.copy()

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
                validated_config = self.config_class(**config)
                return validated_config.to_dict()
            except Exception as e:
                raise ConfigValidationError("template_generated_config", e) from e

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
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.001,
        },
        "log_freq": 20,
        "eval_freq": 200,
        "save_freq": 100,
        "max_checkpoints": 5,
        "num_workers": 4,
        "grad_clip_norm": 1.0,
    },
    required_params=["batch_size", "learning_rate"],
    optional_params={
        "eval_batch_size": 32,
    },
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

    def __init__(self):
        """Initialize template manager."""
        self.templates = {
            "protein_diffusion": PROTEIN_DIFFUSION_TEMPLATE,
            "simple_training": SIMPLE_TRAINING_TEMPLATE,
            "distributed_training": DISTRIBUTED_TEMPLATE,
        }

    def register_template(self, template: ConfigTemplate):
        """Register a new template."""
        self.templates[template.name] = template

    def get_template(self, name: str) -> ConfigTemplate:
        """Get template by name."""
        if name not in self.templates:
            raise ValueError(
                f"Template '{name}' not found. Available: {list(self.templates.keys())}"
            )
        return self.templates[name]

    def generate_config(self, template_name: str, **params) -> dict[str, Any]:
        """Generate configuration from template."""
        template = self.get_template(template_name)
        return template.generate(**params)

    def list_templates(self) -> list[str]:
        """List available template names."""
        return list(self.templates.keys())


# Global template manager instance
template_manager = ConfigTemplateManager()
