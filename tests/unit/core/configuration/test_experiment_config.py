"""Tests for the retained ExperimentConfig and ExperimentTemplateConfig contracts."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from artifex.generative_models.core.configuration import (
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    EvaluationConfig,
    ExperimentConfig,
    ExperimentTemplateConfig,
    ExperimentTemplateOverrides,
    OptimizerConfig,
    TrainingConfig,
    VAEConfig,
)


def _model_cfg() -> VAEConfig:
    encoder = EncoderConfig(
        name="encoder",
        input_shape=(28, 28, 1),
        latent_dim=16,
        hidden_dims=(64, 32),
        activation="relu",
    )
    decoder = DecoderConfig(
        name="decoder",
        latent_dim=16,
        output_shape=(28, 28, 1),
        hidden_dims=(32, 64),
        activation="relu",
    )
    return VAEConfig(name="vae_model", encoder=encoder, decoder=decoder)


def _training_cfg() -> TrainingConfig:
    optimizer = OptimizerConfig(name="adam", optimizer_type="adam", learning_rate=1e-3)
    return TrainingConfig(name="training", optimizer=optimizer)


def _data_cfg() -> DataConfig:
    return DataConfig(name="mnist", dataset_name="mnist")


class TestExperimentConfigBasics:
    """Basic construction and runtime validation."""

    def test_create_minimal(self) -> None:
        config = ExperimentConfig(
            name="experiment",
            model_cfg=_model_cfg(),
            training_cfg=_training_cfg(),
            data_cfg=_data_cfg(),
        )

        assert config.name == "experiment"
        assert isinstance(config.model_cfg, VAEConfig)
        assert config.eval_cfg is None
        assert config.seed == 42
        assert config.deterministic is True
        assert config.output_dir == Path("./experiments")

    def test_create_full(self) -> None:
        eval_cfg = EvaluationConfig(name="eval", metrics=("fid", "ssim"))

        config = ExperimentConfig(
            name="full_experiment",
            model_cfg=_model_cfg(),
            training_cfg=_training_cfg(),
            data_cfg=_data_cfg(),
            eval_cfg=eval_cfg,
            seed=123,
            deterministic=False,
            output_dir=Path("/tmp/experiments"),
            track_carbon=True,
            track_memory=True,
            description="full experiment",
            tags=("vae", "vision"),
        )

        assert config.eval_cfg == eval_cfg
        assert config.seed == 123
        assert config.deterministic is False
        assert config.output_dir == Path("/tmp/experiments")
        assert config.track_carbon is True
        assert config.track_memory is True
        assert config.tags == ("vae", "vision")

    def test_is_frozen(self) -> None:
        config = ExperimentConfig(
            name="experiment",
            model_cfg=_model_cfg(),
            training_cfg=_training_cfg(),
            data_cfg=_data_cfg(),
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.seed = 100

    def test_hash_is_blocked_by_nested_unhashable_configs(self) -> None:
        config = ExperimentConfig(
            name="experiment",
            model_cfg=_model_cfg(),
            training_cfg=_training_cfg(),
            data_cfg=_data_cfg(),
        )

        with pytest.raises(TypeError, match="unhashable type"):
            hash(config)


class TestExperimentConfigValidation:
    """Validation rules for nested configs and defaults."""

    def test_model_cfg_is_required(self) -> None:
        with pytest.raises(ValueError, match="model_cfg is required"):
            ExperimentConfig(
                name="missing_model",
                training_cfg=_training_cfg(),
                data_cfg=_data_cfg(),
            )

    def test_training_cfg_is_required(self) -> None:
        with pytest.raises(ValueError, match="training_cfg is required"):
            ExperimentConfig(
                name="missing_training",
                model_cfg=_model_cfg(),
                data_cfg=_data_cfg(),
            )

    def test_data_cfg_is_required(self) -> None:
        with pytest.raises(ValueError, match="data_cfg is required"):
            ExperimentConfig(
                name="missing_data",
                model_cfg=_model_cfg(),
                training_cfg=_training_cfg(),
            )

    def test_model_cfg_must_be_supported_family_config(self) -> None:
        with pytest.raises(TypeError, match="family-specific typed model config"):
            ExperimentConfig(
                name="invalid_model",
                model_cfg=_training_cfg(),
                training_cfg=_training_cfg(),
                data_cfg=_data_cfg(),
            )

    def test_output_dir_accepts_string_paths(self) -> None:
        config = ExperimentConfig(
            name="string_output_dir",
            model_cfg=_model_cfg(),
            training_cfg=_training_cfg(),
            data_cfg=_data_cfg(),
            output_dir="/tmp/experiments",
        )

        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/tmp/experiments")


class TestExperimentConfigSerialization:
    """Serialization should preserve the retained nested typed config shape."""

    def test_to_dict_contains_nested_family_config(self) -> None:
        config = ExperimentConfig(
            name="experiment",
            model_cfg=_model_cfg(),
            training_cfg=_training_cfg(),
            data_cfg=_data_cfg(),
        )

        payload = config.to_dict()

        assert payload["name"] == "experiment"
        assert payload["model_cfg"]["encoder"]["latent_dim"] == 16
        assert payload["training_cfg"]["optimizer"]["optimizer_type"] == "adam"
        assert payload["data_cfg"]["dataset_name"] == "mnist"

    def test_from_dict_materializes_vae_config(self) -> None:
        payload = {
            "name": "experiment",
            "model_cfg": {
                "name": "vae_model",
                "encoder": {
                    "name": "encoder",
                    "input_shape": [28, 28, 1],
                    "latent_dim": 16,
                    "hidden_dims": [64, 32],
                    "activation": "relu",
                },
                "decoder": {
                    "name": "decoder",
                    "latent_dim": 16,
                    "output_shape": [28, 28, 1],
                    "hidden_dims": [32, 64],
                    "activation": "relu",
                },
            },
            "training_cfg": {
                "name": "training",
                "optimizer": {
                    "name": "adam",
                    "optimizer_type": "adam",
                    "learning_rate": 1e-3,
                },
            },
            "data_cfg": {
                "name": "mnist",
                "dataset_name": "mnist",
            },
        }

        config = ExperimentConfig.from_dict(payload)

        assert isinstance(config.model_cfg, VAEConfig)
        assert config.model_cfg.encoder.latent_dim == 16
        assert config.training_cfg.optimizer.optimizer_type == "adam"
        assert config.data_cfg.dataset_name == "mnist"

    def test_from_dict_rejects_legacy_generic_model_payload(self) -> None:
        payload = {
            "name": "legacy_experiment",
            "model_cfg": {
                "name": "legacy_model",
                "model_class": "artifex.generative_models.models.vae.VAE",
                "input_dim": [28, 28, 1],
            },
            "training_cfg": {
                "name": "training",
                "optimizer": {
                    "name": "adam",
                    "optimizer_type": "adam",
                    "learning_rate": 1e-3,
                },
            },
            "data_cfg": {
                "name": "mnist",
                "dataset_name": "mnist",
            },
        }

        with pytest.raises(ValueError, match="family-specific typed model config"):
            ExperimentConfig.from_dict(payload)

    def test_roundtrip_preserves_nested_types(self) -> None:
        original = ExperimentConfig(
            name="roundtrip",
            model_cfg=_model_cfg(),
            training_cfg=_training_cfg(),
            data_cfg=_data_cfg(),
            eval_cfg=EvaluationConfig(name="eval", metrics=("fid",)),
            seed=7,
            deterministic=False,
        )

        restored = ExperimentConfig.from_dict(original.to_dict())

        assert isinstance(restored.model_cfg, VAEConfig)
        assert restored.model_cfg.encoder.hidden_dims == (64, 32)
        assert restored.eval_cfg.metrics == ("fid",)
        assert restored.seed == 7
        assert restored.deterministic is False


class TestExperimentTemplateConfig:
    """Retained template documents still use the non-BaseConfig schema."""

    def test_from_dict_uses_template_schema_not_runtime_name_field(self) -> None:
        config = ExperimentTemplateConfig.from_dict(
            {
                "experiment_name": "protein_diffusion_cath",
                "seed": 42,
                "output_dir": "{ARTIFEX_OUTPUT_ROOT}/protein_diffusion_cath/",
                "model_config": "models/geometric/protein_point_cloud.yaml",
                "data_config": "data/protein_dataset.yaml",
                "training_config": "training/protein_diffusion_training.yaml",
                "inference_config": "inference/protein_diffusion_inference.yaml",
                "log_level": "INFO",
                "use_wandb": True,
                "wandb_project": "protein-diffusion",
                "overrides": {
                    "model": {"num_residues": 150, "num_points": 600},
                    "training": {"batch_size": 64},
                    "inference": {"target_seq_length": 150},
                },
            }
        )

        assert isinstance(config, ExperimentTemplateConfig)
        assert not hasattr(config, "name")
        assert config.experiment_name == "protein_diffusion_cath"
        assert isinstance(config.overrides, ExperimentTemplateOverrides)
        assert config.overrides.model["num_residues"] == 150
        assert config.overrides.model["num_points"] == 600
        assert config.overrides.training["batch_size"] == 64
        assert config.overrides.inference["target_seq_length"] == 150
