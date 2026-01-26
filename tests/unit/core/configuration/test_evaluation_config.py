"""Tests for EvaluationConfig frozen dataclass.

This module tests the EvaluationConfig frozen dataclass which replaces the
Pydantic-based EvaluationConfig. Tests verify:
1. Basic instantiation and frozen behavior
2. Field validation (types, ranges, required fields)
3. Default values
4. Serialization (to_dict, from_dict)
5. Edge cases and error handling
"""

import dataclasses
from pathlib import Path

import pytest

from artifex.generative_models.core.configuration.evaluation_config import EvaluationConfig


class TestEvaluationConfigBasics:
    """Test basic EvaluationConfig functionality."""

    def test_create_minimal(self):
        """Test creating EvaluationConfig with minimal required fields."""
        config = EvaluationConfig(
            name="eval-config",
            metrics=("fid", "inception_score"),
        )
        assert config.name == "eval-config"
        assert config.metrics == ("fid", "inception_score")
        assert config.metric_params == {}  # default
        assert config.eval_batch_size == 32  # default
        assert config.num_eval_samples is None  # default
        assert config.save_predictions is False  # default
        assert config.save_metrics is True  # default
        assert config.output_dir == Path("./evaluation")  # default

    def test_create_full(self):
        """Test creating EvaluationConfig with all fields."""
        config = EvaluationConfig(
            name="eval-config",
            metrics=("fid", "inception_score", "ssim"),
            metric_params={
                "fid": {"batch_size": 50, "dims": 2048},
                "inception_score": {"splits": 10},
            },
            eval_batch_size=64,
            num_eval_samples=1000,
            save_predictions=True,
            save_metrics=True,
            output_dir=Path("/path/to/evaluation"),
            description="Evaluation configuration for GAN models",
            tags=("gan", "image"),
        )
        assert config.name == "eval-config"
        assert config.metrics == ("fid", "inception_score", "ssim")
        assert config.metric_params["fid"] == {"batch_size": 50, "dims": 2048}
        assert config.eval_batch_size == 64
        assert config.num_eval_samples == 1000
        assert config.save_predictions is True
        assert config.save_metrics is True
        assert config.output_dir == Path("/path/to/evaluation")
        assert config.description == "Evaluation configuration for GAN models"
        assert config.tags == ("gan", "image")

    def test_frozen(self):
        """Test that EvaluationConfig is frozen (immutable)."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.eval_batch_size = 64

    def test_hash(self):
        """Test that EvaluationConfig instances are not hashable due to dict fields."""
        config1 = EvaluationConfig(name="test", metrics=("fid",))
        # Configs with dict fields are not hashable
        with pytest.raises(TypeError, match="unhashable type"):
            hash(config1)


class TestEvaluationConfigValidation:
    """Test EvaluationConfig validation."""

    def test_metrics_required(self):
        """Test that metrics is required (validated in __post_init__)."""
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            EvaluationConfig(name="test")

    def test_metrics_empty_tuple(self):
        """Test that empty metrics tuple is caught."""
        with pytest.raises(ValueError, match="metrics cannot be empty"):
            EvaluationConfig(name="test", metrics=())

    def test_eval_batch_size_zero(self):
        """Test that zero eval_batch_size is invalid."""
        with pytest.raises(ValueError, match="eval_batch_size must be positive"):
            EvaluationConfig(name="test", metrics=("fid",), eval_batch_size=0)

    def test_eval_batch_size_negative(self):
        """Test that negative eval_batch_size is invalid."""
        with pytest.raises(ValueError, match="eval_batch_size must be positive"):
            EvaluationConfig(name="test", metrics=("fid",), eval_batch_size=-1)

    def test_eval_batch_size_positive_valid(self):
        """Test that positive eval_batch_size is valid."""
        config = EvaluationConfig(name="test", metrics=("fid",), eval_batch_size=64)
        assert config.eval_batch_size == 64

    def test_num_eval_samples_zero(self):
        """Test that zero num_eval_samples is invalid."""
        with pytest.raises(ValueError, match="num_eval_samples must be positive"):
            EvaluationConfig(name="test", metrics=("fid",), num_eval_samples=0)

    def test_num_eval_samples_negative(self):
        """Test that negative num_eval_samples is invalid."""
        with pytest.raises(ValueError, match="num_eval_samples must be positive"):
            EvaluationConfig(name="test", metrics=("fid",), num_eval_samples=-1)

    def test_num_eval_samples_none_valid(self):
        """Test that None num_eval_samples is valid."""
        config = EvaluationConfig(name="test", metrics=("fid",), num_eval_samples=None)
        assert config.num_eval_samples is None

    def test_num_eval_samples_positive_valid(self):
        """Test that positive num_eval_samples is valid."""
        config = EvaluationConfig(name="test", metrics=("fid",), num_eval_samples=1000)
        assert config.num_eval_samples == 1000

    def test_single_metric_valid(self):
        """Test that single metric is valid."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        assert config.metrics == ("fid",)

    def test_multiple_metrics_valid(self):
        """Test that multiple metrics are valid."""
        metrics = ("fid", "inception_score", "ssim", "psnr", "lpips")
        config = EvaluationConfig(name="test", metrics=metrics)
        assert config.metrics == metrics


class TestEvaluationConfigDefaults:
    """Test EvaluationConfig default values."""

    def test_default_metric_params(self):
        """Test default metric_params is empty dict."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        assert config.metric_params == {}

    def test_default_eval_batch_size(self):
        """Test default eval_batch_size is 32."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        assert config.eval_batch_size == 32

    def test_default_num_eval_samples(self):
        """Test default num_eval_samples is None."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        assert config.num_eval_samples is None

    def test_default_save_predictions(self):
        """Test default save_predictions is False."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        assert config.save_predictions is False

    def test_default_save_metrics(self):
        """Test default save_metrics is True."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        assert config.save_metrics is True

    def test_default_output_dir(self):
        """Test default output_dir is ./evaluation."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        assert config.output_dir == Path("./evaluation")

    def test_default_description(self):
        """Test default description from BaseConfig."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        assert config.description == ""

    def test_default_tags(self):
        """Test default tags from BaseConfig."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        assert config.tags == ()


class TestEvaluationConfigSerialization:
    """Test EvaluationConfig serialization."""

    def test_to_dict_minimal(self):
        """Test converting minimal config to dict."""
        config = EvaluationConfig(name="test", metrics=("fid",))
        data = config.to_dict()
        assert data["name"] == "test"
        assert data["metrics"] == ("fid",)
        assert data["metric_params"] == {}
        assert data["eval_batch_size"] == 32
        assert data["num_eval_samples"] is None

    def test_to_dict_full(self):
        """Test converting full config to dict."""
        config = EvaluationConfig(
            name="eval-config",
            metrics=("fid", "ssim"),
            metric_params={"fid": {"dims": 2048}},
            eval_batch_size=64,
            num_eval_samples=1000,
            save_predictions=True,
            output_dir=Path("/path/to/eval"),
        )
        data = config.to_dict()
        assert data["metrics"] == ("fid", "ssim")
        assert data["metric_params"] == {"fid": {"dims": 2048}}
        assert data["eval_batch_size"] == 64
        assert data["num_eval_samples"] == 1000
        assert data["save_predictions"] is True
        assert data["output_dir"] == Path("/path/to/eval")

    def test_from_dict_minimal(self):
        """Test creating config from minimal dict."""
        data = {"name": "test", "metrics": ["fid"]}
        config = EvaluationConfig.from_dict(data)
        assert config.name == "test"
        assert config.metrics == ("fid",)  # list converted to tuple
        assert config.eval_batch_size == 32

    def test_from_dict_full(self):
        """Test creating config from full dict."""
        data = {
            "name": "eval-config",
            "metrics": ["fid", "ssim"],
            "metric_params": {"fid": {"dims": 2048}},
            "eval_batch_size": 64,
            "num_eval_samples": 1000,
            "save_predictions": True,
            "save_metrics": False,
            "output_dir": "/path/to/eval",
        }
        config = EvaluationConfig.from_dict(data)
        assert config.metrics == ("fid", "ssim")
        assert config.metric_params == {"fid": {"dims": 2048}}
        assert config.eval_batch_size == 64
        assert config.num_eval_samples == 1000
        assert config.save_predictions is True
        assert config.save_metrics is False
        assert config.output_dir == Path("/path/to/eval")

    def test_from_dict_converts_lists_to_tuples(self):
        """Test that from_dict converts lists to tuples."""
        data = {
            "name": "test",
            "metrics": ["fid", "ssim", "lpips"],
        }
        config = EvaluationConfig.from_dict(data)
        assert isinstance(config.metrics, tuple)
        assert config.metrics == ("fid", "ssim", "lpips")

    def test_from_dict_with_path_string(self):
        """Test that string paths are converted to Path objects."""
        data = {
            "name": "test",
            "metrics": ["fid"],
            "output_dir": "/path/to/output",
        }
        config = EvaluationConfig.from_dict(data)
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/path/to/output")

    def test_roundtrip(self):
        """Test that to_dict -> from_dict roundtrip works."""
        original = EvaluationConfig(
            name="eval-config",
            metrics=("fid", "ssim"),
            metric_params={"fid": {"dims": 2048}},
            eval_batch_size=64,
            num_eval_samples=1000,
        )
        data = original.to_dict()
        restored = EvaluationConfig.from_dict(data)
        assert restored.name == original.name
        assert restored.metrics == original.metrics
        assert restored.metric_params == original.metric_params
        assert restored.eval_batch_size == original.eval_batch_size
        assert restored.num_eval_samples == original.num_eval_samples


class TestEvaluationConfigEdgeCases:
    """Test EvaluationConfig edge cases."""

    def test_output_dir_as_string(self):
        """Test that output_dir accepts string and converts to Path."""
        config = EvaluationConfig(
            name="test",
            metrics=("fid",),
            output_dir="/path/to/output",
        )
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/path/to/output")

    def test_output_dir_as_path(self):
        """Test that output_dir accepts Path objects."""
        config = EvaluationConfig(
            name="test",
            metrics=("fid",),
            output_dir=Path("/path/to/output"),
        )
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/path/to/output")

    def test_empty_metric_params(self):
        """Test that empty metric_params is valid."""
        config = EvaluationConfig(
            name="test",
            metrics=("fid",),
            metric_params={},
        )
        assert config.metric_params == {}

    def test_complex_metric_params(self):
        """Test that complex metric_params are handled."""
        params = {
            "fid": {"dims": 2048, "batch_size": 50},
            "inception_score": {"splits": 10, "resize": True},
            "ssim": {"data_range": 1.0, "multichannel": True},
        }
        config = EvaluationConfig(
            name="test",
            metrics=("fid", "inception_score", "ssim"),
            metric_params=params,
        )
        assert config.metric_params == params

    def test_nested_metric_params(self):
        """Test that nested metric_params are handled."""
        params = {
            "custom_metric": {
                "preprocessing": {"normalize": True, "resize": 256},
                "model": {"architecture": "resnet50", "pretrained": True},
            }
        }
        config = EvaluationConfig(
            name="test",
            metrics=("custom_metric",),
            metric_params=params,
        )
        assert config.metric_params == params

    def test_large_eval_batch_size(self):
        """Test that large eval_batch_size is valid."""
        config = EvaluationConfig(
            name="test",
            metrics=("fid",),
            eval_batch_size=1024,
        )
        assert config.eval_batch_size == 1024

    def test_large_num_eval_samples(self):
        """Test that large num_eval_samples is valid."""
        config = EvaluationConfig(
            name="test",
            metrics=("fid",),
            num_eval_samples=1000000,
        )
        assert config.num_eval_samples == 1000000

    def test_many_metrics(self):
        """Test that many metrics are handled."""
        metrics = (
            "fid",
            "inception_score",
            "ssim",
            "psnr",
            "lpips",
            "kid",
            "precision",
            "recall",
            "density",
            "coverage",
        )
        config = EvaluationConfig(name="test", metrics=metrics)
        assert config.metrics == metrics
        assert len(config.metrics) == 10

    def test_both_save_flags_false(self):
        """Test that both save flags can be False."""
        config = EvaluationConfig(
            name="test",
            metrics=("fid",),
            save_predictions=False,
            save_metrics=False,
        )
        assert config.save_predictions is False
        assert config.save_metrics is False

    def test_both_save_flags_true(self):
        """Test that both save flags can be True."""
        config = EvaluationConfig(
            name="test",
            metrics=("fid",),
            save_predictions=True,
            save_metrics=True,
        )
        assert config.save_predictions is True
        assert config.save_metrics is True

    def test_metric_params_subset_of_metrics(self):
        """Test that metric_params can be a subset of metrics."""
        config = EvaluationConfig(
            name="test",
            metrics=("fid", "ssim", "psnr"),
            metric_params={"fid": {"dims": 2048}},  # only fid has params
        )
        assert "fid" in config.metric_params
        assert "ssim" not in config.metric_params
        assert "psnr" not in config.metric_params

    def test_metric_params_superset_of_metrics(self):
        """Test that metric_params can include metrics not in metrics list."""
        # This is valid - user might have extra params for future use
        config = EvaluationConfig(
            name="test",
            metrics=("fid",),
            metric_params={"fid": {"dims": 2048}, "ssim": {"data_range": 1.0}},
        )
        assert "fid" in config.metric_params
        assert "ssim" in config.metric_params
