"""Tests for the calibrax bridge layer (result_model.py).

Tests round-trip conversions between artifex BenchmarkResult/BenchmarkConfig
and calibrax.core.BenchmarkResult/Metric types.
"""

import pytest
from calibrax.core import BenchmarkResult as CalibraxResult, Metric as CalibraxMetric

from artifex.benchmarks.core import (
    BenchmarkConfig as ArtifexConfig,
    BenchmarkResult as ArtifexResult,
)


class TestImports:
    """Verify bridge module is importable."""

    def test_import_module(self) -> None:
        from artifex.benchmarks.core import result_model  # noqa: F401

    def test_import_to_calibrax_result(self) -> None:
        from artifex.benchmarks.core.result_model import to_calibrax_result  # noqa: F401

    def test_import_from_calibrax_result(self) -> None:
        from artifex.benchmarks.core.result_model import from_calibrax_result  # noqa: F401

    def test_import_config_to_dict(self) -> None:
        from artifex.benchmarks.core.result_model import config_to_dict  # noqa: F401

    def test_import_sanitize_jax_value(self) -> None:
        from artifex.benchmarks.core.result_model import sanitize_jax_value  # noqa: F401


class TestSanitizeJaxValue:
    """Test JAX scalar → Python primitive conversion."""

    def test_python_float_passthrough(self) -> None:
        from artifex.benchmarks.core.result_model import sanitize_jax_value

        assert sanitize_jax_value(3.14) == 3.14
        assert isinstance(sanitize_jax_value(3.14), float)

    def test_python_int_passthrough(self) -> None:
        from artifex.benchmarks.core.result_model import sanitize_jax_value

        assert sanitize_jax_value(42) == 42
        assert isinstance(sanitize_jax_value(42), int)

    def test_python_str_passthrough(self) -> None:
        from artifex.benchmarks.core.result_model import sanitize_jax_value

        assert sanitize_jax_value("hello") == "hello"

    def test_none_passthrough(self) -> None:
        from artifex.benchmarks.core.result_model import sanitize_jax_value

        assert sanitize_jax_value(None) is None

    def test_bool_passthrough(self) -> None:
        from artifex.benchmarks.core.result_model import sanitize_jax_value

        assert sanitize_jax_value(True) is True
        assert sanitize_jax_value(False) is False

    def test_jax_scalar_converted(self) -> None:
        import jax.numpy as jnp

        from artifex.benchmarks.core.result_model import sanitize_jax_value

        jax_val = jnp.float32(2.5)
        result = sanitize_jax_value(jax_val)
        assert result == pytest.approx(2.5)
        assert isinstance(result, float | int)

    def test_jax_int_scalar_converted(self) -> None:
        import jax.numpy as jnp

        from artifex.benchmarks.core.result_model import sanitize_jax_value

        jax_val = jnp.int32(7)
        result = sanitize_jax_value(jax_val)
        assert result == 7

    def test_nested_dict_sanitized(self) -> None:
        import jax.numpy as jnp

        from artifex.benchmarks.core.result_model import sanitize_jax_value

        data = {"a": jnp.float32(1.0), "b": {"c": jnp.int32(2)}}
        result = sanitize_jax_value(data)
        assert result == {"a": pytest.approx(1.0), "b": {"c": 2}}

    def test_list_sanitized(self) -> None:
        import jax.numpy as jnp

        from artifex.benchmarks.core.result_model import sanitize_jax_value

        data = [jnp.float32(1.0), jnp.float32(2.0)]
        result = sanitize_jax_value(data)
        assert result == [pytest.approx(1.0), pytest.approx(2.0)]


class TestConfigToDict:
    """Test artifex BenchmarkConfig → dict conversion."""

    def test_basic_config(self) -> None:
        from artifex.benchmarks.core.result_model import config_to_dict

        config = ArtifexConfig(
            name="test_bench",
            description="A test benchmark",
            metric_names=["fid", "lpips"],
        )
        result = config_to_dict(config)
        assert result["name"] == "test_bench"
        assert result["description"] == "A test benchmark"
        assert result["metric_names"] == ["fid", "lpips"]

    def test_config_with_metadata(self) -> None:
        from artifex.benchmarks.core.result_model import config_to_dict

        config = ArtifexConfig(
            name="bench",
            description="desc",
            metric_names=[],
            metadata={"seed": 42, "batch_size": 64},
        )
        result = config_to_dict(config)
        assert result["metadata"]["seed"] == 42
        assert result["metadata"]["batch_size"] == 64

    def test_config_empty_metadata(self) -> None:
        from artifex.benchmarks.core.result_model import config_to_dict

        config = ArtifexConfig(
            name="bench",
            description="desc",
            metric_names=[],
        )
        result = config_to_dict(config)
        assert result["metadata"] == {}


class TestToCaliraxResult:
    """Test artifex BenchmarkResult → calibrax BenchmarkResult."""

    def test_basic_conversion(self) -> None:
        from artifex.benchmarks.core.result_model import to_calibrax_result

        artifex_result = ArtifexResult(
            benchmark_name="geometric_fid",
            model_name="vae_v1",
            metrics={"fid": 12.5, "lpips": 0.03},
        )
        cal_result = to_calibrax_result(artifex_result)

        assert isinstance(cal_result, CalibraxResult)
        assert cal_result.name == "geometric_fid"
        assert cal_result.tags["model_name"] == "vae_v1"
        assert "fid" in cal_result.metrics
        assert cal_result.metrics["fid"].value == pytest.approx(12.5)
        assert cal_result.metrics["lpips"].value == pytest.approx(0.03)

    def test_metrics_wrapped_as_calibrax_metric(self) -> None:
        from artifex.benchmarks.core.result_model import to_calibrax_result

        artifex_result = ArtifexResult(
            benchmark_name="test",
            model_name="model",
            metrics={"accuracy": 0.95},
        )
        cal_result = to_calibrax_result(artifex_result)
        metric = cal_result.metrics["accuracy"]

        assert isinstance(metric, CalibraxMetric)
        assert metric.value == pytest.approx(0.95)
        assert metric.lower is None
        assert metric.upper is None

    def test_empty_metrics(self) -> None:
        from artifex.benchmarks.core.result_model import to_calibrax_result

        artifex_result = ArtifexResult(
            benchmark_name="empty",
            model_name="model",
            metrics={},
        )
        cal_result = to_calibrax_result(artifex_result)
        assert cal_result.metrics == {}

    def test_metadata_preserved(self) -> None:
        from artifex.benchmarks.core.result_model import to_calibrax_result

        artifex_result = ArtifexResult(
            benchmark_name="test",
            model_name="model",
            metrics={"fid": 10.0},
            metadata={"runtime": 45.2, "device": "gpu"},
        )
        cal_result = to_calibrax_result(artifex_result)
        assert cal_result.metadata["runtime"] == pytest.approx(45.2)
        assert cal_result.metadata["device"] == "gpu"

    def test_none_metadata_becomes_empty(self) -> None:
        from artifex.benchmarks.core.result_model import to_calibrax_result

        # ArtifexResult defaults metadata to {} via field(default_factory=dict)
        artifex_result = ArtifexResult(
            benchmark_name="test",
            model_name="model",
            metrics={},
        )
        cal_result = to_calibrax_result(artifex_result)
        assert cal_result.metadata == {}

    def test_jax_metric_values_sanitized(self) -> None:
        import jax.numpy as jnp

        from artifex.benchmarks.core.result_model import to_calibrax_result

        artifex_result = ArtifexResult(
            benchmark_name="test",
            model_name="model",
            metrics={"fid": float(jnp.float32(15.5))},
        )
        cal_result = to_calibrax_result(artifex_result)
        assert cal_result.metrics["fid"].value == pytest.approx(15.5)

    def test_domain_tag_set(self) -> None:
        from artifex.benchmarks.core.result_model import to_calibrax_result

        artifex_result = ArtifexResult(
            benchmark_name="test",
            model_name="model",
            metrics={},
        )
        cal_result = to_calibrax_result(artifex_result, domain="generative_models")
        assert cal_result.domain == "generative_models"


class TestFromCalibraxResult:
    """Test calibrax BenchmarkResult → artifex BenchmarkResult."""

    def test_basic_conversion(self) -> None:
        from artifex.benchmarks.core.result_model import from_calibrax_result

        cal_result = CalibraxResult(
            name="geometric_fid",
            tags={"model_name": "vae_v1"},
            metrics={
                "fid": CalibraxMetric(value=12.5),
                "lpips": CalibraxMetric(value=0.03),
            },
        )
        artifex_result = from_calibrax_result(cal_result)

        assert isinstance(artifex_result, ArtifexResult)
        assert artifex_result.benchmark_name == "geometric_fid"
        assert artifex_result.model_name == "vae_v1"
        assert artifex_result.metrics["fid"] == pytest.approx(12.5)
        assert artifex_result.metrics["lpips"] == pytest.approx(0.03)

    def test_missing_model_name_tag(self) -> None:
        from artifex.benchmarks.core.result_model import from_calibrax_result

        cal_result = CalibraxResult(
            name="test",
            metrics={"fid": CalibraxMetric(value=10.0)},
        )
        artifex_result = from_calibrax_result(cal_result)
        assert artifex_result.model_name == ""

    def test_empty_metrics(self) -> None:
        from artifex.benchmarks.core.result_model import from_calibrax_result

        cal_result = CalibraxResult(name="test")
        artifex_result = from_calibrax_result(cal_result)
        assert artifex_result.metrics == {}

    def test_metadata_preserved(self) -> None:
        from artifex.benchmarks.core.result_model import from_calibrax_result

        cal_result = CalibraxResult(
            name="test",
            metadata={"runtime": 45.2, "device": "gpu"},
        )
        artifex_result = from_calibrax_result(cal_result)
        assert artifex_result.metadata["runtime"] == pytest.approx(45.2)
        assert artifex_result.metadata["device"] == "gpu"


class TestRoundTrip:
    """Test round-trip conversion preserves data."""

    def test_artifex_to_calibrax_and_back(self) -> None:
        from artifex.benchmarks.core.result_model import (
            from_calibrax_result,
            to_calibrax_result,
        )

        original = ArtifexResult(
            benchmark_name="geometric_fid",
            model_name="vae_v1",
            metrics={"fid": 12.5, "lpips": 0.03, "is": 150.0},
            metadata={"runtime": 45.2, "num_samples": 1000},
        )

        cal_result = to_calibrax_result(original)
        recovered = from_calibrax_result(cal_result)

        assert recovered.benchmark_name == original.benchmark_name
        assert recovered.model_name == original.model_name
        for key in original.metrics:
            assert recovered.metrics[key] == pytest.approx(original.metrics[key])
        assert recovered.metadata["runtime"] == pytest.approx(45.2)
        assert recovered.metadata["num_samples"] == 1000

    def test_calibrax_to_artifex_and_back(self) -> None:
        from artifex.benchmarks.core.result_model import (
            from_calibrax_result,
            to_calibrax_result,
        )

        original = CalibraxResult(
            name="protein_benchmark",
            tags={"model_name": "diffusion_v2"},
            metrics={
                "rmsd": CalibraxMetric(value=1.8),
                "gdt_ts": CalibraxMetric(value=0.85),
            },
            metadata={"device": "cuda:0"},
        )

        artifex_result = from_calibrax_result(original)
        recovered = to_calibrax_result(artifex_result)

        assert recovered.name == original.name
        assert recovered.tags["model_name"] == "diffusion_v2"
        for key in original.metrics:
            assert recovered.metrics[key].value == pytest.approx(original.metrics[key].value)

    def test_serialization_round_trip(self) -> None:
        """Test that calibrax result can be serialized and deserialized."""
        from artifex.benchmarks.core.result_model import to_calibrax_result

        artifex_result = ArtifexResult(
            benchmark_name="test",
            model_name="model",
            metrics={"fid": 12.5},
            metadata={"batch_size": 32},
        )

        cal_result = to_calibrax_result(artifex_result)
        result_dict = cal_result.to_dict()
        restored = CalibraxResult.from_dict(result_dict)

        assert restored.name == cal_result.name
        assert restored.metrics["fid"].value == pytest.approx(12.5)
