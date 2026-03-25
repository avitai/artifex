"""Tests for MetricProtocol conformance across all metric classes.

Verifies that all metric implementations:
1. Have `name` and `higher_is_better` as properties
2. Have `validate_inputs` that raises ValueError on invalid input
3. Pass `isinstance(m, calibrax.core.MetricProtocol)`
4. Conform to MetricBase interface
"""

import inspect

import jax.numpy as jnp
import pytest
from calibrax.core import MetricProtocol
from flax import nnx

from artifex.benchmarks.metrics.core import MetricBase
from artifex.generative_models.core.configuration import EvaluationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_config(
    name: str = "test_metric",
    higher_is_better: bool = True,
    **extra_params,
) -> EvaluationConfig:
    """Create a minimal EvaluationConfig for testing."""
    return EvaluationConfig(
        name=name,
        metrics=[name],
        metric_params={"higher_is_better": higher_is_better, **extra_params},
        eval_batch_size=4,
    )


def _make_rngs() -> nnx.Rngs:
    return nnx.Rngs(42)


# ---------------------------------------------------------------------------
# MetricBase protocol conformance
# ---------------------------------------------------------------------------


class TestMetricBaseProtocol:
    """Test that MetricBase satisfies calibrax MetricProtocol structurally."""

    def test_name_is_property(self) -> None:
        """MetricBase.name must be a property (not just an instance attr)."""
        assert isinstance(MetricBase.__dict__.get("name"), property) or hasattr(MetricBase, "name")

    def test_higher_is_better_is_property(self) -> None:
        """MetricBase.higher_is_better must be a property."""
        assert isinstance(MetricBase.__dict__.get("higher_is_better"), property) or hasattr(
            MetricBase, "higher_is_better"
        )

    def test_has_compute_method(self) -> None:
        assert hasattr(MetricBase, "compute")

    def test_has_validate_inputs_method(self) -> None:
        assert hasattr(MetricBase, "validate_inputs")

    def test_metric_base_is_config_agnostic(self) -> None:
        """MetricBase should not own EvaluationConfig adaptation."""
        assert "config" not in inspect.signature(MetricBase.__init__).parameters


# ---------------------------------------------------------------------------
# Image metrics
# ---------------------------------------------------------------------------


class TestImageMetricConformance:
    """Test image metrics conform to MetricProtocol."""

    @pytest.fixture()
    def rngs(self) -> nnx.Rngs:
        return _make_rngs()

    def test_fid_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.image import FIDMetric

        config = EvaluationConfig(
            name="fid",
            metrics=["fid"],
            metric_params={
                "higher_is_better": False,
                "fid": {"mock_inception": True, "demo_mode": True},
            },
            eval_batch_size=4,
        )
        metric = FIDMetric(rngs=rngs, config=config)
        assert isinstance(metric, MetricProtocol)

    def test_fid_name_property(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.image import FIDMetric

        config = EvaluationConfig(
            name="fid",
            metrics=["fid"],
            metric_params={
                "higher_is_better": False,
                "fid": {"mock_inception": True, "demo_mode": True},
            },
            eval_batch_size=4,
        )
        metric = FIDMetric(rngs=rngs, config=config)
        assert metric.name == "fid"

    def test_fid_higher_is_better(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.image import FIDMetric

        config = EvaluationConfig(
            name="fid",
            metrics=["fid"],
            metric_params={
                "higher_is_better": False,
                "fid": {"mock_inception": True, "demo_mode": True},
            },
            eval_batch_size=4,
        )
        metric = FIDMetric(rngs=rngs, config=config)
        assert metric.higher_is_better is False

    def test_lpips_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.image import LPIPSMetric

        config = EvaluationConfig(
            name="lpips",
            metrics=["lpips"],
            metric_params={
                "lpips": {"higher_is_better": False, "mock_implementation": True, "demo_mode": True}
            },
            eval_batch_size=4,
        )
        metric = LPIPSMetric(rngs=rngs, config=config)
        assert isinstance(metric, MetricProtocol)

    def test_ssim_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.image import SSIMMetric

        config = EvaluationConfig(
            name="ssim",
            metrics=["ssim"],
            metric_params={"higher_is_better": True},
            eval_batch_size=4,
        )
        metric = SSIMMetric(rngs=rngs, config=config)
        assert isinstance(metric, MetricProtocol)

    def test_is_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.image import ISMetric

        config = EvaluationConfig(
            name="is",
            metrics=["inception_score"],
            metric_params={
                "inception_score": {
                    "higher_is_better": True,
                    "mock_inception": True,
                    "demo_mode": True,
                }
            },
            eval_batch_size=4,
        )
        metric = ISMetric(rngs=rngs, config=config)
        assert isinstance(metric, MetricProtocol)

    def test_fid_validate_inputs_raises_on_invalid(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.image import FIDMetric

        config = EvaluationConfig(
            name="fid",
            metrics=["fid"],
            metric_params={
                "higher_is_better": False,
                "fid": {"mock_inception": True, "demo_mode": True},
            },
            eval_batch_size=4,
        )
        metric = FIDMetric(rngs=rngs, config=config)
        with pytest.raises(ValueError):
            metric.validate_inputs("not_an_array", "not_an_array")

    def test_fid_validate_inputs_passes_on_valid(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.image import FIDMetric

        config = EvaluationConfig(
            name="fid",
            metrics=["fid"],
            metric_params={
                "higher_is_better": False,
                "fid": {"mock_inception": True, "demo_mode": True},
            },
            eval_batch_size=4,
        )
        metric = FIDMetric(rngs=rngs, config=config)
        real = jnp.ones((4, 32, 32, 3))
        gen = jnp.ones((4, 32, 32, 3))
        # Should not raise
        metric.validate_inputs(real, gen)


# ---------------------------------------------------------------------------
# Disentanglement metrics
# ---------------------------------------------------------------------------


class TestDisentanglementMetricConformance:
    """Test disentanglement metrics conform to MetricProtocol."""

    @pytest.fixture()
    def rngs(self) -> nnx.Rngs:
        return _make_rngs()

    def test_mig_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.disentanglement import MutualInformationGapMetric

        config = _make_eval_config("mig")
        metric = MutualInformationGapMetric(config=config, rngs=rngs)
        assert isinstance(metric, MetricProtocol)

    def test_sap_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.disentanglement import SeparationMetric

        config = _make_eval_config("sap")
        metric = SeparationMetric(config=config, rngs=rngs)
        assert isinstance(metric, MetricProtocol)

    def test_dci_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.disentanglement import DisentanglementMetric

        config = _make_eval_config("dci")
        metric = DisentanglementMetric(config=config, rngs=rngs)
        assert isinstance(metric, MetricProtocol)

    def test_mig_validate_inputs_raises_on_invalid(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.disentanglement import MutualInformationGapMetric

        config = _make_eval_config("mig")
        metric = MutualInformationGapMetric(config=config, rngs=rngs)
        with pytest.raises(ValueError):
            metric.validate_inputs("invalid", "invalid")

    @pytest.mark.parametrize(
        ("metric_name", "metric_cls_name"),
        [
            ("mig", "MutualInformationGapMetric"),
            ("sap", "SeparationMetric"),
            ("dci", "DisentanglementMetric"),
        ],
    )
    def test_disentanglement_metrics_reject_compat_kwargs(
        self,
        rngs: nnx.Rngs,
        metric_name: str,
        metric_cls_name: str,
    ) -> None:
        module = __import__(
            "artifex.benchmarks.metrics.disentanglement",
            fromlist=[metric_cls_name],
        )
        metric_cls = getattr(module, metric_cls_name)
        metric = metric_cls(config=_make_eval_config(metric_name), rngs=rngs)
        factors = jnp.ones((4, 2))
        latents = jnp.ones((4, 3))

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            metric.compute(
                factors,
                latents,
                latent_representations=latents,
                ground_truth_factors=factors,
            )


# ---------------------------------------------------------------------------
# Geometric metrics
# ---------------------------------------------------------------------------


class TestGeometricMetricConformance:
    """Test geometric metrics conform to MetricProtocol."""

    @pytest.fixture()
    def rngs(self) -> nnx.Rngs:
        return _make_rngs()

    def test_point_cloud_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.geometric import PointCloudMetrics

        config = EvaluationConfig(
            name="point_cloud",
            metrics=["point_cloud"],
            metric_params={"higher_is_better": True, "point_cloud": {}},
            eval_batch_size=4,
        )
        metric = PointCloudMetrics(rngs=rngs, config=config)
        assert isinstance(metric, MetricProtocol)

    def test_point_cloud_validate_inputs_raises(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.geometric import PointCloudMetrics

        config = EvaluationConfig(
            name="point_cloud",
            metrics=["point_cloud"],
            metric_params={"higher_is_better": True, "point_cloud": {}},
            eval_batch_size=4,
        )
        metric = PointCloudMetrics(rngs=rngs, config=config)
        with pytest.raises(ValueError):
            metric.validate_inputs("invalid", "invalid")


# ---------------------------------------------------------------------------
# Audio metrics
# ---------------------------------------------------------------------------


class TestAudioMetricConformance:
    """Test audio metrics conform to MetricProtocol."""

    @pytest.fixture()
    def rngs(self) -> nnx.Rngs:
        return _make_rngs()

    def test_spectral_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.audio import SpectralMetric

        config = EvaluationConfig(
            name="spectral",
            metrics=["spectral"],
            metric_params={"higher_is_better": True, "spectral": {}},
            eval_batch_size=4,
        )
        metric = SpectralMetric(config=config, rngs=rngs)
        assert isinstance(metric, MetricProtocol)

    def test_mcd_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.audio import MelCepstralMetric

        config = EvaluationConfig(
            name="mcd",
            metrics=["mcd"],
            metric_params={"higher_is_better": False, "mcd": {}},
            eval_batch_size=4,
        )
        metric = MelCepstralMetric(config=config, rngs=rngs)
        assert isinstance(metric, MetricProtocol)

    def test_spectral_validate_inputs_raises(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.audio import SpectralMetric

        config = EvaluationConfig(
            name="spectral",
            metrics=["spectral"],
            metric_params={"higher_is_better": True, "spectral": {}},
            eval_batch_size=4,
        )
        metric = SpectralMetric(config=config, rngs=rngs)
        with pytest.raises(ValueError):
            metric.validate_inputs("invalid", "invalid")


# ---------------------------------------------------------------------------
# Molecular flows metrics
# ---------------------------------------------------------------------------


class TestMolecularMetricConformance:
    """Test molecular metrics conform to MetricProtocol."""

    @pytest.fixture()
    def rngs(self) -> nnx.Rngs:
        return _make_rngs()

    def test_molecular_flows_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.molecular_flows import MolecularFlowsMetrics

        config = EvaluationConfig(
            name="molecular_flows",
            metrics=["molecular_flows"],
            metric_params={"higher_is_better": True},
            eval_batch_size=4,
        )
        metric = MolecularFlowsMetrics(rngs=rngs, config=config)
        assert isinstance(metric, MetricProtocol)

    def test_molecular_flows_validate_inputs_raises(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.molecular_flows import MolecularFlowsMetrics

        config = EvaluationConfig(
            name="molecular_flows",
            metrics=["molecular_flows"],
            metric_params={"higher_is_better": True},
            eval_batch_size=4,
        )
        metric = MolecularFlowsMetrics(rngs=rngs, config=config)
        with pytest.raises(ValueError):
            metric.validate_inputs(None, "not_a_dict")


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------


class TestTextMetricConformance:
    """Test text metrics conform to MetricProtocol."""

    @pytest.fixture()
    def rngs(self) -> nnx.Rngs:
        return _make_rngs()

    def test_bleu_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.text import BLEUMetric

        config = EvaluationConfig(
            name="bleu",
            metrics=["bleu"],
            metric_params={"higher_is_better": True, "bleu": {}},
            eval_batch_size=4,
        )
        metric = BLEUMetric(config=config, rngs=rngs)
        assert isinstance(metric, MetricProtocol)

    def test_rouge_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.text import ROUGEMetric

        config = EvaluationConfig(
            name="rouge",
            metrics=["rouge"],
            metric_params={"higher_is_better": True, "rouge": {}},
            eval_batch_size=4,
        )
        metric = ROUGEMetric(config=config, rngs=rngs)
        assert isinstance(metric, MetricProtocol)

    def test_bleu_validate_inputs_raises(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.text import BLEUMetric

        config = EvaluationConfig(
            name="bleu",
            metrics=["bleu"],
            metric_params={"higher_is_better": True, "bleu": {}},
            eval_batch_size=4,
        )
        metric = BLEUMetric(config=config, rngs=rngs)
        with pytest.raises(ValueError):
            metric.validate_inputs(123, 456)


# ---------------------------------------------------------------------------
# Protein-ligand metrics — must extend MetricBase
# ---------------------------------------------------------------------------


class TestProteinLigandMetricConformance:
    """Test protein-ligand metrics conform to MetricProtocol."""

    @pytest.fixture()
    def rngs(self) -> nnx.Rngs:
        return _make_rngs()

    def test_binding_affinity_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.protein_ligand import BindingAffinityMetric

        config = EvaluationConfig(
            name="binding_affinity",
            metrics=["binding_affinity"],
            metric_params={"higher_is_better": False},
            eval_batch_size=4,
        )
        metric = BindingAffinityMetric(rngs=rngs, config=config)
        assert isinstance(metric, MetricProtocol)

    def test_binding_affinity_extends_metric_base(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.protein_ligand import BindingAffinityMetric

        config = EvaluationConfig(
            name="binding_affinity",
            metrics=["binding_affinity"],
            metric_params={"higher_is_better": False},
            eval_batch_size=4,
        )
        metric = BindingAffinityMetric(rngs=rngs, config=config)
        assert isinstance(metric, MetricBase)

    def test_molecular_validity_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.protein_ligand import MolecularValidityMetric

        config = EvaluationConfig(
            name="molecular_validity",
            metrics=["molecular_validity"],
            metric_params={"higher_is_better": True},
            eval_batch_size=4,
        )
        metric = MolecularValidityMetric(rngs=rngs, config=config)
        assert isinstance(metric, MetricProtocol)

    def test_molecular_validity_extends_metric_base(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.protein_ligand import MolecularValidityMetric

        config = EvaluationConfig(
            name="molecular_validity",
            metrics=["molecular_validity"],
            metric_params={"higher_is_better": True},
            eval_batch_size=4,
        )
        metric = MolecularValidityMetric(rngs=rngs, config=config)
        assert isinstance(metric, MetricBase)

    def test_drug_likeness_isinstance_metric_protocol(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.protein_ligand import DrugLikenessMetric

        config = EvaluationConfig(
            name="drug_likeness",
            metrics=["drug_likeness"],
            metric_params={"drug_likeness": {"higher_is_better": True, "demo_mode": True}},
            eval_batch_size=4,
        )
        metric = DrugLikenessMetric(rngs=rngs, config=config)
        assert isinstance(metric, MetricProtocol)

    def test_drug_likeness_extends_metric_base(self, rngs: nnx.Rngs) -> None:
        from artifex.benchmarks.metrics.protein_ligand import DrugLikenessMetric

        config = EvaluationConfig(
            name="drug_likeness",
            metrics=["drug_likeness"],
            metric_params={"drug_likeness": {"higher_is_better": True, "demo_mode": True}},
            eval_batch_size=4,
        )
        metric = DrugLikenessMetric(rngs=rngs, config=config)
        assert isinstance(metric, MetricBase)


# ---------------------------------------------------------------------------
# Style metrics — DRY dedup: no duplicate FID/LPIPS
# ---------------------------------------------------------------------------


class TestStyleMetricsDRY:
    """Test style_metrics.py DRY compliance — no duplicate FID/LPIPS."""

    def test_no_fid_metric_class_in_style_metrics(self) -> None:
        """style_metrics.py must not define its own FIDMetric class."""
        import artifex.benchmarks.metrics.style_metrics as sm

        # FIDMetric should not be defined in style_metrics
        # (it may be imported from image.py, but not defined locally)
        classes_defined_locally = [
            name
            for name, obj in vars(sm).items()
            if isinstance(obj, type) and obj.__module__ == sm.__name__
        ]
        assert "FIDMetric" not in classes_defined_locally

    def test_no_lpips_metric_class_in_style_metrics(self) -> None:
        """style_metrics.py must not define its own LPIPSMetric class."""
        import artifex.benchmarks.metrics.style_metrics as sm

        classes_defined_locally = [
            name
            for name, obj in vars(sm).items()
            if isinstance(obj, type) and obj.__module__ == sm.__name__
        ]
        assert "LPIPSMetric" not in classes_defined_locally

    def test_style_mixing_extends_metric_base(self) -> None:
        """StyleGAN-specific metrics should extend MetricBase."""
        from artifex.benchmarks.metrics.style_metrics import StyleMixingMetric

        assert issubclass(StyleMixingMetric, MetricBase)

    def test_equivariance_extends_metric_base(self) -> None:
        from artifex.benchmarks.metrics.style_metrics import EquivarianceMetric

        assert issubclass(EquivarianceMetric, MetricBase)

    def test_few_shot_extends_metric_base(self) -> None:
        from artifex.benchmarks.metrics.style_metrics import FewShotAdaptationMetric

        assert issubclass(FewShotAdaptationMetric, MetricBase)


# ---------------------------------------------------------------------------
# Evaluation infrastructure metrics
# ---------------------------------------------------------------------------


class TestEvalInfraMetricConformance:
    """Test evaluation infrastructure metrics conform to MetricProtocol."""

    def test_feature_based_extends_metric_base(self) -> None:
        from artifex.generative_models.core.evaluation.metrics.base import FeatureBasedMetric

        assert issubclass(FeatureBasedMetric, MetricBase)
        assert isinstance(FeatureBasedMetric(name="test", batch_size=4), MetricProtocol)

    def test_distribution_metric_extends_metric_base(self) -> None:
        from artifex.generative_models.core.evaluation.metrics.base import DistributionMetric

        assert issubclass(DistributionMetric, MetricBase)
        assert DistributionMetric.__init__ is MetricBase.__init__
        assert isinstance(DistributionMetric(name="test", batch_size=4), MetricProtocol)

    def test_sequence_metric_extends_metric_base(self) -> None:
        from artifex.generative_models.core.evaluation.metrics.base import SequenceMetric

        assert issubclass(SequenceMetric, MetricBase)
        assert SequenceMetric.__init__ is MetricBase.__init__
        assert isinstance(SequenceMetric(name="test", batch_size=4), MetricProtocol)
