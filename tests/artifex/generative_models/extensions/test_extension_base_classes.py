"""Tests for extension base classes hierarchy.

This module tests the extension base class hierarchy following TDD principles.
Tests are written first to define expected behavior, then implementation follows.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    AugmentationExtensionConfig,
    CallbackExtensionConfig,
    ConstraintExtensionConfig,
    EvaluationExtensionConfig,
    ExtensionConfig,
    LossExtensionConfig,
    ModalityExtensionConfig,
    SamplingExtensionConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_rngs():
    """Create mock random number generator keys."""
    return nnx.Rngs(0)


@pytest.fixture
def extension_config():
    """Create a basic extension configuration."""
    return ExtensionConfig(name="test_extension", weight=1.0, enabled=True)


@pytest.fixture
def constraint_config():
    """Create a constraint extension configuration."""
    return ConstraintExtensionConfig(
        name="test_constraint",
        weight=1.0,
        tolerance=0.01,
        projection_enabled=True,
    )


@pytest.fixture
def augmentation_config():
    """Create an augmentation extension configuration."""
    return AugmentationExtensionConfig(
        name="test_augmentation",
        probability=0.5,
        deterministic=False,
    )


@pytest.fixture
def sampling_config():
    """Create a sampling extension configuration."""
    return SamplingExtensionConfig(
        name="test_sampling",
        guidance_scale=7.5,
        temperature=0.8,
    )


@pytest.fixture
def loss_config():
    """Create a loss extension configuration."""
    return LossExtensionConfig(
        name="test_loss",
        weight=0.5,
        weight_schedule="linear",
        warmup_steps=100,
    )


@pytest.fixture
def evaluation_config():
    """Create an evaluation extension configuration."""
    return EvaluationExtensionConfig(
        name="test_eval",
        compute_on_train=True,
        compute_on_eval=True,
    )


@pytest.fixture
def callback_config():
    """Create a callback extension configuration."""
    return CallbackExtensionConfig(
        name="test_callback",
        frequency=10,
        on_train=True,
        on_eval=True,
    )


@pytest.fixture
def modality_config():
    """Create a modality extension configuration."""
    return ModalityExtensionConfig(
        name="test_modality",
        input_key="images",
        output_key="reconstructed",
    )


# =============================================================================
# Extension Base Class Tests
# =============================================================================


class TestExtensionBase:
    """Tests for the base Extension class."""

    def test_extension_is_nnx_module(self, mock_rngs, extension_config):
        """Extension should inherit from nnx.Module."""
        from artifex.generative_models.extensions.base import Extension

        extension = Extension(extension_config, rngs=mock_rngs)
        assert isinstance(extension, nnx.Module)

    def test_extension_stores_config(self, mock_rngs, extension_config):
        """Extension should store the configuration."""
        from artifex.generative_models.extensions.base import Extension

        extension = Extension(extension_config, rngs=mock_rngs)
        assert extension.config == extension_config
        assert extension.config.name == "test_extension"

    def test_extension_is_enabled(self, mock_rngs, extension_config):
        """Extension should have is_enabled method."""
        from artifex.generative_models.extensions.base import Extension

        extension = Extension(extension_config, rngs=mock_rngs)
        assert extension.is_enabled() is True

    def test_extension_disabled(self, mock_rngs):
        """Disabled extension should return False for is_enabled."""
        from artifex.generative_models.extensions.base import Extension

        config = ExtensionConfig(name="disabled", enabled=False)
        extension = Extension(config, rngs=mock_rngs)
        assert extension.is_enabled() is False

    def test_extension_weight_property(self, mock_rngs, extension_config):
        """Extension should have weight property from config."""
        from artifex.generative_models.extensions.base import Extension

        extension = Extension(extension_config, rngs=mock_rngs)
        assert extension.weight == 1.0

    def test_extension_requires_frozen_dataclass_config(self, mock_rngs):
        """Extension should only accept frozen dataclass configs."""
        from artifex.generative_models.extensions.base import Extension

        # Should raise TypeError for dict config
        with pytest.raises(TypeError, match="config must be"):
            Extension({"name": "test"}, rngs=mock_rngs)


# =============================================================================
# ModelExtension Tests
# =============================================================================


class TestModelExtension:
    """Tests for ModelExtension class."""

    def test_model_extension_init(self, mock_rngs, extension_config):
        """ModelExtension should initialize correctly."""
        from artifex.generative_models.extensions.base import ModelExtension

        class TestModelExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {"processed": True}

        extension = TestModelExtension(extension_config, rngs=mock_rngs)
        assert extension.weight == 1.0
        assert extension.is_enabled() is True

    def test_model_extension_call_abstract(self, mock_rngs, extension_config):
        """ModelExtension.__call__ should raise NotImplementedError."""
        from artifex.generative_models.extensions.base import ModelExtension

        extension = ModelExtension(extension_config, rngs=mock_rngs)
        with pytest.raises(NotImplementedError):
            extension({}, {})

    def test_model_extension_loss_fn_default(self, mock_rngs, extension_config):
        """ModelExtension.loss_fn should return 0.0 by default."""
        from artifex.generative_models.extensions.base import ModelExtension

        class TestModelExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        extension = TestModelExtension(extension_config, rngs=mock_rngs)
        loss = extension.loss_fn({}, {})
        assert jnp.allclose(loss, jnp.array(0.0))

    def test_model_extension_loss_fn_disabled(self, mock_rngs):
        """Disabled ModelExtension should return 0.0 loss."""
        from artifex.generative_models.extensions.base import ModelExtension

        config = ExtensionConfig(name="test", enabled=False, weight=5.0)

        class TestModelExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

            def loss_fn(self, batch, model_outputs, **kwargs):
                if not self.is_enabled():
                    return jnp.array(0.0)
                return jnp.array(self.weight)

        extension = TestModelExtension(config, rngs=mock_rngs)
        loss = extension.loss_fn({}, {})
        assert jnp.allclose(loss, jnp.array(0.0))

    def test_model_extension_inheritance(self, mock_rngs, extension_config):
        """ModelExtension should inherit from Extension."""
        from artifex.generative_models.extensions.base import Extension, ModelExtension

        class TestModelExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        extension = TestModelExtension(extension_config, rngs=mock_rngs)
        assert isinstance(extension, Extension)
        assert isinstance(extension, nnx.Module)


# =============================================================================
# ConstraintExtension Tests
# =============================================================================


class TestConstraintExtension:
    """Tests for ConstraintExtension class."""

    def test_constraint_extension_init(self, mock_rngs, constraint_config):
        """ConstraintExtension should initialize correctly."""
        from artifex.generative_models.extensions.base import ConstraintExtension

        class TestConstraintExtension(ConstraintExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

            def validate(self, outputs):
                return {"valid": True}

        extension = TestConstraintExtension(constraint_config, rngs=mock_rngs)
        assert extension.config.tolerance == 0.01
        assert extension.config.projection_enabled is True

    def test_constraint_extension_validate_abstract(self, mock_rngs, constraint_config):
        """ConstraintExtension.validate should be abstract."""
        from artifex.generative_models.extensions.base import ConstraintExtension

        class TestConstraintExtension(ConstraintExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        extension = TestConstraintExtension(constraint_config, rngs=mock_rngs)
        with pytest.raises(NotImplementedError):
            extension.validate({})

    def test_constraint_extension_project_default(self, mock_rngs, constraint_config):
        """ConstraintExtension.project should return outputs unchanged by default."""
        from artifex.generative_models.extensions.base import ConstraintExtension

        class TestConstraintExtension(ConstraintExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

            def validate(self, outputs):
                return {}

        extension = TestConstraintExtension(constraint_config, rngs=mock_rngs)
        outputs = {"positions": jnp.array([1.0, 2.0, 3.0])}
        projected = extension.project(outputs)
        assert jnp.allclose(projected["positions"], outputs["positions"])

    def test_constraint_extension_project_disabled(self, mock_rngs):
        """Disabled ConstraintExtension.project should return unchanged outputs."""
        from artifex.generative_models.extensions.base import ConstraintExtension

        config = ConstraintExtensionConfig(name="test", enabled=False, projection_enabled=True)

        class TestConstraintExtension(ConstraintExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

            def validate(self, outputs):
                return {}

            def project(self, outputs):
                if not self.is_enabled():
                    return outputs
                return {"modified": True}

        extension = TestConstraintExtension(config, rngs=mock_rngs)
        outputs = {"original": True}
        projected = extension.project(outputs)
        assert projected is outputs

    def test_constraint_extension_inheritance(self, mock_rngs, constraint_config):
        """ConstraintExtension should inherit from ModelExtension."""
        from artifex.generative_models.extensions.base import (
            ConstraintExtension,
            ModelExtension,
        )

        class TestConstraintExtension(ConstraintExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

            def validate(self, outputs):
                return {}

        extension = TestConstraintExtension(constraint_config, rngs=mock_rngs)
        assert isinstance(extension, ModelExtension)


# =============================================================================
# AugmentationExtension Tests
# =============================================================================


class TestAugmentationExtension:
    """Tests for AugmentationExtension class."""

    def test_augmentation_extension_init(self, mock_rngs, augmentation_config):
        """AugmentationExtension should initialize correctly."""
        from artifex.generative_models.extensions.base import AugmentationExtension

        class TestAugmentExtension(AugmentationExtension):
            def augment(self, data, *, key=None, deterministic=False):
                return data

        extension = TestAugmentExtension(augmentation_config, rngs=mock_rngs)
        assert extension.config.probability == 0.5
        assert extension.config.deterministic is False

    def test_augmentation_extension_augment_abstract(self, mock_rngs, augmentation_config):
        """AugmentationExtension.augment should be abstract."""
        from artifex.generative_models.extensions.base import AugmentationExtension

        extension = AugmentationExtension(augmentation_config, rngs=mock_rngs)
        with pytest.raises(NotImplementedError):
            extension.augment(jnp.zeros((4, 32, 32, 3)))

    def test_augmentation_extension_call_applies_augment(self, mock_rngs, augmentation_config):
        """AugmentationExtension.__call__ should apply augmentation."""
        from artifex.generative_models.extensions.base import AugmentationExtension

        class TestAugmentExtension(AugmentationExtension):
            def augment(self, data, *, key=None, deterministic=False):
                return data * 2.0

        extension = TestAugmentExtension(augmentation_config, rngs=mock_rngs)
        data = jnp.ones((4, 32, 32, 3))
        result = extension(data, deterministic=False)
        assert jnp.allclose(result, data * 2.0)

    def test_augmentation_extension_deterministic(self, mock_rngs, augmentation_config):
        """Deterministic augmentation should be reproducible."""
        from artifex.generative_models.extensions.base import AugmentationExtension

        class TestAugmentExtension(AugmentationExtension):
            def augment(self, data, *, key=None, deterministic=False):
                if deterministic:
                    return data
                if key is not None:
                    noise = jax.random.normal(key, data.shape)
                    return data + noise * 0.1
                return data

        extension = TestAugmentExtension(augmentation_config, rngs=mock_rngs)
        data = jnp.ones((4, 32, 32, 3))

        # Deterministic should return unchanged
        result_det = extension(data, deterministic=True)
        assert jnp.allclose(result_det, data)

    def test_augmentation_extension_inheritance(self, mock_rngs, augmentation_config):
        """AugmentationExtension should inherit from Extension."""
        from artifex.generative_models.extensions.base import (
            AugmentationExtension,
            Extension,
        )

        class TestAugmentExtension(AugmentationExtension):
            def augment(self, data, *, key=None, deterministic=False):
                return data

        extension = TestAugmentExtension(augmentation_config, rngs=mock_rngs)
        assert isinstance(extension, Extension)


# =============================================================================
# SamplingExtension Tests
# =============================================================================


class TestSamplingExtension:
    """Tests for SamplingExtension class."""

    def test_sampling_extension_init(self, mock_rngs, sampling_config):
        """SamplingExtension should initialize correctly."""
        from artifex.generative_models.extensions.base import SamplingExtension

        extension = SamplingExtension(sampling_config, rngs=mock_rngs)
        assert extension.config.guidance_scale == 7.5
        assert extension.config.temperature == 0.8

    def test_sampling_extension_modify_score_default(self, mock_rngs, sampling_config):
        """SamplingExtension.modify_score should return score unchanged by default."""
        from artifex.generative_models.extensions.base import SamplingExtension

        extension = SamplingExtension(sampling_config, rngs=mock_rngs)
        score = jnp.ones((4, 64))
        timestep = jnp.array([0.5])
        context = {}

        modified = extension.modify_score(score, timestep, context)
        assert jnp.allclose(modified, score)

    def test_sampling_extension_filter_samples_default(self, mock_rngs, sampling_config):
        """SamplingExtension.filter_samples should pass all samples by default."""
        from artifex.generative_models.extensions.base import SamplingExtension

        extension = SamplingExtension(sampling_config, rngs=mock_rngs)
        samples = jnp.ones((8, 64))
        context = {}

        filtered, mask = extension.filter_samples(samples, context)
        assert jnp.allclose(filtered, samples)
        assert jnp.all(mask == 1.0)

    def test_sampling_extension_post_process_sample_default(self, mock_rngs, sampling_config):
        """SamplingExtension.post_process_sample should return sample unchanged."""
        from artifex.generative_models.extensions.base import SamplingExtension

        extension = SamplingExtension(sampling_config, rngs=mock_rngs)
        sample = jnp.ones((1, 64))
        context = {}

        processed = extension.post_process_sample(sample, context)
        assert jnp.allclose(processed, sample)

    def test_sampling_extension_inheritance(self, mock_rngs, sampling_config):
        """SamplingExtension should inherit from Extension."""
        from artifex.generative_models.extensions.base import Extension, SamplingExtension

        extension = SamplingExtension(sampling_config, rngs=mock_rngs)
        assert isinstance(extension, Extension)


# =============================================================================
# LossExtension Tests
# =============================================================================


class TestLossExtension:
    """Tests for LossExtension class."""

    def test_loss_extension_init(self, mock_rngs, loss_config):
        """LossExtension should initialize correctly."""
        from artifex.generative_models.extensions.base import LossExtension

        class TestLossExtension(LossExtension):
            def compute_loss(self, predictions, targets, context):
                return jnp.mean(jnp.square(predictions - targets)), {}

        extension = TestLossExtension(loss_config, rngs=mock_rngs)
        assert extension.config.weight_schedule == "linear"
        assert extension.config.warmup_steps == 100

    def test_loss_extension_compute_loss_abstract(self, mock_rngs, loss_config):
        """LossExtension.compute_loss should be abstract."""
        from artifex.generative_models.extensions.base import LossExtension

        extension = LossExtension(loss_config, rngs=mock_rngs)
        with pytest.raises(NotImplementedError):
            extension.compute_loss(jnp.ones(4), jnp.zeros(4), {})

    def test_loss_extension_get_weight_at_step_constant(self, mock_rngs):
        """LossExtension with constant schedule should return fixed weight."""
        from artifex.generative_models.extensions.base import LossExtension

        config = LossExtensionConfig(
            name="test", weight=0.5, weight_schedule="constant", warmup_steps=0
        )

        class TestLossExtension(LossExtension):
            def compute_loss(self, predictions, targets, context):
                return jnp.array(0.0), {}

        extension = TestLossExtension(config, rngs=mock_rngs)
        assert extension.get_weight_at_step(0) == 0.5
        assert extension.get_weight_at_step(100) == 0.5
        assert extension.get_weight_at_step(1000) == 0.5

    def test_loss_extension_get_weight_at_step_linear(self, mock_rngs):
        """LossExtension with linear schedule should ramp up weight."""
        from artifex.generative_models.extensions.base import LossExtension

        config = LossExtensionConfig(
            name="test", weight=1.0, weight_schedule="linear", warmup_steps=100
        )

        class TestLossExtension(LossExtension):
            def compute_loss(self, predictions, targets, context):
                return jnp.array(0.0), {}

        extension = TestLossExtension(config, rngs=mock_rngs)
        assert extension.get_weight_at_step(0) == 0.0
        assert 0.4 < extension.get_weight_at_step(50) < 0.6  # ~0.5
        assert extension.get_weight_at_step(100) == 1.0
        assert extension.get_weight_at_step(200) == 1.0  # Clamped at max

    def test_loss_extension_inheritance(self, mock_rngs, loss_config):
        """LossExtension should inherit from Extension."""
        from artifex.generative_models.extensions.base import Extension, LossExtension

        class TestLossExtension(LossExtension):
            def compute_loss(self, predictions, targets, context):
                return jnp.array(0.0), {}

        extension = TestLossExtension(loss_config, rngs=mock_rngs)
        assert isinstance(extension, Extension)


# =============================================================================
# EvaluationExtension Tests
# =============================================================================


class TestEvaluationExtension:
    """Tests for EvaluationExtension class."""

    def test_evaluation_extension_init(self, mock_rngs, evaluation_config):
        """EvaluationExtension should initialize correctly."""
        from artifex.generative_models.extensions.base import EvaluationExtension

        class TestEvalExtension(EvaluationExtension):
            def compute_metrics(self, generated, reference=None):
                return {"metric": 1.0}

        extension = TestEvalExtension(evaluation_config, rngs=mock_rngs)
        assert extension.config.compute_on_train is True
        assert extension.config.compute_on_eval is True

    def test_evaluation_extension_compute_metrics_abstract(self, mock_rngs, evaluation_config):
        """EvaluationExtension.compute_metrics should be abstract."""
        from artifex.generative_models.extensions.base import EvaluationExtension

        extension = EvaluationExtension(evaluation_config, rngs=mock_rngs)
        with pytest.raises(NotImplementedError):
            extension.compute_metrics(jnp.ones((4, 64)))

    def test_evaluation_extension_returns_dict(self, mock_rngs, evaluation_config):
        """EvaluationExtension.compute_metrics should return dict of metrics."""
        from artifex.generative_models.extensions.base import EvaluationExtension

        class TestEvalExtension(EvaluationExtension):
            def compute_metrics(self, generated, reference=None):
                mse = jnp.mean(jnp.square(generated))
                return {"mse": float(mse), "count": generated.shape[0]}

        extension = TestEvalExtension(evaluation_config, rngs=mock_rngs)
        metrics = extension.compute_metrics(jnp.ones((4, 64)))
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "count" in metrics

    def test_evaluation_extension_inheritance(self, mock_rngs, evaluation_config):
        """EvaluationExtension should inherit from Extension."""
        from artifex.generative_models.extensions.base import (
            EvaluationExtension,
            Extension,
        )

        class TestEvalExtension(EvaluationExtension):
            def compute_metrics(self, generated, reference=None):
                return {}

        extension = TestEvalExtension(evaluation_config, rngs=mock_rngs)
        assert isinstance(extension, Extension)


# =============================================================================
# CallbackExtension Tests
# =============================================================================


class TestCallbackExtension:
    """Tests for CallbackExtension class."""

    def test_callback_extension_init(self, mock_rngs, callback_config):
        """CallbackExtension should initialize correctly."""
        from artifex.generative_models.extensions.base import CallbackExtension

        extension = CallbackExtension(callback_config, rngs=mock_rngs)
        assert extension.config.frequency == 10
        assert extension.config.on_train is True
        assert extension.config.on_eval is True

    def test_callback_extension_lifecycle_methods(self, mock_rngs, callback_config):
        """CallbackExtension should have lifecycle hook methods."""
        from artifex.generative_models.extensions.base import CallbackExtension

        extension = CallbackExtension(callback_config, rngs=mock_rngs)

        # All lifecycle methods should exist and be callable
        # They should not raise errors when called
        extension.on_train_begin(trainer=None)
        extension.on_train_end(trainer=None)
        extension.on_epoch_begin(trainer=None, epoch=0)
        extension.on_epoch_end(trainer=None, epoch=0, logs={})
        extension.on_batch_begin(trainer=None, batch_idx=0)
        extension.on_batch_end(trainer=None, batch_idx=0, logs={})

    def test_callback_extension_inheritance(self, mock_rngs, callback_config):
        """CallbackExtension should inherit from Extension."""
        from artifex.generative_models.extensions.base import CallbackExtension, Extension

        extension = CallbackExtension(callback_config, rngs=mock_rngs)
        assert isinstance(extension, Extension)


# =============================================================================
# ModalityExtension Tests
# =============================================================================


class TestModalityExtension:
    """Tests for ModalityExtension class."""

    def test_modality_extension_init(self, mock_rngs, modality_config):
        """ModalityExtension should initialize correctly."""
        from artifex.generative_models.extensions.base import ModalityExtension

        class TestModalityExtension(ModalityExtension):
            def preprocess(self, raw_data):
                return {"processed": raw_data}

            def postprocess(self, model_output):
                return model_output

        extension = TestModalityExtension(modality_config, rngs=mock_rngs)
        assert extension.config.input_key == "images"
        assert extension.config.output_key == "reconstructed"

    def test_modality_extension_preprocess_abstract(self, mock_rngs, modality_config):
        """ModalityExtension.preprocess should be abstract."""
        from artifex.generative_models.extensions.base import ModalityExtension

        extension = ModalityExtension(modality_config, rngs=mock_rngs)
        with pytest.raises(NotImplementedError):
            extension.preprocess(jnp.ones((4, 32, 32, 3)))

    def test_modality_extension_postprocess_abstract(self, mock_rngs, modality_config):
        """ModalityExtension.postprocess should be abstract."""
        from artifex.generative_models.extensions.base import ModalityExtension

        extension = ModalityExtension(modality_config, rngs=mock_rngs)
        with pytest.raises(NotImplementedError):
            extension.postprocess(jnp.ones((4, 64)))

    def test_modality_extension_get_input_spec(self, mock_rngs, modality_config):
        """ModalityExtension.get_input_spec should return input specification."""
        from artifex.generative_models.extensions.base import ModalityExtension

        class TestModalityExtension(ModalityExtension):
            def preprocess(self, raw_data):
                return {"processed": raw_data}

            def postprocess(self, model_output):
                return model_output

            def get_input_spec(self):
                return {"shape": (None, 32, 32, 3), "dtype": jnp.float32}

        extension = TestModalityExtension(modality_config, rngs=mock_rngs)
        spec = extension.get_input_spec()
        assert "shape" in spec or isinstance(spec, dict)

    def test_modality_extension_inheritance(self, mock_rngs, modality_config):
        """ModalityExtension should inherit from Extension."""
        from artifex.generative_models.extensions.base import Extension, ModalityExtension

        class TestModalityExtension(ModalityExtension):
            def preprocess(self, raw_data):
                return {}

            def postprocess(self, model_output):
                return model_output

        extension = TestModalityExtension(modality_config, rngs=mock_rngs)
        assert isinstance(extension, Extension)


# =============================================================================
# JIT Compatibility Tests
# =============================================================================


class TestExtensionJITCompatibility:
    """Tests for JAX JIT compatibility of extensions."""

    def test_model_extension_loss_fn_jittable(self, mock_rngs, extension_config):
        """ModelExtension.loss_fn should be JIT-compilable."""
        from artifex.generative_models.extensions.base import ModelExtension

        class JittableExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

            def loss_fn(self, batch, model_outputs, **kwargs):
                positions = model_outputs.get("positions", jnp.zeros(1))
                return jnp.mean(jnp.square(positions))

        extension = JittableExtension(extension_config, rngs=mock_rngs)

        @jax.jit
        def compute_loss(outputs):
            return extension.loss_fn({}, outputs)

        outputs = {"positions": jnp.array([1.0, 2.0, 3.0])}
        loss = compute_loss(outputs)
        assert not jnp.isnan(loss)
        assert loss.shape == ()

    def test_augmentation_extension_augment_jittable(self, mock_rngs, augmentation_config):
        """AugmentationExtension.augment should be JIT-compilable."""
        from artifex.generative_models.extensions.base import AugmentationExtension

        class JittableAugmentation(AugmentationExtension):
            def augment(self, data, *, key=None, deterministic=False):
                return data * 2.0

        extension = JittableAugmentation(augmentation_config, rngs=mock_rngs)

        @jax.jit
        def augment_data(data):
            return extension.augment(data, deterministic=True)

        data = jnp.ones((4, 32, 32, 3))
        result = augment_data(data)
        assert jnp.allclose(result, data * 2.0)

    def test_sampling_extension_modify_score_jittable(self, mock_rngs, sampling_config):
        """SamplingExtension.modify_score should be JIT-compilable."""
        from artifex.generative_models.extensions.base import SamplingExtension

        class GuidanceExtension(SamplingExtension):
            def modify_score(self, score, timestep, context):
                scale = self.config.guidance_scale
                uncond = context.get("uncond_score", jnp.zeros_like(score))
                return uncond + scale * (score - uncond)

        extension = GuidanceExtension(sampling_config, rngs=mock_rngs)

        @jax.jit
        def apply_guidance(score, timestep, uncond_score):
            return extension.modify_score(score, timestep, {"uncond_score": uncond_score})

        score = jnp.ones((4, 64))
        uncond = jnp.zeros((4, 64))
        timestep = jnp.array([0.5])

        result = apply_guidance(score, timestep, uncond)
        expected = uncond + 7.5 * (score - uncond)
        assert jnp.allclose(result, expected)


# =============================================================================
# State Management Tests
# =============================================================================


class TestExtensionStateManagement:
    """Tests for extension state management with nnx.state."""

    def test_extension_state_extraction(self, mock_rngs, extension_config):
        """Extension state should be extractable with nnx.state."""
        from artifex.generative_models.extensions.base import Extension

        extension = Extension(extension_config, rngs=mock_rngs)
        state = nnx.state(extension)
        assert state is not None

    def test_model_extension_state_update(self, mock_rngs, extension_config):
        """ModelExtension state should be updatable with nnx.update."""
        from artifex.generative_models.extensions.base import ModelExtension

        class StatefulExtension(ModelExtension):
            def __init__(self, config, *, rngs):
                super().__init__(config, rngs=rngs)
                self.counter = nnx.Variable(jnp.array(0))

            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        extension1 = StatefulExtension(extension_config, rngs=mock_rngs)
        extension1.counter.value = jnp.array(5)

        extension2 = StatefulExtension(extension_config, rngs=nnx.Rngs(1))
        assert int(extension2.counter.value) == 0

        state1 = nnx.state(extension1)
        nnx.update(extension2, state1)
        assert int(extension2.counter.value) == 5
