"""StyleGAN-specific metrics for image generation evaluation.

Key metrics:
- Style mixing quality
- Few-shot adaptation performance
- Translation and rotation equivariance

FID and LPIPS are imported from image.py (canonical location).
"""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from artifex.benchmarks.metrics.core import _init_metric_from_config, MetricBase
from artifex.benchmarks.metrics.image import FIDMetric
from artifex.benchmarks.runtime_guards import demo_mode_from_mapping, require_demo_mode
from artifex.generative_models.core.configuration import EvaluationConfig


class InceptionV3Feature(nnx.Module):
    """Mock Inception V3 for feature extraction (simplified for development)."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Initialize InceptionV3Feature.

        Args:
            rngs: Random number generators
        """
        super().__init__()

        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=[(3, 3), (3, 3)],
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=64,
            out_features=192,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=[(1, 1), (1, 1)],
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=192,
            out_features=512,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=[(1, 1), (1, 1)],
            rngs=rngs,
        )
        self.conv4 = nnx.Conv(
            in_features=512,
            out_features=2048,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=[(1, 1), (1, 1)],
            rngs=rngs,
        )

        self.final_linear = nnx.Linear(in_features=2048, out_features=2048, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Extract features from images.

        Args:
            x: Input images of shape (batch_size, height, width, channels)

        Returns:
            Feature vectors of shape (batch_size, 2048)
        """
        x = (x + 1.0) / 2.0

        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = nnx.relu(x)
        x = self.conv3(x)
        x = nnx.relu(x)
        x = self.conv4(x)
        x = nnx.relu(x)

        x = jnp.mean(x, axis=(1, 2))
        x = self.final_linear(x)

        return x


class StyleMixingMetric(MetricBase):
    """Metric to evaluate style mixing quality in StyleGAN."""

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig) -> None:
        """Initialize StyleMixingMetric.

        Args:
            rngs: Random number generators
            config: Evaluation configuration
        """
        _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="style_mixing",
            modality="image",
            higher_is_better=True,
        )
        demo_mode = bool(getattr(config, "metadata", {}).get("demo_mode", False)) or (
            demo_mode_from_mapping(config.metric_params)
        )
        require_demo_mode(
            enabled=demo_mode,
            component="StyleMixingMetric",
            detail=(
                "This retained StyleGAN benchmark metric stack is demo-only and uses lightweight "
                "local evaluation backends rather than a benchmark-grade runtime."
            ),
        )
        self._lpips = _create_lpips(rngs)

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute style mixing quality.

        Args:
            real_data: Not used directly
            generated_data: Not used directly
            **kwargs: Must include 'generator' and 'rngs'

        Returns:
            Dictionary with style_mixing_quality
        """
        generator = kwargs.get("generator")
        rngs = kwargs.get("rngs", self.rngs)
        num_samples = kwargs.get("num_samples", 100)

        if generator is None:
            return {"style_mixing_quality": 0.0}

        return self.compute_style_mixing_quality(generator, num_samples=num_samples, rngs=rngs)

    def validate_inputs(self, real_data, generated_data) -> None:
        """Validate inputs (style mixing operates on generators, not data).

        Raises:
            ValueError: If inputs are invalid
        """

    def compute_style_mixing_quality(
        self, generator, num_samples: int = 100, *, rngs: nnx.Rngs
    ) -> dict[str, Any]:
        """Compute style mixing quality metrics.

        Args:
            generator: StyleGAN3 generator
            num_samples: Number of samples for evaluation
            rngs: Random number generators

        Returns:
            Dictionary of style mixing metrics including per-layer details
        """
        batch_size = min(16, num_samples)

        z1 = jax.random.normal(rngs.sample(), (batch_size, generator.latent_dim))
        z2 = jax.random.normal(
            jax.random.fold_in(rngs.sample(), 1), (batch_size, generator.latent_dim)
        )

        img1 = generator(z1, rngs=rngs)
        img2 = generator(z2, rngs=rngs)

        mixing_results = []

        for mixing_layer in [4, 8, 12]:
            w1 = generator.mapping(z1)
            w2 = generator.mapping(z2)

            w_mixed = w1.copy()
            w_mixed = w_mixed.at[:, mixing_layer:].set(w2[:, mixing_layer:])

            img_mixed = generator.synthesis(w_mixed, rngs=rngs)

            dist_to_source = self._lpips.compute_distance(img1, img_mixed)
            dist_to_target = self._lpips.compute_distance(img2, img_mixed)

            mixing_results.append(
                {
                    "layer": mixing_layer,
                    "dist_to_source": float(jnp.mean(dist_to_source)),
                    "dist_to_target": float(jnp.mean(dist_to_target)),
                    "mixing_ratio": float(
                        jnp.mean(dist_to_source / (dist_to_source + dist_to_target + 1e-8))
                    ),
                }
            )

        avg_mixing_quality = np.mean([r["mixing_ratio"] for r in mixing_results])

        return {
            "style_mixing_quality": float(avg_mixing_quality),
            "mixing_details": mixing_results,
            "num_samples": num_samples,
        }


class FewShotAdaptationMetric(MetricBase):
    """Metric to evaluate few-shot domain adaptation performance."""

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig) -> None:
        """Initialize FewShotAdaptationMetric.

        Args:
            rngs: Random number generators
            config: Evaluation configuration
        """
        _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="few_shot",
            modality="image",
            higher_is_better=True,
        )
        demo_mode = bool(getattr(config, "metadata", {}).get("demo_mode", False)) or (
            demo_mode_from_mapping(config.metric_params)
        )
        require_demo_mode(
            enabled=demo_mode,
            component="FewShotAdaptationMetric",
            detail=(
                "This retained StyleGAN benchmark metric stack is demo-only and uses lightweight "
                "local evaluation backends rather than a benchmark-grade runtime."
            ),
        )
        self._fid_feature = InceptionV3Feature(rngs=rngs)
        self._lpips = _create_lpips(rngs)

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute few-shot adaptation metrics.

        Args:
            real_data: Not used directly
            generated_data: Not used directly
            **kwargs: Must include generators and dataset

        Returns:
            Dictionary with adaptation metrics
        """
        return {
            "fid_improvement": 0.0,
            "lpips_improvement": 0.0,
            "adaptation_quality": 0.0,
        }

    def validate_inputs(self, real_data, generated_data) -> None:
        """Validate inputs.

        Raises:
            ValueError: If inputs are invalid
        """

    def evaluate_adaptation(
        self,
        original_generator,
        adapted_generator,
        target_dataset,
        num_samples: int = 1000,
        *,
        rngs: nnx.Rngs,
    ) -> dict[str, float]:
        """Evaluate few-shot adaptation performance.

        Args:
            original_generator: Original pre-trained generator
            adapted_generator: Generator adapted to target domain
            target_dataset: Target domain dataset
            num_samples: Number of samples for evaluation
            rngs: Random number generators

        Returns:
            Dictionary of adaptation metrics
        """
        batch_size = 32

        z = jax.random.normal(rngs.sample(), (num_samples, original_generator.latent_dim))

        num_batches = num_samples // batch_size

        original_images: list[jnp.ndarray] = []
        adapted_images: list[jnp.ndarray] = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            z_batch = z[start_idx:end_idx]

            orig_batch = original_generator(z_batch, rngs=rngs)
            adapt_batch = adapted_generator(z_batch, rngs=rngs)

            original_images.append(orig_batch)
            adapted_images.append(adapt_batch)

        if not original_images or not adapted_images:
            return {
                "fid_improvement": 0.0,
                "lpips_improvement": 0.0,
                "adaptation_quality": 0.0,
            }

        original_images_array = jnp.concatenate(original_images, axis=0)
        adapted_images_array = jnp.concatenate(adapted_images, axis=0)

        target_images: list[jnp.ndarray] = []
        target_batches = list(target_dataset(batch_size))

        for batch in target_batches[:num_batches]:
            target_images.append(batch["images"])

        if not target_images:
            return {
                "fid_improvement": 0.0,
                "lpips_improvement": 0.0,
                "adaptation_quality": 0.0,
            }

        target_images_array = jnp.concatenate(target_images, axis=0)

        target_features = self._fid_feature(target_images_array)
        orig_features = self._fid_feature(original_images_array)
        adapt_features = self._fid_feature(adapted_images_array)

        target_mean = jnp.mean(target_features, axis=0)
        orig_mean = jnp.mean(orig_features, axis=0)
        adapt_mean = jnp.mean(adapt_features, axis=0)

        fid_original = float(jnp.sum((target_mean - orig_mean) ** 2))
        fid_adapted = float(jnp.sum((target_mean - adapt_mean) ** 2))

        fid_improvement = fid_original - fid_adapted

        lpips_original = jnp.mean(
            self._lpips.compute_distance(
                original_images_array[:batch_size],
                original_images_array[batch_size : 2 * batch_size],
            )
        )
        lpips_adapted = jnp.mean(
            self._lpips.compute_distance(
                adapted_images_array[:batch_size],
                adapted_images_array[batch_size : 2 * batch_size],
            )
        )

        diversity_preservation = lpips_adapted / (lpips_original + 1e-8)

        return {
            "fid_original": fid_original,
            "fid_adapted": fid_adapted,
            "fid_improvement": fid_improvement,
            "diversity_preservation": float(diversity_preservation),
            "adaptation_success": (
                1.0 if (fid_improvement > 0 and diversity_preservation > 0.8) else 0.0
            ),
            "num_samples": num_samples,
        }


class EquivarianceMetric(MetricBase):
    """Metric to evaluate translation and rotation equivariance."""

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig) -> None:
        """Initialize EquivarianceMetric.

        Args:
            rngs: Random number generators
            config: Evaluation configuration
        """
        _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="equivariance",
            modality="image",
            higher_is_better=True,
        )
        demo_mode = bool(getattr(config, "metadata", {}).get("demo_mode", False)) or (
            demo_mode_from_mapping(config.metric_params)
        )
        require_demo_mode(
            enabled=demo_mode,
            component="EquivarianceMetric",
            detail=(
                "This retained StyleGAN benchmark metric stack is demo-only and uses lightweight "
                "local evaluation backends rather than a benchmark-grade runtime."
            ),
        )
        self._lpips = _create_lpips(rngs)

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute equivariance metrics.

        Args:
            real_data: Not used directly
            generated_data: Not used directly
            **kwargs: Must include 'generator' and 'rngs'

        Returns:
            Dictionary with equivariance metrics
        """
        generator = kwargs.get("generator")
        rngs = kwargs.get("rngs", self.rngs)

        if generator is None:
            return {"overall_equivariance": 0.0}

        return self.evaluate_equivariance(generator, rngs=rngs)

    def validate_inputs(self, real_data, generated_data) -> None:
        """Validate inputs (equivariance operates on generators).

        Raises:
            ValueError: If inputs are invalid
        """

    def evaluate_equivariance(
        self, generator, num_samples: int = 100, *, rngs: nnx.Rngs
    ) -> dict[str, float]:
        """Evaluate translation and rotation equivariance.

        Args:
            generator: StyleGAN3 generator to test
            num_samples: Number of samples for evaluation
            rngs: Random number generators

        Returns:
            Dictionary of equivariance metrics
        """
        batch_size = min(16, num_samples)

        z = jax.random.normal(rngs.sample(), (batch_size, generator.latent_dim))
        base_images = generator(z, rngs=rngs)

        translation_offset = 0.1
        translated_images = self._apply_translation(base_images, translation_offset)

        translation_consistency = jnp.clip(
            1.0 - jnp.mean(self._lpips.compute_distance(base_images, translated_images)),
            0.0,
            1.0,
        )

        rotation_angle = jnp.pi / 12
        rotated_images = self._apply_rotation(base_images, rotation_angle)

        rotation_consistency = jnp.clip(
            1.0 - jnp.mean(self._lpips.compute_distance(base_images, rotated_images)),
            0.0,
            1.0,
        )

        return {
            "translation_equivariance": float(translation_consistency),
            "rotation_equivariance": float(rotation_consistency),
            "overall_equivariance": float((translation_consistency + rotation_consistency) / 2),
            "num_samples": num_samples,
        }

    def _apply_translation(self, images: jnp.ndarray, offset: float) -> jnp.ndarray:
        """Apply translation to images (mock implementation)."""
        shift_pixels = int(offset * images.shape[1])

        padded = jnp.pad(
            images,
            ((0, 0), (shift_pixels, shift_pixels), (shift_pixels, shift_pixels), (0, 0)),
            mode="edge",
        )

        h, w = images.shape[1], images.shape[2]
        translated = padded[:, shift_pixels : shift_pixels + h, shift_pixels : shift_pixels + w, :]

        return translated

    def _apply_rotation(self, images: jnp.ndarray, angle: float) -> jnp.ndarray:
        """Apply rotation to images (mock implementation)."""
        noise_factor = 0.05 * abs(angle)
        noise = noise_factor * jax.random.normal(jax.random.key(42), images.shape)

        return jnp.clip(images + noise, -1.0, 1.0)


class StyleGANMetrics(MetricBase):
    """StyleGAN evaluation metrics suite.

    Composes FID, LPIPS, style mixing, equivariance, and few-shot metrics.
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig) -> None:
        """Initialize StyleGANMetrics.

        Args:
            rngs: Random number generators
            config: Evaluation configuration
        """
        _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key=None,
            modality="image",
            higher_is_better=True,
        )
        demo_mode = bool(getattr(config, "metadata", {}).get("demo_mode", False)) or (
            demo_mode_from_mapping(config.metric_params)
        )
        require_demo_mode(
            enabled=demo_mode,
            component="StyleGANMetrics",
            detail=(
                "This retained StyleGAN benchmark metric stack is demo-only and uses lightweight "
                "local evaluation backends rather than a benchmark-grade runtime."
            ),
        )
        self.fid_feature = InceptionV3Feature(rngs=rngs)
        fid_config = _make_style_config("fid", higher_is_better=False)
        self.fid_metric = FIDMetric(rngs=rngs, config=fid_config)
        style_config = _make_style_config("style_mixing", higher_is_better=True)
        equiv_config = _make_style_config("equivariance", higher_is_better=True)
        few_shot_config = _make_style_config("few_shot", higher_is_better=True)
        self.style_mixing_metric = StyleMixingMetric(rngs=rngs, config=style_config)
        self.equivariance_metric = EquivarianceMetric(rngs=rngs, config=equiv_config)
        self.few_shot_metric = FewShotAdaptationMetric(rngs=rngs, config=few_shot_config)
        self._lpips = _create_lpips(rngs)

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute all StyleGAN metrics.

        Args:
            real_data: Real images
            generated_data: Generated images
            **kwargs: Must include 'generator' and 'rngs'

        Returns:
            Dictionary of all computed metrics
        """
        generator = kwargs.get("generator")
        rngs = kwargs.get("rngs", self.rngs)
        metrics: dict[str, float] = {}

        # FID via InceptionV3Feature
        real_features = self.fid_feature(real_data)
        fake_features = self.fid_feature(generated_data)
        real_mean = jnp.mean(real_features, axis=0)
        fake_mean = jnp.mean(fake_features, axis=0)
        metrics["fid"] = float(jnp.sum((real_mean - fake_mean) ** 2))

        if generator is not None:
            style_metrics = self.style_mixing_metric.compute_style_mixing_quality(
                generator, rngs=rngs
            )
            metrics.update(style_metrics)

            equivariance_metrics = self.equivariance_metric.evaluate_equivariance(
                generator, rngs=rngs
            )
            metrics.update(equivariance_metrics)

        return metrics

    def compute_all_metrics(
        self,
        generator,
        real_images: jnp.ndarray,
        generated_images: jnp.ndarray,
        target_dataset=None,
        adapted_generator=None,
        *,
        rngs: nnx.Rngs,
    ) -> dict[str, float]:
        """Compute all StyleGAN metrics (delegates to compute).

        Args:
            generator: StyleGAN3 generator
            real_images: Real images
            generated_images: Generated images
            target_dataset: Optional target for few-shot evaluation
            adapted_generator: Optional adapted generator
            rngs: Random number generators

        Returns:
            Dictionary of all computed metrics
        """
        return self.compute(
            real_images,
            generated_images,
            generator=generator,
            rngs=rngs,
        )

    def validate_inputs(self, real_data, generated_data) -> None:
        """Validate inputs for StyleGAN metrics.

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(real_data, jnp.ndarray) or not isinstance(generated_data, jnp.ndarray):
            raise ValueError("Both inputs must be jax arrays")
        if real_data.ndim != 4 or generated_data.ndim != 4:
            raise ValueError("Images must be 4D (batch, height, width, channels)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StyleLPIPS(nnx.Module):
    """Lightweight LPIPS for style metrics (uses perceptual network directly)."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=64,
            kernel_size=(3, 3),
            padding=[(1, 1), (1, 1)],
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=64,
            out_features=128,
            kernel_size=(3, 3),
            padding=[(1, 1), (1, 1)],
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=128,
            out_features=256,
            kernel_size=(3, 3),
            padding=[(1, 1), (1, 1)],
            rngs=rngs,
        )

    def compute_distance(self, images1: jnp.ndarray, images2: jnp.ndarray) -> jnp.ndarray:
        """Compute perceptual distance between image pairs."""
        features1 = self._extract(images1)
        features2 = self._extract(images2)

        total_distance = jnp.zeros(images1.shape[0])
        for f1, f2 in zip(features1, features2):
            feat_distance = jnp.sqrt(jnp.sum((f1 - f2) ** 2, axis=(1, 2, 3)))
            total_distance += feat_distance

        return total_distance / len(features1)

    def _extract(self, x: jnp.ndarray) -> list[jnp.ndarray]:
        features = []
        x = nnx.relu(self.conv1(x))
        features.append(x)
        x = nnx.relu(self.conv2(x))
        features.append(x)
        x = nnx.relu(self.conv3(x))
        features.append(x)
        return features


def _create_lpips(rngs: nnx.Rngs) -> _StyleLPIPS:
    """Create a lightweight LPIPS for style metric internals."""
    return _StyleLPIPS(rngs=rngs)


def _make_style_config(name: str, *, higher_is_better: bool = True) -> EvaluationConfig:
    """Create a minimal EvaluationConfig for a style metric."""
    metric_params: dict[str, object] = {"higher_is_better": higher_is_better, "demo_mode": True}
    if name == "fid":
        metric_params = {
            "fid": {
                "higher_is_better": higher_is_better,
                "mock_inception": True,
                "demo_mode": True,
            }
        }
    elif name == "lpips":
        metric_params = {
            "lpips": {
                "higher_is_better": higher_is_better,
                "mock_implementation": True,
                "demo_mode": True,
            }
        }
    return EvaluationConfig(
        name=name,
        metrics=[name],
        metric_params=metric_params,
        eval_batch_size=4,
        metadata={"demo_mode": True},
    )
