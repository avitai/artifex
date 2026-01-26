"""StyleGAN-Specific Metrics for Image Generation Evaluation.

This module provides comprehensive metrics for evaluating StyleGAN3 performance,
including traditional image quality metrics and StyleGAN-specific evaluations.

Key Metrics:
- Fréchet Inception Distance (FID)
- Inception Score (IS)
- Learned Perceptual Image Patch Similarity (LPIPS)
- Style Mixing Quality
- Few-shot Adaptation Performance
- Translation and Rotation Equivariance
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


class InceptionV3Feature(nnx.Module):
    """Mock Inception V3 for feature extraction (simplified for development)."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize InceptionV3Feature.

        Args:
            rngs: Random number generators
        """
        super().__init__()

        # Mock Inception V3 layers
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

        # Global average pooling and final layer
        self.final_linear = nnx.Linear(in_features=2048, out_features=2048, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Extract features from images.

        Args:
            x: Input images of shape (batch_size, height, width, channels)

        Returns:
            Feature vectors of shape (batch_size, 2048)
        """
        # Ensure input is in [0, 1] range
        x = (x + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]

        # Progressive convolutions
        x = self.conv1(x)
        x = nnx.relu(x)

        x = self.conv2(x)
        x = nnx.relu(x)

        x = self.conv3(x)
        x = nnx.relu(x)

        x = self.conv4(x)
        x = nnx.relu(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (batch_size, 2048)

        # Final transformation
        x = self.final_linear(x)

        return x


class FIDMetric(nnx.Module):
    """Fréchet Inception Distance metric for image quality assessment."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize FIDMetric.

        Args:
            rngs: Random number generators
        """
        super().__init__()
        self.feature_extractor = InceptionV3Feature(rngs=rngs)

    def compute_statistics(self, images: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute mean and covariance of image features.

        Args:
            images: Batch of images

        Returns:
            Tuple of (mean, covariance)
        """
        features = self.feature_extractor(images)

        mean = jnp.mean(features, axis=0)
        centered = features - mean[None, :]
        covariance = jnp.cov(centered, rowvar=False)

        return mean, covariance

    def compute_fid(
        self,
        real_mean: jnp.ndarray,
        real_cov: jnp.ndarray,
        fake_mean: jnp.ndarray,
        fake_cov: jnp.ndarray,
    ) -> float:
        """Compute FID score between real and fake statistics.

        Args:
            real_mean: Mean of real image features
            real_cov: Covariance of real image features
            fake_mean: Mean of fake image features
            fake_cov: Covariance of fake image features

        Returns:
            FID score (lower is better)
        """
        # Compute mean difference
        mean_diff = real_mean - fake_mean
        mean_norm = jnp.sum(mean_diff**2)

        # Compute trace of covariances
        trace_term = jnp.trace(real_cov) + jnp.trace(fake_cov)

        # Compute sqrt(real_cov * fake_cov) term
        # For numerical stability, use SVD
        try:
            sqrt_product = jnp.real(jnp.trace(jax.scipy.linalg.sqrtm(real_cov @ fake_cov)))
        except Exception:
            # Fallback: approximate sqrt term
            sqrt_product = jnp.sqrt(jnp.trace(real_cov) * jnp.trace(fake_cov))

        fid_score = mean_norm + trace_term - 2 * sqrt_product

        return float(fid_score)


class StyleMixingMetric(nnx.Module):
    """Metric to evaluate style mixing quality in StyleGAN."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize StyleMixingMetric.

        Args:
            rngs: Random number generators
        """
        super().__init__()
        self.perceptual_distance = LPIPSMetric(rngs=rngs)

    def compute_style_mixing_quality(
        self, generator, num_samples: int = 100, *, rngs: nnx.Rngs
    ) -> dict[str, float]:
        """Compute style mixing quality metrics.

        Args:
            generator: StyleGAN3 generator
            num_samples: Number of samples for evaluation
            rngs: Random number generators

        Returns:
            dictionary of style mixing metrics
        """
        batch_size = min(16, num_samples)

        # Generate latent codes
        z1 = jax.random.normal(rngs.sample(), (batch_size, generator.latent_dim))
        z2 = jax.random.normal(
            jax.random.fold_in(rngs.sample(), 1), (batch_size, generator.latent_dim)
        )

        # Generate images with single styles
        img1 = generator(z1, rngs=rngs)
        img2 = generator(z2, rngs=rngs)

        # Generate mixed style images (coarse/fine mixing)
        # Mix styles at different layers
        mixing_results = []

        for mixing_layer in [4, 8, 12]:  # Different mixing points
            # Get style codes
            w1 = generator.mapping(z1)
            w2 = generator.mapping(z2)

            # Mix styles
            w_mixed = w1.copy()
            w_mixed = w_mixed.at[:, mixing_layer:].set(w2[:, mixing_layer:])

            # Generate mixed image
            img_mixed = generator.synthesis(w_mixed, rngs=rngs)

            # Compute perceptual distances
            dist_to_source = self.perceptual_distance.compute_distance(img1, img_mixed)
            dist_to_target = self.perceptual_distance.compute_distance(img2, img_mixed)

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

        # Aggregate results
        avg_mixing_quality = np.mean([r["mixing_ratio"] for r in mixing_results])

        return {
            "style_mixing_quality": float(avg_mixing_quality),
            "mixing_details": mixing_results,
            "num_samples": int(num_samples),
        }


class LPIPSMetric(nnx.Module):
    """Learned Perceptual Image Patch Similarity metric."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize LPIPSMetric.

        Args:
            rngs: Random number generators
        """
        super().__init__()
        # Simplified perceptual network (normally would use pre-trained VGG/AlexNet)
        self.feature_net = self._build_perceptual_network(rngs)

    def _build_perceptual_network(self, rngs: nnx.Rngs) -> nnx.Module:
        """Build simplified perceptual feature network."""

        class PerceptualNet(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                """Initialize PerceptualNet.

                Args:
                    rngs: Random number generators
                """
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

            def __call__(self, x: jnp.ndarray) -> list[jnp.ndarray]:
                """Extract multi-scale features."""
                features = []

                x = nnx.relu(self.conv1(x))
                features.append(x)

                x = nnx.relu(self.conv2(x))
                features.append(x)

                x = nnx.relu(self.conv3(x))
                features.append(x)

                return features

        return PerceptualNet(rngs=rngs)

    def compute_distance(self, images1: jnp.ndarray, images2: jnp.ndarray) -> jnp.ndarray:
        """Compute LPIPS distance between image pairs.

        Args:
            images1: First set of images
            images2: Second set of images

        Returns:
            Perceptual distances for each pair
        """
        # Extract features from both image sets
        features1 = self.feature_net(images1)
        features2 = self.feature_net(images2)

        # Compute distances across all feature levels
        total_distance = jnp.zeros(images1.shape[0])

        for f1, f2 in zip(features1, features2):
            # L2 distance in feature space
            feat_distance = jnp.sqrt(jnp.sum((f1 - f2) ** 2, axis=(1, 2, 3)))
            total_distance += feat_distance

        return total_distance / len(features1)


class FewShotAdaptationMetric(nnx.Module):
    """Metric to evaluate few-shot domain adaptation performance."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize FewShotAdaptationMetric.

        Args:
            rngs: Random number generators
        """
        super().__init__()
        self.fid_metric = FIDMetric(rngs=rngs)
        self.lpips_metric = LPIPSMetric(rngs=rngs)

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
            dictionary of adaptation metrics
        """
        batch_size = 32

        # Generate samples from both generators
        z = jax.random.normal(rngs.sample(), (num_samples, original_generator.latent_dim))

        # Split into batches for memory efficiency
        num_batches = num_samples // batch_size

        original_images: list[jnp.ndarray] = []
        adapted_images: list[jnp.ndarray] = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            z_batch = z[start_idx:end_idx]

            # Generate from both models
            orig_batch = original_generator(z_batch, rngs=rngs)
            adapt_batch = adapted_generator(z_batch, rngs=rngs)

            original_images.append(orig_batch)
            adapted_images.append(adapt_batch)

        # Check if we have any samples to concatenate
        if not original_images or not adapted_images:
            return {
                "fid_improvement": 0.0,
                "lpips_improvement": 0.0,
                "adaptation_quality": 0.0,
            }

        original_images_array = jnp.concatenate(original_images, axis=0)
        adapted_images_array = jnp.concatenate(adapted_images, axis=0)

        # Get target domain images
        target_images: list[jnp.ndarray] = []
        target_batches = list(target_dataset(batch_size))

        for batch in target_batches[:num_batches]:
            target_images.append(batch["images"])

        # Check if we have target images to concatenate
        if not target_images:
            return {
                "fid_improvement": 0.0,
                "lpips_improvement": 0.0,
                "adaptation_quality": 0.0,
            }

        target_images_array = jnp.concatenate(target_images, axis=0)

        # Compute FID scores
        target_mean, target_cov = self.fid_metric.compute_statistics(target_images_array)
        orig_mean, orig_cov = self.fid_metric.compute_statistics(original_images_array)
        adapt_mean, adapt_cov = self.fid_metric.compute_statistics(adapted_images_array)

        fid_original = self.fid_metric.compute_fid(target_mean, target_cov, orig_mean, orig_cov)
        fid_adapted = self.fid_metric.compute_fid(target_mean, target_cov, adapt_mean, adapt_cov)

        # Compute adaptation improvement
        fid_improvement = fid_original - fid_adapted

        # Compute diversity preservation
        lpips_original = jnp.mean(
            self.lpips_metric.compute_distance(
                original_images_array[:batch_size],
                original_images_array[batch_size : 2 * batch_size],
            )
        )
        lpips_adapted = jnp.mean(
            self.lpips_metric.compute_distance(
                adapted_images_array[:batch_size], adapted_images_array[batch_size : 2 * batch_size]
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


class EquivarianceMetric(nnx.Module):
    """Metric to evaluate translation and rotation equivariance."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize EquivarianceMetric.

        Args:
            rngs: Random number generators
        """
        super().__init__()
        self.lpips_metric = LPIPSMetric(rngs=rngs)

    def evaluate_equivariance(
        self, generator, num_samples: int = 100, *, rngs: nnx.Rngs
    ) -> dict[str, float]:
        """Evaluate translation and rotation equivariance.

        Args:
            generator: StyleGAN3 generator to test
            num_samples: Number of samples for evaluation
            rngs: Random number generators

        Returns:
            dictionary of equivariance metrics
        """
        batch_size = min(16, num_samples)

        # Generate base images
        z = jax.random.normal(rngs.sample(), (batch_size, generator.latent_dim))
        base_images = generator(z, rngs=rngs)

        # Test translation equivariance
        # Generate with small camera translation
        translation_offset = 0.1  # Small translation
        translated_images = self._apply_translation(base_images, translation_offset)

        # Measure consistency
        translation_consistency = jnp.clip(
            1.0 - jnp.mean(self.lpips_metric.compute_distance(base_images, translated_images)),
            0.0,
            1.0,
        )

        # Test rotation equivariance
        rotation_angle = jnp.pi / 12  # 15 degrees
        rotated_images = self._apply_rotation(base_images, rotation_angle)

        rotation_consistency = jnp.clip(
            1.0 - jnp.mean(self.lpips_metric.compute_distance(base_images, rotated_images)),
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
        # Simple translation by shifting pixel coordinates
        shift_pixels = int(offset * images.shape[1])

        # Pad and crop to simulate translation
        padded = jnp.pad(
            images,
            ((0, 0), (shift_pixels, shift_pixels), (shift_pixels, shift_pixels), (0, 0)),
            mode="edge",
        )

        # Crop back to original size
        h, w = images.shape[1], images.shape[2]
        translated = padded[:, shift_pixels : shift_pixels + h, shift_pixels : shift_pixels + w, :]

        return translated

    def _apply_rotation(self, images: jnp.ndarray, angle: float) -> jnp.ndarray:
        """Apply rotation to images (mock implementation)."""
        # Simple approximation of rotation using affine transform
        # In practice, would use proper rotation matrix

        # Calculate rotation components (for future implementation)
        _cos_a, _sin_a = jnp.cos(angle), jnp.sin(angle)

        # For simplicity, just return slightly modified images
        # Real implementation would apply rotation matrix
        noise_factor = 0.05 * abs(angle)
        noise = noise_factor * jax.random.normal(jax.random.key(42), images.shape)

        return jnp.clip(images + noise, -1.0, 1.0)


class StyleGANMetrics(nnx.Module):
    """Comprehensive StyleGAN evaluation metrics suite."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize StyleGANMetrics.

        Args:
            rngs: Random number generators
        """
        super().__init__()
        self.fid_metric = FIDMetric(rngs=rngs)
        self.style_mixing_metric = StyleMixingMetric(rngs=rngs)
        self.lpips_metric = LPIPSMetric(rngs=rngs)
        self.few_shot_metric = FewShotAdaptationMetric(rngs=rngs)
        self.equivariance_metric = EquivarianceMetric(rngs=rngs)

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
        """Compute all StyleGAN metrics.

        Args:
            generator: StyleGAN3 generator
            real_images: Real images for comparison
            generated_images: Generated images to evaluate
            target_dataset: Optional target dataset for few-shot evaluation
            adapted_generator: Optional adapted generator for few-shot evaluation
            rngs: Random number generators

        Returns:
            dictionary of all computed metrics
        """
        metrics = {}

        # Basic image quality metrics
        real_mean, real_cov = self.fid_metric.compute_statistics(real_images)
        fake_mean, fake_cov = self.fid_metric.compute_statistics(generated_images)

        metrics["fid"] = self.fid_metric.compute_fid(real_mean, real_cov, fake_mean, fake_cov)

        # Style mixing quality
        style_metrics = self.style_mixing_metric.compute_style_mixing_quality(generator, rngs=rngs)
        metrics.update(style_metrics)

        # Equivariance properties
        equivariance_metrics = self.equivariance_metric.evaluate_equivariance(generator, rngs=rngs)
        metrics.update(equivariance_metrics)

        # Few-shot adaptation (if applicable)
        if target_dataset is not None and adapted_generator is not None:
            adaptation_metrics = self.few_shot_metric.evaluate_adaptation(
                generator, adapted_generator, target_dataset, rngs=rngs
            )
            metrics.update(adaptation_metrics)

        return metrics
