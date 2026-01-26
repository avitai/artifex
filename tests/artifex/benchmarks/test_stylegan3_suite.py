"""Tests for StyleGAN3 Benchmark Suite.

This module provides comprehensive tests for the StyleGAN3 implementation,
including model architecture, datasets, metrics, and benchmark orchestration.

Key Test Areas:
- StyleGAN3 model architecture and forward pass
- FFHQ and CelebA dataset functionality
- StyleGAN-specific metrics (FID, style mixing, equivariance)
- Benchmark suite orchestration and evaluation
- Few-shot adaptation capabilities
"""

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.benchmarks.datasets.ffhq import CelebADataset, FFHQDataset
from artifex.benchmarks.metrics.style_metrics import (
    EquivarianceMetric,
    FewShotAdaptationMetric,
    FIDMetric,
    LPIPSMetric,
    StyleGANMetrics,
    StyleMixingMetric,
)
from artifex.benchmarks.suites.stylegan3_suite import (
    create_stylegan3_demo,
    StyleGAN3Benchmark,
    StyleGAN3BenchmarkConfig,
    StyleGAN3Suite,
)
from artifex.generative_models.core.configuration.network_configs import (
    StyleGAN3DiscriminatorConfig,
    StyleGAN3GeneratorConfig,
)
from artifex.generative_models.models.gan.stylegan3 import (
    MappingNetwork,
    StyleGAN3Discriminator,
    StyleGAN3Generator,
    StyleModulatedConv,
    SynthesisBlock,
    SynthesisNetwork,
)


def create_stylegan3_generator(
    latent_dim: int = 512,
    style_dim: int = 512,
    img_resolution: int = 64,
    *,
    rngs: nnx.Rngs,
) -> StyleGAN3Generator:
    """Helper to create a StyleGAN3Generator with config."""
    config = StyleGAN3GeneratorConfig(
        name="test_generator",
        latent_dim=latent_dim,
        hidden_dims=(style_dim,),
        output_shape=(img_resolution, img_resolution, 3),
        activation="leaky_relu",
        style_dim=style_dim,
        mapping_layers=8,
        img_resolution=img_resolution,
        img_channels=3,
    )
    return StyleGAN3Generator(config=config, rngs=rngs)


def create_stylegan3_discriminator(
    img_resolution: int = 64,
    img_channels: int = 3,
    *,
    rngs: nnx.Rngs,
) -> StyleGAN3Discriminator:
    """Helper to create a StyleGAN3Discriminator with config."""
    config = StyleGAN3DiscriminatorConfig(
        name="test_discriminator",
        input_shape=(img_resolution, img_resolution, img_channels),
        hidden_dims=(64,),
        activation="leaky_relu",
        img_resolution=img_resolution,
        img_channels=img_channels,
        base_channels=64,
        max_channels=512,
    )
    return StyleGAN3Discriminator(config=config, rngs=rngs)


@pytest.fixture
def rngs():
    """Standard fixture for random number generators."""
    return nnx.Rngs(42)


class TestStyleGAN3Architecture:
    """Test StyleGAN3 model architecture components."""

    def test_mapping_network_initialization(self, rngs):
        """Test MappingNetwork initialization and forward pass."""
        mapping = MappingNetwork(latent_dim=512, style_dim=512, num_layers=8, rngs=rngs)

        assert mapping.latent_dim == 512
        assert mapping.style_dim == 512
        assert mapping.num_layers == 8
        assert len(mapping.layers) == 8

    def test_mapping_network_forward(self, rngs):
        """Test MappingNetwork forward pass with correct shapes."""
        mapping = MappingNetwork(latent_dim=512, style_dim=512, num_layers=8, rngs=rngs)

        # Test forward pass
        batch_size = 4
        z = jax.random.normal(rngs.sample(), (batch_size, 512))
        w = mapping(z)

        # Check output shape
        assert w.shape == (batch_size, 14, 512)  # 14 synthesis layers

        # Test truncation
        w_truncated = mapping(z, truncation_psi=0.7)
        assert w_truncated.shape == w.shape
        assert not jnp.allclose(w, w_truncated)  # Should be different

    def test_modulated_conv2d(self, rngs):
        """Test ModulatedConv2d layer."""
        conv = StyleModulatedConv(
            in_channels=64, out_channels=128, kernel_size=3, style_dim=512, rngs=rngs
        )

        batch_size = 2
        height, width = 32, 32

        x = jax.random.normal(rngs.sample(), (batch_size, height, width, 64))
        style = jax.random.normal(rngs.sample(), (batch_size, 512))

        output = conv(x, style)
        assert output.shape == (batch_size, height, width, 128)

    def test_synthesis_layer(self, rngs):
        """Test SynthesisBlock functionality."""
        layer = SynthesisBlock(
            in_channels=64,
            out_channels=128,
            style_dim=512,
            upsample=False,  # Changed from resolution and is_torgb
            rngs=rngs,
        )

        batch_size = 2
        x = jax.random.normal(rngs.sample(), (batch_size, 32, 32, 64))
        style1 = jax.random.normal(rngs.sample(), (batch_size, 512))
        style2 = jax.random.normal(rngs.sample(), (batch_size, 512))

        # SynthesisBlock now takes two style vectors
        output = layer(x, style1, style2, rngs=rngs)
        assert output.shape == (batch_size, 32, 32, 128)

    def test_synthesis_network(self, rngs):
        """Test SynthesisNetwork architecture."""
        synthesis = SynthesisNetwork(
            style_dim=512,
            img_resolution=64,  # Reduced to avoid memory issues
            img_channels=3,
            rngs=rngs,
        )

        batch_size = 2
        # Calculate number of w vectors needed
        num_blocks = int(math.log2(64 // 4))  # Number of upsampling blocks
        num_ws = num_blocks * 2 + 2
        w = jax.random.normal(rngs.sample(), (batch_size, num_ws, 512))

        images = synthesis(w, rngs=rngs)
        assert images.shape == (batch_size, 64, 64, 3)

    def test_stylegan3_generator(self, rngs):
        """Test complete StyleGAN3Generator."""
        generator = create_stylegan3_generator(
            latent_dim=256,  # Reduced to avoid memory issues
            style_dim=256,  # Reduced to avoid memory issues
            img_resolution=64,  # Reduced to avoid memory issues
            rngs=rngs,
        )

        batch_size = 2
        z = jax.random.normal(rngs.sample(), (batch_size, 256))

        images = generator(z, rngs=rngs)
        assert images.shape == (batch_size, 64, 64, 3)

        # Test sampling method
        samples = generator.sample(num_samples=3, rngs=rngs)
        assert samples.shape == (3, 64, 64, 3)

    def test_stylegan3_discriminator(self, rngs):
        """Test StyleGAN3Discriminator."""
        discriminator = create_stylegan3_discriminator(
            img_resolution=64,  # Reduced to avoid memory issues
            img_channels=3,
            rngs=rngs,
        )

        batch_size = 2
        images = jax.random.normal(rngs.sample(), (batch_size, 64, 64, 3))

        scores = discriminator(images)
        assert scores.shape == (batch_size, 1)


class TestStyleGANDatasets:
    """Test StyleGAN dataset implementations."""

    def test_ffhq_dataset_initialization(self, rngs):
        """Test FFHQDataset initialization."""
        dataset = FFHQDataset(image_size=256, channels=3, num_samples=1000, rngs=rngs)

        assert dataset.image_size == 256
        assert dataset.channels == 3
        assert dataset.num_samples == 1000

    def test_ffhq_dataset_batching(self, rngs):
        """Test FFHQDataset batch generation."""
        dataset = FFHQDataset(image_size=128, channels=3, num_samples=100, split="train", rngs=rngs)

        batch_size = 8
        batches = list(dataset(batch_size))

        # Check we have batches
        assert len(batches) > 0

        # Check first batch
        batch = batches[0]
        assert "images" in batch
        assert "indices" in batch
        assert batch["images"].shape == (batch_size, 128, 128, 3)
        assert batch["indices"].shape == (batch_size,)

        # Check image range is [-1, 1] for StyleGAN3
        images = batch["images"]
        assert jnp.all(images >= -1.0) and jnp.all(images <= 1.0)

    def test_ffhq_few_shot_batch(self, rngs):
        """Test FFHQ few-shot batch generation."""
        dataset = FFHQDataset(image_size=128, num_samples=1000, rngs=rngs)

        few_shot_batch = dataset.get_few_shot_batch(num_samples=50)

        assert "images" in few_shot_batch
        assert "indices" in few_shot_batch
        assert "num_samples" in few_shot_batch
        assert few_shot_batch["images"].shape == (50, 128, 128, 3)
        assert few_shot_batch["num_samples"] == 50

    def test_celeba_dataset(self, rngs):
        """Test CelebADataset functionality."""
        dataset = CelebADataset(image_size=128, num_samples=1000, rngs=rngs)

        batch_size = 4
        batches = list(dataset(batch_size))

        # Check batch content
        batch = batches[0]
        assert "images" in batch
        assert "attributes" in batch
        assert "indices" in batch

        assert batch["images"].shape == (batch_size, 128, 128, 3)
        assert batch["attributes"].shape == (batch_size, 40)  # 40 CelebA attributes
        assert len(dataset.attributes) == 40


class TestStyleGANMetrics:
    """Test StyleGAN-specific metrics."""

    def test_fid_metric(self, rngs):
        """Test FID metric computation."""
        fid_metric = FIDMetric(rngs=rngs)

        batch_size = 16
        real_images = jax.random.normal(rngs.sample(), (batch_size, 64, 64, 3))
        fake_images = jax.random.normal(
            jax.random.fold_in(rngs.sample(), 1), (batch_size, 64, 64, 3)
        )

        # Compute statistics
        real_mean, real_cov = fid_metric.compute_statistics(real_images)
        fake_mean, fake_cov = fid_metric.compute_statistics(fake_images)

        assert real_mean.shape == (2048,)
        assert real_cov.shape == (2048, 2048)

        # Compute FID
        fid_score = fid_metric.compute_fid(real_mean, real_cov, fake_mean, fake_cov)
        assert isinstance(fid_score, float)
        assert fid_score >= 0.0

    def test_lpips_metric(self, rngs):
        """Test LPIPS perceptual distance metric."""
        lpips_metric = LPIPSMetric(rngs=rngs)

        batch_size = 4
        images1 = jax.random.normal(rngs.sample(), (batch_size, 64, 64, 3))
        images2 = jax.random.normal(jax.random.fold_in(rngs.sample(), 1), (batch_size, 64, 64, 3))

        distances = lpips_metric.compute_distance(images1, images2)
        assert distances.shape == (batch_size,)
        assert jnp.all(distances >= 0.0)

    def test_style_mixing_metric(self, rngs):
        """Test style mixing quality evaluation."""
        # Create mock generator
        generator = create_stylegan3_generator(
            latent_dim=512,
            style_dim=512,
            img_resolution=64,  # Smaller for testing
            rngs=rngs,
        )

        style_metric = StyleMixingMetric(rngs=rngs)

        results = style_metric.compute_style_mixing_quality(
            generator,
            num_samples=32,  # Small for testing
            rngs=rngs,
        )

        assert "style_mixing_quality" in results
        assert "mixing_details" in results
        assert "num_samples" in results

        # Check mixing details
        mixing_details = results["mixing_details"]
        assert len(mixing_details) == 3  # 3 mixing layers tested

        for detail in mixing_details:
            assert "layer" in detail
            assert "mixing_ratio" in detail
            assert 0.0 <= detail["mixing_ratio"] <= 1.0

    def test_equivariance_metric(self, rngs):
        """Test translation and rotation equivariance evaluation."""
        generator = create_stylegan3_generator(
            latent_dim=512, style_dim=512, img_resolution=64, rngs=rngs
        )

        equivariance_metric = EquivarianceMetric(rngs=rngs)

        results = equivariance_metric.evaluate_equivariance(generator, num_samples=16, rngs=rngs)

        assert "translation_equivariance" in results
        assert "rotation_equivariance" in results
        assert "overall_equivariance" in results
        assert "num_samples" in results

        # Check that equivariance scores are in valid range
        for key in ["translation_equivariance", "rotation_equivariance", "overall_equivariance"]:
            score = results[key]
            assert 0.0 <= score <= 1.0

    def test_few_shot_adaptation_metric(self, rngs):
        """Test few-shot adaptation evaluation."""
        # Create generators
        original_generator = create_stylegan3_generator(
            latent_dim=512, style_dim=512, img_resolution=64, rngs=rngs
        )

        adapted_generator = create_stylegan3_generator(
            latent_dim=512,
            style_dim=512,
            img_resolution=64,
            rngs=nnx.Rngs(jax.random.fold_in(rngs.sample(), 1000)),
        )

        # Create target dataset
        target_dataset = CelebADataset(image_size=64, num_samples=200, rngs=rngs)

        adaptation_metric = FewShotAdaptationMetric(rngs=rngs)

        results = adaptation_metric.evaluate_adaptation(
            original_generator=original_generator,
            adapted_generator=adapted_generator,
            target_dataset=target_dataset,
            num_samples=64,  # Small for testing
            rngs=rngs,
        )

        assert "fid_original" in results
        assert "fid_adapted" in results
        assert "fid_improvement" in results
        assert "diversity_preservation" in results
        assert "adaptation_success" in results
        assert "num_samples" in results

    def test_comprehensive_metrics(self, rngs):
        """Test comprehensive StyleGAN metrics suite."""
        generator = create_stylegan3_generator(
            latent_dim=512, style_dim=512, img_resolution=64, rngs=rngs
        )

        metrics = StyleGANMetrics(rngs=rngs)

        # Generate test data
        batch_size = 16
        real_images = jax.random.normal(rngs.sample(), (batch_size, 64, 64, 3))
        generated_images = generator.sample(num_samples=batch_size, rngs=rngs)

        # Compute all metrics
        all_metrics = metrics.compute_all_metrics(
            generator=generator,
            real_images=real_images,
            generated_images=generated_images,
            rngs=rngs,
        )

        # Check that key metrics are present
        expected_metrics = [
            "fid",
            "style_mixing_quality",
            "translation_equivariance",
            "rotation_equivariance",
            "overall_equivariance",
        ]

        for metric in expected_metrics:
            assert metric in all_metrics


class TestStyleGAN3Benchmark:
    """Test StyleGAN3 benchmark suite."""

    def test_benchmark_config(self):
        """Test StyleGAN3BenchmarkConfig."""
        config = StyleGAN3BenchmarkConfig(
            model_name="test_stylegan3", image_size=128, batch_size=8, fid_target=30.0
        )

        assert config.name == "test_stylegan3"
        assert config.image_size == 128
        assert config.batch_size == 8
        assert config.fid_target == 30.0

    def test_benchmark_initialization(self, rngs):
        """Test StyleGAN3Benchmark initialization."""
        config = StyleGAN3BenchmarkConfig(
            model_name="test_stylegan3", image_size=64, batch_size=4, num_evaluation_samples=100
        )

        benchmark = StyleGAN3Benchmark(config, rngs=rngs)

        assert benchmark.benchmark_config == config
        assert benchmark.ffhq_dataset is not None
        assert benchmark.celeba_dataset is not None
        assert benchmark.metrics is not None

    def test_benchmark_model_setup(self, rngs):
        """Test benchmark model setup."""
        config = StyleGAN3BenchmarkConfig(image_size=64, batch_size=4)
        benchmark = StyleGAN3Benchmark(config, rngs=rngs)

        # Test single generator setup
        generator = create_stylegan3_generator(
            latent_dim=512, style_dim=512, img_resolution=64, rngs=rngs
        )

        benchmark.setup_model(generator)
        assert benchmark.generator is not None
        assert benchmark.discriminator is not None

        # Test generator-discriminator tuple setup
        discriminator = create_stylegan3_discriminator(img_resolution=64, rngs=rngs)

        benchmark.setup_model((generator, discriminator))
        # Check object identity without triggering massive printouts on failure
        assert id(benchmark.generator) == id(generator), "Generator object identity mismatch"
        assert id(benchmark.discriminator) == id(discriminator), (
            "Discriminator object identity mismatch"
        )

    def test_benchmark_execution(self, rngs):
        """Test complete benchmark execution."""
        config = StyleGAN3BenchmarkConfig(
            image_size=64,
            batch_size=4,
            num_evaluation_samples=32,  # Small for testing
            fid_target=100.0,  # Relaxed for testing
        )

        benchmark = StyleGAN3Benchmark(config, rngs=rngs)

        # Run benchmark (will create default generator)
        result = benchmark.run_benchmark()

        # Check result structure
        assert result.model_name == config.name
        assert result.dataset_name == "FFHQ"
        assert result.metrics is not None
        assert result.config is not None
        assert result.metadata is not None

        # Check that key metrics are present
        metrics = result.metrics
        expected_metrics = [
            "fid_score",
            "perceptual_diversity",
            "quality_pass",
            "style_mixing_quality",
            "style_mixing_pass",
            "translation_equivariance",
            "rotation_equivariance",
            "equivariance_pass",
            "overall_pass",
            "composite_score",
            "benchmark_success",
        ]

        for metric in expected_metrics:
            assert metric in metrics

    def test_benchmark_suite(self, rngs):
        """Test StyleGAN3Suite functionality."""
        suite = StyleGAN3Suite(image_sizes=[64], rngs=rngs)

        assert len(suite.benchmarks) == 1
        assert 64 in suite.benchmarks

        # Run all benchmarks
        results = suite.run_all_benchmarks()
        assert len(results) == 1
        assert 64 in results

        # Get performance summary
        summary = suite.get_performance_summary(results)

        assert "total_benchmarks" in summary
        assert "passed_benchmarks" in summary
        assert "success_rate" in summary
        assert "results_by_size" in summary
        assert summary["total_benchmarks"] == 1

    def test_stylegan3_demo(self):
        """Test StyleGAN3 demo function."""
        demo_results = create_stylegan3_demo()

        assert "benchmark_results" in demo_results
        assert "performance_summary" in demo_results
        assert "sample_images" in demo_results
        assert "model_info" in demo_results

        # Check sample images shape
        sample_images = demo_results["sample_images"]
        assert sample_images.shape == (4, 256, 256, 3)

        # Check model info
        model_info = demo_results["model_info"]
        assert model_info["architecture"] == "StyleGAN3"
        assert model_info["image_size"] == 256
        assert model_info["latent_dim"] == 512


class TestStyleGANIntegration:
    """Integration tests for complete StyleGAN3 pipeline."""

    def test_end_to_end_generation(self, rngs):
        """Test end-to-end image generation pipeline."""
        # Create generator
        generator = create_stylegan3_generator(
            latent_dim=512, style_dim=512, img_resolution=64, rngs=rngs
        )

        # Test generation pipeline
        batch_size = 4

        # 1. Sample latent codes
        z = jax.random.normal(rngs.sample(), (batch_size, 512))

        # 2. Map to style codes
        w = generator.mapping(z)
        assert w.shape == (
            batch_size,
            8,
            512,
        )  # For 64x64 resolution: 4 blocks * 2 style vectors each = 8

        # 3. Generate images
        images = generator.synthesis(w, rngs=rngs)
        assert images.shape == (batch_size, 64, 64, 3)

        # 4. Test full pipeline
        full_images = generator(z, rngs=rngs)
        assert full_images.shape == (batch_size, 64, 64, 3)

        # Check image values are in valid range
        assert jnp.all(jnp.isfinite(full_images))

    def test_discriminator_evaluation(self, rngs):
        """Test discriminator evaluation of generated images."""
        generator = create_stylegan3_generator(
            latent_dim=512, style_dim=512, img_resolution=64, rngs=rngs
        )

        discriminator = create_stylegan3_discriminator(img_resolution=64, rngs=rngs)

        # Generate images
        batch_size = 4
        images = generator.sample(num_samples=batch_size, rngs=rngs)

        # Evaluate with discriminator
        scores = discriminator(images)
        assert scores.shape == (batch_size, 1)
        assert jnp.all(jnp.isfinite(scores))

    def test_style_interpolation(self, rngs):
        """Test style interpolation between two latent codes."""
        generator = create_stylegan3_generator(
            latent_dim=512, style_dim=512, img_resolution=64, rngs=rngs
        )

        # Create two latent codes
        z1 = jax.random.normal(rngs.sample(), (1, 512))
        z2 = jax.random.normal(jax.random.fold_in(rngs.sample(), 1), (1, 512))

        # Map to style codes
        w1 = generator.mapping(z1)
        w2 = generator.mapping(z2)

        # Interpolate
        num_steps = 5
        interpolated_images = []

        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            w_interp = w1 * (1 - alpha) + w2 * alpha

            img = generator.synthesis(w_interp, rngs=rngs)
            interpolated_images.append(img)

        # Check that we got valid interpolated images
        assert len(interpolated_images) == num_steps
        for img in interpolated_images:
            assert img.shape == (1, 64, 64, 3)
            assert jnp.all(jnp.isfinite(img))

    def test_metrics_consistency(self, rngs):
        """Test that metrics produce consistent results across runs."""
        generator = create_stylegan3_generator(
            latent_dim=512, style_dim=512, img_resolution=64, rngs=rngs
        )

        metrics = StyleGANMetrics(rngs=rngs)

        # Generate fixed test data
        test_key = jax.random.key(123)
        real_images = jax.random.normal(test_key, (16, 64, 64, 3))

        # Generate images with fixed random key
        gen_key = jax.random.key(456)
        z = jax.random.normal(gen_key, (16, 512))
        fake_images = generator(z, rngs=nnx.Rngs(456))

        # Compute metrics twice
        result1 = metrics.compute_all_metrics(
            generator=generator,
            real_images=real_images,
            generated_images=fake_images,
            rngs=nnx.Rngs(789),
        )

        result2 = metrics.compute_all_metrics(
            generator=generator,
            real_images=real_images,
            generated_images=fake_images,
            rngs=nnx.Rngs(789),
        )

        # FID should be consistent for same input
        assert abs(result1["fid"] - result2["fid"]) < 1e-5

        # Style mixing might vary due to random sampling, but should be similar
        style_diff = abs(result1["style_mixing_quality"] - result2["style_mixing_quality"])
        assert style_diff < 0.1  # Allow some variance due to sampling


if __name__ == "__main__":
    # Run basic smoke test
    test_rngs = nnx.Rngs(42)

    # Test generator creation
    generator = create_stylegan3_generator(
        latent_dim=512, style_dim=512, img_resolution=64, rngs=test_rngs
    )

    # Test image generation
    images = generator.sample(num_samples=2, rngs=test_rngs)
    print(f"✅ Generated images with shape: {images.shape}")

    # Test benchmark demo
    demo_results = create_stylegan3_demo()
    print("✅ Demo completed successfully")
    print(f"   Sample images shape: {demo_results['sample_images'].shape}")

    print("✅ All smoke tests passed!")
