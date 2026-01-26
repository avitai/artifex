"""Tests for the Multi-β VAE controllable generation benchmark suite."""

from unittest.mock import patch

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.benchmarks.datasets.celeba import CelebADataset
from artifex.benchmarks.metrics.disentanglement import (
    DisentanglementMetric,
    MutualInformationGapMetric,
    SeparationMetric,
)
from artifex.benchmarks.metrics.image import (
    FIDMetric,
    LPIPSMetric,
    SSIMMetric,
)
from artifex.benchmarks.suites.multi_beta_vae_suite import (
    MultiBetaVAEBenchmark,
    MultiBetaVAEBenchmarkSuite,
)
from artifex.generative_models.core.configuration import EvaluationConfig


def mock_celeba_load(self):
    """Mock the _load_celeba_data method to avoid network access."""
    # Create mock images
    self.images = jnp.ones((self.num_samples, self.image_size, self.image_size, 3)) * 0.5

    # Create mock attributes if needed
    if self.include_attributes:
        self.attributes = jnp.zeros(
            (self.num_samples, len(self.attribute_names)), dtype=jnp.float32
        )
        # Set some random attributes for testing
        key = self.rngs.params() if self.rngs and "params" in self.rngs else jax.random.key(0)
        self.attributes = jax.random.bernoulli(key, 0.5, self.attributes.shape).astype(jnp.float32)

        # Extract selected attributes
        self.selected_attribute_indices = [
            self.attribute_names.index(attr)
            for attr in self.selected_attributes
            if attr in self.attribute_names
        ]
        self.selected_attributes_data = self.attributes[:, self.selected_attribute_indices]
    else:
        self.attributes = None
        self.selected_attributes_data = None

    # Create splits
    if self.split == "all":
        self.train_indices = jnp.arange(0, int(0.8 * self.num_samples))
        self.val_indices = jnp.arange(int(0.8 * self.num_samples), int(0.9 * self.num_samples))
        self.test_indices = jnp.arange(int(0.9 * self.num_samples), self.num_samples)
    elif self.split == "train":
        self.train_indices = jnp.arange(self.num_samples)
        self.val_indices = jnp.array([])
        self.test_indices = jnp.array([])
    elif self.split == "val":
        self.train_indices = jnp.array([])
        self.val_indices = jnp.arange(self.num_samples)
        self.test_indices = jnp.array([])
    elif self.split == "test":
        self.train_indices = jnp.array([])
        self.val_indices = jnp.array([])
        self.test_indices = jnp.arange(self.num_samples)


class MockMultiBetaVAE(nnx.Module):
    """Mock Multi-β VAE model for testing."""

    def __init__(self, model_name="MockVAE", *, rngs: nnx.Rngs):
        """Initialize the mock model."""
        super().__init__()
        self.model_name = model_name
        self.rngs = rngs

    def __call__(self, images, *, rngs=None):
        """Forward pass through the model."""
        return self.encode_decode(images, rngs=rngs)

    def encode_decode(self, images, *, rngs=None):
        """Encode images to latent space and decode back to image space."""
        batch_size = images.shape[0]
        latent_dim = 32

        # Mock latent codes
        if rngs is not None:
            if hasattr(rngs, "encode"):
                key = rngs.encode()
            elif hasattr(rngs, "params"):
                key = rngs.params()
            else:
                key = jax.random.key(0)
        else:
            key = jax.random.key(0)
        latent_codes = jax.random.normal(key, (batch_size, latent_dim))

        # Mock reconstructions (slightly noisy versions of originals)
        if rngs is not None:
            if hasattr(rngs, "decode"):
                noise_key = rngs.decode()
            elif hasattr(rngs, "params"):
                noise_key = rngs.params()
            else:
                noise_key = jax.random.key(1)
        else:
            noise_key = jax.random.key(1)
        noise = jax.random.normal(noise_key, images.shape) * 0.1
        reconstructions = jnp.clip(images + noise, 0.0, 1.0)

        # Mock generated images
        if rngs is not None:
            if hasattr(rngs, "generate"):
                gen_key = rngs.generate()
            elif hasattr(rngs, "params"):
                gen_key = rngs.params()
            else:
                gen_key = jax.random.key(2)
        else:
            gen_key = jax.random.key(2)
        generated_images = jax.random.uniform(gen_key, images.shape)

        return {
            "reconstructions": reconstructions,
            "latent_codes": latent_codes,
            "generated_images": generated_images,
            "reconstruction_loss": 0.1,
        }


@pytest.fixture
def rngs():
    """Create random number generator keys."""
    seed = 42
    key = jax.random.key(seed)
    return nnx.Rngs(dropout=key, params=key)


@pytest.fixture
def mock_dataset(rngs):
    """Create a mock CelebA dataset."""
    with patch.object(CelebADataset, "_load_celeba_data", mock_celeba_load):
        return CelebADataset(
            num_samples=20,
            image_size=32,
            include_attributes=True,
            rngs=rngs,
        )


@pytest.fixture
def mock_model(rngs):
    """Create a mock Multi-β VAE model."""
    return MockMultiBetaVAE(rngs=rngs)


@patch.object(CelebADataset, "_load_celeba_data", mock_celeba_load)
def test_celeba_dataset_initialization(rngs):
    """Test CelebA dataset initialization."""
    dataset = CelebADataset(
        num_samples=10,
        image_size=32,
        include_attributes=True,
        rngs=rngs,
    )

    assert len(dataset) == 10
    assert dataset.image_size == 32
    assert dataset.include_attributes

    # Test batch retrieval
    batch = dataset.get_batch(batch_size=5)
    assert batch["images"].shape == (5, 32, 32, 3)
    assert "attributes" in batch
    assert batch["attributes"].shape[0] == 5

    # Test individual sample retrieval
    sample = dataset[0]
    assert sample["images"].shape == (32, 32, 3)
    assert "attributes" in sample


def test_disentanglement_metrics(rngs):
    """Test disentanglement metrics computation."""
    batch_size = 10
    latent_dim = 20
    num_factors = 5

    # Create mock data
    # Use params key if specific keys are not available
    if hasattr(rngs, "latent"):
        latent_key = rngs.latent()
    elif hasattr(rngs, "params"):
        latent_key = rngs.params()
    else:
        latent_key = jax.random.key(42)

    if hasattr(rngs, "factors"):
        factors_key = rngs.factors()
    elif hasattr(rngs, "params"):
        factors_key = rngs.params()
    else:
        factors_key = jax.random.key(43)

    latent_codes = jax.random.normal(latent_key, (batch_size, latent_dim))
    factors = jax.random.uniform(factors_key, (batch_size, num_factors))

    # Test MIG metric
    mig_config = EvaluationConfig(
        name="mig_test",
        metrics=["mig"],
        metric_params={"mig": {"higher_is_better": True}},
        eval_batch_size=32,
    )
    mig_metric = MutualInformationGapMetric(config=mig_config, rngs=rngs)
    mig_result = mig_metric.compute(factors, latent_codes)
    assert isinstance(mig_result, dict)
    assert "mig_score" in mig_result
    assert 0 <= mig_result["mig_score"] <= 1

    # Test SAP metric
    sap_config = EvaluationConfig(
        name="sap_test",
        metrics=["sap"],
        metric_params={"sap": {"higher_is_better": True}},
        eval_batch_size=32,
    )
    sap_metric = SeparationMetric(config=sap_config, rngs=rngs)
    sap_result = sap_metric.compute(factors, latent_codes)
    assert isinstance(sap_result, dict)
    assert "sap_score" in sap_result

    # Test DCI metric
    dci_config = EvaluationConfig(
        name="dci_test",
        metrics=["dci"],
        metric_params={
            "dci": {
                "higher_is_better": True,
                "weights": {
                    "disentanglement": 0.4,
                    "completeness": 0.4,
                    "informativeness": 0.2,
                },
            }
        },
        eval_batch_size=32,
    )
    dci_metric = DisentanglementMetric(config=dci_config, rngs=rngs)
    dci_results = dci_metric.compute(factors, latent_codes)
    assert isinstance(dci_results, dict)
    assert "dci_score" in dci_results
    assert "disentanglement" in dci_results
    assert "completeness" in dci_results
    assert "informativeness" in dci_results


def test_image_metrics(rngs):
    """Test image quality metrics computation."""
    batch_size = 5
    image_size = 32

    # Create mock images
    if hasattr(rngs, "images1"):
        images1_key = rngs.images1()
    elif hasattr(rngs, "params"):
        images1_key = rngs.params()
    else:
        images1_key = jax.random.key(44)

    if hasattr(rngs, "images2"):
        images2_key = rngs.images2()
    elif hasattr(rngs, "params"):
        images2_key = rngs.params()
    else:
        images2_key = jax.random.key(45)

    images1 = jax.random.uniform(images1_key, (batch_size, image_size, image_size, 3))
    images2 = jax.random.uniform(images2_key, (batch_size, image_size, image_size, 3))

    # Test FID metric
    fid_config = EvaluationConfig(
        name="fid_test",
        metrics=["fid"],
        metric_params={"fid": {"mock_inception": True, "higher_is_better": False}},
        eval_batch_size=16,
    )
    fid_metric = FIDMetric(config=fid_config, rngs=rngs)
    fid_result = fid_metric.compute(images1, images2)
    assert isinstance(fid_result, dict)
    assert "fid_score" in fid_result

    # Test LPIPS metric
    lpips_config = EvaluationConfig(
        name="lpips_test",
        metrics=["lpips"],
        metric_params={"lpips": {"mock_implementation": True, "higher_is_better": False}},
        eval_batch_size=32,
    )
    lpips_metric = LPIPSMetric(config=lpips_config, rngs=rngs)
    lpips_result = lpips_metric.compute(images1, images2)
    assert isinstance(lpips_result, dict)
    assert "lpips_distance" in lpips_result

    # Test SSIM metric
    ssim_config = EvaluationConfig(
        name="ssim_test",
        metrics=["ssim"],
        metric_params={"ssim": {"higher_is_better": True}},
        eval_batch_size=32,
    )
    ssim_metric = SSIMMetric(config=ssim_config, rngs=rngs)
    ssim_result = ssim_metric.compute(images1, images2)
    assert isinstance(ssim_result, dict)
    assert "ssim_score" in ssim_result
    assert 0 <= ssim_result["ssim_score"] <= 1


def test_multi_beta_vae_benchmark(mock_dataset, mock_model, rngs):
    """Test the Multi-β VAE benchmark."""
    benchmark = MultiBetaVAEBenchmark(
        dataset=mock_dataset,
        num_samples=10,
        batch_size=5,
        rngs=rngs,
    )

    # Run benchmark
    result = benchmark.run(mock_model)

    # Check result structure
    assert result.benchmark_name == "multi_beta_vae_controllable_generation"
    assert result.model_name == "MockVAE"
    assert isinstance(result.metrics, dict)
    assert isinstance(result.metadata, dict)

    # Check metrics
    metrics = result.metrics
    assert "mig_score" in metrics or "dci_score" in metrics
    assert "fid_score" in metrics
    assert "lpips_distance" in metrics  # Updated to use lpips_distance
    assert "ssim_score" in metrics


@patch.object(CelebADataset, "_load_celeba_data", mock_celeba_load)
def test_multi_beta_vae_benchmark_suite(mock_model, rngs):
    """Test the Multi-β VAE benchmark suite."""
    # Create benchmark suite
    benchmark_suite = MultiBetaVAEBenchmarkSuite(
        dataset_config={
            "num_samples": 20,
            "image_size": 32,
            "include_attributes": True,
        },
        benchmark_config={
            "num_samples": 10,
            "batch_size": 5,
        },
        rngs=rngs,
    )

    # Check suite initialization
    assert len(benchmark_suite.benchmarks) == 1
    assert isinstance(benchmark_suite.dataset, CelebADataset)

    # Run all benchmarks
    results = benchmark_suite.run_all(mock_model)

    # Check results
    assert len(results) == 1
    benchmark_name = list(results.keys())[0]
    assert benchmark_name == "multi_beta_vae_controllable_generation"

    # Check result structure
    result = results[benchmark_name]
    assert result.benchmark_name == "multi_beta_vae_controllable_generation"
    assert result.model_name == "MockVAE"
    assert isinstance(result.metrics, dict)

    # Check metrics
    metrics = result.metrics
    assert len(metrics) > 0
