"""
Comprehensive test suite for the improved loss function API (Memory Efficient Version).

This test suite validates all components of the loss function library,
ensuring correctness, JAX compatibility, and proper NNX integration,
while being memory-efficient for CI/testing environments.
"""

import jax
import jax.numpy as jnp
from flax import nnx

# Import all the improved loss functions
from artifex.generative_models.core.losses import (
    binary_cross_entropy,
    # Geometric losses
    chamfer_distance,
    # Composable framework
    CompositeLoss,
    # Perceptual losses
    contextual_loss,
    # Convenience functions
    create_gan_loss_suite,
    create_vae_loss_suite,
    l1_regularization,
    l2_regularization,
    LossCollection,
    LossMetrics,
    mae_loss,
    MeshLoss,
    # Individual losses
    mse_loss,
    PerceptualLoss,
    # Base utilities
    reduce_loss,
    ScheduledLoss,
    WeightedLoss,
)


class TestBaseFunctionality:
    """Test base loss functionality."""

    def test_reduce_loss(self):
        """Test loss reduction functionality."""
        # Use smaller arrays to avoid memory issues
        loss = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Test mean reduction
        mean_loss = reduce_loss(loss, reduction="mean")
        assert jnp.allclose(mean_loss, 2.5)

        # Test sum reduction
        sum_loss = reduce_loss(loss, reduction="sum")
        assert jnp.allclose(sum_loss, 10.0)

        # Test no reduction
        none_loss = reduce_loss(loss, reduction="none")
        assert jnp.allclose(none_loss, loss)

        # Test with weights
        weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        weighted_loss = reduce_loss(loss, reduction="mean", weights=weights)
        expected = jnp.mean(loss * weights)
        assert jnp.allclose(weighted_loss, expected)

        # Test axis parameter
        loss_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        axis_loss = reduce_loss(loss_2d, reduction="mean", axis=1)
        expected = jnp.array([1.5, 3.5])
        assert jnp.allclose(axis_loss, expected)

    def test_loss_collection(self):
        """Test functional loss collection."""
        collection = LossCollection()

        # Add some loss functions
        collection.add(mse_loss, weight=1.0, name="content")
        collection.add(mae_loss, weight=0.5, name="style")

        # Test computation with small arrays
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([0.0, 0.0, 0.0])

        total_loss, loss_dict = collection(pred, target)

        # Verify individual losses
        expected_mse = jnp.mean((pred - target) ** 2)
        expected_mae = jnp.mean(jnp.abs(pred - target))

        assert jnp.allclose(loss_dict["content"], expected_mse)
        assert jnp.allclose(loss_dict["style"], expected_mae)

        # Verify total loss
        expected_total = 1.0 * expected_mse + 0.5 * expected_mae
        assert jnp.allclose(total_loss, expected_total)

        # Test weight modification
        collection.set_weight("style", 0.1)
        weights = collection.get_weights()
        assert weights["style"] == 0.1

        # Test removal
        collection.remove("style")
        assert len(collection) == 1


class TestComposableLosses:
    """Test NNX composable loss framework."""

    def test_weighted_loss(self):
        """Test weighted loss wrapper."""
        weight = 2.5
        weighted_loss = WeightedLoss(mse_loss, weight=weight, name="weighted_mse")

        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([0.0, 0.0, 0.0])

        loss_value = weighted_loss(pred, target)
        expected = weight * mse_loss(pred, target)

        assert jnp.allclose(loss_value, expected)
        assert weighted_loss.name == "weighted_mse"

    def test_composite_loss(self):
        """Test composite loss module."""
        content_loss = WeightedLoss(mse_loss, weight=1.0, name="content")
        style_loss = WeightedLoss(mae_loss, weight=0.5, name="style")

        composite = CompositeLoss([content_loss, style_loss], return_components=True)

        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([0.0, 0.0, 0.0])

        total_loss, loss_dict = composite(pred, target)

        # Verify components
        assert "content" in loss_dict
        assert "style" in loss_dict

        # Calculate expected values step by step to debug
        mse_val = mse_loss(pred, target)  # Should be (1^2 + 2^2 + 3^2)/3 = 14/3 ≈ 4.67
        mae_val = mae_loss(pred, target)  # Should be (1 + 2 + 3)/3 = 2.0

        expected_total = 1.0 * mse_val + 0.5 * mae_val

        print(
            f"MSE: {mse_val}, MAE: {mae_val}, Expected total: {expected_total}, Actual total: {total_loss}"
        )

        # Verify individual components
        assert jnp.allclose(loss_dict["content"], mse_val, rtol=1e-5)
        assert jnp.allclose(loss_dict["style"], mae_val, rtol=1e-5)

        # Verify total with more relaxed tolerance
        assert jnp.allclose(total_loss, expected_total, rtol=1e-5)

        # Test without components
        composite_simple = CompositeLoss([content_loss, style_loss], return_components=False)
        simple_loss = composite_simple(pred, target)
        assert jnp.allclose(simple_loss, expected_total, rtol=1e-5)

    def test_scheduled_loss(self):
        """Test scheduled loss functionality."""

        def linear_schedule(step):
            return min(1.0, step / 100.0)

        scheduled = ScheduledLoss(mse_loss, linear_schedule, name="scheduled_mse")

        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([0.0, 0.0, 0.0])

        # Test at different steps
        loss_0 = scheduled(pred, target, step=0)
        loss_50 = scheduled(pred, target, step=50)
        loss_100 = scheduled(pred, target, step=100)

        base_loss = mse_loss(pred, target)

        assert jnp.allclose(loss_0, 0.0 * base_loss)
        assert jnp.allclose(loss_50, 0.5 * base_loss)
        assert jnp.allclose(loss_100, 1.0 * base_loss)


class TestReconstructionLosses:
    """Test reconstruction loss functions."""

    def test_mse_loss(self):
        """Test MSE loss."""
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([0.0, 0.0, 0.0])

        loss = mse_loss(pred, target)
        expected = jnp.mean((pred - target) ** 2)
        assert jnp.allclose(loss, expected)

        # Test with different reductions
        loss_sum = mse_loss(pred, target, reduction="sum")
        expected_sum = jnp.sum((pred - target) ** 2)
        assert jnp.allclose(loss_sum, expected_sum)

    def test_mae_loss(self):
        """Test MAE loss."""
        pred = jnp.array([1.0, -2.0, 3.0])
        target = jnp.array([0.0, 0.0, 0.0])

        loss = mae_loss(pred, target)
        expected = jnp.mean(jnp.abs(pred - target))
        assert jnp.allclose(loss, expected)


class TestGeometricLosses:
    """Test geometric loss functions."""

    def test_chamfer_distance(self):
        """Test Chamfer distance for point clouds."""
        # Create small test case to avoid memory issues
        pred_points = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])  # [1, 2, 3]
        target_points = jnp.array([[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]])  # [1, 2, 3]

        loss = chamfer_distance(pred_points, target_points)

        # Should be small since points are close
        assert loss < 1.0
        assert loss >= 0.0  # Non-negative

        # Test with identical points
        identical_loss = chamfer_distance(pred_points, pred_points)
        assert jnp.allclose(identical_loss, 0.0, atol=1e-6)

    def test_mesh_loss(self):
        """Test mesh loss functionality."""
        mesh_loss = MeshLoss(vertex_weight=1.0, normal_weight=0.1, edge_weight=0.1)

        # Create small mesh data
        vertices = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = jnp.array([[0, 1, 2]])
        normals = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        pred_mesh = (vertices, faces, normals)
        target_mesh = (vertices + 0.1, faces, normals)  # Slightly offset

        loss = mesh_loss(pred_mesh, target_mesh)

        assert loss > 0.0  # Should have non-zero loss due to offset
        assert jnp.isfinite(loss)  # Should be finite

    def test_voxel_losses(self):
        """Test voxel-based losses."""
        # Small voxel grids
        pred = jnp.array([[[0.8, 0.2], [0.3, 0.7]]])  # [1, 2, 2]
        target = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])  # [1, 2, 2]

        # Binary cross-entropy
        bce_loss = binary_cross_entropy(pred, target)
        assert bce_loss > 0.0
        assert jnp.isfinite(bce_loss)


class TestPerceptualLosses:
    """Test perceptual loss functions."""

    def test_contextual_loss(self):
        """Test contextual loss computation with small feature maps."""
        # Create small feature maps to avoid memory issues
        feat_real = jnp.ones((2, 4, 4, 8))  # [B, H, W, C] - small spatial size
        feat_fake = jnp.zeros((2, 4, 4, 8))

        loss = contextual_loss(feat_real, feat_fake, max_samples=16)  # Limit samples

        assert loss > 0.0  # Should have positive loss for different features
        assert jnp.isfinite(loss)

        # Test with identical features
        identical_loss = contextual_loss(feat_real, feat_real, max_samples=16)
        # Should be small (close to 0) for identical features
        assert identical_loss < 1.0

    def test_perceptual_loss_module(self):
        """Test perceptual loss NNX module with small inputs."""
        perceptual = PerceptualLoss(
            content_weight=1.0,
            style_weight=0.1,
            contextual_weight=0.05,
            max_contextual_samples=16,  # Limit for memory efficiency
        )

        # Mock feature extraction with small features
        features_real = {"conv1": jnp.ones((2, 4, 4, 8))}  # Small feature maps
        features_fake = {"conv1": jnp.zeros((2, 4, 4, 8))}

        pred_images = jnp.zeros((2, 16, 16, 3))  # Small images
        target_images = jnp.ones((2, 16, 16, 3))

        loss = perceptual(
            pred_images, target_images, features_pred=features_fake, features_target=features_real
        )

        assert loss > 0.0
        assert jnp.isfinite(loss)


class TestRegularizationLosses:
    """Test regularization loss functions."""

    def test_l1_regularization(self):
        """Test L1 regularization."""
        # Test with simple parameters
        params = {"weight": jnp.array([1.0, -2.0, 3.0])}

        l1_loss = l1_regularization(params, scale=0.1)
        expected = 0.1 * jnp.sum(jnp.abs(jnp.array([1.0, -2.0, 3.0])))
        assert jnp.allclose(l1_loss, expected)

        # Test with small NNX model
        model = nnx.Linear(3, 2, rngs=nnx.Rngs(42))
        model_params = nnx.state(model, nnx.Param)

        l1_model_loss = l1_regularization(model_params, scale=0.01)
        assert l1_model_loss >= 0.0
        assert jnp.isfinite(l1_model_loss)

    def test_l2_regularization(self):
        """Test L2 regularization."""
        params = {"weight": jnp.array([1.0, -2.0, 3.0])}

        l2_loss = l2_regularization(params, scale=0.1)
        expected = 0.1 * jnp.sum(jnp.square(jnp.array([1.0, -2.0, 3.0])))
        assert jnp.allclose(l2_loss, expected)


class TestJAXCompatibility:
    """Test JAX transformation compatibility."""

    def test_jit_compilation(self):
        """Test that loss functions can be JIT compiled."""

        @nnx.jit
        def compiled_loss(pred, target):
            return mse_loss(pred, target)

        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([0.0, 0.0, 0.0])

        loss = compiled_loss(pred, target)
        expected = mse_loss(pred, target)

        assert jnp.allclose(loss, expected)

    def test_gradient_computation(self):
        """Test gradient computation through loss functions."""

        def loss_fn(params):
            pred = params["pred"]
            target = jnp.array([0.0, 0.0, 0.0])
            return mse_loss(pred, target)

        params = {"pred": jnp.array([1.0, 2.0, 3.0])}

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(params)

        # Gradients should exist and be finite
        assert "pred" in grads
        assert jnp.all(jnp.isfinite(grads["pred"]))

    def test_vmap_compatibility(self):
        """Test vectorization compatibility."""
        # Small batch of predictions and targets
        pred_batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        target_batch = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        # Vectorize MSE loss over batch dimension
        vmapped_loss = nnx.vmap(lambda p, t: mse_loss(p, t, reduction="none"))

        losses = vmapped_loss(pred_batch, target_batch)

        assert losses.shape == (3, 2)  # [batch_size, feature_dim]
        assert jnp.all(jnp.isfinite(losses))


class TestNNXIntegration:
    """Test Flax NNX integration."""

    def test_nnx_module_usage(self):
        """Test using loss modules with NNX patterns."""

        # Create a simple small model
        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                super().__init__()
                self.linear = nnx.Linear(3, 2, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(nnx.Rngs(42))

        # Create composite loss with regularization
        content_loss = WeightedLoss(mse_loss, weight=1.0, name="content")
        reg_loss = WeightedLoss(
            lambda pred, target: l1_regularization(nnx.state(model, nnx.Param), scale=0.01),
            weight=1.0,
            name="regularization",
        )

        composite_loss = CompositeLoss([content_loss, reg_loss], return_components=True)

        # Test computation with small data
        x = jnp.ones((2, 3))  # Small batch
        pred = model(x)
        target = jnp.zeros((2, 2))

        total_loss, loss_dict = composite_loss(pred, target)

        assert "content" in loss_dict
        assert "regularization" in loss_dict
        assert jnp.isfinite(total_loss)

    def test_loss_metrics_tracking(self):
        """Test loss metrics tracking."""
        metrics = LossMetrics(momentum=0.9)

        # Simulate training steps
        for step in range(5):
            loss_dict = {
                "content": jnp.array(1.0 / (step + 1)),
                "style": jnp.array(0.5 / (step + 1)),
            }
            metrics.update(loss_dict)

        current_metrics = metrics.get_metrics()

        assert "content" in current_metrics
        assert "style" in current_metrics
        assert all(v > 0 for v in current_metrics.values())


class TestConvenienceFunctions:
    """Test convenience functions for common use cases."""

    def test_gan_loss_suite(self):
        """Test GAN loss suite creation."""
        gen_loss, disc_loss = create_gan_loss_suite(
            generator_loss_type="vanilla", discriminator_loss_type="vanilla"
        )

        # Test with small dummy data
        fake_scores = jnp.array([0.1, 0.2, 0.3])
        real_scores = jnp.array([0.8, 0.9, 0.7])

        gen_loss_value = gen_loss(fake_scores)
        disc_loss_value = disc_loss(real_scores, fake_scores)

        assert jnp.isfinite(gen_loss_value)
        assert jnp.isfinite(disc_loss_value)

    def test_vae_loss_suite(self):
        """Test VAE loss suite creation."""
        vae_loss = create_vae_loss_suite(reconstruction_loss_type="mse", kl_weight=0.1)

        # Note: This test would require properly structured VAE outputs
        # For now, just verify the loss module was created correctly
        assert isinstance(vae_loss, CompositeLoss)
        assert vae_loss.return_components is True


def run_performance_tests():
    """Run performance benchmarks with memory-efficient settings."""
    print("Running performance tests...")

    # Medium-scale test (reduced from original large scale)
    key = jax.random.key(42)
    medium_pred = jax.random.normal(key, (16, 64, 64, 3))  # Reduced from 100x256x256x3
    medium_target = jax.random.normal(key, (16, 64, 64, 3))

    # Time MSE computation
    import time

    @nnx.jit
    def compute_loss(pred, target):
        return mse_loss(pred, target)

    # Warmup
    _ = compute_loss(medium_pred, medium_target)

    # Actual timing
    start_time = time.time()
    for _ in range(10):
        loss = compute_loss(medium_pred, medium_target)
        loss.block_until_ready()  # Ensure computation completes
    end_time = time.time()

    avg_time = (end_time - start_time) / 10
    print(f"Average MSE computation time (16x64x64x3): {avg_time:.4f} seconds")

    # Memory usage test with smaller data
    print("Testing memory efficiency...")

    # Create a complex composite loss
    composite = CompositeLoss(
        [
            WeightedLoss(mse_loss, weight=1.0, name="content"),
            WeightedLoss(mae_loss, weight=0.5, name="style"),
        ],
        return_components=True,
    )

    total_loss, loss_dict = composite(medium_pred, medium_target)
    print(f"Composite loss computed successfully: {float(total_loss):.6f}")


if __name__ == "__main__":
    print("Running memory-efficient test suite...")
    print("=" * 60)

    # Run test classes
    test_classes = [
        TestBaseFunctionality,
        TestComposableLosses,
        TestReconstructionLosses,
        TestGeometricLosses,
        TestPerceptualLosses,
        TestRegularizationLosses,
        TestJAXCompatibility,
        TestNNXIntegration,
        TestConvenienceFunctions,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        test_instance = test_class()

        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    getattr(test_instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {str(e)}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed!")

        # Run performance tests if all functional tests pass
        print("\n" + "=" * 60)
        run_performance_tests()
    else:
        print("❌ Some tests failed. Please review the failures above.")

    print("\nTest suite completed.")
