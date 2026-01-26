"""Tests for geometric loss functions."""

import jax.numpy as jnp
import pytest

from artifex.generative_models.core.losses.geometric import (
    binary_cross_entropy,
    chamfer_distance,
    dice_loss,
    earth_mover_distance,
    focal_loss,
    get_mesh_loss,
    get_point_cloud_loss,
    get_voxel_loss,
)


class TestPointCloudLosses:
    """Test point cloud loss functions."""

    def test_chamfer_distance(self):
        """Test chamfer distance calculation."""
        pred = jnp.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
        target = jnp.array([[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]])

        loss = chamfer_distance(pred, target)
        assert loss.shape == ()  # Scalar output
        assert jnp.isscalar(loss) or loss.size == 1
        assert loss >= 0.0  # Loss should be non-negative

        # Perfect match should have near-zero loss
        assert chamfer_distance(pred, pred) < 1e-5

        # Test with factory function
        factory_loss_fn = get_point_cloud_loss("chamfer")
        assert jnp.allclose(factory_loss_fn(pred, target), loss)

    def test_earth_mover_distance(self):
        """Test earth mover distance calculation."""
        pred = jnp.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
        target = jnp.array([[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]])

        loss = earth_mover_distance(pred, target)
        assert loss.shape == ()  # Scalar output
        assert jnp.isscalar(loss) or loss.size == 1
        assert loss >= 0.0  # Loss should be non-negative

        # Perfect match should have near-zero loss
        assert earth_mover_distance(pred, pred) < 1e-5

        # Test with factory function
        factory_loss_fn = get_point_cloud_loss("earth_mover")
        assert jnp.allclose(factory_loss_fn(pred, target), loss)

    def test_invalid_loss_type(self):
        """Test error on invalid loss type."""
        with pytest.raises(ValueError):
            get_point_cloud_loss("invalid_type")


class TestMeshLosses:
    """Test mesh loss functions."""

    def test_mesh_loss(self):
        """Test mesh loss with different weight configurations."""
        # Create simple test mesh data
        pred_vertices = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        pred_faces = jnp.array([[0, 1, 2]])
        pred_normals = jnp.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])

        target_vertices = jnp.array([[[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.9, 0.1]]])
        target_faces = jnp.array([[0, 1, 2]])
        target_normals = jnp.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])

        pred_mesh = (pred_vertices, pred_faces, pred_normals)
        target_mesh = (target_vertices, target_faces, target_normals)

        # Test with default weights
        default_loss_fn = get_mesh_loss()
        default_loss = default_loss_fn(pred_mesh, target_mesh)
        assert default_loss.shape == ()  # Scalar output
        assert jnp.isscalar(default_loss) or default_loss.size == 1
        assert default_loss >= 0.0  # Loss should be non-negative

        # Test with custom weights
        custom_loss_fn = get_mesh_loss(vertex_weight=0.5, normal_weight=0.3, edge_weight=0.2)
        custom_loss = custom_loss_fn(pred_mesh, target_mesh)
        assert custom_loss.shape == ()  # Scalar output
        assert jnp.isscalar(custom_loss) or custom_loss.size == 1
        assert custom_loss >= 0.0  # Loss should be non-negative

        # Perfect match should have near-zero loss
        perfect_loss = get_mesh_loss()(pred_mesh, pred_mesh)
        assert perfect_loss < 1e-5


class TestVoxelLosses:
    """Test voxel loss functions."""

    def test_binary_cross_entropy(self):
        """Test binary cross entropy loss."""
        pred = jnp.array([[[[0.8, 0.2], [0.3, 0.9]]]])  # Shape [1, 1, 2, 2]
        target = jnp.array([[[[1.0, 0.0], [0.0, 1.0]]]])

        loss = binary_cross_entropy(pred, target)
        assert loss.shape == ()  # Scalar output
        assert jnp.isscalar(loss) or loss.size == 1
        assert loss >= 0.0  # Loss should be non-negative

        # Test with factory function
        factory_loss_fn = get_voxel_loss("bce")
        assert jnp.allclose(factory_loss_fn(pred, target), loss)

    def test_binary_cross_entropy_batch_sum(self):
        """Test BCE with batch_sum reduction (standard for VAE ELBO)."""
        # Shape: (2, 2, 2, 1) - 2 samples, 2x2 images, 1 channel
        pred = jnp.array(
            [
                [[[0.9], [0.1]], [[0.1], [0.9]]],  # Sample 1
                [[[0.8], [0.2]], [[0.3], [0.7]]],  # Sample 2
            ]
        )
        target = jnp.array(
            [
                [[[1.0], [0.0]], [[0.0], [1.0]]],  # Sample 1
                [[[1.0], [0.0]], [[0.0], [1.0]]],  # Sample 2
            ]
        )

        # Test batch_sum reduction
        loss_batch_sum = binary_cross_entropy(pred, target, reduction="batch_sum")
        assert loss_batch_sum.shape == ()  # Scalar output
        assert loss_batch_sum >= 0.0

        # Test sum reduction
        loss_sum = binary_cross_entropy(pred, target, reduction="sum")
        assert loss_sum.shape == ()

        # Test mean reduction
        loss_mean = binary_cross_entropy(pred, target, reduction="mean")
        assert loss_mean.shape == ()

        # batch_sum should be between mean and sum
        # (sum over spatial, mean over batch)
        assert loss_mean <= loss_batch_sum <= loss_sum

    def test_binary_cross_entropy_none_reduction(self):
        """Test BCE with no reduction."""
        pred = jnp.array([[[[0.8, 0.2], [0.3, 0.9]]]])
        target = jnp.array([[[[1.0, 0.0], [0.0, 1.0]]]])

        loss = binary_cross_entropy(pred, target, reduction="none")
        assert loss.shape == pred.shape  # Same shape as input

    def test_dice_loss(self):
        """Test dice loss."""
        pred = jnp.array([[[[0.8, 0.2], [0.3, 0.9]]]])
        target = jnp.array([[[[1.0, 0.0], [0.0, 1.0]]]])

        loss = dice_loss(pred, target)
        assert loss.shape == ()  # Scalar output
        assert jnp.isscalar(loss) or loss.size == 1
        assert 0.0 <= loss <= 1.0  # Dice loss is between 0 and 1

        # Perfect match should have near-zero loss
        perfect_pred = jnp.array([[[[1.0, 0.0], [0.0, 1.0]]]])
        assert dice_loss(perfect_pred, target) < 1e-5

        # Test with factory function
        factory_loss_fn = get_voxel_loss("dice")
        assert jnp.allclose(factory_loss_fn(pred, target), loss)

    def test_focal_loss(self):
        """Test focal loss with different gamma values."""
        pred = jnp.array([[[[0.8, 0.2], [0.3, 0.9]]]])
        target = jnp.array([[[[1.0, 0.0], [0.0, 1.0]]]])

        # Default gamma = 2.0
        loss = focal_loss(pred, target)
        assert loss.shape == ()  # Scalar output
        assert jnp.isscalar(loss) or loss.size == 1
        assert loss >= 0.0  # Loss should be non-negative

        # Custom gamma = 3.0
        loss_gamma3 = focal_loss(pred, target, gamma=3.0)
        assert jnp.isscalar(loss_gamma3) or loss_gamma3.size == 1

        # Higher gamma should decrease the loss for well-classified examples
        # and increase it for misclassified examples

        # Test with factory function
        factory_loss_fn = get_voxel_loss("focal", focal_gamma=2.0)
        assert jnp.allclose(factory_loss_fn(pred, target), loss)

        factory_loss_fn_gamma3 = get_voxel_loss("focal", focal_gamma=3.0)
        assert jnp.allclose(factory_loss_fn_gamma3(pred, target), loss_gamma3)

    def test_mse_voxel_loss(self):
        """Test mean squared error loss for voxels."""
        from artifex.generative_models.core.losses.geometric import mse_voxel_loss

        pred = jnp.array([[[[0.8, 0.2], [0.3, 0.9]]]])
        target = jnp.array([[[[1.0, 0.0], [0.0, 1.0]]]])

        # Compute MSE loss
        loss = mse_voxel_loss(pred, target)
        assert loss.shape == ()  # Scalar output
        assert jnp.isscalar(loss) or loss.size == 1
        assert loss >= 0.0  # Loss should be non-negative

        # Manually compute expected MSE
        expected = jnp.mean(jnp.square(pred - target))
        assert jnp.allclose(loss, expected)

        # Test with reduction="sum"
        loss_sum = mse_voxel_loss(pred, target, reduction="sum")
        expected_sum = jnp.sum(jnp.square(pred - target))
        assert jnp.allclose(loss_sum, expected_sum)

        # Test with reduction="none"
        loss_none = mse_voxel_loss(pred, target, reduction="none")
        assert loss_none.shape == pred.shape

        # Test with factory function
        factory_loss_fn = get_voxel_loss("mse")
        assert jnp.allclose(factory_loss_fn(pred, target), loss)

    def test_invalid_loss_type(self):
        """Test error on invalid loss type."""
        with pytest.raises(ValueError):
            get_voxel_loss("invalid_type")
