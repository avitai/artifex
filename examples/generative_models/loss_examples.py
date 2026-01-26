"""Examples of loss functions for generative models.

This module demonstrates various loss functions used in generative models,
including VAE losses, GAN losses, and custom loss compositions.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

# Import the improved loss functions
from artifex.generative_models.core.losses import (
    chamfer_distance,
    # Composable framework
    CompositeLoss,
    # Convenience functions
    create_gan_loss_suite,
    mae_loss,
    MeshLoss,
    # Individual losses
    mse_loss,
    # Specialized losses
    PerceptualLoss,
    ScheduledLoss,
    total_variation_loss,
    WeightedLoss,
)


# Example 1: Simple Functional Usage
def example_functional_usage():
    """Demonstrate simple functional usage of loss functions."""

    # Generate some dummy data
    key = jax.random.key(42)
    key1, key2 = jax.random.split(key)

    predictions = jax.random.normal(key1, (32, 64, 64, 3))
    targets = jax.random.normal(key2, (32, 64, 64, 3))

    # Simple MSE loss
    content_loss = mse_loss(predictions, targets)
    print(f"Content loss: {content_loss}")

    # MAE loss with custom reduction
    mae_content_loss = mae_loss(predictions, targets, reduction="sum")
    print(f"MAE content loss (sum): {mae_content_loss}")

    return content_loss


# Example 2: Composable Loss with NNX Modules
def example_composable_loss():
    """Demonstrate composable loss using NNX modules."""

    # Generate dummy data
    key = jax.random.key(42)
    key1, key2 = jax.random.split(key)

    predictions = jax.random.normal(key1, (32, 64, 64, 3))
    targets = jax.random.normal(key2, (32, 64, 64, 3))

    # Create individual loss components
    content_loss = WeightedLoss(mse_loss, weight=1.0, name="content")
    style_loss = WeightedLoss(mae_loss, weight=0.1, name="style")

    # Combine them
    composite_loss = CompositeLoss([content_loss, style_loss], return_components=True)

    # Compute loss
    total_loss, loss_dict = composite_loss(predictions, targets)

    print(f"Total loss: {total_loss}")
    print(f"Loss components: {loss_dict}")

    return total_loss, loss_dict


# Example 3: VAE Training with Composable Losses
class SimpleVAE(nnx.Module):
    """Simple VAE implementation for demonstration."""

    def __init__(self, input_dim=784, latent_dim=20, hidden_dim=500, *, rngs: nnx.Rngs):
        """Initialize the VAE model.

        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
            rngs: Random number generator keys
        """
        super().__init__()
        self.encoder = nnx.Sequential(
            nnx.Linear(in_features=input_dim, out_features=hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(in_features=hidden_dim, out_features=latent_dim * 2, rngs=rngs),
        )
        self.decoder = nnx.Sequential(
            nnx.Linear(in_features=latent_dim, out_features=hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(in_features=hidden_dim, out_features=input_dim, rngs=rngs),
            nnx.sigmoid,
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        """Encode input to latent distribution parameters.

        Args:
            x: Input data

        Returns:
            Mean and log variance of the latent distribution
        """
        h = self.encoder(x)
        mean, logvar = jnp.split(h, 2, axis=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar, *, rngs: nnx.Rngs):
        """Sample from the latent distribution using reparameterization.

        Args:
            mean: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            rngs: Random number generator keys

        Returns:
            Sampled latent vector
        """
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rngs.sample(), mean.shape)
        return mean + eps * std

    def decode(self, z):
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector

        Returns:
            Reconstructed data
        """
        return self.decoder(z)

    def __call__(self, x, *, rngs: nnx.Rngs):
        """Forward pass through the VAE.

        Args:
            x: Input data
            rngs: Random number generator keys

        Returns:
            Dictionary with reconstruction, mean, and logvar
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, rngs=rngs)
        recon = self.decode(z)
        return {"reconstruction": recon, "mean": mean, "logvar": logvar}


def example_vae_training():
    """Demonstrate VAE training with composable losses."""

    # Create model and optimizer
    model = SimpleVAE(latent_dim=64, rngs=nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.OfType(nnx.Param))

    # Training step
    @nnx.jit
    def train_step(model, optimizer, x, key):
        def loss_fn(model):
            rngs = nnx.Rngs(sample=key)
            outputs = model(x, rngs=rngs)
            # Debug: check what the model returns
            if isinstance(outputs, dict):
                recon = outputs.get("reconstruction", outputs.get("recon"))
                mu = outputs.get("mean", outputs.get("mu"))
                logvar = outputs.get("logvar", outputs.get("log_var"))
            else:
                recon, mu, logvar = outputs

            # Reconstruction loss
            recon_loss = mse_loss(predictions=recon, targets=x)

            # KL divergence loss (standard normal prior)
            kl_loss = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar))
            kl_loss = kl_loss / x.shape[0]  # Normalize by batch size

            # Note: For this example, we compute losses manually
            # In practice, you'd structure this to work with the composite loss
            total_loss = recon_loss + 0.1 * kl_loss
            return total_loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Generate dummy data
    key = jax.random.key(42)
    key, data_key, train_key = jax.random.split(key, 3)

    x = jax.random.normal(data_key, (32, 784))

    # Training step
    loss = train_step(model, optimizer, x, train_key)
    print(f"VAE training loss: {loss}")

    return loss


# Example 4: GAN Training with Composable Losses
def example_gan_training():
    """Demonstrate GAN training setup with composable losses."""

    # Create GAN loss suite
    generator_loss, discriminator_loss = create_gan_loss_suite(
        generator_loss_type="lsgan", discriminator_loss_type="lsgan"
    )

    print(f"Generator loss: {generator_loss}")
    print(f"Discriminator loss: {discriminator_loss}")

    # Dummy data for demonstration
    key = jax.random.key(42)
    real_scores = jax.random.uniform(key, (32,)) * 0.1 + 0.9  # Near 1.0
    fake_scores = jax.random.uniform(key, (32,)) * 0.1 + 0.1  # Near 0.0

    # Compute losses
    gen_loss = generator_loss(fake_scores)
    disc_loss = discriminator_loss(real_scores, fake_scores)

    print(f"Generator loss value: {gen_loss}")
    print(f"Discriminator loss value: {disc_loss}")

    return gen_loss, disc_loss


# Example 5: Advanced Scheduling and Progressive Training
def example_scheduled_loss():
    """Demonstrate scheduled loss for curriculum learning."""

    # Create a perceptual loss that ramps up over time
    perceptual_loss = PerceptualLoss(content_weight=0.1, style_weight=0.01)

    # Schedule function: start at 0, ramp up to full weight over 1000 steps
    def schedule_fn(step):
        return jnp.minimum(1.0, step / 1000.0)

    # Create scheduled loss
    scheduled_loss = ScheduledLoss(perceptual_loss, schedule_fn, name="scheduled_perceptual")

    # Simulate training steps
    dummy_features = {"conv1": jnp.ones((2, 32, 32, 64)), "conv2": jnp.ones((2, 16, 16, 128))}

    for step in [0, 250, 500, 750, 1000]:
        # Note: This would need proper feature extraction in practice
        loss_value = scheduled_loss(
            pred_images=jnp.ones((2, 64, 64, 3)),
            target_images=jnp.zeros((2, 64, 64, 3)),
            features_pred=dummy_features,
            features_target=dummy_features,
            step=step,
        )
        weight = schedule_fn(step)
        print(f"Step {step}: weight={weight:.3f}, loss={loss_value:.6f}")


# Example 6: 3D Geometric Losses
def example_geometric_losses():
    """Demonstrate 3D geometric loss functions."""

    # Point cloud loss
    key = jax.random.key(42)
    key1, key2 = jax.random.split(key)

    pred_points = jax.random.normal(key1, (4, 1000, 3))  # 4 batches, 1000 points each
    target_points = jax.random.normal(key2, (4, 1000, 3))

    # Chamfer distance
    chamfer_loss = chamfer_distance(pred_points, target_points)
    print(f"Chamfer distance: {chamfer_loss}")

    # Mesh loss
    mesh_loss = MeshLoss(
        vertex_weight=1.0, normal_weight=0.1, edge_weight=0.1, laplacian_weight=0.01
    )

    # Dummy mesh data
    vertices = jax.random.normal(key1, (100, 3))
    faces = jax.random.randint(key2, (50, 3), 0, 100)
    normals = jax.random.normal(key1, (100, 3))

    pred_mesh = (vertices, faces, normals)
    target_mesh = (vertices + 0.1, faces, normals)  # Slightly perturbed

    mesh_loss_value = mesh_loss(pred_mesh, target_mesh)
    print(f"Mesh loss: {mesh_loss_value}")

    return chamfer_loss, mesh_loss_value


# Example 7: Complete Training Loop Template
def example_complete_training():
    """Template for a complete training loop using composable losses."""

    # Mock model and data
    class SimpleModel(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            super().__init__()
            self.conv = nnx.Conv(3, 3, kernel_size=(3, 3), rngs=rngs)

        def __call__(self, x):
            return nnx.sigmoid(self.conv(x))

    rngs = nnx.Rngs(42)
    model = SimpleModel(rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.OfType(nnx.Param))

    @nnx.jit
    def train_step(model, optimizer, images, targets):
        def loss_function(model):
            predictions = model(images)

            # For this example, we'll use a simplified version
            # In practice, you'd need proper feature extraction for perceptual loss
            total_loss = mse_loss(predictions, targets)
            tv_loss = total_variation_loss(predictions)

            return total_loss + 0.001 * tv_loss

        loss, grads = nnx.value_and_grad(loss_function)(model)
        optimizer.update(model, grads)
        return loss

    # Training loop
    key = jax.random.key(42)
    for epoch in range(5):
        # Generate dummy batch
        key, data_key = jax.random.split(key)
        images = jax.random.normal(data_key, (8, 64, 64, 3))
        targets = jax.random.normal(data_key, (8, 64, 64, 3))

        loss = train_step(model, optimizer, images, targets)
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

    return model, optimizer


if __name__ == "__main__":
    print("=" * 50)
    print("Loss Function API Usage Examples")
    print("=" * 50)

    print("\n1. Functional Usage:")
    example_functional_usage()

    print("\n2. Composable Loss:")
    example_composable_loss()

    print("\n3. VAE Training:")
    example_vae_training()

    print("\n4. GAN Training:")
    example_gan_training()

    print("\n5. Scheduled Loss:")
    example_scheduled_loss()

    print("\n6. Geometric Losses:")
    example_geometric_losses()

    print("\n7. Complete Training:")
    example_complete_training()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
