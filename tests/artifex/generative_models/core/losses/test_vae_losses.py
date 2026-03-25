"""Tests for canonical VAE loss helpers."""

import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.losses.vae import (
    vae_elbo_terms,
    vae_kl_components,
    vae_reconstruction_loss,
)


class TestVAEReconstructionLoss:
    """Tests for VAE reconstruction loss helper."""

    def test_mse_uses_batch_sum_reduction(self) -> None:
        """MSE reconstruction loss should use VAE-style batch-sum reduction."""
        targets = jnp.ones((2, 4))
        reconstructed = jnp.zeros((2, 4))

        loss = vae_reconstruction_loss(reconstructed, targets, loss_type="mse")

        # Per element error = 1, summed over 4 features and averaged over 2 samples.
        assert loss == pytest.approx(4.0)

    def test_bce_is_finite(self) -> None:
        """BCE reconstruction loss should be numerically stable."""
        targets = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        reconstructed = jnp.array([[0.9, 0.1], [0.2, 0.8]])

        loss = vae_reconstruction_loss(reconstructed, targets, loss_type="bce")

        assert jnp.isfinite(loss)
        assert loss > 0.0

    def test_rejects_unknown_loss_type(self) -> None:
        """Unknown reconstruction loss types should fail clearly."""
        targets = jnp.ones((1, 2))
        reconstructed = jnp.zeros((1, 2))

        with pytest.raises(ValueError, match="Unknown VAE reconstruction loss type"):
            vae_reconstruction_loss(reconstructed, targets, loss_type="invalid")


class TestVAEKLComponents:
    """Tests for VAE KL helper."""

    def test_zero_for_standard_normal(self) -> None:
        """KL should be zero when the posterior matches the unit Gaussian prior."""
        mean = jnp.zeros((4, 3))
        log_var = jnp.zeros((4, 3))

        kl_loss, kl_per_sample = vae_kl_components(mean, log_var)

        assert kl_loss == pytest.approx(0.0, abs=1e-6)
        assert jnp.allclose(kl_per_sample, jnp.zeros((4,)))

    def test_free_bits_clamps_per_dimension_before_sum(self) -> None:
        """Free bits should clamp each latent dimension before batch reduction."""
        mean = jnp.zeros((2, 3))
        log_var = jnp.zeros((2, 3))

        kl_loss, kl_per_sample = vae_kl_components(mean, log_var, free_bits=0.25)

        assert jnp.allclose(kl_per_sample, jnp.full((2,), 0.75))
        assert kl_loss == pytest.approx(0.75)


class TestVAEELBOTerms:
    """Tests for full VAE ELBO helper."""

    def test_combines_reconstruction_and_kl_terms(self) -> None:
        """ELBO helper should combine canonical reconstruction and KL terms."""
        targets = jnp.ones((2, 4))
        reconstructed = jnp.zeros((2, 4))
        mean = jnp.ones((2, 3))
        log_var = jnp.zeros((2, 3))

        losses = vae_elbo_terms(
            reconstructed=reconstructed,
            targets=targets,
            mean=mean,
            log_var=log_var,
            beta=2.0,
            reconstruction_loss_type="mse",
        )

        assert "total_loss" in losses
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        expected_total = losses["reconstruction_loss"] + 2.0 * losses["kl_loss"]
        assert jnp.isclose(losses["total_loss"], expected_total)
        assert "loss" not in losses

    def test_supports_custom_reconstruction_loss(self) -> None:
        """Custom reconstruction loss functions should still use the shared KL path."""
        targets = jnp.ones((2, 4))
        reconstructed = jnp.zeros((2, 4))
        mean = jnp.zeros((2, 3))
        log_var = jnp.zeros((2, 3))

        def l1_loss(predictions: jax.Array, references: jax.Array) -> jax.Array:
            return jnp.mean(jnp.abs(predictions - references))

        losses = vae_elbo_terms(
            reconstructed=reconstructed,
            targets=targets,
            mean=mean,
            log_var=log_var,
            reconstruction_loss_fn=l1_loss,
        )

        assert losses["reconstruction_loss"] == pytest.approx(1.0)
        assert losses["kl_loss"] == pytest.approx(0.0, abs=1e-6)

    def test_is_jittable(self) -> None:
        """ELBO helper should remain JIT-compilable."""
        targets = jnp.ones((2, 4))
        reconstructed = jnp.full((2, 4), 0.25)
        mean = jnp.ones((2, 3))
        log_var = jnp.zeros((2, 3))

        @jax.jit
        def compute_loss(
            reconstructed: jax.Array,
            targets: jax.Array,
            mean: jax.Array,
            log_var: jax.Array,
        ) -> dict[str, jax.Array]:
            return vae_elbo_terms(
                reconstructed=reconstructed,
                targets=targets,
                mean=mean,
                log_var=log_var,
                beta=2.0,
                reconstruction_loss_type="mse",
            )

        losses = compute_loss(reconstructed, targets, mean, log_var)

        assert jnp.isfinite(losses["total_loss"])
        assert jnp.isfinite(losses["reconstruction_loss"])
        assert jnp.isfinite(losses["kl_loss"])

    def test_gradients_are_finite(self) -> None:
        """ELBO helper should remain differentiable w.r.t. core tensor inputs."""
        targets = jnp.ones((2, 4))
        reconstructed = jnp.full((2, 4), 0.25)
        mean = jnp.ones((2, 3))
        log_var = jnp.zeros((2, 3))

        def scalar_loss(
            reconstructed: jax.Array,
            mean: jax.Array,
            log_var: jax.Array,
        ) -> jax.Array:
            return vae_elbo_terms(
                reconstructed=reconstructed,
                targets=targets,
                mean=mean,
                log_var=log_var,
                beta=2.0,
                reconstruction_loss_type="mse",
            )["total_loss"]

        reconstructed_grad, mean_grad, log_var_grad = jax.grad(
            scalar_loss,
            argnums=(0, 1, 2),
        )(reconstructed, mean, log_var)

        assert jnp.isfinite(reconstructed_grad).all()
        assert jnp.isfinite(mean_grad).all()
        assert jnp.isfinite(log_var_grad).all()
