"""Unit tests for MCMC utilities for energy-based models."""

import pytest

from tests.artifex.generative_models.models.energy.conftest import jax_required


# Import JAX dependencies conditionally
try:
    import jax
    import jax.numpy as jnp

    from artifex.generative_models.models.energy.mcmc import (
        improved_langevin_dynamics,
        langevin_dynamics,
        langevin_dynamics_with_trajectory,
        persistent_contrastive_divergence,
        SampleBuffer,
    )

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class TestLangevinDynamics:
    """Test Langevin dynamics sampling functions."""

    def simple_energy_fn(self, x):
        """Simple quadratic energy function for testing."""
        return jnp.sum(x**2, axis=-1)

    @jax_required
    def test_langevin_dynamics_basic(self, energy_rng_key, energy_test_mlp_data):
        """Test basic Langevin dynamics functionality."""
        initial_samples = energy_test_mlp_data

        final_samples = langevin_dynamics(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=10,
            step_size=0.01,
            noise_scale=0.005,
            rng_key=energy_rng_key,
        )

        # Check output shape
        assert final_samples.shape == initial_samples.shape
        assert jnp.all(jnp.isfinite(final_samples))

    @jax_required
    def test_langevin_dynamics_convergence(self, energy_rng_key):
        """Test that Langevin dynamics moves towards lower energy regions."""
        # Start with high-energy samples (far from zero)
        key = jax.random.key(42)
        initial_samples = 5.0 + jax.random.normal(key, (4, 8))

        # Run Langevin dynamics
        final_samples = langevin_dynamics(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=50,
            step_size=0.1,
            noise_scale=0.01,
            rng_key=energy_rng_key,
        )

        # Check that final samples have lower energy on average
        initial_energies = self.simple_energy_fn(initial_samples)
        final_energies = self.simple_energy_fn(final_samples)

        assert jnp.mean(final_energies) < jnp.mean(initial_energies)

    @jax_required
    def test_langevin_dynamics_parameters(self, energy_rng_key, energy_test_mlp_data):
        """Test Langevin dynamics with different parameters."""
        initial_samples = energy_test_mlp_data

        # Test different step sizes
        for step_size in [0.001, 0.01, 0.1]:
            samples = langevin_dynamics(
                energy_fn=self.simple_energy_fn,
                initial_samples=initial_samples,
                n_steps=5,
                step_size=step_size,
                rng_key=energy_rng_key,
            )
            assert jnp.all(jnp.isfinite(samples))

        # Test different noise scales
        for noise_scale in [0.001, 0.01, 0.1]:
            samples = langevin_dynamics(
                energy_fn=self.simple_energy_fn,
                initial_samples=initial_samples,
                n_steps=5,
                noise_scale=noise_scale,
                rng_key=energy_rng_key,
            )
            assert jnp.all(jnp.isfinite(samples))

    @jax_required
    def test_langevin_dynamics_clipping(self, energy_rng_key):
        """Test that Langevin dynamics respects clipping ranges."""
        # Create samples that might go out of bounds
        key = jax.random.key(42)
        initial_samples = jax.random.normal(key, (4, 8))

        clip_range = (-0.5, 0.5)

        final_samples = langevin_dynamics(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=10,
            step_size=0.1,
            noise_scale=0.1,
            rng_key=energy_rng_key,
            clip_range=clip_range,
        )

        # Check that all samples are within clipping range
        assert jnp.all(final_samples >= clip_range[0])
        assert jnp.all(final_samples <= clip_range[1])

    @jax_required
    def test_langevin_dynamics_gradient_clipping(self, energy_rng_key):
        """Test gradient clipping in Langevin dynamics."""

        # Energy function with potentially large gradients
        def steep_energy_fn(x):
            return 100.0 * jnp.sum(x**2, axis=-1)

        key = jax.random.key(42)
        initial_samples = jax.random.normal(key, (4, 8))

        final_samples = langevin_dynamics(
            energy_fn=steep_energy_fn,
            initial_samples=initial_samples,
            n_steps=10,
            step_size=0.01,
            grad_clip=0.03,
            rng_key=energy_rng_key,
        )

        # Should still produce finite results despite large gradients
        assert jnp.all(jnp.isfinite(final_samples))

    @jax_required
    def test_langevin_dynamics_default_rng(self, energy_test_mlp_data):
        """Test Langevin dynamics with default RNG."""
        initial_samples = energy_test_mlp_data

        # Should work with rng_key=None (uses default)
        final_samples = langevin_dynamics(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=5,
            rng_key=None,
        )

        assert final_samples.shape == initial_samples.shape
        assert jnp.all(jnp.isfinite(final_samples))


class TestLangevinDynamicsWithTrajectory:
    """Test Langevin dynamics with trajectory recording."""

    def simple_energy_fn(self, x):
        """Simple quadratic energy function for testing."""
        return jnp.sum(x**2, axis=-1)

    @jax_required
    def test_langevin_dynamics_with_trajectory_basic(self, energy_rng_key, energy_test_mlp_data):
        """Test basic trajectory recording functionality."""
        initial_samples = energy_test_mlp_data
        n_steps = 10

        final_samples, trajectory = langevin_dynamics_with_trajectory(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=n_steps,
            step_size=0.01,
            rng_key=energy_rng_key,
        )

        # Check final samples
        assert final_samples.shape == initial_samples.shape
        assert jnp.all(jnp.isfinite(final_samples))

        # Check trajectory
        expected_trajectory_length = n_steps + 1  # Initial + n_steps
        assert trajectory.shape[0] == expected_trajectory_length
        assert trajectory.shape[1:] == initial_samples.shape

    @jax_required
    def test_langevin_dynamics_trajectory_save_every(self, energy_rng_key, energy_test_mlp_data):
        """Test trajectory recording with save_every parameter."""
        initial_samples = energy_test_mlp_data
        n_steps = 10
        save_every = 2

        final_samples, trajectory = langevin_dynamics_with_trajectory(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=n_steps,
            save_every=save_every,
            rng_key=energy_rng_key,
        )

        # Check trajectory length (initial + every save_every steps)
        expected_saves = 1 + (n_steps // save_every)  # Initial + saved steps
        assert trajectory.shape[0] == expected_saves

    @jax_required
    def test_langevin_dynamics_trajectory_consistency(self, energy_rng_key, energy_test_mlp_data):
        """Test that trajectory recording produces same results as regular Langevin."""
        initial_samples = energy_test_mlp_data
        n_steps = 5

        # Run regular Langevin dynamics
        final_regular = langevin_dynamics(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=n_steps,
            step_size=0.01,
            noise_scale=0.005,
            rng_key=energy_rng_key,
        )

        # Run Langevin with trajectory (same seed)
        final_trajectory, trajectory = langevin_dynamics_with_trajectory(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=n_steps,
            step_size=0.01,
            noise_scale=0.005,
            rng_key=energy_rng_key,
        )

        # Final results should be very close (within numerical precision)
        assert jnp.allclose(final_regular, final_trajectory, rtol=1e-5)

        # Trajectory should start with initial samples
        assert jnp.allclose(trajectory[0], initial_samples, rtol=1e-6)

        # Trajectory should end with final samples
        assert jnp.allclose(trajectory[-1], final_trajectory, rtol=1e-6)


class TestSampleBuffer:
    """Test the SampleBuffer class."""

    @jax_required
    def test_sample_buffer_initialization(self):
        """Test sample buffer initialization."""
        buffer = SampleBuffer(capacity=100, reinit_prob=0.1)

        assert buffer.capacity == 100
        assert buffer.reinit_prob == 0.1
        assert buffer.buffer == []

    @jax_required
    def test_sample_buffer_update(self, energy_test_mlp_data):
        """Test updating the sample buffer."""
        buffer = SampleBuffer(capacity=50, reinit_prob=0.1)

        # Add samples to buffer
        buffer.update_buffer(energy_test_mlp_data)

        # Check that samples were added
        assert len(buffer.buffer) == 1
        assert jnp.allclose(buffer.buffer[0], energy_test_mlp_data)

    @jax_required
    def test_sample_buffer_capacity_limit(self, energy_test_batch_size):
        """Test that sample buffer respects capacity limits."""
        capacity = 3
        buffer = SampleBuffer(capacity=capacity, reinit_prob=0.1)

        # Add more samples than capacity
        for i in range(5):
            key = jax.random.key(i)
            samples = jax.random.normal(key, (energy_test_batch_size, 8))
            buffer.update_buffer(samples)

        # Buffer should not exceed capacity
        assert len(buffer.buffer) <= capacity

    @jax_required
    def test_sample_buffer_initial_sampling_fresh_samples(self, energy_rng_key):
        """Test sample buffer initial sampling with fresh samples."""
        buffer = SampleBuffer(capacity=50, reinit_prob=1.0)  # Always reinitialize

        batch_size = 4
        sample_shape = (8,)

        samples = buffer.sample_initial(
            batch_size=batch_size,
            rng_key=energy_rng_key,
            sample_shape=sample_shape,
        )

        # Check output shape
        expected_shape = (batch_size, *sample_shape)
        assert samples.shape == expected_shape
        assert jnp.all(jnp.isfinite(samples))

    @jax_required
    def test_sample_buffer_initial_sampling_from_buffer(self, energy_rng_key):
        """Test sample buffer initial sampling from existing buffer."""
        buffer = SampleBuffer(capacity=50, reinit_prob=0.0)  # Never reinitialize

        # Add samples to buffer first
        key = jax.random.key(42)
        buffer_samples = jax.random.normal(key, (8, 10))
        buffer.update_buffer(buffer_samples)

        batch_size = 4
        sample_shape = (10,)

        samples = buffer.sample_initial(
            batch_size=batch_size,
            rng_key=energy_rng_key,
            sample_shape=sample_shape,
        )

        # Check output shape
        expected_shape = (batch_size, *sample_shape)
        assert samples.shape == expected_shape
        assert jnp.all(jnp.isfinite(samples))

    @jax_required
    def test_sample_buffer_mixed_sampling(self, energy_rng_key):
        """Test sample buffer with mixed sampling strategy."""
        buffer = SampleBuffer(capacity=50, reinit_prob=0.5)  # 50% reinitialize

        # Add samples to buffer
        key = jax.random.key(42)
        buffer_samples = jax.random.normal(key, (16, 8))
        buffer.update_buffer(buffer_samples)

        batch_size = 8
        sample_shape = (8,)

        samples = buffer.sample_initial(
            batch_size=batch_size,
            rng_key=energy_rng_key,
            sample_shape=sample_shape,
        )

        # Should produce valid samples
        expected_shape = (batch_size, *sample_shape)
        assert samples.shape == expected_shape
        assert jnp.all(jnp.isfinite(samples))

    @jax_required
    def test_sample_buffer_no_shape_error(self, energy_rng_key):
        """Test that sample buffer raises error when shape is not provided."""
        buffer = SampleBuffer(capacity=50, reinit_prob=0.5)

        with pytest.raises(ValueError, match="sample_shape must be provided"):
            buffer.sample_initial(
                batch_size=4,
                rng_key=energy_rng_key,
                sample_shape=None,
            )

    @jax_required
    def test_sample_buffer_with_predefined_shape(self, energy_rng_key):
        """Test sample buffer with predefined sample shape."""
        sample_shape = (10,)
        buffer = SampleBuffer(capacity=50, reinit_prob=0.5, sample_shape=sample_shape)

        samples = buffer.sample_initial(
            batch_size=4,
            rng_key=energy_rng_key,
        )

        expected_shape = (4, *sample_shape)
        assert samples.shape == expected_shape


class TestImprovedLangevinDynamics:
    """Test improved Langevin dynamics with adaptive features."""

    def simple_energy_fn(self, x):
        """Simple quadratic energy function for testing."""
        return jnp.sum(x**2, axis=-1)

    @jax_required
    def test_improved_langevin_dynamics_basic(self, energy_rng_key, energy_test_mlp_data):
        """Test basic improved Langevin dynamics functionality."""
        initial_samples = energy_test_mlp_data

        final_samples = improved_langevin_dynamics(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=10,
            step_size=0.01,
            rng_key=energy_rng_key,
        )

        # Check output
        assert final_samples.shape == initial_samples.shape
        assert jnp.all(jnp.isfinite(final_samples))

    @jax_required
    def test_improved_langevin_dynamics_adaptive_step_size(
        self, energy_rng_key, energy_test_mlp_data
    ):
        """Test improved Langevin dynamics with adaptive step size."""
        initial_samples = energy_test_mlp_data

        # Test with adaptive step size enabled
        final_samples_adaptive = improved_langevin_dynamics(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=20,
            step_size=0.01,
            adaptive_step_size=True,
            target_acceptance=0.574,
            rng_key=energy_rng_key,
        )

        # Test with adaptive step size disabled
        final_samples_fixed = improved_langevin_dynamics(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=20,
            step_size=0.01,
            adaptive_step_size=False,
            rng_key=energy_rng_key,
        )

        # Both should produce valid results
        assert jnp.all(jnp.isfinite(final_samples_adaptive))
        assert jnp.all(jnp.isfinite(final_samples_fixed))

    @jax_required
    def test_improved_langevin_dynamics_convergence(self, energy_rng_key):
        """Test that improved Langevin dynamics converges effectively."""
        # Start with high-energy samples
        key = jax.random.key(42)
        initial_samples = 3.0 + jax.random.normal(key, (8, 6))

        final_samples = improved_langevin_dynamics(
            energy_fn=self.simple_energy_fn,
            initial_samples=initial_samples,
            n_steps=50,
            step_size=0.1,
            adaptive_step_size=True,
            rng_key=energy_rng_key,
        )

        # Check convergence to lower energy
        initial_energies = self.simple_energy_fn(initial_samples)
        final_energies = self.simple_energy_fn(final_samples)

        assert jnp.mean(final_energies) < jnp.mean(initial_energies)


class TestPersistentContrastiveDivergence:
    """Test persistent contrastive divergence sampling."""

    def simple_energy_fn(self, x):
        """Simple quadratic energy function for testing."""
        return jnp.sum(x**2, axis=-1)

    @jax_required
    def test_persistent_contrastive_divergence_basic(self, energy_rng_key, energy_test_mlp_data):
        """Test basic persistent contrastive divergence functionality."""
        # Create sample buffer
        sample_buffer = SampleBuffer(capacity=64, reinit_prob=0.1)

        # Add some initial samples to buffer
        sample_buffer.update_buffer(energy_test_mlp_data)

        initial_samples, final_samples = persistent_contrastive_divergence(
            energy_fn=self.simple_energy_fn,
            real_samples=energy_test_mlp_data,
            sample_buffer=sample_buffer,
            rng_key=energy_rng_key,
            n_mcmc_steps=10,
        )

        # Check outputs
        assert initial_samples.shape == energy_test_mlp_data.shape
        assert final_samples.shape == energy_test_mlp_data.shape
        assert jnp.all(jnp.isfinite(initial_samples))
        assert jnp.all(jnp.isfinite(final_samples))

    @jax_required
    def test_persistent_contrastive_divergence_parameters(
        self, energy_rng_key, energy_test_mlp_data
    ):
        """Test persistent contrastive divergence with different parameters."""
        sample_buffer = SampleBuffer(capacity=64, reinit_prob=0.1)
        sample_buffer.update_buffer(energy_test_mlp_data)

        # Test different MCMC parameters
        params_list = [
            {"n_mcmc_steps": 5, "step_size": 0.01, "noise_scale": 0.005},
            {"n_mcmc_steps": 20, "step_size": 0.05, "noise_scale": 0.01},
            {"n_mcmc_steps": 10, "step_size": 0.001, "noise_scale": 0.001},
        ]

        for params in params_list:
            initial_samples, final_samples = persistent_contrastive_divergence(
                energy_fn=self.simple_energy_fn,
                real_samples=energy_test_mlp_data,
                sample_buffer=sample_buffer,
                rng_key=energy_rng_key,
                **params,
            )

            assert jnp.all(jnp.isfinite(initial_samples))
            assert jnp.all(jnp.isfinite(final_samples))

    @jax_required
    def test_persistent_contrastive_divergence_buffer_update(
        self, energy_rng_key, energy_test_mlp_data
    ):
        """Test that persistent contrastive divergence updates the sample buffer."""
        sample_buffer = SampleBuffer(capacity=64, reinit_prob=0.1)

        # Initially empty buffer
        initial_buffer_size = len(sample_buffer.buffer)

        # Run persistent contrastive divergence
        initial_samples, final_samples = persistent_contrastive_divergence(
            energy_fn=self.simple_energy_fn,
            real_samples=energy_test_mlp_data,
            sample_buffer=sample_buffer,
            rng_key=energy_rng_key,
            n_mcmc_steps=10,
        )

        # Buffer should be updated
        final_buffer_size = len(sample_buffer.buffer)
        assert final_buffer_size > initial_buffer_size

    @jax_required
    def test_persistent_contrastive_divergence_energy_descent(self, energy_rng_key):
        """Test that persistent contrastive divergence moves toward lower energy."""
        # Start with high-energy samples
        key = jax.random.key(42)
        high_energy_samples = 2.0 + jax.random.normal(key, (4, 8))

        sample_buffer = SampleBuffer(capacity=32, reinit_prob=0.1)
        sample_buffer.update_buffer(high_energy_samples)

        initial_samples, final_samples = persistent_contrastive_divergence(
            energy_fn=self.simple_energy_fn,
            real_samples=high_energy_samples,
            sample_buffer=sample_buffer,
            rng_key=energy_rng_key,
            n_mcmc_steps=30,
            step_size=0.1,
        )

        # Final samples should have lower energy on average
        initial_energies = self.simple_energy_fn(initial_samples)
        final_energies = self.simple_energy_fn(final_samples)

        assert jnp.mean(final_energies) <= jnp.mean(initial_energies)


class TestMCMCIntegration:
    """Integration tests for MCMC utilities."""

    def quadratic_energy_fn(self, x):
        """Quadratic energy function for integration tests."""
        return jnp.sum(x**2, axis=-1)

    @jax_required
    def test_mcmc_sampling_consistency(self, energy_rng_key):
        """Test consistency across different MCMC sampling methods."""
        # Create initial samples
        key = jax.random.key(42)
        initial_samples = jax.random.normal(key, (8, 6))

        # Test parameters
        n_steps = 20
        step_size = 0.01
        noise_scale = 0.005

        # Standard Langevin dynamics
        samples_langevin = langevin_dynamics(
            energy_fn=self.quadratic_energy_fn,
            initial_samples=initial_samples,
            n_steps=n_steps,
            step_size=step_size,
            noise_scale=noise_scale,
            rng_key=energy_rng_key,
        )

        # Improved Langevin dynamics (without adaptation)
        samples_improved = improved_langevin_dynamics(
            energy_fn=self.quadratic_energy_fn,
            initial_samples=initial_samples,
            n_steps=n_steps,
            step_size=step_size,
            noise_scale=noise_scale,
            adaptive_step_size=False,
            rng_key=energy_rng_key,
        )

        # Both should produce finite results
        assert jnp.all(jnp.isfinite(samples_langevin))
        assert jnp.all(jnp.isfinite(samples_improved))

        # Should have similar energy distributions
        energies_langevin = self.quadratic_energy_fn(samples_langevin)
        energies_improved = self.quadratic_energy_fn(samples_improved)

        # Both should reduce energy from initial state
        initial_energies = self.quadratic_energy_fn(initial_samples)
        assert jnp.mean(energies_langevin) <= jnp.mean(initial_energies)
        assert jnp.mean(energies_improved) <= jnp.mean(initial_energies)

    @jax_required
    def test_complete_ebm_sampling_workflow(self, energy_rng_key):
        """Test complete EBM sampling workflow with MCMC utilities."""
        # Simulate a complete EBM training step using MCMC utilities

        # Create "real" data samples
        key = jax.random.key(42)
        real_samples = jax.random.normal(key, (8, 10))

        # Create sample buffer
        sample_buffer = SampleBuffer(capacity=64, reinit_prob=0.1)

        # Simulate multiple training iterations
        for iteration in range(3):
            # Get initial samples from buffer and run MCMC
            _, fake_samples = persistent_contrastive_divergence(
                energy_fn=self.quadratic_energy_fn,
                real_samples=real_samples,
                sample_buffer=sample_buffer,
                rng_key=jax.random.fold_in(energy_rng_key, iteration),
                n_mcmc_steps=15,
            )

            # Verify samples are valid
            assert jnp.all(jnp.isfinite(fake_samples))
            assert fake_samples.shape == real_samples.shape

            # Buffer should grow (up to capacity)
            assert len(sample_buffer.buffer) >= min(iteration + 1, sample_buffer.capacity)

    @jax_required
    def test_mcmc_numerical_stability(self, energy_rng_key):
        """Test MCMC numerical stability with challenging energy functions."""

        # Energy function with steep gradients
        def steep_energy_fn(x):
            return 50.0 * jnp.sum(x**2, axis=-1)

        # Energy function with sharp minima
        def sharp_energy_fn(x):
            return jnp.sum(jnp.abs(x) ** 4, axis=-1)

        # Create initial samples
        key = jax.random.key(42)
        initial_samples = jax.random.normal(key, (4, 6))

        challenging_energy_fns = [steep_energy_fn, sharp_energy_fn]

        for energy_fn in challenging_energy_fns:
            # Test standard Langevin dynamics
            samples_ld = langevin_dynamics(
                energy_fn=energy_fn,
                initial_samples=initial_samples,
                n_steps=10,
                step_size=0.001,  # Small step size for stability
                grad_clip=0.1,  # Strong gradient clipping
                rng_key=energy_rng_key,
            )

            # Test improved Langevin dynamics
            samples_ild = improved_langevin_dynamics(
                energy_fn=energy_fn,
                initial_samples=initial_samples,
                n_steps=10,
                step_size=0.001,
                adaptive_step_size=True,
                rng_key=energy_rng_key,
            )

            # Both should remain numerically stable
            assert jnp.all(jnp.isfinite(samples_ld))
            assert jnp.all(jnp.isfinite(samples_ild))
