"""Property-based tests for energy models.

These tests verify mathematical properties, invariants, and theoretical expectations
of energy-based models using property-based testing and mathematical verification.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.models.energy import (
    improved_langevin_dynamics,
    langevin_dynamics,
    SampleBuffer,
)
from artifex.generative_models.models.energy.base import (
    EnergyBasedModel,
    MLPEnergyFunction,
)
from tests.artifex.generative_models.models.energy.conftest import jax_required


class TestEnergyFunctionProperties:
    """Test mathematical properties of energy functions."""

    @jax_required
    def test_energy_function_determinism(self, energy_rngs):
        """Test that energy functions are deterministic given the same input."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=10,
            rngs=energy_rngs,
        )

        # Create test input
        key = jax.random.key(42)
        test_input = jax.random.normal(key, (4, 10))

        # Compute energy multiple times
        energy1 = energy_fn(test_input)
        energy2 = energy_fn(test_input)
        energy3 = energy_fn(test_input)

        # Should be identical
        assert jnp.allclose(energy1, energy2, atol=1e-8)
        assert jnp.allclose(energy2, energy3, atol=1e-8)

    @jax_required
    def test_energy_function_batch_independence(self, energy_rngs):
        """Test that energy function evaluates each batch element independently."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=10,
            rngs=energy_rngs,
        )

        # Create test inputs
        key = jax.random.key(42)
        input1 = jax.random.normal(key, (1, 10))
        input2 = jax.random.normal(jax.random.key(43), (1, 10))

        # Compute energies individually
        energy1_single = energy_fn(input1)
        energy2_single = energy_fn(input2)

        # Compute energies as batch
        batch_input = jnp.concatenate([input1, input2], axis=0)
        energies_batch = energy_fn(batch_input)

        # Should match individual computations (with relaxed tolerance for GPU numerical differences)
        assert jnp.allclose(energies_batch[0], energy1_single[0], atol=1e-4)
        assert jnp.allclose(energies_batch[1], energy2_single[0], atol=1e-4)

    @jax_required
    def test_energy_function_scaling_properties(self, energy_rngs):
        """Test energy function behavior under input scaling."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=8,
            rngs=energy_rngs,
        )

        # Create test input
        key = jax.random.key(42)
        base_input = jax.random.normal(key, (4, 8))

        # Test different scales
        scales = [0.1, 0.5, 1.0, 2.0, 5.0]
        energies = []

        for scale in scales:
            scaled_input = scale * base_input
            energy = energy_fn(scaled_input)
            energies.append(energy)

        # Energy should change with scale (not be scale-invariant)
        # This verifies that the network is actually using the input
        for i in range(1, len(energies)):
            assert not jnp.allclose(energies[0], energies[i], atol=1e-3)

    @jax_required
    def test_energy_function_gradient_finiteness(self, energy_rngs):
        """Test that energy function gradients are finite."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=6,
            rngs=energy_rngs,
        )

        # Create test input
        key = jax.random.key(42)
        test_input = jax.random.normal(key, (4, 6))

        # Compute gradients
        def total_energy(x):
            return jnp.sum(energy_fn(x))

        grad_fn = jax.grad(total_energy)
        gradients = grad_fn(test_input)

        # Gradients should be finite
        assert jnp.all(jnp.isfinite(gradients))
        assert gradients.shape == test_input.shape

    @jax_required
    def test_energy_function_symmetries(self, energy_rngs):
        """Test energy function behavior with respect to potential symmetries."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=4,
            rngs=energy_rngs,
        )

        # Create test input
        key = jax.random.key(42)
        test_input = jax.random.normal(key, (2, 4))

        # Test that permuting batch order doesn't affect individual energies
        permuted_input = test_input[jnp.array([1, 0])]  # Swap batch elements

        original_energies = energy_fn(test_input)
        permuted_energies = energy_fn(permuted_input)

        # Individual energies should match after undoing permutation
        assert jnp.allclose(original_energies[0], permuted_energies[1], atol=1e-6)
        assert jnp.allclose(original_energies[1], permuted_energies[0], atol=1e-6)


class TestEnergyBasedModelProperties:
    """Test mathematical properties of energy-based models."""

    @jax_required
    def test_energy_log_prob_relationship(self, energy_rngs):
        """Test the fundamental relationship: log_prob = -energy."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=10,
            rngs=energy_rngs,
        )

        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Create test data
        key = jax.random.key(42)
        test_data = jax.random.normal(key, (6, 10))

        # Compute energies and log probabilities
        energies = ebm.energy(test_data)
        log_probs = ebm.unnormalized_log_prob(test_data)

        # Verify relationship
        assert jnp.allclose(log_probs, -energies, atol=1e-8)

    @jax_required
    def test_score_function_gradient_relationship(self, energy_rngs):
        """Test that score function equals negative gradient of energy."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=8,
            rngs=energy_rngs,
        )

        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Create test data
        key = jax.random.key(42)
        test_data = jax.random.normal(key, (4, 8))

        # Compute score using model method
        scores = ebm.score(test_data)

        # Compute score as negative gradient manually
        def energy_sum(x):
            return jnp.sum(ebm.energy(x))

        manual_scores = -jax.grad(energy_sum)(test_data)

        # Should match
        assert jnp.allclose(scores, manual_scores, atol=1e-5)

    @jax_required
    def test_contrastive_divergence_loss_properties(self, energy_rngs):
        """Test properties of contrastive divergence loss."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=10,
            rngs=energy_rngs,
        )

        ebm = EnergyBasedModel(energy_fn, rngs=energy_rngs)

        # Create real and fake data
        key = jax.random.key(42)
        real_data = jax.random.normal(key, (8, 10))
        fake_data = jax.random.normal(jax.random.key(43), (8, 10))

        # Compute loss
        loss_dict = ebm.contrastive_divergence_loss(
            real_data=real_data,
            fake_data=fake_data,
            alpha=0.01,
        )

        # Loss should be finite
        assert jnp.isfinite(loss_dict["loss"])
        assert jnp.isfinite(loss_dict["contrastive_divergence"])
        assert jnp.isfinite(loss_dict["regularization"])

        # CD term should equal difference in energy means
        real_energy_mean = jnp.mean(ebm.energy(real_data))
        fake_energy_mean = jnp.mean(ebm.energy(fake_data))
        expected_cd = real_energy_mean - fake_energy_mean

        assert jnp.allclose(loss_dict["contrastive_divergence"], expected_cd, atol=1e-5)

    @jax_required
    def test_generation_consistency(self, energy_rngs):
        """Test consistency properties of sample generation.

        Since generate() uses internal rngs, reproducibility requires
        creating two separate EBM instances with the same seed.
        """
        # Create two identical energy functions and EBMs with same seed
        energy_fn1 = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=6,
            rngs=nnx.Rngs(1000),
        )
        ebm1 = EnergyBasedModel(energy_fn1, rngs=nnx.Rngs(1000))

        energy_fn2 = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=6,
            rngs=nnx.Rngs(1000),
        )
        ebm2 = EnergyBasedModel(energy_fn2, rngs=nnx.Rngs(1000))

        # Generate samples from each (uses internal rngs)
        samples1 = ebm1.generate(
            n_samples=4,
            shape=(6,),
            n_steps=10,
        )

        samples2 = ebm2.generate(
            n_samples=4,
            shape=(6,),
            n_steps=10,
        )

        # Should be identical with same seed (same init = same RNG sequence)
        assert jnp.allclose(samples1, samples2, atol=1e-6)

        # Generated samples should be finite
        assert jnp.all(jnp.isfinite(samples1))


class TestMCMCProperties:
    """Test mathematical properties of MCMC sampling."""

    def quadratic_energy(self, x):
        """Simple quadratic energy for testing."""
        return jnp.sum(x**2, axis=-1)

    @jax_required
    def test_langevin_dynamics_detailed_balance(self, energy_rng_key):
        """Test that Langevin dynamics satisfies detailed balance in expectation."""
        # For a quadratic energy, Langevin dynamics should converge to Gaussian
        key = jax.random.key(42)
        initial_samples = 2.0 + jax.random.normal(key, (100, 4))

        # Run many steps
        final_samples = langevin_dynamics(
            energy_fn=self.quadratic_energy,
            initial_samples=initial_samples,
            n_steps=100,
            step_size=0.05,
            noise_scale=0.1,
            rng_key=energy_rng_key,
        )

        # For quadratic energy E(x) = ||x||^2, equilibrium is standard Gaussian
        # Check that sample statistics approach theoretical values
        sample_mean = jnp.mean(final_samples, axis=0)
        sample_var = jnp.var(final_samples, axis=0)

        # Mean should be close to zero
        assert jnp.allclose(
            sample_mean, 0.0, atol=0.3
        )  # Allow some deviation due to finite sampling

        # Variance should be reasonable (not exact due to finite time and discretization)
        assert jnp.all(sample_var > 0.1)  # Should have some variance
        assert jnp.all(sample_var < 3.0)  # But not too much

    @jax_required
    def test_langevin_dynamics_energy_descent(self, energy_rng_key):
        """Test that Langevin dynamics reduces energy on average."""
        # Start with high-energy samples
        key = jax.random.key(42)
        initial_samples = 3.0 + jax.random.normal(key, (20, 6))

        # Run Langevin dynamics
        final_samples = langevin_dynamics(
            energy_fn=self.quadratic_energy,
            initial_samples=initial_samples,
            n_steps=50,
            step_size=0.1,
            noise_scale=0.01,
            rng_key=energy_rng_key,
        )

        # Energy should decrease on average
        initial_energies = self.quadratic_energy(initial_samples)
        final_energies = self.quadratic_energy(final_samples)

        assert jnp.mean(final_energies) < jnp.mean(initial_energies)

    @jax_required
    def test_mcmc_invariance_properties(self, energy_rng_key):
        """Test invariance properties of MCMC methods."""
        # Test that translation invariance is broken (as expected for quadratic energy)
        key = jax.random.key(42)
        samples1 = jax.random.normal(key, (10, 4))
        samples2 = samples1 + 1.0  # Translate all samples

        # Run Langevin dynamics
        final1 = langevin_dynamics(
            energy_fn=self.quadratic_energy,
            initial_samples=samples1,
            n_steps=20,
            step_size=0.05,
            rng_key=energy_rng_key,
        )

        final2 = langevin_dynamics(
            energy_fn=self.quadratic_energy,
            initial_samples=samples2,
            n_steps=20,
            step_size=0.05,
            rng_key=energy_rng_key,
        )

        # For quadratic energy, translation is NOT preserved
        # (both should move toward zero)
        mean1 = jnp.mean(final1, axis=0)
        mean2 = jnp.mean(final2, axis=0)

        # Both means should be closer to zero than their initial values
        initial_mean1 = jnp.mean(samples1, axis=0)
        initial_mean2 = jnp.mean(samples2, axis=0)

        assert jnp.linalg.norm(mean1) < jnp.linalg.norm(initial_mean1)
        assert jnp.linalg.norm(mean2) < jnp.linalg.norm(initial_mean2)

    @jax_required
    def test_improved_langevin_convergence_properties(self, energy_rng_key):
        """Test convergence properties of improved Langevin dynamics."""
        key = jax.random.key(42)
        initial_samples = 2.0 + jax.random.normal(key, (16, 6))

        # Run improved Langevin with adaptation
        final_samples = improved_langevin_dynamics(
            energy_fn=self.quadratic_energy,
            initial_samples=initial_samples,
            n_steps=50,
            step_size=0.1,
            adaptive_step_size=True,
            target_acceptance=0.574,
            rng_key=energy_rng_key,
        )

        # Should converge to lower energy
        initial_energies = self.quadratic_energy(initial_samples)
        final_energies = self.quadratic_energy(final_samples)

        # Mean energy should decrease
        assert jnp.mean(final_energies) < jnp.mean(initial_energies)

        # Samples should be finite
        assert jnp.all(jnp.isfinite(final_samples))


class TestSampleBufferProperties:
    """Test properties of the sample buffer."""

    @jax_required
    def test_buffer_capacity_invariant(self):
        """Test that buffer never exceeds its capacity."""
        capacity = 10
        buffer = SampleBuffer(capacity=capacity, reinit_prob=0.1)

        # Add many batches
        key = jax.random.key(42)
        for i in range(20):  # More than capacity
            key, subkey = jax.random.split(key)
            samples = jax.random.normal(subkey, (5, 8))
            buffer.update_buffer(samples)

            # Invariant: buffer size <= capacity
            assert len(buffer.buffer) <= capacity

    @jax_required
    def test_buffer_deterministic_sampling(self, energy_rng_key):
        """Test that buffer sampling is deterministic with fixed RNG."""
        buffer = SampleBuffer(capacity=20, reinit_prob=0.5)

        # Add samples to buffer
        key = jax.random.key(42)
        samples = jax.random.normal(key, (16, 6))
        buffer.update_buffer(samples)

        # Sample multiple times with same RNG
        samples1 = buffer.sample_initial(
            batch_size=8,
            rng_key=energy_rng_key,
            sample_shape=(6,),
        )

        samples2 = buffer.sample_initial(
            batch_size=8,
            rng_key=energy_rng_key,
            sample_shape=(6,),
        )

        # Should be identical with same RNG
        assert jnp.allclose(samples1, samples2, atol=1e-8)

    @jax_required
    def test_buffer_shape_consistency(self, energy_rng_key):
        """Test that buffer maintains shape consistency."""
        buffer = SampleBuffer(capacity=15, reinit_prob=0.3)

        # Add samples of specific shape
        key = jax.random.key(42)
        original_shape = (8, 10)
        samples = jax.random.normal(key, original_shape)
        buffer.update_buffer(samples)

        # Sample from buffer
        sampled = buffer.sample_initial(
            batch_size=4,
            rng_key=energy_rng_key,
            sample_shape=(10,),
        )

        # Shape should be consistent
        assert sampled.shape == (4, 10)

    @jax_required
    def test_buffer_reinit_probability_effect(self, energy_rng_key):
        """Test that reinit probability affects sampling behavior."""
        # Buffer with no reinitialization
        buffer_no_reinit = SampleBuffer(capacity=20, reinit_prob=0.0)
        # Buffer with always reinitialization
        buffer_always_reinit = SampleBuffer(capacity=20, reinit_prob=1.0)

        # Add specific samples to both buffers
        specific_samples = jnp.ones((8, 6))  # All ones

        buffer_no_reinit.update_buffer(specific_samples)
        buffer_always_reinit.update_buffer(specific_samples)

        # Sample from both
        samples_no_reinit = buffer_no_reinit.sample_initial(
            batch_size=4,
            rng_key=energy_rng_key,
            sample_shape=(6,),
        )

        samples_always_reinit = buffer_always_reinit.sample_initial(
            batch_size=4,
            rng_key=energy_rng_key,
            sample_shape=(6,),
        )

        # No-reinit should return stored samples (close to ones)
        # Always-reinit should return fresh random samples (not all ones)

        # Check that no-reinit samples are closer to ones
        distance_no_reinit = jnp.mean(jnp.abs(samples_no_reinit - 1.0))
        distance_always_reinit = jnp.mean(jnp.abs(samples_always_reinit - 1.0))

        # No-reinit should be much closer to the stored ones
        assert distance_no_reinit < distance_always_reinit


class TestNumericalStabilityProperties:
    """Test numerical stability properties."""

    @jax_required
    def test_energy_numerical_stability_extreme_inputs(self, energy_rngs):
        """Test energy function stability with extreme inputs."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=8,
            rngs=energy_rngs,
        )

        # Test with various extreme inputs
        extreme_inputs = [
            jnp.zeros((4, 8)),  # All zeros
            jnp.ones((4, 8)) * 10.0,  # Large positive values
            jnp.ones((4, 8)) * (-10.0),  # Large negative values
            jnp.array([[1e-8] * 8] * 4),  # Very small values
        ]

        for extreme_input in extreme_inputs:
            energies = energy_fn(extreme_input)

            # Should produce finite energies
            assert jnp.all(jnp.isfinite(energies))
            assert energies.shape == (4,)

    @jax_required
    def test_gradient_numerical_stability(self, energy_rngs):
        """Test numerical stability of gradient computations."""
        energy_fn = MLPEnergyFunction(
            hidden_dims=[16, 8],
            input_dim=6,
            rngs=energy_rngs,
        )

        # Create test input with potential numerical challenges
        key = jax.random.key(42)
        test_input = jax.random.normal(key, (4, 6)) * 5.0  # Larger scale

        # Compute second-order gradients (Hessian diagonal)
        def energy_sum(x):
            return jnp.sum(energy_fn(x))

        # First gradient
        grad_fn = jax.grad(energy_sum)
        first_grad = grad_fn(test_input)

        # Second gradient (diagonal of Hessian)
        hessian_diag_fn = jax.grad(lambda x: jnp.sum(grad_fn(x) * x))
        second_grad = hessian_diag_fn(test_input)

        # Both should be finite
        assert jnp.all(jnp.isfinite(first_grad))
        assert jnp.all(jnp.isfinite(second_grad))

    @jax_required
    def test_mcmc_numerical_stability_challenging_energy(self, energy_rng_key):
        """Test MCMC stability with challenging energy functions."""

        # Highly non-convex energy function
        def challenging_energy(x):
            # Multi-modal with sharp peaks
            return jnp.sum(jnp.sin(x * 5.0) ** 2 + 0.1 * x**2, axis=-1)

        # Start with random samples
        key = jax.random.key(42)
        initial_samples = jax.random.normal(key, (8, 4))

        # Run Langevin dynamics with small step size for stability
        final_samples = langevin_dynamics(
            energy_fn=challenging_energy,
            initial_samples=initial_samples,
            n_steps=20,
            step_size=0.01,  # Small step size
            noise_scale=0.02,
            grad_clip=0.1,  # Gradient clipping for stability
            rng_key=energy_rng_key,
        )

        # Should remain numerically stable
        assert jnp.all(jnp.isfinite(final_samples))
        assert final_samples.shape == initial_samples.shape
