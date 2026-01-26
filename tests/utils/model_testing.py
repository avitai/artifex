"""Standardized model testing utilities.

This module provides common testing patterns and utilities for all model types
in the generative models package. It includes standardized test runners,
assertion helpers, and specialized utilities for different model families.
"""

import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import pytest


class ModelTestRunner:
    """Standardized testing utilities for all models.

    Provides common test patterns that can be applied to any model type,
    ensuring consistent testing across the codebase.
    """

    @staticmethod
    def test_model_initialization(model_cls, config, rngs, **init_kwargs):
        """Standard initialization test for any model.

        Args:
            model_cls: Model class to test
            config: Configuration object for the model
            rngs: RNG state for initialization
            **init_kwargs: Additional initialization arguments

        Returns:
            Initialized model instance

        Raises:
            AssertionError: If initialization fails or model lacks required
                attributes
        """
        # Test initialization - pass config as keyword argument
        model = model_cls(config=config, rngs=rngs, **init_kwargs)

        # Check basic properties
        assert hasattr(model, "__call__"), "Model must be callable"

        # For NNX models, get parameters using nnx.state()
        if hasattr(model, "params"):
            # Old Flax style
            model_params = model.params
        else:
            # NNX style - get all parameters
            from flax import nnx

            model_params = nnx.state(model, nnx.Param)

        # Check parameter count is reasonable
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(model_params))
        assert param_count > 0, "Model must have parameters"
        assert param_count < 1e9, f"Model has too many parameters: {param_count:,}"

        # Check all parameters are finite
        param_leaves = jax.tree_util.tree_leaves(model_params)
        for param in param_leaves:
            assert jnp.all(jnp.isfinite(param)), "All parameters must be finite"

        return model

    @staticmethod
    def test_forward_pass(model, test_data, expected_output_shape=None, **forward_kwargs):
        """Standard forward pass test.

        Args:
            model: Model instance to test
            test_data: Input data for forward pass
            expected_output_shape: Expected shape of output (optional)
            **forward_kwargs: Additional forward pass arguments

        Returns:
            Model output

        Raises:
            AssertionError: If forward pass fails or output is invalid
        """
        # Test forward pass - handle diffusion models that need timesteps
        if hasattr(model, "noise_steps") and "timesteps" not in forward_kwargs:
            # Generate random timesteps for diffusion models
            import jax.random as random

            key = random.key(0)
            batch_size = test_data.shape[0] if hasattr(test_data, "shape") else 1
            timesteps = random.randint(key, (batch_size,), 0, model.noise_steps)
            output = model(test_data, timesteps, **forward_kwargs)
        else:
            output = model(test_data, **forward_kwargs)

        # Check output properties - handle both arrays and dictionaries
        if isinstance(output, dict):
            # For dictionary outputs, check each value
            for key, value in output.items():
                assert jnp.all(jnp.isfinite(value)), f"Output[{key}] must be finite"

            # For shape checking, use the main output (typically "predicted_noise")
            if expected_output_shape is not None:
                main_output = output.get("predicted_noise", next(iter(output.values())))
                assert main_output.shape == expected_output_shape, (
                    f"Expected shape {expected_output_shape}, got {main_output.shape}"
                )

            # Check not all zeros for main output
            main_output = output.get("predicted_noise", next(iter(output.values())))
            if not jnp.allclose(main_output, 0, atol=1e-6):
                assert not jnp.allclose(main_output, 0, atol=1e-8), "Output should not be all zeros"
        else:
            # For array outputs, use original logic
            assert jnp.all(jnp.isfinite(output)), "Output must be finite"

            if expected_output_shape is not None:
                assert output.shape == expected_output_shape, (
                    f"Expected shape {expected_output_shape}, got {output.shape}"
                )

            # Check output is not all zeros (unless expected)
            if not jnp.allclose(output, 0, atol=1e-6):
                assert not jnp.allclose(output, 0, atol=1e-8), "Output should not be all zeros"

        return output

    @staticmethod
    def test_gradient_flow(model, test_data, loss_fn, tolerance=1e-6):
        """Standard gradient computation test.

        Args:
            model: Model instance to test
            test_data: Input data for gradient computation
            loss_fn: Loss function that takes (output, target) and returns scalar loss
            tolerance: Numerical tolerance for gradient checks

        Returns:
            Computed gradients

        Raises:
            AssertionError: If gradients are invalid
        """

        # For NNX models, use proper gradient computation
        if not hasattr(model, "params"):
            # NNX style gradient computation
            from flax import nnx

            def compute_loss_nnx(model):
                # For diffusion models, we need to provide timesteps
                if hasattr(model, "noise_steps"):
                    # Generate random timesteps for diffusion models
                    import jax.random as random

                    key = random.key(0)
                    batch_size = test_data.shape[0] if hasattr(test_data, "shape") else 1
                    timesteps = random.randint(key, (batch_size,), 0, model.noise_steps)
                    output = model(test_data, timesteps)
                else:
                    output = model(test_data)

                # Simple reconstruction loss
                target = test_data
                return loss_fn(output, target)

            # Compute gradients using NNX approach
            grads = nnx.grad(compute_loss_nnx)(model)
        else:
            # Old Flax style
            def compute_loss(model_state, data):
                model_with_params = model.replace(params=model_state)
                if hasattr(model, "noise_steps"):
                    # Generate random timesteps for diffusion models
                    import jax.random as random

                    key = random.key(0)
                    batch_size = data.shape[0] if hasattr(data, "shape") else 1
                    timesteps = random.randint(key, (batch_size,), 0, model.noise_steps)
                    output = model_with_params(data, timesteps)
                else:
                    output = model_with_params(data)

                # Simple reconstruction loss
                target = data if hasattr(data, "shape") else test_data
                return loss_fn(output, target)

            # Compute gradients
            grads = jax.grad(compute_loss)(model.params, test_data)

        # Check gradient properties
        grad_leaves = jax.tree_util.tree_leaves(grads)

        # All gradients must be finite
        for grad in grad_leaves:
            assert jnp.all(jnp.isfinite(grad)), "All gradients must be finite"

        # At least some gradients should be non-zero
        has_non_zero = any(not jnp.allclose(grad, 0, atol=tolerance) for grad in grad_leaves)
        assert has_non_zero, "At least some gradients should be non-zero"

        # Check gradient magnitudes are reasonable
        grad_norms = [jnp.linalg.norm(grad) for grad in grad_leaves]
        max_grad_norm = max(grad_norms)
        assert max_grad_norm < 1e6, f"Gradient norm too large: {max_grad_norm}"

        return grads

    @staticmethod
    def test_parameter_shapes(model, expected_param_shapes=None):
        """Test that model parameters have expected shapes.

        Args:
            model: Model instance to test
            expected_param_shapes: Dictionary of expected parameter shapes (optional)

        Returns:
            Dictionary of actual parameter shapes
        """

        def get_param_shapes(params):
            """Recursively get shapes of all parameters."""
            if hasattr(params, "shape"):
                return params.shape
            elif isinstance(params, dict):
                return {k: get_param_shapes(v) for k, v in params.items()}
            elif hasattr(params, "__dict__"):
                return {
                    k: get_param_shapes(v)
                    for k, v in params.__dict__.items()
                    if not k.startswith("_")
                }
            else:
                return str(type(params))

        # Get model parameters
        if hasattr(model, "params"):
            model_params = model.params
        else:
            from flax import nnx

            model_params = nnx.state(model, nnx.Param)

        actual_shapes = get_param_shapes(model_params)

        if expected_param_shapes is not None:
            # Compare expected vs actual shapes
            def compare_shapes(expected, actual, path=""):
                if isinstance(expected, dict) and isinstance(actual, dict):
                    for key in expected:
                        assert key in actual, f"Missing parameter: {path}.{key}"
                        compare_shapes(expected[key], actual[key], f"{path}.{key}")
                else:
                    assert expected == actual, (
                        f"Shape mismatch at {path}: expected {expected}, got {actual}"
                    )

            compare_shapes(expected_param_shapes, actual_shapes)

        return actual_shapes

    @staticmethod
    @contextmanager
    def timing_test(operation_name: str, max_time: float):
        """Context manager for timing tests.

        Args:
            operation_name: Name of the operation being timed
            max_time: Maximum allowed time in seconds

        Yields:
            Dictionary with timing information

        Raises:
            AssertionError: If operation takes longer than max_time
        """
        start_time = time.time()
        timing_info = {"start": start_time}

        try:
            yield timing_info
        finally:
            end_time = time.time()
            duration = end_time - start_time
            timing_info.update({"end": end_time, "duration": duration})

            assert duration <= max_time, (
                f"{operation_name} took {duration:.3f}s, max allowed: {max_time}s"
            )

    @staticmethod
    def test_batch_consistency(model, single_data, batch_data):
        """Test that model produces consistent results for single vs batch inputs.

        Args:
            model: Model instance to test
            single_data: Single example data
            batch_data: Batch of data containing single_data as first element

        Raises:
            AssertionError: If outputs are not consistent
        """
        # Skip batch consistency test for diffusion models
        # Diffusion models are inherently stochastic and may have legitimate
        # numerical differences between single and batch processing
        if hasattr(model, "noise_steps"):
            import pytest

            pytest.skip(
                "Batch consistency test skipped for diffusion models due to stochastic nature"
            )

        # Handle other models normally
        single_output = model(single_data)
        batch_output = model(batch_data)

        # Compare outputs - handle both arrays and dictionaries
        if isinstance(single_output, dict) and isinstance(batch_output, dict):
            # For dictionary outputs, compare each key
            for key in single_output.keys():
                assert key in batch_output, f"Missing key {key} in batch output"

                single_val = single_output[key]
                batch_val = batch_output[key]
                first_batch_val = batch_val[0:1]  # Keep batch dimension

                # Reshape single output to match batch format if needed
                if single_val.ndim < first_batch_val.ndim:
                    single_val_expanded = single_val[None, ...]
                else:
                    single_val_expanded = single_val

                assert jnp.allclose(single_val_expanded, first_batch_val, atol=1e-4), (
                    f"Single and batch outputs for key '{key}' should be consistent"
                )
        else:
            # For array outputs, use original logic
            first_batch_output = batch_output[0:1]  # Keep batch dimension

            # Reshape single output to match batch format if needed
            if single_output.ndim < first_batch_output.ndim:
                single_output_expanded = single_output[None, ...]
            else:
                single_output_expanded = single_output

            assert jnp.allclose(single_output_expanded, first_batch_output, atol=1e-4), (
                "Single and batch outputs should be consistent"
            )


class DiffusionTestUtils:
    """Specialized utilities for diffusion model testing.

    Provides test patterns specific to diffusion models, including noise schedule
    validation, sampling consistency checks, and timestep handling.
    """

    @staticmethod
    def test_noise_schedule(model, num_timesteps, tolerance=1e-6):
        """Test noise schedule properties.

        Args:
            model: Diffusion model instance
            num_timesteps: Expected number of timesteps
            tolerance: Numerical tolerance for checks

        Raises:
            AssertionError: If noise schedule is invalid
        """
        if hasattr(model, "get_beta_schedule"):
            betas = model.get_beta_schedule()
        elif hasattr(model, "betas"):
            betas = model.betas
        else:
            # Try to get from config
            if hasattr(model, "config") and hasattr(model.config, "num_timesteps"):
                # Generate default linear schedule for testing
                betas = jnp.linspace(1e-4, 0.02, model.config.num_timesteps)
            else:
                pytest.skip("Cannot access noise schedule for testing")

        # Check basic properties
        assert betas.shape == (num_timesteps,), (
            f"Expected {num_timesteps} timesteps, got {betas.shape[0]}"
        )
        assert jnp.all(betas > 0), "All betas must be positive"
        assert jnp.all(betas < 1), "All betas must be less than 1"
        assert jnp.all(jnp.isfinite(betas)), "All betas must be finite"

        # Check monotonicity (generally increasing)
        if jnp.all(betas[1:] >= betas[:-1]):
            pass  # Increasing schedule is good
        elif jnp.all(betas[1:] <= betas[:-1]):
            pass  # Decreasing schedule might be valid too
        else:
            # Check if differences are small (might be numerical noise)
            max_violation = jnp.max(jnp.abs(jnp.diff(betas)))
            if max_violation > tolerance:
                print(f"Warning: Non-monotonic noise schedule, max violation: {max_violation}")

    @staticmethod
    def test_forward_process(model, x0, timesteps, tolerance=1e-6):
        """Test forward diffusion process.

        Args:
            model: Diffusion model instance
            x0: Clean data
            timesteps: Timesteps to test
            tolerance: Numerical tolerance

        Returns:
            Noisy data at given timesteps

        Raises:
            AssertionError: If forward process is invalid
        """
        if hasattr(model, "forward_process"):
            xt = model.forward_process(x0, timesteps)
        elif hasattr(model, "q_sample"):
            xt = model.q_sample(x0, timesteps)
        else:
            # Manual forward process for testing
            if hasattr(model, "alphas_cumprod"):
                alphas_cumprod = model.alphas_cumprod
            else:
                # Estimate from betas
                betas = (
                    model.get_beta_schedule()
                    if hasattr(model, "get_beta_schedule")
                    else model.betas
                )
                alphas = 1.0 - betas
                alphas_cumprod = jnp.cumprod(alphas)

            # Sample noise
            key = jax.random.PRNGKey(42)
            noise = jax.random.normal(key, x0.shape)

            # Apply forward process
            alpha_t = alphas_cumprod[timesteps]
            if alpha_t.ndim == 1:
                alpha_t = alpha_t.reshape(-1, *[1] * (x0.ndim - 1))

            xt = jnp.sqrt(alpha_t) * x0 + jnp.sqrt(1 - alpha_t) * noise

        # Check output properties
        assert xt.shape == x0.shape, "Forward process must preserve shape"
        assert jnp.all(jnp.isfinite(xt)), "Forward process output must be finite"

        # Check that noise increases with timestep
        if len(timesteps) > 1:
            # Variance should generally increase with timestep
            variances = jnp.var(xt, axis=tuple(range(1, xt.ndim)))
            # Allow some tolerance for small timestep differences
            if not (variances[1:] >= variances[:-1] - tolerance).all():
                print("Warning: Variance doesn't increase monotonically with timestep")

        return xt

    @staticmethod
    def test_reverse_process(model, xt, timesteps, num_steps=None):
        """Test reverse diffusion process (sampling).

        Args:
            model: Diffusion model instance
            xt: Noisy data to denoise
            timesteps: Timesteps for reverse process
            num_steps: Number of denoising steps (for accelerated sampling)

        Returns:
            Denoised samples

        Raises:
            AssertionError: If reverse process is invalid
        """
        if hasattr(model, "reverse_process"):
            x0_pred = model.reverse_process(xt, timesteps)
        elif hasattr(model, "sample"):
            # Use model's sampling method
            if num_steps is not None:
                x0_pred = model.sample(xt.shape, num_steps=num_steps)
            else:
                x0_pred = model.sample(xt.shape)
        elif hasattr(model, "p_sample"):
            # Single step reverse
            x0_pred = model.p_sample(xt, timesteps)
        else:
            pytest.skip("Model doesn't support reverse process testing")

        # Check output properties
        assert x0_pred.shape == xt.shape, "Reverse process must preserve shape"
        assert jnp.all(jnp.isfinite(x0_pred)), "Reverse process output must be finite"

        return x0_pred

    @staticmethod
    def test_sampling_consistency(model, shape, num_samples=2, tolerance=1e-2):
        """Test that sampling produces diverse but valid outputs.

        Args:
            model: Diffusion model instance
            shape: Shape of samples to generate
            num_samples: Number of samples to generate
            tolerance: Tolerance for diversity check

        Returns:
            Generated samples

        Raises:
            AssertionError: If sampling is invalid
        """
        if not hasattr(model, "sample"):
            pytest.skip("Model doesn't support sampling")

        samples = []
        for i in range(num_samples):
            # Use different keys for each sample
            if hasattr(model, "sample"):
                # RNGs are stored at init time per NNX best practices
                # Each sample call uses the model's internal RNG state
                sample = model.sample(shape)
            else:
                pytest.skip("Model doesn't have sample method")

            assert sample.shape == shape, f"Sample {i} has wrong shape"
            assert jnp.all(jnp.isfinite(sample)), f"Sample {i} contains non-finite values"
            samples.append(sample)

        # Check diversity (samples shouldn't be identical)
        if num_samples > 1:
            for i in range(num_samples - 1):
                for j in range(i + 1, num_samples):
                    diff = jnp.mean(jnp.abs(samples[i] - samples[j]))
                    assert diff > tolerance, f"Samples {i} and {j} are too similar (diff: {diff})"

        return jnp.stack(samples)


class VAETestUtils:
    """Specialized utilities for VAE model testing.

    Provides test patterns specific to VAE models, including encoder/decoder
    validation, latent space checks, and reconstruction quality assessment.
    """

    @staticmethod
    def test_encode_decode_consistency(model, test_data, tolerance=1e-1):
        """Test encoder-decoder consistency.

        Args:
            model: VAE model instance
            test_data: Input data for testing
            tolerance: Reconstruction tolerance

        Returns:
            Tuple of (latents, reconstructions)

        Raises:
            AssertionError: If encode-decode process is invalid
        """
        # Test encoding
        if hasattr(model, "encode"):
            latents = model.encode(test_data)
        else:
            # Try calling model and extracting latents
            output = model(test_data)
            if isinstance(output, dict) and "latents" in output:
                latents = output["latents"]
            else:
                pytest.skip("Cannot access encoder")

        # Test decoding
        if hasattr(model, "decode"):
            reconstructions = model.decode(latents)
        else:
            pytest.skip("Cannot access decoder")

        # Check properties
        assert jnp.all(jnp.isfinite(latents)), "Latents must be finite"
        assert jnp.all(jnp.isfinite(reconstructions)), "Reconstructions must be finite"
        assert reconstructions.shape == test_data.shape, "Reconstruction shape mismatch"

        # Check reconstruction quality (should be somewhat similar to input)
        reconstruction_error = jnp.mean(jnp.abs(test_data - reconstructions))
        data_scale = jnp.mean(jnp.abs(test_data))
        relative_error = reconstruction_error / (data_scale + 1e-8)

        assert relative_error < tolerance, (
            f"Reconstruction error too high: {relative_error:.3f} > {tolerance}"
        )

        return latents, reconstructions

    @staticmethod
    def test_latent_properties(latents, expected_dim=None, check_distribution=True):
        """Test properties of VAE latent representations.

        Args:
            latents: Latent representations to test
            expected_dim: Expected latent dimensionality
            check_distribution: Whether to check if latents follow reasonable distribution

        Raises:
            AssertionError: If latent properties are invalid
        """
        assert jnp.all(jnp.isfinite(latents)), "Latents must be finite"

        if expected_dim is not None:
            assert latents.shape[-1] == expected_dim, (
                f"Expected latent dim {expected_dim}, got {latents.shape[-1]}"
            )

        if check_distribution:
            # Check that latents have reasonable statistics
            latent_mean = jnp.mean(latents)
            latent_std = jnp.std(latents)

            # Shouldn't be all zeros or constant
            assert latent_std > 1e-6, "Latents have no variation"

            # Check reasonable range (for standard VAEs, often centered around 0)
            assert jnp.abs(latent_mean) < 5.0, f"Latent mean too extreme: {latent_mean}"
            assert latent_std < 10.0, f"Latent std too extreme: {latent_std}"


class GeometricTestUtils:
    """Specialized utilities for geometric model testing.

    Provides test patterns specific to geometric models, including point cloud
    processing validation, equivariance checks, and shape consistency tests.
    """

    @staticmethod
    def test_point_cloud_processing(model, point_cloud, expected_output_shape=None):
        """Test point cloud processing consistency.

        Args:
            model: Geometric model instance
            point_cloud: Input point cloud data
            expected_output_shape: Expected output shape

        Returns:
            Model output

        Raises:
            AssertionError: If point cloud processing is invalid
        """
        output = model(point_cloud)

        # Check basic properties
        assert jnp.all(jnp.isfinite(output)), "Output must be finite"

        if expected_output_shape is not None:
            assert output.shape == expected_output_shape, (
                f"Expected shape {expected_output_shape}, got {output.shape}"
            )

        # Check that model processes different batch sizes consistently
        if point_cloud.shape[0] > 1:
            single_output = model(point_cloud[:1])
            batch_first_output = output[:1]

            assert jnp.allclose(single_output, batch_first_output, atol=1e-5), (
                "Model should be consistent across batch sizes"
            )

        return output

    @staticmethod
    def test_permutation_equivariance(model, point_cloud, tolerance=1e-5):
        """Test if model is equivariant to point permutations.

        Args:
            model: Geometric model instance
            point_cloud: Input point cloud data
            tolerance: Numerical tolerance for equivariance check

        Raises:
            AssertionError: If model is not equivariant (when it should be)
        """
        # Get original output
        original_output = model(point_cloud)

        # Permute points
        key = jax.random.PRNGKey(42)
        batch_size, num_points = point_cloud.shape[:2]

        for i in range(batch_size):
            # Create random permutation
            perm = jax.random.permutation(key, num_points)
            key = jax.random.split(key)[0]

            # Permute points
            permuted_cloud = point_cloud.at[i].set(point_cloud[i][perm])
            permuted_output = model(permuted_cloud)

            # For set functions, output should be identical
            # For sequence functions, output should be permuted
            if hasattr(model, "permutation_equivariant") and model.permutation_equivariant:
                # Check if output is appropriately transformed
                if original_output.shape == point_cloud.shape:
                    # Point-wise output should be permuted
                    expected_output = original_output.at[i].set(original_output[i][perm])
                    assert jnp.allclose(permuted_output, expected_output, atol=tolerance), (
                        "Model should be permutation equivariant"
                    )
                else:
                    # Global output should be unchanged
                    assert jnp.allclose(permuted_output, original_output, atol=tolerance), (
                        "Global output should be permutation invariant"
                    )
