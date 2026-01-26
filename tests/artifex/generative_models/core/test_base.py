"""Tests for base neural network abstractions using Flax NNX."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.base import (
    CNN,
    GenerativeModel,
    GenerativeModule,
    MLP,
)


@pytest.fixture
def rngs():
    """Fixture for nnx random number generator."""
    return nnx.Rngs(0)


class ConcreteGenerativeModule(GenerativeModule):
    """Concrete implementation of GenerativeModule for testing."""

    def __init__(
        self,
        hidden_size: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            rngs=rngs,
        )
        self.hidden_size = hidden_size
        self.dense = nnx.Linear(
            in_features=10,
            out_features=hidden_size,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        activation = self._get_default_activation()
        return activation(self.dense(x))


class ConcreteGenerativeModel(GenerativeModel):
    """Concrete implementation of GenerativeModel for testing."""

    def __init__(
        self,
        hidden_size: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            rngs=rngs,
        )
        self.hidden_size = hidden_size
        self.encoder = nnx.Linear(
            in_features=10,
            out_features=hidden_size,
            rngs=rngs,
        )
        self.decoder = nnx.Linear(
            in_features=hidden_size,
            out_features=10,
            rngs=rngs,
        )

    def __call__(
        self, x: jax.Array, *args, rngs: nnx.Rngs | None = None, training: bool = False, **kwargs
    ) -> dict[str, Any]:
        output = self.decoder(nnx.relu(self.encoder(x)))
        return {"output": output}

    def generate(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        """Generate samples from random noise."""
        # Generate random noise
        if rngs is not None:
            # Try to get a sample key, fallback to default if not available
            try:
                key = rngs.sample()
            except (AttributeError, KeyError):
                key = rngs.default()
        else:
            key = jax.random.key(0)
        z = jax.random.normal(key, (n_samples, self.hidden_size))
        return self.decoder(z)

    def loss_fn(
        self,
        batch: tuple[jax.Array, jax.Array],
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute mean squared error loss."""
        x, y = batch
        # Get model outputs or compute them
        if model_outputs is not None:
            output = model_outputs["output"]
        else:
            output = self(x)["output"]
        loss = jnp.mean((output - y) ** 2)
        return {"loss": loss, "mse": loss}


class TestGenerativeModule:
    """Test cases for GenerativeModule."""

    def test_initialization(self, rngs):
        """Test initialization of GenerativeModule."""
        module = ConcreteGenerativeModule(rngs=rngs)
        assert module is not None

    def test_default_activation(self, rngs):
        """Test default activation function."""
        module = ConcreteGenerativeModule(rngs=rngs)
        activation = module._get_default_activation()

        # Test activation function on sample input
        x = jnp.array([0.0, 1.0, -1.0])
        output = activation(x)

        # GELU should map 0 -> 0, positive -> positive, negative -> small negative
        assert jnp.isclose(output[0], 0.0, atol=1e-6)
        assert output[1] > 0.0
        assert output[2] < 0.0
        # GELU attenuates negatives
        assert jnp.abs(output[2]) < jnp.abs(x[2])

    def test_call_implementation(self, rngs):
        """Test call implementation in concrete subclass."""
        module = ConcreteGenerativeModule(rngs=rngs)
        x = jnp.ones((2, 10))

        # Should not raise NotImplementedError
        output = module(x)

        # Check output shape
        assert output.shape == (2, module.hidden_size)


class TestGenerativeModel:
    """Test cases for GenerativeModel."""

    def test_initialization(self, rngs):
        """Test initialization of GenerativeModel."""
        model = ConcreteGenerativeModel(rngs=rngs)
        assert model is not None

    def test_generate_method(self, rngs):
        """Test generate method implementation."""
        model = ConcreteGenerativeModel(rngs=rngs)

        # Generate samples
        samples = model.generate(n_samples=5, rngs=rngs)

        # Check output shape
        assert samples.shape == (5, 10)

        # Generate different samples with different RNG
        new_rngs = nnx.Rngs(1)
        new_samples = model.generate(n_samples=5, rngs=new_rngs)

        # Check that samples are different (with high probability)
        assert not jnp.allclose(samples, new_samples, atol=1e-5)

    def test_loss_fn_method(self, rngs):
        """Test loss_fn method implementation."""
        model = ConcreteGenerativeModel(rngs=rngs)

        # Create dummy batch
        x = jnp.ones((2, 10))
        y = jnp.ones((2, 10))
        batch = (x, y)

        # Generate model outputs
        outputs = model(x)

        # Compute loss with model outputs
        loss_dict = model.loss_fn(batch, outputs)

        # Check loss is a scalar
        assert loss_dict["loss"].shape == ()
        assert jnp.isfinite(loss_dict["loss"])

        # Verify loss value is in dictionary
        assert "loss" in loss_dict
        assert jnp.isfinite(loss_dict["loss"])
        assert "mse" in loss_dict
        assert jnp.isclose(loss_dict["loss"], loss_dict["mse"])

        # Test with different batch inputs
        y2 = jnp.zeros((2, 10))  # Different target
        batch2 = (x, y2)
        loss_dict2 = model.loss_fn(batch2, outputs)

        # Check that loss is different
        assert not jnp.isclose(loss_dict["loss"], loss_dict2["loss"])


class TestMLP:
    """Test cases for MLP."""

    def test_initialization(self, rngs):
        """Test MLP initialization."""
        mlp = MLP(
            hidden_dims=[32, 64, 16],
            in_features=10,
            activation="gelu",
            dropout_rate=0.1,
            rngs=rngs,
        )

        assert len(mlp.layers) == 3
        assert len(mlp.dropouts) > 0  # Check dropouts list is not empty
        assert mlp.hidden_dims.value == [32, 64, 16]

    def test_forward_pass(self, rngs):
        """Test MLP forward pass."""
        mlp = MLP(
            hidden_dims=[32, 16],
            in_features=10,
            activation="relu",
            rngs=rngs,
        )

        x = jnp.ones((4, 10))
        # Set model to eval mode for deterministic behavior
        mlp.eval()
        output = mlp(x)

        assert output.shape == (4, 16)  # Last dimension from hidden_dims
        assert jnp.isfinite(output).all()

    def test_dropout_behavior(self, rngs):
        """Test dropout behavior in training vs eval."""
        mlp = MLP(
            hidden_dims=[32],
            in_features=10,
            dropout_rate=0.5,
            rngs=rngs,
        )

        x = jnp.ones((4, 10))

        # eval mode (deterministic) should give same output
        mlp.eval()
        output1 = mlp(x)
        output2 = mlp(x)
        assert jnp.allclose(output1, output2)

        # train mode (non-deterministic) should give different outputs (with high probability)
        # Note: NNX Dropout modules manage their own RNG state and should advance it
        mlp.train()
        output3 = mlp(x)
        output4 = mlp(x)

        # With 50% dropout rate and reasonable tensor size, outputs should be different
        # Allow for the small possibility they could be the same by checking variance
        outputs_are_different = not jnp.allclose(output3, output4, atol=1e-5)

        # Alternative check: ensure dropout is actually affecting the output
        mlp.eval()
        deterministic_output = mlp(x)
        mlp.train()
        non_deterministic_output = mlp(x)

        # At least one of these should be true:
        # 1. Two non-deterministic calls differ
        # 2. Non-deterministic differs from deterministic
        dropout_is_working = outputs_are_different or not jnp.allclose(
            deterministic_output, non_deterministic_output, atol=1e-5
        )

        assert dropout_is_working, (
            "Dropout doesn't appear to be working. "
            f"Det vs Non-det equal: {jnp.allclose(deterministic_output, non_deterministic_output, atol=1e-5)}, "
            f"Two non-det equal: {jnp.allclose(output3, output4, atol=1e-5)}"
        )

    def test_dropout_layer_directly(self, rngs):
        """Test that NNX Dropout layer works correctly by itself."""
        dropout = nnx.Dropout(rate=0.5, rngs=rngs)
        x = jnp.ones((4, 10))

        # Deterministic mode should not modify input
        output_det1 = dropout(x, deterministic=True)
        output_det2 = dropout(x, deterministic=True)
        assert jnp.allclose(output_det1, output_det2)
        assert jnp.allclose(output_det1, x)  # Should be unchanged

        # Non-deterministic mode should introduce randomness
        output_nondet1 = dropout(x, deterministic=False)
        output_nondet2 = dropout(x, deterministic=False)

        # Should be different from each other and from original
        different_calls = not jnp.allclose(output_nondet1, output_nondet2, atol=1e-5)
        different_from_input = not jnp.allclose(output_nondet1, x, atol=1e-5)

        assert different_calls or different_from_input, (
            "Dropout layer not working: "
            f"Two calls equal: {jnp.allclose(output_nondet1, output_nondet2, atol=1e-5)}, "
            f"Same as input: {jnp.allclose(output_nondet1, x, atol=1e-5)}"
        )


class TestCNN:
    """Test cases for CNN."""

    def test_initialization(self, rngs):
        """Test CNN initialization."""
        cnn = CNN(
            hidden_dims=[32, 64],
            in_features=3,
            kernel_size=(3, 3),
            strides=(2, 2),
            rngs=rngs,
        )

        assert len(cnn.layers) == 2
        assert cnn.hidden_dims.value == [32, 64]

    def test_forward_pass(self, rngs):
        """Test CNN forward pass."""
        cnn = CNN(
            hidden_dims=[16, 32],
            in_features=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        # Input: [batch, height, width, channels]
        x = jnp.ones((2, 32, 32, 3))
        # Set model to eval mode for deterministic behavior
        cnn.eval()
        output = cnn(x)

        # Output should have 32 channels (last in hidden_dims)
        assert output.shape[0] == 2  # batch size
        assert output.shape[-1] == 32  # output channels
        assert jnp.isfinite(output).all()

    def test_transpose_conv(self, rngs):
        """Test CNN with transpose convolution."""
        cnn = CNN(
            hidden_dims=[16],
            in_features=32,
            use_transpose=True,
            kernel_size=(3, 3),
            strides=(2, 2),
            rngs=rngs,
        )

        x = jnp.ones((2, 16, 16, 32))
        # Set model to eval mode for deterministic behavior
        cnn.eval()
        output = cnn(x)

        # With stride 2 transpose conv, spatial dims should increase
        assert output.shape[1] > x.shape[1]  # height increased
        assert output.shape[2] > x.shape[2]  # width increased
        assert output.shape[-1] == 16  # output channels from hidden_dims


class TestOptimizations:
    """Test class for new optimization features."""

    def test_mixed_precision_support(self, rngs):
        """Test mixed precision computation support."""
        # Test basic module functionality (mixed precision will be added later)
        module = ConcreteGenerativeModule(
            hidden_size=32,
            rngs=rngs,
        )

        x = jnp.ones((4, 10), dtype=jnp.float32)
        output = module(x)

        # Check that output is computed correctly
        assert output.shape == (4, 32)
        assert jnp.isfinite(output).all()

    def test_activation_function_caching(self, rngs):
        """Test that activation functions are properly cached."""
        from artifex.generative_models.core.base import get_activation_function

        # Clear cache to start fresh
        get_activation_function.cache_clear()

        # First call should cache the function
        activation1 = get_activation_function("relu")
        cache_info1 = get_activation_function.cache_info()

        # Second call with same activation should hit cache
        activation2 = get_activation_function("relu")
        cache_info2 = get_activation_function.cache_info()

        # Verify caching behavior
        assert activation1 is activation2  # Same function object
        assert cache_info2.hits > cache_info1.hits  # Cache hit occurred

    def test_mlp_scan_forward_pass(self, rngs):
        """Test MLP scan-based forward pass for memory efficiency."""
        # Create deep MLP to trigger scan usage
        mlp = MLP(
            hidden_dims=[64, 64, 64, 64, 64, 64, 64, 64, 64],  # 9 layers
            in_features=32,
            rngs=rngs,
        )

        x = jnp.ones((4, 32))

        # Test regular forward pass
        output_regular = mlp(x, use_scan=False)

        # Test scan-based forward pass
        output_scan = mlp(x, use_scan=True)

        # Outputs should be very similar (allowing for numerical differences)
        assert output_regular.shape == output_scan.shape
        assert jnp.allclose(output_regular, output_scan, atol=1e-5)

    def test_mlp_gradient_checkpointing(self, rngs):
        """Test MLP gradient checkpointing functionality."""
        mlp = MLP(
            hidden_dims=[64, 64, 64],
            in_features=32,
            use_gradient_checkpointing=True,
            rngs=rngs,
        )

        x = jnp.ones((4, 32))
        # Checkpointing is now controlled by init flag, not __call__ kwarg
        output = mlp(x)

        # Check that output is computed correctly with checkpointing
        assert output.shape == (4, 64)
        assert jnp.isfinite(output).all()

    def test_mlp_return_intermediates(self, rngs):
        """Test MLP intermediate activation return functionality."""
        mlp = MLP(
            hidden_dims=[64, 32, 16],
            in_features=32,
            rngs=rngs,
        )

        x = jnp.ones((4, 32))

        # Test with return_intermediates=True
        output, intermediates = mlp(x, return_intermediates=True)

        # Check output and intermediates
        assert output.shape == (4, 16)
        assert len(intermediates) == 3  # One for each layer
        assert intermediates[0].shape == (4, 64)  # First hidden layer
        assert intermediates[1].shape == (4, 32)  # Second hidden layer
        assert intermediates[2].shape == (4, 16)  # Final layer

    def test_cnn_depthwise_separable(self, rngs):
        """Test CNN with depthwise separable convolutions."""
        cnn = CNN(
            hidden_dims=[16, 32],
            in_features=3,
            use_depthwise_separable=True,
            rngs=rngs,
        )

        x = jnp.ones((2, 32, 32, 3))
        output = cnn(x)

        # Check output shape
        assert output.shape[0] == 2
        assert output.shape[-1] == 32
        assert len(output.shape) == 4

    def test_cnn_grouped_convolution(self, rngs):
        """Test CNN with grouped convolutions."""
        cnn = CNN(
            hidden_dims=[16, 32],
            in_features=8,  # Use 8 channels for proper grouping
            groups=2,
            rngs=rngs,
        )

        x = jnp.ones((2, 32, 32, 8))
        output = cnn(x)

        # Check output shape
        assert output.shape[0] == 2
        assert output.shape[-1] == 32
        assert len(output.shape) == 4

    def test_mlp_gradient_checkpointing_with_policy(self, rngs):
        """Test MLP gradient checkpointing with a named policy."""
        mlp = MLP(
            hidden_dims=[64, 64, 64],
            in_features=32,
            use_gradient_checkpointing=True,
            checkpoint_policy="dots_saveable",
            rngs=rngs,
        )

        x = jnp.ones((4, 32))
        output = mlp(x)

        assert output.shape == (4, 64)
        assert jnp.isfinite(output).all()

    def test_mlp_gradient_checkpointing_produces_valid_gradients(self, rngs):
        """Gradients with checkpointing match gradients without within tolerance."""
        mlp_no_ckpt = MLP(
            hidden_dims=[64, 64],
            in_features=32,
            use_gradient_checkpointing=False,
            rngs=rngs,
        )
        mlp_ckpt = MLP(
            hidden_dims=[64, 64],
            in_features=32,
            use_gradient_checkpointing=True,
            rngs=nnx.Rngs(0),  # Same seed for identical weights
        )

        x = jnp.ones((4, 32))

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        grads_no_ckpt = nnx.grad(loss_fn)(mlp_no_ckpt, x)
        grads_ckpt = nnx.grad(loss_fn)(mlp_ckpt, x)

        # Compare gradients on the first layer kernel
        g1 = grads_no_ckpt.layers[0].kernel.value
        g2 = grads_ckpt.layers[0].kernel.value
        assert jnp.allclose(g1, g2, atol=1e-5)

    def test_variable_types_usage(self, rngs):
        """Test that proper Variable types are used for state management."""
        mlp = MLP(
            hidden_dims=[64, 32],
            in_features=16,
            rngs=rngs,
        )

        # Check that static config uses Cache
        assert isinstance(mlp.hidden_dims, nnx.Cache)
        assert isinstance(mlp.in_features, nnx.Cache)
        assert isinstance(mlp.use_batch_norm, nnx.Cache)
        assert isinstance(mlp.use_gradient_checkpointing, bool)

        # Check that mutable config uses Variable
        assert isinstance(mlp.dropout_rate, nnx.Variable)

    def test_jit_compilation(self, rngs):
        """Test that methods are JIT-compatible."""
        mlp = MLP(
            hidden_dims=[64, 32],
            in_features=16,
            rngs=rngs,
        )

        x = jnp.ones((4, 16))

        # Test that the method can be JIT-compiled externally
        jitted_forward = jax.jit(mlp.__call__)
        output = jitted_forward(x)
        assert output.shape == (4, 32)

        # Test with static arguments
        jitted_with_intermediates = jax.jit(mlp.__call__, static_argnames=["return_intermediates"])
        output_with_intermediates, intermediates = jitted_with_intermediates(
            x, return_intermediates=True
        )
        assert output_with_intermediates.shape == (4, 32)
        assert len(intermediates) == 2

    def test_precision_parameter(self, rngs):
        """Test precision parameter functionality."""
        module = ConcreteGenerativeModule(
            hidden_size=32,
            rngs=rngs,
        )

        # Check that precision is stored as Variable
        assert module.precision is None or isinstance(module.precision, nnx.Variable)

    def test_mlp_parameter_counting(self, rngs):
        """Test MLP parameter counting functionality."""
        mlp = MLP(
            hidden_dims=[64, 32, 16],
            in_features=128,
            use_batch_norm=True,
            rngs=rngs,
        )

        param_count = mlp.get_num_params()

        # Should count weights, biases, and batch norm parameters
        assert param_count > 0
        assert isinstance(param_count, int)

        # Rough calculation: (128*64 + 64) + (64*32 + 32) + (32*16 + 16) + BN params
        expected_min = (128 * 64 + 64) + (64 * 32 + 32) + (32 * 16 + 16)
        assert param_count >= expected_min


class TestJITCompatibility:
    """Comprehensive JIT compatibility tests for all base modules."""

    def test_mlp_jit_forward_pass(self, rngs):
        """Test that MLP forward pass can be JIT compiled."""
        mlp = MLP(
            hidden_dims=[64, 32],
            in_features=16,
            rngs=rngs,
        )

        x = jnp.ones((4, 16))

        # Test basic JIT compilation
        @jax.jit
        def forward(model, x):
            return model(x)

        output = forward(mlp, x)
        assert output.shape == (4, 32)
        assert jnp.isfinite(output).all()

    def test_mlp_jit_with_different_batch_sizes(self, rngs):
        """Test MLP JIT compilation with different batch sizes (triggers recompilation)."""
        mlp = MLP(
            hidden_dims=[64, 32],
            in_features=16,
            rngs=rngs,
        )

        @jax.jit
        def forward(model, x):
            return model(x)

        # First batch size
        x1 = jnp.ones((4, 16))
        output1 = forward(mlp, x1)
        assert output1.shape == (4, 32)

        # Different batch size (triggers recompilation)
        x2 = jnp.ones((8, 16))
        output2 = forward(mlp, x2)
        assert output2.shape == (8, 32)

    def test_mlp_jit_gradient_computation(self, rngs):
        """Test that MLP gradient computation can be JIT compiled."""
        mlp = MLP(
            hidden_dims=[64, 32],
            in_features=16,
            rngs=rngs,
        )

        x = jnp.ones((4, 16))

        @jax.jit
        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        # Compute gradients using nnx.grad
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(mlp, x)

        # Check that gradients were computed
        assert grads is not None

    def test_mlp_jit_with_scan(self, rngs):
        """Test that MLP scan-based forward pass is JIT compatible."""
        mlp = MLP(
            hidden_dims=[64, 64, 64, 64, 64, 64, 64, 64],  # 8 layers
            in_features=32,
            rngs=rngs,
        )

        x = jnp.ones((4, 32))

        @jax.jit
        def forward_with_scan(model, x):
            return model(x, use_scan=True)

        output = forward_with_scan(mlp, x)
        assert output.shape == (4, 64)
        assert jnp.isfinite(output).all()

    def test_mlp_jit_with_checkpointing(self, rngs):
        """Test that MLP with gradient checkpointing is JIT compatible."""
        mlp = MLP(
            hidden_dims=[64, 64, 64],
            in_features=32,
            use_gradient_checkpointing=True,
            rngs=rngs,
        )

        x = jnp.ones((4, 32))

        @jax.jit
        def forward_with_checkpoint(model, x):
            return model(x)

        output = forward_with_checkpoint(mlp, x)
        assert output.shape == (4, 64)
        assert jnp.isfinite(output).all()

    def test_mlp_jit_with_dropout(self, rngs):
        """Test that MLP with dropout is JIT compatible."""
        mlp = MLP(
            hidden_dims=[64, 32],
            in_features=16,
            dropout_rate=0.5,
            rngs=rngs,
        )

        x = jnp.ones((4, 16))

        @jax.jit
        def forward(model, x):
            return model(x)

        # Test in eval mode (deterministic)
        mlp.eval()
        output1 = forward(mlp, x)
        output2 = forward(mlp, x)
        assert jnp.allclose(output1, output2)

        # Test in train mode (stochastic)
        mlp.train()
        output3 = forward(mlp, x)
        assert output3.shape == (4, 32)

    def test_mlp_jit_with_batch_norm(self, rngs):
        """Test that MLP with batch normalization is JIT compatible."""
        mlp = MLP(
            hidden_dims=[64, 32],
            in_features=16,
            use_batch_norm=True,
            rngs=rngs,
        )

        x = jnp.ones((4, 16))

        @jax.jit
        def forward(model, x):
            return model(x)

        # Test in eval mode
        mlp.eval()
        output = forward(mlp, x)
        assert output.shape == (4, 32)
        assert jnp.isfinite(output).all()

    def test_mlp_jit_with_batch_norm_and_dropout(self, rngs):
        """Test that MLP with both batch norm and dropout is JIT compatible."""
        mlp = MLP(
            hidden_dims=[64, 32],
            in_features=16,
            use_batch_norm=True,
            dropout_rate=0.3,
            rngs=rngs,
        )

        x = jnp.ones((4, 16))

        @jax.jit
        def forward(model, x):
            return model(x)

        # Test in eval mode (deterministic)
        mlp.eval()
        output1 = forward(mlp, x)
        output2 = forward(mlp, x)
        assert jnp.allclose(output1, output2)
        assert output1.shape == (4, 32)
        assert jnp.isfinite(output1).all()

        # Test in train mode
        mlp.train()
        output3 = forward(mlp, x)
        assert output3.shape == (4, 32)

    def test_cnn_jit_forward_pass(self, rngs):
        """Test that CNN forward pass can be JIT compiled."""
        cnn = CNN(
            hidden_dims=[16, 32],
            in_features=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        x = jnp.ones((2, 32, 32, 3))

        @jax.jit
        def forward(model, x):
            return model(x)

        cnn.eval()
        output = forward(cnn, x)
        assert output.shape[0] == 2
        assert output.shape[-1] == 32
        assert jnp.isfinite(output).all()

    def test_cnn_jit_with_different_batch_sizes(self, rngs):
        """Test CNN JIT compilation with different batch sizes."""
        cnn = CNN(
            hidden_dims=[16, 32],
            in_features=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        @jax.jit
        def forward(model, x):
            return model(x)

        cnn.eval()

        # First batch size
        x1 = jnp.ones((2, 32, 32, 3))
        output1 = forward(cnn, x1)
        assert output1.shape[0] == 2

        # Different batch size
        x2 = jnp.ones((4, 32, 32, 3))
        output2 = forward(cnn, x2)
        assert output2.shape[0] == 4

    def test_cnn_jit_gradient_computation(self, rngs):
        """Test that CNN gradient computation can be JIT compiled."""
        cnn = CNN(
            hidden_dims=[16, 32],
            in_features=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        x = jnp.ones((2, 32, 32, 3))

        @jax.jit
        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        # Compute gradients
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(cnn, x)

        # Check that gradients were computed
        assert grads is not None

    def test_cnn_jit_with_transpose(self, rngs):
        """Test that CNN with transpose convolutions is JIT compatible."""
        cnn = CNN(
            hidden_dims=[16],
            in_features=32,
            use_transpose=True,
            kernel_size=(3, 3),
            strides=(2, 2),
            rngs=rngs,
        )

        x = jnp.ones((2, 16, 16, 32))

        @jax.jit
        def forward(model, x):
            return model(x)

        cnn.eval()
        output = forward(cnn, x)
        assert output.shape[1] > x.shape[1]  # Spatial dims increased
        assert jnp.isfinite(output).all()

    def test_cnn_jit_with_depthwise_separable(self, rngs):
        """Test that CNN with depthwise separable convolutions is JIT compatible."""
        cnn = CNN(
            hidden_dims=[16, 32],
            in_features=3,
            use_depthwise_separable=True,
            rngs=rngs,
        )

        x = jnp.ones((2, 32, 32, 3))

        @jax.jit
        def forward(model, x):
            return model(x)

        output = forward(cnn, x)
        assert output.shape[0] == 2
        assert output.shape[-1] == 32
        assert jnp.isfinite(output).all()

    def test_cnn_jit_with_batch_norm(self, rngs):
        """Test that CNN with batch normalization is JIT compatible."""
        cnn = CNN(
            hidden_dims=[16, 32],
            in_features=3,
            use_batch_norm=True,
            rngs=rngs,
        )

        x = jnp.ones((2, 32, 32, 3))

        @jax.jit
        def forward(model, x):
            return model(x)

        cnn.eval()
        output = forward(cnn, x)
        assert output.shape[0] == 2
        assert output.shape[-1] == 32
        assert jnp.isfinite(output).all()

    def test_cnn_jit_with_dropout(self, rngs):
        """Test that CNN with dropout is JIT compatible."""
        cnn = CNN(
            hidden_dims=[16, 32],
            in_features=3,
            dropout_rate=0.3,
            rngs=rngs,
        )

        x = jnp.ones((2, 32, 32, 3))

        @jax.jit
        def forward(model, x):
            return model(x)

        # Test in eval mode (deterministic)
        cnn.eval()
        output1 = forward(cnn, x)
        output2 = forward(cnn, x)
        assert jnp.allclose(output1, output2)
        assert output1.shape[0] == 2
        assert output1.shape[-1] == 32

        # Test in train mode
        cnn.train()
        output3 = forward(cnn, x)
        assert output3.shape[0] == 2
        assert output3.shape[-1] == 32

    def test_cnn_jit_with_batch_norm_and_dropout(self, rngs):
        """Test that CNN with both batch norm and dropout is JIT compatible."""
        cnn = CNN(
            hidden_dims=[16, 32],
            in_features=3,
            use_batch_norm=True,
            dropout_rate=0.3,
            rngs=rngs,
        )

        x = jnp.ones((2, 32, 32, 3))

        @jax.jit
        def forward(model, x):
            return model(x)

        # Test in eval mode (deterministic)
        cnn.eval()
        output1 = forward(cnn, x)
        output2 = forward(cnn, x)
        assert jnp.allclose(output1, output2)
        assert output1.shape[0] == 2
        assert output1.shape[-1] == 32
        assert jnp.isfinite(output1).all()

        # Test in train mode
        cnn.train()
        output3 = forward(cnn, x)
        assert output3.shape[0] == 2
        assert output3.shape[-1] == 32

    def test_generative_module_jit_forward_pass(self, rngs):
        """Test that GenerativeModule forward pass can be JIT compiled."""
        module = ConcreteGenerativeModule(hidden_size=32, rngs=rngs)

        x = jnp.ones((4, 10))

        @jax.jit
        def forward(model, x):
            return model(x)

        output = forward(module, x)
        assert output.shape == (4, 32)
        assert jnp.isfinite(output).all()

    def test_generative_module_jit_with_different_batch_sizes(self, rngs):
        """Test GenerativeModule JIT with different batch sizes."""
        module = ConcreteGenerativeModule(hidden_size=32, rngs=rngs)

        @jax.jit
        def forward(model, x):
            return model(x)

        # Different batch sizes
        x1 = jnp.ones((4, 10))
        output1 = forward(module, x1)
        assert output1.shape == (4, 32)

        x2 = jnp.ones((8, 10))
        output2 = forward(module, x2)
        assert output2.shape == (8, 32)

    def test_generative_module_jit_gradient_computation(self, rngs):
        """Test that GenerativeModule gradient computation can be JIT compiled."""
        module = ConcreteGenerativeModule(hidden_size=32, rngs=rngs)

        x = jnp.ones((4, 10))

        @jax.jit
        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        # Compute gradients
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(module, x)

        # Check that gradients were computed
        assert grads is not None

    def test_generative_model_jit_forward_pass(self, rngs):
        """Test that GenerativeModel forward pass can be JIT compiled."""
        model = ConcreteGenerativeModel(hidden_size=32, rngs=rngs)

        x = jnp.ones((4, 10))

        @jax.jit
        def forward(model, x):
            return model(x, rngs=None)

        output = forward(model, x)
        assert "output" in output
        assert output["output"].shape == (4, 10)
        assert jnp.isfinite(output["output"]).all()

    def test_generative_model_jit_generate(self, rngs):
        """Test that GenerativeModel generate method can be JIT compiled."""
        model = ConcreteGenerativeModel(hidden_size=32, rngs=rngs)

        # n_samples must be static for JIT compilation (affects array shapes)
        @jax.jit
        def generate(model):
            return model.generate(n_samples=5, rngs=None)

        samples = generate(model)
        assert samples.shape == (5, 10)
        assert jnp.isfinite(samples).all()

    def test_generative_model_jit_loss_fn(self, rngs):
        """Test that GenerativeModel loss_fn can be JIT compiled."""
        model = ConcreteGenerativeModel(hidden_size=32, rngs=rngs)

        x = jnp.ones((4, 10))
        y = jnp.ones((4, 10))
        batch = (x, y)

        @jax.jit
        def compute_loss(model, batch):
            x, y = batch
            outputs = model(x, rngs=None)
            return model.loss_fn(batch, outputs, rngs=None)

        loss_dict = compute_loss(model, batch)
        assert "loss" in loss_dict
        assert jnp.isfinite(loss_dict["loss"])

    def test_generative_model_jit_with_different_batch_sizes(self, rngs):
        """Test GenerativeModel JIT with different batch sizes."""
        model = ConcreteGenerativeModel(hidden_size=32, rngs=rngs)

        @jax.jit
        def forward(model, x):
            return model(x, rngs=None)

        # Different batch sizes
        x1 = jnp.ones((4, 10))
        output1 = forward(model, x1)
        assert output1["output"].shape == (4, 10)

        x2 = jnp.ones((8, 10))
        output2 = forward(model, x2)
        assert output2["output"].shape == (8, 10)

    def test_generative_model_jit_gradient_computation(self, rngs):
        """Test that GenerativeModel gradient computation can be JIT compiled."""
        model = ConcreteGenerativeModel(hidden_size=32, rngs=rngs)

        x = jnp.ones((4, 10))
        y = jnp.ones((4, 10))
        batch = (x, y)

        @jax.jit
        def loss_fn(model, batch):
            x, y = batch
            outputs = model(x, rngs=None)
            loss_dict = model.loss_fn(batch, outputs, rngs=None)
            return loss_dict["loss"]

        # Compute gradients
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(model, batch)

        # Check that gradients were computed
        assert grads is not None

    def test_all_modules_jit_end_to_end(self, rngs):
        """Test end-to-end JIT compilation of a simple training step."""
        model = ConcreteGenerativeModel(hidden_size=32, rngs=rngs)

        x = jnp.ones((4, 10))
        y = jnp.ones((4, 10))
        batch = (x, y)

        @jax.jit
        def train_step(model, batch):
            x, y = batch

            # Forward pass
            outputs = model(x, rngs=None)

            # Compute loss
            loss_dict = model.loss_fn(batch, outputs, rngs=None)

            # Compute gradients
            def loss_fn(model):
                outputs = model(x, rngs=None)
                loss_dict = model.loss_fn(batch, outputs, rngs=None)
                return loss_dict["loss"]

            grads = nnx.grad(loss_fn)(model)

            return loss_dict, grads

        loss_dict, grads = train_step(model, batch)

        # Verify results
        assert "loss" in loss_dict
        assert jnp.isfinite(loss_dict["loss"])
        assert grads is not None
