# Model Parallelism

Model parallelism techniques for training large models that don't fit on a single device. Artifex supports tensor parallelism, pipeline parallelism, and hybrid strategies using JAX's sharding capabilities.

<div class="grid cards" markdown>

- :material-ab-testing:{ .lg .middle } **Tensor Parallelism**

    ---

    Split weight matrices across devices within a single layer

    [:octicons-arrow-right-24: Learn more](#tensor-parallelism)

- :material-pipeline:{ .lg .middle } **Pipeline Parallelism**

    ---

    Distribute model layers across devices in a pipeline

    [:octicons-arrow-right-24: Learn more](#pipeline-parallelism)

- :material-layers-triple:{ .lg .middle } **Hybrid Strategies**

    ---

    Combine multiple parallelism techniques for maximum scale

    [:octicons-arrow-right-24: Learn more](#hybrid-parallelism)

- :material-memory:{ .lg .middle } **Memory Optimization**

    ---

    Techniques to maximize model size on limited memory

    [:octicons-arrow-right-24: Learn more](#memory-optimization)

</div>

## Overview

Model parallelism becomes necessary when:

- **Model Too Large**: Parameters exceed single device memory
- **Activation Memory**: Forward/backward activations don't fit
- **Batch Size Constraints**: Can't reduce batch size further
- **Extreme Scale**: Training models with billions of parameters

### Parallelism Strategies Comparison

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Data Parallel** | Model fits on one device | Simple, efficient | Limited by model size |
| **Tensor Parallel** | Large layers, fast interconnect | Balances compute | Communication overhead |
| **Pipeline Parallel** | Many layers, slower interconnect | Minimal communication | Pipeline bubbles |
| **Hybrid** | Extreme scale | Maximum efficiency | Complex to implement |

## Tensor Parallelism

Tensor parallelism splits individual weight matrices across multiple devices.

### Megatron-Style Tensor Parallelism

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import nnx

class ColumnParallelLinear(nnx.Module):
    """Linear layer with column-parallel weight matrix.

    Splits weight matrix along output dimension:
        Y = X @ W  where W is split as [W1, W2]
        Y = [X @ W1, X @ W2]  (concatenate outputs)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        mesh: Mesh,
        gather_output: bool = True,
    ):
        super().__init__()
        self.gather_output = gather_output

        # Create weight sharded along output dimension
        self.weight = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(),
                (in_features, out_features)
            )
        )

        # Shard along columns (output dimension)
        weight_sharding = NamedSharding(mesh, P(None, "model"))
        self.weight.value = jax.device_put(self.weight.value, weight_sharding)

        # Bias sharded same way
        self.bias = nnx.Param(jnp.zeros(out_features))
        bias_sharding = NamedSharding(mesh, P("model"))
        self.bias.value = jax.device_put(self.bias.value, bias_sharding)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with column parallelism.

        Args:
            x: Input activations (batch, in_features)

        Returns:
            Output activations (batch, out_features)
        """
        # Input is replicated across model parallel devices
        # Output is sharded along out_features dimension

        # Matrix multiplication (automatically parallelized)
        output = x @ self.weight.value + self.bias.value

        if self.gather_output:
            # Gather output across model parallel devices
            output = jax.lax.all_gather(
                output,
                axis_name="model",
                axis=1,  # Gather along feature dimension
                tiled=True
            )

        return output


class RowParallelLinear(nnx.Module):
    """Linear layer with row-parallel weight matrix.

    Splits weight matrix along input dimension:
        Y = X @ W  where W is split as [W1; W2]
        Y = X1 @ W1 + X2 @ W2  (sum partial results)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        mesh: Mesh,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.input_is_parallel = input_is_parallel

        # Create weight sharded along input dimension
        self.weight = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(),
                (in_features, out_features)
            )
        )

        # Shard along rows (input dimension)
        weight_sharding = NamedSharding(mesh, P("model", None))
        self.weight.value = jax.device_put(self.weight.value, weight_sharding)

        # Bias replicated (only added once after reduce)
        self.bias = nnx.Param(jnp.zeros(out_features))
        bias_sharding = NamedSharding(mesh, P())
        self.bias.value = jax.device_put(self.bias.value, bias_sharding)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with row parallelism.

        Args:
            x: Input activations (batch, in_features)
                Either sharded or replicated depending on input_is_parallel

        Returns:
            Output activations (batch, out_features), replicated
        """
        if not self.input_is_parallel:
            # Split input across model parallel devices if not already split
            x = jax.lax.all_split(x, axis_name="model", split_axis=1)

        # Matrix multiplication (each device has partial result)
        partial_output = x @ self.weight.value

        # All-reduce to sum partial results
        output = jax.lax.psum(partial_output, axis_name="model")

        # Add bias (only once after reduction)
        output = output + self.bias.value

        return output
```

### Transformer with Tensor Parallelism

```python
class ParallelTransformerLayer(nnx.Module):
    """Transformer layer with Megatron-style tensor parallelism."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        *,
        rngs: nnx.Rngs,
        mesh: Mesh,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Attention: column-parallel for Q, K, V projections
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,  # Q, K, V concatenated
            rngs=rngs,
            mesh=mesh,
            gather_output=False,  # Keep sharded for attention
        )

        # Attention: row-parallel for output projection
        self.output_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            rngs=rngs,
            mesh=mesh,
            input_is_parallel=True,  # Input from sharded attention
        )

        # Feed-forward: column-parallel for first layer
        self.ffn_layer1 = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            rngs=rngs,
            mesh=mesh,
            gather_output=False,  # Keep sharded
        )

        # Feed-forward: row-parallel for second layer
        self.ffn_layer2 = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            rngs=rngs,
            mesh=mesh,
            input_is_parallel=True,
        )

        # Layer norms (replicated)
        self.ln1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_size, rngs=rngs)

    def attention(self, x: jax.Array) -> jax.Array:
        """Multi-head attention with tensor parallelism.

        Args:
            x: (batch, seq_len, hidden_size)

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape
        head_dim = hidden_size // self.num_heads

        # Q, K, V projection (column-parallel)
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * hidden_size) sharded

        # Split into Q, K, V
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        # Each device has subset of heads
        q = q.reshape(batch_size, seq_len, -1, head_dim)
        k = k.reshape(batch_size, seq_len, -1, head_dim)
        v = v.reshape(batch_size, seq_len, -1, head_dim)

        # Attention computation (parallelized across heads)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(head_dim)
        attention_weights = nnx.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bkhd->bqhd", attention_weights, v)

        # Reshape back
        context = context.reshape(batch_size, seq_len, -1)

        # Output projection (row-parallel)
        output = self.output_proj(context)  # All-reduce happens here

        return output

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through transformer layer.

        Args:
            x: (batch, seq_len, hidden_size)

        Returns:
            (batch, seq_len, hidden_size)
        """
        # Attention block with residual
        residual = x
        x = self.ln1(x)
        x = self.attention(x)
        x = residual + x

        # Feed-forward block with residual
        residual = x
        x = self.ln2(x)
        x = self.ffn_layer1(x)
        x = nnx.gelu(x)
        x = self.ffn_layer2(x)
        x = residual + x

        return x


# Create model with tensor parallelism
devices = jax.devices()
mesh = Mesh(devices.reshape(1, 4), axis_names=("data", "model"))

with mesh:
    # Create transformer layer parallelized across 4 devices
    layer = ParallelTransformerLayer(
        hidden_size=768,
        num_heads=12,
        ffn_hidden_size=3072,
        rngs=nnx.Rngs(0),
        mesh=mesh,
    )

    # Forward pass (automatic parallelization)
    x = jnp.ones((2, 128, 768))  # (batch=2, seq_len=128, hidden=768)
    output = layer(x)
    print(f"Output shape: {output.shape}")  # (2, 128, 768)
```

### Sequence Parallelism

For long sequences, also shard along sequence dimension:

```python
class SequenceParallelTransformerLayer(nnx.Module):
    """Transformer with both tensor and sequence parallelism."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        mesh: Mesh,
    ):
        super().__init__()

        # Same as before but with sequence parallelism annotations
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size,
            rngs=rngs, mesh=mesh, gather_output=False
        )
        self.output_proj = RowParallelLinear(
            hidden_size, hidden_size,
            rngs=rngs, mesh=mesh, input_is_parallel=True
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with sequence parallelism.

        Args:
            x: (batch, seq_len, hidden_size)
                Sharded along both seq_len and hidden_size dimensions

        Returns:
            (batch, seq_len, hidden_size) with same sharding
        """
        # Layer norm computed independently on each sequence chunk
        x_norm = nnx.LayerNorm(self.hidden_size)(x)

        # Attention with sequence parallelism
        # All-gather along sequence dimension for attention computation
        x_gathered = jax.lax.all_gather(
            x_norm,
            axis_name="sequence",
            axis=1,  # Gather along sequence dimension
            tiled=True
        )

        # Compute attention on full sequence
        attn_output = self.attention(x_gathered)

        # Split back along sequence dimension
        attn_output = jax.lax.all_split(
            attn_output,
            axis_name="sequence",
            split_axis=1
        )

        # Residual connection
        output = x + attn_output

        return output


# Create mesh with sequence parallelism
devices = jax.devices()  # 8 devices
mesh = Mesh(
    devices.reshape(2, 2, 2),
    axis_names=("data", "model", "sequence")
)

# Shard input along both model and sequence dimensions
sharding = NamedSharding(mesh, P("data", "sequence", "model"))
x = jax.device_put(x, sharding)
```

## Pipeline Parallelism

Pipeline parallelism splits model layers across devices and pipelines microbatches.

### GPipe-Style Pipeline

```python
from typing import Callable
import jax
import jax.numpy as jnp
from flax import nnx

class PipelineParallelModel(nnx.Module):
    """Model with GPipe-style pipeline parallelism."""

    def __init__(
        self,
        layer_configs: list[dict],
        num_microbatches: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.num_microbatches = num_microbatches

        # Create stages (groups of layers)
        self.stages = []
        for config in layer_configs:
            stage = self._create_stage(config, rngs)
            self.stages.append(stage)

    def _create_stage(self, config: dict, rngs: nnx.Rngs) -> nnx.Module:
        """Create a pipeline stage from config."""
        layers = []
        for layer_spec in config["layers"]:
            if layer_spec["type"] == "linear":
                layer = nnx.Linear(
                    in_features=layer_spec["in_features"],
                    out_features=layer_spec["out_features"],
                    rngs=rngs,
                )
            elif layer_spec["type"] == "conv":
                layer = nnx.Conv(
                    in_features=layer_spec["in_channels"],
                    out_features=layer_spec["out_channels"],
                    kernel_size=layer_spec["kernel_size"],
                    rngs=rngs,
                )
            layers.append(layer)

        # Wrap layers in a sequential module
        return nnx.Sequential(*layers)

    def forward_stage(self, stage_id: int, x: jax.Array) -> jax.Array:
        """Forward pass through one pipeline stage."""
        return self.stages[stage_id](x)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with pipeline parallelism."""
        return self._pipeline_forward(x)

    def _pipeline_forward(self, x: jax.Array) -> jax.Array:
        """Execute pipeline forward pass with microbatching."""
        batch_size = x.shape[0]
        microbatch_size = batch_size // self.num_microbatches
        num_stages = len(self.stages)

        # Split input into microbatches
        microbatches = [
            x[i * microbatch_size:(i + 1) * microbatch_size]
            for i in range(self.num_microbatches)
        ]

        # Pipeline execution buffer
        # buffer[stage_id] holds the activation for that stage
        buffer = [None] * num_stages
        outputs = []

        # Pipeline schedule: fill, steady-state, drain
        for time_step in range(num_stages + self.num_microbatches - 1):
            # Process each stage at this time step
            for stage_id in range(num_stages - 1, -1, -1):
                microbatch_id = time_step - stage_id

                if 0 <= microbatch_id < self.num_microbatches:
                    # Get input for this stage
                    if stage_id == 0:
                        stage_input = microbatches[microbatch_id]
                    else:
                        stage_input = buffer[stage_id - 1]

                    # Compute this stage
                    stage_output = self.forward_stage(stage_id, stage_input)

                    # Store in buffer or output
                    if stage_id == num_stages - 1:
                        outputs.append(stage_output)
                    else:
                        buffer[stage_id] = stage_output

        # Concatenate outputs
        return jnp.concatenate(outputs, axis=0)


# Create pipeline model
layer_configs = [
    # Stage 0: Input layers
    {
        "layers": [
            {"type": "linear", "in_features": 784, "out_features": 512},
            {"type": "linear", "in_features": 512, "out_features": 512},
        ]
    },
    # Stage 1: Middle layers
    {
        "layers": [
            {"type": "linear", "in_features": 512, "out_features": 256},
            {"type": "linear", "in_features": 256, "out_features": 256},
        ]
    },
    # Stage 2: Output layers
    {
        "layers": [
            {"type": "linear", "in_features": 256, "out_features": 128},
            {"type": "linear", "in_features": 128, "out_features": 10},
        ]
    },
]

model = PipelineParallelModel(
    layer_configs=layer_configs,
    num_microbatches=4,
    rngs=nnx.Rngs(0),
)

# Forward pass with pipeline parallelism
x = jnp.ones((32, 784))  # Batch of 32
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 10)
```

### 1F1B (One Forward One Backward) Schedule

More memory-efficient pipeline schedule:

```python
class OneFOneBPipeline(nnx.Module):
    """Pipeline with 1F1B (One Forward One Backward) schedule."""

    def __init__(
        self,
        stages: list[nnx.Module],
        num_microbatches: int = 4,
    ):
        super().__init__()
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.num_stages = len(stages)

    def forward_backward_step(
        self,
        stage_id: int,
        forward_input: jax.Array | None,
        backward_grad: jax.Array | None,
    ) -> tuple:
        """Perform one forward and one backward step for a stage."""
        outputs = {}

        # Forward pass if input available
        if forward_input is not None:
            def forward_fn(stage, x):
                return self.stages[stage_id](x)

            # Compute forward and store for backward
            outputs["forward_output"], outputs["forward_vjp"] = jax.vjp(
                lambda x: forward_fn(stage_id, x),
                forward_input
            )

        # Backward pass if gradient available
        if backward_grad is not None and "forward_vjp" in outputs:
            # Compute gradients
            outputs["backward_grad"] = outputs["forward_vjp"](backward_grad)[0]

        return outputs

    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict]:
        """Forward-backward pass with 1F1B schedule."""
        batch_size = x.shape[0]
        microbatch_size = batch_size // self.num_microbatches

        # Split into microbatches
        microbatches = [
            x[i * microbatch_size:(i + 1) * microbatch_size]
            for i in range(self.num_microbatches)
        ]

        # Execution state
        forward_cache = [[None] * self.num_microbatches
                        for _ in range(self.num_stages)]
        backward_grads = [[None] * self.num_microbatches
                         for _ in range(self.num_stages)]

        outputs = []

        # 1F1B Schedule:
        # 1. Warmup: Fill pipeline with forward passes
        # 2. Steady state: Alternate forward and backward
        # 3. Cooldown: Drain backward passes

        warmup_steps = self.num_stages - 1
        steady_steps = self.num_microbatches - warmup_steps

        # Warmup phase
        for step in range(warmup_steps):
            for stage_id in range(step + 1):
                microbatch_id = step - stage_id

                if stage_id == 0:
                    stage_input = microbatches[microbatch_id]
                else:
                    stage_input = forward_cache[stage_id - 1][microbatch_id]

                result = self.forward_backward_step(
                    stage_id, stage_input, None
                )
                forward_cache[stage_id][microbatch_id] = result["forward_output"]

        # Steady state: 1 forward + 1 backward per step
        for step in range(steady_steps):
            microbatch_id = warmup_steps + step

            # Forward pass for new microbatch
            for stage_id in range(self.num_stages):
                if stage_id == 0:
                    stage_input = microbatches[microbatch_id]
                else:
                    stage_input = forward_cache[stage_id - 1][microbatch_id]

                result = self.forward_backward_step(
                    stage_id, stage_input, None
                )
                forward_cache[stage_id][microbatch_id] = result["forward_output"]

            # Backward pass for oldest microbatch in pipeline
            backward_microbatch_id = step
            for stage_id in range(self.num_stages - 1, -1, -1):
                if stage_id == self.num_stages - 1:
                    # Loss gradient (assume 1.0 for now)
                    grad = jnp.ones_like(
                        forward_cache[stage_id][backward_microbatch_id]
                    )
                else:
                    grad = backward_grads[stage_id + 1][backward_microbatch_id]

                result = self.forward_backward_step(
                    stage_id,
                    forward_cache[stage_id][backward_microbatch_id],
                    grad
                )
                if "backward_grad" in result:
                    backward_grads[stage_id][backward_microbatch_id] = result["backward_grad"]

        # Cooldown: Drain remaining backward passes
        for step in range(warmup_steps):
            backward_microbatch_id = steady_steps + step

            for stage_id in range(self.num_stages - 1, -1, -1):
                if stage_id == self.num_stages - 1:
                    grad = jnp.ones_like(
                        forward_cache[stage_id][backward_microbatch_id]
                    )
                else:
                    grad = backward_grads[stage_id + 1][backward_microbatch_id]

                result = self.forward_backward_step(
                    stage_id,
                    forward_cache[stage_id][backward_microbatch_id],
                    grad
                )
                if "backward_grad" in result:
                    backward_grads[stage_id][backward_microbatch_id] = result["backward_grad"]

        # Collect outputs
        final_outputs = [
            forward_cache[self.num_stages - 1][i]
            for i in range(self.num_microbatches)
        ]

        return jnp.concatenate(final_outputs, axis=0), backward_grads[0]
```

## Hybrid Parallelism

Combine multiple parallelism strategies for maximum scale.

### 3D Parallelism

Combine data, tensor, and pipeline parallelism:

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import nnx

class HybridParallelTransformer(nnx.Module):
    """Transformer with 3D parallelism (data + tensor + pipeline)."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_pipeline_stages: int,
        *,
        rngs: nnx.Rngs,
        mesh: Mesh,
    ):
        super().__init__()

        # Divide layers into pipeline stages
        layers_per_stage = num_layers // num_pipeline_stages

        self.stages = []
        for stage_id in range(num_pipeline_stages):
            stage_layers = []

            for _ in range(layers_per_stage):
                # Each layer uses tensor parallelism
                layer = ParallelTransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_hidden_size=4 * hidden_size,
                    rngs=rngs,
                    mesh=mesh,
                )
                stage_layers.append(layer)

            self.stages.append(nnx.Sequential(*stage_layers))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with 3D parallelism.

        Args:
            x: (batch, seq_len, hidden_size)
                Sharded along batch (data parallel),
                hidden_size (tensor parallel),
                and layers (pipeline parallel)

        Returns:
            (batch, seq_len, hidden_size) with same sharding
        """
        # Pipeline forward through stages
        for stage in self.stages:
            x = stage(x)

        return x


# Create 3D parallel mesh
devices = jax.devices()  # e.g., 16 devices
mesh = Mesh(
    devices.reshape(2, 4, 2),  # (data, model, pipeline)
    axis_names=("data", "model", "pipeline")
)

# Create hybrid parallel model
with mesh:
    model = HybridParallelTransformer(
        num_layers=24,
        hidden_size=1024,
        num_heads=16,
        num_pipeline_stages=2,  # 12 layers per stage
        rngs=nnx.Rngs(0),
        mesh=mesh,
    )

    # Define input sharding
    input_sharding = NamedSharding(mesh, P("data", None, "model"))

    # Forward pass
    x = jnp.ones((16, 512, 1024))  # (batch, seq_len, hidden)
    x = jax.device_put(x, input_sharding)
    output = model(x)
```

### Automatic Parallelism Selection

Choose parallelism strategy based on model size and available devices:

```python
def select_parallelism_strategy(
    model_params: int,
    available_devices: int,
    device_memory_gb: float,
) -> dict:
    """Select optimal parallelism strategy.

    Args:
        model_params: Number of model parameters (billions)
        available_devices: Number of available devices
        device_memory_gb: Memory per device (GB)

    Returns:
        Dictionary with parallelism configuration
    """
    # Estimate memory requirements (rough approximation)
    # Parameters: 4 bytes per param (fp32) or 2 bytes (fp16)
    # Gradients: Same as parameters
    # Optimizer states: 2x parameters (Adam)
    # Activations: Depends on batch size, roughly 2x parameters
    memory_per_param_bytes = 2  # fp16
    total_memory_gb = model_params * memory_per_param_bytes * 5 / 1e9

    params_per_device = model_params / available_devices

    if total_memory_gb <= device_memory_gb:
        # Model fits on one device: use data parallelism
        return {
            "strategy": "data_parallel",
            "data_parallel_size": available_devices,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "mesh_shape": (available_devices,),
            "mesh_axis_names": ("data",),
        }

    elif params_per_device * memory_per_param_bytes * 5 / 1e9 <= device_memory_gb:
        # Model fits with data parallelism: use it
        return {
            "strategy": "data_parallel",
            "data_parallel_size": available_devices,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "mesh_shape": (available_devices,),
            "mesh_axis_names": ("data",),
        }

    else:
        # Need model parallelism
        # Prefer tensor parallelism for fast interconnect
        # Fall back to pipeline for slower interconnect

        # Heuristic: Use tensor parallelism up to 8 devices
        # Then add pipeline parallelism
        if available_devices <= 8:
            return {
                "strategy": "tensor_parallel",
                "data_parallel_size": 1,
                "tensor_parallel_size": available_devices,
                "pipeline_parallel_size": 1,
                "mesh_shape": (1, available_devices),
                "mesh_axis_names": ("data", "model"),
            }

        else:
            # 3D parallelism
            # Allocate: 25% data, 50% tensor, 25% pipeline
            # Adjust to factors of available_devices

            # Find factors
            import math

            def find_factors(n):
                factors = []
                for i in range(1, int(math.sqrt(n)) + 1):
                    if n % i == 0:
                        factors.append((i, n // i))
                return factors

            # Simple heuristic: balance tensor and pipeline
            tensor_size = 4  # Typical good value
            if available_devices % tensor_size == 0:
                remaining = available_devices // tensor_size

                # Split remaining between data and pipeline
                data_size = max(1, remaining // 2)
                pipeline_size = remaining // data_size

                return {
                    "strategy": "hybrid_3d",
                    "data_parallel_size": data_size,
                    "tensor_parallel_size": tensor_size,
                    "pipeline_parallel_size": pipeline_size,
                    "mesh_shape": (data_size, tensor_size, pipeline_size),
                    "mesh_axis_names": ("data", "model", "pipeline"),
                }

            # Fallback: Just use available devices for tensor parallelism
            return {
                "strategy": "tensor_parallel",
                "data_parallel_size": 1,
                "tensor_parallel_size": available_devices,
                "pipeline_parallel_size": 1,
                "mesh_shape": (1, available_devices),
                "mesh_axis_names": ("data", "model"),
            }


# Example usage
strategy = select_parallelism_strategy(
    model_params=175,  # 175B parameters (GPT-3 scale)
    available_devices=64,
    device_memory_gb=40,  # A100 40GB
)

print(f"Selected strategy: {strategy['strategy']}")
print(f"Data parallel: {strategy['data_parallel_size']}")
print(f"Tensor parallel: {strategy['tensor_parallel_size']}")
print(f"Pipeline parallel: {strategy['pipeline_parallel_size']}")
print(f"Mesh shape: {strategy['mesh_shape']}")
```

## Memory Optimization

Techniques to maximize model size on limited memory.

### Activation Checkpointing

Trade computation for memory by recomputing activations:

```python
from jax.ad_checkpoint import checkpoint as jax_checkpoint

class CheckpointedTransformerLayer(nnx.Module):
    """Transformer layer with activation checkpointing."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        use_remat: bool = True,
    ):
        super().__init__()
        self.use_remat = use_remat

        # Standard transformer layer components
        self.attention = MultiHeadAttention(hidden_size, num_heads, rngs=rngs)
        self.ffn = FeedForward(hidden_size, 4 * hidden_size, rngs=rngs)
        self.ln1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_size, rngs=rngs)

    def _forward_attention(self, x: jax.Array) -> jax.Array:
        """Attention block (may be checkpointed)."""
        residual = x
        x = self.ln1(x)
        x = self.attention(x)
        x = residual + x
        return x

    def _forward_ffn(self, x: jax.Array) -> jax.Array:
        """Feed-forward block (may be checkpointed)."""
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        return x

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with optional checkpointing."""
        if self.use_remat:
            # Checkpoint both attention and FFN
            # Activations will be recomputed during backward pass
            x = jax_checkpoint(self._forward_attention)(x)
            x = jax_checkpoint(self._forward_ffn)(x)
        else:
            # Standard forward pass
            x = self._forward_attention(x)
            x = self._forward_ffn(x)

        return x


# Compare memory usage
def measure_memory(use_checkpointing: bool):
    """Measure memory usage with/without checkpointing."""
    layer = CheckpointedTransformerLayer(
        hidden_size=1024,
        num_heads=16,
        rngs=nnx.Rngs(0),
        use_remat=use_checkpointing,
    )

    # Dummy forward-backward pass
    x = jnp.ones((32, 512, 1024))  # Large batch and sequence

    def loss_fn(layer, x):
        output = layer(x)
        return jnp.mean(output ** 2)

    # Compute gradients (triggers backward pass)
    loss, grads = nnx.value_and_grad(loss_fn)(layer, x)

    return loss, grads

# Without checkpointing: ~10GB peak memory
# With checkpointing: ~5GB peak memory (50% reduction)
# But ~30% slower due to recomputation
```

### Selective Checkpointing

Checkpoint only memory-intensive operations:

```python
class SelectiveCheckpointedLayer(nnx.Module):
    """Layer with selective activation checkpointing."""

    def __init__(
        self,
        hidden_size: int,
        *,
        rngs: nnx.Rngs,
        checkpoint_attention: bool = True,
        checkpoint_ffn: bool = False,
    ):
        super().__init__()
        self.checkpoint_attention = checkpoint_attention
        self.checkpoint_ffn = checkpoint_ffn

        self.attention = MultiHeadAttention(hidden_size, 16, rngs=rngs)
        self.ffn = FeedForward(hidden_size, 4 * hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with selective checkpointing."""
        # Attention: Large activations (seq_len^2), worth checkpointing
        if self.checkpoint_attention:
            x = jax_checkpoint(self.attention)(x)
        else:
            x = self.attention(x)

        # FFN: Smaller activations, maybe don't checkpoint
        if self.checkpoint_ffn:
            x = jax_checkpoint(self.ffn)(x)
        else:
            x = self.ffn(x)

        return x

# Rule of thumb:
# - Checkpoint attention (quadratic memory in sequence length)
# - Don't checkpoint FFN (linear memory, fast recompute)
# - Checkpoint every N layers in deep models
```

### Gradient Accumulation

Simulate larger batches with gradient accumulation:

```python
@jax.jit
def train_step_with_accumulation(
    model_state,
    batch,
    optimizer_state,
    num_accumulation_steps: int = 4
):
    """Training step with gradient accumulation for memory efficiency."""
    model_graphdef, _ = nnx.split(model)

    # Split batch into sub-batches
    sub_batch_size = batch["data"].shape[0] // num_accumulation_steps

    # Initialize accumulated gradients
    accumulated_grads = None
    total_loss = 0.0

    # Accumulate gradients over sub-batches
    for i in range(num_accumulation_steps):
        start_idx = i * sub_batch_size
        end_idx = (i + 1) * sub_batch_size

        sub_batch = {
            "data": batch["data"][start_idx:end_idx]
        }

        # Compute gradients for sub-batch
        def loss_fn(state):
            model = nnx.merge(model_graphdef, state)
            output = model(sub_batch["data"])
            return output["loss"]

        loss, grads = nnx.value_and_grad(loss_fn)(model_state)

        # Accumulate gradients
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = jax.tree.map(
                lambda acc, g: acc + g,
                accumulated_grads,
                grads
            )

        total_loss += loss

    # Average accumulated gradients
    accumulated_grads = jax.tree.map(
        lambda g: g / num_accumulation_steps,
        accumulated_grads
    )

    # Single optimizer update
    updates, optimizer_state = optimizer.update(
        accumulated_grads,
        optimizer_state
    )
    model_state = optax.apply_updates(model_state, updates)

    return model_state, optimizer_state, total_loss / num_accumulation_steps

# Memory usage: 1/num_accumulation_steps of full batch
# Gradient noise: Same as full batch
# Training speed: ~num_accumulation_steps slower
```

## Troubleshooting

Common issues and solutions in model parallelism.

### Communication Overhead

**Problem**: Training slower than expected due to communication.

**Solutions**:

1. **Profile Communication**:

```python
import jax.profiler

# Profile training step
with jax.profiler.trace("/tmp/trace"):
    train_step(model_state, batch)

# Look for:
# - All-reduce time (should be <20% of step time)
# - All-gather time
# - Point-to-point communication
```

2. **Reduce Tensor Parallelism**:

```python
# If communication > computation, reduce parallelism
# Bad: 8-way tensor parallel on slow interconnect
mesh = Mesh(devices.reshape(1, 8), ("data", "model"))

# Good: 2-way tensor parallel, 4-way data parallel
mesh = Mesh(devices.reshape(4, 2), ("data", "model"))
```

3. **Use Larger Microbatches**:

```python
# Larger microbatches amortize communication
# Bad: Too many small microbatches
model(x, num_microbatches=16)  # High overhead

# Good: Fewer larger microbatches
model(x, num_microbatches=4)  # Lower overhead
```

### Load Imbalance

**Problem**: Some devices idle while others compute.

**Solutions**:

1. **Balance Pipeline Stages**:

```python
# Profile each stage
for stage_id, stage in enumerate(model.stages):
    start = time.time()
    stage(x)
    duration = time.time() - start
    print(f"Stage {stage_id}: {duration:.3f}s")

# Rebalance: Move layers from slow stages to fast stages
```

2. **Adjust Parallelism Dimensions**:

```python
# If tensor parallel devices imbalanced, adjust mesh
# Bad: Imbalanced load
mesh = Mesh(devices.reshape(2, 4), ("data", "model"))

# Good: More balanced
mesh = Mesh(devices.reshape(4, 2), ("data", "model"))
```

### Memory Fragmentation

**Problem**: Out of memory despite sufficient total memory.

**Solutions**:

1. **Use Gradient Checkpointing**:

```python
# Reduce peak memory with checkpointing
layer = CheckpointedTransformerLayer(
    hidden_size=1024,
    num_heads=16,
    rngs=rngs,
    use_remat=True,  # Enable checkpointing
)
```

2. **Increase Pipeline Stages**:

```python
# More pipeline stages = less memory per device
# But more communication and pipeline bubbles
model = PipelineParallelModel(
    layer_configs=layer_configs,
    num_microbatches=8,  # Increase microbatches too
    rngs=rngs,
)
```

## Best Practices

### DO

- ✅ **Start with data parallelism** - simplest and most efficient
- ✅ **Profile before optimizing** - measure actual bottlenecks
- ✅ **Use tensor parallelism for large layers** - effective for transformers
- ✅ **Use pipeline parallelism for many layers** - good for deep models
- ✅ **Combine strategies for extreme scale** - 3D parallelism
- ✅ **Use activation checkpointing** - when memory-constrained
- ✅ **Balance pipeline stages** - equal computation per stage
- ✅ **Match parallelism to interconnect** - tensor needs fast links
- ✅ **Test on small scale first** - validate before scaling
- ✅ **Monitor communication overhead** - should be <20%

### DON'T

- ❌ **Don't over-parallelize** - diminishing returns beyond 8-16 devices
- ❌ **Don't mix strategies randomly** - profile and measure
- ❌ **Don't ignore load imbalance** - causes bubbles and idle time
- ❌ **Don't checkpoint everything** - balance memory vs. compute
- ❌ **Don't use pipeline for small models** - overhead not worth it
- ❌ **Don't use tensor parallel on slow interconnect** - communication dominates
- ❌ **Don't forget gradient averaging** - affects convergence
- ❌ **Don't assume linear scaling** - measure actual speedup
- ❌ **Don't ignore pipeline bubbles** - can waste 20-30% of time
- ❌ **Don't skip testing** - parallelism bugs are subtle

## Summary

Model parallelism enables training large models:

1. **Tensor Parallelism**: Split weight matrices across devices
2. **Pipeline Parallelism**: Distribute layers across devices
3. **Hybrid Parallelism**: Combine strategies for extreme scale
4. **Memory Optimization**: Checkpointing and gradient accumulation

Key trade-offs:

- Tensor parallel: High communication, good for large layers
- Pipeline parallel: Low communication, pipeline bubbles
- Hybrid: Scales to extreme sizes, complex to implement

Choose based on:

- Model size and architecture
- Available devices and interconnect
- Memory constraints
- Target throughput

## Next Steps

<div class="grid cards" markdown>

- :material-content-save:{ .lg .middle } **Checkpointing**

    ---

    Learn about gradient checkpointing and model checkpointing

    [:octicons-arrow-right-24: Checkpointing guide](checkpointing.md)

- :material-cube-outline:{ .lg .middle } **Custom Architectures**

    ---

    Build custom model architectures with parallelism

    [:octicons-arrow-right-24: Architecture guide](architectures.md)

- :material-chart-line:{ .lg .middle } **Distributed Training**

    ---

    Return to distributed training overview

    [:octicons-arrow-right-24: Distributed guide](distributed.md)

- :material-speedometer:{ .lg .middle } **Training Guide**

    ---

    Complete training documentation and best practices

    [:octicons-arrow-right-24: Training guide](../training/training-guide.md)

</div>
