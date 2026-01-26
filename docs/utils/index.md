# Utilities

Comprehensive utility modules for JAX operations, logging, visualization, I/O, and development tools.

## Overview

<div class="grid cards" markdown>

- :material-cog:{ .lg .middle } **JAX Utilities**

    ---

    Device management, PRNG handling, dtype utilities, and Flax helpers

- :material-chart-line:{ .lg .middle } **Logging & Metrics**

    ---

    MLflow, Weights & Biases, and custom logging integrations

- :material-image:{ .lg .middle } **Visualization**

    ---

    Attention maps, latent space plots, image grids, and protein visualization

- :material-speedometer:{ .lg .middle } **Profiling**

    ---

    Memory profiling, performance tracking, and XProf integration

</div>

## JAX Utilities

### Device Management

```python
from artifex.utils.jax import get_device, get_available_devices

# Get default device
device = get_device()  # Returns GPU if available, else CPU

# List all devices
devices = get_available_devices()
print(f"Available: {devices}")
```

[:octicons-arrow-right-24: Device Utilities](device.md)

### PRNG Handling

```python
from artifex.utils.jax import create_prng_key, split_key

# Create a key
key = create_prng_key(42)

# Split for multiple uses
key1, key2, key3 = split_key(key, num=3)
```

[:octicons-arrow-right-24: PRNG Utilities](prng.md)

### Data Types

```python
from artifex.utils.jax import get_dtype, ensure_dtype

# Get appropriate dtype
dtype = get_dtype("float32")

# Convert array to dtype
array = ensure_dtype(array, "bfloat16")
```

[:octicons-arrow-right-24: Dtype Utilities](dtype.md)

### Shape Utilities

```python
from artifex.utils.jax import flatten_batch, unflatten_batch

# Flatten batch dimensions
flat, shape = flatten_batch(tensor, num_batch_dims=2)

# Restore batch dimensions
restored = unflatten_batch(flat, shape)
```

[:octicons-arrow-right-24: Shape Utilities](shapes.md)

### Flax Utilities

```python
from artifex.utils.jax import count_params, get_param_shapes

# Count model parameters
num_params = count_params(model)

# Get parameter shapes
shapes = get_param_shapes(model)
```

[:octicons-arrow-right-24: Flax Utilities](flax_utils.md)

## Logging & Metrics

### Logger

```python
from artifex.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Training started")
logger.debug("Batch size: 128")
```

[:octicons-arrow-right-24: Logger Reference](logger.md)

### Weights & Biases

```python
from artifex.utils.logging import WandbLogger

logger = WandbLogger(
    project="my-project",
    name="experiment-001",
    config=config_dict,
)

logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
logger.log_image("samples", image_array)
```

[:octicons-arrow-right-24: W&B Integration](wandb.md)

### MLflow

```python
from artifex.utils.logging import MLflowLogger

logger = MLflowLogger(
    experiment_name="vae-experiments",
    tracking_uri="http://localhost:5000",
)

logger.log_params({"learning_rate": 1e-3})
logger.log_metrics({"loss": 0.5}, step=100)
```

[:octicons-arrow-right-24: MLflow Integration](mlflow.md)

### Metrics Tracking

```python
from artifex.utils.logging import MetricsTracker

tracker = MetricsTracker()
tracker.update("loss", 0.5)
tracker.update("loss", 0.4)

avg = tracker.compute("loss")  # Returns average
tracker.reset()
```

[:octicons-arrow-right-24: Metrics Tracking](metrics.md)

## Visualization

### Image Grids

```python
from artifex.utils.visualization import create_image_grid, save_image_grid

# Create grid from batch
grid = create_image_grid(images, nrow=8)

# Save to file
save_image_grid(images, "samples.png", nrow=8)
```

[:octicons-arrow-right-24: Image Grid](image_grid.md)

### Latent Space Visualization

```python
from artifex.utils.visualization import plot_latent_space

# Plot 2D latent space with labels
plot_latent_space(
    latents,
    labels=labels,
    method="tsne",  # or "pca", "umap"
    save_path="latent_space.png",
)
```

[:octicons-arrow-right-24: Latent Space](latent_space.md)

### Attention Visualization

```python
from artifex.utils.visualization import visualize_attention

# Visualize attention weights
visualize_attention(
    attention_weights,
    tokens=tokens,
    save_path="attention.png",
)
```

[:octicons-arrow-right-24: Attention Visualization](attention_vis.md)

### Plotting

```python
from artifex.utils.visualization import plot_training_curves

# Plot loss curves
plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    save_path="training_curves.png",
)
```

[:octicons-arrow-right-24: Plotting Utilities](plotting.md)

### Protein Visualization

```python
from artifex.utils.visualization import visualize_protein_structure

# Visualize protein backbone
visualize_protein_structure(
    coordinates=coords,
    sequence=sequence,
    save_path="protein.png",
)
```

[:octicons-arrow-right-24: Protein Visualization](protein.md)

## I/O Utilities

### File Operations

```python
from artifex.utils.io import save_checkpoint, load_checkpoint

# Save model checkpoint
save_checkpoint(model, optimizer, "checkpoint.ckpt")

# Load checkpoint
model, optimizer = load_checkpoint("checkpoint.ckpt")
```

[:octicons-arrow-right-24: File Utilities](file.md)

### Format Conversion

```python
from artifex.utils.io import convert_format

# Convert between formats
convert_format(
    input_path="model.ckpt",
    output_path="model.safetensors",
    format="safetensors",
)
```

[:octicons-arrow-right-24: Format Utilities](formats.md)

### Serialization

```python
from artifex.utils.io import serialize_config, deserialize_config

# Serialize to YAML
yaml_str = serialize_config(config, format="yaml")

# Deserialize from JSON
config = deserialize_config(json_str, format="json")
```

[:octicons-arrow-right-24: Serialization](serialization.md)

## Profiling

### Memory Profiling

```python
from artifex.utils.profiling import memory_profiler

with memory_profiler() as prof:
    output = model(input)

print(f"Peak memory: {prof.peak_memory_mb:.2f} MB")
```

[:octicons-arrow-right-24: Memory Profiling](memory.md)

### Performance Profiling

```python
from artifex.utils.profiling import profile_function

@profile_function
def train_step(batch):
    return model(batch)

# Profiling results printed automatically
```

[:octicons-arrow-right-24: Performance Profiling](performance.md)

### XProf Integration

```python
from artifex.utils.profiling import start_xprof, stop_xprof

start_xprof(log_dir="profiles/")
# ... training code ...
stop_xprof()
```

[:octicons-arrow-right-24: XProf Integration](xprof.md)

## Image Utilities

### Color Operations

```python
from artifex.utils.image import rgb_to_grayscale, normalize_image

grayscale = rgb_to_grayscale(image)
normalized = normalize_image(image, mean=0.5, std=0.5)
```

[:octicons-arrow-right-24: Color Utilities](color.md)

### Image Metrics

```python
from artifex.utils.image import compute_psnr, compute_ssim

psnr = compute_psnr(generated, reference)
ssim = compute_ssim(generated, reference)
```

[:octicons-arrow-right-24: Image Metrics](metrics.md)

### Transforms

```python
from artifex.utils.image import resize, center_crop, random_flip

resized = resize(image, size=(256, 256))
cropped = center_crop(image, size=(224, 224))
flipped = random_flip(image, key=prng_key)
```

[:octicons-arrow-right-24: Image Transforms](transforms.md)

## Numerical Utilities

### Math Operations

```python
from artifex.utils.numerical import log_sum_exp, softmax_temperature

# Numerically stable log-sum-exp
result = log_sum_exp(logits, axis=-1)

# Softmax with temperature
probs = softmax_temperature(logits, temperature=0.7)
```

[:octicons-arrow-right-24: Math Utilities](math.md)

### Numerical Stability

```python
from artifex.utils.numerical import safe_log, safe_divide

# Safe log with epsilon
log_x = safe_log(x, eps=1e-8)

# Safe division
result = safe_divide(a, b, eps=1e-8)
```

[:octicons-arrow-right-24: Stability Utilities](stability.md)

### Statistics

```python
from artifex.utils.numerical import running_mean, exponential_moving_average

# Compute running statistics
mean = running_mean(values)
ema = exponential_moving_average(values, decay=0.99)
```

[:octicons-arrow-right-24: Statistics Utilities](stats.md)

## Text Utilities

```python
from artifex.utils.text import compute_bleu, compute_rouge

bleu = compute_bleu(predictions, references)
rouge = compute_rouge(predictions, references)
```

[:octicons-arrow-right-24: Text Metrics](metrics.md)

## Development Utilities

### Timer

```python
from artifex.utils import Timer

with Timer("training_step"):
    output = train_step(batch)
# Prints: training_step took 0.123s
```

[:octicons-arrow-right-24: Timer](timer.md)

### Registry

```python
from artifex.utils import Registry

models = Registry("models")

@models.register("my_model")
class MyModel:
    pass

model_class = models.get("my_model")
```

[:octicons-arrow-right-24: Registry](registry.md)

### Environment

```python
from artifex.utils import get_env, set_env

# Get environment variable with default
value = get_env("MY_VAR", default="default_value")
```

[:octicons-arrow-right-24: Environment](env.md)

### Dependency Analyzer

```python
from artifex.utils import analyze_dependencies

# Analyze module dependencies
deps = analyze_dependencies("artifex.generative_models")
```

[:octicons-arrow-right-24: Dependency Analyzer](dependency_analyzer.md)

## Module Reference

| Category | Modules |
|----------|---------|
| **JAX** | [device](device.md), [dtype](dtype.md), [flax_utils](flax_utils.md), [prng](prng.md), [shapes](shapes.md) |
| **Logging** | [logger](logger.md), [metrics](metrics.md), [mlflow](mlflow.md), [wandb](wandb.md), [file_utils](file_utils.md) |
| **Visualization** | [attention_vis](attention_vis.md), [image_grid](image_grid.md), [latent_space](latent_space.md), [plotting](plotting.md), [protein](protein.md) |
| **I/O** | [file](file.md), [formats](formats.md), [serialization](serialization.md) |
| **Profiling** | [memory](memory.md), [performance](performance.md), [xprof](xprof.md) |
| **Image** | [color](color.md), [metrics](metrics.md), [transforms](transforms.md) |
| **Numerical** | [math](math.md), [stability](stability.md), [stats](stats.md) |
| **Text** | [metrics](metrics.md), [postprocessing](postprocessing.md), [processing](processing.md) |
| **Utils** | [env](env.md), [registry](registry.md), [timer](timer.md), [types](types.md) |

## Related Documentation

- [Training Guide](../user-guide/training/training-guide.md) - Using utilities in training
- [Logging & Tracking](../user-guide/training/logging.md) - Experiment tracking
- [Performance Profiling](../user-guide/training/profiling.md) - Profiling guide
