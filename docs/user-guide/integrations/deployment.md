# Model Deployment

Deploy trained Artifex models to production environments, including model export, serving, and optimization for inference.

<div class="grid cards" markdown>

- :material-export:{ .lg .middle } **Model Export**

    ---

    Export models for deployment and inference

    [:octicons-arrow-right-24: Learn more](#model-export)

- :material-server:{ .lg .middle } **Model Serving**

    ---

    Serve models with REST APIs and batch processing

    [:octicons-arrow-right-24: Learn more](#model-serving)

- :material-speedometer:{ .lg .middle } **Optimization**

    ---

    Optimize models for production performance

    [:octicons-arrow-right-24: Learn more](#optimization)

- :material-cloud:{ .lg .middle } **Cloud Deployment**

    ---

    Deploy to cloud platforms and containers

    [:octicons-arrow-right-24: Learn more](#cloud-deployment)

</div>

## Model Export

Export trained models for deployment.

### Export Model State

```python
from artifex.generative_models.core.checkpointing import save_checkpoint
from flax import nnx
import orbax.checkpoint as ocp

# Save trained model
checkpoint_manager, checkpoint_dir = setup_checkpoint_manager(
    base_dir="./models/production/vae_v1"
)

save_checkpoint(checkpoint_manager, model, step=final_step)

print(f"Model exported to {checkpoint_dir}")
```

### Export with Metadata

```python
import json
import orbax.checkpoint as ocp
from flax import nnx

def export_model_with_metadata(
    model: nnx.Module,
    config: dict,
    metrics: dict,
    export_dir: str,
):
    """Export model with configuration and metrics."""
    # Save model checkpoint
    checkpoint_manager, _ = setup_checkpoint_manager(export_dir)
    save_checkpoint(checkpoint_manager, model, step=0)

    # Save configuration
    config_path = f"{export_dir}/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save metrics
    metrics_path = f"{export_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model info
    info = {
        "model_type": config.get("model_type"),
        "input_shape": config.get("input_shape"),
        "latent_dim": config.get("latent_dim"),
        "trained_steps": metrics.get("total_steps"),
        "final_loss": float(metrics.get("final_loss", 0.0)),
    }

    info_path = f"{export_dir}/model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return export_dir

# Export
export_dir = export_model_with_metadata(
    model=model,
    config={"model_type": "vae", "latent_dim": 20, "input_shape": [28, 28, 1]},
    metrics={"total_steps": 10000, "final_loss": 0.15},
    export_dir="./models/production/vae_v1",
)
```

### Load Exported Model

```python
import json
from artifex.generative_models.core.checkpointing import (
    setup_checkpoint_manager,
    load_checkpoint,
)

def load_exported_model(export_dir: str):
    """Load exported model with metadata."""
    # Load configuration
    with open(f"{export_dir}/config.json") as f:
        config = json.load(f)

    # Create model template
    from artifex.generative_models.models.vae import create_vae_model
    from artifex.generative_models.core.configuration import ModelConfig

    model_config = ModelConfig(**config)
    model_template = create_vae_model(model_config, rngs=nnx.Rngs(0))

    # Load checkpoint
    checkpoint_manager, _ = setup_checkpoint_manager(export_dir)
    model, step = load_checkpoint(checkpoint_manager, model_template)

    # Load metrics
    with open(f"{export_dir}/metrics.json") as f:
        metrics = json.load(f)

    return model, config, metrics

# Load
model, config, metrics = load_exported_model("./models/production/vae_v1")
print(f"Loaded model trained for {metrics['total_steps']} steps")
```

## Model Serving

Serve models for inference requests.

### Simple Inference Server

```python
from flask import Flask, request, jsonify
import jax.numpy as jnp
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model at startup
model, config, metrics = load_exported_model("./models/production/vae_v1")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/generate", methods=["POST"])
def generate():
    """Generate samples from the model."""
    data = request.get_json()

    # Get parameters
    num_samples = data.get("num_samples", 1)
    seed = data.get("seed", 0)

    # Generate samples
    key = jax.random.key(seed)
    z = jax.random.normal(key, (num_samples, config["latent_dim"]))

    samples = model.decode(z)
    samples = np.array(samples)

    # Convert to list for JSON
    samples_list = samples.tolist()

    return jsonify({
        "samples": samples_list,
        "num_samples": num_samples,
        "shape": list(samples.shape),
    })

@app.route("/encode", methods=["POST"])
def encode():
    """Encode image to latent representation."""
    # Get image from request
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = image_array.reshape(1, *config["input_shape"])

    # Encode
    output = model.encode(jnp.array(image_array))
    latent = np.array(output["mean"][0])

    return jsonify({
        "latent": latent.tolist(),
        "latent_dim": len(latent),
    })

@app.route("/reconstruct", methods=["POST"])
def reconstruct():
    """Reconstruct image from input."""
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = image_array.reshape(1, *config["input_shape"])

    # Reconstruct
    output = model(jnp.array(image_array))
    reconstruction = np.array(output["reconstruction"][0])

    return jsonify({
        "reconstruction": reconstruction.tolist(),
        "shape": list(reconstruction.shape),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

### Batch Inference

```python
import jax
import jax.numpy as jnp
from typing import Iterator

def batch_inference(
    model: nnx.Module,
    data_iterator: Iterator,
    batch_size: int = 32,
) -> list:
    """Process data in batches for efficient inference."""
    results = []

    batch = []
    for sample in data_iterator:
        batch.append(sample)

        if len(batch) >= batch_size:
            # Process batch
            batch_array = jnp.array(batch)
            output = model(batch_array)

            # Store results
            results.extend(np.array(output["reconstruction"]))

            # Clear batch
            batch = []

    # Process remaining samples
    if batch:
        batch_array = jnp.array(batch)
        output = model(batch_array)
        results.extend(np.array(output["reconstruction"]))

    return results

# Usage
def data_generator():
    """Generator for inference data."""
    for i in range(1000):
        yield np.random.randn(28, 28, 1)

results = batch_inference(model, data_generator(), batch_size=64)
print(f"Processed {len(results)} samples")
```

## Optimization

Optimize models for production performance.

### JIT Compilation

```python
import jax

# JIT-compile inference functions
@jax.jit
def generate_jit(model_state, z):
    """JIT-compiled generation."""
    model = nnx.merge(model_graphdef, model_state)
    return model.decode(z)

@jax.jit
def encode_jit(model_state, x):
    """JIT-compiled encoding."""
    model = nnx.merge(model_graphdef, model_state)
    return model.encode(x)

# Split model once
model_graphdef, model_state = nnx.split(model)

# Fast inference
z = jax.random.normal(jax.random.key(0), (10, 20))
samples = generate_jit(model_state, z)

# First call: compilation + execution (~slow)
# Subsequent calls: cached execution (~fast)
```

### Batched Generation

```python
@jax.jit
def batched_generate(model_state, keys):
    """Generate multiple samples in parallel."""
    # Vectorize over batch
    def generate_single(key):
        z = jax.random.normal(key, (latent_dim,))
        model = nnx.merge(model_graphdef, model_state)
        return model.decode(z[None, :])[0]

    # vmap over keys
    samples = jax.vmap(generate_single)(keys)
    return samples

# Generate 100 samples in parallel
keys = jax.random.split(jax.random.key(0), 100)
samples = batched_generate(model_state, keys)
```

### Mixed Precision

```python
import jax.numpy as jnp

def convert_to_bfloat16(model_state):
    """Convert model to bfloat16 for faster inference."""
    def to_bf16(x):
        if x.dtype == jnp.float32:
            return x.astype(jnp.bfloat16)
        return x

    return jax.tree.map(to_bf16, model_state)

# Convert model
model_state_bf16 = convert_to_bfloat16(model_state)

# Inference in bfloat16 (2x faster on modern GPUs)
@jax.jit
def generate_bf16(model_state, z):
    z = z.astype(jnp.bfloat16)
    model = nnx.merge(model_graphdef, model_state)
    output = model.decode(z)
    return output.astype(jnp.float32)  # Convert back for output

z = jax.random.normal(jax.random.key(0), (10, 20))
samples = generate_bf16(model_state_bf16, z)
```

## Cloud Deployment

Deploy models to cloud platforms.

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY models/ models/
COPY serve.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "serve.py"]
```

```bash
# Build image
docker build -t artifex-vae-server .

# Run container
docker run -p 8000:8000 artifex-vae-server

# Test
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 5}'
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: artifex-vae-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: artifex-vae-server
  template:
    metadata:
      labels:
        app: artifex-vae-server
    spec:
      containers:
      - name: server
        image: artifex-vae-server:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: artifex-vae-service
spec:
  selector:
    app: artifex-vae-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

```bash
# Deploy to Kubernetes
kubectl apply -f deployment.yaml

# Check status
kubectl get pods
kubectl get services
```

### Cloud Functions (Serverless)

```python
# Google Cloud Function
import functions_framework
import jax.numpy as jnp
import numpy as np

# Load model once (cold start)
model = None

def load_model():
    """Load model on cold start."""
    global model
    if model is None:
        model, _, _ = load_exported_model("gs://my-bucket/models/vae_v1")
    return model

@functions_framework.http
def generate_samples(request):
    """Cloud Function for generation."""
    # Load model
    model = load_model()

    # Parse request
    request_json = request.get_json()
    num_samples = request_json.get("num_samples", 1)
    seed = request_json.get("seed", 0)

    # Generate
    key = jax.random.key(seed)
    z = jax.random.normal(key, (num_samples, 20))
    samples = model.decode(z)

    return {
        "samples": np.array(samples).tolist(),
        "num_samples": num_samples,
    }
```

## Monitoring and Logging

Monitor deployed models in production.

### Logging Inference Metrics

```python
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitored_inference(model, input_data):
    """Inference with monitoring."""
    start_time = time.time()

    try:
        # Inference
        output = model(input_data)

        # Log success
        duration = time.time() - start_time
        logger.info(
            f"Inference successful: "
            f"batch_size={input_data.shape[0]}, "
            f"duration={duration:.3f}s"
        )

        return output

    except Exception as e:
        # Log error
        duration = time.time() - start_time
        logger.error(
            f"Inference failed: "
            f"error={str(e)}, "
            f"duration={duration:.3f}s"
        )
        raise
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
inference_counter = Counter(
    "model_inference_total",
    "Total number of inference requests"
)

inference_duration = Histogram(
    "model_inference_duration_seconds",
    "Inference duration in seconds"
)

inference_errors = Counter(
    "model_inference_errors_total",
    "Total number of inference errors"
)

def monitored_inference_prometheus(model, input_data):
    """Inference with Prometheus metrics."""
    inference_counter.inc()

    with inference_duration.time():
        try:
            output = model(input_data)
            return output
        except Exception as e:
            inference_errors.inc()
            raise

# Start Prometheus metrics server
start_http_server(9090)
```

## Best Practices

### DO

- ✅ **Export models with metadata** - include config and metrics
- ✅ **Use JIT compilation** - significant speedup for inference
- ✅ **Implement health checks** - monitor server status
- ✅ **Add logging** - track inference requests and errors
- ✅ **Use batching** - process multiple requests efficiently
- ✅ **Set resource limits** - prevent out-of-memory errors
- ✅ **Version models** - track deployed model versions
- ✅ **Monitor latency** - track inference performance
- ✅ **Use load balancers** - distribute traffic across replicas
- ✅ **Test before deploying** - validate in staging environment

### DON'T

- ❌ **Don't skip JIT** - leave performance on the table
- ❌ **Don't ignore errors** - log and handle gracefully
- ❌ **Don't process one at a time** - use batching
- ❌ **Don't deploy without health checks** - can't monitor status
- ❌ **Don't hardcode configurations** - use environment variables
- ❌ **Don't skip resource limits** - can crash containers
- ❌ **Don't deploy untested models** - validate first
- ❌ **Don't ignore monitoring** - can't debug production issues
- ❌ **Don't use debug mode** - slow and verbose
- ❌ **Don't expose internal errors** - sanitize error messages

## Summary

Model deployment in Artifex:

1. **Export**: Save models with checkpoints and metadata
2. **Serve**: REST APIs, batch processing, cloud functions
3. **Optimize**: JIT compilation, batching, mixed precision
4. **Deploy**: Docker, Kubernetes, serverless platforms
5. **Monitor**: Logging, metrics, health checks

Key considerations:

- Performance: JIT, batching, mixed precision
- Reliability: Health checks, error handling, monitoring
- Scalability: Load balancing, auto-scaling, replicas
- Maintainability: Versioning, logging, configuration

## Next Steps

<div class="grid cards" markdown>

- :material-speedometer:{ .lg .middle } **Inference Guide**

    ---

    Learn more about inference and generation

    [:octicons-arrow-right-24: Inference guide](../inference/overview.md)

- :material-chart-line:{ .lg .middle } **Training Guide**

    ---

    Return to training documentation

    [:octicons-arrow-right-24: Training guide](../training/training-guide.md)

- :material-api:{ .lg .middle } **API Reference**

    ---

    Explore the complete API documentation

    [:octicons-arrow-right-24: API docs](../../api/configuration.md)

- :material-help-circle:{ .lg .middle } **FAQ**

    ---

    Common questions about deployment

    [:octicons-arrow-right-24: FAQ](../../community/faq.md)

</div>
