# Comprehensive Geometric Benchmark Demo

**Level:** Advanced | **Runtime:** ~10-15 minutes (CPU) / ~3-5 minutes (GPU) | **Format:** Python + Jupyter

**Prerequisites:** Understanding of 3D geometry, point clouds, and transformer architectures | **Target Audience:** Users training 3D generative models

## Overview

This example demonstrates a complete end-to-end pipeline for training and evaluating point cloud generation models with Artifex. Learn how to load ShapeNet datasets, train transformer-based geometric models, use Chamfer distance loss, and evaluate with comprehensive 3D metrics.

## What You'll Learn

<div class="grid cards" markdown>

- :material-database: **ShapeNet Dataset**

    ---

    PyTorch3D-style data loading with automatic fallbacks to synthetic data

- :material-cube-outline: **Point Cloud Models**

    ---

    Transformer-based architecture for generating 3D point clouds

- :material-function-variant: **Chamfer Distance**

    ---

    Primary loss function for measuring point cloud similarity

- :material-chart-line: **Training Pipeline**

    ---

    Complete training with Adam optimizer, cosine scheduler, and checkpointing

- :material-chart-box: **Evaluation Metrics**

    ---

    Diversity, coverage, quality, and geometric fidelity scores

- :material-compare: **Benchmark Suite**

    ---

    Compare results against standard geometric benchmarks

</div>

## Files

This example is available in two formats:

- **Python Script**: [`geometric_benchmark_demo.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/geometric/geometric_benchmark_demo.py)
- **Jupyter Notebook**: [`geometric_benchmark_demo.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/geometric/geometric_benchmark_demo.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the complete demo (trains for 50 epochs)
python examples/generative_models/geometric/geometric_benchmark_demo.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/geometric/geometric_benchmark_demo.ipynb
```

## Key Concepts

### 1. Point Cloud Representation

Point clouds are sets of 3D coordinates representing object surfaces:

```python
# Point cloud shape: (batch_size, num_points, 3)
point_cloud = jnp.array([
    [[x1, y1, z1],
     [x2, y2, z2],
     ...
     [xN, yN, zN]]
])  # Shape: (1, 1024, 3)
```

**Key Properties:**

- **Unordered**: No canonical ordering of points
- **Variable size**: Different objects may have different numbers of points
- **Surface representation**: Points typically lie on object surface
- **Normalized**: Usually normalized to unit sphere or box

### 2. ShapeNet Dataset

Large-scale 3D object dataset with 51,300 models across 55 categories:

```python
from artifex.benchmarks.datasets.geometric import ShapeNetDataset

dataset = ShapeNetDataset(
    data_path="./data/shapenet",
    config=data_config,
    rngs=rngs
)

# Get batch
batch = dataset.get_batch(batch_size=8, split="train")
# batch = {
#     "point_clouds": (8, 1024, 3),  # 8 samples, 1024 points each
#     "labels": (8,),                 # Category labels
#     "synsets": ["02691156", ...],   # Category IDs
# }
```

**Synset Categories (examples):**

- `02691156`: Airplane
- `02958343`: Car
- `03001627`: Chair
- `04379243`: Table
- More: See [ShapeNet documentation](https://shapenet.org/)

**Automatic Fallbacks:**

1. Try downloading ShapeNet data
2. Fall back to ModelNet if available
3. Generate synthetic data if needed

### 3. Chamfer Distance Loss

Primary loss function for point clouds, measuring bidirectional nearest-neighbor distances:

$$L_{\text{Chamfer}}(X, Y) = \frac{1}{|X|}\sum_{x \in X} \min_{y \in Y} \|x - y\|^2 + \frac{1}{|Y|}\sum_{y \in Y} \min_{x \in X} \|x - y\|^2$$

```python
from artifex.generative_models.core.losses.geometric import chamfer_distance

# Compute Chamfer distance
loss = chamfer_distance(pred_points, target_points)

# pred_points: (batch, num_points, 3)
# target_points: (batch, num_points, 3)
# loss: scalar value (lower is better)
```

**Interpretation:**

- **First term**: Average distance from predicted to closest real point
- **Second term**: Average distance from real to closest predicted point
- **Symmetric**: Penalizes both missing points and spurious points

### 4. Point Cloud Model Architecture

Transformer-based model for generating point clouds:

```python
from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.models.geometric.point_cloud import PointCloudModel

# Network config with transformer architecture
network_config = PointCloudNetworkConfig(
    name="benchmark_network",
    hidden_dims=(128,),  # Tuple for frozen dataclass
    activation="gelu",
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    dropout_rate=0.1,
)

# Point cloud config
model_config = PointCloudConfig(
    name="point_cloud_model",
    network=network_config,
    num_points=1024,
    dropout_rate=0.1,
)

model = PointCloudModel(config=model_config, rngs=rngs)
```

**Architecture:**

- **Encoder**: Point cloud → latent embedding (via self-attention)
- **Transformer layers**: Multi-head self-attention with residual connections
- **Decoder**: Latent embedding → reconstructed point cloud
- **Permutation invariance**: Order-independent processing via attention

### 5. Training Configuration

Complete training setup with optimizer and scheduler:

```python
from artifex.generative_models.core.configuration import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)

# Optimizer
optimizer_config = OptimizerConfig(
    name="optimizer",
    optimizer_type="adam",
    learning_rate=1e-4,
    weight_decay=1e-5,
    beta1=0.9,
    beta2=0.999,
)

# Learning rate schedule
scheduler_config = SchedulerConfig(
    name="scheduler",
    scheduler_type="cosine",
    warmup_steps=100,
    min_lr_ratio=0.01,
)

# Training
training_config = TrainingConfig(
    name="training",
    batch_size=8,
    num_epochs=50,
    optimizer=optimizer_config,
    scheduler=scheduler_config,
)
```

### 6. Evaluation Metrics

Comprehensive metrics for point cloud generation:

```python
from artifex.benchmarks.metrics.geometric import PointCloudMetrics

metrics = PointCloudMetrics(rngs=rngs, config=eval_config)

results = metrics.compute(
    real_data=real_point_clouds,
    generated_data=generated_point_clouds
)

# results = {
#     "1nn_accuracy": 0.85,          # 1-NN classification accuracy
#     "coverage": 0.72,              # Coverage of real distribution
#     "geometric_fidelity": 0.68,    # Geometric quality score
#     "chamfer_distance": 0.012,     # Average Chamfer distance
# }
```

**Metric Definitions:**

- **1-NN Accuracy**: Classification accuracy using 1-nearest neighbor
  - Tests if generated samples are realistic
  - Higher is better (target: >0.8)

- **Coverage**: Fraction of real samples covered by generated samples
  - Tests distribution diversity
  - Higher is better (target: >0.6)

- **Geometric Fidelity**: Quality of geometric structure
  - Measures surface smoothness and completeness
  - Higher is better (target: >0.7)

- **Chamfer Distance**: Average point-to-point distance
  - Direct reconstruction quality
  - Lower is better (target: <0.02)

### 7. Training Pipeline

Complete training loop with logging and checkpointing:

```python
class GeometricDemoTrainer:
    def train(self):
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(trainer, epoch)

            # Validation phase
            val_metrics = self._validate_epoch(trainer, epoch)

            # Update learning rate
            current_lr = self._update_learning_rate(trainer, epoch)

            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, current_lr)

            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self._save_checkpoint(trainer, epoch)

            # Visualize progress
            if (epoch + 1) % 25 == 0:
                self._visualize_progress(trainer, epoch)

        # Final evaluation
        final_metrics = self._final_evaluation(trainer)

        return trainer, final_metrics
```

## Code Structure

The example consists of three main components:

1. **GeometricDemoTrainer** - Complete trainer orchestrating:
   - Dataset setup (ShapeNet with fallbacks)
   - Model initialization (transformer architecture)
   - Training loop (optimizer, scheduler, logging)
   - Evaluation (comprehensive metrics)
   - Visualization (training curves, samples)

2. **Training Pipeline** - Real optimization:
   - Forward pass through model
   - Chamfer distance loss computation
   - Gradient computation and parameter updates
   - Learning rate scheduling

3. **Evaluation Suite** - Comprehensive metrics:
   - Diversity score (sample variation)
   - Coverage score (distribution coverage)
   - Quality score (geometric properties)
   - Comparison with benchmarks

## Features Demonstrated

- ✅ PyTorch3D-style ShapeNet dataset loading
- ✅ Automatic fallback to synthetic data
- ✅ Transformer-based point cloud model
- ✅ Chamfer distance loss function
- ✅ Adam optimizer with cosine decay schedule
- ✅ Complete training loop with real optimization
- ✅ Training/validation split with proper evaluation
- ✅ Checkpointing and model saving
- ✅ Training visualization (loss curves, samples)
- ✅ Comprehensive evaluation metrics
- ✅ Benchmark comparison
- ✅ Production-ready logging and reporting

## Experiments to Try

1. **Use Real ShapeNet Data**

   ```python
   demo_config = {
       "dataset": {
           "data_path": "./data/shapenet",
           "data_source": "auto",  # Try real data download
           # ...
       }
   }
   ```

2. **Add More Categories**

   ```python
   demo_config = {
       "dataset": {
           "synsets": [
               "02691156",  # Airplane
               "02958343",  # Car
               "03001627",  # Chair
           ],
           # ...
       }
   }
   ```

3. **Increase Model Capacity**

   ```python
   demo_config = {
       "model": {
           "embed_dim": 256,     # More expressive
           "num_layers": 8,      # Deeper network
           "num_heads": 16,      # More attention
       }
   }
   ```

4. **Longer Training**

   ```python
   demo_config = {
       "training": {
           "num_epochs": 200,    # More training
           "batch_size": 16,     # Larger batches (if GPU allows)
       }
   }
   ```

5. **Different Optimizers**

   ```python
   demo_config = {
       "training": {
           "optimizer": {
               "optimizer_type": "adamw",
               "weight_decay": 1e-4,  # More regularization
           }
       }
   }
   ```

## Next Steps

<div class="grid cards" markdown>

- :material-arrow-right: **Advanced Architectures**

    ---

    Try PointNet++, DGCNN, or diffusion models

    [:octicons-arrow-right-24: Advanced 3D Models](#)

- :material-arrow-right: **Conditional Generation**

    ---

    Generate point clouds conditioned on category

    [:octicons-arrow-right-24: Conditional 3D](#)

- :material-arrow-right: **Mesh Generation**

    ---

    Extend to surface reconstruction and meshing

    [:octicons-arrow-right-24: Mesh Models](#)

- :material-arrow-right: **Loss Functions**

    ---

    Explore geometric loss functions

    [:octicons-arrow-right-24: Loss Examples](../losses/loss-examples.md)

</div>

## Troubleshooting

### Dataset Download Fails

**Symptom:** Error downloading ShapeNet data

**Solution:** The example automatically falls back to synthetic data

```python
# Synthetic data is generated automatically
# To try real data:
demo_config["dataset"]["data_source"] = "auto"
```

### Training Too Slow

**Symptom:** Training takes >20 minutes

**Solution:** Reduce epochs or batch size

```python
demo_config["training"]["num_epochs"] = 25  # Faster
demo_config["training"]["batch_size"] = 4   # Less memory
```

### CUDA Out of Memory

**Symptom:** `CUDA out of memory` error during training

**Solution:** Reduce batch size or model size

```python
demo_config["training"]["batch_size"] = 4
demo_config["model"]["embed_dim"] = 64
demo_config["dataset"]["num_points"] = 512  # Fewer points
```

### Poor Generation Quality

**Symptom:** Generated point clouds look random

**Cause:** Insufficient training or model capacity

**Solution:** Train longer or increase model size

```python
demo_config["training"]["num_epochs"] = 100
demo_config["model"]["embed_dim"] = 256
demo_config["model"]["num_layers"] = 8
```

### Loss Not Decreasing

**Symptom:** Training loss plateaus or increases

**Cause:** Learning rate too high or optimizer issue

**Solution:** Reduce learning rate or adjust optimizer

```python
demo_config["training"]["optimizer"]["learning_rate"] = 5e-5  # Lower LR
demo_config["training"]["optimizer"]["weight_decay"] = 1e-6   # Less regularization
```

## Additional Resources

### Documentation

- [Geometric Benchmark Suite](../../benchmarks/geometric_suite.md) - Complete benchmarking guide
- [Point Cloud Models](../../models/point_cloud.md) - Model architecture details
- [Chamfer Distance](../../core/geometric.md) - Loss function documentation

### Related Examples

- [Loss Examples](../losses/loss-examples.md) - Geometric loss functions
- [Framework Features Demo](../framework/framework-features-demo.md) - Configuration system

### Papers and Resources

- **PointNet**: [PointNet: Deep Learning on Point Sets (Qi et al., 2017)](https://arxiv.org/abs/1612.00593)
- **PointNet++**: [PointNet++: Deep Hierarchical Feature Learning (Qi et al., 2017)](https://arxiv.org/abs/1706.02413)
- **ShapeNet**: [ShapeNet: An Information-Rich 3D Model Repository (Chang et al., 2015)](https://arxiv.org/abs/1512.03012)
- **Point Cloud Transformers**: [PCT: Point Cloud Transformer (Guo et al., 2021)](https://arxiv.org/abs/2012.09688)
- **Chamfer Distance**: [Learning Representations and Generative Models for 3D Point Clouds](https://arxiv.org/abs/1707.02392)

### External Tools

- **PyTorch3D**: [PyTorch library for 3D deep learning](https://pytorch3d.org/)
- **Open3D**: [Modern library for 3D data processing](http://www.open3d.org/)
- **Kaolin**: [PyTorch library for 3D deep learning research](https://kaolin.readthedocs.io/)
