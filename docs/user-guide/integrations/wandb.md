# Weights & Biases Integration

Complete guide to experiment tracking, hyperparameter sweeps, and artifact management using Weights & Biases with Artifex.

## Overview

Weights & Biases (W&B) is a powerful platform for tracking machine learning experiments. Artifex integrates seamlessly with W&B to log metrics, visualize training progress, run hyperparameter sweeps, and manage model artifacts.

!!! tip "W&B Benefits"
    - **Experiment Tracking**: Log metrics, hyperparameters, and system info
    - **Visualizations**: Interactive charts and sample galleries
    - **Hyperparameter Sweeps**: Automated optimization
    - **Artifact Management**: Version models and datasets
    - **Team Collaboration**: Share results with teammates

<div class="grid cards" markdown>

- :material-chart-line:{ .lg .middle } **Experiment Tracking**

    ---

    Log metrics, losses, and visualizations

    [:octicons-arrow-right-24: Tracking Guide](#experiment-tracking)

- :material-tune:{ .lg .middle } **Hyperparameter Sweeps**

    ---

    Automate hyperparameter optimization

    [:octicons-arrow-right-24: Sweeps Guide](#hyperparameter-sweeps)

- :material-package-variant:{ .lg .middle } **Artifact Management**

    ---

    Version and track models and datasets

    [:octicons-arrow-right-24: Artifacts Guide](#artifact-management)

- :material-file-document:{ .lg .middle } **Reports**

    ---

    Create shareable experiment reports

    [:octicons-arrow-right-24: Reports Guide](#report-generation)

</div>

---

## Quick Start with Built-in Callback

For most use cases, use the built-in `WandbLoggerCallback`:

```python
from artifex.generative_models.training.callbacks import (
    WandbLoggerCallback,
    WandbLoggerConfig,
)

callback = WandbLoggerCallback(WandbLoggerConfig(
    project="my-project",
    name="experiment-1",
    tags=["vae", "baseline"],
    config={"learning_rate": 1e-3},
))

trainer.fit(callbacks=[callback])
```

See [Logging Callbacks](../../training/logging.md) for full documentation.

The sections below cover advanced W&B features not available through the callback.

---

## Prerequisites

Install and configure W&B:

```bash
# Install wandb
pip install wandb

# Or using uv
uv pip install wandb

# Login
wandb login
```

---

## Experiment Tracking

### Basic Logging

Track metrics during training.

```python
import wandb
from flax import nnx
import jax.numpy as jnp

class WandBTrainer:
    """Trainer with W&B logging."""

    def __init__(
        self,
        model,
        config: dict,
        project_name: str = "artifex-experiments",
    ):
        self.model = model
        self.config = config

        # Initialize W&B
        wandb.init(
            project=project_name,
            config=config,
            name=f"{config['model_type']}-{config.get('experiment_name', 'default')}",
        )

        # Log model architecture
        wandb.watch(self.model, log_freq=100)

    def train_step(self, batch, step: int):
        """Single training step with logging."""
        # Compute loss (simplified)
        loss, metrics = self.compute_loss(batch)

        # Log metrics
        wandb.log({
            "train/loss": float(loss),
            "train/step": step,
            **{f"train/{k}": float(v) for k, v in metrics.items()}
        }, step=step)

        return loss, metrics

    def validation_step(self, val_data, step: int):
        """Validation with logging."""
        val_loss, val_metrics = self.evaluate(val_data)

        # Log validation metrics
        wandb.log({
            "val/loss": float(val_loss),
            **{f"val/{k}": float(v) for k, v in val_metrics.items()}
        }, step=step)

        return val_loss, val_metrics

    def log_images(self, images, step: int, key: str = "samples"):
        """Log generated images."""
        # Convert to numpy and denormalize
        images_np = np.array(images)
        images_np = ((images_np + 1) / 2 * 255).astype(np.uint8)

        # Log to W&B
        wandb.log({
            key: [wandb.Image(img) for img in images_np]
        }, step=step)

    def finish(self):
        """Finish W&B run."""
        wandb.finish()
```

### Advanced Metrics Logging

Track custom metrics and visualizations.

```python
class AdvancedWandBLogger:
    """Advanced W&B logging with custom metrics."""

    def __init__(self, project: str, config: dict):
        wandb.init(project=project, config=config)
        self.step = 0

    def log_training_metrics(
        self,
        loss: float,
        reconstruction_loss: float,
        kl_loss: float,
        learning_rate: float,
    ):
        """Log detailed training metrics."""
        wandb.log({
            # Loss components
            "loss/total": loss,
            "loss/reconstruction": reconstruction_loss,
            "loss/kl_divergence": kl_loss,

            # Optimization
            "optimization/learning_rate": learning_rate,
            "optimization/step": self.step,

            # Loss ratios
            "analysis/recon_kl_ratio": reconstruction_loss / (kl_loss + 1e-8),
        }, step=self.step)

        self.step += 1

    def log_histograms(self, model):
        """Log parameter histograms."""
        state = nnx.state(model)

        for name, param in state.items():
            if isinstance(param, jnp.ndarray):
                wandb.log({
                    f"histograms/{name}": wandb.Histogram(np.array(param))
                }, step=self.step)

    def log_latent_space(
        self,
        latent_codes: jax.Array,
        labels: jax.Array = None,
    ):
        """Log latent space visualization."""
        # Convert to numpy
        latent_np = np.array(latent_codes)

        # Log scatter plot
        if labels is not None:
            labels_np = np.array(labels)
            data = [[x, y, label] for (x, y), label in zip(latent_np[:, :2], labels_np)]
            table = wandb.Table(data=data, columns=["x", "y", "label"])
            wandb.log({
                "latent_space": wandb.plot.scatter(
                    table, "x", "y", "label",
                    title="Latent Space Visualization"
                )
            }, step=self.step)
        else:
            wandb.log({
                "latent_space": wandb.Histogram(latent_np)
            }, step=self.step)

    def log_generation_quality(
        self,
        real_images: jax.Array,
        fake_images: jax.Array,
    ):
        """Log generation quality metrics."""
        from artifex.benchmarks.metrics import compute_fid, compute_inception_score

        # Compute metrics
        fid = compute_fid(real_images, fake_images)
        inception_score, _ = compute_inception_score(fake_images)

        wandb.log({
            "quality/fid": fid,
            "quality/inception_score": inception_score,
        }, step=self.step)
```

---

## Hyperparameter Sweeps

### Sweep Configuration

Define hyperparameter search space.

```python
# sweep_config.yaml or Python dict
sweep_config = {
    "method": "bayes",  # bayes, grid, or random
    "metric": {
        "name": "val/loss",
        "goal": "minimize",
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "latent_dim": {
            "values": [64, 128, 256, 512],
        },
        "beta": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 10.0,
        },
        "batch_size": {
            "values": [32, 64, 128],
        },
        "architecture": {
            "values": ["conv", "resnet", "transformer"],
        },
    },
}

# Create sweep
sweep_id = wandb.sweep(
    sweep=sweep_config,
    project="artifex-sweeps",
)
```

### Running Sweeps

Execute hyperparameter sweep.

```python
def train_with_sweep():
    """Training function for W&B sweep."""
    # Initialize W&B run
    run = wandb.init()

    # Get hyperparameters from sweep
    config = wandb.config

    # Build model with sweep config
    model = build_model(
        latent_dim=config.latent_dim,
        architecture=config.architecture,
    )

    # Train model
    for epoch in range(config.num_epochs):
        # Training loop
        train_loss = train_epoch(
            model,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            beta=config.beta,
        )

        # Validation
        val_loss = validate(model)

        # Log metrics
        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "epoch": epoch,
        })

    # Finish run
    wandb.finish()


# Run sweep agent
wandb.agent(
    sweep_id=sweep_id,
    function=train_with_sweep,
    count=50,  # Number of runs
)
```

### Multi-Objective Optimization

Optimize for multiple metrics simultaneously.

```python
sweep_config_multi = {
    "method": "bayes",
    "metric": {
        "name": "combined_score",
        "goal": "maximize",
    },
    "parameters": {
        # ... same as before
    },
}

def train_with_multi_objective():
    """Train with multiple objectives."""
    run = wandb.init()
    config = wandb.config

    # Training...
    val_loss = validate(model)
    fid_score = compute_fid(model)
    inference_time = benchmark_inference(model)

    # Combine objectives
    # Lower is better for loss and FID, lower is better for time
    combined_score = -val_loss - fid_score / 100 - inference_time

    wandb.log({
        "val/loss": val_loss,
        "quality/fid": fid_score,
        "performance/inference_time": inference_time,
        "combined_score": combined_score,
    })

    wandb.finish()
```

---

## Artifact Management

### Model Artifacts

Version and track trained models.

```python
class ArtifactManager:
    """Manage W&B artifacts for models and datasets."""

    def __init__(self, project: str):
        self.project = project

    def save_model_artifact(
        self,
        model,
        artifact_name: str,
        metadata: dict = None,
    ):
        """Save model as W&B artifact.

        Args:
            model: Trained model
            artifact_name: Artifact name (e.g., "vae-mnist")
            metadata: Additional metadata
        """
        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata=metadata or {},
        )

        # Save model to temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model"

            # Export model
            state = nnx.state(model)
            with open(f"{model_path}/params.pkl", "wb") as f:
                import pickle
                pickle.dump(state, f)

            # Add to artifact
            artifact.add_dir(model_path)

        # Log artifact
        wandb.log_artifact(artifact)

        print(f"Model saved as artifact: {artifact_name}")

    def load_model_artifact(
        self,
        artifact_name: str,
        version: str = "latest",
    ):
        """Load model from artifact.

        Args:
            artifact_name: Artifact name
            version: Artifact version ("latest" or "v0", "v1", etc.)

        Returns:
            Loaded model
        """
        # Download artifact
        artifact = wandb.use_artifact(
            f"{artifact_name}:{version}",
            type="model",
        )
        artifact_dir = artifact.download()

        # Load model
        import pickle
        with open(f"{artifact_dir}/params.pkl", "rb") as f:
            state = pickle.load(f)

        # Reconstruct model
        # (Simplified - actual implementation needs proper model reconstruction)
        model = reconstruct_model(state)

        return model

    def save_dataset_artifact(
        self,
        data: jax.Array,
        artifact_name: str,
        description: str = "",
    ):
        """Save dataset as artifact.

        Args:
            data: Dataset array
            artifact_name: Artifact name
            description: Dataset description
        """
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description=description,
        )

        # Save as numpy array
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = f"{tmpdir}/data.npy"
            np.save(data_path, np.array(data))
            artifact.add_file(data_path)

        wandb.log_artifact(artifact)

    def link_artifacts(
        self,
        model_artifact: str,
        dataset_artifact: str,
    ):
        """Link model to training dataset.

        Args:
            model_artifact: Model artifact name
            dataset_artifact: Dataset artifact name
        """
        # Get artifacts
        model = wandb.use_artifact(f"{model_artifact}:latest")
        dataset = wandb.use_artifact(f"{dataset_artifact}:latest")

        # Link them
        model.link(f"trained_on_{dataset_artifact}")
        wandb.log_artifact(model)
```

### Artifact Lineage

Track relationships between artifacts.

```python
def create_artifact_lineage():
    """Create artifact lineage for full experiment tracking."""
    run = wandb.init(project="artifex-lineage")

    # Log dataset
    dataset_artifact = wandb.Artifact("mnist-train", type="dataset")
    dataset_artifact.add_file("data/mnist_train.npy")
    wandb.log_artifact(dataset_artifact)

    # Use dataset in training
    dataset = wandb.use_artifact("mnist-train:latest")
    dataset_dir = dataset.download()

    # Train model...
    model = train_model(dataset_dir)

    # Log trained model (automatically linked to dataset)
    model_artifact = wandb.Artifact("vae-model", type="model")
    model_artifact.add_dir("models/vae")
    wandb.log_artifact(model_artifact)

    # Log evaluation results
    eval_artifact = wandb.Artifact("vae-evaluation", type="evaluation")
    eval_artifact.add_file("results/metrics.json")
    wandb.log_artifact(eval_artifact)

    wandb.finish()
```

---

## Report Generation

### Creating Reports

Generate shareable experiment reports.

```python
class ReportGenerator:
    """Generate W&B reports."""

    @staticmethod
    def create_experiment_report(
        project: str,
        runs: list[str],
        title: str,
    ) -> str:
        """Create comparison report.

        Args:
            project: W&B project name
            runs: List of run IDs to compare
            title: Report title

        Returns:
            Report URL
        """
        import wandb

        # Create report
        report = wandb.apis.reports.Report(
            project=project,
            title=title,
            description="Experiment comparison report",
        )

        # Add run comparison
        report.blocks = [
            wandb.apis.reports.RunComparer(
                diff_only=False,
                runsets=[
                    wandb.apis.reports.Runset(
                        project=project,
                        filters={"$or": [{"name": run_id} for run_id in runs]},
                    )
                ],
            ),
            wandb.apis.reports.LinePlot(
                title="Training Loss Comparison",
                x="step",
                y=["train/loss"],
                runsets=[
                    wandb.apis.reports.Runset(
                        project=project,
                        filters={"$or": [{"name": run_id} for run_id in runs]},
                    )
                ],
            ),
        ]

        # Save and get URL
        report.save()
        return report.url
```

### Custom Visualizations

Create custom plots for reports.

```python
def log_custom_visualizations(model, test_data):
    """Log custom visualizations to W&B."""

    # 1. Sample Grid
    samples = model.sample(num_samples=64)
    wandb.log({
        "visualizations/sample_grid": wandb.Image(
            create_image_grid(samples, nrow=8)
        )
    })

    # 2. Reconstruction Comparison
    reconstructions = model.reconstruct(test_data[:8])
    comparison = np.concatenate([test_data[:8], reconstructions], axis=0)
    wandb.log({
        "visualizations/reconstruction": wandb.Image(
            create_image_grid(comparison, nrow=8)
        )
    })

    # 3. Latent Space 2D Projection
    from sklearn.manifold import TSNE

    latents, labels = encode_dataset(model, test_data)
    tsne = TSNE(n_components=2)
    latents_2d = tsne.fit_transform(np.array(latents))

    # Create scatter plot
    data = [[x, y, int(label)] for (x, y), label in zip(latents_2d, labels)]
    table = wandb.Table(data=data, columns=["x", "y", "label"])
    wandb.log({
        "visualizations/latent_tsne": wandb.plot.scatter(
            table, "x", "y", "label",
            title="Latent Space t-SNE"
        )
    })

    # 4. Interpolation Video
    interpolation_frames = create_interpolation(model, num_frames=60)
    wandb.log({
        "visualizations/interpolation": wandb.Video(
            interpolation_frames, fps=30, format="gif"
        )
    })
```

---

## Integration with Artifex Training

### Complete Training Example

Full integration example.

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.core import DeviceManager

class ArtifexWandBTrainer(Trainer):
    """Artifex Trainer with W&B integration."""

    def __init__(
        self,
        model,
        config: dict,
        wandb_project: str = "artifex",
        **kwargs
    ):
        super().__init__(model, config, **kwargs)

        # Initialize W&B
        wandb.init(
            project=wandb_project,
            config=config,
            name=config.get("experiment_name"),
        )

        self.wandb_log_frequency = config.get("wandb_log_frequency", 100)

    def on_train_step_end(self, step: int, loss: float, metrics: dict):
        """Called after each training step."""
        if step % self.wandb_log_frequency == 0:
            wandb.log({
                "train/loss": float(loss),
                "train/step": step,
                **{f"train/{k}": float(v) for k, v in metrics.items()}
            }, step=step)

    def on_validation_end(self, epoch: int, val_metrics: dict):
        """Called after validation."""
        wandb.log({
            "val/epoch": epoch,
            **{f"val/{k}": float(v) for k, v in val_metrics.items()}
        }, step=epoch)

        # Log sample images
        samples = self.model.sample(num_samples=16, rngs=self.rngs)
        wandb.log({
            "samples": [wandb.Image(img) for img in np.array(samples)]
        }, step=epoch)

    def on_training_end(self):
        """Called when training completes."""
        # Save final model as artifact
        artifact = wandb.Artifact("final_model", type="model")
        artifact.add_dir(self.checkpoint_dir)
        wandb.log_artifact(artifact)

        wandb.finish()


# Usage
config = {
    "model_type": "vae",
    "latent_dim": 128,
    "learning_rate": 1e-4,
    "batch_size": 128,
    "num_epochs": 100,
    "experiment_name": "vae-mnist-baseline",
}

trainer = ArtifexWandBTrainer(
    model=model,
    config=config,
    wandb_project="artifex-experiments",
)

trainer.train(train_data, val_data)
```

---

## Best Practices

### DO

!!! success "Recommended Practices"
    ✅ **Log hyperparameters** at the start of each run

    ✅ **Use meaningful run names** for easy identification

    ✅ **Tag runs** with experiment type (baseline, ablation, etc.)

    ✅ **Save artifacts** for reproducibility

    ✅ **Create reports** for team sharing

    ✅ **Use sweeps** for systematic hyperparameter search

### DON'T

!!! danger "Avoid These Mistakes"
    ❌ **Don't log too frequently** (causes overhead)

    ❌ **Don't hardcode API keys** (use environment variables)

    ❌ **Don't skip run names** (default names are hard to track)

    ❌ **Don't log high-resolution images** every step (use subsampling)

    ❌ **Don't forget to call** `wandb.finish()`

    ❌ **Don't create too many artifacts** (version strategically)

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Slow logging** | Logging too frequently | Reduce log frequency to every 100-1000 steps |
| **Missing metrics** | Not calling `wandb.log()` | Ensure metrics are logged in training loop |
| **Artifact upload fails** | Large file size | Use compression or split artifacts |
| **Sweep not starting** | Invalid config | Validate sweep config with W&B docs |
| **Run not appearing** | Network issues | Check internet connection, retry |
| **Memory leak** | Not finishing runs | Always call `wandb.finish()` |

---

## Summary

W&B integration provides powerful experiment tracking:

- **Metrics Logging**: Track training progress in real-time
- **Hyperparameter Sweeps**: Automate optimization
- **Artifacts**: Version models and datasets
- **Reports**: Share results with teammates
- **Visualization**: Interactive charts and galleries

Start tracking your experiments systematically!

---

## Next Steps

<div class="grid cards" markdown>

- :material-monitor-dashboard:{ .lg .middle } **TensorBoard**

    ---

    Alternative visualization with TensorBoard

    [:octicons-arrow-right-24: TensorBoard Guide](tensorboard.md)

- :material-cloud-upload:{ .lg .middle } **HuggingFace Hub**

    ---

    Share models with the community

    [:octicons-arrow-right-24: HuggingFace Integration](huggingface.md)

- :material-chart-line:{ .lg .middle } **Benchmarking**

    ---

    Evaluate models with comprehensive metrics

    [:octicons-arrow-right-24: Evaluation Framework](../../benchmarks/index.md)

- :material-cog:{ .lg .middle } **Training Guide**

    ---

    Learn advanced training techniques

    [:octicons-arrow-right-24: Training Guide](../training/training-guide.md)

</div>
