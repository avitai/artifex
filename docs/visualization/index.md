# Visualization

Artifex provides visualization tools for analyzing generative model outputs, training progress, latent spaces, and benchmark results. These tools help you understand model behavior and communicate results effectively.

## Overview

<div class="grid cards" markdown>

- :material-view-grid:{ .lg .middle } **Sample Visualization**

    ---

    Display generated samples in organized grids for quality assessment

    [:octicons-arrow-right-24: Sample Grids](#sample-grids)

- :material-chart-scatter-plot:{ .lg .middle } **Latent Space**

    ---

    Visualize and analyze learned latent representations

    [:octicons-arrow-right-24: Latent Space](#latent-space-visualization)

- :material-chart-line:{ .lg .middle } **Training Metrics**

    ---

    Plot loss curves, metrics, and training progress

    [:octicons-arrow-right-24: Training Plots](#training-plots)

- :material-molecule:{ .lg .middle } **Protein Structures**

    ---

    3D visualization of protein structures and molecular data

    [:octicons-arrow-right-24: Protein Visualization](#protein-visualization)

</div>

## Quick Start

```python
from artifex.utils.visualization import (
    create_image_grid,
    plot_latent_space,
    plot_training_curves,
)

# Create a grid of generated samples
grid = create_image_grid(samples, nrow=8, padding=2)

# Visualize latent space with t-SNE
plot_latent_space(latents, labels=labels, method="tsne")

# Plot training progress
plot_training_curves(
    history,
    metrics=["loss", "reconstruction_loss", "kl_loss"],
)
```

---

## Sample Grids

Display generated samples in organized grids for visual inspection.

### Basic Grid

```python
from artifex.utils.visualization.image_grid import create_image_grid
import matplotlib.pyplot as plt

# Generate samples
samples = model.sample(num_samples=64, rngs=rngs)

# Create grid
grid = create_image_grid(
    samples,
    nrow=8,           # Images per row
    padding=2,        # Padding between images
    normalize=True,   # Normalize to [0, 1]
)

# Display
plt.figure(figsize=(12, 12))
plt.imshow(grid)
plt.axis("off")
plt.title("Generated Samples")
plt.savefig("samples.png", dpi=150, bbox_inches="tight")
```

### Comparison Grid

```python
from artifex.utils.visualization.image_grid import create_comparison_grid

# Create comparison between models
comparison = create_comparison_grid(
    samples_dict={
        "VAE": vae_samples,
        "GAN": gan_samples,
        "Diffusion": diffusion_samples,
    },
    nrow=4,
)

plt.figure(figsize=(15, 10))
plt.imshow(comparison)
plt.axis("off")
plt.savefig("model_comparison.png", dpi=150)
```

### Reconstruction Grid

```python
from artifex.utils.visualization.image_grid import create_reconstruction_grid

# Show original vs reconstructed
recon_grid = create_reconstruction_grid(
    original=test_images[:16],
    reconstructed=model.reconstruct(test_images[:16]),
    nrow=4,
)

plt.figure(figsize=(10, 5))
plt.imshow(recon_grid)
plt.axis("off")
plt.title("Original (top) vs Reconstructed (bottom)")
plt.savefig("reconstructions.png", dpi=150)
```

---

## Latent Space Visualization

Analyze learned latent representations through dimensionality reduction.

### t-SNE Visualization

```python
from artifex.utils.visualization.latent_space import plot_latent_tsne

# Encode images to latent space
latents = model.encode(images)["mean"]

# Visualize with t-SNE
fig = plot_latent_tsne(
    latents,
    labels=labels,
    perplexity=30,
    n_iter=1000,
    title="Latent Space (t-SNE)",
)
fig.savefig("latent_tsne.png", dpi=150)
```

### PCA Visualization

```python
from artifex.utils.visualization.latent_space import plot_latent_pca

# Visualize with PCA
fig = plot_latent_pca(
    latents,
    labels=labels,
    n_components=2,
    title="Latent Space (PCA)",
)
fig.savefig("latent_pca.png", dpi=150)
```

### Latent Traversal

```python
from artifex.utils.visualization.latent_space import plot_latent_traversal

# Traverse individual latent dimensions
traversal_grid = plot_latent_traversal(
    model,
    base_latent=z,
    dimensions=[0, 1, 2, 3],  # Dimensions to traverse
    range_vals=(-3, 3),
    num_steps=10,
)

plt.figure(figsize=(12, 6))
plt.imshow(traversal_grid)
plt.xlabel("Latent Value")
plt.ylabel("Dimension")
plt.title("Latent Dimension Traversal")
plt.savefig("latent_traversal.png", dpi=150)
```

### Interpolation

```python
from artifex.utils.visualization.latent_space import plot_interpolation

# Interpolate between two samples
interp_grid = plot_interpolation(
    model,
    start_image=image1,
    end_image=image2,
    num_steps=10,
    method="slerp",  # or "linear"
)

plt.figure(figsize=(15, 2))
plt.imshow(interp_grid)
plt.axis("off")
plt.title("Latent Space Interpolation")
plt.savefig("interpolation.png", dpi=150)
```

---

## Training Plots

Visualize training progress and metrics.

### Loss Curves

```python
from artifex.utils.visualization.plotting import plot_losses

# Plot training and validation losses
fig = plot_losses(
    train_losses=history["train_loss"],
    val_losses=history["val_loss"],
    title="Training Progress",
)
fig.savefig("loss_curves.png", dpi=150)
```

### Multi-Metric Plot

```python
from artifex.utils.visualization.plotting import plot_metrics

# Plot multiple metrics
fig = plot_metrics(
    history,
    metrics=["loss", "reconstruction_loss", "kl_loss", "fid"],
    smooth=True,
    window=10,
)
fig.savefig("training_metrics.png", dpi=150)
```

### Learning Rate Schedule

```python
from artifex.utils.visualization.plotting import plot_lr_schedule

# Visualize learning rate over training
fig = plot_lr_schedule(
    lr_history=history["learning_rate"],
    title="Learning Rate Schedule",
)
fig.savefig("lr_schedule.png", dpi=150)
```

---

## Benchmark Visualization

Visualize and compare benchmark results.

### Metric Comparison

```python
from artifex.benchmarks.visualization.comparison import plot_model_comparison

# Compare models on multiple metrics
fig = plot_model_comparison(
    results={
        "VAE": {"fid": 45.2, "is": 8.1, "lpips": 0.12},
        "GAN": {"fid": 32.1, "is": 9.2, "lpips": 0.08},
        "Diffusion": {"fid": 18.5, "is": 10.1, "lpips": 0.05},
    },
    metrics=["fid", "is", "lpips"],
)
fig.savefig("model_comparison.png", dpi=150)
```

### Radar Chart

```python
from artifex.benchmarks.visualization.plots import plot_radar

# Create radar chart for model comparison
fig = plot_radar(
    models=["VAE", "GAN", "Diffusion"],
    metrics={
        "Quality": [0.7, 0.85, 0.95],
        "Diversity": [0.9, 0.75, 0.88],
        "Speed": [0.95, 0.8, 0.3],
        "Stability": [0.95, 0.6, 0.9],
    },
)
fig.savefig("radar_comparison.png", dpi=150)
```

### FID Over Training

```python
from artifex.benchmarks.visualization.plots import plot_fid_progression

# Track FID during training
fig = plot_fid_progression(
    fid_values=history["fid"],
    steps=history["step"],
    title="FID Score During Training",
)
fig.savefig("fid_progression.png", dpi=150)
```

---

## Protein Visualization

Specialized visualization for protein structures.

### 3D Structure Plot

```python
from artifex.utils.visualization.protein_viz import plot_protein_structure

# Visualize protein backbone
fig = plot_protein_structure(
    coordinates=protein_coords,  # (num_residues, 4, 3) for N, CA, C, O
    color_by="residue",          # or "chain", "secondary_structure"
    show_bonds=True,
)
fig.savefig("protein_structure.png", dpi=150)
```

### Ramachandran Plot

```python
from artifex.utils.visualization.protein_viz import plot_ramachandran

# Plot backbone dihedral angles
fig = plot_ramachandran(
    phi_angles=phi,
    psi_angles=psi,
    title="Ramachandran Plot",
)
fig.savefig("ramachandran.png", dpi=150)
```

### Bond Length Distribution

```python
from artifex.utils.visualization.protein_viz import plot_bond_distributions

# Visualize bond length/angle distributions
fig = plot_bond_distributions(
    generated_proteins=generated_coords,
    reference_proteins=real_coords,
    metrics=["bond_length", "bond_angle"],
)
fig.savefig("bond_distributions.png", dpi=150)
```

### Documentation

- [Protein Visualization](protein_viz.md) - Full protein visualization API

---

## Interactive Dashboard

Create interactive dashboards for model analysis.

```python
from artifex.benchmarks.visualization.dashboard import create_dashboard

# Create interactive dashboard
dashboard = create_dashboard(
    model=model,
    dataset=test_dataset,
    metrics=["fid", "is", "reconstruction"],
)

# Launch in browser
dashboard.run(port=8050)
```

The dashboard provides:

- Real-time sample generation
- Latent space exploration
- Metric tracking
- Model comparison tools

---

## Saving and Exporting

### High-Quality Export

```python
from artifex.utils.visualization import save_figure

# Save with publication-quality settings
save_figure(
    fig,
    path="figure.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)
```

### Animation Export

```python
from artifex.utils.visualization import create_animation

# Create training animation
anim = create_animation(
    frames=sample_frames,  # List of sample grids over training
    fps=10,
    title="Training Progress",
)
anim.save("training.gif", writer="pillow")
```

---

## Configuration

### Style Settings

```python
from artifex.utils.visualization import set_style

# Set consistent plotting style
set_style(
    style="seaborn",
    font_scale=1.2,
    palette="viridis",
)
```

### Figure Defaults

```python
from artifex.utils.visualization import configure_defaults

configure_defaults(
    figsize=(10, 8),
    dpi=150,
    colormap="plasma",
    grid=True,
)
```

---

## Best Practices

!!! success "DO"
    - Use consistent color schemes across plots
    - Include axis labels and titles
    - Save figures in vector format (PDF, SVG) for publications
    - Use appropriate resolution for the output medium

!!! danger "DON'T"
    - Don't use rainbow colormaps for sequential data
    - Don't overcrowd plots with too many elements
    - Don't forget to normalize images before display
    - Don't use low resolution for print materials

---

## Summary

Artifex visualization tools provide:

- **Sample Grids**: Display and compare generated samples
- **Latent Space**: t-SNE, PCA, traversals, and interpolation
- **Training Plots**: Loss curves, metrics, learning rate schedules
- **Benchmarks**: Model comparisons, radar charts, progression plots
- **Protein Viz**: 3D structures, Ramachandran plots, distributions
- **Dashboard**: Interactive exploration and analysis

Use these tools to understand model behavior, diagnose issues, and communicate results effectively.
