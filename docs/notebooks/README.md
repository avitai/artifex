# Notebooks

This directory contains Jupyter notebooks with examples, tutorials, and demonstrations of the Artifex library features.

## Available Notebooks

### Protein Structure Generation

- `protein_diffusion.ipynb`: Implementation of a diffusion model for protein structure generation using JAX and Flax.
  - Demonstrates how to generate 3D protein structures
  - Shows integration with protein-specific constraints
  - Includes visualization of generated structures

### New Model Examples

Artifex now includes comprehensive examples for new model types:

- **Energy-Based Models**: See `examples/` directory for EBM training and sampling examples
- **Advanced MCMC Sampling**: Langevin dynamics and persistent contrastive divergence
- **GPU Optimization**: Matrix multiplication fixes and CUDA setup examples

## Running the Notebooks

To run these notebooks locally:

1. Ensure you have installed Artifex with development dependencies:

   ```bash
   # Recommended: Use the cuda-dev environment for full GPU support
   uv sync --extra cuda-dev

   # Or just development dependencies
   pip install -e ".[dev]"
   ```

2. Set up your CUDA environment (for GPU examples):

   ```bash
   ./scripts/fresh_cuda_setup.sh
   ```

3. Launch Jupyter:

   ```bash
   jupyter lab
   ```

   or

   ```bash
   jupyter notebook
   ```

4. Navigate to the notebook you want to run.

## Notebook Structure

Each notebook generally follows this structure:

1. **Introduction and Setup**: Overview of the example and necessary imports
2. **Data Preparation**: Loading and preprocessing data
3. **Model Configuration**: Setting up the model architecture
4. **Training**: Training the model on the data
5. **Evaluation**: Evaluating model performance
6. **Generation**: Using the model to generate new samples
7. **Visualization**: Visualizing the results

## Adding New Notebooks

When adding new notebooks, please:

1. Ensure all dependencies are documented
2. Include comprehensive explanations with markdown cells
3. Add the notebook to this README
4. Run `pre-commit` to ensure the notebook is properly formatted and stripped of outputs
