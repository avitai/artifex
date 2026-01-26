# Installation Guide

This guide provides comprehensive instructions for installing Artifex on various platforms and configurations.

## System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB (16GB recommended)
- **Disk Space**: 2GB for installation, 10GB+ for datasets
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10/11 (via WSL2)

### Recommended Requirements

- **Python**: 3.10 or 3.11 (tested and recommended)
- **RAM**: 16GB+ for training models
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for GPU acceleration)
  - Compute Capability 7.0+ (V100, RTX 20xx, RTX 30xx, RTX 40xx, A100)
  - CUDA 12.0 or higher
- **Disk Space**: 50GB+ for large-scale experiments

## Installation Options

Artifex provides multiple installation methods to suit different use cases:

=== "PyPI (Recommended for Users)"

    The simplest way to get started:

    ```bash
    # CPU-only installation
    pip install artifex

    # With GPU support
    pip install artifex[cuda]

    # With all extras (documentation, development tools)
    pip install artifex[all]
    ```

    **Note**: PyPI package coming soon. For now, install from source.

=== "uv (Recommended for Developers)"

    [uv](https://github.com/astral-sh/uv) is a fast Python package manager. Artifex provides an automated setup script that handles everything:

    **Quick Setup (Recommended)**:

    ```bash
    # Clone the repository
    git clone https://github.com/avitai/artifex.git
    cd artifex

    # Run unified setup script (auto-detects GPU)
    ./setup.sh

    # Activate environment
    source ./activate.sh
    ```

    **What `setup.sh` does**:

    - Installs `uv` package manager if not present
    - Detects GPU/CUDA availability automatically
    - Creates `.venv` virtual environment
    - Installs all dependencies (`uv sync --extra all` for GPU or `--extra dev` for CPU)
    - Creates `.env` file with:
        - CUDA library paths (GPU mode)
        - JAX platform configuration
        - Environment variables for optimal performance
    - Generates `activate.sh` script for easy activation
    - Verifies installation with JAX GPU tests

    **What `activate.sh` does**:

    - Activates the `.venv` virtual environment
    - Loads environment variables from `.env`
    - Configures CUDA paths (GPU mode)
    - Verifies JAX installation and GPU detection
    - Displays environment status and helpful commands

    **Setup Options**:

    ```bash
    # CPU-only setup (skip GPU detection)
    ./setup.sh --cpu-only

    # Clean setup (removes caches)
    ./setup.sh --deep-clean

    # Force reinstall
    ./setup.sh --force

    # Verbose output
    ./setup.sh --verbose
    ```

    **Manual Setup (Advanced)**:

    ```bash
    # Install uv if needed
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create virtual environment
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install Artifex with all dependencies
    uv sync --all-extras

    # Or install specific extras
    uv sync --extra cuda-dev  # CUDA development environment
    uv sync --extra dev       # Development tools only
    ```

=== "pip (Traditional)"

    Using standard pip with Artifex's setup script:

    **Quick Setup with Scripts**:

    ```bash
    # Clone the repository
    git clone https://github.com/avitai/artifex.git
    cd artifex

    # Run setup script (installs uv automatically)
    ./setup.sh

    # Activate environment
    source ./activate.sh
    ```

    The setup script works with both `uv` and `pip`. It will:
    - Auto-detect and install `uv` if needed
    - Create virtual environment and configure CUDA
    - Generate activation script with proper environment setup

    **Manual pip Installation**:

    ```bash
    # Clone the repository
    git clone https://github.com/avitai/artifex.git
    cd artifex

    # Create and activate virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install in development mode
    pip install -e .

    # Or with extras
    pip install -e '.[dev]'        # Development tools
    pip install -e '.[cuda]'       # CUDA support
    pip install -e '.[all]'        # Everything

    # For GPU support, manually configure environment:
    export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.*/site-packages/nvidia/*/lib:$LD_LIBRARY_PATH
    export JAX_PLATFORMS="cuda,cpu"
    export XLA_PYTHON_CLIENT_PREALLOCATE="false"
    ```

    **Note**: Using `./setup.sh` is recommended even for pip users as it properly configures CUDA paths and environment variables automatically.

=== "Docker"

    For containerized deployment:

    ```bash
    # Pull the latest image
    docker pull ghcr.io/avitai/artifex:latest

    # Run with GPU support
    docker run --gpus all -it ghcr.io/avitai/artifex:latest

    # Run with volume mount for data
    docker run --gpus all -v $(pwd)/data:/workspace/data \
      -it ghcr.io/avitai/artifex:latest

    # Or build locally
    docker build -t artifex:local .
    docker run --gpus all -it artifex:local
    ```

## Local Virtual Environment Setup

Artifex's `setup.sh` and `activate.sh` scripts provide automated local development environment configuration with GPU support.

!!! tip "Need Hardware-Specific Configuration?"
    For detailed information on customizing the setup for different GPUs (NVIDIA, AMD), TPUs, Apple Silicon, or multi-GPU systems, see the **[Hardware Setup Guide](hardware-setup-guide.md)**. This comprehensive guide explains how `setup.sh` and `.env.example` work and how to customize them for your specific hardware.

### Understanding the Setup Process

When you run `./setup.sh`, it creates a complete, self-contained development environment:

**Files Created:**

```
artifex/
├── .venv/                    # Virtual environment
│   ├── bin/activate         # Standard venv activation
│   ├── lib/python3.*/       # Python packages
│   └── ...
├── .env                      # Environment configuration
├── activate.sh              # Unified activation script
└── uv.lock                  # Dependency lock file
```

**The `.env` File (GPU Mode)**:

```bash
# CUDA library paths (from local venv installation)
export LD_LIBRARY_PATH=".venv/lib/python3.*/site-packages/nvidia/*/lib:$LD_LIBRARY_PATH"

# JAX GPU configuration
export JAX_PLATFORMS="cuda,cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.8"

# Project paths
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
export PYTEST_CUDA_ENABLED="true"
```

**The `.env` File (CPU Mode)**:

```bash
# JAX CPU configuration
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64="0"

# Project paths
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
export PYTEST_CUDA_ENABLED="false"
```

### Using the Environment

**First Time Setup:**

```bash
# Run setup once
./setup.sh

# Activate environment
source ./activate.sh
```

**Daily Workflow:**

```bash
# Simply activate the environment
source ./activate.sh

# Your environment is now ready with:
# - Virtual environment activated
# - CUDA paths configured (if GPU)
# - JAX optimally configured
# - Project in PYTHONPATH

# Work on your code...
uv run pytest tests/
python your_script.py

# Deactivate when done
deactivate
```

### Activation Script Features

The `activate.sh` script provides:

1. **Smart Process Detection**: Checks for running processes before deactivation
2. **GPU Verification**: Tests GPU availability and displays device info
3. **Environment Status**: Shows Python version, JAX backend, available devices
4. **Helpful Commands**: Displays common development commands
5. **Error Handling**: Provides clear messages if setup is incomplete

**Example activation output:**

```
🚀 Activating Artifex Development Environment
=============================================
✅ Virtual environment activated
✅ Environment configuration loaded
   🎮 GPU Mode: CUDA enabled

🔍 Environment Status:
   Python: Python 3.11.5
   Virtual Environment: /path/to/artifex/.venv

🧪 JAX Configuration:
   JAX version: 0.4.35
   Default backend: gpu
   Available devices: 1 total
   🎉 GPU devices: 1 (['cuda(id=0)'])
   ✅ CUDA acceleration ready!

🚀 Ready for Development!
```

### Setup Script Options

Customize setup for different scenarios:

```bash
# Standard setup (auto-detects GPU)
./setup.sh

# CPU-only setup (laptop/CI)
./setup.sh --cpu-only

# Clean installation (remove all caches)
./setup.sh --deep-clean

# Force reinstall over existing environment
./setup.sh --force

# Verbose output for debugging
./setup.sh --verbose

# Combine options
./setup.sh --force --deep-clean --verbose
```

### Troubleshooting Local Setup

**Problem**: `./setup.sh: Permission denied`

```bash
# Make script executable
chmod +x setup.sh activate.sh
./setup.sh
```

**Problem**: GPU not detected after setup

```bash
# Check NVIDIA drivers
nvidia-smi

# Re-run setup with force
./setup.sh --force

# Check activation output for GPU status
source ./activate.sh
```

**Problem**: Environment variables not loaded

```bash
# Verify .env file exists
cat .env

# Use 'source' not 'bash'
source ./activate.sh  # ✅ Correct
bash activate.sh       # ❌ Won't load environment
```

## GPU Setup

### CUDA Installation

Artifex requires CUDA 12.0+ for GPU acceleration.

#### Linux (Ubuntu/Debian)

```bash
# Check if CUDA is already installed
nvcc --version
nvidia-smi

# If not installed, download CUDA Toolkit 12.0+
# Visit: https://developer.nvidia.com/cuda-downloads

# Example for Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-0

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### macOS

CUDA is not supported on macOS. Use Metal backend (experimental) or CPU mode.

#### Windows (WSL2)

Windows users should use WSL2 with Ubuntu:

```powershell
# In PowerShell (Admin)
wsl --install

# Follow Ubuntu installation steps inside WSL2
```

### Automated GPU Setup (Linux)

Artifex provides automated CUDA setup through the main setup script:

```bash
# Complete environment setup with CUDA (recommended)
./setup.sh

# This automatically:
# - Detects GPU and CUDA availability
# - Installs JAX with CUDA support
# - Configures CUDA library paths
# - Sets up JAX environment variables
# - Creates activation script with GPU verification
# - Tests GPU functionality

# For CPU-only systems:
./setup.sh --cpu-only
```

**What gets configured automatically:**

- `LD_LIBRARY_PATH`: Points to CUDA libraries in `.venv/lib/python3.*/site-packages/nvidia/*/lib`
- `JAX_PLATFORMS`: Set to `"cuda,cpu"` for GPU or `"cpu"` for CPU-only
- `XLA_PYTHON_CLIENT_PREALLOCATE`: Disabled for dynamic memory allocation
- `XLA_PYTHON_CLIENT_MEM_FRACTION`: Set to 0.8 (use 80% of GPU memory)
- JAX CUDA plugins: Automatically matched to JAX version

See [Local Virtual Environment Setup](#local-virtual-environment-setup) for detailed explanation.

### JAX GPU Installation

After CUDA is installed:

```bash
# Install JAX with CUDA support
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify GPU is detected
python -c "import jax; print(jax.devices())"
# Should print: [cuda(id=0)] or similar
```

### Troubleshooting GPU Installation

**Problem**: `jax.devices()` returns `[cpu(id=0)]` instead of GPU

**Solutions**:

1. **Set LD_LIBRARY_PATH**:

   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. **Use the setup script**:

   ```bash
   ./scripts/fresh_cuda_setup.sh
   ```

3. **Reinstall JAX with CUDA**:

   ```bash
   pip uninstall jax jaxlib
   pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

4. **Check CUDA installation**:

   ```bash
   nvcc --version          # Should show CUDA version
   nvidia-smi              # Should show GPU info
   python -c "import jax; print(jax.lib.xla_bridge.get_backend().platform)"  # Should print 'gpu'
   ```

**Problem**: CUDA out of memory

**Solutions**:

- Reduce batch size
- Enable mixed precision training (BF16/FP16)
- Use gradient accumulation
- Clear GPU cache: `jax.clear_caches()`

**Problem**: Slow training on GPU

**Solutions**:

- Ensure XLA is enabled (automatic with JAX)
- Use JIT compilation: `@jax.jit`
- Check GPU utilization: `nvidia-smi dmon`
- Increase batch size for better GPU utilization

## TPU Setup

For Google Cloud TPU:

```bash
# Install JAX with TPU support
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Verify TPU is detected
python -c "import jax; print(jax.devices())"
# Should print TPU devices
```

## Verification

After installation, verify your setup:

```python
# test_installation.py
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.models.vae import VAE
from artifex.generative_models.core.configuration import (
    VAEConfig,
    EncoderConfig,
    DecoderConfig,
)

print("JAX version:", jax.__version__)
print("JAX backend:", jax.default_backend())
print("Available devices:", jax.devices())

# Test simple computation
x = jnp.array([1.0, 2.0, 3.0])
print("JAX computation test:", jnp.sum(x))

# Test Artifex imports
encoder = EncoderConfig(
    name="test_encoder",
    input_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(64, 128),
    activation="relu",
)
decoder = DecoderConfig(
    name="test_decoder",
    latent_dim=32,
    output_shape=(28, 28, 1),
    hidden_dims=(128, 64),
    activation="relu",
)
config = VAEConfig(
    name="test_vae",
    encoder=encoder,
    decoder=decoder,
    encoder_type="dense",
    kl_weight=1.0,
)
rngs = nnx.Rngs(0)
model = VAE(config, rngs=rngs)
print("Artifex model created successfully!")

# Test forward pass (model uses internal RNGs)
batch = jax.random.normal(jax.random.PRNGKey(0), (4, 28, 28, 1))
outputs = model(batch)
print("Forward pass successful!")
print("Output keys:", list(outputs.keys()))
```

Run the verification:

```bash
python test_installation.py
```

Expected output:

```console
JAX version: 0.8.2
JAX backend: gpu (or cpu)
Available devices: [CudaDevice(id=0)] (or [CpuDevice(id=0)])
JAX computation test: 6.0
Artifex model created successfully!
Forward pass successful!
Output keys: ['reconstructed', 'mean', 'log_var', 'z']
```

## Development Installation

For contributing to Artifex:

**Quick Development Setup:**

```bash
# Clone repository
git clone https://github.com/avitai/artifex.git
cd artifex

# Run setup script (includes dev tools)
./setup.sh

# Activate environment
source ./activate.sh

# Install pre-commit hooks
uv run pre-commit install

# Verify development setup
uv run pytest tests/ -x              # Run tests
uv run ruff check src/               # Linting
uv run ruff format src/              # Formatting
uv run pyright src/                  # Type checking
```

**What's included in development setup:**

- All core dependencies (`uv sync --extra all` or `--extra dev`)
- Development tools: pytest, ruff, pyright, pre-commit
- GPU support (if available)
- Documentation tools (mkdocs, mkdocstrings)
- Benchmarking utilities
- Test coverage tools

**Daily development workflow:**

```bash
# Start your work session
source ./activate.sh

# Make changes to code
# ...

# Run tests before committing
uv run pytest tests/ -x

# Run pre-commit checks
uv run pre-commit run --all-files

# Commit your changes
git add .
git commit -m "Your commit message"

# Deactivate when done
deactivate
```

## Platform-Specific Notes

### Linux

- **Recommended**: Ubuntu 20.04 LTS or Ubuntu 22.04 LTS
- Ensure NVIDIA drivers are up-to-date: `sudo ubuntu-drivers autoinstall`
- For multi-GPU: Set `CUDA_VISIBLE_DEVICES="0,1"`

### macOS

- **Apple Silicon (M1/M2/M3)**: JAX has experimental support

  ```bash
  pip install jax-metal  # For Apple M1/M2/M3
  ```

- **Intel Macs**: CPU-only mode is fully supported
- XCode Command Line Tools required: `xcode-select --install`

### Windows

- **WSL2 Required**: Native Windows support is limited
- Install Ubuntu 22.04 from Microsoft Store
- Follow Linux installation inside WSL2
- GPU support requires Windows 11 + CUDA in WSL2

## Environment Variables

Configure Artifex behavior with environment variables:

```bash
# JAX Configuration
export JAX_PLATFORMS=gpu              # or 'cpu', 'tpu'
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Disable memory preallocation
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75  # Use 75% of GPU memory

# Artifex Configuration
export ARTIFEX_CACHE_DIR=~/.cache/artifex  # Cache directory
export ARTIFEX_DATA_DIR=~/artifex_data     # Data directory
export ARTIFEX_LOG_LEVEL=INFO               # Logging level

# Development
export ENABLE_BLACKJAX_TESTS=1      # Enable expensive BlackJAX tests
```

Add to `~/.bashrc` or `~/.zshrc` for persistence.

## Common Issues

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'artifex'`

**Solutions**:

```bash
# Verify installation
pip list | grep artifex

# Reinstall
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### Version Conflicts

**Problem**: Dependency version conflicts

**Solutions**:

```bash
# Clean install
deactivate
rm -rf .venv uv.lock
uv venv
source .venv/bin/activate
uv sync --all-extras
```

### Memory Issues

**Problem**: Out of memory during installation

**Solutions**:

```bash
# Increase pip timeout and disable parallel builds
pip install --no-cache-dir -e .

# Or use uv which is more memory efficient
uv sync --all-extras
```

## Updating Artifex

Keep Artifex up-to-date:

```bash
# From PyPI (when available)
pip install --upgrade artifex

# From source
cd artifex
git pull origin main
uv sync --all-extras  # or: pip install -e .
```

## Uninstallation

Remove Artifex completely:

```bash
# If installed from PyPI
pip uninstall artifex

# If installed from source
pip uninstall artifex

# Remove cache and data (optional)
rm -rf ~/.cache/artifex
rm -rf ~/artifex_data

# Remove virtual environment
deactivate
rm -rf .venv
```

## Next Steps

After successful installation:

1. **Quick Start**: Follow the [Quickstart Guide](quickstart.md) to train your first model
2. **Core Concepts**: Learn about [Artifex architecture](core-concepts.md)
3. **Examples**: Explore ready-to-run [Examples](../examples/index.md)

## Getting Help

If you encounter issues:

- **Documentation**: Check this guide and the [documentation index](../index.md)
- **GitHub Issues**: [Report bugs or request features](https://github.com/avitai/artifex/issues)
- **Discussions**: [Ask questions](https://github.com/avitai/artifex/discussions)
- **Discord**: Join our community (coming soon)

## Hardware Recommendations

### For Research/Development

- **CPU**: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 16-32GB
- **GPU**: NVIDIA RTX 3060 (12GB) or better
- **Storage**: SSD with 100GB+ free space

### For Production

- **CPU**: 16+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 64GB+
- **GPU**: NVIDIA A100 (40/80GB) or H100
- **Storage**: NVMe SSD with 500GB+ free space

### For Large-Scale Training

- **Multi-GPU**: 4-8x NVIDIA A100 or H100
- **RAM**: 256GB+
- **Network**: High-speed interconnect (NVLink, InfiniBand)
- **Storage**: Parallel file system (Lustre, GPFS)

---

**Last Updated**: 2025-10-13

**Installation Support**: For installation help, open an [issue](https://github.com/avitai/artifex/issues).
