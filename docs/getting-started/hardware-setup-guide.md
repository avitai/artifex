# Hardware Setup Guide: Customizing for Your GPU/TPU

This comprehensive guide explains how Artifex's setup system works and how to customize it for different hardware configurations including various NVIDIA GPUs, AMD GPUs, Apple Silicon, and TPUs.

## Table of Contents

- [Overview](#overview)
- [Architecture & Philosophy](#architecture-philosophy)
- [The Setup Pipeline](#the-setup-pipeline)
- [Customizing for Different Hardware](#customizing-for-different-hardware)
- [Multi-GPU Configurations](#multi-gpu-configurations)
- [Memory Management](#memory-management)
- [TPU Setup](#tpu-setup)
- [AMD GPU Setup (ROCm)](#amd-gpu-setup-rocm)
- [Apple Silicon (M1/M2/M3)](#apple-silicon-m1m2m3)
- [Troubleshooting](#troubleshooting)

## Overview

Artifex uses a three-component setup system:

1. **`setup.sh`**: Main orchestration script that detects hardware and creates the environment
2. **`.env.example`**: Template for `.env` file with GPU-specific configurations
3. **`activate.sh`**: Generated activation script that loads the environment

### Quick Facts

- **Hardware Agnostic**: Despite being tested on RTX 4090, the setup is designed for ANY NVIDIA GPU
- **No Hardware-Specific Code**: The system auto-detects and configures for your hardware
- **CUDA Version Independent**: Uses CUDA libraries installed in your virtual environment
- **Extensible**: Easy to customize for TPUs, AMD GPUs, or custom hardware

## Architecture Philosophy

### Design Principles

#### 1. Virtual Environment Isolation

All CUDA libraries are installed **inside** the virtual environment, not system-wide:

```
.venv/
└── lib/
    └── python3.*/
        └── site-packages/
            └── nvidia/
                ├── cublas/lib/
                ├── cudnn/lib/
                ├── cufft/lib/
                └── ... (all CUDA libraries)
```

**Why?** This allows:

- Multiple projects with different CUDA versions
- No system-wide CUDA installation required
- Clean, reproducible environments
- Easy cleanup (just delete `.venv/`)

#### 2. Dynamic Hardware Detection

The setup script detects your hardware at runtime:

```bash
# From setup.sh
detect_cuda_support() {
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$gpu_info" ]; then
            echo "✅ NVIDIA GPU detected: $gpu_info"
            return 0
        fi
    fi
    return 1  # No GPU found
}
```

#### 3. Template-Based Configuration

The `.env.example` provides a base configuration that gets customized during setup:

- **Generic Base**: Works for any NVIDIA GPU
- **Customizable**: Easy to modify for specific needs
- **Version-Independent**: Auto-detects Python version

## The Setup Pipeline

### Step-by-Step Execution

When you run `./setup.sh`, here's what happens:

#### 1. Pre-flight Checks

```bash
# Checks and installs uv if needed
ensure_uv_installed()

# Detects GPU hardware
detect_cuda_support()  # Returns true for GPU, false for CPU
```

#### 2. Environment Cleaning (if needed)

```bash
perform_cleaning()
# - Removes old .venv/
# - Clears caches
# - Removes old config files
```

#### 3. Environment File Creation

```bash
create_env_file() {
    local has_cuda=$1  # true or false from detection

    if [ "$has_cuda" = true ]; then
        # Use .env.example if it exists
        if [ -f ".env.example" ]; then
            # Copy and customize template
            sed "s|PROJECT_DIR=\"\$(pwd)\"|PROJECT_DIR=\"$(pwd)\"|g" \
                .env.example > .env
        else
            # Use embedded fallback template
            # (same as .env.example)
        fi
    else
        # Create CPU-only .env
        # (simpler configuration)
    fi
}
```

**Key Point**: The template is **hardware-agnostic**. It doesn't know about RTX 4090 vs A100 vs V100.

#### 4. Virtual Environment Creation

```bash
setup_environment() {
    uv venv
    source .venv/bin/activate
    source .env

    if [ "$has_cuda" = true ]; then
        # Install GPU packages
        uv sync --extra all

        # Install matching CUDA plugins for JAX
        JAX_VERSION=$(python -c "import jax; print(jax.__version__)")
        uv pip install --force-reinstall \
            "jax-cuda12-pjrt==$JAX_VERSION" \
            "jax-cuda12-plugin==$JAX_VERSION"
    else
        # CPU-only installation
        uv sync --extra dev
    fi
}
```

#### 5. Activation Script Generation

```bash
create_activation_script() {
    cat > activate.sh << 'EOF'
#!/bin/bash
# 1. Activates .venv
# 2. Sources .env (loads environment variables)
# 3. Runs JAX verification tests
# 4. Displays system information
EOF
    chmod +x activate.sh
}
```

#### 6. Verification

```bash
verify_installation() {
    # Tests JAX import
    # Tests GPU detection (if applicable)
    # Runs simple computations
}
```

## Understanding `.env.example`

### Template Structure

The template has several key sections:

#### Section 1: Project Path Detection

```bash
# Use absolute path for the project directory (will be replaced during setup)
PROJECT_DIR="/path/to/artifex"  # ← Replaced by setup.sh with actual path

# Dynamically detect Python version
if [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
    PYTHON_VERSION=$("${PROJECT_DIR}/.venv/bin/python" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
    # ... fallback detection
fi
VENV_CUDA_BASE="${PROJECT_DIR}/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia"
```

**Purpose**: Locate CUDA libraries in the virtual environment, regardless of Python version.

#### Section 2: Path Filtering

```bash
# Filter out old CUDA paths from existing LD_LIBRARY_PATH
if [ -n "$LD_LIBRARY_PATH" ]; then
    FILTERED_LD_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | \
        grep -v -E '(nvidia|cuda|cudnn|nccl|...)' | \
        tr '\n' ':' | sed 's/:$//')
fi
```

**Purpose**: Remove system-wide CUDA paths to avoid version conflicts. Preserve non-CUDA paths.

#### Section 3: CUDA Library Paths

```bash
# Include ALL CUDA libraries installed by jax-cuda12-plugin
NEW_CUDA_PATHS="${VENV_CUDA_BASE}/cublas/lib:${VENV_CUDA_BASE}/cusolver/lib:..."

export LD_LIBRARY_PATH="${NEW_CUDA_PATHS}:${FILTERED_LD_PATH}"
export CUDA_HOME="${VENV_CUDA_BASE}"
export PATH="${VENV_CUDA_BASE}/cuda_nvcc/bin:${PATH}"
```

**Purpose**: Make JAX use CUDA libraries from venv, not system.

#### Section 4: JAX Configuration

```bash
export JAX_PLATFORMS="cuda,cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
```

**Purpose**: Configure JAX for optimal GPU performance.

#### Section 5: Development Settings

```bash
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_DIR}"
export PYTEST_CUDA_ENABLED="true"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
```

**Purpose**: Project-specific development configuration.

### Why This Template Works for All NVIDIA GPUs

The template is **hardware-agnostic** because:

1. **No GPU-Specific Settings**: It doesn't configure compute capability, SM count, or GPU-specific features
2. **CUDA Auto-Detection**: JAX and XLA handle GPU-specific optimizations automatically
3. **Dynamic Library Loading**: Libraries are loaded based on what's installed, not hardcoded
4. **Memory Fraction**: Uses 90% by default, works on any GPU size

## Customizing for Different Hardware

### NVIDIA GPU Variations

The default template works for **all NVIDIA GPUs** (V100, A100, RTX 3090, RTX 4090, H100, etc.), but you can optimize for specific use cases.

#### Consumer GPUs (RTX 3060, 3090, 4090)

**Typical Configuration** (already optimal in template):

```bash
# For 8-24GB GPUs
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"
export CUDA_MODULE_LOADING="LAZY"
```

**For Lower Memory GPUs (6-8GB)**:

Edit `.env.example`:

```bash
# Conservative memory usage for RTX 3060, GTX 1660, etc.
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.7"  # Use 70% of GPU memory
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"  # Better memory management
export TF_FORCE_GPU_ALLOW_GROWTH="true"  # Gradual memory allocation
```

#### Data Center GPUs (V100, A100, H100)

**For A100 (40GB/80GB)**:

```bash
# Aggressive memory usage for large GPU
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"  # Use 95% of memory
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export NCCL_DEBUG="INFO"  # Enable NCCL logging for multi-GPU
export NCCL_IB_DISABLE="0"  # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL="5"  # Enable GPUDirect RDMA
```

**For H100**:

```bash
# H100-specific optimizations
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"
export XLA_FLAGS="--xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_latency_hiding_scheduler=true"
export CUDA_MODULE_LOADING="EAGER"  # H100 benefits from eager loading
export NCCL_PROTO="Simple"  # Optimal for H100 NVLink
```

#### Professional GPUs (Quadro, Tesla)

```bash
# Similar to consumer but may benefit from ECC
export CUDA_FORCE_PTX_JIT="1"  # Force PTX JIT compilation
export XLA_FLAGS="--xla_gpu_cuda_graph_level=3"  # Aggressive CUDA graph optimizations
```

### Custom Template Creation

Create a custom template for your hardware:

```bash
# 1. Copy the original template
cp .env.example .env.example.rtx3060

# 2. Edit for your GPU
nano .env.example.rtx3060

# 3. Modify the create_env_file function in setup.sh to use your template
# Edit setup.sh, line ~274:
if [ -f ".env.example.rtx3060" ]; then
    sed "s|PROJECT_DIR=\"\$(pwd)\"|PROJECT_DIR=\"$(pwd)\"|g" \
        .env.example.rtx3060 > .env
fi
```

### Example: RTX 3060 (12GB) Optimized Template

Create `.env.example.rtx3060`:

```bash
# Artifex Environment Configuration - RTX 3060 Optimized
# 12GB VRAM, consumer GPU optimizations

PROJECT_DIR="$(pwd)"

# [... Python version detection same as original ...]

# CUDA paths configuration
# [... same as original until JAX Configuration ...]

# JAX Configuration for RTX 3060 (12GB VRAM)
export JAX_PLATFORMS="cuda,cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.75"  # Conservative for 12GB
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"  # Better memory management
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

# Memory optimization for consumer GPU
export CUDA_MODULE_LOADING="LAZY"
export CUDA_CACHE_MAXSIZE="268435456"  # 256MB cache
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# JAX CUDA Plugin Configuration
export JAX_CUDA_PLUGIN_VERIFY="false"

# Reduce CUDA warnings
export TF_CPP_MIN_LOG_LEVEL="1"

# Performance settings
export JAX_ENABLE_X64="0"  # Keep 32-bit for better performance

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_DIR}"
export PYTEST_CUDA_ENABLED="true"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
```

## Multi-GPU Configurations

### Modifying for Multiple GPUs

#### 1. Update `.env.example` for Multi-GPU

Add to the template:

```bash
# Multi-GPU Configuration
# Set which GPUs to use (0,1,2,3 for 4 GPUs)
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Use all 4 GPUs
# Or: export CUDA_VISIBLE_DEVICES="0,1"  # Use only first 2 GPUs

# NCCL Configuration for Multi-GPU
export NCCL_DEBUG="INFO"  # Enable NCCL logging
export NCCL_DEBUG_SUBSYS="ALL"  # Debug all subsystems
export NCCL_IB_DISABLE="0"  # Enable InfiniBand (if available)
export NCCL_SOCKET_IFNAME="eth0"  # Network interface for NCCL
export NCCL_P2P_LEVEL="LOC"  # Enable peer-to-peer transfers

# XLA Multi-GPU Settings
export XLA_FLAGS="--xla_gpu_enable_async_all_reduce=true \
                  --xla_gpu_enable_async_all_gather=true \
                  --xla_gpu_all_reduce_combine_threshold_bytes=134217728 \
                  --xla_gpu_enable_nccl_comm_splitting=true"

# Memory per GPU (adjust based on total GPU memory)
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"  # 90% per GPU
```

#### 2. Modify `setup.sh` for Multi-GPU Detection

Add GPU counting to `detect_cuda_support()`:

```bash
detect_cuda_support() {
    if [ "$CPU_ONLY" = true ]; then
        return 1
    fi

    if command -v nvidia-smi &> /dev/null; then
        # Get all GPU names
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
        gpu_count=$(echo "$gpu_info" | wc -l)

        if [ -n "$gpu_info" ] && [ $gpu_count -gt 0 ]; then
            log_success "NVIDIA GPU(s) detected: $gpu_count GPU(s)"
            echo "$gpu_info" | nl -w2 -s'. '  # Number and list GPUs

            # Store GPU count for later use
            export DETECTED_GPU_COUNT=$gpu_count
            return 0
        fi
    fi

    return 1
}
```

#### 3. Dynamic Multi-GPU Configuration

Modify `create_env_file()` to use GPU count:

```bash
create_env_file() {
    local has_cuda=$1

    if [ "$has_cuda" = true ]; then
        # Copy template
        sed "s|PROJECT_DIR=\"\$(pwd)\"|PROJECT_DIR=\"$(pwd)\"|g" \
            .env.example > .env

        # Add multi-GPU settings if multiple GPUs detected
        if [ "${DETECTED_GPU_COUNT:-1}" -gt 1 ]; then
            cat >> .env << EOF

# Auto-detected Multi-GPU Configuration ($DETECTED_GPU_COUNT GPUs)
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((DETECTED_GPU_COUNT-1)))"
export NCCL_DEBUG="WARN"
export XLA_FLAGS="\${XLA_FLAGS} --xla_gpu_enable_async_all_reduce=true"
EOF
            log_success "Multi-GPU configuration added for $DETECTED_GPU_COUNT GPUs"
        fi
    fi
}
```

### Testing Multi-GPU Setup

After setup, verify multi-GPU:

```python
import jax
print(f"Number of devices: {len(jax.devices())}")
print(f"Devices: {jax.devices()}")

# Test multi-device computation
import jax.numpy as jnp
x = jnp.arange(1000)
# Data parallel computation across all GPUs
results = jax.pmap(lambda x: x ** 2)(x.reshape(len(jax.devices()), -1))
print(f"Results shape: {results.shape}")
```

## Memory Management

### Understanding Memory Fractions

`XLA_PYTHON_CLIENT_MEM_FRACTION` controls GPU memory allocation:

```bash
# Formula: usable_memory = total_memory * fraction
# RTX 4090 (24GB): 24GB * 0.9 = 21.6GB available to JAX
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"
```

### GPU-Specific Memory Recommendations

| GPU Model | VRAM | Recommended Fraction | Reasoning |
|-----------|------|---------------------|-----------|
| RTX 3060 | 12GB | 0.7-0.75 | Leave room for OS/display |
| RTX 3080 | 10GB | 0.75-0.8 | Same as above |
| RTX 3090 | 24GB | 0.85-0.9 | More memory available |
| RTX 4090 | 24GB | 0.85-0.9 | Plenty of memory |
| A100 (40GB) | 40GB | 0.9-0.95 | Server GPU, no display |
| A100 (80GB) | 80GB | 0.9-0.95 | Maximize usage |
| H100 | 80GB | 0.95 | Maximum usage |
| V100 (16GB) | 16GB | 0.8-0.85 | Balanced |
| V100 (32GB) | 32GB | 0.85-0.9 | More room |
| T4 | 16GB | 0.75-0.8 | Shared environments |

### Dynamic Memory Allocation

For variable workloads, use dynamic allocation:

```bash
# Add to .env.example
export XLA_PYTHON_CLIENT_PREALLOCATE="false"  # Don't preallocate
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"  # Use platform allocator
export TF_FORCE_GPU_ALLOW_GROWTH="true"  # Gradual allocation
```

### Out-of-Memory Handling

Add to your template:

```bash
# OOM handling
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.7"  # Conservative
export JAX_PLATFORMS="cuda,cpu"  # Fallback to CPU if GPU OOM
export JAX_DEBUG_NANS="false"  # Disable NaN checking to save memory
```

## TPU Setup

### Creating a TPU-Specific Template

Create `.env.example.tpu`:

```bash
# Artifex Environment Configuration - TPU Optimized
# Google Cloud TPU configuration

PROJECT_DIR="$(pwd)"

# JAX Configuration for TPU
export JAX_PLATFORMS="tpu,cpu"  # TPU first, CPU fallback
export TPU_CHIPS_PER_HOST="8"  # Standard TPU v3/v4 configuration
export TPU_NAME="local"  # Or your TPU name

# XLA TPU Flags
export XLA_FLAGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true \
                  --xla_tpu_data_parallel_opt_different_sized_ops=true \
                  --xla_tpu_enable_async_collective_fusion=true \
                  --xla_tpu_enable_async_collective_fusion_multiple_steps=true"

# TPU Performance Settings
export JAX_ENABLE_X64="0"  # TPUs work better with 32-bit
export XLA_PYTHON_CLIENT_PREALLOCATE="true"  # TPUs benefit from preallocation
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_DIR}"
export PYTEST_CUDA_ENABLED="false"
export PYTEST_TPU_ENABLED="true"

# Reduce verbosity
export TF_CPP_MIN_LOG_LEVEL="1"
```

### Modifying `setup.sh` for TPU Detection

Add TPU detection:

```bash
# Add to setup.sh
detect_tpu_support() {
    if [ "$CPU_ONLY" = true ]; then
        return 1
    fi

    # Check for TPU via environment variable
    if [ -n "$TPU_NAME" ]; then
        log_success "TPU detected: $TPU_NAME"
        return 0
    fi

    # Check for TPU via gcloud
    if command -v gcloud &> /dev/null; then
        if gcloud compute tpus list 2>/dev/null | grep -q "READY"; then
            log_success "TPU available via gcloud"
            return 0
        fi
    fi

    log_info "No TPU detected"
    return 1
}
```

### TPU-Specific Installation

Modify `setup_environment()`:

```bash
setup_environment() {
    local has_cuda=$1
    local has_tpu=$2  # Add TPU parameter

    uv venv
    source .venv/bin/activate

    if [ "$has_tpu" = true ]; then
        log_info "Installing with TPU support..."
        uv sync --extra dev

        # Install JAX with TPU support
        pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

        log_success "TPU installation successful"
    elif [ "$has_cuda" = true ]; then
        # GPU installation (as before)
        # ...
    fi
}
```

### Using Custom TPU Template

```bash
# Run setup with TPU template
TPU_NAME="your-tpu-name" ./setup.sh

# Or modify setup.sh to detect and use TPU template
# In create_env_file():
if [ -f ".env.example.tpu" ]; then
    sed "s|PROJECT_DIR=\"\$(pwd)\"|PROJECT_DIR=\"$(pwd)\"|g" \
        .env.example.tpu > .env
fi
```

## AMD GPU Setup (ROCm)

### Creating AMD/ROCm Template

Create `.env.example.rocm`:

```bash
# Artifex Environment Configuration - AMD ROCm
# AMD GPU configuration using ROCm

PROJECT_DIR="$(pwd)"

# ROCm Configuration
export ROCM_PATH="/opt/rocm"  # Default ROCm installation
export HIP_VISIBLE_DEVICES="0"  # GPU to use

# Add ROCm to paths
export PATH="${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"

# JAX Configuration for ROCm
export JAX_PLATFORMS="rocm,cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"

# ROCm-specific settings
export HSA_OVERRIDE_GFX_VERSION="10.3.0"  # Adjust for your GPU
export GPU_DEVICE_ORDINAL="0"

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_DIR}"
export PYTEST_CUDA_ENABLED="false"
export PYTEST_ROCM_ENABLED="true"
```

### AMD GPU Detection

Add to `setup.sh`:

```bash
detect_rocm_support() {
    if [ "$CPU_ONLY" = true ]; then
        return 1
    fi

    if command -v rocm-smi &> /dev/null; then
        gpu_info=$(rocm-smi --showproductname 2>/dev/null | grep "Card series" | head -1)
        if [ -n "$gpu_info" ]; then
            log_success "AMD GPU detected: $gpu_info"
            return 0
        fi
    fi

    log_info "No AMD GPU detected"
    return 1
}
```

### ROCm Installation

```bash
setup_environment() {
    # ... (existing code)

    if [ "$has_rocm" = true ]; then
        log_info "Installing with ROCm support..."
        uv sync --extra dev

        # Install JAX with ROCm (if available)
        # Note: JAX ROCm support is experimental
        pip install jax-rocm

        log_success "ROCm installation complete"
    fi
}
```

## Apple Silicon (M1/M2/M3)

### Creating Apple Silicon Template

Create `.env.example.metal`:

```bash
# Artifex Environment Configuration - Apple Silicon
# M1/M2/M3 GPU configuration using Metal

PROJECT_DIR="$(pwd)"

# JAX Configuration for Metal
export JAX_PLATFORMS="METAL,cpu"  # Note: Experimental
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

# Metal Performance Settings
export JAX_ENABLE_X64="0"  # Metal works better with 32-bit

# Apple Neural Engine (if supported)
# export USE_ANE="1"  # Experimental

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_DIR}"
export PYTEST_CUDA_ENABLED="false"
export PYTEST_METAL_ENABLED="true"

# Reduce verbosity
export TF_CPP_MIN_LOG_LEVEL="1"
```

### Apple Silicon Detection

```bash
detect_apple_silicon() {
    if [ "$(uname)" = "Darwin" ]; then
        # Check for Apple Silicon
        if [ "$(uname -m)" = "arm64" ]; then
            chip_info=$(sysctl -n machdep.cpu.brand_string)
            log_success "Apple Silicon detected: $chip_info"
            return 0
        fi
    fi
    return 1
}
```

### Metal Installation

```bash
setup_environment() {
    # ... (existing code)

    if [ "$has_metal" = true ]; then
        log_info "Installing with Metal support..."
        uv sync --extra dev

        # Install JAX with Metal support (experimental)
        pip install jax-metal

        log_success "Metal installation complete"
    fi
}
```

## Advanced Customization Examples

### Example 1: Multi-Node Multi-GPU Cluster

For distributed training across multiple nodes:

```bash
# .env.example.cluster
# Multi-node configuration

PROJECT_DIR="$(pwd)"

# [... standard CUDA paths ...]

# Multi-Node Configuration
export MASTER_ADDR="192.168.1.100"  # Head node IP
export MASTER_PORT="29500"
export WORLD_SIZE="8"  # Total number of GPUs across all nodes
export NODE_RANK="0"  # This node's rank (0 for head, 1, 2, ... for workers)
export NPROC_PER_NODE="4"  # GPUs per node

# NCCL Multi-Node Settings
export NCCL_SOCKET_IFNAME="eth0"  # Network interface
export NCCL_IB_DISABLE="0"  # Enable InfiniBand
export NCCL_IB_HCA="mlx5_0"  # InfiniBand adapter
export NCCL_DEBUG="INFO"
export NCCL_ALGO="Ring"  # Or "Tree" depending on topology

# XLA Distributed Settings
export JAX_COORDINATOR_ADDRESS="${MASTER_ADDR}:${MASTER_PORT}"
export JAX_NUM_PROCESSES="${WORLD_SIZE}"
export JAX_PROCESS_INDEX="${NODE_RANK}"

# Memory settings for large scale
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"
export XLA_FLAGS="--xla_gpu_enable_async_all_reduce=true \
                  --xla_gpu_enable_async_all_gather=true \
                  --xla_gpu_all_reduce_combine_threshold_bytes=268435456"
```

### Example 2: Low-Power/Edge GPU Configuration

For Jetson or other edge devices:

```bash
# .env.example.edge
# Edge device configuration (Jetson, etc.)

PROJECT_DIR="$(pwd)"

# Conservative memory usage
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.6"  # Leave room for system
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"

# Power-efficient settings
export CUDA_MODULE_LOADING="LAZY"
export JAX_ENABLE_X64="0"  # 32-bit for efficiency
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Reduce parallelism to save resources
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

# Thermal management (if applicable)
# export CUDA_CACHE_DISABLE="1"  # Reduce disk I/O
```

### Example 3: Mixed Precision Training

For automatic mixed precision:

```bash
# Add to .env.example
# Mixed Precision Training Configuration
export JAX_ENABLE_X64="0"  # Force 32-bit
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_xla_runtime_executable=true"

# For BF16 on supported GPUs (Ampere and newer)
export JAX_DEFAULT_DTYPE_BITS="32"
export JAX_ENABLE_BFLOAT16="true"  # If using BF16 explicitly
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: GPU Not Detected After Setup

**Problem**: `activate.sh` shows CPU-only despite having GPU.

**Solution**:

```bash
# Check NVIDIA driver
nvidia-smi

# If driver is fine, check library paths
source .env
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep nvidia

# Verify JAX can see GPU
python -c "import jax; print(jax.devices())"

# If still not working, force GPU reinstall
./setup.sh --force
```

#### Issue 2: CUDA Version Mismatch

**Problem**: JAX complains about CUDA version.

**Solution**:

```bash
# Check CUDA versions
nvcc --version  # System CUDA
python -c "import jax; print(jax.lib.xla_bridge.get_backend().platform_version)"  # JAX CUDA

# Reinstall matching versions
pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### Issue 3: Out of Memory Errors

**Problem**: Training crashes with OOM.

**Solution**:

Edit `.env`:

```bash
# Reduce memory fraction
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.6"  # Was 0.9

# Enable dynamic allocation
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Reload environment
deactivate
source ./activate.sh
```

#### Issue 4: Slow Training on Multi-GPU

**Problem**: Multi-GPU training slower than expected.

**Solution**:

Edit `.env`:

```bash
# Enable NCCL optimizations
export NCCL_DEBUG="INFO"  # Check for errors
export NCCL_IB_DISABLE="0"  # Enable InfiniBand if available
export NCCL_P2P_LEVEL="NVL"  # Use NVLink if available

# Enable XLA optimizations
export XLA_FLAGS="--xla_gpu_enable_async_all_reduce=true \
                  --xla_gpu_enable_latency_hiding_scheduler=true \
                  --xla_gpu_enable_highest_priority_async_stream=true"

# Check actual NCCL performance
NCCL_DEBUG=INFO python your_training_script.py 2>&1 | grep -i "nccl"
```

#### Issue 5: Template Not Being Used

**Problem**: Custom template not applied.

**Solution**:

```bash
# Check if template exists
ls -la .env.example*

# Verify setup.sh uses your template
grep ".env.example" setup.sh

# Manually create .env from template
sed "s|PROJECT_DIR=\"\$(pwd)\"|PROJECT_DIR=\"$(pwd)\"|g" \
    .env.example.custom > .env

# Source it
source .env
source .venv/bin/activate
```

## Testing Your Configuration

### Validation Script

Create `test_hardware_setup.py`:

```python
#!/usr/bin/env python3
"""Test hardware setup and configuration."""

import jax
import jax.numpy as jnp
import os
import sys

def test_jax_installation():
    """Test JAX installation."""
    print(f"✅ JAX version: {jax.__version__}")
    print(f"✅ JAX backend: {jax.default_backend()}")


def test_devices():
    """Test device detection."""
    devices = jax.devices()
    print(f"\n📱 Available devices: {len(devices)}")

    for i, device in enumerate(devices):
        print(f"  {i+1}. {device}")
        print(f"     Platform: {device.platform}")
        print(f"     Device kind: {device.device_kind}")

    return len(devices) > 0

def test_computation():
    """Test basic computation."""
    print("\n🧮 Testing computation...")

    try:
        x = jnp.arange(1000)
        y = jnp.sum(x ** 2)
        print(f"✅ Computation successful: {float(y)}")
        return True
    except Exception as e:
        print(f"❌ Computation failed: {e}")
        return False

def test_gpu_specific():
    """Test GPU-specific features."""
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']

    if not gpu_devices:
        print("\n💻 No GPU devices (CPU-only mode)")
        return True

    print(f"\n🎮 Testing {len(gpu_devices)} GPU device(s)...")

    try:
        # Test each GPU
        for i, gpu in enumerate(gpu_devices):
            with jax.default_device(gpu):
                x = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
                y = jnp.dot(x, x.T)
                print(f"✅ GPU {i} test passed: {y.shape}")

        # Test multi-GPU if available
        if len(gpu_devices) > 1:
            print(f"\n🔄 Testing multi-GPU with {len(gpu_devices)} devices...")
            x = jnp.arange(len(gpu_devices) * 100).reshape(len(gpu_devices), -1)
            result = jax.pmap(lambda x: x ** 2)(x)
            print(f"✅ Multi-GPU test passed: {result.shape}")

        return True
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_environment_variables():
    """Test environment variables."""
    print("\n📋 Environment Variables:")

    important_vars = [
        'JAX_PLATFORMS',
        'XLA_PYTHON_CLIENT_MEM_FRACTION',
        'XLA_PYTHON_CLIENT_PREALLOCATE',
        'LD_LIBRARY_PATH',
        'CUDA_VISIBLE_DEVICES',
        'CUDA_HOME',
    ]

    for var in important_vars:
        value = os.environ.get(var, 'Not set')
        # Truncate long values
        if len(str(value)) > 80:
            value = str(value)[:77] + "..."
        print(f"  {var}: {value}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Hardware Setup Validation")
    print("=" * 60)

    test_jax_installation()
    devices_ok = test_devices()
    compute_ok = test_computation()
    gpu_ok = test_gpu_specific()
    test_environment_variables()

    print("\n" + "=" * 60)
    if devices_ok and compute_ok and gpu_ok:
        print("✅ All tests passed! Hardware setup is correct.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check configuration.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run validation:

```bash
# After setup and activation
python test_hardware_setup.py
```

## Summary

### Quick Reference

**For NVIDIA GPUs (any model)**:

- Use default `.env.example` (works for all)
- Run `./setup.sh`
- Adjust `XLA_PYTHON_CLIENT_MEM_FRACTION` based on VRAM

**For Multi-GPU**:

- Set `CUDA_VISIBLE_DEVICES`
- Add NCCL configuration
- Enable async all-reduce in XLA_FLAGS

**For TPU**:

- Create `.env.example.tpu`
- Install JAX with TPU support
- Set `JAX_PLATFORMS="tpu,cpu"`

**For AMD ROCm**:

- Create `.env.example.rocm`
- Install JAX with ROCm (if available)
- Set `JAX_PLATFORMS="rocm,cpu"`

**For Apple Silicon**:

- Create `.env.example.metal`
- Install jax-metal
- Set `JAX_PLATFORMS="METAL,cpu"`

### Key Takeaways

1. **The template is hardware-agnostic** - it works for any NVIDIA GPU by default
2. **Customization is optional** - only needed for edge cases or optimizations
3. **Memory fraction is the main tunable** - adjust based on your GPU VRAM
4. **Multi-GPU requires explicit configuration** - add NCCL and XLA settings
5. **Test your setup** - run validation script after customization

### Getting Help

If you have issues:

1. Check this guide's troubleshooting section
2. Run the validation script
3. Check environment variables: `source .env && env | grep -E '(JAX|XLA|CUDA)'`
4. Open an issue with your hardware specs and error messages

---

**Last Updated**: 2025-10-15

**Maintainer**: Artifex Team

**Feedback**: Open an issue at [github.com/avitai/artifex](https://github.com/avitai/artifex/issues)
