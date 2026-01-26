# Artifex Core Module

This module contains the core components of the artifex generative models library, including device management, performance infrastructure, and base abstractions.

## Device Management System

The device management system provides GPU/CPU optimization with type-safe configuration and testing.

### Architecture Overview

Key components:

- **DeviceManager**: Central device management with unified API
- **DeviceDetector**: Protocol-based device detection with CUDA implementations
- **DeviceConfiguration**: Type-safe configuration with enum-based strategies
- **JAXDeviceManager**: JAX-specific device operations
- **DeviceTestingSuite**: Testing framework for device validation

### Key Features

1. **Type-Safe Configuration**: Enum-based device types and memory strategies
2. **Automatic Detection**: CUDA capability detection with detailed reporting
3. **Configurable Memory Strategies**: Conservative, Balanced, and Aggressive allocation
4. **Progressive Testing**: Critical, important, and optional tests
5. **Model-Size Aware**: Configuration recommendations based on model size

### Quick Start

```python
from artifex.generative_models.core.device_manager import get_device_manager

# Get global device manager
manager = get_device_manager()
print(f"GPU Available: {manager.has_gpu}")
print(f"Device Count: {manager.device_count}")
print(f"Device Info: {manager.get_device_info()}")
```

### Configuration Options

```python
from artifex.generative_models.core.device_manager import (
    DeviceConfiguration,
    MemoryStrategy,
    configure_for_generative_models
)

# Custom configuration
config = DeviceConfiguration(
    memory_strategy=MemoryStrategy.BALANCED,
    memory_fraction=0.8,
    enable_x64=False,
    platform_priority=["cuda", "cpu"]
)

# Pre-configured for generative models
manager = configure_for_generative_models(
    memory_strategy=MemoryStrategy.BALANCED,
    enable_mixed_precision=True
)
```

### Memory Strategies

| Strategy | Memory Fraction | Use Case |
|----------|----------------|----------|
| **Conservative** | 60% | Stable training, multiple processes |
| **Balanced** | 75% | General purpose (recommended) |
| **Aggressive** | 90% | Maximum performance, single model |
| **Custom** | User-defined | Specific requirements |

### Device Testing

```python
from artifex.generative_models.core.device_testing import (
    run_device_tests,
    print_test_results
)

# Run device tests
suite = run_device_tests()
print_test_results(suite)

# Check system health
print(f"System Health: {suite.is_healthy}")
print(f"Success Rate: {suite.success_rate}")
```

### Command Line Tools

```bash
# Quick device status
python scripts/gpu_utils.py

# Detailed device information
python scripts/gpu_utils.py --comprehensive

# Run device tests
python scripts/gpu_utils.py --test

# Test critical operations only
python scripts/gpu_utils.py --test-critical

# Configure for generative models
python scripts/gpu_utils.py --configure-generative
```

### Advanced Usage

#### Model-Size Aware Configuration

```python
# Optimize configuration for model size
large_model_config = manager.optimize_for_model_size(
    parameter_count=1e9  # 1B parameters
)
print(f"Recommended strategy: {large_model_config.memory_strategy}")
```

#### Data Distribution Across Devices

```python
import jax.numpy as jnp

# Distribute data across available devices
data = jnp.ones((64, 784))
distributed_data = manager.jax_manager.distribute_data(data)
print(f"Data distributed across {len(distributed_data)} devices")
```

#### Custom Device Detection

```python
from artifex.generative_models.core.device_manager import DeviceDetector

class CustomDetector(DeviceDetector):
    def detect_capabilities(self):
        # Custom detection logic
        return DeviceCapabilities(
            device_type=DeviceType.GPU,
            device_count=1,
            total_memory_mb=8192,
            compute_capability="8.6"
        )

# Use custom detector
custom_manager = DeviceManager(detector=CustomDetector())
```

### Environment Variables

The system configures optimal environment variables:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_ENABLE_X64=0
TF_CPP_MIN_LOG_LEVEL=1
JAX_PLATFORMS=cuda,cpu
XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
```

### Performance Infrastructure

The core module includes performance analysis tools:

```python
from artifex.generative_models.core.performance import (
    HardwareDetector,
    PerformanceEstimator
)

# Hardware detection
detector = HardwareDetector()
specs = detector.detect_hardware()
print(f"Platform: {specs.platform}")
print(f"Memory: {specs.memory_gb} GB")

# Performance estimation
estimator = PerformanceEstimator()
flops = estimator.estimate_flops_linear(
    batch_size=32,
    input_size=784,
    output_size=128
)
print(f"Estimated FLOPs: {flops:,}")
```

### Integration with Training

```python
# Configure device manager for training
manager = configure_for_generative_models(
    memory_strategy=MemoryStrategy.BALANCED,
    enable_mixed_precision=True
)

# Use in training loop
def train_step(model, batch):
    device = manager.jax_manager.get_default_device()
    batch = jax.device_put(batch, device)
    return model.train_step(batch)
```

### Best Practices

1. **Initialize Early**: Set up device management at program start
2. **Choose Appropriate Strategy**: Based on your use case
3. **Test Regularly**: Run device tests for optimal performance
4. **Monitor Performance**: Track device health
5. **Handle Errors**: Implement proper error handling

### Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU not detected | Check CUDA installation and drivers |
| Memory allocation errors | Reduce memory fraction or use Conservative strategy |
| Performance issues | Run device tests and check hardware |
| Configuration conflicts | Use `force_reinit=True` with get_device_manager |

### API Reference

For complete API documentation, see module docstrings:

- `device_manager.py`: Core device management classes
- `device_testing.py`: Testing framework and utilities
- `performance.py`: Performance analysis tools
- `adapters.py`: Model adapters for scaling

The device management system provides a foundation for GPU/CPU operations in Artifex, ensuring optimal performance across different hardware configurations.
