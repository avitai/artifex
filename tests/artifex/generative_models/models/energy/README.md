# Energy Model Test Suite

This directory contains comprehensive unit tests, integration tests, and property-based tests for the energy-based models in the `artifex.generative_models.models.energy` package.

## Test Structure

### 1. **`conftest.py`** - Test Infrastructure

- **JAX Conditional Imports**: Graceful handling of environments with/without JAX
- **Fixture System**: Comprehensive fixtures for RNG keys, test data, configurations
- **Test Data Generators**: Synthetic data for MLP and CNN testing
- **Configuration Fixtures**: MCMC, buffer, and loss configurations for consistent testing
- **Tolerance Settings**: Numerical tolerance settings for different test scenarios

### 2. **`test_base.py`** - Base Classes Unit Tests

- **`TestEnergyFunction`**: Abstract base class validation
- **`TestMLPEnergyFunction`**:
  - Initialization with various configurations
  - Forward pass validation
  - Dropout functionality
  - Different activation functions
  - Bias control options
- **`TestCNNEnergyFunction`**:
  - Initialization and configuration
  - Forward pass with image data
  - Various kernel sizes
  - Multiple activation functions
  - Different input channel configurations
- **`TestEnergyBasedModel`**:
  - Energy computation methods
  - Score function validation
  - Contrastive divergence loss
  - Sample generation
  - Loss function integration
  - Data type consistency

### 3. **`test_ebm.py`** - EBM Implementation Tests

- **`TestEBM`**: Main EBM class functionality
  - MLP and CNN initialization
  - Custom energy function support
  - Training steps and buffer integration
  - Configuration management
  - Error handling for invalid inputs
- **`TestDeepEBM`**: Deep EBM variant testing
  - Advanced CNN architectures
  - Residual connections and spectral normalization
  - Configuration options validation
- **`TestDeepCNNEnergyFunction`**: Deep CNN energy functions
- **`TestEnergyBlock`**: Energy block components
  - Residual connections
  - Stride configurations
  - Multiple activation functions
- **`TestFactoryFunctions`**: Pre-configured model creation
  - MNIST, CIFAR, and simple EBM factories
  - Custom parameter passing
- **`TestEBMIntegration`**: Complete workflow validation

### 4. **`test_mcmc.py`** - MCMC Utilities Tests

- **`TestLangevinDynamics`**: Basic Langevin dynamics
  - Parameter variations (step size, noise scale)
  - Gradient and value clipping
  - Convergence properties
  - Default RNG handling
- **`TestLangevinDynamicsWithTrajectory`**: Trajectory recording
  - Save interval configurations
  - Consistency with regular Langevin
- **`TestSampleBuffer`**: Sample buffer management
  - Capacity management
  - Sampling strategies (initialization probability)
  - Shape consistency
  - Error conditions
- **`TestImprovedLangevinDynamics`**: Advanced sampling
  - Adaptive step size
  - Convergence testing
- **`TestPersistentContrastiveDivergence`**: PCD sampling
  - Buffer integration
  - Parameter variations
  - Energy descent validation
- **`TestMCMCIntegration`**: Cross-method consistency

### 5. **`test_integration.py`** - Integration Tests

- **`TestEnergyModelWorkflows`**: End-to-end workflows
  - Complete training pipelines for MLP and CNN EBMs
  - Factory function workflows
  - Deep EBM advanced workflows
- **`TestCrossComponentIntegration`**: Component interaction
  - Energy function interchange
  - MCMC-EBM integration
  - Buffer-EBM integration
  - Data type consistency across components
- **`TestErrorHandlingIntegration`**: Error scenarios
  - Mismatched dimensions
  - Empty buffer handling
  - Invalid MCMC parameters
- **`TestPerformanceIntegration`**: Performance characteristics
  - Batch size scaling
  - Memory efficiency with large buffers
- **`TestReproducibilityIntegration`**: Deterministic behavior
  - Fixed seed reproducibility
  - Generation consistency
  - MCMC reproducibility

### 6. **`test_properties.py`** - Property-Based Tests

- **`TestEnergyFunctionProperties`**: Mathematical properties
  - Determinism verification
  - Batch independence
  - Scaling behavior
  - Gradient finiteness
  - Symmetry properties
- **`TestEnergyBasedModelProperties`**: Theoretical relationships
  - Energy-log probability relationship
  - Score function gradient relationship
  - Contrastive divergence properties
  - Generation consistency
- **`TestMCMCProperties`**: MCMC theoretical properties
  - Detailed balance (in expectation)
  - Energy descent
  - Invariance properties
  - Convergence characteristics
- **`TestSampleBufferProperties`**: Buffer invariants
  - Capacity constraints
  - Deterministic sampling
  - Shape consistency
  - Initialization probability effects
- **`TestNumericalStabilityProperties`**: Numerical robustness
  - Extreme input handling
  - Gradient stability
  - Challenging energy functions
  - Data type preservation

## Test Features

### Conditional JAX Support

All tests include conditional JAX imports with fallback behavior for environments without JAX installed, ensuring the test suite doesn't break due to missing dependencies.

### Comprehensive Fixtures

- **RNG Management**: Multiple RNG keys for different purposes
- **Data Generation**: Synthetic datasets for various test scenarios
- **Configuration**: Standardized configurations for consistent testing
- **Tolerances**: Appropriate numerical tolerances for different test types

### Property-Based Testing

Tests verify mathematical properties and invariants that should hold for energy models:

- Energy function determinism
- MCMC convergence properties
- Buffer capacity invariants
- Numerical stability

### Integration Testing

Complete workflows testing realistic usage scenarios:

- Training loops with multiple steps
- Sample generation pipelines
- Cross-component interactions
- Error handling in integrated scenarios

### Performance Testing

Tests for performance characteristics:

- Batch size scaling
- Memory efficiency
- Large buffer handling

### Reproducibility Testing

Ensures deterministic behavior with fixed seeds:

- Training reproducibility
- Generation consistency
- MCMC reproducibility

## Test Coverage

The test suite provides comprehensive coverage of:

- **Core Functionality**: All public methods and classes
- **Edge Cases**: Error conditions and boundary cases
- **Integration**: Cross-component interactions
- **Properties**: Mathematical and theoretical properties
- **Performance**: Scaling and efficiency characteristics
- **Reproducibility**: Deterministic behavior verification

## Usage

### Running All Tests

```bash
python -m pytest tests/artifex/generative_models/models/energy/ -v
```

### Running Specific Test Files

```bash
python -m pytest tests/artifex/generative_models/models/energy/test_base.py -v
python -m pytest tests/artifex/generative_models/models/energy/test_ebm.py -v
python -m pytest tests/artifex/generative_models/models/energy/test_mcmc.py -v
python -m pytest tests/artifex/generative_models/models/energy/test_integration.py -v
python -m pytest tests/artifex/generative_models/models/energy/test_properties.py -v
```

### Running Tests with JAX Requirement

```bash
python -m pytest tests/artifex/generative_models/models/energy/ -v -m "not skipif"
```

## Notes

- Tests are designed to work in both CPU and GPU environments
- GPU memory limitations may cause some tests to fail in resource-constrained environments
- All tests include appropriate numerical tolerances for finite precision arithmetic
- Property-based tests verify theoretical expectations and mathematical relationships
- Integration tests ensure components work together correctly in realistic scenarios

The test suite provides confidence in the correctness, robustness, and performance of the energy model implementations.
