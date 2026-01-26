# Modality Architecture Test Suite

This directory contains comprehensive tests for the modality architecture that enables the separation of model architectures from domain-specific (modality) functionality.

## Test Structure

The test suite is organized as follows:

### Core Components

- **Registry Tests** (`test_registry.py`): Tests for the modality registry system that allows registering and retrieving modalities by name.
- **Registry Initialization Tests** (`test_registry_init.py`): Tests that verify modalities are properly registered during package initialization.

### Protein Modality

- **Modality Tests** (`protein/test_modality.py`): Tests for the protein modality implementation, including extension and adapter retrieval.
- **Adapters Tests** (`protein/test_adapters.py`): Tests for protein-specific model adapters that adapt different model types to work with protein data.
- **Utilities Tests** (`protein/test_utils.py`): Tests for utility functions that support the protein modality.

### Extension System

- **Base Extensions Tests** (`../extensions/test_base_extensions.py`): Tests for the base extension classes that provide the foundation for model extensions.
- **Protein Extensions Tests** (`../extensions/protein/test_protein_extensions.py`): Tests for protein-specific extensions like bond length and angle constraints.

### Factory Functions

- **Factory Tests** (`../factories/test_factories.py`): Tests for the factory functions that create models with specific modalities and adaptations.

### Integration Tests

- **Modality Architecture Integration** (`../integration/test_modality_architecture.py`): End-to-end tests that verify all components of the modality architecture work together correctly.

## Test Coverage

The test suite covers:

1. **Registration**: Testing the modality registry for registering, retrieving, and listing modalities.
2. **Extensions**: Testing extension creation, configuration, and functionality.
3. **Adapters**: Testing adapters that modify model behavior for specific modalities.
4. **Factory Functions**: Testing creation of models with modality-specific adaptations.
5. **Integration**: Testing the entire system working together from modality registration to model creation and usage.

## Running the Tests

Run the tests using pytest:

```bash
python -m pytest tests/artifex/generative_models/modalities
python -m pytest tests/artifex/generative_models/extensions
python -m pytest tests/artifex/generative_models/factories
python -m pytest tests/artifex/generative_models/integration
```

## Adding New Modality Tests

When adding tests for a new modality:

1. Create a new directory `tests/artifex/generative_models/modalities/<modality_name>/`
2. Implement tests for the modality implementation, adapters, and utilities
3. Add tests for any new extensions in `tests/artifex/generative_models/extensions/<modality_name>/`
4. Update the integration tests to include the new modality

Follow the patterns established in the existing tests to ensure consistency.
