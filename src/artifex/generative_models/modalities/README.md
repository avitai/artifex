# Modality Framework for Generative Models

This package provides a modality framework that enables separation of model architectures from domain-specific data types (modalities) in generative models.

## Directory Structure

```
modalities/
├── __init__.py         # Exports registry functions
├── base.py             # Base protocols for modalities
├── registry.py         # Registry for modality classes
├── protein/            # Protein modality
│   ├── __init__.py
│   ├── adapters.py     # Protein model adapters
│   ├── config.py       # Protein modality config
│   ├── modality.py     # Protein modality implementation
│   └── utils.py        # Utility functions
└── README.md           # This file
```

## Key Features

- **Separation of Concerns**: Cleanly separates model architectures from domain-specific functionality
- **Extensibility**: Easy to add new modalities without modifying existing code
- **Adaptability**: Modality adapters customize generic models for specific data types
- **Registration**: Registry system for discovering and instantiating modalities

## Core Components

- **Modality Protocol**: Defines interface for adapting models to specific data types
- **ModelAdapter Protocol**: Defines interface for modality-specific model adapters
- **Modality Registry**: Registry for looking up modality implementations by name

## Usage Examples

### Using the Registry

```python
from artifex.generative_models.modalities import (
    register_modality,
    get_modality,
    list_modalities,
)

# List available modalities
available_modalities = list_modalities()
print(available_modalities)  # {'protein': <class 'ProteinModality'>, ...}

# Get a specific modality
protein_modality_cls = get_modality("protein")
protein_modality = protein_modality_cls()
```

### Creating Models with Modalities

```python
from artifex.generative_models.factory import create_model
from artifex.generative_models.core.configuration import ModelConfiguration

# Create a geometric model adapted for protein data
config = ModelConfiguration(
    name="protein_geometric",
    model_class="artifex.generative_models.models.geometric.PointCloudModel",
    input_dim=(1024, 3),  # 1024 points, 3D coordinates
    hidden_dims=[256, 128],
    output_dim=(1024, 3),  # Output same as input for geometric models
    parameters={
        "num_points": 1024,
        "num_layers": 3,
        "modality": "protein",  # Specify modality
    }
)

protein_geometric_model = create_model(config, rngs=rngs)
```

### Using Model Zoo for Pre-configured Models

```python
from artifex.generative_models.zoo import zoo

# Create a pre-configured protein model
protein_model = zoo.create_model("protein_point_cloud", rngs=rngs)
```

## Adding New Modalities

To add a new modality:

1. Create a new package in `modalities/` for your modality
2. Implement the `Modality` protocol for your data type
3. Implement specific `ModelAdapter` classes for each model type
4. Register your modality using `register_modality`

## Design Principles

The modality framework follows these principles:

1. **Domain-Specific Separation**: Modalities encapsulate all domain-specific logic
2. **Adapter Pattern**: Adapters customize models for different data types
3. **Factory Method Pattern**: Factory functions create models with appropriate adapters
4. **Registry Pattern**: Registry provides lookup for available modalities
