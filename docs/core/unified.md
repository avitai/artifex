# Unified

**Module:** `generative_models.core.configuration.unified`

**Source:** `generative_models/core/configuration/unified.py`

## Overview

Unified configuration system for artifex generative models.

This module provides a centralized configuration management system that
replaces the fragmented configuration approaches across the codebase.

Key Features:

1. Type-safe configuration with Pydantic validation
2. Hierarchical configuration inheritance
3. Centralized registry for all configuration types
4. Consistent validation and serialization
5. Easy extension mechanism for new modalities/models

## Classes

### BaseConfiguration

```python
class BaseConfiguration
```

### Config

```python
class Config
```

### ConfigurationRegistry

```python
class ConfigurationRegistry
```

### ConfigurationType

```python
class ConfigurationType
```

### DataConfig

```python
class DataConfig
```

### EvaluationConfig

```python
class EvaluationConfig
```

### ExperimentConfig

```python
class ExperimentConfig
```

### ModalityConfiguration

```python
class ModalityConfiguration
```

### ModelConfig

```python
class ModelConfig
```

### OptimizerConfig

```python
class OptimizerConfig
```

### SchedulerConfig

```python
class SchedulerConfig
```

### TrainingConfig

```python
class TrainingConfig
```

## Functions

### **init**

```python
def __init__()
```

### create_from_template

```python
def create_from_template()
```

### create_from_template

```python
def create_from_template()
```

### from_yaml

```python
def from_yaml()
```

### get

```python
def get()
```

### get_config

```python
def get_config()
```

### list_configs

```python
def list_configs()
```

### list_configs

```python
def list_configs()
```

### load_from_directory

```python
def load_from_directory()
```

### merge

```python
def merge()
```

### register

```python
def register()
```

### register_config

```python
def register_config()
```

### register_template

```python
def register_template()
```

### resolve_configs

```python
def resolve_configs()
```

### to_yaml

```python
def to_yaml()
```

### validate_activation

```python
def validate_activation()
```

### validate_compatibility

```python
def validate_compatibility()
```

### validate_optimizer_type

```python
def validate_optimizer_type()
```

### validate_scheduler_type

```python
def validate_scheduler_type()
```

## Module Statistics

- **Classes:** 12
- **Functions:** 19
- **Imports:** 6
