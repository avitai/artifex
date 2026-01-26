# Migrate Configs

**Module:** `generative_models.core.configuration.migrate_configs`

**Source:** `generative_models/core/configuration/migrate_configs.py`

## Overview

Script to migrate all configurations in the codebase to the new unified system.

This script will:

1. Find all files using old configuration patterns
2. Create new configuration objects
3. Update the code to use the new system
4. Generate migration reports

## Classes

### ConfigMigrator

```python
class ConfigMigrator
```

## Functions

### **init**

```python
def __init__()
```

### create_migration_plan

```python
def create_migration_plan()
```

### find_config_patterns

```python
def find_config_patterns()
```

### generate_config_templates

```python
def generate_config_templates()
```

### generate_migration_code

```python
def generate_migration_code()
```

### main

```python
def main()
```

### migrate_metric_classes

```python
def migrate_metric_classes()
```

### migrate_model_factories

```python
def migrate_model_factories()
```

### visit_Dict

```python
def visit_Dict()
```

### visit_FunctionDef

```python
def visit_FunctionDef()
```

## Module Statistics

- **Classes:** 1
- **Functions:** 10
- **Imports:** 5
