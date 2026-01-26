# Checkpointing

**Module:** `generative_models.core.checkpointing`

**Source:** `generative_models/core/checkpointing.py`

## Overview

Checkpointing utilities for saving and loading model state using Orbax. Provides functions for basic model checkpointing, optimizer state checkpointing, checkpoint validation, and corruption recovery.

## Functions

### setup_checkpoint_manager

```python
def setup_checkpoint_manager(base_dir: str) -> tuple[ocp.CheckpointManager, str]
```

Setup Orbax checkpoint manager.

**Parameters:**

- `base_dir`: Directory path for checkpoints

**Returns:**

- Tuple of (CheckpointManager, absolute_path)

---

### save_checkpoint

```python
def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module,
    step: int,
) -> ocp.CheckpointManager
```

Save model checkpoint using Orbax.

**Parameters:**

- `checkpoint_manager`: The Orbax CheckpointManager instance
- `model`: The NNX model to save
- `step`: The step number for this checkpoint

**Returns:**

- The checkpoint manager (for chaining)

---

### load_checkpoint

```python
def load_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    target_model_template: Optional[nnx.Module | nnx.GraphDef] = None,
    step: Optional[int] = None,
) -> tuple[Optional[Any], Optional[int]]
```

Load model checkpoint using Orbax.

**Parameters:**

- `checkpoint_manager`: The Orbax CheckpointManager instance
- `target_model_template`: Optional model template for restoration
- `step`: Specific step to restore (None = latest)

**Returns:**

- Tuple of (restored_model_or_state, step) or (None, None) if not found

---

### save_checkpoint_with_optimizer

```python
def save_checkpoint_with_optimizer(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step: int,
) -> ocp.CheckpointManager
```

Save both model and optimizer state to checkpoint.

**Parameters:**

- `checkpoint_manager`: The Orbax CheckpointManager instance
- `model`: The NNX model to save
- `optimizer`: The NNX Optimizer to save
- `step`: The step number for this checkpoint

**Returns:**

- The checkpoint manager (for chaining)

---

### load_checkpoint_with_optimizer

```python
def load_checkpoint_with_optimizer(
    checkpoint_manager: ocp.CheckpointManager,
    model_template: nnx.Module,
    optimizer_template: nnx.Optimizer,
    step: Optional[int] = None,
) -> tuple[Optional[nnx.Module], Optional[nnx.Optimizer], Optional[int]]
```

Load both model and optimizer state from checkpoint.

**Parameters:**

- `checkpoint_manager`: The Orbax CheckpointManager instance
- `model_template`: Model with same structure as saved model
- `optimizer_template`: Optimizer with same structure as saved
- `step`: Specific step to restore (None = latest)

**Returns:**

- Tuple of (model, optimizer, step) or (None, None, None) if not found

---

### validate_checkpoint

```python
def validate_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module,
    step: int,
    validation_data: Any,
    tolerance: float = 1e-5,
) -> bool
```

Validate that a checkpoint loads correctly and produces consistent outputs.

**Parameters:**

- `checkpoint_manager`: The Orbax CheckpointManager instance
- `model`: The current model whose state was saved
- `step`: The step number to validate
- `validation_data`: Input data to test model outputs
- `tolerance`: Maximum allowed difference between outputs

**Returns:**

- True if checkpoint is valid, False otherwise

---

### recover_from_corruption

```python
def recover_from_corruption(
    checkpoint_dir: str,
    model_template: nnx.Module,
) -> tuple[Optional[nnx.Module], Optional[int]]
```

Attempt to recover from corrupted checkpoints.

Tries loading checkpoints from newest to oldest until one succeeds.

**Parameters:**

- `checkpoint_dir`: Directory containing checkpoints
- `model_template`: Model with same structure as saved model

**Returns:**

- Tuple of (recovered_model, step) or (None, None) if recovery failed

## Module Statistics

- **Classes:** 0
- **Functions:** 7
- **Imports:** 6
