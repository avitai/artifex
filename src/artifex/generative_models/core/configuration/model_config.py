"""ModelConfig frozen dataclass configuration.

Replaces Pydantic ModelConfiguration with frozen dataclass.

Design:
- Frozen dataclass inheriting from BaseConfig
- All validation in __post_init__ using DRY utilities
- Tuples for immutable sequences (hidden_dims, tags)
- Support for int or tuple input_dim and output_dim
- Model-specific parameters in 'parameters' field
"""

import dataclasses
from typing import Any

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.validation import (
    validate_activation,
    validate_dropout_rate,
    validate_positive_tuple,
)


@dataclasses.dataclass(frozen=True)
class ModelConfig(BaseConfig):
    """Configuration for generative models.

    This class provides a unified configuration interface for all models in Artifex.

    Parameter Handling Guidelines:
    ------------------------------

    1. The `parameters` field is for FUNCTIONAL configuration:
       - Model-specific hyperparameters (e.g., beta for VAE, noise_steps for diffusion)
       - Architecture details not covered by standard fields
       - Anything that affects model behavior or training

    2. The `metadata` field is for NON-FUNCTIONAL information:
       - Experiment tracking IDs
       - Documentation and notes
       - Hyperparameter search spaces
       - Dataset versions
       - Author information

    Examples:
    ---------
    Correct usage:
    ```python
    config = ModelConfig(
        name="vae_experiment",
        model_class="artifex.generative_models.models.vae.VAE",
        input_dim=(28, 28, 1),
        hidden_dims=(256, 128),
        output_dim=64,
        # Functional parameters that affect model behavior
        parameters={
            "beta": 1.0,
            "kl_weight": 0.5,
            "reconstruction_loss": "mse",
        },
        # Non-functional metadata for tracking
        metadata={
            "experiment_id": "exp_001",
            "dataset_version": "v2.1",
            "notes": "Testing lower KL weight",
        }
    )
    ```

    Attributes:
        model_class: Fully qualified model class name
        input_dim: Input dimensions (int for 1D, tuple for multi-dimensional)
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimensions (None, int, or tuple)
        activation: Activation function name
        dropout_rate: Dropout rate [0.0, 1.0]
        use_batch_norm: Whether to use batch normalization
        rngs_seeds: Random seeds for NNX Rngs (dict of key -> seed)
        parameters: Model-specific functional parameters
    """

    # Model architecture - Required fields (must come AFTER BaseConfig fields!)
    # BaseConfig has 'name' as required field, so these go after all BaseConfig defaults
    model_class: str = ""  # Will validate in __post_init__
    input_dim: int | tuple[int, ...] = ()  # Will validate in __post_init__

    # Model architecture - Optional fields with defaults
    hidden_dims: tuple[int, ...] = (128, 256, 512)
    output_dim: int | tuple[int, ...] | None = None

    # Model parameters - Optional with defaults
    activation: str = "gelu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True

    # NNX specific
    rngs_seeds: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"params": 0, "dropout": 1}
    )

    # Additional model-specific parameters
    parameters: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Validate all fields.

        Validation uses DRY utilities from validation.py.
        Follows fail-fast principle - raise on first error.
        """
        # Call parent validation first
        super().__post_init__()

        # Validate required fields (they have dummy defaults for dataclass compatibility)
        if not self.model_class:
            raise ValueError("model_class is required and cannot be empty")
        if not self.input_dim:
            raise ValueError("input_dim is required and cannot be empty")

        # Validate activation function
        validate_activation(self.activation)

        # Validate dropout rate
        validate_dropout_rate(self.dropout_rate)

        # Validate hidden_dims (non-empty tuple of positive ints)
        validate_positive_tuple(self.hidden_dims, "hidden_dims")

        # Validate input_dim
        self._validate_dimension(self.input_dim, "input_dim", allow_none=False)

        # Validate output_dim (can be None)
        if self.output_dim is not None:
            self._validate_dimension(self.output_dim, "output_dim", allow_none=False)

        # Validate rngs_seeds is not empty
        if not self.rngs_seeds:
            raise ValueError("rngs_seeds cannot be empty")

    def _validate_dimension(
        self, value: int | tuple[int, ...], field_name: str, allow_none: bool = True
    ) -> None:
        """Validate dimension field (can be int or tuple).

        Args:
            value: Dimension value to validate
            field_name: Name of field for error messages
            allow_none: Whether None is allowed

        Raises:
            ValueError: If validation fails
        """
        if value is None:
            if not allow_none:
                raise ValueError(f"{field_name} cannot be None")
            return

        # If int, check positive
        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
            return

        # If tuple, check all positive
        if isinstance(value, tuple):
            if not value:  # Empty tuple
                raise ValueError(f"{field_name} cannot be empty")
            for i, dim in enumerate(value):
                if not isinstance(dim, int):
                    raise ValueError(f"{field_name}[{i}] must be int, got {type(dim).__name__}")
                if dim <= 0:
                    raise ValueError(f"{field_name}[{i}] must be positive, got {dim}")
            return

        raise ValueError(f"{field_name} must be int or tuple[int, ...], got {type(value).__name__}")
