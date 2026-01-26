"""DRY validation utilities for configuration dataclasses.

These utilities eliminate code duplication across config classes.
All validation functions follow fail-fast principle (raise immediately on invalid input).

Usage in config __post_init__:
    def __post_init__(self) -> None:
        validate_positive_int(self.latent_dim, "latent_dim")
        validate_positive_tuple(self.hidden_dims, "hidden_dims")
        validate_dropout_rate(self.dropout_rate)
        validate_activation(self.activation)
"""


def validate_positive_int(value: int, field_name: str) -> None:
    """Validate that an integer is positive (> 0).

    Args:
        value: Integer to validate
        field_name: Name of the field (for error messages)

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


def validate_non_negative_int(value: int, field_name: str) -> None:
    """Validate that an integer is non-negative (>= 0).

    Args:
        value: Integer to validate
        field_name: Name of the field (for error messages)

    Raises:
        ValueError: If value is negative
    """
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")


def validate_positive_float(value: float, field_name: str) -> None:
    """Validate that a float is positive (> 0.0).

    Args:
        value: Float to validate
        field_name: Name of the field (for error messages)

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive, got {value}")


def validate_non_negative_float(value: float, field_name: str) -> None:
    """Validate that a float is non-negative (>= 0.0).

    Args:
        value: Float to validate
        field_name: Name of the field (for error messages)

    Raises:
        ValueError: If value is negative
    """
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")


def validate_positive_tuple(values: tuple[int, ...], field_name: str) -> None:
    """Validate that a tuple of integers is non-empty and all values are positive.

    Args:
        values: Tuple of integers to validate
        field_name: Name of the field (for error messages)

    Raises:
        ValueError: If tuple is empty or any value is not positive
    """
    if not values:
        raise ValueError(f"{field_name} must have at least 1 element")

    if not all(v > 0 for v in values):
        raise ValueError(f"All {field_name} must be positive, got {values}")


def validate_positive_int_tuple(values: tuple[int, ...], field_name: str) -> None:
    """Validate that a tuple of integers contains only positive values.

    Alias for validate_positive_tuple with a more explicit name.

    Args:
        values: Tuple of integers to validate
        field_name: Name of the field (for error messages)

    Raises:
        ValueError: If any value is not positive
    """
    if not all(v > 0 for v in values):
        raise ValueError(f"All values in {field_name} must be positive, got {values}")


def validate_dropout_rate(rate: float) -> None:
    """Validate that dropout rate is in valid range [0.0, 1.0].

    Args:
        rate: Dropout rate to validate

    Raises:
        ValueError: If rate is not in [0.0, 1.0]
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"dropout_rate must be in [0.0, 1.0], got {rate}")


def validate_probability(value: float, field_name: str) -> None:
    """Validate that a probability is in valid range [0.0, 1.0].

    Generic version of validate_dropout_rate for any probability field.

    Args:
        value: Probability to validate
        field_name: Name of the field (for error messages)

    Raises:
        ValueError: If value is not in [0.0, 1.0]
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0], got {value}")


def validate_range(value: float, field_name: str, min_val: float, max_val: float) -> None:
    """Validate that a value is within a specified range (inclusive).

    Args:
        value: Value to validate
        field_name: Name of the field (for error messages)
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Raises:
        ValueError: If value is outside the valid range
    """
    if not min_val <= value <= max_val:
        raise ValueError(f"{field_name} must be between {min_val} and {max_val}, got {value}")


def validate_learning_rate(lr: float) -> None:
    """Validate that learning rate is positive.

    Args:
        lr: Learning rate to validate

    Raises:
        ValueError: If learning rate is not positive
    """
    if lr <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {lr}")


def validate_activation(activation: str) -> None:
    """Validate that activation function is supported.

    Checks against common Flax NNX and JAX activation functions.

    Args:
        activation: Activation function name

    Raises:
        ValueError: If activation is not recognized
    """
    # Common activations in Flax NNX and JAX
    valid_activations = {
        "relu",
        "gelu",
        "silu",
        "swish",  # Alias for silu
        "tanh",
        "sigmoid",
        "elu",
        "leaky_relu",
        "softplus",
        "softsign",
        "relu6",
        "hard_tanh",
        "celu",
        "selu",
        "glu",
    }

    if activation not in valid_activations:
        raise ValueError(
            f"Unknown activation function: {activation}. Valid options: {sorted(valid_activations)}"
        )
