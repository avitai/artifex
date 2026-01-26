"""Configuration schema for hyperparameter search."""

from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator

from artifex.configs.schema.base import BaseConfig


class SearchType(str, Enum):
    """Supported hyperparameter search types."""

    GRID = "grid"  # Grid search
    RANDOM = "random"  # Random search
    BAYESIAN = "bayesian"  # Bayesian optimization
    POPULATION = "population"  # Population-based training


class ParameterDistribution(BaseConfig):
    """Base class for parameter distributions."""

    name: str = Field("", description="Name of the parameter")
    param_path: str = Field("", description="Path to parameter in configuration (dot notation)")


class CategoricalDistribution(ParameterDistribution):
    """Distribution for categorical parameters."""

    type: Literal["categorical"] = Field("categorical", description="Type of distribution")
    categories: list[Any] = Field(..., description="list of possible values for this parameter")

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: list[Any]) -> list[Any]:
        """Validate that values list is not empty."""
        if not v:
            raise ValueError("Categories list cannot be empty")
        return v


class LogUniformDistribution(ParameterDistribution):
    """Distribution for log-uniform distributed parameters."""

    type: Literal["log_uniform"] = Field("log_uniform", description="Type of distribution")
    low: float = Field(..., description="Lower bound of the distribution (must be > 0)")
    high: float = Field(..., description="Upper bound of the distribution")

    @field_validator("low")
    @classmethod
    def validate_low(cls, v: float) -> float:
        """Validate that low bound is positive."""
        if v <= 0:
            raise ValueError("Lower bound must be positive for log-uniform distribution")
        return v

    @field_validator("high")
    @classmethod
    def validate_high(cls, v: float, info: Any) -> float:
        """Validate that high > low."""
        values = info.data
        if "low" in values and v <= values["low"]:
            raise ValueError("Upper bound must be greater than lower bound")
        return v


class UniformDistribution(ParameterDistribution):
    """Distribution for uniformly distributed parameters."""

    type: Literal["uniform"] = Field("uniform", description="Type of distribution")
    low: float | int = Field(..., description="Minimum value for the parameter")
    high: float | int = Field(..., description="Maximum value for the parameter")
    q: float | None = Field(None, description="Quantization parameter (step size)")
    log_scale: bool = Field(False, description="Whether to use log scale")

    @field_validator("high")
    @classmethod
    def validate_range(cls, v: float | int, info: Any) -> float | int:
        """Validate that high > low."""
        values = info.data
        if "low" in values and v <= values["low"]:
            raise ValueError("high must be greater than low")
        return v

    @field_validator("q")
    @classmethod
    def validate_q(cls, v: float | None) -> float | None:
        """Validate quantization parameter."""
        if v is not None and v <= 0:
            raise ValueError("Quantization parameter must be positive")
        return v


class ChoiceDistribution(ParameterDistribution):
    """Distribution for discrete choice parameters."""

    type: Literal["choice"] = Field("choice", description="Type of distribution")
    choices: list[Any] = Field(..., description="list of choices for this parameter")
    weights: list[float] | None = Field(None, description="Optional weights for choices")

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v: list[Any]) -> list[Any]:
        """Validate that choices list is not empty."""
        if not v:
            raise ValueError("Choices list cannot be empty")
        return v

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: list[float] | None, info: Any) -> list[float] | None:
        """Validate weights if provided."""
        values = info.data
        if v is not None:
            if "choices" not in values:
                raise ValueError("Cannot validate weights without choices")

            if len(v) != len(values["choices"]):
                raise ValueError("Number of weights must match number of choices")

            if any(w < 0 for w in v):
                raise ValueError("Weights cannot be negative")

            if sum(v) == 0:
                raise ValueError("Sum of weights cannot be zero")
        return v


class HyperparamSearchConfig(BaseConfig):
    """Configuration for hyperparameter search."""

    name: str = Field(
        "hyperparam_search",
        description="Name for this hyperparameter search configuration",
    )
    search_type: SearchType = Field(SearchType.RANDOM, description="Type of hyperparameter search")
    num_trials: int = Field(10, description="Number of trials to run")
    max_parallel_trials: int = Field(1, description="Maximum number of parallel trials")

    # Search space
    search_space: dict[
        str,
        CategoricalDistribution | UniformDistribution | ChoiceDistribution | LogUniformDistribution,
    ] = Field({}, description="Parameter distributions to search over")

    # Early stopping
    early_stopping: bool = Field(False, description="Whether to use early stopping")
    patience: int = Field(10, description="Number of trials without improvement before stopping")

    # Pruning (for supported algorithms)
    pruning: bool = Field(False, description="Whether to use pruning")
    pruning_interval: int = Field(1, description="Interval for pruning checks")

    # Optimization direction
    direction: Literal["minimize", "maximize"] = Field(
        "minimize", description="Direction for optimization"
    )
    metric: str = Field("validation_loss", description="Metric to optimize")

    # Random seed for reproducibility
    seed: int = Field(42, description="Random seed for hyperparameter search")

    # Optional experiment tracking
    tracking_uri: str | None = Field(None, description="URI for tracking service")

    @field_validator("num_trials", "max_parallel_trials", "patience", "pruning_interval")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that value is a positive integer."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("max_parallel_trials")
    @classmethod
    def validate_max_parallel(cls, v: int, info: Any) -> int:
        """Validate max_parallel_trials."""
        values = info.data
        if "num_trials" in values and v > values["num_trials"]:
            raise ValueError("max_parallel_trials cannot be greater than num_trials")
        return v

    @field_validator("search_space")
    @classmethod
    def validate_search_space(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate that search_space is not empty."""
        if not v:
            raise ValueError("Search space cannot be empty")
        return v
