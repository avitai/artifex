"""Hyperparameter search frozen dataclass configuration.

Design:
- Frozen dataclass inheriting from BaseConfig
- All validation in __post_init__ using DRY utilities
- StrEnum for SearchType (Python 3.11+)
- Immutable tuples instead of mutable lists for sequence fields
- Literal type discriminators for distribution subclasses
"""

import dataclasses
import enum
from typing import Any, Literal

from artifex.generative_models.core.configuration.base_dataclass import (
    BaseConfig,
    ConfigDocument,
)
from artifex.generative_models.core.configuration.validation import (
    validate_positive_float,
    validate_positive_int,
)


class SearchType(enum.StrEnum):
    """Supported hyperparameter search types."""

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    POPULATION = "population"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ParameterDistribution(ConfigDocument):
    """Base class for parameter distributions.

    Attributes:
        name: Name of the parameter (empty string allowed for distributions).
        param_path: Dot-notation path to the parameter in configuration.
    """

    name: str = ""
    param_path: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CategoricalDistribution(ParameterDistribution):
    """Distribution for categorical parameters.

    Attributes:
        type: Distribution type discriminator.
        categories: Tuple of possible values for this parameter (must be non-empty).
    """

    type: Literal["categorical"] = "categorical"
    categories: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Validate categorical distribution fields."""
        super(CategoricalDistribution, self).__post_init__()
        if not self.categories:
            raise ValueError("categories must not be empty")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LogUniformDistribution(ParameterDistribution):
    """Distribution for log-uniform distributed parameters.

    Both bounds must be positive, and high must exceed low.

    Attributes:
        type: Distribution type discriminator.
        low: Lower bound of the distribution (must be > 0).
        high: Upper bound of the distribution (must be > low).
    """

    type: Literal["log_uniform"] = "log_uniform"
    low: float = 1e-6
    high: float = 1.0

    def __post_init__(self) -> None:
        """Validate log-uniform distribution fields."""
        super(LogUniformDistribution, self).__post_init__()
        validate_positive_float(self.low, "low")
        if self.high <= self.low:
            raise ValueError(f"high must be greater than low, got high={self.high}, low={self.low}")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UniformDistribution(ParameterDistribution):
    """Distribution for uniformly distributed parameters.

    Attributes:
        type: Distribution type discriminator.
        low: Minimum value for the parameter.
        high: Maximum value for the parameter (must be > low).
        q: Quantization step size (must be > 0 if set).
        log_scale: Whether to sample in log scale.
    """

    type: Literal["uniform"] = "uniform"
    low: float | int = 0.0
    high: float | int = 1.0
    q: float | None = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        """Validate uniform distribution fields."""
        super(UniformDistribution, self).__post_init__()
        if self.high <= self.low:
            raise ValueError(f"high must be greater than low, got high={self.high}, low={self.low}")
        if self.q is not None:
            validate_positive_float(self.q, "q")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ChoiceDistribution(ParameterDistribution):
    """Distribution for discrete choice parameters.

    Attributes:
        type: Distribution type discriminator.
        choices: Tuple of choices for this parameter (must be non-empty).
        weights: Optional tuple of weights for each choice. If provided,
                 must match the length of choices, be non-negative, and sum to > 0.
    """

    type: Literal["choice"] = "choice"
    choices: tuple[Any, ...] = ()
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate choice distribution fields."""
        super(ChoiceDistribution, self).__post_init__()
        if not self.choices:
            raise ValueError("choices must not be empty")

        if self.weights is not None:
            if len(self.weights) != len(self.choices):
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match "
                    f"number of choices ({len(self.choices)})"
                )
            if any(w < 0 for w in self.weights):
                raise ValueError("weights must be non-negative")
            if sum(self.weights) == 0:
                raise ValueError("sum of weights must be greater than zero")


# Union of all distribution types for the search_space field
Distribution = (
    CategoricalDistribution | UniformDistribution | ChoiceDistribution | LogUniformDistribution
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class HyperparamSearchConfig(BaseConfig):
    """Configuration for hyperparameter search.

    Attributes:
        name: Name for this hyperparameter search configuration.
        search_type: Type of hyperparameter search strategy.
        num_trials: Number of trials to run (must be positive).
        max_parallel_trials: Maximum parallel trials (must be positive, <= num_trials).
        search_space: Mapping of parameter names to their distributions.
        early_stopping: Whether to use early stopping.
        patience: Trials without improvement before stopping (must be positive).
        pruning: Whether to use trial pruning.
        pruning_interval: Interval for pruning checks (must be positive).
        direction: Optimization direction.
        metric: Name of the metric to optimize.
        seed: Random seed for reproducibility.
        tracking_uri: Optional URI for experiment tracking service.
    """

    name: str = "hyperparam_search"
    search_type: SearchType = SearchType.RANDOM
    num_trials: int = 10
    max_parallel_trials: int = 1
    search_space: dict[str, Distribution] = dataclasses.field(default_factory=dict)
    early_stopping: bool = False
    patience: int = 10
    pruning: bool = False
    pruning_interval: int = 1
    direction: Literal["minimize", "maximize"] = "minimize"
    metric: str = "validation_loss"
    seed: int = 42
    tracking_uri: str | None = None

    def __post_init__(self) -> None:
        """Validate all fields.

        Uses DRY validation utilities where possible.
        Follows fail-fast principle: raises on first error.
        """
        super(HyperparamSearchConfig, self).__post_init__()

        validate_positive_int(self.num_trials, "num_trials")
        validate_positive_int(self.max_parallel_trials, "max_parallel_trials")
        validate_positive_int(self.patience, "patience")
        validate_positive_int(self.pruning_interval, "pruning_interval")

        if self.max_parallel_trials > self.num_trials:
            raise ValueError(
                f"max_parallel_trials ({self.max_parallel_trials}) "
                f"cannot be greater than num_trials ({self.num_trials})"
            )
