"""Standalone test for hyperparameter search configuration."""

from enum import Enum
from typing import Any

import pytest
from pydantic import BaseModel, Field, model_validator


class BaseConfig(BaseModel):
    """Base configuration class."""

    name: str = Field("test_config", description="Configuration name")

    class Config:
        """Pydantic config."""

        extra = "forbid"


class SearchType(str, Enum):
    """Type of hyperparameter search."""

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


class ParameterDistribution(BaseModel):
    """Base class for parameter distributions."""

    param_type: str


class UniformDistribution(ParameterDistribution):
    """Uniform distribution for continuous parameters."""

    param_type: str = "uniform"
    low: float
    high: float

    @model_validator(mode="after")
    def high_greater_than_low(self):
        """Validate that high is greater than low."""
        if self.high <= self.low:
            raise ValueError("high must be greater than low")
        return self


class CategoricalDistribution(ParameterDistribution):
    """Categorical distribution for discrete choices."""

    param_type: str = "categorical"
    choices: list[Any]

    @model_validator(mode="after")
    def validate_choices(self):
        """Validate that choices is not empty."""
        if not self.choices:
            raise ValueError("choices cannot be empty")
        return self


class ChoiceDistribution(ParameterDistribution):
    """Simple choice distribution for values."""

    param_type: str = "choice"
    values: list[int | float | str | bool]

    @model_validator(mode="after")
    def validate_values(self):
        """Validate that values is not empty."""
        if not self.values:
            raise ValueError("values cannot be empty")
        return self


class HyperparamSearchConfig(BaseConfig):
    """Configuration for hyperparameter search."""

    search_type: SearchType = Field(SearchType.GRID, description="Type of hyperparameter search")
    num_trials: int = Field(10, description="Number of trials for the search")
    max_parallel_trials: int = Field(1, description="Maximum number of parallel trials")
    parameters: dict[str, ParameterDistribution] = Field(
        {}, description="Parameter distributions to search over"
    )

    @model_validator(mode="after")
    def validate_values(self):
        """Validate config values."""
        # Validate positive values
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive")

        if self.max_parallel_trials <= 0:
            raise ValueError("max_parallel_trials must be positive")

        # Validate max_parallel_trials
        if self.max_parallel_trials > self.num_trials:
            raise ValueError("max_parallel_trials cannot be greater than num_trials")

        return self


def test_search_type_enum():
    """Test search type enum values."""
    assert SearchType.GRID.value == "grid"
    assert SearchType.RANDOM.value == "random"
    assert SearchType.BAYESIAN.value == "bayesian"


def test_hyperparam_search_config_instantiation():
    """Test that HyperparamSearchConfig can be instantiated with valid values."""
    config = HyperparamSearchConfig(
        name="test_hyperparam_search",
        search_type=SearchType.RANDOM,
        num_trials=20,
        max_parallel_trials=4,
        parameters={
            "learning_rate": UniformDistribution(low=0.0001, high=0.1),
            "batch_size": ChoiceDistribution(values=[16, 32, 64, 128]),
            "optimizer": CategoricalDistribution(choices=["adam", "sgd", "adamw"]),
        },
    )

    assert config.name == "test_hyperparam_search"
    assert config.search_type == SearchType.RANDOM
    assert config.num_trials == 20
    assert config.max_parallel_trials == 4
    assert len(config.parameters) == 3
    assert isinstance(config.parameters["learning_rate"], UniformDistribution)
    assert isinstance(config.parameters["batch_size"], ChoiceDistribution)
    assert isinstance(config.parameters["optimizer"], CategoricalDistribution)


def test_uniform_distribution():
    """Test uniform distribution."""
    dist = UniformDistribution(low=0.0, high=1.0)
    assert dist.param_type == "uniform"
    assert dist.low == 0.0
    assert dist.high == 1.0

    # Test validation
    with pytest.raises(ValueError):
        UniformDistribution(low=2.0, high=1.0)


def test_categorical_distribution():
    """Test categorical distribution."""
    dist = CategoricalDistribution(choices=["a", "b", "c"])
    assert dist.param_type == "categorical"
    assert dist.choices == ["a", "b", "c"]

    # Test validation
    with pytest.raises(ValueError):
        CategoricalDistribution(choices=[])


def test_choice_distribution():
    """Test choice distribution."""
    dist = ChoiceDistribution(values=[1, 2, 3])
    assert dist.param_type == "choice"
    assert dist.values == [1, 2, 3]

    # Test validation
    with pytest.raises(ValueError):
        ChoiceDistribution(values=[])
