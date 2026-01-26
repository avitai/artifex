"""Simplified standalone test for hyperparameter configuration."""

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


class HyperparameterType(str, Enum):
    """Type of hyperparameter."""

    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class HyperparameterConfig(BaseConfig):
    """Simple hyperparameter configuration."""

    param_name: str = Field(..., description="Name of the hyperparameter")
    param_type: HyperparameterType = Field(..., description="Type of the hyperparameter")
    value: Any = Field(None, description="Current value of the hyperparameter")

    @model_validator(mode="after")
    def validate_value_type(self):
        """Validate that the value matches the declared type."""
        if self.value is None:
            return self

        if self.param_type == HyperparameterType.INT and not isinstance(self.value, int):
            raise ValueError(f"{self.param_name} should be an integer")

        if self.param_type == HyperparameterType.FLOAT and not isinstance(self.value, float):
            raise ValueError(f"{self.param_name} should be a float")

        if self.param_type == HyperparameterType.BOOLEAN and not isinstance(self.value, bool):
            raise ValueError(f"{self.param_name} should be a boolean")

        return self


def test_hyperparameter_enum():
    """Test hyperparameter type enum values."""
    assert HyperparameterType.INT.value == "int"
    assert HyperparameterType.FLOAT.value == "float"
    assert HyperparameterType.CATEGORICAL.value == "categorical"
    assert HyperparameterType.BOOLEAN.value == "boolean"


def test_hyperparameter_config_instantiation():
    """Test that HyperparameterConfig can be instantiated with valid values."""
    # Integer parameter
    int_param = HyperparameterConfig(
        name="int_param",
        param_name="batch_size",
        param_type=HyperparameterType.INT,
        value=32,
    )
    assert int_param.param_name == "batch_size"
    assert int_param.param_type == HyperparameterType.INT
    assert int_param.value == 32

    # Float parameter
    float_param = HyperparameterConfig(
        name="float_param",
        param_name="learning_rate",
        param_type=HyperparameterType.FLOAT,
        value=0.001,
    )
    assert float_param.param_name == "learning_rate"
    assert float_param.param_type == HyperparameterType.FLOAT
    assert float_param.value == 0.001


def test_hyperparameter_validation():
    """Test validation of hyperparameter values."""
    # Should raise error for wrong type
    with pytest.raises(ValueError):
        HyperparameterConfig(
            param_name="batch_size",
            param_type=HyperparameterType.INT,
            value=32.5,
        )

    with pytest.raises(ValueError):
        HyperparameterConfig(
            param_name="learning_rate",
            param_type=HyperparameterType.FLOAT,
            value=1,
        )

    with pytest.raises(ValueError):
        HyperparameterConfig(
            param_name="use_bias",
            param_type=HyperparameterType.BOOLEAN,
            value="true",
        )

    # None value should be accepted for any type
    null_param = HyperparameterConfig(
        param_name="param", param_type=HyperparameterType.INT, value=None
    )
    assert null_param.value is None


def test_categorical_validation():
    """Test validation for categorical parameters."""
    # Create a categorical parameter with valid choices
    cat_param = HyperparameterConfig(
        param_name="optimizer",
        param_type=HyperparameterType.CATEGORICAL,
        value="adam",
    )
    assert cat_param.param_name == "optimizer"
    assert cat_param.param_type == HyperparameterType.CATEGORICAL
    assert cat_param.value == "adam"

    # Categorical parameters accept any type of value
    cat_param2 = HyperparameterConfig(
        param_name="optimizer_config",
        param_type=HyperparameterType.CATEGORICAL,
        value={"lr": 0.01, "beta1": 0.9},
    )
    assert cat_param2.value == {"lr": 0.01, "beta1": 0.9}
