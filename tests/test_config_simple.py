"""Simple test to isolate the configuration issue."""

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Direct import to test
from enum import Enum

from pydantic import BaseModel, Field


class ConfigurationType(str, Enum):
    MODEL = "model"
    EXPERIMENT = "experiment"


class BaseConfig(BaseModel):
    name: str = Field(..., description="Name")
    type: ConfigurationType = Field(..., description="Type")

    class Config:
        extra = "forbid"


class ModelConfig(BaseConfig):
    type: ConfigurationType = Field(ConfigurationType.MODEL, frozen=True)
    model_class: str = Field(..., description="Model class")


# This is where the error occurs
class ExperimentConfiguration(BaseConfig):
    type: ConfigurationType = Field(ConfigurationType.EXPERIMENT, frozen=True)
    # The issue is likely here - model_config conflicts with pydantic's model_config
    model_cfg: ModelConfig | str = Field(..., description="Model configuration")


def test_simple():
    """Test simple configuration creation."""
    model = ModelConfig(name="test", model_class="Test")
    print(f"Model created: {model.name}")

    exp = ExperimentConfiguration(name="exp", model_cfg=model)
    print(f"Experiment created: {exp.name}")


if __name__ == "__main__":
    test_simple()
