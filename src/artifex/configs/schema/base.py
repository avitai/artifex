from pydantic import BaseModel, Field

# Import BaseConfig from the new location in core protocols
from artifex.generative_models.core.protocols.configuration import BaseConfig


# Re-export BaseConfig for backward compatibility
__all__ = ["BaseConfig", "ExperimentConfig"]


class ExperimentConfig(BaseModel):
    """Container for a complete experiment configuration."""

    experiment_name: str = Field(..., description="Name of the experiment")
    seed: int = Field(42, description="Random seed for reproducibility")
    output_dir: str = Field("./outputs", description="Directory for saving outputs")

    # References to other configurations
    model_config_ref: str = Field(..., description="Reference to model configuration")
    data_config: str = Field(..., description="Reference to data configuration")
    training_config: str = Field(..., description="Reference to training configuration")
    inference_config: str | None = Field(None, description="Reference to inference configuration")

    # Optional logging and monitoring settings
    log_level: str = Field("INFO", description="Logging level")
    use_wandb: bool = Field(False, description="Whether to use Weights & Biases for logging")
    wandb_project: str | None = Field(None, description="Weights & Biases project name")

    model_config = {
        "extra": "forbid",
    }
