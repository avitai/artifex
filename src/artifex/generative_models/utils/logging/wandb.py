"""
Weights & Biases logger implementation for the Artifex library.

This module provides a logger implementation that integrates with Weights & Biases
for experiment tracking, visualization, and collaboration.
"""

import os
from typing import Any, Literal

import numpy as np

from artifex.generative_models.utils.logging.logger import Logger


class WandbLogger(Logger):
    """
    Logger implementation that integrates with Weights & Biases.

    This logger logs metrics, media, and other artifacts to W&B, enabling
    experiment tracking, visualization, and collaboration.
    """

    def __init__(
        self,
        name: str,
        project: str,
        entity: str | None = None,
        log_dir: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        level: int = 20,  # INFO level
        console_log: bool = True,
        resume: Literal["allow", "never", "must", "auto"] | bool | None = None,
        anonymous: Literal["never", "allow", "must"] | None = None,
    ):
        """
        Initialize the Weights & Biases logger.

        Args:
            name: Name of the W&B run.
            project: Name of the W&B project.
            entity: Username or team name where the project is located.
            log_dir: Directory to save logs locally (in addition to W&B).
            config: Dictionary of hyperparameters to log.
            tags: list of tags for the run.
            notes: Notes about the run.
            level: Logging level.
            console_log: Whether to also log to console.
            resume: Whether to resume a previous run.
                Can be "auto", "allow", "must", or None.
            anonymous: Whether to use an anonymous account.
                Can be "must", "allow", or None.

        Raises:
            ImportError: If wandb is not installed.
        """
        super().__init__(name, log_dir, level)

        try:
            import wandb
        except ImportError as err:
            self.error("Weights & Biases is not installed. Install with `pip install wandb`.")
            raise ImportError(
                "Weights & Biases is required for WandbLogger but not installed. "
                "Install it with `pip install wandb`."
            ) from err

        self.wandb = wandb
        self.run = None

        # Initialize W&B run
        self.run = self.wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            dir=log_dir,
            resume=resume,
            anonymous=anonymous,
        )

        self.info(f"Initialized W&B run: {self.run.name} (ID: {self.run.id})")

        # Save run ID for potential resuming
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "wandb_run_id.txt"), "w") as f:
                f.write(self.run.id)

    def log_scalar(
        self,
        name: str,
        value: float,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log a scalar value to W&B.

        Args:
            name: Name of the scalar.
            value: Scalar value to log.
            step: Global step value to record. If None, W&B uses internal step.
            **kwargs: Additional keyword arguments passed to wandb.log().
        """
        log_dict = {name: value}
        self.wandb.log(log_dict, step=step, **kwargs)

        step_str = f"[Step {step}] " if step is not None else ""
        self.info(f"{step_str}{name}: {value:.6g}")

    def log_scalars(
        self,
        scalars: dict[str, float],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log multiple scalar values to W&B.

        Args:
            scalars: Dictionary of scalar names to values.
            step: Global step value to record. If None, W&B uses internal step.
            **kwargs: Additional keyword arguments passed to wandb.log().
        """
        # Filter out non-numeric values
        log_dict = {
            name: value for name, value in scalars.items() if isinstance(value, (int, float))
        }

        self.wandb.log(log_dict, step=step, **kwargs)

        step_str = f"[Step {step}] " if step is not None else ""
        scalar_str = ", ".join(f"{name}: {value:.6g}" for name, value in log_dict.items())
        self.info(f"{step_str}{scalar_str}")

    def log_image(
        self,
        name: str,
        image: np.ndarray | list[np.ndarray],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log an image or list of images to W&B.

        Args:
            name: Name of the image.
            image: Image or list of images to log.
            step: Global step value to record. If None, W&B uses internal step.
            **kwargs: Additional keyword arguments passed to wandb.log().
        """
        try:
            # Convert single image to list for uniform handling
            if not isinstance(image, list):
                image = [image]

            # Convert JAX arrays to numpy if needed
            images = []
            for img in image:
                if hasattr(img, "device_buffer"):
                    img = np.array(img)
                images.append(img)

            # Log the images
            self.wandb.log(
                {name: [self.wandb.Image(img) for img in images]},
                step=step,
                **kwargs,
            )

            step_str = f"[Step {step}] " if step is not None else ""
            self.info(f"{step_str}Logged {len(images)} images for {name}")

        except Exception as e:
            step_str = f"[Step {step}] " if step is not None else ""
            self.warning(f"{step_str}Failed to log images: {e}")

    def log_histogram(
        self,
        name: str,
        values: np.ndarray | list[float],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log a histogram of values to W&B.

        Args:
            name: Name of the histogram.
            values: Values to build histogram.
            step: Global step value to record. If None, W&B uses internal step.
            **kwargs: Additional keyword arguments passed to wandb.log().
        """
        values_arr = np.asarray(values)

        # Log basic statistics as metrics
        statistics = {
            f"{name}/min": float(values_arr.min()),
            f"{name}/max": float(values_arr.max()),
            f"{name}/mean": float(values_arr.mean()),
            f"{name}/std": float(values_arr.std()),
            f"{name}/median": float(np.median(values_arr)),
        }

        # Log the histogram
        histogram_data = self.wandb.Histogram(values_arr.tolist())
        log_dict = {**statistics, name: histogram_data}
        self.wandb.log(log_dict, step=step, **kwargs)

        step_str = f"[Step {step}] " if step is not None else ""
        self.info(
            f"{step_str}Logged histogram {name}: "
            f"min={values_arr.min():.6g}, max={values_arr.max():.6g}, "
            f"mean={values_arr.mean():.6g}, std={values_arr.std():.6g}"
        )

    def log_text(
        self,
        name: str,
        text: str,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log text to W&B.

        Args:
            name: Name of the text entry.
            text: Text to log.
            step: Global step value to record. If None, W&B uses internal step.
            **kwargs: Additional keyword arguments passed to wandb.log().
        """
        self.wandb.log({name: self.wandb.Html(f"<pre>{text}</pre>")}, step=step, **kwargs)

        step_str = f"[Step {step}] " if step is not None else ""
        self.info(f"{step_str}Logged text for {name}")

    def log_hyperparams(
        self,
        params: dict[str, Any],
        **kwargs,
    ) -> None:
        """
        Log hyperparameters to W&B.

        Args:
            params: Dictionary of hyperparameter names to values.
            **kwargs: Additional keyword arguments.
        """
        # Update W&B config with new parameters
        if self.run is not None:
            for name, value in params.items():
                self.run.config[name] = value

        self.info(f"Logged {len(params)} hyperparameters to W&B")

    def log_code(self, root: str | None = None, **kwargs) -> None:
        """
        Log code to W&B for experiment reproducibility.

        Args:
            root: Root directory of the code to log.
                If None, the current directory is used.
            **kwargs: Additional keyword arguments passed to wandb.run.log_code().
        """
        try:
            if self.run is not None:
                self.run.log_code(root=root, **kwargs)
            self.info("Logged code to W&B for reproducibility")
        except Exception as e:
            self.warning(f"Failed to log code to W&B: {e}")

    def log_model(
        self,
        model_path: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Log a model to W&B Artifacts.

        Args:
            model_path: Path to the model file or directory.
            name: Name of the artifact. If None, uses "{run_name}_model".
            metadata: Dictionary of metadata to log with the model.
            **kwargs: Additional keyword arguments passed to
                wandb.Artifact.add_file() or add_dir().
        """
        try:
            if self.run is not None:
                name = name or f"{self.run.name}_model"
                artifact = self.wandb.Artifact(
                    name=name,
                    type="model",
                    metadata=metadata,
                )

                if os.path.isdir(model_path):
                    artifact.add_dir(model_path, **kwargs)
                else:
                    artifact.add_file(model_path, **kwargs)

                self.run.log_artifact(artifact)
                self.info(f"Logged model from {model_path} to W&B Artifacts")
        except Exception as e:
            self.warning(f"Failed to log model to W&B: {e}")

    def finish(self, exit_code: int = 0) -> None:
        """
        Finish the W&B run.

        Args:
            exit_code: Exit code to report to W&B.
        """
        if self.run:
            self.wandb.finish(exit_code=exit_code)
            self.info(f"Finished W&B run: {self.run.name} (ID: {self.run.id})")
            self.run = None

    def close(self) -> None:
        """Close the logger and finish the W&B run."""
        self.finish()
        super().close()
