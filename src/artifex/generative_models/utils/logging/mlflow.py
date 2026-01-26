"""
MLflow logger implementation for the Artifex library.

This module provides a logger implementation that integrates with MLflow
for experiment tracking, including metrics, parameters, artifacts, and models.
"""

import os
from typing import Any

import numpy as np

from artifex.generative_models.utils.logging.logger import Logger


class MLFlowLogger(Logger):
    """
    Logger implementation that integrates with MLflow.

    This logger logs metrics and artifacts to MLflow, enabling experiment
    tracking and comparison.
    """

    def __init__(
        self,
        name: str,
        log_dir: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        level: int = 20,  # INFO level
        console_log: bool = True,
    ):
        """
        Initialize the MLflow logger.

        Args:
            name: Name of the logger.
            log_dir: Directory to save logs locally (in addition to MLflow).
            experiment_name: Name of the MLflow experiment.
                If None, uses the logger name.
            run_name: Name of the MLflow run. If None, MLflow auto-generates one.
            run_id: ID of an existing MLflow run to continue logging to.
                If provided, experiment_name and run_name are ignored.
            tracking_uri: URI of the MLflow tracking server.
                If None, uses the default set in the environment.
            registry_uri: URI of the MLflow model registry.
                If None, uses the same as tracking_uri.
            level: Logging level.
            console_log: Whether to also log to console.

        Raises:
            ImportError: If mlflow is not installed.
        """
        super().__init__(name, log_dir, level)

        try:
            import mlflow
        except ImportError as err:
            self.error("MLflow is not installed. Install it with `pip install mlflow`.")
            raise ImportError(
                "MLflow is required for MLFlowLogger but not installed. "
                "Install it with `pip install mlflow`."
            ) from err

        self.mlflow = mlflow

        # Set tracking URI if provided
        if tracking_uri is not None:
            self.mlflow.set_tracking_uri(tracking_uri)

        # Set registry URI if provided
        if registry_uri is not None:
            self.mlflow.set_registry_uri(registry_uri)

        # Set up experiment
        experiment_name = experiment_name or name
        self.experiment_name = experiment_name
        self.active_run = None

        # Start or resume run
        if run_id is not None:
            self.active_run = self.mlflow.start_run(run_id=run_id)
            self.info(f"Resumed MLflow run: {run_id}")
        else:
            experiment = self.mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.mlflow.create_experiment(experiment_name)
                self.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                self.info(f"Using existing MLflow experiment: {experiment_name}")

            self.active_run = self.mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
            self.info(f"Started new MLflow run: {self.active_run.info.run_id}")

        self.run_id = self.active_run.info.run_id

        # Configure local artifact logging
        if log_dir is not None:
            self.artifact_dir = os.path.join(log_dir, "artifacts")
            os.makedirs(self.artifact_dir, exist_ok=True)
        else:
            self.artifact_dir = None

    def log_scalar(
        self,
        name: str,
        value: float,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log a scalar value to MLflow.

        Args:
            name: Name of the scalar.
            value: Scalar value to log.
            step: Global step value to record.
            **kwargs: Additional keyword arguments.
        """
        self.mlflow.log_metric(name, value, step=step)

        step_str = f"[Step {step}] " if step is not None else ""
        self.info(f"{step_str}{name}: {value:.6g}")

    def log_scalars(
        self,
        scalars: dict[str, float],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log multiple scalar values to MLflow.

        Args:
            scalars: Dictionary of scalar names to values.
            step: Global step value to record.
            **kwargs: Additional keyword arguments.
        """
        metrics = {
            name: value for name, value in scalars.items() if isinstance(value, (int, float))
        }
        self.mlflow.log_metrics(metrics, step=step)

        step_str = f"[Step {step}] " if step is not None else ""
        scalar_str = ", ".join(f"{name}: {value:.6g}" for name, value in scalars.items())
        self.info(f"{step_str}{scalar_str}")

    def log_image(
        self,
        name: str,
        image: np.ndarray | list[np.ndarray],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log an image or list of images to MLflow.

        Args:
            name: Name of the image.
            image: Image or list of images to log.
            step: Global step value to record.
            **kwargs: Additional keyword arguments.
        """
        step_str = f"_step{step}" if step is not None else ""
        log_step_str = f"[Step {step}] " if step is not None else ""

        try:
            import matplotlib.pyplot as plt

            # Create temporary directory for images
            if self.artifact_dir:
                images_dir = os.path.join(self.artifact_dir, "images")
                os.makedirs(images_dir, exist_ok=True)

            if isinstance(image, list):
                # Handle multiple images
                self.info(f"{log_step_str}Logging {len(image)} images for {name}")

                # Create a grid of images
                fig, axes = plt.subplots(1, len(image), figsize=(3 * len(image), 3))
                if len(image) == 1:
                    axes = [axes]

                for i, img in enumerate(image):
                    if len(img.shape) == 3 and img.shape[2] in [1, 3, 4]:
                        axes[i].imshow(img)
                    else:
                        axes[i].imshow(img, cmap="gray")
                    axes[i].set_axis_off()

                # Save and log the figure
                if self.artifact_dir:
                    filename = f"{name}{step_str}.png"
                    filepath = os.path.join(images_dir, filename)
                    fig.savefig(filepath, bbox_inches="tight")
                    self.mlflow.log_artifact(filepath, "images")
                else:
                    # Save to a temporary file if no artifact_dir
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                        fig.savefig(tmp.name, bbox_inches="tight")
                        self.mlflow.log_artifact(tmp.name, "images")

                plt.close(fig)

            else:
                # Handle single image
                self.info(f"{log_step_str}Logging image {name} with shape {image.shape}")

                # Convert JAX arrays to numpy
                if hasattr(image, "device_buffer"):
                    image = np.array(image)

                # Create and save the image
                fig, ax = plt.subplots(figsize=(6, 6))

                if len(image.shape) == 3 and image.shape[2] in [1, 3, 4]:
                    ax.imshow(image)
                else:
                    ax.imshow(image, cmap="gray")
                ax.set_axis_off()

                # Save and log the figure
                if self.artifact_dir:
                    filename = f"{name}{step_str}.png"
                    filepath = os.path.join(images_dir, filename)
                    fig.savefig(filepath, bbox_inches="tight")
                    self.mlflow.log_artifact(filepath, "images")
                else:
                    # Save to a temporary file if no artifact_dir
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                        fig.savefig(tmp.name, bbox_inches="tight")
                        self.mlflow.log_artifact(tmp.name, "images")

                plt.close(fig)

        except ImportError as e:
            self.warning(f"Failed to log image: {e}")
            if isinstance(image, list):
                self.info(f"{log_step_str}Logged {len(image)} images for {name}")
            else:
                self.info(f"{log_step_str}Logged image {name} with shape {image.shape}")

    def log_histogram(
        self,
        name: str,
        values: np.ndarray | list[float],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log a histogram of values to MLflow.

        Args:
            name: Name of the histogram.
            values: Values to build histogram.
            step: Global step value to record.
            **kwargs: Additional keyword arguments.
        """
        step_str = f"_step{step}" if step is not None else ""
        log_step_str = f"[Step {step}] " if step is not None else ""
        values_arr = np.asarray(values)

        # Log basic statistics as metrics
        self.log_scalars(
            {
                f"{name}/min": float(values_arr.min()),
                f"{name}/max": float(values_arr.max()),
                f"{name}/mean": float(values_arr.mean()),
                f"{name}/std": float(values_arr.std()),
                f"{name}/median": float(np.median(values_arr)),
            },
            step=step,
        )

        try:
            import matplotlib.pyplot as plt

            # Create histogram plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(values_arr.flatten(), bins=50)
            ax.set_title(f"Histogram of {name}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

            # Save and log the histogram
            if self.artifact_dir:
                histograms_dir = os.path.join(self.artifact_dir, "histograms")
                os.makedirs(histograms_dir, exist_ok=True)
                filename = f"{name}{step_str}.png"
                filepath = os.path.join(histograms_dir, filename)
                fig.savefig(filepath)
                self.mlflow.log_artifact(filepath, "histograms")
            else:
                # Save to a temporary file if no artifact_dir
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                    fig.savefig(tmp.name)
                    self.mlflow.log_artifact(tmp.name, "histograms")

            plt.close(fig)

        except ImportError:
            self.warning(f"{log_step_str}Failed to create histogram plot for {name}")

    def log_text(
        self,
        name: str,
        text: str,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log text to MLflow.

        Args:
            name: Name of the text entry.
            text: Text to log.
            step: Global step value to record.
            **kwargs: Additional keyword arguments.
        """
        step_str = f"_step{step}" if step is not None else ""
        log_step_str = f"[Step {step}] " if step is not None else ""

        # Log to file and then log as artifact
        if self.artifact_dir:
            texts_dir = os.path.join(self.artifact_dir, "texts")
            os.makedirs(texts_dir, exist_ok=True)
            filename = f"{name}{step_str}.txt"
            filepath = os.path.join(texts_dir, filename)
            with open(filepath, "w") as f:
                f.write(text)
            self.mlflow.log_artifact(filepath, "texts")
        else:
            # Save to a temporary file if no artifact_dir
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as tmp:
                tmp.write(text)
                tmp.flush()
                self.mlflow.log_artifact(tmp.name, "texts")

        self.info(f"{log_step_str}Logged text for {name}")

    def log_hyperparams(self, params: dict[str, Any], **kwargs) -> None:
        """
        Log hyperparameters to MLflow.

        Args:
            params: Dictionary of hyperparameter names to values.
            **kwargs: Additional keyword arguments.
        """
        # MLflow only accepts simple types like str, int, float, bool
        filtered_params = {}
        for name, value in params.items():
            if isinstance(value, (str, int, float, bool)):
                filtered_params[name] = value
            else:
                # Convert complex types to string representation
                try:
                    filtered_params[name] = str(value)
                except Exception:
                    self.warning(f"Could not convert parameter {name} to string")

        # Log the parameters
        self.mlflow.log_params(filtered_params)

        self.info(f"Logged {len(filtered_params)} hyperparameters to MLflow")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        **kwargs,
    ) -> None:
        """
        Log a model to MLflow.

        Args:
            model: The model to log.
            artifact_path: Path within the MLflow run's artifact directory.
            **kwargs: Additional keyword arguments passed to mlflow.log_model.
        """
        self.mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=model,
            **kwargs,
        )
        self.info(f"Logged model to MLflow at {artifact_path}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """
        Log an artifact to MLflow.

        Args:
            local_path: Path to the file to log.
            artifact_path: Path within the MLflow run's artifact directory.
        """
        self.mlflow.log_artifact(local_path, artifact_path)
        self.info(f"Logged artifact from {local_path} to MLflow")

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        """
        Log all artifacts in a directory to MLflow.

        Args:
            local_dir: Path to the directory containing artifacts to log.
            artifact_path: Path within the MLflow run's artifact directory.
        """
        self.mlflow.log_artifacts(local_dir, artifact_path)
        self.info(f"Logged artifacts from {local_dir} to MLflow")

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.active_run:
            self.mlflow.end_run()
            self.info(f"Ended MLflow run: {self.run_id}")
            self.active_run = None

    def close(self) -> None:
        """Close the logger and end the MLflow run."""
        self.end_run()
        super().close()
