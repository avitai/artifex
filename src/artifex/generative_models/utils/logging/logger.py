"""
Base logger interface for Artifex.

This module provides a base logger class for logging training progress,
metrics, and other information during model training and evaluation.
"""

import csv
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp


class Logger(ABC):
    """
    Abstract base class for all loggers.

    This class defines the interface that all concrete logger implementations
    must follow. It provides methods for logging scalar values, images,
    histograms, and text.
    """

    def __init__(
        self,
        name: str,
        log_dir: str | None = None,
        level: int = logging.INFO,
    ):
        """
        Initialize the logger.

        Args:
            name: Name of the logger.
            log_dir: Directory to save logs. If None, logs are not saved to disk.
            level: Logging level (from the logging module).
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level

        # Create log directory if it doesn't exist
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        # Initialize Python's built-in logger for console output
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        # Remove any existing handlers to avoid duplicate logs
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

    def log(self, msg: str, level: int = logging.INFO) -> None:
        """
        Log a message with the specified level.

        Args:
            msg: Message to log.
            level: Logging level.
        """
        self._logger.log(level, msg)

    def debug(self, msg: str) -> None:
        """Log a debug message."""
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log an info message."""
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self._logger.error(msg)

    def critical(self, msg: str) -> None:
        """Log a critical message."""
        self._logger.critical(msg)

    @abstractmethod
    def log_scalar(self, name: str, value: float, step: int | None = None, **kwargs) -> None:
        """
        Log a scalar value.

        Args:
            name: Name of the scalar.
            value: Scalar value to log.
            step: Global step value to record (e.g., training iteration).
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def log_scalars(self, scalars: dict[str, float], step: int | None = None, **kwargs) -> None:
        """
        Log multiple scalar values.

        Args:
            scalars: dictionary of scalar names to values.
            step: Global step value to record (e.g., training iteration).
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def log_image(
        self,
        name: str,
        image: jax.Array | list[jax.Array],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log an image or list of images.

        Args:
            name: Name of the image.
            image: Image or list of images to log.
            step: Global step value to record.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def log_histogram(
        self,
        name: str,
        values: jax.Array | list[float],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log a histogram of values.

        Args:
            name: Name of the histogram.
            values: Values to build histogram.
            step: Global step value to record.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def log_text(self, name: str, text: str, step: int | None = None, **kwargs) -> None:
        """
        Log text.

        Args:
            name: Name of the text entry.
            text: Text to log.
            step: Global step value to record.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def log_hyperparams(self, params: dict[str, Any], **kwargs) -> None:
        """
        Log hyperparameters.

        Args:
            params: dictionary of hyperparameter names to values.
            **kwargs: Additional keyword arguments.
        """
        pass

    def close(self) -> None:
        """
        Close the logger and release any resources.

        This method should be called when the logger is no longer needed.
        """
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)


class ConsoleLogger(Logger):
    """Logger that outputs to the console."""

    def log_scalar(self, name: str, value: float, step: int | None = None, **kwargs) -> None:
        """Log a scalar value to console."""
        step_str = f"[Step {step}] " if step is not None else ""
        self.info(f"{step_str}{name}: {value:.6g}")

    def log_scalars(self, scalars: dict[str, float], step: int | None = None, **kwargs) -> None:
        """Log multiple scalar values to console."""
        step_str = f"[Step {step}] " if step is not None else ""
        scalar_str = ", ".join(f"{name}: {value:.6g}" for name, value in scalars.items())
        self.info(f"{step_str}{scalar_str}")

    def log_image(
        self,
        name: str,
        image: jax.Array | list[jax.Array],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Log image info to console (not the actual image)."""
        step_str = f"[Step {step}] " if step is not None else ""
        if isinstance(image, list):
            self.info(f"{step_str}Logged {len(image)} images for {name}")
        else:
            self.info(f"{step_str}Logged image {name} with shape {image.shape}")

    def log_histogram(
        self,
        name: str,
        values: jax.Array | list[float],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Log histogram info to console (not the actual histogram)."""
        step_str = f"[Step {step}] " if step is not None else ""
        values_arr = jnp.asarray(values)
        self.info(
            f"{step_str}Logged histogram {name}: "
            f"min={float(values_arr.min()):.6g}, max={float(values_arr.max()):.6g}, "
            f"mean={float(values_arr.mean()):.6g}, std={float(values_arr.std()):.6g}"
        )

    def log_text(self, name: str, text: str, step: int | None = None, **kwargs) -> None:
        """Log text to console."""
        step_str = f"[Step {step}] " if step is not None else ""
        self.info(f"{step_str}{name}:\n{text}")

    def log_hyperparams(self, params: dict[str, Any], **kwargs) -> None:
        """Log hyperparameters to console."""
        self.info("Hyperparameters:")
        for name, value in params.items():
            self.info(f"  {name}: {value}")


class FileLogger(Logger):
    """Logger that outputs to a file."""

    def __init__(
        self,
        name: str,
        log_dir: str,
        filename: str | None = None,
        level: int = logging.INFO,
    ):
        """
        Initialize the file logger.

        Args:
            name: Name of the logger.
            log_dir: Directory to save logs.
            filename: Log filename. If None, a default name is generated.
            level: Logging level.
        """
        super().__init__(name, log_dir, level)

        # FileLogger requires a valid log_dir
        if log_dir is None:
            raise ValueError("FileLogger requires a valid log_dir")

        # Create a unique filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.log"

        self.log_file = os.path.join(log_dir, filename)

        # Add file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # Create a separate CSV file for scalar metrics
        self.metrics_file = os.path.join(log_dir, f"{name}_{timestamp}_metrics.csv")
        self.metrics_header_written = False

        self.info(f"Log file created at {self.log_file}")
        self.info(f"Metrics file created at {self.metrics_file}")

    def log_scalar(self, name: str, value: float, step: int | None = None, **kwargs) -> None:
        """Log a scalar value to file."""
        step_str = f"[Step {step}] " if step is not None else ""
        self.info(f"{step_str}{name}: {value:.6g}")

        # Also log to CSV
        self._log_scalar_to_csv({name: value}, step)

    def log_scalars(self, scalars: dict[str, float], step: int | None = None, **kwargs) -> None:
        """Log multiple scalar values to file."""
        step_str = f"[Step {step}] " if step is not None else ""
        scalar_str = ", ".join(f"{name}: {value:.6g}" for name, value in scalars.items())
        self.info(f"{step_str}{scalar_str}")

        # Also log to CSV
        self._log_scalar_to_csv(scalars, step)

    def _log_scalar_to_csv(
        self,
        scalars: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log scalar values to CSV file."""

        # Prepare row data
        fieldnames = ["timestamp", "step", *scalars.keys()]
        row = {
            "timestamp": datetime.now().isoformat(),
            "step": step if step is not None else "",
        }
        row.update({k: str(v) for k, v in scalars.items()})

        # Write to CSV file
        file_exists = os.path.isfile(self.metrics_file)
        with open(self.metrics_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header if new file or first write
            if not file_exists or not self.metrics_header_written:
                writer.writeheader()
                self.metrics_header_written = True

            writer.writerow(row)

    def log_image(
        self,
        name: str,
        image: jax.Array | list[jax.Array],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Log image info to file and save image if possible."""
        step_str = f"[Step {step}] " if step is not None else ""

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Create images directory
            images_dir = os.path.join(self.log_dir, "images")  # type: ignore[arg-type]
            os.makedirs(images_dir, exist_ok=True)

            if isinstance(image, list):
                # Save a grid of images
                self.info(f"{step_str}Saving {len(image)} images for {name}")
                fig, axes = plt.subplots(1, len(image), figsize=(3 * len(image), 3))
                if len(image) == 1:
                    axes = [axes]

                for i, img in enumerate(image):
                    # Convert JAX array to numpy for matplotlib
                    img_np = np.asarray(img)
                    if len(img_np.shape) == 3 and img_np.shape[2] in [1, 3, 4]:
                        # RGB or RGBA image
                        axes[i].imshow(img_np)
                    else:
                        # Grayscale or other format
                        axes[i].imshow(img_np, cmap="gray")
                    axes[i].set_axis_off()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                step_suffix = f"_step{step}" if step is not None else ""
                filename = f"{name}_{timestamp}{step_suffix}.png"
                fig.savefig(os.path.join(images_dir, filename), bbox_inches="tight")
                plt.close(fig)

            else:
                # Save single image
                self.info(f"{step_str}Saving image {name} with shape {image.shape}")
                fig, ax = plt.subplots(figsize=(6, 6))

                # Convert JAX array to numpy for matplotlib
                image_np = np.asarray(image)
                if len(image_np.shape) == 3 and image_np.shape[2] in [1, 3, 4]:
                    # RGB or RGBA image
                    ax.imshow(image_np)
                else:
                    # Grayscale or other format
                    ax.imshow(image_np, cmap="gray")
                ax.set_axis_off()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                step_suffix = f"_step{step}" if step is not None else ""
                filename = f"{name}_{timestamp}{step_suffix}.png"
                fig.savefig(os.path.join(images_dir, filename), bbox_inches="tight")
                plt.close(fig)

        except ImportError:
            self.warning(f"{step_str}Failed to save image {name}: matplotlib not installed")
            if isinstance(image, list):
                self.info(f"{step_str}Logged {len(image)} images for {name}")
            else:
                self.info(f"{step_str}Logged image {name} with shape {image.shape}")

    def log_histogram(
        self,
        name: str,
        values: jax.Array | list[float],
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Log histogram info to file and save histogram if possible."""
        step_str = f"[Step {step}] " if step is not None else ""
        values_arr = jnp.asarray(values)
        self.info(
            f"{step_str}Logged histogram {name}: "
            f"min={float(values_arr.min()):.6g}, max={float(values_arr.max()):.6g}, "
            f"mean={float(values_arr.mean()):.6g}, std={float(values_arr.std()):.6g}"
        )

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Create histograms directory
            histograms_dir = os.path.join(self.log_dir, "histograms")  # type: ignore[arg-type]
            os.makedirs(histograms_dir, exist_ok=True)

            # Plot histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            # Convert JAX array to numpy for matplotlib
            values_np = np.asarray(values_arr)
            ax.hist(values_np.flatten(), bins=50)
            ax.set_title(f"Histogram of {name}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_suffix = f"_step{step}" if step is not None else ""
            filename = f"{name}_{timestamp}{step_suffix}.png"
            fig.savefig(os.path.join(histograms_dir, filename), bbox_inches="tight")
            plt.close(fig)

        except ImportError:
            self.warning(f"{step_str}Failed to save histogram {name}: matplotlib not installed")

    def log_text(self, name: str, text: str, step: int | None = None, **kwargs) -> None:
        """Log text to file."""
        step_str = f"[Step {step}] " if step is not None else ""
        self.info(f"{step_str}{name}:\n{text}")

        # Also save to separate text file for longer content
        try:
            # Create texts directory
            texts_dir = os.path.join(self.log_dir, "texts")  # type: ignore[arg-type]
            os.makedirs(texts_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_suffix = f"_step{step}" if step is not None else ""
            filename = f"{name}_{timestamp}{step_suffix}.txt"

            with open(os.path.join(texts_dir, filename), "w") as f:
                f.write(text)
        except Exception as e:
            self.warning(f"Failed to save text to file: {e}")

    def log_hyperparams(self, params: dict[str, Any], **kwargs) -> None:
        """Log hyperparameters to file."""
        self.info("Hyperparameters:")
        for name, value in params.items():
            self.info(f"  {name}: {value}")

        # Also save to separate file
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hyperparams_{timestamp}.txt"

            with open(os.path.join(self.log_dir, filename), "w") as f:  # type: ignore[arg-type]
                for name, value in params.items():
                    f.write(f"{name}: {value}\n")
        except Exception as e:
            self.warning(f"Failed to save hyperparameters to file: {e}")


def create_logger(
    name: str,
    log_dir: str | None = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
    level: int = logging.INFO,
) -> Logger:
    """
    Create a logger with console and/or file output.

    Args:
        name: Name of the logger.
        log_dir: Directory to save logs. If None and log_to_file is True,
            logs are saved to './logs/{name}'.
        log_to_console: Whether to log to console.
        log_to_file: Whether to log to file.
        level: Logging level.

    Returns:
        A Logger instance.
    """
    if log_to_file and log_dir is None:
        log_dir = os.path.join("logs", name)

    if log_to_console and not log_to_file:
        return ConsoleLogger(name, log_dir, level)
    elif log_to_file:
        if log_dir is None:
            raise ValueError("log_dir must be provided when log_to_file is True")
        return FileLogger(name, log_dir, level=level)
    else:
        # Default to console logger if both are False
        return ConsoleLogger(name, log_dir, level)
