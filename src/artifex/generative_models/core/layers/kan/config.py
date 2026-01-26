"""KAN layer configuration.

Provides a frozen dataclass for configuring Kolmogorov-Arnold Network layers.
This is a layer-level config (no BaseConfig inheritance).
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class KANConfig:
    """Configuration for KAN layers.

    Attributes:
        k: B-spline order (polynomial degree). Only used by spline layers.
        grid_intervals: Number of grid intervals (G in jaxKAN).
        grid_range: Initial range for grid endpoints.
        grid_e: Grid mixing parameter. 1.0 = uniform, 0.0 = sample-dependent.
        degree: Polynomial degree for basis layers (D in jaxKAN).
        residual: Whether to use a residual activation function.
        external_weights: Whether to apply external edge weights.
        add_bias: Whether to include bias terms.
        init_scheme: Initialization scheme configuration.
    """

    k: int = 3
    grid_intervals: int = 3
    grid_range: tuple[float, float] = (-1.0, 1.0)
    grid_e: float = 0.05
    degree: int = 5
    residual: bool = True
    external_weights: bool = True
    add_bias: bool = True
    init_scheme: str = "default"

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.k < 0:
            raise ValueError(f"Spline order k must be >= 0, got {self.k}")
        if self.grid_intervals < 1:
            raise ValueError(f"grid_intervals must be >= 1, got {self.grid_intervals}")
        if self.grid_range[0] >= self.grid_range[1]:
            raise ValueError(f"grid_range[0] must be < grid_range[1], got {self.grid_range}")
        if not 0.0 <= self.grid_e <= 1.0:
            raise ValueError(f"grid_e must be in [0, 1], got {self.grid_e}")
        if self.degree < 1:
            raise ValueError(f"degree must be >= 1, got {self.degree}")
        valid_schemes = {
            "default",
            "power",
            "lecun",
            "glorot",
            "glorot_fine",
            "custom",
        }
        if self.init_scheme not in valid_schemes:
            raise ValueError(
                f"Unknown init_scheme '{self.init_scheme}'. Valid: {sorted(valid_schemes)}"
            )
