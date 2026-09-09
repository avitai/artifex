"""Runtime-oriented device helpers for generative-model workflows.

Device identity, placement and the hardware batch-size table are substrax's
(``substrax.devices``); this module keeps the one artifex-specific rule, scaling
the hardware's batch size by model size.
"""

from __future__ import annotations

from substrax.devices import get_batch_size_recommendation


__all__ = ["get_recommended_batch_size"]

# Parameter counts at which the hardware's batch size is halved or doubled.
LARGE_MODEL_PARAMETERS = 100_000_000
SMALL_MODEL_PARAMETERS = 1_000_000


def get_recommended_batch_size(model_params: int, base_batch_size: int | None = None) -> int:
    """Return a batch size for the detected hardware, scaled by model size.

    Args:
        model_params: Number of trainable parameters in the model.
        base_batch_size: Starting point. ``None`` takes substrax's optimal batch
            size for the detected hardware.

    Returns:
        The starting point halved above 100M parameters and doubled below 1M,
        never below 1 and never above the hardware's estimated memory ceiling
        when substrax states one.
    """
    recommendation = get_batch_size_recommendation()
    base = recommendation.optimal_batch_size if base_batch_size is None else base_batch_size
    if model_params > LARGE_MODEL_PARAMETERS:
        multiplier = 0.5
    elif model_params < SMALL_MODEL_PARAMETERS:
        multiplier = 2.0
    else:
        multiplier = 1.0
    batch_size = max(1, int(base * multiplier))
    ceiling = recommendation.max_memory_batch_size
    return batch_size if ceiling is None else min(batch_size, ceiling)
