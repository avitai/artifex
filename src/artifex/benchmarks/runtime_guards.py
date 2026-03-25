"""Shared truthfulness guards for retained benchmark demo surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEMO_MODE_KEYS = (
    "demo_mode",
    "allow_demo_mode",
    "allow_mock",
    "allow_mock_data",
    "allow_mock_metric",
    "allow_synthetic_data",
    "allow_synthetic_fallback",
    "mock_inception",
    "mock_implementation",
    "use_mock",
)

DEMO_MODE_VALUE_KEYS = {
    "data_source": {"demo", "mock", "synthetic"},
    "mode": {"demo", "mock", "synthetic"},
}


def demo_mode_from_mapping(values: Mapping[str, Any] | None) -> bool:
    """Return whether a mapping explicitly opts into a retained demo workflow."""
    if not values:
        return False
    if any(bool(values.get(key)) for key in DEMO_MODE_KEYS):
        return True
    for key, allowed_values in DEMO_MODE_VALUE_KEYS.items():
        raw_value = values.get(key)
        if isinstance(raw_value, str) and raw_value.lower() in allowed_values:
            return True
    return False


def require_demo_mode(*, enabled: bool, component: str, detail: str) -> None:
    """Fail fast when a retained demo-only benchmark surface is used as real runtime."""
    if enabled:
        return
    raise RuntimeError(
        f"{component} is retained only for explicit benchmark demos. {detail} "
        "Pass demo_mode=True to opt into the retained synthetic or mock workflow."
    )
