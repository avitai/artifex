"""Command-line interface for generative models."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from artifex.cli.__main__ import app, main


def __getattr__(name: str) -> object:
    """Lazy import to avoid RuntimeWarning when run as ``python -m artifex.cli``."""
    if name in ("app", "main"):
        from artifex.cli.__main__ import app, main  # noqa: F811

        globals().update({"app": app, "main": main})
        return globals()[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["app", "main"]
