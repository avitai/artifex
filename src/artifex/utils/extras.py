"""Errors for the optional subsystems that ship behind package extras."""

from __future__ import annotations


def missing_extra(package: str, extra: str) -> ImportError:
    """Return the error to raise when ``package`` from the ``extra`` extra is absent.

    Args:
        package: The import name that failed.
        extra: The ``avitai-artifex`` extra that installs it.

    Returns:
        An ``ImportError`` whose message names the extra to install.
    """
    return ImportError(
        f"{package} is not installed. It ships with the `{extra}` extra: "
        f'pip install "avitai-artifex[{extra}]"'
    )
