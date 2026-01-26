#!/usr/bin/env python
"""Check compatibility between JAX, NNX, and other dependencies.

This script verifies that the installed versions of JAX, NNX, and other
dependencies are compatible with each other.
"""

import importlib.metadata
import sys

from packaging import version


def get_version(package_name: str) -> str | None:
    """Get the version of an installed package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_compatibility() -> tuple[bool, dict[str, str | None]]:
    """Check if installed package versions are compatible.

    Returns:
        Tuple of (is_compatible, versions_dict)
    """
    versions = {
        "jax": get_version("jax"),
        "flax": get_version("flax"),
        "jaxlib": get_version("jaxlib"),
        "optax": get_version("optax"),
        "orbax-checkpoint": get_version("orbax-checkpoint"),
    }

    # If core packages are not installed, they can't be compatible
    if not versions["jax"] or not versions["flax"]:
        return False, versions

    # Parse versions
    jax_v = version.parse(versions["jax"])
    flax_v = version.parse(versions["flax"])

    # Check if jaxlib version matches jax version
    jaxlib_compatible = True
    if versions["jaxlib"]:
        jaxlib_v = version.parse(versions["jaxlib"])
        if jaxlib_v != jax_v:
            jaxlib_compatible = False

    # Check optax version if installed
    optax_compatible = True
    if versions["optax"]:
        optax_v = version.parse(versions["optax"])
        # Check optax compatibility with jax
        if optax_v <= version.parse("0.1.9") and jax_v >= version.parse("0.6.0"):
            optax_compatible = False
        elif optax_v >= version.parse("0.2.0") and jax_v < version.parse("0.5.1"):
            optax_compatible = False

    # Check orbax-checkpoint version if installed
    orbax_compatible = True
    if versions["orbax-checkpoint"]:
        orbax_v = version.parse(versions["orbax-checkpoint"])
        # orbax 0.11.0-0.11.5 requires JAX >= 0.4.34
        if version.parse("0.11.0") <= orbax_v <= version.parse("0.11.5"):
            if jax_v < version.parse("0.4.34"):
                orbax_compatible = False
        # orbax 0.11.6, 0.11.8+ require JAX >= 0.5.0
        elif orbax_v == version.parse("0.11.6") or orbax_v >= version.parse("0.11.8"):
            if jax_v < version.parse("0.5.0"):
                orbax_compatible = False
        # orbax 0.11.7 requires JAX == 0.5.0
        elif orbax_v == version.parse("0.11.7"):
            if jax_v != version.parse("0.5.0"):
                orbax_compatible = False

    # Check Flax version for NNX compatibility
    flax_nnx_compatible = True
    if flax_v < version.parse("0.10.0"):
        flax_nnx_compatible = False  # NNX requires Flax >= 0.10.0

    # Check if Flax version is compatible with JAX version
    flax_jax_compatible = True
    if flax_v >= version.parse("0.10.0"):
        if jax_v < version.parse("0.5.1"):
            flax_jax_compatible = False

    # Our project is configured for:
    # - JAX 0.6.1
    # - Flax 0.10.6
    # - optax 0.2.4
    # - orbax-checkpoint 0.11.13
    # Check if versions match our target versions
    # (uncommenting this if needed for additional checks)
    # target_compatible = (jax_v == version.parse("0.6.1") and
    #                      flax_v == version.parse("0.10.6"))

    # All checks must pass
    is_compatible = (
        flax_nnx_compatible
        and flax_jax_compatible
        and jaxlib_compatible
        and optax_compatible
        and orbax_compatible
    )

    return is_compatible, versions


def main():
    """Run the compatibility check and print results."""
    is_compatible, versions = check_compatibility()

    print("Installed versions:")
    for pkg, ver in versions.items():
        print(f"  {pkg}: {ver or 'Not installed'}")

    if is_compatible:
        print("\n✅ Compatible: Dependencies are compatible")
        print("\nRecommended configurations:")
        print("  - JAX 0.6.1")
        print("  - Flax 0.10.6")
        print("  - optax 0.2.4")
        print("  - orbax-checkpoint 0.11.13")
        sys.exit(0)
    else:
        print("\n❌ Incompatible: Dependency version conflicts detected")
        print("\nRecommended compatible versions:")
        print("  - JAX 0.6.1")
        print("  - Flax 0.10.6")
        print("  - optax 0.2.4")
        print("  - orbax-checkpoint 0.11.13")
        print("\nCheck docs/dependencies.md for more information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
