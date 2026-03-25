#!/usr/bin/env python3
"""Validate installed JAX ecosystem packages against the repo dependency contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DOCS_REFS = [
    "docs/getting-started/installation.md",
    "docs/getting-started/hardware-setup-guide.md",
]
PACKAGES_TO_CHECK = ("jax", "flax", "optax", "orbax-checkpoint", "jaxlib")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the compatibility report as JSON",
    )
    return parser.parse_args(argv)


def get_version(package_name: str) -> str | None:
    """Return the installed version for one package, if present."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_project_minimum_versions() -> dict[str, str]:
    """Return the repo-declared minimum versions for the JAX stack."""
    with PYPROJECT_PATH.open("rb") as handle:
        pyproject = tomllib.load(handle)

    minimum_versions: dict[str, str] = {}
    for raw_requirement in pyproject["project"]["dependencies"]:
        requirement = Requirement(raw_requirement)
        if requirement.name not in {"jax", "flax", "optax", "orbax-checkpoint"}:
            continue

        lower_bounds = [
            Version(spec.version)
            for spec in requirement.specifier
            if spec.operator in {">=", "==", "==="}
        ]
        if not lower_bounds:
            continue
        minimum_versions[requirement.name] = str(max(lower_bounds))

    return minimum_versions


def build_report() -> dict[str, Any]:
    """Build the metadata-driven compatibility report."""
    project_minimum_versions = load_project_minimum_versions()
    installed_versions = {package: get_version(package) for package in PACKAGES_TO_CHECK}

    missing_packages: list[str] = []
    below_minimum: dict[str, dict[str, str]] = {}
    for package_name, minimum_version in project_minimum_versions.items():
        installed_version = installed_versions.get(package_name)
        if installed_version is None:
            missing_packages.append(package_name)
            continue

        specifier = SpecifierSet(f">={minimum_version}")
        if installed_version not in specifier:
            below_minimum[package_name] = {
                "installed": installed_version,
                "required_minimum": minimum_version,
            }

    jax_version = installed_versions.get("jax")
    jaxlib_version = installed_versions.get("jaxlib")
    jaxlib_matches_jax = (
        jax_version is not None and jaxlib_version is not None and jax_version == jaxlib_version
    )

    issues: list[str] = []
    if missing_packages:
        issues.append("Missing required packages: " + ", ".join(sorted(missing_packages)))
    if below_minimum:
        issues.extend(
            (
                f"{package} {details['installed']} is below the repo minimum "
                f"{details['required_minimum']}"
            )
            for package, details in sorted(below_minimum.items())
        )
    if jax_version and jaxlib_version and not jaxlib_matches_jax:
        issues.append(f"jaxlib {jaxlib_version} does not match jax {jax_version}")

    return {
        "installed_versions": installed_versions,
        "project_minimum_versions": project_minimum_versions,
        "docs_refs": DOCS_REFS,
        "missing_packages": sorted(missing_packages),
        "below_minimum": below_minimum,
        "jaxlib_matches_jax": jaxlib_matches_jax,
        "scope_note": (
            "This check validates the repo-declared minimum versions and JAX/JAXLIB parity. "
            "It does not claim full upstream compatibility beyond that contract."
        ),
        "is_compatible": not issues,
        "issues": issues,
    }


def render_human_report(report: dict[str, Any]) -> str:
    """Render the compatibility report for terminal output."""
    lines = ["Installed versions:"]
    for package_name in PACKAGES_TO_CHECK:
        installed_version = report["installed_versions"][package_name]
        lines.append(f"  {package_name}: {installed_version or 'Not installed'}")

    lines.append("")
    lines.append("Project-declared minimum versions:")
    for package_name, minimum_version in report["project_minimum_versions"].items():
        lines.append(f"  - {package_name} >= {minimum_version}")

    lines.append("")
    if report["is_compatible"]:
        lines.append("Compatible with the repo-declared dependency contract.")
    else:
        lines.append("Incompatible with the repo-declared dependency contract.")
        for issue in report["issues"]:
            lines.append(f"  - {issue}")

    lines.append("")
    lines.append(report["scope_note"])
    lines.append("Relevant docs:")
    for relative_path in report["docs_refs"]:
        lines.append(f"  - {relative_path}")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compatibility check."""
    args = parse_args(argv)
    report = build_report()

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(f"{render_human_report(report)}\n")

    return 0 if report["is_compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
