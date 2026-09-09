#!/usr/bin/env python3
"""Derive the model-family surface from the tree and check it against the docs.

Two run modes::

    uv run python scripts/derive_status.py            # print the derived matrix
    uv run python scripts/derive_status.py --check     # exit 1 on any drift (CI)

``docs/models/index.md`` carries a table naming every generative model family, its
owner package and representative exports. That table rots as packages and exports
change; this script imports each package, checks each export, lists every family
package under ``models/`` that the table does not name, and prints the matrix of
what each family ships (trainer, default configs, tests, examples).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("derive_status")

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = "artifex.generative_models.models"
MODELS_DIR = REPO_ROOT / "src" / "artifex" / "generative_models" / "models"
# Shared building blocks under models/, not families: backbones and common layers.
NON_FAMILY_PACKAGES = frozenset({"backbones", "common"})
ROW = re.compile(r"^\| ([^|]+) \| `([\w.]+)` \| (.+) \|$")
EXPORT = re.compile(r"`(\w+)`")


@dataclass(frozen=True, slots=True, kw_only=True)
class FamilyRow:
    """One row of the documented family table."""

    family: str
    package: str
    exports: tuple[str, ...]

    @property
    def name(self) -> str:
        """The package's last segment, which names the family directory."""
        return self.package.rsplit(".", 1)[-1]


@dataclass(frozen=True, slots=True, kw_only=True)
class FamilyStatus:
    """What the tree holds for one documented family."""

    row: FamilyRow
    importable: bool
    missing_exports: tuple[str, ...]
    trainer: bool
    configs: bool
    tests: bool
    examples: bool

    @property
    def is_drifted(self) -> bool:
        """Whether the docs claim something the tree does not have."""
        return not self.importable or bool(self.missing_exports)


def parse_family_table(index: Path) -> list[FamilyRow]:
    """Read the ``Family Packages`` table rows from the models index page."""
    rows: list[FamilyRow] = []
    for line in index.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if match is None or match.group(1).strip() in {"Family", "---"}:
            continue
        rows.append(
            FamilyRow(
                family=match.group(1).strip(),
                package=match.group(2),
                exports=tuple(EXPORT.findall(match.group(3))),
            )
        )
    if not rows:
        raise SystemExit(f"no family table rows found in {index}")
    return rows


def probe_packages(root: Path, rows: list[FamilyRow]) -> dict[str, dict[str, object]]:
    """Import every documented package in a subprocess and report missing exports."""
    script = (
        "import importlib, json, sys\n"
        "rows = json.loads(sys.argv[1])\n"
        "out = {}\n"
        "for package, exports in rows.items():\n"
        "    try:\n"
        "        module = importlib.import_module(package)\n"
        "    except ImportError as error:\n"
        "        out[package] = {'importable': False, 'missing': exports, 'error': str(error)}\n"
        "        continue\n"
        "    missing = [name for name in exports if not hasattr(module, name)]\n"
        "    out[package] = {'importable': True, 'missing': missing}\n"
        "print(json.dumps(out))\n"
    )
    payload = json.dumps({row.package: list(row.exports) for row in rows})
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(root / "src"), os.environ.get("PYTHONPATH")])
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", script, payload],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    return json.loads(result.stdout)


def family_packages(models_dir: Path) -> set[str]:
    """Every package directory under models/ that is a family."""
    return {
        path.name
        for path in models_dir.iterdir()
        if path.is_dir()
        and (path / "__init__.py").is_file()
        and path.name not in NON_FAMILY_PACKAGES
    }


def collect_status(root: Path, rows: list[FamilyRow]) -> list[FamilyStatus]:
    """Pair each documented family with what the tree holds for it."""
    probe = probe_packages(root, rows)
    generative = root / "src" / "artifex" / "generative_models"
    statuses = []
    for row in rows:
        result = probe[row.package]
        statuses.append(
            FamilyStatus(
                row=row,
                importable=bool(result["importable"]),
                missing_exports=tuple(result["missing"]),
                trainer=(generative / "training" / "trainers" / f"{row.name}_trainer.py").is_file(),
                configs=(
                    root / "src" / "artifex" / "configs" / "defaults" / "models" / row.name
                ).is_dir(),
                tests=(
                    root / "tests" / "artifex" / "generative_models" / "models" / row.name
                ).is_dir(),
                examples=(root / "examples" / "generative_models" / row.name).is_dir(),
            )
        )
    return statuses


def render_matrix(statuses: list[FamilyStatus], undocumented: set[str]) -> str:
    """Format the family matrix for human reading."""
    widths = {"family": 16, "exports": 8, "trainer": 8, "configs": 8, "tests": 6, "examples": 9}

    def row(cells: dict[str, str], drift: str) -> str:
        return (
            " ".join(f"{cells[column]:<{width}}" for column, width in widths.items()) + f" {drift}"
        )

    header = row({column: column for column in widths}, "drift")
    mark = {True: "yes", False: "-"}
    lines = [header, "-" * len(header)]
    for status in statuses:
        exports = (
            f"{len(status.row.exports) - len(status.missing_exports)}/{len(status.row.exports)}"
        )
        cells = {
            "family": status.row.name,
            "exports": exports,
            "trainer": mark[status.trainer],
            "configs": mark[status.configs],
            "tests": mark[status.tests],
            "examples": mark[status.examples],
        }
        lines.append(row(cells, "DRIFT" if status.is_drifted else "ok"))
    for name in sorted(undocumented):
        cells = {column: "-" for column in widths}
        cells["family"] = name
        lines.append(row(cells, "DRIFT (not in the docs table)"))
    return "\n".join(lines)


def main() -> int:
    """Print the derived matrix; with ``--check``, exit non-zero on drift."""
    parser = argparse.ArgumentParser(description="Derive and verify the documented model surface.")
    parser.add_argument("--check", action="store_true", help="exit non-zero on any drift")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--models-index",
        type=Path,
        default=None,
        help="the models index page; defaults to docs/models/index.md under the root",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = args.repo_root.resolve()
    index = (args.models_index or root / "docs" / "models" / "index.md").resolve()
    rows = parse_family_table(index)
    statuses = collect_status(root, rows)
    undocumented = family_packages(root / "src" / "artifex" / "generative_models" / "models") - {
        row.name for row in rows
    }
    print(render_matrix(statuses, undocumented))  # noqa: T201 CLI entry point report

    drifted = [status for status in statuses if status.is_drifted]
    for status in drifted:
        if not status.importable:
            logger.error("drift: %s does not import", status.row.package)
        for name in status.missing_exports:
            logger.error("drift: %s has no export %s", status.row.package, name)
    for name in sorted(undocumented):
        logger.error(
            "drift: %s.%s is a family package the docs table does not name", PACKAGE_ROOT, name
        )
    if args.check and (drifted or undocumented):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
