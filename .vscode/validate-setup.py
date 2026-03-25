#!/usr/bin/env python3
# ruff: noqa: T201
"""Validate the Artifex-local VS Code workspace contract."""

from __future__ import annotations

import subprocess
from pathlib import Path


def check_file_exists(path: Path, description: str) -> bool:
    """Check whether one required workspace file exists."""
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists


def check_command(command: list[str], description: str) -> bool:
    """Check whether one uv-based workspace command succeeds."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"❌ {description}: {' '.join(command)}")
        return False

    status = "✅" if result.returncode == 0 else "❌"
    print(f"{status} {description}: {' '.join(command)}")
    return result.returncode == 0


def main() -> int:
    """Validate the checked-in Artifex VS Code workspace assumptions."""
    workspace = Path.cwd()

    print("🔍 Validating Artifex VS Code Workspace")
    print("=" * 50)

    print("\n📁 Workspace Files:")
    required_files = [
        (workspace / ".vscode" / "settings.json", "VS Code settings"),
        (workspace / ".vscode" / "tasks.json", "VS Code tasks"),
        (workspace / ".vscode" / "extensions.json", "VS Code extensions"),
        (workspace / ".vscode" / "launch.json", "VS Code launch config"),
        (workspace / "pyproject.toml", "Project configuration"),
        (workspace / ".pre-commit-config.yaml", "Pre-commit configuration"),
    ]
    files_ok = all(check_file_exists(path, description) for path, description in required_files)

    print("\n🔧 uv Workspace Commands:")
    command_checks = [
        (["uv", "--version"], "uv available"),
        (["uv", "run", "python", "--version"], "workspace Python"),
        (["uv", "run", "ruff", "--version"], "workspace Ruff"),
        (["uv", "run", "pyright", "--version"], "workspace Pyright"),
        (["uv", "run", "pre-commit", "--version"], "workspace Pre-commit"),
        (["uv", "run", "pytest", "--version"], "workspace Pytest"),
    ]
    commands_ok = all(
        check_command(command, description) for command, description in command_checks
    )

    print("\n📋 Workspace Notes:")
    artifex_env = workspace / ".artifex.env"
    if artifex_env.exists():
        print(f"✅ Generated runtime env present: {artifex_env}")
    else:
        print("⚠️  .artifex.env not found. Run ./setup.sh if you want the managed backend layer.")

    print("\n" + "=" * 50)
    if files_ok and commands_ok:
        print("🎉 Artifex VS Code workspace configuration looks ready.")
        return 0

    print("⚠️  Some Artifex workspace checks failed.")
    print("💡 Run ./setup.sh or uv sync, then retry this validator.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
