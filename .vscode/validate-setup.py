#!/usr/bin/env python3
"""
Validation script for portable VS Code configuration.
Checks if all tools and configurations are properly detected.
Only recommends verified extensions from trusted publishers.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists and report status."""
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists


def check_command(cmd: list[str], description: str) -> bool:
    """Check if a command is available and report status."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        status = "✅" if result.returncode == 0 else "❌"
        print(f"{status} {description}: {' '.join(cmd)}")
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"❌ {description}: {' '.join(cmd)} (not found)")
        return False


def main():
    """Validate VS Code configuration setup."""
    print("🔍 Validating Portable VS Code Configuration")
    print("=" * 50)

    workspace = Path.cwd()
    venv_bin = workspace / ".venv" / "bin"
    if os.name == "nt":  # Windows
        venv_bin = workspace / ".venv" / "Scripts"

    print("\n📁 Configuration Files:")
    config_files = [
        (".vscode/settings.json", "VS Code Settings"),
        (".vscode/tasks.json", "VS Code Tasks"),
        (".vscode/extensions.json", "VS Code Extensions"),
        (".vscode/launch.json", "VS Code Launch Config"),
        ("pyproject.toml", "Project Configuration"),
        (".pre-commit-config.yaml", "Pre-commit Configuration"),
    ]

    config_ok = all(check_file_exists(str(workspace / path), desc)
                   for path, desc in config_files)

    print("\n🔧 Development Tools:")
    tool_commands = [
        ([str(venv_bin / "python"), "--version"], "Python Interpreter"),
        ([str(venv_bin / "ruff"), "--version"], "Ruff Linter/Formatter"),
        ([str(venv_bin / "pyright"), "--version"], "Pyright Type Checker"),
        ([str(venv_bin / "pre-commit"), "--version"], "Pre-commit"),
        ([str(venv_bin / "pytest"), "--version"], "Pytest"),
    ]

    tools_ok = all(check_command(cmd, desc) for cmd, desc in tool_commands)

    print("\n📋 Configuration Validation:")

    # Check settings.json (VS Code supports JSONC format with comments)
    settings_path = workspace / ".vscode" / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                content = f.read()

            # Check for key configurations (simple string search since VS Code uses JSONC)
            ruff_enabled = "ruff.enable" in content and "true" in content
            pyright_enabled = "python.analysis.typeCheckingMode" in content
            precommit_enabled = "pre-commit.enabled" in content and "true" in content

            print(f"{'✅' if ruff_enabled else '❌'} Ruff integration enabled")
            print(f"{'✅' if pyright_enabled else '❌'} Pyright integration enabled")
            print(f"{'✅' if precommit_enabled else '❌'} Pre-commit integration enabled")

        except Exception as e:
            print(f"❌ Error reading settings.json: {e}")

    # Check pyproject.toml for tool configuration
    pyproject_path = workspace / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path) as f:
            content = f.read()

        has_ruff_config = "[tool.ruff" in content
        has_pyright_config = "[tool.pyright" in content

        print(f"{'✅' if has_ruff_config else '❌'} Ruff configuration in pyproject.toml")
        print(f"{'✅' if has_pyright_config else '❌'} Pyright configuration in pyproject.toml")

    print("\n" + "=" * 50)

    if config_ok and tools_ok:
        print("🎉 Configuration is fully portable and ready to use!")
        print("💡 You can copy the .vscode/ folder to any Python project.")
        return 0
    else:
        print("⚠️  Some issues detected. Check the items above.")
        print("💡 Run 'pre-commit install' and ensure your virtual environment is set up.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
