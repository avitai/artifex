# Artifex VS Code Workspace

This `.vscode/` folder is an Artifex-local workspace convenience layer. It is
not a generic template for other repositories.

## Scope

These files assume:

- the Artifex repository root is the open VS Code workspace
- dependencies were installed through the repo workflow (`./setup.sh` or
  `uv sync ...`)
- the selected interpreter comes from the Artifex workspace environment
- repo tasks should follow the same `uv run ...` contract documented in
  [scripts/README.md](../scripts/README.md) and [TESTING.md](../TESTING.md)

## Included Surfaces

- `settings.json`: editor defaults for this repository
- `tasks.json`: Artifex task shortcuts for `uv run` checks and test commands
- `launch.json`: debugger entries for current-file execution and pytest
- `extensions.json`: recommended extensions for the Artifex workflow
- `keybindings.json`: local keybinding conveniences
- `validate-setup.py`: Artifex-specific workspace validation helper

## Quick Start

1. Run `./setup.sh` or `uv sync` for the Artifex workspace.
2. Open the repository root in VS Code.
3. Select the Artifex interpreter from `.venv`.
4. Optionally validate the workspace:

```bash
uv run python .vscode/validate-setup.py
```

## Supported Tasks

- Pre-commit against all files or staged files
- Ruff check and format for the current file
- Pyright on the current file
- Pytest for the current file or the full suite
- Curated docs validation

## Debugger Expectations

- Current-file debugging runs the selected file from the Artifex workspace
- Pytest launch entries run inside the open repository root
- No launch entry assumes a generic web app such as `main:app` or `app.py`

## Maintenance Notes

- Keep this folder truthful to the Artifex repo contract
- Prefer `uv run ...` over bare tool invocations
- Remove stale tasks or launch configs instead of keeping fake generality
- If the repo setup flow changes, update this folder and the validation helper
  together
