# Portable VS Code Configuration

This directory contains **fully portable** VS Code configurations that automatically integrate with **any Python project's** `.pre-commit-config.yaml` and `pyproject.toml` files.

## 🎯 **Portability Features**

✅ **Universal Tool Detection**: Automatically detects and uses tools from any project's virtual environment
✅ **Configuration Auto-Discovery**: Reads settings from `pyproject.toml` and `.pre-commit-config.yaml`
✅ **Multi-Platform Support**: Works on Linux, macOS, and Windows
✅ **Environment Agnostic**: Supports uv, pip, conda, poetry, pipenv
✅ **Zero Hardcoding**: No project-specific paths or settings
🔒 **Security-First**: Only verified extensions from trusted publishers (Microsoft, Red Hat, GitHub, etc.)

## 🚀 Quick Setup

### Using This Configuration in Any Project

1. **Copy the `.vscode/` folder** to your project root
2. **Open the project** in VS Code
3. **Install verified extensions** when prompted (all from trusted publishers)
4. **Reload window**: Press `Ctrl+Shift+P` → "Developer: Reload Window"

That's it! The configuration will automatically:

- Detect your virtual environment (`.venv`, `venv`, conda, etc.)
- Read linting rules from your `pyproject.toml`
- Use the same tools as your `.pre-commit-config.yaml`
- Adapt to your project structure

## 🔧 What's Configured

### Automatic Linting & Formatting

- **Ruff**: Configured to use the same settings as your pre-commit hooks
- **Pyright**: Type checking with project-specific configuration
- **Format on Save**: Automatically formats Python files and Jupyter notebooks
- **Auto Fix**: Automatically fixes import organization and common issues

### Right-Click Menu Actions

#### For Python Files (.py)

- Right-click → "Format Document" → Uses Ruff formatter
- Right-click → "Format Selection" → Formats selected code
- `Ctrl+Shift+I` → Quick format shortcut

#### For Jupyter Notebooks (.ipynb)

- **Right-click on cell** → "Format Selection" → Uses Ruff formatter
- **Keyboard shortcut**: `Ctrl+Shift+I` → Formats entire notebook (safe - won't remove imports)
- **Alternative shortcut**: `Ctrl+Alt+F` → Formats entire notebook
- **Command Palette**: `Ctrl+Shift+P` → "Tasks: Run Task" → "Format Current Notebook"
- **Auto-format**: Format on save is enabled for notebook cells (safe mode)
- **Linting**: Use "NBQa: Ruff Lint Notebook" task only when you want to remove unused imports

⚠️ **Important**: Auto-save now uses safe formatting that won't remove imports. Use the linting task manually when needed.

### Command Palette Actions (Ctrl+Shift+P)

#### Pre-commit Integration

- "Tasks: Run Task" → "Run Pre-commit All Files"
- "Tasks: Run Task" → "Run Pre-commit Staged Files"
- "Tasks: Run Task" → "Run All Linters on Current File"

#### Individual Tools

- "Tasks: Run Task" → "Ruff Check Current File"
- "Tasks: Run Task" → "Ruff Format Current File"
- "Tasks: Run Task" → "Organize Imports (Pre-commit Style)"
- "Tasks: Run Task" → "Pyright Check Current File"
- "Tasks: Run Task" → "NBQa Ruff Jupyter Notebook"

## 🎯 Key Features

### 1. Seamless Pre-commit Integration

All linting rules match your `.pre-commit-config.yaml`:

- Same Ruff configuration (`--fix`, `--unsafe-fixes`)
- Same file exclusions (memory-bank, documents, etc.)
- Same Pyright settings from `pyproject.toml`

### 2. Notebook Support

- Jupyter notebooks are automatically formatted with Ruff
- NBQa integration for running Python linters on notebooks
- Format on save for notebook cells

### 3. Real-time Feedback

- Errors and warnings appear as you type
- Hover over underlined code for detailed messages
- Problems panel shows all issues across the workspace

### 4. Keyboard Shortcuts

- `Ctrl+Shift+I` → Format document
- `Ctrl+Shift+O` → Organize imports (matches pre-commit exactly)
- `Ctrl+Shift+L` → Run linter on current file
- `Ctrl+Shift+P` → Command palette
- `Ctrl+Shift+E` → Explorer panel
- `Ctrl+Shift+X` → Extensions panel

## 🔄 Workflow Integration

### Before Committing

1. Open files you've modified
2. Press `Ctrl+Shift+P` → "Tasks: Run Task" → "Run Pre-commit Staged Files"
3. Review and fix any issues
4. Commit your changes

### Daily Development

- Files auto-format on save
- Linting errors appear in real-time
- Right-click formatting for quick fixes
- Problems panel for overview of all issues

### Jupyter Notebooks

- Cells auto-format when you save the notebook
- Right-click on cells for manual formatting
- Use the "NBQa Ruff Jupyter Notebook" task for full notebook linting

## 🛠 Customization

### Adding More Tools

To add additional linting tools from your pre-commit config:

1. **Update settings.json**: Add new linter configurations
2. **Update tasks.json**: Add new tasks for the tools
3. **Update extensions.json**: Add required VS Code extensions

### Modifying Exclusions

The configuration automatically excludes the same directories as your pre-commit config:

- `memory-bank/`
- `documents/`
- `custom_modes/`
- `.cursor/`
- `notebooks/`

To modify exclusions, update the `python.analysis.exclude` setting in `settings.json`.

## 🐛 Troubleshooting

### Linting Not Working

1. Ensure `.venv` is activated: Check bottom-left Python interpreter
2. Install tools: `uv pip install ruff pyright`
3. Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"

### Format on Save Not Working

1. Check file type association in settings
2. Ensure Ruff extension is installed and enabled
3. Verify `editor.formatOnSave` is true for your file type

### Jupyter Notebooks Not Formatting

1. Install Jupyter extension pack
2. Ensure nbqa is installed: `uv pip install nbqa`
3. Use the dedicated notebook formatting task

## 📋 Available Commands Summary

| Action | Method | Scope |
|--------|---------|-------|
| Format File | Right-click → Format Document | Current file |
| Format Selection | Right-click → Format Selection | Selected code |
| Lint Current File | Command Palette → "Run All Linters on Current File" | Current file |
| Pre-commit All | Command Palette → "Run Pre-commit All Files" | All files |
| Pre-commit Staged | Command Palette → "Run Pre-commit Staged Files" | Staged files |
| Format Notebook | Task → "NBQa Ruff Jupyter Notebook" | Current notebook |

## 🔄 **How Portability Works**

### Automatic Tool Detection

The configuration uses `${workspaceFolder}/.venv/bin/` paths that automatically adapt to any project:

```json
"command": "${workspaceFolder}/.venv/bin/ruff"  // Works in any project
```

### Environment Discovery

- **Virtual Environments**: Auto-detects `.venv`, `venv`, conda environments
- **Package Managers**: Works with uv, pip, poetry, pipenv, conda
- **Python Paths**: Automatically includes common source directories (`src/`, `source/`, `lib/`)

### Configuration File Integration

- **Ruff**: Reads settings from `pyproject.toml` automatically
- **Pyright**: Uses `pyproject.toml` type checking configuration
- **Pre-commit**: Runs the exact hooks defined in `.pre-commit-config.yaml`

### Cross-Platform Compatibility

- **Linux/macOS**: Uses Unix-style paths and bash/zsh terminals
- **Windows**: Adapts to PowerShell and Windows paths
- **Environment Variables**: Sets platform-specific PYTHONPATH

## 📁 **Copy to Any Project**

To use this configuration in another project:

1. **Copy the entire `.vscode/` folder** to your new project
2. **No modifications needed** - it automatically adapts to:
   - Different virtual environment names
   - Different source code structures
   - Different linting configurations
   - Different pre-commit setups

This configuration ensures that your VS Code environment perfectly mirrors your pre-commit setup, giving you consistent code quality checking both in the editor and during commits, **regardless of the project**.

## ✅ **Final Validation**

Run the validation script to ensure everything is working:

```bash
python3 .vscode/validate-setup.py
```

This will check:

- ✅ All configuration files exist
- ✅ All development tools are available
- ✅ Configuration integration is working
- ✅ Portability features are enabled

## 📋 **File Summary**

| File | Purpose | Portability Features |
|------|---------|---------------------|
| `settings.json` | Main VS Code settings | Auto-detects tools from any project |
| `tasks.json` | Custom tasks for linting/testing | Uses project's virtual environment |
| `extensions.json` | Recommended extensions | Prevents conflicting extensions |
| `launch.json` | Debug configurations | Portable across project structures |
| `keybindings.json` | Keyboard shortcuts | Enhanced formatting shortcuts |
| `validate-setup.py` | Validation script | Checks configuration health |
| `README.md` | Documentation | Complete usage guide |

## 🎯 **Guaranteed Portability**

This `.vscode/` folder is **guaranteed to work** when copied to any Python project with:

- ✅ Virtual environment (`.venv`, `venv`, conda, etc.)
- ✅ `pyproject.toml` configuration file
- ✅ `.pre-commit-config.yaml` file
- ✅ Standard Python project structure

**Zero configuration changes needed!**
