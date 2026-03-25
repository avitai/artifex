# Contributing to Artifex

Thank you for your interest in contributing to Artifex! This document provides guidelines and information for contributors.

## Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/avitai/artifex.git
cd artifex

# Run setup script (creates venv, installs dependencies, detects GPU)
./setup.sh

# Activate the environment
source ./activate.sh
```

### Prerequisites

- Python 3.11+
- uv package manager
- Git

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/amazing-feature
```

### 2. Make Your Changes

- Follow the coding standards below
- Write tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Run all pre-commit checks
uv run pre-commit run --all-files

# Run tests
uv run pytest tests/ -v
```

### 4. Commit Your Changes

```bash
git commit -m "Add amazing feature"
```

### 5. Push and Create PR

```bash
git push origin feature/amazing-feature
```

Then open a Pull Request on GitHub.

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use type annotations for all functions
- Maximum line length: 100 characters (Ruff formatter)
- Use descriptive variable names

### Framework Requirements

**Always use Flax NNX:**

```python
# CORRECT
from flax import nnx

class MyModule(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.dense = nnx.Linear(10, 10, rngs=rngs)
```

**Never use Flax Linen or PyTorch.**

### Configuration Style

Use frozen dataclass configurations:

```python
from artifex.generative_models.core.configuration import VAEConfig, EncoderConfig

# CORRECT: Frozen dataclass config
config = VAEConfig(
    name="my_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    kl_weight=1.0,
)

# WRONG: Dictionary config
config = {"name": "my_vae", "params": {...}}  # Don't do this
```

### Testing Requirements

- Write tests for all new functionality
- Tests should be in the appropriate `tests/` subdirectory
- Aim for minimum 80% coverage on new code
- Use pytest fixtures for common setup

```python
def test_feature():
    """Test description."""
    # Arrange
    config = create_config()

    # Act
    result = function_under_test(config)

    # Assert
    assert result.property == expected_value
```

## Code Quality Tools

### Pre-commit Hooks

Install and run pre-commit hooks:

```bash
# Install hooks
uv run pre-commit install

# Run on all files
uv run pre-commit run --all-files
```

### Linting and Formatting

```bash
# Linting
uv run ruff check src/

# Formatting
uv run ruff format src/

# Type checking
uv run pyright src/
```

## Testing

### Running Tests

```bash
# Run the standard suite
uv run pytest

# Run specific test file
uv run pytest tests/path/to/test.py -xvs

# Run with coverage
uv run pytest --cov=src/artifex --cov-report=html

# Marker-focused runs
uv run pytest -m gpu
uv run pytest -m blackjax
```

### Test Organization

- `tests/artifex/`: package, integration, and repo-contract tests over live Artifex owners
- `tests/unit/`: narrower low-level unit coverage where that layout already exists
- Use `test_device` for device-aware tests and `gpu_test_fixture` for explicitly GPU-required tests
- Mark GPU tests with `@pytest.mark.gpu`
- Do not add local replica suites or shadow suites that bypass live imports

## Documentation

### Writing Documentation

- Use Markdown for all documentation
- Include code examples where appropriate
- Keep examples runnable and tested
- Update relevant docs when changing code

### Documentation Structure

```
docs/
├── getting-started/    # Installation and first steps
├── user-guide/         # How-to guides
├── api/                # API reference
├── examples/           # Example documentation
└── development/        # Contributor guides
```

## Creating Examples

When contributing new examples, use the templates in `docs/examples/templates/`:

- `example_template.py` - Python script with Jupytext markers
- `example_template.ipynb` - Jupyter notebook version

The canonical authoring guide is the
[Example Documentation Design Guide](docs/development/example-documentation-design.md).
Use it as the standard for example structure, device labeling, docs layout, and
sync workflow.

### Example Guidelines

1. **Use dual-format for tutorials**: Reader-facing tutorial examples should ship as `.py` and `.ipynb` Jupytext pairs. Verification or maintenance scripts under `examples/` may remain `.py` only.
2. **Follow the template structure**: Include learning objectives, prerequisites, step-by-step implementation, and exercises
3. **Use Flax NNX patterns**: Follow the module initialization and RNG handling patterns shown in the template
4. **Add educational content**: Explain concepts, not just code
5. **Use current repo workflows**: Show `source ./activate.sh` and `uv run ...` commands
6. **Sync dual-format examples with the repo tool**: Regenerate paired notebooks with `scripts/jupytext_converter.py`

### Running Examples

```bash
# Activate the repo environment
source ./activate.sh

# Run as Python script
uv run python examples/path/to/example.py

# Sync the paired notebook for dual-format tutorial examples
uv run python scripts/jupytext_converter.py sync examples/path/to/example.py
```

## Pull Request Guidelines

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally (`uv run pytest tests/ -v`)
- [ ] Pre-commit hooks pass (`uv run pre-commit run --all-files`)
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive

### PR Title Format

Use conventional commit format:

- `feat: Add new feature`
- `fix: Fix bug in component`
- `docs: Update documentation`
- `refactor: Refactor module`
- `test: Add tests for feature`
- `chore: Update dependencies`

### PR Description

Include:

- Brief description of changes
- Motivation and context
- How to test the changes
- Any breaking changes

## Issue Guidelines

### Bug Reports

Include:

- Artifex version
- Python version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior

### Feature Requests

Include:

- Use case description
- Proposed solution
- Alternative approaches considered

## Community

### Getting Help

- Check existing documentation
- Search existing issues
- Open a new issue if needed

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
