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
- Maximum line length: 88 characters (Black formatter)
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
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/path/to/test.py -xvs

# Run with coverage
uv run pytest --cov=src/artifex --cov-report=html

# GPU-aware testing
./scripts/smart_test_runner.sh tests/ -v
```

### Test Organization

- `tests/standalone/`: Isolated component tests
- `tests/artifex/`: Integrated system tests
- Mark GPU tests with `@pytest.mark.gpu`

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

### Example Guidelines

1. **Use dual-format**: Examples should work as both `.py` scripts and `.ipynb` notebooks via Jupytext
2. **Follow the template structure**: Include learning objectives, prerequisites, step-by-step implementation, and exercises
3. **Use Flax NNX patterns**: Follow the module initialization and RNG handling patterns shown in the template
4. **Add educational content**: Explain concepts, not just code

### Running Examples

```bash
# Run as Python script
python examples/path/to/example.py

# Convert to notebook (if needed)
jupytext --to notebook examples/path/to/example.py
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
