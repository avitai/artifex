# Contributing to Artifex

Thank you for your interest in contributing to Artifex! This guide will help you get started with contributing code, documentation, and other improvements.

## Getting Started

### Development Setup

1. **Fork and Clone**:

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/artifex.git
cd artifex
```

2. **Install with Development Dependencies**:

```bash
# Install all development dependencies
uv sync --all-extras

# Or install specific extras
uv sync --extra cuda-dev  # For GPU development
```

3. **Install Pre-commit Hooks**:

```bash
# Install pre-commit hooks for code quality
uv run pre-commit install

# Run hooks on all files
uv run pre-commit run --all-files
```

4. **Verify Installation**:

```bash
# Run tests to verify setup
uv run pytest tests/ -v
```

### Development Workflow

1. **Create a Feature Branch**:

```bash
git checkout -b feature/my-new-feature
```

2. **Make Changes and Test**:

```bash
# Make your changes
# ...

# Run tests
uv run pytest tests/path/to/test_file.py -xvs

# Run linting
uv run ruff check src/
uv run ruff format src/

# Run type checking
uv run pyright src/
```

3. **Commit Changes**:

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new feature description"
```

4. **Push and Create Pull Request**:

```bash
# Push to your fork
git push origin feature/my-new-feature

# Create pull request on GitHub
```

## Code Standards

### Flax NNX Requirements

Artifex uses **Flax NNX exclusively**. All neural network code must use Flax NNX:

```python
from flax import nnx

class MyModule(nnx.Module):
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        super().__init__()  # ALWAYS call this
        self.dense = nnx.Linear(features, features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dense(x)
```

**Do NOT use**:

- Flax Linen
- PyTorch or TensorFlow
- Numpy operations inside modules (use `jax.numpy`)

### Code Style

1. **Type Hints**: All functions must have type hints:

```python
def my_function(x: jax.Array, y: int) -> dict[str, jax.Array]:
    """Function docstring."""
    return {"result": x * y}
```

2. **Docstrings**: Use Google-style docstrings:

```python
def train_model(config: ModelConfig) -> dict:
    """Train a generative model.

    Args:
        config: Model configuration

    Returns:
        Dictionary with training results

    Raises:
        ValueError: If configuration is invalid
    """
    pass
```

3. **Formatting**: Code must pass Ruff formatting:

```bash
uv run ruff format src/
uv run ruff check src/
```

4. **Type Checking**: Code must pass Pyright:

```bash
uv run pyright src/
```

## Testing

### Writing Tests

1. **Test Structure**: Mirror source structure:

```
src/artifex/generative_models/models/vae.py
tests/artifex/generative_models/models/test_vae.py
```

2. **Test Template**:

```python
import pytest
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.models.vae import create_vae_model

def test_vae_creation():
    """Test VAE model creation."""
    config = ModelConfig(
        model_type="vae",
        latent_dim=10,
        # ...
    )

    model = create_vae_model(config, rngs=nnx.Rngs(0))

    assert model is not None
    assert model.latent_dim == 10

def test_vae_forward_pass():
    """Test VAE forward pass."""
    model = create_vae_model(config, rngs=nnx.Rngs(0))

    x = jnp.ones((2, 28, 28, 1))
    output = model(x)

    assert "reconstruction" in output
    assert output["reconstruction"].shape == x.shape
```

3. **GPU Tests**: Mark GPU-specific tests:

```python
@pytest.mark.gpu
def test_gpu_training():
    """Test training on GPU."""
    # GPU-specific test
    pass
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/artifex/generative_models/models/test_vae.py -xvs

# Run with coverage
uv run pytest --cov=src/artifex --cov-report=html

# Run GPU tests (requires CUDA)
uv run pytest -m gpu
```

## Documentation

### Writing Documentation

1. **Structure**: Follow existing patterns
2. **Examples**: Include working code examples
3. **Cross-references**: Link to related docs
4. **No AI Traces**: Never mention AI assistants, Claude, etc.

### Building Documentation

```bash
# Install documentation dependencies
uv sync --extra docs

# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```

## Pull Request Process

### Before Submitting

- [ ] Tests pass locally
- [ ] Code is formatted (Ruff)
- [ ] Type checking passes (Pyright)
- [ ] Documentation updated if needed
- [ ] Commit messages are descriptive

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added new tests
- [ ] All tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated Checks**: CI runs tests, linting, type checking
2. **Code Review**: Maintainer reviews code
3. **Revisions**: Address feedback if needed
4. **Merge**: Approved PRs are merged

## Commit Messages

Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `ci`: CI/CD changes

**Examples**:

```
feat(vae): add β-VAE implementation
fix(training): resolve NaN loss issue
docs(quickstart): add installation steps
test(gan): increase test coverage
```

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

### Enforcement

Violations may result in:

1. Warning
2. Temporary ban
3. Permanent ban

Report issues to maintainers.

## Getting Help

- **Issues**: Open GitHub issue for bugs/features
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check docs first

## Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes
- Documentation credits

Thank you for contributing to Artifex!
