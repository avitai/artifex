# Command Line Interface

The retained runtime owners are `artifex.cli.__main__` and `artifex.cli.config`.

The retained top-level CLI is intentionally small. `python -m artifex.cli --help`
currently exposes only the `config` sub-app for Configuration management commands.

## Retained Surface

- `artifex config create`
- `artifex config validate`
- `artifex config show`
- `artifex config diff`
- `artifex config version`
- `artifex config list`
- `artifex config get`

## Quick Start

```bash
# Show the retained top-level catalog
python -m artifex.cli --help

# Inspect config commands
python -m artifex.cli config --help

# Create a typed training config
python -m artifex.cli config create simple_training config.yaml \
  --param batch_size=32 \
  --param learning_rate=0.001

# Validate the generated config
python -m artifex.cli config validate config.yaml
```

## Current Scope

The shipped CLI owns configuration management only. Training, generation,
evaluation, serving, benchmarking, and conversion workflows are not currently
available as top-level runtime commands.

Legacy command pages remain in the docs tree only as retirement notices so
existing links fail truthfully instead of pointing at phantom runtime entrypoints.

## Next References

- [Config Command Reference](config.md)
- [Training Configuration Guide](../user-guide/training/configuration.md)
- [Factory Overview](../factory/index.md)
