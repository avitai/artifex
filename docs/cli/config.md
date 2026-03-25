# Config

**Status:** `Supported runtime CLI surface`

**Module:** `artifex.cli.config`

**Source:** `src/artifex/cli/config.py`

`artifex.cli.config` is the retained CLI wrapper for the typed configuration runtime.

## Supported Commands

- `create`
- `validate`
- `show`
- `diff`
- `version`
- `list`
- `get`

## Quick Start

```bash
artifex config create simple_training training.yaml \
  -p batch_size=64 \
  -p learning_rate=0.0002

artifex config create distributed_training distributed.json \
  -f json \
  -p world_size=4 \
  -p num_nodes=1 \
  -p num_processes_per_node=4

artifex config validate training.yaml
artifex config show training.yaml -f json
```

`create` accepts flat `key=value` template params only. Dead bundled file paths
and nested dotted keys are rejected instead of being fabricated into untyped
payloads.

These commands are intentionally narrow: they work with the repo-owned typed
config documents and reject arbitrary YAML mappings.
