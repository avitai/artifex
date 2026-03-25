# CLI Entrypoint

**Status:** `Supported runtime CLI surface`

**Module:** `artifex.cli.__main__`

**Source:** `src/artifex/cli/__main__.py`

`artifex.cli.__main__` owns the Typer application bootstrap for the shipped CLI.
The retained public surface is small: one `app`, one callback, and one
`main()` entrypoint.

## Current Runtime Members

- `app`: the top-level `typer.Typer` application
- `main_callback`: the callback that owns the `--version` option
- `main`: the `python -m artifex.cli` entrypoint

## Runtime Shape

```python
from artifex.cli.__main__ import app, main

app()   # invoked by the CLI runtime
main()  # module entrypoint wrapper
```

The entrypoint currently mounts only the `config` sub-app from
`artifex.cli.config`. No separate help helper is part of the shipped runtime.
