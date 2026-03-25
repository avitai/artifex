# Config Error Handling

The retained config loader keeps file, parse, and typed-schema failures inside
one repo-owned error hierarchy.

## Module

`artifex.configs.utils.error_handling`

## Available Error Types

- `ConfigError`: base loader failure for retained config helpers
- `ConfigNotFoundError`: the requested config path or shipped asset could not be resolved
- `ConfigValidationError`: the YAML document or typed dataclass payload failed validation
- `ConfigLoadError`: the file could not be parsed or loaded for a non-validation reason

## Current Contract

- public loader helpers exposed from `artifex.configs` should raise only this
  hierarchy for config-loading failures
- raw `dacite`, YAML parser, and file-loading exceptions should not leak past
  the retained loader helpers or config CLI commands
- most callers use this module indirectly through the public loader helpers
  exposed from `artifex.configs`
