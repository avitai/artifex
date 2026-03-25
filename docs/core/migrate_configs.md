# Config Migration

Status: Coming soon

`artifex.generative_models.core.configuration.migrate_configs` does not ship in
the current runtime.

## Current Migration Path

Artifex currently uses manual migration onto the supported typed config
surface:

- move imports to `artifex.configs` or
  `artifex.generative_models.core.configuration`
- replace legacy `*Configuration` names with the surviving `*Config` classes
- rematerialize typed configs through `from_dict()` or `from_yaml()`
- move family-specific config usage to the concrete `*_config.py` owners that
  still exist in the runtime

A future migration assistant may be added later, but manual migration is the
current supported path.

## Related Pages

- [Configuration Overview](configuration.md)
- [Unified Configuration Backend](unified.md)
- [Training Configuration Guide](../user-guide/training/configuration.md)
