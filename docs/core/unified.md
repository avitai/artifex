# Unified Configuration Backend

Artifex uses one configuration backend across the repo:

- frozen dataclass runtime configs
- typed retained template documents
- shared `from_dict()` / `from_yaml()` materialization

The old `unified.py`-style catch-all module is not the supported contract.
Use:

- `artifex.configs` for public imports
- `artifex.generative_models.core.configuration` for the reviewed runtime
  package surface

For the current config guide, see:

- [docs/configs/index.md](../configs/index.md)
- [docs/user-guide/training/configuration.md](../user-guide/training/configuration.md)
