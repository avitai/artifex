# Extensions Architecture

## Overview

The extensions architecture enables modular addition of domain-specific
features. Artifex currently ships registry-backed extension families for
protein, chemical, vision, NLP, and audio processing domains.

Protein remains the reference implementation for the typed bundle workflow, but it is not the only supported shared extension family. The curated top-level barrel stays lean and convenient, while the broader registry-backed family subpackages remain public through their own package paths and through `ExtensionsRegistry`.

## Related Documentation

- [Extensions Guide](../extensions/index.md)
- [Registry](../extensions/registry.md)
- [Config Surface](../configs/extensions.md)
