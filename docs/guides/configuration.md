# Configuration Guide

!!! info "Coming Soon"
    This guide is being developed. Check back for comprehensive configuration documentation.

## Overview

Learn how to configure Artifex models and training:

- Model configuration classes
- Training configuration
- Data configuration
- Nested and composed configurations

## Related Documentation

- [Training Configuration](../user-guide/training/configuration.md)
- [Factory Guide](factory.md)
- [Extensions Guide](extensions.md)

## Configuration Patterns

### Frozen Dataclasses

All configurations use frozen dataclasses for immutability.

### Nested Configuration

Complex models use nested configuration objects.

### Validation

Configurations are validated at creation time using Pydantic.
