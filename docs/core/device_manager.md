# Device Manager

**Module:** `generative_models.core.device_manager`

## Overview

The device manager is a runtime helper, not a backend bootstrap layer.

It provides:

- inspection of the active JAX runtime
- access to visible devices
- default-device selection
- simple batch sharding across visible devices

It does not expose backend configuration knobs such as memory strategies,
environment-variable injection, or forced platform ordering.

## Public Classes

### DeviceCapabilities

```python
class DeviceCapabilities
```

### DeviceManager

```python
class DeviceManager
```

### DeviceType

```python
class DeviceType
```

## Public Functions

### get_default_device

```python
def get_default_device()
```

### get_device_manager

```python
def get_device_manager()
```

### has_gpu

```python
def has_gpu()
```

### print_device_info

```python
def print_device_info()
```
