"""Registry for mapping configuration types to model implementations.

NOTE: The primary factory for model creation is in `artifex.generative_models.factory`.
This registry provides an alternative string-based lookup mechanism for models.
"""

import dataclasses
from typing import Any, Callable, TypeVar


# Type for model factories
T = TypeVar("T")
ModelFactory = Callable[..., T]


class ModelRegistry:
    """
    Registry for mapping model names to their factory functions.

    This registry supports any callable factory. Config validation is
    handled by the dataclass configs' __post_init__ methods.
    """

    def __init__(self):
        """Initialize the model registry."""
        self._factories: dict[str, ModelFactory] = {}

    def register(self, name: str, factory: ModelFactory) -> None:
        """
        Register a model factory.

        Args:
            name: Name of the model
            factory: Factory function

        Raises:
            ValueError: If name is already registered
        """
        if name in self._factories:
            raise ValueError(f"Model '{name}' is already registered")

        self._factories[name] = factory

    def get_factory(self, name: str) -> ModelFactory:
        """
        Get a factory by name.

        Args:
            name: Name of the model

        Returns:
            Factory function

        Raises:
            KeyError: If name is not registered
        """
        if name not in self._factories:
            available = list(self._factories.keys())
            raise KeyError(f"Model '{name}' not found. Available models: {available}")

        return self._factories[name]

    def create_model(self, name: str, config: Any, **kwargs) -> Any:
        """
        Create a model using a registered factory.

        Args:
            name: Name of the model
            config: Model configuration (dataclass config)
            **kwargs: Additional keyword arguments for the factory

        Returns:
            Model instance

        Raises:
            TypeError: If config is not a dataclass
            KeyError: If name is not registered
        """
        # Validate config is a dataclass
        if not dataclasses.is_dataclass(config):
            raise TypeError(
                f"config must be a dataclass, got {type(config).__name__}. "
                "Use a dataclass config (DDPMConfig, VAEConfig, etc.) instead."
            )

        factory = self.get_factory(name)
        return factory(config=config, **kwargs)

    def list_models(self) -> list[str]:
        """
        List all registered model names.

        Returns:
            List of model names
        """
        return sorted(self._factories.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a model is registered."""
        return name in self._factories

    def __len__(self) -> int:
        """Get the number of registered models."""
        return len(self._factories)


# Global registry instance
_global_registry = ModelRegistry()


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """
    Decorator to register a model factory.

    Usage:
        @register_model("my_model")
        def create_my_model(*, config: MyConfig, rngs) -> MyModel:
            return MyModel(config=config, rngs=rngs)

    Args:
        name: Name of the model

    Returns:
        Decorator function
    """

    def decorator(factory: ModelFactory) -> ModelFactory:
        _global_registry.register(name, factory)
        return factory

    return decorator


def get_model_factory(name: str) -> ModelFactory:
    """
    Get a model factory from the global registry.

    Args:
        name: Name of the model

    Returns:
        Factory function

    Raises:
        KeyError: If name is not registered
    """
    return _global_registry.get_factory(name)


def create_model(name: str, config: Any, **kwargs) -> Any:
    """
    Create a model using the global registry.

    Args:
        name: Name of the model
        config: Model configuration (dataclass config)
        **kwargs: Additional keyword arguments for the factory

    Returns:
        Model instance

    Raises:
        TypeError: If config is not a dataclass
        KeyError: If name is not registered
    """
    return _global_registry.create_model(name, config, **kwargs)


def list_registered_models() -> list[str]:
    """
    List all models in the global registry.

    Returns:
        List of model names
    """
    return _global_registry.list_models()
