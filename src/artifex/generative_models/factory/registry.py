"""Internal registry types for the explicit model factory implementation."""

from typing import Any, Protocol, runtime_checkable

from flax import nnx


@runtime_checkable
class ModelBuilder(Protocol):
    """Protocol for model builders.

    Builders accept dataclass configs directly in their build() method.
    The config type determines the model to build.

    Example:
        class MyBuilder:
            def build(self, config: MyConfig | OtherConfig, *, rngs: nnx.Rngs, **kwargs) -> Any:
                if isinstance(config, MyConfig):
                    return MyModel(config=config, rngs=rngs, **kwargs)
                elif isinstance(config, OtherConfig):
                    return OtherModel(config=config, rngs=rngs, **kwargs)
    """

    def build(self, config: Any, *, rngs: nnx.Rngs, **kwargs) -> Any:
        """Build a model from configuration.

        Args:
            config: Dataclass model configuration (e.g., DDPMConfig, VAEConfig)
            rngs: Random number generators
            **kwargs: Additional arguments

        Returns:
            Model instance

        Raises:
            TypeError: If config type is not supported by this builder
        """
        ...


class DuplicateBuilderError(Exception):
    """Raised when attempting to register a duplicate builder."""

    pass


class BuilderNotFoundError(Exception):
    """Raised when a builder is not found in the registry."""

    pass


class ModelTypeRegistry:
    """Registry for model type builders."""

    def __init__(self):
        """Initialize the registry."""
        self._builders: dict[str, ModelBuilder] = {}

    def register(self, model_type: str, builder: ModelBuilder) -> None:
        """Register a model builder.

        Args:
            model_type: Type identifier for the model
            builder: Builder instance

        Raises:
            DuplicateBuilderError: If builder already registered
        """
        if model_type in self._builders:
            raise DuplicateBuilderError(f"Builder for '{model_type}' already registered")

        self._builders[model_type] = builder

    def get_builder(self, model_type: str) -> ModelBuilder:
        """Get a builder by model type.

        Args:
            model_type: Type identifier for the model

        Returns:
            Builder instance

        Raises:
            BuilderNotFoundError: If builder not found
        """
        if model_type not in self._builders:
            available = ", ".join(self._builders.keys())
            raise BuilderNotFoundError(
                f"No builder registered for type '{model_type}'. Available types: {available}"
            )

        return self._builders[model_type]

    def list_builders(self) -> list[str]:
        """List all registered builder types.

        Returns:
            List of registered model types
        """
        return list(self._builders.keys())

    def clear(self) -> None:
        """Clear all registered builders."""
        self._builders.clear()
