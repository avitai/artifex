"""Internal registry types for the explicit model factory implementation."""

from typing import Any, Protocol, runtime_checkable

from calibrax.core import Registry
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


class BuilderNotFoundError(Exception):
    """Raised when a builder is not found in the registry."""

    pass


class ModelTypeRegistry(Registry[ModelBuilder]):
    """Registry for model type builders, on calibrax's ``Registry``.

    ``register`` refuses a model type that is already registered with calibrax's
    ``DuplicateRegistrationError`` and keeps the builder registered first; ``list_names``,
    ``remove`` and ``clear`` come from the same base.
    """

    def get_builder(self, model_type: str) -> ModelBuilder:
        """Get a builder by model type.

        Args:
            model_type: Type identifier for the model

        Returns:
            Builder instance

        Raises:
            BuilderNotFoundError: If builder not found
        """
        if model_type not in self:
            available = ", ".join(self.list_names())
            raise BuilderNotFoundError(
                f"No builder registered for type '{model_type}'. Available types: {available}"
            )

        return self.get(model_type)
