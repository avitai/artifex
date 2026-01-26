"""Test for correct Protocol usage patterns.

This test demonstrates the correct way to use Protocol types
and identifies incorrect patterns in the codebase.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from flax import nnx


class TestProtocolPatterns:
    """Test correct Protocol usage patterns."""

    def test_protocol_vs_abc_difference(self):
        """Test the difference between Protocol and ABC."""

        # CORRECT: Using Protocol for structural typing
        @runtime_checkable
        class ModelProtocol(Protocol):
            """A protocol defining the interface for models."""

            def forward(self, x): ...
            def loss(self, x, y): ...

        # CORRECT: Using ABC for inheritance-based abstract classes
        class BaseModel(ABC):
            """An abstract base class for models."""

            @abstractmethod
            def forward(self, x):
                """Forward pass."""
                pass

            @abstractmethod
            def loss(self, x, y):
                """Compute loss."""
                pass

        # Protocol can be used for isinstance checks with @runtime_checkable
        class ConcreteModel:
            def forward(self, x):
                return x * 2

            def loss(self, x, y):
                return jnp.mean((x - y) ** 2)

        model = ConcreteModel()
        assert isinstance(model, ModelProtocol)  # Works because of @runtime_checkable

    def test_incorrect_protocol_naming(self):
        """Test incorrect naming of ABC classes as Protocol."""

        # INCORRECT: Class named Protocol but inherits from ABC
        # This is what's wrong in the codebase
        class BenchmarkProtocol(nnx.Module, ABC):  # Wrong! Not a Protocol
            """This is incorrectly named - it's an ABC, not a Protocol."""

            @abstractmethod
            def run(self):
                pass

        # This is NOT a Protocol in the typing sense
        assert not issubclass(BenchmarkProtocol, Protocol)

    def test_correct_nnx_patterns(self):
        """Test correct patterns for NNX modules."""

        # CORRECT: Abstract base class for NNX modules
        class BaseNNXModule(nnx.Module, ABC):
            """Abstract base class for NNX modules."""

            @abstractmethod
            def __call__(self, x):
                """Forward pass."""
                pass

        # CORRECT: Protocol for type checking NNX modules
        @runtime_checkable
        class NNXModuleProtocol(Protocol):
            """Protocol for NNX modules."""

            def __call__(self, x): ...

        # CORRECT: Concrete implementation
        class ConcreteNNXModule(BaseNNXModule):
            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__()
                self.dense = nnx.Linear(10, 10, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        # Test instantiation
        module = ConcreteNNXModule(rngs=nnx.Rngs(42))
        assert isinstance(module, nnx.Module)
        assert isinstance(module, BaseNNXModule)
        assert isinstance(module, NNXModuleProtocol)

    def test_protocol_for_duck_typing(self):
        """Test using Protocol for duck typing."""

        @runtime_checkable
        class Trainable(Protocol):
            """Protocol for trainable objects."""

            def train_step(self, batch): ...
            def evaluate(self, data): ...

        # Any class implementing these methods satisfies the protocol
        class MyTrainer:
            def train_step(self, batch):
                return {"loss": 0.1}

            def evaluate(self, data):
                return {"accuracy": 0.95}

        class MyOtherTrainer:
            def train_step(self, batch):
                return {"loss": 0.2}

            def evaluate(self, data):
                return {"accuracy": 0.90}

        # Both satisfy the protocol without inheritance
        trainer1 = MyTrainer()
        trainer2 = MyOtherTrainer()

        assert isinstance(trainer1, Trainable)
        assert isinstance(trainer2, Trainable)

        # This is the power of Protocol - structural typing
        def run_training(trainer: Trainable):
            """Function that accepts any Trainable object."""
            result = trainer.train_step({"x": 1})
            return result

        # Both work without explicit inheritance
        assert run_training(trainer1) == {"loss": 0.1}
        assert run_training(trainer2) == {"loss": 0.2}


class TestRecommendedRefactoring:
    """Test recommended refactoring patterns."""

    def test_benchmark_protocol_refactoring(self):
        """Demonstrate how BenchmarkProtocol should be refactored."""

        # CURRENT (INCORRECT) in codebase:
        # class BenchmarkProtocol(nnx.Module, ABC):
        #     ...

        # RECOMMENDED: Rename to reflect it's an ABC
        class BenchmarkBase(nnx.Module, ABC):
            """Abstract base class for benchmarks."""

            @abstractmethod
            def run_training(self):
                pass

            @abstractmethod
            def run_evaluation(self):
                pass

        # AND/OR create an actual Protocol for duck typing
        @runtime_checkable
        class BenchmarkProtocol(Protocol):
            """Protocol for benchmark implementations."""

            def run_training(self) -> dict: ...
            def run_evaluation(self) -> dict: ...
            def get_performance_targets(self) -> dict: ...

        # Now we have both:
        # 1. BenchmarkBase for inheritance
        # 2. BenchmarkProtocol for duck typing

        class ConcreteBenchmark(BenchmarkBase):
            def run_training(self):
                return {"loss": 0.1}

            def run_evaluation(self):
                return {"accuracy": 0.95}

            def get_performance_targets(self):
                return {"accuracy": 0.9}

        benchmark = ConcreteBenchmark()

        # It's both a BenchmarkBase (inheritance) and BenchmarkProtocol (duck typing)
        assert isinstance(benchmark, BenchmarkBase)
        assert isinstance(benchmark, BenchmarkProtocol)
