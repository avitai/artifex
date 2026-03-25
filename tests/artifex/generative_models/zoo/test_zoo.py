"""Tests for the removed legacy model-zoo surface."""

import pytest

from artifex.generative_models.zoo import ModelZoo, zoo


class TestModelZoo:
    """Legacy model zoo should fail fast with migration guidance."""

    def test_model_zoo_constructor_raises_removed_contract_error(self):
        """Constructing ModelZoo should direct callers to typed config creation."""
        with pytest.raises(RuntimeError, match="typed config"):
            ModelZoo()

    def test_global_zoo_proxy_raises_removed_contract_error(self):
        """The retained zoo symbol should fail rather than pretend presets still work."""
        with pytest.raises(RuntimeError, match="typed config"):
            zoo.list_configs()
