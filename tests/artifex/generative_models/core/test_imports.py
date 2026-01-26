"""Tests for import integrity in core modules."""

import importlib
import unittest


class TestCoreImports(unittest.TestCase):
    """Test cases for verifying imports work correctly."""

    def test_interfaces_import(self):
        """Test importing the interfaces module."""
        module = importlib.import_module("artifex.generative_models.core.interfaces")
        self.assertTrue(hasattr(module, "Distribution"))
        self.assertTrue(isinstance(getattr(module, "Distribution"), type))

    def test_distributions_import(self):
        """Test importing the distributions module."""
        module = importlib.import_module("artifex.generative_models.core.distributions.base")
        self.assertTrue(hasattr(module, "Distribution"))
        self.assertTrue(isinstance(getattr(module, "Distribution"), type))

    def test_sampling_mcmc_import(self):
        """Test importing the sampling.mcmc module."""
        module = importlib.import_module("artifex.generative_models.core.sampling.mcmc")
        self.assertTrue(hasattr(module, "mcmc_sampling"))
        self.assertTrue(callable(getattr(module, "mcmc_sampling")))

    def test_sampling_ancestral_import(self):
        """Test importing the sampling.ancestral module."""
        module = importlib.import_module("artifex.generative_models.core.sampling.ancestral")
        self.assertTrue(hasattr(module, "ancestral_sampling"))
        self.assertTrue(callable(getattr(module, "ancestral_sampling")))

    def test_import_all_together(self):
        """Test importing multiple modules together to verify no circular imports."""
        # These imports should not raise any exceptions
        from artifex.generative_models.core.interfaces import Distribution
        from artifex.generative_models.core.sampling.ancestral import ancestral_sampling
        from artifex.generative_models.core.sampling.mcmc import mcmc_sampling

        # Verify the imports work
        self.assertTrue(isinstance(Distribution, type))
        self.assertTrue(callable(mcmc_sampling))
        self.assertTrue(callable(ancestral_sampling))


if __name__ == "__main__":
    unittest.main()
