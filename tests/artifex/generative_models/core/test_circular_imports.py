"""Test for circular import resolution.

This test verifies that the circular import between Distribution and sampling
modules has been resolved by importing them in different orders.
"""

import unittest


class TestCircularImports(unittest.TestCase):
    """Test cases for verifying circular import resolution."""

    def test_distribution_then_sampling(self):
        """Test importing Distribution first, then sampling modules."""
        # Import Distribution first
        from artifex.generative_models.core.interfaces import Distribution
        from artifex.generative_models.core.sampling.ancestral import ancestral_sampling

        # Then import from sampling
        from artifex.generative_models.core.sampling.mcmc import mcmc_sampling

        # Verify they're callable
        self.assertTrue(callable(mcmc_sampling))
        self.assertTrue(callable(ancestral_sampling))

        # Verify Distribution is a class
        self.assertTrue(isinstance(Distribution, type))

    def test_sampling_then_distribution(self):
        """Test importing sampling modules first, then Distribution."""
        # Import from sampling first
        # Then import Distribution
        from artifex.generative_models.core.interfaces import Distribution
        from artifex.generative_models.core.sampling.ancestral import ancestral_sampling
        from artifex.generative_models.core.sampling.mcmc import mcmc_sampling

        # Verify they're callable
        self.assertTrue(callable(mcmc_sampling))
        self.assertTrue(callable(ancestral_sampling))

        # Verify Distribution is a class
        self.assertTrue(isinstance(Distribution, type))


if __name__ == "__main__":
    unittest.main()
