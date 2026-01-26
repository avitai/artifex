"""Tests for the test_helpers module."""

import os
import unittest
from unittest import mock

import pytest

from tests.utils.test_helpers import (
    get_mock_reason,
    is_running_on_cpu,
    is_running_on_gpu,
    should_run_diffusion_tests,
    should_run_flow_tests,
    should_run_gan_tests,
    should_run_integration_tests,
    should_run_training_pipeline_tests,
    should_run_vae_tests,
    verify_model_initialization_shapes,
)


class TestEnvironmentVariableControl(unittest.TestCase):
    """Test the environment variable based test execution control functions."""

    def setUp(self):
        """Set up the test environment by clearing environment variables."""
        # Save original environment variables
        self.original_env = os.environ.copy()

        # Clear all test-related environment variables
        for var in [
            "RUN_INTEGRATION_TESTS",
            "SKIP_INTEGRATION_TESTS",
            "RUN_VAE_TESTS",
            "SKIP_VAE_TESTS",
            "RUN_FLOW_TESTS",
            "SKIP_FLOW_TESTS",
            "RUN_GAN_TESTS",
            "SKIP_GAN_TESTS",
            "RUN_DIFFUSION_TESTS",
            "SKIP_DIFFUSION_TESTS",
            "RUN_TRAINING_PIPELINE_TESTS",
            "SKIP_TRAINING_PIPELINE_TESTS",
            "JAX_PLATFORMS",
        ]:
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore the original environment variables."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_should_run_integration_tests_default(self):
        """Test the default behavior of should_run_integration_tests."""
        # By default, integration tests should run
        self.assertTrue(should_run_integration_tests())

    def test_should_run_integration_tests_skip(self):
        """Test skipping integration tests with SKIP_INTEGRATION_TESTS."""
        # When SKIP_INTEGRATION_TESTS is set, integration tests should not run
        os.environ["SKIP_INTEGRATION_TESTS"] = "1"
        self.assertFalse(should_run_integration_tests())

    def test_should_run_integration_tests_run(self):
        """Test running integration tests with RUN_INTEGRATION_TESTS."""
        # When RUN_INTEGRATION_TESTS is set, integration tests should run
        # even if SKIP_INTEGRATION_TESTS is set
        os.environ["SKIP_INTEGRATION_TESTS"] = "1"
        os.environ["RUN_INTEGRATION_TESTS"] = "1"
        self.assertTrue(should_run_integration_tests())

    def test_should_run_vae_tests_default(self):
        """Test the default behavior of should_run_vae_tests."""
        # By default, VAE tests should follow integration test setting
        self.assertTrue(should_run_vae_tests())

        os.environ["SKIP_INTEGRATION_TESTS"] = "1"
        self.assertFalse(should_run_vae_tests())

    def test_should_run_vae_tests_specific(self):
        """Test specific control of VAE tests."""
        # RUN_VAE_TESTS should override SKIP_INTEGRATION_TESTS
        os.environ["SKIP_INTEGRATION_TESTS"] = "1"
        os.environ["RUN_VAE_TESTS"] = "1"
        self.assertTrue(should_run_vae_tests())

        # SKIP_VAE_TESTS should override RUN_INTEGRATION_TESTS
        os.environ.clear()
        os.environ["RUN_INTEGRATION_TESTS"] = "1"
        os.environ["SKIP_VAE_TESTS"] = "1"
        self.assertFalse(should_run_vae_tests())

        # RUN_VAE_TESTS should override SKIP_VAE_TESTS
        os.environ["RUN_VAE_TESTS"] = "1"
        os.environ["SKIP_VAE_TESTS"] = "1"
        self.assertTrue(should_run_vae_tests())

    def test_should_run_flow_tests(self):
        """Test the behavior of should_run_flow_tests."""
        # By default, Flow tests should follow integration test setting
        self.assertTrue(should_run_flow_tests())

        os.environ["SKIP_INTEGRATION_TESTS"] = "1"
        self.assertFalse(should_run_flow_tests())

        # RUN_FLOW_TESTS should override SKIP_INTEGRATION_TESTS
        os.environ["RUN_FLOW_TESTS"] = "1"
        self.assertTrue(should_run_flow_tests())

    def test_should_run_gan_tests(self):
        """Test the behavior of should_run_gan_tests."""
        # By default, GAN tests should follow integration test setting
        self.assertTrue(should_run_gan_tests())

        os.environ["SKIP_INTEGRATION_TESTS"] = "1"
        self.assertFalse(should_run_gan_tests())

        # RUN_GAN_TESTS should override SKIP_INTEGRATION_TESTS
        os.environ["RUN_GAN_TESTS"] = "1"
        self.assertTrue(should_run_gan_tests())

    def test_should_run_diffusion_tests(self):
        """Test the behavior of should_run_diffusion_tests."""
        # By default, Diffusion tests should be disabled due to known GroupNorm issues
        self.assertFalse(should_run_diffusion_tests())

        # SKIP_INTEGRATION_TESTS should have no effect (already false by default)
        os.environ["SKIP_INTEGRATION_TESTS"] = "1"
        self.assertFalse(should_run_diffusion_tests())

        # RUN_DIFFUSION_TESTS should override the default behavior
        os.environ["RUN_DIFFUSION_TESTS"] = "1"
        self.assertTrue(should_run_diffusion_tests())

    def test_should_run_training_pipeline_tests(self):
        """Test the behavior of should_run_training_pipeline_tests."""
        # By default, pipeline tests should follow integration test setting
        self.assertTrue(should_run_training_pipeline_tests())

        os.environ["SKIP_INTEGRATION_TESTS"] = "1"
        self.assertFalse(should_run_training_pipeline_tests())

        # RUN_TRAINING_PIPELINE_TESTS should override SKIP_INTEGRATION_TESTS
        os.environ["RUN_TRAINING_PIPELINE_TESTS"] = "1"
        self.assertTrue(should_run_training_pipeline_tests())


class TestPlatformDetection(unittest.TestCase):
    """Test the platform detection utilities."""

    def setUp(self):
        """Set up the test environment by clearing environment variables."""
        # Save original environment variables
        self.original_env = os.environ.copy()

        # Clear JAX_PLATFORMS environment variable
        if "JAX_PLATFORMS" in os.environ:
            del os.environ["JAX_PLATFORMS"]

    def tearDown(self):
        """Restore the original environment variables."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_is_running_on_cpu(self):
        """Test the is_running_on_cpu function."""
        # By default, JAX_PLATFORMS is not set to 'cpu'
        self.assertFalse(is_running_on_cpu())

        # When JAX_PLATFORMS is set to 'cpu', it should return True
        os.environ["JAX_PLATFORMS"] = "cpu"
        self.assertTrue(is_running_on_cpu())

        # When JAX_PLATFORMS is set to something else, it should return False
        os.environ["JAX_PLATFORMS"] = "gpu"
        self.assertFalse(is_running_on_cpu())

    def test_is_running_on_gpu(self):
        """Test the is_running_on_gpu function."""
        # By default, JAX_PLATFORMS is not set to 'cpu', so should be GPU
        self.assertTrue(is_running_on_gpu())

        # When JAX_PLATFORMS is set to 'cpu', it should return False
        os.environ["JAX_PLATFORMS"] = "cpu"
        self.assertFalse(is_running_on_gpu())

        # When JAX_PLATFORMS is set to something else, it should return True
        os.environ["JAX_PLATFORMS"] = "gpu"
        self.assertTrue(is_running_on_gpu())


class TestMockUtilities(unittest.TestCase):
    """Test the mock utilities."""

    def test_get_mock_reason(self):
        """Test the get_mock_reason function."""
        # Test with default environment variable
        reason = get_mock_reason("vae")
        self.assertEqual(
            reason,
            "vae integration test skipped. "
            "Set RUN_VAE_TESTS=1 to enable or RUN_INTEGRATION_TESTS=1 to "
            "enable all integration tests.",
        )

        # Test with custom environment variable
        reason = get_mock_reason("vae", "CUSTOM_VAR")
        self.assertEqual(
            reason,
            "vae integration test skipped. "
            "Set CUSTOM_VAR=1 to enable or RUN_INTEGRATION_TESTS=1 to "
            "enable all integration tests.",
        )


class TestShapeVerification(unittest.TestCase):
    """Test the shape verification utilities."""

    def test_verify_model_initialization_shapes(self):
        """Test the verify_model_initialization_shapes function."""

        # Create a mock model class and config
        class MockModel:
            def __init__(self, config):
                self.config = config
                self.param1 = mock.MagicMock(shape=(1, 2, 3))
                self.param2 = mock.MagicMock(shape=(4, 5, 6))

        config = {}
        expected_shapes = {
            "param1": (1, 2, 3),
            "param2": (4, 5, 6),
        }

        # Test with matching shapes
        self.assertTrue(verify_model_initialization_shapes(MockModel, config, expected_shapes))

        # Test with missing parameter
        with pytest.raises(AssertionError) as excinfo:
            verify_model_initialization_shapes(MockModel, config, {"param3": (7, 8, 9)})
        self.assertIn("Model missing expected parameter param3", str(excinfo.value))

        # Test with mismatched shape
        with pytest.raises(AssertionError) as excinfo:
            verify_model_initialization_shapes(MockModel, config, {"param1": (7, 8, 9)})
        self.assertIn(
            "Parameter param1 has shape (1, 2, 3), expected (7, 8, 9)",
            str(excinfo.value),
        )
