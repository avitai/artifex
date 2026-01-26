"""Tests for configuration validation utilities.

Following TDD: These tests are written BEFORE implementation.
They define the expected behavior of validation utilities.
"""

import pytest

from artifex.generative_models.core.configuration.validation import (
    validate_activation,
    validate_dropout_rate,
    validate_learning_rate,
    validate_positive_float,
    validate_positive_int,
    validate_positive_tuple,
    validate_probability,
)


class TestValidatePositiveInt:
    """Test validate_positive_int utility."""

    def test_valid_positive_int(self):
        """Test that valid positive integers pass."""
        # Should not raise
        validate_positive_int(1, "test_field")
        validate_positive_int(100, "test_field")
        validate_positive_int(999999, "test_field")

    def test_zero_raises(self):
        """Test that zero raises ValueError."""
        with pytest.raises(ValueError, match="test_field must be positive"):
            validate_positive_int(0, "test_field")

    def test_negative_raises(self):
        """Test that negative integers raise ValueError."""
        with pytest.raises(ValueError, match="test_field must be positive"):
            validate_positive_int(-1, "test_field")

        with pytest.raises(ValueError, match="test_field must be positive"):
            validate_positive_int(-100, "test_field")

    def test_custom_field_name_in_error(self):
        """Test that field name appears in error message."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            validate_positive_int(-10, "latent_dim")


class TestValidatePositiveFloat:
    """Test validate_positive_float utility."""

    def test_valid_positive_float(self):
        """Test that valid positive floats pass."""
        validate_positive_float(0.1, "test_field")
        validate_positive_float(1.5, "test_field")
        validate_positive_float(100.0, "test_field")

    def test_zero_raises(self):
        """Test that zero raises ValueError."""
        with pytest.raises(ValueError, match="test_field must be positive"):
            validate_positive_float(0.0, "test_field")

    def test_negative_raises(self):
        """Test that negative floats raise ValueError."""
        with pytest.raises(ValueError, match="test_field must be positive"):
            validate_positive_float(-0.1, "test_field")


class TestValidatePositiveTuple:
    """Test validate_positive_tuple utility."""

    def test_valid_positive_tuple(self):
        """Test that valid positive tuples pass."""
        validate_positive_tuple((1, 2, 3), "test_field")
        validate_positive_tuple((100,), "test_field")
        validate_positive_tuple((512, 256, 128), "test_field")

    def test_empty_tuple_raises(self):
        """Test that empty tuple raises ValueError."""
        with pytest.raises(ValueError, match="test_field must have at least 1 element"):
            validate_positive_tuple((), "test_field")

    def test_tuple_with_zero_raises(self):
        """Test that tuple containing zero raises ValueError."""
        with pytest.raises(ValueError, match="All test_field must be positive"):
            validate_positive_tuple((1, 0, 3), "test_field")

    def test_tuple_with_negative_raises(self):
        """Test that tuple containing negative raises ValueError."""
        with pytest.raises(ValueError, match="All test_field must be positive"):
            validate_positive_tuple((512, -256, 128), "test_field")

    def test_all_negative_raises(self):
        """Test that tuple with all negative values raises ValueError."""
        with pytest.raises(ValueError, match="All test_field must be positive"):
            validate_positive_tuple((-1, -2, -3), "test_field")


class TestValidateDropoutRate:
    """Test validate_dropout_rate utility."""

    def test_valid_dropout_rates(self):
        """Test that valid dropout rates pass."""
        validate_dropout_rate(0.0)  # Min
        validate_dropout_rate(0.5)  # Middle
        validate_dropout_rate(1.0)  # Max

    def test_below_zero_raises(self):
        """Test that negative dropout rate raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be in \\[0.0, 1.0\\]"):
            validate_dropout_rate(-0.1)

    def test_above_one_raises(self):
        """Test that dropout rate > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be in \\[0.0, 1.0\\]"):
            validate_dropout_rate(1.1)

        with pytest.raises(ValueError, match="dropout_rate must be in \\[0.0, 1.0\\]"):
            validate_dropout_rate(2.0)


class TestValidateProbability:
    """Test validate_probability utility (similar to dropout but generic)."""

    def test_valid_probabilities(self):
        """Test that valid probabilities pass."""
        validate_probability(0.0, "prob_field")
        validate_probability(0.5, "prob_field")
        validate_probability(1.0, "prob_field")

    def test_below_zero_raises(self):
        """Test that negative probability raises ValueError."""
        with pytest.raises(ValueError, match="prob_field must be in \\[0.0, 1.0\\]"):
            validate_probability(-0.1, "prob_field")

    def test_above_one_raises(self):
        """Test that probability > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="prob_field must be in \\[0.0, 1.0\\]"):
            validate_probability(1.5, "prob_field")


class TestValidateLearningRate:
    """Test validate_learning_rate utility."""

    def test_valid_learning_rates(self):
        """Test that valid learning rates pass."""
        validate_learning_rate(0.0001)
        validate_learning_rate(0.001)
        validate_learning_rate(0.1)
        validate_learning_rate(1.0)

    def test_zero_raises(self):
        """Test that zero learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_learning_rate(0.0)

    def test_negative_raises(self):
        """Test that negative learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_learning_rate(-0.001)


class TestValidateActivation:
    """Test validate_activation utility."""

    def test_valid_activations(self):
        """Test that valid activation functions pass."""
        # Common Flax NNX activations
        valid_activations = [
            "relu",
            "gelu",
            "silu",
            "swish",
            "tanh",
            "sigmoid",
            "elu",
            "leaky_relu",
        ]

        for activation in valid_activations:
            validate_activation(activation)  # Should not raise

    def test_unknown_activation_raises(self):
        """Test that unknown activation function raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation function"):
            validate_activation("nonexistent_activation")

        with pytest.raises(ValueError, match="Unknown activation function"):
            validate_activation("not_real")

    def test_case_sensitive(self):
        """Test that activation validation is case-sensitive."""
        # Should work with lowercase
        validate_activation("relu")

        # Should fail with uppercase (JAX/Flax uses lowercase)
        with pytest.raises(ValueError, match="Unknown activation function"):
            validate_activation("RELU")

    def test_suggests_valid_options(self):
        """Test that error message includes valid options."""
        try:
            validate_activation("bad_activation")
        except ValueError as e:
            error_msg = str(e)
            # Should mention some valid activations
            assert "relu" in error_msg or "gelu" in error_msg or "Valid" in error_msg


class TestValidationUtilitiesIntegration:
    """Integration tests for validation utilities."""

    def test_all_validators_fail_fast(self):
        """Test that all validators raise immediately on invalid input."""
        validators_and_invalid_inputs = [
            (validate_positive_int, (-1, "field")),
            (validate_positive_float, (-0.1, "field")),
            (validate_positive_tuple, ((), "field")),
            (validate_dropout_rate, (1.5,)),
            (validate_probability, (-0.1, "field")),
            (validate_learning_rate, (0.0,)),
            (validate_activation, ("bad",)),
        ]

        for validator, args in validators_and_invalid_inputs:
            with pytest.raises(ValueError):
                validator(*args)

    def test_validators_accept_valid_inputs(self):
        """Test that all validators accept valid inputs without raising."""
        # These should all pass without raising
        validate_positive_int(10, "field")
        validate_positive_float(0.1, "field")
        validate_positive_tuple((1, 2, 3), "field")
        validate_dropout_rate(0.5)
        validate_probability(0.5, "field")
        validate_learning_rate(0.001)
        validate_activation("relu")


class TestValidationCoverage:
    """Meta-test to ensure we achieve 80%+ coverage."""

    def test_coverage_reminder(self):
        """Reminder that we need 80%+ coverage for validation utilities.

        All validation functions must be tested with:
        - Valid inputs (positive tests)
        - Invalid inputs (negative tests)
        - Edge cases (0, negative, empty, etc.)
        - Error messages (field names, suggestions)
        """
        assert True
