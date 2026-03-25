"""Tests for Hyperparameter search configuration frozen dataclass classes.

Tests cover SearchType enum, ParameterDistribution base,
CategoricalDistribution, LogUniformDistribution, UniformDistribution,
ChoiceDistribution, and HyperparamSearchConfig.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.base_dataclass import (
    BaseConfig,
    ConfigDocument,
)
from artifex.generative_models.core.configuration.hyperparam_config import (
    CategoricalDistribution,
    ChoiceDistribution,
    HyperparamSearchConfig,
    LogUniformDistribution,
    ParameterDistribution,
    SearchType,
    UniformDistribution,
)


# =============================================================================
# SearchType Enum Tests
# =============================================================================
class TestSearchType:
    """Test SearchType enum."""

    def test_enum_values(self):
        """Test that enum values match expected strings."""
        assert SearchType.GRID.value == "grid"
        assert SearchType.RANDOM.value == "random"
        assert SearchType.BAYESIAN.value == "bayesian"
        assert SearchType.POPULATION.value == "population"

    def test_str_enum_behavior(self):
        """Test that SearchType works as a string."""
        assert str(SearchType.GRID) == "grid"
        assert f"{SearchType.RANDOM}" == "random"

    def test_from_string(self):
        """Test creating enum from string value."""
        assert SearchType("grid") == SearchType.GRID
        assert SearchType("bayesian") == SearchType.BAYESIAN

    def test_invalid_search_type(self):
        """Test that invalid search type raises ValueError."""
        with pytest.raises(ValueError):
            SearchType("invalid")


# =============================================================================
# ParameterDistribution Tests
# =============================================================================
class TestParameterDistribution:
    """Test ParameterDistribution base class."""

    def test_create_with_defaults(self):
        """Test creation with default empty name."""
        dist = ParameterDistribution()
        assert dist.name == ""
        assert dist.param_path == ""

    def test_allows_empty_name(self):
        """Test that empty name is allowed (unlike BaseConfig)."""
        dist = ParameterDistribution(name="")
        assert dist.name == ""

    def test_frozen(self):
        """Test that distribution is frozen."""
        dist = ParameterDistribution()
        with pytest.raises(dataclasses.FrozenInstanceError):
            dist.name = "new"  # type: ignore

    def test_inherits_from_config_document_not_named_runtime_base(self):
        """Distributions are typed documents, not named runtime configs."""
        dist = ParameterDistribution()
        assert isinstance(dist, ConfigDocument)
        assert not isinstance(dist, BaseConfig)


# =============================================================================
# CategoricalDistribution Tests
# =============================================================================
class TestCategoricalDistribution:
    """Test CategoricalDistribution."""

    def test_create_with_categories(self):
        """Test creation with valid categories."""
        dist = CategoricalDistribution(categories=("a", "b", "c"))
        assert dist.categories == ("a", "b", "c")
        assert dist.type == "categorical"

    def test_empty_categories_raises(self):
        """Test that empty categories raises ValueError."""
        with pytest.raises(ValueError, match="categories.*not be empty"):
            CategoricalDistribution(categories=())

    def test_frozen(self):
        """Test that distribution is frozen."""
        dist = CategoricalDistribution(categories=("x",))
        with pytest.raises(dataclasses.FrozenInstanceError):
            dist.categories = ("y",)  # type: ignore

    def test_with_mixed_types(self):
        """Test categories with mixed types."""
        dist = CategoricalDistribution(categories=(1, "two", 3.0, True))
        assert len(dist.categories) == 4

    def test_serialization_roundtrip(self):
        """Test roundtrip serialization."""
        original = CategoricalDistribution(
            name="optimizer",
            categories=("adam", "sgd", "adamw"),
        )
        data = original.to_dict()
        restored = CategoricalDistribution.from_dict(data)
        assert original == restored


# =============================================================================
# LogUniformDistribution Tests
# =============================================================================
class TestLogUniformDistribution:
    """Test LogUniformDistribution."""

    def test_create_with_defaults(self):
        """Test creation with default low/high."""
        dist = LogUniformDistribution()
        assert dist.low == 1e-6
        assert dist.high == 1.0
        assert dist.type == "log_uniform"

    def test_create_with_custom_bounds(self):
        """Test creation with custom bounds."""
        dist = LogUniformDistribution(low=1e-4, high=1e-1)
        assert dist.low == 1e-4
        assert dist.high == 1e-1

    def test_low_must_be_positive(self):
        """Test that low <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="low"):
            LogUniformDistribution(low=0.0, high=1.0)

    def test_low_negative_raises(self):
        """Test that negative low raises ValueError."""
        with pytest.raises(ValueError, match="low"):
            LogUniformDistribution(low=-0.1, high=1.0)

    def test_high_must_exceed_low(self):
        """Test that high <= low raises ValueError."""
        with pytest.raises(ValueError, match="high.*greater.*low"):
            LogUniformDistribution(low=0.5, high=0.5)

    def test_high_less_than_low_raises(self):
        """Test that high < low raises ValueError."""
        with pytest.raises(ValueError, match="high.*greater.*low"):
            LogUniformDistribution(low=0.5, high=0.1)

    def test_serialization_roundtrip(self):
        """Test roundtrip serialization."""
        original = LogUniformDistribution(low=1e-5, high=0.01)
        data = original.to_dict()
        restored = LogUniformDistribution.from_dict(data)
        assert original == restored


# =============================================================================
# UniformDistribution Tests
# =============================================================================
class TestUniformDistribution:
    """Test UniformDistribution."""

    def test_create_with_defaults(self):
        """Test creation with default bounds."""
        dist = UniformDistribution()
        assert dist.low == 0.0
        assert dist.high == 1.0
        assert dist.q is None
        assert dist.log_scale is False
        assert dist.type == "uniform"

    def test_create_with_custom_bounds(self):
        """Test creation with custom bounds."""
        dist = UniformDistribution(low=-1.0, high=2.0)
        assert dist.low == -1.0
        assert dist.high == 2.0

    def test_high_must_exceed_low(self):
        """Test that high <= low raises ValueError."""
        with pytest.raises(ValueError, match="high.*greater.*low"):
            UniformDistribution(low=1.0, high=1.0)

    def test_high_less_than_low_raises(self):
        """Test that high < low raises ValueError."""
        with pytest.raises(ValueError, match="high.*greater.*low"):
            UniformDistribution(low=5.0, high=2.0)

    def test_quantization_step(self):
        """Test creation with quantization step."""
        dist = UniformDistribution(low=0.0, high=10.0, q=0.5)
        assert dist.q == 0.5

    def test_invalid_quantization_step_zero(self):
        """Test that q=0 raises ValueError."""
        with pytest.raises(ValueError, match="q"):
            UniformDistribution(low=0.0, high=1.0, q=0.0)

    def test_invalid_quantization_step_negative(self):
        """Test that negative q raises ValueError."""
        with pytest.raises(ValueError, match="q"):
            UniformDistribution(low=0.0, high=1.0, q=-0.5)

    def test_log_scale(self):
        """Test creation with log_scale."""
        dist = UniformDistribution(low=0.0, high=1.0, log_scale=True)
        assert dist.log_scale is True

    def test_serialization_roundtrip(self):
        """Test roundtrip serialization."""
        original = UniformDistribution(low=-5.0, high=5.0, q=0.1, log_scale=True)
        data = original.to_dict()
        restored = UniformDistribution.from_dict(data)
        assert original == restored


# =============================================================================
# ChoiceDistribution Tests
# =============================================================================
class TestChoiceDistribution:
    """Test ChoiceDistribution."""

    def test_create_with_choices(self):
        """Test creation with valid choices."""
        dist = ChoiceDistribution(choices=(16, 32, 64, 128))
        assert dist.choices == (16, 32, 64, 128)
        assert dist.type == "choice"
        assert dist.weights is None

    def test_empty_choices_raises(self):
        """Test that empty choices raises ValueError."""
        with pytest.raises(ValueError, match="choices.*not be empty"):
            ChoiceDistribution(choices=())

    def test_with_weights(self):
        """Test creation with matching weights."""
        dist = ChoiceDistribution(
            choices=("a", "b", "c"),
            weights=(0.5, 0.3, 0.2),
        )
        assert dist.weights == (0.5, 0.3, 0.2)

    def test_weights_length_mismatch(self):
        """Test that mismatched weights/choices lengths raise ValueError."""
        with pytest.raises(ValueError, match="weights.*choices"):
            ChoiceDistribution(choices=("a", "b"), weights=(0.5,))

    def test_negative_weight_raises(self):
        """Test that negative weight raises ValueError."""
        with pytest.raises(ValueError, match="weights.*non-negative"):
            ChoiceDistribution(choices=("a", "b"), weights=(-0.1, 1.1))

    def test_all_zero_weights_raises(self):
        """Test that all-zero weights raises ValueError."""
        with pytest.raises(ValueError, match="sum.*weights.*greater.*zero"):
            ChoiceDistribution(choices=("a", "b"), weights=(0.0, 0.0))

    def test_serialization_roundtrip(self):
        """Test roundtrip serialization."""
        original = ChoiceDistribution(
            choices=(1, 2, 3),
            weights=(0.5, 0.3, 0.2),
        )
        data = original.to_dict()
        restored = ChoiceDistribution.from_dict(data)
        assert original == restored


# =============================================================================
# HyperparamSearchConfig Tests
# =============================================================================
class TestHyperparamSearchConfigBasics:
    """Test basic functionality of HyperparamSearchConfig."""

    def test_create_with_defaults(self):
        """Test creation with default values."""
        config = HyperparamSearchConfig()
        assert config.name == "hyperparam_search"
        assert config.search_type == SearchType.RANDOM
        assert config.num_trials == 10

    def test_frozen(self):
        """Test that config is frozen."""
        config = HyperparamSearchConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.num_trials = 20  # type: ignore

    def test_inherits_from_base_config(self):
        """Test inheritance from BaseConfig."""
        config = HyperparamSearchConfig()
        assert isinstance(config, BaseConfig)


class TestHyperparamSearchConfigDefaults:
    """Test default values of HyperparamSearchConfig."""

    def test_default_search_type(self):
        """Test search_type defaults to RANDOM."""
        config = HyperparamSearchConfig()
        assert config.search_type == SearchType.RANDOM

    def test_default_max_parallel_trials(self):
        """Test max_parallel_trials defaults to 1."""
        config = HyperparamSearchConfig()
        assert config.max_parallel_trials == 1

    def test_default_search_space_empty(self):
        """Test search_space defaults to empty dict."""
        config = HyperparamSearchConfig()
        assert config.search_space == {}

    def test_default_early_stopping(self):
        """Test early_stopping defaults to False."""
        config = HyperparamSearchConfig()
        assert config.early_stopping is False

    def test_default_patience(self):
        """Test patience defaults to 10."""
        config = HyperparamSearchConfig()
        assert config.patience == 10

    def test_default_pruning(self):
        """Test pruning defaults to False."""
        config = HyperparamSearchConfig()
        assert config.pruning is False

    def test_default_direction(self):
        """Test direction defaults to 'minimize'."""
        config = HyperparamSearchConfig()
        assert config.direction == "minimize"

    def test_default_metric(self):
        """Test metric defaults to 'validation_loss'."""
        config = HyperparamSearchConfig()
        assert config.metric == "validation_loss"

    def test_default_seed(self):
        """Test seed defaults to 42."""
        config = HyperparamSearchConfig()
        assert config.seed == 42

    def test_default_tracking_uri_none(self):
        """Test tracking_uri defaults to None."""
        config = HyperparamSearchConfig()
        assert config.tracking_uri is None


class TestHyperparamSearchConfigValidation:
    """Test validation of HyperparamSearchConfig."""

    def test_invalid_num_trials_zero(self):
        """Test that num_trials=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_trials"):
            HyperparamSearchConfig(num_trials=0)

    def test_invalid_num_trials_negative(self):
        """Test that negative num_trials raises ValueError."""
        with pytest.raises(ValueError, match="num_trials"):
            HyperparamSearchConfig(num_trials=-1)

    def test_invalid_max_parallel_trials_zero(self):
        """Test that max_parallel_trials=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_parallel_trials"):
            HyperparamSearchConfig(max_parallel_trials=0)

    def test_invalid_patience_zero(self):
        """Test that patience=0 raises ValueError."""
        with pytest.raises(ValueError, match="patience"):
            HyperparamSearchConfig(patience=0)

    def test_invalid_pruning_interval_zero(self):
        """Test that pruning_interval=0 raises ValueError."""
        with pytest.raises(ValueError, match="pruning_interval"):
            HyperparamSearchConfig(pruning_interval=0)

    def test_max_parallel_exceeds_num_trials(self):
        """Test that max_parallel_trials > num_trials raises ValueError."""
        with pytest.raises(ValueError, match="max_parallel_trials.*num_trials"):
            HyperparamSearchConfig(num_trials=5, max_parallel_trials=10)

    def test_max_parallel_equals_num_trials_allowed(self):
        """Test that max_parallel_trials == num_trials is allowed."""
        config = HyperparamSearchConfig(num_trials=5, max_parallel_trials=5)
        assert config.max_parallel_trials == 5


class TestHyperparamSearchConfigWithSearchSpace:
    """Test HyperparamSearchConfig with search space distributions."""

    def test_with_uniform_distribution(self):
        """Test config with uniform distribution in search space."""
        config = HyperparamSearchConfig(
            search_space={
                "learning_rate": UniformDistribution(low=1e-4, high=1e-1),
            },
        )
        assert "learning_rate" in config.search_space
        assert isinstance(config.search_space["learning_rate"], UniformDistribution)

    def test_with_mixed_distributions(self):
        """Test config with multiple distribution types."""
        config = HyperparamSearchConfig(
            search_space={
                "lr": LogUniformDistribution(low=1e-5, high=0.01),
                "batch_size": ChoiceDistribution(choices=(16, 32, 64)),
                "optimizer": CategoricalDistribution(
                    categories=("adam", "sgd"),
                ),
                "dropout": UniformDistribution(low=0.0, high=0.5),
            },
        )
        assert len(config.search_space) == 4

    def test_with_all_search_types(self):
        """Test that all SearchType values are accepted."""
        for search_type in SearchType:
            config = HyperparamSearchConfig(search_type=search_type)
            assert config.search_type == search_type


class TestHyperparamSearchConfigSerialization:
    """Test serialization of HyperparamSearchConfig."""

    def test_to_dict_simple(self):
        """Test to_dict for simple config."""
        config = HyperparamSearchConfig(
            num_trials=20,
            direction="maximize",
            metric="accuracy",
        )
        data = config.to_dict()
        assert data["num_trials"] == 20
        assert data["direction"] == "maximize"
        assert data["metric"] == "accuracy"

    def test_roundtrip_simple(self):
        """Test roundtrip serialization without search space."""
        original = HyperparamSearchConfig(
            name="my_search",
            search_type=SearchType.BAYESIAN,
            num_trials=50,
            max_parallel_trials=4,
            early_stopping=True,
            patience=5,
            direction="maximize",
            metric="f1_score",
            seed=123,
        )
        data = original.to_dict()
        restored = HyperparamSearchConfig.from_dict(data)
        assert restored.name == "my_search"
        assert restored.num_trials == 50
        assert restored.max_parallel_trials == 4
        assert restored.early_stopping is True
        assert restored.direction == "maximize"
