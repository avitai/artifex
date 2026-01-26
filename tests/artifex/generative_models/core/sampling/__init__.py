"""Tests for artifex.generative_models.core.sampling."""


def should_skip_blackjax_tests():
    """Check if BlackJAX tests should be skipped.

    Returns:
        bool: True if tests should be skipped, False otherwise.

    Environment variables that control BlackJAX test execution:

    - SKIP_BLACKJAX_TESTS: When set to any non-empty value, disables BlackJAX
      tests
      Example: SKIP_BLACKJAX_TESTS=1 pytest

    - ENABLE_BLACKJAX_TESTS: When set to any non-empty value, enables BlackJAX
      tests regardless of SKIP_BLACKJAX_TESTS setting
      Example: ENABLE_BLACKJAX_TESTS=1 pytest
    """
    import os

    # If explicitly enabled, run the tests
    if os.environ.get("ENABLE_BLACKJAX_TESTS", "0") == "1":
        return False

    # If explicitly disabled, skip the tests
    if os.environ.get("SKIP_BLACKJAX_TESTS", "0") != "0":
        return True

    # Default to skipping (since BlackJAX is optional)
    return True
