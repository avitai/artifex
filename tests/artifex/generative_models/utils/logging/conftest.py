"""
Configuration file for pytest in the logging tests directory.

This file ensures proper import paths for the logging tests.
"""

import tempfile

import pytest


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for test logs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
