import sys
import unittest
from io import StringIO
from unittest.mock import patch

# Import the CLI module directly
from artifex.configs.cli import main


class CaptureOutput:
    """Context manager to capture stdout and stderr."""

    def __init__(self):
        self.stdout = StringIO()
        self.stderr = StringIO()
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


class TestConfigCreate(unittest.TestCase):
    """Test the config create command."""

    @patch("artifex.configs.cli.handle_create")
    def test_create_default(self, mock_create):
        """Test creating a default config."""
        mock_create.return_value = 0

        with patch.object(
            sys,
            "argv",
            ["cli.py", "create", "test_config.yaml", "output_config.yaml"],
        ):
            result = main()

        self.assertEqual(result, 0)
        mock_create.assert_called_once()

    @patch("artifex.configs.cli.handle_create")
    def test_create_with_params(self, mock_create):
        """Test creating a config with parameters."""
        mock_create.return_value = 0

        with patch.object(
            sys,
            "argv",
            [
                "cli.py",
                "create",
                "template_config.yaml",
                "output_config.yaml",
                "--format",
                "yaml",
            ],
        ):
            result = main()

        self.assertEqual(result, 0)
        mock_create.assert_called_once()

    @patch("artifex.configs.cli.handle_create")
    def test_create_error(self, mock_create):
        """Test error handling in create command."""
        mock_create.return_value = 1

        with patch.object(
            sys,
            "argv",
            ["cli.py", "create", "invalid_config.yaml", "output_config.yaml"],
        ):
            result = main()

        self.assertEqual(result, 1)
        mock_create.assert_called_once()


class TestConfigValidate(unittest.TestCase):
    """Test the config validate command."""

    @patch("artifex.configs.cli.handle_validate")
    def test_validate_success(self, mock_validate):
        """Test validating a config successfully."""
        mock_validate.return_value = 0

        with patch.object(sys, "argv", ["cli.py", "validate", "test_config.yaml"]):
            result = main()

        self.assertEqual(result, 0)
        mock_validate.assert_called_once()

    @patch("artifex.configs.cli.handle_validate")
    def test_validate_with_schema(self, mock_validate):
        """Test validating a config with schema specified."""
        mock_validate.return_value = 0

        with patch.object(
            sys,
            "argv",
            ["cli.py", "validate", "test_config.yaml", "--schema", "diffusion"],
        ):
            result = main()

        self.assertEqual(result, 0)
        mock_validate.assert_called_once()

    @patch("artifex.configs.cli.handle_validate")
    def test_validate_failure(self, mock_validate):
        """Test handling validation failure."""
        mock_validate.return_value = 1

        with patch.object(sys, "argv", ["cli.py", "validate", "invalid_config.yaml"]):
            result = main()

        self.assertEqual(result, 1)
        mock_validate.assert_called_once()


class TestConfigShow(unittest.TestCase):
    """Test the config show command."""

    @patch("artifex.configs.cli.handle_show")
    def test_show_default(self, mock_show):
        """Test showing a config in default format."""
        mock_show.return_value = 0

        with patch.object(sys, "argv", ["cli.py", "show", "test_config.yaml"]):
            result = main()

        self.assertEqual(result, 0)
        mock_show.assert_called_once()

    @patch("artifex.configs.cli.handle_show")
    def test_show_json_format(self, mock_show):
        """Test showing a config in JSON format."""
        mock_show.return_value = 0

        with patch.object(
            sys,
            "argv",
            ["cli.py", "show", "test_config.yaml", "--format", "json"],
        ):
            result = main()

        self.assertEqual(result, 0)
        mock_show.assert_called_once()

    @patch("artifex.configs.cli.handle_show")
    def test_show_error(self, mock_show):
        """Test error handling in show command."""
        mock_show.return_value = 1

        with patch.object(sys, "argv", ["cli.py", "show", "nonexistent_config.yaml"]):
            result = main()

        self.assertEqual(result, 1)
        mock_show.assert_called_once()


class TestConfigVersion(unittest.TestCase):
    """Test the config version command."""

    @patch("artifex.configs.cli.handle_version")
    def test_version_command(self, mock_version):
        """Test the version command."""
        mock_version.return_value = 0

        with patch.object(sys, "argv", ["cli.py", "version", "test_config.yaml"]):
            result = main()

        self.assertEqual(result, 0)
        mock_version.assert_called_once()

    @patch("artifex.configs.cli.handle_version")
    def test_version_error(self, mock_version):
        """Test error handling in version command."""
        mock_version.return_value = 1

        with patch.object(sys, "argv", ["cli.py", "version", "nonexistent_config.yaml"]):
            result = main()

        self.assertEqual(result, 1)
        mock_version.assert_called_once()


class TestConfigDiff(unittest.TestCase):
    """Test the config diff command."""

    @patch("artifex.configs.cli.handle_diff")
    def test_diff_command(self, mock_diff):
        """Test the diff command."""
        mock_diff.return_value = 0

        with patch.object(sys, "argv", ["cli.py", "diff", "config1.yaml", "config2.yaml"]):
            result = main()

        self.assertEqual(result, 0)
        mock_diff.assert_called_once()


if __name__ == "__main__":
    unittest.main()
