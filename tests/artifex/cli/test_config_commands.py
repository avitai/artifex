"""Tests for config CLI commands using typer CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from artifex.cli.__main__ import app
from artifex.generative_models.core.configuration import TrainingConfig


runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[3]


class TestMainApp:
    """Test the main CLI app."""

    def test_help(self) -> None:
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "config" in result.output

    def test_version(self) -> None:
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "artifex" in result.output


class TestConfigCreate:
    """Test the config create command."""

    @patch("artifex.cli.config.create_config")
    def test_create_default(self, mock_create) -> None:
        """Test creating a default config."""
        mock_create.return_value = 0
        result = runner.invoke(app, ["config", "create", "simple_training", "output.yaml"])
        assert result.exit_code == 0
        mock_create.assert_called_once()

    @patch("artifex.cli.config.create_config")
    def test_create_with_format(self, mock_create) -> None:
        """Test creating a config with format option."""
        mock_create.return_value = 0
        result = runner.invoke(
            app,
            ["config", "create", "simple_training", "output.json", "-f", "json"],
        )
        assert result.exit_code == 0
        mock_create.assert_called_once()
        args = mock_create.call_args[0][0]
        assert args.format == "json"

    @patch("artifex.cli.config.create_config")
    def test_create_with_param(self, mock_create) -> None:
        """Test creating a config with typed template params."""
        mock_create.return_value = 0
        result = runner.invoke(
            app,
            [
                "config",
                "create",
                "simple_training",
                "output.yaml",
                "-p",
                "batch_size=64",
            ],
        )
        assert result.exit_code == 0
        args = mock_create.call_args[0][0]
        assert args.param == ["batch_size=64"]


class TestConfigValidate:
    """Test the config validate command."""

    @patch("artifex.cli.config.validate_config_file")
    def test_validate_success(self, mock_validate) -> None:
        """Test validating a config successfully."""
        mock_validate.return_value = 0
        result = runner.invoke(app, ["config", "validate", "test.yaml"])
        assert result.exit_code == 0
        mock_validate.assert_called_once()

    @patch("artifex.cli.config.validate_config_file")
    def test_validate_failure(self, mock_validate) -> None:
        """Test handling validation failure."""
        mock_validate.return_value = 1
        result = runner.invoke(app, ["config", "validate", "invalid.yaml"])
        assert result.exit_code == 1


class TestConfigShow:
    """Test the config show command."""

    @patch("artifex.cli.config.show_config")
    def test_show_default(self, mock_show) -> None:
        """Test showing a config in default format."""
        mock_show.return_value = 0
        result = runner.invoke(app, ["config", "show", "test.yaml"])
        assert result.exit_code == 0
        mock_show.assert_called_once()
        args = mock_show.call_args[0][0]
        assert args.format == "yaml"

    @patch("artifex.cli.config.show_config")
    def test_show_json(self, mock_show) -> None:
        """Test showing a config in JSON format."""
        mock_show.return_value = 0
        result = runner.invoke(app, ["config", "show", "test.yaml", "-f", "json"])
        assert result.exit_code == 0
        args = mock_show.call_args[0][0]
        assert args.format == "json"


class TestConfigDiff:
    """Test the config diff command."""

    @patch("artifex.cli.config.diff_config")
    def test_diff(self, mock_diff) -> None:
        """Test diffing two configs."""
        mock_diff.return_value = 0
        result = runner.invoke(app, ["config", "diff", "config1.yaml", "config2.yaml"])
        assert result.exit_code == 0
        mock_diff.assert_called_once()
        args = mock_diff.call_args[0][0]
        assert args.config1 == "config1.yaml"
        assert args.config2 == "config2.yaml"


class TestConfigVersion:
    """Test the config version command."""

    @patch("artifex.cli.config.version_config")
    def test_version_command(self, mock_version) -> None:
        """Test the version command."""
        mock_version.return_value = 0
        result = runner.invoke(app, ["config", "version", "test.yaml"])
        assert result.exit_code == 0
        mock_version.assert_called_once()

    @patch("artifex.cli.config.version_config")
    def test_version_with_description(self, mock_version) -> None:
        """Test versioning with description."""
        mock_version.return_value = 0
        result = runner.invoke(
            app,
            ["config", "version", "test.yaml", "-d", "Initial version"],
        )
        assert result.exit_code == 0
        args = mock_version.call_args[0][0]
        assert args.description == "Initial version"


class TestConfigList:
    """Test the config list command."""

    @patch("artifex.cli.config.list_configs")
    def test_list_default(self, mock_list) -> None:
        """Test listing configs with defaults."""
        mock_list.return_value = 0
        result = runner.invoke(app, ["config", "list"])
        assert result.exit_code == 0
        mock_list.assert_called_once()

    @patch("artifex.cli.config.list_configs")
    def test_list_with_limit(self, mock_list) -> None:
        """Test listing configs with limit."""
        mock_list.return_value = 0
        result = runner.invoke(app, ["config", "list", "-n", "5"])
        assert result.exit_code == 0
        args = mock_list.call_args[0][0]
        assert args.limit == 5


class TestConfigGet:
    """Test the config get command."""

    @patch("artifex.cli.config.get_config")
    def test_get_by_version(self, mock_get) -> None:
        """Test getting a config by version."""
        mock_get.return_value = 0
        result = runner.invoke(app, ["config", "get", "v1"])
        assert result.exit_code == 0
        mock_get.assert_called_once()
        args = mock_get.call_args[0][0]
        assert args.version_or_hash == "v1"

    @patch("artifex.cli.config.get_config")
    def test_get_with_output(self, mock_get) -> None:
        """Test getting a config with output file."""
        mock_get.return_value = 0
        result = runner.invoke(app, ["config", "get", "v1", "-o", "output.yaml"])
        assert result.exit_code == 0
        args = mock_get.call_args[0][0]
        assert args.output == "output.yaml"


class TestConfigCommandsEndToEnd:
    """End-to-end CLI tests against the retained typed config surface."""

    def test_create_generates_typed_training_config(self, tmp_path: Path) -> None:
        """create should generate a typed config from a supported template id."""
        output_path = tmp_path / "training.yaml"

        result = runner.invoke(
            app,
            [
                "config",
                "create",
                "simple_training",
                str(output_path),
                "-p",
                "batch_size=64",
                "-p",
                "learning_rate=0.0002",
            ],
        )

        assert result.exit_code == 0
        assert "Created configuration" in result.output

        config = TrainingConfig.from_yaml(output_path)
        assert config.batch_size == 64
        assert config.optimizer.learning_rate == 0.0002

    def test_create_rejects_unknown_template_identifier(self, tmp_path: Path) -> None:
        """create should not accept dead bundled asset paths as template ids."""
        output_path = tmp_path / "invalid.yaml"

        result = runner.invoke(
            app,
            [
                "config",
                "create",
                "src/artifex/cli/config/train/vae.yaml",
                str(output_path),
                "-p",
                "batch_size=64",
            ],
        )

        assert result.exit_code == 1
        assert "Template" in result.output
        assert "Traceback" not in result.output
        assert not output_path.exists()

    def test_create_rejects_dotted_params(self, tmp_path: Path) -> None:
        """create should not fabricate nested shapes from dotted keys."""
        output_path = tmp_path / "invalid.yaml"

        result = runner.invoke(
            app,
            [
                "config",
                "create",
                "simple_training",
                str(output_path),
                "-p",
                "optimizer.learning_rate=0.0002",
            ],
        )

        assert result.exit_code == 1
        assert "nested dotted keys" in result.output
        assert "Traceback" not in result.output

    def test_validate_accepts_retained_experiment_template(self) -> None:
        """validate should succeed on a retained typed experiment template."""
        config_path = (
            REPO_ROOT / "src/artifex/configs/experiments/protein_diffusion_experiment.yaml"
        )

        result = runner.invoke(app, ["config", "validate", str(config_path)])

        assert result.exit_code == 0
        assert "Configuration is valid" in result.output
        assert "ExperimentTemplateConfig" in result.output

    def test_validate_rejects_arbitrary_raw_yaml_mapping(self, tmp_path: Path) -> None:
        """validate should reject mappings that are not supported typed config documents."""
        config_path = tmp_path / "unsupported.yaml"
        config_path.write_text("name: unsupported\nfoo: bar\n", encoding="utf-8")

        result = runner.invoke(app, ["config", "validate", str(config_path)])

        assert result.exit_code == 1
        assert "supported typed config document" in result.output
        assert "Traceback" not in result.output

    def test_show_renders_supported_typed_config_as_json(self) -> None:
        """show should render a supported typed asset rather than raw YAML."""
        config_path = (
            REPO_ROOT / "src/artifex/configs/defaults/training/protein_diffusion_training.yaml"
        )

        result = runner.invoke(app, ["config", "show", str(config_path), "-f", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["name"] == "protein_diffusion_training"
        assert payload["optimizer"]["optimizer_type"] == "adamw"

    def test_show_rejects_unsupported_mapping_with_controlled_error(self, tmp_path: Path) -> None:
        """show should fail cleanly instead of echoing arbitrary YAML mappings."""
        config_path = tmp_path / "unsupported.yaml"
        config_path.write_text("name: unsupported\nfoo: bar\n", encoding="utf-8")

        result = runner.invoke(app, ["config", "show", str(config_path)])

        assert result.exit_code == 1
        assert "supported typed config document" in result.output
        assert "Traceback" not in result.output

    def test_create_supports_json_output_for_distributed_template(self, tmp_path: Path) -> None:
        """create should serialize typed distributed configs to JSON cleanly."""
        output_path = tmp_path / "distributed.json"

        result = runner.invoke(
            app,
            [
                "config",
                "create",
                "distributed_training",
                str(output_path),
                "-f",
                "json",
                "-p",
                "world_size=4",
                "-p",
                "num_nodes=1",
                "-p",
                "num_processes_per_node=4",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["name"] == "distributed_training"
        assert payload["world_size"] == 4
        assert payload["backend"] == "nccl"

    def test_validate_experiment_template_summary_uses_typed_sections(self) -> None:
        """validate should summarize typed override sections instead of blessing raw keys."""
        config_path = (
            REPO_ROOT / "src/artifex/configs/experiments/protein_diffusion_distributed.yaml"
        )

        result = runner.invoke(app, ["config", "validate", str(config_path)])

        assert result.exit_code == 0
        assert "overrides: model, data, training, inference, distributed" in result.output
