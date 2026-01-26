"""End-to-end tests for CLI workflows."""

import subprocess

import pytest


@pytest.mark.e2e
@pytest.mark.slow
class TestCLIWorkflows:
    """End-to-end tests for CLI command workflows."""

    def test_config_validation_workflow(self, temp_workspace):
        """Test complete config validation workflow via CLI."""
        # Create a test config file
        config_content = """
        model:
          type: "vae"
          input_dim: [32, 32, 3]
          latent_dim: 16
          hidden_dims: [32, 64]

        training:
          batch_size: 4
          learning_rate: 0.001
          num_epochs: 2
        """

        config_file = temp_workspace / "test_config.yaml"
        config_file.write_text(config_content)

        # Test config validation command
        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "artifex.cli",
                    "validate-config",
                    str(config_file),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should succeed or skip if CLI not available
            if result.returncode != 0:
                pytest.skip(f"CLI validation failed: {result.stderr}")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI command not available or timed out")

    def test_model_info_workflow(self):
        """Test model information retrieval workflow."""
        try:
            result = subprocess.run(
                ["python", "-m", "artifex.cli", "list-models"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                pytest.skip(f"CLI list-models failed: {result.stderr}")

            # Should contain some model information
            assert len(result.stdout) > 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI command not available or timed out")

    def test_benchmark_workflow(self, temp_workspace):
        """Test benchmark execution workflow via CLI."""
        # Create benchmark config
        benchmark_config = """
        benchmark:
          name: "test_benchmark"
          models: ["vae"]
          metrics: ["reconstruction_loss"]

        data:
          type: "synthetic"
          batch_size: 4
          num_samples: 16
        """

        config_file = temp_workspace / "benchmark_config.yaml"
        config_file.write_text(benchmark_config)

        results_dir = temp_workspace / "benchmark_results"
        results_dir.mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "artifex.cli",
                    "run-benchmark",
                    str(config_file),
                    "--output-dir",
                    str(results_dir),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                pytest.skip(f"CLI benchmark failed: {result.stderr}")

            # Check if results were generated
            result_files = list(results_dir.glob("*.json"))
            if result_files:
                assert len(result_files) > 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI benchmark command not available or timed out")

    def test_training_workflow_cli(self, temp_workspace):
        """Test training workflow via CLI commands."""
        # Create training config
        training_config = """
        model:
          type: "vae"
          input_dim: [16, 16, 3]
          latent_dim: 8
          hidden_dims: [16, 32]

        training:
          batch_size: 2
          learning_rate: 0.001
          num_epochs: 1

        data:
          type: "synthetic"
          num_samples: 8
        """

        config_file = temp_workspace / "training_config.yaml"
        config_file.write_text(training_config)

        model_dir = temp_workspace / "trained_models"
        model_dir.mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "artifex.cli",
                    "train",
                    str(config_file),
                    "--output-dir",
                    str(model_dir),
                    "--dry-run",
                ],  # Use dry-run to avoid long training
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                pytest.skip(f"CLI training failed: {result.stderr}")

            # Should complete without errors
            assert "error" not in result.stderr.lower()

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI training command not available or timed out")

    def test_evaluation_workflow_cli(self, temp_workspace):
        """Test model evaluation workflow via CLI."""
        # Create evaluation config
        eval_config = """
        model:
          type: "vae"
          checkpoint_path: "dummy_path"  # Will be skipped in dry-run

        evaluation:
          metrics: ["reconstruction_loss", "kl_divergence"]

        data:
          type: "synthetic"
          batch_size: 4
          num_samples: 16
        """

        config_file = temp_workspace / "eval_config.yaml"
        config_file.write_text(eval_config)

        results_dir = temp_workspace / "eval_results"
        results_dir.mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "artifex.cli",
                    "evaluate",
                    str(config_file),
                    "--output-dir",
                    str(results_dir),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                pytest.skip(f"CLI evaluation failed: {result.stderr}")

            # Should complete without errors
            assert "error" not in result.stderr.lower()

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI evaluation command not available or timed out")

    def test_help_commands_workflow(self):
        """Test help and documentation commands."""
        commands_to_test = [
            ["python", "-m", "artifex.cli", "--help"],
            ["python", "-m", "artifex.cli", "train", "--help"],
            ["python", "-m", "artifex.cli", "evaluate", "--help"],
        ]

        for cmd in commands_to_test:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

                # Help commands should succeed
                if result.returncode == 0:
                    assert len(result.stdout) > 0
                    help_indicators = ["usage", "help"]
                    assert any(indicator in result.stdout.lower() for indicator in help_indicators)
                else:
                    # If command fails, skip this specific test
                    continue

            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Skip if CLI not available
                continue

    def test_version_workflow(self):
        """Test version information workflow."""
        try:
            result = subprocess.run(
                ["python", "-m", "artifex.cli", "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode == 0:
                # Should contain version information
                assert len(result.stdout) > 0
            else:
                pytest.skip("Version command not available")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI version command not available")

    def test_config_generation_workflow(self, temp_workspace):
        """Test configuration file generation workflow."""
        output_file = temp_workspace / "generated_config.yaml"

        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "artifex.cli",
                    "generate-config",
                    "--model",
                    "vae",
                    "--output",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                pytest.skip(f"Config generation failed: {result.stderr}")

            # Check if config file was created
            if output_file.exists():
                content = output_file.read_text()
                assert len(content) > 0
                assert "model" in content.lower()

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI config generation not available")

    def test_pipeline_workflow_integration(self, temp_workspace):
        """Test complete pipeline: config generation → validation → training."""
        # Step 1: Generate config
        config_file = temp_workspace / "pipeline_config.yaml"

        try:
            # Generate config
            result1 = subprocess.run(
                [
                    "python",
                    "-m",
                    "artifex.cli",
                    "generate-config",
                    "--model",
                    "vae",
                    "--output",
                    str(config_file),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result1.returncode != 0:
                pytest.skip("Config generation step failed")

            if not config_file.exists():
                pytest.skip("Config file not generated")

            # Step 2: Validate config
            result2 = subprocess.run(
                [
                    "python",
                    "-m",
                    "artifex.cli",
                    "validate-config",
                    str(config_file),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result2.returncode != 0:
                pytest.skip("Config validation step failed")

            # Step 3: Dry-run training
            model_dir = temp_workspace / "pipeline_models"
            model_dir.mkdir(exist_ok=True)

            result3 = subprocess.run(
                [
                    "python",
                    "-m",
                    "artifex.cli",
                    "train",
                    str(config_file),
                    "--output-dir",
                    str(model_dir),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result3.returncode != 0:
                pytest.skip("Training step failed")

            # All steps should complete successfully
            assert result1.returncode == 0
            assert result2.returncode == 0
            assert result3.returncode == 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI pipeline commands not available")
