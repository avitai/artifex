"""Tests for configuration loading utilities."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from artifex.configs.utils.config_loader import (
    get_config_path,
    get_data_config,
    get_inference_config,
    get_protein_extensions_config,
    get_training_config,
    load_experiment_config,
    load_yaml_config,
)
from artifex.configs.utils.error_handling import (
    ConfigLoadError,
    ConfigNotFoundError,
    ConfigValidationError,
)
from artifex.generative_models.core.configuration import (
    ChoiceDistribution,
    DataConfig,
    DistributedBackend,
    DistributedConfig,
    ExperimentTemplateConfig,
    ExperimentTemplateOverrides,
    HyperparamSearchConfig,
    InferenceConfig,
    MeshConfig,
    OptimizerConfig,
    PointCloudConfig,
    ProteinDiffusionInferenceConfig,
    ProteinExtensionsConfig,
    ProteinPointCloudConfig,
    SchedulerConfig,
    SearchType,
    TrainingConfig,
    VoxelConfig,
)
from artifex.generative_models.core.configuration.model_creation import (
    materialize_model_creation_config,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "src" / "artifex" / "configs"
DEFAULTS_ROOT = CONFIG_ROOT / "defaults"
EXPERIMENTS_ROOT = CONFIG_ROOT / "experiments"


class TestGetConfigPath:
    """Test config path resolution."""

    def test_absolute_path_exists(self, tmp_path: Path) -> None:
        """Test that existing absolute paths are returned directly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value")
        result = get_config_path(str(config_file))
        assert result == config_file

    def test_absolute_path_not_found(self) -> None:
        """Test that missing absolute paths raise ConfigNotFoundError."""
        with pytest.raises(ConfigNotFoundError):
            get_config_path("/nonexistent/path/config.yaml")

    def test_relative_path_not_found(self) -> None:
        """Test that unresolvable relative paths raise ConfigNotFoundError."""
        with pytest.raises(ConfigNotFoundError):
            get_config_path("nonexistent_config_xyz.yaml")

    def test_relative_name_not_found(self) -> None:
        """Test that unresolvable names raise ConfigNotFoundError."""
        with pytest.raises(ConfigNotFoundError):
            get_config_path("nonexistent_config_xyz")


class TestLoadYamlConfig:
    """Test YAML config loading."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """Test loading a valid YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"model": {"dim": 128}}))
        result = load_yaml_config(config_file)
        assert result["model"]["dim"] == 128

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test that loading a missing file raises appropriate error."""
        with pytest.raises((ConfigNotFoundError, ConfigLoadError)):
            load_yaml_config(tmp_path / "missing.yaml")

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """Empty YAML files are invalid config assets."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        with pytest.raises(ConfigValidationError):
            load_yaml_config(config_file)

    def test_load_nested_yaml(self, tmp_path: Path) -> None:
        """Test loading a YAML file with nested structure."""
        data = {
            "training": {"lr": 0.001, "epochs": 100},
            "model": {"layers": [64, 128, 256]},
        }
        config_file = tmp_path / "nested.yaml"
        config_file.write_text(yaml.dump(data))
        result = load_yaml_config(config_file)
        assert result["training"]["lr"] == 0.001
        assert result["model"]["layers"] == [64, 128, 256]

    def test_load_non_mapping_yaml(self, tmp_path: Path) -> None:
        """Only mapping-shaped YAML is accepted as a config document."""
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigValidationError):
            load_yaml_config(config_file)

    def test_shipped_yaml_assets_are_non_empty_mappings(self) -> None:
        """Tracked runtime config assets must be loadable mapping documents."""
        yaml_files = sorted(CONFIG_ROOT.rglob("*.yaml"))
        assert yaml_files, "expected shipped YAML assets under src/artifex/configs"

        empty_files = [
            path.relative_to(REPO_ROOT) for path in yaml_files if path.stat().st_size == 0
        ]
        assert not empty_files, f"empty shipped YAML assets are not allowed: {empty_files}"

        bad_shapes: list[str] = []
        for path in yaml_files:
            try:
                payload = load_yaml_config(path)
            except (ConfigLoadError, ConfigValidationError) as exc:
                bad_shapes.append(f"{path.relative_to(REPO_ROOT)} -> {exc}")
                continue
            if not isinstance(payload, dict):
                bad_shapes.append(f"{path.relative_to(REPO_ROOT)} -> {type(payload).__name__}")

        assert not bad_shapes, "invalid shipped YAML assets:\n" + "\n".join(bad_shapes)

    def test_retained_config_asset_inventory_matches_runtime_contract(self) -> None:
        """The shipped config tree should keep only the reviewed retained assets."""
        expected_defaults = {
            "defaults/data/protein_dataset.yaml",
            "defaults/distributed/default.yaml",
            "defaults/extensions/protein.yaml",
            "defaults/hyperparam/diffusion_hyperparameter_search.yaml",
            "defaults/inference/protein_diffusion_inference.yaml",
            "defaults/models/geometric/mesh.yaml",
            "defaults/models/geometric/point_cloud.yaml",
            "defaults/models/geometric/protein_point_cloud.yaml",
            "defaults/models/geometric/voxel.yaml",
            "defaults/training/protein_diffusion_training.yaml",
        }
        expected_experiments = {
            "experiments/protein_diffusion_distributed.yaml",
            "experiments/protein_diffusion_experiment.yaml",
        }

        actual_defaults = {
            path.relative_to(CONFIG_ROOT).as_posix()
            for path in sorted(DEFAULTS_ROOT.rglob("*.yaml"))
        }
        actual_experiments = {
            path.relative_to(CONFIG_ROOT).as_posix()
            for path in sorted(EXPERIMENTS_ROOT.rglob("*.yaml"))
        }

        assert actual_defaults == expected_defaults
        assert actual_experiments == expected_experiments

    def test_shipped_data_asset_matches_data_config_surface(self) -> None:
        """The retained data asset must instantiate the current DataConfig."""
        config = get_data_config("protein_dataset")

        assert isinstance(config, DataConfig)
        assert config.name == "protein_dataset_default"
        assert config.dataset_name == "protein_dataset"
        assert config.data_dir == Path("{ARTIFEX_DATA_ROOT}/protein_dataset")
        assert config.split == "train"
        assert config.num_workers == 4
        assert config.prefetch_factor == 2
        assert config.pin_memory is False
        assert config.augmentation is False
        assert config.metadata["cache_dir"] == "{ARTIFEX_CACHE_ROOT}/protein_dataset"
        assert config.metadata["splits_file"] == "{ARTIFEX_DATA_ROOT}/protein_dataset/splits.json"
        assert config.metadata["max_seq_length"] == 128
        assert config.metadata["backbone_atom_indices"] == [0, 1, 2, 3]
        assert config.metadata["normalize_coordinates"] is True
        assert config.metadata["center_coordinates"] is True
        assert config.metadata["random_rotation"] is False

    def test_shipped_training_asset_matches_training_config_surface(self) -> None:
        """The retained default training asset must instantiate TrainingConfig."""
        config = get_training_config("protein_diffusion_training")

        assert isinstance(config, TrainingConfig)
        assert config.name == "protein_diffusion_training"
        assert config.batch_size == 128
        assert config.num_epochs == 100
        assert config.log_frequency == 20
        assert config.save_frequency == 100
        assert config.gradient_clip_norm == 1.0
        assert config.optimizer.learning_rate == pytest.approx(1.0e-4)
        assert config.scheduler is not None
        assert config.scheduler.scheduler_type == "cosine"
        assert config.scheduler.warmup_steps == 500

    def test_shipped_inference_asset_matches_specific_inference_surface(self) -> None:
        """The retained protein inference asset should load as its concrete typed config."""
        config = get_inference_config("protein_diffusion_inference")

        assert isinstance(config, InferenceConfig)
        assert isinstance(config, ProteinDiffusionInferenceConfig)
        assert config.name == "protein_diffusion_inference"
        assert (
            config.checkpoint_path
            == "{ARTIFEX_CHECKPOINT_ROOT}/protein_diffusion/checkpoint_best.pt"
        )
        assert config.output_dir == "{ARTIFEX_OUTPUT_ROOT}/protein_diffusion/samples/"
        assert config.device == "cpu"
        assert config.sampler == "ddpm"
        assert config.target_seq_length == 128
        assert config.calculate_metrics is True

    def test_shipped_protein_extension_asset_matches_bundle_surface(self) -> None:
        """The retained protein extension asset must instantiate ProteinExtensionsConfig."""
        config = get_protein_extensions_config("protein")

        assert isinstance(config, ProteinExtensionsConfig)
        assert config.name == "default_protein_extensions"
        assert config.bond_length is not None
        assert config.bond_length.ideal_bond_lengths["N-CA"] == pytest.approx(1.45)
        assert config.bond_angle is not None
        assert config.bond_angle.ideal_bond_angles["N-CA-C"] == pytest.approx(1.94)
        assert config.backbone is not None
        assert config.backbone.bond_length_weight == pytest.approx(1.0)
        assert config.backbone.bond_angle_weight == pytest.approx(0.5)
        assert config.mixin is not None
        assert config.mixin.embedding_dim == 16
        assert config.mixin.num_aa_types == 21

    def test_shipped_distributed_asset_matches_distributed_config_surface(self) -> None:
        """The retained distributed asset should be a portable typed baseline."""
        config = DistributedConfig.from_yaml(DEFAULTS_ROOT / "distributed" / "default.yaml")

        assert isinstance(config, DistributedConfig)
        assert config.name == "distributed_default"
        assert config.enabled is False
        assert config.backend is DistributedBackend.GLOO
        assert config.world_size == 1
        assert config.num_nodes == 1
        assert config.num_processes_per_node == 1
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.mixed_precision == "no"

    def test_shipped_hyperparam_asset_matches_typed_search_surface(self) -> None:
        """The retained hyperparameter asset should use the typed search schema."""
        config = HyperparamSearchConfig.from_yaml(
            DEFAULTS_ROOT / "hyperparam" / "diffusion_hyperparameter_search.yaml"
        )

        assert isinstance(config, HyperparamSearchConfig)
        assert config.name == "diffusion_hyperparameter_search"
        assert config.search_type is SearchType.RANDOM
        assert config.num_trials == 20
        assert config.max_parallel_trials == 2
        assert config.metric == "validation_loss"
        assert isinstance(config.search_space["network.embed_dim"], ChoiceDistribution)
        assert config.search_space["network.embed_dim"].choices == (64, 128, 256, 512)
        assert isinstance(config.search_space["network.num_layers"], ChoiceDistribution)
        assert config.search_space["network.num_layers"].choices == (4, 6, 8, 12)
        assert config.search_space["network.dropout_rate"].type == "uniform"
        assert config.search_space["training.optimizer.learning_rate"].type == "uniform"
        assert config.search_space["training.optimizer.weight_decay"].type == "uniform"
        assert config.search_space["training.scheduler.warmup_steps"].type == "choice"

    def test_shipped_geometric_model_assets_match_typed_config_surfaces(self) -> None:
        """Retained geometric defaults should use the current nested dataclass schema."""
        mesh_yaml = (DEFAULTS_ROOT / "models" / "geometric" / "mesh.yaml").read_text(
            encoding="utf-8"
        )
        voxel_yaml = (DEFAULTS_ROOT / "models" / "geometric" / "voxel.yaml").read_text(
            encoding="utf-8"
        )
        point_cloud = PointCloudConfig.from_yaml(
            DEFAULTS_ROOT / "models" / "geometric" / "point_cloud.yaml"
        )
        mesh = MeshConfig.from_yaml(DEFAULTS_ROOT / "models" / "geometric" / "mesh.yaml")
        voxel = VoxelConfig.from_yaml(DEFAULTS_ROOT / "models" / "geometric" / "voxel.yaml")

        assert point_cloud.name == "default_point_cloud_model"
        assert point_cloud.network is not None
        assert point_cloud.network.embed_dim == 128
        assert point_cloud.network.num_layers == 4
        assert point_cloud.num_points == 512
        assert point_cloud.dropout_rate == pytest.approx(0.1)

        assert mesh.name == "default_mesh_model"
        assert mesh.network is not None
        assert mesh.network.embed_dim == 256
        assert mesh.network.edge_features_dim == 64
        assert mesh.num_vertices == 512
        assert mesh.vertex_dim == 3
        assert not hasattr(mesh, "num_faces")
        assert "num_faces" not in mesh_yaml

        assert voxel.name == "default_voxel_model"
        assert voxel.network is not None
        assert voxel.network.base_channels == 64
        assert voxel.network.num_layers == 4
        assert voxel.voxel_size == 16
        assert voxel.loss_type == "focal"
        assert voxel.focal_gamma == pytest.approx(2.0)
        for stale_key in (
            "\nresolution:",
            "\nuse_conditioning:",
            "\nconditioning_dim:",
            "\nchannels:",
            "\nmodel_type:",
        ):
            assert stale_key not in voxel_yaml

    def test_shipped_protein_model_asset_matches_model_creation_surface(self) -> None:
        """The retained protein model asset should materialize as ProteinPointCloudConfig."""
        config = materialize_model_creation_config(
            load_yaml_config(DEFAULTS_ROOT / "models" / "geometric" / "protein_point_cloud.yaml")
        )

        assert isinstance(config, ProteinPointCloudConfig)
        assert config.name == "protein_point_cloud_default"
        assert config.network is not None
        assert config.network.embed_dim == 128
        assert config.num_residues == 128
        assert config.num_atoms_per_residue == 4
        assert config.num_points == 512
        assert config.backbone_indices == (0, 1, 2, 3)
        assert config.extensions is None

    def test_typed_training_loader_wraps_schema_errors(self, tmp_path: Path) -> None:
        """Typed loaders should translate schema drift into repo-owned validation errors."""
        config_file = tmp_path / "invalid_training.yaml"
        config_file.write_text("description: missing required name\n", encoding="utf-8")

        with pytest.raises(ConfigValidationError) as exc_info:
            get_training_config(str(config_file))

        message = str(exc_info.value)
        assert "Configuration validation failed" in message
        assert "invalid_training.yaml" in message
        assert "missing value for field" in message

    def test_root_extension_catalog_file_is_removed(self) -> None:
        """Protein extensions should ship as one bundle document, not a duplicate root catalog."""
        assert not (CONFIG_ROOT / "defaults" / "extensions.yaml").exists()

    def test_experiment_training_overrides_match_training_config_surface(self) -> None:
        """Experiment training overrides must only use supported training keys."""
        training_fields = {field.name for field in dataclasses.fields(TrainingConfig)}
        optimizer_fields = {field.name for field in dataclasses.fields(OptimizerConfig)}
        scheduler_fields = {field.name for field in dataclasses.fields(SchedulerConfig)}
        invalid_overrides: list[str] = []

        for path in sorted((CONFIG_ROOT / "experiments").glob("*.yaml")):
            payload = load_yaml_config(path)
            overrides = payload.get("overrides", {})
            training_overrides = overrides.get("training", {})
            for key, value in training_overrides.items():
                if key == "optimizer":
                    if not isinstance(value, dict):
                        invalid_overrides.append(
                            f"{path.relative_to(REPO_ROOT)} -> training.optimizer must be a mapping"
                        )
                        continue
                    bad_keys = sorted(set(value) - optimizer_fields)
                    if bad_keys:
                        invalid_overrides.append(
                            f"{path.relative_to(REPO_ROOT)} -> training.optimizer invalid keys: {bad_keys}"
                        )
                    continue
                if key == "scheduler":
                    if not isinstance(value, dict):
                        invalid_overrides.append(
                            f"{path.relative_to(REPO_ROOT)} -> training.scheduler must be a mapping"
                        )
                        continue
                    bad_keys = sorted(set(value) - scheduler_fields)
                    if bad_keys:
                        invalid_overrides.append(
                            f"{path.relative_to(REPO_ROOT)} -> training.scheduler invalid keys: {bad_keys}"
                        )
                    continue
                if key not in training_fields:
                    invalid_overrides.append(
                        f"{path.relative_to(REPO_ROOT)} -> training invalid key: {key}"
                    )

        assert not invalid_overrides, "invalid training overrides:\n" + "\n".join(invalid_overrides)

    def test_shipped_experiment_templates_use_placeholders_for_local_paths(self) -> None:
        """Retained experiment templates should not hardcode repo-local data or output paths."""
        placeholder_roots = {
            "output_dir": "{ARTIFEX_OUTPUT_ROOT}",
            "data_path": "{ARTIFEX_DATA_ROOT}",
            "cache_dir": "{ARTIFEX_CACHE_ROOT}",
            "splits_file": "{ARTIFEX_DATA_ROOT}",
        }
        violations: list[str] = []

        def check_mapping(path: Path, mapping: dict[str, object]) -> None:
            for key, value in mapping.items():
                if isinstance(value, dict):
                    check_mapping(path, value)
                    continue
                if key not in placeholder_roots or not isinstance(value, str):
                    continue
                if value.startswith("./"):
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)} -> {key} uses repo-local path {value!r}"
                    )
                    continue
                if placeholder_roots[key] not in value:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)} -> {key} must include {placeholder_roots[key]!r}"
                    )

        for path in sorted((CONFIG_ROOT / "experiments").glob("*.yaml")):
            check_mapping(path, load_yaml_config(path))

        assert not violations, "invalid retained experiment templates:\n" + "\n".join(violations)

    def test_shipped_inference_defaults_are_portable(self) -> None:
        """Retained inference defaults should avoid machine-local paths and GPU forcing."""
        payload = load_yaml_config(
            CONFIG_ROOT / "defaults" / "inference" / "protein_diffusion_inference.yaml"
        )

        assert (
            payload["checkpoint_path"]
            == "{ARTIFEX_CHECKPOINT_ROOT}/protein_diffusion/checkpoint_best.pt"
        )
        assert payload["output_dir"] == "{ARTIFEX_OUTPUT_ROOT}/protein_diffusion/samples/"
        assert payload["device"] == "cpu"

    def test_load_experiment_config_returns_typed_template(self) -> None:
        """Experiment templates should load as a typed template object, not a raw dict."""
        config = load_experiment_config("protein_diffusion_experiment")

        assert isinstance(config, ExperimentTemplateConfig)
        assert not hasattr(config, "name")
        assert config.experiment_name == "protein_diffusion_cath"
        assert config.model_config == "models/geometric/protein_point_cloud.yaml"
        assert config.data_config == "data/protein_dataset.yaml"
        assert config.training_config == "training/protein_diffusion_training.yaml"
        assert config.inference_config == "inference/protein_diffusion_inference.yaml"
        assert str(config.output_dir) == "{ARTIFEX_OUTPUT_ROOT}/protein_diffusion_cath"
        assert isinstance(config.overrides, ExperimentTemplateOverrides)
        assert config.overrides.model["num_residues"] == 150
        assert config.overrides.model["num_points"] == 600
        assert config.overrides.training["batch_size"] == 64
        assert config.overrides.inference["target_seq_length"] == 150

    def test_load_experiment_config_wraps_template_schema_errors(self, tmp_path: Path) -> None:
        """Typed experiment loading should not leak raw dacite/schema exceptions."""
        config_file = tmp_path / "not_a_template.yaml"
        config_file.write_text("name: plain_training_config\nbatch_size: 32\n", encoding="utf-8")

        with pytest.raises(ConfigValidationError) as exc_info:
            load_experiment_config(str(config_file))

        message = str(exc_info.value)
        assert "Configuration validation failed" in message
        assert "not_a_template.yaml" in message
        assert 'can not match "' in message
        assert '"name"' in message
        assert '"batch_size"' in message

    def test_shipped_model_defaults_are_single_config_documents(self) -> None:
        """Runtime model defaults must not hide multiple configs in one catalog file."""
        invalid_catalogs: list[str] = []

        for path in sorted((CONFIG_ROOT / "defaults" / "models").rglob("*.yaml")):
            payload = load_yaml_config(path)
            if set(payload) == {"configs"} and isinstance(payload["configs"], dict):
                invalid_catalogs.append(str(path.relative_to(REPO_ROOT)))

        assert not invalid_catalogs, (
            "runtime model defaults must be single-config documents:\n"
            + "\n".join(invalid_catalogs)
        )

    def test_shipped_experiment_templates_reference_loadable_retained_assets(self) -> None:
        """Every retained experiment template should point at loadable retained assets."""
        experiment = load_experiment_config("protein_diffusion_experiment")
        distributed_experiment = load_experiment_config("protein_diffusion_distributed")

        experiment_model = materialize_model_creation_config(
            load_yaml_config(DEFAULTS_ROOT / experiment.model_config)
        )
        distributed_model = materialize_model_creation_config(
            load_yaml_config(DEFAULTS_ROOT / distributed_experiment.model_config)
        )

        assert isinstance(experiment_model, ProteinPointCloudConfig)
        assert isinstance(distributed_model, ProteinPointCloudConfig)
        assert isinstance(DataConfig.from_yaml(DEFAULTS_ROOT / experiment.data_config), DataConfig)
        assert isinstance(
            TrainingConfig.from_yaml(DEFAULTS_ROOT / experiment.training_config),
            TrainingConfig,
        )
        assert isinstance(
            ProteinDiffusionInferenceConfig.from_yaml(DEFAULTS_ROOT / experiment.inference_config),
            ProteinDiffusionInferenceConfig,
        )
        assert distributed_experiment.distributed_config == "distributed/default.yaml"
        assert isinstance(
            DistributedConfig.from_yaml(DEFAULTS_ROOT / distributed_experiment.distributed_config),
            DistributedConfig,
        )
        assert (
            distributed_experiment.hyperparam_config
            == "hyperparam/diffusion_hyperparameter_search.yaml"
        )
        assert isinstance(
            HyperparamSearchConfig.from_yaml(
                DEFAULTS_ROOT / distributed_experiment.hyperparam_config
            ),
            HyperparamSearchConfig,
        )
