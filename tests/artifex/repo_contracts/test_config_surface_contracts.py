"""Repository contracts for the configuration runtime surface."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import json
import pkgutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_CONFIG_DOCS = {
    "base.md",
    "config_loader.md",
    "data.md",
    "distributed.md",
    "error_handling.md",
    "extensions.md",
    "hyperparam.md",
    "index.md",
    "inference.md",
    "templates.md",
    "training.md",
}
REMOVED_CONFIG_DOCS = {
    "cli.md",
    "conversion.md",
    "dit..md",
    "extension_config.md",
    "io.md",
    "merge.md",
    "validation.md",
}


def _run_python(code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_config_runtime_surface_does_not_ship_embedded_test_modules() -> None:
    """Runtime config utilities should not ship stray pytest/demo modules."""
    assert not (REPO_ROOT / "src/artifex/configs/utils/templates.py").exists()

    for runtime_module in (REPO_ROOT / "src/artifex/configs/utils").glob("*.py"):
        contents = runtime_module.read_text()
        assert "import pytest" not in contents
        assert "from pytest" not in contents


def test_core_config_classes_define_annotated_post_init_validation_hooks() -> None:
    """Concrete core config classes should carry explicit validation hooks."""
    config_dir = REPO_ROOT / "src/artifex/generative_models/core/configuration"
    missing_hooks: list[str] = []
    missing_annotations: list[str] = []

    for path in sorted(config_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or not node.name.endswith("Config"):
                continue

            post_init = next(
                (
                    stmt
                    for stmt in node.body
                    if isinstance(stmt, ast.FunctionDef) and stmt.name == "__post_init__"
                ),
                None,
            )
            if post_init is None:
                missing_hooks.append(f"{path.name}:{node.name}")
                continue
            if post_init.returns is None:
                missing_annotations.append(f"{path.name}:{node.name}")

    assert missing_hooks == []
    assert missing_annotations == []


def test_core_config_documents_are_frozen_slotted_and_keyword_only() -> None:
    """Core config dataclasses should use the shared frozen/slotted/kw-only contract."""
    from artifex.generative_models.core.configuration import __path__ as configuration_paths
    from artifex.generative_models.core.configuration.base_dataclass import ConfigDocument

    violations: list[str] = []
    module_prefix = "artifex.generative_models.core.configuration."

    for module_info in pkgutil.iter_modules(configuration_paths, module_prefix):
        module = importlib.import_module(module_info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is ConfigDocument:
                continue
            if not issubclass(obj, ConfigDocument):
                continue
            if not obj.__module__.startswith(module_prefix):
                continue

            params = getattr(obj, "__dataclass_params__", None)
            field_kwargs = [field.kw_only for field in dataclasses.fields(obj) if field.init]
            if (
                not dataclasses.is_dataclass(obj)
                or params is None
                or not params.frozen
                or not all(field_kwargs)
                or not hasattr(obj, "__slots__")
            ):
                violations.append(f"{obj.__module__}.{obj.__name__}")

    assert violations == []


def test_template_docs_reference_the_supported_surface() -> None:
    """Docs should point to the canonical template module, not a deleted shim."""
    stale_references = [
        "configs.utils.templates",
        "configs/utils/templates.py",
    ]

    docs_to_check = [
        "docs/configs/templates.md",
        "docs/roadmap/planned-modules.md",
    ]

    for relative_path in docs_to_check:
        contents = (REPO_ROOT / relative_path).read_text()
        for stale_reference in stale_references:
            assert stale_reference not in contents

    template_docs = (REPO_ROOT / "docs/configs/templates.md").read_text()
    assert "generative_models.core.configuration.management.templates" in template_docs


def test_package_local_config_readme_matches_current_surface() -> None:
    """The config package README should describe the current dataclass-based API only."""
    readme = (REPO_ROOT / "src/artifex/configs/README.md").read_text()

    banned_references = [
        "apply_env_overrides",
        "Pydantic-based",
        "ModelConfiguration",
        "PointCloudDiffusionConfig",
        "artifex.generative_models.configs",
        "./test.py",
        "templates/protein_diffusion.yaml",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in readme

    required_references = [
        "frozen dataclass",
        "TrainingConfig",
        "load_experiment_config",
        "template_manager",
        "ConfigVersionRegistry",
    ]

    for required_reference in required_references:
        assert required_reference in readme


def test_config_docs_index_matches_current_surface() -> None:
    """The config docs index should describe the current runtime API only."""
    docs_index = (REPO_ROOT / "docs/configs/index.md").read_text()

    banned_references = [
        "apply_env_overrides",
        "get_env_name",
        "get_env_config_path",
        "load_env_config",
        "ConfigLoader",
        "load_config(",
        "validate_config",
        "convert_config",
        "merge_configs",
        "get_template(",
        "save_config(",
        "ConfigError",
        "handle_config_error",
        "CLIConfig",
        "ConfigDocument",
        "HyperparamConfig",
        "ExtensionsConfig",
        "load_yaml_config",
        "create_config_from_yaml",
        "get_model_config",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in docs_index

    required_references = [
        "artifex.configs",
        "TrainingConfig",
        "load_experiment_config",
        "ExperimentTemplateConfig",
        "template_manager",
        "ConfigVersionRegistry",
    ]

    for required_reference in required_references:
        assert required_reference in docs_index


def test_config_shim_packages_are_removed() -> None:
    """Nested config shim packages should not survive as alternative APIs."""
    assert not (REPO_ROOT / "src/artifex/configs/schema/__init__.py").exists()
    assert not (REPO_ROOT / "src/artifex/configs/schema/base.py").exists()
    assert not (REPO_ROOT / "src/artifex/configs/utils/__init__.py").exists()


def test_top_level_config_package_keeps_deeper_modules_lazy() -> None:
    """Importing artifex.configs should not eagerly import the full config runtime."""
    payload = _run_python(
        "import json, sys; "
        "import artifex.configs as configs; "
        "print(json.dumps({"
        "'loader_loaded': 'artifex.configs.utils.config_loader' in sys.modules, "
        "'environment_loaded': "
        "'artifex.generative_models.core.configuration.environment' in sys.modules, "
        "'templates_loaded': "
        "'artifex.generative_models.core.configuration.management.templates' in sys.modules, "
        "'versioning_loaded': "
        "'artifex.generative_models.core.configuration.management.versioning' in sys.modules, "
        "'configuration_loaded': "
        "'artifex.generative_models.core.configuration' in sys.modules, "
        "'jax_loaded': 'jax' in sys.modules, "
        "'config_template_visible': hasattr(configs, 'ConfigTemplate'), "
        "'apply_env_overrides_visible': hasattr(configs, 'apply_env_overrides'), "
        "'get_env_name_visible': hasattr(configs, 'get_env_name'), "
        "'get_env_config_path_visible': hasattr(configs, 'get_env_config_path'), "
        "'load_env_config_visible': hasattr(configs, 'load_env_config'), "
        "'all': list(getattr(configs, '__all__'))"
        "}))"
    )

    assert payload["loader_loaded"] is False
    assert payload["environment_loaded"] is False
    assert payload["templates_loaded"] is False
    assert payload["versioning_loaded"] is False
    assert payload["configuration_loaded"] is False
    assert payload["jax_loaded"] is False
    assert payload["config_template_visible"] is False
    assert payload["apply_env_overrides_visible"] is False
    assert payload["get_env_name_visible"] is False
    assert payload["get_env_config_path_visible"] is False
    assert payload["load_env_config_visible"] is False
    assert "TrainingConfig" in payload["all"]
    assert "get_training_config" in payload["all"]
    assert "load_experiment_config" in payload["all"]
    assert "template_manager" in payload["all"]
    assert "ConfigDocument" not in payload["all"]
    assert "ConfigTemplate" not in payload["all"]
    assert "apply_env_overrides" not in payload["all"]
    assert "get_env_name" not in payload["all"]
    assert "get_env_config_path" not in payload["all"]
    assert "load_env_config" not in payload["all"]
    assert "load_yaml_config" not in payload["all"]
    assert "create_config_from_yaml" not in payload["all"]
    assert "get_model_config" not in payload["all"]


def test_top_level_config_exports_resolve_lazily() -> None:
    """Documented config exports should still resolve on explicit access."""
    payload = _run_python(
        "import json; "
        "import artifex.configs as configs; "
        "print(json.dumps({"
        "'training_config_module': configs.TrainingConfig.__module__, "
        "'get_training_module': configs.get_training_config.__module__, "
        "'load_experiment_module': configs.load_experiment_config.__module__, "
        "'template_manager_module': configs.template_manager.__class__.__module__, "
        "'config_version_registry_module': configs.ConfigVersionRegistry.__module__"
        "}))"
    )

    assert payload["training_config_module"] == (
        "artifex.generative_models.core.configuration.training_config"
    )
    assert payload["get_training_module"] == "artifex.configs.utils.config_loader"
    assert payload["load_experiment_module"] == "artifex.configs.utils.config_loader"
    assert payload["template_manager_module"] == (
        "artifex.generative_models.core.configuration.management.templates"
    )
    assert payload["config_version_registry_module"] == (
        "artifex.generative_models.core.configuration.management.versioning"
    )


def test_core_protocols_package_does_not_export_concrete_template_helpers() -> None:
    """The protocols package should expose protocols only, not concrete config helpers."""
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.core.protocols as protocols; "
        "from artifex.generative_models.core.configuration.management import templates; "
        "print(json.dumps({"
        "'has_config_template': hasattr(protocols, 'ConfigTemplate'), "
        "'all': list(getattr(protocols, '__all__')), "
        "'config_template_module': templates.ConfigTemplate.__module__"
        "}))"
    )

    assert payload["has_config_template"] is False
    assert "ConfigTemplate" not in payload["all"]
    assert payload["config_template_module"] == (
        "artifex.generative_models.core.configuration.management.templates"
    )


def test_core_configuration_package_keeps_children_lazy() -> None:
    """Importing core.configuration should not eagerly load its child modules."""
    payload = _run_python(
        "import json, sys; "
        "import artifex.generative_models.core.configuration as configuration; "
        "print(json.dumps({"
        "'base_dataclass_loaded': "
        "'artifex.generative_models.core.configuration.base_dataclass' in sys.modules, "
        "'training_loaded': "
        "'artifex.generative_models.core.configuration.training_config' in sys.modules, "
        "'hyperparam_loaded': "
        "'artifex.generative_models.core.configuration.hyperparam_config' in sys.modules, "
        "'templates_loaded': "
        "'artifex.generative_models.core.configuration.management.templates' in sys.modules, "
        "'versioning_loaded': "
        "'artifex.generative_models.core.configuration.management.versioning' in sys.modules, "
        "'jax_loaded': 'jax' in sys.modules, "
        "'config_document_visible': hasattr(configuration, 'ConfigDocument'), "
        "'validate_positive_float_visible': hasattr(configuration, 'validate_positive_float'), "
        "'all': list(getattr(configuration, '__all__'))"
        "}))"
    )

    assert payload["base_dataclass_loaded"] is False
    assert payload["training_loaded"] is False
    assert payload["hyperparam_loaded"] is False
    assert payload["templates_loaded"] is False
    assert payload["versioning_loaded"] is False
    assert payload["jax_loaded"] is False
    assert payload["config_document_visible"] is False
    assert payload["validate_positive_float_visible"] is False
    assert "TrainingConfig" in payload["all"]
    assert "BaseConfig" in payload["all"]
    assert "ParameterDistribution" in payload["all"]
    assert "ConfigDocument" not in payload["all"]
    assert "validate_positive_float" not in payload["all"]


def test_core_configuration_exports_resolve_lazily() -> None:
    """Documented core.configuration exports should resolve on explicit access."""
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.core.configuration as configuration; "
        "print(json.dumps({"
        "'training_module': configuration.TrainingConfig.__module__, "
        "'base_config_module': configuration.BaseConfig.__module__, "
        "'parameter_distribution_module': configuration.ParameterDistribution.__module__, "
        "'experiment_template_module': configuration.ExperimentTemplateConfig.__module__"
        "}))"
    )

    assert payload["training_module"] == (
        "artifex.generative_models.core.configuration.training_config"
    )
    assert payload["base_config_module"] == (
        "artifex.generative_models.core.configuration.base_dataclass"
    )
    assert payload["parameter_distribution_module"] == (
        "artifex.generative_models.core.configuration.hyperparam_config"
    )
    assert payload["experiment_template_module"] == (
        "artifex.generative_models.core.configuration.experiment_config"
    )


def test_experiment_templates_stay_distinct_from_named_runtime_configs() -> None:
    """Retained experiment templates should not be forced into BaseConfig."""
    payload = _run_python(
        "import json; "
        "from artifex.configs import BaseConfig, ExperimentConfig, ExperimentTemplateConfig; "
        "from artifex.generative_models.core.configuration.base_dataclass import ConfigDocument; "
        "print(json.dumps({"
        "'template_is_config_document': issubclass(ExperimentTemplateConfig, ConfigDocument), "
        "'template_is_base_config': issubclass(ExperimentTemplateConfig, BaseConfig), "
        "'experiment_is_base_config': issubclass(ExperimentConfig, BaseConfig)"
        "}))"
    )

    assert payload["template_is_config_document"] is True
    assert payload["template_is_base_config"] is False
    assert payload["experiment_is_base_config"] is True


def test_config_docs_directory_keeps_required_pages_and_removed_pages_absent() -> None:
    """Config docs should keep the supported core pages without reviving removed ones."""
    docs_dir = REPO_ROOT / "docs/configs"
    actual_docs = {path.name for path in docs_dir.glob("*.md")}
    assert REQUIRED_CONFIG_DOCS.issubset(actual_docs)
    assert REMOVED_CONFIG_DOCS.isdisjoint(actual_docs)


def test_docs_do_not_refer_to_removed_config_shim_imports() -> None:
    """Docs should import config symbols from the supported surface only."""
    disallowed_references = [
        "artifex.configs.schema",
        "from artifex.configs.utils import",
    ]

    docs_to_check = [
        "docs/configs/index.md",
        "docs/configs/base.md",
        "docs/configs/data.md",
        "docs/configs/distributed.md",
        "docs/configs/extensions.md",
        "docs/configs/hyperparam.md",
        "docs/configs/inference.md",
        "docs/configs/templates.md",
        "docs/configs/training.md",
        "docs/examples/protein/protein-extensions-with-config.md",
        "docs/user-guide/advanced/distributed.md",
    ]

    for relative_path in docs_to_check:
        contents = (REPO_ROOT / relative_path).read_text()
        for disallowed_reference in disallowed_references:
            assert disallowed_reference not in contents


def test_config_loader_docs_match_current_surface() -> None:
    """Loader docs should describe only the supported typed loader APIs."""
    loader_docs = (REPO_ROOT / "docs/configs/config_loader.md").read_text()

    banned_references = [
        "create_config_from_yaml",
        "get_model_config",
        "load_yaml_config",
        "ConfigDocument",
    ]
    required_references = [
        "from artifex.configs import (",
        "get_config_path",
        "get_data_config",
        "get_inference_config",
        "get_protein_extensions_config",
        "get_training_config",
        "load_experiment_config",
        "TrainingConfig.from_yaml",
        "ExperimentTemplateConfig",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in loader_docs

    for required_reference in required_references:
        assert required_reference in loader_docs


def test_config_error_handling_docs_match_current_surface() -> None:
    """Error-handling docs should describe the retained loader failure boundary only."""
    error_docs = (REPO_ROOT / "docs/configs/error_handling.md").read_text()

    banned_references = [
        "format_config_error",
        "format_validation_error",
        "error_handler",
        "config_type",
    ]
    required_references = [
        "ConfigError",
        "ConfigLoadError",
        "ConfigNotFoundError",
        "ConfigValidationError",
        "public loader helpers exposed from `artifex.configs`",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in error_docs

    for required_reference in required_references:
        assert required_reference in error_docs


def test_hyperparam_docs_match_supported_top_level_import_surface() -> None:
    """Hyperparameter docs should not advertise imports missing from artifex.configs."""
    hyperparam_docs = (REPO_ROOT / "docs/configs/hyperparam.md").read_text()

    banned_references = [
        "from artifex.configs import ChoiceDistribution, HyperparamSearchConfig, SearchType, LogUniformDistribution",
        "- `LogUniformDistribution`",
    ]
    required_references = [
        "from artifex.configs import ChoiceDistribution, HyperparamSearchConfig, SearchType",
        "ChoiceDistribution",
        "UniformDistribution",
        "CategoricalDistribution",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in hyperparam_docs

    for required_reference in required_references:
        assert required_reference in hyperparam_docs


def test_cli_config_docs_match_surviving_typed_cli_surface() -> None:
    """CLI config docs should describe the surviving wrapper and typed templates only."""
    cli_docs = (REPO_ROOT / "docs/cli/config.md").read_text()
    core_docs = (REPO_ROOT / "docs/core/config_commands.md").read_text()

    banned_references = [
        "cli.utils.config",
        "cli/utils/config.py",
        "apply_overrides",
        "parse_override",
        "raw YAML",
    ]
    required_cli_references = [
        "artifex config create",
        "simple_training",
        "distributed_training",
        "artifex config validate",
        "artifex config show",
    ]
    required_core_references = [
        "artifex.generative_models.core.cli.config_commands",
        "typed config templates",
        "supported typed config documents",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in cli_docs
        assert banned_reference not in core_docs

    for required_reference in required_cli_references:
        assert required_reference in cli_docs

    for required_reference in required_core_references:
        assert required_reference in core_docs


def test_dead_cli_asset_tree_and_environment_helper_are_removed() -> None:
    """Dead CLI placeholder assets and the unowned environment helper should be gone."""
    cli_config_dir = REPO_ROOT / "src/artifex/cli/config"
    cli_templates_dir = REPO_ROOT / "src/artifex/cli/templates"

    assert sorted(cli_config_dir.rglob("*.yaml")) == []
    assert not (cli_config_dir / "{__init__.py}").exists()
    assert not (cli_templates_dir / "{__init__.py}").exists()
    assert not (
        REPO_ROOT / "src/artifex/generative_models/core/configuration/environment.py"
    ).exists()
    assert not (REPO_ROOT / "docs/core/environment.md").exists()

    mkdocs_config = (REPO_ROOT / "mkdocs.yml").read_text()
    core_index = (REPO_ROOT / "docs/core/index.md").read_text()
    assert "core/environment.md" not in mkdocs_config
    assert "[environment](environment.md)" not in core_index


def test_user_facing_docs_and_package_readmes_use_current_config_contract() -> None:
    """Tracked docs should teach the current dataclass config surface only."""
    docs_to_check = [
        "src/artifex/generative_models/README.md",
        "src/artifex/generative_models/modalities/README.md",
        "src/artifex/generative_models/models/energy/README.md",
        "docs/user-guide/training/configuration.md",
        "docs/user-guide/training/overview.md",
        "docs/api/core/configuration.md",
        "docs/core/unified.md",
        "docs/api/training/trainer.md",
        "docs/guides/configuration.md",
        "docs/user-guide/modalities/text.md",
        "docs/user-guide/data/overview.md",
        "docs/examples/protein/protein-ligand-benchmark-demo.md",
        "docs/examples/protein/protein-model-with-modality.md",
        "docs/examples/framework/framework-features-demo.md",
        "docs/getting-started/core-concepts.md",
    ]
    banned_references = [
        "Pydantic",
        "ModelConfiguration",
        "ModalityConfiguration",
        "BaseConfiguration",
    ]

    for relative_path in docs_to_check:
        contents = (REPO_ROOT / relative_path).read_text()
        for banned_reference in banned_references:
            assert banned_reference not in contents

    required_page_markers = {
        "src/artifex/generative_models/README.md": [
            "artifex.configs",
            "VAEConfig",
        ],
        "src/artifex/generative_models/modalities/README.md": [
            "ModalityConfig",
        ],
        "src/artifex/generative_models/models/energy/README.md": [
            "EBMConfig",
            "EnergyNetworkConfig",
        ],
        "docs/user-guide/training/configuration.md": [
            "frozen dataclasses",
            "TrainingConfig",
            "ExperimentTemplateConfig",
        ],
        "docs/api/core/configuration.md": [
            "artifex.configs",
            "from_dict()",
            "from_yaml()",
        ],
    }

    for relative_path, required_references in required_page_markers.items():
        contents = (REPO_ROOT / relative_path).read_text()
        for required_reference in required_references:
            assert required_reference in contents


def test_core_config_tooling_docs_match_live_owner_modules() -> None:
    """Core config tooling docs should stay on the surviving typed-config surface."""
    unified_docs = (REPO_ROOT / "docs/core/unified.md").read_text()
    configuration_docs = (REPO_ROOT / "docs/core/configuration.md").read_text()
    gan_docs = (REPO_ROOT / "docs/core/gan.md").read_text()
    validation_docs = (REPO_ROOT / "docs/core/validation.md").read_text()
    migrate_docs = (REPO_ROOT / "docs/core/migrate_configs.md").read_text()
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.core.configuration as config; "
        "from artifex.generative_models.core.configuration import gan_config; "
        "import importlib.util; "
        "print(json.dumps({"
        "'config_exports': sorted(config.__all__), "
        "'gan_module': gan_config.__name__, "
        "'migrate_exists': importlib.util.find_spec('artifex.generative_models.core.configuration.migrate_configs') is not None"
        "}))"
    )

    for required in [
        "artifex.configs",
        "artifex.generative_models.core.configuration",
        "from_dict()",
        "from_yaml()",
    ]:
        assert required in unified_docs

    for required in [
        "BaseConfig",
        "ModalityConfig",
        "BaseModalityConfig",
        "artifex.generative_models.core.configuration",
    ]:
        assert required in configuration_docs

    for required in [
        "artifex.generative_models.core.configuration.gan_config",
        "GANConfig",
        "DCGANConfig",
        "WGANConfig",
        "LSGANConfig",
        "ConditionalGANConfig",
        "CycleGANConfig",
    ]:
        assert required in gan_docs

    for banned in [
        "`generative_models.core.configuration.gan`",
        "`generative_models/core/configuration/gan.py`",
        "GANConfiguration",
        "DCGANConfiguration",
        "WGANConfiguration",
        "create_gan_config_from_model_config",
        "image_shape",
    ]:
        assert banned not in gan_docs

    for required in [
        "artifex.generative_models.core.configuration.validation",
        "validate_positive_int",
        "validate_non_negative_float",
        "validate_activation",
    ]:
        assert required in validation_docs

    for banned in [
        "ensure_model_configuration_compatibility",
        "validate_base_configuration",
        "validate_configuration_type",
        "validate_model_configuration",
    ]:
        assert banned not in validation_docs

    for required in [
        "Status: Coming soon",
        "artifex.generative_models.core.configuration.migrate_configs",
        "does not ship",
        "manual migration",
    ]:
        assert required in migrate_docs

    for banned in [
        "ConfigMigrator",
        "create_migration_plan",
        "find_config_patterns",
        "generate_migration_code",
        "generate_migration_plan",
        "migrate_metric_classes",
        "migrate_model_factories",
    ]:
        assert banned not in migrate_docs

    for expected in [
        "GANConfig",
        "DCGANConfig",
        "WGANConfig",
        "LSGANConfig",
        "ConditionalGANConfig",
        "CycleGANConfig",
    ]:
        assert expected in payload["config_exports"]

    assert payload["gan_module"] == "artifex.generative_models.core.configuration.gan_config"
    assert payload["migrate_exists"] is False


def test_protein_config_examples_match_supported_runtime_contract() -> None:
    """Protein config examples should use the current typed bundle runtime API."""
    disallowed_references = [
        "artifex.configs.schema",
        "from artifex.configs.utils import",
        "model_dump()",
        "Pydantic",
        "build_protein_extension_mapping",
        "normalize_loaded_protein_extension_mapping",
        "plain mapping",
    ]

    example_paths = [
        "docs/examples/protein/protein-extensions-with-config.md",
        "examples/generative_models/protein/protein_extensions_with_config.py",
        "examples/generative_models/protein/protein_extensions_with_config.ipynb",
    ]

    for relative_path in example_paths:
        contents = (REPO_ROOT / relative_path).read_text()
        for disallowed_reference in disallowed_references:
            assert disallowed_reference not in contents

    python_example = (
        REPO_ROOT / "examples/generative_models/protein/protein_extensions_with_config.py"
    ).read_text()
    required_references = [
        "ProteinExtensionsConfig",
        "get_protein_extensions_config",
        "custom_bundle = ProteinExtensionsConfig(",
        "create_protein_extensions(custom_bundle, rngs=rngs)",
    ]

    for required_reference in required_references:
        assert required_reference in python_example
