import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_python(code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_core_package_import_keeps_children_lazy() -> None:
    """Importing the core package should not eagerly load its heavy children."""
    payload = _run_python(
        "import json, sys; "
        "import artifex.generative_models.core as core; "
        "print(json.dumps({"
        "'cli_loaded': 'artifex.generative_models.core.cli' in sys.modules, "
        "'configuration_loaded': 'artifex.generative_models.core.configuration' in sys.modules, "
        "'distributions_loaded': 'artifex.generative_models.core.distributions' in sys.modules, "
        "'evaluation_loaded': 'artifex.generative_models.core.evaluation' in sys.modules, "
        "'layers_loaded': 'artifex.generative_models.core.layers' in sys.modules, "
        "'losses_loaded': 'artifex.generative_models.core.losses' in sys.modules, "
        "'protocols_loaded': 'artifex.generative_models.core.protocols' in sys.modules, "
        "'sampling_loaded': 'artifex.generative_models.core.sampling' in sys.modules, "
        "'checkpointing_loaded': 'artifex.generative_models.core.checkpointing' in sys.modules, "
        "'device_manager_loaded': 'artifex.generative_models.core.device_manager' in sys.modules, "
        "'device_testing_loaded': 'artifex.generative_models.core.device_testing' in sys.modules, "
        "'gradient_checkpointing_loaded': "
        "'artifex.generative_models.core.gradient_checkpointing' in sys.modules, "
        "'jax_loaded': 'jax' in sys.modules, "
        "'all': list(getattr(core, '__all__'))"
        "}))"
    )

    assert payload["cli_loaded"] is False
    assert payload["configuration_loaded"] is False
    assert payload["distributions_loaded"] is False
    assert payload["evaluation_loaded"] is False
    assert payload["layers_loaded"] is False
    assert payload["losses_loaded"] is False
    assert payload["protocols_loaded"] is False
    assert payload["sampling_loaded"] is False
    assert payload["checkpointing_loaded"] is False
    assert payload["device_manager_loaded"] is False
    assert payload["device_testing_loaded"] is False
    assert payload["gradient_checkpointing_loaded"] is False
    assert payload["jax_loaded"] is False
    assert payload["all"] == [
        "cli",
        "configuration",
        "distributions",
        "evaluation",
        "layers",
        "losses",
        "protocols",
        "sampling",
        "load_checkpoint",
        "save_checkpoint",
        "setup_checkpoint_manager",
        "CHECKPOINT_POLICIES",
        "apply_remat",
        "resolve_checkpoint_policy",
        "DeviceManager",
        "print_test_results",
        "run_device_tests",
    ]


def test_core_exports_resolve_lazily() -> None:
    """Core exports should remain accessible on explicit attribute access."""
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.core as core; "
        "print(json.dumps({"
        "'sampling_module': core.sampling.__name__, "
        "'load_checkpoint_module': core.load_checkpoint.__module__, "
        "'device_manager_module': core.DeviceManager.__module__, "
        "'run_device_tests_module': core.run_device_tests.__module__, "
        "'print_test_results_module': core.print_test_results.__module__"
        "}))"
    )

    assert payload["sampling_module"] == "artifex.generative_models.core.sampling"
    assert payload["load_checkpoint_module"] == "artifex.generative_models.core.checkpointing"
    assert payload["device_manager_module"] == "artifex.generative_models.core.device_manager"
    assert payload["run_device_tests_module"] == "artifex.generative_models.core.device_testing"
    assert payload["print_test_results_module"] == "artifex.generative_models.core.device_testing"


def test_core_overview_docs_match_live_core_surface() -> None:
    """The core overview should route readers through live namespaces only."""
    docs = (REPO_ROOT / "docs/core/index.md").read_text(encoding="utf-8")
    payload = _run_python(
        "import json; "
        "from artifex.generative_models import core; "
        "from artifex.generative_models.core.sampling import "
        "BlackJAXNUTS, mcmc_sampling, sde_sampling; "
        "from artifex.generative_models.core.evaluation.metrics import "
        "FrechetInceptionDistance, InceptionScore, PrecisionRecall; "
        "from artifex.generative_models.core.layers import "
        "FlashMultiHeadAttention, TransformerEncoder, ResNetBlock; "
        "from artifex.generative_models.core.protocols import "
        "BatchableDatasetProtocol, MetricBase, NoiseScheduleProtocol; "
        "print(json.dumps({"
        "'core_exports': list(core.__all__), "
        "'sampling_module': BlackJAXNUTS.__module__, "
        "'mcmc_module': mcmc_sampling.__module__, "
        "'sde_module': sde_sampling.__module__, "
        "'fid_module': FrechetInceptionDistance.__module__, "
        "'inception_module': InceptionScore.__module__, "
        "'precision_recall_module': PrecisionRecall.__module__, "
        "'flash_attention_module': FlashMultiHeadAttention.__module__, "
        "'transformer_module': TransformerEncoder.__module__, "
        "'resnet_module': ResNetBlock.__module__, "
        "'dataset_protocol_module': BatchableDatasetProtocol.__module__, "
        "'metric_protocol_module': MetricBase.__module__, "
        "'noise_schedule_module': NoiseScheduleProtocol.__module__"
        "}))"
    )

    required_terms = [
        "from artifex.generative_models import core",
        "core.sampling",
        "artifex.generative_models.core.evaluation.metrics",
        "BlackJAXNUTS",
        "mcmc_sampling",
        "sde_sampling",
        "FrechetInceptionDistance",
        "InceptionScore",
        "PrecisionRecall",
        "FlashMultiHeadAttention",
        "TransformerEncoder",
        "ResNetBlock",
        "BatchableDatasetProtocol",
        "MetricBase",
        "NoiseScheduleProtocol",
    ]
    banned_patterns = [
        r"\bancestral_sample\b",
        r"\blangevin_dynamics\b",
        r"\bhmc_sample\b",
        r"\bBlackJAXSampler\b",
        r"\bode_solver\b",
        r"\bsde_solver\b",
        r"\bartifex\.generative_models\.core\.metrics\b",
        r"\bMultiHeadAttention\b",
        r"\bFlashAttention\b",
        r"\bCrossAttention\b",
        r"\bTransformerBlock\b",
        r"\bResNetBlockV2\b",
    ]

    for required in required_terms:
        assert required in docs

    for banned in banned_patterns:
        assert re.search(banned, docs) is None

    assert "sampling" in payload["core_exports"]
    assert "evaluation" in payload["core_exports"]
    assert "layers" in payload["core_exports"]
    assert "protocols" in payload["core_exports"]
    assert payload["sampling_module"] == "artifex.generative_models.core.sampling.blackjax_samplers"
    assert payload["mcmc_module"] == "artifex.generative_models.core.sampling.mcmc"
    assert payload["sde_module"] == "artifex.generative_models.core.sampling.sde"
    assert payload["fid_module"].startswith(
        "artifex.generative_models.core.evaluation.metrics.image"
    )
    assert payload["inception_module"].startswith(
        "artifex.generative_models.core.evaluation.metrics.image"
    )
    assert payload["precision_recall_module"].startswith(
        "artifex.generative_models.core.evaluation.metrics.general"
    )
    assert (
        payload["flash_attention_module"] == "artifex.generative_models.core.layers.flash_attention"
    )
    assert payload["transformer_module"] == "artifex.generative_models.core.layers.transformers"
    assert payload["resnet_module"] == "artifex.generative_models.core.layers.resnet"
    assert payload["dataset_protocol_module"] == "calibrax.core.protocols"
    assert payload["metric_protocol_module"] == "artifex.generative_models.core.protocols.metrics"
    assert payload["noise_schedule_module"] == "artifex.generative_models.core.protocols.training"


def test_models_package_import_keeps_children_lazy() -> None:
    """Importing the models package should not eagerly import model families."""
    payload = _run_python(
        "import json, sys; "
        "import artifex.generative_models.models as models; "
        "print(json.dumps({"
        "'diffusion_loaded': 'artifex.generative_models.models.diffusion' in sys.modules, "
        "'geometric_loaded': 'artifex.generative_models.models.geometric' in sys.modules, "
        "'vae_loaded': 'artifex.generative_models.models.vae' in sys.modules, "
        "'jax_loaded': 'jax' in sys.modules, "
        "'all': list(getattr(models, '__all__'))"
        "}))"
    )

    assert payload["diffusion_loaded"] is False
    assert payload["geometric_loaded"] is False
    assert payload["vae_loaded"] is False
    assert payload["jax_loaded"] is False
    assert payload["all"] == ["diffusion", "geometric", "vae"]


def test_models_exports_resolve_lazily() -> None:
    """Models exports should stay focused on concrete model families only."""
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.models as models; "
        "print(json.dumps({"
        "'diffusion_module': models.diffusion.__name__, "
        "'vae_module': models.vae.__name__, "
        "'geometric_module': models.geometric.__name__, "
        "'registry_in_dir': 'registry' in dir(models), "
        "'has_registry_attr': hasattr(models, 'registry')"
        "}))"
    )

    assert payload["diffusion_module"] == "artifex.generative_models.models.diffusion"
    assert payload["vae_module"] == "artifex.generative_models.models.vae"
    assert payload["geometric_module"] == "artifex.generative_models.models.geometric"
    assert payload["registry_in_dir"] is False
    assert payload["has_registry_attr"] is False


def test_factory_package_import_keeps_default_builders_lazy() -> None:
    """Importing the factory package should keep its public surface lazy."""
    payload = _run_python(
        "import json, sys; "
        "import artifex.generative_models.factory as factory; "
        "print(json.dumps({"
        "'core_loaded': 'artifex.generative_models.factory.core' in sys.modules, "
        "'registry_loaded': 'artifex.generative_models.factory.registry' in sys.modules, "
        "'autoregressive_builder_loaded': "
        "'artifex.generative_models.factory.builders.autoregressive' in sys.modules, "
        "'diffusion_builder_loaded': "
        "'artifex.generative_models.factory.builders.diffusion' in sys.modules, "
        "'ebm_builder_loaded': 'artifex.generative_models.factory.builders.ebm' in sys.modules, "
        "'flow_builder_loaded': 'artifex.generative_models.factory.builders.flow' in sys.modules, "
        "'gan_builder_loaded': 'artifex.generative_models.factory.builders.gan' in sys.modules, "
        "'geometric_builder_loaded': "
        "'artifex.generative_models.factory.builders.geometric' in sys.modules, "
        "'vae_builder_loaded': 'artifex.generative_models.factory.builders.vae' in sys.modules, "
        "'all': list(getattr(factory, '__all__'))"
        "}))"
    )

    assert payload["core_loaded"] is False
    assert payload["registry_loaded"] is False
    assert payload["autoregressive_builder_loaded"] is False
    assert payload["diffusion_builder_loaded"] is False
    assert payload["ebm_builder_loaded"] is False
    assert payload["flow_builder_loaded"] is False
    assert payload["gan_builder_loaded"] is False
    assert payload["geometric_builder_loaded"] is False
    assert payload["vae_builder_loaded"] is False
    assert payload["all"] == ["ModelFactory", "create_model", "create_model_with_extensions"]


def test_factory_exports_resolve_without_preloading_builders() -> None:
    """Accessing factory exports should not eagerly register builders."""
    payload = _run_python(
        "import json, sys; "
        "import artifex.generative_models.factory as factory; "
        "print(json.dumps({"
        "'create_model_module': factory.create_model.__module__, "
        "'create_model_with_extensions_module': factory.create_model_with_extensions.__module__, "
        "'model_factory_module': factory.ModelFactory.__module__, "
        "'has_registry_symbol': hasattr(factory, 'ModelTypeRegistry'), "
        "'has_builder_symbol': hasattr(factory, 'ModelBuilder'), "
        "'builder_module_loaded': "
        "'artifex.generative_models.factory.builders.diffusion' in sys.modules"
        "}))"
    )

    assert payload["create_model_module"] == "artifex.generative_models.factory.core"
    assert (
        payload["create_model_with_extensions_module"] == "artifex.generative_models.factory.core"
    )
    assert payload["model_factory_module"] == "artifex.generative_models.factory.core"
    assert payload["has_registry_symbol"] is False
    assert payload["has_builder_symbol"] is False
    assert payload["builder_module_loaded"] is False


def test_extensions_package_import_keeps_children_lazy() -> None:
    """Importing the extensions package should not eagerly import protein extensions."""
    payload = _run_python(
        "import json, sys; "
        "import artifex.generative_models.extensions as extensions; "
        "print(json.dumps({"
        "'base_loaded': 'artifex.generative_models.extensions.base' in sys.modules, "
        "'registry_loaded': 'artifex.generative_models.extensions.registry' in sys.modules, "
        "'protein_loaded': 'artifex.generative_models.extensions.protein' in sys.modules, "
        "'jax_loaded': 'jax' in sys.modules, "
        "'all': list(getattr(extensions, '__all__'))"
        "}))"
    )

    assert payload["base_loaded"] is False
    assert payload["registry_loaded"] is False
    assert payload["protein_loaded"] is False
    assert payload["jax_loaded"] is False
    assert payload["all"] == [
        "ExtensionDict",
        "ModelExtension",
        "ConstraintExtension",
        "ExtensionsRegistry",
        "ExtensionType",
        "get_extensions_registry",
        "BondAngleExtension",
        "BondLengthExtension",
        "ProteinBackboneConstraint",
        "ProteinDihedralConstraint",
        "ProteinMixinExtension",
        "create_protein_extensions",
    ]


def test_extensions_exports_resolve_lazily() -> None:
    """Extension exports should remain accessible on explicit attribute access."""
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.extensions as extensions; "
        "print(json.dumps({"
        "'registry_module': extensions.get_extensions_registry.__module__, "
        "'protein_factory_module': extensions.create_protein_extensions.__module__, "
        "'extension_type_module': extensions.ExtensionType.__module__"
        "}))"
    )

    assert payload["registry_module"] == "artifex.generative_models.extensions.registry"
    assert payload["protein_factory_module"].startswith(
        "artifex.generative_models.extensions.protein"
    )
    assert payload["extension_type_module"] == "artifex.generative_models.extensions.registry"


def test_removed_factory_forwarders_are_not_shipped_or_documented() -> None:
    """Removed duplicate factory surfaces should not reappear in code or docs."""
    assert not (REPO_ROOT / "src/artifex/generative_models/models/factories.py").exists()
    assert not (REPO_ROOT / "src/artifex/generative_models/models/registry.py").exists()
    assert not (
        REPO_ROOT / "src/artifex/generative_models/models/diffusion/future_recommendations.md"
    ).exists()
    assert not (REPO_ROOT / "docs/models/factories.md").exists()
    assert not (REPO_ROOT / "docs/models/registry.md").exists()

    files_to_check = [
        "docs/generative_models/index.md",
        "docs/models/index.md",
        "docs/papers/artifex_arxiv_preprint.md",
        "src/artifex/generative_models/factory/README.md",
    ]
    banned_strings = [
        "artifex.generative_models.factories",
        "artifex.generative_models.models.factories",
        "artifex.generative_models.models.registry",
        "ModelRegistry",
        "register_model(",
    ]

    for relative_path in files_to_check:
        contents = (REPO_ROOT / relative_path).read_text()
        for banned in banned_strings:
            assert banned not in contents
