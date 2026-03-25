"""Repository contracts for the public model-creation surface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CURATED_MODEL_CREATION_FILES = {
    "docs/api/core/base.md",
    "docs/api/core/configuration.md",
    "docs/api/data/loaders.md",
    "docs/api/models/diffusion.md",
    "docs/api/models/flow.md",
    "docs/api/models/gan.md",
    "docs/api/models/vae.md",
    "docs/community/contributing.md",
    "docs/core/architecture.md",
    "docs/core/gan.md",
    "docs/examples/basic/diffusion-mnist-demo.md",
    "docs/examples/overview.md",
    "docs/examples/protein/protein-model-with-modality.md",
    "src/artifex/generative_models/README.md",
    "src/artifex/generative_models/factory/README.md",
    "docs/getting-started/core-concepts.md",
    "docs/generative_models/index.md",
    "docs/papers/artifex_arxiv_preprint.md",
    "docs/user-guide/integrations/deployment.md",
    "docs/user-guide/training/overview.md",
    "docs/user-guide/training/training-guide.md",
    "docs/examples/framework/framework-features-demo.md",
    "docs/examples/advanced/advanced-training.md",
    "examples/generative_models/geometric/geometric_models_demo.py",
    "examples/generative_models/framework_features_demo.py",
    "examples/generative_models/advanced_training_example.py",
    "examples/generative_models/protein/protein_model_with_modality.py",
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


def test_core_configuration_surface_no_longer_exports_model_config() -> None:
    """The supported config surface should expose family-specific configs, not ModelConfig."""
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.core.configuration as configuration; "
        "print(json.dumps({"
        "'has_model_config': hasattr(configuration, 'ModelConfig'), "
        "'exports': sorted(configuration.__all__)"
        "}))"
    )

    assert payload["has_model_config"] is False
    assert "ModelConfig" not in payload["exports"]


def test_curated_model_creation_docs_and_examples_use_family_specific_configs_only() -> None:
    """High-signal public surfaces should not teach the removed generic creation story."""
    banned_tokens = (
        "ModelConfig",
        "model_class=",
        "from artifex.generative_models.zoo import zoo",
        "zoo.create_model(",
    )

    for relative_path in CURATED_MODEL_CREATION_FILES:
        contents = (REPO_ROOT / relative_path).read_text()
        for banned_token in banned_tokens:
            assert banned_token not in contents, f"{relative_path} still contains {banned_token!r}"


def test_model_zoo_docs_are_explicitly_migrated_not_taught_as_live_runtime() -> None:
    """The zoo docs page should be a migration page, not a working preset tutorial."""
    contents = (REPO_ROOT / "docs/zoo/index.md").read_text()

    assert "removed" in contents.lower()
    assert "typed config" in contents.lower()
    assert "zoo.create_model(" not in contents
