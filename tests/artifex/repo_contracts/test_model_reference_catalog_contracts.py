from __future__ import annotations

import json
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


def test_models_index_uses_live_supported_model_owners() -> None:
    """The model index should teach the live owner packages and imports only."""
    docs = (REPO_ROOT / "docs/models/index.md").read_text(encoding="utf-8")
    payload = _run_python(
        "import json; "
        "from pathlib import Path; "
        "import artifex.generative_models.models.diffusion as diffusion; "
        "print(json.dumps({"
        "'diffusion_exports': sorted(diffusion.__all__), "
        "'common_unet_exists': Path('src/artifex/generative_models/models/common/unet.py').exists(), "
        "'common_conditioning_exists': Path('src/artifex/generative_models/models/common/conditioning.py').exists(), "
        "'stylegan_exists': Path('src/artifex/generative_models/models/gan/stylegan.py').exists()"
        "}))"
    )

    for banned in [
        "from artifex.generative_models.models.common.unet import UNet",
        "from artifex.generative_models.models.diffusion import DiT",
        "models/common/unet.py",
        "models/common/conditioning.py",
    ]:
        assert banned not in docs

    for required in [
        "DiTModel",
        "artifex.generative_models.models.diffusion.stable_diffusion import StableDiffusionModel",
        "artifex.generative_models.models.backbones.unet",
        "UNetBackboneConfig",
        "not re-exported from `artifex.generative_models.models.diffusion`",
    ]:
        assert required in docs

    assert "DiTModel" in payload["diffusion_exports"]
    assert "StableDiffusionModel" not in payload["diffusion_exports"]
    assert payload["common_unet_exists"] is False
    assert payload["common_conditioning_exists"] is False
    assert payload["stylegan_exists"] is False


def test_curated_model_pages_match_live_owner_surfaces() -> None:
    """Curated docs.models pages should list only live owner symbols."""
    dit_docs = (REPO_ROOT / "docs/models/dit.md").read_text(encoding="utf-8")
    guidance_docs = (REPO_ROOT / "docs/models/guidance.md").read_text(encoding="utf-8")
    stable_diffusion_docs = (REPO_ROOT / "docs/models/stable_diffusion.md").read_text(
        encoding="utf-8"
    )
    wgan_docs = (REPO_ROOT / "docs/models/wgan.md").read_text(encoding="utf-8")
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.models.diffusion as diffusion; "
        "import artifex.generative_models.models.gan as gan; "
        "print(json.dumps({"
        "'diffusion_exports': sorted(diffusion.__all__), "
        "'gan_exports': sorted(gan.__all__)"
        "}))"
    )

    for required in ["artifex.generative_models.models.diffusion.dit", "DiTModel"]:
        assert required in dit_docs
    for banned in [
        "create_dit_backbone",
        "sample_step",
        "generate()",
        "def __call__",
        "def __init__",
    ]:
        assert banned not in dit_docs

    for required in [
        "ClassifierFreeGuidance",
        "ClassifierGuidance",
        "ConditionalDiffusionMixin",
        "GuidedDiffusionModel",
        "apply_guidance",
        "linear_guidance_schedule",
        "cosine_guidance_schedule",
    ]:
        assert required in guidance_docs
        assert required in payload["diffusion_exports"]
    for banned in [
        "classifier_fn",
        "guided_sample_step",
        "generate()",
        "def __call__",
        "def __init__",
    ]:
        assert banned not in guidance_docs

    for required in [
        "artifex.generative_models.models.diffusion.stable_diffusion",
        "StableDiffusionModel",
        "not re-exported from `artifex.generative_models.models.diffusion`",
    ]:
        assert required in stable_diffusion_docs
    for banned in [
        "TextEncoder",
        "generate_with_text",
        "interpolate_text",
        "compute_text_similarity",
        "encode_text",
    ]:
        assert banned not in stable_diffusion_docs

    for required in ["WGAN", "WGANGenerator", "WGANDiscriminator", "compute_gradient_penalty"]:
        assert required in wgan_docs
        assert required in payload["gan_exports"]
    for banned in [
        "discriminator_fn",
        "generator_loss",
        "discriminator_loss",
        "generate()",
        "def __call__",
        "def __init__",
    ]:
        assert banned not in wgan_docs


def test_planned_model_pages_are_marked_and_separated_from_supported_nav() -> None:
    """Planned or removed model pages should not stay in the supported catalog."""
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    model_reference_block = mkdocs.split("      - Model Reference:\n", 1)[1].split(
        "      - Coming Soon:\n", 1
    )[0]
    coming_soon_block = mkdocs.split("      - Coming Soon:\n", 1)[1].split(
        "      - Notebooks:\n", 1
    )[0]

    planned_pages = {
        "models/conditioning.md": [
            "Status: Coming soon",
            "artifex.generative_models.models.diffusion.guidance",
        ],
        "models/stylegan.md": [
            "Status: Coming soon",
            "artifex.generative_models.models.gan.stylegan3",
        ],
        "models/unet.md": [
            "Status: Coming soon",
            "artifex.generative_models.models.backbones.unet",
        ],
    }

    for relative_path, required_terms in planned_pages.items():
        assert relative_path not in model_reference_block
        assert relative_path in coming_soon_block
        contents = (REPO_ROOT / "docs" / Path(relative_path)).read_text(encoding="utf-8")
        for required_term in required_terms:
            assert required_term in contents
