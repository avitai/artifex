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


def test_inference_overview_uses_family_owned_loading_and_retained_optimizer_only() -> None:
    """Inference overview docs should teach the live loading surface only."""
    overview_docs = (REPO_ROOT / "docs/user-guide/inference/overview.md").read_text(
        encoding="utf-8"
    )
    payload = _run_python(
        "import json; "
        "from pathlib import Path; "
        "from artifex.generative_models.core.configuration import DecoderConfig, EncoderConfig, VAEConfig; "
        "from artifex.generative_models.inference.optimization.production import ProductionOptimizer; "
        "print(json.dumps({"
        "'optimization_owner_exists': Path('src/artifex/generative_models/inference/optimization/production.py').exists(), "
        "'vae_config_cls': VAEConfig.__name__, "
        "'encoder_config_cls': EncoderConfig.__name__, "
        "'decoder_config_cls': DecoderConfig.__name__, "
        "'production_optimizer_cls': ProductionOptimizer.__name__"
        "}))"
    )

    for banned in [
        "StreamingGenerator",
        "AsyncInferenceServer",
        "load_exported_model(",
        'input_shape=config["input_shape"]',
        'latent_dim=config["latent_dim"]',
        "DeviceManager",
    ]:
        assert banned not in overview_docs

    for required in [
        "EncoderConfig",
        "DecoderConfig",
        "VAEConfig",
        "load_checkpoint",
        "setup_checkpoint_manager",
        "ProductionOptimizer",
        "artifex.generative_models.inference.optimization.production",
        "jit_compilation",
        "shared streaming helper",
    ]:
        assert required in overview_docs

    assert payload["optimization_owner_exists"] is True
    assert payload["vae_config_cls"] == "VAEConfig"
    assert payload["encoder_config_cls"] == "EncoderConfig"
    assert payload["decoder_config_cls"] == "DecoderConfig"
    assert payload["production_optimizer_cls"] == "ProductionOptimizer"


def test_sampling_guide_uses_live_family_sampling_exports_only() -> None:
    """Sampling guide should stay within the retained family-owned runtime surface."""
    sampling_docs = (REPO_ROOT / "docs/user-guide/inference/sampling.md").read_text(
        encoding="utf-8"
    )
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.models.diffusion as diffusion; "
        "import artifex.generative_models.models.flow as flow; "
        "import artifex.generative_models.models.gan as gan; "
        "print(json.dumps({"
        "'diffusion_exports': sorted(diffusion.__all__), "
        "'flow_exports': sorted(flow.__all__), "
        "'gan_exports': sorted(gan.__all__)"
        "}))"
    )

    for banned in [
        "from artifex.generative_models.models.flow import FlowModel",
        "class StyleGANSampler",
        "class DDPMSampler",
        "class DDIMSampler",
        "class DPMSolver",
        "mix_styles(",
    ]:
        assert banned not in sampling_docs

    for required in [
        "from artifex.generative_models.models.vae import VAE",
        "from artifex.generative_models.models.gan import GAN",
        "from artifex.generative_models.models.diffusion import DDPMModel",
        "from artifex.generative_models.models.flow import NormalizingFlow",
        'scheduler="ddim"',
        "vae.sample(",
        "gan.generate(",
        "ddpm.generate(",
        "flow.sample(",
        "flow.log_prob(",
    ]:
        assert required in sampling_docs

    assert "FlowModel" not in payload["flow_exports"]
    assert "NormalizingFlow" in payload["flow_exports"]
    assert "DDPMModel" in payload["diffusion_exports"]
    assert "GAN" in payload["gan_exports"]
