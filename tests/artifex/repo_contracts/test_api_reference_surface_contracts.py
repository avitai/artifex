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


def test_core_base_api_docs_use_live_shared_interface_contract() -> None:
    """Shared API docs should reflect the live narrow base-model interface."""
    docs = (REPO_ROOT / "docs/api/core/base.md").read_text(encoding="utf-8")

    banned_terms = [
        "rngs: nnx.Rngs | None = None,\n    training: bool = False,",
        "outputs = model(batch, timesteps, training=True, rngs=rngs)",
        "loss_dict = model.loss_fn(batch, outputs, rngs=rngs)",
        "samples = model.generate(n_samples=16, rngs=rngs)",
        "latent = model.encode(batch, rngs=rngs)",
    ]
    required_terms = [
        "model.train()",
        "model.eval()",
        "stored module RNG state",
        "single-objective",
        "multi-objective",
        "generator_objective",
        "discriminator_objective",
    ]

    for banned in banned_terms:
        assert banned not in docs

    for required in required_terms:
        assert required in docs


def test_diffusion_api_docs_match_live_exports_and_signatures() -> None:
    """Diffusion API docs should use live exports and signatures only."""
    docs = (REPO_ROOT / "docs/api/models/diffusion.md").read_text(encoding="utf-8")
    payload = _run_python(
        "import json; import artifex.generative_models.models.diffusion as m; "
        "print(json.dumps({'exports': sorted(m.__all__)}))"
    )

    for expected in [
        "DiffusionModel",
        "DDPMModel",
        "ScoreDiffusionModel",
        "LDMModel",
        "DiTModel",
        "ClassifierFreeGuidance",
        "ClassifierGuidance",
        "GuidedDiffusionModel",
        "apply_guidance",
    ]:
        assert expected in payload["exports"]
        assert expected in docs

    banned_terms = [
        "__call__(x, timesteps, *, rngs=None, training=False, **kwargs)",
        "q_sample(x_start, t, noise=None, *, rngs=None)",
        "generate(n_samples=1, *, rngs=None, shape=None, clip_denoised=True, **kwargs)",
        "loss_fn(batch, model_outputs, *, rngs=None, **kwargs)",
        "DDIMModel",
        "StableDiffusionModel",
        "TextEncoder",
    ]
    required_terms = [
        "DiffusionModel.__call__(x, timesteps, *, conditioning=None, **kwargs)",
        "DiffusionModel.q_sample(x_start, t, noise=None)",
        'DDPMModel.sample(n_samples_or_shape, scheduler="ddpm", steps=None)',
        "model.train()",
        "model.eval()",
    ]

    for banned in banned_terms:
        assert banned not in docs

    for required in required_terms:
        assert required in docs


def test_vae_api_docs_match_live_constructor_and_method_signatures() -> None:
    """VAE API docs should use the config-based constructor and stored RNG story."""
    docs = (REPO_ROOT / "docs/api/models/vae.md").read_text(encoding="utf-8")
    payload = _run_python(
        "import json; import artifex.generative_models.models.vae as m; "
        "print(json.dumps({'exports': sorted(m.__all__)}))"
    )

    for expected in ["VAE", "BetaVAE", "ConditionalVAE", "VQVAE"]:
        assert expected in payload["exports"]
        assert expected in docs

    banned_terms = [
        "encoder: nnx.Module",
        "decoder: nnx.Module",
        "latent_dim: int",
        "vae.encode(x, rngs=rngs)",
        "vae.decode(z, rngs=rngs)",
        "vae.reparameterize(mean, log_var, rngs=rngs)",
        "outputs = vae(x, rngs=rngs)",
    ]
    required_terms = [
        "VAE(config: VAEConfig, *, rngs: nnx.Rngs, precision=None)",
        "encode(x)",
        "decode(z)",
        "reparameterize(mean, log_var)",
        "loss_fn(batch, model_outputs, *, beta=None, reconstruction_loss_fn=None)",
        "stored RNG state",
    ]

    for banned in banned_terms:
        assert banned not in docs

    for required in required_terms:
        assert required in docs


def test_sampling_and_energy_api_docs_use_live_top_level_surfaces() -> None:
    """Sampling and energy API docs should not publish dead helper paths."""
    sampling_docs = (REPO_ROOT / "docs/api/sampling.md").read_text(encoding="utf-8")
    ebm_docs = (REPO_ROOT / "docs/api/models/ebm.md").read_text(encoding="utf-8")
    payload = _run_python(
        "import json; import artifex.generative_models.core.sampling as sampling; "
        "from artifex.generative_models.models.energy import EnergyBasedModel; "
        "print(json.dumps({'sampling_exports': sorted(sampling.__all__), 'energy_module': EnergyBasedModel.__module__}))"
    )

    assert payload["energy_module"] == "artifex.generative_models.models.energy.base"

    for expected in [
        "BlackJAXHMC",
        "BlackJAXMALA",
        "BlackJAXNUTS",
        "mcmc_sampling",
        "sde_sampling",
    ]:
        assert expected in payload["sampling_exports"]
        assert expected in sampling_docs

    assert "sample_from_model" not in sampling_docs
    assert "temperature_sample" not in sampling_docs
    assert "models.ebm.base" not in ebm_docs
    assert "from artifex.generative_models.models.energy import EnergyBasedModel" in ebm_docs
