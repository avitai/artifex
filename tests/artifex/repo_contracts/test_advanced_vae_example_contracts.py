from __future__ import annotations

import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = REPO_ROOT / "examples" / "generative_models" / "image" / "vae" / "advanced_vae.py"
DOC_PATH = REPO_ROOT / "docs" / "examples" / "advanced" / "advanced-vae.md"


def test_advanced_vae_surface_exports_capacity_model_and_uses_real_perplexity() -> None:
    """The advanced VAE example should rely on real exported runtime surfaces."""
    vae_module = importlib.import_module("artifex.generative_models.models.vae")
    example_text = EXAMPLE_PATH.read_text(encoding="utf-8")
    docs_text = DOC_PATH.read_text(encoding="utf-8")

    assert "BetaVAEWithCapacity" in getattr(vae_module, "__all__", [])
    assert hasattr(vae_module, "BetaVAEWithCapacity")

    assert 'losses["perplexity"]' in example_text
    assert 'losses.get("perplexity", 0.0)' not in example_text

    for text in [example_text, docs_text]:
        assert "Monitoring VQ-VAE codebook usage and perplexity" in text
        assert "BetaVAEWithCapacity" in text

    assert "from artifex.generative_models.models.vae import BetaVAEWithCapacity" in docs_text
