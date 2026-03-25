"""Contracts for the shared generative model interface story."""

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_shared_model_protocol_surface_is_narrow() -> None:
    """The base protocol should center on inference/generation, not universal objectives."""
    models_base = importlib.import_module("artifex.generative_models.models.base")

    assert hasattr(models_base, "GenerativeModelProtocol")
    assert hasattr(models_base, "TrainableGenerativeModelProtocol")

    protocol_dict = models_base.GenerativeModelProtocol.__dict__
    trainable_protocol_dict = models_base.TrainableGenerativeModelProtocol.__dict__

    assert "__call__" in protocol_dict
    assert "generate" in protocol_dict
    assert "loss_fn" not in protocol_dict
    assert "sample" not in protocol_dict
    assert "loss_fn" in trainable_protocol_dict


def test_shared_interface_docs_do_not_teach_universal_loss_or_sample_alias() -> None:
    """Shared docs should not teach `loss_fn` and `sample` as universal model requirements."""
    docs_to_check = [
        PROJECT_ROOT / "docs" / "api" / "core" / "base.md",
        PROJECT_ROOT / "docs" / "getting-started" / "core-concepts.md",
    ]

    banned_terms = (
        "Every generative model must implement three key methods",
        "3. **`loss_fn`**: Loss computation for training",
        "4. sample: Alias for generate (backward compatibility)",
        "Generate samples (alias for generate).",
    )
    required_terms = (
        "single-objective",
        "multi-objective",
        "generator_objective",
        "discriminator_objective",
    )

    for path in docs_to_check:
        content = path.read_text(encoding="utf-8")
        content_lower = content.lower()
        for term in banned_terms:
            assert term not in content, (
                f"{path} should not teach stale shared-interface term {term}"
            )
        for term in required_terms:
            assert term in content_lower, f"{path} should document {term}"
