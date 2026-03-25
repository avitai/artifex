"""Contracts for the public generative loss surface."""

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_core_losses_public_surface_does_not_export_suite_builders() -> None:
    """Loss modules should expose primitives, not duplicate management facades."""
    losses_module = importlib.import_module("artifex.generative_models.core.losses")

    assert not hasattr(losses_module, "create_gan_loss_suite")
    assert not hasattr(losses_module, "create_image_generation_loss_suite")
    assert not hasattr(losses_module, "CompositeLoss")
    assert not hasattr(losses_module, "WeightedLoss")
    assert not hasattr(losses_module, "ScheduledLoss")
    assert not hasattr(losses_module, "LossCollection")
    assert not hasattr(losses_module, "LossMetrics")
    assert not hasattr(losses_module, "LossScheduler")
    assert not hasattr(losses_module, "create_loss_suite")
    assert not hasattr(losses_module, "create_weighted_loss")


def test_loss_docs_do_not_teach_deleted_suite_builders() -> None:
    """User-facing docs should teach explicit objectives instead of suite facades."""
    docs_to_check = [
        PROJECT_ROOT / "docs" / "examples" / "losses" / "loss-examples.md",
        PROJECT_ROOT / "docs" / "api" / "core" / "losses.md",
        PROJECT_ROOT / "docs" / "examples" / "framework" / "framework-features-demo.md",
        PROJECT_ROOT / "docs" / "core" / "index.md",
        PROJECT_ROOT / "docs" / "core" / "architecture.md",
        PROJECT_ROOT / "examples" / "EXAMPLES_GUIDE.md",
    ]

    banned_terms = (
        "CompositeLoss",
        "WeightedLoss",
        "ScheduledLoss",
        "create_gan_loss_suite",
        "create_image_generation_loss_suite",
        "LossCollection",
        "LossMetrics",
        "LossScheduler",
        "create_loss_suite",
        "create_weighted_loss",
    )

    for doc_path in docs_to_check:
        content = doc_path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{doc_path} should not reference {term}"

    assert not (PROJECT_ROOT / "docs" / "core" / "composable.md").exists()


def test_core_loss_docs_teach_primitive_and_calibrax_backed_surface() -> None:
    """The core loss docs should describe the surviving narrow public surface."""
    loss_docs = PROJECT_ROOT / "docs" / "api" / "core" / "losses.md"
    content = loss_docs.read_text(encoding="utf-8")

    banned_terms = (
        "40+ Loss Functions",
        "Composable Framework",
        "complete catalog of loss functions",
        "Easily combine multiple losses with weights and scheduling",
    )
    required_terms = (
        "CalibraX",
        "primitives",
        "explicit",
    )

    for term in banned_terms:
        assert term not in content, (
            f"{loss_docs} should not overstate the public loss surface with {term}"
        )

    for term in required_terms:
        assert term in content, f"{loss_docs} should document the narrower {term} story"


def test_gan_docs_do_not_teach_fake_single_objective_loss_fn() -> None:
    """GAN docs should teach explicit generator/discriminator objectives only."""
    gan_docs = PROJECT_ROOT / "docs" / "api" / "models" / "gan.md"
    content = gan_docs.read_text(encoding="utf-8")

    banned_terms = (
        "#### `loss_fn(",
        "Compute GAN loss for training.",
        '"loss": Total loss (generator + discriminator)',
        "gan.loss_fn(",
        "losses = gan.loss_fn",
    )
    required_terms = (
        "generator_objective",
        "discriminator_objective",
        "NotImplementedError",
        "GAN training requires separate generator and discriminator objectives",
    )

    for term in banned_terms:
        assert term not in content, (
            f"{gan_docs} should not reference stale GAN loss entrypoint {term}"
        )

    for term in required_terms:
        assert term in content, f"{gan_docs} should document {term}"


def test_cyclegan_docs_do_not_teach_fake_single_objective_loss_fn() -> None:
    """CycleGAN docs should describe explicit generator/discriminator objectives."""
    cyclegan_docs = PROJECT_ROOT / "docs" / "models" / "cyclegan.md"
    content = cyclegan_docs.read_text(encoding="utf-8")

    banned_terms = (
        "### loss_fn",
        "def loss_fn()",
    )
    required_terms = (
        "generator_objective",
        "discriminator_objective",
        "CycleGAN training does not expose a combined `loss_fn(...)`",
    )

    for term in banned_terms:
        assert term not in content, (
            f"{cyclegan_docs} should not reference stale CycleGAN loss entrypoint {term}"
        )

    for term in required_terms:
        assert term in content, f"{cyclegan_docs} should document {term}"


def test_model_loss_examples_and_docs_use_total_loss_contract() -> None:
    """Model-facing examples should use the canonical `total_loss` contract."""
    files_and_banned_terms = {
        PROJECT_ROOT / "examples" / "generative_models" / "image" / "vae" / "advanced_vae.py": (
            'losses["loss"]',
            "losses['loss']",
            'history = {"loss":',
            'epoch_metrics = {"loss":',
        ),
        PROJECT_ROOT / "examples" / "generative_models" / "energy" / "simple_ebm_example.py": (
            'loss_dict["loss"]',
            "loss_dict['loss']",
        ),
        PROJECT_ROOT
        / "examples"
        / "generative_models"
        / "protein"
        / "protein_model_extension.py": (
            'loss_outputs["loss"]',
            "loss_outputs['loss']",
        ),
        PROJECT_ROOT / "docs" / "examples" / "advanced" / "advanced-vae.md": (
            "output['loss']",
            'output["loss"]',
            '"loss": total_loss',
            '"loss": recon_loss + kl_loss',
            '"loss": [],',
            "metrics['loss']",
            'metrics["loss"]',
        ),
        PROJECT_ROOT / "docs" / "examples" / "energy" / "simple-ebm.md": (
            "loss_dict['loss']",
            'loss_dict["loss"]',
        ),
        PROJECT_ROOT / "docs" / "user-guide" / "models" / "vae-guide.md": (
            'return losses["loss"], losses',
            "return losses['loss'], losses",
        ),
        PROJECT_ROOT / "docs" / "user-guide" / "models" / "ebm-guide.md": (
            "loss_dict['loss']",
            'loss_dict["loss"]',
        ),
        PROJECT_ROOT / "docs" / "user-guide" / "models" / "autoregressive-guide.md": (
            "loss_dict['loss']",
            'loss_dict["loss"]',
        ),
        PROJECT_ROOT / "docs" / "api" / "core" / "base.md": (
            "outputs['loss']",
            'outputs["loss"]',
            "loss_dict['loss']",
            'loss_dict["loss"]',
            '"loss": recon_loss',
        ),
        PROJECT_ROOT / "docs" / "api" / "models" / "diffusion.md": (
            "outputs['loss']",
            'outputs["loss"]',
            '"loss": Primary loss value for optimization',
            "'loss': Primary loss value for optimization",
            "loss_dict['loss']",
            'loss_dict["loss"]',
            'return loss_dict["loss"]',
            "return loss_dict['loss']",
            'return {"loss": loss}',
        ),
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "advanced-features.md": (
            'return outputs["loss"], outputs',
            "return outputs['loss'], outputs",
            'scaled_loss = scaler.scale_loss(outputs["loss"])',
            "scaled_loss = scaler.scale_loss(outputs['loss'])",
            'return unscaled_grads, outputs["loss"]',
            "return unscaled_grads, outputs['loss']",
        ),
    }

    for path, banned_terms in files_and_banned_terms.items():
        content = path.read_text(encoding="utf-8")
        for term in banned_terms:
            assert term not in content, f"{path} should not reference stale model loss term {term}"


def test_geometric_docs_and_examples_only_teach_runtime_backed_loss_owners() -> None:
    """Geometric docs/examples should not advertise dead config-based loss knobs."""
    per_file_contracts = {
        PROJECT_ROOT
        / "examples"
        / "generative_models"
        / "geometric"
        / "geometric_losses_demo.py": {
            "forbidden": (
                'loss_type="chamfer"',
                'loss_type="earth_mover"',
                "vertex_loss_weight=",
                "normal_loss_weight=",
                "edge_loss_weight=",
            ),
            "required": (
                "chamfer_distance",
                "earth_mover_distance",
                "hausdorff_distance",
                "get_mesh_loss",
            ),
        },
        PROJECT_ROOT
        / "examples"
        / "generative_models"
        / "geometric"
        / "geometric_models_demo.py": {
            "forbidden": (
                "- `loss_type`: Distance metric",
                "- `template_type`: Initial mesh template",
                "geometric loss weights for vertices, normals, and edges",
            ),
            "required": (
                "PointCloudConfig(",
                "MeshConfig(",
                "VoxelConfig(",
            ),
        },
        PROJECT_ROOT / "docs" / "examples" / "geometric" / "geometric-losses-demo.md": {
            "forbidden": (
                'loss_type="chamfer"',
                'loss_type="earth_mover"',
                "vertex_loss_weight=",
                "normal_loss_weight=",
                "edge_loss_weight=",
                "template_type",
            ),
            "required": (
                "chamfer_distance",
                "earth_mover_distance",
                "hausdorff_distance",
                "get_mesh_loss",
            ),
        },
        PROJECT_ROOT / "docs" / "examples" / "geometric" / "geometric-models-demo.md": {
            "forbidden": (
                'loss_type="chamfer"',
                '"loss_type": "emd"',
                'template_type="sphere"',
                '"template_type": "cube"',
                "vertex_loss_weight=",
                "normal_loss_weight=",
                "edge_loss_weight=",
                "- `loss_type`: Distance metric",
                "- `template_type`: Initial mesh template",
            ),
            "required": (
                "PointCloudConfig(",
                "MeshConfig(",
                "VoxelConfig(",
            ),
        },
    }

    for path, contract in per_file_contracts.items():
        content = path.read_text(encoding="utf-8")
        for term in contract["forbidden"]:
            assert term not in content, (
                f"{path} should not advertise stale geometric loss knob {term}"
            )
        for term in contract["required"]:
            assert term in content, f"{path} should document the retained geometric surface {term}"
