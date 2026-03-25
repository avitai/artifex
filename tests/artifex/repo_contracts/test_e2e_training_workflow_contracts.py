from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_TESTS = (
    REPO_ROOT / "tests/artifex/integration/e2e/test_training_workflows.py"
).read_text()
E2E_CONFTEST = (REPO_ROOT / "tests/artifex/integration/e2e/conftest.py").read_text()


def test_e2e_training_workflows_use_live_public_gan_and_flow_owners() -> None:
    """Retained e2e workflows should import only live public GAN and flow owners."""
    required_fragments = [
        "DCGAN",
        "DCGANConfig",
        "ConvGeneratorConfig",
        "ConvDiscriminatorConfig",
        "RealNVP",
        "RealNVPConfig",
        "CouplingNetworkConfig",
    ]
    for fragment in required_fragments:
        assert fragment in WORKFLOW_TESTS

    banned_fragments = [
        "GANModel",
        "FlowModel",
        "except ImportError",
        "pytest.skip(",
        "generator_hidden_dims",
        "discriminator_hidden_dims",
        "num_flows",
    ]
    for fragment in banned_fragments:
        assert fragment not in WORKFLOW_TESTS


def test_e2e_fixture_layer_does_not_mutate_backend_selection_state() -> None:
    """Shared e2e fixtures should not rewrite JAX backend-selection environment state."""
    assert "JAX_PLATFORMS" not in E2E_CONFTEST
    assert "os.environ" not in E2E_CONFTEST
