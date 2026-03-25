from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CURRENT_TFDS_IMPORT = "from datarax.sources import from_tfds"
BANNED_TFDS_REFERENCES = [
    "TfdsDataSourceConfig",
    "TFDSSource",
    "from datarax.sources import TfdsDataSourceConfig, TFDSSource",
]


def test_mnist_training_examples_use_live_datarax_tfds_helper() -> None:
    """The retained MNIST training examples should use the current DataRax TFDS API."""
    paths = [
        REPO_ROOT / "examples/generative_models/image/diffusion/diffusion_mnist_training.py",
        REPO_ROOT / "examples/generative_models/image/flow/flow_mnist.py",
    ]

    for path in paths:
        contents = path.read_text(encoding="utf-8")
        assert CURRENT_TFDS_IMPORT in contents, f"{path} is missing the live DataRax TFDS helper"
        assert "from_tfds(" in contents, (
            f"{path} no longer constructs its TFDS source through the live helper"
        )
        for banned_reference in BANNED_TFDS_REFERENCES:
            assert banned_reference not in contents, f"{path} still references {banned_reference!r}"


def test_mnist_training_docs_and_landing_page_use_live_datarax_tfds_helper() -> None:
    """The retained MNIST tutorial pages should teach the same current TFDS API."""
    paths = [
        REPO_ROOT / "docs/examples/basic/diffusion-mnist.md",
        REPO_ROOT / "docs/examples/basic/flow-mnist.md",
    ]

    for path in paths:
        contents = path.read_text(encoding="utf-8")
        assert CURRENT_TFDS_IMPORT in contents, f"{path} is missing the live DataRax TFDS helper"
        assert "from_tfds(" in contents, f"{path} no longer teaches the live TFDS helper call"
        for banned_reference in BANNED_TFDS_REFERENCES:
            assert banned_reference not in contents, f"{path} still references {banned_reference!r}"


def test_diffusion_mnist_docs_point_to_the_correct_example_pairs() -> None:
    """The diffusion demo page and the training page should link to their actual paired sources."""
    expected_links = {
        REPO_ROOT
        / "docs/examples/basic/diffusion-mnist-demo.md": "examples/generative_models/image/diffusion/diffusion_mnist",
        REPO_ROOT
        / "docs/examples/basic/diffusion-mnist.md": "examples/generative_models/image/diffusion/diffusion_mnist_training",
        REPO_ROOT
        / "docs/examples/basic/flow-mnist.md": "examples/generative_models/image/flow/flow_mnist",
    }

    for path, stem in expected_links.items():
        contents = path.read_text(encoding="utf-8")
        assert f"{stem}.py" in contents, f"{path} should link to {stem}.py"
        assert f"{stem}.ipynb" in contents, f"{path} should link to {stem}.ipynb"
