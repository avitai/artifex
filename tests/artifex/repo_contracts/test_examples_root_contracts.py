from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "examples" / "README.md"
GUIDE_PATH = REPO_ROOT / "examples" / "EXAMPLES_GUIDE.md"
RUN_ALL_PATH = REPO_ROOT / "examples" / "run_all_examples.sh"
VERIFY_PATH = REPO_ROOT / "examples" / "verify_examples.py"
COMMAND_PATTERN = re.compile(r"uv run python (?P<path>examples/[\w./-]+\.py)")
BARE_PYTHON_PATTERN = re.compile(r"(?m)^python examples/[\w./-]+\.py$")

BROKEN_OR_NONCANONICAL_PATHS = {
    "examples/generative_models/audio/simple_audio_generation.py",
    "examples/generative_models/diffusion/simple_diffusion_example.py",
    "examples/generative_models/image/diffusion/diffusion_mnist_training.py",
    "examples/generative_models/image/gan/advanced_gan.py",
    "examples/generative_models/multimodal/simple_image_text.py",
    "examples/generative_models/protein/protein_diffusion_example.py",
    "examples/generative_models/protein/protein_diffusion_tech_validation.py",
    "examples/generative_models/text/simple_text_generation.py",
    "examples/utils/demo_utils.py",
    "examples/verify_examples.py",
}


def _extract_uv_commands(contents: str) -> set[str]:
    return set(COMMAND_PATTERN.findall(contents))


def test_root_example_docs_use_uv_and_avoid_backend_forcing() -> None:
    """Contributor-facing root example docs should follow the checked-in design guide."""
    for path in (README_PATH, GUIDE_PATH):
        contents = path.read_text(encoding="utf-8")
        assert "source ./activate.sh" in contents
        assert "uv run python" in contents
        assert not BARE_PYTHON_PATTERN.search(contents), (
            f"{path} still documents direct python execution"
        )
        assert "export JAX_PLATFORMS" not in contents
        assert "unset JAX_PLATFORMS" not in contents


def test_root_example_doc_commands_resolve_to_real_files() -> None:
    """Every documented root example command should point at a real checked-in example."""
    commands = _extract_uv_commands(README_PATH.read_text(encoding="utf-8"))
    commands.update(_extract_uv_commands(GUIDE_PATH.read_text(encoding="utf-8")))

    assert commands

    for relative_path in commands:
        assert (REPO_ROOT / relative_path).exists(), (
            f"Documented example path is missing: {relative_path}"
        )


def test_run_all_helper_has_curated_scope() -> None:
    """The run-all helper should target an explicit reviewed subset, not auto-discover the whole tree."""
    contents = RUN_ALL_PATH.read_text(encoding="utf-8")
    documented_paths = set(re.findall(r"examples/[\w./-]+\.py", contents))

    assert "uv run python" in contents
    assert "find examples" not in contents
    assert documented_paths

    for relative_path in documented_paths:
        assert (REPO_ROOT / relative_path).exists(), (
            f"run_all_examples.sh references missing file: {relative_path}"
        )

    for banned_path in BROKEN_OR_NONCANONICAL_PATHS:
        assert banned_path not in contents


def test_examples_verifier_derives_scope_from_live_readme() -> None:
    """The retained verifier should read the live README and avoid backend forcing or overclaiming."""
    contents = VERIFY_PATH.read_text(encoding="utf-8")

    assert "README.md" in contents
    assert "read_text" in contents
    assert 'os.environ["JAX_PLATFORMS"]' not in contents
    assert 'jax.config.update("jax_platforms"' not in contents
    assert "All README examples are working correctly!" not in contents
    assert "Complete verification script for README examples." not in contents
