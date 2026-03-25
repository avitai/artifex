from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


BENCHMARK_DOCS = [
    REPO_ROOT / "docs" / "examples" / "geometric" / "geometric-benchmark-demo.md",
    REPO_ROOT / "docs" / "examples" / "protein" / "protein-ligand-benchmark-demo.md",
    REPO_ROOT / "docs" / "examples" / "vae" / "multi-beta-vae-benchmark-demo.md",
]


def test_benchmark_docs_are_explicitly_demo_only() -> None:
    for path in BENCHMARK_DOCS:
        contents = path.read_text(encoding="utf-8")
        assert "**Status:** `Demo-only benchmark walkthrough`" in contents
        assert "demo_mode=True" in contents or 'data_source="synthetic"' in contents
        assert "complete benchmark suite" not in contents.lower()
        assert "automatic fallbacks" not in contents.lower()


def test_benchmark_index_describes_narrow_public_surface() -> None:
    contents = (REPO_ROOT / "docs" / "benchmarks" / "index.md").read_text(encoding="utf-8")

    assert "There is no supported public benchmark CLI runner" in contents
    assert "demo_mode=True" in contents
    assert "complete benchmarking framework" not in contents
    assert "Run benchmarks on your models with just a few lines of code" not in contents


def test_benchmark_cli_modules_are_status_helpers_only() -> None:
    benchmark_cli = (
        REPO_ROOT / "src" / "artifex" / "benchmarks" / "cli" / "benchmark_runner.py"
    ).read_text(encoding="utf-8")
    optimization_cli = (
        REPO_ROOT / "src" / "artifex" / "benchmarks" / "cli" / "optimization_benchmark.py"
    ).read_text(encoding="utf-8")

    assert "Retained benchmark CLI status helper" in benchmark_cli
    assert "There is no supported public benchmark CLI runner" in benchmark_cli
    assert "DummyModel" not in benchmark_cli
    assert "simulated" not in benchmark_cli.lower()

    assert "deprecated optimization benchmark CLI" in optimization_cli
    assert "There is no supported public optimization benchmark CLI" in optimization_cli


def test_benchmark_examples_opt_into_demo_mode_in_code() -> None:
    geometric = (
        REPO_ROOT / "examples" / "generative_models" / "geometric" / "geometric_benchmark_demo.py"
    ).read_text(encoding="utf-8")
    protein = (
        REPO_ROOT
        / "examples"
        / "generative_models"
        / "protein"
        / "protein_ligand_benchmark_demo.py"
    ).read_text(encoding="utf-8")
    vae = (
        REPO_ROOT / "examples" / "generative_models" / "vae" / "multi_beta_vae_benchmark_demo.py"
    ).read_text(encoding="utf-8")

    assert '"data_source": "synthetic"' in geometric
    assert '"demo_mode": True' in geometric
    assert '"demo_mode": True' in protein
    assert "demo_mode=True" in protein
    assert "demo_mode=True" in vae
