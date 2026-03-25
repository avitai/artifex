from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def normalized_text(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_core_runtime_docs_publish_explicit_estimation_and_sampling_contracts() -> None:
    performance = normalized_text(REPO_ROOT / "docs/core/performance.md")
    diffusion = normalized_text(REPO_ROOT / "docs/core/diffusion.md")
    trainer = normalized_text(REPO_ROOT / "docs/training/diffusion_trainer.md")

    performance_fragments = [
        "heuristic estimates rather than measured hardware facts",
        "`memory_source`",
        "`peak_flops_source`",
        "`memory_bandwidth_source`",
        (
            "`analyze_roofline(...)` requires explicit `peak_flops_per_second` and "
            "`memory_bandwidth_gb_per_second` values"
        ),
    ]
    for fragment in performance_fragments:
        assert fragment in performance

    diffusion_fragments = [
        "wrapper-only",
        "delegates to `model.sample(...)`",
        "does not implement a standalone generic direct-sampling path",
    ]
    for fragment in diffusion_fragments:
        assert fragment in diffusion

    trainer_fragments = [
        "`t` must have shape `(batch,)` or `(1,)`",
        "must match the data batch size or contain one element",
    ]
    for fragment in trainer_fragments:
        assert fragment in trainer


def test_blackjax_docs_and_examples_only_publish_live_wrapper_controls() -> None:
    explicit_rng_docs = [
        REPO_ROOT / "docs/core/blackjax_samplers.md",
        REPO_ROOT / "docs/examples/sampling/blackjax-example.md",
        REPO_ROOT / "docs/examples/sampling/blackjax-integration-examples.md",
        REPO_ROOT / "docs/examples/sampling/blackjax-sampling-examples.md",
    ]
    for path in explicit_rng_docs:
        contents = normalized_text(path)
        assert "explicit JAX key or `nnx.Rngs`" in contents, (
            f"{path} must require explicit RNG ownership"
        )

    dead_wrapper_control_paths = [
        REPO_ROOT / "docs/core/blackjax_samplers.md",
        REPO_ROOT / "docs/examples/sampling/blackjax-example.md",
        REPO_ROOT / "docs/examples/sampling/blackjax-integration-examples.md",
        REPO_ROOT / "docs/examples/sampling/blackjax-sampling-examples.md",
        REPO_ROOT / "examples/generative_models/sampling/blackjax_sampling_examples.py",
        REPO_ROOT / "examples/generative_models/sampling/test_blackjax_integration.py",
    ]
    for path in dead_wrapper_control_paths:
        contents = path.read_text(encoding="utf-8")
        assert "max_num_doublings" not in contents, (
            f"{path} still publishes dead NUTS wrapper controls"
        )
        assert "max_depth" not in contents, f"{path} still publishes dead NUTS wrapper controls"
