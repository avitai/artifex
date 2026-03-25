from __future__ import annotations

import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "fine_tuning"
CURRENT_RUNTIME_TRAINERS = {
    "REINFORCETrainer": "../training/reinforce.md",
    "PPOTrainer": "../training/ppo.md",
    "GRPOTrainer": "../training/grpo.md",
    "DPOTrainer": "../training/dpo.md",
}
COMING_SOON_TOPICS = (
    "LoRA",
    "Prefix Tuning",
    "Prompt Tuning",
    "Distillation",
    "Few-Shot Learning",
    "Transfer Learning",
    "RLHF",
)


def test_fine_tuning_catalog_keeps_only_truthful_overview_page() -> None:
    """The fine-tuning docs should collapse to one truthful overview page."""
    actual_pages = {path.name for path in DOCS_ROOT.glob("*.md")}
    assert actual_pages == {"index.md"}

    contents = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")

    assert "**Status:** `Current runtime fine-tuning surface`" in contents
    assert "There is no standalone `artifex.fine_tuning` package" in contents
    assert "artifex.generative_models.training" in contents
    assert "not shipped yet" in contents
    assert "../roadmap/planned-modules.md" in contents

    for trainer_name, link_target in CURRENT_RUNTIME_TRAINERS.items():
        assert trainer_name in contents
        assert link_target in contents

    for topic in COMING_SOON_TOPICS:
        assert topic in contents

    for banned in [
        "from artifex.fine_tuning",
        "artifex.fine_tuning.adapters",
        "artifex.generative_models.fine_tuning",
        "LoRAAdapter",
        "PrefixTuning",
        "PromptTuning",
        "DistillationTrainer",
        "FewShotTrainer",
        "TransferTrainer",
        "## Quick Start",
    ]:
        assert banned not in contents


def test_fine_tuning_overview_points_to_live_training_exports() -> None:
    """The overview should reference only exported RL trainers from training."""
    training = importlib.import_module("artifex.generative_models.training")
    exported = set(getattr(training, "__all__", []))

    for trainer_name in CURRENT_RUNTIME_TRAINERS:
        assert trainer_name in exported
        assert hasattr(training, trainer_name)


def test_fine_tuning_nav_and_architecture_drop_the_phantom_package() -> None:
    """Navigation and architecture docs should not publish a fine_tuning package."""
    mkdocs_contents = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    architecture_contents = (REPO_ROOT / "docs/core/architecture.md").read_text(encoding="utf-8")

    assert "      - Fine-Tuning: fine_tuning/index.md" in mkdocs_contents
    assert "      - Fine-Tuning Reference:\n" not in mkdocs_contents
    assert "          - fine_tuning/" not in mkdocs_contents

    assert "fine_tuning/" not in architecture_contents
    assert "RL fine-tuning helpers" in architecture_contents
    assert "training/rl" in architecture_contents
