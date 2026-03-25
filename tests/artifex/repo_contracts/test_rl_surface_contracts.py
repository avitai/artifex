"""Contracts for the RL training surface and public docs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_rl_docs_use_typed_rollout_batches_for_policy_gradient_trainers() -> None:
    """Policy-gradient RL docs should teach the typed rollout contracts."""
    per_file_contracts = {
        PROJECT_ROOT / "docs" / "training" / "reinforce.md": (
            "SequenceRolloutBatch",
            "GeneratedSequenceBatch",
            "GeneratedBatch",
        ),
        PROJECT_ROOT / "docs" / "training" / "ppo.md": (
            "SequenceRolloutBatch",
            "old_log_probs=",
            "advantages=",
            "returns=",
        ),
        PROJECT_ROOT / "docs" / "training" / "grpo.md": (
            "GroupRolloutBatch",
            "SequenceRolloutBatch",
            "group_size=",
            "sequence_rewards=",
        ),
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "rl-training.md": (
            "SequenceRolloutBatch",
            "GroupRolloutBatch",
            "GeneratedSequenceBatch",
            "PreferenceBatch",
        ),
    }

    forbidden_tokens = (
        '"states":',
        '"actions":',
        '"rewards":',
        '"old_log_probs":',
        '"group_size":',
        "Categorical Policy Contract",
        "categorical policy",
    )

    for path, required_tokens in per_file_contracts.items():
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"{path} still contains stale dict-batch RL text: {token}"
        for token in required_tokens:
            assert token in text, f"{path} should document the typed RL rollout contract: {token}"


def test_dpo_docs_use_typed_preference_batches_and_response_masks() -> None:
    """DPO docs should teach typed preference batches, not raw dict payloads."""
    docs_to_check = [
        PROJECT_ROOT / "docs" / "training" / "dpo.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "rl-training.md",
    ]

    forbidden_tokens = (
        '"chosen":',
        '"rejected":',
        '"chosen_log_probs":',
        '"rejected_log_probs":',
        '"ref_log_probs":',
        "DPOTrainer(model, optimizer, config)",
        "DPOTrainer(model, None, optimizer, config)",
    )
    required_tokens = (
        "PreferenceBatch",
        "GeneratedSequenceBatch",
        "response_mask",
        "chosen_loss_mask",
        "rejected_loss_mask",
    )

    for path in docs_to_check:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"{path} still contains stale DPO batch text: {token}"
        for token in required_tokens:
            assert token in text, (
                f"{path} should document the typed DPO preference contract: {token}"
            )


def test_grpo_docs_do_not_advertise_removed_gamma_or_batch_ref_log_probs() -> None:
    """GRPO docs should match the real config and reference-model contract."""
    docs_to_check = [
        PROJECT_ROOT / "docs" / "training" / "grpo.md",
        PROJECT_ROOT / "docs" / "user-guide" / "training" / "rl-training.md",
    ]

    forbidden_tokens = (
        "| `gamma` | `0.99` | Discount factor |",
        '"ref_log_probs":',
    )
    required_tokens = (
        "reference_model",
        "GroupRolloutBatch",
        "old_log_probs=",
    )

    for path in docs_to_check:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"{path} still contains stale GRPO contract text: {token}"
        for token in required_tokens:
            assert token in text, f"{path} should document the current GRPO contract: {token}"


def test_rl_docs_do_not_reference_nonexistent_fine_tuning_rl_package() -> None:
    """Fine-tuning docs should point to the public RL package that actually exists."""
    path = PROJECT_ROOT / "docs" / "fine_tuning" / "index.md"
    text = path.read_text(encoding="utf-8")

    forbidden_tokens = (
        "artifex.fine_tuning.rl",
        "RLHFTrainer",
    )
    required_tokens = (
        "artifex.generative_models.training",
        "DPOTrainer",
        "PPOTrainer",
        "GRPOTrainer",
        "REINFORCETrainer",
    )

    for token in forbidden_tokens:
        assert token not in text, (
            f"{path} still references a nonexistent RL package or trainer: {token}"
        )

    for token in required_tokens:
        assert token in text, f"{path} should point to the public RL training surface: {token}"


def test_rl_guide_describes_rl_trainers_as_standalone_helpers() -> None:
    """The RL guide should not claim RL trainers inherit from the shared Trainer."""
    path = PROJECT_ROOT / "docs" / "user-guide" / "training" / "rl-training.md"
    text = path.read_text(encoding="utf-8")

    forbidden_tokens = (
        "Trainer (base)",
        "Inherits core functionality",
        "on_train_batch_end",
    )
    required_tokens = (
        "standalone optimizer helpers",
        "CallbackList",
        "SequenceRolloutBatch",
    )

    for token in forbidden_tokens:
        assert token not in text, (
            f"{path} still overstates the RL trainer hierarchy/callback API: {token}"
        )

    for token in required_tokens:
        assert token in text, f"{path} should explain the real RL trainer shape: {token}"
