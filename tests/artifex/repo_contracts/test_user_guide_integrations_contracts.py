from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_python(code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_deployment_and_huggingface_guides_match_live_helper_owners() -> None:
    """Deployment/model-hub guides should teach only retained integration owners."""
    deployment_docs = (REPO_ROOT / "docs/user-guide/integrations/deployment.md").read_text(
        encoding="utf-8"
    )
    huggingface_docs = (REPO_ROOT / "docs/user-guide/integrations/huggingface.md").read_text(
        encoding="utf-8"
    )
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.models as models; "
        "import artifex.generative_models.models.vae as vae; "
        "import artifex.generative_models.utils.logging as logging_utils; "
        "print(json.dumps({"
        "'models_exports': sorted(models.__all__), "
        "'vae_exports': sorted(vae.__all__), "
        "'logging_exports': sorted(logging_utils.__all__)"
        "}))"
    )

    for banned in [
        "ModelConfig",
        "create_vae_model",
        "load_exported_model(",
        "@app.route",
        "class ModelUploader",
        "class ModelDownloader",
    ]:
        assert banned not in deployment_docs
        assert banned not in huggingface_docs

    for required in [
        "save_checkpoint",
        "load_checkpoint",
        "setup_checkpoint_manager",
        "ProductionOptimizer",
        "OptimizationTarget",
        "family-owned",
    ]:
        assert required in deployment_docs

    for required in [
        "Status: Coming soon",
        "does not currently ship built-in HuggingFace Hub upload/download helpers",
        "core.checkpointing",
        "deployment.md",
    ]:
        assert required in huggingface_docs

    assert "load_model" not in payload["models_exports"]
    assert "VAE" in payload["vae_exports"]
    assert "WandbLogger" in payload["logging_exports"]


def test_tensorboard_and_wandb_guides_use_real_callback_and_trainer_entrypoints() -> None:
    """Logging integration guides should use real callbacks and Trainer.train only."""
    tensorboard_docs = (REPO_ROOT / "docs/user-guide/integrations/tensorboard.md").read_text(
        encoding="utf-8"
    )
    wandb_docs = (REPO_ROOT / "docs/user-guide/integrations/wandb.md").read_text(encoding="utf-8")
    payload = _run_python(
        "import json; "
        "from artifex.generative_models.training.callbacks import ("
        "CallbackList, TensorBoardLoggerCallback, TensorBoardLoggerConfig, "
        "WandbLoggerCallback, WandbLoggerConfig, JAXProfiler, ProfilingConfig); "
        "from artifex.generative_models.training.trainer import Trainer; "
        "from artifex.generative_models.utils.logging import WandbLogger; "
        "print(json.dumps({"
        "'trainer_has_train': hasattr(Trainer, 'train'), "
        "'trainer_has_fit': hasattr(Trainer, 'fit'), "
        "'callback_list_cls': CallbackList.__name__, "
        "'tensorboard_callback_cls': TensorBoardLoggerCallback.__name__, "
        "'tensorboard_config_cls': TensorBoardLoggerConfig.__name__, "
        "'wandb_callback_cls': WandbLoggerCallback.__name__, "
        "'wandb_config_cls': WandbLoggerConfig.__name__, "
        "'profiler_cls': JAXProfiler.__name__, "
        "'profiling_config_cls': ProfilingConfig.__name__, "
        "'wandb_logger_cls': WandbLogger.__name__"
        "}))"
    )

    for banned in [
        "trainer.fit(",
        "class TensorBoardLogger",
        "class ImageLogger",
        "class HistogramLogger",
        "class WandBTrainer",
        "class AdvancedWandBLogger",
        "on_train_step_end",
    ]:
        assert banned not in tensorboard_docs
        assert banned not in wandb_docs

    for required in [
        "TensorBoardLoggerCallback",
        "TensorBoardLoggerConfig",
        "CallbackList",
        "trainer.train(",
        "JAXProfiler",
        "ProfilingConfig",
    ]:
        assert required in tensorboard_docs

    for required in [
        "WandbLoggerCallback",
        "WandbLoggerConfig",
        "CallbackList",
        "trainer.train(",
        "WandbLogger",
    ]:
        assert required in wandb_docs

    assert payload["trainer_has_train"] is True
    assert payload["trainer_has_fit"] is False
    assert payload["callback_list_cls"] == "CallbackList"
    assert payload["tensorboard_callback_cls"] == "TensorBoardLoggerCallback"
    assert payload["tensorboard_config_cls"] == "TensorBoardLoggerConfig"
    assert payload["wandb_callback_cls"] == "WandbLoggerCallback"
    assert payload["wandb_config_cls"] == "WandbLoggerConfig"
    assert payload["profiler_cls"] == "JAXProfiler"
    assert payload["profiling_config_cls"] == "ProfilingConfig"
    assert payload["wandb_logger_cls"] == "WandbLogger"
