"""Repository contracts for the narrowed core-layer surface."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FLASH_DOC = REPO_ROOT / "docs/core/flash_attention.md"
RESIDUAL_DOC = REPO_ROOT / "docs/core/residual.md"
GRAPH_DOC = REPO_ROOT / "docs/models/graph.md"
CORE_LAYERS_INIT = REPO_ROOT / "src/artifex/generative_models/core/layers/__init__.py"
RESIDUAL_RUNTIME = REPO_ROOT / "src/artifex/generative_models/core/layers/residual.py"
PIXELCNN_RUNTIME = REPO_ROOT / "src/artifex/generative_models/models/autoregressive/pixel_cnn.py"


def _run_python(code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _normalized_text(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_flash_attention_surface_drops_fake_backend_switch() -> None:
    """Flash attention should expose one honest helper path with no backend selector."""
    payload = _run_python(
        textwrap.dedent(
            """
            import inspect
            import json

            import artifex.generative_models.core.layers.flash_attention as flash_attention_module

            print(json.dumps({
                'has_backend_enum': hasattr(flash_attention_module, 'AttentionBackend'),
                'has_flash_attention': hasattr(flash_attention_module, 'flash_attention'),
                'has_flash_attention_triton': hasattr(flash_attention_module, 'flash_attention_triton'),
                'init_signature': str(inspect.signature(flash_attention_module.FlashMultiHeadAttention.__init__)),
            }))
            """
        )
    )

    assert payload["has_backend_enum"] is False
    assert payload["has_flash_attention"] is True
    assert payload["has_flash_attention_triton"] is False
    assert "backend:" not in payload["init_signature"]

    flash_doc = _normalized_text(FLASH_DOC).lower()
    assert "attentionbackend" not in flash_doc
    assert "flash_attention" in flash_doc
    assert "jax fallback" in flash_doc
    for banned in (
        "flash_cudnn",
        "jax_native",
        "kvax optimizations",
        "significant performance improvements",
        "additional features",
        "triton-first path",
        "flash_attention_triton",
    ):
        assert banned not in flash_doc


def test_masked_pixelcnn_residual_surface_is_local_to_pixelcnn() -> None:
    """Placeholder masked residual blocks should not remain on the shared core-layer surface."""
    payload = _run_python(
        textwrap.dedent(
            """
            import importlib
            import json

            results = {}
            checks = (
                ('artifex.generative_models.core.layers.residual', 'MaskedConv2DResidualBlock'),
                ('artifex.generative_models.core.layers', 'MaskedConv2DResidualBlock'),
                ('artifex.generative_models.core.layers', 'PixelCNNResidualBlock'),
            )
            for module_name, attr_name in checks:
                module = importlib.import_module(module_name)
                results[f'{module_name}:{attr_name}'] = hasattr(module, attr_name)

            print(json.dumps(results))
            """
        )
    )

    for key, value in payload.items():
        assert value is False, key

    residual_doc = _normalized_text(RESIDUAL_DOC)
    residual_runtime = RESIDUAL_RUNTIME.read_text(encoding="utf-8")
    core_layers_init = CORE_LAYERS_INIT.read_text(encoding="utf-8")
    pixelcnn_runtime = PIXELCNN_RUNTIME.read_text(encoding="utf-8")

    for banned in ("MaskedConv2DResidualBlock", "PixelCNNResidualBlock", "masked_conv2d"):
        assert banned not in residual_doc
        assert banned not in residual_runtime
        assert banned not in core_layers_init

    assert "class PixelCNNResidualBlock" in pixelcnn_runtime
    assert (
        "from artifex.generative_models.core.layers import PixelCNNResidualBlock"
        not in pixelcnn_runtime
    )


def test_egnn_layer_contract_is_hidden_dim_only() -> None:
    """EGNNLayer should no longer advertise a separate node_dim constructor contract."""
    payload = _run_python(
        textwrap.dedent(
            """
            import inspect
            import json

            from artifex.generative_models.core.layers.egnn import EGNNLayer

            print(json.dumps({
                'init_params': list(inspect.signature(EGNNLayer.__init__).parameters),
            }))
            """
        )
    )

    assert "node_dim" not in payload["init_params"]
    assert "hidden_dim" in payload["init_params"]

    graph_doc = _normalized_text(GRAPH_DOC)
    assert "EGNNLayer" in graph_doc
    assert "node_dim" not in graph_doc
