"""Repository contracts for the narrowed core-layer surface."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import cast


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


FLASH_RUNTIME = REPO_ROOT / "src/artifex/generative_models/core/layers/flash_attention.py"
BACKEND_RUNTIME = REPO_ROOT / "src/artifex/generative_models/core/layers/attention_backend.py"


def test_flash_attention_surface_drops_the_orphaned_triton_kernel() -> None:
    """The Triton kernel was defined but never called, so it must be gone."""
    payload = _run_python(
        textwrap.dedent(
            """
            import json

            import artifex.generative_models.core.layers.flash_attention as flash_attention_module

            print(json.dumps({
                name: hasattr(flash_attention_module, name)
                for name in (
                    'flash_dot_product_attention',
                    'flash_attention_triton',
                    'flash_attention_forward_kernel',
                    'TRITON_AVAILABLE',
                    'FlashAttentionConfig',
                    'AttentionMask',
                    'PADDING_SEGMENT_ID',
                )
            }))
            """
        )
    )

    assert payload["flash_dot_product_attention"] is True
    for removed in (
        "flash_attention_triton",
        "flash_attention_forward_kernel",
        "TRITON_AVAILABLE",
        "FlashAttentionConfig",
        "AttentionMask",
        "PADDING_SEGMENT_ID",
    ):
        assert payload[removed] is False, f"{removed} should have been removed"

    runtime = FLASH_RUNTIME.read_text(encoding="utf-8")
    assert "triton" not in runtime.lower()

    # The layers package must not export a symbol named after one of its own
    # submodules: doing so rebinds the package attribute and makes
    # `import ...layers.flash_attention as m` return the symbol, not the module.
    exported = _run_python(
        textwrap.dedent(
            """
            import json
            import pkgutil

            import artifex.generative_models.core.layers as layers

            submodules = {info.name for info in pkgutil.iter_modules(layers.__path__)}
            print(json.dumps(sorted(submodules & set(layers.__all__))))
            """
        )
    )
    assert exported == []

    flash_doc = _normalized_text(FLASH_DOC).lower()
    for banned in (
        "jax_native",
        "kvax optimizations",
        "significant performance improvements",
        "additional features",
        "triton-first path",
        "flash_attention_triton",
    ):
        assert banned not in flash_doc


def test_every_advertised_attention_backend_is_actually_dispatched() -> None:
    """Each backend the enum names must be reachable in the runtime, not just declared.

    The previous surface advertised backends that no code path could select. A
    backend now counts as real only when the dispatcher passes its value to
    ``jax.nn.dot_product_attention`` or delegates to the nnx kernel.
    """
    payload = _run_python(
        textwrap.dedent(
            """
            import json

            from artifex.generative_models.core.layers.attention_backend import AttentionBackend

            print(json.dumps({'members': [backend.value for backend in AttentionBackend]}))
            """
        )
    )

    assert set(cast(list[str], payload["members"])) == {"cudnn", "xla"}

    runtime = FLASH_RUNTIME.read_text(encoding="utf-8")
    # The fused backend must be requested explicitly, since implementation=None
    # silently falls back to XLA inside JAX.
    assert 'implementation="cudnn"' in runtime
    # The portable backend must delegate to the reference kernel rather than
    # reimplementing dropout, masking and sowing.
    assert "_nnx_attention(" in runtime

    backend_runtime = BACKEND_RUNTIME.read_text(encoding="utf-8")
    for constraint in ("CUDNN_SUPPORTED_DTYPES", "HEAD_DIM_MULTIPLE", "HEAD_DIM_MAX_HOPPER"):
        assert constraint in backend_runtime


def test_live_dropout_is_never_fused() -> None:
    """The fused kernel bakes its dropout seed into the compiled executable.

    ``jax/_src/cudnn/fused_attention_stablehlo.py`` marks ``seed`` a static
    argument and serialises it into the custom-call backend config, so a jitted
    training step would reuse one dropout mask forever. Selecting the fused
    kernel with live dropout would be silently wrong, not merely fast.
    """
    payload = _run_python(
        textwrap.dedent(
            """
            import json

            import jax.numpy as jnp

            from artifex.generative_models.core.layers.attention_backend import (
                AttentionBackend,
                select_attention_backend,
            )

            eligible = dict(
                dtype=jnp.bfloat16,
                head_dim=64,
                device_kind='gpu',
                compute_capability=(9, 0),
            )
            print(json.dumps({
                'inference': select_attention_backend(
                    **eligible, deterministic=True, dropout_rate=0.0, sow_weights=False
                ).value,
                'live_dropout': select_attention_backend(
                    **eligible, deterministic=False, dropout_rate=0.1, sow_weights=False
                ).value,
                'sow_weights': select_attention_backend(
                    **eligible, deterministic=True, dropout_rate=0.0, sow_weights=True
                ).value,
            }))
            """
        )
    )

    assert payload["inference"] == "cudnn"
    assert payload["live_dropout"] == "xla"
    assert payload["sow_weights"] == "xla"


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
