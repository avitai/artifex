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


def test_device_manager_module_import_is_side_effect_free() -> None:
    """Importing the module should not import JAX or mutate backend env vars."""
    payload = _run_python(
        "import json, os, sys; "
        "os.environ.pop('JAX_PLATFORMS', None); "
        "os.environ.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None); "
        "import artifex.generative_models.core.device_manager as dm; "
        "print(json.dumps({"
        "'jax_loaded': 'jax' in sys.modules, "
        "'jax_platforms': os.environ.get('JAX_PLATFORMS'), "
        "'memory_fraction': os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'), "
        "'module': dm.__name__"
        "}))"
    )

    assert payload["module"] == "artifex.generative_models.core.device_manager"
    assert payload["jax_loaded"] is False
    assert payload["jax_platforms"] is None
    assert payload["memory_fraction"] is None


def test_device_testing_module_import_is_side_effect_free() -> None:
    """Importing device diagnostics should not import JAX eagerly."""
    payload = _run_python(
        "import json, sys; "
        "import artifex.generative_models.core.device_testing as dt; "
        "print(json.dumps({"
        "'jax_loaded': 'jax' in sys.modules, "
        "'flax_loaded': 'flax' in sys.modules, "
        "'module': dt.__name__, "
        "'has_runner': hasattr(dt, 'DeviceTestRunner')"
        "}))"
    )

    assert payload["module"] == "artifex.generative_models.core.device_testing"
    assert payload["jax_loaded"] is False
    assert payload["flax_loaded"] is False
    assert payload["has_runner"] is False


def test_device_docs_use_runtime_manager_surface() -> None:
    """Device docs should describe runtime introspection, not backend setup knobs."""
    files_to_check = [
        "docs/api/core/device-manager.md",
        "docs/core/device_manager.md",
        "docs/getting-started/core-concepts.md",
        "docs/utils/device.md",
        "src/artifex/generative_models/core/README.md",
    ]
    banned_references = [
        "DeviceConfiguration",
        "MemoryStrategy",
        "configure_for_generative_models",
        "DeviceTestRunner",
        "force_reinit",
        "platform_priority",
        "environment_variables",
        "optimize_for_model_size",
        "setup_device_for_training",
    ]

    for relative_path in files_to_check:
        contents = (REPO_ROOT / relative_path).read_text()
        for banned_reference in banned_references:
            assert banned_reference not in contents

    api_doc = (REPO_ROOT / "docs/api/core/device-manager.md").read_text()
    assert "DeviceManager" in api_doc
    assert "verify_gpu_setup.py" in api_doc
    assert "get_default_device" in api_doc


def test_device_utils_surface_keeps_runtime_helpers_only() -> None:
    """Device utils should not re-export removed setup/config abstractions."""
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.utils.jax.device as device_utils; "
        "print(json.dumps({"
        "'has_setup_device_for_training': hasattr(device_utils, 'setup_device_for_training'), "
        "'has_device_configuration': hasattr(device_utils, 'DeviceConfiguration'), "
        "'has_memory_strategy': hasattr(device_utils, 'MemoryStrategy'), "
        "'has_device_test_runner': hasattr(device_utils, 'DeviceTestRunner'), "
        "'has_configure_for_generative_models': "
        "hasattr(device_utils, 'configure_for_generative_models'), "
        "'has_verify_device_setup': hasattr(device_utils, 'verify_device_setup'), "
        "'has_recommended_batch_size': hasattr(device_utils, 'get_recommended_batch_size')"
        "}))"
    )

    assert payload["has_setup_device_for_training"] is False
    assert payload["has_device_configuration"] is False
    assert payload["has_memory_strategy"] is False
    assert payload["has_device_test_runner"] is False
    assert payload["has_configure_for_generative_models"] is False
    assert payload["has_verify_device_setup"] is True
    assert payload["has_recommended_batch_size"] is True
