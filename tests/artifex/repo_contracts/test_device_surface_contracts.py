"""Contracts for the device surface artifex delegates to substrax."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


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


@pytest.mark.parametrize(
    "module",
    [
        "artifex.generative_models.core.device_manager",
        "artifex.generative_models.core.device_testing",
    ],
)
def test_the_duplicated_device_modules_are_gone(module: str) -> None:
    """Device identity and diagnostics have one home: substrax and scripts/gpu_utils.py."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_device_utils_keep_the_batch_size_rule_only() -> None:
    """The device utilities module ships one helper, built on substrax's table."""
    payload = _run_python(
        "import json; "
        "import artifex.generative_models.utils.jax.device as device_utils; "
        "print(json.dumps({"
        "'all': sorted(device_utils.__all__), "
        "'has_verify_device_setup': hasattr(device_utils, 'verify_device_setup'), "
        "'has_device_manager': hasattr(device_utils, 'DeviceManager'), "
        "'uses_substrax': device_utils.get_batch_size_recommendation.__module__"
        "}))"
    )

    assert payload["all"] == ["get_recommended_batch_size"]
    assert payload["has_verify_device_setup"] is False
    assert payload["has_device_manager"] is False
    assert payload["uses_substrax"] == "substrax.devices.placement"


def test_device_docs_describe_the_substrax_surface() -> None:
    """Device docs name substrax and no deleted artifex class or setup knob."""
    files_to_check = [
        "docs/getting-started/core-concepts.md",
        "docs/utils/device.md",
        "src/artifex/generative_models/core/README.md",
    ]
    banned_references = [
        "DeviceManager",
        "device_manager",
        "run_device_tests",
        "DeviceConfiguration",
        "MemoryStrategy",
        "DeviceTestRunner",
        "setup_device_for_training",
    ]

    for relative_path in files_to_check:
        contents = (REPO_ROOT / relative_path).read_text()
        assert "substrax" in contents, relative_path
        for banned_reference in banned_references:
            assert banned_reference not in contents, (relative_path, banned_reference)


def test_gpu_diagnostics_script_runs_the_critical_checks() -> None:
    """The developer diagnostics run from scripts/ on substrax's inventory."""
    result = subprocess.run(
        [sys.executable, "scripts/gpu_utils.py", "--test-critical"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={"JAX_PLATFORMS": "cpu", "PATH": "", "PYTHONPATH": ""},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Basic Computation" in result.stderr + result.stdout
    assert "Neural Network Operations" in result.stderr + result.stdout
