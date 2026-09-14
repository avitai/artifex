"""The JAX runtime a test run starts with: the backend and the emulated CPU devices.

Tests run on the CPU with eight emulated devices unless ``ARTIFEX_TEST_JAX_PLATFORMS`` names an
accelerator, and an inherited ``JAX_PLATFORMS`` does not move them onto a GPU, so a local run
matches CI. ``substrax.runtime.resolve_test_runtime`` decides; these tests pin artifex's choices
and check, through a pytest session in a fresh interpreter, that ``tests/conftest.py`` applies
them before jax is imported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from substrax.testing import run_python
from tests.jax_test_environment import resolve_test_environment


REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE = Path(__file__).with_name("jax_runtime_probe.py")
PROBE_OUTPUT_ENV = "ARTIFEX_RUNTIME_PROBE_OUTPUT"
SESSION_TIMEOUT = 180.0


def test_tests_default_to_eight_emulated_cpu_devices() -> None:
    env = resolve_test_environment({}, cuda_plugin_available=True)

    assert env == {"JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "8"}


def test_an_inherited_cuda_selection_still_runs_tests_on_emulated_cpus() -> None:
    env = resolve_test_environment({"JAX_PLATFORMS": "cuda,cpu"}, cuda_plugin_available=True)

    assert env == {"JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "8"}


def test_an_explicit_accelerator_request_is_used_without_emulation() -> None:
    env = resolve_test_environment(
        {"ARTIFEX_TEST_JAX_PLATFORMS": "cuda"}, cuda_plugin_available=True
    )

    assert env == {"JAX_PLATFORMS": "cuda"}


def _session_runtime(tmp_path: Path, env: dict[str, str]) -> list[object]:
    """Run the probe under the repository's conftest; return its backend and device count."""
    output = tmp_path / "runtime.json"
    result = run_python(
        "import sys, pytest; sys.exit(pytest.main(sys.argv[1:]))",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-q",
        str(PROBE),
        timeout=SESSION_TIMEOUT,
        env={**env, PROBE_OUTPUT_ENV: str(output)},
        cwd=REPO_ROOT,
    )
    result.check()
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("requested", "devices"), [("1", 1), ("2", 2), ("8", 8)])
def test_a_pytest_session_sees_the_cpu_devices_it_asks_for(
    tmp_path: Path, requested: str, devices: int
) -> None:
    env = {"ARTIFEX_TEST_JAX_PLATFORMS": "cpu", "ARTIFEX_TEST_DEVICE_COUNT": requested}

    assert _session_runtime(tmp_path, env) == ["cpu", devices]


def test_the_substrax_pytest_plugin_is_enabled(pytestconfig: pytest.Config) -> None:
    """It fails a test that changes global jax configuration and adds the device markers."""
    assert pytestconfig.pluginmanager.hasplugin("substrax.testing.pytest_plugin")
