import ast
import importlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.utils.fresh_interpreter import run_repo_json, run_repo_python


REPO_ROOT = Path(__file__).resolve().parents[3]


def run_repo_shell(command: str) -> subprocess.CompletedProcess[str]:
    """Run a shell command inside the repository root."""
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_cuda12_managed_env(managed_env: Path) -> None:
    """Write the managed env file setup.sh generates for the cuda12 backend."""
    setup_env = REPO_ROOT / "scripts" / "setup_env.py"
    run_repo_python(setup_env, "write", "--backend", "cuda12", "--output", str(managed_env)).check()


def test_pyproject_uses_explicit_cuda12_extra():
    """The package should expose an explicit CUDA extra instead of legacy GPU shims."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "cuda12" in extras
    assert "cuda-dev" in extras
    assert "gpu" not in extras
    assert any("jax[cuda12]" in requirement for requirement in extras["cuda12"])
    assert any("nvidia-cudnn-cu12==" in requirement for requirement in extras["cuda12"])
    assert any("nvidia-cuda-runtime-cu12==" in requirement for requirement in extras["cuda12"])
    assert "tool.artifex.cuda" not in (REPO_ROOT / "pyproject.toml").read_text()


def test_backend_bootstrap_does_not_hardcode_system_cuda() -> None:
    """Bootstrap files should not depend on system CUDA paths or hardcoded fallback lists."""
    files_to_check = [
        ".env.example",
        "activate.sh",
        "conftest.py",
        "scripts/setup_env.py",
        "scripts/verify_gpu_setup.py",
        "setup.sh",
        "tests/conftest.py",
    ]

    banned_strings = [
        "/usr/local/cuda",
        "cuda,cpu",
        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK",
    ]

    for relative_path in files_to_check:
        contents = (REPO_ROOT / relative_path).read_text()
        for banned in banned_strings:
            assert banned not in contents

    pyproject_contents = (REPO_ROOT / "pyproject.toml").read_text()
    assert "LD_LIBRARY_PATH =" not in pyproject_contents
    assert ".artifex.env" in (REPO_ROOT / "setup.sh").read_text()
    assert ".artifex.env" in (REPO_ROOT / "activate.sh").read_text()
    assert ".artifex.env" in (REPO_ROOT / ".gitignore").read_text()


def test_user_facing_docs_reference_explicit_cuda12_install() -> None:
    """Top-level docs should direct users to the explicit CUDA extra."""
    files_to_check = [
        "README.md",
        "TESTING.md",
        "docs/getting-started/installation.md",
        "docs/index.md",
        "docs/user-guide/inference/overview.md",
        "docs/examples/diffusion/simple-diffusion.md",
        "docs/examples/basic/flow-mnist.md",
        "docs/examples/advanced/advanced-gan.md",
    ]

    for relative_path in files_to_check:
        contents = (REPO_ROOT / relative_path).read_text()
        assert "artifex[cuda]" not in contents

    installation_guide = (REPO_ROOT / "docs/getting-started/installation.md").read_text()
    assert '"avitai-artifex[cuda12]"' in installation_guide
    assert '"artifex[cuda12]"' not in installation_guide


def test_activate_clears_stale_managed_backend_variables(tmp_path: Path) -> None:
    """Re-sourcing activate.sh should not keep stale managed backend variables."""
    managed_env = tmp_path / ".artifex.env"
    _write_cuda12_managed_env(managed_env)

    result = subprocess.run(
        [
            "bash",
            "-lc",
            (
                "export JAX_PLATFORMS='cuda,cpu'; "
                f"export ARTIFEX_MANAGED_ENV_FILE='{managed_env}'; "
                "source ./activate.sh >/tmp/artifex-activate-contract.log 2>&1; "
                f"'{sys.executable}' - <<'PY'\n"
                "import json, os\n"
                "print(json.dumps({\n"
                "    'artifex_backend': os.environ.get('ARTIFEX_BACKEND'),\n"
                "    'jax_platforms': os.environ.get('JAX_PLATFORMS'),\n"
                "    'env_root': os.environ.get('ARTIFEX_ENV_ROOT'),\n"
                "}))\n"
                "PY"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["artifex_backend"] == "cuda12"
    assert payload["jax_platforms"] is None
    assert payload["env_root"] == str(REPO_ROOT)


def test_activate_strips_inherited_cuda_library_paths_for_cuda12_backend(
    tmp_path: Path,
) -> None:
    """CUDA12 activation should drop stale toolkit library paths while keeping unrelated ones."""
    managed_env = tmp_path / ".artifex.env"
    _write_cuda12_managed_env(managed_env)

    result = subprocess.run(
        [
            "bash",
            "-lc",
            (
                "export LD_LIBRARY_PATH='/tmp/keep:/opt/toolchains/cuda-12.4/lib64:"
                "/tmp/also-keep:/opt/toolchains/cuda-12.4/targets/x86_64-linux/lib'; "
                f"export ARTIFEX_MANAGED_ENV_FILE='{managed_env}'; "
                "source ./activate.sh >/tmp/artifex-activate-contract.log 2>&1; "
                f"'{sys.executable}' - <<'PY'\n"
                "import json, os\n"
                "print(json.dumps({'ld_library_path': os.environ.get('LD_LIBRARY_PATH')}))\n"
                "PY"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ld_library_path"] == "/tmp/keep:/tmp/also-keep"


def test_setup_env_writes_the_memory_fraction_name_jaxlib_reads(tmp_path: Path) -> None:
    """The managed env file sets XLA_CLIENT_MEM_FRACTION, never the deprecated name jax refuses."""
    managed_env = tmp_path / ".artifex.env"
    _write_cuda12_managed_env(managed_env)

    contents = managed_env.read_text()
    assert "export XLA_CLIENT_MEM_FRACTION=0.75" in contents
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in contents


def test_reactivating_a_shell_drops_the_deprecated_memory_fraction(tmp_path: Path) -> None:
    """A shell activated before the rename keeps no XLA_PYTHON_CLIENT_MEM_FRACTION after activate.

    jax refuses a process that sets it beside XLA_CLIENT_MEM_FRACTION. activate.sh unsets the
    names the previous managed file listed, which include the deprecated one, before sourcing
    the new file.
    """
    managed_env = tmp_path / ".artifex.env"
    _write_cuda12_managed_env(managed_env)

    result = subprocess.run(
        [
            "bash",
            "-lc",
            (
                "export ARTIFEX_MANAGED_ENV_VARS='XLA_PYTHON_CLIENT_MEM_FRACTION'; "
                "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75; "
                f"export ARTIFEX_MANAGED_ENV_FILE='{managed_env}'; "
                "source ./activate.sh >/tmp/artifex-activate-contract.log 2>&1; "
                f"'{sys.executable}' - <<'PY'\n"
                "import json, os\n"
                "print(json.dumps({\n"
                "    'deprecated': os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),\n"
                "    'current': os.environ.get('XLA_CLIENT_MEM_FRACTION'),\n"
                "}))\n"
                "PY"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload == {"deprecated": None, "current": "0.75"}


def test_env_example_is_comment_only_user_override_template() -> None:
    """The checked-in env example should not behave like an authoritative policy file."""
    env_example = (REPO_ROOT / ".env.example").read_text().splitlines()
    active_lines = [
        line for line in env_example if line.strip() and not line.lstrip().startswith("#")
    ]

    assert active_lines == []
    assert any(".artifex.env" in line for line in env_example)
    assert any("Copy this file to `.env`" in line for line in env_example)
    assert "PYTEST_CUDA_ENABLED" not in "\n".join(env_example)
    assert "LD_LIBRARY_PATH" not in "\n".join(env_example)


def test_setup_dry_run_routes_env_generation_through_setup_env_owner() -> None:
    """The shell bootstrap should delegate generated env contents to setup_env.py."""
    result = run_repo_shell("./setup.sh --dry-run")

    assert "python3 scripts/setup_env.py write --backend " in result.stdout
    assert ".artifex.env" in result.stdout
    assert ".env.example" not in result.stdout
    assert "verify_gpu_setup.py" not in result.stdout
    assert "LD_LIBRARY_PATH" not in result.stdout


def test_root_conftest_import_does_not_import_jax_or_mutate_environment() -> None:
    """Importing the root pytest conftest should stay lightweight and side-effect free."""
    payload = run_repo_json(
        """
import json
import os
import sys

keys = [
    "TF_CPP_MIN_LOG_LEVEL",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
]
for key in keys:
    os.environ.pop(key, None)
before = {key: os.environ.get(key) for key in keys}
import conftest  # noqa: F401
after = {key: os.environ.get(key) for key in keys}
print(json.dumps({"before": before, "after": after, "jax_loaded": "jax" in sys.modules}))
"""
    )

    assert payload["before"] == payload["after"]
    assert payload["jax_loaded"] is False


def test_tests_conftest_import_keeps_jax_deferred() -> None:
    """Importing the shared tests conftest should not eagerly initialize the JAX stack."""
    payload = run_repo_json(
        """
import json
import sys
import tests.conftest  # noqa: F401
print(json.dumps({"jax_loaded": "jax" in sys.modules}))
"""
    )

    assert payload["jax_loaded"] is False


def test_pytest_header_defers_runtime_probe_without_opt_in(monkeypatch) -> None:
    """Pytest startup should not probe the live JAX runtime unless explicitly requested."""
    pytest_conftest = importlib.import_module("tests.conftest")
    devices = importlib.import_module("substrax.devices")

    info = devices.DeviceInfo(
        platform="gpu",
        kind=devices.DeviceKind.GPU,
        count=2,
        device_kinds=("NVIDIA RTX 4090", "NVIDIA RTX 4090"),
    )

    monkeypatch.setenv("ARTIFEX_BACKEND", "cuda12")
    monkeypatch.setattr(
        devices,
        "detect_devices",
        lambda: (_ for _ in ()).throw(AssertionError("runtime probe should stay deferred")),
    )

    config = SimpleNamespace(
        _metadata={},
        getoption=lambda name: False,
    )

    header = pytest_conftest.pytest_report_header(config)

    assert "Artifex backend: cuda12" in header
    assert any(line.startswith("JAX runtime probe: deferred") for line in header)
    assert config._metadata["Artifex backend"] == "cuda12"
    assert config._metadata["JAX runtime probe"] == "deferred"

    monkeypatch.setattr(devices, "detect_devices", lambda: info)
    opt_in_config = SimpleNamespace(
        _metadata={},
        getoption=lambda name: name == "--artifex-probe-jax-runtime",
    )

    probed_header = pytest_conftest.pytest_report_header(opt_in_config)
    assert "JAX default backend: gpu" in probed_header
    assert "JAX visible devices: 2 (NVIDIA RTX 4090, NVIDIA RTX 4090)" in probed_header
    assert "Accelerator available for testing: True" in probed_header
    assert opt_in_config._metadata["Accelerator available for testing"] == "True"
    assert opt_in_config._metadata["JAX default backend"] == "gpu"
    assert opt_in_config._metadata["Artifex backend"] == "cuda12"


def test_tests_conftest_uses_plugin_registration_for_shared_fixtures() -> None:
    """Shared fixtures and the substrax JAX test plugin are registered as pytest plugins."""
    conftest_contents = (REPO_ROOT / "tests/conftest.py").read_text()
    assignments = [
        node
        for node in ast.parse(conftest_contents).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pytest_plugins"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    assert ast.literal_eval(assignments[0].value) == [
        "substrax.testing.pytest_plugin",
        "tests.utils.pytest_hooks",
        "tests.artifex.fixtures.base",
    ]
    assert "print(" not in conftest_contents


def test_import_artifex_does_not_preload_generative_stack() -> None:
    """Importing the top-level package should not eagerly load generative subpackages."""
    payload = run_repo_json(
        (
            "import json, sys; "
            "import artifex; "
            "print(json.dumps({"
            "'has_generative_models': 'artifex.generative_models' in sys.modules, "
            "'has_jax': 'jax' in sys.modules"
            "}))"
        )
    )

    assert payload["has_generative_models"] is False
    assert payload["has_jax"] is False


def test_import_generative_models_keeps_subpackages_lazy() -> None:
    """Importing artifex.generative_models should not eagerly import its heavy children."""
    payload = run_repo_json(
        (
            "import json, sys; "
            "import artifex.generative_models as gm; "
            "print(json.dumps({"
            "'core_loaded': 'artifex.generative_models.core' in sys.modules, "
            "'models_loaded': 'artifex.generative_models.models' in sys.modules, "
            "'extensions_loaded': 'artifex.generative_models.extensions' in sys.modules, "
            "'utils_loaded': 'artifex.generative_models.utils' in sys.modules, "
            "'jax_loaded': 'jax' in sys.modules, "
            "'all': list(getattr(gm, '__all__'))"
            "}))"
        )
    )

    assert payload["core_loaded"] is False
    assert payload["models_loaded"] is False
    assert payload["extensions_loaded"] is False
    assert payload["utils_loaded"] is False
    assert payload["jax_loaded"] is False
    assert payload["all"] == ["core", "extensions", "models", "utils"]


def test_generative_models_exports_resolve_lazily() -> None:
    """Lazy package attributes should still resolve to the documented modules."""
    payload = run_repo_json(
        (
            "import json, sys; "
            "import artifex; "
            "gm = artifex.generative_models; "
            "core = gm.core; "
            "print(json.dumps({"
            "'gm_module': gm.__name__, "
            "'core_module': core.__name__, "
            "'gm_loaded': 'artifex.generative_models' in sys.modules, "
            "'core_loaded': 'artifex.generative_models.core' in sys.modules"
            "}))"
        )
    )

    assert payload["gm_module"] == "artifex.generative_models"
    assert payload["core_module"] == "artifex.generative_models.core"
    assert payload["gm_loaded"] is True
    assert payload["core_loaded"] is True


def test_jax_configuration_module_is_gone() -> None:
    """JAX process configuration has one home, substrax.runtime."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("artifex.generative_models.core.jax_config")


def test_pytest_env_sets_only_variables_the_stack_reads() -> None:
    """The test environment table names no removed artifex knob and no TensorFlow-only variable."""
    pytest_env = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["pytest_env"]

    assert set(pytest_env) == {
        "JAX_ENABLE_X64",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
        "TF_CPP_MIN_LOG_LEVEL",
        "XLA_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    }
    assert pytest_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == {"skip_if_set": True, "value": "false"}


def test_device_requirements_use_the_substrax_markers() -> None:
    """GPU-only tests use the plugin's accelerator and devices markers, not artifex's own."""
    pytest_options = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["pytest"]
    markers = {line.split(":")[0].strip() for line in pytest_options["ini_options"]["markers"]}

    assert not markers & {"gpu", "requires_gpu", "cuda", "cpu", "skip_on_gpu"}
    assert "gpu_available" not in (REPO_ROOT / "conftest.py").read_text()
    assert not (REPO_ROOT / "tests/utils/gpu_test_utils.py").exists()
    for name in ("ci.yml", "upstream-compat.yml"):
        workflow = (REPO_ROOT / ".github/workflows" / name).read_text()
        assert "not gpu" not in workflow
        assert "not cuda" not in workflow
