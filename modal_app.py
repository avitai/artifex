"""Modal launcher for Artifex GPU verification and attention backend benchmarking.

The local development card is an RTX 4090 (compute capability 8.9, Ada). That is
enough to exercise one branch of the attention backend selector, but not the
Hopper-only paths: head dims above 128, packed layouts, and multi-head latent
attention all require compute capability 9.0. Running the same container on
several Modal GPUs covers the architectures the selector has to decide between.

Usage::

    modal run deploy/modal_app.py --task probe
    modal run deploy/modal_app.py --task bench --gpu L40S
    modal run deploy/modal_app.py --task tests --gpu A100-80GB

Requires a Modal account (``modal setup`` for browser auth). Modal is a host-side
launcher only; it is deliberately not an Artifex dependency.
"""

from __future__ import annotations

import os
import subprocess

import modal


APP_NAME = "artifex-gpu"
REPO_PATH = "/root/artifex"

# Ada, matching the local RTX 4090's compute capability.
DEFAULT_GPU = "L40S"

# Excludes for the build-time repo copy: local virtualenvs and caches, generated
# documentation, private notes, and every ``.env*`` file. activate.sh sources
# .artifex.env, which pins ARTIFEX_ENV_ROOT to a local path and would point the
# container at a virtualenv that does not exist inside the image.
_IMAGE_IGNORE = [
    "**/.git",
    "**/.venv",
    "**/.env*",
    "**/.artifex.env",
    "**/node_modules",
    "**/site",
    "**/site-offline",
    "**/htmlcov",
    "**/memory-bank",
    "**/temp",
    "**/outputs",
    "**/checkpoints",
    "**/benchmark_results",
    "**/test_results",
    "**/__pycache__",
    "**/*.pyc",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.mypy_cache",
]

app = modal.App(APP_NAME)

# Pin uv to the version that generated uv.lock, for build stability.
_UV_VERSION = "0.11.25"

# `--frozen` installs the exact pinned versions from uv.lock without re-resolving.
# `--locked` additionally re-checks consistency, which fails spuriously on Modal
# because its managed Python differs from the local interpreter and the
# git-sourced dependency graph re-resolves slightly differently even when the
# lock is correct.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(f"uv=={_UV_VERSION}")
    .add_local_dir(".", REPO_PATH, copy=True, ignore=_IMAGE_IGNORE)
    .run_commands(f"cd {REPO_PATH} && uv sync --extra cuda-dev --frozen --no-dev")
    .workdir(REPO_PATH)
)

# XLA_PYTHON_CLIENT_PREALLOCATE=false keeps a short benchmark from grabbing the
# whole card, matching the local .artifex.env recipe.
_RUN_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
    "TF_CPP_MIN_LOG_LEVEL": "1",
}


def _run(argv: list[str]) -> None:
    """Run a command inside the synced project environment."""
    os.environ.update(_RUN_ENV)
    subprocess.run(
        ["uv", "run", "--no-sync", *argv],
        cwd=REPO_PATH,
        check=True,
    )


@app.function(image=image, gpu=DEFAULT_GPU, timeout=15 * 60)
def probe() -> None:
    """Report the backend, device and attention-kernel eligibility of this GPU.

    This is the cheap sanity check: it confirms the image builds, that JAX sees a
    CUDA device, and which fused attention paths the card actually admits.
    """
    _run(["python", "-u", "deploy/gpu_probe.py"])


@app.function(image=image, gpu=DEFAULT_GPU, timeout=60 * 60)
def bench() -> None:
    """Benchmark the available attention backends against each other."""
    _run(["python", "-u", "deploy/bench_attention.py"])


@app.function(image=image, gpu=DEFAULT_GPU, timeout=2 * 60 * 60)
def tests(extra_args: list[str]) -> None:
    """Run the GPU-marked suite plus the attention layers on a real device."""
    _run(
        [
            "pytest",
            "-m",
            "gpu or requires_gpu",
            "tests/artifex/generative_models/core/layers",
            "-q",
            "--no-header",
            "--no-cov",
            *extra_args,
        ]
    )


@app.local_entrypoint()
def main(task: str = "probe", gpu: str = DEFAULT_GPU, extra: str = "") -> None:
    """Dispatch a task to Modal.

    Args:
        task: One of ``"probe"``, ``"bench"``, or ``"tests"``.
        gpu: Modal GPU spec, for example ``"L40S"``, ``"A100-80GB"`` or ``"H100"``.
        extra: Extra CLI arguments forwarded verbatim, as one space-separated string.
    """
    extra_args = extra.split() if extra else []
    tasks = {"probe": probe, "bench": bench, "tests": tests}
    if task not in tasks:
        msg = f"Unknown task {task!r}; expected one of {sorted(tasks)}"
        raise ValueError(msg)

    if task == "tests":
        tasks[task].with_options(gpu=gpu).remote(extra_args)
    else:
        tasks[task].with_options(gpu=gpu).remote()
