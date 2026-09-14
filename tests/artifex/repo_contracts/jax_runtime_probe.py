"""Record the backend and device count of the pytest session that runs this file.

``test_jax_test_environment_contracts.py`` runs it in a fresh pytest session. Its name does not
match ``python_files``, so the suite never collects it on its own.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax


def test_record_runtime() -> None:
    output = Path(os.environ["ARTIFEX_RUNTIME_PROBE_OUTPUT"])

    output.write_text(json.dumps([jax.default_backend(), jax.device_count()]), encoding="utf-8")
