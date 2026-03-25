#!/bin/bash
# Fast documentation server for development.

set -euo pipefail

echo "Starting fast documentation server with mkdocs-dev.yml..."
echo "Use scripts/build_docs.py for the strict validation and build path."

exec uv run mkdocs serve -f mkdocs-dev.yml --dirtyreload "$@"
