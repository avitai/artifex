#!/bin/bash
# Fast documentation server for development
# This skips API documentation generation for much faster builds

set -e

echo "🚀 Starting fast documentation server..."
echo ""
echo "This uses mkdocs-dev.yml which:"
echo "  ✓ Skips mkdocstrings (API docs)"
echo "  ✓ Watches only docs/ directory"
echo "  ✓ Uses dirty reload for incremental builds"
echo ""
echo "For full documentation build, use: mkdocs serve"
echo ""

# Use development config with dirty reload for fastest iteration
.venv/bin/mkdocs serve -f mkdocs-dev.yml --dirtyreload

# Alternative: If you already have a built site, just serve it statically:
# python -m http.server 8000 --directory site
