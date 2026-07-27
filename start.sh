#!/bin/bash
# TAGM — Transformer Alignment Geometric Metrology
# Start script. Runnable from anywhere; resolves paths relative to itself.

set -e
cd "$(dirname "$0")"

# Interpreter resolution, most specific first:
#   1. An already-activated virtualenv (VIRTUAL_ENV) — respect the user's choice.
#   2. A repo-local .venv — the local-install layout from the README.
#   3. python3 on PATH — the Codespaces / system-env layout.
#   4. python — last resort (some containers ship only this name).
# Dependency installs go through "$PY" -m pip: macOS system Python has no
# `pip` shim at all, and a bare `pip` can also silently belong to a
# different interpreter than the one that will run the server.
if [ -n "$VIRTUAL_ENV" ]; then
  PY=python
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

echo "=================================================="
echo "  TAGM — Transformer Alignment Geometric Metrology"
echo "=================================================="
echo ""

# HuggingFace token (optional, speeds up downloads + avoids rate limits)
if [ -n "$HF_TOKEN" ]; then
  echo "-> HF_TOKEN detected"
else
  echo "-> No HF_TOKEN set (optional: export HF_TOKEN=hf_... for faster downloads)"
fi

# Clean bytecode cache to ensure code changes take effect
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Auto-install missing dependencies
"$PY" -c "import uvicorn" 2>/dev/null || {
  echo "-> Installing dependencies..."
  "$PY" -m pip install -q -r requirements.txt
}

echo "-> Starting server on port 8000..."
echo "   Open http://localhost:8000"
echo "   Log file: tagm.log"
echo ""

# Host selection is delegated to src/__main__.py, which binds 0.0.0.0 inside
# a container (Codespaces/devcontainer/Docker — the forwarding proxy cannot
# reach loopback, which shows up as a blank page on the forwarded URL) and
# 127.0.0.1 everywhere else. Override with TAGM_HOST.
exec "$PY" -m src --port 8000 "$@"
