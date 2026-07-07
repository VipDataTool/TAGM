#!/bin/bash
# TAGM — Transformer Alignment Geometric Metrology
# Start script. Run from the repo root.

set -e

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
python -c "import uvicorn" 2>/dev/null || {
  echo "-> Installing dependencies..."
  pip install -q -r requirements.txt
}

echo "-> Starting server on port 8000..."
echo "   Open http://localhost:8000"
echo "   Log file: tagm.log"
echo ""

cd "$(dirname "$0")"
exec python -m src --host 0.0.0.0 --port 8000 "$@"
