#!/bin/bash
# TASM Analyzer - Start Script
# Run from the tasm/ directory

set -e

echo "=================================================="
echo "  TASM -- The Alignment Stress Map Analyzer"
echo "  Runtime Per-Token Sensitivity Attribution"
echo "=================================================="
echo ""

# HuggingFace token (optional, speeds up downloads + avoids rate limits)
# Set before running: export HF_TOKEN=hf_yourTokenHere
if [ -n "$HF_TOKEN" ]; then
  echo "-> HF_TOKEN detected"
else
  echo "-> No HF_TOKEN set (optional: export HF_TOKEN=hf_... for faster downloads)"
fi

# Install dependencies
echo "→ Installing dependencies..."
pip install -q torch transformers accelerate fastapi "uvicorn[standard]" \
    python-multipart matplotlib numpy scipy aiofiles reportlab 2>/dev/null

echo "→ Starting server on port 8000..."
echo "  Open http://localhost:8000 in your browser"
echo ""

cd "$(dirname "$0")"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300
