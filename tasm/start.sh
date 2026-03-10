#!/bin/bash
# TASM Analyzer - Start Script
# Run from the tasm/ directory

set -e

echo "╔══════════════════════════════════════════════════╗"
echo "║  TASM — The Alignment Stress Map Analyzer        ║"
echo "║  Runtime Per-Token Sensitivity Attribution        ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Install dependencies
echo "→ Installing dependencies..."
pip install -q torch transformers accelerate fastapi "uvicorn[standard]" \
    python-multipart matplotlib numpy scipy aiofiles 2>/dev/null

echo "→ Starting server on port 8000..."
echo "  Open http://localhost:8000 in your browser"
echo ""

cd "$(dirname "$0")"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
