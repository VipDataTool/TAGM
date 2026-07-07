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

cd "$(dirname "$0")"

# Clean bytecode cache to ensure code changes take effect
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Auto-install missing dependencies
python -c "import uvicorn" 2>/dev/null || {
  echo "-> Installing dependencies..."
  pip install -q -r requirements.txt
}

PORT="${TAGM_PORT:-8000}"

# Kill any stale server still holding the port (orphaned terminals in
# codespaces leave the old process running, so edits never take effect
# and the new launch dies with "address already in use").
if fuser -k "${PORT}/tcp" 2>/dev/null; then
  echo "-> Killed stale process on port ${PORT}, waiting for it to release..."
  sleep 1
fi

# ── Why the readiness poll exists ─────────────────────────────────
# uvicorn cannot open the port until Python finishes importing the
# app — and importing the app pulls torch, transformers, sklearn, and
# every analysis module. On a cold codespace that's a 10-30 second
# wall during which the port is CLOSED: any browser tab loaded in that
# window gets a blank page from the forwarding proxy. Nothing is
# wrong; the server just isn't up yet. The loop below polls the
# /api/health endpoint and prints an unambiguous READY banner, so
# "is it up?" is answered by the terminal instead of by a blank tab.

echo "-> Starting server on port ${PORT} (imports take 10-30s cold)..."
python -m src --host 0.0.0.0 --port "$PORT" "$@" &
SERVER_PID=$!

# Ensure Ctrl-C tears down the background server
trap 'kill "$SERVER_PID" 2>/dev/null; exit 0' INT TERM

READY=0
for i in $(seq 1 120); do
  # Bail out early if the server process died (port conflict, import error)
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "!! Server process exited during startup — check tagm.log:"
    tail -n 20 tagm.log 2>/dev/null || true
    exit 1
  fi
  if curl -sf -o /dev/null "http://localhost:${PORT}/api/health" 2>/dev/null; then
    READY=1
    break
  fi
  printf "."
  sleep 1
done
echo ""

if [ "$READY" = "1" ]; then
  echo "=================================================="
  echo "  READY — open http://localhost:${PORT}"
  echo "  (codespaces: use the forwarded-port URL)"
  echo "  Log file: tagm.log"
  echo "=================================================="
else
  echo "!! Server did not become ready within 120s — check tagm.log:"
  tail -n 20 tagm.log 2>/dev/null || true
fi

# Hand the foreground back to the server so logs stream and Ctrl-C works
wait "$SERVER_PID"
