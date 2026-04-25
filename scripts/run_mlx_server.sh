#!/usr/bin/env bash
#
# Run our trained War Room 7B model as a local OpenAI-compatible server
# on Apple Silicon via MLX. Designed for M-series Macs with >=16GB of
# unified memory — M4 Pro with 48GB runs bf16 comfortably at ~30-40 tok/s.
#
# After this starts, the server is at http://localhost:8080/v1 and speaks
# the OpenAI Chat Completions API. Our Gradio Space's "Trained Adapter"
# preset can point at this URL and Agent Mode will drive the trained model.
#
# Usage (from repo root):
#   bash scripts/run_mlx_server.sh
#
# Stop with Ctrl-C. Server uses ~16GB of unified memory while running.
#
# One-time setup is done by run_mlx_server_setup.sh (runs pip install).

set -euo pipefail

# Which model to serve. Defaults to our merged 7B (adapter folded into
# base weights). MLX will convert on first run and cache the converted
# weights under ~/.cache/mlx/.
MODEL="${MODEL:-brodie1of1/war-room-7b-merged}"

# Port — default 8080 to avoid clashing with the Gradio UI (7860).
PORT="${PORT:-8080}"

# Activate our venv if present (python 3.12 with openenv-core).
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Verify mlx-lm is installed.
if ! python -c "import mlx_lm" 2>/dev/null; then
    echo "mlx-lm not installed. Run scripts/run_mlx_server_setup.sh first."
    exit 1
fi

echo "=== MLX inference server ==="
echo "Model:     $MODEL"
echo "Port:      $PORT"
echo "URL:       http://localhost:${PORT}/v1"
echo ""
echo "Point Gradio 'Trained Adapter' preset at that URL to use this."
echo ""

# `mlx_lm.server` is the official OpenAI-compatible shim. It auto-converts
# HF models on first load (downloads, dequantizes/requantizes as needed).
exec python -m mlx_lm.server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --log-level INFO
