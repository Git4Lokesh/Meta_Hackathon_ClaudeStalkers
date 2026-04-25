#!/usr/bin/env bash
#
# One-time setup for the local MLX inference server on Apple Silicon.
# Installs mlx-lm (Apple's LLM inference library) into the repo's .venv.
#
# Usage:  bash scripts/run_mlx_server_setup.sh

set -euo pipefail

if [ ! -f .venv/bin/activate ]; then
    echo "No .venv found. Create it first:  python3.12 -m venv .venv"
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing mlx-lm and its deps (takes ~1-2 min) ..."
pip install --upgrade pip --quiet
pip install --upgrade "mlx-lm>=0.20" "mlx>=0.20" --quiet

echo ""
echo "✅ Setup complete."
echo ""
echo "Next:  bash scripts/run_mlx_server.sh"
echo "       The server will start at http://localhost:8080/v1"
