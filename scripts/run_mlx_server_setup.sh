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

# Use the venv's python directly (not pip on PATH, which may belong to
# the system Python 3.9 user-site).
VPYTHON=".venv/bin/python"

# Ensure pip is bootstrapped inside the venv (ensurepip sometimes leaves
# it out depending on how the venv was created).
if ! "$VPYTHON" -m pip --version >/dev/null 2>&1; then
    "$VPYTHON" -m ensurepip --upgrade
fi

echo "Installing mlx-lm and its deps (takes ~1-2 min) ..."
"$VPYTHON" -m pip install --upgrade pip --quiet
"$VPYTHON" -m pip install --upgrade "mlx-lm>=0.20" "mlx>=0.20" --quiet

echo ""
echo "✅ Setup complete."
echo ""
echo "Next:  bash scripts/run_mlx_server.sh"
echo "       The server will start at http://localhost:8080/v1"
