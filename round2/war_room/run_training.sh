#!/usr/bin/env bash
# =============================================================
# Generic GPU Training Runner for Multi-Agent War Room
# Works on: HF Spaces GPU, HF Jobs, any Linux machine with CUDA
# =============================================================
#
# Usage:
#   bash round2/war_room/run_training.sh [quick|full]
#
#   quick  — Qwen2.5-1.5B on T4 (~15 min)
#   full   — Qwen2.5-7B on A100 (~30-60 min, recommended)
#
# Prereqs:
#   - CUDA-capable GPU
#   - HF_TOKEN environment variable set (optional but recommended)
# =============================================================

set -e

MODE="${1:-full}"

echo "================================================"
echo "🔥 WAR ROOM — GRPO TRAINING"
echo "================================================"

# Install dependencies
echo "[1/3] Installing dependencies..."
pip install -q "trl>=0.15.0" "peft>=0.14.0" "transformers>=4.46.0" \
    datasets accelerate bitsandbytes
pip install -q fastapi pydantic uvicorn openai matplotlib rich

# Install editable project
pip install -e . --quiet

# Check GPU
echo "[2/3] Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'✅ GPU: {name} ({mem:.0f}GB)')
else:
    print('❌ No GPU available — training will be too slow')
    exit(1)
"

# Run training
echo "[3/3] Running GRPO training (mode: $MODE)..."
export PYTHONPATH=.

if [ "$MODE" = "quick" ]; then
    echo "Using T4-optimized script with Qwen2.5-1.5B..."
    python3 round2/war_room/train_t4_quick.py
else
    echo "Installing Unsloth for 4-bit Qwen2.5-7B..."
    pip install -q unsloth
    echo "Using full GRPO with Qwen2.5-7B..."
    python3 round2/war_room/train_colab.py --episodes 30 --tasks task1 task2 task3
fi

# Generate charts
echo ""
echo "Generating training curves..."
python3 round2/war_room/generate_charts.py

echo ""
echo "================================================"
echo "✅ TRAINING COMPLETE"
echo "================================================"
echo "Results in: outputs/war_room_grpo/"
echo "  - metrics.json"
echo "  - training_curves.png"
echo "  - baseline_vs_trained.png"
