#!/usr/bin/env bash
#
# Launch GRPO training for Multi-Agent War Room on Hugging Face Jobs.
#
# Hardware: l40sx1 (1x L40S 48GB, $1.80/hr) — fits Qwen2.5-7B-Instruct
#           in 4-bit quant comfortably with headroom for LoRA + activations
# Timeout:  75 minutes hard cap ($2.25 max spend per run)
# Model:    Qwen2.5-7B-Instruct (no SFT; --lenient-format gives partial
#           reward so GRPO sees non-zero signal from step 1)
#
# Prerequisites:
#   1. `hf auth login`  (one-time, uses your HF token)
#   2. repo pushed to GitHub (we clone inside the container)
#
# Usage:  bash hf_job_launch.sh
#
# Monitor:  hf jobs logs <JOB_ID>   (printed by this script)
#           hf jobs ps              (list running jobs)
#           hf jobs cancel <JOB_ID> (abort)

set -euo pipefail

REPO_URL="https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git"
GIT_REV="${GIT_REV:-main}"
FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-75m}"
EPISODES="${EPISODES:-30}"
TASKS="${TASKS:-task1 task2 task3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Inner command: executed inside the container
INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/5] Clone repo ==="
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 --branch main \
    https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git /workspace/repo
cd /workspace/repo

echo "=== [2/5] Install dependencies ==="
# Pin trl<0.19 so FSDPModule import (torch.distributed.fsdp.fully_shard,
# new in torch 2.5) is not required. Container ships torch 2.4.
pip install --quiet --no-cache-dir \
    "trl>=0.15.0,<0.19" "peft>=0.14.0" "transformers>=4.46.0,<4.50" \
    datasets accelerate bitsandbytes \
    fastapi pydantic uvicorn matplotlib
pip install --quiet --no-cache-dir -e .

echo "=== [3/5] Environment info ==="
python - <<'PY'
import torch, os
print(f"Python : {os.sys.version.split()[0]}")
print(f"Torch  : {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
PY

echo "=== [4/5] Run GRPO training ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/war_room_grpo

# --no-unsloth: unsloth 4-bit kernels sometimes clash with L40S; use
#               stock transformers+bitsandbytes path which is proven
# --lenient-format: gives partial credit for COMMAND/MESSAGE_TO/MESSAGE
#                   keywords so GRPO gets non-zero reward from step 1
#                   (no SFT warm-up needed)
python round2/war_room/train_colab.py \
    --model "$MODEL_NAME" \
    --episodes "$EPISODES" \
    --tasks $TASKS \
    --lenient-format \
    --no-unsloth \
    --output outputs/war_room_grpo \
    --lr 5e-6 \
    --lora-r 16 \
    --generations 4 \
    --batch-size 1 \
  || { echo "Training failed with exit $?"; ls -la outputs/war_room_grpo/ || true; exit 1; }

echo "=== [5/5] Generate training curves ==="
python round2/war_room/generate_charts.py || echo "chart gen failed (non-fatal)"

echo "=== Artifacts ==="
ls -la outputs/war_room_grpo/ || true
if [ -f outputs/war_room_grpo/metrics.json ]; then
    echo "--- metrics.json ---"
    cat outputs/war_room_grpo/metrics.json
fi

echo "=== [6/6] Push artifacts to HF Hub ==="
# Upload LoRA adapter + metrics + charts to a model repo we can pull
# from locally. Repo name is derived from the HF username via whoami.
python - <<'PY'
import os, sys
from huggingface_hub import HfApi, create_repo
api = HfApi(token=os.environ.get("HF_TOKEN"))
user = api.whoami()["name"]
repo_id = f"{user}/war-room-grpo-adapter"
print(f"Pushing to {repo_id}...")
create_repo(repo_id, repo_type="model", exist_ok=True, token=os.environ.get("HF_TOKEN"))
api.upload_folder(
    folder_path="outputs/war_room_grpo",
    repo_id=repo_id,
    repo_type="model",
    commit_message="GRPO adapter + training curves from HF Jobs run",
    token=os.environ.get("HF_TOKEN"),
    ignore_patterns=["checkpoint-*/**"],  # skip intermediate checkpoints
)
print(f"✅ Artifacts pushed: https://huggingface.co/{repo_id}")
PY

# NOTE: Job output is persisted in the HF Jobs artifact store.
# Download with: hf jobs logs <JOB_ID> > training.log
# The LoRA adapter is in outputs/war_room_grpo/ — to get it back to
# your local repo, we'll either push it as an HF model or print it
# and re-create locally.
EOF
)

echo "=== HF Jobs launcher ==="
echo "Flavor      : $FLAVOR (see 'hf jobs hardware' for pricing)"
echo "Timeout     : $TIMEOUT"
echo "Repo rev    : $GIT_REV"
echo "Model       : $MODEL"
echo "Episodes    : $EPISODES"
echo "Tasks       : $TASKS"
echo ""
echo "Launching..."

hf jobs run \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --secrets HF_TOKEN \
    -e MODEL_NAME="$MODEL" \
    -e EPISODES="$EPISODES" \
    -e TASKS="$TASKS" \
    --detach \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    bash -c "$INNER_CMD"

echo ""
echo "Job submitted. Monitor with:"
echo "  hf jobs ps"
echo "  hf jobs logs <JOB_ID>"
