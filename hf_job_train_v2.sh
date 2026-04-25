#!/usr/bin/env bash
#
# Parameterized GRPO training launcher for War Room v2 on HF Jobs.
#
# Why this exists separate from hf_job_launch.sh:
#   - hf_job_launch.sh hard-codes the output repo as "war-room-grpo-adapter"
#     which would overwrite the v1 adapter
#   - We want one script for both Stage A (smoke, 30 ep) and Stage B (full,
#     250 ep), differing only by env vars
#   - We also need to match the new heuristics for procedural / example_custom
#     tasks (committed in this branch on main)
#
# Configuration (override via env vars before invocation):
#
#   STAGE         "smoke" | "full" — bumps EPISODES + TIMEOUT defaults
#   REPO_NAME     HF model repo name suffix
#                 e.g. "war-room-grpo-adapter-smoke" or "...-v2"
#   EPISODES      Number of training episodes (default: 30 for smoke, 250 full)
#   TASKS         Space-separated task ids
#                 (default: "task1 task2 task3 task4 procedural")
#   TIMEOUT       hh:mm or Nm cap (default: 30m smoke, 4h full)
#   FLAVOR        HF Jobs hardware (default: l40sx1)
#   GIT_REV       Branch / commit to clone (default: main)
#   MODEL         Base model (default: Qwen/Qwen2.5-7B-Instruct)
#
# Usage examples:
#
#   # Stage A — smoke (default)
#   STAGE=smoke REPO_NAME=war-room-grpo-adapter-smoke bash hf_job_train_v2.sh
#
#   # Stage B — full run
#   STAGE=full REPO_NAME=war-room-grpo-adapter-v2 bash hf_job_train_v2.sh
#
# Monitor:
#   hf jobs ps
#   hf jobs logs <JOB_ID>
#   hf jobs cancel <JOB_ID>

set -euo pipefail

STAGE="${STAGE:-smoke}"

# Stage-aware defaults
if [ "$STAGE" = "full" ]; then
    DEFAULT_EPISODES=250
    DEFAULT_TIMEOUT="4h"
    DEFAULT_REPO="war-room-grpo-adapter-v2"
else
    DEFAULT_EPISODES=30
    DEFAULT_TIMEOUT="30m"
    DEFAULT_REPO="war-room-grpo-adapter-smoke"
fi

EPISODES="${EPISODES:-$DEFAULT_EPISODES}"
TIMEOUT="${TIMEOUT:-$DEFAULT_TIMEOUT}"
REPO_NAME="${REPO_NAME:-$DEFAULT_REPO}"
TASKS="${TASKS:-task1 task2 task3 task4 procedural}"
FLAVOR="${FLAVOR:-l40sx1}"
GIT_REV="${GIT_REV:-main}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/6] Clone repo ==="
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 --branch "$GIT_REV" \
    https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git /workspace/repo
cd /workspace/repo
echo "Cloned $(git log -1 --oneline)"

echo "=== [2/6] Install dependencies ==="
pip install --quiet --no-cache-dir \
    "trl>=0.15.0,<0.19" "peft>=0.14.0" "transformers>=4.46.0,<4.50" \
    datasets accelerate bitsandbytes \
    fastapi pydantic uvicorn matplotlib
pip install --quiet --no-cache-dir -e .

echo "=== [3/6] Environment info ==="
python - <<'PY'
import torch, os
print(f"Python : {os.sys.version.split()[0]}")
print(f"Torch  : {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
PY

echo "=== [4/6] Run GRPO training ==="
echo "  episodes: $EPISODES"
echo "  tasks   : $TASKS"
echo "  model   : $MODEL_NAME"
echo "  output  : outputs/war_room_grpo"
export PYTHONPATH=/workspace/repo
mkdir -p outputs/war_room_grpo

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

echo "=== [5/6] Generate training curves ==="
python round2/war_room/generate_charts.py || echo "chart gen failed (non-fatal)"

echo "=== Artifacts ==="
ls -la outputs/war_room_grpo/ || true
if [ -f outputs/war_room_grpo/metrics.json ]; then
    echo "--- metrics.json ---"
    cat outputs/war_room_grpo/metrics.json
fi

echo "=== [6/6] Push artifacts to HF Hub ==="
python - <<PY
import os
from huggingface_hub import HfApi, create_repo
api = HfApi(token=os.environ.get("HF_TOKEN"))
user = api.whoami()["name"]
repo_id = f"{user}/${REPO_NAME}"
print(f"Pushing to {repo_id}...")
create_repo(repo_id, repo_type="model", exist_ok=True, token=os.environ.get("HF_TOKEN"))
api.upload_folder(
    folder_path="outputs/war_room_grpo",
    repo_id=repo_id,
    repo_type="model",
    commit_message="GRPO adapter v2 (procedural-aware heuristics)",
    token=os.environ.get("HF_TOKEN"),
    ignore_patterns=["checkpoint-*/**"],
)
print(f"Artifacts pushed: https://huggingface.co/{repo_id}")
PY
EOF
)

# Substitute REPO_NAME into the heredoc since heredocs with quoted EOF
# don't expand. We do it the simple way: use envsubst-style braces.
INNER_CMD="${INNER_CMD//\$\{REPO_NAME\}/$REPO_NAME}"

echo "=== HF Jobs War Room v2 launcher ==="
echo "Stage       : $STAGE"
echo "Episodes    : $EPISODES"
echo "Tasks       : $TASKS"
echo "Output repo : $REPO_NAME"
echo "Timeout     : $TIMEOUT"
echo "Flavor      : $FLAVOR"
echo "Git rev     : $GIT_REV"
echo "Model       : $MODEL"
echo ""

read -p "Proceed? (y/N): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

echo "Launching..."

hf jobs run \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --secrets HF_TOKEN \
    -e MODEL_NAME="$MODEL" \
    -e EPISODES="$EPISODES" \
    -e TASKS="$TASKS" \
    -e GIT_REV="$GIT_REV" \
    --detach \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    bash -c "$INNER_CMD"

echo ""
echo "Job submitted. Monitor with:"
echo "  hf jobs ps"
echo "  hf jobs logs <JOB_ID>"
