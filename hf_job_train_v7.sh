#!/usr/bin/env bash
#
# GRPO training v7 (full) — SFT r32 + reward-fix grader + 9 tasks × 200 ep
# → GeminiHugger/war-room-grpo-adapter-v7-rewardfix
#
# This is the *full* v7 run (the one that was missing as a file for a while).
# Shorter / brodie1of1 window run:  hf_job_train_v7_fast_brodie.sh
# v6 SFT (original reward, same 9 tasks):  hf_job_train_v6_sft.sh
#
# Clones:  feature/v7-reward-fix  (grader + train_colab floor clamp, etc.)
# SFT:     GeminiHugger/war-room-sft-v1-r32
#
# Usage:  bash hf_job_train_v7.sh
#         FLAVOR=h200 TIMEOUT=5h bash hf_job_train_v7.sh

set -euo pipefail
cd "$(dirname "$0")"

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-5h}"
EPISODES="${EPISODES:-200}"
TASKS="${TASKS:-task1 task2 task3 task4 task5 task6 example_custom procedural_easy procedural_hard}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-GeminiHugger/war-room-sft-v1-r32}"
LORA_R="${LORA_R:-32}"
LR="${LR:-1e-5}"
ADAPTER_REPO="${ADAPTER_REPO:-GeminiHugger/war-room-grpo-adapter-v7-rewardfix}"
BRANCH="${BRANCH:-feature/v7-reward-fix}"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/6] Clone (v7 branch) ==="
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 --branch "$BRANCH_ARG" \
    https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git /workspace/repo
cd /workspace/repo
echo "HEAD: $(git log -1 --oneline)"
grep -E '^(TIME_PRESSURE_PENALTY|PENALTY_CAP_FRACTION|FATAL_SCORE) = ' round2/war_room/grader.py

echo "=== [2/6] deps ==="
pip install --quiet --no-cache-dir \
    "trl>=0.15.0,<0.19" "peft>=0.14.0" "transformers>=4.46.0,<4.50" \
    datasets accelerate bitsandbytes \
    fastapi pydantic uvicorn matplotlib
pip install --quiet --no-cache-dir -e .

python - <<'PY'
from round2.war_room.grader import (
    TIME_PRESSURE_PENALTY, PENALTY_CAP_FRACTION, FATAL_SCORE,
)
assert TIME_PRESSURE_PENALTY == 0.005
assert PENALTY_CAP_FRACTION == 0.10
assert FATAL_SCORE == 0.001
print("v7 reward-fix verified")
PY

echo "=== [3/6] GPU ==="
python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== [4/6] train ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/war_room_grpo_v7
python round2/war_room/train_colab.py \
    --model "$MODEL_NAME" \
    --sft-checkpoint "$SFT_ARG" \
    --episodes "$EPISODES" \
    --tasks $TASKS_ARG \
    --lenient-format \
    --no-unsloth \
    --output outputs/war_room_grpo_v7 \
    --lr "$LR_ARG" \
    --lora-r "$LORA_R_ARG" \
    --generations 4 \
    --batch-size 1 \
  || { echo "train failed $?" ; ls -la outputs/war_room_grpo_v7/ || true; exit 1; }

echo "=== [5/6] charts ==="
python round2/war_room/generate_charts.py \
  --metrics outputs/war_room_grpo_v7/metrics.json || true

ls -la outputs/war_room_grpo_v7/

echo "=== [6/6] push ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("ADAPTER_REPO_ARG", "GeminiHugger/war-room-grpo-adapter-v7-rewardfix")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/war_room_grpo_v7",
    repo_id=repo,
    repo_type="model",
    commit_message="GRPO v7 full: SFT r32 + reward-fix + 9 tasks x 200 ep",
    token=token,
    ignore_patterns=["checkpoint-*/**"],
)
print("https://huggingface.co/" + repo)
PY
EOF
)

echo "=== hf_job_train_v7 (full) ==="
echo "Branch: $BRANCH  |  Adapter: $ADAPTER_REPO  |  Flavor: $FLAVOR"
echo ""

hf jobs run \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --secrets HF_TOKEN \
    -e MODEL_NAME="$MODEL" \
    -e EPISODES="$EPISODES" \
    -e TASKS_ARG="$TASKS" \
    -e LR_ARG="$LR" \
    -e LORA_R_ARG="$LORA_R" \
    -e SFT_ARG="$SFT_CHECKPOINT" \
    -e ADAPTER_REPO_ARG="$ADAPTER_REPO" \
    -e BRANCH_ARG="$BRANCH" \
    --detach \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    bash -c "$INNER_CMD"

echo ""
echo "Monitor:  hf jobs logs <JOB_ID>"
echo "Eval:     bash hf_job_llm_eval_v7.sh"
