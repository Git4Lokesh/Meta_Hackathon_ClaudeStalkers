#!/usr/bin/env bash
#
# GRPO training v4 launcher — reward-surgery codepath.
#
# Baseline for comparison:
#   - v1 (broken)   : delta_composite on scripted eval = -0.017
#   - v2            : delta = -0.001
#   - v3 (multirole): delta = +0.046  ← first positive
#   - Lakshminath v2 (multirole, 6 tasks): not yet evaluated; mean team 0.263
#
# v4 changes on top of v3:
#   - Reward surgery (round2/war_room/grader.py)
#       SOLVE_BONUS +0.10 when all milestones hit
#       PENALTY_CAP_FRACTION 0.40 cap on penalty/credit ratio
#       Oracle now scores 0.99/0.99/0.98 (up from 0.99/0.95/0.88)
#       Fixes easy/hard inversion: procedural_easy (0.11 mean v3) should
#       now score higher than procedural_hard when solved correctly
#   - LoRA rank 16 -> 32 (2x capacity, ~160M -> ~320M trainable params)
#   - LR 5e-6 -> 1e-5 (2x pace)
#   - Task mix matches Lakshminath's v2: task1, task3, task4, procedural_easy,
#     procedural_hard, example_custom (broader = more environment diversity
#     per hackathon doc 2 §36)
#   - Episodes: 200 for an apples-to-apples comparison with his run
#
# Budget:
#   Expected runtime: 2.5-3h on L40S
#   Expected spend  : ~$5
#   Hard cap        : 4h timeout (~$7)
#
# Usage:
#   bash hf_job_train_v4.sh

set -euo pipefail

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-4h}"
EPISODES="${EPISODES:-200}"
TASKS="${TASKS:-task1 task3 task4 procedural_easy procedural_hard example_custom}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LORA_R="${LORA_R:-32}"
LR="${LR:-1e-5}"
ADAPTER_REPO="${ADAPTER_REPO:-brodie1of1/war-room-grpo-adapter-v4}"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/6] Clone repo ==="
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 --branch main \
    https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git /workspace/repo
cd /workspace/repo
echo "HEAD: $(git log -1 --oneline)"

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

echo "=== [4/6] Run GRPO training v4 ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/war_room_grpo_v4

python round2/war_room/train_colab.py \
    --model "$MODEL_NAME" \
    --episodes "$EPISODES" \
    --tasks $TASKS_ARG \
    --lenient-format \
    --no-unsloth \
    --output outputs/war_room_grpo_v4 \
    --lr "$LR_ARG" \
    --lora-r "$LORA_R_ARG" \
    --generations 4 \
    --batch-size 1 \
  || { echo "Training failed with exit $?"; ls -la outputs/war_room_grpo_v4/ || true; exit 1; }

echo "=== [5/6] Generate training curves ==="
python round2/war_room/generate_charts.py || echo "chart gen failed (non-fatal)"

echo "=== Artifacts ==="
ls -la outputs/war_room_grpo_v4/ || true
if [ -f outputs/war_room_grpo_v4/metrics.json ]; then
    echo "--- metrics.json (head) ---"
    head -c 1500 outputs/war_room_grpo_v4/metrics.json
fi

echo "=== [6/6] Push artifacts to HF Hub ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("ADAPTER_REPO_ARG", "brodie1of1/war-room-grpo-adapter-v4")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/war_room_grpo_v4",
    repo_id=repo,
    repo_type="model",
    commit_message="GRPO v4 adapter: reward surgery + rank 32 + lr 1e-5 + broad task mix",
    token=token,
    ignore_patterns=["checkpoint-*/**"],
)
print(f"Artifacts pushed: https://huggingface.co/{repo}")
PY
EOF
)

echo "=== HF Jobs — War Room v4 (reward surgery) ==="
echo "Flavor      : $FLAVOR"
echo "Timeout     : $TIMEOUT"
echo "Episodes    : $EPISODES"
echo "Tasks       : $TASKS"
echo "LoRA rank   : $LORA_R"
echo "Learn rate  : $LR"
echo "Adapter     : $ADAPTER_REPO"
echo "Model       : $MODEL"
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
    -e ADAPTER_REPO_ARG="$ADAPTER_REPO" \
    --detach \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    bash -c "$INNER_CMD"

echo ""
echo "Job submitted. Monitor with:"
echo "  hf jobs logs <JOB_ID>"
echo ""
echo "After training, eval with:"
echo "  ADAPTER_REPO=$ADAPTER_REPO bash hf_job_llm_eval.sh"
