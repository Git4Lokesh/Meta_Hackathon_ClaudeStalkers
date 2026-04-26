#!/usr/bin/env bash
#
# GRPO training v7-fast — SFT warm-up + reward fix + FOCUSED 4-task training.
# brodie1of1 account. Designed to fit inside the 5pm IST submission window.
#
# Why this exists:
#   Lakshminath's full v7 run (200 eps x 9 tasks) is on track to finish
#   ~7:30-8:30 pm IST, after the 5pm submission deadline. This launcher
#   runs a trimmed version on brodie1of1 that fits in the remaining time:
#     - 50 episodes per task (v3 landed +0.046 on 100 eps, so 50 on top
#       of SFT + reward fix should still be enough to beat v3)
#     - 4 tasks (task1, task2, task3, procedural_easy — exactly the ones
#       the head-to-head eval measures, plus one procedural for generality)
#     - L40S ~1 hour training + ~25 min eval = ~1h 25m total
#
# Same v7 patch as Lakshminath's job:
#   TIME_PRESSURE_PENALTY  0.01 -> 0.005
#   PENALTY_CAP_FRACTION   0.40 -> 0.10
#   FATAL_SCORE            0.01 -> 0.001
#   current_score() clamp  0.01 -> 0.001
#
# Budget: ~$1.80 at $1.80/h on L40S
# Timeout: 2h (hard cap ~$3.60)
#
# Usage:
#   bash hf_job_train_v7_fast_brodie.sh
#
# Prereqs:
#   - GeminiHugger/war-room-sft-v1-r32 is public (verified).
#   - HF token with write access to brodie1of1/war-room-grpo-adapter-v7-fast

set -euo pipefail

FLAVOR="${FLAVOR:-h200}"
TIMEOUT="${TIMEOUT:-90m}"
EPISODES="${EPISODES:-50}"
TASKS="${TASKS:-task1 task2 task3 task4 task5 task6 example_custom procedural_easy procedural_hard}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-GeminiHugger/war-room-sft-v1-r32}"
LORA_R="${LORA_R:-32}"
LR="${LR:-1e-5}"
ADAPTER_REPO="${ADAPTER_REPO:-brodie1of1/war-room-grpo-adapter-v7-fast}"
BRANCH="${BRANCH:-feature/v7-reward-fix}"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/6] Clone repo ==="
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 --branch "$BRANCH_ARG" \
    https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git /workspace/repo
cd /workspace/repo
echo "HEAD: $(git log -1 --oneline)"
echo "--- grader.py source-level verify (no Python deps yet) ---"
grep -E '^(TIME_PRESSURE_PENALTY|PENALTY_CAP_FRACTION|FATAL_SCORE) = ' round2/war_room/grader.py

echo "=== [2/6] Install dependencies ==="
pip install --quiet --no-cache-dir \
    "trl>=0.15.0,<0.19" "peft>=0.14.0" "transformers>=4.46.0,<4.50" \
    datasets accelerate bitsandbytes \
    fastapi pydantic uvicorn matplotlib
pip install --quiet --no-cache-dir -e .

echo "--- Verify reward-fix patch via Python import ---"
python - <<'PY'
from round2.war_room.grader import (
    TIME_PRESSURE_PENALTY, PENALTY_CAP_FRACTION, FATAL_SCORE,
)
print(f"TIME_PRESSURE_PENALTY = {TIME_PRESSURE_PENALTY}")
print(f"PENALTY_CAP_FRACTION  = {PENALTY_CAP_FRACTION}")
print(f"FATAL_SCORE           = {FATAL_SCORE}")
assert TIME_PRESSURE_PENALTY == 0.005, "v7 reward fix not applied!"
assert PENALTY_CAP_FRACTION == 0.10,   "v7 reward fix not applied!"
assert FATAL_SCORE == 0.001,           "v7 reward fix not applied!"
print("v7 reward-fix patch verified.")
PY

echo "=== [3/6] Environment info ==="
python - <<'PY'
import torch, os
print(f"Python : {os.sys.version.split()[0]}")
print(f"Torch  : {torch.__version__}")
print(f"CUDA   : {torch.version.cuda} | GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

echo "=== [4/6] Train ==="
python round2/war_room/train_colab.py \
    --model "$MODEL_NAME" \
    --sft-checkpoint "$SFT_ARG" \
    --episodes "$EPISODES" \
    --tasks $TASKS_ARG \
    --lenient-format \
    --no-unsloth \
    --output outputs/war_room_grpo_v7_fast \
    --lr "$LR_ARG" \
    --lora-r "$LORA_R_ARG" \
    --generations 4 \
    --batch-size 1 \
  || { echo "Training failed with exit $?"; ls -la outputs/war_room_grpo_v7_fast/ || true; exit 1; }

echo "=== [5/6] Generate training curves ==="
python round2/war_room/generate_charts.py || echo "chart gen failed (non-fatal)"

echo "=== Artifacts ==="
ls -la outputs/war_room_grpo_v7_fast/ || true
if [ -f outputs/war_room_grpo_v7_fast/metrics.json ]; then
    echo "--- metrics.json (head) ---"
    head -c 1500 outputs/war_room_grpo_v7_fast/metrics.json
fi

echo "=== [6/6] Push adapter to HF Hub ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("ADAPTER_REPO_ARG", "brodie1of1/war-room-grpo-adapter-v7-fast")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/war_room_grpo_v7_fast",
    repo_id=repo,
    repo_type="model",
    commit_message="GRPO v7-fast: reward fix + SFT warm-up + 50 eps x 4 tasks (submission-window variant)",
    token=token,
    ignore_patterns=["checkpoint-*/**"],
)
print(f"Artifacts pushed: https://huggingface.co/{repo}")
PY
EOF
)

echo "=== HF Jobs — War Room v7-fast (brodie1of1, 50 eps x 4 tasks) ==="
echo "Flavor         : $FLAVOR"
echo "Timeout        : $TIMEOUT"
echo "Episodes       : $EPISODES"
echo "Tasks          : $TASKS"
echo "LoRA rank      : $LORA_R"
echo "Learn rate     : $LR"
echo "SFT checkpoint : $SFT_CHECKPOINT"
echo "Adapter        : $ADAPTER_REPO"
echo "Branch         : $BRANCH"
echo "Model          : $MODEL"
echo ""
echo "Reward fix     : pen_cap 0.40->0.10, time_pen 0.01->0.005, floor 0.01->0.001"
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
echo "Job submitted. Monitor with:"
echo "  hf jobs logs <JOB_ID>"
echo ""
echo "When training completes, immediately launch eval:"
echo "  ADAPTER_REPO=$ADAPTER_REPO UPLOAD_REPO=brodie1of1/war-room-eval-results bash hf_job_llm_eval.sh"
