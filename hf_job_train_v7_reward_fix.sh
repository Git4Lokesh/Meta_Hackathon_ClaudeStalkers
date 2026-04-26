#!/usr/bin/env bash
#
# GRPO training v7 — reward shaping fix + SFT warm-up + 9-task GRPO.
# GeminiHugger account.
#
# Why v7:
#   v5 critical analysis revealed task2/3/5/6 were stuck at score=0.01
#   across 800 episodes per task -- not because the 7B model couldn't learn,
#   but because the reward function destroyed the gradient. The model was
#   actually hitting 1-3 milestones per episode (95-98% of episodes) on
#   tasks 5/6 but raw_score = milestone_credit - penalty_applied was
#   negative or near-zero, which got clamped to the 0.01 floor.
#
# Fix shipped on this branch (feature/v7-reward-fix):
#   round2/war_room/grader.py:
#     - TIME_PRESSURE_PENALTY:  0.01  -> 0.005   (halve per-round penalty)
#     - PENALTY_CAP_FRACTION:   0.40  -> 0.10    (tighter penalty cap)
#     - FATAL_SCORE / clamp:    0.01  -> 0.001   (lower floor preserves
#                                                 sub-floor gradient)
#   round2/war_room/train_colab.py:
#     - env_reward clamp lower bound: 0.01 -> 0.001 (match grader)
#
# Validated curves (functional sim, all 7 tasks):
#   task1  ms=5: 0.99   (preserved, was 0.99 in v5)
#   ex_cust ms=3: 0.99  (preserved, was 0.95 in v5)
#   task2  ms=2: 0.20   (was 0.01, gradient restored)
#   task3  ms=2: 0.15   (was 0.01)
#   task5  ms=3: 0.20   (was 0.01)
#   task6  ms=3: 0.20   (was 0.01)
#
# Combined with v6's SFT warm-up:
#   - SFT teaches output format (### TRIAGE/DIAGNOSIS/REMEDIATION blocks)
#   - GRPO with v7 reward gets gradient on hard tasks
#   - Should outperform both v5 (no SFT, broken reward) and v6 (SFT,
#     broken reward) on aggregate per-task means.
#
# Same task mix, episodes, rank, LR as v6 so we can do a clean A/B.
#
# Budget:
#   Expected runtime: 3.5-4.5h on L40S
#   Expected spend  : ~$7
#   Hard cap        : 5h timeout (~$10)
#
# Usage:
#   bash hf_job_train_v7_reward_fix.sh
#
# Prereq: GeminiHugger/war-room-sft-v1-r32 must exist on HF Hub
#   (already pushed via scripts/upcast_sft_adapter.py during v6 prep).

set -euo pipefail

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

echo "--- Verify reward-fix patch via Python import (post-install) ---"
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
print(f"CUDA   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"SFT checkpoint: {os.environ.get('SFT_ARG')}")
PY

echo "=== [4/6] Run GRPO on SFT checkpoint w/ v7 reward fix ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/war_room_grpo_v7_rewardfix

python round2/war_room/train_colab.py \
    --model "$MODEL_NAME" \
    --sft-checkpoint "$SFT_ARG" \
    --episodes "$EPISODES" \
    --tasks $TASKS_ARG \
    --lenient-format \
    --no-unsloth \
    --output outputs/war_room_grpo_v7_rewardfix \
    --lr "$LR_ARG" \
    --lora-r "$LORA_R_ARG" \
    --generations 4 \
    --batch-size 1 \
  || { echo "Training failed with exit $?"; ls -la outputs/war_room_grpo_v7_rewardfix/ || true; exit 1; }

echo "=== [5/6] Generate training curves ==="
python round2/war_room/generate_charts.py || echo "chart gen failed (non-fatal)"

echo "=== Artifacts ==="
ls -la outputs/war_room_grpo_v7_rewardfix/ || true
if [ -f outputs/war_room_grpo_v7_rewardfix/metrics.json ]; then
    echo "--- metrics.json (head) ---"
    head -c 1500 outputs/war_room_grpo_v7_rewardfix/metrics.json
fi

echo "=== [6/6] Push adapter to HF Hub ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("ADAPTER_REPO_ARG", "GeminiHugger/war-room-grpo-adapter-v7-rewardfix")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/war_room_grpo_v7_rewardfix",
    repo_id=repo,
    repo_type="model",
    commit_message="GRPO v7: reward fix (penalty 0.40->0.10, time 0.01->0.005, floor 0.01->0.001) + SFT warm-up + 9 tasks",
    token=token,
    ignore_patterns=["checkpoint-*/**"],
)
print(f"Artifacts pushed: https://huggingface.co/{repo}")
PY
EOF
)

echo "=== HF Jobs — War Room v7 (reward-fix + SFT + 9 tasks, GeminiHugger) ==="
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
echo "After training, eval with:"
echo "  ADAPTER_REPO=$ADAPTER_REPO bash hf_job_llm_eval.sh"
