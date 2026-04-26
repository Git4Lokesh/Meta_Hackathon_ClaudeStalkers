#!/usr/bin/env bash
#
# GRPO training v5 launcher — Lakshminath / GeminiHugger account.
#
# Builds on Lokesh's v4 recipe (reward surgery + rank 32 + lr 1e-5) but
# expands the task mix to cover EVERY scripted task (task1..task6) plus
# example_custom and procedural_easy/hard. Goal: maximize task diversity
# to validate generalization claims on the widest possible set.
#
# Differences vs hf_job_train_v4.sh:
#   - 9 tasks instead of 6 (adds task2, task5, task6)
#   - Uses our feature branch with two prep fixes:
#       * task6: per-grader state (no module-level _logs_read global) so
#         later episodes don't start with milestone already satisfied
#       * train_colab.py: MAX_EPISODE_ROUNDS 16 -> 20, so task5 fits
#         cleanly and task6 only loses 5 of 25 rounds
#   - Adapter pushed to GeminiHugger/war-room-grpo-adapter-v5 (not brodie1of1)
#   - Timeout 5h (longer per-episode work due to extra rounds)
#
# Risks (read before launching):
#   - task5/task6 milestones rely on specific phrasings ("tampered", "DNS",
#     "root cause") the base model rarely emits spontaneously. If they
#     don't fire often enough, those tasks contribute zero gradient.
#   - task6.max_rounds=25 truncated to 20; the last belief_convergence_bonus
#     and remediation_restarts_services may fire less than they could.
#   - These risks are why we *also* keep Lokesh's 6-task v4 baseline as a
#     comparison point.
#
# Budget:
#   Expected runtime: 3.5-4.5h on L40S
#   Expected spend  : ~$7
#   Hard cap        : 5h timeout (~$10)
#
# Usage:
#   bash hf_job_train_v5.sh

set -euo pipefail

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-5h}"
EPISODES="${EPISODES:-200}"
TASKS="${TASKS:-task1 task2 task3 task4 task5 task6 example_custom procedural_easy procedural_hard}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LORA_R="${LORA_R:-32}"
LR="${LR:-1e-5}"
ADAPTER_REPO="${ADAPTER_REPO:-GeminiHugger/war-room-grpo-adapter-v5}"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/6] Clone repo ==="
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 --branch feature/grpo-multirole-outputs-fast \
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

echo "=== [4/6] Run GRPO training v5 ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/war_room_grpo_v5

python round2/war_room/train_colab.py \
    --model "$MODEL_NAME" \
    --episodes "$EPISODES" \
    --tasks $TASKS_ARG \
    --lenient-format \
    --no-unsloth \
    --output outputs/war_room_grpo_v5 \
    --lr "$LR_ARG" \
    --lora-r "$LORA_R_ARG" \
    --generations 4 \
    --batch-size 1 \
  || { echo "Training failed with exit $?"; ls -la outputs/war_room_grpo_v5/ || true; exit 1; }

echo "=== [5/6] Generate training curves ==="
python round2/war_room/generate_charts.py || echo "chart gen failed (non-fatal)"

echo "=== Artifacts ==="
ls -la outputs/war_room_grpo_v5/ || true
if [ -f outputs/war_room_grpo_v5/metrics.json ]; then
    echo "--- metrics.json (head) ---"
    head -c 1500 outputs/war_room_grpo_v5/metrics.json
fi

echo "=== [6/6] Push artifacts to HF Hub ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("ADAPTER_REPO_ARG", "GeminiHugger/war-room-grpo-adapter-v5")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/war_room_grpo_v5",
    repo_id=repo,
    repo_type="model",
    commit_message="GRPO v5 adapter: reward surgery + rank 32 + lr 1e-5 + 9 task families",
    token=token,
    ignore_patterns=["checkpoint-*/**"],
)
print(f"Artifacts pushed: https://huggingface.co/{repo}")
PY
EOF
)

echo "=== HF Jobs — War Room v5 (reward surgery + 9 tasks, GeminiHugger) ==="
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
