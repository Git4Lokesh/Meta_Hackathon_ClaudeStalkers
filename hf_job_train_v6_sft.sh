#!/usr/bin/env bash
#
# GRPO training v6 — SFT warm-up + 9-task GRPO. GeminiHugger account.
#
# Combines:
#   - Lokesh's SFT warm-up adapter (GeminiHugger/war-room-sft-v1) so the
#     model emits the multirole format on step 1 instead of having to
#     learn it from scratch through GRPO trial-and-error.
#   - v5's full 9-task mix (task1..task6 + example_custom +
#     procedural_easy/hard) so we don't sacrifice generalization breadth.
#   - v5's reward surgery + task3 verifier loosening + task6 per-grader
#     state fix (all live in feature/grpo-multirole-outputs-fast).
#
# Caveat to know about: the SFT dataset only contains examples for 6 of
# the 9 tasks (task1, task2, task3, procedural_easy/medium/hard). task4,
# task5, task6, example_custom are GRPO-cold but they still benefit from
# the SFT-taught output format (### TRIAGE / ### DIAGNOSIS /
# ### REMEDIATION block structure transfers across tasks).
#
# Differences vs hf_job_train_v5.sh (no-SFT baseline):
#   - Adds --sft-checkpoint GeminiHugger/war-room-sft-v1
#   - Adapter target: GeminiHugger/war-room-grpo-adapter-v6-sft
#   - Same task mix, episodes (200), rank (32), LR (1e-5)
#
# Budget:
#   Expected runtime: 3.5-4.5h on L40S
#   Expected spend  : ~$7
#   Hard cap        : 5h timeout (~$10)
#
# Usage:
#   bash hf_job_train_v6_sft.sh
#
# Prereq: GeminiHugger/war-room-sft-v1 must exist on HF Hub. Launched via
#   ADAPTER_REPO=GeminiHugger/war-room-sft-v1 bash hf_job_sft.sh

set -euo pipefail

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-5h}"
EPISODES="${EPISODES:-200}"
TASKS="${TASKS:-task1 task2 task3 task4 task5 task6 example_custom procedural_easy procedural_hard}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-GeminiHugger/war-room-sft-v1}"
LORA_R="${LORA_R:-32}"
LR="${LR:-1e-5}"
ADAPTER_REPO="${ADAPTER_REPO:-GeminiHugger/war-room-grpo-adapter-v6-sft}"

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
print(f"SFT checkpoint: {os.environ.get('SFT_ARG')}")
PY

echo "=== [4/6] Run GRPO on SFT checkpoint (v6) ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/war_room_grpo_v6_sft

python round2/war_room/train_colab.py \
    --model "$MODEL_NAME" \
    --sft-checkpoint "$SFT_ARG" \
    --episodes "$EPISODES" \
    --tasks $TASKS_ARG \
    --lenient-format \
    --no-unsloth \
    --output outputs/war_room_grpo_v6_sft \
    --lr "$LR_ARG" \
    --lora-r "$LORA_R_ARG" \
    --generations 4 \
    --batch-size 1 \
  || { echo "Training failed with exit $?"; ls -la outputs/war_room_grpo_v6_sft/ || true; exit 1; }

echo "=== [5/6] Generate training curves ==="
python round2/war_room/generate_charts.py || echo "chart gen failed (non-fatal)"

echo "=== Artifacts ==="
ls -la outputs/war_room_grpo_v6_sft/ || true
if [ -f outputs/war_room_grpo_v6_sft/metrics.json ]; then
    echo "--- metrics.json (head) ---"
    head -c 1500 outputs/war_room_grpo_v6_sft/metrics.json
fi

echo "=== [6/6] Push adapter to HF Hub ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("ADAPTER_REPO_ARG", "GeminiHugger/war-room-grpo-adapter-v6-sft")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/war_room_grpo_v6_sft",
    repo_id=repo,
    repo_type="model",
    commit_message="GRPO v6-SFT: SFT warm-up + reward surgery + 9 task families",
    token=token,
    ignore_patterns=["checkpoint-*/**"],
)
print(f"Artifacts pushed: https://huggingface.co/{repo}")
PY
EOF
)

echo "=== HF Jobs — War Room v6 (SFT warm-up + 9 tasks, GeminiHugger) ==="
echo "Flavor         : $FLAVOR"
echo "Timeout        : $TIMEOUT"
echo "Episodes       : $EPISODES"
echo "Tasks          : $TASKS"
echo "LoRA rank      : $LORA_R"
echo "Learn rate     : $LR"
echo "SFT checkpoint : $SFT_CHECKPOINT"
echo "Adapter        : $ADAPTER_REPO"
echo "Model          : $MODEL"
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
    --detach \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    bash -c "$INNER_CMD"

echo ""
echo "Job submitted. Monitor with:"
echo "  hf jobs logs <JOB_ID>"
echo ""
echo "After training, eval with:"
echo "  ADAPTER_REPO=$ADAPTER_REPO bash hf_job_llm_eval.sh"
