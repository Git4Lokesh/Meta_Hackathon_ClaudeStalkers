#!/usr/bin/env bash
#
# Launch GRPO training v2: bigger budget, larger LoRA.
#
# Motivation:
#   v1 run (91 steps, rank-16 LoRA) produced an adapter that did NOT
#   measurably beat base Qwen 7B on the head-to-head eval
#   (composite 0.285 base vs 0.269 trained — delta -0.017).
#   The training signal was real (reward curves climbed), but the
#   budget was too thin to push a 7B model off its prior.
#
# Changes vs v1:
#   - --episodes 100        (was 30) → 300 GRPO steps vs 90
#   - --lora-r 32           (was 16) → 2× adapter capacity
#   - --lr 2e-5             (was 5e-6) → 4× larger, push harder
#   - timeout 50m           (was 75m, but we only need ~22 min actual)
#
# Hardware: l40sx1 (1x L40S 48GB, $1.80/hr)
# Expected: ~22-25 min runtime (~$0.70-0.75 spend)
# Timeout:  50 minutes hard cap (~$1.50 max spend)
#
# Adapter target repo:
#   brodie1of1/war-room-grpo-adapter-v2
#   (kept separate from v1 so we can A/B compare)
#
# Prerequisites: `hf auth login`, repo pushed to GitHub
#
# Usage:  bash hf_job_launch_v2.sh

set -euo pipefail

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-50m}"
EPISODES="${EPISODES:-100}"
TASKS="${TASKS:-task1 task2 task3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LORA_R="${LORA_R:-32}"
LR="${LR:-2e-5}"
ADAPTER_REPO="${ADAPTER_REPO:-brodie1of1/war-room-grpo-adapter-v2}"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/5] Clone repo ==="
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 --branch main \
    https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git /workspace/repo
cd /workspace/repo

echo "=== [2/5] Install dependencies ==="
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

echo "=== [4/5] Run GRPO training (v2 — bigger) ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/war_room_grpo_v2

python round2/war_room/train_colab.py \
    --model "$MODEL_NAME" \
    --episodes "$EPISODES" \
    --tasks $TASKS \
    --lenient-format \
    --no-unsloth \
    --output outputs/war_room_grpo_v2 \
    --lr "$LR_ARG" \
    --lora-r "$LORA_R_ARG" \
    --generations 4 \
    --batch-size 1 \
  || { echo "Training failed with exit $?"; ls -la outputs/war_room_grpo_v2/ || true; exit 1; }

echo "=== [5/5] Generate training curves ==="
python round2/war_room/generate_charts.py || echo "chart gen failed (non-fatal)"

echo "=== Artifacts ==="
ls -la outputs/war_room_grpo_v2/ || true
if [ -f outputs/war_room_grpo_v2/metrics.json ]; then
    echo "--- metrics.json (head) ---"
    head -c 2000 outputs/war_room_grpo_v2/metrics.json
fi

echo "=== [6/6] Push artifacts to HF Hub ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo_id = os.environ.get("ADAPTER_REPO_ARG", "brodie1of1/war-room-grpo-adapter-v2")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
print(f"Pushing to {repo_id}...")
create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/war_room_grpo_v2",
    repo_id=repo_id,
    repo_type="model",
    commit_message="GRPO v2 adapter: 300 steps, rank-32 LoRA, lr=2e-5",
    token=token,
    ignore_patterns=["checkpoint-*/**"],
)
print(f"✅ Artifacts pushed: https://huggingface.co/{repo_id}")
PY
EOF
)

echo "=== HF Jobs launcher v2 ==="
echo "Flavor      : $FLAVOR"
echo "Timeout     : $TIMEOUT"
echo "Model       : $MODEL"
echo "Episodes    : $EPISODES  (→ ~$((EPISODES * 3)) GRPO steps)"
echo "LoRA rank   : $LORA_R"
echo "Learn rate  : $LR"
echo "Adapter     : $ADAPTER_REPO"
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
    -e LORA_R_ARG="$LORA_R" \
    -e LR_ARG="$LR" \
    -e ADAPTER_REPO_ARG="$ADAPTER_REPO" \
    --detach \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    bash -c "$INNER_CMD"

echo ""
echo "Job submitted. Monitor with:"
echo "  hf jobs ps"
echo "  hf jobs logs <JOB_ID>"
echo ""
echo "After training, re-run head-to-head eval against the v2 adapter:"
echo "  ADAPTER_REPO=$ADAPTER_REPO bash hf_job_llm_eval.sh"
