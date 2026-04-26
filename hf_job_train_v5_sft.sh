#!/usr/bin/env bash
#
# GRPO training v5-SFT: start from the SFT warm-up adapter and run GRPO.
#
# Why:
#   v3/v4 train from base Qwen whose round-0 multirole completions are
#   often generic ("COMMAND: top" etc). That produced flat reward_std in
#   the early training steps and slow convergence. Starting from
#   brodie1of1/war-room-sft-v1 — which has eval_loss 0.024 and
#   mean_token_accuracy 99.1% on the multirole format — means the model
#   already emits the right structure and a reasonable fault-keyword
#   message on step 1. GRPO then refines milestone behavior.
#
# Per hackathon doc §16: "Use GRPO/PPO-style RL only after the model
# can at least occasionally succeed." SFT gets us there.
#
# Config:
#   - Same task mix, rank, LR as v4 (reward surgery active)
#   - 100 episodes (shorter than v4's 200; SFT gave us the head start)
#   - --sft-checkpoint brodie1of1/war-room-sft-v1
#
# Budget:
#   Expected runtime: 1.5-2h on L40S
#   Expected spend  : ~$3-3.50
#
# Usage: bash hf_job_train_v5_sft.sh

set -euo pipefail

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-3h}"
EPISODES="${EPISODES:-100}"
TASKS="${TASKS:-task1 task2 task3 procedural_easy procedural_medium procedural_hard}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-brodie1of1/war-room-sft-v1}"
LORA_R="${LORA_R:-32}"
LR="${LR:-1e-5}"
ADAPTER_REPO="${ADAPTER_REPO:-brodie1of1/war-room-grpo-adapter-v5-sft}"

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

echo "=== [4/6] Run GRPO on SFT checkpoint ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/war_room_grpo_v5_sft

python round2/war_room/train_colab.py \
    --model "$MODEL_NAME" \
    --sft-checkpoint "$SFT_ARG" \
    --episodes "$EPISODES" \
    --tasks $TASKS_ARG \
    --lenient-format \
    --no-unsloth \
    --output outputs/war_room_grpo_v5_sft \
    --lr "$LR_ARG" \
    --lora-r "$LORA_R_ARG" \
    --generations 4 \
    --batch-size 1 \
  || { echo "Training failed with exit $?"; ls -la outputs/war_room_grpo_v5_sft/ || true; exit 1; }

echo "=== [5/6] Generate training curves ==="
python round2/war_room/generate_charts.py || echo "chart gen failed (non-fatal)"

echo "=== Artifacts ==="
ls -la outputs/war_room_grpo_v5_sft/ || true
if [ -f outputs/war_room_grpo_v5_sft/metrics.json ]; then
    echo "--- metrics.json (head) ---"
    head -c 1500 outputs/war_room_grpo_v5_sft/metrics.json
fi

echo "=== [6/6] Push adapter to HF Hub ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("ADAPTER_REPO_ARG", "brodie1of1/war-room-grpo-adapter-v5-sft")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/war_room_grpo_v5_sft",
    repo_id=repo,
    repo_type="model",
    commit_message="GRPO v5-SFT: SFT warm-up + reward surgery + rank 32",
    token=token,
    ignore_patterns=["checkpoint-*/**"],
)
print(f"Artifacts pushed: https://huggingface.co/{repo}")
PY
EOF
)

echo "=== HF Jobs — War Room v5-SFT ==="
echo "Flavor         : $FLAVOR"
echo "Timeout        : $TIMEOUT"
echo "Episodes       : $EPISODES"
echo "Tasks          : $TASKS"
echo "LoRA rank      : $LORA_R"
echo "Learn rate     : $LR"
echo "SFT checkpoint : $SFT_CHECKPOINT"
echo "Adapter        : $ADAPTER_REPO"
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
echo "After training, eval with:"
echo "  ADAPTER_REPO=$ADAPTER_REPO bash hf_job_llm_eval.sh"
