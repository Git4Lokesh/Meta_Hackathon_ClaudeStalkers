#!/usr/bin/env bash
#
# GRPO training v6-SFT — brodie1of1 account, with the PEFT key-name fix.
#
# Differences from Lakshminath's hf_job_train_v6_sft.sh:
#   - Uses brodie1of1/war-room-sft-v1 as the SFT checkpoint (his
#     GeminiHugger/war-room-sft-v1 doesn't exist yet; ours does).
#   - Target adapter repo: brodie1of1/war-room-grpo-adapter-v6-sft
#   - Runs from main branch (PEFT key-name fix landed as commit 55e71c8
#     which was merged to main as dd2ac79).
#
# Same training shape as his (200 eps, 9 tasks, rank 32, lr 1e-5) so
# results are apples-to-apples.
#
# Budget:
#   Runtime: 3.5-4.5h on L40S
#   Spend:   ~$7
#   Timeout: 5h (~$10)

set -euo pipefail

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-5h}"
EPISODES="${EPISODES:-200}"
TASKS="${TASKS:-task1 task2 task3 task4 task5 task6 example_custom procedural_easy procedural_hard}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-brodie1of1/war-room-sft-v1}"
LORA_R="${LORA_R:-32}"
LR="${LR:-1e-5}"
ADAPTER_REPO="${ADAPTER_REPO:-brodie1of1/war-room-grpo-adapter-v6-sft}"

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

echo "=== [3/6] Upcast SFT adapter r=16 -> r=32 ==="
# Lakshminath's upcast_sft_adapter.py zero-pads the rank-16 SFT LoRA
# weights to rank 32 so they load into the v6 GRPO config. Writes to
# outputs/war_room_sft_v1_upcast/ locally. No push — we load from disk.
export PYTHONPATH=/workspace/repo
python scripts/upcast_sft_adapter.py \
    --src-repo "$SFT_ARG" \
    --dst-dir outputs/war_room_sft_v1_upcast \
    --new-rank "$LORA_R_ARG" \
    --push-to none \
  || { echo "SFT upcast failed with exit $?"; exit 1; }

echo "=== [4/6] Environment info ==="
python - <<'PY'
import torch, os
print(f"Python : {os.sys.version.split()[0]}")
print(f"Torch  : {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
PY

echo "=== [5/6] Run GRPO on SFT checkpoint ==="
mkdir -p outputs/war_room_grpo_v6_sft

python round2/war_room/train_colab.py \
    --model "$MODEL_NAME" \
    --sft-checkpoint outputs/war_room_sft_v1_upcast \
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

echo "=== [6/6] Artifacts + push ==="
ls -la outputs/war_room_grpo_v6_sft/ || true
if [ -f outputs/war_room_grpo_v6_sft/metrics.json ]; then
    echo "--- metrics.json (head) ---"
    head -c 1500 outputs/war_room_grpo_v6_sft/metrics.json
fi

python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("ADAPTER_REPO_ARG", "brodie1of1/war-room-grpo-adapter-v6-sft")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/war_room_grpo_v6_sft",
    repo_id=repo,
    repo_type="model",
    commit_message="GRPO v6-SFT: SFT warm-up (fixed key naming) + 9 tasks + rank 32",
    token=token,
    ignore_patterns=["checkpoint-*/**"],
)
print(f"Artifacts pushed: https://huggingface.co/{repo}")
PY
EOF
)

echo "=== HF Jobs — War Room v6-SFT (brodie1of1) ==="
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
echo "  ADAPTER_REPO=$ADAPTER_REPO UPLOAD_REPO=brodie1of1/war-room-eval-results bash hf_job_llm_eval.sh"
