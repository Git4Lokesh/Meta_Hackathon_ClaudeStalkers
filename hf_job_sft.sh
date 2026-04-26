#!/usr/bin/env bash
#
# HF Job: SFT warm-up for War Room multirole format.
#
# Goal:
#   Shift Qwen2.5-7B-Instruct's prior toward emitting the exact
#   ### TRIAGE / ### DIAGNOSIS / ### REMEDIATION format with correct
#   fault keywords, so the subsequent GRPO run starts with a policy
#   that already *occasionally* succeeds (per hackathon doc §16,
#   §45).
#
# Dataset: outputs/sft_dataset/train.jsonl (355 oracle-validated
# examples, committed to main as b8c5a6a).
#
# Expected:
#   - Runtime: ~15-25 min on L40S
#   - Cost:    ~$0.60
#   - Output:  brodie1of1/war-room-sft-v1 adapter on HF Hub
#
# Next step after this completes: launch GRPO with --sft-checkpoint
# pointing at this adapter.
#
# Usage:
#   bash hf_job_sft.sh

set -euo pipefail

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-45m}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-16}"
ADAPTER_REPO="${ADAPTER_REPO:-brodie1of1/war-room-sft-v1}"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/5] Clone repo ==="
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 --branch main \
    https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git /workspace/repo
cd /workspace/repo
echo "HEAD: $(git log -1 --oneline)"

echo "=== [2/5] Install dependencies ==="
pip install --quiet --no-cache-dir \
    "trl>=0.15.0,<0.19" "peft>=0.14.0" "transformers>=4.46.0,<4.50" \
    datasets accelerate bitsandbytes matplotlib
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
print(f"SFT dataset size:")
n = sum(1 for _ in open('outputs/sft_dataset/train.jsonl'))
print(f"  {n} examples")
PY

echo "=== [4/5] Run SFT warm-up ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/war_room_sft_v1

python round2/war_room/sft_train.py \
    --model "$MODEL_NAME" \
    --dataset outputs/sft_dataset/train.jsonl \
    --output outputs/war_room_sft_v1 \
    --epochs "$EPOCHS_ARG" \
    --lr "$LR_ARG" \
    --lora-r "$LORA_R_ARG" \
  || { echo "SFT failed with exit $?"; ls -la outputs/war_room_sft_v1/ || true; exit 1; }

echo "=== Artifacts ==="
ls -la outputs/war_room_sft_v1/ || true
if [ -f outputs/war_room_sft_v1/sft_metrics.json ]; then
    echo "--- sft_metrics.json (tail) ---"
    python -c "import json; d=json.load(open('outputs/war_room_sft_v1/sft_metrics.json')); [print(e) for e in d['log_history'][-5:]]"
fi

echo "=== [5/5] Push adapter to HF Hub ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("ADAPTER_REPO_ARG", "brodie1of1/war-room-sft-v1")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/war_room_sft_v1",
    repo_id=repo,
    repo_type="model",
    commit_message="SFT warm-up adapter (multirole format + fault keywords)",
    token=token,
    ignore_patterns=["checkpoint-*/**"],
)
print(f"Artifacts pushed: https://huggingface.co/{repo}")
PY
EOF
)

echo "=== HF Jobs — War Room SFT warm-up ==="
echo "Flavor      : $FLAVOR"
echo "Timeout     : $TIMEOUT"
echo "Model       : $MODEL"
echo "Epochs      : $EPOCHS"
echo "LR          : $LR"
echo "LoRA rank   : $LORA_R"
echo "Adapter     : $ADAPTER_REPO"
echo ""

hf jobs run \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --secrets HF_TOKEN \
    -e MODEL_NAME="$MODEL" \
    -e EPOCHS_ARG="$EPOCHS" \
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
echo "After this completes, launch GRPO on top of the SFT checkpoint."
