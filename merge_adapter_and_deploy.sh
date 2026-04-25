#!/usr/bin/env bash
#
# Merge our LoRA adapter into the Qwen2.5-7B-Instruct base model and push
# the merged weights to a new HF model repo. This is required because
# HF Inference Endpoints need a full model, not an adapter.
#
# Strategy: run as an HF Job on a cheap GPU (l4x1, $0.80/hr). The merge
# itself takes ~5 min. We set a 20m timeout as safety net — max spend ~$0.27.
#
# Output: brodie1of1/war-room-7b-merged (full model repo)
#
# After this, we can deploy that repo as an Inference Endpoint on A10G
# with autoscale-to-zero.

set -euo pipefail

FLAVOR="${FLAVOR:-l4x1}"
TIMEOUT="${TIMEOUT:-20m}"
ADAPTER_REPO="${ADAPTER_REPO:-brodie1of1/war-room-grpo-adapter}"
MERGED_REPO="${MERGED_REPO:-brodie1of1/war-room-7b-merged}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/4] Install dependencies ==="
pip install --quiet --no-cache-dir \
    "transformers>=4.46.0,<4.50" "peft>=0.14.0,<0.19" \
    "accelerate>=0.30" "safetensors>=0.4.0" \
    "huggingface_hub>=0.24.0" "torch>=2.4.0"

echo "=== [2/4] Merge LoRA adapter into base model ==="
python - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.environ["BASE_MODEL"]
ADAPTER = os.environ["ADAPTER_REPO"]
OUT = "/workspace/merged"

print(f"Loading base model: {BASE}")
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto",
)

print(f"Loading adapter: {ADAPTER}")
model = PeftModel.from_pretrained(model, ADAPTER)

print("Merging LoRA weights into base...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUT}")
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")
tokenizer.save_pretrained(OUT)
print("✅ Merge complete")
PY

echo "=== [3/4] Push merged model to HF Hub ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ["MERGED_REPO"]
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, private=False, token=token)
api.upload_folder(
    folder_path="/workspace/merged",
    repo_id=repo,
    repo_type="model",
    commit_message="Qwen2.5-7B-Instruct + GRPO LoRA adapter (merged)",
    token=token,
)
print(f"✅ Pushed merged model to https://huggingface.co/{repo}")
PY

echo "=== [4/4] Done ==="
EOF
)

echo "=== Merge launcher ==="
echo "Flavor      : $FLAVOR"
echo "Timeout     : $TIMEOUT"
echo "Adapter     : $ADAPTER_REPO"
echo "Base        : $BASE_MODEL"
echo "Output      : $MERGED_REPO"
echo ""

hf jobs run \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --secrets HF_TOKEN \
    -e ADAPTER_REPO="$ADAPTER_REPO" \
    -e MERGED_REPO="$MERGED_REPO" \
    -e BASE_MODEL="$BASE_MODEL" \
    --detach \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    bash -c "$INNER_CMD"
