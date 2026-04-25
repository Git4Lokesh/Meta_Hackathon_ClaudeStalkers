#!/usr/bin/env bash
#
# Launch the base-vs-trained LLM head-to-head eval on a Hugging Face Job.
#
# Why HF Jobs instead of local / Colab / Inference Endpoint:
#   - Local MLX: 15 GB download over home WiFi is unreliable
#   - Inference Endpoint: A10G/L40S both OOMed on the 7B merged weights
#     when mounted through their managed runtime
#   - Colab T4: works, but slow and flaky when judges re-run
#   - HF Jobs: the SAME L40S that successfully trained the adapter, so we
#     KNOW the model fits. One-shot, exits cleanly.
#
# Hardware: l40sx1 (1x L40S 48GB, $1.80/hr)
# Timeout:  30 minutes hard cap (~$0.90 worst-case spend)
# Scope:    3 tasks × 5 seeds × 2 models (base + base+adapter) = 30 rollouts
#
# Inside the container:
#   1. Clone the repo
#   2. Install deps
#   3. Load Qwen 7B in bf16 + run 15 rollouts via LiveAgentRunner
#   4. Apply LoRA adapter via peft + run another 15 rollouts
#   5. Save results + plot to outputs/llm_eval/
#   6. Upload artifacts to a HF model repo so we can pull them back
#
# Usage:  bash hf_job_llm_eval.sh

set -euo pipefail

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-30m}"
SEEDS="${SEEDS:-11 22 33 44 55}"
TASKS="${TASKS:-task1 task2 task3}"
ADAPTER_REPO="${ADAPTER_REPO:-brodie1of1/war-room-grpo-adapter}"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/6] Clone repo ==="
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 --branch main \
    https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git /workspace/repo
cd /workspace/repo

echo "=== [2/6] Install deps ==="
pip install --quiet --no-cache-dir \
    "trl>=0.15.0,<0.19" "peft>=0.14.0" "transformers>=4.46.0,<4.50" \
    accelerate bitsandbytes \
    fastapi pydantic uvicorn matplotlib "openai>=1.0.0"
pip install --quiet --no-cache-dir -e .

echo "=== [3/6] Environment info ==="
python - <<'PY'
import torch
print(f"Torch:  {torch.__version__}")
print(f"CUDA:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
PY

echo "=== [4/6] Run head-to-head eval ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/llm_eval

# The eval script inside the container spins up a local OpenAI-compatible
# shim around transformers so our existing LiveAgentRunner code path is
# reused unchanged. See round2/war_room/eval_llm_on_gpu.py.
python round2/war_room/eval_llm_on_gpu.py \
    --seeds $SEEDS_ARG \
    --tasks $TASKS_ARG \
    --adapter-repo "$ADAPTER_REPO_ARG" \
    --output-dir outputs/llm_eval \
  || { echo "Eval failed with exit $?"; ls -la outputs/llm_eval/ || true; exit 1; }

echo "=== [5/6] Artifacts ==="
ls -la outputs/llm_eval/ || true
if [ -f outputs/llm_eval/summary.json ]; then
    echo "--- summary.json ---"
    cat outputs/llm_eval/summary.json
fi

echo "=== [6/6] Push results to HF Hub ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("ADAPTER_REPO_ARG", "brodie1of1/war-room-grpo-adapter")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
# Upload results alongside the adapter so it all lives in one place.
api.upload_folder(
    folder_path="outputs/llm_eval",
    path_in_repo="eval",
    repo_id=repo,
    repo_type="model",
    commit_message="Base-vs-trained head-to-head eval results",
    token=token,
)
print(f"✅ Results pushed to https://huggingface.co/{repo}/tree/main/eval")
PY
EOF
)

# Serialize seed / task lists so the inner bash passes them as args
SEEDS_ARG_JOINED=$(echo "$SEEDS" | tr ' ' ' ')
TASKS_ARG_JOINED=$(echo "$TASKS" | tr ' ' ' ')

echo "=== HF Jobs — LLM head-to-head ==="
echo "Flavor   : $FLAVOR"
echo "Timeout  : $TIMEOUT"
echo "Tasks    : $TASKS_ARG_JOINED"
echo "Seeds    : $SEEDS_ARG_JOINED"
echo "Adapter  : $ADAPTER_REPO"
echo ""

hf jobs run \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --secrets HF_TOKEN \
    -e SEEDS_ARG="$SEEDS_ARG_JOINED" \
    -e TASKS_ARG="$TASKS_ARG_JOINED" \
    -e ADAPTER_REPO_ARG="$ADAPTER_REPO" \
    --detach \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    bash -c "$INNER_CMD"

echo ""
echo "Monitor with:  hf jobs logs <JOB_ID>"
echo "After completion, pull results:"
echo "  bash pull_hf_artifacts.sh brodie1of1  # adapter repo now has eval/"
