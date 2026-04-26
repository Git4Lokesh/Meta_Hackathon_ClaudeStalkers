#!/usr/bin/env bash
#
# Capture side-by-side rollout traces for base vs trained on a specific
# (task, seed) pair. Output is a single JSON file with per-round
# observations + completions + parsed actions for each model.
#
# Used to generate the blog/README 'worked example' — the one place
# judges see literal verbatim model output.
#
# Target pair: task2 seed=33  (where v3 beats base 0.25 vs 0.04, the
# biggest relative delta in our head-to-head eval).
#
# Runtime: ~5 min on L40S
# Cost:    ~$0.20

set -euo pipefail

FLAVOR="${FLAVOR:-l40sx1}"
TIMEOUT="${TIMEOUT:-15m}"
TASK="${TASK:-task2}"
SEED="${SEED:-33}"
ADAPTER_REPO="${ADAPTER_REPO:-brodie1of1/war-room-grpo-adapter-v3}"
UPLOAD_REPO="${UPLOAD_REPO:-brodie1of1/war-room-eval-results}"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail

echo "=== [1/5] Clone repo ==="
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 --branch main \
    https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git /workspace/repo
cd /workspace/repo

echo "=== [2/5] Install deps ==="
pip install --quiet --no-cache-dir \
    "trl>=0.15.0,<0.19" "peft>=0.14.0" "transformers>=4.46.0,<4.50" \
    accelerate bitsandbytes \
    fastapi pydantic uvicorn matplotlib
pip install --quiet --no-cache-dir -e .

echo "=== [3/5] GPU info ==="
python - <<'PY'
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo "=== [4/5] Capture rollout ==="
export PYTHONPATH=/workspace/repo
mkdir -p outputs/worked_example
python round2/war_room/capture_rollout.py \
    --task "$TASK_ARG" \
    --seed "$SEED_ARG" \
    --adapter-repo "$ADAPTER_REPO_ARG" \
    --output-dir outputs/worked_example

ls -la outputs/worked_example/

echo "=== [5/5] Push to HF ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo = os.environ.get("UPLOAD_REPO_ARG", "brodie1of1/war-room-eval-results")
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
create_repo(repo, repo_type="model", exist_ok=True, token=token)
api.upload_folder(
    folder_path="outputs/worked_example",
    path_in_repo="worked_example",
    repo_id=repo,
    repo_type="model",
    commit_message="Worked example rollout traces",
    token=token,
)
print(f"Pushed to https://huggingface.co/{repo}")
PY
EOF
)

echo "Launching rollout capture..."
hf jobs run \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --secrets HF_TOKEN \
    -e TASK_ARG="$TASK" \
    -e SEED_ARG="$SEED" \
    -e ADAPTER_REPO_ARG="$ADAPTER_REPO" \
    -e UPLOAD_REPO_ARG="$UPLOAD_REPO" \
    --detach \
    pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    bash -c "$INNER_CMD"
