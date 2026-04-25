#!/usr/bin/env bash
# Pull GRPO training artifacts from the HF model repo that the job pushed to.
# Run this AFTER the training job completes successfully.
#
# Usage:  bash pull_hf_artifacts.sh [HF_USERNAME]
#         Defaults to brodie1of1 if not provided.
#
# Result: Files land in outputs/war_room_grpo/ locally.

set -euo pipefail

USER="${1:-brodie1of1}"
REPO_ID="${USER}/war-room-grpo-adapter"
DEST="outputs/war_room_grpo"

echo "Pulling ${REPO_ID} -> ${DEST}/"
mkdir -p "${DEST}"

# hf download puts files in a cache, then we copy/link into place
hf download "${REPO_ID}" --local-dir "${DEST}"

echo ""
echo "=== Artifacts pulled ==="
ls -la "${DEST}"

echo ""
echo "Key files:"
echo "  ${DEST}/adapter_model.safetensors  (LoRA weights)"
echo "  ${DEST}/metrics.json                (training curves data)"
echo "  ${DEST}/training_curves.png         (reward plot)"
echo "  ${DEST}/baseline_vs_trained.png     (comparison plot)"
