#!/usr/bin/env bash
# LLM head-to-head eval for v7 full adapter on a high-end HF Job GPU.
#
# HF does not offer H100 in `hf jobs hardware`; this uses **h200** (1× H200) by
# default. Override: FLAVOR=a100-large or FLAVOR=l40sx1
#
# Upload: results go to UPLOAD_REPO (default: same adapter repo under eval/).
# If you cannot push to the adapter org, set e.g. UPLOAD_REPO=brodie1of1/war-room-eval-logs
#
# Prereq:  hf auth login  and  HF token with **write** to UPLOAD_REPO
# Usage:    bash hf_job_llm_eval_v7.sh

set -euo pipefail
cd "$(dirname "$0")"

export FLAVOR="${FLAVOR:-h200}"
export TIMEOUT="${TIMEOUT:-45m}"
export ADAPTER_REPO="${ADAPTER_REPO:-GeminiHugger/war-room-grpo-adapter-v7-rewardfix}"
export UPLOAD_REPO="${UPLOAD_REPO:-$ADAPTER_REPO}"
export SEEDS="${SEEDS:-11 22 33 44 55}"
export TASKS="${TASKS:-task1 task2 task3}"

echo "v7 eval — adapter: $ADAPTER_REPO"
echo "  flavor: $FLAVOR  (H100 not in catalog; h200 = closest fast GPU)"
echo "  upload: $UPLOAD_REPO/eval/"

exec bash hf_job_llm_eval.sh
