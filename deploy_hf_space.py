"""Deploy the current repo to the brodie1of1/war-room HF Space.

Uses huggingface_hub's API rather than git push so we don't have to
wrangle credentials on the command line. The HF_TOKEN env var is read
automatically.

Usage:
    source .venv/bin/activate
    python deploy_hf_space.py

This uploads the current working tree (minus .git, __pycache__, etc.)
to the Space, which triggers a Docker build on HF's side.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


SPACE_ID = "brodie1of1/war-room"
REPO_ROOT = Path(__file__).parent

IGNORE_PATTERNS = [
    ".git/**",
    ".git",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    ".pytest_cache/**",
    ".venv/**",
    "*.egg-info/**",
    "outputs/war_room_grpo/adapter_model.safetensors",
    "outputs/war_room_grpo/checkpoint-*/**",
    "outputs/war_room_grpo/tokenizer.json",
    "outputs/war_room_grpo/vocab.json",
    "outputs/war_room_grpo/merges.txt",
    "outputs/sft_dataset*.json",
    "outputs/war_room_sft/**",
    "/tmp/**",
    ".kiro/**",
]


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
    if not token:
        print(
            "ERROR: HF_TOKEN is not set. Run:\n"
            "  export HF_TOKEN='your_token_here'\n"
            "before retrying.",
            file=sys.stderr,
        )
        return 1

    api = HfApi(token=token)

    print(f"Ensuring Space {SPACE_ID} exists...")
    create_repo(
        SPACE_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=token,
    )

    print(f"Uploading repo from {REPO_ROOT} -> {SPACE_ID}...")
    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Deploy Gradio demo + OpenEnv API (unified app)",
        token=token,
        ignore_patterns=IGNORE_PATTERNS,
    )

    print("\n✅ Deployment triggered.")
    print(f"   View:  https://huggingface.co/spaces/{SPACE_ID}")
    print(f"   Logs:  https://huggingface.co/spaces/{SPACE_ID}/logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
