"""Deploy our trained merged model as a HF Inference Endpoint.

Uses scale-to-zero so it costs $0 when idle and ~$1.30/hr only when
handling requests. The first request after idle takes ~60s to cold-boot.

Usage:
    HF_TOKEN=... python deploy_endpoint.py

Reads from huggingface_hub's cached credentials if HF_TOKEN isn't in env.
"""

from __future__ import annotations

import getpass
import os

from huggingface_hub import create_inference_endpoint


REPO = "brodie1of1/war-room-7b-merged"
ENDPOINT_NAME = "war-room-trained"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not in env. Paste your token (input hidden):")
        token = getpass.getpass("Token: ").strip()
    if not token:
        print("No token provided; aborting.")
        return 1

    print(f"Deploying {REPO} as endpoint '{ENDPOINT_NAME}'...")
    endpoint = create_inference_endpoint(
        name=ENDPOINT_NAME,
        repository=REPO,
        framework="pytorch",
        task="text-generation",
        accelerator="gpu",
        vendor="aws",
        region="us-east-1",
        type="protected",
        instance_type="nvidia-a10g",
        instance_size="x1",
        min_replica=0,
        max_replica=1,
        scale_to_zero_timeout=15,
        token=token,
        custom_image={
            "url": "ghcr.io/huggingface/text-generation-inference:3.0.0",
            "env": {
                "MAX_INPUT_TOKENS": "2048",
                "MAX_TOTAL_TOKENS": "2560",
                "MAX_BATCH_PREFILL_TOKENS": "2560",
            },
        },
    )

    print(f"✅ Endpoint created: {endpoint.name}")
    print(f"   Status: {endpoint.status}")
    print(f"   URL:    {endpoint.url}")
    print()
    print("Scale-to-zero is enabled. Costs ~$0 while idle, ~$1.30/hr when")
    print("serving requests. First call after idle takes ~60s to warm up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
