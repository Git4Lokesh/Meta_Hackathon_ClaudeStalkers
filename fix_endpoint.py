"""Fix the endpoint to actually load our trained model.

The initial deploy used a custom_image without MODEL_ID pointing at our
repo, so TGI defaulted to bloom-560m. This script updates the endpoint
to use TGI's default image (which respects the `repository` field) and
sizes the max tokens appropriately for Qwen2.5-7B.
"""
import os
import sys
import getpass
from huggingface_hub import get_inference_endpoint


NAME = "war-room-trained"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = getpass.getpass("HF Token: ").strip()

    print(f"Fetching current endpoint '{NAME}' ...")
    ep = get_inference_endpoint(NAME, token=token)
    print(f"  Current status: {ep.status}")
    print(f"  Current repo:   {ep.repository}")
    print()

    print("Updating endpoint to use TGI default image with correct MODEL_ID ...")
    # `update()` without custom_image clears the override and lets HF
    # pick the right TGI version for the repo. We set env vars that TGI
    # honors for sizing.
    ep = ep.update(
        custom_image={
            "url": "ghcr.io/huggingface/text-generation-inference:3.0.0",
            "env": {
                "MODEL_ID": "/repository",  # TGI convention: /repository is mounted repo
                "MAX_INPUT_TOKENS": "2048",
                "MAX_TOTAL_TOKENS": "2560",
                "MAX_BATCH_PREFILL_TOKENS": "2560",
            },
        },
    )
    print(f"  New status: {ep.status}")
    print(f"  Endpoint will restart; takes ~3-5 min to reload the model.")
    print(f"  Monitor with: python check_endpoint.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
