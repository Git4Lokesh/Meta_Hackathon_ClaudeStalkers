"""Delete the misconfigured endpoint and redeploy it correctly.

The first deploy passed a `custom_image` that broke MODEL_ID resolution
(TGI defaulted to bloom-560m). We drop that override so HF's standard
endpoint config points TGI at `brodie1of1/war-room-7b-merged` directly.
"""
import os
import sys
import time
import getpass
from huggingface_hub import (
    create_inference_endpoint,
    get_inference_endpoint,
)


NAME = "war-room-trained"
REPO = "brodie1of1/war-room-7b-merged"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = getpass.getpass("HF Token: ").strip()

    # Try to fetch existing endpoint; delete if present
    try:
        ep = get_inference_endpoint(NAME, token=token)
        print(f"Found existing endpoint '{NAME}' (status={ep.status}). Deleting...")
        ep.delete()
        # Give HF a few seconds to finalize the deletion
        time.sleep(10)
        print("Deleted.")
    except Exception as e:
        print(f"No existing endpoint (or fetch failed): {e}")

    print(f"Creating endpoint with default TGI image pointing at {REPO} ...")
    ep = create_inference_endpoint(
        name=NAME,
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
        # no custom_image: let HF pick the right TGI image and set MODEL_ID
    )

    print(f"✅ Endpoint created: {ep.name}")
    print(f"   Status:   {ep.status}")
    print(f"   URL:      {ep.url}")
    print()
    print("Will take ~5-10 min to boot + load the 15GB model.")
    print("Poll status with:  python check_endpoint.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
