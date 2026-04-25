"""Delete the failed endpoint and redeploy on a GPU that can actually
fit Qwen 7B.

Previous failure: A10G-small ran OOM at 30GB host RAM during model load.
Qwen2.5-7B in bf16 is 15GB weights + KV + activations ~ 30-35GB peak.

Fix: L40S x1 has 48GB VRAM and 62GB host RAM — the exact hardware we
trained on, confirmed working. $1.80/hr when active, $0 idle.
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
NAMESPACE = "brodie1of1"  # skip whoami lookup that keeps rate-limiting


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = getpass.getpass("HF Token: ").strip()

    # Skip the existence check — it calls whoami which is rate-limited.
    # If an endpoint with this name already exists, the create will fail
    # with a clear error; we can delete manually via the HF web UI.
    print(f"Creating endpoint on L40S (same hardware we trained on) ...")
    ep = create_inference_endpoint(
        name=NAME,
        repository=REPO,
        namespace=NAMESPACE,
        framework="pytorch",
        task="text-generation",
        accelerator="gpu",
        vendor="aws",
        region="us-east-1",
        type="protected",
        instance_type="nvidia-l40s",
        instance_size="x1",
        min_replica=0,
        max_replica=1,
        scale_to_zero_timeout=15,
        token=token,
    )

    print(f"✅ Endpoint created: {ep.name}")
    print(f"   Status:   {ep.status}")
    print(f"   URL:      {ep.url}")
    print()
    print("L40S has 48GB VRAM + 62GB host RAM — fits 7B bf16 comfortably.")
    print("Takes ~5-10 min to boot and load weights.")
    print("Poll status with:  python check_endpoint.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
