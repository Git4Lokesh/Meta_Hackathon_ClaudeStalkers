"""Quick endpoint status check."""
import os
import sys
import getpass
from huggingface_hub import get_inference_endpoint


NAME = "war-room-trained"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = getpass.getpass("HF Token: ").strip()
    ep = get_inference_endpoint(NAME, token=token)
    print(f"Name:    {ep.name}")
    print(f"Status:  {ep.status}")
    print(f"URL:     {ep.url}")
    print(f"Repo:    {ep.repository}")
    print(f"Replicas: {ep.raw.get('status', {}).get('targetReplica', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
