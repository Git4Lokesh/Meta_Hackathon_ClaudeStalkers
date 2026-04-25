"""Probe the endpoint directly to see what's wrong."""
import os, sys, getpass, json
import requests
from huggingface_hub import get_inference_endpoint


URL = "https://k6cu78bokhtwi9ns.us-east-1.aws.endpoints.huggingface.cloud"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = getpass.getpass("HF Token: ").strip()

    ep = get_inference_endpoint("war-room-trained", token=token)
    print("=" * 60)
    print(f"Name:   {ep.name}")
    print(f"Status: {ep.status}")
    print(f"URL:    {ep.url}")
    print()
    print("Raw status dict:")
    print(json.dumps(ep.raw.get("status", {}), indent=2))

    headers = {"Authorization": f"Bearer {token}"}
    print()
    print(f"Probing {URL}/health ...")
    try:
        r = requests.get(f"{URL}/health", headers=headers, timeout=10)
        print(f"  /health status: {r.status_code}")
        print(f"  body (first 200 chars): {r.text[:200]!r}")
    except Exception as e:
        print(f"  /health failed: {e}")

    print()
    print(f"Probing {URL}/info ...")
    try:
        r = requests.get(f"{URL}/info", headers=headers, timeout=10)
        print(f"  /info status: {r.status_code}")
        print(f"  body (first 500 chars): {r.text[:500]!r}")
    except Exception as e:
        print(f"  /info failed: {e}")

    print()
    print(f"Probing {URL}/v1/chat/completions with minimal request ...")
    try:
        r = requests.post(
            f"{URL}/v1/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": "tgi",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            },
            timeout=60,
        )
        print(f"  status: {r.status_code}")
        print(f"  body (first 500 chars): {r.text[:500]!r}")
    except Exception as e:
        print(f"  failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
