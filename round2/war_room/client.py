"""Lightweight HTTP client for the War Room environment server.

Follows the OpenEnv client pattern (like EchoEnv) — thin wrapper around
the FastAPI endpoints exposed by ``round2.war_room.app``.

Usage::

    from round2.war_room.client import WarRoomClient

    client = WarRoomClient("http://localhost:7860")
    obs = client.reset(task_id="task1", seed=42)
    obs = client.step({"triage": {"command": "get_dashboard"}, ...})
    state = client.state()
"""

from __future__ import annotations


class WarRoomClient:
    """Client for interacting with the War Room environment server."""

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "task1", seed: int = 42) -> dict:
        """Reset the environment."""
        import requests

        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict) -> dict:
        """Step the environment with a multi-agent action."""
        import requests

        resp = requests.post(f"{self.base_url}/step", json=action)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        """Get current environment state."""
        import requests

        resp = requests.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Check server health."""
        import requests

        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()
