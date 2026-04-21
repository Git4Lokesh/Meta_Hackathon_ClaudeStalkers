"""SRE Environment Client.

Provides a simple interface for interacting with the SRE environment,
both locally and (in the future) via HTTP/WebSocket.
"""

from __future__ import annotations
from typing import Optional, Any

from sre_env.server.models import SREAction, SREObservation, SREState
from sre_env.server.sre_environment import SREEnvironment


class SREClient:
    """Client for the SRE Incident Response environment.
    
    For local usage, wraps SREEnvironment directly.
    Can be extended to support remote HTTP/WebSocket connections.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """Initialize client. If no base_url, uses local environment."""
        self._env = SREEnvironment()
        self._base_url = base_url
    
    def reset(self, task_id: str = "task1", seed: int = 42, **kwargs) -> SREObservation:
        """Reset the environment for a task."""
        return self._env.reset(task_id=task_id, seed=seed, **kwargs)
    
    def step(self, command: str, agent_role: str = "system") -> SREObservation:
        """Execute a command and return observation."""
        return self._env.step(SREAction(command=command, agent_role=agent_role))
    
    def execute_command(self, command: str, agent_role: str = "system") -> str:
        """Convenience: execute command and return just the output string."""
        obs = self.step(command, agent_role=agent_role)
        return obs.output
    
    def get_system_overview(self) -> dict:
        """Get system status summary."""
        return self._env.get_system_overview()
    
    def get_available_commands(self) -> list:
        """List supported commands."""
        return self._env.get_available_commands()
    
    @property
    def state(self) -> SREState:
        return self._env.state
