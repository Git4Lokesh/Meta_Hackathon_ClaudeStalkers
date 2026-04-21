"""Base class for war room task definitions."""
from abc import ABC, abstractmethod
from sre_env.server.simulated_system import SimulatedSystem
from round2.war_room.grader import MultiAgentGrader


class WarRoomTaskBase(ABC):
    task_id: str
    name: str
    description: str
    max_rounds: int
    difficulty: str

    @abstractmethod
    def create_initial_state(self, seed: int) -> SimulatedSystem:
        """Build SimulatedSystem with task-specific fault injection."""
        ...

    @abstractmethod
    def create_grader(self) -> MultiAgentGrader:
        """Return configured MultiAgentGrader with milestones."""
        ...

    @abstractmethod
    def get_alert_config(self) -> dict[str, int]:
        """Return prominence overrides for AlertEngine. {service: prominence}"""
        ...

    def get_phantom_alerts(self) -> list:
        """Return phantom (fake) alerts for theory-of-mind testing. Default: none."""
        return []
