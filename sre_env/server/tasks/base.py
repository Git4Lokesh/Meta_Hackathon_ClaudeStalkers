"""TaskDefinitionBase: abstract base class for task scenario definitions."""

from abc import ABC, abstractmethod

from sre_env.server.simulated_system import SimulatedSystem


class TaskDefinitionBase(ABC):
    """Base class for task scenario definitions.

    Each concrete task (e.g. ServiceRestart, MemoryLeak, CascadingFailure)
    subclasses this and provides:
    * ``create_initial_state(seed)`` — deterministic system setup
    * ``create_grader()`` — configured ``TaskGrader`` with milestones/penalties
    """

    task_id: str
    name: str
    description: str
    max_steps: int
    difficulty: str  # "easy", "medium", "hard"

    @abstractmethod
    def create_initial_state(self, seed: int) -> SimulatedSystem:
        """Build the SimulatedSystem with task-specific fault injection.

        Must be deterministic given the same seed.
        """
        ...

    @abstractmethod
    def create_grader(self) -> "TaskGrader":  # noqa: F821
        """Return a configured TaskGrader with milestones and penalties."""
        ...
