"""War room task definitions and registry."""

from round2.war_room.tasks.base import WarRoomTaskBase
from round2.war_room.tasks.task1_coordinated_restart import CoordinatedRestartTask
from round2.war_room.tasks.task2_memory_misdirection import MemoryMisdirectionTask
from round2.war_room.tasks.task3_cascading_conflicting import CascadingConflictingTask
from round2.war_room.tasks.task4_simultaneous import SimultaneousIncidentsTask
from round2.war_room.tasks.task5_rogue_insider import RogueInsiderTask
from round2.war_room.tasks.task6_blame_game import BlameGameTask
from round2.war_room.tasks.procedural import ProceduralTask
from round2.war_room.tasks.example_custom_task import ExampleCustomTask

WAR_ROOM_TASK_REGISTRY: dict[str, type[WarRoomTaskBase]] = {
    "task1": CoordinatedRestartTask,
    "task2": MemoryMisdirectionTask,
    "task3": CascadingConflictingTask,
    "task4": SimultaneousIncidentsTask,
    "task5": RogueInsiderTask,
    "task6": BlameGameTask,
    # Procedural tasks are keyed by difficulty band to allow curriculum sampling.
    # "procedural" is a factory alias used by the environment to instantiate
    # a new procedural task per reset (overriding the base pattern).
    "procedural": ProceduralTask,
    "procedural_easy": lambda: ProceduralTask(difficulty=0.2),
    "procedural_medium": lambda: ProceduralTask(difficulty=0.5),
    "procedural_hard": lambda: ProceduralTask(difficulty=0.9),
    # Worked example: shows how a user adds their own task in ~30 lines.
    "example_custom": ExampleCustomTask,
}

__all__ = [
    "WarRoomTaskBase",
    "CoordinatedRestartTask",
    "MemoryMisdirectionTask",
    "CascadingConflictingTask",
    "SimultaneousIncidentsTask",
    "RogueInsiderTask",
    "BlameGameTask",
    "ProceduralTask",
    "ExampleCustomTask",
    "WAR_ROOM_TASK_REGISTRY",
]
