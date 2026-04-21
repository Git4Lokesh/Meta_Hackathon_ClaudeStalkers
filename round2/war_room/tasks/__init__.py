"""War room task definitions and registry."""

from round2.war_room.tasks.base import WarRoomTaskBase
from round2.war_room.tasks.task1_coordinated_restart import CoordinatedRestartTask
from round2.war_room.tasks.task2_memory_misdirection import MemoryMisdirectionTask
from round2.war_room.tasks.task3_cascading_conflicting import CascadingConflictingTask
from round2.war_room.tasks.task4_simultaneous import SimultaneousIncidentsTask

WAR_ROOM_TASK_REGISTRY: dict[str, type[WarRoomTaskBase]] = {
    "task1": CoordinatedRestartTask,
    "task2": MemoryMisdirectionTask,
    "task3": CascadingConflictingTask,
    "task4": SimultaneousIncidentsTask,
}

__all__ = [
    "WarRoomTaskBase",
    "CoordinatedRestartTask",
    "MemoryMisdirectionTask",
    "CascadingConflictingTask",
    "SimultaneousIncidentsTask",
    "WAR_ROOM_TASK_REGISTRY",
]
