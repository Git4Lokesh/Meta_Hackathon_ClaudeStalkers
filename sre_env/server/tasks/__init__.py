"""SRE task definitions.

Exports all task classes and a TASK_REGISTRY mapping task_id → task class.
"""

from sre_env.server.tasks.task1_service_restart import ServiceRestartTask
from sre_env.server.tasks.task2_memory_leak import MemoryLeakTask
from sre_env.server.tasks.task3_cascading_failure import CascadingFailureTask

__all__ = [
    "ServiceRestartTask",
    "MemoryLeakTask",
    "CascadingFailureTask",
    "TASK_REGISTRY",
]

TASK_REGISTRY: dict[str, type] = {
    "task1": ServiceRestartTask,
    "task2": MemoryLeakTask,
    "task3": CascadingFailureTask,
}
