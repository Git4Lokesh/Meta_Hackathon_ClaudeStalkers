"""Example of adding a custom War Room task using only the public primitives.

This module exists to demonstrate how to extend the environment **without
touching any existing grader / reward / environment code**. Drop a new
module like this in ``round2/war_room/tasks/``, register it in
``WAR_ROOM_TASK_REGISTRY`` in ``round2/war_room/tasks/__init__.py``, and
the Gradio UI, training loop, and OpenEnv REST API all start serving it
automatically.

The task below injects a disk-full condition on ``api_gateway`` and uses
the three compositional milestone primitives exported from
``procedural.py`` — ``diag_mentions_milestone``,
``triage_mentions_milestone``, and ``service_running_milestone`` — to
define the rubric declaratively.
"""

from __future__ import annotations

import random

from round2.war_room.grader import MultiAgentGrader
from round2.war_room.tasks.base import WarRoomTaskBase
from round2.war_room.tasks.procedural import (
    FaultSpec,
    _apply_disk_full,
    _build_base_system,
    diag_mentions_milestone,
    service_running_milestone,
    triage_mentions_milestone,
)
from sre_env.server.simulated_system import SimulatedSystem


class ApiGatewayDiskFullTask(WarRoomTaskBase):
    """A single-fault disk-full scenario on ``api_gateway``.

    Shows the minimal amount of code you need to author a new task. The
    entire rubric is four lines of composed primitives — no lambdas.
    """

    task_id = "example_disk_full"
    name = "Example: API Gateway Disk Full"
    description = (
        "api_gateway's access log has filled the disk. Agents must detect "
        "the ENOSPC condition, rotate or truncate the log file, and bring "
        "api_gateway back to a healthy state."
    )
    max_rounds = 12
    difficulty = "medium"

    def __init__(self) -> None:
        self._fault = FaultSpec(
            fault_type="disk_full",
            target_service="api_gateway",
            params={
                "disk_percent": 98,
                "log_path": "/var/log/api_gateway/access.log",
            },
        )

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        system = _build_base_system(random.Random(seed))
        _apply_disk_full(system, self._fault)
        return system

    def create_grader(self) -> MultiAgentGrader:
        svc = self._fault.target_service
        return MultiAgentGrader(
            milestones=[
                triage_mentions_milestone(
                    name="triage_flags_disk_full",
                    svc=svc,
                    credit=0.10,
                ),
                diag_mentions_milestone(
                    name="diagnosis_identifies_enospc",
                    svc=svc,
                    keywords=["disk", "space", "enospc", "full"],
                    credit=0.25,
                    must_include_svc=False,
                ),
                service_running_milestone(
                    name="remediation_restores_api_gateway",
                    svc=svc,
                    credit=0.40,
                    description="api_gateway status returns to running",
                ),
            ],
        )

    def get_alert_config(self) -> dict[str, int]:
        # Optional: bump api_gateway's dashboard prominence so Triage sees
        # it first. Leaving the default is fine too.
        return {"api_gateway": 1}
