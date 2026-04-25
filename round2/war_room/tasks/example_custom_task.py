"""Example: how a user adds their own War Room task.

This is a working, copy-pasteable template that demonstrates the three
extension points of the environment:

  1. Subclass ``WarRoomTaskBase`` (4 methods)
  2. Compose milestones from the named primitive helpers in ``procedural.py``
  3. Register the task in ``WAR_ROOM_TASK_REGISTRY``

After registration the task immediately works with:

  - The OpenEnv API:        POST /reset {"task_id": "example_custom"}
  - The Gradio dashboard:   appears in the task dropdown
  - Deterministic eval:     eval_deterministic.py --tasks example_custom
  - GRPO training:          train_colab.py --tasks example_custom

No changes to the trainer, reward functions, or evaluation harness are
required — the entire pipeline is task-agnostic by design.

Run it::

    PYTHONPATH=. python -c "
    from round2.war_room.environment import WarRoomEnvironment
    env = WarRoomEnvironment()
    obs = env.reset(task_id='example_custom', seed=1)
    print(obs.diagnosis.text[:400])
    "
"""

from __future__ import annotations

from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.models import ServiceRecord

from round2.war_room.grader import MultiAgentGrader
from round2.war_room.tasks.base import WarRoomTaskBase
from round2.war_room.tasks.procedural import (
    diagnosis_says_about,
    service_running,
    triage_mentions,
)


class ExampleCustomTask(WarRoomTaskBase):
    """A 30-line custom task: ``payments_service`` has crashed.

    Demonstrates the minimum surface area for adding a new scenario:
    inject a fault, declare which milestones prove resolution, expose
    an alert configuration. Everything else is automatic.
    """

    task_id = "example_custom"
    name = "Payments Service Crash"
    description = (
        "The payments_service has crashed. Triage must escalate, "
        "diagnosis must inspect the service, remediation must restart it."
    )
    max_rounds = 12
    difficulty = "easy"

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        system = SimulatedSystem()
        # Boot a couple of healthy services
        system.service_registry.services["postgres"] = ServiceRecord(
            name="postgres", status="running", port=5432, dependencies=[],
        )
        # The faulted service: payments_service is crashed
        system.service_registry.services["payments_service"] = ServiceRecord(
            name="payments_service",
            status="crashed",
            port=9000,
            dependencies=["postgres"],
        )
        system.log_buffer.append(
            timestamp=system.current_time,
            severity="ERROR",
            source="payments_service",
            message="payments_service crashed: stripe webhook handler raised KeyError('amount')",
        )
        return system

    def create_grader(self) -> MultiAgentGrader:
        # Composed declaratively from the milestone primitive library —
        # zero lambdas, zero copy-pasted boilerplate.
        return MultiAgentGrader(
            milestones=[
                triage_mentions("payments_service", credit=0.20),
                diagnosis_says_about("payments_service", ["crash", "error"], credit=0.20),
                service_running("payments_service", credit=0.60),
            ],
        )

    def get_alert_config(self) -> dict[str, int]:
        # Triage's dashboard sees the payments_service alert with
        # high prominence — a 3 means "shown first".
        return {"payments_service": 3}
