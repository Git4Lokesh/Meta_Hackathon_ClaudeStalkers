"""Task 1 — Coordinated Service Restart (Easy).

Multi-agent version: three agents must coordinate to identify and restart
a crashed nginx service through communication.
"""

from __future__ import annotations

from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.tasks.task1_service_restart import ServiceRestartTask
from round2.war_room.grader import MultiAgentGrader, MultiAgentMilestone
from round2.war_room.models import MultiAgentAction
from round2.war_room.communication import CommunicationChannel
from round2.war_room.tasks.base import WarRoomTaskBase


class CoordinatedRestartTask(WarRoomTaskBase):
    task_id = "task1"
    name = "Coordinated Service Restart"
    description = (
        "nginx has crashed. Three agents must coordinate to identify, "
        "diagnose, and restart the service."
    )
    max_rounds = 10
    difficulty = "easy"

    def __init__(self) -> None:
        self._round1_task = ServiceRestartTask()

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        return self._round1_task.create_initial_state(seed)

    def create_grader(self) -> MultiAgentGrader:
        milestones = [
            MultiAgentMilestone(
                name="triage_escalates_nginx",
                credit=0.10,
                description="Triage sends message mentioning nginx to diagnosis",
                check=lambda actions, system, outputs, channel: _triage_mentions_nginx(channel),
            ),
            MultiAgentMilestone(
                name="diagnosis_reads_logs",
                credit=0.15,
                description="Diagnosis reads nginx error log",
                check=lambda actions, system, outputs, channel: (
                    any(k in actions.diagnosis.command for k in ("cat", "tail", "grep"))
                    and "nginx" in actions.diagnosis.command
                    and ("error" in actions.diagnosis.command or "log" in actions.diagnosis.command)
                ),
            ),
            MultiAgentMilestone(
                name="diagnosis_messages_findings",
                credit=0.15,
                description="Diagnosis sends message to remediation mentioning nginx and crash/restart",
                check=lambda actions, system, outputs, channel: _diagnosis_messages_findings(channel),
            ),
            MultiAgentMilestone(
                name="remediation_restarts_nginx",
                credit=0.40,
                description="nginx service status becomes running",
                check=lambda actions, system, outputs, channel: (
                    system.service_registry.services.get("nginx") is not None
                    and system.service_registry.services["nginx"].status == "running"
                ),
            ),
            MultiAgentMilestone(
                name="verification",
                credit=0.20,
                description="Any agent verifies nginx is running",
                check=lambda actions, system, outputs, channel: (
                    system.service_registry.services.get("nginx") is not None
                    and system.service_registry.services["nginx"].status == "running"
                    and _any_agent_verifies_nginx(actions)
                ),
            ),
        ]

        return MultiAgentGrader(milestones=milestones)

    def get_alert_config(self) -> dict[str, int]:
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triage_mentions_nginx(channel: CommunicationChannel) -> bool:
    """Check if triage sent a message mentioning nginx."""
    for msg in channel.get_full_history():
        if msg.from_agent == "triage" and "nginx" in msg.content.lower():
            return True
    return False


def _diagnosis_messages_findings(channel: CommunicationChannel) -> bool:
    """Check if diagnosis sent a message to remediation mentioning nginx and crash/restart."""
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis" and msg.to_agent in ("remediation", "all"):
            content_lower = msg.content.lower()
            if "nginx" in content_lower and ("crash" in content_lower or "restart" in content_lower):
                return True
    return False


def _any_agent_verifies_nginx(actions: MultiAgentAction) -> bool:
    """Check if any agent runs a verification command for nginx."""
    for role in ["triage", "diagnosis", "remediation"]:
        cmd = getattr(actions, role).command
        if ("curl" in cmd and ("80" in cmd or "localhost" in cmd)) or \
           cmd.strip() == "systemctl status nginx":
            return True
    return False
