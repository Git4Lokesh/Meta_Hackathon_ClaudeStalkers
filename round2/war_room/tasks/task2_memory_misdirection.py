"""Task 2 — Memory Leak with Misdirection (Medium).

Multi-agent version: a high-CPU red herring process distracts from the
real memory leak. Agents must prioritize correctly and communicate
precise diagnostic findings.
"""

from __future__ import annotations

from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.models import ServiceRecord
from sre_env.server.tasks.task2_memory_leak import MemoryLeakTask
from round2.war_room.grader import MultiAgentGrader, MultiAgentMilestone
from round2.war_room.models import MultiAgentAction
from round2.war_room.communication import CommunicationChannel
from round2.war_room.tasks.base import WarRoomTaskBase


class MemoryMisdirectionTask(WarRoomTaskBase):
    task_id = "task2"
    name = "Memory Leak with Misdirection"
    description = (
        "A process is leaking memory while a high-CPU red herring distracts. "
        "Agents must prioritize correctly."
    )
    max_rounds = 15
    difficulty = "medium"

    def __init__(self) -> None:
        self._round1_task = MemoryLeakTask()
        self._leaking_pid: int = 0

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        system = self._round1_task.create_initial_state(seed)
        self._leaking_pid = self._round1_task._leaking_pid

        # Add high-CPU red herring: api_gateway service + process
        system.service_registry.services["api_gateway"] = ServiceRecord(
            name="api_gateway", status="running", port=8082, dependencies=[],
        )
        api_pid = system.process_table.add_process(
            "api_gateway", cpu=88.0, mem=128.0,
            status="running", service_name="api_gateway",
        )
        system.service_registry.services["api_gateway"].pid = api_pid

        return system

    def create_grader(self) -> MultiAgentGrader:
        leaking_pid = self._leaking_pid

        milestones = [
            MultiAgentMilestone(
                name="triage_prioritizes_memory",
                credit=0.10,
                description="Triage sends message mentioning memory or OOM (not just CPU)",
                check=lambda actions, system, outputs, channel: _triage_prioritizes_memory(channel),
            ),
            MultiAgentMilestone(
                name="diagnosis_identifies_pid",
                credit=0.20,
                description="Diagnosis runs ps aux and sends message containing the leaking PID",
                check=lambda actions, system, outputs, channel: (
                    "ps" in actions.diagnosis.command
                    and _diagnosis_sends_pid(channel, leaking_pid)
                ),
            ),
            MultiAgentMilestone(
                name="diagnosis_reads_oom",
                credit=0.10,
                description="Diagnosis reads syslog and output contains OOM",
                check=lambda actions, system, outputs, channel: (
                    any(k in actions.diagnosis.command for k in ("cat", "tail", "grep"))
                    and "syslog" in actions.diagnosis.command
                    and "OOM" in outputs.get("diagnosis", "")
                ),
            ),
            MultiAgentMilestone(
                name="remediation_kills_correct",
                credit=0.30,
                description="Leaking process PID no longer in process table",
                check=lambda actions, system, outputs, channel: (
                    system.process_table.processes.get(leaking_pid) is None
                ),
            ),
            MultiAgentMilestone(
                name="remediation_restarts_service",
                credit=0.20,
                description="data_processor service status is running",
                check=lambda actions, system, outputs, channel: (
                    system.service_registry.services.get("data_processor") is not None
                    and system.service_registry.services["data_processor"].status == "running"
                ),
            ),
            MultiAgentMilestone(
                name="verification",
                credit=0.10,
                description="Verify data_processor is healthy",
                check=lambda actions, system, outputs, channel: (
                    system.service_registry.services.get("data_processor") is not None
                    and system.service_registry.services["data_processor"].status == "running"
                    and _any_agent_verifies_data_processor(actions)
                ),
            ),
        ]

        return MultiAgentGrader(milestones=milestones)

    def get_alert_config(self) -> dict[str, int]:
        # api_gateway shown first (red herring), data_processor hidden
        return {"api_gateway": 5, "data_processor": 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triage_prioritizes_memory(channel: CommunicationChannel) -> bool:
    """Check if triage sent a message mentioning memory or OOM (not just CPU)."""
    for msg in channel.get_full_history():
        if msg.from_agent == "triage":
            content_lower = msg.content.lower()
            if "memory" in content_lower or "oom" in content_lower:
                return True
    return False


def _diagnosis_sends_pid(channel: CommunicationChannel, leaking_pid: int) -> bool:
    """Check if diagnosis sent a message containing the leaking PID."""
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis":
            if str(leaking_pid) in msg.content:
                return True
    return False


def _any_agent_verifies_data_processor(actions: MultiAgentAction) -> bool:
    """Check if any agent runs a verification command for data_processor."""
    for role in ["triage", "diagnosis", "remediation"]:
        cmd = getattr(actions, role).command
        if ("curl" in cmd and "8081" in cmd) or \
           cmd.strip() == "systemctl status data_processor":
            return True
    return False
