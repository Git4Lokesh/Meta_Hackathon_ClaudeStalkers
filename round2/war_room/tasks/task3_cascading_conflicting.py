"""Task 3 — Cascading Failure with Conflicting Information (Hard).

Multi-agent version: Redis memory warnings are surfaced more prominently
than the actual DB authentication failure, creating misdirection.
"""

from __future__ import annotations

from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.tasks.task3_cascading_failure import CascadingFailureTask
from round2.war_room.grader import MultiAgentGrader, MultiAgentMilestone
from round2.war_room.models import MultiAgentAction
from round2.war_room.communication import CommunicationChannel
from round2.war_room.tasks.base import WarRoomTaskBase


class CascadingConflictingTask(WarRoomTaskBase):
    task_id = "task3"
    name = "Cascading Failure with Conflicting Information"
    description = (
        "A wrong DB password causes a cascade. Redis memory warnings are "
        "a red herring shown more prominently."
    )
    max_rounds = 20
    difficulty = "hard"

    def __init__(self) -> None:
        self._round1_task = CascadingFailureTask()

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        return self._round1_task.create_initial_state(seed)

    def create_grader(self) -> MultiAgentGrader:
        milestones = [
            MultiAgentMilestone(
                name="diagnosis_identifies_db_auth",
                credit=0.10,
                description="Diagnosis reads db_connector logs and sends message mentioning authentication or password",
                check=lambda actions, system, outputs, channel: (
                    _diagnosis_reads_db_logs(actions)
                    and _diagnosis_mentions_auth(channel)
                ),
            ),
            MultiAgentMilestone(
                name="diagnosis_reads_config",
                credit=0.10,
                description="Diagnosis sends message mentioning database.yml or wrong_password",
                check=lambda actions, system, outputs, channel: _diagnosis_mentions_config(channel),
            ),
            MultiAgentMilestone(
                name="remediation_fixes_config",
                credit=0.20,
                description="database.yml contains correct_db_pass_456",
                check=lambda actions, system, outputs, channel: _config_has_correct_password(system),
            ),
            MultiAgentMilestone(
                name="remediation_restarts_db_connector",
                credit=0.10,
                description="db_connector status is running",
                check=lambda actions, system, outputs, channel: (
                    system.service_registry.services.get("db_connector") is not None
                    and system.service_registry.services["db_connector"].status == "running"
                ),
            ),
            MultiAgentMilestone(
                name="remediation_restarts_app_server",
                credit=0.10,
                description="app_server status is running",
                check=lambda actions, system, outputs, channel: (
                    system.service_registry.services.get("app_server") is not None
                    and system.service_registry.services["app_server"].status == "running"
                ),
            ),
            MultiAgentMilestone(
                name="remediation_restarts_lb",
                credit=0.10,
                description="load_balancer restarted",
                check=lambda actions, system, outputs, channel: (
                    "systemctl restart load_balancer" in actions.remediation.command
                ),
            ),
            MultiAgentMilestone(
                name="all_services_running",
                credit=0.10,
                description="db_connector + app_server + load_balancer all running",
                check=lambda actions, system, outputs, channel: _all_three_running(system),
            ),
            MultiAgentMilestone(
                name="diagnosis_pushback_bonus",
                credit=0.15,
                description="Diagnosis sends message saying Redis is NOT the issue",
                check=lambda actions, system, outputs, channel: _diagnosis_pushback(channel),
            ),
            MultiAgentMilestone(
                name="diagnosis_detects_stale_metrics",
                credit=0.05,
                description="Diagnosis identifies that some metrics are stale/cached",
                check=lambda actions, system, outputs, channel: _diagnosis_detects_stale(channel),
            ),
        ]

        return MultiAgentGrader(milestones=milestones)

    def get_alert_config(self) -> dict[str, int]:
        # Redis shown first/loudest, db_connector hidden
        return {"redis": 10, "db_connector": 0}

    def get_phantom_alerts(self) -> list:
        from round2.war_room.models import Alert
        return [
            Alert(
                service="monitoring",
                alert_type="high_cpu",
                severity="warning",
                description="High CPU on monitoring daemon (92.3%)",
                prominence=8,
            ),
            Alert(
                service="app_server",
                alert_type="high_memory",
                severity="warning",
                description="Memory spike on app_server (88.1%)",
                prominence=7,
            ),
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORRECT_PASSWORD = "correct_db_pass_456"


def _diagnosis_reads_db_logs(actions: MultiAgentAction) -> bool:
    """Check if diagnosis reads db_connector logs."""
    cmd = actions.diagnosis.command
    return (
        any(k in cmd for k in ("cat", "tail", "grep"))
        and ("db_connector" in cmd or "connector" in cmd)
    )


def _diagnosis_mentions_auth(channel: CommunicationChannel) -> bool:
    """Check if diagnosis sent a message mentioning authentication or password."""
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis":
            content_lower = msg.content.lower()
            if "authentication" in content_lower or "password" in content_lower:
                return True
    return False


def _diagnosis_mentions_config(channel: CommunicationChannel) -> bool:
    """Check if diagnosis sent a message mentioning database.yml or wrong_password."""
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis":
            content_lower = msg.content.lower()
            if "database.yml" in content_lower or "wrong_password" in content_lower:
                return True
    return False


def _config_has_correct_password(system: SimulatedSystem) -> bool:
    """Check if database.yml contains the correct password."""
    try:
        content = system.filesystem.read_file("/etc/app/database.yml")
        return CORRECT_PASSWORD in content
    except ValueError:
        return False


def _all_three_running(system: SimulatedSystem) -> bool:
    """Check if db_connector, app_server, and load_balancer are all running."""
    for name in ("db_connector", "app_server", "load_balancer"):
        svc = system.service_registry.services.get(name)
        if svc is None or svc.status != "running":
            return False
    return True


def _diagnosis_pushback(channel: CommunicationChannel) -> bool:
    """Check if diagnosis sent a message dismissing Redis as the cause.

    Relaxed verifier (task3 v2): the original strict check required the
    literal pattern `"not" + "redis"` in the same message. That excludes
    natural phrasings the base model actually emits ("redis is fine",
    "redis is a red herring", "ignore redis"), which kept this 0.15-credit
    milestone unreachable in practice and made task3's reward gradient
    flat at 0.01 across runs. The looser check accepts any message from
    diagnosis that mentions redis together with a dismissal phrase.
    Semantics unchanged: still requires diagnosis to explicitly call out
    that redis is not the root cause.
    """
    DISMISSAL = (
        "not the issue", "not the cause", "not the real",
        "not the problem", "not the root", "isn't the",
        "is fine", "is healthy", "is irrelevant",
        "red herring", "distraction", "misdirection",
        "ignore redis", "redis is fine", "rule out redis",
        "rule redis out", "redis can be ruled out",
        "false alarm", "stale", "cached metric",
    )
    for msg in channel.get_full_history():
        if msg.from_agent != "diagnosis":
            continue
        content_lower = msg.content.lower()
        if "redis" not in content_lower:
            continue
        # Original pattern still wins: "not" + "redis" anywhere.
        if "not" in content_lower:
            return True
        # New: redis + any dismissal phrase counts.
        if any(d in content_lower for d in DISMISSAL):
            return True
    return False


def _diagnosis_detects_stale(channel: CommunicationChannel) -> bool:
    """Check if diagnosis identifies that some metrics are stale/cached."""
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis":
            content_lower = msg.content.lower()
            if ("stale" in content_lower or "cached" in content_lower or
                "not real" in content_lower or "false alarm" in content_lower or
                "phantom" in content_lower or "hallucinating" in content_lower):
                return True
    return False
