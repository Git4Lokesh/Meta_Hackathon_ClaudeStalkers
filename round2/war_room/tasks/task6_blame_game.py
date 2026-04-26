"""Task 6 — The Blame Game: Conflicting Agent Reports (Expert+).

A subtle config error chain where:
1. A cron job updated /etc/resolv.conf with wrong DNS servers → services can't resolve hosts
2. Multiple services fail intermittently with DIFFERENT error messages
3. Each service's logs suggest a different root cause (network, auth, timeout)

The key challenge: three red-herring explanations compete, and
agents must resist anchoring on the FIRST explanation they find.

Theory-of-Mind twist: Each agent's initial investigation will find
a plausible but WRONG explanation. They must communicate, find the
pattern (all errors → DNS), and converge on the real root cause.

This is the hardest ToM test:
- Triage says "network is down" (wrong — network is fine, DNS is bad)
- Diagnosis first reads nginx logs → "upstream timeout" (symptom, not cause)
- Diagnosis must go deeper → check resolv.conf → find the DNS issue
- All three agents must update their beliefs when evidence changes
"""

from __future__ import annotations

from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.tasks.task1_service_restart import ServiceRestartTask
from round2.war_room.grader import MultiAgentGrader, MultiAgentMilestone
from round2.war_room.models import MultiAgentAction, Alert
from round2.war_room.communication import CommunicationChannel
from round2.war_room.tasks.base import WarRoomTaskBase


class BlameGameTask(WarRoomTaskBase):
    task_id = "task6"
    name = "The Blame Game: Conflicting Agent Reports"
    description = (
        "DNS poisoning causes cascading failures with conflicting symptoms. "
        "Each service logs a different error. Agents must resist anchoring "
        "on the first explanation and find the common root cause."
    )
    max_rounds = 25
    difficulty = "expert+"

    def __init__(self) -> None:
        self._base_task = ServiceRestartTask()

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        system = self._base_task.create_initial_state(seed)

        # 1. Sabotage DNS — the actual root cause
        system.filesystem.write_file(
            "/etc/resolv.conf",
            "# Modified by cron job at 02:30\n"
            "nameserver 10.255.255.1\n"  # Non-existent DNS server
            "nameserver 10.255.255.2\n"
            "# Previous: nameserver 8.8.8.8\n"
            "# Previous: nameserver 8.8.4.4\n"
        )

        # 2. Inject conflicting error logs — each suggests a different cause

        # nginx logs suggest "upstream timeout" (symptom, not cause)
        system.filesystem.write_file(
            "/var/log/nginx/error.log",
            "[2024-01-15 03:00:01] ERROR upstream connection timed out (110)\n"
            "[2024-01-15 03:00:02] WARN  no live upstreams while connecting to upstream\n"
            "[2024-01-15 03:00:03] ERROR connect() failed: Connection timed out to app_server:8080\n"
            "[2024-01-15 03:00:04] CRIT  upstream prematurely closed connection\n"
            "[2024-01-15 03:00:05] ERROR failed to resolve host name 'app-backend.internal.local'\n"
        )

        # app_server logs suggest "database auth failure" (misleading)
        system.filesystem.write_file(
            "/var/log/app_server/error.log",
            "[2024-01-15 03:00:01] ERROR Failed to connect to database at db.internal.local:5432\n"
            "[2024-01-15 03:00:01] ERROR java.net.UnknownHostException: db.internal.local\n"
            "[2024-01-15 03:00:02] WARN  Connection pool exhausted — no available connections\n"
            "[2024-01-15 03:00:03] ERROR Service degraded: cannot reach any external dependency\n"
            "[2024-01-15 03:00:04] INFO  Retrying connection to db.internal.local (attempt 5/5)\n"
        )

        # redis logs suggest "replication failure" (another red herring)
        system.filesystem.write_file(
            "/var/log/redis/redis.log",
            "[2024-01-15 03:00:01] WARNING  Unable to connect to replica at redis-replica.internal.local:6380\n"
            "[2024-01-15 03:00:02] ERROR  MASTER aborted replication: sync failed\n"
            "[2024-01-15 03:00:03] WARNING  Background save failed with error: network unreachable\n"
            "[2024-01-15 03:00:04] ERROR  Failed to resolve redis-replica.internal.local\n"
        )

        # syslog has the real clue — mentions DNS and resolv.conf
        system.filesystem.write_file(
            "/var/log/syslog",
            "[2024-01-15 02:30:00] INFO  cron.daily: running dns-update.sh\n"
            "[2024-01-15 02:30:01] NOTICE /etc/resolv.conf updated by cron job\n"
            "[2024-01-15 02:30:02] WARNING DNS resolution failing for all .internal.local domains\n"
            "[2024-01-15 02:45:00] ERROR systemd-resolved: failed to resolve 'app-backend.internal.local': NXDOMAIN\n"
            "[2024-01-15 02:50:00] ERROR systemd-resolved: failed to resolve 'db.internal.local': SERVFAIL\n"
            "[2024-01-15 02:55:00] ERROR systemd-resolved: failed to resolve 'redis-replica.internal.local': NXDOMAIN\n"
            "[2024-01-15 03:00:00] CRITICAL Multiple services failing DNS resolution — check /etc/resolv.conf\n"
        )

        # Make services degraded (running but returning errors)
        for svc_name in ("app_server", "nginx"):
            svc = system.service_registry.services.get(svc_name)
            if svc:
                svc.status = "degraded"

        return system

    def create_grader(self) -> MultiAgentGrader:
        # Per-grader (per-episode) state. The previous module-level _logs_read
        # set persisted across episodes during training, causing later episodes
        # to start with the milestone already satisfied. Bind a fresh set
        # to each grader instance instead.
        logs_read: set[str] = set()

        def _diag_reads_multi(actions: MultiAgentAction) -> bool:
            cmd = actions.diagnosis.command.lower()
            if any(k in cmd for k in ("cat", "tail", "grep")):
                for log_type in ("nginx", "app_server", "redis", "syslog", "resolv"):
                    if log_type in cmd:
                        logs_read.add(log_type)
            return len(logs_read) >= 2

        milestones = [
            MultiAgentMilestone(
                name="triage_escalates_multiple_failures",
                credit=0.05,
                description="Triage notices multiple services failing and escalates",
                check=lambda a, s, o, c: _triage_notices_multiple(c),
            ),
            MultiAgentMilestone(
                name="diagnosis_reads_multiple_logs",
                credit=0.10,
                description="Diagnosis reads at least 2 different service logs",
                check=lambda a, s, o, c: _diag_reads_multi(a),
            ),
            MultiAgentMilestone(
                name="diagnosis_notices_pattern",
                credit=0.15,
                description="Diagnosis notices 'resolve' or 'DNS' pattern across logs",
                check=lambda a, s, o, c: _diagnosis_notices_dns_pattern(c),
            ),
            MultiAgentMilestone(
                name="diagnosis_reads_resolv_conf",
                credit=0.10,
                description="Diagnosis reads /etc/resolv.conf",
                check=lambda a, s, o, c: "resolv.conf" in a.diagnosis.command,
            ),
            MultiAgentMilestone(
                name="diagnosis_identifies_root_cause",
                credit=0.15,
                description="Diagnosis messages that DNS/resolv.conf is the root cause",
                check=lambda a, s, o, c: _diagnosis_identifies_dns(c),
            ),
            MultiAgentMilestone(
                name="diagnosis_pushback_on_initial_theory",
                credit=0.10,
                description="Diagnosis pushes back on wrong initial theory (upstream timeout, auth, etc.)",
                check=lambda a, s, o, c: _diagnosis_pushback_theory(c),
            ),
            MultiAgentMilestone(
                name="remediation_fixes_dns",
                credit=0.20,
                description="resolv.conf restored with correct DNS servers",
                check=lambda a, s, o, c: _dns_fixed(s),
            ),
            MultiAgentMilestone(
                name="remediation_restarts_services",
                credit=0.10,
                description="Degraded services restarted after DNS fix",
                check=lambda a, s, o, c: (
                    _service_running(s, "nginx") and _service_running(s, "app_server")
                ),
            ),
            MultiAgentMilestone(
                name="belief_convergence_bonus",
                credit=0.05,
                description="All agents agree on DNS as root cause (visible in messages)",
                check=lambda a, s, o, c: _belief_convergence(c),
            ),
        ]
        return MultiAgentGrader(milestones=milestones)

    def get_alert_config(self) -> dict[str, int]:
        # All services show warnings — deliberately confusing
        return {"nginx": 5, "app_server": 5, "redis": 5}

    def get_phantom_alerts(self) -> list:
        return [
            Alert(
                service="network",
                alert_type="connectivity",
                severity="warning",
                description="Intermittent network connectivity issues detected",
                prominence=9,  # Very prominent — misleading!
            ),
            Alert(
                service="database",
                alert_type="connection_pool",
                severity="warning",
                description="Database connection pool exhausted (0/20 available)",
                prominence=8,
            ),
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triage_notices_multiple(channel: CommunicationChannel) -> bool:
    for msg in channel.get_full_history():
        if msg.from_agent == "triage":
            cl = msg.content.lower()
            services_mentioned = sum(1 for s in ("nginx", "app_server", "redis", "database")
                                     if s in cl or s.replace("_", " ") in cl)
            if services_mentioned >= 2 or "multiple" in cl or "several" in cl:
                return True
    return False


def _diagnosis_notices_dns_pattern(channel: CommunicationChannel) -> bool:
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis":
            cl = msg.content.lower()
            if ("dns" in cl or "resolve" in cl or "resolv" in cl or
                "name resolution" in cl or "nxdomain" in cl):
                return True
    return False


def _diagnosis_identifies_dns(channel: CommunicationChannel) -> bool:
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis":
            cl = msg.content.lower()
            if ("root cause" in cl or "found" in cl or "the problem is" in cl or
                "the issue is" in cl or "real cause" in cl):
                if "dns" in cl or "resolv.conf" in cl or "nameserver" in cl:
                    return True
    return False


def _diagnosis_pushback_theory(channel: CommunicationChannel) -> bool:
    """Check if diagnosis pushes back on a wrong initial theory."""
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis":
            cl = msg.content.lower()
            if "not" in cl and any(k in cl for k in
                ("upstream", "timeout", "auth", "network", "database", "replication")):
                return True
            if any(k in cl for k in ("red herring", "symptom", "misleading", "not the cause")):
                return True
    return False


def _dns_fixed(system: SimulatedSystem) -> bool:
    try:
        content = system.filesystem.read_file("/etc/resolv.conf")
        return "8.8.8.8" in content or "1.1.1.1" in content
    except (ValueError, FileNotFoundError):
        return False


def _service_running(system: SimulatedSystem, name: str) -> bool:
    svc = system.service_registry.services.get(name)
    return svc is not None and svc.status == "running"


def _belief_convergence(channel: CommunicationChannel) -> bool:
    """Check if multiple agents acknowledge DNS as the root cause."""
    agents_agree = set()
    for msg in channel.get_full_history():
        cl = msg.content.lower()
        if "dns" in cl or "resolv" in cl or "nameserver" in cl:
            agents_agree.add(msg.from_agent)
    return len(agents_agree) >= 2
