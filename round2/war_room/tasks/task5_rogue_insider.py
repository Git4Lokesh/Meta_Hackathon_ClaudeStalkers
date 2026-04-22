"""Task 5 — Rogue Insider: Corrupted Logs & Config Sabotage (Expert).

A rogue insider has:
1. Changed the app_server's config to point to a malicious external host
2. Injected fake "healthy" entries into log files to hide the evidence
3. Disabled the monitoring service to prevent detection

Agents must:
- Notice monitoring is down (Triage can see it on dashboard)
- Discover the tampered logs (Diagnosis finds inconsistencies)
- Find the sabotaged config (Diagnosis reads config files)
- Fix the config and restart services (Remediation)
- Push back on the fake log entries (Theory-of-Mind: logs say "OK" but system isn't)

This tests a NEW theory-of-mind angle: instead of phantom alerts on the
dashboard, the LOGS themselves are corrupted. Diagnosis must recognize
that log evidence is fabricated.
"""

from __future__ import annotations

from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.tasks.task1_service_restart import ServiceRestartTask
from round2.war_room.grader import MultiAgentGrader, MultiAgentMilestone
from round2.war_room.models import MultiAgentAction, Alert
from round2.war_room.communication import CommunicationChannel
from round2.war_room.tasks.base import WarRoomTaskBase


class RogueInsiderTask(WarRoomTaskBase):
    task_id = "task5"
    name = "Rogue Insider: Corrupted Logs & Config Sabotage"
    description = (
        "A rogue insider has sabotaged the app_server config, injected "
        "fake log entries, and disabled monitoring. Agents must detect "
        "the deception in the logs themselves."
    )
    max_rounds = 20
    difficulty = "expert"

    def __init__(self) -> None:
        self._base_task = ServiceRestartTask()

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        system = self._base_task.create_initial_state(seed)

        # 1. Sabotage: change app_server config to point to malicious host
        system.filesystem.write_file(
            "/etc/app/config.yml",
            "---\n"
            "app_server:\n"
            "  host: malicious-external-host.evil.com\n"
            "  port: 8080\n"
            "  database_url: postgresql://localhost:5432/app\n"
            "  secret_key: COMPROMISED_KEY_12345\n"
            "  # Original host was: 0.0.0.0\n"
        )

        # 2. Inject fake "healthy" log entries to hide the sabotage
        system.filesystem.write_file(
            "/var/log/app_server/access.log",
            "[2024-01-15 03:00:00] INFO  Application started successfully\n"
            "[2024-01-15 03:00:01] INFO  Health check passed: all systems nominal\n"
            "[2024-01-15 03:00:02] INFO  Database connection established\n"
            "[2024-01-15 03:00:03] INFO  Serving traffic on 0.0.0.0:8080\n"
            "[2024-01-15 03:00:04] INFO  Health check passed: all systems nominal\n"
            "[2024-01-15 03:00:05] INFO  Request processed: /api/v1/users [200 OK]\n"
        )

        # 3. The REAL evidence — hidden in syslog
        system.filesystem.write_file(
            "/var/log/syslog",
            "[2024-01-15 02:55:00] WARNING  Unauthorized config change detected by audit daemon\n"
            "[2024-01-15 02:55:01] ALERT  /etc/app/config.yml modified by uid=0 at 02:54:30\n"
            "[2024-01-15 02:55:02] WARNING  app_server host changed from 0.0.0.0 to malicious-external-host.evil.com\n"
            "[2024-01-15 02:55:03] CRITICAL  Secret key rotated without authorization\n"
            "[2024-01-15 02:56:00] ALERT  monitoring service was killed by SIGKILL at 02:55:30\n"
        )

        # 4. Kill monitoring service
        monitoring = system.service_registry.services.get("monitoring")
        if monitoring and monitoring.pid is not None:
            system.kill_process(monitoring.pid)

        # 5. Make app_server degraded (it's pointing to wrong host)
        app_svc = system.service_registry.services.get("app_server")
        if app_svc:
            app_svc.status = "degraded"

        return system

    def create_grader(self) -> MultiAgentGrader:
        milestones = [
            MultiAgentMilestone(
                name="triage_notices_monitoring_down",
                credit=0.05,
                description="Triage notices monitoring is down and escalates",
                check=lambda a, s, o, c: _triage_notices_monitoring(c),
            ),
            MultiAgentMilestone(
                name="diagnosis_reads_syslog",
                credit=0.10,
                description="Diagnosis reads syslog (where real evidence is)",
                check=lambda a, s, o, c: (
                    "syslog" in a.diagnosis.command and
                    any(k in a.diagnosis.command for k in ("cat", "tail", "grep"))
                ),
            ),
            MultiAgentMilestone(
                name="diagnosis_detects_tampered_logs",
                credit=0.15,
                description="Diagnosis sends message about fake/tampered logs",
                check=lambda a, s, o, c: _diagnosis_detects_tampered_logs(c),
            ),
            MultiAgentMilestone(
                name="diagnosis_finds_sabotaged_config",
                credit=0.15,
                description="Diagnosis identifies malicious host in config",
                check=lambda a, s, o, c: _diagnosis_finds_sabotage(c),
            ),
            MultiAgentMilestone(
                name="remediation_fixes_config",
                credit=0.20,
                description="Config restored to safe host",
                check=lambda a, s, o, c: _config_restored(s),
            ),
            MultiAgentMilestone(
                name="remediation_restarts_app_server",
                credit=0.10,
                description="app_server restarted and running",
                check=lambda a, s, o, c: (
                    s.service_registry.services.get("app_server") is not None and
                    s.service_registry.services["app_server"].status == "running"
                ),
            ),
            MultiAgentMilestone(
                name="remediation_restarts_monitoring",
                credit=0.10,
                description="monitoring service restarted",
                check=lambda a, s, o, c: (
                    s.service_registry.services.get("monitoring") is not None and
                    s.service_registry.services["monitoring"].status == "running"
                ),
            ),
            MultiAgentMilestone(
                name="all_services_restored",
                credit=0.10,
                description="app_server + monitoring both running",
                check=lambda a, s, o, c: (
                    _service_running(s, "app_server") and _service_running(s, "monitoring")
                ),
            ),
            MultiAgentMilestone(
                name="diagnosis_log_skepticism_bonus",
                credit=0.05,
                description="Diagnosis explicitly mentions logs are fabricated/fake",
                check=lambda a, s, o, c: _diagnosis_log_skepticism(c),
            ),
        ]
        return MultiAgentGrader(milestones=milestones)

    def get_alert_config(self) -> dict[str, int]:
        return {"monitoring": 5, "app_server": 2}

    def get_phantom_alerts(self) -> list:
        """The fake log entries serve as the deception — no dashboard phantoms needed."""
        return [
            Alert(
                service="app_server",
                alert_type="health_check",
                severity="info",
                description="Health check passed: all systems nominal",
                prominence=6,
            ),
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triage_notices_monitoring(channel: CommunicationChannel) -> bool:
    for msg in channel.get_full_history():
        if msg.from_agent == "triage" and "monitoring" in msg.content.lower():
            return True
    return False


def _diagnosis_detects_tampered_logs(channel: CommunicationChannel) -> bool:
    """Diagnosis mentions that logs are fake/tampered/fabricated."""
    keywords = ["tamper", "fake", "fabricat", "injected", "suspicious",
                 "doesn't match", "don't trust", "misleading", "planted"]
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis":
            content_lower = msg.content.lower()
            if any(k in content_lower for k in keywords):
                return True
    return False


def _diagnosis_finds_sabotage(channel: CommunicationChannel) -> bool:
    """Diagnosis identifies the malicious host in config."""
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis":
            content_lower = msg.content.lower()
            if ("malicious" in content_lower or "evil" in content_lower or
                "sabotage" in content_lower or "unauthorized" in content_lower or
                "compromised" in content_lower or "config" in content_lower):
                if ("host" in content_lower or "config.yml" in content_lower):
                    return True
    return False


def _config_restored(system: SimulatedSystem) -> bool:
    """Check if config has been restored to safe values."""
    try:
        content = system.filesystem.read_file("/etc/app/config.yml")
        return ("0.0.0.0" in content or "localhost" in content) and "malicious" not in content
    except (ValueError, FileNotFoundError):
        return False


def _service_running(system: SimulatedSystem, name: str) -> bool:
    svc = system.service_registry.services.get(name)
    return svc is not None and svc.status == "running"


def _diagnosis_log_skepticism(channel: CommunicationChannel) -> bool:
    """Check if diagnosis explicitly says the logs are not trustworthy."""
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis":
            content_lower = msg.content.lower()
            if any(k in content_lower for k in ("log", "access.log", "entries")):
                if any(k in content_lower for k in ("fake", "fabricat", "tamper",
                       "planted", "not real", "don't trust", "suspicious")):
                    return True
    return False
