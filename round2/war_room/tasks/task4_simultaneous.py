"""Task 4 — Simultaneous Incidents (Expert).

Composes Task 1 (nginx crash) + Task 2 (memory leak) into a single
scenario with two independent milestone tracks.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.models import ServiceRecord, MetricPoint
from sre_env.server.tasks.task1_service_restart import ServiceRestartTask
from sre_env.server.tasks.task2_memory_leak import MemoryLeakTask
from round2.war_room.grader import MultiAgentGrader, MultiAgentMilestone
from round2.war_room.models import MultiAgentAction
from round2.war_room.communication import CommunicationChannel
from round2.war_room.tasks.base import WarRoomTaskBase


class SimultaneousIncidentsTask(WarRoomTaskBase):
    task_id = "task4"
    name = "Simultaneous Incidents"
    description = (
        "Two concurrent incidents: nginx crash AND memory-leaking process. "
        "Agents must handle both independently."
    )
    max_rounds = 25
    difficulty = "expert"

    def __init__(self) -> None:
        self._task1 = ServiceRestartTask()
        self._task2 = MemoryLeakTask()
        self._leaking_pid: int = 0

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        """Build a system that has BOTH nginx crashed AND a memory-leaking process."""
        rng = random.Random(seed)
        system = SimulatedSystem()
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        crash_time = base_time - timedelta(minutes=5)
        system.current_time = base_time

        # ---- Services ----
        # nginx crashed (from Task 1)
        system.service_registry.services["nginx"] = ServiceRecord(
            name="nginx", status="crashed", port=80, dependencies=[],
        )
        # data_processor degraded (from Task 2)
        system.service_registry.services["data_processor"] = ServiceRecord(
            name="data_processor", status="degraded", port=8081,
            dependencies=["postgres"],
        )
        # Healthy services
        system.service_registry.services["postgres"] = ServiceRecord(
            name="postgres", status="running", port=5432, dependencies=[],
        )
        system.service_registry.services["redis"] = ServiceRecord(
            name="redis", status="running", port=6379, dependencies=[],
        )
        system.service_registry.services["app_server"] = ServiceRecord(
            name="app_server", status="running", port=8080,
            dependencies=["postgres", "redis"],
        )
        system.service_registry.services["monitoring"] = ServiceRecord(
            name="monitoring", status="running", port=9090, dependencies=[],
        )

        # ---- Processes ----
        # Memory-leaking process (from Task 2)
        leak_mem = rng.uniform(2500.0, 3200.0)
        leak_pid = system.process_table.add_process(
            "data_processor_worker", cpu=45.0, mem=leak_mem,
            status="running", service_name="data_processor",
        )
        system.service_registry.services["data_processor"].pid = leak_pid
        self._leaking_pid = leak_pid

        # nginx has NO process (it crashed)

        # Healthy service processes
        pg_pid = system.process_table.add_process(
            "postgres", cpu=3.0, mem=256.0, status="running", service_name="postgres",
        )
        system.service_registry.services["postgres"].pid = pg_pid

        redis_pid = system.process_table.add_process(
            "redis", cpu=1.0, mem=64.0, status="running", service_name="redis",
        )
        system.service_registry.services["redis"].pid = redis_pid

        app_pid = system.process_table.add_process(
            "app_server", cpu=12.0, mem=512.0, status="running", service_name="app_server",
        )
        system.service_registry.services["app_server"].pid = app_pid

        mon_pid = system.process_table.add_process(
            "monitoring", cpu=2.0, mem=96.0, status="running", service_name="monitoring",
        )
        system.service_registry.services["monitoring"].pid = mon_pid

        # Background processes
        bg_procs = [
            ("cron", 0.1, 8.0), ("sshd", 0.2, 12.0), ("systemd", 0.5, 32.0),
            ("journald", 0.3, 24.0), ("dbus-daemon", 0.1, 6.0),
            ("rsyslogd", 0.2, 16.0), ("networkd", 0.1, 10.0),
            ("resolved", 0.1, 8.0), ("logrotate", 0.0, 4.0), ("atd", 0.0, 2.0),
        ]
        for name, cpu, mem in bg_procs:
            system.process_table.add_process(
                name, cpu=max(0.0, cpu + rng.uniform(-0.05, 0.05)),
                mem=mem, status="sleeping",
            )

        # ---- Filesystem ----
        # nginx error log (from Task 1)
        error_lines = [
            "2024/01/15 09:54:58 [error] 1234#0: *5 open() \"/usr/share/nginx/html/favicon.ico\" failed",
            "2024/01/15 09:55:00 [emerg] worker process exited with signal 11 (core dumped)",
            "2024/01/15 09:55:01 [alert] worker process 1234 exited on signal 11",
            "2024/01/15 09:55:02 [emerg] cannot bind to 0.0.0.0:80 (Address already in use)",
            "2024/01/15 09:55:03 [emerg] master process exiting due to fatal error",
        ]
        system.filesystem.write_file(
            "/var/log/nginx/error.log", "\n".join(error_lines) + "\n",
        )

        # syslog with both nginx crash and OOM entries
        syslog_lines = [
            "Jan 15 09:54:59 server kernel: nginx[1234]: segfault at 0",
            "Jan 15 09:55:00 server systemd[1]: nginx.service: Main process exited, code=killed, status=11/SEGV",
            "Jan 15 09:55:01 server systemd[1]: nginx.service: Failed with result 'signal'.",
            f"Jan 15 09:30:01 server kernel: Out of memory: Kill process {leak_pid} (data_processor_worker) score 900",
            f"Jan 15 09:30:02 server kernel: Killed process {leak_pid} (data_processor_worker) total-vm:3200000kB",
            "Jan 15 09:35:00 server systemd[1]: data_processor.service: Main process exited, code=killed, status=9/KILL",
        ]
        system.filesystem.write_file("/var/log/syslog", "\n".join(syslog_lines) + "\n")

        # data_processor app log (from Task 2)
        dp_log_lines = [
            "2024-01-15 09:20:00 INFO  [main] Data processor started on port 8081",
            "2024-01-15 09:25:00 WARN  [worker] Memory usage at 70%",
            "2024-01-15 09:30:00 ERROR [worker] Memory usage critical at 90%",
            "2024-01-15 09:35:00 FATAL [main] Worker process killed by OOM killer",
        ]
        system.filesystem.write_file(
            "/var/log/data_processor/app.log", "\n".join(dp_log_lines) + "\n",
        )

        # /proc/meminfo showing high usage
        meminfo = """MemTotal:        8192000 kB
MemFree:          512000 kB
MemAvailable:     768000 kB
Buffers:          128000 kB
Cached:           256000 kB
SwapTotal:       2048000 kB
SwapFree:         512000 kB
"""
        system.filesystem.write_file("/proc/meminfo", meminfo)

        # ---- LogBuffer ----
        services_for_logs = ["postgres", "redis", "app_server", "monitoring"]
        normal_msgs = {
            "postgres": ["checkpoint starting: time", "connection received: host=127.0.0.1"],
            "redis": ["Background saving started", "DB saved on disk"],
            "app_server": ["GET /api/health 200 2ms", "POST /api/data 201 15ms"],
            "monitoring": ["Scraping metrics", "Alert rule evaluation complete"],
        }

        t = base_time - timedelta(minutes=30)
        for i in range(40):
            svc = rng.choice(services_for_logs)
            sev = rng.choice(["DEBUG", "INFO", "INFO", "INFO"])
            msg = rng.choice(normal_msgs[svc])
            system.log_buffer.append(timestamp=t, severity=sev, source=svc, message=msg)
            t += timedelta(seconds=rng.randint(20, 60))

        # nginx crash log entries
        system.log_buffer.append(
            timestamp=crash_time, severity="ERROR", source="nginx",
            message="worker process exited with signal 11 (core dumped)",
        )
        system.log_buffer.append(
            timestamp=crash_time + timedelta(seconds=1), severity="FATAL", source="nginx",
            message="master process exiting due to fatal error",
        )

        # OOM / data_processor entries
        oom_t = base_time - timedelta(minutes=25)
        system.log_buffer.append(
            timestamp=oom_t, severity="ERROR", source="data_processor",
            message="Memory usage critical at 90%",
        )
        system.log_buffer.append(
            timestamp=oom_t + timedelta(minutes=5), severity="FATAL", source="data_processor",
            message=f"Out of memory: Kill process {leak_pid} (data_processor_worker) score 900",
        )

        # ---- MetricsStore ----
        for svc_name in ["nginx", "data_processor", "postgres", "redis", "app_server", "monitoring"]:
            t = base_time - timedelta(minutes=35)
            for minute in range(35):
                ts = t + timedelta(minutes=minute)
                if svc_name == "nginx" and ts >= crash_time:
                    point = MetricPoint(
                        timestamp=ts, cpu_percent=0.0, memory_percent=0.0,
                        disk_percent=45.0, network_mbps=0.0,
                    )
                elif svc_name == "data_processor":
                    mem_pct = min(30.0 + (minute * 2.0), 95.0)
                    point = MetricPoint(
                        timestamp=ts, cpu_percent=rng.uniform(30.0, 50.0),
                        memory_percent=mem_pct, disk_percent=45.0,
                        network_mbps=rng.uniform(1.0, 5.0),
                    )
                else:
                    point = MetricPoint(
                        timestamp=ts, cpu_percent=rng.uniform(1.0, 15.0),
                        memory_percent=rng.uniform(2.0, 10.0),
                        disk_percent=45.0, network_mbps=rng.uniform(0.5, 3.0),
                    )
                system.metrics_store.add_point(svc_name, point)

        return system

    def create_grader(self) -> MultiAgentGrader:
        leaking_pid = self._leaking_pid

        milestones = [
            # ---- Nginx track (0.30 total) ----
            MultiAgentMilestone(
                name="triage_escalates_nginx",
                credit=0.10,
                description="Triage sends message mentioning nginx",
                check=lambda actions, system, outputs, channel: _triage_mentions_nginx(channel),
            ),
            MultiAgentMilestone(
                name="diagnosis_reads_nginx_logs",
                credit=0.10,
                description="Diagnosis reads nginx error logs",
                check=lambda actions, system, outputs, channel: (
                    any(k in actions.diagnosis.command for k in ("cat", "tail", "grep"))
                    and "nginx" in actions.diagnosis.command
                ),
            ),
            MultiAgentMilestone(
                name="remediation_restarts_nginx",
                credit=0.10,
                description="nginx service status becomes running",
                check=lambda actions, system, outputs, channel: (
                    system.service_registry.services.get("nginx") is not None
                    and system.service_registry.services["nginx"].status == "running"
                ),
            ),
            # ---- Memory track (0.50 total) ----
            MultiAgentMilestone(
                name="triage_escalates_memory",
                credit=0.10,
                description="Triage sends message mentioning memory or OOM",
                check=lambda actions, system, outputs, channel: _triage_mentions_memory(channel),
            ),
            MultiAgentMilestone(
                name="diagnosis_identifies_leak_pid",
                credit=0.15,
                description="Diagnosis identifies the leaking PID",
                check=lambda actions, system, outputs, channel: (
                    "ps" in actions.diagnosis.command
                    and _diagnosis_sends_leak_pid(channel, leaking_pid)
                ),
            ),
            MultiAgentMilestone(
                name="remediation_kills_leak",
                credit=0.15,
                description="Leaking process no longer in process table",
                check=lambda actions, system, outputs, channel: (
                    system.process_table.processes.get(leaking_pid) is None
                ),
            ),
            MultiAgentMilestone(
                name="remediation_restarts_data_processor",
                credit=0.10,
                description="data_processor service status is running",
                check=lambda actions, system, outputs, channel: (
                    system.service_registry.services.get("data_processor") is not None
                    and system.service_registry.services["data_processor"].status == "running"
                ),
            ),
            # ---- Both resolved (0.20) ----
            MultiAgentMilestone(
                name="all_incidents_resolved",
                credit=0.20,
                description="Both nginx and data_processor are running",
                check=lambda actions, system, outputs, channel: (
                    _nginx_running(system) and _data_processor_running(system)
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
    for msg in channel.get_full_history():
        if msg.from_agent == "triage" and "nginx" in msg.content.lower():
            return True
    return False


def _triage_mentions_memory(channel: CommunicationChannel) -> bool:
    for msg in channel.get_full_history():
        if msg.from_agent == "triage":
            content_lower = msg.content.lower()
            if "memory" in content_lower or "oom" in content_lower:
                return True
    return False


def _diagnosis_sends_leak_pid(channel: CommunicationChannel, leaking_pid: int) -> bool:
    for msg in channel.get_full_history():
        if msg.from_agent == "diagnosis" and str(leaking_pid) in msg.content:
            return True
    return False


def _nginx_running(system: SimulatedSystem) -> bool:
    svc = system.service_registry.services.get("nginx")
    return svc is not None and svc.status == "running"


def _data_processor_running(system: SimulatedSystem) -> bool:
    svc = system.service_registry.services.get("data_processor")
    return svc is not None and svc.status == "running"
