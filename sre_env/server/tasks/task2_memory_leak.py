"""Task 2 — Memory Leak Diagnosis (Medium).

Scenario: A process is leaking memory, causing OOM kills and service
degradation. The agent must identify the leaking process, kill it,
restart the service, and verify health.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from sre_env.server.grader import Milestone, Penalty, TaskGrader
from sre_env.server.models import MetricPoint, ServiceRecord
from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.tasks.base import TaskDefinitionBase


class MemoryLeakTask(TaskDefinitionBase):
    task_id = "task2"
    name = "Memory Leak Diagnosis"
    description = (
        "A process is leaking memory, causing OOM kills and service degradation."
    )
    max_steps = 30
    difficulty = "medium"

    # Stored at creation time so the grader can reference it.
    _leaking_pid: int = 0

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        rng = random.Random(seed)
        system = SimulatedSystem()
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        system.current_time = base_time

        # ---- Services (5) ----
        system.service_registry.services["data_processor"] = ServiceRecord(
            name="data_processor", status="degraded", port=8081,
            dependencies=["postgres"],
        )
        system.service_registry.services["postgres"] = ServiceRecord(
            name="postgres", status="running", port=5432, dependencies=[],
        )
        system.service_registry.services["redis"] = ServiceRecord(
            name="redis", status="running", port=6379, dependencies=[],
        )
        system.service_registry.services["nginx"] = ServiceRecord(
            name="nginx", status="running", port=80, dependencies=[],
        )
        system.service_registry.services["monitoring"] = ServiceRecord(
            name="monitoring", status="running", port=9090, dependencies=[],
        )

        # ---- Processes ----
        # The leaking process
        leak_mem = rng.uniform(2500.0, 3200.0)
        leak_pid = system.process_table.add_process(
            "data_processor_worker", cpu=45.0, mem=leak_mem,
            status="running", service_name="data_processor",
        )
        system.service_registry.services["data_processor"].pid = leak_pid
        self._leaking_pid = leak_pid

        # Normal service processes
        pg_pid = system.process_table.add_process(
            "postgres", cpu=3.0, mem=256.0, status="running", service_name="postgres",
        )
        system.service_registry.services["postgres"].pid = pg_pid

        redis_pid = system.process_table.add_process(
            "redis", cpu=1.0, mem=64.0, status="running", service_name="redis",
        )
        system.service_registry.services["redis"].pid = redis_pid

        nginx_pid = system.process_table.add_process(
            "nginx", cpu=5.0, mem=128.0, status="running", service_name="nginx",
        )
        system.service_registry.services["nginx"].pid = nginx_pid

        mon_pid = system.process_table.add_process(
            "monitoring", cpu=2.0, mem=96.0, status="running", service_name="monitoring",
        )
        system.service_registry.services["monitoring"].pid = mon_pid

        # Background processes
        bg_procs = [
            ("cron", 0.1, 8.0),
            ("sshd", 0.2, 12.0),
            ("systemd", 0.5, 32.0),
            ("journald", 0.3, 24.0),
            ("dbus-daemon", 0.1, 6.0),
            ("rsyslogd", 0.2, 16.0),
            ("networkd", 0.1, 10.0),
            ("resolved", 0.1, 8.0),
            ("logrotate", 0.0, 4.0),
            ("atd", 0.0, 2.0),
        ]
        for name, cpu, mem in bg_procs:
            system.process_table.add_process(
                name, cpu=max(0.0, cpu + rng.uniform(-0.05, 0.05)),
                mem=mem, status="sleeping",
            )

        # ---- Filesystem ----
        # /var/log/syslog with OOM killer entries
        syslog_lines = [
            "Jan 15 09:30:00 server kernel: [12345.678] data_processor_worker invoked oom-killer: gfp_mask=0x6200ca(GFP_HIGHUSER_MOVABLE), order=0",
            f"Jan 15 09:30:01 server kernel: [12345.679] Out of memory: Kill process {leak_pid} (data_processor_worker) score 900 or sacrifice child",
            f"Jan 15 09:30:02 server kernel: [12345.680] Killed process {leak_pid} (data_processor_worker) total-vm:3200000kB, anon-rss:2560000kB",
            "Jan 15 09:35:00 server systemd[1]: data_processor.service: Main process exited, code=killed, status=9/KILL",
            "Jan 15 09:35:01 server systemd[1]: data_processor.service: Triggering OnFailure= dependencies.",
            "Jan 15 09:40:00 server kernel: [12400.000] Memory cgroup out of memory: Killed process data_processor_worker",
            "Jan 15 09:50:00 server cron[500]: (root) CMD (/usr/bin/logrotate /etc/logrotate.conf)",
            "Jan 15 09:55:00 server systemd[1]: Started Session 42 of user root.",
        ]
        system.filesystem.write_file("/var/log/syslog", "\n".join(syslog_lines) + "\n")

        # data_processor app log
        dp_log_lines = [
            "2024-01-15 09:20:00 INFO  [main] Data processor started on port 8081",
            "2024-01-15 09:25:00 WARN  [worker] Memory usage at 70% (2048MB/2900MB)",
            "2024-01-15 09:28:00 WARN  [worker] Memory usage at 80% (2320MB/2900MB)",
            "2024-01-15 09:30:00 ERROR [worker] Memory usage critical at 90% (2610MB/2900MB)",
            "2024-01-15 09:30:01 ERROR [worker] GC unable to reclaim memory, possible leak in batch processor",
            "2024-01-15 09:35:00 FATAL [main] Worker process killed by OOM killer",
            "2024-01-15 09:35:01 WARN  [main] Service degraded: primary worker unavailable",
            "2024-01-15 09:40:00 ERROR [main] Health check failing: worker not responding",
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

        # ---- LogBuffer (50+ entries) ----
        services_for_logs = ["postgres", "redis", "nginx", "monitoring"]
        normal_msgs = {
            "postgres": [
                "checkpoint starting: time",
                "automatic vacuum of table public.events",
                "connection received: host=127.0.0.1",
            ],
            "redis": [
                "Background saving started",
                "DB saved on disk",
                "Client connected from 127.0.0.1",
            ],
            "nginx": [
                "GET /api/health 200 2ms",
                "POST /api/data 201 15ms",
                "upstream connection established",
            ],
            "monitoring": [
                "Scraping metrics from data_processor:8081",
                "Alert rule evaluation complete",
                "Health check passed for postgres",
            ],
        }

        t = base_time - timedelta(minutes=30)
        for i in range(40):
            svc = rng.choice(services_for_logs)
            sev = rng.choice(["DEBUG", "INFO", "INFO", "INFO"])
            msg = rng.choice(normal_msgs[svc])
            system.log_buffer.append(timestamp=t, severity=sev, source=svc, message=msg)
            t += timedelta(seconds=rng.randint(20, 60))

        # OOM / data_processor entries
        oom_t = base_time - timedelta(minutes=25)
        system.log_buffer.append(
            timestamp=oom_t, severity="WARN", source="data_processor",
            message="Memory usage at 70% (2048MB/2900MB)",
        )
        system.log_buffer.append(
            timestamp=oom_t + timedelta(minutes=3), severity="WARN", source="data_processor",
            message="Memory usage at 80% (2320MB/2900MB)",
        )
        system.log_buffer.append(
            timestamp=oom_t + timedelta(minutes=5), severity="ERROR", source="data_processor",
            message="Memory usage critical at 90% (2610MB/2900MB)",
        )
        system.log_buffer.append(
            timestamp=oom_t + timedelta(minutes=5, seconds=1), severity="FATAL", source="data_processor",
            message=f"Out of memory: Kill process {leak_pid} (data_processor_worker) score 900",
        )
        system.log_buffer.append(
            timestamp=oom_t + timedelta(minutes=10), severity="ERROR", source="data_processor",
            message="Worker process killed by OOM killer",
        )
        system.log_buffer.append(
            timestamp=oom_t + timedelta(minutes=10, seconds=1), severity="WARN", source="data_processor",
            message="Service degraded: primary worker unavailable",
        )
        # A few more normal entries
        for i in range(8):
            svc = rng.choice(services_for_logs)
            system.log_buffer.append(
                timestamp=base_time - timedelta(minutes=rng.randint(0, 4)),
                severity="INFO", source=svc,
                message=rng.choice(normal_msgs[svc]),
            )

        # ---- MetricsStore (30+ minutes) ----
        for svc_name in ["data_processor", "postgres", "redis", "nginx", "monitoring"]:
            t = base_time - timedelta(minutes=35)
            for minute in range(35):
                ts = t + timedelta(minutes=minute)
                if svc_name == "data_processor":
                    # Memory climbing over time to >90%
                    mem_pct = min(30.0 + (minute * 2.0), 95.0)
                    point = MetricPoint(
                        timestamp=ts,
                        cpu_percent=rng.uniform(30.0, 50.0),
                        memory_percent=mem_pct,
                        disk_percent=45.0,
                        network_mbps=rng.uniform(1.0, 5.0),
                    )
                else:
                    point = MetricPoint(
                        timestamp=ts,
                        cpu_percent=rng.uniform(1.0, 15.0),
                        memory_percent=rng.uniform(2.0, 10.0),
                        disk_percent=45.0,
                        network_mbps=rng.uniform(0.5, 3.0),
                    )
                system.metrics_store.add_point(svc_name, point)

        return system

    def create_grader(self) -> TaskGrader:
        leaking_pid = self._leaking_pid

        milestones = [
            Milestone(
                name="check_memory",
                credit=0.10,
                description="Agent checks system memory",
                check=lambda cmd, sys, out: (
                    "free" in cmd or "top" in cmd or "/proc/meminfo" in cmd
                ),
            ),
            Milestone(
                name="identify_process",
                credit=0.20,
                description="Agent identifies the high-memory process",
                check=lambda cmd, sys, out: (
                    "ps aux" in cmd or "top" in cmd
                ),
            ),
            Milestone(
                name="read_oom_logs",
                credit=0.15,
                description="Agent reads OOM killer logs",
                check=lambda cmd, sys, out: (
                    any(k in cmd for k in ("cat", "tail", "grep"))
                    and "syslog" in cmd
                    and ("OOM" in out or "Out of memory" in out)
                ),
            ),
            Milestone(
                name="kill_process",
                credit=0.25,
                description="Agent kills the leaking process",
                check=lambda cmd, sys, out: (
                    sys.process_table.processes.get(leaking_pid) is None
                    and "kill" in cmd
                ),
            ),
            Milestone(
                name="restart_service",
                credit=0.20,
                description="Agent restarts the data_processor service",
                check=lambda cmd, sys, out: (
                    sys.service_registry.services.get("data_processor") is not None
                    and sys.service_registry.services["data_processor"].status == "running"
                ),
            ),
            Milestone(
                name="verify_healthy",
                credit=0.10,
                description="Agent verifies the service is healthy",
                check=lambda cmd, sys, out: (
                    sys.service_registry.services.get("data_processor") is not None
                    and sys.service_registry.services["data_processor"].status == "running"
                    and (
                        ("curl" in cmd and "8081" in cmd)
                        or cmd.strip() == "systemctl status data_processor"
                    )
                ),
            ),
        ]

        penalties = [
            Penalty(
                name="kill_wrong_process",
                amount=0.10,
                description="Agent killed a process that was not the leaking one",
                check=lambda cmd, sys, out: (
                    "kill" in cmd
                    and not any(
                        str(leaking_pid) in tok
                        for tok in cmd.split()
                    )
                    and "No such process" not in out
                ),
            ),
        ]

        return TaskGrader(milestones=milestones, penalties=penalties)
