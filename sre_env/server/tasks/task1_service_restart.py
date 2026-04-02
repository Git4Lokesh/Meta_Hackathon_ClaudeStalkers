"""Task 1 — Service Restart (Easy).

Scenario: nginx has crashed. The agent must identify the crashed service,
read error logs, restart nginx, and verify it is running.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from sre_env.server.grader import Milestone, Penalty, TaskGrader
from sre_env.server.models import MetricPoint
from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.tasks.base import TaskDefinitionBase


class ServiceRestartTask(TaskDefinitionBase):
    task_id = "task1"
    name = "Service Restart"
    description = "A web server (nginx) has crashed. Identify and restart it."
    max_steps = 20
    difficulty = "easy"

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        rng = random.Random(seed)
        system = SimulatedSystem()
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        crash_time = base_time - timedelta(minutes=5)
        system.current_time = base_time

        # ---- Services (5) ----
        system.service_registry.services["nginx"] = _svc(
            "nginx", "crashed", 80, [],
        )
        system.service_registry.services["postgres"] = _svc(
            "postgres", "running", 5432, [],
        )
        system.service_registry.services["redis"] = _svc(
            "redis", "running", 6379, [],
        )
        system.service_registry.services["app_server"] = _svc(
            "app_server", "running", 8080, ["postgres", "redis"],
        )
        system.service_registry.services["monitoring"] = _svc(
            "monitoring", "running", 9090, [],
        )

        # ---- Processes for running services ----
        pg_pid = system.process_table.add_process(
            "postgres", cpu=2.5, mem=256.0, status="running", service_name="postgres",
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
            "monitoring", cpu=3.0, mem=128.0, status="running", service_name="monitoring",
        )
        system.service_registry.services["monitoring"].pid = mon_pid

        # nginx has NO process (it crashed)

        # ---- Background processes (~10) ----
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
            ("agetty", 0.0, 1.5),
        ]
        for name, cpu, mem in bg_procs:
            cpu_jitter = rng.uniform(-0.05, 0.05)
            system.process_table.add_process(
                name, cpu=max(0.0, cpu + cpu_jitter), mem=mem, status="sleeping",
            )

        # ---- Filesystem ----
        # nginx error log
        error_lines = [
            "2024/01/15 09:54:58 [error] 1234#0: *5 open() \"/usr/share/nginx/html/favicon.ico\" failed (2: No such file or directory)",
            "2024/01/15 09:55:00 [emerg] worker process exited with signal 11 (core dumped)",
            "2024/01/15 09:55:01 [alert] worker process 1234 exited on signal 11",
            "2024/01/15 09:55:02 [emerg] cannot bind to 0.0.0.0:80 (Address already in use)",
            "2024/01/15 09:55:03 [emerg] master process exiting due to fatal error",
        ]
        system.filesystem.write_file(
            "/var/log/nginx/error.log", "\n".join(error_lines) + "\n",
        )

        # nginx access log
        access_lines = [
            '192.168.1.10 - - [15/Jan/2024:09:50:00 +0000] "GET / HTTP/1.1" 200 612',
            '192.168.1.11 - - [15/Jan/2024:09:51:30 +0000] "GET /api/health HTTP/1.1" 200 2',
            '192.168.1.12 - - [15/Jan/2024:09:53:00 +0000] "POST /api/data HTTP/1.1" 201 45',
            '192.168.1.10 - - [15/Jan/2024:09:54:00 +0000] "GET /static/style.css HTTP/1.1" 200 1024',
        ]
        system.filesystem.write_file(
            "/var/log/nginx/access.log", "\n".join(access_lines) + "\n",
        )

        # nginx config
        nginx_conf = """worker_processes auto;
error_log /var/log/nginx/error.log;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    sendfile on;
    keepalive_timeout 65;

    server {
        listen 80;
        server_name localhost;
        location / {
            root /usr/share/nginx/html;
            index index.html;
        }
    }
}
"""
        system.filesystem.write_file("/etc/nginx/nginx.conf", nginx_conf)

        # syslog with nginx crash entries
        syslog_lines = [
            "Jan 15 09:50:00 server systemd[1]: Started nginx - high performance web server.",
            "Jan 15 09:54:59 server kernel: nginx[1234]: segfault at 0 ip 00007f3a rsp 00007ffd error 4",
            "Jan 15 09:55:00 server systemd[1]: nginx.service: Main process exited, code=killed, status=11/SEGV",
            "Jan 15 09:55:01 server systemd[1]: nginx.service: Failed with result 'signal'.",
            "Jan 15 09:55:02 server systemd[1]: nginx.service: Scheduled restart job, restart counter is at 3.",
            "Jan 15 09:55:03 server systemd[1]: nginx.service: Start request repeated too quickly. Refusing to start.",
            "Jan 15 09:56:00 server cron[500]: (root) CMD (/usr/bin/logrotate /etc/logrotate.conf)",
        ]
        system.filesystem.write_file("/var/log/syslog", "\n".join(syslog_lines) + "\n")

        # app_server log (normal)
        app_log_lines = [
            "2024-01-15 09:50:00 INFO  [main] Application started on port 8080",
            "2024-01-15 09:52:00 INFO  [http] GET /api/health 200 2ms",
            "2024-01-15 09:54:00 INFO  [http] POST /api/data 201 15ms",
            "2024-01-15 09:55:30 WARN  [http] upstream nginx not responding, using cached response",
        ]
        system.filesystem.write_file("/var/log/app_server/app.log", "\n".join(app_log_lines) + "\n")

        # /proc/meminfo
        meminfo = """MemTotal:        8192000 kB
MemFree:         4096000 kB
MemAvailable:    5120000 kB
Buffers:          256000 kB
Cached:           768000 kB
SwapTotal:       2048000 kB
SwapFree:        2048000 kB
"""
        system.filesystem.write_file("/proc/meminfo", meminfo)

        # ---- LogBuffer (50+ entries) ----
        # Normal entries from various services over the last 30 minutes
        services_for_logs = ["postgres", "redis", "app_server", "monitoring"]
        severities = ["DEBUG", "INFO", "INFO", "INFO", "WARN"]
        messages_by_svc = {
            "postgres": [
                "checkpoint starting: time",
                "automatic vacuum of table public.events",
                "connection received: host=127.0.0.1 port=5432",
                "statement: SELECT 1",
            ],
            "redis": [
                "Background saving started",
                "DB saved on disk",
                "Client connected from 127.0.0.1",
                "Accepted connection from 127.0.0.1:6379",
            ],
            "app_server": [
                "GET /api/health 200 2ms",
                "POST /api/data 201 15ms",
                "Connection pool: 5 active, 10 idle",
                "Cache hit ratio: 0.85",
            ],
            "monitoring": [
                "Scraping metrics from nginx:9113",
                "Scraping metrics from postgres:9187",
                "Alert rule evaluation complete",
                "Health check passed for all targets",
            ],
        }

        t = base_time - timedelta(minutes=30)
        for i in range(45):
            svc = rng.choice(services_for_logs)
            sev = rng.choice(severities)
            msg = rng.choice(messages_by_svc[svc])
            system.log_buffer.append(timestamp=t, severity=sev, source=svc, message=msg)
            t += timedelta(seconds=rng.randint(20, 60))

        # nginx crash entries in the last 5 minutes
        crash_t = crash_time
        system.log_buffer.append(
            timestamp=crash_t, severity="ERROR", source="nginx",
            message="worker process exited with signal 11 (core dumped)",
        )
        system.log_buffer.append(
            timestamp=crash_t + timedelta(seconds=1), severity="FATAL", source="nginx",
            message="cannot bind to 0.0.0.0:80 (Address already in use)",
        )
        system.log_buffer.append(
            timestamp=crash_t + timedelta(seconds=2), severity="FATAL", source="nginx",
            message="master process exiting due to fatal error",
        )
        system.log_buffer.append(
            timestamp=crash_t + timedelta(seconds=3), severity="ERROR", source="nginx",
            message="nginx.service: Failed with result 'signal'",
        )
        system.log_buffer.append(
            timestamp=crash_t + timedelta(seconds=4), severity="ERROR", source="nginx",
            message="nginx.service: Start request repeated too quickly. Refusing to start.",
        )
        # A few more normal entries after the crash
        for i in range(5):
            svc = rng.choice(services_for_logs)
            system.log_buffer.append(
                timestamp=base_time - timedelta(minutes=rng.randint(0, 4)),
                severity="INFO", source=svc,
                message=rng.choice(messages_by_svc[svc]),
            )

        # ---- MetricsStore (30+ minutes of history) ----
        for svc_name in ["nginx", "postgres", "redis", "app_server", "monitoring"]:
            t = base_time - timedelta(minutes=35)
            for minute in range(35):
                ts = t + timedelta(minutes=minute)
                if svc_name == "nginx" and ts >= crash_time:
                    # nginx metrics drop to 0 after crash
                    point = MetricPoint(
                        timestamp=ts, cpu_percent=0.0, memory_percent=0.0,
                        disk_percent=45.0, network_mbps=0.0,
                    )
                elif svc_name == "nginx":
                    point = MetricPoint(
                        timestamp=ts,
                        cpu_percent=rng.uniform(5.0, 15.0),
                        memory_percent=rng.uniform(3.0, 8.0),
                        disk_percent=45.0,
                        network_mbps=rng.uniform(10.0, 50.0),
                    )
                else:
                    point = MetricPoint(
                        timestamp=ts,
                        cpu_percent=rng.uniform(1.0, 20.0),
                        memory_percent=rng.uniform(2.0, 10.0),
                        disk_percent=45.0,
                        network_mbps=rng.uniform(0.5, 5.0),
                    )
                system.metrics_store.add_point(svc_name, point)

        return system

    def create_grader(self) -> TaskGrader:
        milestones = [
            Milestone(
                name="read_error_log",
                credit=0.15,
                description="Agent reads the nginx error log",
                check=lambda cmd, sys, out: (
                    any(k in cmd for k in ("cat", "tail", "grep"))
                    and "nginx" in cmd
                    and ("error" in cmd or "log" in cmd)
                ),
            ),
            Milestone(
                name="check_service_status",
                credit=0.25,
                description="Agent checks nginx service status",
                check=lambda cmd, sys, out: cmd.strip() == "systemctl status nginx",
            ),
            Milestone(
                name="restart_nginx",
                credit=0.50,
                description="Agent restarts nginx successfully",
                check=lambda cmd, sys, out: (
                    sys.service_registry.services.get("nginx") is not None
                    and sys.service_registry.services["nginx"].status == "running"
                ),
            ),
            Milestone(
                name="verify_running",
                credit=0.10,
                description="Agent verifies nginx is running after restart",
                check=lambda cmd, sys, out: (
                    sys.service_registry.services.get("nginx") is not None
                    and sys.service_registry.services["nginx"].status == "running"
                    and (
                        ("curl" in cmd and ("80" in cmd or "localhost" in cmd))
                        or cmd.strip() == "systemctl status nginx"
                    )
                ),
            ),
        ]
        return TaskGrader(milestones=milestones, penalties=[])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(name: str, status: str, port: int, deps: list[str]):
    from sre_env.server.models import ServiceRecord
    return ServiceRecord(name=name, status=status, port=port, dependencies=deps)
