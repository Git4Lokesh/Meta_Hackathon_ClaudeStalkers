"""Task 3 — Cascading Failure Resolution (Hard).

Scenario: A database has incorrect credentials causing a multi-service
cascade. The agent must trace the root cause, fix the config, and restart
services in dependency order.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from sre_env.server.grader import Milestone, Penalty, TaskGrader
from sre_env.server.models import MetricPoint, ServiceRecord
from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.tasks.base import TaskDefinitionBase

CORRECT_PASSWORD = "correct_db_pass_456"


class CascadingFailureTask(TaskDefinitionBase):
    task_id = "task3"
    name = "Cascading Failure Resolution"
    description = (
        "A database has incorrect credentials causing a multi-service cascade."
    )
    max_steps = 40
    difficulty = "hard"

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        rng = random.Random(seed)
        system = SimulatedSystem()
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        system.current_time = base_time

        # ---- Services (6) ----
        system.service_registry.services["postgres"] = ServiceRecord(
            name="postgres", status="running", port=5432, dependencies=[],
        )
        system.service_registry.services["db_connector"] = ServiceRecord(
            name="db_connector", status="crashed", port=5433,
            dependencies=["postgres"],
        )
        system.service_registry.services["app_server"] = ServiceRecord(
            name="app_server", status="degraded", port=8080,
            dependencies=["db_connector"],
        )
        system.service_registry.services["load_balancer"] = ServiceRecord(
            name="load_balancer", status="running", port=80,
            dependencies=["app_server"],
        )
        system.service_registry.services["monitoring"] = ServiceRecord(
            name="monitoring", status="running", port=9090, dependencies=[],
        )
        system.service_registry.services["redis"] = ServiceRecord(
            name="redis", status="running", port=6379, dependencies=[],
        )

        # ---- Processes ----
        # postgres is running
        pg_pid = system.process_table.add_process(
            "postgres", cpu=3.0, mem=256.0, status="running", service_name="postgres",
        )
        system.service_registry.services["postgres"].pid = pg_pid

        # db_connector is crashed — no process

        # app_server is degraded but still has a process (retrying)
        app_pid = system.process_table.add_process(
            "app_server", cpu=85.0, mem=512.0, status="running", service_name="app_server",
        )
        system.service_registry.services["app_server"].pid = app_pid

        # load_balancer is running
        lb_pid = system.process_table.add_process(
            "load_balancer", cpu=8.0, mem=128.0, status="running", service_name="load_balancer",
        )
        system.service_registry.services["load_balancer"].pid = lb_pid

        # monitoring
        mon_pid = system.process_table.add_process(
            "monitoring", cpu=2.0, mem=96.0, status="running", service_name="monitoring",
        )
        system.service_registry.services["monitoring"].pid = mon_pid

        # redis
        redis_pid = system.process_table.add_process(
            "redis", cpu=1.0, mem=64.0, status="running", service_name="redis",
        )
        system.service_registry.services["redis"].pid = redis_pid

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
        # database.yml with wrong password
        db_config = """database:
  host: localhost
  port: 5432
  user: app
  password: wrong_password_123
  dbname: production
"""
        system.filesystem.write_file("/etc/app/database.yml", db_config)

        # db_connector log
        db_conn_log = [
            "2024-01-15 09:45:00 INFO  [main] db_connector starting on port 5433",
            "2024-01-15 09:45:01 INFO  [main] Connecting to postgres at localhost:5432",
            "2024-01-15 09:45:02 FATAL [auth] authentication failed for user 'app'",
            "2024-01-15 09:45:02 FATAL [main] Cannot establish database connection",
            "2024-01-15 09:45:03 ERROR [main] Retrying connection (attempt 1/3)...",
            "2024-01-15 09:45:05 FATAL [auth] authentication failed for user 'app'",
            "2024-01-15 09:45:06 ERROR [main] Retrying connection (attempt 2/3)...",
            "2024-01-15 09:45:08 FATAL [auth] authentication failed for user 'app'",
            "2024-01-15 09:45:09 FATAL [main] All connection attempts failed. Shutting down.",
        ]
        system.filesystem.write_file(
            "/var/log/db_connector/connector.log", "\n".join(db_conn_log) + "\n",
        )

        # app_server log
        app_log = [
            "2024-01-15 09:40:00 INFO  [main] Application started on port 8080",
            "2024-01-15 09:45:10 ERROR [db] connection timeout to db_connector:5433, retrying...",
            "2024-01-15 09:45:15 ERROR [db] connection timeout to db_connector:5433, retrying...",
            "2024-01-15 09:45:20 ERROR [db] connection timeout to db_connector:5433, retrying...",
            "2024-01-15 09:46:00 WARN  [main] Service degraded: database unavailable",
            "2024-01-15 09:50:00 ERROR [db] connection timeout to db_connector:5433, retrying...",
            "2024-01-15 09:55:00 ERROR [db] connection timeout to db_connector:5433, retrying...",
        ]
        system.filesystem.write_file(
            "/var/log/app_server/app.log", "\n".join(app_log) + "\n",
        )

        # load_balancer log
        lb_log = [
            "2024-01-15 09:40:00 INFO  [main] Load balancer started on port 80",
            "2024-01-15 09:45:30 WARN  [health] backend health check failed for app_server:8080",
            "2024-01-15 09:46:00 WARN  [health] backend health check failed for app_server:8080",
            "2024-01-15 09:47:00 WARN  [health] backend health check failed for app_server:8080",
            "2024-01-15 09:50:00 WARN  [health] backend app_server:8080 marked as unhealthy",
            "2024-01-15 09:55:00 ERROR [routing] no healthy backends available, returning 503",
        ]
        system.filesystem.write_file(
            "/var/log/load_balancer/lb.log", "\n".join(lb_log) + "\n",
        )

        # syslog
        syslog_lines = [
            "Jan 15 09:45:00 server systemd[1]: Starting db_connector.service...",
            "Jan 15 09:45:09 server systemd[1]: db_connector.service: Main process exited, code=exited, status=1/FAILURE",
            "Jan 15 09:45:10 server systemd[1]: db_connector.service: Failed with result 'exit-code'.",
            "Jan 15 09:46:00 server app_server[2000]: WARN: database connection unavailable",
            "Jan 15 09:50:00 server load_balancer[3000]: WARN: backend health check failing",
            "Jan 15 09:55:00 server cron[500]: (root) CMD (/usr/bin/logrotate /etc/logrotate.conf)",
        ]
        system.filesystem.write_file("/var/log/syslog", "\n".join(syslog_lines) + "\n")

        # ---- Config validator for db_connector ----
        def db_connector_validator(sys: SimulatedSystem) -> bool:
            """db_connector can only run if the password is correct."""
            try:
                content = sys.filesystem.read_file("/etc/app/database.yml")
                return CORRECT_PASSWORD in content
            except ValueError:
                return False

        system.config_validators["db_connector"] = db_connector_validator

        # ---- LogBuffer (50+ entries) ----
        services_for_logs = ["postgres", "redis", "monitoring"]
        normal_msgs = {
            "postgres": [
                "checkpoint starting: time",
                "automatic vacuum of table public.events",
                "connection received: host=127.0.0.1",
                "statement: SELECT 1",
            ],
            "redis": [
                "Background saving started",
                "DB saved on disk",
                "Client connected from 127.0.0.1",
            ],
            "monitoring": [
                "Scraping metrics from postgres:9187",
                "Alert rule evaluation complete",
                "Health check passed for redis",
            ],
        }

        t = base_time - timedelta(minutes=30)
        for i in range(35):
            svc = rng.choice(services_for_logs)
            sev = rng.choice(["DEBUG", "INFO", "INFO", "INFO"])
            msg = rng.choice(normal_msgs[svc])
            system.log_buffer.append(timestamp=t, severity=sev, source=svc, message=msg)
            t += timedelta(seconds=rng.randint(20, 60))

        # Cascading failure log entries
        fail_t = base_time - timedelta(minutes=15)
        system.log_buffer.append(
            timestamp=fail_t, severity="FATAL", source="db_connector",
            message="authentication failed for user 'app'",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(seconds=5), severity="FATAL", source="db_connector",
            message="All connection attempts failed. Shutting down.",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(seconds=10), severity="ERROR", source="app_server",
            message="connection timeout to db_connector:5433, retrying...",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(seconds=15), severity="ERROR", source="app_server",
            message="connection timeout to db_connector:5433, retrying...",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(minutes=1), severity="WARN", source="app_server",
            message="Service degraded: database unavailable",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(minutes=1, seconds=30), severity="WARN", source="load_balancer",
            message="backend health check failed for app_server:8080",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(minutes=2), severity="WARN", source="load_balancer",
            message="backend app_server:8080 marked as unhealthy",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(minutes=5), severity="ERROR", source="load_balancer",
            message="no healthy backends available, returning 503",
        )
        # Extra normal entries
        for i in range(10):
            svc = rng.choice(services_for_logs)
            system.log_buffer.append(
                timestamp=base_time - timedelta(minutes=rng.randint(0, 4)),
                severity="INFO", source=svc,
                message=rng.choice(normal_msgs[svc]),
            )

        # ---- MetricsStore (30+ minutes) ----
        for svc_name in [
            "postgres", "db_connector", "app_server",
            "load_balancer", "monitoring", "redis",
        ]:
            t = base_time - timedelta(minutes=35)
            for minute in range(35):
                ts = t + timedelta(minutes=minute)
                if svc_name == "db_connector":
                    # crashed — zero metrics
                    point = MetricPoint(
                        timestamp=ts, cpu_percent=0.0, memory_percent=0.0,
                        disk_percent=45.0, network_mbps=0.0,
                    )
                elif svc_name == "app_server":
                    # CPU spikes from retry loops
                    point = MetricPoint(
                        timestamp=ts,
                        cpu_percent=rng.uniform(70.0, 95.0),
                        memory_percent=rng.uniform(10.0, 20.0),
                        disk_percent=45.0,
                        network_mbps=rng.uniform(0.5, 2.0),
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
        milestones = [
            Milestone(
                name="read_lb_logs",
                credit=0.05,
                description="Agent reads load balancer logs",
                check=lambda cmd, sys, out: (
                    any(k in cmd for k in ("cat", "tail", "grep"))
                    and ("load_balancer" in cmd or "lb" in cmd)
                ),
            ),
            Milestone(
                name="read_app_logs",
                credit=0.10,
                description="Agent reads app_server logs",
                check=lambda cmd, sys, out: (
                    any(k in cmd for k in ("cat", "tail", "grep"))
                    and ("app_server" in cmd or "app.log" in cmd)
                ),
            ),
            Milestone(
                name="read_db_logs",
                credit=0.10,
                description="Agent reads db_connector logs",
                check=lambda cmd, sys, out: (
                    any(k in cmd for k in ("cat", "tail", "grep"))
                    and ("db_connector" in cmd or "connector" in cmd)
                ),
            ),
            Milestone(
                name="read_config",
                credit=0.15,
                description="Agent reads the database config",
                check=lambda cmd, sys, out: (
                    any(k in cmd for k in ("cat", "grep"))
                    and "database.yml" in cmd
                ),
            ),
            Milestone(
                name="fix_config",
                credit=0.20,
                description="Agent fixes the database password",
                check=lambda cmd, sys, out: (
                    _config_has_correct_password(sys)
                ),
            ),
            Milestone(
                name="restart_db_connector",
                credit=0.10,
                description="Agent restarts db_connector",
                check=lambda cmd, sys, out: (
                    sys.service_registry.services.get("db_connector") is not None
                    and sys.service_registry.services["db_connector"].status == "running"
                ),
            ),
            Milestone(
                name="restart_app_server",
                credit=0.10,
                description="Agent restarts app_server",
                check=lambda cmd, sys, out: (
                    sys.service_registry.services.get("app_server") is not None
                    and sys.service_registry.services["app_server"].status == "running"
                ),
            ),
            Milestone(
                name="restart_load_balancer",
                credit=0.10,
                description="Agent restarts load_balancer after app_server is healthy",
                check=lambda cmd, sys, out: (
                    sys.service_registry.services.get("load_balancer") is not None
                    and sys.service_registry.services["load_balancer"].status == "running"
                    and "systemctl restart load_balancer" in cmd
                ),
            ),
            Milestone(
                name="verify_all",
                credit=0.10,
                description="Agent verifies all services are running",
                check=lambda cmd, sys, out: (
                    _all_three_running(sys)
                    and ("curl" in cmd or "systemctl status" in cmd)
                ),
            ),
        ]

        penalties = [
            Penalty(
                name="restart_before_deps",
                amount=0.05,
                description="Agent restarted a service with unmet dependencies",
                check=lambda cmd, sys, out: (
                    "systemctl restart" in cmd
                    and _restarted_with_unmet_deps(cmd, sys)
                ),
            ),
        ]

        return TaskGrader(milestones=milestones, penalties=penalties)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_has_correct_password(sys: SimulatedSystem) -> bool:
    try:
        content = sys.filesystem.read_file("/etc/app/database.yml")
        return CORRECT_PASSWORD in content
    except ValueError:
        return False


def _all_three_running(sys: SimulatedSystem) -> bool:
    for name in ("db_connector", "app_server", "load_balancer"):
        svc = sys.service_registry.services.get(name)
        if svc is None or svc.status != "running":
            return False
    return True


def _restarted_with_unmet_deps(cmd: str, sys: SimulatedSystem) -> bool:
    """Check if the restart command targeted a service with unmet deps."""
    parts = cmd.strip().split()
    # Expected: systemctl restart <service_name>
    if len(parts) < 3:
        return False
    svc_name = parts[2]
    # Check if the service had unmet deps *before* the restart executed.
    # Since the grader runs after execution, we check if the restart failed
    # (service is not running) which indicates unmet deps.
    svc = sys.service_registry.services.get(svc_name)
    if svc is None:
        return False
    # If the service is not running after a restart attempt, deps were unmet
    # or config was bad. We specifically check the dependency chain.
    unmet = sys.service_registry.unmet_dependencies(svc_name)
    # The penalty fires if there were unmet deps at the time of the command.
    # Since the grader evaluates after execution, if the service is crashed
    # and has unmet deps, the penalty applies.
    return svc.status != "running" and len(unmet) > 0
