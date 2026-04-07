"""Task 3 — Cascading Failure Resolution (Hard).

Scenario: A database has incorrect credentials causing a multi-service
cascade. The agent must trace the root cause, fix the config, and restart
services in dependency order.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from sre_env.server.grader import Milestone, Penalty, StateCheck, FatalAction, TaskGrader
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
            "Jan 15 09:48:00 server redis[4000]: WARN: Memory usage above 70% threshold",
            "Jan 15 09:49:00 server monitoring[5000]: WARN: High CPU detected on app_server",
            "Jan 15 09:51:00 server monitoring[5000]: WARN: Redis memory usage above threshold",
        ]
        system.filesystem.write_file("/var/log/syslog", "\n".join(syslog_lines) + "\n")

        # Redis red herring logs
        redis_log = [
            "2024-01-15 09:40:00 INFO  [main] Redis server started on port 6379",
            "2024-01-15 09:48:00 WARN  [memory] Memory usage above 70% threshold (5734MB/8192MB)",
            "2024-01-15 09:50:00 WARN  [memory] Consider increasing maxmemory or enabling eviction",
            "2024-01-15 09:52:00 INFO  [keyspace] DB 0: 15234 keys, 0 expires",
            "2024-01-15 09:55:00 WARN  [memory] Memory usage at 72% - approaching limit",
        ]
        system.filesystem.write_file("/var/log/redis/redis.log", "\n".join(redis_log) + "\n")

        # Monitoring alert log
        monitoring_log = [
            "2024-01-15 09:40:00 INFO  [main] Monitoring service started",
            "2024-01-15 09:45:00 WARN  [alert] High CPU detected on app_server (85%)",
            "2024-01-15 09:46:00 WARN  [alert] app_server response time > 5000ms",
            "2024-01-15 09:48:00 WARN  [alert] Redis memory usage above threshold",
            "2024-01-15 09:50:00 ERROR [alert] Multiple services degraded - possible cascading failure",
            "2024-01-15 09:55:00 WARN  [alert] db_connector health check timeout",
        ]
        system.filesystem.write_file("/var/log/monitoring/alerts.log", "\n".join(monitoring_log) + "\n")

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

        # Redis red herring log entries
        system.log_buffer.append(
            timestamp=fail_t + timedelta(minutes=3), severity="WARN", source="redis",
            message="Memory usage above 70% threshold (5734MB/8192MB)",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(minutes=4), severity="WARN", source="redis",
            message="Consider increasing maxmemory or enabling eviction",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(minutes=6), severity="WARN", source="redis",
            message="Memory usage at 72% - approaching limit",
        )

        # Monitoring alert log entries
        system.log_buffer.append(
            timestamp=fail_t + timedelta(seconds=20), severity="WARN", source="monitoring",
            message="High CPU detected on app_server (85%)",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(minutes=3, seconds=30), severity="WARN", source="monitoring",
            message="Redis memory usage above threshold",
        )
        system.log_buffer.append(
            timestamp=fail_t + timedelta(minutes=5, seconds=10), severity="ERROR", source="monitoring",
            message="Multiple services degraded - possible cascading failure",
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
                credit=0.05,
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
                credit=0.10,
                description="Agent reads the database config",
                check=lambda cmd, sys, out: (
                    any(k in cmd for k in ("cat", "grep"))
                    and "database.yml" in cmd
                ),
            ),
            Milestone(
                name="fix_config",
                credit=0.15,
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
                credit=0.05,
                description="Agent verifies all services are running",
                check=lambda cmd, sys, out: (
                    _all_three_running(sys)
                    and ("curl" in cmd or "systemctl status" in cmd)
                ),
            ),
        ]

        # State-based verification: all cascade services running + correct config
        state_checks = [
            StateCheck(
                name="cascade_resolved",
                credit=0.10,
                description="All three cascade services are running (state-based)",
                check=lambda sys: _all_three_running(sys),
            ),
            StateCheck(
                name="config_correct",
                credit=0.10,
                description="Database config has correct password (state-based)",
                check=lambda sys: _config_has_correct_password(sys),
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

        # Fatal actions
        fatal_actions = [
            FatalAction(
                name="killed_postgres",
                description="Killed the running postgres database — the root dependency",
                check=lambda cmd, sys, out: (
                    "kill" in cmd
                    and "No such process" not in out
                    and _killed_service_process(cmd, sys, "postgres")
                ),
            ),
            FatalAction(
                name="stopped_postgres",
                description="Stopped the running postgres database",
                check=lambda cmd, sys, out: (
                    "systemctl stop" in cmd
                    and "postgres" in cmd
                ),
            ),
            FatalAction(
                name="corrupted_config",
                description="Replaced database.yml config with invalid content",
                check=lambda cmd, sys, out: (
                    "edit" in cmd
                    and "database.yml" in cmd
                    and _config_is_corrupted(sys)
                ),
            ),
            FatalAction(
                name="killed_redis",
                description="Killed the healthy redis service",
                check=lambda cmd, sys, out: (
                    "kill" in cmd
                    and "No such process" not in out
                    and _killed_service_process(cmd, sys, "redis")
                ),
            ),
        ]

        # Health function: cascade services + config correctness
        def health_fn(sys: SimulatedSystem) -> float:
            services = sys.service_registry.services
            if not services:
                return 1.0
            running = sum(1 for s in services.values() if s.status == "running")
            svc_health = running / len(services)
            config_ok = 1.0 if _config_has_correct_password(sys) else 0.0
            # Weight: 70% services, 30% config
            return svc_health * 0.7 + config_ok * 0.3

        return TaskGrader(
            milestones=milestones,
            penalties=penalties,
            state_checks=state_checks,
            fatal_actions=fatal_actions,
            health_fn=health_fn,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_has_correct_password(sys: SimulatedSystem) -> bool:
    try:
        content = sys.filesystem.read_file("/etc/app/database.yml")
        return CORRECT_PASSWORD in content
    except ValueError:
        return False


def _config_is_corrupted(sys: SimulatedSystem) -> bool:
    """Check if the database.yml has been corrupted (missing required keys)."""
    try:
        content = sys.filesystem.read_file("/etc/app/database.yml")
        # Must still contain basic structure
        required = ["host:", "port:", "user:", "password:", "dbname:"]
        return not all(key in content for key in required)
    except ValueError:
        return True  # File deleted = corrupted


def _all_three_running(sys: SimulatedSystem) -> bool:
    for name in ("db_connector", "app_server", "load_balancer"):
        svc = sys.service_registry.services.get(name)
        if svc is None or svc.status != "running":
            return False
    return True


def _restarted_with_unmet_deps(cmd: str, sys: SimulatedSystem) -> bool:
    """Check if the restart command targeted a service with unmet deps."""
    parts = cmd.strip().split()
    if len(parts) < 3:
        return False
    svc_name = parts[2]
    svc = sys.service_registry.services.get(svc_name)
    if svc is None:
        return False
    unmet = sys.service_registry.unmet_dependencies(svc_name)
    return svc.status != "running" and len(unmet) > 0


def _killed_service_process(cmd: str, sys: SimulatedSystem, service_name: str) -> bool:
    """Check if a kill command resulted in the given service's process being killed."""
    svc = sys.service_registry.services.get(service_name)
    if svc is None:
        return False
    proc = sys.process_table.get_by_service(service_name)
    if proc is None and svc.status in ("crashed",):
        return True
    return False
