"""SimulatedSystem aggregate root holding all infrastructure state.

Composes ProcessTable, VirtualFilesystem, ServiceRegistry, LogBuffer, and
MetricsStore.  Enforces cross-component consistency invariants so that
service status changes cascade to logs, process kills cascade to services,
and metrics reflect the current state.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

from sre_env.server.models import (
    LogBuffer,
    MetricPoint,
    MetricsStore,
    ProcessTable,
    ServiceRegistry,
    VirtualFilesystem,
)


class SimulatedSystem:
    """Aggregate root holding all infrastructure state."""

    def __init__(self) -> None:
        self.process_table = ProcessTable()
        self.filesystem = VirtualFilesystem()
        self.service_registry = ServiceRegistry()
        self.log_buffer = LogBuffer()
        self.metrics_store = MetricsStore()
        self.current_time: datetime = datetime(2024, 1, 15, 10, 0, 0)

        # Tasks can register config validators keyed by service name.
        # A validator receives the SimulatedSystem and returns True if the
        # config is valid (service may run) or False (service should crash).
        self.config_validators: Dict[str, Callable[["SimulatedSystem"], bool]] = {}

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def kill_process(self, pid: int) -> str:
        """Kill a process and cascade to the service registry.

        * Removes the process from the ProcessTable.
        * If the process was linked to a service, sets that service to
          ``"crashed"`` and logs the event.
        * Returns terminal-style output.
        """
        record = self.process_table.kill_process(pid)
        if record is None:
            return f"No such process: {pid}"

        output = f"Process {pid} ({record.name}) killed"

        if record.service_name:
            svc = self.service_registry.get_service(record.service_name)
            if svc is not None:
                self.service_registry.set_status(record.service_name, "crashed")
                svc.pid = None
                self.log_buffer.append(
                    timestamp=self.current_time,
                    severity="ERROR",
                    source=record.service_name,
                    message=f"Service crashed: process {pid} was killed",
                )
                output += f"\nService {record.service_name} is now crashed"

        self.log_buffer.append(
            timestamp=self.current_time,
            severity="INFO",
            source="system",
            message=f"Process {pid} ({record.name}) killed by user",
        )
        return output

    def restart_service(self, name: str) -> str:
        """Restart a service if its dependencies are met.

        * Checks the service exists (error if not).
        * Checks for unmet dependencies (error if any).
        * Runs any registered config validator — if it returns False the
          service starts but immediately crashes (Task 3 wrong-password
          scenario).
        * Otherwise creates a new process in the ProcessTable, sets the
          service to ``"running"``, and logs the event.
        * Returns terminal-style output.
        """
        svc = self.service_registry.get_service(name)
        if svc is None:
            return f"Unit {name} not found"

        unmet = self.service_registry.unmet_dependencies(name)
        if unmet:
            dep_list = ", ".join(unmet)
            self.log_buffer.append(
                timestamp=self.current_time,
                severity="ERROR",
                source=name,
                message=f"Failed to restart: unmet dependencies: {dep_list}",
            )
            return (
                f"Job for {name}.service failed: dependency {unmet[0]} is not running"
            )

        # Create a new process for the service
        new_pid = self.process_table.add_process(
            name=name,
            cpu=1.0,
            mem=64.0,
            status="running",
            service_name=name,
        )

        # Check config validator (e.g. wrong password in Task 3)
        validator = self.config_validators.get(name)
        if validator is not None and not validator(self):
            # Service starts but immediately crashes
            self.service_registry.set_status(name, "crashed")
            svc.pid = new_pid
            svc.started_at = self.current_time
            # Kill the process we just created
            self.process_table.kill_process(new_pid)
            svc.pid = None
            self.log_buffer.append(
                timestamp=self.current_time,
                severity="FATAL",
                source=name,
                message=f"Service {name} started but immediately crashed due to configuration error",
            )
            return (
                f"Service {name} started but immediately crashed "
                f"(check logs for details)"
            )

        # Successful restart
        self.service_registry.set_status(name, "running")
        svc.pid = new_pid
        svc.started_at = self.current_time

        self.log_buffer.append(
            timestamp=self.current_time,
            severity="INFO",
            source=name,
            message=f"Service {name} started successfully (PID {new_pid})",
        )
        return f"Service {name} restarted successfully (PID {new_pid})"

    def edit_file(self, path: str, old: str, new: str) -> str:
        """Replace content in the virtual filesystem.

        Delegates to ``VirtualFilesystem.edit_file()`` and returns a
        human-readable success / failure message.
        """
        try:
            success = self.filesystem.edit_file(path, old, new)
        except ValueError as exc:
            return str(exc)

        if success:
            self.log_buffer.append(
                timestamp=self.current_time,
                severity="INFO",
                source="system",
                message=f"File edited: {path}",
            )
            return f"File {path} updated successfully"
        return f"edit: pattern not found in {path}"

    # ------------------------------------------------------------------
    # Time & metrics
    # ------------------------------------------------------------------

    def advance_time(self, seconds: int = 5) -> None:
        """Advance the simulated clock and generate metric data points.

        For each running service a ``MetricPoint`` is appended to the
        ``MetricsStore`` reflecting current resource usage.
        """
        self.current_time += timedelta(seconds=seconds)

        for svc_name, svc in self.service_registry.services.items():
            proc = self.process_table.get_by_service(svc_name)
            if svc.status == "running" and proc is not None:
                point = MetricPoint(
                    timestamp=self.current_time,
                    cpu_percent=proc.cpu_percent,
                    memory_percent=min(proc.memory_mb / 8192.0 * 100.0, 100.0),
                    disk_percent=45.0,
                    network_mbps=1.0,
                )
            else:
                # Non-running services still get a zero-activity data point
                point = MetricPoint(
                    timestamp=self.current_time,
                    cpu_percent=0.0,
                    memory_percent=0.0,
                    disk_percent=45.0,
                    network_mbps=0.0,
                )
            self.metrics_store.add_point(svc_name, point)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of the entire system state."""
        return {
            "process_table": self.process_table.model_dump(),
            "filesystem": self.filesystem.model_dump(),
            "service_registry": self.service_registry.model_dump(),
            "log_buffer": self.log_buffer.model_dump(),
            "metrics_store": self.metrics_store.model_dump(),
            "current_time": self.current_time.isoformat(),
        }
