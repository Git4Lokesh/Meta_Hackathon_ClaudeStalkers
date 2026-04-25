"""Procedural Task Generator — RLVE-style infinite task diversity.

Generates War Room tasks procedurally from a small library of "fault primitives":
  - crash        — a service transitions to "crashed" status
  - memory_leak  — a process exceeds memory budget (OOM risk)
  - cascade      — config misconfiguration triggers dependency cascade
  - auth_failure — database auth fails causing upstream services to degrade

The difficulty parameter controls:
  - Number of concurrent faults (1-3)
  - Number of phantom alerts / red herrings (0-4)
  - Max rounds (tighter time pressure = harder)

Each ProceduralTask instance is reproducible given a seed — the same
(difficulty, seed) pair always produces the same initial state. This means
the environment satisfies the RLVE adaptive-difficulty spec: it procedurally
generates new tasks as the model improves, never saturating on a fixed
distribution.

Usage:
    task = ProceduralTask(difficulty=0.5)
    system = task.create_initial_state(seed=42)
    grader = task.create_grader()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from sre_env.server.simulated_system import SimulatedSystem
from round2.war_room.communication import CommunicationChannel
from round2.war_room.grader import (
    MultiAgentGrader,
    MultiAgentMilestone,
)
from round2.war_room.models import Alert, MultiAgentAction
from round2.war_room.tasks.base import WarRoomTaskBase


# ---------------------------------------------------------------------------
# Fault primitives
# ---------------------------------------------------------------------------

# Catalog of services that can be faulted. Each entry is (service_name, port, deps).
_SERVICE_CATALOG: list[tuple[str, int, list[str]]] = [
    ("nginx", 80, []),
    ("app_server", 8080, ["postgres", "redis"]),
    ("api_gateway", 8000, ["app_server"]),
    ("data_processor", 8081, ["postgres", "redis"]),
    ("load_balancer", 443, ["app_server", "api_gateway"]),
    ("db_connector", 5433, ["postgres"]),
    ("postgres", 5432, []),
    ("redis", 6379, []),
    ("monitoring", 9090, []),
    ("cache_server", 11211, []),
]

# Which services are considered CRITICAL — killing them terminates the episode.
_CRITICAL_SERVICES = {"postgres", "redis", "monitoring"}


@dataclass
class FaultSpec:
    """Describes a single fault to inject into the simulated system."""
    fault_type: str  # "crash" | "memory_leak" | "cascade" | "auth_failure"
    target_service: str
    params: dict = field(default_factory=dict)


def _sample_fault(
    rng: random.Random,
    already_faulted: set[str],
    allowed_types: list[str] | None = None,
) -> FaultSpec:
    """Sample a random fault that doesn't overlap with already-faulted services."""
    fault_types = allowed_types or ["crash", "memory_leak", "cascade", "auth_failure"]
    fault_type = rng.choice(fault_types)

    # Pick a target appropriate for the fault type
    if fault_type == "crash":
        candidates = [s[0] for s in _SERVICE_CATALOG if s[0] not in already_faulted and s[0] not in _CRITICAL_SERVICES]
    elif fault_type == "memory_leak":
        candidates = ["data_processor", "app_server", "api_gateway"]
        candidates = [s for s in candidates if s not in already_faulted]
    elif fault_type == "cascade":
        candidates = ["db_connector", "app_server", "load_balancer"]
        candidates = [s for s in candidates if s not in already_faulted]
    else:  # auth_failure
        candidates = ["db_connector"]
        candidates = [s for s in candidates if s not in already_faulted]

    if not candidates:
        # Fallback — use any non-critical service
        candidates = [s[0] for s in _SERVICE_CATALOG if s[0] not in _CRITICAL_SERVICES and s[0] not in already_faulted]
    if not candidates:
        # Hard fallback — reuse nginx (shouldn't happen in practice)
        candidates = ["nginx"]

    target = rng.choice(candidates)
    params: dict = {}
    if fault_type == "memory_leak":
        params["leak_pid"] = 1000 + rng.randint(0, 99)
        params["memory_mb"] = rng.randint(2000, 3500)
    return FaultSpec(fault_type=fault_type, target_service=target, params=params)


# ---------------------------------------------------------------------------
# State builders (one per fault type)
# ---------------------------------------------------------------------------

def _svc(name: str, status: str, port: int, deps: list[str]):
    """Build a service registry entry."""
    from sre_env.server.models import ServiceRecord
    return ServiceRecord(name=name, status=status, port=port, dependencies=deps)


def _build_base_system(rng: random.Random) -> SimulatedSystem:
    """Build a healthy baseline system with all catalog services running."""
    system = SimulatedSystem()
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    system.current_time = base_time

    # Deploy catalog services, all initially healthy
    for name, port, deps in _SERVICE_CATALOG:
        system.service_registry.services[name] = _svc(name, "running", port, deps)

    # Add processes for each service
    svc_resource_profile = {
        "nginx": (5.0, 64.0),
        "app_server": (12.0, 512.0),
        "api_gateway": (8.0, 256.0),
        "data_processor": (15.0, 1024.0),
        "load_balancer": (3.0, 128.0),
        "db_connector": (4.0, 128.0),
        "postgres": (2.5, 256.0),
        "redis": (1.0, 64.0),
        "monitoring": (3.0, 128.0),
        "cache_server": (2.0, 96.0),
    }
    for svc_name, (cpu, mem) in svc_resource_profile.items():
        if svc_name in system.service_registry.services:
            pid = system.process_table.add_process(
                svc_name, cpu=cpu, mem=mem, status="running", service_name=svc_name,
            )
            system.service_registry.services[svc_name].pid = pid

    return system


def _apply_crash(system: SimulatedSystem, fault: FaultSpec) -> None:
    """Mark a service as crashed and kill its process."""
    svc_name = fault.target_service
    svc = system.service_registry.services.get(svc_name)
    if svc is None:
        return
    svc.status = "crashed"
    if svc.pid is not None:
        # Remove the process
        if svc.pid in system.process_table.processes:
            del system.process_table.processes[svc.pid]
        svc.pid = None
    # Add error log entry
    system.log_buffer.append(
        timestamp=system.current_time,
        severity="ERROR",
        source=svc_name,
        message=f"{svc_name} crashed with signal 11 (SIGSEGV)",
    )


def _apply_memory_leak(system: SimulatedSystem, fault: FaultSpec) -> None:
    """Add a leaking process to the target service."""
    svc_name = fault.target_service
    svc = system.service_registry.services.get(svc_name)
    if svc is None:
        return
    # Add a high-memory worker process
    leak_pid = system.process_table.add_process(
        f"{svc_name}_worker",
        cpu=8.0,
        mem=float(fault.params.get("memory_mb", 2500)),
        status="running",
        service_name=svc_name,
    )
    # Mark service as degraded
    svc.status = "degraded"
    # Syslog entry indicating OOM
    system.log_buffer.append(
        timestamp=system.current_time,
        severity="ERROR",
        source="kernel",
        message=f"Out of memory: Killed process {leak_pid} ({svc_name}_worker)",
    )


def _apply_cascade(system: SimulatedSystem, fault: FaultSpec) -> None:
    """Trigger a cascading failure via bad config."""
    svc_name = fault.target_service
    svc = system.service_registry.services.get(svc_name)
    if svc is None:
        return
    svc.status = "crashed"
    if svc.pid is not None and svc.pid in system.process_table.processes:
        del system.process_table.processes[svc.pid]
        svc.pid = None
    # Mark dependent services as degraded
    for dep_svc_name, dep_svc in system.service_registry.services.items():
        if svc_name in dep_svc.dependencies:
            dep_svc.status = "degraded"
    # Error log
    system.log_buffer.append(
        timestamp=system.current_time,
        severity="ERROR",
        source=svc_name,
        message="cascade: upstream dependency failure",
    )


def _apply_auth_failure(system: SimulatedSystem, fault: FaultSpec) -> None:
    """Inject a database auth failure with a wrong password in config."""
    svc_name = fault.target_service
    svc = system.service_registry.services.get(svc_name)
    if svc is None:
        return
    svc.status = "crashed"
    if svc.pid is not None and svc.pid in system.process_table.processes:
        del system.process_table.processes[svc.pid]
        svc.pid = None
    # Create a config file with wrong password (following Task 3's pattern)
    system.filesystem.write_file(
        "/etc/app/database.yml",
        "host: postgres\nport: 5432\nuser: appuser\npassword: wrong_password_123\n",
    )
    # Auth error log
    system.log_buffer.append(
        timestamp=system.current_time,
        severity="FATAL",
        source=svc_name,
        message="FATAL: password authentication failed for user 'appuser'",
    )


_FAULT_APPLIERS: dict[str, Callable[[SimulatedSystem, FaultSpec], None]] = {
    "crash": _apply_crash,
    "memory_leak": _apply_memory_leak,
    "cascade": _apply_cascade,
    "auth_failure": _apply_auth_failure,
}


# ---------------------------------------------------------------------------
# Phantom alerts (theory-of-mind challenge)
# ---------------------------------------------------------------------------

_PHANTOM_ALERT_POOL: list[tuple[str, str, str]] = [
    ("redis", "high_memory", "Redis memory usage at 72% (threshold: 70%) — WARNING"),
    ("monitoring", "high_cpu", "monitoring CPU spike at 92% (threshold: 80%) — CRITICAL"),
    ("cache_server", "high_memory", "cache_server memory at 78% — WARNING"),
    ("load_balancer", "high_cpu", "load_balancer CPU at 84% — WARNING"),
]


def _sample_phantom_alerts(rng: random.Random, n: int, faulted_services: set[str]) -> list[Alert]:
    """Sample n phantom alerts that don't overlap with actually-faulted services."""
    available = [p for p in _PHANTOM_ALERT_POOL if p[0] not in faulted_services]
    rng.shuffle(available)
    alerts = []
    for svc, alert_type, desc in available[:n]:
        alerts.append(
            Alert(
                service=svc,
                alert_type=alert_type,
                severity="warning",
                description=desc,
                prominence=3,  # high prominence — these are the red herrings
            )
        )
    return alerts


# ---------------------------------------------------------------------------
# Milestone generation (per fault type)
# ---------------------------------------------------------------------------

def _make_milestones_for_fault(fault: FaultSpec) -> list[MultiAgentMilestone]:
    """Build milestones that check for resolution of a specific fault."""
    svc = fault.target_service
    milestones: list[MultiAgentMilestone] = []

    if fault.fault_type == "crash":
        milestones.append(
            MultiAgentMilestone(
                name=f"diagnosis_reads_{svc}_logs",
                credit=0.15,
                description=f"Diagnosis reads {svc} error logs",
                check=lambda actions, system, outputs, channel, svc=svc: (
                    bool(outputs.get("diagnosis", "")) and svc in outputs.get("diagnosis", "").lower()
                ),
            )
        )
        milestones.append(
            MultiAgentMilestone(
                name=f"remediation_restarts_{svc}",
                credit=0.30,
                description=f"Remediation restarts {svc}",
                check=lambda actions, system, outputs, channel, svc=svc: (
                    system.service_registry.services.get(svc)
                    and system.service_registry.services[svc].status == "running"
                ),
            )
        )
    elif fault.fault_type == "memory_leak":
        milestones.append(
            MultiAgentMilestone(
                name=f"diagnosis_identifies_{svc}_leak",
                credit=0.15,
                description=f"Diagnosis identifies memory leak in {svc}",
                check=lambda actions, system, outputs, channel, svc=svc: (
                    ("memory" in outputs.get("diagnosis", "").lower() or "oom" in outputs.get("diagnosis", "").lower())
                    and svc in outputs.get("diagnosis", "").lower()
                ),
            )
        )
        milestones.append(
            MultiAgentMilestone(
                name=f"remediation_kills_{svc}_worker",
                credit=0.30,
                description=f"Remediation kills the leaking {svc} worker process",
                check=lambda actions, system, outputs, channel, svc=svc: (
                    # Leaky worker process is gone
                    not any(
                        p.name.startswith(f"{svc}_worker")
                        for p in system.process_table.processes.values()
                    )
                ),
            )
        )
    elif fault.fault_type == "cascade":
        milestones.append(
            MultiAgentMilestone(
                name=f"diagnosis_identifies_{svc}_cascade",
                credit=0.15,
                description=f"Diagnosis identifies {svc} as cascade root cause",
                check=lambda actions, system, outputs, channel, svc=svc: (
                    "cascade" in outputs.get("diagnosis", "").lower()
                    or "dependency" in outputs.get("diagnosis", "").lower()
                ),
            )
        )
        milestones.append(
            MultiAgentMilestone(
                name=f"remediation_restores_{svc}",
                credit=0.30,
                description=f"Remediation restores {svc} and dependents",
                check=lambda actions, system, outputs, channel, svc=svc: (
                    system.service_registry.services.get(svc)
                    and system.service_registry.services[svc].status == "running"
                ),
            )
        )
    elif fault.fault_type == "auth_failure":
        milestones.append(
            MultiAgentMilestone(
                name=f"diagnosis_identifies_{svc}_auth",
                credit=0.15,
                description="Diagnosis identifies authentication failure",
                check=lambda actions, system, outputs, channel: (
                    "auth" in outputs.get("diagnosis", "").lower()
                    or "password" in outputs.get("diagnosis", "").lower()
                ),
            )
        )
        milestones.append(
            MultiAgentMilestone(
                name="remediation_fixes_password",
                credit=0.30,
                description="Remediation fixes the wrong password in config",
                check=lambda actions, system, outputs, channel: _password_fixed(system),
            )
        )

    # Universal communication milestone
    milestones.append(
        MultiAgentMilestone(
            name=f"triage_escalates_{svc}",
            credit=0.10,
            description=f"Triage sends message mentioning {svc}",
            check=lambda actions, system, outputs, channel, svc=svc: _triage_mentions(channel, svc),
        )
    )

    return milestones


def _triage_mentions(channel: CommunicationChannel, keyword: str) -> bool:
    """True if triage sent any message containing the keyword."""
    for msg in channel.get_full_history():
        if msg.from_agent == "triage" and keyword.lower() in msg.content.lower():
            return True
    return False


def _password_fixed(system: SimulatedSystem) -> bool:
    """True if /etc/app/database.yml no longer contains the wrong password."""
    try:
        content = system.filesystem.read_file("/etc/app/database.yml")
    except (ValueError, Exception):
        return False
    return "wrong_password_123" not in content


# ---------------------------------------------------------------------------
# ProceduralTask — the main public class
# ---------------------------------------------------------------------------

class ProceduralTask(WarRoomTaskBase):
    """RLVE-style procedurally generated War Room task.

    Given a difficulty level [0.0, 1.0] and a seed, produces a reproducible
    incident scenario with fault injection, phantom alerts, and milestones.
    Higher difficulty = more concurrent faults + more red herrings + tighter
    time budget.
    """

    task_id = "procedural"
    name = "Procedurally Generated Incident"
    description = "A procedurally generated multi-fault incident scenario."
    difficulty = "variable"

    def __init__(self, difficulty: float = 0.5) -> None:
        """
        Args:
            difficulty: [0.0, 1.0]. 0.0 = 1 fault, 0 phantoms, 30 rounds.
                1.0 = 3 faults, 4 phantoms, 15 rounds.
        """
        self.difficulty_level = max(0.0, min(1.0, difficulty))
        # Placeholders — populated in create_initial_state
        self._faults: list[FaultSpec] = []
        self._phantom_alerts: list[Alert] = []
        # Scale max rounds based on difficulty (more faults = tighter budget)
        self.max_rounds = int(30 - 15 * self.difficulty_level)

    def _sample_scenario(self, seed: int) -> tuple[list[FaultSpec], list[Alert]]:
        """Sample fault specs + phantom alerts for this (difficulty, seed)."""
        rng = random.Random(seed)

        # Number of faults: 1 at difficulty 0, 3 at difficulty 1
        n_faults = 1 + int(round(self.difficulty_level * 2))
        # Number of phantom alerts: 0 at difficulty 0, 4 at difficulty 1
        n_phantoms = int(round(self.difficulty_level * 4))

        faults: list[FaultSpec] = []
        faulted_services: set[str] = set()
        for _ in range(n_faults):
            f = _sample_fault(rng, faulted_services)
            faults.append(f)
            faulted_services.add(f.target_service)

        phantoms = _sample_phantom_alerts(rng, n_phantoms, faulted_services)
        return faults, phantoms

    def create_initial_state(self, seed: int) -> SimulatedSystem:
        """Build the simulated system with sampled faults applied."""
        self._faults, self._phantom_alerts = self._sample_scenario(seed)

        rng = random.Random(seed)
        system = _build_base_system(rng)

        # Apply each fault in order
        for fault in self._faults:
            applier = _FAULT_APPLIERS[fault.fault_type]
            applier(system, fault)

        return system

    def create_grader(self) -> MultiAgentGrader:
        """Build a grader with milestones for each injected fault."""
        milestones: list[MultiAgentMilestone] = []
        for fault in self._faults:
            milestones.extend(_make_milestones_for_fault(fault))

        return MultiAgentGrader(
            milestones=milestones,
            fatal_checks=[_make_fatal_check()],
        )

    def get_alert_config(self) -> dict[str, int]:
        """Promote phantom alerts to high prominence so Triage sees them first."""
        return {alert.service: alert.prominence for alert in self._phantom_alerts}

    def get_phantom_alerts(self) -> list[Alert]:
        return list(self._phantom_alerts)

    def summary(self) -> str:
        """Short human-readable description of the sampled scenario."""
        fault_descs = [f"{f.fault_type}:{f.target_service}" for f in self._faults]
        phantom_descs = [a.service for a in self._phantom_alerts]
        return (
            f"difficulty={self.difficulty_level:.2f}, "
            f"faults=[{', '.join(fault_descs)}], "
            f"phantoms=[{', '.join(phantom_descs) or 'none'}], "
            f"max_rounds={self.max_rounds}"
        )


def _make_fatal_check() -> Callable:
    """Fatal check: killing a critical service during the episode ends the game.

    Uses a closure that snapshots the initial crashed set — so we only
    penalize NEW critical-service crashes caused by the agent, not ones
    that were part of the initial fault injection.
    """
    initial_crashed: set[str] = set()

    def check(actions: MultiAgentAction, system: SimulatedSystem, outputs: dict[str, str]) -> bool:
        # On first invocation, record which critical services were already crashed
        if not initial_crashed and system is not None:
            for svc_name in _CRITICAL_SERVICES:
                svc = system.service_registry.services.get(svc_name)
                if svc is not None and svc.status == "crashed":
                    initial_crashed.add(svc_name)

        # Fatal only if a PREVIOUSLY healthy critical service is now crashed
        for svc_name in _CRITICAL_SERVICES:
            if svc_name in initial_crashed:
                continue
            svc = system.service_registry.services.get(svc_name)
            if svc is not None and svc.status == "crashed":
                return True
        return False
    return check
