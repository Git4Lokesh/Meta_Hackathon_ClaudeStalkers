"""Procedural Task Generator — RLVE-style infinite task diversity.

Generates War Room tasks procedurally from a small library of "fault primitives":
  - crash        — a service transitions to "crashed" status
  - memory_leak  — a process exceeds memory budget (OOM risk)
  - cascade      — config misconfiguration triggers dependency cascade
  - auth_failure — database auth fails causing upstream services to degrade
  - disk_full    — disk reaches near-capacity, blocking writes for a service

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
    fault_type: str  # "crash" | "memory_leak" | "cascade" | "auth_failure" | "disk_full"
    target_service: str
    params: dict = field(default_factory=dict)


# Registered fault types. Adding a new primitive is a matter of:
#   1. Add a name here
#   2. Provide a candidate-service rule in _sample_fault
#   3. Register an applier in _FAULT_APPLIERS
#   4. Register milestone builders in _make_milestones_for_fault
ALL_FAULT_TYPES: list[str] = [
    "crash",
    "memory_leak",
    "cascade",
    "auth_failure",
    "disk_full",
]


def _sample_fault(
    rng: random.Random,
    already_faulted: set[str],
    allowed_types: list[str] | None = None,
) -> FaultSpec:
    """Sample a random fault that doesn't overlap with already-faulted services."""
    fault_types = allowed_types or ALL_FAULT_TYPES
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
    elif fault_type == "auth_failure":
        candidates = ["db_connector"]
        candidates = [s for s in candidates if s not in already_faulted]
    else:  # disk_full
        # disk_full targets services that produce significant log/data volume
        candidates = ["data_processor", "app_server", "monitoring", "nginx"]
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
    elif fault_type == "disk_full":
        params["disk_pct"] = rng.randint(96, 100)
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


def _apply_disk_full(system: SimulatedSystem, fault: FaultSpec) -> None:
    """Fill the root filesystem to near-capacity, blocking the target service.

    The agent must:
      * notice via ``df`` / log entries that disk is critical, and
      * free space — modeled here as killing the runaway logger worker that
        the fault spawned. ``ProcessTable.kill_process`` triggers a hook
        registered via ``system.config_validators[svc_name]`` (or, more
        directly, the ``WarRoomEnvironment`` post-step hook) — for now we
        rely on a periodic check inside the env to refresh disk_usage when
        the worker is gone. Simpler: the milestone primitive ``disk_freed``
        watches for ``disk_usage["/"] < 90``, and we re-evaluate that on
        each step in ``_post_step_disk_recovery``.
    """
    svc_name = fault.target_service
    svc = system.service_registry.services.get(svc_name)
    pct = int(fault.params.get("disk_pct", 98))

    # Mutate disk state — df now reflects the full disk
    if not hasattr(system, "disk_usage"):
        system.disk_usage = {"/": pct}
    else:
        system.disk_usage["/"] = pct

    # Spawn a runaway "logger" worker — the disk_freed milestone fires once
    # this process is killed (we recompute disk_usage from the worker's
    # presence in a post-step hook installed by the procedural task).
    logger_pid = system.process_table.add_process(
        f"{svc_name}_logger",
        cpu=2.0,
        mem=128.0,
        status="running",
        service_name=svc_name,
    )
    fault.params["logger_pid"] = logger_pid

    # Drop a large "log file" representing the offender
    try:
        system.filesystem.write_file(
            f"/var/log/{svc_name}/runaway.log",
            "X" * 1024,
        )
    except Exception:
        pass

    if svc is not None:
        # Service is degraded — it's still alive but cannot write
        svc.status = "degraded"

    system.log_buffer.append(
        timestamp=system.current_time,
        severity="ERROR",
        source=svc_name,
        message=f"{svc_name}: write failed: No space left on device (/)",
    )
    system.log_buffer.append(
        timestamp=system.current_time,
        severity="WARN",
        source="kernel",
        message=f"disk usage on / is {pct}% — critical threshold exceeded",
    )


_FAULT_APPLIERS: dict[str, Callable[[SimulatedSystem, FaultSpec], None]] = {
    "crash": _apply_crash,
    "memory_leak": _apply_memory_leak,
    "cascade": _apply_cascade,
    "auth_failure": _apply_auth_failure,
    "disk_full": _apply_disk_full,
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
# Milestone PRIMITIVES (composable building blocks)
# ---------------------------------------------------------------------------
# These helpers return milestone-check callables. They are deliberately
# small and named — assembling a new task's grader becomes declarative:
#
#     milestones = [
#         triage_mentions(svc),
#         diagnosis_says_about(svc, ["auth", "password"]),
#         service_running(svc),
#     ]
#
# Adding a new fault type is then a 3-step operation: define an applier
# in _FAULT_APPLIERS, list which primitives prove it, and you're done.


def triage_mentions(svc: str, credit: float = 0.10) -> MultiAgentMilestone:
    """Triage sends any message containing the service name."""
    def _check(actions, system, outputs, channel) -> bool:
        for msg in channel.get_full_history():
            if msg.from_agent == "triage" and svc.lower() in msg.content.lower():
                return True
        return False
    return MultiAgentMilestone(
        name=f"triage_escalates_{svc}",
        credit=credit,
        description=f"Triage sends message mentioning {svc}",
        check=_check,
    )


def diagnosis_says_about(svc: str, keywords: list[str], credit: float = 0.15) -> MultiAgentMilestone:
    """Diagnosis output mentions the service AND at least one of the keywords."""
    kw_lower = [k.lower() for k in keywords]
    name_kw = "_".join(kw_lower[:1] + ["etc"]) if len(kw_lower) > 1 else (kw_lower[0] if kw_lower else "any")

    def _check(actions, system, outputs, channel) -> bool:
        diag = outputs.get("diagnosis", "").lower()
        if not diag:
            return False
        if svc.lower() not in diag:
            return False
        return any(k in diag for k in kw_lower)

    return MultiAgentMilestone(
        name=f"diagnosis_identifies_{svc}_{name_kw}",
        credit=credit,
        description=f"Diagnosis identifies {svc} via keywords {keywords}",
        check=_check,
    )


def diagnosis_inspects(svc: str, credit: float = 0.15) -> MultiAgentMilestone:
    """Diagnosis ran any inspection command whose output mentions the service."""
    def _check(actions, system, outputs, channel) -> bool:
        diag = outputs.get("diagnosis", "")
        return bool(diag) and svc.lower() in diag.lower()
    return MultiAgentMilestone(
        name=f"diagnosis_inspects_{svc}",
        credit=credit,
        description=f"Diagnosis inspects {svc} (output mentions service)",
        check=_check,
    )


def service_running(svc: str, credit: float = 0.30) -> MultiAgentMilestone:
    """Target service is back to running."""
    def _check(actions, system, outputs, channel) -> bool:
        s = system.service_registry.services.get(svc)
        return bool(s) and s.status == "running"
    return MultiAgentMilestone(
        name=f"remediation_restores_{svc}",
        credit=credit,
        description=f"Remediation restores {svc} to running",
        check=_check,
    )


def worker_killed(svc: str, credit: float = 0.30) -> MultiAgentMilestone:
    """All ``{svc}_worker`` processes have been removed."""
    def _check(actions, system, outputs, channel) -> bool:
        return not any(
            p.name.startswith(f"{svc}_worker")
            for p in system.process_table.processes.values()
        )
    return MultiAgentMilestone(
        name=f"remediation_kills_{svc}_worker",
        credit=credit,
        description=f"Remediation kills the leaking {svc} worker process",
        check=_check,
    )


def password_fixed(credit: float = 0.30) -> MultiAgentMilestone:
    """Wrong password no longer present in /etc/app/database.yml."""
    def _check(actions, system, outputs, channel) -> bool:
        try:
            content = system.filesystem.read_file("/etc/app/database.yml")
        except (ValueError, Exception):
            return False
        return "wrong_password_123" not in content
    return MultiAgentMilestone(
        name="remediation_fixes_password",
        credit=credit,
        description="Remediation fixes the wrong password in config",
        check=_check,
    )


def disk_freed(svc: str, credit: float = 0.30) -> MultiAgentMilestone:
    """Disk pressure resolved — ``{svc}_logger`` runaway worker has been killed.

    Implementation: we model "disk recovery" via process-table state. When
    the runaway logger spawned by ``_apply_disk_full`` is gone, we declare
    the disk freed. (We also auto-refresh ``system.disk_usage`` to reflect
    this so subsequent ``df`` calls show a healthy disk.)
    """
    def _check(actions, system, outputs, channel) -> bool:
        runaway_alive = any(
            p.name == f"{svc}_logger"
            for p in system.process_table.processes.values()
        )
        if not runaway_alive:
            # Reflect the recovery in disk_usage so df shows a healthy disk
            if hasattr(system, "disk_usage"):
                system.disk_usage["/"] = 45
            return True
        return False

    return MultiAgentMilestone(
        name=f"remediation_frees_disk_{svc}",
        credit=credit,
        description=f"Remediation frees disk by killing runaway {svc}_logger",
        check=_check,
    )


# ---------------------------------------------------------------------------
# Milestone composition per fault type
# ---------------------------------------------------------------------------

def _make_milestones_for_fault(fault: FaultSpec) -> list[MultiAgentMilestone]:
    """Build milestones that check for resolution of a specific fault.

    Composed from the named primitives above. Each fault type maps to a
    short, declarative recipe — adding a new fault means adding one branch.
    """
    svc = fault.target_service
    milestones: list[MultiAgentMilestone] = []

    if fault.fault_type == "crash":
        milestones.append(diagnosis_inspects(svc))
        milestones.append(service_running(svc))
    elif fault.fault_type == "memory_leak":
        milestones.append(diagnosis_says_about(svc, ["memory", "oom"]))
        milestones.append(worker_killed(svc))
    elif fault.fault_type == "cascade":
        milestones.append(diagnosis_says_about(svc, ["cascade", "dependency"]))
        milestones.append(service_running(svc))
    elif fault.fault_type == "auth_failure":
        milestones.append(diagnosis_says_about(svc, ["auth", "password"]))
        milestones.append(password_fixed())
    elif fault.fault_type == "disk_full":
        milestones.append(diagnosis_says_about(svc, ["disk", "space", "full"]))
        milestones.append(disk_freed(svc))

    # Universal communication milestone
    milestones.append(triage_mentions(svc))

    return milestones


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
