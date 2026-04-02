"""Core Pydantic models for the SRE Incident Response environment."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime


# ---------------------------------------------------------------------------
# OpenEnv Core Models (Pydantic BaseModel, structured for openenv compat)
# ---------------------------------------------------------------------------

class SREAction(BaseModel):
    """Agent action containing a Linux-style command string."""
    command: str = Field(..., description="Linux-style command string")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SREObservation(BaseModel):
    """Observation returned after each step."""
    output: str = Field("", description="Terminal-style text output")
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SREState(BaseModel):
    """Environment state snapshot."""
    episode_id: Optional[str] = None
    step_count: int = 0
    simulated_system: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized snapshot of SimulatedSystem",
    )


# ---------------------------------------------------------------------------
# Infrastructure Models
# ---------------------------------------------------------------------------

class ProcessRecord(BaseModel):
    """A single process in the simulated process table."""
    pid: int
    name: str
    cpu_percent: float = Field(ge=0.0, le=100.0)
    memory_mb: float = Field(ge=0.0)
    status: str = Field(pattern=r"^(running|sleeping|zombie|stopped)$")
    service_name: Optional[str] = None


class ProcessTable(BaseModel):
    """Dictionary-based process table mapping PIDs to ProcessRecords."""
    processes: Dict[int, ProcessRecord] = Field(default_factory=dict)
    next_pid: int = 1000

    def add_process(
        self,
        name: str,
        cpu: float,
        mem: float,
        status: str,
        service_name: Optional[str] = None,
    ) -> int:
        """Add a new process and return its PID."""
        pid = self.next_pid
        self.processes[pid] = ProcessRecord(
            pid=pid,
            name=name,
            cpu_percent=cpu,
            memory_mb=mem,
            status=status,
            service_name=service_name,
        )
        self.next_pid += 1
        return pid

    def kill_process(self, pid: int) -> Optional[ProcessRecord]:
        """Remove a process by PID. Returns the removed record or None."""
        return self.processes.pop(pid, None)

    def get_by_service(self, service_name: str) -> Optional[ProcessRecord]:
        """Find the process associated with a given service name."""
        for proc in self.processes.values():
            if proc.service_name == service_name:
                return proc
        return None


class FSNode(BaseModel):
    """A node in the virtual filesystem tree."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    is_dir: bool
    content: Optional[str] = None
    children: Dict[str, "FSNode"] = Field(default_factory=dict)


class VirtualFilesystem(BaseModel):
    """Dictionary-based tree representing a Linux-like filesystem."""
    root: FSNode = Field(
        default_factory=lambda: FSNode(name="/", is_dir=True)
    )

    def _resolve(self, path: str) -> List[str]:
        """Split an absolute path into component parts."""
        path = path.strip()
        if not path.startswith("/"):
            path = "/" + path
        parts = [p for p in path.split("/") if p]
        return parts

    def _get_node(self, path: str) -> Optional[FSNode]:
        """Traverse the tree and return the node at *path*, or None."""
        parts = self._resolve(path)
        node = self.root
        for part in parts:
            if not node.is_dir or part not in node.children:
                return None
            node = node.children[part]
        return node

    def _get_parent(self, path: str):
        """Return (parent_node, child_name) for *path*."""
        parts = self._resolve(path)
        if not parts:
            return None, ""
        parent = self.root
        for part in parts[:-1]:
            if not parent.is_dir or part not in parent.children:
                return None, ""
            parent = parent.children[part]
        return parent, parts[-1]

    def mkdir_p(self, path: str) -> None:
        """Create directory and all intermediate parents (like mkdir -p)."""
        parts = self._resolve(path)
        node = self.root
        for part in parts:
            if part not in node.children:
                node.children[part] = FSNode(name=part, is_dir=True)
            node = node.children[part]

    def read_file(self, path: str) -> str:
        """Read file content. Raises ValueError if not found or is a dir."""
        node = self._get_node(path)
        if node is None:
            raise ValueError(f"No such file or directory: {path}")
        if node.is_dir:
            raise ValueError(f"Is a directory: {path}")
        return node.content or ""

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file, creating parent dirs as needed."""
        parts = self._resolve(path)
        if not parts:
            raise ValueError("Cannot write to root")
        # Ensure parent directories exist
        parent = self.root
        for part in parts[:-1]:
            if part not in parent.children:
                parent.children[part] = FSNode(name=part, is_dir=True)
            parent = parent.children[part]
        fname = parts[-1]
        parent.children[fname] = FSNode(
            name=fname, is_dir=False, content=content
        )

    def edit_file(self, path: str, old: str, new: str) -> bool:
        """Replace first occurrence of *old* with *new* in file at *path*.
        Returns True on success, False if pattern not found.
        Raises ValueError if file doesn't exist.
        """
        node = self._get_node(path)
        if node is None:
            raise ValueError(f"No such file or directory: {path}")
        if node.is_dir:
            raise ValueError(f"Is a directory: {path}")
        current = node.content or ""
        if old not in current:
            return False
        node.content = current.replace(old, new, 1)
        return True

    def list_dir(self, path: str) -> List[str]:
        """List immediate children of a directory."""
        node = self._get_node(path)
        if node is None:
            raise ValueError(f"No such file or directory: {path}")
        if not node.is_dir:
            raise ValueError(f"Not a directory: {path}")
        return sorted(node.children.keys())

    def exists(self, path: str) -> bool:
        """Check whether a path exists in the filesystem."""
        return self._get_node(path) is not None


class ServiceRecord(BaseModel):
    """A service tracked in the service registry."""
    name: str
    status: str = Field(pattern=r"^(running|stopped|crashed|degraded)$")
    port: int
    dependencies: List[str] = Field(default_factory=list)
    health_endpoint: str = ""
    pid: Optional[int] = None
    started_at: Optional[datetime] = None


class ServiceRegistry(BaseModel):
    """Registry of all services and their dependency relationships."""
    services: Dict[str, ServiceRecord] = Field(default_factory=dict)

    def get_service(self, name: str) -> Optional[ServiceRecord]:
        """Return the ServiceRecord for *name*, or None."""
        return self.services.get(name)

    def set_status(self, name: str, status: str) -> None:
        """Update the status of a service. Raises ValueError if unknown."""
        svc = self.services.get(name)
        if svc is None:
            raise ValueError(f"Unit {name} not found")
        svc.status = status

    def get_dependencies(self, name: str) -> List[str]:
        """Return the dependency list for *name*."""
        svc = self.services.get(name)
        if svc is None:
            return []
        return list(svc.dependencies)

    def unmet_dependencies(self, name: str) -> List[str]:
        """Return names of dependencies that are NOT 'running'."""
        svc = self.services.get(name)
        if svc is None:
            return []
        unmet: List[str] = []
        for dep_name in svc.dependencies:
            dep = self.services.get(dep_name)
            if dep is None or dep.status != "running":
                unmet.append(dep_name)
        return unmet

    def dependency_graph_is_dag(self) -> bool:
        """Return True if the service dependency graph has no cycles."""
        # Kahn's algorithm: edges go dep -> svc (svc depends on dep)
        adj: Dict[str, List[str]] = {name: [] for name in self.services}
        in_degree: Dict[str, int] = {name: 0 for name in self.services}

        for svc in self.services.values():
            for dep in svc.dependencies:
                if dep in self.services:
                    adj[dep].append(svc.name)
                    in_degree[svc.name] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return visited == len(self.services)


class LogEntry(BaseModel):
    """A single structured log entry."""
    timestamp: datetime
    severity: str = Field(pattern=r"^(DEBUG|INFO|WARN|ERROR|FATAL)$")
    source: str
    message: str


class LogBuffer(BaseModel):
    """Ordered collection of log entries."""
    entries: List[LogEntry] = Field(default_factory=list)

    def append(
        self,
        timestamp: datetime,
        severity: str,
        source: str,
        message: str,
    ) -> None:
        """Add a new log entry to the buffer."""
        self.entries.append(
            LogEntry(
                timestamp=timestamp,
                severity=severity,
                source=source,
                message=message,
            )
        )

    def query(
        self,
        source: Optional[str] = None,
        severity: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[LogEntry]:
        """Filter log entries by source, severity, and/or timestamp."""
        results: List[LogEntry] = []
        for entry in self.entries:
            if source is not None and entry.source != source:
                continue
            if severity is not None and entry.severity != severity:
                continue
            if since is not None and entry.timestamp < since:
                continue
            results.append(entry)
        return results

    def tail(self, n: int, source: Optional[str] = None) -> List[LogEntry]:
        """Return the last *n* entries, optionally filtered by source."""
        if source is not None:
            filtered = [e for e in self.entries if e.source == source]
        else:
            filtered = list(self.entries)
        return filtered[-n:]


class MetricPoint(BaseModel):
    """A single time-series data point for system metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_mbps: float


class MetricsStore(BaseModel):
    """Time-series metrics storage for host and per-service metrics."""
    host_metrics: List[MetricPoint] = Field(default_factory=list)
    service_metrics: Dict[str, List[MetricPoint]] = Field(
        default_factory=dict
    )

    def add_point(self, service: str, point: MetricPoint) -> None:
        """Append a metric data point for a service."""
        if service not in self.service_metrics:
            self.service_metrics[service] = []
        self.service_metrics[service].append(point)

    def get_range(
        self,
        service: str,
        start: datetime,
        end: datetime,
    ) -> List[MetricPoint]:
        """Return metric points for *service* within [start, end]."""
        points = self.service_metrics.get(service, [])
        return [p for p in points if start <= p.timestamp <= end]

    def latest(self, service: str) -> Optional[MetricPoint]:
        """Return the most recent metric point for *service*, or None."""
        points = self.service_metrics.get(service, [])
        if not points:
            return None
        return points[-1]


# ---------------------------------------------------------------------------
# Command / Task / Grader Models
# ---------------------------------------------------------------------------

class ParsedCommand(BaseModel):
    """Intermediate representation of a parsed command."""
    command: str
    args: List[str] = Field(default_factory=list)
    flags: Dict[str, Optional[str]] = Field(default_factory=dict)
    raw: str = ""


class TaskDefinition(BaseModel):
    """Definition of a task scenario."""
    task_id: str
    name: str
    description: str
    max_steps: int
    difficulty: str


class MilestoneDefinition(BaseModel):
    """A grading milestone with partial credit."""
    name: str
    credit: float = Field(ge=0.0, le=1.0)
    description: str


class PenaltyDefinition(BaseModel):
    """A grading penalty definition."""
    name: str
    amount: float = Field(ge=0.0, le=1.0)
    description: str


class GraderResult(BaseModel):
    """Result of grading an episode."""
    score: float = Field(ge=0.0, le=1.0)
    milestones_achieved: List[str]
    penalties_applied: List[str]
    done: bool
