"""SRE Incident Response Environment.

Main environment class implementing the OpenEnv interface:
- reset(task_id, seed) → SREObservation
- step(command) → SREObservation
- state → SREState

This is a standalone implementation that follows the OpenEnv interface
pattern (reset/step/state) but does not require the openenv-core package
at import time.  It can later be adapted to extend MCPEnvironment when
deployed.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from sre_env.server.command_parser import CommandParser
from sre_env.server.grader import TaskGrader
from sre_env.server.models import SREAction, SREObservation, SREState
from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.tasks import TASK_REGISTRY


class SREEnvironment:
    """OpenEnv-compliant SRE Incident Response environment.

    Exposes MCP-style tools for agent interaction:
    - execute_command(command: str) -> str
    - get_system_overview() -> dict
    - get_available_commands() -> list
    """

    def __init__(self) -> None:
        self._system: Optional[SimulatedSystem] = None
        self._grader: Optional[TaskGrader] = None
        self._parser = CommandParser()
        self._state = SREState()
        self._task_id: Optional[str] = None
        self._max_steps: int = 0
        self._done: bool = False
        self._initial_observation: Optional[SREObservation] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "task1",
        **kwargs: Any,
    ) -> SREObservation:
        """Initialize the environment for a specific task.

        Args:
            seed: Random seed for deterministic state generation (default: 42).
            episode_id: Optional episode identifier.
            task_id: Task to load ("task1", "task2", "task3").

        Returns:
            Initial SREObservation with system overview.

        Raises:
            ValueError: If *task_id* is not valid.
        """
        if task_id not in TASK_REGISTRY:
            valid = ", ".join(sorted(TASK_REGISTRY.keys()))
            raise ValueError(f"Invalid task_id '{task_id}'. Valid: {valid}")

        seed = seed if seed is not None else 42

        # Create task definition and initial state
        task_cls = TASK_REGISTRY[task_id]
        task_def = task_cls()
        self._system = task_def.create_initial_state(seed)
        self._grader = task_def.create_grader()
        self._task_id = task_id
        self._max_steps = task_def.max_steps
        self._done = False

        # Initialize state
        self._state = SREState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            simulated_system=self._system.snapshot(),
        )

        # Build initial observation with system overview
        overview = self._build_system_overview()
        initial_output = (
            f"=== SRE Incident Response ===\n"
            f"Task: {task_def.name} ({task_def.difficulty})\n"
            f"Description: {task_def.description}\n"
            f"Max steps: {task_def.max_steps}\n"
            f"\n{overview}\n"
            f"\nType 'help' for available commands."
        )

        obs = SREObservation(
            output=initial_output,
            done=False,
            reward=0.0,
            metadata={
                "task_id": task_id,
                "task_name": task_def.name,
                "difficulty": task_def.difficulty,
                "max_steps": task_def.max_steps,
                "step": 0,
            },
        )
        self._initial_observation = obs
        return obs

    def step(self, action: SREAction, **kwargs: Any) -> SREObservation:
        """Execute a command and return the observation.

        Args:
            action: SREAction with command string.

        Returns:
            SREObservation with command output, reward, and done flag.

        Raises:
            RuntimeError: If ``reset()`` hasn't been called.
        """
        if self._system is None or self._grader is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )

        if self._done:
            return SREObservation(
                output="Episode is complete. Call reset() to start a new episode.",
                done=True,
                reward=self._grader.current_score(),
                metadata=self._terminal_metadata(),
            )

        # Increment step count
        self._state.step_count += 1

        # Execute command against simulated system
        command = action.command.strip()
        output = self._parser.execute(command, self._system)

        # Advance simulated time
        self._system.advance_time(5)

        # Evaluate grader (includes all advanced reward signals)
        reward = self._grader.evaluate(command, self._system, output)

        # Check if done: fatal action, step limit, or all milestones achieved
        fatal = self._grader.fatal_triggered
        all_milestones = all(
            m.name in self._grader.achieved for m in self._grader.milestones
        )
        all_state_checks = all(
            sc.name in self._grader.state_achieved
            for sc in self._grader.state_checks
        ) if self._grader.state_checks else True
        step_limit_reached = self._state.step_count >= self._max_steps
        self._done = fatal or (all_milestones and all_state_checks) or step_limit_reached

        # Update state snapshot
        self._state.simulated_system = self._system.snapshot()

        # Build metadata with health score for observability
        metadata: dict[str, Any] = {
            "task_id": self._task_id,
            "step": self._state.step_count,
            "max_steps": self._max_steps,
            "health_score": self._grader.prev_health if self._grader.prev_health is not None else 1.0,
        }

        # Add fatal action info if triggered
        if fatal:
            output = (
                f"⚠️  FATAL ACTION: {self._grader.fatal_name}\n"
                f"Episode terminated immediately. Score: 0.0\n\n"
                f"Original output: {output}"
            )

        if self._done:
            metadata.update(self._terminal_metadata())

        return SREObservation(
            output=output,
            done=self._done,
            reward=reward,
            metadata=metadata,
        )

    @property
    def state(self) -> SREState:
        """Return the current environment state."""
        return self._state

    # ------------------------------------------------------------------
    # MCP tool implementations
    # ------------------------------------------------------------------

    def execute_command(self, command: str) -> str:
        """MCP tool: Execute a Linux-style command."""
        action = SREAction(command=command)
        obs = self.step(action)
        return obs.output

    def get_system_overview(self) -> dict:
        """MCP tool: Get high-level system status summary."""
        if self._system is None:
            return {"error": "Environment not initialized"}

        services: dict[str, Any] = {}
        for name, svc in self._system.service_registry.services.items():
            services[name] = {
                "status": svc.status,
                "port": svc.port,
                "pid": svc.pid,
                "dependencies": svc.dependencies,
            }

        procs = len(self._system.process_table.processes)
        total_cpu = sum(
            p.cpu_percent
            for p in self._system.process_table.processes.values()
        )
        total_mem = sum(
            p.memory_mb
            for p in self._system.process_table.processes.values()
        )

        return {
            "services": services,
            "process_count": procs,
            "total_cpu_percent": round(total_cpu, 1),
            "total_memory_mb": round(total_mem, 1),
            "current_time": self._system.current_time.isoformat(),
        }

    def get_available_commands(self) -> list:
        """MCP tool: List supported commands."""
        return CommandParser.SUPPORTED_COMMANDS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_overview(self) -> str:
        """Build a text summary of the current system state."""
        if self._system is None:
            return ""

        lines = ["System Overview:"]
        lines.append("  Services:")
        for name, svc in sorted(
            self._system.service_registry.services.items()
        ):
            if svc.status == "running":
                status_icon = "●"
            elif svc.status == "stopped":
                status_icon = "○"
            else:
                status_icon = "✗"
            lines.append(
                f"    {status_icon} {name}: {svc.status} (port {svc.port})"
            )

        procs = self._system.process_table.processes
        total_cpu = sum(p.cpu_percent for p in procs.values())
        total_mem = sum(p.memory_mb for p in procs.values())
        lines.append(f"\n  Processes: {len(procs)} running")
        lines.append(f"  CPU: {total_cpu:.1f}% total")
        lines.append(
            f"  Memory: {total_mem:.0f} MB used / 8192 MB total"
        )

        return "\n".join(lines)

    def _terminal_metadata(self) -> dict:
        """Build terminal metadata with score, milestones, penalties."""
        result = self._grader.result(done=True)
        return {
            "score": result.score,
            "milestones_achieved": result.milestones_achieved,
            "penalties_applied": result.penalties_applied,
        }
