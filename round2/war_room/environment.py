"""WarRoomEnvironment: Multi-Agent Incident War Room.

OpenEnv-compliant environment where three specialized agents (Triage,
Diagnosis, Remediation) cooperate through a shared communication channel
to diagnose and fix production infrastructure failures.
"""

from __future__ import annotations
from typing import Any, Optional
from uuid import uuid4

from round2.war_room.models import (
    MultiAgentAction, MultiAgentObservation, AgentObservation,
    AgentAction, WarRoomState, Message,
)
from round2.war_room.communication import CommunicationChannel
from round2.war_room.alert_engine import AlertEngine
from round2.war_room.grader import MultiAgentGrader
from round2.war_room.observation_builders import (
    build_triage_observation,
    build_diagnosis_observation,
    build_remediation_observation,
)
from round2.war_room.role_permissions import validate_command
from round2.war_room.tasks import WAR_ROOM_TASK_REGISTRY
from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.command_parser import CommandParser


class WarRoomEnvironment:
    """Multi-Agent Incident War Room environment."""

    def __init__(self) -> None:
        self._system: Optional[SimulatedSystem] = None
        self._channel: Optional[CommunicationChannel] = None
        self._alert_engine: Optional[AlertEngine] = None
        self._grader: Optional[MultiAgentGrader] = None
        self._parser = CommandParser()
        self._task_id: Optional[str] = None
        self._max_rounds: int = 0
        self._round_number: int = 0
        self._done: bool = False
        self._episode_id: str = ""
        self._prev_outputs: dict[str, str] = {
            "triage": "",
            "diagnosis": "",
            "remediation": "",
        }

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = "task1",
        seed: int = 42,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MultiAgentObservation:
        """Initialize the environment for a specific task."""
        if task_id not in WAR_ROOM_TASK_REGISTRY:
            valid = ", ".join(sorted(WAR_ROOM_TASK_REGISTRY.keys()))
            raise ValueError(f"Invalid task_id '{task_id}'. Valid: {valid}")

        # Create task and initial state
        task_cls = WAR_ROOM_TASK_REGISTRY[task_id]
        task_def = task_cls()
        self._system = task_def.create_initial_state(seed)
        self._grader = task_def.create_grader()
        self._channel = CommunicationChannel()
        self._alert_engine = AlertEngine(
            prominence_overrides=task_def.get_alert_config(),
        )
        self._task_id = task_id
        self._max_rounds = task_def.max_rounds
        self._round_number = 0
        self._done = False
        self._episode_id = episode_id or str(uuid4())
        self._prev_outputs = {
            "triage": "",
            "diagnosis": "",
            "remediation": "",
        }

        # Generate initial alerts
        self._alert_engine.evaluate(self._system)

        # Build initial observations
        alerts = self._alert_engine.get_active_alerts()
        triage_text = build_triage_observation(
            self._system, alerts, self._channel, 0, self._max_rounds,
        )
        diagnosis_text = build_diagnosis_observation(
            self._system, self._channel, 0, self._max_rounds,
        )
        remediation_text = build_remediation_observation(
            self._system, self._channel, 0, self._max_rounds,
        )

        return MultiAgentObservation(
            triage=AgentObservation(text=triage_text),
            diagnosis=AgentObservation(text=diagnosis_text),
            remediation=AgentObservation(text=remediation_text),
            team_reward=0.0,
            done=False,
            metadata={
                "task_id": task_id,
                "task_name": task_def.name,
                "difficulty": task_def.difficulty,
                "max_rounds": self._max_rounds,
                "round": 0,
            },
        )

    def step(
        self, action: MultiAgentAction, **kwargs: Any
    ) -> MultiAgentObservation:
        """Process one round of multi-agent actions."""
        if self._system is None or self._grader is None or self._channel is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )

        if self._done:
            return self._build_terminal_observation()

        self._round_number += 1
        outputs: dict[str, str] = {}

        # Process each agent in order: triage → diagnosis → remediation
        for role in ("triage", "diagnosis", "remediation"):
            agent_action: AgentAction = getattr(action, role)
            command = agent_action.command.strip()

            # Handle send_message action
            if agent_action.message is not None:
                msg = agent_action.message
                self._channel.send(
                    from_agent=role,
                    to_agent=msg.to_agent,
                    content=msg.content,
                    timestamp=self._system.current_time,
                )

            # Handle command
            if command:
                allowed, error_msg = validate_command(role, command)
                if not allowed:
                    outputs[role] = error_msg or f"Command not allowed for {role}"
                    self._grader.penalties_applied.append("role_violation")
                elif role == "triage":
                    outputs[role] = self._handle_triage_command(command)
                else:
                    outputs[role] = self._parser.execute(command, self._system)
            else:
                outputs[role] = ""

        # Advance simulated time
        self._system.advance_time(5)

        # Advance communication round
        self._channel.advance_round()

        # Update alerts
        self._alert_engine.evaluate(self._system)

        # Evaluate grader
        reward_result = self._grader.evaluate(
            action, self._system, outputs, self._channel,
        )

        # Check termination conditions
        all_milestones = all(
            m.name in self._grader.achieved for m in self._grader.milestones
        )
        round_limit = self._round_number >= self._max_rounds
        fatal = self._grader.fatal_triggered
        comm_breakdown = self._channel.rounds_without_any_messages() >= 3

        self._done = all_milestones or round_limit or fatal or comm_breakdown

        if comm_breakdown and not fatal:
            self._grader.penalties_applied.append("communication_breakdown")

        # Store outputs for next round's observations
        self._prev_outputs = outputs

        # Build observations
        alerts = self._alert_engine.get_active_alerts()
        triage_text = build_triage_observation(
            self._system, alerts, self._channel,
            self._round_number, self._max_rounds,
        )
        diagnosis_text = build_diagnosis_observation(
            self._system, self._channel,
            self._round_number, self._max_rounds,
            prev_output=outputs.get("diagnosis", ""),
        )
        remediation_text = build_remediation_observation(
            self._system, self._channel,
            self._round_number, self._max_rounds,
            prev_output=outputs.get("remediation", ""),
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "task_id": self._task_id,
            "round": self._round_number,
            "max_rounds": self._max_rounds,
        }
        if self._done:
            metadata["score"] = reward_result.team_reward
            metadata["milestones_achieved"] = reward_result.milestones_achieved
            metadata["penalties_applied"] = reward_result.penalties_applied
            metadata["credit_assignment"] = reward_result.credit_assignment

        return MultiAgentObservation(
            triage=AgentObservation(
                text=triage_text,
                reward=reward_result.individual_rewards.get("triage", 0.0),
                messages=self._channel.get_messages_for(
                    "triage", self._round_number,
                ),
            ),
            diagnosis=AgentObservation(
                text=diagnosis_text,
                reward=reward_result.individual_rewards.get("diagnosis", 0.0),
                messages=self._channel.get_messages_for(
                    "diagnosis", self._round_number,
                ),
            ),
            remediation=AgentObservation(
                text=remediation_text,
                reward=reward_result.individual_rewards.get(
                    "remediation", 0.0,
                ),
                messages=self._channel.get_messages_for(
                    "remediation", self._round_number,
                ),
            ),
            team_reward=reward_result.team_reward,
            done=self._done,
            metadata=metadata,
        )

    @property
    def state(self) -> WarRoomState:
        """Return the full environment state."""
        return WarRoomState(
            episode_id=self._episode_id,
            round_number=self._round_number,
            max_rounds=self._max_rounds,
            task_id=self._task_id or "",
            simulated_system=(
                self._system.snapshot() if self._system else {}
            ),
            communication_history=(
                self._channel.get_full_history() if self._channel else []
            ),
            alerts=[
                a
                for a in (
                    self._alert_engine.get_active_alerts()
                    if self._alert_engine
                    else []
                )
            ],
            per_agent_tracking={
                "milestones": {
                    m: True
                    for m in sorted(self._grader.achieved)
                } if self._grader else {},
                "credit_assignment": dict(self._grader.credit_assignment)
                if self._grader
                else {},
            },
            done=self._done,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_triage_command(self, command: str) -> str:
        """Handle triage-specific commands."""
        cmd = command.strip()

        if cmd == "get_dashboard":
            return self._alert_engine.get_dashboard_summary()
        elif cmd == "get_alerts":
            return self._alert_engine.format_alerts()
        elif cmd == "get_health_summary":
            return self._build_health_summary()
        elif cmd.startswith("escalate"):
            return self._handle_escalate(cmd)
        elif cmd.startswith("send_message"):
            return "Message sent via communication channel."
        else:
            return f"Unknown triage command: {cmd}"

    def _build_health_summary(self) -> str:
        """Build a health summary for the triage agent."""
        if self._system is None:
            return "System not initialized."

        services = self._system.service_registry.services
        running = sum(
            1 for s in services.values() if s.status == "running"
        )
        total = len(services)
        procs = self._system.process_table.processes
        total_cpu = sum(p.cpu_percent for p in procs.values())
        total_mem = sum(p.memory_mb for p in procs.values())

        lines = [
            "Health Summary:",
            f"  Services: {running}/{total} healthy",
            f"  Processes: {len(procs)} running",
            f"  CPU: {total_cpu:.1f}% total",
            f"  Memory: {total_mem:.0f} MB / 8192 MB",
        ]

        unhealthy = [
            s for s in services.values() if s.status != "running"
        ]
        if unhealthy:
            lines.append("\n  Unhealthy services:")
            for s in unhealthy:
                lines.append(
                    f"    ✗ {s.name}: {s.status} (port {s.port})"
                )

        return "\n".join(lines)

    def _handle_escalate(self, command: str) -> str:
        """Handle escalate command: escalate <agent> <description>."""
        parts = command.split(maxsplit=2)
        if len(parts) < 3:
            return "Usage: escalate <agent> <description>"

        target_agent = parts[1]
        description = parts[2]

        if target_agent not in ("diagnosis", "remediation", "all"):
            return (
                f"Invalid target agent: {target_agent}. "
                "Use 'diagnosis', 'remediation', or 'all'."
            )

        if self._channel and self._system:
            self._channel.send(
                from_agent="triage",
                to_agent=target_agent,
                content=f"[ESCALATION] {description}",
                timestamp=self._system.current_time,
            )

        return f"Escalated to {target_agent}: {description}"

    def _build_terminal_observation(self) -> MultiAgentObservation:
        """Build observation for when episode is already done."""
        score = self._grader.current_score() if self._grader else 0.01
        return MultiAgentObservation(
            triage=AgentObservation(
                text="Episode complete. Call reset() to start a new episode.",
            ),
            diagnosis=AgentObservation(
                text="Episode complete. Call reset() to start a new episode.",
            ),
            remediation=AgentObservation(
                text="Episode complete. Call reset() to start a new episode.",
            ),
            team_reward=score,
            done=True,
            metadata={
                "task_id": self._task_id,
                "round": self._round_number,
                "score": score,
                "milestones_achieved": sorted(self._grader.achieved)
                if self._grader
                else [],
                "penalties_applied": list(self._grader.penalties_applied)
                if self._grader
                else [],
            },
        )
