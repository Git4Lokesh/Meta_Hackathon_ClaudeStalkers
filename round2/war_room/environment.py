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
from round2.war_room.adaptive import PerformanceTracker
from round2.war_room.belief_tracker import BeliefStateTracker
from sre_env.server.simulated_system import SimulatedSystem
from sre_env.server.command_parser import CommandParser

# OpenEnv base class — try both import paths, graceful fallback if not installed.
# Prefer the canonical `openenv.core` path; `openenv_core` is the deprecated alias.
# We also catch TypeError because older openenv-core uses `kw_only=True` (needs Python 3.10+).
try:
    from openenv.core.env_server import Environment as OpenEnvBase
except (ImportError, TypeError):
    try:
        from openenv_core.env_server import Environment as OpenEnvBase
    except (ImportError, TypeError):
        class OpenEnvBase:
            """Stub when openenv-core is not installed (or Python < 3.10)."""
            pass

EXECUTIVE_PANIC_MESSAGES = [
    "🔥 CEO is asking for an update! Are we back up yet?! Revenue is dropping!",
    "⚠️ The board meeting is in 30 minutes. I need this fixed NOW. Have you tried restarting everything?",
    "💰 We're losing $10K per minute! Just restart the database, that usually fixes things!",
    "📞 Customer support is overwhelmed. Can someone just reboot all the servers?",
    "🚨 The VP of Engineering wants to know why we haven't rolled back yet. ROLL BACK NOW!",
    "😤 This is taking too long. I'm going to SSH in and fix it myself if you don't hurry up.",
]


class WarRoomEnvironment(OpenEnvBase):
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
        self._performance_tracker = PerformanceTracker()
        self._executive_enabled: bool = True
        self._executive_interval: int = 3  # Inject panic every N rounds
        self._belief_tracker: Optional[BeliefStateTracker] = None

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
        self._task_def = task_def  # exposed for introspecting heuristics/eval
        self._system = task_def.create_initial_state(seed)
        self._grader = task_def.create_grader()
        self._channel = CommunicationChannel()
        self._alert_engine = AlertEngine(
            prominence_overrides=task_def.get_alert_config(),
            phantom_alerts=task_def.get_phantom_alerts(),
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

        # Initialize Belief State Tracker with ground truth from system state
        ground_truth = {}
        for svc_name, svc in self._system.service_registry.services.items():
            ground_truth[svc_name] = {"status": svc.status}
        phantom_entities = [a.service for a in task_def.get_phantom_alerts()]
        self._belief_tracker = BeliefStateTracker(
            ground_truth=ground_truth,
            phantom_entities=phantom_entities,
        )

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

        # Panicked Executive injection
        if self._executive_enabled and self._round_number % self._executive_interval == 0:
            import random
            panic_msg = random.choice(EXECUTIVE_PANIC_MESSAGES)
            self._channel.send(
                from_agent="executive",
                to_agent="all",
                content=panic_msg,
                timestamp=self._system.current_time,
            )

        # Update Belief State Tracker
        if self._belief_tracker:
            self._belief_tracker.set_round(self._round_number)
            # Track commands that reveal beliefs
            for role in ("triage", "diagnosis", "remediation"):
                agent_action = getattr(action, role)
                if agent_action.command:
                    self._belief_tracker.record_command(role, agent_action.command)
                # Track messages — infer beliefs from content
                if agent_action.message:
                    msg = agent_action.message
                    content_lower = msg.content.lower()
                    # Detect pushback messages (Theory of Mind)
                    if "not" in content_lower or "false" in content_lower or "stale" in content_lower:
                        for entity in list(self._belief_tracker._ground_truth.keys()):
                            if entity.lower() in content_lower:
                                self._belief_tracker.record_pushback(
                                    role, msg.to_agent, entity, msg.content,
                                )
            # Update ground truth with current service states
            for svc_name, svc in self._system.service_registry.services.items():
                self._belief_tracker.update_ground_truth(svc_name, "status", svc.status)
            # Detect belief conflicts
            self._belief_tracker.detect_conflicts()

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

        # Record episode for adaptive difficulty
        if self._done and self._grader:
            score = self._grader.current_score()
            self._performance_tracker.record_episode(
                self._task_id or "", score, self._round_number,
            )

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
            "reward_components": reward_result.reward_components,
            "penalty_reasons": reward_result.penalty_reasons,
            "penalties_applied": reward_result.penalties_applied,
        }
        if self._done:
            metadata["score"] = reward_result.team_reward
            metadata["milestones_achieved"] = reward_result.milestones_achieved
            metadata["credit_assignment"] = reward_result.credit_assignment
            metadata["adaptive_difficulty"] = self._performance_tracker.summary()
            # Include Belief State and Deception Score
            if self._belief_tracker:
                metadata["belief_state"] = self._belief_tracker.get_snapshot()
                metadata["deception_score"] = self._belief_tracker.get_deception_score()

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

    def get_performance_summary(self) -> dict:
        """Return adaptive difficulty performance summary."""
        return self._performance_tracker.summary()

    def inject_chaos(self) -> str:
        """Inject a random failure into the running system.

        Called by the 'INJECT CHAOS' button in the Gradio dashboard.
        Kills a random healthy service's process, causing a new incident
        mid-episode that agents must handle.
        """
        if self._system is None or self._done:
            return "Cannot inject chaos: no active episode."

        import random

        # Find healthy services (running, not already being fixed)
        healthy = [
            name for name, svc in self._system.service_registry.services.items()
            if svc.status == "running" and svc.pid is not None
        ]

        if not healthy:
            return "No healthy services to disrupt!"

        # Pick a random healthy service and kill it
        target = random.choice(healthy)
        svc = self._system.service_registry.services[target]
        pid = svc.pid

        # Kill the process
        self._system.kill_process(pid)

        # Inject a chaos message into the channel
        self._channel.send(
            from_agent="chaos_monkey",
            to_agent="all",
            content=f"🐒💥 CHAOS MONKEY: Killed {target} (PID {pid})! A new incident has been injected!",
            timestamp=self._system.current_time,
        )

        # Update alerts
        self._alert_engine.evaluate(self._system)

        return f"💥 Chaos injected: killed {target} (PID {pid})"

    def inject_external_message(
        self,
        content: str,
        from_agent: str = "executive",
        to_agent: str = "all",
    ) -> str:
        """Inject an external message into the communication channel."""
        if self._system is None or self._channel is None:
            return "Cannot inject message: no active episode."
        payload = content.strip()
        if not payload:
            return "Cannot inject message: empty content."
        self._channel.send(
            from_agent=from_agent,
            to_agent=to_agent,
            content=payload,
            timestamp=self._system.current_time,
        )
        return f"Injected message from {from_agent} to {to_agent}."

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
            dashboard = self._alert_engine.get_dashboard_summary()
            # Record triage beliefs from dashboard alerts
            if self._belief_tracker and self._alert_engine:
                for alert in self._alert_engine.get_active_alerts():
                    status = "critical" if alert.severity == "critical" else "warning"
                    self._belief_tracker.record_observation(
                        "triage", "dashboard",
                        {alert.service: status},
                    )
            return dashboard
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
