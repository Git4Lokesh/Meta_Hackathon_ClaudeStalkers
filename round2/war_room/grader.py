"""MultiAgentGrader for the Multi-Agent Incident War Room."""

from __future__ import annotations
from typing import Callable, Optional
from round2.war_room.models import MultiAgentAction, RewardResult, Message
from round2.war_room.communication import CommunicationChannel
from sre_env.server.simulated_system import SimulatedSystem

# Constants
TIME_PRESSURE_PENALTY = 0.01
NOOP_PENALTY = 0.01
COMMUNICATION_USEFUL_BONUS = 0.05
COMMUNICATION_INCORRECT_PENALTY = 0.02
FATAL_SCORE = 0.01
ROLE_VIOLATION_PENALTY = 0.01

TASK_WEIGHTS = {"task1": 0.15, "task2": 0.25, "task3": 0.35, "task4": 0.25}


class MultiAgentMilestone:
    """A milestone that tracks which agent contributed."""
    def __init__(self, name: str, credit: float, description: str,
                 check: Callable[[MultiAgentAction, SimulatedSystem, dict[str, str], CommunicationChannel], bool]):
        self.name = name
        self.credit = credit
        self.description = description
        # check(actions, system, outputs_per_agent, channel) -> bool
        self.check = check


class MultiAgentGrader:
    def __init__(
        self,
        milestones: list[MultiAgentMilestone],
        fatal_checks: list[Callable[[MultiAgentAction, SimulatedSystem, dict[str, str]], bool]] | None = None,
    ):
        self.milestones = milestones
        self.fatal_checks = fatal_checks or []

        # Tracking
        self.achieved: set[str] = set()
        self.credit_assignment: dict[str, str] = {}  # {milestone: agent_role}
        self.penalties_applied: list[str] = []
        self.round_count: int = 0
        self.fatal_triggered: bool = False
        self.fatal_name: str = ""

    def evaluate(
        self,
        actions: MultiAgentAction,
        system: SimulatedSystem,
        outputs: dict[str, str],  # {role: command_output}
        channel: CommunicationChannel,
    ) -> RewardResult:
        """Evaluate one round of multi-agent actions."""
        self.round_count += 1

        if self.fatal_triggered:
            return self._build_result(done=True)

        # 1. Check fatal actions
        for check in self.fatal_checks:
            if check(actions, system, outputs):
                self.fatal_triggered = True
                self.penalties_applied.append("FATAL")
                return self._build_result(done=True)

        # 2. Check milestones
        for milestone in self.milestones:
            if milestone.name not in self.achieved:
                if milestone.check(actions, system, outputs, channel):
                    self.achieved.add(milestone.name)
                    # Credit assignment: determine which agent contributed
                    self._assign_credit(milestone.name, actions, outputs)

        # 3. Time pressure penalty
        self.penalties_applied.append("time_pressure")

        # 4. No-op penalties (per agent)
        for role in ["triage", "diagnosis", "remediation"]:
            agent_action = getattr(actions, role)
            if not agent_action.command.strip() and agent_action.message is None:
                self.penalties_applied.append(f"noop:{role}")

        # 5. Communication rewards (check if previous round's messages led to milestones)
        # This is evaluated by checking if new milestones were achieved this round
        # and if there were messages in the previous round

        return self._build_result(done=False)

    def current_score(self) -> float:
        """Compute current team score."""
        if self.fatal_triggered:
            return FATAL_SCORE

        credit = sum(m.credit for m in self.milestones if m.name in self.achieved)

        penalty = 0.0
        for p in self.penalties_applied:
            if p == "time_pressure":
                penalty += TIME_PRESSURE_PENALTY
            elif p.startswith("noop:"):
                penalty += NOOP_PENALTY
            elif p == "FATAL":
                return FATAL_SCORE
            elif p == "role_violation":
                penalty += ROLE_VIOLATION_PENALTY
            elif p == "comm_incorrect":
                penalty += COMMUNICATION_INCORRECT_PENALTY

        raw = credit - penalty
        return max(0.01, min(0.99, raw))

    def _assign_credit(self, milestone_name: str, actions: MultiAgentAction, outputs: dict[str, str]):
        """Determine which agent contributed to a milestone."""
        # Simple heuristic: the agent whose command output is non-empty and relevant
        for role in ["remediation", "diagnosis", "triage"]:  # Priority order
            agent_action = getattr(actions, role)
            if agent_action.command.strip():
                self.credit_assignment[milestone_name] = role
                return
        self.credit_assignment[milestone_name] = "team"

    def _build_result(self, done: bool) -> RewardResult:
        score = self.current_score()
        return RewardResult(
            team_reward=score,
            individual_rewards={
                "triage": score,
                "diagnosis": score,
                "remediation": score,
            },
            communication_reward=0.0,
            milestones_achieved=sorted(self.achieved),
            penalties_applied=list(self.penalties_applied),
            credit_assignment=dict(self.credit_assignment),
            done=done,
        )

    @staticmethod
    def composite_score(per_task_scores: dict[str, float]) -> float:
        """Compute weighted composite score across tasks."""
        total = 0.0
        for task_id, weight in TASK_WEIGHTS.items():
            score = per_task_scores.get(task_id, 0.01)
            score = max(0.01, min(0.99, score))
            total += weight * score
        return max(0.01, min(0.99, total))
