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

# Milestone → relevant keywords mapping
MILESTONE_KEYWORDS: dict[str, list[str]] = {
    "triage_escalates_nginx": ["nginx", "down", "crash"],
    "diagnosis_reads_logs": ["log", "error", "nginx"],
    "diagnosis_messages_findings": ["nginx", "crash", "restart", "signal"],
    "remediation_restarts_nginx": ["restart", "nginx"],
    "triage_prioritizes_memory": ["memory", "oom", "leak"],
    "diagnosis_identifies_pid": ["pid", "process", "kill"],
    "diagnosis_reads_oom": ["oom", "memory", "kill"],
    "remediation_kills_correct": ["kill", "pid"],
    "diagnosis_identifies_db_auth": ["auth", "password", "database"],
    "diagnosis_reads_config": ["database.yml", "password", "config"],
    "remediation_fixes_config": ["edit", "password", "fix"],
}


def _message_relevant_to_milestone(content_lower: str, milestone_name: str) -> bool:
    """Return True if the message content matches ≥2 keywords for the milestone."""
    keywords = MILESTONE_KEYWORDS.get(milestone_name, [])
    if not keywords:
        return False
    return sum(1 for kw in keywords if kw in content_lower) >= 2


def _message_is_incorrect(content: str, system: SimulatedSystem) -> bool:
    """Return True if the message contains factual claims contradicting system state."""
    content_lower = content.lower()
    for name, svc in system.service_registry.services.items():
        # Claims service is running when it's not
        if f"{name} is running" in content_lower and svc.status != "running":
            return True
        if f"{name} is healthy" in content_lower and svc.status != "running":
            return True
        # Claims service is down when it's actually running
        if f"{name} is down" in content_lower and svc.status == "running":
            return True
    return False


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
        self._comm_bonus_count: int = 0
        self._max_comm_bonuses: int = 5  # Cap at 5 useful message bonuses per episode
        self.enable_comm_bonus: bool = True
        self.enable_anti_hack_penalty: bool = True
        self._last_reward_components: dict[str, float] = {}

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
        achieved_before = set(self.achieved)
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

        # 5. Communication quality scoring
        comm_reward = self._evaluate_communication(
            actions, system, channel, achieved_before,
        )

        return self._build_result(done=False, communication_reward=comm_reward)

    def _evaluate_communication(
        self,
        actions: MultiAgentAction,
        system: SimulatedSystem,
        channel: CommunicationChannel,
        achieved_before: set[str],
    ) -> float:
        """Score communication quality: useful messages and incorrect claims."""
        new_milestones = self.achieved - achieved_before
        comm_reward = 0.0

        # Check if messages from previous round contributed to milestones this round
        if new_milestones:
            prev_round = self.round_count - 1
            if prev_round >= 0 and channel:
                prev_messages = [
                    m for m in channel.get_full_history()
                    if m.round_number == prev_round
                ]
                for msg in prev_messages:
                    content_lower = msg.content.lower()
                    for milestone_name in new_milestones:
                        if _message_relevant_to_milestone(content_lower, milestone_name):
                            if self._comm_bonus_count < self._max_comm_bonuses:
                                comm_reward += COMMUNICATION_USEFUL_BONUS
                                self._comm_bonus_count += 1
                                self.penalties_applied.append(
                                    f"comm_useful:{msg.from_agent}",
                                )
                            break

        # Check for incorrect messages this round
        for role in ["triage", "diagnosis", "remediation"]:
            agent_action = getattr(actions, role)
            if agent_action.message and agent_action.message.content:
                if _message_is_incorrect(agent_action.message.content, system):
                    comm_reward -= COMMUNICATION_INCORRECT_PENALTY
                    self.penalties_applied.append(f"comm_incorrect:{role}")

        return comm_reward

    def current_score(self) -> float:
        """Compute current team score."""
        if self.fatal_triggered:
            return FATAL_SCORE

        credit = sum(m.credit for m in self.milestones if m.name in self.achieved)

        penalty = 0.0
        bonus = 0.0
        for p in self.penalties_applied:
            if p == "time_pressure":
                penalty += TIME_PRESSURE_PENALTY
            elif p.startswith("noop:"):
                penalty += NOOP_PENALTY
            elif p == "FATAL":
                return FATAL_SCORE
            elif p == "role_violation":
                penalty += ROLE_VIOLATION_PENALTY
            elif p.startswith("comm_incorrect"):
                penalty += COMMUNICATION_INCORRECT_PENALTY
            elif p.startswith("comm_useful"):
                bonus += COMMUNICATION_USEFUL_BONUS

        raw = credit - penalty + bonus
        self._last_reward_components = {
            "milestone_credit": credit,
            "penalty_total": penalty,
            "communication_bonus": bonus,
            "raw_score": raw,
            "final_score": max(0.01, min(0.99, raw)),
        }
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

    def _build_result(self, done: bool, communication_reward: float = 0.0) -> RewardResult:
        score = self.current_score()
        penalty_reasons = sorted({p.split(":", 1)[0] for p in self.penalties_applied})
        return RewardResult(
            team_reward=score,
            individual_rewards={
                "triage": score,
                "diagnosis": score,
                "remediation": score,
            },
            communication_reward=communication_reward,
            reward_components=dict(self._last_reward_components),
            milestones_achieved=sorted(self.achieved),
            penalties_applied=list(self.penalties_applied),
            penalty_reasons=penalty_reasons,
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
