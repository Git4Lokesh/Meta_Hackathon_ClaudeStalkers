"""TaskGrader: deterministic scoring engine for SRE tasks.

Tracks milestones (partial credit) and penalties.  Evaluates each action
and produces a cumulative reward clamped to [0.0, 1.0].
"""

from __future__ import annotations

from typing import Callable, Optional

from sre_env.server.models import GraderResult
from sre_env.server.simulated_system import SimulatedSystem


class Milestone:
    """A grading milestone with partial credit."""

    def __init__(
        self,
        name: str,
        credit: float,
        description: str,
        check: Callable[[str, SimulatedSystem, str], bool],
    ) -> None:
        self.name = name
        self.credit = credit  # 0.0 to 1.0
        self.description = description
        # check(command, system_after_execution, command_output) -> bool
        self.check = check


class Penalty:
    """A grading penalty."""

    def __init__(
        self,
        name: str,
        amount: float,
        description: str,
        check: Callable[[str, SimulatedSystem, str], bool],
    ) -> None:
        self.name = name
        self.amount = amount  # positive value, will be subtracted
        self.description = description
        # check(command, system_after_execution, command_output) -> bool
        self.check = check


NOOP_PENALTY_NAME = "noop"
NOOP_PENALTY_AMOUNT = 0.02


class TaskGrader:
    """Deterministic scoring engine for a task.

    * Milestones are tracked in a ``set[str]`` — each can only be awarded
      once (idempotent).
    * Penalties are tracked in a ``list[str]`` — duplicates are allowed
      (e.g. multiple no-op penalties).
    * No-op detection: if the command is identical to the previous command,
      ``consecutive_noop_count`` is incremented and a 0.02 penalty is
      applied.  A different command resets the counter.
    * Score is always clamped to [0.0, 1.0].
    """

    def __init__(
        self,
        milestones: list[Milestone],
        penalties: list[Penalty],
    ) -> None:
        self.milestones = milestones
        self.penalties = penalties
        self.achieved: set[str] = set()
        self.penalties_applied: list[str] = []
        self.prev_command: Optional[str] = None
        self.consecutive_noop_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        command: str,
        system: SimulatedSystem,
        output: str,
    ) -> float:
        """Evaluate a single action and return the current cumulative reward.

        1. Check each milestone — award if its check passes and it hasn't
           been achieved yet.
        2. Check each penalty — apply if its check passes.
        3. Check for no-op (same command as previous) — 0.02 penalty per
           consecutive no-op.
        4. Update ``prev_command``.
        5. Return cumulative score clamped to [0.0, 1.0].
        """
        # 1. Milestones
        for milestone in self.milestones:
            if milestone.name not in self.achieved:
                if milestone.check(command, system, output):
                    self.achieved.add(milestone.name)

        # 2. Task-specific penalties
        for penalty in self.penalties:
            if penalty.check(command, system, output):
                self.penalties_applied.append(penalty.name)

        # 3. No-op detection
        if self.prev_command is not None and command == self.prev_command:
            self.consecutive_noop_count += 1
            self.penalties_applied.append(NOOP_PENALTY_NAME)
        else:
            self.consecutive_noop_count = 0

        # 4. Update previous command
        self.prev_command = command

        # 5. Return clamped score
        return self.current_score()

    def current_score(self) -> float:
        """Compute current score: milestone credits minus penalties, clamped to [0.0, 1.0]."""
        credit = sum(
            m.credit for m in self.milestones if m.name in self.achieved
        )

        # Build a lookup of penalty amounts by name (task-specific penalties)
        penalty_amounts: dict[str, float] = {
            p.name: p.amount for p in self.penalties
        }

        total_penalty = 0.0
        for name in self.penalties_applied:
            if name == NOOP_PENALTY_NAME:
                total_penalty += NOOP_PENALTY_AMOUNT
            elif name in penalty_amounts:
                total_penalty += penalty_amounts[name]

        raw = credit - total_penalty
        return max(0.0, min(1.0, raw))

    def result(self, done: bool) -> GraderResult:
        """Build a GraderResult with current state."""
        return GraderResult(
            score=self.current_score(),
            milestones_achieved=sorted(self.achieved),
            penalties_applied=list(self.penalties_applied),
            done=done,
        )
