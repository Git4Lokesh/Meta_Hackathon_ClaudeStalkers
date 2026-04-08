"""TaskGrader: advanced scoring engine for SRE tasks.

Implements five reward design concepts for frontier-level RL training:

1. **Step Efficiency Penalty** (MTTR optimisation) — small per-step cost
   forces agents to form hypotheses and act efficiently.
2. **State-Based Verification** — rewards system *state* (e.g. health
   endpoint returning 200) rather than just the command that was executed.
3. **Asymmetric Costly-Action Penalty** — mutating commands that don't
   contribute to progress incur a risk penalty.
4. **Progress Reversal Penalty** — if an agent achieves a milestone but
   then ruins it, points are deducted and the milestone is revoked.
5. **Critical Failure Conditions** — catastrophic actions (e.g. killing
   the only database) immediately terminate the episode with score 0.

Tracks milestones (partial credit) and penalties.  Evaluates each action
and produces a cumulative reward clamped to [0.0, 1.0].
"""

from __future__ import annotations

from typing import Callable, Optional

from sre_env.server.models import GraderResult
from sre_env.server.simulated_system import SimulatedSystem


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

STEP_PENALTY = 0.01
MUTATING_ACTION_COST = 0.05
PROGRESS_REVERSAL_PENALTY = 0.15
HEALTH_DROP_MULTIPLIER = 0.5
FATAL_SCORE = 0.01

NOOP_PENALTY_NAME = "noop"
NOOP_PENALTY_AMOUNT = 0.02

INVALID_CMD_PENALTY_NAME = "invalid_command"
INVALID_CMD_PENALTY_AMOUNT = 0.01

# Commands that read state — zero risk cost
READ_ONLY_COMMANDS = frozenset({
    "cat", "grep", "tail", "head", "ls", "ps", "top",
    "df", "free", "netstat", "echo", "help", "curl",
})

# Commands that mutate state — carry a risk cost when they don't help
MUTATING_COMMANDS = frozenset({"kill", "systemctl", "edit"})


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

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


class StateCheck:
    """State-based verification — rewards system *state* not the action.

    Unlike milestones, these only inspect the SimulatedSystem (no command
    or output).  They are evaluated every step and act as milestones that
    can also be *revoked* if the state regresses.
    """

    def __init__(
        self,
        name: str,
        credit: float,
        description: str,
        check: Callable[[SimulatedSystem], bool],
    ) -> None:
        self.name = name
        self.credit = credit
        self.description = description
        # check(system) -> bool
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


class FatalAction:
    """A catastrophic action that immediately terminates the episode."""

    def __init__(
        self,
        name: str,
        description: str,
        check: Callable[[str, SimulatedSystem, str], bool],
    ) -> None:
        self.name = name
        self.description = description
        # check(command, system_after_execution, command_output) -> bool
        self.check = check


# ---------------------------------------------------------------------------
# Health score helper
# ---------------------------------------------------------------------------

# Default implementation — counts fraction of services that are "running".
# Tasks can override this by providing a custom health_fn to the grader.

def default_health_score(system: SimulatedSystem) -> float:
    """Compute a [0, 1] health score for the simulated system."""
    services = system.service_registry.services
    if not services:
        return 1.0
    running = sum(1 for s in services.values() if s.status == "running")
    return running / len(services)


# ---------------------------------------------------------------------------
# TaskGrader
# ---------------------------------------------------------------------------

class TaskGrader:
    """Advanced scoring engine for a task.

    Features:
    * **Milestones** — tracked in a ``set[str]``; each can only be awarded
      once (idempotent).
    * **State checks** — evaluated every step; can be *revoked* if state
      regresses (progress reversal).
    * **Penalties** — tracked in a ``list[str]``; duplicates allowed.
    * **Step penalty** — ``-0.01`` per step (MTTR optimisation).
    * **Costly-action penalty** — ``-0.05`` for mutating commands that
      don't earn a new milestone.
    * **Progress reversal** — ``-0.15`` per revoked state check.
    * **Fatal actions** — immediately terminate the episode with score 0.
    * **No-op detection** — ``-0.02`` per consecutive duplicate command.
    * Score is always clamped to [0.0, 1.0].
    """

    def __init__(
        self,
        milestones: list[Milestone],
        penalties: list[Penalty],
        state_checks: list[StateCheck] | None = None,
        fatal_actions: list[FatalAction] | None = None,
        health_fn: Callable[[SimulatedSystem], float] | None = None,
    ) -> None:
        self.milestones = milestones
        self.penalties = penalties
        self.state_checks = state_checks or []
        self.fatal_actions = fatal_actions or []
        self.health_fn = health_fn or default_health_score

        # Tracking
        self.achieved: set[str] = set()          # milestone names
        self.state_achieved: set[str] = set()    # state check names
        self.penalties_applied: list[str] = []
        self.prev_command: Optional[str] = None
        self.consecutive_noop_count: int = 0
        self.step_count: int = 0
        self.prev_health: Optional[float] = None
        self.fatal_triggered: bool = False
        self.fatal_name: Optional[str] = None

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

        Order of operations:
        1. Check fatal actions — terminate immediately if triggered.
        2. Check milestones — award new ones.
        3. Check state-based verifications — award or REVOKE.
        4. Check task-specific penalties.
        5. Costly-action penalty (mutating + no new progress).
        6. Step penalty (MTTR).
        7. Progress reversal penalty (health-score drop).
        8. No-op detection.
        9. Invalid command detection.
        10. Update internal state.
        """
        if self.fatal_triggered:
            return FATAL_SCORE

        self.step_count += 1
        cmd_base = command.strip().split()[0] if command.strip() else ""

        # ── 1. Fatal actions ──────────────────────────────────────────
        for fatal in self.fatal_actions:
            if fatal.check(command, system, output):
                self.fatal_triggered = True
                self.fatal_name = fatal.name
                self.penalties_applied.append(f"FATAL:{fatal.name}")
                return FATAL_SCORE

        # ── 2. Milestones (action-based) ──────────────────────────────
        achieved_before = set(self.achieved)
        for milestone in self.milestones:
            if milestone.name not in self.achieved:
                if milestone.check(command, system, output):
                    self.achieved.add(milestone.name)

        # ── 3. State checks (state-based verification) ────────────────
        state_before = set(self.state_achieved)
        for sc in self.state_checks:
            if sc.check(system):
                self.state_achieved.add(sc.name)
            elif sc.name in self.state_achieved:
                # Progress reversal — was achieved, now regressed
                self.state_achieved.discard(sc.name)
                self.penalties_applied.append(f"reversal:{sc.name}")

        # ── 4. Task-specific penalties ────────────────────────────────
        for penalty in self.penalties:
            if penalty.check(command, system, output):
                self.penalties_applied.append(penalty.name)

        # ── 5. Costly-action penalty ─────────────────────────────────
        new_milestones = self.achieved - achieved_before
        new_state = self.state_achieved - state_before
        earned_progress = bool(new_milestones) or bool(new_state)

        if cmd_base in MUTATING_COMMANDS and not earned_progress:
            self.penalties_applied.append("costly_action")

        # ── 6. Step penalty (MTTR) ───────────────────────────────────
        self.penalties_applied.append("step_penalty")

        # ── 7. Progress reversal via health score drop ───────────────
        current_health = self.health_fn(system)
        if self.prev_health is not None:
            health_drop = self.prev_health - current_health
            if health_drop > 0.05:  # >5% health drop triggers penalty
                self.penalties_applied.append("health_drop")
        self.prev_health = current_health

        # ── 8. No-op detection ───────────────────────────────────────
        if self.prev_command is not None and command == self.prev_command:
            self.consecutive_noop_count += 1
            self.penalties_applied.append(NOOP_PENALTY_NAME)
        else:
            self.consecutive_noop_count = 0

        # ── 9. Invalid command detection ─────────────────────────────
        if "invalid usage" in output.lower() or "command not found" in output.lower():
            self.penalties_applied.append(INVALID_CMD_PENALTY_NAME)

        # ── 10. Update previous command ──────────────────────────────
        self.prev_command = command

        return self.current_score()

    def current_score(self) -> float:
        """Compute current score: credits minus penalties, clamped [0.0, 1.0]."""
        if self.fatal_triggered:
            return FATAL_SCORE

        # Credits from milestones
        credit = sum(
            m.credit for m in self.milestones if m.name in self.achieved
        )
        # Credits from state checks
        credit += sum(
            sc.credit for sc in self.state_checks if sc.name in self.state_achieved
        )

        # Build penalty lookup
        penalty_amounts: dict[str, float] = {
            p.name: p.amount for p in self.penalties
        }

        total_penalty = 0.0
        for name in self.penalties_applied:
            if name == NOOP_PENALTY_NAME:
                total_penalty += NOOP_PENALTY_AMOUNT
            elif name == INVALID_CMD_PENALTY_NAME:
                total_penalty += INVALID_CMD_PENALTY_AMOUNT
            elif name == "step_penalty":
                total_penalty += STEP_PENALTY
            elif name == "costly_action":
                total_penalty += MUTATING_ACTION_COST
            elif name.startswith("reversal:"):
                total_penalty += PROGRESS_REVERSAL_PENALTY
            elif name == "health_drop":
                total_penalty += HEALTH_DROP_MULTIPLIER * 0.1  # scaled
            elif name.startswith("FATAL:"):
                return FATAL_SCORE
            elif name in penalty_amounts:
                total_penalty += penalty_amounts[name]

        raw = credit - total_penalty
        return max(0.01, min(0.99, raw))

    def result(self, done: bool) -> GraderResult:
        """Build a GraderResult with current state."""
        return GraderResult(
            score=self.current_score(),
            milestones_achieved=sorted(self.achieved | self.state_achieved),
            penalties_applied=list(self.penalties_applied),
            done=done,
            fatal_action_triggered=self.fatal_triggered,
            health_score=self.prev_health if self.prev_health is not None else 1.0,
            step_penalties=self.step_count,
        )
