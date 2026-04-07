"""Tests for advanced reward design features.

Tests all 5 reward concepts:
1. Step Efficiency Penalty (MTTR)
2. State-Based Verification
3. Asymmetric Costly-Action Penalty
4. Progress Reversal Penalty
5. Critical Failure Conditions (Fatal Actions)
"""

import pytest

from sre_env.server.grader import (
    FatalAction,
    Milestone,
    Penalty,
    StateCheck,
    TaskGrader,
    STEP_PENALTY,
    MUTATING_ACTION_COST,
    PROGRESS_REVERSAL_PENALTY,
    FATAL_SCORE,
)
from sre_env.server.models import SREAction
from sre_env.server.sre_environment import SREEnvironment


# ---- Helpers ----

def make_simple_env():
    """Create a simple SREEnvironment for testing."""
    env = SREEnvironment()
    return env


# ===========================================================================
# 1. Step Efficiency Penalty Tests
# ===========================================================================


class TestStepEfficiencyPenalty:
    """Step penalty: -0.01 per step, forces MTTR optimisation."""

    def test_step_penalty_accumulates(self):
        """Score decreases with each step even for read-only commands."""
        env = make_simple_env()
        obs = env.reset(task_id="task1", seed=42)

        # Take 5 read-only steps
        scores = []
        for _ in range(5):
            obs = env.step(SREAction(command="ps aux"))
            scores.append(obs.reward)

        # Each step should incur STEP_PENALTY
        # Score should decrease over time (from step penalties)
        assert scores[-1] < scores[0] or scores[0] == 0.0  # may start at 0

    def test_fewer_steps_higher_score(self):
        """An efficient agent gets a higher score than a wasteful one."""
        # Efficient: 3 steps
        env1 = make_simple_env()
        env1.reset(task_id="task1", seed=42)
        env1.step(SREAction(command="cat /var/log/nginx/error.log"))
        env1.step(SREAction(command="systemctl status nginx"))
        obs1 = env1.step(SREAction(command="systemctl restart nginx"))
        score_efficient = obs1.reward

        # Wasteful: 10 steps of fluff then same fix
        env2 = make_simple_env()
        env2.reset(task_id="task1", seed=42)
        for _ in range(7):
            env2.step(SREAction(command="ls /"))
        env2.step(SREAction(command="cat /var/log/nginx/error.log"))
        env2.step(SREAction(command="systemctl status nginx"))
        obs2 = env2.step(SREAction(command="systemctl restart nginx"))
        score_wasteful = obs2.reward

        assert score_efficient > score_wasteful, (
            f"Efficient agent ({score_efficient:.3f}) should score higher "
            f"than wasteful agent ({score_wasteful:.3f})"
        )

    def test_step_penalty_in_penalties_list(self):
        """Step penalty should appear in the penalties_applied list."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)
        env.step(SREAction(command="ps aux"))

        result = env._grader.result(done=False)
        assert "step_penalty" in result.penalties_applied


# ===========================================================================
# 2. State-Based Verification Tests
# ===========================================================================


class TestStateBasedVerification:
    """State checks verify system state, not just commands."""

    def test_state_check_awarded_when_state_matches(self):
        """State check is awarded when system state matches."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        # Before restart: nginx_healthy state check should NOT be achieved
        assert "nginx_healthy" not in env._grader.state_achieved

        # After restart: nginx_healthy should be achieved
        env.step(SREAction(command="systemctl restart nginx"))
        assert "nginx_healthy" in env._grader.state_achieved

    def test_state_check_contributes_to_score(self):
        """State check credit is included in the score."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        obs_before = env.step(SREAction(command="ps aux"))
        score_before = obs_before.reward

        obs_after = env.step(SREAction(command="systemctl restart nginx"))
        score_after = obs_after.reward

        # The restart should earn milestone credit + state check credit
        assert score_after > score_before

    def test_state_check_in_milestones_achieved(self):
        """State checks appear in milestones_achieved in GraderResult."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)
        env.step(SREAction(command="systemctl restart nginx"))

        result = env._grader.result(done=False)
        assert "nginx_healthy" in result.milestones_achieved

    def test_task2_memory_state_check(self):
        """Task 2 memory state check rewards when memory is below threshold."""
        env = make_simple_env()
        env.reset(task_id="task2", seed=42)

        # Initially memory is high — state check should NOT be achieved
        env.step(SREAction(command="ps aux"))
        # The leaking process uses ~2800MB, so total > 4096MB
        # Check if memory_under_control is NOT in state_achieved
        assert "memory_under_control" not in env._grader.state_achieved

    def test_task3_cascade_state_check(self):
        """Task 3 config_correct state check works."""
        env = make_simple_env()
        env.reset(task_id="task3", seed=42)

        # Initially config is wrong
        env.step(SREAction(command="ps aux"))
        assert "config_correct" not in env._grader.state_achieved


# ===========================================================================
# 3. Asymmetric Costly-Action Penalty Tests
# ===========================================================================


class TestCostlyActionPenalty:
    """Mutating commands that don't earn milestones get penalized."""

    def test_readonly_no_penalty(self):
        """Read-only commands should NOT incur costly_action penalty."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        env.step(SREAction(command="ps aux"))
        env.step(SREAction(command="cat /var/log/nginx/error.log"))
        env.step(SREAction(command="top"))

        result = env._grader.result(done=False)
        assert "costly_action" not in result.penalties_applied

    def test_mutating_without_progress_penalized(self):
        """A mutating command that doesn't earn a milestone gets penalized."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        # Kill a non-existent PID — mutating, earns nothing
        env.step(SREAction(command="kill 99999"))

        result = env._grader.result(done=False)
        assert "costly_action" in result.penalties_applied

    def test_mutating_with_progress_no_penalty(self):
        """A mutating command that earns a milestone should NOT be penalized."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        # Restart nginx — mutating, but earns restart_nginx milestone
        env.step(SREAction(command="systemctl restart nginx"))

        result = env._grader.result(done=False)
        assert "costly_action" not in result.penalties_applied


# ===========================================================================
# 4. Progress Reversal Penalty Tests
# ===========================================================================


class TestProgressReversalPenalty:
    """Agent loses points when previously achieved state checks regress."""

    def test_state_check_revoked_on_regression(self):
        """If nginx is restarted then killed, the state check is revoked."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        # Restart nginx — achieves nginx_healthy
        env.step(SREAction(command="systemctl restart nginx"))
        assert "nginx_healthy" in env._grader.state_achieved

        # Now stop nginx — should revoke nginx_healthy
        env.step(SREAction(command="systemctl stop nginx"))
        assert "nginx_healthy" not in env._grader.state_achieved

    def test_reversal_penalty_applied(self):
        """Progress reversal should add a reversal penalty."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        env.step(SREAction(command="systemctl restart nginx"))
        env.step(SREAction(command="systemctl stop nginx"))

        result = env._grader.result(done=False)
        reversal_penalties = [p for p in result.penalties_applied if p.startswith("reversal:")]
        assert len(reversal_penalties) > 0, "Expected reversal penalty"


# ===========================================================================
# 5. Critical Failure Conditions Tests
# ===========================================================================


class TestCriticalFailureConditions:
    """Fatal actions immediately terminate episode with score 0."""

    def test_fatal_action_terminates_episode(self):
        """Killing a healthy critical service should end the episode."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        # Get postgres PID
        postgres_proc = env._system.process_table.get_by_service("postgres")
        assert postgres_proc is not None
        postgres_pid = postgres_proc.pid

        # Kill postgres — fatal action
        obs = env.step(SREAction(command=f"kill -9 {postgres_pid}"))

        assert obs.done is True, "Episode should be terminated"
        assert obs.reward == FATAL_SCORE, f"Score should be {FATAL_SCORE}, got {obs.reward}"

    def test_fatal_action_score_zero(self):
        """Fatal action should set score to exactly 0.0."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        # First earn some credit
        env.step(SREAction(command="cat /var/log/nginx/error.log"))
        env.step(SREAction(command="systemctl status nginx"))

        # Now kill postgres — should zero out everything
        postgres_proc = env._system.process_table.get_by_service("postgres")
        obs = env.step(SREAction(command=f"kill -9 {postgres_proc.pid}"))

        assert obs.reward == 0.0

    def test_fatal_in_grader_result(self):
        """GraderResult should reflect fatal action."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        postgres_proc = env._system.process_table.get_by_service("postgres")
        env.step(SREAction(command=f"kill -9 {postgres_proc.pid}"))

        result = env._grader.result(done=True)
        assert result.fatal_action_triggered is True
        assert result.score == 0.0

    def test_fatal_message_in_output(self):
        """Output should contain FATAL ACTION warning."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        postgres_proc = env._system.process_table.get_by_service("postgres")
        obs = env.step(SREAction(command=f"kill -9 {postgres_proc.pid}"))

        assert "FATAL ACTION" in obs.output

    def test_stopped_critical_service_is_fatal(self):
        """Stopping a healthy critical service should be fatal."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        obs = env.step(SREAction(command="systemctl stop postgres"))
        assert obs.done is True
        assert obs.reward == 0.0

    def test_task3_killing_postgres_is_fatal(self):
        """In task3, killing postgres (root dependency) is fatal."""
        env = make_simple_env()
        env.reset(task_id="task3", seed=42)

        postgres_proc = env._system.process_table.get_by_service("postgres")
        obs = env.step(SREAction(command=f"kill -9 {postgres_proc.pid}"))

        assert obs.done is True
        assert obs.reward == 0.0

    def test_no_further_steps_after_fatal(self):
        """After fatal, additional steps should return done=True."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        postgres_proc = env._system.process_table.get_by_service("postgres")
        env.step(SREAction(command=f"kill -9 {postgres_proc.pid}"))

        # Try another step
        obs = env.step(SREAction(command="ps aux"))
        assert obs.done is True


# ===========================================================================
# Integration: Score Verification
# ===========================================================================


class TestScoreIntegration:
    """End-to-end score verification for known sequences."""

    def test_perfect_task1_score(self):
        """Optimal task1 sequence should achieve a high score."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        # Optimal sequence: 4 steps
        env.step(SREAction(command="cat /var/log/nginx/error.log"))
        env.step(SREAction(command="systemctl status nginx"))
        env.step(SREAction(command="systemctl restart nginx"))
        obs = env.step(SREAction(command="curl http://localhost:80/health"))

        # Should have all milestones + state checks - small step penalties
        # Total credit: 0.15 + 0.20 + 0.40 + 0.10 + 0.15(state) = 1.0
        # Step penalties: 4 * 0.01 = 0.04
        # Expected: ~0.96
        assert obs.reward >= 0.90, f"Expected >= 0.90, got {obs.reward:.3f}"

    def test_health_score_in_metadata(self):
        """Health score should be in step metadata."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        obs = env.step(SREAction(command="ps aux"))
        assert "health_score" in obs.metadata

    def test_step_penalties_in_result(self):
        """step_penalties count should be in GraderResult."""
        env = make_simple_env()
        env.reset(task_id="task1", seed=42)

        env.step(SREAction(command="ps aux"))
        env.step(SREAction(command="ls /"))

        result = env._grader.result(done=False)
        assert result.step_penalties == 2
