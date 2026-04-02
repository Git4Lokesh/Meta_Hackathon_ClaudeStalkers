"""Unit tests for SREEnvironment."""

import pytest

from sre_env.server.models import SREAction
from sre_env.server.sre_environment import SREEnvironment


class TestSREEnvironmentReset:
    """Tests for reset() behaviour."""

    def test_reset_returns_observation(self):
        env = SREEnvironment()
        obs = env.reset(task_id="task1", seed=42)
        assert obs.done is False
        assert obs.reward == 0.0
        assert "SRE Incident Response" in obs.output
        assert obs.metadata["task_id"] == "task1"

    def test_reset_invalid_task_raises(self):
        env = SREEnvironment()
        with pytest.raises(ValueError, match="Invalid task_id"):
            env.reset(task_id="nonexistent")

    def test_reset_determinism(self):
        env = SREEnvironment()
        obs1 = env.reset(task_id="task1", seed=99)
        snap1 = env.state.simulated_system

        obs2 = env.reset(task_id="task1", seed=99)
        snap2 = env.state.simulated_system

        assert snap1 == snap2
        assert obs1.output == obs2.output

    def test_reset_all_tasks(self):
        env = SREEnvironment()
        for tid in ("task1", "task2", "task3"):
            obs = env.reset(task_id=tid, seed=42)
            assert obs.done is False
            assert obs.metadata["task_id"] == tid

    def test_reset_default_seed(self):
        env = SREEnvironment()
        obs = env.reset(task_id="task1")
        assert obs.done is False


class TestSREEnvironmentStep:
    """Tests for step() behaviour."""

    def test_step_before_reset_raises(self):
        env = SREEnvironment()
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(SREAction(command="help"))

    def test_step_returns_observation(self):
        env = SREEnvironment()
        env.reset(task_id="task1", seed=42)
        obs = env.step(SREAction(command="help"))
        assert isinstance(obs.output, str)
        assert obs.done is False
        assert obs.metadata["step"] == 1

    def test_step_increments_step_count(self):
        env = SREEnvironment()
        env.reset(task_id="task1", seed=42)
        env.step(SREAction(command="help"))
        env.step(SREAction(command="ps aux"))
        assert env.state.step_count == 2

    def test_step_after_done_returns_complete_message(self):
        env = SREEnvironment()
        env.reset(task_id="task1", seed=42)
        # Force done
        env._done = True
        obs = env.step(SREAction(command="help"))
        assert obs.done is True
        assert "Episode is complete" in obs.output

    def test_step_limit_enforcement(self):
        env = SREEnvironment()
        env.reset(task_id="task1", seed=42)
        # Task 1 has max_steps=20
        for _ in range(20):
            obs = env.step(SREAction(command="echo test"))
        assert obs.done is True
        assert "score" in obs.metadata

    def test_terminal_metadata_keys(self):
        env = SREEnvironment()
        env.reset(task_id="task1", seed=42)
        # Exhaust steps
        for _ in range(20):
            obs = env.step(SREAction(command="echo test"))
        assert "score" in obs.metadata
        assert "milestones_achieved" in obs.metadata
        assert "penalties_applied" in obs.metadata


class TestSREEnvironmentState:
    """Tests for state property."""

    def test_state_initial(self):
        env = SREEnvironment()
        assert env.state.step_count == 0

    def test_state_after_reset(self):
        env = SREEnvironment()
        env.reset(task_id="task1", seed=42)
        assert env.state.episode_id is not None
        assert env.state.step_count == 0
        assert env.state.simulated_system != {}

    def test_state_updates_after_step(self):
        env = SREEnvironment()
        env.reset(task_id="task1", seed=42)
        snap_before = dict(env.state.simulated_system)
        env.step(SREAction(command="systemctl restart nginx"))
        # Snapshot should have changed (time advanced, service restarted)
        assert env.state.simulated_system != snap_before


class TestSREEnvironmentMCPTools:
    """Tests for MCP tool methods."""

    def test_execute_command(self):
        env = SREEnvironment()
        env.reset(task_id="task1", seed=42)
        result = env.execute_command("help")
        assert "Available commands" in result

    def test_get_system_overview_before_reset(self):
        env = SREEnvironment()
        overview = env.get_system_overview()
        assert "error" in overview

    def test_get_system_overview_after_reset(self):
        env = SREEnvironment()
        env.reset(task_id="task1", seed=42)
        overview = env.get_system_overview()
        assert "services" in overview
        assert "process_count" in overview
        assert overview["process_count"] > 0

    def test_get_available_commands(self):
        env = SREEnvironment()
        cmds = env.get_available_commands()
        assert "cat" in cmds
        assert "ps" in cmds
        assert "help" in cmds
