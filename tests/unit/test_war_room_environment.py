"""Unit tests for WarRoomEnvironment."""

import pytest
from datetime import datetime

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import (
    AgentAction,
    MultiAgentAction,
    MultiAgentObservation,
    Message,
    WarRoomState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_action() -> MultiAgentAction:
    """All agents do nothing."""
    return MultiAgentAction(
        triage=AgentAction(command=""),
        diagnosis=AgentAction(command=""),
        remediation=AgentAction(command=""),
    )


def _action_with_commands(
    triage_cmd: str = "",
    diag_cmd: str = "",
    remed_cmd: str = "",
) -> MultiAgentAction:
    return MultiAgentAction(
        triage=AgentAction(command=triage_cmd),
        diagnosis=AgentAction(command=diag_cmd),
        remediation=AgentAction(command=remed_cmd),
    )


def _action_with_message(
    from_role: str,
    to_agent: str,
    content: str,
) -> MultiAgentAction:
    """Build an action where one agent sends a message."""
    msg = Message(
        from_agent=from_role,
        to_agent=to_agent,
        content=content,
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        round_number=0,
    )
    kwargs: dict = {
        "triage": AgentAction(command=""),
        "diagnosis": AgentAction(command=""),
        "remediation": AgentAction(command=""),
    }
    kwargs[from_role] = AgentAction(command="", message=msg)
    return MultiAgentAction(**kwargs)


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestWarRoomReset:
    """Tests for reset() behaviour."""

    @pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
    def test_reset_returns_valid_observation_all_tasks(self, task_id: str):
        env = WarRoomEnvironment()
        obs = env.reset(task_id=task_id, seed=42)

        assert isinstance(obs, MultiAgentObservation)
        assert obs.done is False
        assert obs.team_reward == 0.0
        # All three agents get non-empty text
        assert obs.triage.text
        assert obs.diagnosis.text
        assert obs.remediation.text
        # Metadata populated
        assert obs.metadata["task_id"] == task_id
        assert obs.metadata["round"] == 0
        assert obs.metadata["max_rounds"] > 0

    def test_reset_invalid_task_id_raises_value_error(self):
        env = WarRoomEnvironment()
        with pytest.raises(ValueError, match="Invalid task_id"):
            env.reset(task_id="nonexistent")

    def test_reset_invalid_task_id_lists_valid(self):
        env = WarRoomEnvironment()
        with pytest.raises(ValueError, match="task1"):
            env.reset(task_id="bad")

    def test_reset_clears_previous_state(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        # Step once to advance state
        env.step(_noop_action())
        assert env.state.round_number == 1

        # Reset should clear
        env.reset(task_id="task1", seed=42)
        assert env.state.round_number == 0
        assert env.state.done is False


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestWarRoomStep:
    """Tests for step() behaviour."""

    def test_step_before_reset_raises_runtime_error(self):
        env = WarRoomEnvironment()
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(_noop_action())

    def test_step_returns_observation(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        obs = env.step(_noop_action())

        assert isinstance(obs, MultiAgentObservation)
        assert obs.triage.text
        assert obs.diagnosis.text
        assert obs.remediation.text
        assert obs.metadata["round"] == 1

    def test_step_processes_all_three_agents(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)

        action = _action_with_commands(
            triage_cmd="get_dashboard",
            diag_cmd="ps aux",
            remed_cmd="curl http://localhost:80",
        )
        obs = env.step(action)
        assert obs.metadata["round"] == 1

    def test_round_counting(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)

        for i in range(1, 4):
            obs = env.step(_noop_action())
            assert obs.metadata["round"] == i

    def test_done_on_round_limit(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)  # max_rounds=10

        for _ in range(10):
            obs = env.step(_noop_action())

        assert obs.done is True
        assert "score" in obs.metadata

    def test_step_after_done_returns_terminal(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        # Force done
        env._done = True

        obs = env.step(_noop_action())
        assert obs.done is True
        assert "Episode complete" in obs.triage.text


# ---------------------------------------------------------------------------
# Role permission tests
# ---------------------------------------------------------------------------

class TestWarRoomPermissions:
    """Tests for role-based command validation."""

    def test_triage_allowed_command(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)

        action = _action_with_commands(triage_cmd="get_dashboard")
        obs = env.step(action)
        # Should not contain error about permissions
        assert obs.triage.text  # observation is populated

    def test_triage_disallowed_command(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)

        action = _action_with_commands(triage_cmd="kill -9 1234")
        obs = env.step(action)
        # Penalty should be recorded
        assert "role_violation" in env._grader.penalties_applied

    def test_diagnosis_disallowed_command(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)

        action = _action_with_commands(diag_cmd="systemctl restart nginx")
        obs = env.step(action)
        assert "role_violation" in env._grader.penalties_applied

    def test_remediation_disallowed_log_access(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)

        action = _action_with_commands(remed_cmd="cat /var/log/nginx/error.log")
        obs = env.step(action)
        assert "role_violation" in env._grader.penalties_applied


# ---------------------------------------------------------------------------
# Communication tests
# ---------------------------------------------------------------------------

class TestWarRoomCommunication:
    """Tests for communication channel integration."""

    def test_message_stored_in_channel(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)

        action = _action_with_message("triage", "diagnosis", "nginx is down")
        env.step(action)

        history = env._channel.get_full_history()
        assert len(history) >= 1
        assert any(m.content == "nginx is down" for m in history)

    def test_escalate_sends_message(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)

        action = _action_with_commands(
            triage_cmd="escalate diagnosis nginx is crashed",
        )
        env.step(action)

        history = env._channel.get_full_history()
        assert any("[ESCALATION]" in m.content for m in history)


# ---------------------------------------------------------------------------
# State property tests
# ---------------------------------------------------------------------------

class TestWarRoomState:
    """Tests for the state property."""

    def test_state_before_reset(self):
        env = WarRoomEnvironment()
        s = env.state
        assert isinstance(s, WarRoomState)
        assert s.round_number == 0
        assert s.done is False

    def test_state_after_reset(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        s = env.state

        assert s.task_id == "task1"
        assert s.round_number == 0
        assert s.episode_id != ""
        assert s.simulated_system != {}
        assert s.done is False

    def test_state_updates_after_step(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        env.step(_noop_action())

        s = env.state
        assert s.round_number == 1
