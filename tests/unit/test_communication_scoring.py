"""Tests for communication quality scoring in the MultiAgentGrader."""

import pytest
from datetime import datetime

from round2.war_room.grader import (
    MultiAgentGrader,
    MultiAgentMilestone,
    _message_relevant_to_milestone,
    _message_is_incorrect,
    COMMUNICATION_USEFUL_BONUS,
    COMMUNICATION_INCORRECT_PENALTY,
)
from round2.war_room.models import MultiAgentAction, AgentAction, Message
from round2.war_room.communication import CommunicationChannel
from round2.war_room.environment import WarRoomEnvironment


class TestMessageRelevance:
    """Test _message_relevant_to_milestone helper."""
    
    def test_relevant_message_matches(self):
        assert _message_relevant_to_milestone("nginx is down and crashed", "triage_escalates_nginx")
    
    def test_irrelevant_message_no_match(self):
        assert not _message_relevant_to_milestone("everything looks fine", "triage_escalates_nginx")
    
    def test_needs_at_least_two_keywords(self):
        # Only one keyword match — not enough
        assert not _message_relevant_to_milestone("nginx is fine", "triage_escalates_nginx")

    def test_memory_keywords(self):
        assert _message_relevant_to_milestone("memory leak causing oom kills", "triage_prioritizes_memory")
    
    def test_auth_keywords(self):
        assert _message_relevant_to_milestone("authentication failed, wrong password in database", "diagnosis_identifies_db_auth")
    
    def test_unknown_milestone_returns_false(self):
        assert not _message_relevant_to_milestone("anything", "unknown_milestone")


class TestMessageIncorrectness:
    """Test _message_is_incorrect helper."""
    
    def test_correct_claim_not_flagged(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        # nginx is crashed in task1
        assert not _message_is_incorrect("nginx is down", env._system)
    
    def test_incorrect_running_claim_flagged(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        # nginx is crashed, claiming it's running is incorrect
        assert _message_is_incorrect("nginx is running", env._system)
    
    def test_incorrect_down_claim_flagged(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        # postgres is running, claiming it's down is incorrect
        assert _message_is_incorrect("postgres is down", env._system)
    
    def test_vague_message_not_flagged(self):
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        assert not _message_is_incorrect("something seems wrong", env._system)


class TestCommunicationScoringIntegration:
    """Integration tests for communication scoring in the environment."""
    
    def test_useful_message_earns_bonus(self):
        """A message about nginx before the restart milestone should earn comm bonus."""
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        
        # Round 1: triage sends useful message about nginx
        action1 = MultiAgentAction(
            triage=AgentAction(
                command="get_dashboard",
                message=Message(
                    from_agent="triage", to_agent="diagnosis",
                    content="nginx is down and crashed, please investigate",
                    timestamp=datetime.now(), round_number=1,
                ),
            ),
            diagnosis=AgentAction(command=""),
            remediation=AgentAction(command=""),
        )
        env.step(action1)
        
        # Round 2: diagnosis reads logs (triggers milestone)
        action2 = MultiAgentAction(
            triage=AgentAction(command=""),
            diagnosis=AgentAction(command="cat /var/log/nginx/error.log"),
            remediation=AgentAction(command=""),
        )
        obs = env.step(action2)
        
        # Check that comm_useful was recorded
        useful_penalties = [p for p in env._grader.penalties_applied if p.startswith("comm_useful")]
        assert len(useful_penalties) > 0, "Expected communication useful bonus"
    
    def test_incorrect_message_incurs_penalty(self):
        """A message with incorrect facts should incur a penalty."""
        env = WarRoomEnvironment()
        env.reset(task_id="task1", seed=42)
        
        # Send a message claiming nginx is running (it's actually crashed)
        action = MultiAgentAction(
            triage=AgentAction(
                command="get_dashboard",
                message=Message(
                    from_agent="triage", to_agent="diagnosis",
                    content="nginx is running fine, check redis instead",
                    timestamp=datetime.now(), round_number=1,
                ),
            ),
            diagnosis=AgentAction(command=""),
            remediation=AgentAction(command=""),
        )
        env.step(action)
        
        incorrect_penalties = [p for p in env._grader.penalties_applied if p.startswith("comm_incorrect")]
        assert len(incorrect_penalties) > 0, "Expected communication incorrect penalty"
