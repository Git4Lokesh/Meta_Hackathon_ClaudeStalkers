"""Pydantic data models for the Multi-Agent Incident War Room."""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


class Message(BaseModel):
    """A single message on the communication channel."""
    from_agent: str          # "triage" | "diagnosis" | "remediation"
    to_agent: str            # "triage" | "diagnosis" | "remediation" | "all"
    content: str
    timestamp: datetime
    round_number: int


class AgentAction(BaseModel):
    """A single agent's action for one round."""
    command: str = ""                    # The command to execute (or empty for no-op)
    message: Optional[Message] = None   # Optional message to send
    thought: str = ""                   # Agent's internal Chain-of-Thought


class MultiAgentAction(BaseModel):
    """All three agents' actions for one round."""
    triage: AgentAction = Field(default_factory=AgentAction)
    diagnosis: AgentAction = Field(default_factory=AgentAction)
    remediation: AgentAction = Field(default_factory=AgentAction)


class AgentObservation(BaseModel):
    """One agent's observation for a round."""
    text: str = ""                      # Serialized role-specific view
    reward: float = 0.0                 # Individual agent reward
    messages: list[Message] = Field(default_factory=list)  # New messages for this agent


class MultiAgentObservation(BaseModel):
    """Observation returned from reset/step."""
    triage: AgentObservation = Field(default_factory=AgentObservation)
    diagnosis: AgentObservation = Field(default_factory=AgentObservation)
    remediation: AgentObservation = Field(default_factory=AgentObservation)
    team_reward: float = 0.0
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Alert(BaseModel):
    """A structured alert for the Triage Agent."""
    service: str
    alert_type: str                     # "service_down", "high_cpu", "high_memory"
    severity: str                       # "critical", "warning", "info"
    description: str
    prominence: int = 0                 # Higher = shown first (for misdirection tasks)


class RewardResult(BaseModel):
    """Detailed reward breakdown from the grader."""
    team_reward: float = 0.0
    individual_rewards: dict[str, float] = Field(default_factory=dict)  # {role: reward}
    communication_reward: float = 0.0
    milestones_achieved: list[str] = Field(default_factory=list)
    penalties_applied: list[str] = Field(default_factory=list)
    credit_assignment: dict[str, str] = Field(default_factory=dict)  # {milestone: agent_role}
    done: bool = False


class WarRoomState(BaseModel):
    """Full environment state."""
    episode_id: str = ""
    round_number: int = 0
    max_rounds: int = 0
    task_id: str = ""
    simulated_system: dict[str, Any] = Field(default_factory=dict)
    communication_history: list[Message] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    per_agent_tracking: dict[str, dict[str, Any]] = Field(default_factory=dict)
    done: bool = False
