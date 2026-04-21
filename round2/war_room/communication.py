"""CommunicationChannel for the Multi-Agent Incident War Room."""

from datetime import datetime
from round2.war_room.models import Message


class CommunicationChannel:
    def __init__(self):
        self._messages: list[Message] = []
        self._current_round: int = 0
        self._rounds_since_any_message: int = 0

    def send(self, from_agent: str, to_agent: str, content: str, timestamp: datetime) -> None:
        """Append a message to the channel."""
        self._messages.append(Message(
            from_agent=from_agent, to_agent=to_agent,
            content=content, timestamp=timestamp,
            round_number=self._current_round,
        ))
        self._rounds_since_any_message = 0

    def advance_round(self) -> None:
        """Call at the end of each round to track communication gaps."""
        msgs_this_round = [m for m in self._messages if m.round_number == self._current_round]
        if not msgs_this_round:
            self._rounds_since_any_message += 1
        self._current_round += 1

    def get_messages_for(self, agent: str, since_round: int) -> list[Message]:
        """Get messages addressed to this agent (or 'all') since a given round."""
        return [
            m for m in self._messages
            if m.round_number >= since_round
            and (m.to_agent == agent or m.to_agent == "all")
        ]

    def get_full_history(self) -> list[Message]:
        """Return all messages in order."""
        return list(self._messages)

    def rounds_without_any_messages(self) -> int:
        """Number of consecutive rounds with zero messages from all agents."""
        return self._rounds_since_any_message

    @property
    def current_round(self) -> int:
        return self._current_round
