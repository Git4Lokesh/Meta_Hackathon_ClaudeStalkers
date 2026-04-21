"""Adaptive difficulty for the Multi-Agent War Room.

Tracks agent performance across episodes and adjusts difficulty
parameters to keep the environment challenging as agents improve.
"""

from dataclasses import dataclass, field


@dataclass
class DifficultyConfig:
    """Adjustable difficulty parameters."""
    extra_red_herrings: int = 0          # Additional misleading alerts
    round_limit_reduction: int = 0       # Reduce max rounds (harder time pressure)
    noise_log_entries: int = 0           # Extra irrelevant log entries
    communication_penalty_multiplier: float = 1.0  # Scale comm penalties


@dataclass
class PerformanceTracker:
    """Tracks agent performance across episodes for adaptive difficulty."""
    episode_scores: list[float] = field(default_factory=list)
    episode_rounds: list[int] = field(default_factory=list)
    episode_tasks: list[str] = field(default_factory=list)

    def record_episode(self, task_id: str, score: float, rounds: int) -> None:
        self.episode_scores.append(score)
        self.episode_rounds.append(rounds)
        self.episode_tasks.append(task_id)

    def recent_avg_score(self, n: int = 5) -> float:
        if not self.episode_scores:
            return 0.0
        recent = self.episode_scores[-n:]
        return sum(recent) / len(recent)

    def get_difficulty(self) -> DifficultyConfig:
        """Compute difficulty based on recent performance."""
        avg = self.recent_avg_score()

        if avg >= 0.8:
            # Agent is crushing it — make it harder
            return DifficultyConfig(
                extra_red_herrings=2,
                round_limit_reduction=3,
                noise_log_entries=10,
                communication_penalty_multiplier=1.5,
            )
        elif avg >= 0.6:
            # Agent is doing well — moderate increase
            return DifficultyConfig(
                extra_red_herrings=1,
                round_limit_reduction=1,
                noise_log_entries=5,
                communication_penalty_multiplier=1.2,
            )
        else:
            # Agent is struggling — keep it standard
            return DifficultyConfig()

    def summary(self) -> dict:
        return {
            "total_episodes": len(self.episode_scores),
            "recent_avg_score": round(self.recent_avg_score(), 3),
            "difficulty_level": (
                "hard" if self.recent_avg_score() >= 0.8
                else "medium" if self.recent_avg_score() >= 0.6
                else "standard"
            ),
        }
