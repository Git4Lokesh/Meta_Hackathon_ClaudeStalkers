"""Round 2 reward determinism and anti-hack tests."""

from __future__ import annotations

from datetime import datetime

from round2.war_room.anti_hack import check_episode
from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import AgentAction, Message, MultiAgentAction


def _scripted_task1_rounds() -> list[MultiAgentAction]:
    return [
        MultiAgentAction(
            triage=AgentAction(
                command="get_dashboard",
                message=Message(
                    from_agent="triage",
                    to_agent="diagnosis",
                    content="nginx is down",
                    timestamp=datetime.now(),
                    round_number=1,
                ),
            ),
        ),
        MultiAgentAction(
            diagnosis=AgentAction(
                command="cat /var/log/nginx/error.log",
                message=Message(
                    from_agent="diagnosis",
                    to_agent="remediation",
                    content="nginx crashed, restart nginx",
                    timestamp=datetime.now(),
                    round_number=2,
                ),
            ),
        ),
        MultiAgentAction(remediation=AgentAction(command="systemctl restart nginx")),
        MultiAgentAction(remediation=AgentAction(command="curl http://localhost:80/health")),
    ]


def test_reward_determinism_same_seed_same_actions():
    env1 = WarRoomEnvironment()
    env2 = WarRoomEnvironment()
    obs1 = env1.reset(task_id="task1", seed=42)
    obs2 = env2.reset(task_id="task1", seed=42)

    for action in _scripted_task1_rounds():
        obs1 = env1.step(action)
        obs2 = env2.step(action)

    assert obs1.team_reward == obs2.team_reward
    assert obs1.metadata.get("reward_components") == obs2.metadata.get("reward_components")


def test_reward_components_exposed_in_metadata():
    env = WarRoomEnvironment()
    env.reset(task_id="task1", seed=42)
    obs = env.step(MultiAgentAction(triage=AgentAction(command="get_dashboard")))
    assert "reward_components" in obs.metadata
    assert "penalty_reasons" in obs.metadata


def test_anti_hack_detects_loop_and_spam():
    loop_result = check_episode(
        commands=["ps aux", "ps aux", "ps aux"],
        messages=["", ""],
    )
    assert loop_result.is_hacking
    spam_result = check_episode(
        commands=["cat /var/log/syslog"],
        messages=["restart nginx now please", "restart nginx now please"],
    )
    assert spam_result.is_hacking
