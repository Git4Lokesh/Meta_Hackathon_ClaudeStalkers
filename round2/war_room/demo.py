"""Interactive demo of the Multi-Agent Incident War Room.

Shows three agents cooperating to solve a production incident,
with formatted output showing the communication channel.

Usage: PYTHONPATH=. python3 round2/war_room/demo.py
"""

from datetime import datetime
from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import MultiAgentAction, AgentAction, Message


def demo_task1():
    """Demo Task 1: Coordinated Service Restart."""
    env = WarRoomEnvironment()
    obs = env.reset(task_id="task1", seed=42)

    print("=" * 60)
    print("🔧 MULTI-AGENT INCIDENT WAR ROOM")
    print("   Task 1: Coordinated Service Restart (Easy)")
    print("=" * 60)
    print()

    # Show initial state
    print("📊 TRIAGE AGENT sees:")
    print(obs.triage.text[:300] + "...")
    print()

    steps = [
        {
            "desc": "Triage checks dashboard and escalates to Diagnosis",
            "triage": AgentAction(
                command="get_dashboard",
                message=Message(from_agent="triage", to_agent="diagnosis",
                    content="URGENT: nginx is DOWN. Please investigate /var/log/nginx/error.log",
                    timestamp=datetime.now(), round_number=1),
            ),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(command=""),
        },
        {
            "desc": "Diagnosis reads logs and reports findings to Remediation",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(
                command="cat /var/log/nginx/error.log",
                message=Message(from_agent="diagnosis", to_agent="remediation",
                    content="nginx crashed with signal 11 (segfault). Config is valid. Needs restart.",
                    timestamp=datetime.now(), round_number=2),
            ),
            "remediation": AgentAction(command=""),
        },
        {
            "desc": "Remediation restarts nginx",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(
                command="systemctl restart nginx",
                message=Message(from_agent="remediation", to_agent="all",
                    content="nginx restarted. Verifying health...",
                    timestamp=datetime.now(), round_number=3),
            ),
        },
        {
            "desc": "Remediation verifies the fix",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(command="curl http://localhost:80/health"),
        },
    ]

    for i, step in enumerate(steps, 1):
        print(f"{'─' * 60}")
        print(f"ROUND {i}: {step['desc']}")
        print(f"{'─' * 60}")

        action = MultiAgentAction(
            triage=step["triage"],
            diagnosis=step["diagnosis"],
            remediation=step["remediation"],
        )
        obs = env.step(action)

        # Show what each agent did
        for role in ["triage", "diagnosis", "remediation"]:
            a = getattr(action, role)
            if a.command:
                print(f"  [{role.upper()}] Command: {a.command}")
            if a.message:
                print(f"  [{role.upper()}] 💬 → {a.message.to_agent}: {a.message.content}")

        print(f"\n  Team Reward: {obs.team_reward:.2f} | Done: {obs.done}")

        if obs.done:
            print(f"\n  ✅ INCIDENT RESOLVED!")
            print(f"  Score: {obs.metadata.get('score', obs.team_reward):.3f}")
            print(f"  Milestones: {obs.metadata.get('milestones_achieved', [])}")
            break
        print()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_task1()
