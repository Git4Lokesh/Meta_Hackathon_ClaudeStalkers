"""Rich terminal demo of the Multi-Agent Incident War Room.

Shows agents cooperating in a Slack-like incident channel format.

Usage:
    pip install rich
    PYTHONPATH=. python3 round2/war_room/demo_rich.py
"""

import sys
from datetime import datetime
from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import MultiAgentAction, AgentAction, Message

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


ROLE_COLORS = {
    "triage": "yellow",
    "diagnosis": "cyan",
    "remediation": "green",
}

ROLE_ICONS = {
    "triage": "🚨",
    "diagnosis": "🔎",
    "remediation": "🛠️",
}


def demo_task3_rich():
    """Demo Task 3 with rich formatting — the theory-of-mind showcase."""
    if not HAS_RICH:
        print("Install rich for the beautiful demo: pip install rich")
        print("Falling back to plain demo...")
        from round2.war_room.demo import demo_task1
        demo_task1()
        return

    console = Console()
    env = WarRoomEnvironment()
    obs = env.reset(task_id="task3", seed=42)

    console.print()
    console.print(Panel.fit(
        "[bold red]🔧 INCIDENT WAR ROOM[/bold red]\n"
        "[dim]Task 3: Cascading Failure with Conflicting Information (Hard)[/dim]\n\n"
        "[yellow]⚡ Three agents must trace a root cause across service boundaries.[/yellow]\n"
        "[yellow]⚡ Redis warnings are a RED HERRING — the real issue is a wrong DB password.[/yellow]\n"
        "[yellow]⚡ Diagnosis must push back on Triage's initial misdirection.[/yellow]",
        title="[bold]#incident-response[/bold]",
        border_style="red",
    ))
    console.print()

    steps = [
        {
            "desc": "Triage checks dashboard — sees Redis warnings prominently",
            "triage": AgentAction(
                command="get_dashboard",
                message=Message(from_agent="triage", to_agent="diagnosis",
                    content="🚨 URGENT: Redis memory at 72%! Also seeing some DB connector issues. Check Redis first — it looks critical.",
                    timestamp=datetime.now(), round_number=1),
            ),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(command=""),
        },
        {
            "desc": "Diagnosis checks Redis logs — finds nothing critical",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(
                command="cat /var/log/redis/redis.log",
                message=Message(from_agent="diagnosis", to_agent="triage",
                    content="Redis logs look normal — memory at 72% is within tolerance. Those metrics might be stale. Let me check the DB connector instead.",
                    timestamp=datetime.now(), round_number=2),
            ),
            "remediation": AgentAction(command=""),
        },
        {
            "desc": "Diagnosis reads DB connector logs — finds the REAL root cause",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(
                command="cat /var/log/db_connector/connector.log",
                message=Message(from_agent="diagnosis", to_agent="remediation",
                    content="FOUND IT: db_connector crashed due to authentication failure. Wrong password in /etc/app/database.yml. Replace 'wrong_password_123' with 'correct_db_pass_456'. Then restart db_connector → app_server → load_balancer IN THAT ORDER.",
                    timestamp=datetime.now(), round_number=3),
            ),
            "remediation": AgentAction(command=""),
        },
        {
            "desc": "Remediation reads config to confirm",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(command="cat /etc/app/database.yml"),
        },
        {
            "desc": "Remediation fixes the password",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(
                command='edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"',
                message=Message(from_agent="remediation", to_agent="all",
                    content="Password fixed. Starting dependency-ordered restart...",
                    timestamp=datetime.now(), round_number=5),
            ),
        },
        {
            "desc": "Restart db_connector",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(command="systemctl restart db_connector"),
        },
        {
            "desc": "Restart app_server",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(command="systemctl restart app_server"),
        },
        {
            "desc": "Restart load_balancer",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(command="systemctl restart load_balancer"),
        },
        {
            "desc": "Verify all services",
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(command="curl http://localhost:80/health"),
        },
    ]

    for i, step in enumerate(steps, 1):
        console.print(f"\n[dim]{'─' * 60}[/dim]")
        console.print(f"[bold]Round {i}[/bold] [dim]{step['desc']}[/dim]")
        console.print(f"[dim]{'─' * 60}[/dim]")

        action = MultiAgentAction(
            triage=step["triage"],
            diagnosis=step["diagnosis"],
            remediation=step["remediation"],
        )
        obs = env.step(action)

        # Show what each agent did in Slack-like format
        for role in ["triage", "diagnosis", "remediation"]:
            a = getattr(action, role)
            color = ROLE_COLORS[role]
            icon = ROLE_ICONS[role]

            if a.command:
                console.print(f"  {icon} [{color} bold]@{role.capitalize()}[/{color} bold] [dim]ran:[/dim] [white on dark_green] {a.command} [/white on dark_green]")
            if a.message:
                target = a.message.to_agent
                console.print(f"  {icon} [{color} bold]@{role.capitalize()}[/{color} bold] → @{target.capitalize()}: [italic]{a.message.content}[/italic]")

        # Show reward
        reward_color = "green" if obs.team_reward > 0.5 else "yellow" if obs.team_reward > 0.2 else "red"
        console.print(f"\n  [{reward_color}]Score: {obs.team_reward:.2f}[/{reward_color}] [dim]| Done: {obs.done}[/dim]")

        if obs.done:
            console.print()
            score = obs.metadata.get('score', obs.team_reward)
            milestones = obs.metadata.get('milestones_achieved', [])

            if score >= 0.7:
                console.print(Panel.fit(
                    f"[bold green]✅ INCIDENT RESOLVED[/bold green]\n\n"
                    f"Score: [bold]{score:.3f}[/bold]\n"
                    f"Rounds: {i}\n"
                    f"Milestones: {len(milestones)}\n\n"
                    f"[dim]{', '.join(milestones)}[/dim]",
                    title="[bold green]Resolution[/bold green]",
                    border_style="green",
                ))
            else:
                console.print(f"[yellow]Score: {score:.3f} | Milestones: {milestones}[/yellow]")
            break

    console.print()


def demo_task1_rich():
    """Demo Task 1 with rich formatting."""
    if not HAS_RICH:
        from round2.war_room.demo import demo_task1
        demo_task1()
        return

    console = Console()
    env = WarRoomEnvironment()
    obs = env.reset(task_id="task1", seed=42)

    console.print()
    console.print(Panel.fit(
        "[bold red]🔧 INCIDENT WAR ROOM[/bold red]\n"
        "[dim]Task 1: Coordinated Service Restart (Easy)[/dim]\n\n"
        "[yellow]⚡ nginx has crashed. Three agents must coordinate to fix it.[/yellow]",
        title="[bold]#incident-response[/bold]",
        border_style="red",
    ))

    steps = [
        {
            "triage": AgentAction(
                command="get_dashboard",
                message=Message(from_agent="triage", to_agent="diagnosis",
                    content="nginx is DOWN. Please check /var/log/nginx/error.log",
                    timestamp=datetime.now(), round_number=1),
            ),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(command=""),
        },
        {
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(
                command="cat /var/log/nginx/error.log",
                message=Message(from_agent="diagnosis", to_agent="remediation",
                    content="nginx crashed with signal 11. Needs restart.",
                    timestamp=datetime.now(), round_number=2),
            ),
            "remediation": AgentAction(command=""),
        },
        {
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(
                command="systemctl restart nginx",
                message=Message(from_agent="remediation", to_agent="all",
                    content="nginx restarted. Verifying...",
                    timestamp=datetime.now(), round_number=3),
            ),
        },
        {
            "triage": AgentAction(command=""),
            "diagnosis": AgentAction(command=""),
            "remediation": AgentAction(command="curl http://localhost:80/health"),
        },
    ]

    for i, step in enumerate(steps, 1):
        console.print(f"\n[dim]{'─' * 50}[/dim]")
        action = MultiAgentAction(triage=step["triage"], diagnosis=step["diagnosis"], remediation=step["remediation"])
        obs = env.step(action)

        for role in ["triage", "diagnosis", "remediation"]:
            a = getattr(action, role)
            icon = ROLE_ICONS[role]
            color = ROLE_COLORS[role]
            if a.command:
                console.print(f"  {icon} [{color} bold]@{role.capitalize()}[/{color} bold]: [white on dark_green] {a.command} [/white on dark_green]")
            if a.message:
                console.print(f"  {icon} [{color} bold]@{role.capitalize()}[/{color} bold] → @{a.message.to_agent}: [italic]{a.message.content}[/italic]")

        reward_color = "green" if obs.team_reward > 0.5 else "yellow"
        console.print(f"  [dim]Score: [{reward_color}]{obs.team_reward:.2f}[/{reward_color}][/dim]")

        if obs.done:
            score = obs.metadata.get('score', obs.team_reward)
            console.print(Panel.fit(
                f"[bold green]✅ RESOLVED in {i} rounds — Score: {score:.3f}[/bold green]",
                border_style="green",
            ))
            break
    console.print()


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "task3"
    if task == "task1":
        demo_task1_rich()
    else:
        demo_task3_rich()
