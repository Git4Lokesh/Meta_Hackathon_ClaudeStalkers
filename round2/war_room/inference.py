"""
Multi-Agent Inference Script for the Incident War Room
======================================================

Three LLM agents (Triage, Diagnosis, Remediation) cooperate to solve
production incidents through a shared communication channel.

Features Slack-like Rich terminal UI when `rich` is installed.

Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (when --plain):
    [START] task=<task_name> env=war_room model=<model_name>
    [STEP]  round=<n> triage_action=<cmd> diagnosis_action=<cmd> remediation_action=<cmd> reward=<0.00> done=<true|false>
    [END]   success=<true|false> rounds=<n> score=<0.000>
"""

import os
import re
import sys
import json
import textwrap
import argparse
from typing import Optional, List
from datetime import datetime

from openai import OpenAI

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import (
    MultiAgentAction, AgentAction, Message, MultiAgentObservation,
)

# ---- Rich UI (optional) ----
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "war_room"
SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 300

TASKS = [
    {"task_id": "task1", "name": "coordinated-restart", "max_rounds": 10},
    {"task_id": "task2", "name": "memory-misdirection", "max_rounds": 15},
    {"task_id": "task3", "name": "cascading-conflicting", "max_rounds": 20},
    {"task_id": "task4", "name": "simultaneous-incidents", "max_rounds": 25},
]

# ---- Rich formatting config ----
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
TASK_DESCRIPTIONS = {
    "task1": "Coordinated Service Restart (Easy) — nginx has crashed",
    "task2": "Memory Leak with Misdirection (Medium) — multiple alerts, one is a red herring",
    "task3": "Cascading Failure with Conflicting Info (Hard) — phantom metrics + wrong DB password",
    "task4": "Simultaneous Incidents (Expert) — two incidents at once",
}


# ---- Role-specific system prompts ----
TRIAGE_SYSTEM = textwrap.dedent("""\
You are the TRIAGE agent in an SRE incident war room. You monitor the dashboard and coordinate the team.

Your capabilities:
- get_dashboard: See service statuses and alerts
- get_alerts: List active alerts
- get_health_summary: System health overview
- escalate <agent> <description>: Assign work to diagnosis or remediation
- send_message <to> <content>: Communicate with other agents

Your workflow:
1. Check the dashboard to understand what's happening
2. Identify the most critical issue
3. Escalate to the diagnosis agent with a clear description
4. Monitor progress and redirect if needed

IMPORTANT: Be skeptical of metrics — they may be stale or cached. Cross-check with other data sources.

RESPOND WITH EXACTLY ONE LINE in this format:
COMMAND: <your_command>
MESSAGE_TO: <diagnosis|remediation|all|none>
MESSAGE: <your message or empty>

Example:
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: nginx is down, please check /var/log/nginx/error.log
""")

DIAGNOSIS_SYSTEM = textwrap.dedent("""\
You are the DIAGNOSIS agent in an SRE incident war room. You investigate issues by reading logs and inspecting the system.

Your capabilities:
- cat <path>: Read log files
- grep <pattern> <path>: Search in files
- tail [-n N] <path>: Recent log entries
- ps aux: Process table
- top: System overview
- journalctl [-u service]: Journal logs
- dmesg: Kernel messages
- send_message <to> <content>: Share findings

Your workflow:
1. Read messages from triage to understand what to investigate
2. Read relevant log files to find the root cause
3. If logs contradict the metrics triage reported, PUSH BACK — the metrics may be stale or incorrect
4. Send your findings to the remediation agent with specific details (PIDs, file paths, exact errors)

IMPORTANT: Don't blindly trust the triage agent's assessment. Verify claims against actual logs.
If you find that an alert is a false alarm, tell triage explicitly.

RESPOND WITH EXACTLY ONE LINE in this format:
COMMAND: <your_command>
MESSAGE_TO: <triage|remediation|all|none>
MESSAGE: <your findings or empty>

Example:
COMMAND: cat /var/log/nginx/error.log
MESSAGE_TO: remediation
MESSAGE: nginx crashed with signal 11, needs restart. No config issues found.
""")

REMEDIATION_SYSTEM = textwrap.dedent("""\
You are the REMEDIATION agent in an SRE incident war room. You fix issues by restarting services, editing configs, and killing processes.

Your capabilities:
- systemctl restart <service>: Restart a service
- systemctl stop <service>: Stop a service
- edit <path> "<old>" "<new>": Edit config files
- kill -9 <PID>: Kill a process
- curl <url>: Check service health
- cat <config_path>: Read config files (NOT log files)
- send_message <to> <content>: Communicate

Your workflow:
1. Wait for findings from the diagnosis agent
2. Apply the fix (restart, edit config, kill process)
3. Verify the fix worked (curl health endpoint)
4. Report back to the team

IMPORTANT: Restart services in dependency order. Check messages for specific instructions.

RESPOND WITH EXACTLY ONE LINE in this format:
COMMAND: <your_command>
MESSAGE_TO: <triage|diagnosis|all|none>
MESSAGE: <status update or empty>

Example:
COMMAND: systemctl restart nginx
MESSAGE_TO: all
MESSAGE: nginx restarted successfully, verifying health
""")

ROLE_SYSTEMS = {
    "triage": TRIAGE_SYSTEM,
    "diagnosis": DIAGNOSIS_SYSTEM,
    "remediation": REMEDIATION_SYSTEM,
}


# ---- Rich UI Renderer ----

class RichRenderer:
    """Renders War Room output as a Slack-like incident channel."""

    def __init__(self):
        self.console = Console() if HAS_RICH else None

    def render_task_header(self, task_id: str, task_name: str, model: str):
        if not self.console:
            print(f"\n[START] task={task_name} env={BENCHMARK} model={model}")
            return
        desc = TASK_DESCRIPTIONS.get(task_id, task_name)
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold red]🔧 INCIDENT WAR ROOM[/bold red]\n"
            f"[dim]{desc}[/dim]\n\n"
            f"[dim]Model: {model}[/dim]\n"
            f"[dim]Agents: Triage · Diagnosis · Remediation[/dim]",
            title="[bold]#incident-response[/bold]",
            border_style="red",
        ))
        self.console.print()

    def render_round(self, round_num: int, max_rounds: int,
                     action: MultiAgentAction, obs: MultiAgentObservation):
        if not self.console:
            t_cmd = action.triage.command or "(none)"
            d_cmd = action.diagnosis.command or "(none)"
            r_cmd = action.remediation.command or "(none)"
            print(
                f"[STEP] round={round_num} triage_action={t_cmd} "
                f"diagnosis_action={d_cmd} remediation_action={r_cmd} "
                f"reward={obs.team_reward:.2f} done={str(obs.done).lower()}"
            )
            return

        self.console.print(f"\n[dim]{'─' * 60}[/dim]")
        self.console.print(
            f"[bold]Round {round_num}/{max_rounds}[/bold] "
            f"[dim]│[/dim] "
            f"Score: [{self._reward_color(obs.team_reward)}]{obs.team_reward:.2f}"
            f"[/{self._reward_color(obs.team_reward)}]"
        )
        self.console.print(f"[dim]{'─' * 60}[/dim]")

        for role in ["triage", "diagnosis", "remediation"]:
            a: AgentAction = getattr(action, role)
            color = ROLE_COLORS[role]
            icon = ROLE_ICONS[role]

            if a.command:
                self.console.print(
                    f"  {icon} [{color} bold]@{role.capitalize()}[/{color} bold] "
                    f"[dim]ran:[/dim] [white on dark_green] {a.command} [/white on dark_green]"
                )
            if a.message:
                target = a.message.to_agent.capitalize()
                self.console.print(
                    f"  {icon} [{color} bold]@{role.capitalize()}[/{color} bold] "
                    f"→ @{target}: [italic]{a.message.content}[/italic]"
                )

            # Show the observation snippet for this agent (truncated)
            agent_obs: str = getattr(obs, role).text
            if agent_obs and a.command:
                # Show first 2 lines of output
                lines = agent_obs.strip().splitlines()
                preview = "\n".join(lines[:2])
                if len(lines) > 2:
                    preview += f"\n[dim]... ({len(lines)-2} more lines)[/dim]"
                self.console.print(f"      [dim]{preview}[/dim]")

    def render_resolution(self, obs: MultiAgentObservation, rounds: int):
        score = obs.metadata.get("score", obs.team_reward)
        milestones = obs.metadata.get("milestones_achieved", [])
        penalties = obs.metadata.get("penalties_applied", [])
        credit = obs.metadata.get("credit_assignment", {})

        if not self.console:
            success = score >= 0.5
            print(f"[END] success={str(success).lower()} rounds={rounds} score={score:.3f}")
            return

        self.console.print()

        if score >= 0.7:
            border = "green"
            icon = "✅"
            status = "INCIDENT RESOLVED"
        elif score >= 0.3:
            border = "yellow"
            icon = "⚠️"
            status = "PARTIAL RESOLUTION"
        else:
            border = "red"
            icon = "❌"
            status = "RESOLUTION FAILED"

        # Build milestone table
        milestone_text = ""
        if milestones:
            for m in milestones:
                agent = credit.get(m, "team")
                agent_icon = ROLE_ICONS.get(agent, "👥")
                milestone_text += f"\n  {agent_icon} {m} [dim]({agent})[/dim]"

        # Unique penalties (filter time_pressure spam)
        unique_penalties = set(p for p in penalties if p not in ("time_pressure",))
        penalty_text = ""
        if unique_penalties:
            penalty_text = f"\n\n[dim]Penalties: {', '.join(sorted(unique_penalties))}[/dim]"

        self.console.print(Panel.fit(
            f"[bold {border}]{icon} {status}[/bold {border}]\n\n"
            f"Score: [bold]{score:.3f}[/bold]  │  Rounds: {rounds}  │  "
            f"Milestones: {len(milestones)}\n"
            f"{milestone_text}{penalty_text}",
            title=f"[bold {border}]Resolution[/bold {border}]",
            border_style=border,
        ))
        self.console.print()

    def render_summary(self, results: list[dict]):
        if not self.console:
            print("\n" + "=" * 60)
            print("MULTI-AGENT WAR ROOM RESULTS")
            print("=" * 60)
            for r in results:
                print(f"  {r['task_name']}: score={r['score']:.3f} "
                      f"rounds={r['rounds']} success={r['success']}")
            from round2.war_room.grader import MultiAgentGrader
            composite = MultiAgentGrader.composite_score(
                {r["task_id"]: r["score"] for r in results}
            )
            print(f"  Composite: {composite:.3f}")
            print("=" * 60)
            return

        from round2.war_room.grader import MultiAgentGrader
        composite = MultiAgentGrader.composite_score(
            {r["task_id"]: r["score"] for r in results}
        )

        table = Table(
            title="Multi-Agent War Room Results",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Task", style="bold")
        table.add_column("Score", justify="center")
        table.add_column("Rounds", justify="center")
        table.add_column("Status", justify="center")

        for r in results:
            score = r["score"]
            if score >= 0.7:
                score_style = "bold green"
                status = "✅ Resolved"
            elif score >= 0.3:
                score_style = "bold yellow"
                status = "⚠️ Partial"
            else:
                score_style = "bold red"
                status = "❌ Failed"

            table.add_row(
                r["task_name"],
                f"[{score_style}]{score:.3f}[/{score_style}]",
                str(r["rounds"]),
                status,
            )

        table.add_section()
        comp_style = "bold green" if composite >= 0.5 else "bold red"
        table.add_row(
            "[bold]Composite[/bold]",
            f"[{comp_style}]{composite:.3f}[/{comp_style}]",
            "",
            "",
        )

        self.console.print()
        self.console.print(table)
        self.console.print()

    @staticmethod
    def _reward_color(reward: float) -> str:
        if reward >= 0.7:
            return "green"
        elif reward >= 0.3:
            return "yellow"
        return "red"


# ---- Response parsing ----

def parse_agent_response(text: str, role: str, round_num: int) -> AgentAction:
    """Parse LLM response into command + optional message."""
    text = text.strip()

    # Remove markdown code blocks
    text = re.sub(r'```\w*\n?', '', text)
    text = text.strip('`').strip()

    command = ""
    msg_to = ""
    msg_content = ""

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("COMMAND:"):
            command = line.split(":", 1)[1].strip()
        elif line.upper().startswith("MESSAGE_TO:"):
            msg_to = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("MESSAGE:"):
            msg_content = line.split(":", 1)[1].strip()

    # Fallback: if no COMMAND: prefix found, try to extract a command from the first line
    if not command:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("MESSAGE"):
                command = line
                break

    message = None
    if msg_to and msg_to != "none" and msg_content:
        message = Message(
            from_agent=role,
            to_agent=msg_to,
            content=msg_content,
            timestamp=datetime.now(),
            round_number=round_num,
        )

    return AgentAction(command=command, message=message)


# ---- LLM calls ----

def get_agent_response(
    client: OpenAI,
    role: str,
    observation_text: str,
    round_num: int,
    conversation: list[dict],
    max_retries: int = 2,
) -> AgentAction:
    """Get one agent's action from the LLM."""
    messages = list(conversation)
    messages.append({
        "role": "user",
        "content": f"[Round {round_num}]\n{observation_text}\n\nWhat do you do?",
    })

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            action = parse_agent_response(text, role, round_num)

            # Update conversation
            conversation.append({"role": "user", "content": f"[Round {round_num}]\n{observation_text}"})
            conversation.append({"role": "assistant", "content": text})

            return action
        except Exception as exc:
            print(f"[DEBUG] {role} request failed (attempt {attempt+1}): {exc}", flush=True)

    return AgentAction(command="")


# ---- Main ----

def run_task(client: OpenAI, env: WarRoomEnvironment, task_config: dict,
             renderer: RichRenderer) -> dict:
    """Run one task with 3 LLM agents."""
    task_id = task_config["task_id"]
    task_name = task_config["name"]
    max_rounds = task_config["max_rounds"]

    rounds_taken = 0
    score = 0.01
    success = False

    renderer.render_task_header(task_id, task_name, MODEL_NAME)

    try:
        obs = env.reset(task_id=task_id, seed=SEED)

        # Per-agent conversation histories
        conversations = {
            role: [{"role": "system", "content": ROLE_SYSTEMS[role]}]
            for role in ["triage", "diagnosis", "remediation"]
        }

        for round_num in range(1, max_rounds + 1):
            if obs.done:
                break

            # Get each agent's action from the LLM
            triage_action = get_agent_response(
                client, "triage", obs.triage.text, round_num, conversations["triage"],
            )
            diag_action = get_agent_response(
                client, "diagnosis", obs.diagnosis.text, round_num, conversations["diagnosis"],
            )
            remed_action = get_agent_response(
                client, "remediation", obs.remediation.text, round_num, conversations["remediation"],
            )

            # Step the environment
            action = MultiAgentAction(
                triage=triage_action,
                diagnosis=diag_action,
                remediation=remed_action,
            )
            obs = env.step(action)
            rounds_taken = round_num

            renderer.render_round(round_num, max_rounds, action, obs)

            if obs.done:
                break

        score = obs.metadata.get("score", obs.team_reward)
        score = max(0.01, min(0.99, score))
        success = score >= 0.5

        renderer.render_resolution(obs, rounds_taken)

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    return {
        "task_id": task_id,
        "task_name": task_name,
        "score": score,
        "rounds": rounds_taken,
        "success": success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent War Room Inference")
    parser.add_argument("--plain", action="store_true",
                        help="Disable Rich UI — use plain [START]/[STEP]/[END] format")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Run specific tasks (e.g., --tasks task1 task3)")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: HF_TOKEN or API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # Decide rendering mode
    use_rich = HAS_RICH and not args.plain
    renderer = RichRenderer() if use_rich else RichRenderer.__new__(RichRenderer)
    if not use_rich:
        renderer.console = None

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = WarRoomEnvironment()

    # Filter tasks if specified
    task_list = TASKS
    if args.tasks:
        task_list = [t for t in TASKS if t["task_id"] in args.tasks]

    results = []
    for task_config in task_list:
        result = run_task(client, env, task_config, renderer)
        results.append(result)

    renderer.render_summary(results)


if __name__ == "__main__":
    main()
