"""
Multi-Agent Inference Script for the Incident War Room
======================================================

Three LLM agents (Triage, Diagnosis, Remediation) cooperate to solve
production incidents through a shared communication channel.

Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=war_room model=<model_name>
    [STEP]  round=<n> triage_action=<cmd> diagnosis_action=<cmd> remediation_action=<cmd> reward=<0.00> done=<true|false>
    [END]   success=<true|false> rounds=<n> score=<0.000>
"""

import os
import re
import sys
import json
import textwrap
from typing import Optional, List
from datetime import datetime

from openai import OpenAI

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import (
    MultiAgentAction, AgentAction, Message, MultiAgentObservation,
)

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

# Role-specific system prompts
TRIAGE_SYSTEM = textwrap.dedent("""
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

RESPOND WITH EXACTLY ONE LINE in this format:
COMMAND: <your_command>
MESSAGE_TO: <diagnosis|remediation|all|none>
MESSAGE: <your message or empty>

Example:
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: nginx is down, please check /var/log/nginx/error.log
""").strip()

DIAGNOSIS_SYSTEM = textwrap.dedent("""
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
3. Send your findings to the remediation agent with specific details (PIDs, file paths, exact errors)

RESPOND WITH EXACTLY ONE LINE in this format:
COMMAND: <your_command>
MESSAGE_TO: <triage|remediation|all|none>
MESSAGE: <your findings or empty>

Example:
COMMAND: cat /var/log/nginx/error.log
MESSAGE_TO: remediation
MESSAGE: nginx crashed with signal 11, needs restart. No config issues found.
""").strip()

REMEDIATION_SYSTEM = textwrap.dedent("""
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
""").strip()

ROLE_SYSTEMS = {
    "triage": TRIAGE_SYSTEM,
    "diagnosis": DIAGNOSIS_SYSTEM,
    "remediation": REMEDIATION_SYSTEM,
}


# ---- Logging ----

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(round_num: int, triage_cmd: str, diag_cmd: str, remed_cmd: str,
             reward: float, done: bool) -> None:
    done_val = str(done).lower()
    print(
        f"[STEP] round={round_num} triage_action={triage_cmd} "
        f"diagnosis_action={diag_cmd} remediation_action={remed_cmd} "
        f"reward={reward:.2f} done={done_val}",
        flush=True,
    )

def log_end(success: bool, rounds: int, score: float) -> None:
    print(
        f"[END] success={str(success).lower()} rounds={rounds} score={score:.3f}",
        flush=True,
    )


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

def run_task(client: OpenAI, env: WarRoomEnvironment, task_config: dict) -> dict:
    """Run one task with 3 LLM agents."""
    task_id = task_config["task_id"]
    task_name = task_config["name"]
    max_rounds = task_config["max_rounds"]

    rounds_taken = 0
    score = 0.01
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

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

            log_step(
                round_num=round_num,
                triage_cmd=triage_action.command or "(none)",
                diag_cmd=diag_action.command or "(none)",
                remed_cmd=remed_action.command or "(none)",
                reward=obs.team_reward,
                done=obs.done,
            )

            if obs.done:
                break

        score = obs.metadata.get("score", obs.team_reward)
        score = max(0.01, min(0.99, score))
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, rounds=rounds_taken, score=score)

    return {
        "task_id": task_id,
        "task_name": task_name,
        "score": score,
        "rounds": rounds_taken,
        "success": success,
    }


def main() -> None:
    if not API_KEY:
        print("Error: HF_TOKEN or API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = WarRoomEnvironment()

    results = []
    for task_config in TASKS:
        result = run_task(client, env, task_config)
        results.append(result)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("MULTI-AGENT WAR ROOM RESULTS", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(f"  {r['task_name']}: score={r['score']:.3f} rounds={r['rounds']} success={r['success']}", flush=True)

    from round2.war_room.grader import MultiAgentGrader
    composite = MultiAgentGrader.composite_score({r["task_id"]: r["score"] for r in results})
    print(f"  Composite: {composite:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
