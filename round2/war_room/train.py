"""TRL GRPO Training Script for the Multi-Agent Incident War Room.

Trains LLM agents to cooperate in incident response using Group Relative
Policy Optimization (GRPO) from HuggingFace TRL.

Usage (Colab):
    !pip install trl unsloth openai
    !python round2/war_room/train.py --model unsloth/Qwen2.5-7B --episodes 100

Local demo (no GPU needed):
    PYTHONPATH=. python round2/war_room/train.py --episodes 50

Environment variables:
    HF_TOKEN: HuggingFace token for model access
"""

import argparse
import os
import json
import random
import sys
from typing import Optional
from datetime import datetime

# Environment imports (always available)
from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import MultiAgentAction, AgentAction, Message


# ---- Role-specific system prompts ----

TRIAGE_PROMPT = """You are the TRIAGE agent in an SRE incident war room.
You can see the monitoring dashboard and alerts. Your job is to:
1. Identify which services are down or degraded
2. Prioritize incidents by severity
3. Escalate to the diagnosis agent with clear descriptions

Available commands: get_dashboard, get_alerts, get_health_summary, escalate <agent> <description>, send_message <to> <content>

Respond with a JSON object: {"command": "...", "message_to": "...", "message_content": "..."}
Leave message fields empty if not sending a message."""

DIAGNOSIS_PROMPT = """You are the DIAGNOSIS agent in an SRE incident war room.
You can read logs and inspect processes. Your job is to:
1. Investigate issues escalated by the triage agent
2. Read relevant log files to identify root causes
3. Report findings to the remediation agent with specific details (PIDs, file paths, error messages)

Available commands: cat, grep, tail, ps, top, journalctl, dmesg, send_message <to> <content>

Respond with a JSON object: {"command": "...", "message_to": "...", "message_content": "..."}
Leave message fields empty if not sending a message."""

REMEDIATION_PROMPT = """You are the REMEDIATION agent in an SRE incident war room.
You can restart services, edit configs, and kill processes. Your job is to:
1. Apply fixes based on diagnosis agent's findings
2. Restart services in the correct dependency order
3. Verify fixes are working

Available commands: systemctl restart/stop <svc>, edit <path> <old> <new>, kill -9 <PID>, curl <url>, cat <config_path>, send_message <to> <content>

Respond with a JSON object: {"command": "...", "message_to": "...", "message_content": "..."}
Leave message fields empty if not sending a message."""

ROLE_PROMPTS = {
    "triage": TRIAGE_PROMPT,
    "diagnosis": DIAGNOSIS_PROMPT,
    "remediation": REMEDIATION_PROMPT,
}


# ---- Agent action parsing ----

def parse_agent_response(text: str, role: str, round_num: int) -> AgentAction:
    """Parse LLM response into an AgentAction."""
    text = text.strip()

    # Try JSON parsing first
    try:
        data = json.loads(text)
        command = data.get("command", "")
        msg_to = data.get("message_to", "")
        msg_content = data.get("message_content", "")

        message = None
        if msg_to and msg_content:
            message = Message(
                from_agent=role,
                to_agent=msg_to,
                content=msg_content,
                timestamp=datetime.now(),
                round_number=round_num,
            )

        return AgentAction(command=command, message=message)
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: treat entire response as a command
    return AgentAction(command=text)


# ---- Reward function for GRPO ----

def compute_episode_reward(
    env: WarRoomEnvironment,
    task_id: str,
    seed: int,
    agent_responses: dict[str, list[str]],  # {role: [response_per_round]}
) -> dict[str, float]:
    """Run a full episode and return per-agent rewards."""
    obs = env.reset(task_id=task_id, seed=seed)

    max_rounds = obs.metadata["max_rounds"]
    final_reward = 0.0

    for round_num in range(1, max_rounds + 1):
        if obs.done:
            break

        # Build actions from agent responses
        actions = {}
        for role in ["triage", "diagnosis", "remediation"]:
            if round_num - 1 < len(agent_responses.get(role, [])):
                response = agent_responses[role][round_num - 1]
                actions[role] = parse_agent_response(response, role, round_num)
            else:
                actions[role] = AgentAction(command="")

        multi_action = MultiAgentAction(**actions)
        obs = env.step(multi_action)
        final_reward = obs.team_reward

    return {
        "team": final_reward,
        "triage": obs.triage.reward,
        "diagnosis": obs.diagnosis.reward,
        "remediation": obs.remediation.reward,
    }


# ---- Heuristic agents (for demo/baseline) ----

def _heuristic_action_task1(round_num: int, skill_level: float) -> MultiAgentAction:
    """Task 1: Coordinated nginx restart."""
    # Skill 0.0 = random/bad actions, Skill 1.0 = optimal
    rng = random.Random()

    if skill_level < 0.3:
        # Untrained: random/wrong commands, no communication
        bad_steps = [
            MultiAgentAction(
                triage=AgentAction(command="get_dashboard"),
                diagnosis=AgentAction(command="ps aux"),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command="get_alerts"),
                diagnosis=AgentAction(command="cat /var/log/syslog"),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command="get_health_summary"),
                diagnosis=AgentAction(command="top"),
                remediation=AgentAction(command="curl http://localhost:80/health"),
            ),
        ]
        if round_num < len(bad_steps):
            return bad_steps[round_num]
        return MultiAgentAction()

    elif skill_level < 0.7:
        # Partially trained: correct actions but wasteful
        steps = [
            MultiAgentAction(
                triage=AgentAction(command="get_dashboard"),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(
                    command="get_alerts",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="nginx is down, check logs",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command="cat /var/log/nginx/error.log"),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(
                    command="",
                    message=Message(from_agent="diagnosis", to_agent="remediation",
                        content="nginx crashed with signal 11, needs restart",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command="systemctl restart nginx"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="curl http://localhost:80/health"),
            ),
        ]
        if round_num < len(steps):
            return steps[round_num]
        return MultiAgentAction()

    else:
        # Well-trained: efficient, parallel actions, good communication
        steps = [
            MultiAgentAction(
                triage=AgentAction(
                    command="get_dashboard",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="nginx is DOWN. Please check /var/log/nginx/error.log",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(
                    command="cat /var/log/nginx/error.log",
                    message=Message(from_agent="diagnosis", to_agent="remediation",
                        content="nginx crashed with signal 11. Needs restart.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(
                    command="systemctl restart nginx",
                    message=Message(from_agent="remediation", to_agent="all",
                        content="nginx restarted. Verifying...",
                        timestamp=datetime.now(), round_number=round_num),
                ),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="curl http://localhost:80/health"),
            ),
        ]
        if round_num < len(steps):
            return steps[round_num]
        return MultiAgentAction()


def _heuristic_action_task2(round_num: int, skill_level: float) -> MultiAgentAction:
    """Task 2: Memory leak with misdirection."""
    if skill_level < 0.3:
        # Untrained: chases the CPU red herring, no coordination
        bad_steps = [
            MultiAgentAction(
                triage=AgentAction(
                    command="get_dashboard",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="High CPU on api_gateway! Check it now!",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command="top"),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart api_gateway"),
            ),
        ]
        if round_num < len(bad_steps):
            return bad_steps[round_num]
        return MultiAgentAction()

    elif skill_level < 0.7:
        # Partially trained: finds memory issue but slow
        steps = [
            MultiAgentAction(
                triage=AgentAction(
                    command="get_dashboard",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="Multiple alerts: high CPU on api_gateway AND memory issue on data_processor. Please check both.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command="ps aux"),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command="cat /var/log/syslog"),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(
                    command="",
                    message=Message(from_agent="diagnosis", to_agent="remediation",
                        content="data_processor_worker PID 1000 leaking memory. Kill it and restart data_processor.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command="kill -9 1000"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart data_processor"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="curl http://localhost:8081/health"),
            ),
        ]
        if round_num < len(steps):
            return steps[round_num]
        return MultiAgentAction()

    else:
        # Well-trained: prioritizes memory, ignores CPU red herring
        steps = [
            MultiAgentAction(
                triage=AgentAction(
                    command="get_dashboard",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="Memory issue on data_processor looks critical (possible OOM). Also see high CPU on api_gateway but memory is priority.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(
                    command="ps aux",
                    message=Message(from_agent="diagnosis", to_agent="remediation",
                        content="data_processor_worker PID 1000 using 2800MB — OOM risk. Kill it and restart data_processor.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command="cat /var/log/syslog"),
                remediation=AgentAction(command="kill -9 1000"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart data_processor"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="curl http://localhost:8081/health"),
            ),
        ]
        if round_num < len(steps):
            return steps[round_num]
        return MultiAgentAction()


def _heuristic_action_task3(round_num: int, skill_level: float) -> MultiAgentAction:
    """Task 3: Cascading failure with conflicting information."""
    if skill_level < 0.3:
        # Untrained: chases Redis red herring, ignores DB auth issue
        bad_steps = [
            MultiAgentAction(
                triage=AgentAction(
                    command="get_dashboard",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="Redis memory is critical! Check Redis NOW!",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command="cat /var/log/redis/redis.log"),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart redis"),
            ),
        ]
        if round_num < len(bad_steps):
            return bad_steps[round_num]
        return MultiAgentAction()

    elif skill_level < 0.7:
        # Partially trained: eventually finds DB issue but wastes rounds on Redis
        steps = [
            MultiAgentAction(
                triage=AgentAction(
                    command="get_dashboard",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="Redis memory warning AND db_connector issues. Check Redis first.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command="cat /var/log/redis/redis.log"),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(
                    command="cat /var/log/db_connector/connector.log",
                    message=Message(from_agent="diagnosis", to_agent="triage",
                        content="Redis logs look normal. Checking db_connector...",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(
                    command="",
                    message=Message(from_agent="diagnosis", to_agent="remediation",
                        content="FOUND: db_connector failed authentication. Wrong password in /etc/app/database.yml. Change wrong_password_123 to correct_db_pass_456.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command='edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"'),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart db_connector"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart app_server"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart load_balancer"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="curl http://localhost:80/health"),
            ),
        ]
        if round_num < len(steps):
            return steps[round_num]
        return MultiAgentAction()

    else:
        # Well-trained: pushes back on Redis, detects stale metrics
        steps = [
            MultiAgentAction(
                triage=AgentAction(
                    command="get_dashboard",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="Multiple alerts: Redis memory warning, monitoring CPU spike, and db_connector issues.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(
                    command="cat /var/log/db_connector/connector.log",
                    message=Message(from_agent="diagnosis", to_agent="triage",
                        content="Redis is NOT the issue — those metrics look stale. The real problem is db_connector: authentication failed due to wrong password in /etc/app/database.yml.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(
                    command="",
                    message=Message(from_agent="diagnosis", to_agent="remediation",
                        content="Root cause: wrong password in /etc/app/database.yml. Replace 'wrong_password_123' with 'correct_db_pass_456'. Then restart: db_connector → app_server → load_balancer.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command='edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"'),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart db_connector"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart app_server"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart load_balancer"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="curl http://localhost:80/health"),
            ),
        ]
        if round_num < len(steps):
            return steps[round_num]
        return MultiAgentAction()


def _heuristic_action_task4(round_num: int, skill_level: float) -> MultiAgentAction:
    """Task 4: Simultaneous incidents (nginx + memory leak)."""
    if skill_level < 0.3:
        # Untrained: only addresses one incident
        bad_steps = [
            MultiAgentAction(
                triage=AgentAction(command="get_dashboard"),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(
                    command="",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="nginx is down! Fix it!",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command="cat /var/log/nginx/error.log"),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart nginx"),
            ),
        ]
        if round_num < len(bad_steps):
            return bad_steps[round_num]
        return MultiAgentAction()

    elif skill_level < 0.7:
        # Partially trained: addresses both but sequentially and slowly
        steps = [
            MultiAgentAction(
                triage=AgentAction(
                    command="get_dashboard",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="TWO incidents: 1) nginx DOWN 2) data_processor memory critical. Check both.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command="cat /var/log/nginx/error.log"),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(
                    command="ps aux",
                    message=Message(from_agent="diagnosis", to_agent="remediation",
                        content="nginx crashed signal 11 — restart it. Also data_processor_worker PID 1000 leaking memory — kill it.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command="systemctl restart nginx"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command="cat /var/log/syslog"),
                remediation=AgentAction(command="kill -9 1000"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart data_processor"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="curl http://localhost:80/health"),
            ),
        ]
        if round_num < len(steps):
            return steps[round_num]
        return MultiAgentAction()

    else:
        # Well-trained: handles both incidents in parallel, efficient communication
        steps = [
            MultiAgentAction(
                triage=AgentAction(
                    command="get_dashboard",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="TWO simultaneous incidents: nginx crashed AND data_processor memory leak (possible OOM). Investigate both.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(
                    command="cat /var/log/nginx/error.log",
                    message=Message(from_agent="diagnosis", to_agent="remediation",
                        content="nginx: signal 11 crash → restart. data_processor_worker PID 1000 leaking memory → kill & restart.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command=""),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command="ps aux"),
                remediation=AgentAction(command="systemctl restart nginx"),
            ),
            MultiAgentAction(
                triage=AgentAction(
                    command="",
                    message=Message(from_agent="triage", to_agent="diagnosis",
                        content="Both incidents: nginx AND memory leak (OOM).",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                diagnosis=AgentAction(
                    command="cat /var/log/syslog",
                    message=Message(from_agent="diagnosis", to_agent="remediation",
                        content="Confirmed: OOM killer hit PID 1000 (data_processor_worker). Kill and restart data_processor.",
                        timestamp=datetime.now(), round_number=round_num),
                ),
                remediation=AgentAction(command="kill -9 1000"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="systemctl restart data_processor"),
            ),
            MultiAgentAction(
                triage=AgentAction(command=""),
                diagnosis=AgentAction(command=""),
                remediation=AgentAction(command="curl http://localhost:80/health"),
            ),
        ]
        if round_num < len(steps):
            return steps[round_num]
        return MultiAgentAction()


HEURISTIC_DISPATCH = {
    "task1": _heuristic_action_task1,
    "task2": _heuristic_action_task2,
    "task3": _heuristic_action_task3,
    "task4": _heuristic_action_task4,
}


# ---- Training loop ----

def train(
    model_name: str = "unsloth/Qwen2.5-7B",
    num_episodes: int = 100,
    tasks: list[str] = None,
    output_dir: str = "outputs/war_room_training",
):
    """Train agents using GRPO on the War Room environment.

    This function is designed to be called from Colab with compute credits.
    It uses TRL's GRPOTrainer for optimization.

    In demo mode (no TRL), runs heuristic agents with progressively
    increasing skill levels to simulate training improvement.
    """
    tasks = tasks or ["task1", "task2", "task3", "task4"]

    print(f"Training config:")
    print(f"  Model: {model_name}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Tasks: {tasks}")
    print(f"  Output: {output_dir}")

    # Try importing TRL/Unsloth (only available in Colab with GPU)
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        HAS_TRL = True
    except ImportError:
        HAS_TRL = False
        print("WARNING: TRL not installed. Running in demo mode (simulated training).")
        print("Install with: pip install trl unsloth")

    env = WarRoomEnvironment()

    # Curriculum: cycle through tasks with increasing difficulty
    task_curriculum = []
    for epoch in range(num_episodes):
        progress = epoch / max(num_episodes - 1, 1)
        if progress < 0.25:
            task_curriculum.append("task1")
        elif progress < 0.5:
            task_curriculum.append("task2" if epoch % 2 == 0 else "task1")
        elif progress < 0.75:
            task_curriculum.append("task3" if epoch % 2 == 0 else "task2")
        else:
            task_curriculum.append("task4" if epoch % 3 == 0 else "task3")

    # Training metrics
    metrics = {
        "episode": [],
        "task": [],
        "team_reward": [],
        "rounds_used": [],
        "milestones_achieved": [],
    }

    if not HAS_TRL:
        print("\n--- Demo Mode: Simulated training with progressive skill levels ---\n")

        for ep in range(num_episodes):
            task_id = task_curriculum[ep]

            # Skill level increases over training (with noise)
            raw_skill = ep / max(num_episodes - 1, 1)
            noise = random.gauss(0, 0.08)
            skill_level = max(0.0, min(1.0, raw_skill + noise))

            obs = env.reset(task_id=task_id, seed=ep)
            rounds = 0

            heuristic_fn = HEURISTIC_DISPATCH.get(task_id, _heuristic_action_task1)

            for r in range(obs.metadata["max_rounds"]):
                if obs.done:
                    break
                rounds += 1

                action = heuristic_fn(r, skill_level)
                obs = env.step(action)

            score = obs.metadata.get("score", obs.team_reward)
            milestones = obs.metadata.get("milestones_achieved", [])

            metrics["episode"].append(ep)
            metrics["task"].append(task_id)
            metrics["team_reward"].append(score)
            metrics["rounds_used"].append(rounds)
            metrics["milestones_achieved"].append(len(milestones))

            # Print every 5th episode
            if ep % 5 == 0 or ep == num_episodes - 1:
                m_list = ", ".join(milestones[:3])
                if len(milestones) > 3:
                    m_list += f", +{len(milestones)-3} more"
                print(
                    f"  Ep {ep:3d} | task={task_id} | skill={skill_level:.2f} | "
                    f"score={score:.3f} | rounds={rounds:2d} | "
                    f"milestones={len(milestones)} [{m_list}]"
                )

        print("\n--- Simulated training complete. Install TRL for real GRPO training. ---")
    else:
        # Real GRPO training — delegates to train_colab.py
        print("\n--- Starting GRPO Training ---\n")
        print("Delegating to train_colab.py for full GRPO pipeline...\n")

        from round2.war_room.train_colab import train_grpo
        grpo_metrics = train_grpo(
            model_name=model_name,
            num_episodes=num_episodes,
            tasks=tasks,
            output_dir=output_dir,
        )

        # Merge GRPO metrics into our metrics dict
        if grpo_metrics:
            for key in metrics:
                if key in grpo_metrics:
                    metrics[key] = grpo_metrics[key]

        print(f"\nGRPO training complete. Model saved to {output_dir}")

    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train War Room agents with GRPO")
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B", help="Model name")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--output", default="outputs/war_room_training", help="Output directory")
    args = parser.parse_args()

    train(model_name=args.model, num_episodes=args.episodes, output_dir=args.output)


if __name__ == "__main__":
    main()
