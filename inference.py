"""
Inference Script for Multi-Agent SRE Incident War Room
======================================================

Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=sre_env model=<model_name>
    [STEP]  step=<n> role=<agent_role> action=<action_str> reward=<0.00> done=<true|false>
    [END]   success=<true|false> steps=<n> score=<0.000>
"""

import os
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

from sre_env.client import SREClient

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "sre_env"
SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 200

TASKS = [
    {"task_id": "task1", "name": "service-restart", "max_steps": 20},
    {"task_id": "task2", "name": "memory-leak-diagnosis", "max_steps": 30},
    {"task_id": "task3", "name": "cascading-failure", "max_steps": 40},
]

ROLES = ["triage", "diagnosis", "remediation"]

ROLE_PROMPTS = {
    "triage": textwrap.dedent("""
        You are the SRE TRIAGE Agent.
        Your job is to identify failing systems by checking metrics and status, then delegate an investigation to the diagnosis agent.
        
        You can ONLY run these commands: top, free, netstat, ps, df, message, help.
        
        To delegate, use the message command. Example:
        message diagnosis "I see high CPU on app_server, please investigate the logs."
        
        IMPORTANT: Respond with ONLY a single Linux command. No explanations.
    """).strip(),

    "diagnosis": textwrap.dedent("""
        You are the SRE DIAGNOSIS Agent.
        Your job is to read logs and files to find the root cause of an issue handed to you by Triage.
        
        You can ONLY run these commands: cat, grep, tail, head, journalctl, dmesg, ls, message, help.
        You CANNOT restart services or kill processes.
        
        Once you find the root cause, delegate the fix to remediation. Example:
        message remediation "The database password in /etc/app/config is wrong. Please change it and restart the service."
        
        IMPORTANT: Respond with ONLY a single Linux command. No explanations.
    """).strip(),

    "remediation": textwrap.dedent("""
        You are the SRE REMEDIATION Agent.
        Your job is to mutate system state and verify fixes based on Diagnosis's advice.
        
        You can ONLY run these commands: kill, systemctl, edit, curl, message, help.
        
        Once the fix is applied, you can message triage to verify metrics again. Example:
        message triage "I have restarted the service, please check if CPU usage dropped."
        
        IMPORTANT: Respond with ONLY a single Linux command. No explanations.
    """).strip()
}

TASK_PROMPTS = {
    "task1": "TASK: A web service has gone down. Triage should check status, Diagnosis should read /var/log/nginx/error.log, Remediation should restart nginx and curl http://localhost:80/health.",
    "task2": "TASK: There is a memory leak incident. Triage should check 'free', Diagnosis should read /var/log/syslog, Remediation should kill the high memory process and restart data_processor.",
    "task3": "TASK: A cascading failure across load balancer, app server, and db connector. Fix the password in /etc/app/database.yml and restart services in dependency order.",
}


# ---- Structured logging ----

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, role: str, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} role={role.upper()} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---- Command extraction ----

KNOWN_COMMANDS = [
    "cat", "grep", "tail", "head", "ls", "ps", "top",
    "kill", "systemctl", "curl", "df", "free", "netstat",
    "edit", "echo", "help", "journalctl", "dmesg", "message"
]


def extract_command(llm_response: str) -> Optional[str]:
    """Extract a Linux command from LLM response text."""
    text = llm_response.strip()
    text = re.sub(r'```\w*\n?', '', text)
    text = text.strip('`').strip()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        first_word = line.split()[0] if line.split() else ""
        if first_word in KNOWN_COMMANDS:
            return line
    return None


# ---- LLM interaction ----

def get_model_message(
    client: OpenAI,
    messages: list,
    max_retries: int = 3,
) -> str:
    """Get a command from the LLM. Returns 'help' on failure."""
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
            command = extract_command(text)
            if command is not None:
                return command
        except Exception as exc:
            print(f"[DEBUG] Model request failed (attempt {attempt+1}): {exc}", flush=True)
    return "help"


# ---- Main ----

def run_task(client: OpenAI, env_client: SREClient, task_config: dict) -> dict:
    """Run a single task with multiple agents and return results."""
    task_id = task_config["task_id"]
    task_name = task_config["name"]
    max_steps = task_config["max_steps"]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_client.reset(task_id=task_id, seed=SEED)
        
        task_prompt = TASK_PROMPTS.get(task_id, "Find and fix the SRE incident.")
        
        # Initialize thread histories for all three agents
        agent_messages = {}
        for role in ROLES:
            system_prompt = f"{ROLE_PROMPTS[role]}\n\n{task_prompt}"
            agent_messages[role] = [
                {"role": "system", "content": system_prompt}
            ]

        # Triage goes first
        active_agent = "triage"
        agent_messages[active_agent].append({
            "role": "user", 
            "content": f"You are investigating a production incident.\n\nINITIAL OBSERVATION:\n{obs.output}\n\nWhat command do you want to run?"
        })

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            # Get action from active persona
            command = get_model_message(client, agent_messages[active_agent])
            
            # Check for role transition via message command
            target_agent = active_agent
            if command and command.startswith("message "):
                parts = command.split(" ", 2)
                if len(parts) >= 2 and parts[1].lower() in ROLES:
                    target_agent = parts[1].lower()

            # Execute step in environment pretending to be the active agent
            obs = env_client.step(command, agent_role=active_agent)

            reward = obs.reward or 0.0
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, role=active_agent, action=command, reward=reward, done=done, error=error)

            # Record action in sender's history
            agent_messages[active_agent].append({"role": "assistant", "content": command})
            
            if done:
                break

            # Route observation to the correct agent
            if target_agent != active_agent:
                # Transition to new agent
                active_agent = target_agent
                agent_messages[active_agent].append({
                    "role": "user",
                    "content": f"[Transferred to YOU ({active_agent.upper()})] Action Result:\n{obs.output}\n\nWhat command do you want to run next?"
                })
            else:
                # Same agent continues
                agent_messages[active_agent].append({
                    "role": "user",
                    "content": f"[Step {step}] Output:\n{obs.output}\n\nWhat command do you want to run next?",
                })

        # Score is the final grader score from the last observation
        if obs.metadata and "score" in obs.metadata:
            score = obs.metadata["score"]
        elif rewards:
            score = rewards[-1]  # last cumulative reward

        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "task_name": task_name,
        "score": score,
        "steps": steps_taken,
        "success": success,
        "rewards": rewards,
    }


def main() -> None:
    if not API_KEY:
        print("Error: HF_TOKEN or API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = SREClient()

    results = []
    for task_config in TASKS:
        result = run_task(client, env_client, task_config)
        results.append(result)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("Multi-Agent Simulation SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(f"  {r['task_name']}: score={r['score']:.3f} steps={r['steps']} success={r['success']}", flush=True)
    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"  Average: {avg:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
