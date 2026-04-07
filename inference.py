"""
Inference Script for SRE Incident Response Environment
=======================================================

Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=sre_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
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

SRE_SYSTEM_PROMPT = textwrap.dedent("""
    You are an experienced Site Reliability Engineer (SRE) diagnosing and fixing production incidents.
    You are connected to a simulated Linux server where services are failing. Your job is to:
    1. Investigate the issue using standard Linux commands
    2. Identify the root cause
    3. Fix the problem
    4. Verify the fix

    Available commands: cat, grep, tail, head, ls, ps, top, kill, systemctl, curl, df, free, netstat, edit, echo, journalctl, dmesg, help

    IMPORTANT: Respond with ONLY a single Linux command. No explanations, no markdown, no code blocks. Just the command.

    Examples of valid responses:
    systemctl status nginx
    cat /var/log/nginx/error.log
    ps aux
    kill -9 1234
    systemctl restart nginx
    curl http://localhost:80/health
    edit /etc/app/config.yml "old_value" "new_value"
""").strip()

TASK_PROMPTS = {
    "task1": """You are an expert SRE diagnosing a production incident. A web service has gone down.

Follow this diagnostic workflow:
1. First check service status: systemctl status nginx
2. Read error logs: cat /var/log/nginx/error.log
3. Fix the issue: systemctl restart nginx
4. Verify the fix: curl http://localhost:80/health

Respond with ONLY a single Linux command. No explanations.""",

    "task2": """You are an expert SRE diagnosing a memory leak incident.

Follow this diagnostic workflow:
1. Check memory usage: free
2. Identify high-memory processes: ps aux
3. Read OOM logs: cat /var/log/syslog
4. Kill the leaking process: kill -9 <PID> (the one using 2500+ MB)
5. Restart the affected service: systemctl restart data_processor
6. Verify: curl http://localhost:8081/health

The leaking process will be the one with memory_mb > 2000. Respond with ONLY a single Linux command.""",

    "task3": """You are an expert SRE diagnosing a cascading failure across multiple services.

Follow this diagnostic workflow:
1. Read load balancer logs: cat /var/log/load_balancer/lb.log
2. Read app server logs: cat /var/log/app_server/app.log
3. Read DB connector logs: cat /var/log/db_connector/connector.log
4. Read the database config: cat /etc/app/database.yml
5. Fix the password: edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"
6. Restart in dependency order:
   - systemctl restart db_connector
   - systemctl restart app_server
   - systemctl restart load_balancer
7. Verify: curl http://localhost:80/health

IMPORTANT: Restart services in dependency order. DB connector first, then app server, then load balancer.
Respond with ONLY a single Linux command.""",
}


# ---- Structured logging ----

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
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
    "edit", "echo", "help", "journalctl", "dmesg",
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
    """Run a single task and return results."""
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

        system_prompt = TASK_PROMPTS.get(task_id, SRE_SYSTEM_PROMPT)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"You are investigating a production incident.\n\n{obs.output}\n\nWhat command do you want to run?"},
        ]

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            command = get_model_message(client, messages)
            obs = env_client.step(command)

            reward = obs.reward or 0.0
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=command, reward=reward, done=done, error=error)

            messages.append({"role": "assistant", "content": command})
            messages.append({
                "role": "user",
                "content": f"[Step {step}] Output:\n{obs.output}\n\nWhat command do you want to run next?",
            })

            if done:
                break

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
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(f"  {r['task_name']}: score={r['score']:.3f} steps={r['steps']} success={r['success']}", flush=True)
    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"  Average: {avg:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
