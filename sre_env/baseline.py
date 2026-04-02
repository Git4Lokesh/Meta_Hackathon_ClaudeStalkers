"""Baseline inference script for the SRE Incident Response environment.

Uses the OpenAI API to run an LLM agent through all three tasks and
report reproducible scores.

Usage:
    python -m sre_env.baseline --seed 42 --model gpt-4o
    
Environment variables:
    OPENAI_API_KEY: Required for authentication
"""

import argparse
import os
import re
import sys
from typing import Optional

from sre_env.client import SREClient
from sre_env.server.models import SREObservation


# System prompt for the LLM
SRE_SYSTEM_PROMPT = """You are an experienced Site Reliability Engineer (SRE) diagnosing and fixing production incidents.

You are connected to a simulated Linux server where services are failing. Your job is to:
1. Investigate the issue using standard Linux commands
2. Identify the root cause
3. Fix the problem
4. Verify the fix

Available commands: cat, grep, tail, head, ls, ps, top, kill, systemctl, curl, df, free, netstat, edit, echo, help

IMPORTANT: Respond with ONLY a single Linux command. No explanations, no markdown, no code blocks. Just the command.

Examples of valid responses:
systemctl status nginx
cat /var/log/nginx/error.log
ps aux
kill -9 1234
systemctl restart nginx
curl http://localhost:80/health
edit /etc/app/config.yml "old_value" "new_value"
"""


def extract_command(llm_response: str) -> Optional[str]:
    """Extract a Linux command from LLM response text.
    
    Returns the command string, or None if no valid command found.
    """
    # Clean up the response
    text = llm_response.strip()
    
    # Remove markdown code blocks if present
    text = re.sub(r'```\w*\n?', '', text)
    text = text.strip('`').strip()
    
    # Take the first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Check if it starts with a known command
        first_word = line.split()[0] if line.split() else ""
        known_commands = [
            "cat", "grep", "tail", "head", "ls", "ps", "top",
            "kill", "systemctl", "curl", "df", "free", "netstat",
            "edit", "echo", "help",
        ]
        if first_word in known_commands:
            return line
    
    return None


def run_task(client: SREClient, task_id: str, seed: int, model: str, max_retries: int = 3) -> dict:
    """Run a single task with the LLM agent."""
    from openai import OpenAI
    
    api_client = OpenAI()
    
    obs = client.reset(task_id=task_id, seed=seed)
    messages = [
        {"role": "system", "content": SRE_SYSTEM_PROMPT},
        {"role": "user", "content": f"You are investigating a production incident.\n\n{obs.output}\n\nWhat command do you want to run?"},
    ]
    
    steps = 0
    while not obs.done:
        # Get LLM response
        command = None
        for attempt in range(max_retries):
            response = api_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,  # deterministic
                max_tokens=200,
            )
            llm_text = response.choices[0].message.content or ""
            command = extract_command(llm_text)
            if command is not None:
                break
        
        if command is None:
            command = "help"
        
        # Execute command
        obs = client.step(command)
        steps += 1
        
        # Update conversation
        messages.append({"role": "assistant", "content": command})
        messages.append({
            "role": "user",
            "content": f"[Step {steps}] Output:\n{obs.output}\n\nWhat command do you want to run next?",
        })
    
    return {
        "task_id": task_id,
        "score": obs.metadata.get("score", 0.0),
        "steps": steps,
        "milestones": obs.metadata.get("milestones_achieved", []),
        "penalties": obs.metadata.get("penalties_applied", []),
    }


def print_summary(results: list[dict]):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)
    print(f"{'Task':<15}{'Score':<10}{'Steps':<10}{'Milestones'}")
    print("-" * 70)
    for r in results:
        milestones = ", ".join(r["milestones"]) if r["milestones"] else "none"
        print(f"{r['task_id']:<15}{r['score']:<10.2f}{r['steps']:<10}{milestones}")
    print("=" * 70)
    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"Average score: {avg:.2f}")


def main():
    parser = argparse.ArgumentParser(description="SRE Incident Response Baseline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    parser.add_argument("--tasks", type=str, default="task1,task2,task3", help="Comma-separated task IDs")
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set it: export OPENAI_API_KEY='your-key-here'", file=sys.stderr)
        sys.exit(1)
    
    client = SREClient()
    task_ids = [t.strip() for t in args.tasks.split(",")]
    
    results = []
    for task_id in task_ids:
        print(f"\nRunning {task_id}...")
        result = run_task(client, task_id, args.seed, args.model)
        results.append(result)
        print(f"  Score: {result['score']:.2f} ({result['steps']} steps)")
    
    print_summary(results)


if __name__ == "__main__":
    main()
