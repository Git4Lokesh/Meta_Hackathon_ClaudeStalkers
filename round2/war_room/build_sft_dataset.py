"""SFT Dataset Builder for the War Room Diagnosis Agent.

Generates (prompt, completion) pairs by running the well-trained heuristic
agents (skill_level=1.0) from train.py against the War Room environment.
Each pair captures:
  - Prompt: what the Diagnosis agent sees at a given round
  - Completion: the correctly-formatted ``COMMAND:/MESSAGE_TO:/MESSAGE:``
    response the heuristic would produce

Purpose: SFT the Qwen base model to produce valid format BEFORE GRPO.
The qwen1.5B_output.md run showed the base model never produces valid
format on its own, giving GRPO zero reward signal. SFT fixes this.

Usage:
    PYTHONPATH=. python round2/war_room/build_sft_dataset.py \\
        --output outputs/sft_dataset.json \\
        --pairs-per-task 30

Output format:
    JSON list of {"prompt": [...], "completion": str, "task_id": str}
    Compatible with HuggingFace datasets and TRL SFTTrainer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import AgentAction, MultiAgentAction, Message
from round2.war_room.train import HEURISTIC_DISPATCH


DIAGNOSIS_SYSTEM_PROMPT = """You are the DIAGNOSIS agent in an SRE incident war room.
You investigate issues by reading logs and inspecting the system.

Your capabilities:
- cat <path>: Read log files
- grep <pattern> <path>: Search in files
- tail [-n N] <path>: Recent log entries
- ps aux: Process table
- top: System overview
- journalctl [-u service]: Journal logs
- dmesg: Kernel messages

IMPORTANT RULES:
- Don't blindly trust metrics from Triage — they may be stale or cached
- Cross-reference alerts with actual log data
- If logs contradict the metrics, push back and say so
- Send specific findings to remediation (PIDs, file paths, exact errors)

Respond in this format:
COMMAND: <your_command>
MESSAGE_TO: <triage|remediation|all|none>
MESSAGE: <your findings>"""


TRIAGE_MESSAGES = {
    "task1": "Message from @Triage: nginx is DOWN. Please check /var/log/nginx/error.log",
    "task2": "Message from @Triage: Multiple alerts — high memory on data_processor AND high CPU on api_gateway. Investigate both.",
    "task3": "Message from @Triage: Redis memory at 72%! Also monitoring CPU spike at 92%. db_connector showing issues.",
    "task4": "Message from @Triage: TWO incidents: nginx crashed AND data_processor memory leak. Investigate both.",
}


def _format_agent_action_as_completion(action: AgentAction) -> str:
    """Convert an AgentAction into the expected COMMAND:/MESSAGE_TO:/MESSAGE: string."""
    command = action.command.strip() if action.command else ""

    if action.message is not None:
        msg_to = action.message.to_agent
        msg_content = action.message.content
    else:
        msg_to = "none"
        msg_content = ""

    lines = [
        f"COMMAND: {command}" if command else "COMMAND:",
        f"MESSAGE_TO: {msg_to}",
        f"MESSAGE: {msg_content}" if msg_content else "MESSAGE:",
    ]
    return "\n".join(lines)


def _build_prompt(diag_observation: str, task_id: str) -> str:
    """Build the user prompt the Diagnosis agent would see."""
    triage_msg = TRIAGE_MESSAGES.get(task_id, "Check the system.")
    return (
        f"{diag_observation}\n\n"
        f"{triage_msg}\n\n"
        f"What command do you want to run? What message do you want to send?"
    )


def build_dataset(
    tasks: list[str] | None = None,
    pairs_per_task: int = 30,
    seed: int = 42,
    skill_level: float = 1.0,
) -> list[dict]:
    """Generate SFT pairs by running optimal heuristic agents.

    For each task, reset the environment at multiple seeds and capture
    the heuristic Diagnosis action at each round.  Only rounds where
    the heuristic produces a non-empty action are kept.
    """
    tasks = tasks or ["task1", "task2", "task3", "task4"]
    rows: list[dict] = []

    for task_id in tasks:
        if task_id not in HEURISTIC_DISPATCH:
            print(f"  ⚠️  {task_id} has no heuristic dispatch — skipping")
            continue

        heuristic_fn = HEURISTIC_DISPATCH[task_id]
        pairs_collected = 0
        seed_idx = 0

        while pairs_collected < pairs_per_task:
            current_seed = seed + seed_idx
            seed_idx += 1

            env = WarRoomEnvironment()
            env._executive_enabled = False  # clean training signal
            try:
                obs = env.reset(task_id=task_id, seed=current_seed)
            except Exception as exc:
                print(f"  ⚠️  reset failed on {task_id} seed={current_seed}: {exc}")
                continue

            max_rounds = obs.metadata.get("max_rounds", 10)

            for r in range(max_rounds):
                if obs.done:
                    break

                multi_action = heuristic_fn(r, skill_level)
                diag_action = multi_action.diagnosis

                # Only include rounds where Diagnosis produces substantive output:
                # prefer pairs with BOTH a command and a message for best SFT signal
                has_command = bool(diag_action.command and diag_action.command.strip())
                has_message = bool(
                    diag_action.message
                    and diag_action.message.content
                    and len(diag_action.message.content.split()) >= 3
                )
                # Accept if we have at least a meaningful command OR a substantive message
                if has_command and has_message:
                    quality = "full"
                elif has_command:
                    quality = "cmd_only"
                elif has_message:
                    quality = "msg_only"
                else:
                    quality = None

                if quality is not None:
                    prompt_text = _build_prompt(obs.diagnosis.text, task_id)
                    completion_text = _format_agent_action_as_completion(diag_action)

                    rows.append({
                        "prompt": [
                            {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt_text},
                        ],
                        "completion": [
                            {"role": "assistant", "content": completion_text},
                        ],
                        "task_id": task_id,
                        "round_num": r,
                        "quality": quality,
                    })
                    pairs_collected += 1

                    if pairs_collected >= pairs_per_task:
                        break

                # Step the environment so the next round's observation is fresh
                obs = env.step(multi_action)

            # Safety: avoid infinite loop if no valid pair found
            if seed_idx > pairs_per_task * 5:
                print(
                    f"  ⚠️  {task_id}: only collected {pairs_collected}/"
                    f"{pairs_per_task} after {seed_idx} seeds — stopping"
                )
                break

        print(f"  ✅ {task_id}: {pairs_collected} pairs")

    random.seed(seed)
    random.shuffle(rows)
    return rows


def save_dataset(rows: list[dict], output_path: str) -> None:
    """Save rows as a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)


def print_dataset_stats(rows: list[dict]) -> None:
    """Print summary stats about the generated dataset."""
    from collections import Counter

    total = len(rows)
    task_counts = Counter(row["task_id"] for row in rows)
    round_counts = Counter(row["round_num"] for row in rows)
    quality_counts = Counter(row.get("quality", "unknown") for row in rows)

    # Sample completion length stats
    completion_lens = [
        len(row["completion"][0]["content"].split())
        for row in rows
    ]
    avg_len = sum(completion_lens) / max(len(completion_lens), 1)

    print()
    print("=" * 50)
    print(f"Dataset stats: {total} pairs total")
    print("=" * 50)
    print("By task:")
    for task_id, count in sorted(task_counts.items()):
        print(f"  {task_id}: {count}")
    print(f"By quality: {dict(quality_counts)}")
    print(f"By round (top 5): {dict(round_counts.most_common(5))}")
    print(f"Avg completion word count: {avg_len:.1f}")
    print()
    print("Sample FULL-quality pair:")
    print("-" * 50)
    full_quality = [r for r in rows if r.get("quality") == "full"]
    sample = full_quality[0] if full_quality else rows[0]
    print("PROMPT (user):", sample["prompt"][1]["content"][:200] + "...")
    print()
    print("COMPLETION:")
    print(sample["completion"][0]["content"])
    print("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SFT warm-up dataset for the War Room Diagnosis agent",
    )
    parser.add_argument(
        "--output",
        default="outputs/sft_dataset.json",
        help="Output JSON path (default: outputs/sft_dataset.json)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["task1", "task2", "task3", "task4"],
        help="Task IDs to generate pairs for",
    )
    parser.add_argument(
        "--pairs-per-task",
        type=int,
        default=30,
        help="Number of pairs per task (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed (default: 42)",
    )
    args = parser.parse_args()

    print("Building SFT dataset for War Room Diagnosis agent...")
    print(f"  Tasks: {args.tasks}")
    print(f"  Pairs per task: {args.pairs_per_task}")
    print(f"  Output: {args.output}")
    print()

    rows = build_dataset(
        tasks=args.tasks,
        pairs_per_task=args.pairs_per_task,
        seed=args.seed,
        skill_level=1.0,
    )

    save_dataset(rows, args.output)
    print_dataset_stats(rows)
    print(f"\n✅ Saved {len(rows)} pairs to {args.output}")


if __name__ == "__main__":
    main()
