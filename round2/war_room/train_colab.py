"""
Colab-Ready GRPO Training Script for Multi-Agent War Room
==========================================================

Copy-paste this into a Colab cell with GPU runtime (A100 recommended).

This trains a single Qwen2.5-7B model to play the DIAGNOSIS agent role,
using the War Room environment as the reward function. Heuristic agents
handle Triage and Remediation roles during training.

Usage in Colab:
    # Cell 1: Setup
    !git clone https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git
    %cd Meta_Hackathon_ClaudeStalkers
    !pip install -q trl>=0.15.0 peft>=0.14.0 transformers>=4.46.0 datasets accelerate
    !pip install -q unsloth
    !pip install -q fastapi pydantic uvicorn openai matplotlib
    !pip install -e . --quiet

    # Cell 2: Train
    !PYTHONPATH=. python round2/war_room/train_colab.py --episodes 30 --task task1

    # Cell 3: Visualize
    !PYTHONPATH=. python round2/war_room/visualize.py
"""

import argparse
import json
import os
import sys
import random
from datetime import datetime
from typing import Optional

# ---- Environment imports ----
from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import MultiAgentAction, AgentAction, Message


# ============================================================
# REWARD FUNCTION: Runs completion through the War Room environment
# ============================================================

# Global environment instance for reward computation
_reward_env = WarRoomEnvironment()
_reward_task_id = "task1"
_reward_seed = 42


def _build_heuristic_triage(round_num: int, task_id: str) -> AgentAction:
    """Heuristic Triage agent — checks dashboard and escalates."""
    if round_num == 0:
        # First round: always check dashboard and escalate
        msg_content = {
            "task1": "nginx is DOWN. Please check /var/log/nginx/error.log",
            "task2": "Multiple alerts: high memory on data_processor, high CPU on api_gateway. Investigate both.",
            "task3": "Multiple alerts: Redis memory warning, monitoring CPU spike, and db_connector issues.",
            "task4": "TWO incidents: nginx crashed AND data_processor memory leak.",
        }.get(task_id, "Check the system for failing services.")

        return AgentAction(
            command="get_dashboard",
            message=Message(
                from_agent="triage", to_agent="diagnosis",
                content=msg_content,
                timestamp=datetime.now(), round_number=round_num,
            ),
        )
    return AgentAction(command="")


def _build_heuristic_remediation(round_num: int, task_id: str, diagnosis_msg: str) -> AgentAction:
    """Heuristic Remediation agent — follows Diagnosis instructions."""
    msg_lower = diagnosis_msg.lower()

    # React to diagnosis messages
    if "restart" in msg_lower and "nginx" in msg_lower:
        return AgentAction(command="systemctl restart nginx")
    if "kill" in msg_lower:
        # Extract PID from message
        import re
        pid_match = re.search(r'pid\s*(\d+)', msg_lower)
        if pid_match:
            return AgentAction(command=f"kill -9 {pid_match.group(1)}")
    if "edit" in msg_lower and "password" in msg_lower:
        return AgentAction(
            command='edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"'
        )
    if "restart" in msg_lower and "db_connector" in msg_lower:
        return AgentAction(command="systemctl restart db_connector")
    if "restart" in msg_lower and "app_server" in msg_lower:
        return AgentAction(command="systemctl restart app_server")
    if "restart" in msg_lower and "load_balancer" in msg_lower:
        return AgentAction(command="systemctl restart load_balancer")
    if "restart" in msg_lower and "data_processor" in msg_lower:
        return AgentAction(command="systemctl restart data_processor")
    if "curl" in msg_lower or "verify" in msg_lower:
        return AgentAction(command="curl http://localhost:80/health")

    return AgentAction(command="")


def _parse_diagnosis_completion(text: str, round_num: int) -> AgentAction:
    """Parse a model completion into a Diagnosis AgentAction."""
    text = text.strip()
    # Remove markdown
    import re
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

    # Fallback
    if not command:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.upper().startswith("MESSAGE"):
                command = line
                break

    message = None
    if msg_to and msg_to != "none" and msg_content:
        message = Message(
            from_agent="diagnosis", to_agent=msg_to,
            content=msg_content,
            timestamp=datetime.now(), round_number=round_num,
        )

    return AgentAction(command=command, message=message)


def war_room_reward(completions, task_id=None, observation=None, round_num=None, **kwargs):
    """
    GRPO reward function: runs each diagnosis completion through the
    War Room environment and returns the team reward.

    Args:
        completions: list of lists of dicts, e.g. [[{'content': '...'}], ...]
        task_id: which task to evaluate on (passed via dataset column)
        observation: the observation text the model saw
        round_num: which round this is

    Returns:
        list of float rewards
    """
    rewards = []
    tid = task_id[0] if isinstance(task_id, list) else (task_id or _reward_task_id)

    for i, completion in enumerate(completions):
        try:
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)

            # Reset environment for this evaluation
            env = WarRoomEnvironment()
            obs = env.reset(task_id=tid, seed=_reward_seed + i)

            max_rounds = obs.metadata["max_rounds"]
            last_diag_msg = ""

            # Run a full episode with the model's response in round 1
            # and heuristic follow-up
            for r in range(min(max_rounds, 8)):  # Cap at 8 rounds for speed
                if obs.done:
                    break

                rn = r  # round number for messages

                # Build diagnosis action from model completion (round 0)
                # or from a simple follow-up heuristic (later rounds)
                if r == 0:
                    diag_action = _parse_diagnosis_completion(text, rn)
                else:
                    # Simple follow-up: read additional logs if first round
                    # didn't find the answer
                    follow_up_cmds = {
                        "task1": ["cat /var/log/nginx/error.log", ""],
                        "task2": ["ps aux", "cat /var/log/syslog"],
                        "task3": [
                            "cat /var/log/db_connector/connector.log",
                            "cat /var/log/redis/redis.log",
                        ],
                        "task4": ["cat /var/log/nginx/error.log", "ps aux"],
                    }
                    cmds = follow_up_cmds.get(tid, [""])
                    cmd = cmds[r - 1] if r - 1 < len(cmds) else ""
                    diag_action = AgentAction(command=cmd)

                # Extract any message diagnosis sent
                if diag_action.message and diag_action.message.content:
                    last_diag_msg = diag_action.message.content

                triage_action = _build_heuristic_triage(r, tid)
                remed_action = _build_heuristic_remediation(r, tid, last_diag_msg)

                action = MultiAgentAction(
                    triage=triage_action,
                    diagnosis=diag_action,
                    remediation=remed_action,
                )
                obs = env.step(action)

            score = obs.metadata.get("score", obs.team_reward)
            rewards.append(float(max(0.0, min(1.0, score))))

        except Exception as e:
            print(f"[REWARD] Error computing reward: {e}", flush=True)
            rewards.append(0.0)

    return rewards


# ============================================================
# DATASET: Generate training prompts from War Room observations
# ============================================================

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


def generate_training_dataset(tasks=None, prompts_per_task=10, seed=42):
    """Generate training prompts by resetting the environment for each task."""
    tasks = tasks or ["task1", "task2", "task3"]
    env = WarRoomEnvironment()
    dataset_rows = []

    for task_id in tasks:
        for i in range(prompts_per_task):
            obs = env.reset(task_id=task_id, seed=seed + i)

            # Build the prompt the diagnosis agent would see
            diag_obs = obs.diagnosis.text

            # Add triage escalation message
            triage_msgs = {
                "task1": "Message from @Triage: nginx is DOWN. Please check /var/log/nginx/error.log",
                "task2": "Message from @Triage: Multiple alerts — high memory on data_processor AND high CPU on api_gateway. Investigate both.",
                "task3": "Message from @Triage: Redis memory at 72% looks critical! Also monitoring CPU spike at 92%. And db_connector showing some issues.",
                "task4": "Message from @Triage: TWO incidents at once: nginx crashed AND data_processor memory leak. Investigate both.",
            }

            prompt_text = (
                f"{diag_obs}\n\n"
                f"{triage_msgs.get(task_id, 'Check the system.')}\n\n"
                f"What command do you want to run? What message do you want to send?"
            )

            dataset_rows.append({
                "prompt": [
                    {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "task_id": task_id,
                "round_num": 0,
            })

    random.seed(seed)
    random.shuffle(dataset_rows)
    return dataset_rows


# ============================================================
# TRAINING LOOP
# ============================================================

def train_grpo(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_episodes: int = 30,
    tasks: list[str] = None,
    output_dir: str = "outputs/war_room_grpo",
    use_unsloth: bool = True,
    lora_r: int = 16,
    learning_rate: float = 5e-6,
    batch_size: int = 1,
    num_generations: int = 4,  # G in GRPO — completions per prompt
):
    """Train the Diagnosis agent using GRPO with the War Room as reward."""

    tasks = tasks or ["task1", "task2", "task3"]
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MULTI-AGENT WAR ROOM — GRPO TRAINING")
    print("=" * 60)
    print(f"  Model:       {model_name}")
    print(f"  Tasks:       {tasks}")
    print(f"  Episodes:    {num_episodes}")
    print(f"  LoRA rank:   {lora_r}")
    print(f"  LR:          {learning_rate}")
    print(f"  Generations: {num_generations}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    # ---- Step 1: Load model ----
    print("\n[1/4] Loading model...")

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=None,  # auto-detect
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_r,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            print(f"  ✅ Loaded with Unsloth (4-bit)")
        except ImportError:
            print("  ⚠️  Unsloth not available, falling back to transformers")
            use_unsloth = False

    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        print(f"  ✅ Loaded with transformers + LoRA")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Step 2: Build dataset ----
    print("\n[2/4] Generating training prompts...")

    dataset_rows = generate_training_dataset(
        tasks=tasks,
        prompts_per_task=num_episodes,
    )

    from datasets import Dataset
    train_dataset = Dataset.from_list(dataset_rows)
    print(f"  ✅ {len(train_dataset)} training prompts generated")

    # ---- Step 3: Configure GRPO ----
    print("\n[3/4] Setting up GRPO trainer...")

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_completion_length=256,
        max_prompt_length=1536,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to="none",  # or "wandb" if available
        bf16=True,
        gradient_accumulation_steps=4,
        seed=42,
        # GRPO-specific
        temperature=0.7,
        log_completions=True,
    )

    # TRL >= 0.16 uses processing_class; older versions use tokenizer
    try:
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            reward_funcs=war_room_reward,
            train_dataset=train_dataset,
        )
    except TypeError:
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            reward_funcs=war_room_reward,
            train_dataset=train_dataset,
        )

    print(f"  ✅ GRPOTrainer configured")

    # ---- Step 4: Train! ----
    print("\n[4/4] Starting GRPO training...")
    print("  This will take ~30-60 minutes on A100.\n")

    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n  ✅ Model saved to {output_dir}")

    # ---- Step 5: Extract and save metrics ----
    print("\n[5/5] Saving metrics...")

    log_history = trainer.state.log_history
    metrics = {
        "episode": list(range(len(log_history))),
        "task": [tasks[i % len(tasks)] for i in range(len(log_history))],
        "team_reward": [
            h.get("reward", h.get("rewards/war_room_reward", 0.0))
            for h in log_history
        ],
        "rounds_used": [5] * len(log_history),  # approximate
        "milestones_achieved": [
            int(h.get("reward", 0) * 9)  # approximate milestones from reward
            for h in log_history
        ],
        "loss": [h.get("loss", 0.0) for h in log_history],
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ Metrics saved to {metrics_path}")

    # Generate charts
    try:
        from round2.war_room.visualize import plot_matplotlib
        plot_matplotlib(metrics, output_dir)
    except Exception as e:
        print(f"  ⚠️  Could not generate charts: {e}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Run inference:  PYTHONPATH=. python round2/war_room/inference.py --tasks task1 task3")
    print(f"  2. Push to HF Hub: huggingface-cli upload {output_dir}")
    print(f"  3. Visualize:      PYTHONPATH=. python round2/war_room/visualize.py --metrics {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for War Room (Colab-ready)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--episodes", type=int, default=30, help="Prompts per task")
    parser.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"])
    parser.add_argument("--output", default="outputs/war_room_grpo", help="Output dir")
    parser.add_argument("--no-unsloth", action="store_true", help="Disable Unsloth")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--generations", type=int, default=4, help="GRPO completions per prompt")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    train_grpo(
        model_name=args.model,
        num_episodes=args.episodes,
        tasks=args.tasks,
        output_dir=args.output,
        use_unsloth=not args.no_unsloth,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        num_generations=args.generations,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
