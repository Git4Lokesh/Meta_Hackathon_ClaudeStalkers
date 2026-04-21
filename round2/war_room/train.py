"""TRL GRPO Training Script for the Multi-Agent Incident War Room.

Trains LLM agents to cooperate in incident response using Group Relative
Policy Optimization (GRPO) from HuggingFace TRL.

Usage (Colab):
    !pip install trl unsloth openai
    !python round2/war_room/train.py --model unsloth/Qwen2.5-7B --episodes 100

Environment variables:
    HF_TOKEN: HuggingFace token for model access
"""

import argparse
import os
import json
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


# ---- Training loop (GRPO) ----

def train(
    model_name: str = "unsloth/Qwen2.5-7B",
    num_episodes: int = 100,
    tasks: list[str] = None,
    output_dir: str = "outputs/war_room_training",
):
    """Train agents using GRPO on the War Room environment.

    This function is designed to be called from Colab with compute credits.
    It uses TRL's GRPOTrainer for optimization.
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
        print("WARNING: TRL not installed. Running in demo mode (no actual training).")
        print("Install with: pip install trl unsloth")

    env = WarRoomEnvironment()

    # Curriculum: cycle through tasks with increasing difficulty
    task_curriculum = []
    for epoch in range(num_episodes):
        # Weight harder tasks more as training progresses
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
        # Demo mode: run episodes with random/fixed actions to show the environment works
        print("\n--- Demo Mode: Running episodes without training ---\n")

        for ep, task_id in enumerate(task_curriculum[:10]):  # Just 10 episodes in demo
            obs = env.reset(task_id=task_id, seed=ep)
            rounds = 0

            for r in range(obs.metadata["max_rounds"]):
                if obs.done:
                    break
                rounds += 1

                # Simple heuristic agent (not trained)
                action = _heuristic_action(task_id, r, obs)
                obs = env.step(action)

            score = obs.metadata.get("score", obs.team_reward)
            milestones = obs.metadata.get("milestones_achieved", [])

            metrics["episode"].append(ep)
            metrics["task"].append(task_id)
            metrics["team_reward"].append(score)
            metrics["rounds_used"].append(rounds)
            metrics["milestones_achieved"].append(len(milestones))

            print(f"  Episode {ep}: task={task_id} score={score:.3f} rounds={rounds} milestones={len(milestones)}")

        print("\n--- Demo complete. Install TRL for actual training. ---")
    else:
        # Real GRPO training
        print("\n--- Starting GRPO Training ---\n")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=1e-5,
            logging_steps=1,
        )

        # Define reward function for GRPO
        def reward_fn(completions, **kwargs):
            """Compute rewards for a batch of completions."""
            rewards = []
            for completion in completions:
                # Parse completion into agent responses
                # This is simplified — real implementation would need proper parsing
                reward = 0.5  # placeholder
                rewards.append(reward)
            return rewards

        trainer = GRPOTrainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
            reward_funcs=[reward_fn],
        )

        trainer.train()
        trainer.save_model(output_dir)
        print(f"\nModel saved to {output_dir}")

    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


def _heuristic_action(task_id: str, round_num: int, obs) -> MultiAgentAction:
    """Simple heuristic agent for demo mode."""
    # Task 1: coordinated restart
    if task_id == "task1":
        steps = [
            # Round 0: triage checks dashboard, sends message
            {"triage": "get_dashboard", "diag": "", "remed": "",
             "msg_from": "triage", "msg_to": "diagnosis", "msg": "nginx is down, please investigate"},
            # Round 1: diagnosis reads logs, sends findings
            {"triage": "", "diag": "cat /var/log/nginx/error.log", "remed": "",
             "msg_from": "diagnosis", "msg_to": "remediation", "msg": "nginx crashed with signal 11, needs restart"},
            # Round 2: remediation restarts
            {"triage": "", "diag": "", "remed": "systemctl restart nginx"},
            # Round 3: verify
            {"triage": "", "diag": "", "remed": "curl http://localhost:80/health"},
        ]
    elif task_id == "task2":
        steps = [
            {"triage": "get_dashboard", "diag": "", "remed": "",
             "msg_from": "triage", "msg_to": "diagnosis", "msg": "high memory alert, possible OOM"},
            {"triage": "", "diag": "ps aux", "remed": ""},
            {"triage": "", "diag": "cat /var/log/syslog", "remed": "",
             "msg_from": "diagnosis", "msg_to": "remediation", "msg": "data_processor_worker PID 1000 leaking memory, kill it"},
            {"triage": "", "diag": "", "remed": "kill -9 1000"},
            {"triage": "", "diag": "", "remed": "systemctl restart data_processor"},
            {"triage": "", "diag": "", "remed": "curl http://localhost:8081/health"},
        ]
    else:
        steps = [{"triage": "get_dashboard", "diag": "ps aux", "remed": ""}]

    if round_num >= len(steps):
        return MultiAgentAction()

    s = steps[round_num]

    msg = None
    if "msg_from" in s and s.get("msg"):
        msg = Message(
            from_agent=s["msg_from"],
            to_agent=s["msg_to"],
            content=s["msg"],
            timestamp=datetime.now(),
            round_number=round_num,
        )

    triage_action = AgentAction(command=s.get("triage", ""))
    diag_action = AgentAction(command=s.get("diag", ""))
    remed_action = AgentAction(command=s.get("remed", ""))

    # Attach message to the sending agent
    if msg:
        if msg.from_agent == "triage":
            triage_action = AgentAction(command=s.get("triage", ""), message=msg)
        elif msg.from_agent == "diagnosis":
            diag_action = AgentAction(command=s.get("diag", ""), message=msg)
        elif msg.from_agent == "remediation":
            remed_action = AgentAction(command=s.get("remed", ""), message=msg)

    return MultiAgentAction(
        triage=triage_action,
        diagnosis=diag_action,
        remediation=remed_action,
    )


def main():
    parser = argparse.ArgumentParser(description="Train War Room agents with GRPO")
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B", help="Model name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--output", default="outputs/war_room_training", help="Output directory")
    args = parser.parse_args()

    train(model_name=args.model, num_episodes=args.episodes, output_dir=args.output)


if __name__ == "__main__":
    main()
