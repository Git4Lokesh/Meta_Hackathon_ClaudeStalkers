"""Capture full rollout text (prompts + completions + actions + observations)
for a specific (task, seed) pair using base Qwen and base+trained. Used to
generate a 'worked example' for the blog and README.

Usage:
  python round2/war_room/capture_rollout.py \\
      --task task2 --seed 33 \\
      --adapter-repo brodie1of1/war-room-grpo-adapter-v3 \\
      --output-dir outputs/worked_example
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _role_system_prompt(role: str) -> str:
    rules = {
        "triage": (
            "You are the TRIAGE agent. You see the dashboard. "
            "Do NOT forward panicked executive messages. "
            "Pick the ONE real issue. "
        ),
        "diagnosis": (
            "You are the DIAGNOSIS agent. You read logs. "
            "If logs contradict Triage's metrics, push back explicitly. "
            "Send findings with exact PID, file path, error line. "
        ),
        "remediation": (
            "You are the REMEDIATION agent. You fix things. "
            "NEVER touch a service Diagnosis did not mention. "
            "NEVER kill a healthy or already-crashed service. "
            "After a restart, curl the health endpoint to verify. "
        ),
    }
    return (
        f"{rules[role]}\n"
        "RESPOND WITH EXACTLY THREE LINES in this format:\n"
        "COMMAND: <your_command>\n"
        "MESSAGE_TO: <triage|diagnosis|remediation|all|none>\n"
        "MESSAGE: <your message or empty>"
    )


def _parse_response(text: str, role: str, round_num: int):
    from round2.war_room.models import AgentAction, Message

    text = (text or "").strip()
    text = re.sub(r"```\w*\n?", "", text).strip("`").strip()

    command = ""
    msg_to = ""
    msg_content = ""
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("COMMAND:"):
            command = stripped.split(":", 1)[1].strip()
        elif upper.startswith("MESSAGE_TO:"):
            msg_to = stripped.split(":", 1)[1].strip().lower()
        elif upper.startswith("MESSAGE:"):
            msg_content = stripped.split(":", 1)[1].strip()

    if not command:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.upper().startswith("MESSAGE"):
                command = stripped
                break

    message = None
    if msg_to and msg_to != "none" and msg_content:
        message = Message(
            from_agent=role, to_agent=msg_to,
            content=msg_content, timestamp=datetime.now(),
            round_number=round_num,
        )
    return AgentAction(command=command, message=message)


def _generate(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int = 160) -> str:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return text


def capture(model, tokenizer, task_id: str, seed: int, label: str) -> dict[str, Any]:
    from round2.war_room.environment import WarRoomEnvironment
    from round2.war_room.models import MultiAgentAction

    env = WarRoomEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    max_rounds = obs.metadata.get("max_rounds", 10)

    rounds_log: list[dict[str, Any]] = []

    for r in range(max_rounds):
        if obs.done:
            break
        round_data = {"round": r + 1, "roles": {}}

        actions: dict[str, Any] = {}
        for role, role_obs in (
            ("triage", obs.triage.text),
            ("diagnosis", obs.diagnosis.text),
            ("remediation", obs.remediation.text),
        ):
            raw = _generate(
                model, tokenizer,
                system_prompt=_role_system_prompt(role),
                user_prompt=f"[Round {r + 1}]\n{role_obs}\n\nWhat do you do?",
            )
            action = _parse_response(raw, role, r + 1)
            round_data["roles"][role] = {
                "observation": role_obs[:400],
                "raw_completion": raw,
                "parsed_command": action.command,
                "parsed_message": (
                    action.message.content if action.message else None
                ),
            }
            actions[role] = action
        obs = env.step(MultiAgentAction(**actions))
        round_data["score_so_far"] = env._grader.current_score() if env._grader else None
        round_data["milestones_hit"] = sorted(env._grader.achieved) if env._grader else []
        rounds_log.append(round_data)

    return {
        "label": label,
        "task": task_id,
        "seed": seed,
        "final_score": float(obs.metadata.get("score", obs.team_reward)),
        "total_rounds": env._round_number,
        "milestones": sorted(env._grader.achieved) if env._grader else [],
        "rounds": rounds_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="task2")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--output-dir", default="outputs/worked_example")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    print(f"\nCapturing BASE rollout on {args.task} seed={args.seed}...")
    base_trace = capture(model, tokenizer, args.task, args.seed, "base")
    print(f"  final_score={base_trace['final_score']}  milestones={len(base_trace['milestones'])}")

    print(f"\nLoading LoRA adapter {args.adapter_repo}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter_repo)
    model.eval()

    print(f"\nCapturing TRAINED rollout on {args.task} seed={args.seed}...")
    trained_trace = capture(model, tokenizer, args.task, args.seed, "trained")
    print(f"  final_score={trained_trace['final_score']}  milestones={len(trained_trace['milestones'])}")

    output = {
        "base": base_trace,
        "trained": trained_trace,
    }
    out_path = os.path.join(args.output_dir, f"{args.task}_seed{args.seed}_rollout.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved trace to {out_path}")

    print("\n=== Summary ===")
    print(f"BASE    : {base_trace['final_score']:.2f}  ({len(base_trace['milestones'])} milestones)")
    print(f"TRAINED : {trained_trace['final_score']:.2f}  ({len(trained_trace['milestones'])} milestones)")
    print(f"Delta   : {trained_trace['final_score'] - base_trace['final_score']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
