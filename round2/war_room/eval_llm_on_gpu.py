"""On-GPU head-to-head eval: base Qwen 7B vs base + our GRPO LoRA adapter.

Runs entirely inside an HF Job container with a single GPU (L40S by
default). Uses ``transformers.generate`` directly — no OpenAI shim
needed because the model is local.

For each task × seed:
  1. Build a fresh WarRoomEnvironment
  2. Each round, call the loaded model once per role (triage / diagnosis /
     remediation) with role-specific system prompts
  3. Parse the model's response into an AgentAction
  4. Step the environment, collect metrics

We run ``len(SEEDS) × len(TASKS)`` rollouts with the base model, then
apply the LoRA adapter via ``PeftModel`` and run the same rollouts again.

Outputs (written to --output-dir, default outputs/llm_eval/):
  results.json      — per-rollout rows ({model, task, seed, score, ...})
  summary.json      — aggregated per-(model, task) plus overall composite
  head_to_head.png  — grouped bar chart, base vs trained per task

Usage (inside HF Jobs):
  python round2/war_room/eval_llm_on_gpu.py \\
      --seeds 11 22 33 44 55 \\
      --tasks task1 task2 task3 \\
      --output-dir outputs/llm_eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any

# Delay heavy imports until after arg parsing so --help is fast.


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_REPO = "brodie1of1/war-room-grpo-adapter"

DEFAULT_TASKS = ["task1", "task2", "task3"]
DEFAULT_SEEDS = [11, 22, 33, 44, 55]


def _role_system_prompt(role: str) -> str:
    # Short, identical-ish to what LiveAgentRunner uses. Kept here so this
    # script has no runtime dependency on LiveAgentRunner (avoids dragging
    # the OpenAI client into the container).
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
    """Turn the model's text into an AgentAction."""
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
        # Fallback: first non-empty non-MESSAGE line is the command.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.upper().startswith("MESSAGE"):
                command = stripped
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


def _generate(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int = 160) -> str:
    """Greedy generation via the chat template."""
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
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return text


def _run_one_episode(model, tokenizer, task_id: str, seed: int) -> dict[str, Any]:
    """Run a single episode of the given task with the given model."""
    from round2.war_room.environment import WarRoomEnvironment
    from round2.war_room.models import MultiAgentAction

    env = WarRoomEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    max_rounds = obs.metadata.get("max_rounds", 10)

    rounds = 0
    for r in range(max_rounds):
        if obs.done:
            break
        rounds += 1

        # Observe → generate → parse for each role.
        actions: dict[str, Any] = {}
        for role, role_obs in (
            ("triage", obs.triage.text),
            ("diagnosis", obs.diagnosis.text),
            ("remediation", obs.remediation.text),
        ):
            try:
                raw = _generate(
                    model, tokenizer,
                    system_prompt=_role_system_prompt(role),
                    user_prompt=f"[Round {r + 1}]\n{role_obs}\n\nWhat do you do?",
                )
            except Exception as exc:
                return {
                    "task": task_id, "seed": seed,
                    "score": 0.0, "rounds": rounds,
                    "milestones": 0, "pushbacks": 0,
                    "resolved": False, "error": str(exc)[:300],
                }
            actions[role] = _parse_response(raw, role, r + 1)
        obs = env.step(MultiAgentAction(**actions))

    score = float(obs.metadata.get("score", obs.team_reward))
    milestones = obs.metadata.get("milestones_achieved") or []
    tracker = getattr(env, "_belief_tracker", None)
    pushbacks = 0
    if tracker is not None:
        try:
            pushbacks = len(tracker.get_snapshot().get("tom_events", []))
        except Exception:
            pass
    resolved = (
        bool(obs.done)
        and env._grader is not None
        and len(milestones) == len(env._grader.milestones)
    )
    return {
        "task": task_id, "seed": seed,
        "score": round(score, 4),
        "rounds": rounds,
        "milestones": len(milestones),
        "pushbacks": pushbacks,
        "resolved": resolved,
        "error": None,
    }


def _run_model(
    label: str,
    model,
    tokenizer,
    tasks: list[str],
    seeds: list[int],
    model_name: str,
) -> list[dict[str, Any]]:
    print(f"\n=== {label} ===")
    rows: list[dict[str, Any]] = []
    for task_id in tasks:
        for seed in seeds:
            t0 = time.time()
            row = _run_one_episode(model, tokenizer, task_id, seed)
            row["model"] = model_name
            elapsed = time.time() - t0
            status = "ERR" if row.get("error") else f"{row['score']:.2f}"
            print(
                f"  {task_id} seed={seed}: score={status} "
                f"rounds={row['rounds']} push={row['pushbacks']} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
            rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[(r["model"], r["task"])].append(r)

    per_model_task: dict[str, dict[str, Any]] = defaultdict(dict)
    for (model, task), group in by_key.items():
        scores = [g["score"] for g in group if g["error"] is None]
        roundsu = [g["rounds"] for g in group if g["error"] is None]
        pushes = [g["pushbacks"] for g in group if g["error"] is None]
        resolved = [g["resolved"] for g in group if g["error"] is None]
        n = max(len(scores), 1)
        per_model_task[model][task] = {
            "avg_score": round(mean(scores) if scores else 0.0, 4),
            "avg_rounds": round(mean(roundsu) if roundsu else 0.0, 2),
            "avg_pushbacks": round(mean(pushes) if pushes else 0.0, 2),
            "resolved_rate": round(sum(resolved) / n, 3),
            "n_seeds": n,
        }

    overall: dict[str, Any] = {}
    for model in per_model_task:
        ms = [per_model_task[model][t]["avg_score"] for t in per_model_task[model]]
        overall[model] = round(mean(ms) if ms else 0.0, 4)
    delta = None
    if "base" in overall and "trained" in overall:
        delta = round(overall["trained"] - overall["base"], 4)
    return {
        "per_model_task": dict(per_model_task),
        "overall_composite": overall,
        "delta_composite": delta,
    }


def _render_plot(summary: dict[str, Any], tasks: list[str], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    per = summary["per_model_task"]
    models = [m for m in ["base", "trained"] if m in per]
    model_colors = {"base": "#8b949e", "trained": "#3fb950"}
    model_labels = {"base": "Base Qwen 7B", "trained": "Base + GRPO adapter"}

    if not models:
        ax.text(0.5, 0.5, "No results", color="#8b949e", ha="center", va="center")
        plt.savefig(path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    n_models = len(models)
    group_width = 0.7
    bar_w = group_width / n_models
    x_positions = list(range(len(tasks)))
    for i, m in enumerate(models):
        heights = [per[m].get(t, {}).get("avg_score", 0.0) for t in tasks]
        offsets = [x + (i - (n_models - 1) / 2) * bar_w for x in x_positions]
        ax.bar(offsets, heights, bar_w,
               label=model_labels[m], color=model_colors[m],
               edgecolor="#0d1117", linewidth=0.8)
        for xi, h in zip(offsets, heights):
            ax.text(xi, h + 0.02, f"{h:.2f}",
                    color="#c9d1d9", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tasks, color="#c9d1d9")
    ax.set_ylabel("Average team score (0–1)", color="#8b949e")
    ax.set_title(
        "Head-to-head: Base Qwen 7B vs Base + GRPO adapter "
        f"({len(tasks)} tasks × {summary['per_model_task'][models[0]][tasks[0]]['n_seeds']} seeds)",
        color="#c9d1d9", fontweight="bold", pad=12,
    )
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#484f58")
    ax.spines["bottom"].set_color("#21262d")
    ax.spines["left"].set_color("#21262d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.12, color="#30363d", axis="y")
    legend = ax.legend(
        facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
        loc="upper right",
    )
    for text in legend.get_texts():
        text.set_color("#c9d1d9")

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="On-GPU base vs trained eval")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", default="outputs/llm_eval")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter-repo", default=ADAPTER_REPO)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-trained", action="store_true")
    args = parser.parse_args()

    print(f"tasks: {args.tasks}")
    print(f"seeds: {args.seeds}")
    print(f"output: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Lazy imports so --help stays fast.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading tokenizer and base model {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    all_rows: list[dict[str, Any]] = []

    if not args.skip_base:
        print(f"\n[BASE] running {len(args.tasks) * len(args.seeds)} rollouts ...")
        all_rows.extend(_run_model(
            label=f"BASE ({args.base_model})",
            model=model, tokenizer=tokenizer,
            tasks=args.tasks, seeds=args.seeds,
            model_name="base",
        ))

    if not args.skip_trained:
        print(f"\n[TRAINED] loading LoRA adapter {args.adapter_repo} ...")
        from peft import PeftModel
        # Important: wrap the *base model* (same reference) with the adapter.
        model = PeftModel.from_pretrained(model, args.adapter_repo)
        model.eval()
        print(f"[TRAINED] running {len(args.tasks) * len(args.seeds)} rollouts ...")
        all_rows.extend(_run_model(
            label=f"TRAINED ({args.adapter_repo})",
            model=model, tokenizer=tokenizer,
            tasks=args.tasks, seeds=args.seeds,
            model_name="trained",
        ))

    # Aggregate + write outputs
    summary = _aggregate(all_rows)
    results_path = os.path.join(args.output_dir, "results.json")
    summary_path = os.path.join(args.output_dir, "summary.json")
    plot_path = os.path.join(args.output_dir, "head_to_head.png")

    with open(results_path, "w") as f:
        json.dump({"rows": all_rows, "tasks": args.tasks, "seeds": args.seeds}, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _render_plot(summary, args.tasks, plot_path)

    print()
    print("=" * 60)
    print("  HEAD-TO-HEAD SUMMARY")
    print("=" * 60)
    for m, s in summary["overall_composite"].items():
        print(f"  {m:<10s} composite: {s:.3f}")
    if summary["delta_composite"] is not None:
        print(f"  delta     (trained − base): {summary['delta_composite']:+.3f}")
    print(f"\n  rows    → {results_path}")
    print(f"  summary → {summary_path}")
    print(f"  chart   → {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
