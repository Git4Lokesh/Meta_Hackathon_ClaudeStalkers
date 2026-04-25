"""Reward ablation runner for Round 2 War Room.

Compares fixed-seed episodes across reward settings:
- full
- milestone_only
- no_comm_bonus
- no_anti_hack
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.grader import MultiAgentGrader
from round2.war_room.train import HEURISTIC_DISPATCH

TASKS = ["task1", "task2", "task3", "task4"]
SEEDS = [7, 42, 99]


def _run_episode(task_id: str, seed: int, config: str) -> dict[str, Any]:
    env = WarRoomEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)

    if env._grader:
        if config == "milestone_only":
            env._grader._max_comm_bonuses = 0
            env._grader.penalties_applied = []
        elif config == "no_comm_bonus":
            env._grader._max_comm_bonuses = 0

    heuristic = HEURISTIC_DISPATCH[task_id]
    rounds = 0
    for r in range(obs.metadata.get("max_rounds", 10)):
        if obs.done:
            break
        rounds += 1
        skill = 1.0
        obs = env.step(heuristic(r, skill))

    score = obs.metadata.get("score", obs.team_reward)
    milestones = obs.metadata.get("milestones_achieved", [])
    if not milestones and env._grader:
        milestones = sorted(env._grader.achieved)

    if config == "milestone_only" and env._grader:
        milestone_sum = sum(m.credit for m in env._grader.milestones if m.name in env._grader.achieved)
        score = max(0.01, min(0.99, milestone_sum))

    return {
        "config": config,
        "task": task_id,
        "seed": seed,
        "score": round(float(score), 4),
        "rounds": rounds,
        "milestones": len(milestones),
        "resolved": bool(obs.done and len(milestones) == len(env._grader.milestones) if env._grader else False),
    }


def run_ablation(output_dir: str) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    configs = ["full", "milestone_only", "no_comm_bonus", "no_anti_hack"]
    for config in configs:
        for task in TASKS:
            for seed in SEEDS:
                rows.append(_run_episode(task, seed, config))

    by_config = {}
    for config in configs:
        subset = [r for r in rows if r["config"] == config]
        avg_score = sum(r["score"] for r in subset) / len(subset)
        avg_rounds = sum(r["rounds"] for r in subset) / len(subset)
        resolved_rate = sum(1 for r in subset if r["resolved"]) / len(subset)
        per_task = {}
        for task in TASKS:
            t_rows = [r for r in subset if r["task"] == task]
            per_task[task] = round(sum(r["score"] for r in t_rows) / len(t_rows), 4)
        by_config[config] = {
            "avg_score": round(avg_score, 4),
            "avg_rounds": round(avg_rounds, 2),
            "resolved_rate": round(resolved_rate, 4),
            "per_task_avg_score": per_task,
        }

    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump({"rows": rows, "summary": by_config}, f, indent=2)

    csv_path = os.path.join(output_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "task", "seed", "score", "rounds", "milestones", "resolved"])
        writer.writeheader()
        writer.writerows(rows)

    return {"rows": rows, "summary": by_config}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reward ablation for War Room")
    parser.add_argument("--output", default="outputs/reward_ablation")
    args = parser.parse_args()
    out = run_ablation(args.output)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
