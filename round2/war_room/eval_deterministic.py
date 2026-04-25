"""Deterministic base-vs-trained evaluation helper.

This script evaluates fixed-seed episodes using the heuristic baseline proxy
for reproducible comparisons required in hackathon demos.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.grader import MultiAgentGrader
from round2.war_room.train import HEURISTIC_DISPATCH

TASKS = ["task1", "task2", "task3", "task4"]
SEEDS = [11, 22, 33, 44, 55]


def eval_skill(skill_level: float) -> dict[str, Any]:
    per_task_scores = {}
    rows = []
    for task in TASKS:
        task_scores = []
        for seed in SEEDS:
            env = WarRoomEnvironment()
            obs = env.reset(task_id=task, seed=seed)
            heuristic = HEURISTIC_DISPATCH[task]
            for r in range(obs.metadata.get("max_rounds", 10)):
                if obs.done:
                    break
                obs = env.step(heuristic(r, skill_level))
            score = float(obs.metadata.get("score", obs.team_reward))
            task_scores.append(score)
            rows.append({"task": task, "seed": seed, "score": round(score, 4), "skill": skill_level})
        per_task_scores[task] = sum(task_scores) / len(task_scores)
    composite = MultiAgentGrader.composite_score(per_task_scores)
    return {"rows": rows, "per_task_scores": per_task_scores, "composite": composite}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic evaluation for War Room")
    parser.add_argument("--output", default="outputs/war_room_eval")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    baseline = eval_skill(0.0)
    trained = eval_skill(1.0)
    payload = {
        "seeds": SEEDS,
        "baseline": baseline,
        "trained": trained,
        "delta_composite": trained["composite"] - baseline["composite"],
    }
    out_path = os.path.join(args.output, "deterministic_eval.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(
        {
            "baseline_composite": round(baseline["composite"], 4),
            "trained_composite": round(trained["composite"], 4),
            "delta": round(payload["delta_composite"], 4),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
