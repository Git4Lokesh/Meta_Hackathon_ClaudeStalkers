"""Before/After Demo: Baseline vs Trained Heuristic Agents.

Runs untrained (skill_level=0.0) and trained (skill_level=1.0) heuristic
agents side-by-side across all four War Room tasks and prints a comparison
table suitable for copy-pasting into presentation slides.

Usage:
    PYTHONPATH=. python round2/war_room/demo_comparison.py
"""

from __future__ import annotations

import time
from typing import Any

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.grader import MultiAgentGrader

# Graceful import of heuristic dispatch
try:
    from round2.war_room.train import HEURISTIC_DISPATCH
except ImportError:
    HEURISTIC_DISPATCH: dict = {}  # type: ignore[no-redef]

TASK_IDS = ["task1", "task2", "task3", "task4"]


def run_episode(task_id: str, seed: int, skill_level: float) -> dict[str, Any]:
    """Run a single episode with heuristic agents at the given skill level.

    Returns a dict with:
        rounds_used  – number of rounds executed
        milestones   – count of milestones achieved
        score        – final team score (0.0 on error)
        messages_sent – total messages sent during the episode
        resolved     – True if all milestones were achieved
    """
    heuristic_fn = HEURISTIC_DISPATCH.get(task_id)
    if heuristic_fn is None:
        return {
            "rounds_used": 0,
            "milestones": 0,
            "score": 0.0,
            "messages_sent": 0,
            "resolved": False,
        }

    try:
        env = WarRoomEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)
        max_rounds = obs.metadata.get("max_rounds", 10)
        rounds = 0

        for r in range(max_rounds):
            if obs.done:
                break
            rounds += 1
            action = heuristic_fn(r, skill_level)
            obs = env.step(action)

        score = obs.metadata.get("score", obs.team_reward)
        milestones = obs.metadata.get("milestones_achieved", [])
        messages_sent = len(env._channel.get_full_history()) if env._channel else 0

        # Resolved = all milestones achieved
        total_milestones = len(env._grader.milestones) if env._grader else 0
        resolved = len(milestones) == total_milestones and total_milestones > 0

        return {
            "rounds_used": rounds,
            "milestones": len(milestones),
            "score": round(score, 4),
            "messages_sent": messages_sent,
            "resolved": resolved,
        }
    except Exception:
        return {
            "rounds_used": 0,
            "milestones": 0,
            "score": 0.0,
            "messages_sent": 0,
            "resolved": False,
        }


def format_comparison_table(results: list[dict[str, Any]]) -> str:
    """Produce a clean ASCII table comparing baseline vs trained results.

    *results* is a list of dicts, one per task, each containing:
        task_id, baseline (dict), trained (dict)

    Returns a string suitable for copy-pasting into slides.
    """
    header = (
        f"{'Task':<8} | {'Metric':<14} | {'Baseline':>10} | {'Trained':>10} | {'Delta':>10}"
    )
    sep = "-" * len(header)
    lines: list[str] = [
        "",
        "=" * len(header),
        "  MULTI-AGENT WAR ROOM — BASELINE vs TRAINED COMPARISON",
        "=" * len(header),
        "",
        header,
        sep,
    ]

    for entry in results:
        tid = entry["task_id"]
        bl = entry["baseline"]
        tr = entry["trained"]

        if bl["score"] == 0.0 and bl["rounds_used"] == 0:
            # Heuristic dispatch missing — show N/A
            lines.append(f"{tid:<8} | {'N/A':<14} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")
            lines.append(sep)
            continue

        metrics = [
            ("Rounds", bl["rounds_used"], tr["rounds_used"], True),
            ("Milestones", bl["milestones"], tr["milestones"], False),
            ("Score", bl["score"], tr["score"], False),
            ("Messages", bl["messages_sent"], tr["messages_sent"], False),
            ("Resolved", "Yes" if bl["resolved"] else "No",
             "Yes" if tr["resolved"] else "No", None),
        ]

        for i, (label, bv, tv, lower_better) in enumerate(metrics):
            task_col = tid if i == 0 else ""
            if isinstance(bv, float):
                bv_str = f"{bv:.4f}"
                tv_str = f"{tv:.4f}"
                delta = tv - bv
                delta_str = f"{delta:+.4f}"
            elif isinstance(bv, int):
                bv_str = str(bv)
                tv_str = str(tv)
                delta = tv - bv
                sign = "+" if delta > 0 else ""
                delta_str = f"{sign}{delta}"
            else:
                bv_str = str(bv)
                tv_str = str(tv)
                delta_str = "-"

            lines.append(
                f"{task_col:<8} | {label:<14} | {bv_str:>10} | {tv_str:>10} | {delta_str:>10}"
            )
        lines.append(sep)

    return "\n".join(lines)


def main() -> None:
    """Run task1–task4 with seed=42, compare baseline vs trained."""
    seed = 42
    results: list[dict[str, Any]] = []
    per_task_baseline: dict[str, float] = {}
    per_task_trained: dict[str, float] = {}

    print("Running baseline vs trained comparison (seed=42)...\n")
    start = time.time()

    for task_id in TASK_IDS:
        if task_id not in HEURISTIC_DISPATCH:
            print(f"  {task_id}: N/A (no heuristic dispatch)")
            results.append({
                "task_id": task_id,
                "baseline": {"rounds_used": 0, "milestones": 0, "score": 0.0,
                              "messages_sent": 0, "resolved": False},
                "trained": {"rounds_used": 0, "milestones": 0, "score": 0.0,
                             "messages_sent": 0, "resolved": False},
            })
            continue

        bl = run_episode(task_id, seed, skill_level=0.0)
        tr = run_episode(task_id, seed, skill_level=1.0)

        per_task_baseline[task_id] = bl["score"]
        per_task_trained[task_id] = tr["score"]

        results.append({
            "task_id": task_id,
            "baseline": bl,
            "trained": tr,
        })
        print(f"  {task_id}: baseline={bl['score']:.4f}  trained={tr['score']:.4f}")

    elapsed = time.time() - start

    # Print comparison table
    table = format_comparison_table(results)
    print(table)

    # Composite scores
    composite_bl = MultiAgentGrader.composite_score(per_task_baseline)
    composite_tr = MultiAgentGrader.composite_score(per_task_trained)
    delta = composite_tr - composite_bl

    print()
    print(f"  Composite Score (baseline): {composite_bl:.4f}")
    print(f"  Composite Score (trained):  {composite_tr:.4f}")
    print(f"  Improvement:                {delta:+.4f}")
    print()
    print(f"  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
