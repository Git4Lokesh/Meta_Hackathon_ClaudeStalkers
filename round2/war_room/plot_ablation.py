"""Generate ablation comparison chart from ablation_results.json.

Produces a grouped bar chart per task plus an overall avg-score bar chart,
saved to the same output directory.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CONFIG_ORDER = ["full", "milestone_only", "no_comm_bonus", "no_anti_hack"]
CONFIG_COLORS = {
    "full": "#22c55e",
    "milestone_only": "#f59e0b",
    "no_comm_bonus": "#ef4444",
    "no_anti_hack": "#3b82f6",
}
CONFIG_LABELS = {
    "full": "Full reward",
    "milestone_only": "Milestone only",
    "no_comm_bonus": "No comm bonus",
    "no_anti_hack": "No anti-hack",
}


def _per_task_chart(summary: dict[str, Any], out_path: str) -> None:
    tasks = sorted({t for cfg in summary.values() for t in cfg["per_task_avg_score"].keys()})
    x = np.arange(len(tasks))
    width = 0.2

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, cfg in enumerate(CONFIG_ORDER):
        if cfg not in summary:
            continue
        scores = [summary[cfg]["per_task_avg_score"].get(t, 0.0) for t in tasks]
        offset = (i - (len(CONFIG_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            scores,
            width,
            label=CONFIG_LABELS[cfg],
            color=CONFIG_COLORS[cfg],
            edgecolor="#0a0a1a",
            linewidth=0.6,
        )
        for b, s in zip(bars, scores):
            ax.text(
                b.get_x() + b.get_width() / 2,
                s + 0.01,
                f"{s:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#c9d1d9",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Avg episode score (0–1)")
    ax.set_xlabel("Task")
    ax.set_title("Reward Ablation: per-task average score (fixed seeds, optimal heuristic)")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="lower right", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _overall_chart(summary: dict[str, Any], out_path: str) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 4))

    cfgs = [c for c in CONFIG_ORDER if c in summary]
    scores = [summary[c]["avg_score"] for c in cfgs]
    colors = [CONFIG_COLORS[c] for c in cfgs]
    labels = [CONFIG_LABELS[c] for c in cfgs]

    bars = ax.bar(labels, scores, color=colors, edgecolor="#0a0a1a", linewidth=0.6)
    for b, s in zip(bars, scores):
        ax.text(
            b.get_x() + b.get_width() / 2,
            s + 0.01,
            f"{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#c9d1d9",
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Avg episode score (0–1)")
    ax.set_title("Reward Ablation: overall average score across tasks/seeds")
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward ablation results")
    parser.add_argument("--input", default="outputs/reward_ablation/ablation_results.json")
    parser.add_argument("--output-dir", default="outputs/reward_ablation")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    summary = data["summary"]

    os.makedirs(args.output_dir, exist_ok=True)
    per_task_path = os.path.join(args.output_dir, "ablation_per_task.png")
    overall_path = os.path.join(args.output_dir, "ablation_overall.png")
    _per_task_chart(summary, per_task_path)
    _overall_chart(summary, overall_path)
    print(json.dumps({"per_task": per_task_path, "overall": overall_path}, indent=2))


if __name__ == "__main__":
    main()
