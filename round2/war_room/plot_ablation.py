"""Plot the reward ablation results as a grouped bar chart.

Reads outputs/reward_ablation/ablation_results.json produced by
reward_ablation.py and emits outputs/reward_ablation/ablation_results.png
suitable for the pitch deck and the README.

The chart groups by reward configuration (full / milestone_only / no_comm_bonus
/ no_anti_hack) with one bar per task. Dark theme to match the Gradio UI.

Usage:
    PYTHONPATH=. python round2/war_room/plot_ablation.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_ORDER = ["full", "milestone_only", "no_comm_bonus", "no_anti_hack"]
CONFIG_LABELS = {
    "full": "Full reward",
    "milestone_only": "Milestones only",
    "no_comm_bonus": "No comm bonus",
    "no_anti_hack": "No anti-hack gate",
}
TASK_ORDER = ["task1", "task2", "task3", "task4"]
TASK_LABELS = {
    "task1": "T1 · Restart",
    "task2": "T2 · Leak",
    "task3": "T3 · Cascade",
    "task4": "T4 · Parallel",
}
TASK_COLORS = {
    "task1": "#58a6ff",
    "task2": "#3fb950",
    "task3": "#d29922",
    "task4": "#f85149",
}


def _aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Return {config: {task: avg_score}} from the raw ablation rows."""
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        bucket[row["config"]][row["task"]].append(float(row["score"]))
    return {
        cfg: {task: sum(scores) / len(scores) for task, scores in by_task.items()}
        for cfg, by_task in bucket.items()
    }


def plot(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        payload = json.load(f)
    rows = payload["rows"]

    agg = _aggregate(rows)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    n_configs = len(CONFIG_ORDER)
    n_tasks = len(TASK_ORDER)
    group_width = 0.72
    bar_width = group_width / n_tasks

    x_positions = list(range(n_configs))
    for i, task in enumerate(TASK_ORDER):
        heights = [agg.get(cfg, {}).get(task, 0.0) for cfg in CONFIG_ORDER]
        offsets = [x + (i - (n_tasks - 1) / 2) * bar_width for x in x_positions]
        ax.bar(
            offsets,
            heights,
            bar_width,
            label=TASK_LABELS[task],
            color=TASK_COLORS[task],
            edgecolor="#0d1117",
            linewidth=0.5,
        )
        for xi, h in zip(offsets, heights):
            ax.text(
                xi,
                h + 0.02,
                f"{h:.2f}",
                color="#c9d1d9",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_ORDER], color="#c9d1d9")
    ax.set_ylabel("Average team score (0–1)", color="#8b949e")
    ax.set_title(
        "Reward ablation — scripted expert policy, 3 seeds × 4 tasks",
        color="#c9d1d9",
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#484f58")
    ax.spines["bottom"].set_color("#21262d")
    ax.spines["left"].set_color("#21262d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.12, color="#30363d", axis="y")

    legend = ax.legend(
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="#c9d1d9",
        ncol=n_tasks,
        loc="upper right",
    )
    for text in legend.get_texts():
        text.set_color("#c9d1d9")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward ablation results")
    parser.add_argument(
        "--input",
        default="outputs/reward_ablation/ablation_results.json",
        help="Path to ablation JSON from reward_ablation.py",
    )
    parser.add_argument(
        "--output",
        default="outputs/reward_ablation/ablation_results.png",
        help="Output PNG path",
    )
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
