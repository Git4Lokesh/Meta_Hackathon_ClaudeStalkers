"""Metrics visualization for the Multi-Agent War Room.

Generates reward curve charts from training metrics to demonstrate
agent improvement over episodes.

Usage:
    python round2/war_room/visualize.py [--metrics outputs/war_room_training/metrics.json]
    
If matplotlib is not available, outputs text-based charts.
"""

import json
import argparse
import os
import sys


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def text_chart(values: list[float], title: str, width: int = 50) -> str:
    """Generate a simple text-based bar chart."""
    if not values:
        return f"{title}: No data"
    
    lines = [f"\n{title}", "─" * (width + 15)]
    max_val = max(values) if values else 1.0
    
    for i, v in enumerate(values):
        bar_len = int((v / max(max_val, 0.01)) * width)
        bar = "█" * bar_len
        lines.append(f"  Ep {i:3d} │{bar} {v:.3f}")
    
    lines.append("─" * (width + 15))
    avg = sum(values) / len(values)
    lines.append(f"  Average: {avg:.3f}")
    return "\n".join(lines)


def plot_matplotlib(metrics: dict, output_dir: str):
    """Generate publication-quality matplotlib charts."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    
    episodes = metrics["episode"]
    rewards = metrics["team_reward"]
    rounds_used = metrics["rounds_used"]
    milestones = metrics["milestones_achieved"]
    tasks = metrics["task"]
    
    # Color scheme
    COLORS = {
        "task1": "#4CAF50",  # green
        "task2": "#2196F3",  # blue
        "task3": "#FF9800",  # orange
        "task4": "#E91E63",  # pink
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Multi-Agent War Room — Training Progress",
        fontsize=18, fontweight="bold", y=0.98,
    )
    fig.patch.set_facecolor("#FAFAFA")
    
    # ---- 1. Reward curve with rolling average ----
    ax = axes[0][0]
    ax.set_facecolor("#F5F5F5")
    
    # Scatter points colored by task
    for i, (ep, r, t) in enumerate(zip(episodes, rewards, tasks)):
        ax.scatter(ep, r, c=COLORS.get(t, "gray"), s=20, alpha=0.6, zorder=3)
    
    # Rolling average
    window = 5
    if len(rewards) >= window:
        rolling = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            rolling.append(sum(rewards[start:i+1]) / (i - start + 1))
        ax.plot(episodes, rolling, "red", linewidth=2.5, label=f"{window}-ep rolling avg", zorder=4)
    
    # Trend line
    if len(rewards) > 2:
        z = np.polyfit(episodes, rewards, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(episodes), max(episodes), 100)
        ax.plot(x_smooth, np.clip(p(x_smooth), 0, 1), "--", color="#666", 
                linewidth=1.5, alpha=0.7, label="Trend", zorder=2)
    
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Team Reward", fontsize=11)
    ax.set_title("Reward Over Training Episodes", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add phase annotations
    n = len(episodes)
    if n > 10:
        ax.axvspan(0, n*0.25, alpha=0.05, color='green', label='_')
        ax.axvspan(n*0.25, n*0.5, alpha=0.05, color='blue', label='_')
        ax.axvspan(n*0.5, n*0.75, alpha=0.05, color='orange', label='_')
        ax.axvspan(n*0.75, n, alpha=0.05, color='red', label='_')
        ax.text(n*0.125, 1.02, "Phase 1\n(Easy)", ha='center', fontsize=7, color='green')
        ax.text(n*0.375, 1.02, "Phase 2\n(Medium)", ha='center', fontsize=7, color='blue')
        ax.text(n*0.625, 1.02, "Phase 3\n(Hard)", ha='center', fontsize=7, color='orange')
        ax.text(n*0.875, 1.02, "Phase 4\n(Expert)", ha='center', fontsize=7, color='red')
    
    # ---- 2. Rounds to resolve ----
    ax = axes[0][1]
    ax.set_facecolor("#F5F5F5")
    
    for i, (ep, r, t) in enumerate(zip(episodes, rounds_used, tasks)):
        ax.scatter(ep, r, c=COLORS.get(t, "gray"), s=20, alpha=0.6, zorder=3)
    
    if len(rounds_used) >= window:
        rolling_r = []
        for i in range(len(rounds_used)):
            start = max(0, i - window + 1)
            rolling_r.append(sum(rounds_used[start:i+1]) / (i - start + 1))
        ax.plot(episodes, rolling_r, "green", linewidth=2.5, label=f"{window}-ep rolling avg", zorder=4)
    
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Rounds Used", fontsize=11)
    ax.set_title("Rounds to Resolve (↓ = more efficient)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ---- 3. Milestones per episode ----
    ax = axes[1][0]
    ax.set_facecolor("#F5F5F5")
    
    bar_colors = [COLORS.get(t, "gray") for t in tasks]
    ax.bar(episodes, milestones, color=bar_colors, alpha=0.7, width=0.8)
    
    if len(milestones) >= window:
        rolling_m = []
        for i in range(len(milestones)):
            start = max(0, i - window + 1)
            rolling_m.append(sum(milestones[start:i+1]) / (i - start + 1))
        ax.plot(episodes, rolling_m, "red", linewidth=2, label=f"{window}-ep rolling avg")
        ax.legend(fontsize=9)
    
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Milestones Achieved", fontsize=11)
    ax.set_title("Milestones Per Episode", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # ---- 4. Average reward by task ----
    ax = axes[1][1]
    ax.set_facecolor("#F5F5F5")
    
    task_rewards: dict[str, list[float]] = {}
    for t, r in zip(tasks, rewards):
        task_rewards.setdefault(t, []).append(r)
    
    # Split into first half vs second half for before/after comparison
    task_first_half: dict[str, list[float]] = {}
    task_second_half: dict[str, list[float]] = {}
    mid = len(rewards) // 2
    for i, (t, r) in enumerate(zip(tasks, rewards)):
        if i < mid:
            task_first_half.setdefault(t, []).append(r)
        else:
            task_second_half.setdefault(t, []).append(r)
    
    all_tasks = sorted(set(tasks))
    x_pos = range(len(all_tasks))
    bar_width = 0.35
    
    first_avgs = [
        sum(task_first_half.get(t, [0.01])) / max(len(task_first_half.get(t, [0.01])), 1)
        for t in all_tasks
    ]
    second_avgs = [
        sum(task_second_half.get(t, [0.01])) / max(len(task_second_half.get(t, [0.01])), 1)
        for t in all_tasks
    ]
    
    bars1 = ax.bar(
        [x - bar_width/2 for x in x_pos], first_avgs, 
        bar_width, label="Before Training", color="#BDBDBD", alpha=0.8,
    )
    bars2 = ax.bar(
        [x + bar_width/2 for x in x_pos], second_avgs, 
        bar_width, label="After Training", 
        color=[COLORS.get(t, "gray") for t in all_tasks], alpha=0.9,
    )
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8, color='gray')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Average Reward", fontsize=11)
    ax.set_title("Before vs After Training", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(all_tasks)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Task color legend
    patches = [mpatches.Patch(color=c, label=t) for t, c in COLORS.items() if t in set(tasks)]
    fig.legend(
        handles=patches, loc='lower center', ncol=4, fontsize=9,
        bbox_to_anchor=(0.5, 0.01), frameon=True, fancybox=True,
    )
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    chart_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Charts saved to {chart_path}")
    return chart_path


def main():
    parser = argparse.ArgumentParser(description="Visualize War Room training metrics")
    parser.add_argument("--metrics", default="outputs/war_room_training/metrics.json")
    parser.add_argument("--output", default="outputs/war_room_training")
    args = parser.parse_args()
    
    if not os.path.exists(args.metrics):
        print(f"Metrics file not found: {args.metrics}")
        print("Run training first: python round2/war_room/train.py")
        sys.exit(1)
    
    metrics = load_metrics(args.metrics)
    
    # Always show text charts
    print(text_chart(metrics["team_reward"], "Team Reward Over Episodes"))
    print(text_chart(metrics["rounds_used"], "Rounds Used Per Episode"))
    
    # Try matplotlib
    try:
        plot_matplotlib(metrics, args.output)
    except ImportError:
        print("\nmatplotlib not installed — text charts only.")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
