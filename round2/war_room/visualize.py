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
    """Generate matplotlib charts if available."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    episodes = metrics["episode"]
    rewards = metrics["team_reward"]
    rounds_used = metrics["rounds_used"]
    milestones = metrics["milestones_achieved"]
    tasks = metrics["task"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multi-Agent War Room — Training Progress", fontsize=16, fontweight="bold")
    
    # 1. Reward curve
    ax = axes[0][0]
    ax.plot(episodes, rewards, "b-o", markersize=3, linewidth=1)
    # Rolling average
    if len(rewards) >= 5:
        rolling = [sum(rewards[max(0,i-4):i+1])/min(i+1,5) for i in range(len(rewards))]
        ax.plot(episodes, rolling, "r-", linewidth=2, label="5-ep rolling avg")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Team Reward")
    ax.set_title("Reward Over Episodes")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 2. Rounds to resolve
    ax = axes[0][1]
    ax.plot(episodes, rounds_used, "g-o", markersize=3, linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rounds Used")
    ax.set_title("Rounds to Resolve (↓ = more efficient)")
    ax.grid(True, alpha=0.3)
    
    # 3. Milestones achieved
    ax = axes[1][0]
    ax.bar(episodes, milestones, color="purple", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Milestones Achieved")
    ax.set_title("Milestones Per Episode")
    ax.grid(True, alpha=0.3)
    
    # 4. Reward by task
    ax = axes[1][1]
    task_rewards = {}
    for t, r in zip(tasks, rewards):
        task_rewards.setdefault(t, []).append(r)
    for task_id, task_r in sorted(task_rewards.items()):
        ax.bar(task_id, sum(task_r)/len(task_r), alpha=0.7, label=task_id)
    ax.set_xlabel("Task")
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Reward by Task")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(chart_path, dpi=150)
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
