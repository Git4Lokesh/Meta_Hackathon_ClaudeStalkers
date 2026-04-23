#!/usr/bin/env python3
"""Generate publication-quality training charts from metrics.json.

Produces two PNG files:
  1. training_curves.png  — reward curve, per-reward breakdown, milestones bar
  2. baseline_vs_trained.png — per-task score comparison (heuristic demo data)

Usage:
    python round2/war_room/generate_charts.py
    python round2/war_room/generate_charts.py --metrics path/to/metrics.json
    python round2/war_room/generate_charts.py --output-dir outputs/charts/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Chart styling helpers
# ---------------------------------------------------------------------------

BG = "#0d1117"
FG = "#c9d1d9"
MUTED = "#8b949e"
GRID = "#30363d"
BORDER = "#21262d"

COLORS = {
    "blue": "#58a6ff",
    "red": "#f85149",
    "green": "#3fb950",
    "purple": "#bc8cff",
    "orange": "#d29922",
    "cyan": "#39d2c0",
}


def _style_ax(ax):
    """Apply dark GitHub-style theme to an axis."""
    ax.set_facecolor(BG)
    ax.tick_params(colors="#484f58")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(BORDER)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.1, color=GRID)


# ---------------------------------------------------------------------------
# Training curves (from metrics.json)
# ---------------------------------------------------------------------------

def generate_training_curves(metrics: dict, output_dir: str) -> str:
    """Generate a 3-panel training curves figure and return the saved path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = metrics["episode"]
    rewards = metrics["team_reward"]
    milestones = metrics.get("milestones_achieved", [0] * len(episodes))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "GRPO Training — Multi-Agent War Room",
        color=FG, fontsize=16, fontweight="bold", y=1.02,
    )

    # ── Panel 1: Reward curve ──
    ax1 = axes[0]
    _style_ax(ax1)
    ax1.plot(
        episodes, rewards,
        color=COLORS["blue"], marker="o", markersize=4,
        linewidth=2, markerfacecolor=COLORS["blue"], markeredgecolor=BG,
    )
    ax1.fill_between(episodes, rewards, alpha=0.15, color=COLORS["blue"])
    if len(rewards) >= 3:
        rolling = [
            sum(rewards[max(0, i - 2): i + 1]) / min(i + 1, 3)
            for i in range(len(rewards))
        ]
        ax1.plot(
            episodes, rolling,
            color=COLORS["red"], linewidth=2, linestyle="--",
            label="3-ep rolling avg",
        )
        ax1.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=FG)
    ax1.set_xlabel("Episode", color=MUTED)
    ax1.set_ylabel("Team Reward (0–1)", color=MUTED)
    ax1.set_title("Reward ↑", color=FG, fontweight="bold")
    ax1.set_ylim(0, 1)
    # Best-episode annotation
    if rewards:
        best_idx = rewards.index(max(rewards))
        ax1.annotate(
            f"Best: {rewards[best_idx]:.3f}",
            xy=(episodes[best_idx], rewards[best_idx]),
            xytext=(episodes[best_idx], min(rewards[best_idx] + 0.08, 0.98)),
            arrowprops=dict(arrowstyle="->", color="#FFD700"),
            color="#FFD700", fontweight="bold", fontsize=10,
        )

    # ── Panel 2: Per-reward-function breakdown ──
    ax2 = axes[1]
    _style_ax(ax2)
    breakdown_keys = {
        "format_reward": ("Format", COLORS["green"]),
        "communication_reward": ("Communication", COLORS["cyan"]),
        "anti_hack_reward": ("Anti-Hack", COLORS["orange"]),
    }
    has_breakdown = False
    for key, (label, color) in breakdown_keys.items():
        if key in metrics and metrics[key]:
            ax2.plot(episodes, metrics[key], color=color, linewidth=2, label=label)
            has_breakdown = True
    if not has_breakdown:
        # Fallback: show rounds_used as efficiency proxy
        rounds_used = metrics.get("rounds_used", [])
        if rounds_used:
            ax2.plot(
                episodes, rounds_used,
                color=COLORS["green"], marker="s", markersize=4,
                linewidth=2, markerfacecolor=COLORS["green"], markeredgecolor=BG,
            )
            ax2.fill_between(episodes, rounds_used, alpha=0.15, color=COLORS["green"])
            ax2.set_ylabel("Rounds Used", color=MUTED)
            ax2.set_title("Efficiency ↓ (Lower = Faster)", color=FG, fontweight="bold")
        else:
            ax2.text(
                0.5, 0.5, "No per-reward breakdown\navailable in metrics",
                transform=ax2.transAxes, ha="center", va="center",
                color=MUTED, fontsize=12,
            )
            ax2.set_title("Reward Breakdown", color=FG, fontweight="bold")
    else:
        ax2.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=FG)
        ax2.set_ylabel("Reward Component (0–1)", color=MUTED)
        ax2.set_title("Per-Reward Breakdown", color=FG, fontweight="bold")
        ax2.set_ylim(0, 1)
    ax2.set_xlabel("Episode", color=MUTED)

    # ── Panel 3: Milestones bar chart ──
    ax3 = axes[2]
    _style_ax(ax3)
    ax3.bar(
        episodes, milestones,
        color=COLORS["purple"], alpha=0.8,
        edgecolor="#8957e5", linewidth=0.5,
    )
    ax3.set_xlabel("Episode", color=MUTED)
    ax3.set_ylabel("Milestones Achieved", color=MUTED)
    ax3.set_title("Milestones per Episode", color=FG, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  📈 Saved {chart_path}")
    return chart_path


# ---------------------------------------------------------------------------
# Baseline vs Trained comparison chart
# ---------------------------------------------------------------------------

def _run_demo_comparison() -> dict[str, dict[str, float]]:
    """Run heuristic demo comparison and return per-task scores."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from round2.war_room.demo_comparison import run_episode, TASK_IDS

    results: dict[str, dict[str, float]] = {"baseline": {}, "trained": {}}
    for task_id in TASK_IDS:
        bl = run_episode(task_id, seed=42, skill_level=0.0)
        tr = run_episode(task_id, seed=42, skill_level=1.0)
        results["baseline"][task_id] = bl["score"]
        results["trained"][task_id] = tr["score"]
    return results


def generate_baseline_vs_trained(output_dir: str) -> str:
    """Generate a grouped bar chart comparing baseline vs trained scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    print("  Running heuristic demo comparison …")
    data = _run_demo_comparison()

    tasks = sorted(data["baseline"].keys())
    baseline_scores = [data["baseline"][t] for t in tasks]
    trained_scores = [data["trained"][t] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    bars_bl = ax.bar(x - width / 2, baseline_scores, width, label="Baseline",
                     color=COLORS["red"], alpha=0.85, edgecolor=BG)
    bars_tr = ax.bar(x + width / 2, trained_scores, width, label="Trained",
                     color=COLORS["green"], alpha=0.85, edgecolor=BG)

    # Value labels on bars
    for bar in bars_bl:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", color=MUTED, fontsize=9)
    for bar in bars_tr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", color=MUTED, fontsize=9)

    ax.set_xlabel("Task", color=MUTED)
    ax.set_ylabel("Score (0–1)", color=MUTED)
    ax.set_title("Baseline vs Trained — Per-Task Scores", color=FG, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, color=MUTED)
    ax.set_ylim(0, 1.15)
    ax.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=FG)

    # Caption annotation
    avg_bl = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    avg_tr = sum(trained_scores) / len(trained_scores) if trained_scores else 0
    ax.text(
        0.5, -0.12,
        f"Avg baseline: {avg_bl:.3f}  →  Avg trained: {avg_tr:.3f}  (Δ {avg_tr - avg_bl:+.3f})",
        transform=ax.transAxes, ha="center", color=MUTED, fontsize=10,
    )

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "baseline_vs_trained.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  📊 Saved {chart_path}")
    return chart_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training charts for the War Room environment.",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/war_room_grpo/metrics.json",
        help="Path to metrics.json (default: outputs/war_room_grpo/metrics.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output PNGs (default: same dir as metrics.json)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.metrics).parent)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Training curves (if metrics.json exists)
    if os.path.isfile(args.metrics):
        print(f"📂 Loading metrics from {args.metrics}")
        with open(args.metrics) as f:
            metrics = json.load(f)
        generate_training_curves(metrics, output_dir)
    else:
        print(f"⚠️  {args.metrics} not found — skipping training curves.")
        print("   (Run training first, or pass --metrics path/to/metrics.json)")

    # 2. Baseline vs Trained (always available — no GPU needed)
    print("\n📊 Generating baseline vs trained comparison …")
    try:
        generate_baseline_vs_trained(output_dir)
    except Exception as e:
        print(f"⚠️  Could not generate baseline_vs_trained chart: {e}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
