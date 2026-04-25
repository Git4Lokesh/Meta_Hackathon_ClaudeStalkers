"""Generate generalization plots from generalization_eval.json.

Produces two charts:
  - Avg score per difficulty (baseline vs trained-style), with resolved rate annotation
  - Resolved-rate per difficulty
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

DIFFICULTY_ORDER = ["procedural_easy", "procedural_medium", "procedural_hard"]
SHORT_LABELS = {
    "procedural_easy": "Easy",
    "procedural_medium": "Medium",
    "procedural_hard": "Hard",
}


def _score_chart(summary: dict, out_path: str, n_seeds: int) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))

    diffs = [d for d in DIFFICULTY_ORDER if d in summary]
    x = np.arange(len(diffs))
    width = 0.35

    base_scores = [summary[d]["baseline"]["avg_score"] for d in diffs]
    trained_scores = [summary[d]["trained"]["avg_score"] for d in diffs]
    base_res = [summary[d]["baseline"]["resolved_rate"] for d in diffs]
    trained_res = [summary[d]["trained"]["resolved_rate"] for d in diffs]

    bars1 = ax.bar(x - width / 2, base_scores, width, label="Baseline (no-coord)", color="#ef4444", edgecolor="#0a0a1a")
    bars2 = ax.bar(x + width / 2, trained_scores, width, label="Trained-style (adaptive)", color="#22c55e", edgecolor="#0a0a1a")

    for b, s, rr in zip(bars1, base_scores, base_res):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.02, f"{s:.2f}\n({rr:.0%})", ha="center", fontsize=8, color="#c9d1d9")
    for b, s, rr in zip(bars2, trained_scores, trained_res):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.02, f"{s:.2f}\n({rr:.0%})", ha="center", fontsize=8, color="#c9d1d9")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[d] for d in diffs])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Avg episode score (0–1)")
    ax.set_xlabel("Procedural difficulty")
    ax.set_title(f"Generalization across procedural seeds (n={n_seeds} per cell)")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper left", framealpha=0.85)

    note = "Bars labeled with avg score and (% resolved). Same env/reward/seeds; only the policy differs."
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="#8b949e", style="italic")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot generalization eval results")
    parser.add_argument("--input", default="outputs/generalization_eval/generalization_eval.json")
    parser.add_argument("--output-dir", default="outputs/generalization_eval")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    summary = data["summary"]
    n_seeds = len(data.get("seeds", []))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "generalization_score.png")
    _score_chart(summary, out_path, n_seeds)
    print(json.dumps({"score_chart": out_path}, indent=2))


if __name__ == "__main__":
    main()
