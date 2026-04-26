"""
Plot the v6 vs v7 milestone-reward trajectory across every snapshot we have.

Pulls the merged_steps.csv that Lakshminath's parse_logs.py produces and
draws a single chart showing:
  - per-step milestone_reward_mean for both runs on the same axes
  - rolling mean overlay to smooth the bimodal GRPO noise
  - dashed horizontal line at the 0.01 floor showing where v5/v6
    rollouts pile up and v7 does not

Run:
    PYTHONPATH=. .venv/bin/python scripts/plot_v6_v7_trajectory.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = REPO_ROOT / "outputs" / "v6_vs_v7_comparison" / "merged_steps.csv"
OUT_PATH = REPO_ROOT / "outputs" / "v6_vs_v7_comparison" / "v6_v7_trajectory.png"


def _load_runs(path: Path) -> dict[str, dict[str, list[float]]]:
    runs: dict[str, dict[str, list[float]]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row["run"]
            runs.setdefault(run, {"step": [], "milestone": [], "epoch": []})
            runs[run]["step"].append(int(row["step_index"]))
            runs[run]["milestone"].append(float(row["rewards/reward_milestone/mean"]))
            runs[run]["epoch"].append(float(row["epoch"]))
    return runs


def _rolling_mean(vals: list[float], window: int = 5) -> list[float]:
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(sum(vals[lo : i + 1]) / (i - lo + 1))
    return out


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Missing {IN_PATH}. Run parse_logs.py first.")

    runs = _load_runs(IN_PATH)
    if "v6" not in runs or "v7" not in runs:
        raise SystemExit(f"Expected v6 and v7 runs, got {list(runs.keys())}")

    BG = "#0d1117"
    FG = "#c9d1d9"
    MUTED = "#8b949e"
    GRID = "#21262d"
    V6_COLOR = "#f85149"  # red — old reward
    V7_COLOR = "#3fb950"  # green — fixed reward

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(BG)

    for run_name, color, label in [
        ("v6", V6_COLOR, "v6 (SFT + original reward)"),
        ("v7", V7_COLOR, "v7 (SFT + reward surgery)"),
    ]:
        d = runs[run_name]
        if not d["step"]:
            continue
        ax.scatter(
            d["step"], d["milestone"],
            s=18, alpha=0.35, color=color, edgecolors="none",
            label=f"{label} — per-step",
        )
        smoothed = _rolling_mean(d["milestone"], window=5)
        ax.plot(
            d["step"], smoothed,
            color=color, linewidth=2.2,
            label=f"{label} — rolling mean (w=5)",
        )

    # Reward-function floor line — the exact value v5/v6 rollouts cluster at
    ax.axhline(0.01, color=MUTED, linestyle="--", linewidth=1,
               label="0.01 reward floor (v6 clamp)")

    ax.set_xlabel("Training step", color=FG, fontsize=11)
    ax.set_ylabel("rewards/reward_milestone/mean (0–1)", color=FG, fontsize=11)
    ax.set_title(
        "v6 vs v7 milestone reward per training step\n"
        "v7's reward-surgery patch lifts the model off the 0.01 floor",
        color=FG, fontsize=12, pad=12,
    )
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.4, linestyle="-")
    ax.set_ylim(-0.02, 1.05)

    leg = ax.legend(
        loc="upper left", framealpha=0.9, facecolor="#161b22",
        edgecolor=GRID, labelcolor=FG, fontsize=9,
    )
    for text in leg.get_texts():
        text.set_color(FG)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
