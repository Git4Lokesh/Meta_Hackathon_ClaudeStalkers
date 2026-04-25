"""Generalization evaluation across procedural War Room scenarios.

Judges asked for evidence the environment generalizes beyond the 4 scripted
tasks. This script leans on the procedural task generator
(``round2/war_room/tasks/procedural.py``) to sample random incident scenarios
at three difficulty bands and compares two policies:

* **Baseline (random/no-op)** — every agent emits an empty action every round.
  Represents an untrained agent that never coordinates.
* **Reactive (trained-style)** — a generic heuristic policy that reads
  ``env._system`` to find crashed services and runaway processes, and has
  triage/diagnosis exchange messages so communication milestones fire.

Per-difficulty we collect: final score, rounds used, milestones achieved, and
whether the episode resolved. Results are written to
``outputs/war_room_eval/generalization.{json,png}`` and a summary table is
printed to stdout.

Usage::

    PYTHONPATH=. python round2/war_room/eval_generalization.py --seeds 30
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")  # headless backend — safe for CI/CLI
import matplotlib.pyplot as plt

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import AgentAction, Message, MultiAgentAction


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIFFICULTIES: list[str] = ["procedural_easy", "procedural_medium", "procedural_hard"]
N_SEEDS: int = 30
OUTPUT_DIR: str = "outputs/war_room_eval"
JSON_PATH: str = os.path.join(OUTPUT_DIR, "generalization.json")
PNG_PATH: str = os.path.join(OUTPUT_DIR, "generalization.png")

# Services we consider "critical" — don't try to restart these (we can't tell
# whether they were legitimately faulted). The reactive policy only handles
# non-critical crashes.
_CRITICAL_SERVICES = {"postgres", "redis", "monitoring"}

# RSS threshold (MB) above which the reactive policy will kill a process.
_RSS_KILL_THRESHOLD_MB = 2000.0


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def baseline_policy(
    env: WarRoomEnvironment,
    obs: Any,
    round_num: int,
) -> MultiAgentAction:
    """Random/no-op baseline: every agent does nothing every round."""
    return MultiAgentAction()


def reactive_policy(
    env: WarRoomEnvironment,
    obs: Any,
    round_num: int,
) -> MultiAgentAction:
    """Generic reactive (trained-style) policy.

    Reads the system state (``env._system``) to generate sensible actions:

    * Round 0: triage runs ``get_dashboard`` and broadcasts an escalation
      message to diagnosis that names the first crashed/degraded service.
    * Round 1: diagnosis runs ``cat /var/log/syslog`` and broadcasts its
      findings to remediation, citing the first crashed service.
    * Every round:
        - If any process exceeds the RSS threshold remediation does ``kill -9 <pid>``.
        - Otherwise remediation restarts the first crashed non-critical service.
        - Triage keeps polling ``get_health_summary`` / ``get_alerts`` so the
          no-op penalty stays bounded once the system is nominal.
        - Diagnosis re-reads syslog if crashes remain, otherwise stays idle
          (only one no-op slot after nominal).
    """
    system = env._system
    timestamp = system.current_time if system is not None else datetime.utcnow()

    crashed_svc = _first_crashed_service(system)
    degraded_svc = _first_degraded_service(system)
    focus_svc = crashed_svc or degraded_svc

    # --- Triage ---------------------------------------------------------
    if round_num == 0:
        escalation_target = focus_svc or "unknown_service"
        triage_action = AgentAction(
            command="get_dashboard",
            message=Message(
                from_agent="triage",
                to_agent="diagnosis",
                content=(
                    f"{escalation_target} looks down/degraded — please check "
                    f"syslog and report root cause on {escalation_target}."
                ),
                timestamp=timestamp,
                round_number=round_num,
            ),
        )
    elif focus_svc is not None:
        # Keep triage busy polling alerts while there's still an active issue.
        triage_action = AgentAction(
            command="get_alerts",
            message=Message(
                from_agent="triage",
                to_agent="all",
                content=(
                    f"Status update: {focus_svc} still not healthy — "
                    "continuing triage."
                ),
                timestamp=timestamp,
                round_number=round_num,
            ),
        )
    else:
        # Nominal — do a lightweight health check (keeps no-op penalty down).
        triage_action = AgentAction(command="get_health_summary")

    # --- Diagnosis ------------------------------------------------------
    if round_num == 1 or (focus_svc is not None and round_num > 1):
        svc_for_diag = focus_svc or "the affected service"
        content = (
            f"Syslog shows {svc_for_diag} crashed (memory/auth/cascade) — "
            f"remediation should restart or kill leaking pid."
        )
        diagnosis_action = AgentAction(
            command="cat /var/log/syslog",
            message=Message(
                from_agent="diagnosis",
                to_agent="remediation",
                content=content,
                timestamp=timestamp,
                round_number=round_num,
            ),
        )
    else:
        diagnosis_action = AgentAction()

    # --- Remediation ----------------------------------------------------
    rss_kill_target = _first_oversized_pid(system)
    if rss_kill_target is not None:
        remediation_action = AgentAction(command=f"kill -9 {rss_kill_target}")
    elif crashed_svc is not None:
        remediation_action = AgentAction(
            command=f"systemctl restart {crashed_svc}",
        )
    elif degraded_svc is not None:
        remediation_action = AgentAction(
            command=f"systemctl restart {degraded_svc}",
        )
    else:
        remediation_action = AgentAction()

    return MultiAgentAction(
        triage=triage_action,
        diagnosis=diagnosis_action,
        remediation=remediation_action,
    )


def _first_crashed_service(system: Any) -> str | None:
    """Return the name of the first crashed/stopped non-critical service."""
    if system is None:
        return None
    for name, svc in system.service_registry.services.items():
        if name in _CRITICAL_SERVICES:
            continue
        if svc.status in ("crashed", "stopped"):
            return name
    return None


def _first_degraded_service(system: Any) -> str | None:
    """Return the name of the first degraded non-critical service (not already crashed)."""
    if system is None:
        return None
    for name, svc in system.service_registry.services.items():
        if name in _CRITICAL_SERVICES:
            continue
        if svc.status == "degraded":
            return name
    return None


def _first_oversized_pid(system: Any) -> int | None:
    """Return the first PID whose RSS exceeds the threshold, else None."""
    if system is None:
        return None
    for pid, proc in system.process_table.processes.items():
        rss_mb = getattr(proc, "memory_mb", 0.0) or 0.0
        if rss_mb > _RSS_KILL_THRESHOLD_MB:
            return pid
    return None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    task_id: str,
    seed: int,
    policy: Callable[[WarRoomEnvironment, Any, int], MultiAgentAction],
) -> dict[str, Any]:
    """Run a single episode and return per-episode metrics."""
    env = WarRoomEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    max_rounds = obs.metadata.get("max_rounds", 15)

    rounds_used = 0
    for round_num in range(max_rounds):
        if obs.done:
            break
        action = policy(env, obs, round_num)
        obs = env.step(action)
        rounds_used += 1

    score = float(obs.metadata.get("score", obs.team_reward))
    milestones_achieved = obs.metadata.get("milestones_achieved", [])
    total_milestones = (
        len(env._grader.milestones) if env._grader is not None else 0
    )
    resolved = (
        total_milestones > 0 and len(milestones_achieved) == total_milestones
    )

    return {
        "seed": seed,
        "score": round(score, 4),
        "rounds_used": rounds_used,
        "milestones": len(milestones_achieved),
        "total_milestones": total_milestones,
        "resolved": bool(resolved),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode results into summary statistics."""
    scores = [r["score"] for r in per_seed]
    rounds_used = [r["rounds_used"] for r in per_seed]
    milestones = [r["milestones"] for r in per_seed]
    resolved_count = sum(1 for r in per_seed if r["resolved"])
    n = max(1, len(per_seed))
    return {
        "scores": scores,
        "rounds_used": rounds_used,
        "milestones": milestones,
        "resolved_rate": round(resolved_count / n, 4),
        "avg_score": round(sum(scores) / n, 4),
        "avg_rounds_used": round(sum(rounds_used) / n, 2),
        "avg_milestones": round(sum(milestones) / n, 2),
        "n_episodes": len(per_seed),
    }


def evaluate(n_seeds: int) -> dict[str, Any]:
    """Run baseline + reactive policies across all difficulties and seeds."""
    by_difficulty: dict[str, dict[str, Any]] = {}
    for difficulty in DIFFICULTIES:
        baseline_rows = [
            run_episode(difficulty, seed, baseline_policy)
            for seed in range(n_seeds)
        ]
        reactive_rows = [
            run_episode(difficulty, seed, reactive_policy)
            for seed in range(n_seeds)
        ]
        by_difficulty[difficulty] = {
            "baseline": _aggregate(baseline_rows),
            "reactive": _aggregate(reactive_rows),
        }
    return {
        "n_seeds": n_seeds,
        "difficulties": DIFFICULTIES,
        "by_difficulty": by_difficulty,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _short(label: str) -> str:
    return label.replace("procedural_", "")


def print_summary(results: dict[str, Any]) -> None:
    """Pretty-print a summary table to stdout."""
    header = (
        f"{'Difficulty':<8} | {'Policy':<9} | {'AvgScore':>9} | "
        f"{'Resolved%':>10} | {'AvgRounds':>10} | {'AvgMile':>8}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  WAR ROOM — GENERALIZATION SUMMARY")
    print(f"  seeds={results['n_seeds']} × difficulties={len(results['difficulties'])}")
    print("=" * len(header))
    print()
    print(header)
    print(sep)
    for difficulty, payload in results["by_difficulty"].items():
        label = _short(difficulty)
        for policy_name in ("baseline", "reactive"):
            p = payload[policy_name]
            print(
                f"{label:<8} | {policy_name:<9} | "
                f"{p['avg_score']:>9.4f} | "
                f"{p['resolved_rate'] * 100:>9.1f}% | "
                f"{p['avg_rounds_used']:>10.2f} | "
                f"{p['avg_milestones']:>8.2f}"
            )
        print(sep)
    print()


def render_plot(results: dict[str, Any], png_path: str) -> None:
    """Render the 2-panel matplotlib summary plot."""
    difficulties = results["difficulties"]
    labels = [_short(d) for d in difficulties]

    baseline_scores = [
        results["by_difficulty"][d]["baseline"]["avg_score"] for d in difficulties
    ]
    reactive_scores = [
        results["by_difficulty"][d]["reactive"]["avg_score"] for d in difficulties
    ]
    baseline_resolved = [
        results["by_difficulty"][d]["baseline"]["resolved_rate"] * 100
        for d in difficulties
    ]
    reactive_resolved = [
        results["by_difficulty"][d]["reactive"]["resolved_rate"] * 100
        for d in difficulties
    ]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")

    bar_w = 0.35
    x = list(range(len(labels)))
    x_base = [i - bar_w / 2 for i in x]
    x_reac = [i + bar_w / 2 for i in x]

    baseline_color = "#3b82f6"  # blue
    reactive_color = "#22c55e"  # green

    # --- Panel 1: Avg Score -------------------------------------------
    ax = axes[0]
    ax.set_facecolor("#0d1117")
    b1 = ax.bar(
        x_base, baseline_scores, width=bar_w,
        label="Baseline (no-op)", color=baseline_color, edgecolor="#1f2937",
    )
    b2 = ax.bar(
        x_reac, reactive_scores, width=bar_w,
        label="Reactive (trained-style)", color=reactive_color, edgecolor="#1f2937",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg score (0.0 – 1.0)")
    ax.set_xlabel("Difficulty")
    ax.set_title("Average Score by Difficulty")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.7)
    for bar in list(b1) + list(b2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.3f}",
            ha="center", va="bottom", fontsize=9, color="#e5e7eb",
        )

    # --- Panel 2: Resolved rate ---------------------------------------
    ax = axes[1]
    ax.set_facecolor("#0d1117")
    b1 = ax.bar(
        x_base, baseline_resolved, width=bar_w,
        label="Baseline (no-op)", color=baseline_color, edgecolor="#1f2937",
    )
    b2 = ax.bar(
        x_reac, reactive_resolved, width=bar_w,
        label="Reactive (trained-style)", color=reactive_color, edgecolor="#1f2937",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Resolved rate (%)")
    ax.set_xlabel("Difficulty")
    ax.set_title("Resolved Rate by Difficulty")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.7)
    for bar in list(b1) + list(b2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.5,
            f"{height:.1f}%",
            ha="center", va="bottom", fontsize=9, color="#e5e7eb",
        )

    fig.suptitle(
        f"War Room — Generalization across {results['n_seeds']} seeds × "
        f"{len(difficulties)} difficulties",
        fontsize=13, color="#e5e7eb",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate War Room generalization on procedural scenarios.",
    )
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help=f"Number of random seeds per difficulty (default: {N_SEEDS}).",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Directory for JSON + PNG outputs.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "generalization.json")
    png_path = os.path.join(args.output_dir, "generalization.png")

    start = time.time()
    results = evaluate(n_seeds=args.seeds)
    results["elapsed_sec"] = round(time.time() - start, 2)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    render_plot(results, png_path)
    print_summary(results)
    print(f"  Wrote {json_path}")
    print(f"  Wrote {png_path}")
    print(f"  Elapsed: {results['elapsed_sec']}s")


if __name__ == "__main__":
    main()
