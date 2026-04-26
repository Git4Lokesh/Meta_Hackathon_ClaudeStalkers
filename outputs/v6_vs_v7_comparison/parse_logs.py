#!/usr/bin/env python3
"""Parse HF Jobs logs for v6 (SFT, original reward) and v7 (SFT + reward fix)
and emit comparison-ready JSON + CSV.

Usage:
    python parse_logs.py /path/to/v6.log /path/to/v7.log

Inputs (raw HF jobs logs) contain GRPOTrainer step dicts of the form:
    {'loss': 0.0286, ..., 'rewards/reward_milestone/mean': 0.054999..., ...}
emitted once per training step. We extract the numeric fields, tag them with
the run name, and write per-run JSON + a CSV merged for plotting.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path

# Step dict keys we care about (rest dropped to keep CSVs small).
WANTED_KEYS = [
    "loss",
    "grad_norm",
    "learning_rate",
    "completions/mean_length",
    "completions/clipped_ratio",
    "rewards/reward_milestone/mean",
    "rewards/reward_milestone/std",
    "rewards/reward_format_lenient/mean",
    "rewards/reward_communication/mean",
    "rewards/reward_anti_hack/mean",
    "reward",
    "reward_std",
    "kl",
    "epoch",
]

STEP_RE = re.compile(r"^\{'loss': .*\}$")


def parse_step_lines(log_path: Path) -> list[dict]:
    """Walk the log file once, collecting every GRPOTrainer step dict."""
    steps: list[dict] = []
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{'loss'"):
                continue
            try:
                row = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            steps.append(row)
    return steps


def trim(rows: list[dict]) -> list[dict]:
    """Project to WANTED_KEYS, dropping anything we don't plot."""
    return [{k: r.get(k) for k in WANTED_KEYS} for r in rows]


def summary(rows: list[dict]) -> dict:
    """Produce a compact summary block for the readme & blog."""
    if not rows:
        return {"steps_logged": 0}

    last_n = rows[-min(20, len(rows)) :]
    rm_floor_count = sum(
        1
        for r in rows
        if r.get("rewards/reward_milestone/mean") is not None
        and r["rewards/reward_milestone/mean"] <= 0.011
    )
    rm_solve_count = sum(
        1
        for r in rows
        if r.get("rewards/reward_milestone/mean") is not None
        and r["rewards/reward_milestone/mean"] >= 0.95
    )
    rm_partial_count = sum(
        1
        for r in rows
        if r.get("rewards/reward_milestone/mean") is not None
        and 0.011 < r["rewards/reward_milestone/mean"] < 0.95
    )

    def m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def m_last(key):
        vals = [r[key] for r in last_n if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "steps_logged": len(rows),
        "epoch_first": rows[0].get("epoch"),
        "epoch_last": rows[-1].get("epoch"),
        "milestone_reward_distribution": {
            "at_floor_le_0p011": rm_floor_count,
            "partial_0p011_to_0p95": rm_partial_count,
            "near_solve_ge_0p95": rm_solve_count,
            "fraction_partial": (
                rm_partial_count / len(rows) if rows else None
            ),
        },
        "lifetime_means": {
            "loss": m("loss"),
            "grad_norm": m("grad_norm"),
            "kl": m("kl"),
            "reward": m("reward"),
            "rewards/reward_milestone/mean": m("rewards/reward_milestone/mean"),
            "rewards/reward_format_lenient/mean": m(
                "rewards/reward_format_lenient/mean"
            ),
            "rewards/reward_communication/mean": m(
                "rewards/reward_communication/mean"
            ),
        },
        "last_20_steps_means": {
            "loss": m_last("loss"),
            "grad_norm": m_last("grad_norm"),
            "kl": m_last("kl"),
            "reward": m_last("reward"),
            "rewards/reward_milestone/mean": m_last(
                "rewards/reward_milestone/mean"
            ),
        },
    }


def write_csv(rows: list[dict], path: Path, run_name: str) -> None:
    if not rows:
        path.write_text("step_index,run\n")
        return
    cols = ["step_index", "run"] + WANTED_KEYS
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(rows):
            w.writerow([i, run_name] + [r.get(k) for k in WANTED_KEYS])


def main():
    if len(sys.argv) != 3:
        print("Usage: parse_logs.py <v6.log> <v7.log>", file=sys.stderr)
        sys.exit(2)

    v6_log = Path(sys.argv[1])
    v7_log = Path(sys.argv[2])

    here = Path(__file__).resolve().parent

    print("[1/3] Parsing v6 log:", v6_log)
    v6_rows = trim(parse_step_lines(v6_log))
    (here / "v6_steps.json").write_text(json.dumps(v6_rows, indent=2))
    write_csv(v6_rows, here / "v6_steps.csv", "v6")
    print(f"      {len(v6_rows)} step rows")

    print("[2/3] Parsing v7 log:", v7_log)
    v7_rows = trim(parse_step_lines(v7_log))
    (here / "v7_steps.json").write_text(json.dumps(v7_rows, indent=2))
    write_csv(v7_rows, here / "v7_steps.csv", "v7")
    print(f"      {len(v7_rows)} step rows")

    print("[3/3] Building summary.json")
    summary_blob = {
        "v6_summary": summary(v6_rows),
        "v7_summary": summary(v7_rows),
        "v6_job_id": "69ed9454d70108f37acdf848",
        "v7_job_id": "69edb1bdd70108f37acdfbb1",
        "v6_branch": "feature/grpo-multirole-outputs-fast",
        "v7_branch": "feature/v7-reward-fix",
        "v6_adapter_target": "GeminiHugger/war-room-grpo-adapter-v6-sft",
        "v7_adapter_target": "GeminiHugger/war-room-grpo-adapter-v7-rewardfix",
    }
    (here / "summary.json").write_text(json.dumps(summary_blob, indent=2))

    # Also stitch a merged CSV for easier plotting
    merged_path = here / "merged_steps.csv"
    with merged_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step_index", "run"] + WANTED_KEYS)
        for i, r in enumerate(v6_rows):
            w.writerow([i, "v6"] + [r.get(k) for k in WANTED_KEYS])
        for i, r in enumerate(v7_rows):
            w.writerow([i, "v7"] + [r.get(k) for k in WANTED_KEYS])

    print("\nDone.")
    print("Artifacts in", here)


if __name__ == "__main__":
    main()
