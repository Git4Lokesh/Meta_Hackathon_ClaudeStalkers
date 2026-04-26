"""Inspect Lakshminath's war_room_grpo_multirole_v2 metrics.json in detail."""
from __future__ import annotations
import json
import statistics
from collections import defaultdict, Counter


with open("outputs/war_room_grpo_multirole_v2/metrics.json") as f:
    m = json.load(f)

print("=" * 72)
print("LAKSHMINATH'S multirole_v2 — training metrics inspection")
print("=" * 72)

print(f"\nKeys in metrics.json: {sorted(m.keys())}")

rewards = m["team_reward"]
print(f"\nTotal recorded rows: {len(rewards)}")
print(
    f"team_reward — mean={statistics.mean(rewards):.3f}  "
    f"median={statistics.median(rewards):.3f}  "
    f"stdev={statistics.pstdev(rewards):.3f}  "
    f"min={min(rewards):.3f}  max={max(rewards):.3f}",
)

# Quartile progression — are rewards climbing?
n = len(rewards)
q = n // 4
print("\nReward curve (quartile means — is training actually improving?):")
for i in range(4):
    phase = rewards[i * q:(i + 1) * q]
    print(f"  Q{i + 1}  (rows {i * q:5d}-{(i + 1) * q:5d}): mean={sum(phase)/len(phase):.3f}")

# Per-task breakdown
if "task" in m:
    per_task_r: dict[str, list[float]] = defaultdict(list)
    per_task_m: dict[str, list[int]] = defaultdict(list)
    per_task_r_used: dict[str, list[int]] = defaultdict(list)
    for task, r, milestones, ru in zip(
        m["task"], rewards,
        m.get("milestones_achieved", [0] * n),
        m.get("rounds_used", [0] * n),
    ):
        per_task_r[task].append(r)
        per_task_m[task].append(milestones)
        per_task_r_used[task].append(ru)

    print("\nPer-task performance:")
    print(f"  {'task':24s}  {'n':>5s}  {'reward_mean':>11s}  {'reward_max':>10s}  "
          f"{'milestone_mean':>14s}  {'rounds_mean':>11s}")
    for t in sorted(per_task_r.keys()):
        rs = per_task_r[t]
        ms = per_task_m[t]
        ru = per_task_r_used[t]
        print(
            f"  {t:24s}  {len(rs):5d}  {sum(rs)/len(rs):11.3f}  "
            f"{max(rs):10.3f}  {sum(ms)/len(ms):14.2f}  {sum(ru)/len(ru):11.1f}",
        )

# How many episodes hit high reward (≥0.8)?
high = sum(1 for r in rewards if r >= 0.8)
mid = sum(1 for r in rewards if 0.3 <= r < 0.8)
low = sum(1 for r in rewards if r < 0.3)
print(
    f"\nReward distribution:\n"
    f"  high (≥0.80): {high:5d} ({100*high/n:.1f}%)\n"
    f"  mid  (0.3-0.8): {mid:5d} ({100*mid/n:.1f}%)\n"
    f"  low  (<0.30): {low:5d} ({100*low/n:.1f}%)",
)

# Milestones distribution
if "milestones_achieved" in m:
    ms_counter = Counter(m["milestones_achieved"])
    print(f"\nMilestones achieved per episode:")
    for k in sorted(ms_counter.keys()):
        print(f"  {k} milestones: {ms_counter[k]:5d}  ({100*ms_counter[k]/n:.1f}%)")

# Format / comm / anti-hack reward channels
for key in ("format_reward_avg", "communication_reward_avg", "anti_hack_triggers"):
    if key in m:
        vals = m[key]
        if vals and any(v for v in vals):
            print(
                f"\n{key}: mean={sum(vals)/len(vals):.3f}  "
                f"n_nonzero={sum(1 for v in vals if v)}/{len(vals)}  "
                f"max={max(vals):.3f}",
            )
        else:
            print(f"\n{key}: ALL ZERO — channel not being recorded")
