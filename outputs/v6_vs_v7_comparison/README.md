# v6 vs v7 — live comparison snapshot

> Where: `outputs/v6_vs_v7_comparison/`
> Branch: `feature/v6-v7-comparison`
> Generated: 2026-04-26 (mid-run snapshot — both jobs still training)

This folder contains the head-to-head data we are using to test whether the v7 reward shaping fix actually unsticks task2/3/5/6 from the 0.01 floor that crippled v5/v6.

## TL;DR

| Metric | v6 (SFT, original reward) | v7 (SFT + reward fix) |
|---|---|---|
| Job ID | `69ed9454d70108f37acdf848` | `69edb1bdd70108f37acdfbb1` |
| Branch | `feature/grpo-multirole-outputs-fast` | `feature/v7-reward-fix` |
| Adapter target | `GeminiHugger/war-room-grpo-adapter-v6-sft` | `GeminiHugger/war-room-grpo-adapter-v7-rewardfix` |
| Epoch at snapshot | **0.56–0.58** (mid-run) | **0.09** (just started) |
| Steps captured in this snapshot | 27 | 14 |
| `reward_milestone/mean` lifetime | 0.319 | **0.413** |
| Fraction of steps with **partial** credit (0.011 < x < 0.95) | 70.4% | **85.7%** |
| Fraction of steps **at floor** (≤0.011) | 18.5% | **7.1%** |
| Fraction of steps near solve (≥0.95) | 11.1% | 7.1% |
| KL | 0.80 | 1.49 |
| grad_norm | 0.63 | 2.15 |

The pattern we predicted is showing up live on the wire: **v6 produces a bimodal "0.01 floor or 0.99 solve" reward distribution** (the gradient is dead on partial progress), and **v7 produces a continuous distribution with mass at 0.04, 0.19, 0.44** etc — exactly the partial-credit signal that GRPO needs to learn task2/3/5/6.

The grad_norm is also higher on v7 (2.15 vs 0.63) — that's the policy actually getting useful gradient on the harder tasks, not staying flat.

## What's in this folder

```
outputs/v6_vs_v7_comparison/
├── README.md              ← you are here
├── BLOG_POST.md           ← long-form writeup of the diagnosis & fix
├── parse_logs.py          ← reproducible parser (HF logs → CSV/JSON)
├── summary.json           ← compact machine-readable summary
├── comparison_charts.png  ← visual summary (4-panel)
├── v6_steps.json          ← step-by-step v6 metrics (json)
├── v6_steps.csv           ← same, csv
├── v7_steps.json          ← step-by-step v7 metrics (json)
├── v7_steps.csv           ← same, csv
├── merged_steps.csv       ← v6 + v7 stacked, easiest for plotting
├── v6_raw.log             ← full HF jobs logs at snapshot time
└── v7_raw.log             ← full HF jobs logs at snapshot time
```

## How to read this for Kiro / anyone else

The most important file is **`summary.json`** — it has the numbers above in one machine-readable blob. If you want to plot anything, **`merged_steps.csv`** has every captured step from both runs in one CSV with a `run` column.

The four panels in `comparison_charts.png`:

1. **top-left** — `reward_milestone/mean` per training step. v6 alternates between the 0.01 floor and 0.99 solve. v7 sits in the 0.05–0.45 partial-credit zone with no floor pile-up.
2. **top-right** — Histogram of `reward_milestone/mean` values. v6 has tall bars at the extremes; v7's mass is in the middle bins.
3. **bottom-left** — Total `reward` per step. v7 already at 0.5–0.65 by step 14, comparable to v6 at step 27.
4. **bottom-right** — KL per step. Higher on v7 because it's just started and the policy is moving more (richer gradient → bigger updates).

## How to reproduce / extend

```bash
# 1. Pull the latest logs from HF
hf jobs logs 69ed9454d70108f37acdf848 > /tmp/v6.log
hf jobs logs 69edb1bdd70108f37acdfbb1 > /tmp/v7.log

# 2. Re-run the parser (fast, no network)
python outputs/v6_vs_v7_comparison/parse_logs.py /tmp/v6.log /tmp/v7.log
```

The parser is idempotent — re-run it any time during training to refresh the comparison data. After both jobs finish you can also pull the **per-episode** metrics (the file the trainer writes inside the adapter repo as `metrics.json`) and do the per-task breakdown that v5 already has in `outputs/war_room_grpo_v5_alltasks/RESULTS.md`.

## What we expect at the end of the runs

If the diagnosis is right, then when both jobs finish:

| Metric | v5 (no SFT, broken reward) | v6 (SFT, broken reward) | v7 (SFT, fixed reward) |
|---|---|---|---|
| `task2` mean reward | 0.01 (stuck at floor) | likely also ~0.01 | **>0.10** |
| `task3` mean reward | 0.01 | ~0.01 | **>0.10** |
| `task5` mean reward | 0.01 | ~0.01 | **>0.10** |
| `task6` mean reward | 0.01 | ~0.01 | **>0.10** |
| `task1` mean reward | 0.46 | 0.46–0.55 | ≈0.50 (preserved) |
| `example_custom` mean reward | 0.95 | 0.95+ | ≈0.95 (preserved) |

If task2/3/5/6 don't move on v7 either, the fallback hypothesis is **cold-start** — the base model literally cannot emit the keywords those graders look for ("tampered", "DNS misconfiguration"), and we'd need targeted SFT examples for those tasks.

## Where the v7 reward fix lives

The actual code change is on **`feature/v7-reward-fix`**, not this branch. This branch is just the comparison data + writeups. See the v7 branch for:

- `round2/war_room/grader.py` — the 4 constant changes
- `round2/war_room/train_colab.py` — matching env_reward floor
- `hf_job_train_v7_reward_fix.sh` — the launcher
- `docs/V7_REWARD_FIX_HANDOFF.md` — full explanation

## Live job links

- v6: https://huggingface.co/jobs/GeminiHugger/69ed9454d70108f37acdf848
- v7: https://huggingface.co/jobs/GeminiHugger/69edb1bdd70108f37acdfbb1
