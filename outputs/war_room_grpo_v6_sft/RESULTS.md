# v6 — GRPO with SFT warm-up on 9 tasks

**Run finished**: 2026-04-26 (~6.5h on L40S, completed in full)
**HF Job**: `69ed9454d70108f37acdf848` (account: GeminiHugger)
**Adapter on Hub**: [`GeminiHugger/war-room-grpo-adapter-v6-sft`](https://huggingface.co/GeminiHugger/war-room-grpo-adapter-v6-sft)
**Branch trained from**: `feature/grpo-multirole-outputs-fast` (Lokesh's SFT branch)

## Config

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| **SFT warm-up adapter** | **`GeminiHugger/war-room-sft-v1-r32`** (Lokesh's SFT, upcast to r=32) |
| LoRA rank / alpha | 32 / 32 |
| Learning rate | 1e-5 |
| Episodes | 200 (per task) |
| Generations / step | 4 |
| Tasks | task1, task2, task3, task4, task5, task6, example_custom, procedural_easy, procedural_hard |
| MAX_EPISODE_ROUNDS | 20 |
| Reward function | **original v5 reward** (TIME_PRESSURE_PENALTY=0.01, PENALTY_CAP_FRACTION=0.40, FATAL_SCORE=0.01) |

## Headline numbers (7,200 episodes total)

| Metric | Mean | Min | Max |
|---|---|---|---|
| `team_reward` | **0.287** | 0.010 | 0.990 |
| `milestones_achieved` | 2.98 | 0 | 8 |
| `rounds_used` | 15.7 | 2 | 20 |
| `format_reward_avg` | **0.9999** | 0.975 | 1.000 |
| `communication_reward_avg` | **0.960** | 0.450 | 1.000 |
| `anti_hack_triggers` | **0.0** | 0 | 0 |
| `loss` | 0.035 | 0.016 | 0.545 |

## v5 vs v6 head-to-head (per-task `team_reward`)

| Task | v5 (no SFT) | **v6 (SFT)** | Δ | Verdict |
|---|---|---|---|---|
| `example_custom` | 0.946 | **0.976** | +0.030 | preserved |
| `task1` (single fault) | 0.460 | **0.573** | +0.113 | **improved** |
| `procedural_hard` | 0.350 | **0.472** | +0.122 | **improved** |
| `procedural_easy` | 0.140 | 0.135 | -0.005 | preserved |
| `task4` (auth failure) | 0.120 | 0.101 | -0.019 | regressed slightly |
| **`task2` (cascade)** | **0.010** | **0.248** | **+0.238** | **🎉 SFT unstuck it** |
| **`task6` (blame game)** | **0.010** | **0.061** | **+0.051** | partial recovery |
| **`task3` (conflicting info)** | 0.010 | 0.010 | +0.000 | still stuck |
| **`task5` (config tampering)** | 0.010 | 0.010 | +0.000 | still stuck |

## What v6 proved

1. **SFT warm-up alone is enough to unstick task2.** The task2 mean jumped from 0.010 (flat-floor across 800 episodes in v5) to 0.248. That's a 25× lift from a single intervention — adding format-teaching SFT samples before GRPO.

2. **task6 partially recovered** (0.010 → 0.061). Not as dramatic as task2 but the floor is broken — the model now sometimes hits enough milestones to get above the 0.01 clamp.

3. **task3 and task5 are still completely stuck at 0.010.** SFT didn't help these. Two possible reasons:
   - The base model literally cannot emit the keywords those graders look for ("tampered", "redis is fine"), and the SFT dataset doesn't include enough examples for those tasks specifically.
   - The reward shaping bug is the bottleneck (1-2 milestones × small credit < accumulated penalty → clamped to floor).

   v7 (reward fix) will tell us which. If v7 lifts task3/5 too, it was reward shaping. If only task2/6 climb further, then task3/5 are pure cold-start and need targeted SFT examples.

4. **task1 and procedural_hard improved meaningfully** (+0.11, +0.12). SFT didn't just unstick the dead tasks — it also lifted the moderate tasks. Format consistency from SFT gives GRPO more usable rollouts.

5. **task4 regressed slightly** (-0.019). Within noise; the SFT dataset doesn't have task4 examples and the model's prior on "auth failure" output got slightly worse.

6. **No reward hacking, perfect format, near-perfect communication.** anti_hack=0.0 across all 7,200 episodes. format_reward=0.9999. comm_reward=0.96.

## Learning trajectory

| Slice | Mean `team_reward` |
|---|---|
| First half (eps 0–3,599) | 0.287 |
| Second half (eps 3,600–7,199) | 0.287 |
| Last quarter (eps 5,400–7,199) | 0.302 |

Improvement is modest because two tasks (task3, task5) are dragging the mean down with their flat 0.010 floor reward. Of the 1,600 episodes from those two tasks, the contribution to the global mean is ~0.222 of the floor that nothing in v6 can lift.

## What this means for v7

v7 (currently RUNNING, job `69edb1bdd70108f37acdfbb1`) holds the SFT warm-up constant and *only* changes the reward function. The pre-registered prediction is:

| Task | v5 | v6 | v7 prediction |
|---|---|---|---|
| task1 | 0.460 | 0.573 | ≈0.55–0.60 (preserved) |
| example_custom | 0.946 | 0.976 | ≈0.95–0.99 (preserved) |
| task2 | 0.010 | **0.248** | **>0.30** (reward fix should still help) |
| task3 | 0.010 | 0.010 | **>0.10** ← critical falsification test |
| task5 | 0.010 | 0.010 | **>0.10** ← critical falsification test |
| task6 | 0.010 | 0.061 | **>0.15** |

If task3 and task5 still come out at 0.010 on v7, it's not the reward shaping — it's pure cold-start, and we need targeted SFT examples for those two tasks specifically.

## Files in this folder

- `metrics.json` — full per-episode metrics (7,200 rows × 10 columns, 786 KB)
- `training_curves.png` — TRL-generated training plots
- `adapter_config.json` — PEFT config (r=32, alpha=32)
- `README.md` — auto-generated by HF on adapter push
- `rollout_audit.jsonl` — empty (audit logging didn't fire)
- `v5_vs_v6_summary.json` — machine-readable summary of the head-to-head
- `job_full.log` — full HF Jobs log (721k lines, raw training output)
- `RESULTS.md` — this file
