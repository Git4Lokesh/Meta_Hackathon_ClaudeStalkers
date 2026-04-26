# v7 reward shaping fix — handoff for Lokesh

**TL;DR**: v5 wasn't broken because of model size or SFT — it was broken because the reward function destroyed the gradient on tasks 2/3/5/6. v6 (your SFT warm-up branch) inherits the same broken reward. v7 fixes the reward, runs in parallel, and is the apples-to-apples test that isolates the impact.

## Branches in play

| Branch | Status | Adapter | Purpose |
|---|---|---|---|
| `feature/grpo-multirole-outputs-fast` | merged into v5/v6 | `GeminiHugger/war-room-grpo-adapter-v5`, `…-v6-sft` | original reward + 9 tasks |
| `feature/v7-reward-fix` | **this branch** | `GeminiHugger/war-room-grpo-adapter-v7-rewardfix` | fixed reward + SFT + 9 tasks |

## What I changed (3 files, 4 constants)

### `round2/war_room/grader.py`
```diff
- TIME_PRESSURE_PENALTY = 0.01
+ TIME_PRESSURE_PENALTY = 0.005

- FATAL_SCORE = 0.01
+ FATAL_SCORE = 0.001

- PENALTY_CAP_FRACTION = 0.40
+ PENALTY_CAP_FRACTION = 0.10

  # current_score() lower clamp
- return max(0.01, min(0.99, raw))
+ return max(0.001, min(0.99, raw))
```

### `round2/war_room/train_colab.py`
```diff
- "env_reward": float(max(0.01, min(0.99, score))),
+ "env_reward": float(max(0.001, min(0.99, score))),
```

That's it. No changes to the policy, the curriculum, the SFT adapter, the GRPO loss, or any task definition.

## Why v5 was stuck (the diagnosis that prompted this)

Per-task v5 means after 7,200 episodes:

| Task | mean | what was happening |
|---|---|---|
| task1 | 0.460 | learning |
| example_custom | 0.946 | basically solved |
| procedural_hard | 0.350 | partial |
| procedural_easy | 0.140 | weak |
| task4 | 0.120 | barely learning |
| **task2** | **0.010** | flat at floor, all 800 episodes |
| **task3** | **0.010** | flat at floor, all 800 episodes |
| **task5** | **0.010** | flat at floor, all 800 episodes |
| **task6** | **0.010** | flat at floor, all 800 episodes |

The "stuck at 0.01" was suspicious — exactly the lower clamp. Tracing through the math:

```
raw = milestone_credit - min(penalty_raw, PENALTY_CAP_FRACTION × total_credit)
      + communication_bonus + solve_bonus

penalty_raw ≈ TIME_PRESSURE_PENALTY × rounds_used + NOOP_PENALTY × noops
            ≈ 0.01 × 20 + 0.01 × ~5  =  ~0.25

penalty_cap  =  0.40 × 1.0  =  0.40   (so cap never bites)
penalty_applied  ≈  0.25
```

So for task5/task6, each milestone is worth ~0.05–0.20. The model was actually hitting 1–3 milestones per episode (95–98% of episodes hit at least one), earning ~0.05–0.30 of credit, but penalty_applied of ~0.25 wiped it out. `raw` was negative or near-zero → clamp at 0.01 → **flat reward surface → no GRPO gradient → no learning, ever**.

The 7B model wasn't the problem. The reward function was.

## Why these specific values

I simulated 5 candidate parameter sets against the actual milestone structures of all 7 tasks. The winner satisfies:

1. **Don't break what works**: task1 still scores 0.99 at full solve (was 0.99). example_custom still scores 0.99 (was 0.95).
2. **Restore gradient where it died**: task2/3/5/6 all show monotonically increasing reward as more milestones get hit, instead of flat 0.01.
3. **Minimal blast radius**: 3 constant changes, no logic rewrites.

Functional simulation of patched curves (real graders, real milestones, max time penalty applied):

```
task     #ms maxR  totC   ms=0    ms=1    ms=2    ms=3    ms=4    ms=5   ms=ALL
task1      5   10  1.00   0.001   0.100   0.250   0.400   0.600   0.990   0.990
task2      6   15  1.00   0.001   0.100   0.200   0.300   0.500   0.700   0.990
task3      9   20  1.00   0.001   0.050   0.150   0.250   0.350   0.450   0.990
task4      8   25  1.00   0.001   0.100   0.200   0.300   0.400   0.500   0.990
task5      9   20  1.00   0.001   0.050   0.100   0.200   0.300   0.400   0.990
task6      9   25  1.00   0.001   0.050   0.100   0.200   0.300   0.400   0.990
ex_cust    3   12  1.00   0.001   0.200   0.400   0.990     -       -     0.990
```

Compare ms=2 column: previously the entire column was `0.010` for task2/3/5/6. Now it's `0.05–0.20`. That's the gradient GRPO needs.

## How to reproduce

```bash
git checkout feature/v7-reward-fix
bash hf_job_train_v7_reward_fix.sh
```

The launcher:
- Clones this branch on the HF runner.
- Source-greps `grader.py` to confirm the constants are patched (works without pydantic installed).
- After `pip install`, asserts via Python import that the constants are exactly `0.005 / 0.10 / 0.001`.
- Runs GRPO with `--sft-checkpoint GeminiHugger/war-room-sft-v1-r32` (the upcast-to-r32 version of your SFT adapter), 200 episodes, 9 tasks, identical hyperparams to v6.
- Pushes to `GeminiHugger/war-room-grpo-adapter-v7-rewardfix`.

## What to look at when v7 finishes

The two metrics that prove or disprove the diagnosis:

1. **task2/3/5/6 mean team_reward**: should jump from `0.010` → at least `0.10` if the diagnosis is right.
2. **task1 + example_custom mean team_reward**: should be approximately unchanged (`0.46` and `0.95`). If they collapse, my parameter choices were too aggressive and need to be walked back.

If task2/3/5/6 still don't move, the cold-start hypothesis (model literally cannot emit "tampered", "DNS misconfiguration") is the next thing to test, and your SFT dataset would need targeted samples for those tasks.

## Files added on this branch

- `round2/war_room/grader.py` (modified — 4 constants + comments)
- `round2/war_room/train_colab.py` (modified — 1 line)
- `hf_job_train_v7_reward_fix.sh` (new — launcher with built-in patch verification)
- `docs/V7_REWARD_FIX_HANDOFF.md` (new — this document)
- `outputs/war_room_grpo_v5_alltasks/` (new — v5 baseline metrics + analysis, already on main)

## Live jobs

| Job ID | Branch | Status (when this was written) |
|---|---|---|
| `69ed9454d70108f37acdf848` | v6 (SFT, original reward) | RUNNING |
| `69edb1bdd70108f37acdfbb1` | v7 (SFT + reward fix) | RUNNING |

Both are on `GeminiHugger`. View at `https://huggingface.co/jobs/GeminiHugger/<job_id>`.
