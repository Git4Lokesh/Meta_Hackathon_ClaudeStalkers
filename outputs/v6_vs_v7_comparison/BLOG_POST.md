# Why our 7B GRPO model couldn't learn 4 of our 9 tasks — and how we fixed it without touching the model

*A debugging story from training a multi-agent SRE incident-response model with GRPO on Hugging Face Jobs.*

## The setup

We're training Qwen2.5-7B-Instruct with GRPO (group-relative policy optimization) to act as a 3-agent SRE incident response team — a **triage** agent, a **diagnosis** agent, and a **remediation** agent — that has to solve simulated production incidents.

Each task has a hand-written `MultiAgentGrader` that watches the team's actions and credits *milestones* — discrete things like "triage agent escalated the right service", "diagnosis agent read the relevant log file", "remediation agent restarted the right process". Hit a milestone, get some credit. Miss the whole episode, get punished by a small per-round time-pressure penalty.

We have 9 task families (`task1` through `task6`, plus `example_custom` and two procedural variants). Run number five, **v5**, was supposed to be our cleanest training run yet: 9 tasks, 200 episodes per task, 7,200 episodes total, ~5 hours on an L40S.

## The result that didn't make sense

After v5 finished, we pulled the metrics and got this:

| Task | mean team_reward | what was happening |
|---|---|---|
| `task1` (single fault) | 0.460 | learning fine |
| `example_custom` | 0.946 | basically solved |
| `procedural_hard` | 0.350 | partial |
| `procedural_easy` | 0.140 | weak |
| `task4` (auth failure) | 0.120 | barely learning |
| **`task2` (cascade)** | **0.010** | **flat at floor across all 800 episodes** |
| **`task3` (conflicting info)** | **0.010** | **flat at floor across all 800 episodes** |
| **`task5` (config tampering)** | **0.010** | **flat at floor across all 800 episodes** |
| **`task6` (blame game / DNS)** | **0.010** | **flat at floor across all 800 episodes** |

That `0.010` is suspicious. It's not just low — it's *exactly* the lower clamp of our reward function. Across 800 episodes per task, the model has *never* produced a meaningfully different reward. It's not "learning slowly"; it's "no signal at all."

The first instinct was the obvious one: maybe a 7B model just isn't smart enough for the harder tasks. They are harder! task5 has a "rogue insider tampered with the config" backstory and you need to actually read multiple config files to spot the tampering.

But "model is too small" doesn't predict the *exact* number `0.010` showing up 800 times in a row.

## The diagnosis

So we went and looked at what the model was actually *doing* during these 800 episodes. The trainer logs `milestones_achieved` per episode, and on tasks 5 and 6 the model was hitting **1–3 milestones per episode in 95–98% of episodes**. It was making real progress. It just wasn't being *paid* for that progress.

Tracing through the reward math in `round2/war_room/grader.py`:

```python
raw = milestone_credit - min(penalty_raw, PENALTY_CAP_FRACTION × total_credit)
      + communication_bonus + solve_bonus

penalty_raw  = TIME_PRESSURE_PENALTY × rounds_used + NOOP_PENALTY × noops
             ≈ 0.01 × 20 + 0.01 × 5  =  ~0.25

penalty_cap  = 0.40 × 1.0  =  0.40   # so the cap doesn't kick in
penalty_applied  ≈  0.25
```

For task5, milestones are worth 0.05–0.20 each, total 1.0. Hit 2 milestones (typical), earn ~0.10 credit. Penalty is 0.25. Raw score is **negative**.

Then comes the line that destroys the gradient:

```python
return max(0.01, min(0.99, raw))
```

`max(0.01, -0.15) = 0.01`. Every episode where the agent does meaningful but partial work returns the *same* reward. GRPO computes its policy update from the *variance* of rewards within a group. When every reward in a group is 0.01, variance is zero, and the gradient is zero. The model gets no information about which actions were better than others.

It wasn't a model size problem. It was a reward shaping bug.

## The fix

Three constants in `round2/war_room/grader.py`, plus one matching value in `train_colab.py`:

```diff
- TIME_PRESSURE_PENALTY = 0.01
+ TIME_PRESSURE_PENALTY = 0.005    # halve the per-round time penalty

- PENALTY_CAP_FRACTION = 0.40
+ PENALTY_CAP_FRACTION = 0.10      # cap penalty at 10% of total credit, not 40%

- FATAL_SCORE = 0.01
+ FATAL_SCORE = 0.001              # lower the floor so sub-floor cases are differentiated

  # in current_score():
- return max(0.01, min(0.99, raw))
+ return max(0.001, min(0.99, raw))
```

That's it. No change to the policy, the model size, the curriculum, the SFT adapter, the GRPO loss, or any task definition.

### Why these specific values

The constraint was: **don't break what already works**. v5's task1 was scoring 0.46 and example_custom was scoring 0.95. We can't lift the floor on the hard tasks at the cost of regressing the easy ones.

So before launching v7, we wrote a simulator that takes the actual milestone structures from `task1` through `task6` and `example_custom` and tries five different parameter combinations. The winner had to satisfy:

1. **Preserve the easy tasks**: task1 at full solve still ~0.99, example_custom at full solve still ~0.99.
2. **Restore the gradient on the hard tasks**: task2/3/5/6 must show a *monotonically increasing* reward as more milestones are hit, instead of flat 0.01.
3. **Minimal blast radius**: we want the smallest possible change to the reward function. Three constants is tractable; rewriting the grader is not.

The functional simulation (real graders, real milestones, max time penalty applied) shows what the new curves look like:

```
task     #ms maxR  totC   ms=0    ms=1    ms=2    ms=3    ms=4    ms=5   ms=ALL
task1      5   10  1.00   0.001   0.100   0.250   0.400   0.600   0.990   0.990  ← preserved
ex_cust    3   12  1.00   0.001   0.200   0.400   0.990     -       -     0.990  ← preserved
task2      6   15  1.00   0.001   0.100   0.200   0.300   0.500   0.700   0.990  ← was 0.01 flat
task3      9   20  1.00   0.001   0.050   0.150   0.250   0.350   0.450   0.990  ← was 0.01 flat
task5      9   20  1.00   0.001   0.050   0.100   0.200   0.300   0.400   0.990  ← was 0.01 flat
task6      9   25  1.00   0.001   0.050   0.100   0.200   0.300   0.400   0.990  ← was 0.01 flat
```

Compare the `ms=2` column. Previously the entire column was `0.010` for tasks 2/3/5/6. Now it's `0.05–0.20`. That's the gradient GRPO needs.

## The experiment

To isolate the effect of the reward fix from everything else, we set up a clean head-to-head:

- **v6**: Lokesh's SFT-warm-up branch. Uses the SFT adapter (so the model already knows the output format) + GRPO + 9 tasks + **original reward function**. This is the "what we'd ship without the fix" baseline.
- **v7**: Same SFT warm-up adapter, same 9 tasks, same hyperparameters (LoRA r=32, lr=1e-5, 200 episodes, 4 generations per step), but with the reward fix applied. This is the "single change is the reward function" arm.

Both jobs running on a Hugging Face Jobs L40S, in parallel.

## What we're seeing live (mid-run)

We're snapshotting the runs while they're still training. v6 is at epoch ~0.57; v7 is at epoch ~0.09 because it started later. From the captured step metrics (the `{loss': ..., 'rewards/reward_milestone/mean': ..., ...}` dicts the trainer prints):

| Metric | v6 (orig reward) | v7 (reward fix) |
|---|---|---|
| `reward_milestone/mean` lifetime | 0.319 | **0.413** |
| Fraction of steps with **partial** credit (0.011 < x < 0.95) | 70.4% | **85.7%** |
| Fraction of steps **at floor** (≤0.011) | 18.5% | **7.1%** |
| Fraction of steps near solve (≥0.95) | 11.1% | 7.1% |

The thing we expected to see is showing up. v6's milestone reward is bimodal — either the policy hits the 0.01 floor (model fails entirely) or it nails the full solve (0.99). It rarely sits *in between*, because the reward function literally can't represent "the team made meaningful but incomplete progress."

v7's reward, on the other hand, sits comfortably in the 0.05–0.45 range across most steps. The intermediate region is *populated*. The model is being paid for partial milestone progress, and GRPO has variance to work with.

The grad_norm tells the same story from another angle: 0.63 on v6 vs **2.15 on v7**. Higher gradient norm = the policy is actually getting useful update directions, not staying flat.

## What we're going to look at when both runs finish

The acid test is the per-task `team_reward` distribution. v5 had `task2/task3/task5/task6` pinned at exactly 0.010 across all 800 of their episodes. If the diagnosis is right, then v7 should produce:

- `task2/3/5/6` mean reward **>0.10** (up from 0.010) — the proof that the gradient is restored.
- `task1` and `example_custom` mean reward approximately unchanged from v5 (0.46 and 0.95) — the proof that we didn't break anything.

If task2/3/5/6 still don't move on v7 either, then the next hypothesis up the stack is **cold-start**: the base model literally cannot emit the words those graders look for ("tampered", "DNS misconfiguration"), and we'd need targeted SFT samples for those tasks before GRPO can do anything.

But based on the live numbers, the cold-start hypothesis is looking less likely. The model *is* hitting milestones in v6 already. It just wasn't being paid for it.

## What this debugging episode taught me

1. **Stuck-at-exactly-the-floor metric is always a reward shape problem, not a capacity problem.** Models that can't learn produce noisy rewards near zero, not exactly-floor rewards forever.
2. **GRPO's gradient comes from reward *variance within a group***, not absolute reward magnitude. A flat reward surface gives zero gradient even if the surface is "high enough."
3. **Penalty cap fractions matter more than they look.** A `PENALTY_CAP_FRACTION = 0.40` setting that "preserves time pressure as a signal" can quietly destroy the gradient on long-horizon tasks where penalty saturates.
4. **Always check what your model is *actually doing* before blaming model size.** The model was hitting 1–3 milestones in 95% of episodes on the "stuck" tasks. That is not the behavior of a model that can't learn the task; it's the behavior of a model whose teacher can't tell it whether it did well or not.
5. **Simulate before launching expensive runs.** A 200-line Python script that re-implements the score function and tries five parameter sets caught the right values before we burned $7 on a job that would also fail.

## Reproducing all of this

The actual reward fix is on **`feature/v7-reward-fix`** in [Git4Lokesh/Meta_Hackathon_ClaudeStalkers](https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers/tree/feature/v7-reward-fix). The comparison data (this folder, including the raw HF logs) is on **`feature/v6-v7-comparison`**.

```bash
# Reward fix branch
git checkout feature/v7-reward-fix
bash hf_job_train_v7_reward_fix.sh

# Comparison data branch (this writeup)
git checkout feature/v6-v7-comparison
ls outputs/v6_vs_v7_comparison/
```

The parser is idempotent: re-pull the HF logs and re-run `parse_logs.py` to refresh `summary.json`, the per-step CSVs, and the comparison chart at any point during training.

— *written 2026-04-26 while both jobs were still running*
