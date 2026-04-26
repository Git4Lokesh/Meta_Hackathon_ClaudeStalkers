# Kiro card — v6 vs v7

**Hypothesis under test**: Tasks 2/3/5/6 stay at the 0.01 reward floor in v5/v6 because the reward function destroys the gradient on partial progress, not because the 7B model is too small.

**Independent variable**: 4 constants in `round2/war_room/grader.py` (and a matching one in `train_colab.py`).
- `TIME_PRESSURE_PENALTY`: 0.01 → 0.005
- `PENALTY_CAP_FRACTION`: 0.40 → 0.10
- `FATAL_SCORE` (lower clamp): 0.01 → 0.001
- `current_score()` lower clamp: 0.01 → 0.001

**Held constant across v6 and v7**: SFT warm-up adapter, base model, LoRA rank 32, lr 1e-5, 200 episodes/task, 9 tasks (task1–6 + example_custom + procedural_easy/hard), 4 generations per step, batch size 1, anti-hack reward, format reward, communication reward.

**Live snapshot numbers**:

```yaml
v6_job_id: 69ed9454d70108f37acdf848
v7_job_id: 69edb1bdd70108f37acdfbb1

v6_at_snapshot:
  epoch: 0.56-0.58
  steps_captured: 27
  reward_milestone_mean_lifetime: 0.319
  partial_credit_step_fraction: 0.704
  at_floor_step_fraction: 0.185
  near_solve_step_fraction: 0.111
  grad_norm_avg: 0.63
  kl_avg: 0.80

v7_at_snapshot:
  epoch: 0.09
  steps_captured: 14
  reward_milestone_mean_lifetime: 0.413
  partial_credit_step_fraction: 0.857
  at_floor_step_fraction: 0.071
  near_solve_step_fraction: 0.071
  grad_norm_avg: 2.15
  kl_avg: 1.49
```

**Predicted final outcomes** (when both jobs finish):

```yaml
v7_predictions:
  task1_mean_team_reward: ~0.50    # was 0.46 in v5; should not regress
  example_custom_mean_team_reward: ~0.95   # was 0.946 in v5; should not regress
  task2_mean_team_reward: ">0.10"   # was 0.010 in v5; gradient restored
  task3_mean_team_reward: ">0.10"   # was 0.010 in v5; gradient restored
  task5_mean_team_reward: ">0.10"   # was 0.010 in v5; gradient restored
  task6_mean_team_reward: ">0.10"   # was 0.010 in v5; gradient restored
```

**Falsification criterion**: If task2/3/5/6 still come out at 0.010 mean on v7 too, the reward shaping was not the bottleneck. The fallback hypothesis is **cold-start** — the base model can't emit the keywords those graders look for — which would imply a different intervention (targeted SFT examples for those tasks).

**Files in this folder for downstream tooling**:

```
summary.json          # the headline numbers, machine-readable
v6_steps.csv          # per-step v6 metrics
v7_steps.csv          # per-step v7 metrics
merged_steps.csv      # v6 + v7 stacked, with `run` column
comparison_charts.png # 4-panel visual summary
v6_raw.log            # raw HF jobs logs at snapshot
v7_raw.log            # raw HF jobs logs at snapshot
parse_logs.py         # idempotent parser to refresh data
```

**Reproduction**:

```bash
git checkout feature/v6-v7-comparison
hf jobs logs 69ed9454d70108f37acdf848 > /tmp/v6.log
hf jobs logs 69edb1bdd70108f37acdfbb1 > /tmp/v7.log
python outputs/v6_vs_v7_comparison/parse_logs.py /tmp/v6.log /tmp/v7.log
```
