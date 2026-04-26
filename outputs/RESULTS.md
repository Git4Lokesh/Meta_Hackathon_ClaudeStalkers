# Training run archives

Every GRPO run we've done, in chronological order, with headline metrics and what we learned. The `current best` call-out points at whichever adapter wins the head-to-head eval today.

## Head-to-head summary (base Qwen 2.5-7B-Instruct vs each adapter)

Composite score = average of task1/task2/task3 scripted-task eval across 5 seeds per task.

| Adapter | Training shape | Episodes | Composite delta | Task2 lift | Notes |
|---|---|---|---:|---:|---|
| v1 | Round-0 diagnosis only, strict format | 91 | **−0.017** | − | Train-eval shape mismatch; first iteration |
| v2 | Procedural-only, round-0 only | 300 | **−0.001** | − | Same shape bug, bigger budget |
| v3 | **Multirole** structured completion | 300 | **+0.046** | **+0.140 (4×)** | Fixed train-eval alignment — first positive |
| multirole_v2 (Lakshminath) | Multirole, 6-task mix | 800 | **+0.021** | +0.062 | More tasks = more breadth but slightly lower transfer |
| **v4** (brodie1of1) | Multirole + reward surgery, rank 32 | 800 | _eval in flight_ | _eval in flight_ | Same codepath as multirole_v2 + penalty cap + solve bonus; task4 no longer stuck at 0.01 in training |
| **v5-SFT** (brodie1of1) | **SFT warm-up + GRPO** with relaxed task3 | ~300 | _training in flight_ | _pending_ | SFT on 355 oracle examples lands `eval_loss=0.024`; task3 pushback should fire for the first time because SFT teaches "Redis is NOT the real issue" |

## Current best: `v3` on the public head-to-head eval

- Adapter: [`brodie1of1/war-room-grpo-adapter-v3`](https://huggingface.co/brodie1of1/war-room-grpo-adapter-v3)
- Base Qwen 7B composite: 0.269
- v3 composite: 0.315
- Delta: **+0.046** (+17% relative)
- Biggest lift: task2 (memory leak + red herring) 0.048 → 0.188, a 4× improvement

See `outputs/llm_eval/v3/` for the full head-to-head chart, `results.json` (per-rollout rows), `summary.json` (aggregates). See `outputs/worked_example/task2_seed33_rollout.json` for a verbatim before/after trace of base vs v3 on one specific seed — the trace is what the blog post section "What the model actually learned" is built from.

## Per-run details

### v1 (deprecated) — `outputs/war_room_grpo_v1_broken/`

- First adapter we trained. Delta was −0.017.
- Root cause: training only graded a single Diagnosis completion at round 0, while the head-to-head eval runs the model for all three roles (Triage / Diagnosis / Remediation) across all rounds. Training was optimising a strictly different problem from what eval measured.
- Secondary issue: `_EPISODE_TELEMETRY` empty because TRL's `GRPOTrainer.__init__` rejected the `rollout_func` kwarg and silently fell back to its default rollout. Made `rounds_used` and `milestones_achieved` columns all zero in metrics.json.

### v2 — procedural-only, same shape bug

- Same train-eval shape mismatch as v1, just with 300 steps and procedural task sampling.
- Delta −0.001. Slightly better but still net negative.
- Confirmed more training doesn't help if the reward signal doesn't correspond to the eval metric.

### v3 — multirole completion (first positive)

- Adapter: `brodie1of1/war-room-grpo-adapter-v3`
- Training prompt rewritten to ask for a structured `### TRIAGE / ### DIAGNOSIS / ### REMEDIATION` block that drives all three agents at round 0 of each episode.
- 100 episodes × 3 procedural difficulties = 300 gradient updates. ~25 minutes on L40S.
- Head-to-head delta: +0.046 composite, +0.140 on task2 specifically.
- **This is the first adapter that's better than base** and remains the public `current best` until v4/v5-SFT results land.

### multirole_v2 (Lakshminath, `GeminiHugger/war-room-grpo-multirole-v2`)

- 200 episodes × 6 task mix (task1, task3, task4, procedural_easy, procedural_hard, example_custom).
- 2h 36m on L40S.
- Per-task training `team_reward` mean:
  - example_custom: 0.85 (saturated, demo task is easy)
  - procedural_hard: 0.34
  - task1: 0.29
  - procedural_easy: 0.09
  - task3: 0.01 (verifier fix hadn't landed yet)
  - task4: 0.01 (fatal-check interaction, see v4)
- Head-to-head eval: composite 0.289 vs base 0.269 → delta **+0.021**.
- Lower transfer to scripted eval than v3 despite 2.7× more training steps. Hypothesis: the broader task mix causes the model to memorise specific scripted patterns rather than generalise, which then doesn't transfer to held-out eval seeds. Procedural-only training (v3) transferred better.

### v4 — multirole + reward surgery

- Adapter: `brodie1of1/war-room-grpo-adapter-v4`
- Same shape as multirole_v2 but with:
  - Reward surgery: `SOLVE_BONUS = 0.10` when all milestones hit, `PENALTY_CAP_FRACTION = 0.40` cap on time-pressure + noop penalties as a fraction of available milestone credit.
  - LoRA rank 16 → 32.
  - LR 5e-6 → 1e-5.
- 200 episodes × 6 tasks = 800 gradient updates on L40S.
- Training metrics (from `brodie1of1/war-room-grpo-adapter-v4/metrics.json`):
  - Overall `team_reward` mean: **0.338** (up from multirole_v2's 0.263)
  - Quartile progression: 0.293 → 0.358 → 0.359 → 0.344 (rose then plateaued)
  - **task4 no longer stuck at 0.01** — now 0.109 mean, 3.6 milestones/episode. Reward-surgery fixed the 25-round-penalty swamp.
  - example_custom: 0.924 (vs 0.845)
  - procedural_hard: 0.496 (vs 0.339)
  - task3 still 0.01 — verifier relax landed _after_ v4's training code was frozen.
- Head-to-head eval: in flight (job 69ed889ed2c8bd8662bcf088).

### v5-SFT — SFT warm-up then GRPO (in flight)

- Adapter target: `brodie1of1/war-room-grpo-adapter-v5-sft`
- Pipeline:
  1. **SFT warm-up** on 355 oracle-generated multirole examples with per-task validation thresholds (`outputs/sft_dataset/train.jsonl`). Produced `brodie1of1/war-room-sft-v1` with `eval_loss=0.024`, `mean_token_accuracy=0.991` in 5.3 minutes on L40S.
  2. **GRPO** on top of the SFT checkpoint: 100 episodes × 6 tasks, rank 32, LR 1e-5, same reward-surgery config as v4, on the **relaxed task3 verifier** (accepts `"red herring"`, `"not the root"`, `"phantom"`, etc. — 20 dismissal phrases in total).
- Hypothesis: GRPO on the SFT checkpoint starts from a model that already emits correct multirole format and hits task3 pushback milestones. That gives GRPO successful rollouts to reinforce rather than starting from the noise floor on task3.
- Smoke check (before launching the full run): a single SFT completion on `task3 seed=56227` hits 5 milestones including `diagnosis_pushback_bonus` at round 1. So the reward signal is live on task3 for the first time in any of our runs.

## Reproduction commands

```bash
# Smoke training (validates pipeline on procedural_easy in ~3 min on L40S)
STAGE=smoke REPO_NAME=war-room-grpo-smoke \
  TASKS="procedural_easy" EPISODES=5 \
  bash hf_job_train_v2.sh

# v3 configuration (our current public best)
STAGE=full REPO_NAME=war-room-grpo-adapter-v3 \
  TASKS="procedural_easy procedural_medium procedural_hard" \
  EPISODES=100 \
  bash hf_job_train_v2.sh

# v4 configuration (reward surgery + rank 32)
bash hf_job_train_v4.sh

# SFT warm-up (builds the dataset + trains the SFT adapter)
PYTHONPATH=. python scripts/build_sft_dataset.py --seeds-per-task 60
bash hf_job_sft.sh

# v5-SFT (GRPO on the SFT checkpoint)
bash hf_job_train_v5_sft.sh

# Head-to-head eval on any adapter
ADAPTER_REPO=brodie1of1/war-room-grpo-adapter-v3 \
  UPLOAD_REPO=brodie1of1/war-room-eval-results \
  bash hf_job_llm_eval.sh
```

## Evidence files

| File | What it shows |
|---|---|
| `outputs/war_room_grpo_v3/metrics.json` | v3 training per-episode telemetry (1200 rows) |
| `outputs/war_room_grpo_v3/training_curves.png` | v3 training curves |
| `outputs/llm_eval/v3/head_to_head.png` | v3 head-to-head chart (the README/blog hero image) |
| `outputs/llm_eval/v3/results.json` | v3 per-rollout rows (30 rollouts: 3 tasks × 5 seeds × 2 models) |
| `outputs/llm_eval/v3/summary.json` | v3 aggregates + composite delta |
| `outputs/worked_example/task2_seed33_rollout.json` | Verbatim base vs trained rollout used in the blog post |
| `outputs/war_room_grpo_multirole_v2/metrics.json` | Lakshminath's multirole_v2 training telemetry (4800 rows) |
| `outputs/war_room_grpo_multirole_v2/training_curves.png` | multirole_v2 training curves |
| `outputs/war_room_grpo_v1_broken/metrics.json` | v1 broken telemetry (kept for contrast) |
| `outputs/reward_ablation/ablation_overall.png` | Reward ablation chart — turning off each component |
| `outputs/reward_ablation/ablation_per_task.png` | Reward ablation per task |
| `outputs/generalization_eval/generalization_score.png` | 60-seed procedural generalisation chart |
| `outputs/generalization_eval/generalization_eval.json` | Raw 60-seed generalisation data |
