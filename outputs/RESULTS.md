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
| v4 (brodie1of1) | Multirole + reward surgery, rank 32 | 800 | **−0.007** | −0.005 | Training metrics better than multirole_v2 (mean 0.338 vs 0.263, task4 unstuck). But transfer to scripted eval was worse — broader 6-task mix seems to hurt out-of-distribution generalisation compared to v3's procedural-only curriculum. |
| v5-SFT (brodie1of1) | SFT warm-up + GRPO, rank 16→32 | 100 | **regressed (bug)** | n/a | **Failed silently due to PEFT key-naming bug.** SFT adapter trained fine (`eval_loss=0.024`) but `load_state_dict(strict=False)` in the GRPO loader dropped all 392 SFT keys because PEFT saves as `…lora_A.weight` but the live `PeftModel` expects `…lora_A.default.weight` (adapter-name segment). Training ran on the base model with no SFT warm-start. Task2/task3 stuck at 0.01. Fix landed in commit `55e71c8`: key rename on load + rank upcast script. |
| v6-SFT (brodie1of1, **cancelled**) | SFT + GRPO, fixed loader, rank 32, lr 1e-5 | 200 (reached epoch 0.46) | **cancelled** | n/a | Backup run. Same pipeline as Lakshminath's v6-SFT but on brodie1of1 account. At epoch 0.46 after 4h, milestone reward was hitting the 0.01 floor on ~40% of rollouts — the exact issue Lakshminath's v7 reward-fix targets. Would not have finished before the 5pm submission deadline. Cancelled to save budget and trust v7 as the candidate. |
| **v6-SFT (GeminiHugger, in flight)** | SFT + GRPO, fixed loader, rank 32 | _see poll log_ | _training_ | _training_ | **Primary SFT candidate.** Early training signal at epoch 0.26: avg reward 0.355, good% 50% (vs v5 lifetime 0.195 / 21%). SFT + original reward shaping. |
| **v7-rewardfix (GeminiHugger, in flight)** | SFT + GRPO + **reward surgery**, rank 32 | 200 | _training_ | _training_ | **Primary candidate overall.** Same pipeline as v6-SFT-gemini plus `TIME_PRESSURE_PENALTY 0.01→0.005`, `PENALTY_CAP_FRACTION 0.40→0.10`, `FATAL_SCORE 0.01→0.001`. Designed to unstick task2/3/5/6 which sat flat at the 0.01 reward floor across all 800 episodes of v5 despite the model hitting 1-2 milestones per episode. **07:40Z live snapshot (v7 epoch 0.23 vs v6 epoch 0.72):** v7 milestone mean is 0.594 — **2.6× v6's 0.230 at triple the training depth**. v7 has produced **zero at-floor rollouts across three consecutive snapshots**; v6 held at 18.5% → 25% → 25.7% at-floor as training progressed. v7 hit its first near-solve rollout at epoch 0.23. If v7 beats v3's +0.046 composite delta on head-to-head eval, it becomes hero. |

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

### v5-SFT — SFT warm-up then GRPO (FAILED silently — documented for honesty)

- Adapter: `brodie1of1/war-room-grpo-adapter-v5-sft` (keep published as evidence; its metrics file is the paper trail)
- Intended pipeline:
  1. SFT warm-up on 355 oracle-generated multirole examples (`outputs/sft_dataset/train.jsonl`). Produced `brodie1of1/war-room-sft-v1` with `eval_loss=0.024`, `mean_token_accuracy=0.991` in 5.3 minutes on L40S. **This part worked.**
  2. GRPO on top of the SFT checkpoint. **This part silently dropped the SFT warm-start.**
- What broke: the GRPO loader called `model.load_state_dict(state, strict=False)` on the SFT adapter weights. PEFT saves state-dict keys as `base_model.model.model.layers.X...lora_A.weight` but a live `PeftModel` stores adapter weights in a `nn.ModuleDict({"default": ...})` — so the in-memory keys have an extra `.default.` segment: `...lora_A.default.weight`. With `strict=False`, all 392 SFT keys were reported as "unexpected" and ignored. Training ran on the base model instead of the SFT-warmed model. Task2 and task3 stuck at 0.01 throughout.
- Fix (commit `55e71c8`): the loader now renames keys on load, inserting `.default` before `.weight` on `lora_A` / `lora_B` keys. A companion script `scripts/upcast_sft_adapter.py` zero-pads rank-16 SFT to rank-32 for GRPO when ranks differ. A pre-flight verifier `scripts/verify_sft_load.py` asserts the load succeeded.
- Why this matters for the submission: SFT → GRPO is a published technique in the hackathon-relevant literature, and a silent key-naming mismatch between libraries is exactly the kind of bug that makes RL look dead when the issue is really load plumbing. We're leaving this run in the archive because debugging it produced the loader fix and the verification scripts that v6-SFT depends on.

### v6-SFT — fixed loader, SFT warm-up, GRPO on top (in flight)

- Two parallel runs with the same pipeline, different accounts:
  - `brodie1of1/war-room-grpo-adapter-v6-sft` — backup. 200 episodes × 9 tasks, rank 32, lr 1e-5, 5h timeout.
  - `GeminiHugger/war-room-grpo-adapter-v6-sft` — primary, owned by Lakshminath.
- Uses the commit-`55e71c8` loader (PEFT key-rename + rank upcast).
- Pre-flight verified: upcast script zero-pads r=16 → r=32 correctly producing 392 keys; every key matches expected PEFT format after `.default` rename; config shows r=32, lora_alpha=32.
- Training poll on Lakshminath's run (T+60min, epoch ≈ 0.26):
  - Avg reward climbing monotonically: 0.248 → 0.317 → 0.325 → 0.355
  - Floor% stable/low: 36% → 33% → 26% → 29%
  - Good% (rollouts ≥0.5) jumping: 26% → 14% → 33% → 50%
  - **v5 lifetime avg reward was 0.195; v6-SFT at 26% epochs is already at 0.355 — 82% higher.**
- Head-to-head eval will launch when training completes. Hero_Swap Protocol (per `.kiro/specs/hackathon-final-submission/design.md`) swaps v3 out iff v6-SFT composite delta > +0.046 AND task2 delta ≥ 0.

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
