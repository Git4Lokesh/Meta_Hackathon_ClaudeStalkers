# Training Run Archives

This folder holds the historical GRPO training runs for the War Room
adapter. Each run lives in its own subdirectory so future runs don't
overwrite earlier results.

## Active runs

### `war_room_grpo_multirole_v2/` ← current best

- **Adapter on Hub:** [GeminiHugger/war-room-grpo-multirole-v2](https://huggingface.co/GeminiHugger/war-room-grpo-multirole-v2)
- **Episodes:** 200 (4800 reward calls × 4 generations / step)
- **Tasks:** task1, task3, task4, procedural_easy, procedural_hard, example_custom
- **Runtime:** 2h 36m on L40S
- **Headline metrics** (from `metrics.json`):

  | metric | value |
  | --- | --- |
  | `team_reward` mean | 0.263 (max 0.99) |
  | `rounds_used` mean | 12.6 |
  | `milestones_achieved` mean | 2.78 |
  | `format_reward_avg` mean | 1.00 |
  | `communication_reward_avg` mean | 0.71 |
  | anti-hack triggers | 0 / 4800 |

- **Per-task `team_reward` mean:**

  | task | mean reward | mean milestones |
  | --- | --- | --- |
  | example_custom | 0.85 | 2.89 |
  | procedural_hard | 0.34 | 6.31 |
  | task1 | 0.29 | 2.59 |
  | procedural_easy | 0.09 | 2.17 |
  | task3 | 0.01 | 0.00 (known unreachable milestones) |
  | task4 | 0.01 | 2.70 |

- **What's included:**
  - `adapter_model.safetensors` — LoRA weights (rank 16)
  - `adapter_config.json` — adapter spec
  - `metrics.json` — per-episode telemetry
  - `training_curves.png` — reward / loss / KL plots
  - `baseline_vs_trained.png` — head-to-head visualization
  - tokenizer files

### `war_room_grpo_multirole_smoke/` ← validation run for v2

- **Adapter on Hub:** [GeminiHugger/war-room-grpo-multirole-smoke](https://huggingface.co/GeminiHugger/war-room-grpo-multirole-smoke)
- **Episodes:** 40 (smoke test before launching v2)
- **Purpose:** Verify the multi-role + telemetry fixes worked before
  spending L40S credits on the full run.
- Only `metrics.json`, `training_curves.png`, `adapter_config.json`
  are kept here (full adapter is on the Hub if needed).

## Archived runs

### `war_room_grpo_v1_broken/`

The previous v1 run, kept for reference. It has two known issues that
were fixed in v2:

1. **Train-eval mismatch** — only graded a single Diagnosis completion at
   round 0 while eval calls the LLM 3× per round (Triage / Diagnosis /
   Remediation). Completion length stayed at ~37 tokens; the model never
   learned the multi-role plan format.
2. **`_EPISODE_TELEMETRY` empty** — `rounds_used` and
   `milestones_achieved` recorded as 0 for every episode in
   `metrics.json` because TRL's `GRPOTrainer.__init__` rejected the
   `rollout_func` kwarg and fell back to its default rollout.

Both issues are resolved in `war_room_grpo_multirole_v2/`. See the
commit history of `round2/war_room/train_colab.py` for the fix
(structured multi-role completion + inline-path telemetry).

## How to reproduce

```bash
# Smoke (40 ep, ~30 min, ~$1 on L40S)
STAGE=smoke REPO_NAME=war-room-grpo-multirole-smoke \
  TASKS="task1 task3 procedural_easy procedural_hard example_custom" \
  EPISODES=40 TIMEOUT=45m bash hf_job_train_v2.sh

# Full (200 ep, ~2.5h, ~$5 on L40S)
STAGE=full REPO_NAME=war-room-grpo-multirole-v2 \
  TASKS="task1 task3 task4 procedural_easy procedural_hard example_custom" \
  EPISODES=200 TIMEOUT=4h bash hf_job_train_v2.sh
```
