# v7 reward-fix training (full 200-ep × 9-task run)

**Source:** [GeminiHugger/war-room-grpo-adapter-v7-rewardfix](https://huggingface.co/GeminiHugger/war-room-grpo-adapter-v7-rewardfix) (pulled 2026-04-26).

**Config (from training pipeline / job):** Qwen2.5-7B-Instruct, GRPO + LoRA, SFT warm-up; `grader.py` v7: `TIME_PRESSURE` 0.01→0.005, `PENALTY_CAP` 0.40→0.10, `FATAL` 0.01→0.001, training clamp 0.001. **`metrics.json` has 7,200 rows = 9 tasks × 800 rows per task** (one row per GRPO **episode** in the run — here 200 scheduler passes × 4 `num_generations` = 800 steps of environment interaction per task family, matching the v5/v6 logging shape).

## Headline training-time numbers (from `metrics.json`)

| Stat | Value |
|------|------:|
| Rows (episodes logged) | 7,200 |
| `team_reward` mean (all) | 0.460 |
| `team_reward` min / max | 0.001 / 0.990 |

## Per-task mean `team_reward` (n = 800 each)

| Task | Mean | Notes |
|------|------:|--------|
| example_custom | 0.971 | Saturated |
| procedural_hard | 0.761 | Strong |
| task1 | 0.724 | |
| task2 | **0.565** | Large lift vs v5/v6 floor era |
| task4 | 0.481 | |
| task5 | 0.101 | Still hard |
| task6 | 0.234 | |
| procedural_easy | 0.292 | |
| **task3** | **0.012** | Still near floor on average; reward-fix unlocks *some* partial credit, not “solved” |

**Last 5 `team_reward` in file:** 0.585, 0.001, 0.001, 0.001, 0.001 (typical bimodal GRPO tail).

## Files in this folder

| File | Purpose |
|------|--------|
| `metrics.json` | Full per-episode/rollout metrics (training-time) |
| `training_curves.png` | TRL-generated plots |
| `adapter_config.json` | PEFT / LoRA config |
| `job_tail.log` | Last ~2k lines of the HF job log (progress + step dicts) |
| `rollout_audit.jsonl` | Empty in this push (0 bytes) — no sampled audit in Hub snapshot |

**Large files** (`adapter_model.safetensors`, tokenizers) are in this directory when pulled via `hf download` but are **not committed to Git**; pull from the Hub link above.

## What to run next

- **LLM head-to-head eval** (same script as v3 / v6): compare base vs this adapter on scripted task1–3. Training-time means here do *not* guarantee eval transfer (see v7-fast and README iteration notes).

## Job reference

- HF Job ID (historical): `69edb1bdd70108f37acdfbb1` (completed; `hf jobs ps` no longer lists it after retention).
