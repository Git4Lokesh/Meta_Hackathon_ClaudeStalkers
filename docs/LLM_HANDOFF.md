# War Room / OpenEnv — LLM Handoff Summary

**Purpose:** Self-contained brief for another model or teammate. No prior chat context required.

---

## 1. What this project is

- **Round 2 “War Room”:** A multi-agent SRE incident simulator (Triage, Diagnosis, Remediation) built on a Python “environment-native” stack (`sre_env` + `round2/war_room`).
- **Training:** GRPO (Group Relative Policy Optimization) on **Qwen2.5-7B-Instruct** with **LoRA** (rank 16), using TRL’s `GRPOTrainer`.
- **Claim:** Treat the stack as an **OpenEnv-style** platform: procedural/custom tasks, curriculum sampling, HF Jobs for GPU training, metrics and charts.

---

## 2. Repository map (high signal)

| Area | Path |
|------|------|
| Simulated infra / commands | `sre_env/server/` (`simulated_system.py`, `command_parser.py`) |
| War room env + rewards | `round2/war_room/environment.py` |
| GRPO training script | `round2/war_room/train_colab.py` |
| Task registry | `round2/war_room/tasks/__init__.py` |
| Scripted tasks | `round2/war_room/tasks/task*.py` |
| Procedural generator + fault primitives | `round2/war_room/tasks/procedural.py` |
| Custom task example | `round2/war_room/tasks/example_custom_task.py` |
| Generalization eval (heuristic baseline) | `round2/war_room/eval_generalization.py` |
| LLM eval (API / MLX) | `round2/war_room/eval_llm_head_to_head.py`, `eval_llm_on_gpu.py` |
| HF Jobs launcher | `hf_job_train_v2.sh` |
| Unit tests | `tests/unit/` |

---

## 3. Tasks and “datasets”

- There is **no static dataset** for new tasks in the classical sense. Prompts come from **environment observations** after `reset(task_id, seed)`; rewards come from **milestone / team reward** in the grader.
- **Scripted:** `task1`–`task4` (curated scenarios).
- **Procedural:** `procedural_easy`, `procedural_hard`, etc. — faults sampled from primitives (`crash`, `memory_leak`, `cascade`, `auth_failure`, `disk_full`, …).
- **User extension:** Register a task class in `tasks/__init__.py`; optional milestone helpers live in `procedural.py` (e.g. `triage_mentions`, `diagnosis_says_about`, …).
- **Curriculum:** `CurriculumScheduler` in `train_colab.py` respects `allowed_tasks` and can uniform-sample custom task lists.

---

## 4. Training pipeline (current behavior)

### 4.1 Multi-role structured completion (train–eval alignment)

**Problem (v1):** Training effectively graded a **single Diagnosis-style** completion at round 0 while eval / live agents use **three roles per round**. TRL often **dropped** `rollout_func` from `GRPOTrainer.__init__`, so the custom rollout rarely ran; rewards came from `_milestone_reward_inline`, which still only reflected diagnosis-centric behavior and **short** completions (~37 tokens).

**Fix (v2):** One completion must drive **all three agents at round 0**:

- System prompt: `MULTIROLE_SYSTEM_PROMPT` in `train_colab.py`.
- User prompt: **Triage + Diagnosis + Remediation** observations + triage handoff (`generate_training_dataset`).
- Model output: three blocks — `### TRIAGE`, `### DIAGNOSIS`, `### REMEDIATION`, each with `COMMAND:` / `MESSAGE_TO:` / `MESSAGE:`.
- Parsing: `_parse_multirole_completion` → `MultiAgentAction` for round 0; later rounds use **fault-aware heuristics** (legacy heuristics for task1–4) so episodes can reach late milestones.
- `multirole=False` path still exists for backward compatibility / tests.

### 4.2 Metrics telemetry

**Problem:** `_EPISODE_TELEMETRY` stayed empty when only the inline reward path ran → `metrics.json` had `rounds_used` and `milestones_achieved` as **0** everywhere.

**Fix:** `_milestone_reward_inline` **appends** to `_EPISODE_TELEMETRY` per completion (task, seed, env_reward, rounds_used, milestones_hit). Final `metrics.json` writer reads that list.

### 4.3 Other training-related knobs

- `MAX_EPISODE_ROUNDS = 16` in `train_colab.py` (was 8; longer tasks were truncating).
- `environment.py`: `metadata` always includes `score` and `milestones_achieved` even when `done` is false (external round cap).
- Reward fns: `reward_milestone`, `reward_format` / `reward_format_lenient`, `reward_communication`, `reward_anti_hack` — format rewards updated for multi-role structure.

---

## 5. HF Jobs and artifacts

- **Script:** `hf_job_train_v2.sh` — clones `main` from GitHub, installs TRL/peft stack, runs `train_colab.py`, charts, uploads folder to a **HF model repo**.
- **Pinned TRL:** `trl>=0.15,<0.19` in the job (API drift handled in `_build_trainer` in `train_colab.py`).

**Published adapters (user `GeminiHugger` on HF):**

| Repo | Role |
|------|------|
| `war-room-grpo-multirole-smoke` | Short smoke before full run |
| `war-room-grpo-multirole-v2` | **Main** full run (200 episodes config; metrics show 4800 rows = reward calls with `generations=4`) |

**Local mirror (after download):**

- `outputs/war_room_grpo_multirole_v2/` — adapter weights, tokenizer, `metrics.json`, plots.
- `outputs/war_room_grpo_multirole_smoke/` — lighter smoke snapshot.
- `outputs/war_room_grpo_v1_broken/` — renamed old `war_room_grpo` (broken metrics / wrong training shape).
- `outputs/RESULTS.md` — human-readable index of runs.

---

## 6. Results snapshot (v2, from `metrics.json`)

- **Rows:** 4800 (not 200 “episodes” — one row per reward-group sample path).
- **Global:** `team_reward` mean ~0.26, max 0.99; `rounds_used` mean ~12.6, **no all-zero rows**; `milestones_achieved` mean ~2.8; format lenient ~1.0; communication ~0.71; anti-hack triggers 0 in that run.
- **Per-task means (illustrative):** strong on `example_custom` (~0.85 team reward); `procedural_hard` shows high milestone counts (~6.3 mean); `task3` still **flat** (known grader / scenario issues — do not interpret as “multi-role failed” without separate oracle audit).

---

## 7. Known limitations / follow-ups

1. **Eval vs training topology:** Training uses **one** structured completion for **round 0** for all roles; live eval may still call the model **per role per round**. Alignment is **much** better than v1 but not bitwise identical to “3× calls every round.”
2. **Task3 / task4** in v2 metrics: investigate grader reachability and prompts (teammate scripts: `scripts/oracle_audit.py`, `scripts/verify_gradient.py`).
3. **`rollout_func`:** May still be ignored by TRL; training is designed to work via **inline** milestone path + telemetry.
4. **`rollout_audit.jsonl`:** Can be empty on HF upload depending on job; primary signal is `metrics.json` + TRL `log_history`.

---

## 8. Commands cheat sheet

```bash
# Tests
PYTHONPATH=. python -m pytest tests/unit -q

# Local sanity (imports, parsers, small episode — no GPU required for pieces)
PYTHONPATH=. python -c "from round2.war_room.train_colab import generate_training_dataset; print(len(generate_training_dataset(prompts_per_task=2)))"

# HF training (interactive confirm in script)
STAGE=full REPO_NAME=war-room-grpo-multirole-v2 \
  TASKS="task1 task3 task4 procedural_easy procedural_hard example_custom" \
  EPISODES=200 TIMEOUT=4h bash hf_job_train_v2.sh

# Pull adapter locally
hf download GeminiHugger/war-room-grpo-multirole-v2 --local-dir outputs/war_room_grpo_multirole_v2
```

---

## 9. Git pointer

Multi-role + telemetry fixes landed on **`main`** as commit **`6fa5fe8`** (“train: structured multi-role completion + populate _EPISODE_TELEMETRY”). Subsequent commits may exist; treat `train_colab.py` on `main` as source of truth.

---

## 10. What to tell the next LLM to do first

1. Read `round2/war_room/train_colab.py` — search `MULTIROLE_SYSTEM_PROMPT`, `_parse_multirole_completion`, `_milestone_reward_inline`, `_EPISODE_TELEMETRY`.
2. Read `outputs/RESULTS.md` and skim `outputs/war_room_grpo_multirole_v2/metrics.json` (large JSON — sample or aggregate with a small script).
3. If improving eval parity: align `eval_llm_head_to_head.py` / `live_agent.py` prompts with the same `### TRIAGE` / `### DIAGNOSIS` / `### REMEDIATION` contract **or** document the intentional difference.

---

*Generated for handoff. Repo: Meta_Hackathon_ClaudeStalkers.*
