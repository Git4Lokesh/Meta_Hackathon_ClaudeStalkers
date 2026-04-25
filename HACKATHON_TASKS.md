# Hackathon On-Site Task Split

**Team**: ClaudeStalkers (Siddharth + Lakshminath)
**Time budget**: ~36 hours until pitch
**Priority order**: Validate → SFT warm-start → Procedural tasks → Deploy

## 🚨 Critical Finding from Qwen 1.5B run

`qwen1.5B_output.md` confirms the run produced **zero reward across all steps** (238 lines, `reward: 0`, `loss: 0`, `grad_norm: 0`). The 1.5B model never produced valid `COMMAND:/MESSAGE_TO:/MESSAGE:` format, so the reward function had no signal to optimize.

This is the "success probability must be > 0" failure from the hackathon guide §15. **SFT warm-start before GRPO is now non-negotiable** to avoid repeating this on the 7B model.

---

## Phase 1: Validate Baseline (first 3-4 hours on-site)

**Owner**: Both (pair-program, this is the foundation)

Once HF credits are allocated:

1. Confirm which compute mechanism HF gave us (Jobs / Spaces GPU / dev mode). Ask Scaler team on Discord.
2. Clone repo on the GPU instance: `git clone https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git`
3. Activate 3.12 venv: `cd Meta_Hackathon_ClaudeStalkers && source .venv/bin/activate || python3.12 -m venv .venv && source .venv/bin/activate`
4. Install: `pip install -e .` and training deps per `run_training.sh`
5. Run smoke test on heuristic demo: `PYTHONPATH=. python round2/war_room/demo_comparison.py` — confirms env works on their infra
6. **Do NOT run GRPO training yet** — Phase 2 first

**Done criterion**: demo_comparison prints the 0.01 → 0.80 table, no errors.

---

## Phase 2: Features (split the work)

### Task A — SFT Warm-Start Pipeline (HIGH PRIORITY)
**Owner**: Lakshminath
**Estimated time**: 3-4 hours
**Why it matters**: Without SFT, 7B GRPO will likely collapse to zero reward like 1.5B did.

**Deliverables:**
1. Create `round2/war_room/sft_warmup.py`:
   - Generate 50-100 synthetic Diagnosis completions using the heuristic agents from `train.py` at `skill_level=1.0`
   - Extract (prompt, target_completion) pairs where target is the structured `COMMAND:/MESSAGE_TO:/MESSAGE:` format
   - Save as a HuggingFace dataset
2. Create `round2/war_room/sft_train.ipynb` (Colab-runnable):
   - Load Qwen2.5-7B-Instruct (or Qwen2.5-1.5B-Instruct for quick validation)
   - SFT on the synthetic dataset for 1-2 epochs, LoRA rank 16
   - Save the adapter to `outputs/war_room_sft/`
3. Update `train_colab.py` to accept an optional `--sft-checkpoint` argument that loads the SFT adapter before GRPO

**Success check**: After SFT, running inference on a fresh War Room observation should produce ≥ 60% format compliance (has all three COMMAND/MESSAGE_TO/MESSAGE fields).

### Task B — Procedural Task Generator (MEDIUM PRIORITY)
**Owner**: Siddharth
**Estimated time**: 4-5 hours
**Why it matters**: Converts the environment from RLVR (6 fixed tasks) to RLVE (infinite procedurally generated tasks). Directly hits the RLVE theme the judging criteria rewards.

**Deliverables:**
1. Create `round2/war_room/tasks/procedural.py` with a `ProceduralTask(difficulty: float)` class that:
   - Randomly picks a fault type: `crash`, `memory_leak`, `cascade`, `auth_failure`
   - Randomly picks 1-3 services to fault based on difficulty
   - Injects 0-4 red herrings (phantom alerts) based on difficulty
   - Computes milestones procedurally from the fault injection
2. Extend `WAR_ROOM_TASK_REGISTRY` with `procedural` as a task_id that instantiates `ProceduralTask`
3. Update `CurriculumScheduler` to sample procedural tasks with increasing difficulty
4. Write 5-10 property-based tests in `tests/property/test_procedural.py` using hypothesis

**Success check**: `env.reset(task_id="procedural", seed=N)` produces a valid episode for any seed N, with milestones that can be achieved by the heuristic agents.

### Task C — Deploy to HF Spaces (QUICK WIN)
**Owner**: Siddharth (during idle time while training runs)
**Estimated time**: 1-2 hours

**Deliverables:**
1. Create HF Space at `huggingface.co/spaces/brodie1of1/war-room` (if not already)
2. Push the current repo to the Space: `git remote add hf https://huggingface.co/spaces/brodie1of1/war-room && git push hf main`
3. Verify the Space builds and runs on port 7860
4. Verify the HTML homepage loads, `/health` returns OK, and `/reset` works
5. Update README with the live Space URL

**Success check**: `curl https://brodie1of1-war-room.hf.space/health` returns 200 OK.

---

## Phase 3: Scale Training (after Task A completes)

**Owner**: Lakshminath (runs training), Siddharth (monitors, handles failures)
**Estimated time**: 2-4 hours (depends on compute allocation)

1. Run SFT warm-up on Qwen2.5-7B → produces `outputs/war_room_sft/adapter_model.safetensors`
2. Run full GRPO with `--sft-checkpoint outputs/war_room_sft` → produces `outputs/war_room_grpo/metrics.json`
3. Run `python round2/war_room/generate_charts.py` → produces `training_curves.png`
4. Commit both to repo
5. Uncomment plot embeds in README

**Success check**: `training_curves.png` shows reward trending upward over episodes (not flat at zero).

---

## Phase 4: Stretch Goals (only if Phase 1-3 complete with time to spare)

### Task D — Train 2 agents simultaneously (HIGH RISK, HIGH REWARD)
**Owner**: Both
**Estimated time**: 6-8 hours

Extend `train_colab.py` to train Diagnosis AND Remediation with separate LoRA adapters. Triage stays heuristic. Use alternating GRPO updates. **Only attempt if everything else is done AND we have >8 hours left.** High chance of convergence issues.

### Task E — Rollout inspection dashboard
**Owner**: whoever finishes first
**Estimated time**: 2 hours

Add a page to `gradio_app.py` that shows the rollout audit log (sampled completions + reward breakdowns + anti-hack triggers) during training. Visually impressive for the pitch.

---

## Phase 5: Pre-Pitch (last 4-6 hours)

**Owner**: Both (pair work)

1. Pitch practice — minimum 3 full run-throughs, time under 3 minutes
2. Record backup demo video in case live demo fails
3. Update `pitch_outline.md` with actual training numbers
4. Polish README with final metrics, link the HF Space, link the blog, link the Colab notebook
5. Final `git push` and lock the repo — **no more commits** after submission deadline

---

## Hard Rules

1. After EVERY task, run `PYTHONPATH=. python -m pytest tests/ -x -q`. If tests fail, revert the commit.
2. Commit to main only after local tests pass. No broken builds on main.
3. If GRPO training starts producing non-zero rewards, **stop adding features** and let training run. Real training evidence > theoretical features.
4. Communicate on Discord before touching shared files (train_colab.py, environment.py, grader.py). Non-overlapping files are fair game.
5. If stuck for >30 min on any task, ask the other person or skip to the next priority.

---

## Git Branching Strategy

Both of you work on `main`. Pull before every work session. If conflicts arise:
- On merge conflict, communicate first, don't resolve blindly
- Training outputs (`outputs/`) are in `.gitignore` — only commit the final `metrics.json` and `training_curves.png` from the winning run
