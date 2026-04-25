# Hackathon On-Site Task Split — Updated

**Team**: ClaudeStalkers
**Active on-site**: Lokesh (+ Kiro)
**Time budget**: ~36 hours until pitch

## 🚨 Critical Finding from Qwen 1.5B run

`qwen1.5B_output.md` shows **zero reward across all 238 steps** (`reward: 0`, `loss: 0`, `grad_norm: 0`). The 1.5B model never produced valid `COMMAND:/MESSAGE_TO:/MESSAGE:` format, so GRPO had no gradient signal.

**Implication**: before running ANY GRPO training on 7B, we need either (a) an SFT warm-start to teach format, or (b) a more forgiving reward function that gives non-zero signal for partial format compliance. **Going in without these is high-risk.**

---

## Revised Plan (Lokesh + Kiro)

### Phase 0: Pre-Credit Work (do NOW, no GPU needed)

**High-impact, you + Kiro pair-program these:**

#### Task A — SFT Warm-Start Pipeline
**Why**: Prevents repeating the zero-reward collapse on 7B. The hackathon guide explicitly recommends "tiny amount of task-format SFT" before RL.

**Deliverables** (Kiro will scaffold, you review and commit):
1. `round2/war_room/build_sft_dataset.py` — generates 100-200 synthetic (prompt, target) pairs from heuristic agents at skill_level=1.0
2. `round2/war_room/sft_train.ipynb` — Colab-runnable notebook that SFTs Qwen2.5-7B-Instruct on the dataset (LoRA rank 16, 1-2 epochs)
3. Update `train_colab.py` to accept `--sft-checkpoint` flag that loads the SFT adapter before GRPO

**Done when**: Dataset generates 100+ valid pairs locally, notebook runs without errors (can be validated on-site once GPU is available).

#### Task B — Procedural Task Generator
**Why**: Converts environment from RLVR (6 fixed tasks) → RLVE (infinite procedurally generated tasks). The judging criteria explicitly rewards RLVE-style adaptive environments.

**Deliverables** (Kiro scaffolds, you review):
1. `round2/war_room/tasks/procedural.py` — `ProceduralTask(difficulty: float)` class
2. Randomly injects fault types: `crash`, `memory_leak`, `cascade`, `auth_failure`
3. Randomly adds 0-4 phantom alerts based on difficulty
4. Register `"procedural"` in `WAR_ROOM_TASK_REGISTRY`
5. Update `CurriculumScheduler` to sample procedural tasks
6. Property tests in `tests/property/test_procedural.py` (5-10 tests using hypothesis)

**Done when**: `env.reset(task_id="procedural", seed=N)` works for any N, all 146 existing tests still pass.

#### Task C — Fallback Reward Relaxation (insurance)
**Why**: If SFT warm-start doesn't work, we need a backup plan. A more forgiving reward gives the model *some* signal even with imperfect format.

**Deliverables**:
1. Add a `--lenient-format` flag to `train_colab.py`
2. When set, format_reward gives partial credit: 0.3 for any of (COMMAND / MESSAGE) keywords present, 1.0 for perfect format
3. Keep the strict version as default

**Done when**: Training with `--lenient-format` gives non-zero reward signal even on degenerate outputs.

---

### Phase 1: Validate on Compute (when credits arrive, ~3-4 hours)

**Priority order on the A100 (or whatever HF gives):**

1. **Smoke test** (5 min): `PYTHONPATH=. python round2/war_room/demo_comparison.py` → should print the 0.01 → 0.80 table
2. **SFT warm-up** (20-40 min): Run the SFT notebook from Task A with Qwen2.5-7B-Instruct → produces `outputs/war_room_sft/`
3. **Verify format** (5 min): Run inference on 20 War Room observations with the SFT model → confirm ≥ 60% format compliance
4. **GRPO training** (45-90 min): `python round2/war_room/train_colab.py --episodes 30 --sft-checkpoint outputs/war_room_sft`
5. **Generate charts** (2 min): `python round2/war_room/generate_charts.py`
6. **Commit and push** (2 min): commit `metrics.json` and `training_curves.png`
7. **Update README** (5 min): uncomment the plot embeds

**If Step 2 or 4 fails**: fall back to `--lenient-format` from Task C, try again.

---

### Phase 2: Deploy + Polish (1-2 hours)

1. **Push to HF Spaces** (15 min):
   ```bash
   git remote add hf https://huggingface.co/spaces/brodie1of1/war-room
   git push hf main
   ```
   Verify the Space builds and runs.

2. **Polish README** (15 min):
   - Add final training numbers to results section
   - Uncomment plot embeds
   - Verify all links work (HF Space, blog, notebook, Colab badge)

3. **Record backup video** (30 min): Screen-record the demo_comparison + gradio dashboard in case live demo fails

---

### Phase 3: Pitch Prep (final 4-6 hours)

**Do at least 3 full practice runs of the 3-minute pitch:**

1. Lead with the 3 AM incident story (30 sec)
2. Show environment architecture briefly (45 sec)
3. **Live demo moment**: trained agent pushing back on a phantom alert — this is your jaw-drop moment (60 sec)
4. Show training curves and metrics (30 sec)
5. Close with "first multi-agent Theory of Mind environment on OpenEnv" (15 sec)

**Q&A prep is in `round2/war_room/pitch_outline.md`** — review it.

---

## Stretch Goals (only if Phase 1 produces real curves AND you have >8 hours left)

### Task D — Train 2 agents simultaneously (HIGH RISK)
Extend `train_colab.py` to train Diagnosis AND Remediation with separate LoRA adapters. Alternating GRPO updates. Only attempt if everything is stable and you have buffer time.

### Task E — Live belief tracker demo in Gradio
Add a live panel to `gradio_app.py` that shows the Belief State Tracker updating in real-time as the trained agent plays. Visually impressive for pitch.

---

## Hard Rules

1. **Test after every change**: `PYTHONPATH=. python -m pytest tests/ -x -q`. If broken, revert.
2. **Don't touch `environment.py`, `grader.py`, `anti_hack.py`** unless absolutely needed — these are stable and tested.
3. **If GRPO starts producing non-zero rewards, STOP adding features** and let it train. Real training evidence beats theoretical code.
4. **No commits to main after submission deadline**. Lock it.
5. **If stuck >30 min**, pivot to the next task or ask for help on Discord.

---

## Immediate Next Action (right now, before credits arrive)

Start Task A (SFT dataset builder). Ask Kiro to scaffold `round2/war_room/build_sft_dataset.py`. Review the code, commit. Then move to Task B (procedural tasks). This puts you in the strongest possible position when credits drop.
