# Improving Model Performance — Options Matrix

After our first successful GRPO run (Qwen2.5-7B-Instruct on L40S, 30 episodes × 3 tasks):

| Metric                  | Current  | Target   |
| ----------------------- | -------- | -------- |
| Team reward (mean)      | 0.33–0.41| 0.50+    |
| Milestones / episode    | 2.36 / 4 | 3.5 / 4  |
| Format compliance       | 100%     | 100%     |
| Anti-hack triggers      | 0        | 0        |
| Milestone reward (raw)  | 0.01–0.06| 0.20+    |

The **milestone reward** is the bottleneck. Format and comm are near-ceiling
already; the question is whether the agents actually resolve incidents.

## Option ladder (cheapest → most impactful)

### 1. Run more episodes (cheap — $0.40, 10 min)

Just let the existing pipeline cook longer. `train_colab.py --episodes 60`
gives 180 training prompts instead of 90, same LR, same model. Most of
the gain comes from the first 90 episodes (we have data for 30 episodes ×
3 tasks). Doubling that should push milestone reward from ~0.03 → ~0.10.

    bash hf_job_launch.sh   # with EPISODES=60 in hf_job_launch.sh

**Risk**: low. Same infrastructure as proven run.

### 2. Harder tasks in the mix (cheap — same cost)

Currently training on task1/task2/task3. Adding task4 (simultaneous
incidents) forces the model to split attention and coordinate harder. The
procedural task generator can inject more variety still.

    EPISODES=60 TASKS="task1 task2 task3 task4" bash hf_job_launch.sh

**Risk**: medium. task4 is 25 rounds, longer episodes = slower training.

### 3. Multi-turn rollout (medium complexity — no $ change)

Right now the model only plays round 0 of each episode. Heuristic
co-agents play rounds 1-N and get the reward signal. If the model plays
all rounds, the credit assignment is direct: the model's actions are what
determines the milestones it achieves.

Implementation: `make_rollout_func` in `train_colab.py` already has the
hook; just need to have it re-invoke the model each round with updated
observations, rather than filling in heuristics after round 0.

**Risk**: high. Touches training-time rollout logic, 3-4 hours of dev +
validation. Don't attempt unless option 1 or 2 plateau.

### 4. Bigger LoRA + longer training (higher $ — $1.50, 45 min)

Bump `--lora-r 32` (up from 16) and `--episodes 90`. More LoRA parameters
= more capacity to learn task-specific behavior. Longer training = more
gradient steps against that capacity.

    EPISODES=90 TIMEOUT=90m bash hf_job_launch.sh
    # after editing hf_job_launch.sh to add --lora-r 32

**Risk**: medium. Larger LoRA slightly increases VRAM pressure on L40S
but still fits. Longer training still well within budget.

### 5. SFT warm-up before GRPO (high complexity — $1.80, 60 min)

The 1.5B SFT run failed because of label masking ambiguity, but on 7B
with L40S we have enough VRAM and compute for a proper two-stage run:

1. SFT for 1 epoch on 160 heuristic pairs, learning rate 1e-5, LoRA r=16
2. Resume GRPO from the SFT checkpoint with `--sft-checkpoint`

The combination should push format reward from 1.0 (already at ceiling)
to retain that *and* give GRPO a warm-started policy that already knows
the action space, making milestone reward climb faster.

**Risk**: medium-high. SFT on 7B should be much more stable than 1.5B
(more capacity to absorb the signal without overfitting), but we haven't
validated it. Dev time ~30 min, training ~45 min.

### 6. Curriculum learning (builds on 1 or 4)

`train_colab.py` already has `CurriculumScheduler`. Currently it starts
with task1 only, then adds task2 at 30%, task3/4 at 60%. This prevents
the model from being crushed by hard tasks before it's learned the easy
ones.

The scheduler is already in the code. No work needed; just make sure
`--tasks task1 task2 task3 task4` is passed so the curriculum has all
four to schedule.

### 7. Teacher-forced RL (complex — several hours of dev)

Use the heuristic teacher's action as a "correct" signal mixed with the
model's action. DAGGER-style imitation learning: every few rounds,
replace the model's action with the teacher's for data collection. Lets
the model see expert trajectories without pure SFT on them.

**Not recommended for this deadline** — too much risk for the remaining
time budget.

## Recommended sequence (hackathon path)

Given ~30 hours to pitch and $29 remaining credits:

1. **Commit what we have.** Current training curve is real, defensible,
   and shows non-zero learning. Write the blog post around "format
   compliance is perfect, milestones are the growth edge."
2. **Run option 1 or 2** while you eat/sleep. $0.40 spend, uses ~15% of
   one hour of budget. If reward improves, we swap in the new
   `metrics.json` for the pitch.
3. **Only if step 2 plateaus**, try option 4 (bigger LoRA + longer). One
   more shot, still under $3 total burn.
4. **Do not attempt option 5 (SFT) unless 1-4 all failed.** SFT added
   complexity and three failed runs last time; we have a working pipeline
   now and shouldn't throw that away.

## What we will NOT do

- Re-introduce SFT on the critical path. It caused three failed runs.
- Rewrite the rollout_func. High risk, minimal expected gain.
- Switch base model. Qwen2.5-7B is performing; changing it means
  re-validating everything.
- Add more reward functions. The four we have are well-designed; adding
  a fifth adds variance without adding capability.
