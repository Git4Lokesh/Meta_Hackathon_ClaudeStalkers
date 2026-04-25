# Hackathon Task Split — Live Update

**Team**: ClaudeStalkers
**Time budget**: ~36 hours until pitch

## 🎯 Updated Strategy: Validate on 1.5B NOW

**Why not wait for HF credits?**

The original 1.5B run collapsed because the model couldn't produce valid format. We've now shipped two fixes:

1. **SFT warm-up pipeline** — teaches format before GRPO kicks in
2. **Lenient-format reward** — gives non-zero signal even for imperfect output

Either alone could solve it. Both together almost certainly will. And Colab's free T4 means we can validate the full pipeline RIGHT NOW without burning expensive A100 credits.

**The plan**: run SFT→GRPO on 1.5B first to prove the pipeline works and get an upward reward curve. Then when A100 credits arrive, re-run with 7B for the real-deal submission results.

---

## Lokesh's Tasks (active in Kiro)

### ✅ DONE (already committed + pushed)
- Anti-hack detection module
- Decomposed reward functions (format, milestone, communication, anti-hack)
- Rollout auditor + adaptive curriculum
- OpenEnv Environment base class inheritance (Python 3.12)
- SFT dataset builder (`build_sft_dataset.py`)
- SFT training notebook (`sft_train.ipynb`)
- Lenient-format fallback reward
- `--sft-checkpoint` and `--lenient-format` CLI flags in `train_colab.py`
- Procedural task generator (`tasks/procedural.py`) — 4 fault types × difficulty scaling
- 20 new tests (166 total passing)

### 🎯 NEXT (while Lakshminath handles deployment + slides)

**Task X1 — Plain-Python SFT script** (30 min)
Create `round2/war_room/sft_train.py` as a non-notebook alternative to `sft_train.ipynb`. Useful if we end up running on HF Jobs or SSH rather than Colab.

**Task X2 — Live belief tracker panel in Gradio** (2 hours)
Extend `gradio_app.py` to show the `BeliefStateTracker` state live as episodes play. This is the "jaw drop" moment for the pitch — visually showing agents updating beliefs and pushing back on phantom alerts.

**Task X3 — Multi-turn rollout support** (3-4 hours, stretch)
Currently the rollout runs the model on round 0, then heuristics for later rounds. A proper multi-turn rollout where the model plays all rounds would be more impressive. High risk of breaking things — only attempt after training on 1.5B validates.

---

## Lakshminath's Tasks (parallel, non-overlapping)

**These don't touch any code Lokesh is working on — zero merge conflict risk.**

### Task L1 — Run 1.5B Validation Training (PRIORITY, ~30 min)

Open Colab with the latest repo (after Lokesh's most recent push, commit `c8e2ae6`):

```bash
# Cell 1: clone + install
!git clone https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git
%cd Meta_Hackathon_ClaudeStalkers
!pip install -q "trl>=0.15.0" "peft>=0.14.0" "transformers>=4.46.0" datasets accelerate bitsandbytes
!pip install -e . --quiet

# Cell 2: build SFT dataset (small, fast)
!PYTHONPATH=. python round2/war_room/build_sft_dataset.py --output outputs/sft_dataset.json --pairs-per-task 40

# Cell 3: SFT on 1.5B (~10 min on T4)
# Edit sft_train.ipynb to use Qwen/Qwen2.5-1.5B-Instruct, then run all cells

# Cell 4: GRPO with SFT checkpoint + lenient format (insurance)
!PYTHONPATH=. python round2/war_room/train_colab.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 15 \
    --tasks task1 task2 \
    --sft-checkpoint outputs/war_room_sft \
    --lenient-format \
    --no-unsloth

# Cell 5: generate charts and download
!PYTHONPATH=. python round2/war_room/generate_charts.py
from google.colab import files
files.download('outputs/war_room_grpo/metrics.json')
files.download('outputs/war_room_grpo/training_curves.png')
```

**Success criterion**: `rewards/reward_fn/mean` column in training logs is > 0 at any step. If yes, commit the files to the repo.

**Failure criterion**: Still all-zero after 20 steps. Message Lokesh on Discord immediately — we need to debug.

### Task L2 — Deploy to HF Spaces (1-2 hours)

Create HF Space at `huggingface.co/spaces/brodie1of1/war-room`:

```bash
# From the repo root
git remote add hf https://huggingface.co/spaces/brodie1of1/war-room
git push hf main
```

Verify:
- Space builds without errors
- `https://brodie1of1-war-room.hf.space/health` returns 200 OK
- `https://brodie1of1-war-room.hf.space/` shows the HTML landing page
- Reset/step work via Swagger at `/docs`

If it fails to build, check the Dockerfile logs in the Space console. Likely fix is missing dependencies in `requirements.txt`.

### Task L3 — Record Backup Demo Video (1 hour)

Screen-record ~2 minutes showing:
1. `demo_comparison.py` output (the 0.01 → 0.80 improvement table)
2. `gradio_app.py` running locally with an agent playing an episode
3. Key moment: trained agent pushing back on phantom alert (task 3)
4. Training curve from L1 (if available)

Upload to YouTube unlisted. Add the link to README. This is insurance in case the live demo fails on pitch day.

### Task L4 — Create Pitch Slides (2-3 hours)

Use `round2/war_room/pitch_outline.md` as the content template. Create 5-6 slides in Google Slides or Keynote:

1. Title + hook (3 AM incident story)
2. Environment architecture diagram
3. 6 escalating tasks + Theory of Mind innovation
4. 5-signal reward design
5. Before/after training results (include `baseline_vs_trained.png` from `outputs/war_room_grpo/`)
6. Try-it-yourself (HF Space URL + GitHub link)

Keep it clean, dark theme, large type. Embed the video link from L3 on the last slide.

---

## Hard Rules

1. **Test after every commit**: `source .venv/bin/activate && PYTHONPATH=. python -m pytest tests/ -x -q`. If broken, revert.
2. **Don't both push training metrics** — only one of us commits the final `metrics.json` + `training_curves.png`. Agree on Discord who pushes.
3. **Sync on Discord before touching `environment.py`, `grader.py`, `train_colab.py`** — these are stable and Lokesh is actively editing them.
4. **Lock the repo after submission deadline.** No more commits.

---

## Communication Channel

**On Discord** (`hackathon` channel): status updates every hour, or whenever a task completes. Lokesh posts git commit hashes when he pushes.

---

## Post-Validation (after Task L1 produces real curves)

Decision tree:
- **If 1.5B training shows clear upward curve** → Lokesh tunes hyperparams, Lakshminath starts pitch slides (L4)
- **If 1.5B curve is flat but non-zero** → we have *something* to show. Good enough. Wait for A100 credits for the real run.
- **If 1.5B still zero even with SFT + lenient** → we have a bug. Debug together on Discord.

When A100 credits arrive, swap `--model Qwen/Qwen2.5-1.5B-Instruct` for `--model Qwen/Qwen2.5-7B-Instruct` and rerun. Same pipeline, bigger model.
