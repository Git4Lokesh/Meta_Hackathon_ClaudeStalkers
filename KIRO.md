# KIRO.md — train v7-SFT yourself

Hey Lokesh — quick instructions to train v7 (SFT warm-up + reward shaping fix + 9 tasks) on your own HF account / GPU.

If you just want to run it, skip to **[Run it](#run-it)**.

## What v7 is

Same training recipe as v6, with one change: the reward function in `round2/war_room/grader.py` was destroying the gradient on tasks 2/3/5/6 (they were stuck at exactly `score=0.01` for 800 episodes each in v5). The fix is 4 constants:

```diff
# round2/war_room/grader.py
- TIME_PRESSURE_PENALTY = 0.01
+ TIME_PRESSURE_PENALTY = 0.005

- PENALTY_CAP_FRACTION = 0.40
+ PENALTY_CAP_FRACTION = 0.10

- FATAL_SCORE = 0.01
+ FATAL_SCORE = 0.001

  # current_score():
- return max(0.01, min(0.99, raw))
+ return max(0.001, min(0.99, raw))

# round2/war_room/train_colab.py
- "env_reward": float(max(0.01, min(0.99, score))),
+ "env_reward": float(max(0.001, min(0.99, score))),
```

Same model (Qwen2.5-7B-Instruct), same LoRA rank (32), same lr (1e-5), same 200 episodes per task, same 9 tasks, same SFT warm-up adapter, same `--generations 4 --batch-size 1`.

Full diagnosis & validated reward curves: [`docs/V7_REWARD_FIX_HANDOFF.md`](docs/V7_REWARD_FIX_HANDOFF.md).

## Prerequisites

1. **HF token in your shell** — already have `hf` CLI logged in or `HF_TOKEN` env var set.
2. **The upcast SFT adapter must exist on the Hub.**
   - Currently published at `GeminiHugger/war-room-sft-v1-r32` (your r=16 SFT zero-padded to r=32 so it loads into the r=32 GRPO model).
   - If you want to use your own account's copy, run `scripts/upcast_sft_adapter.py` first — it pulls `brodie1of1/war-room-sft-v1` and pushes the upcast version to whatever repo you set in `ADAPTER_REPO`.
3. **GPU**: `l40sx1` works (~$7, ~3.5–4.5h). `h100x1` is ~2× faster for ~$8. The launcher defaults to L40S.

## Run it

```bash
git checkout feature/v7-reward-fix
git pull

# Default: trains on YOUR HF account, GPU = L40S, pushes to your repo
ADAPTER_REPO=<your-username>/war-room-grpo-adapter-v7-rewardfix \
    bash hf_job_train_v7_reward_fix.sh
```

That's it. The launcher will:

1. Submit an HF job, clone this branch on the runner, install deps.
2. **Source-grep `grader.py`** to confirm the patch is there before doing anything expensive.
3. **Assert via Python import** after `pip install` that `TIME_PRESSURE_PENALTY=0.005`, `PENALTY_CAP_FRACTION=0.10`, `FATAL_SCORE=0.001`. Hard-fails if not.
4. Run GRPO for 200 episodes × 9 tasks (~1,800 steps).
5. Generate training curves.
6. Push the adapter to `$ADAPTER_REPO`.

## Knobs you might want to tweak

All are env vars before the `bash` command:

| var | default | what to change it to |
|---|---|---|
| `FLAVOR` | `l40sx1` | `h100x1` for ~2× speed, ~$8 |
| `TIMEOUT` | `5h` | `3h` if H100 (it'll finish faster) |
| `EPISODES` | `200` | `100` for a quick smoke test (~$3, ~2h) |
| `TASKS` | all 9 | quote-list to subset, e.g. `"task5 task6"` to focus on the previously-stuck ones |
| `MODEL` | `Qwen/Qwen2.5-7B-Instruct` | another base model if you want |
| `SFT_CHECKPOINT` | `GeminiHugger/war-room-sft-v1-r32` | your own r=32 SFT adapter |
| `LORA_R` | `32` | must match `SFT_CHECKPOINT`'s rank |
| `LR` | `1e-5` | `5e-6` if you see KL spike too high |
| `ADAPTER_REPO` | `GeminiHugger/war-room-grpo-adapter-v7-rewardfix` | **change to your account!** |

Example with H100 + your account:

```bash
FLAVOR=h100x1 \
TIMEOUT=3h \
ADAPTER_REPO=lokesh/war-room-grpo-adapter-v7-rewardfix \
    bash hf_job_train_v7_reward_fix.sh
```

## Monitoring during training

```bash
# List your jobs
hf jobs ps

# Tail the logs (replace JOB_ID with what was printed)
hf jobs logs <JOB_ID>

# Pull comparison data live (uses Lakshminath's already-running v6/v7)
git checkout feature/v6-v7-comparison
bash outputs/v6_vs_v7_comparison/poll_and_push.sh
```

The poll script also drops `summary.json` with the latest milestone-reward distribution at every poll. The single-line "is it learning" check is:

```python
import json
s = json.load(open("outputs/v6_vs_v7_comparison/summary.json"))
print(s["v7_summary"]["lifetime_means"])
```

## What "success" looks like

When the job finishes, pull the adapter's `metrics.json` from your HF repo and check per-task means. The pre-registered prediction:

| Task | v5 (broken reward) | v6 (broken reward + SFT) | v7 (fixed reward + SFT, prediction) |
|---|---|---|---|
| task1 | 0.460 | similar | ≈0.50 (preserved) |
| example_custom | 0.946 | similar | ≈0.95 (preserved) |
| task2 | 0.010 | likely 0.010 | **>0.10** ⭐ |
| task3 | 0.010 | likely 0.010 | **>0.10** ⭐ |
| task5 | 0.010 | likely 0.010 | **>0.10** ⭐ |
| task6 | 0.010 | likely 0.010 | **>0.10** ⭐ |

If task2/3/5/6 still come out at 0.010 on v7 too, the fallback hypothesis is **cold-start** (base model can't emit the keywords those graders look for). At that point we need targeted SFT examples for those tasks, not another reward tweak.

## After training: push results to GitHub

To save your run's metrics and adapter info to GitHub so we can compare:

```bash
# 1. Pull the adapter's metrics from HF
huggingface-cli download <your-repo>/war-room-grpo-adapter-v7-rewardfix \
    metrics.json training_curves.png \
    --local-dir outputs/v7_lokesh

# 2. Add a brief RESULTS.md (mirror outputs/war_room_grpo_v5_alltasks/RESULTS.md format)

# 3. Commit on a fresh branch
git checkout -b results/v7-lokesh
git add outputs/v7_lokesh/
git commit -m "v7-rewardfix results from Lokesh's training run"
git push -u origin results/v7-lokesh
```

Or just push the raw `outputs/v7_lokesh/metrics.json` — the README/analysis can be auto-generated later. Don't worry about formatting it nicely.

## Branch map

| branch | purpose |
|---|---|
| **`feature/v7-reward-fix`** | reward fix code (this branch you're on) |
| `feature/v6-v7-comparison` | live polling data + comparison artifacts (the 4-panel chart, summary.json, etc.) |
| `feature/grpo-multirole-outputs-fast` | your SFT branch — v5 + v6 baselines |

## Live jobs (Lakshminath's account)

- **v6** (your SFT, original reward): `69ed9454d70108f37acdf848` — RUNNING, ~epoch 0.85
- **v7** (your SFT + reward fix): `69edb1bdd70108f37acdfbb1` — RUNNING, ~epoch 0.30

Latest live numbers always at: `outputs/v6_vs_v7_comparison/summary.json` on the comparison branch (refreshed every 10 min).

## Questions / sanity checks

If anything fails, the most likely culprits in order:

1. **Wrong HF token / no permission to push to `ADAPTER_REPO`** → set `ADAPTER_REPO` to a repo your token can write to.
2. **SFT adapter rank mismatch** → if you bring your own SFT, make sure it's r=32 (matches the GRPO model). Use `scripts/upcast_sft_adapter.py` if you only have r=16.
3. **L40S OOM** → drop `--generations 4` to `--generations 2` (edit the launcher).
4. **Patch verification fails at job start** → you forgot `git pull` and are running an old branch.

Ping me if anything else breaks.
