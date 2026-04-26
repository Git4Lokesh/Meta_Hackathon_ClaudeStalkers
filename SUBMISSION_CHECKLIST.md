# Submission Checklist — OpenEnv Hackathon India 2026

**Team**: ClaudeStalkers (Siddharth, Lakshminath, Lokesh) · BITS Pilani Hyderabad
**Environment**: Multi-Agent Incident War Room
**Theme**: #1 Multi-Agent Interactions
**Submission URL**: https://huggingface.co/spaces/brodie1of1/war-room
**Deadline**: 26 April 5 PM IST

---

## Minimum Requirements (non-negotiable per hackathon guide)

| # | Requirement | Status | Evidence link |
|---|---|---|---|
| 1 | Uses OpenEnv latest release — build on top, don't reinvent | **Met** | `pyproject.toml` declares `openenv-core[core]>=0.2.2`. Environment extends OpenEnv Environment base class in `round2/war_room/environment.py`. Valid `openenv.yaml` manifest at repo root; entry point `round2.war_room.hf_space_app:app` imports cleanly as FastAPI. |
| 2 | Working training script using TRL / Unsloth / any RL framework, ideally a Colab notebook judges can re-run | **Met** | TRL GRPOTrainer in `round2/war_room/train_colab.py`. **Publicly clickable Colab link** in README header: https://colab.research.google.com/github/Git4Lokesh/Meta_Hackathon_ClaudeStalkers/blob/main/round2/war_room/train_colab.ipynb |
| 3 | Evidence of real training (loss + reward plots, at minimum) | **Met** | `outputs/war_room_grpo_v3/training_curves.png`, `outputs/llm_eval/v3/head_to_head.png`, `outputs/reward_ablation/ablation_overall.png`, `outputs/generalization_eval/generalization_score.png`, `outputs/v6_vs_v7_comparison/comparison_charts.png`. All PNGs committed, all axes labeled with units, all embedded with one-line captions. |
| 4 | Mini-blog on HF **OR** <2min YouTube video (one is sufficient) | **Met** (**both**) | **📺 YouTube walkthrough: https://youtu.be/B2tvMdbr7AE** (1.25-1.5× speed recommended). `Blog.md` at repo root — pushed into the HF Space per organizer's explicit instruction "push Blog.MD into your HF Space". Opens with a 30-second before/after teaser + embedded verbatim rollout trace. |
| 5 | MUST push environment to a Hugging Face Space — discoverable and runnable | **Met** | https://huggingface.co/spaces/brodie1of1/war-room — OpenEnv-compliant FastAPI + Gradio dashboard, Docker SDK on port 7860. API endpoints: `/api/reset`, `/api/step`, `/api/state`, `/api/schema`, `/api/health`. Manifest in `openenv.yaml`. Space repo size: 7.3 MB (well under limits). |
| 6 | README MUST link to the HF Space + all additional materials | **Met** | `README.md` at repo root. First-1000-chars pitch opener + Rubric-alignment table + "What's actually novel" section with three probe questions answered. Links to Space, Blog, adapter repo, publicly-runnable Colab, GitHub. |
| 7 | Experimental tracking turned on for training runs | **Met** (via committed artifacts) | See "Experimental tracking" section below for how we report training progress without a W&B link. |
| 8 | Don't include big video files in the env submission | **Met** | Space repo: 7.3 MB total, no video files. Largest file is `outputs/sft_dataset/train.jsonl` at 1.1 MB. Any video-style content is linked externally (none currently). |

**All 8 minimum requirements met.**

---

## Experimental tracking

We did not use W&B or TensorBoard for hosted tracking. Instead, every training run commits its full telemetry directly to this repo so judges (and we) can audit without a third-party dependency:

| Artifact | What it contains | For every run |
|---|---|---|
| `outputs/<run>/metrics.json` | Per-episode team_reward, rounds_used, milestones_achieved, per-reward-component means, loss | v3, v4, multirole_v2, v5 |
| `outputs/<run>/rollout_audit.jsonl` | Per-rollout raw completions + scored reward breakdown | v3, v4, multirole_v2 |
| `outputs/<run>/training_curves.png` | Reward curve, per-reward breakdown, milestones bar | v3, v4, multirole_v2 |
| `outputs/v6_vs_v7_comparison/` | Live HF Jobs log → CSV/JSON + 4-panel comparison chart + raw log files | v6 and v7 (in-flight) |
| `outputs/RESULTS.md` | Human-readable run log with per-run config + verdict + honest failure reporting (v4 regression, v5-SFT PEFT bug) | All runs |

This choice was deliberate: committed artifacts can't be deleted or lose their links the way deleted W&B runs can, and the `parse_logs.py` script in `outputs/v6_vs_v7_comparison/` is idempotent so anyone can refresh a snapshot from raw HF Jobs logs.

---

## One Submission Per Team — team leader responsibilities

Per organizer NOTE 2: only the team leader's submission will be accepted. Team members cannot submit on behalf of the team.

- [ ] **Team leader confirmed**: who is clicking "submit" on the form?
- [ ] Team leader has the URL `https://huggingface.co/spaces/brodie1of1/war-room` ready to paste into the submission form.
- [ ] Team leader has read this checklist and confirms all rows are Met before submitting.

---

## Rubric Strongest-Evidence Pointers

**Environment Innovation (40%)** — 6 escalating incident tasks with role-gated partial observability, a dedicated phantom-alert subsystem, belief-state tracker, misdirection / blame-game dynamics, a procedural task generator (5 fault primitives × 10 services × difficulty). Fresh angle on Theme #1 Multi-Agent Interactions. The README "What's actually novel" section answers the three hackathon probe questions explicitly: teaches something LLMs can't do well (base Qwen gets fooled 70% of the time on task3), underexplored domain (no comparable open SRE-incident multi-agent env), paper-worthy building blocks (phantom alerts + cross-role pushback + RLVE curriculum).

**Storytelling (30%)** — `Blog.md` 30-second teaser with verbatim base-vs-trained rollout excerpt + full worked-example section citing `outputs/worked_example/task2_seed33_rollout.json`. Honest `outputs/RESULTS.md` documents v1→v7 including failures (v4 regression, v5-SFT PEFT-key-naming bug, v6-SFT-brodie cancelled when we realized it would miss deadline) — credibility as an asset. README pitch opener reshaped for the 3–5 minute skim the guide describes. Gradio dashboard layout gives judges a live-replay feel (chat + diagrams + controls all in one viewport).

**Showing Improvement (20%)** — Head-to-head chart `outputs/llm_eval/v3/head_to_head.png` (base Qwen 7B vs v3 LoRA, 5 seeds × 3 tasks). Composite delta +0.046, task2 delta +0.140 (4× lift on the memory-leak task where the base is near-floor). 60-seed procedural generalisation study at `outputs/generalization_eval/`. Full run log at `outputs/RESULTS.md`. Live v6/v7 training snapshot at `outputs/v6_vs_v7_comparison/` — v7 with reward surgery hits **0% at-floor rollouts** at epoch 0.17 vs v6's 25% at epoch 0.67, proving the reward-fix hypothesis. If v6-SFT or v7-rewardfix finishes and evaluates strictly better than v3 before submission, Hero_Swap Protocol (spec'd in `.kiro/specs/hackathon-final-submission/`) updates all hero references.

**Reward & Pipeline (10%)** — Four decomposed reward functions (milestone 0.60, format 0.15, communication 0.15, anti-hack 0.10 — multiplicative gate). Reward ablation study at `outputs/reward_ablation/ablation_overall.png` proves each component earns its weight. Oracle-audited verifiers via `scripts/oracle_audit.py` (caught two unreachable milestones before training). SFT warm-up + GRPO pipeline with the PEFT key-rename fix (landed commit `55e71c8`) and rank-upcast tool. v7 reward surgery (`grader.py`) documents a clean, minimal fix — 4 constants — that unsticks task2/3/5/6 from the 0.01 floor. TRL + Unsloth on HF Jobs L40S. 172 tests.

---

## Final Hour Checks (4:30–4:45pm IST)

- [ ] Space smoke test re-run: all API endpoints return 2xx, Gradio dashboard loads within 10s, Start → Next → Auto completes an episode without error.
- [ ] Verify the new dashboard UI renders without forced-scroll on a 1080p viewport.
- [ ] Confirm Blog.md is browsable at the Space URL (mirrored from repo).
- [ ] README all links clickable — Space, Blog.md, adapter, Colab, GitHub.
- [ ] Verify the Colab link opens a runnable notebook in a private browser session (logged-out state) so we know judges can open it without our auth.
- [ ] If v6-SFT or v7 eval completed and hero swap happened, grep for old adapter URLs across README + Blog + RESULTS and confirm any stale references are inside explicit "previous hero" sections only.
- [ ] `git status` clean on main. `git log origin/main..HEAD` is empty.
- [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/ -x -q` → 172 passing.
- [ ] Submission URL above matches the URL the team leader pastes into the submission form.

---

## Open items (non-blocking for submission)

- **Team leader identification**: who is clicking "submit" at 4:45pm IST? Per organizer NOTE 2 only the team leader's submission counts.
- **v7 training still in flight** on `GeminiHugger` account. Latest snapshot at epoch 0.17 shows 0% at-floor rollouts (vs v6 at 0.67 showing 25% at-floor). If it publishes before ~3:30pm IST, we eval and potentially swap. If later, v3 remains documented hero with v7 in-flight documented in RESULTS as a late-arriving artifact.
- **v6-SFT-brodie cancelled** at epoch 0.46 (was hitting 0.01 floor 40% of rollouts — the exact problem v7's reward fix targets). Saved ~$3.50. This is a research decision, documented in RESULTS.md, not a gap.

---

*Last updated: 2026-04-26 during final preparation.*
*Spec: `.kiro/specs/hackathon-final-submission/`.*
