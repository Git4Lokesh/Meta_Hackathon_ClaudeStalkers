# Submission Checklist — OpenEnv Hackathon India 2026

**Team**: ClaudeStalkers (Siddharth, Lakshminath, Lokesh) · BITS Pilani Hyderabad
**Environment**: Multi-Agent Incident War Room
**Theme**: #1 Multi-Agent Interactions
**Submission URL**: https://huggingface.co/spaces/brodie1of1/war-room
**Deadline**: 5pm today

---

## Minimum Requirements (non-negotiable per hackathon guide)

| # | Requirement | Status | Evidence link |
|---|---|---|---|
| 1 | Uses OpenEnv latest release | **Met** | `pyproject.toml` declares `openenv-core[core]>=0.2.2`. Environment extends OpenEnv Environment base class in `round2/war_room/environment.py`. |
| 2 | TRL or Unsloth training script, ideally Colab notebook | **Met** | TRL GRPOTrainer in `round2/war_room/train_colab.py`. Colab-runnable notebook at `round2/war_room/train_colab.ipynb` (linked from README). |
| 3 | Evidence of real training (loss + reward plots as committed PNGs) | **Met** | `outputs/war_room_grpo_v3/training_curves.png`, `outputs/llm_eval/v3/head_to_head.png`, `outputs/reward_ablation/ablation_overall.png`, `outputs/generalization_eval/generalization_score.png`. All PNGs committed, all axes labeled, all embedded in README with captions. |
| 4 | Mini-blog on HF OR <2min YouTube video | **Met** (blog) | `Blog.md` at repo root — pushed into the HF Space per organizer's explicit instruction "push Blog.MD into your HF Space". Opens with a 30-second before/after teaser + embedded verbatim rollout trace. ~2.8k words, engineering-log voice. |
| 5 | HF Space deployment | **Met** | https://huggingface.co/spaces/brodie1of1/war-room — OpenEnv-compliant FastAPI + Gradio dashboard, Docker SDK on port 7860. API endpoints: `/api/reset`, `/api/step`, `/api/state`, `/api/schema`, `/api/health`. |
| 6 | README with problem motivation, env explanation, results, all links | **Met** | `README.md` at repo root. Pitch opener answers 4 judge-guide questions in first ~1000 chars (capability gap, agent see-do-reward, what changed after training, why it matters). Rubric-alignment table + environment-design-to-impact table below the fold. Links to Space, Blog, adapter repo, Colab, GitHub. |

**All 6 minimum requirements met.**

---

## Rubric Strongest-Evidence Pointers

**Environment Innovation (40%)** — 6 escalating incident tasks with role-gated partial observability, a dedicated phantom-alert subsystem, belief-state tracker, misdirection / blame-game dynamics, a procedural task generator (5 fault primitives × 10 services × difficulty). Fresh angle on Theme #1 Multi-Agent Interactions. Evidence: `round2/PROBLEM_STATEMENT.md`, `round2/war_room/environment.py`, `round2/war_room/tasks/`, README "What's actually novel" section.

**Storytelling (30%)** — `Blog.md` 30-second teaser with verbatim base-vs-trained rollout excerpt + full worked-example section citing `outputs/worked_example/task2_seed33_rollout.json`. Honest `outputs/RESULTS.md` documents v1→v6-SFT including failures (v4 regression, v5-SFT PEFT-key-naming bug) — credibility as an asset. README pitch opener reshaped for the 3–5 minute skim the guide describes. Gradio dashboard layout gives judges a live-replay feel (chat + diagrams + controls all in one viewport).

**Showing Improvement (20%)** — Head-to-head chart `outputs/llm_eval/v3/head_to_head.png` (base Qwen 7B vs v3 LoRA, 5 seeds × 3 tasks). Composite delta +0.046, task2 delta +0.140 (4× lift on the memory-leak task where the base is near-floor). 60-seed procedural generalisation study at `outputs/generalization_eval/`. Full run log at `outputs/RESULTS.md`. **v6-SFT training in flight — if it beats v3 on composite delta and task2, Hero_Swap Protocol (spec'd in `.kiro/specs/hackathon-final-submission/`) will update all hero references before submit.**

**Reward & Pipeline (10%)** — Four decomposed reward functions (milestone 0.60, format 0.15, communication 0.15, anti-hack 0.10 — multiplicative gate). Reward ablation study at `outputs/reward_ablation/ablation_overall.png` proves each component earns its weight. Oracle-audited verifiers via `scripts/oracle_audit.py` (caught two unreachable milestones before training). SFT warm-up + GRPO pipeline. TRL + Unsloth on HF Jobs L40S. 172 tests.

---

## Final Hour Checks (4:30–4:45pm)

- [ ] Space smoke test re-run: all API endpoints return 2xx, Gradio dashboard loads within 10s, Start → Next → Auto completes an episode without error.
- [ ] Verify the new dashboard UI renders without overflow on a 1080p viewport (chat, status cards, plots all visible at once).
- [ ] Confirm Blog.md is browsable at the Space URL (mirrored from repo).
- [ ] README all links clickable — Space, Blog.md, adapter, Colab, GitHub.
- [ ] If v6-SFT eval completed and hero swap happened, grep for `v3` adapter references across README + Blog + RESULTS and confirm any stale references are inside explicit "previous hero" sections only.
- [ ] `git status` clean on main.
- [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/ -x -q` → 172 passing.
- [ ] Submission URL above matches the URL the team pastes into the submission form.

---

## Open items (non-blocking for submission)

- Colab link public-access check: the README points to `round2/war_room/train_colab.ipynb` on GitHub which is publicly visible as long as the GitHub repo is public. Confirm in a private browser window during final-hour review.
- v6-SFT training still in flight on both `brodie1of1` and `GeminiHugger` accounts. Primary candidate (GeminiHugger) was at epoch 0.26 with avg reward 0.355 (v5 lifetime: 0.195) at the last poll. If it publishes before 3pm, eval and potentially swap. If after 3pm, v3 remains hero (documented risk in spec).

---

*Last updated: before final-hour review.*
*Spec: `.kiro/specs/hackathon-final-submission/`.*
