# Pitch Deck — Multi-Agent Incident War Room

3 min pitch, 5 content slides + 1 closing. Copy the content into Keynote or Google Slides, dark theme. Each slide has a single visual and a single sentence of script.

---

## Slide 1 — The Hook

**Visual**: a Slack-style incident channel with three agent avatars messaging each other. Phantom Redis alert banner in the corner.

**Headline**: *When the monitoring dashboard lies, can AI agents push back on each other?*

**Script (20s)**:

> "At 3 AM when production goes down, it's never one person who fixes it. It's a team — one reading alerts, one digging through logs, one applying the fix. They communicate, and they push back when the evidence doesn't match the alarm. We built the environment that trains LLMs to do exactly that."

---

## Slide 2 — The Environment

**Visual**: the three-agent diagram from the README (Triage / Diagnosis / Remediation → Communication Channel → Simulated System). Each agent's capabilities in a box.

**Headline**: *Strict role-based partial observability forces cooperation.*

**Script (30s)**:

> "Three agents, each sees a different slice. Triage sees the dashboard but not logs. Diagnosis reads logs but can't restart services. Remediation fixes things but is blind to what's broken. They can only coordinate through a shared message board — and that message board is part of the action space, with its own reward signal."

---

## Slide 3 — The Innovation: Theory of Mind

**Visual**: screenshot of the Belief State Tracker showing Redis=fine (ground truth) but Triage believes Redis=critical (stale metric). Purple "Theory of Mind moment" callout showing Diagnosis pushed back.

**Headline**: *Phantom alerts force agents to reason about others' false beliefs.*

**Script (45s)**:

> "The environment injects phantom alerts — stale cached metrics that look alarming but aren't real. Triage sees "Redis memory 72%!" and panics. Diagnosis checks the Redis logs, finds nothing wrong, and has to make a choice: follow the panicked teammate, or push back with evidence? We built a Belief State Tracker that records what every agent thinks vs ground truth, and a Deception Resistance Score that measures pushback quality. This is Theory of Mind made trainable."

---

## Slide 4 — The Reward, Proved

**Visual**: the two ablation charts side by side (`outputs/reward_ablation/ablation_overall.png` + `ablation_per_task.png`). Tiny table caption.

**Headline**: *5 independent reward signals. Ablated and proved non-redundant.*

**Script (30s)**:

> "Five reward streams: milestone progress, format, communication quality, anti-hack gate, deception resistance. We ablate each one across fixed seeds — removing the communication bonus drops Task 2 by 22%. Removing the anti-hack gate doesn't affect heuristics but matters for RL. Every component earns its weight."

---

## Slide 5 — Training Results

**Visual**: two panels side by side — left: `outputs/war_room_grpo/training_curves.png` (GRPO reward curves), right: `outputs/generalization_eval/generalization_score.png` (baseline vs trained across difficulties).

**Headline**: *Real training. Real generalization. Real numbers.*

**Script (40s)**:

> "We trained Qwen2.5-7B with GRPO on an L40S via HuggingFace Jobs — 91 episodes, $1.10 in compute. Composite score on scripted tasks: 0.01 to 0.80. On 60 procedurally generated unseen scenarios: baseline flatlines at 0.01, trained policy climbs to 0.97 on hard with 55% full resolution. The environment has a gradient a trained policy can climb — that's the contribution."

---

## Slide 6 — Close

**Visual**: single QR code + three URLs: HF Space, GitHub, trained adapter repo.

**Headline**: *Try it in 30 seconds. `demo_comparison.py`. No GPU.*

**Script (15s)**:

> "Multi-agent cooperation under deception, with a real trainable reward signal. Everything is reproducible. Clone the repo and run `demo_comparison.py`. Thanks."

---

## Q&A prep (the two questions you will get)

**Q: Why these three agents and not one?**
A: Real teams are structured that way for a reason: role specialization, blast-radius control, rotation. Partial observability forces communication, and communication is exactly the skill we're training. A single agent would collapse the problem to log-reading; we care about cross-agent belief modeling.

**Q: How do you know it isn't reward-hacked?**
A: Three layers. Anti-hack multiplicative gate detects loops, repetition, and spam and zeros out reward. Five independent reward functions — hard to game one without failing others. Ablation study confirms each component earns its weight. And the `outputs/war_room_grpo/rollout_audit.jsonl` logs every generation for post-hoc inspection.

**Q: Why Qwen 7B and not a bigger model?**
A: Three reasons: fits on a single consumer GPU (L40S, $1.80/hr) so the pipeline is reproducible by any team; Qwen 2.5-Instruct follows our COMMAND/MESSAGE format from step 1 without SFT; and 91 episodes is enough to show a learning signal while staying under $2 in compute.

**Q: What would you do with more time?**
A: Three things. One, run for more episodes to close the milestone gap. Two, add a rogue-insider agent that plays adversarially against the team rather than just noise. Three, ingest real Prometheus/PagerDuty traces as replay fixtures. But those are post-hackathon — the environment is the contribution, and the environment is done.
