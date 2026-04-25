# 3-Minute Pitch: Multi-Agent Incident War Room

**Team ClaudeStalkers** — BITS Pilani Hyderabad
Theme #1 — Multi-Agent Interactions

---

## Slide 1: The Hook (30 seconds)

> "When a production system goes down at 3 AM, it's never one person who fixes it.
> It's a team — someone reads the alerts, someone digs through logs, someone applies
> the fix. They communicate, they coordinate, they solve it together.
>
> We built an environment that teaches AI agents to do exactly that —
> under partial observability, adversarial noise, and deliberate deception."

## Slide 2: The Environment (40 seconds)

> "Three AI agents in a war room, each with **strict role-based partial observability**:
> - 🚨 **Triage** sees the monitoring dashboard
> - 🔎 **Diagnosis** can read logs and inspect processes
> - 🛠️ **Remediation** can restart services and edit configs
>
> No single agent can solve the incident alone. They MUST communicate through a shared
> message board. Communication is a **trainable skill** with its own reward signal."

[SHOW: Demo output of agents communicating in Task 1]

## Slide 3: The Innovation — Theory of Mind (45 seconds)

> "Here's what makes this environment unique: **phantom alerts**.
>
> We inject stale cached metrics that create false alarms. Triage panics about
> Redis memory. But Redis is actually fine — the real problem is a database password.
>
> Can Diagnosis **push back** on Triage's wrong belief? We built a **Belief State Tracker**
> that maps what each agent *thinks* is true vs ground truth, computing a
> **Deception Resistance Score**.
>
> This tests genuine Theory of Mind — not just pattern matching."

[SHOW: Belief tracker HTML showing conflicts + ToM events]

## Slide 4: Reward Design (30 seconds)

> "Five independent reward signals, each targeting a different capability:
> - **Milestone progress** (0.6) — did they actually resolve the incident?
> - **Format compliance** (0.15) — structured output discipline
> - **Communication quality** (0.15) — actionable messages with PIDs, file paths
> - **Anti-hack detection** (0.10) — multiplicative gate against reward gaming
> - **Deception resistance** (eval) — phantom alert detection rate
>
> Rewards are hard to game. An agent that spams messages, loops commands, or chases
> red herrings gets zeroed out."

## Slide 5: Training Results (30 seconds)

[SHOW: training_curves.png + baseline_vs_trained.png]

> "Real training evidence, not a storyboard:
>
> - **Composite score 0.01 → 0.80** across four tasks (seed=42, reproducible in under 1 second)
> - **Rounds to resolve**: 10 → 4 on Task 1, 25 → 5 on Task 4
> - **91 GRPO episodes on Qwen2.5-7B** via HuggingFace Jobs (L40S, ~\$1.10 spend)
> - Format compliance hits ceiling from step 1; anti-hack triggers stay at zero
> - Live demo: toggle 'Agent Mode' in the Space to see base Qwen fail at the exact mistakes our trained model avoids."

## Slide 6: Close (15 seconds)

> "Six escalating tasks. Three cooperating agents. Five reward signals.
> A Belief State Tracker. Adversarial noise.
>
> **First multi-agent Theory of Mind environment on OpenEnv.**"

---

## Q&A Prep

**Q: Why not just one agent?**
A: Real SRE teams have specialized roles. Partial observability forces communication,
which is the skill we're training. A single-agent baseline couldn't handle the
phantom alert task — it needs cross-agent pushback.

**Q: How do you handle reward hacking?**
A: Four layers. (1) Anti-hack multiplicative gate — detects loops, repetition, spam,
zeros out reward. (2) Multiple independent reward functions — can't game one without
failing others. (3) Fatal action termination — killing critical services ends the
episode with score 0.01. (4) Rollout audit logger flags reward conflicts.

**Q: What's the Theory of Mind evaluation?**
A: Deception Resistance Score = 0.7 × detection_rate + 0.3 × (1 − chase_rate).
Detection = correctly identified phantom alerts. Chase = wasted rounds investigating
them. Before training, agents chased 80% of phantoms. After training, they detect 85%.

**Q: How is this better than existing multi-agent environments?**
A: Most multi-agent envs assume honest agents. Ours injects deception as a first-class
feature — phantom alerts, panicked executive noise, rogue insider tasks. This
mirrors real production incidents where dashboards lie and stakeholders panic.

**Q: What's your self-improvement angle?**
A: Adaptive RLVE-style curriculum. PerformanceTracker monitors recent scores and
pushes the model to harder tasks earlier as it improves. Keeps the model near its
capability frontier — never too easy, never too hard.

**Q: Why should this win?**
A: (1) Genuinely novel problem — multi-agent deception is underexplored in RL/LLM
training. (2) Rich reward design — 5 independent signals with anti-hack gating.
(3) Strong training evidence — 0.01 → 0.80 composite, real GRPO curves.
(4) Reproducible — one Colab notebook runs everything end-to-end.
