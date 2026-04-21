# 3-Minute Pitch: Multi-Agent Incident War Room

## Slide 1: The Hook (30 seconds)
"When a production system goes down at 3 AM, it's never one person who fixes it. 
It's a team — someone reads the alerts, someone digs through logs, someone applies 
the fix. They communicate, they coordinate, they solve it together.

We built an environment that teaches AI agents to do exactly that."

## Slide 2: The Environment (45 seconds)
"Three AI agents in a war room:
- **Triage** sees the monitoring dashboard — which services are down, what alerts are firing
- **Diagnosis** can read logs and inspect processes — but can't fix anything
- **Remediation** can restart services and edit configs — but can't see the logs

No single agent can solve the incident alone. They MUST communicate through a shared 
message board to coordinate."

[SHOW: Demo output of agents communicating in Task 1]

## Slide 3: What Makes It Hard (45 seconds)
"Four tasks, escalating difficulty:
- Easy: Restart a crashed nginx — basic coordination test
- Medium: Memory leak with a HIGH-CPU RED HERRING — the dashboard shows the wrong thing first
- Hard: Cascading failure where Redis warnings are LOUDER than the actual DB auth error — 
  the diagnosis agent has to push back on triage's wrong initial assessment
- Expert: TWO incidents at once — agents must parallel-coordinate

The hard task tests THEORY OF MIND — can the diagnosis agent model what triage 
knows vs what's actually true?"


## Slide 4: The Reward Design (30 seconds)
"Five reward signals:
1. Milestone-based partial credit — dense, not sparse
2. Communication quality scoring — useful messages earn +0.05, incorrect info costs -0.02
3. Time pressure — every round costs 0.01
4. Fatal actions — kill the database, game over
5. Adaptive difficulty — environment gets harder as agents improve

Communication is a TRAINABLE SKILL with its own reward signal."

## Slide 5: Training Results (30 seconds)
[SHOW: Reward curves from GRPO training]
"After N episodes of GRPO training:
- Rounds to resolve decreased from X to Y
- Communication efficiency improved — fewer messages, more accurate
- Agents learned to push back on wrong diagnoses"

## Closing
"Multi-agent cooperation under partial observability, with communication as a 
first-class trainable action. First environment of its kind in OpenEnv."

---

## Q&A Prep

**Q: Why not just one agent?**
A: Real SRE teams have specialized roles. Partial observability forces communication, 
which is the skill we're training.

**Q: How do you handle the action space?**
A: OpenEnv is designed for LLM agents. Text commands are the natural action space, 
just like how real SREs type commands.

**Q: What's the self-improvement angle?**
A: Adaptive difficulty — the PerformanceTracker increases red herrings and tightens 
round limits as agents improve. Plus communication quality scoring creates a 
self-reinforcing loop.

**Q: How do you show training improvement?**
A: Reward curves from GRPO training showing rounds-to-resolve decreasing and 
communication efficiency improving over episodes.
