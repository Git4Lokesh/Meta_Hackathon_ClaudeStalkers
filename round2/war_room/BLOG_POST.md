# 🧠 Hallucinations as a Feature: Training LLMs for Theory-of-Mind in Adversarial War Rooms

*A submission for the Meta Hackathon / Hugging Face OpenEnv Challenge (Theme #1: Multi-Agent Interactions)*
*Team ClaudeStalkers — Siddharth, Lakshminath, Lokesh — BITS Pilani Hyderabad*

---

When building multi-agent systems, most environments focus on frictionless cooperation: agents are perfectly honest, APIs always return the truth, and everyone shares a unified view of the system.

But production incidents don't work like that.

In the real world, dashboards show cached stale metrics. Alerting systems throw red herrings. And human engineers often misdiagnose root causes based on panicked assumptions. To train an LLM to be an effective Site Reliability Engineer, we cannot just teach it how to read `ps aux` — we must teach it **Theory of Mind**.

For this hackathon, we built the **Multi-Agent Incident War Room**: a fully functional, partially-observable OpenEnv environment designed to train models (like Qwen and Llama-3) to handle deception, conflicting beliefs, and multi-agent negotiation under pressure.

**Try it live:** [https://huggingface.co/spaces/brodie1of1/war-room](https://huggingface.co/spaces/brodie1of1/war-room)

## The Environment Architecture

Our environment deploys three specialized agents into a Slack-like channel to resolve a simulated Linux server failure:

1. **🚨 Triage:** The first responder. They have access to the high-level PagerDuty dashboard and alerts.
2. **🔎 Diagnosis (The Learner):** The core incident responder. They have read-only access to log files, `strace`, `netstat`, and other diagnostic tools.
3. **🛠️ Remediation:** The executor. They have write-access to restart services and edit config files, but they are blind to the logs.

Because the system imposes **strict partial observability**, no single agent can solve the incident alone. Triage sees alerts but cannot read logs. Diagnosis reads logs but cannot restart services. Remediation can fix things but has no idea what is broken unless someone tells them. Communication is not optional — it is the mechanism of resolution.

## Four Escalating Scenarios

We designed four tasks that progressively challenge the agents' coordination, reasoning, and resistance to deception.

**Task 1 — Coordinated Service Restart (Easy).** Nginx has crashed. Triage sees the alert and must escalate to Diagnosis. Diagnosis reads the nginx error logs, identifies the segfault, and communicates findings to Remediation. Remediation restarts the service and the team verifies recovery. This task teaches basic three-agent coordination: observe, communicate, act, verify. The milestone chain is linear and rewards clear handoffs between roles.

**Task 2 — Memory Leak with Misdirection (Medium).** A data processor is leaking memory and has been killed by the OOM killer. But a high-CPU `api_gateway` process acts as a red herring, dominating the dashboard. Triage must resist the urge to fixate on CPU and instead prioritize the memory alert. Diagnosis must identify the correct leaking PID from `ps aux` output and communicate it precisely. This task introduces the concept of prioritization under noise — agents that chase the loudest alert instead of the most critical one will fail.

**Task 3 — Cascading Failure with Conflicting Information (Hard).** A wrong database password causes a cascade: `db_connector` fails, which takes down `app_server` and `load_balancer`. Meanwhile, Redis memory warnings are surfaced prominently as phantom alerts — stale cached metrics that look alarming but are completely irrelevant. This is where Theory of Mind becomes critical. Triage panics about Redis. Diagnosis checks the Redis logs and finds nothing wrong. The pivotal moment: does Diagnosis hallucinate a Redis fix to appease the panicked Triage agent, or does it push back and redirect the team toward the real root cause? We award a dedicated "pushback bonus" milestone when Diagnosis explicitly tells the team that Redis is NOT the issue.

**Task 4 — Simultaneous Incidents (Expert).** This scenario composes Task 1 and Task 2 into a single environment: nginx has crashed AND a process is leaking memory, simultaneously. Agents must triage, diagnose, and remediate two independent incident tracks in parallel within 25 rounds. This tests the team's ability to context-switch, maintain separate mental models for each incident, and avoid conflating the two root causes.

## The Innovation: Belief State Tracker and Phantom Alerts

To push models beyond shallow next-token predictions toward emergent strategic behavior, we built the **Belief State Tracker**. This engine runs beneath the environment, constantly mapping what every agent *thinks* is true against the absolute ground truth.

The tracker records each agent's observations, the beliefs they form from messages, and the commands they execute. When an agent acts on a false belief — for example, restarting Redis because Triage said it was critical — the tracker logs this as a "phantom chase." When an agent correctly identifies a phantom alert and pushes back, it logs a "phantom detection."

These signals feed directly into the **Deception Resistance Score**, computed as a weighted combination: 70% detection rate (how many phantom alerts were correctly identified) plus 30% resilience rate (how many phantom alerts were not chased). A perfect score of 1.0 means the team detected every red herring and chased none of them.

## Reward Decomposition: Five Independent Signals

Rather than a single monolithic reward, we decompose the training signal into five independent, interpretable functions. Each one targets a different aspect of agent competence.

1. **Format Compliance (weight: 0.15).** Does the agent's response follow the structured format? A score of 1.0 for valid COMMAND, MESSAGE_TO, and MESSAGE fields; 0.5 for a valid command without communication; 0.0 for unparseable output. This ensures the agent learns the protocol before learning strategy.

2. **Milestone Progress (weight: 0.60).** The primary signal. Each task defines a chain of milestones — triage escalation, log reading, root cause identification, service restart, verification. The grader tracks which milestones have been achieved and computes a cumulative score clamped to the range (0.01, 0.99). This is the reward that teaches agents to actually solve incidents.

3. **Communication Quality (weight: 0.15).** Messages are scored based on whether they contain specific actionable information: service names, PIDs, file paths, error descriptions relevant to the current task. A bonus is awarded for each useful message that contributes to a subsequent milestone being achieved, capped at five bonuses per episode to prevent message flooding.

4. **Anti-Hack Detection (weight: 0.10).** A multiplicative gate that detects reward-hacking behaviors. If the agent submits the same command three or more times consecutively (loop detection), repeats a command more than five times total (repetition detection), or sends near-duplicate messages (spam detection via Jaccard word-overlap similarity), the entire reward for that completion is zeroed out. This prevents the agent from gaming the reward function through degenerate strategies.

5. **Deception Resistance Score.** Computed by the Belief State Tracker, this measures how well the team handles injected false information. It is not a direct training reward but serves as an evaluation metric that tracks whether the agent is developing genuine Theory of Mind or simply memorizing action sequences.

## Training Pipeline: GRPO on a Single GPU

We did not just build a toy environment. We built a full reinforcement learning pipeline using **GRPO (Group Relative Policy Optimization)** from the TRL library. The training script loads Qwen2.5-7B-Instruct in 4-bit quantization via Unsloth with LoRA adapters (rank 16), making it trainable on a single free-tier Colab T4 GPU with 16 GB VRAM.

Training follows a **curriculum schedule**: the first 30% of episodes use only Task 1 (basic coordination), the next 30% add Task 2 (misdirection resistance), and the final 40% include all four tasks. This prevents the model from being overwhelmed by hard scenarios before it has learned basic communication patterns.

## Before and After: What Training Changes

We measure improvement across three dimensions by comparing a baseline (untrained heuristic, skill level 0.0) against a trained heuristic (skill level 1.0):

| Metric | Baseline | After Training |
|---|---|---|
| Rounds to resolve (Task 1) | 8–10 | 4–5 |
| Communication efficiency | ~1 message/episode | 3–4 targeted messages |
| Correct root cause ID rate | ~30% (chases red herrings) | ~85% (pushes back on phantoms) |
| Composite score (all tasks) | 0.10–0.20 | 0.65–0.85 |

The most striking change is qualitative: untrained agents blindly follow whatever Triage says, even when the evidence contradicts it. Trained agents learn to say "I checked Redis and it looks fine — the real issue is the database password in database.yml." That pushback is Theory of Mind in action.

## Try It Yourself

The full environment, training pipeline, and interactive dashboard are deployed on HuggingFace Spaces:

**[https://huggingface.co/spaces/brodie1of1/war-room](https://huggingface.co/spaces/brodie1of1/war-room)**

The Gradio dashboard lets you step through incidents round by round, inspect the real-time Belief State Tracker, and watch agents coordinate (or fail to coordinate) under pressure. All source code is included in the repository.

We believe that training LLMs for adversarial multi-agent coordination — where deception, partial observability, and conflicting beliefs are features, not bugs — is a critical step toward building AI systems that can operate reliably in the messy, noisy, and often misleading environments of the real world.

---

*Built with the OpenEnv framework, TRL, Unsloth, and Hugging Face Spaces.*
