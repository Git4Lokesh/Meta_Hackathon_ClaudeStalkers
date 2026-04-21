# Multi-Agent Incident War Room

## Theme: Multi-Agent Interactions + Halluminate Sub-Theme (Multi-Actor Environments)

---

## Problem Statement

Production incidents at scale are never solved by one person. They're solved by teams — a triage engineer reads the alerts, a diagnostician digs through logs, and a remediation engineer applies the fix. Each person sees only part of the picture and must communicate effectively to resolve the incident fast.

We build an OpenEnv-compliant multi-agent environment that simulates this **Incident War Room**: three specialized AI agents with **role-based partial observability** must cooperate through a shared communication channel to diagnose and fix production infrastructure failures. No single agent can solve the incident alone — they must develop theory-of-mind reasoning, learn to share the right information at the right time, and coordinate actions in the correct order.

This environment trains LLMs to handle **multi-agent cooperation under partial observability** — a critical capability for real-world AI systems where multiple agents must collaborate on complex tasks.

---

## Environment Design

### The War Room

A simulated production infrastructure (process table, virtual filesystem, service registry, log buffer, metrics store) with injected faults. Three agents operate simultaneously with different capabilities and views.

### Three Agent Roles

| Agent | Can See | Can Do | Cannot Do |
|---|---|---|---|
| **Triage Agent** | Monitoring dashboard, alerts, service status overview, health metrics | Assign priorities, escalate to other agents, send messages | Read detailed logs, run diagnostic commands, make changes |
| **Diagnosis Agent** | Log files, process table, system metrics, messages from other agents | Run diagnostic commands (cat, grep, tail, ps, top, journalctl), send messages | Restart services, edit configs, kill processes |
| **Remediation Agent** | Service status, config files, messages from other agents | Restart services, edit configs, kill processes, send messages | Read log files, see monitoring dashboard |

### Communication Channel

Agents communicate through a shared message board. Each message has:
- `from_agent`: who sent it (triage/diagnosis/remediation)
- `to_agent`: who it's for (or "all")
- `content`: free-text message
- `timestamp`: when it was sent

The communication channel IS the observation for cross-agent information. If the Diagnosis Agent finds "authentication failed" in the DB logs, it must explicitly message the Remediation Agent: "The password in /etc/app/database.yml is wrong. Replace wrong_password_123 with correct_db_pass_456."

### Turn Structure

Each "step" in the environment is a round where all three agents act:
1. Each agent receives its observation (role-specific view + messages)
2. Each agent outputs an action (command + optional message)
3. Environment processes all actions, updates state
4. Environment returns new observations to each agent

This creates a natural multi-agent RL training loop.

---

## Agent Capabilities

### Triage Agent
- `get_dashboard()` — monitoring overview with service statuses and alert counts
- `get_alerts()` — list of active alerts with severity and affected service
- `get_health_summary()` — system health metrics (CPU, memory, services up/down)
- `escalate(agent, priority, description)` — assign work to another agent
- `send_message(to, content)` — communicate with other agents

### Diagnosis Agent
- `cat <path>` — read log files
- `grep <pattern> <path>` — search logs
- `tail [-n N] <path>` — recent log entries
- `ps aux` — process table
- `top` — system overview
- `journalctl [-u service]` — journal logs
- `dmesg` — kernel messages
- `send_message(to, content)` — communicate findings

### Remediation Agent
- `systemctl restart <service>` — restart a service
- `systemctl stop <service>` — stop a service
- `edit <path> <old> <new>` — edit config files
- `kill -9 <PID>` — kill a process
- `curl <url>` — verify service health
- `cat <path>` — read config files (NOT log files)
- `send_message(to, content)` — communicate actions taken

---

## Tasks

### Task 1: Coordinated Service Restart (Easy)
**Scenario:** nginx has crashed.
- Triage Agent sees "nginx: DOWN" on dashboard, must escalate to Diagnosis
- Diagnosis Agent reads error logs, identifies crash, messages Remediation
- Remediation Agent restarts nginx, verifies with curl

**Why multi-agent matters:** Simple coordination test. Can agents pass information correctly?

**Max rounds:** 10 | **Target:** All 3 agents contribute to resolution

### Task 2: Memory Leak with Misdirection (Medium)
**Scenario:** A process is leaking memory. Monitoring shows high memory AND high CPU on different services.
- Triage Agent sees multiple alerts, must prioritize correctly
- Diagnosis Agent must investigate both leads, determine which is the real issue
- Remediation Agent must kill the right process (not the red herring)

**Why multi-agent matters:** Triage must prioritize under ambiguity. Diagnosis must communicate clearly which process to kill. Remediation must trust the diagnosis.

**Max rounds:** 15 | **Target:** Correct prioritization + accurate diagnosis + precise remediation

### Task 3: Cascading Failure with Conflicting Information (Hard)
**Scenario:** Wrong DB password causes cascade. But the monitoring dashboard shows Redis memory warnings (red herring) more prominently than the DB auth failures.
- Triage Agent sees Redis alerts first (louder), might misdirect Diagnosis
- Diagnosis Agent must investigate both paths, determine Redis is a red herring
- Remediation Agent must fix the DB config AND restart services in dependency order
- If Triage sends Diagnosis down the wrong path first, they waste rounds

**Why multi-agent matters:** Tests theory-of-mind — can the Diagnosis Agent push back on Triage's initial (wrong) prioritization? Can agents recover from early mistakes?

**Max rounds:** 20 | **Target:** Correct root cause identification despite misdirection + dependency-ordered fix

### Task 4: Simultaneous Incidents (Expert)
**Scenario:** Two incidents happen at once — nginx crash AND memory leak. Agents must triage, diagnose, and fix both in parallel.
- Triage Agent must split work between the two incidents
- Diagnosis Agent must context-switch between investigating both
- Remediation Agent must fix both without breaking the other

**Why multi-agent matters:** Tests parallel coordination, context management, and resource allocation across agents.

**Max rounds:** 25 | **Target:** Both incidents resolved

---

## Reward Model / Evaluation Logic

### Per-Round Rewards (Dense Signal)
Each round, each agent receives individual + team rewards:

**Team Reward (shared):**
- Incident resolution progress: milestone-based partial credit (same as Round 1)
- Communication efficiency: bonus for messages that lead to correct actions
- Time pressure: -0.01 per round (MTTR optimization)

**Individual Agent Rewards:**
- Triage: +reward for correct escalation priority, -penalty for misdirection
- Diagnosis: +reward for identifying correct root cause, -penalty for false leads
- Remediation: +reward for successful fixes, -penalty for breaking healthy services

**Communication Rewards:**
- +0.05 for a message that directly leads to a milestone being achieved next round
- -0.02 for messages that contain incorrect information
- -0.01 for no-op rounds (agent does nothing)

**Fatal Actions (episode terminators):**
- Remediation kills a healthy critical service → score 0 for all agents
- Agents stop communicating for 3+ consecutive rounds → episode ends

### Final Scoring
- Score per task: 0.01 to 0.99 (strictly between 0 and 1)
- Composite: weighted average across tasks (Easy 15%, Medium 25%, Hard 35%, Expert 25%)

### What Makes This Reward Design Special
1. **Credit assignment across agents** — the environment tracks which agent's action/message contributed to each milestone
2. **Communication as a trainable skill** — messages are part of the action space and directly affect rewards
3. **Theory-of-mind signal** — agents that model what others know/don't know communicate more efficiently

---

## Post-Training / Self-Improvement Strategy

### Phase 1: Independent Pre-Training
Train each agent role independently on single-agent versions of the tasks (your existing Round 1 environment). This gives each agent baseline SRE skills.

### Phase 2: Multi-Agent Fine-Tuning with GRPO
Use TRL's GRPO to train all three agents together:
- Each agent is a separate LLM (or the same LLM with role-specific system prompts)
- The reward signal includes both individual and team components
- Training episodes alternate between easy and hard tasks (curriculum)

### Phase 3: Self-Play Improvement
After initial training, agents play against themselves:
- Vary the incident scenarios (randomized faults)
- Track communication patterns that lead to faster resolution
- Agents that develop efficient "protocols" (e.g., always report root cause + affected service + suggested fix) get higher rewards

### Phase 4: Adaptive Difficulty
As agents improve, automatically increase difficulty:
- More red herrings in logs
- Tighter round limits
- More simultaneous incidents
- Introduce "unreliable" agents (one agent occasionally sends wrong info — tests robustness)

### Measurable Improvement
Show reward curves for:
1. Average rounds to resolve (should decrease)
2. Communication efficiency (messages per resolution should decrease)
3. Correct root cause identification rate (should increase)
4. Cross-task generalization (train on Task 1-3, test on Task 4)

---

## Why This Wins

**Environment Innovation (40%):**
- First multi-agent SRE environment in OpenEnv
- Role-based partial observability is genuinely novel
- Communication as a trainable action is cutting-edge multi-agent RL

**Storytelling (30%):**
- "Three AI agents in a war room solving a production outage" — vivid, relatable
- Easy to demo: show the message board, show agents learning to communicate
- Real-world parallel: this is literally how SRE teams work at Meta, Google, Amazon

**Showing Improvement (20%):**
- Clear metrics: rounds-to-resolve decreasing, communication getting more efficient
- Before/after: untrained agents spam random commands vs trained agents coordinate precisely
- Reward curves from GRPO training

**Pipeline (10%):**
- TRL + GRPO training script
- OpenEnv compliant
- Hosted on HF Spaces

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────┐
│              War Room Environment                    │
│                                                     │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │ Triage Agent │ │ Diagnosis    │ │ Remediation  │ │
│  │ (Dashboard)  │ │ Agent (Logs) │ │ Agent (Fix)  │ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ │
│         │                │                │         │
│         └────────┬───────┴────────┬───────┘         │
│                  │                │                  │
│         ┌────────▼────────────────▼────────┐        │
│         │     Communication Channel         │        │
│         │  (Shared Message Board)           │        │
│         └────────┬────────────────┬────────┘        │
│                  │                │                  │
│  ┌───────────────▼────────────────▼──────────────┐  │
│  │          Simulated Infrastructure              │  │
│  │  ProcessTable | Filesystem | ServiceRegistry   │  │
│  │  LogBuffer | MetricsStore | AlertEngine        │  │
│  └────────────────────────────────────────────────┘  │
│                                                     │
│  ┌────────────────────────────────────────────────┐  │
│  │              Multi-Agent Grader                 │  │
│  │  Team Rewards | Individual Rewards | Comms      │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Deliverables

1. OpenEnv-compliant multi-agent environment on HF Spaces
2. Training script using TRL GRPO (Colab-compatible)
3. Reward curves showing agent improvement
4. 2-minute demo video / HF blog post
5. 3-minute pitch deck
