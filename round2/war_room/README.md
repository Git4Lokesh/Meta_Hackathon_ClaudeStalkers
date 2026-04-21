---
title: Multi-Agent Incident War Room
emoji: 🔧
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# 🔧 Multi-Agent Incident War Room

**Three AI agents. One war room. Zero downtime.**

An OpenEnv-compliant multi-agent RL environment where three specialized SRE agents — Triage, Diagnosis, and Remediation — cooperate through a shared communication channel to diagnose and fix production infrastructure failures. No single agent can solve the incident alone.

## Why This Matters

Production incidents at scale are never solved by one person. They're solved by teams — a triage engineer reads the alerts, a diagnostician digs through logs, and a remediation engineer applies the fix. Each person sees only part of the picture and must communicate effectively to resolve the incident fast.

This environment trains LLMs to handle **multi-agent cooperation under partial observability** — a critical capability for real-world AI systems.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   War Room Environment                    │
│                                                          │
│  ┌──────────────┐ ┌───────────────┐ ┌─────────────────┐ │
│  │ Triage Agent  │ │ Diagnosis     │ │ Remediation     │ │
│  │ 📊 Dashboard  │ │ 🔍 Logs       │ │ 🔧 Fix          │ │
│  └──────┬────────┘ └──────┬────────┘ └──────┬──────────┘ │
│         │                 │                  │            │
│         └────────┬────────┴─────────┬────────┘            │
│                  │                  │                     │
│         ┌────────▼──────────────────▼────────┐           │
│         │     💬 Communication Channel        │           │
│         │     (Shared Message Board)          │           │
│         └────────┬──────────────────┬────────┘           │
│                  │                  │                     │
│  ┌───────────────▼──────────────────▼──────────────────┐ │
│  │           Simulated Infrastructure                   │ │
│  │  ProcessTable │ Filesystem │ ServiceRegistry         │ │
│  │  LogBuffer │ MetricsStore │ AlertEngine              │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              Multi-Agent Grader                       │ │
│  │  Team Rewards │ Individual Rewards │ Credit Assignment│ │
│  └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## The Three Agent Roles

| Agent | Sees | Can Do | Cannot Do |
|---|---|---|---|
| **📊 Triage** | Dashboard, alerts, health metrics, messages | `get_dashboard`, `get_alerts`, `get_health_summary`, `escalate`, `send_message` | Read logs, run diagnostics, make changes |
| **🔍 Diagnosis** | Log files, process table, messages | `cat`, `grep`, `tail`, `ps`, `top`, `journalctl`, `dmesg`, `send_message` | Restart services, edit configs, kill processes |
| **🔧 Remediation** | Service status, config files, messages | `systemctl restart/stop`, `edit`, `kill`, `curl`, `cat` (configs only), `send_message` | Read log files, see dashboard |

Each agent has **partial observability** — they can only see what their role allows. The communication channel is the only way to share information across roles.

---

## Tasks

### Task 1: Coordinated Service Restart (Easy)
- **Scenario:** nginx has crashed
- **Max rounds:** 10
- **Milestones:** Triage escalates → Diagnosis reads logs → Diagnosis messages findings → Remediation restarts → Verification
- **Tests:** Basic coordination — can agents pass information correctly?

### Task 2: Memory Leak with Misdirection (Medium)
- **Scenario:** Memory leak + high-CPU red herring
- **Max rounds:** 15
- **Milestones:** Triage prioritizes memory over CPU → Diagnosis identifies correct PID → Remediation kills correct process
- **Tests:** Prioritization under ambiguity, accurate diagnosis, trust between agents

### Task 3: Cascading Failure with Conflicting Info (Hard)
- **Scenario:** Wrong DB password causes cascade; Redis alerts are louder than DB auth failures
- **Max rounds:** 20
- **Milestones:** Diagnosis identifies DB auth (not Redis) → Remediation fixes password → Dependency-ordered restart
- **Tests:** Theory-of-mind — can Diagnosis push back on Triage's wrong initial prioritization?

### Task 4: Simultaneous Incidents (Expert)
- **Scenario:** nginx crash AND memory leak at the same time
- **Max rounds:** 25
- **Milestones:** Independent tracks for each incident, both must be resolved
- **Tests:** Parallel coordination, context management, resource allocation

---

## Communication Channel

The communication channel is the core innovation. Agents communicate through structured messages:

```python
Message(
    from_agent="diagnosis",
    to_agent="remediation",
    content="nginx crashed with signal 11 (segfault). Config is valid. Needs restart.",
    timestamp=datetime.now(),
    round_number=3,
)
```

Messages are delivered at the start of the next round. If agents stop communicating for 3+ consecutive rounds, the episode terminates (communication breakdown).

Communication is a **trainable skill** — messages that lead to correct actions earn +0.05 reward, while incorrect information costs -0.02.

---

## Reward Model

### Team Reward (shared)
- **Milestone progress:** Partial credit for each milestone achieved (0.10–0.40 per milestone)
- **Time pressure:** -0.01 per round (incentivizes fast resolution)
- **Communication efficiency:** +0.05 for useful messages, -0.02 for incorrect info

### Individual Rewards
- **Triage:** +reward for correct escalation priority, -penalty for misdirection
- **Diagnosis:** +reward for identifying correct root cause, -penalty for false leads
- **Remediation:** +reward for successful fixes, -penalty for breaking healthy services

### Fatal Actions (episode terminators)
- Remediation kills a healthy critical service → score 0.01 for all agents
- Communication breakdown (3+ silent rounds) → episode ends

### Composite Scoring
```
Composite = 0.15 × Task1 + 0.25 × Task2 + 0.35 × Task3 + 0.25 × Task4
```
All per-task scores clamped to (0.01, 0.99).

---

## Setup

### Local Development

```bash
# Clone and install
git clone <repo-url>
cd <repo>
pip install -e ".[dev]"

# Run the demo
PYTHONPATH=. python3 round2/war_room/demo.py

# Run inference (requires API key)
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
PYTHONPATH=. python3 round2/war_room/inference.py

# Run tests
pytest tests/ -v
```

### Docker

```bash
docker build -f round2/war_room/Dockerfile -t war-room .
docker run -p 8000:8000 war-room

# API endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": "task1", "seed": 42}'
```

### Colab Training

```python
!pip install trl unsloth openai
!git clone <repo-url>
%cd <repo>
!pip install -e .

# Train with GRPO
!python round2/war_room/train.py --model unsloth/Qwen2.5-7B --episodes 100
```

---

## Demo Output

```
============================================================
🔧 MULTI-AGENT INCIDENT WAR ROOM
   Task 1: Coordinated Service Restart (Easy)
============================================================

────────────────────────────────────────────────────────────
ROUND 1: Triage checks dashboard and escalates to Diagnosis
────────────────────────────────────────────────────────────
  [TRIAGE] Command: get_dashboard
  [TRIAGE] 💬 → diagnosis: URGENT: nginx is DOWN. Please investigate /var/log/nginx/error.log

  Team Reward: 0.09 | Done: False

────────────────────────────────────────────────────────────
ROUND 2: Diagnosis reads logs and reports findings
────────────────────────────────────────────────────────────
  [DIAGNOSIS] Command: cat /var/log/nginx/error.log
  [DIAGNOSIS] 💬 → remediation: nginx crashed with signal 11 (segfault). Needs restart.

  Team Reward: 0.17 | Done: False

────────────────────────────────────────────────────────────
ROUND 3: Remediation restarts nginx
────────────────────────────────────────────────────────────
  [REMEDIATION] Command: systemctl restart nginx
  [REMEDIATION] 💬 → all: nginx restarted. Verifying health...

  Team Reward: 0.54 | Done: False

────────────────────────────────────────────────────────────
ROUND 4: Remediation verifies the fix
────────────────────────────────────────────────────────────
  [REMEDIATION] Command: curl http://localhost:80/health

  ✅ INCIDENT RESOLVED!
  Score: 0.760
  Milestones: ['diagnosis_messages_findings', 'diagnosis_reads_logs',
               'remediation_restarts_nginx', 'triage_escalates_nginx',
               'verification']

============================================================
```

---

## API Reference

### `POST /reset`
Initialize environment for a task.
```json
{"task_id": "task1", "seed": 42}
```

### `POST /step`
Submit one round of multi-agent actions.
```json
{
  "triage": {"command": "get_dashboard", "message": {"from_agent": "triage", "to_agent": "diagnosis", "content": "nginx is down", "timestamp": "...", "round_number": 1}},
  "diagnosis": {"command": "cat /var/log/nginx/error.log"},
  "remediation": {"command": ""}
}
```

### `GET /state`
Full environment state snapshot.

### `GET /health`
Health check endpoint.

### `GET /schema`
Action and observation JSON schemas.

---

## Training Strategy

1. **Phase 1 — Independent Pre-Training:** Each agent role trained on single-agent tasks (Round 1 environment)
2. **Phase 2 — Multi-Agent GRPO:** All three agents trained together with TRL GRPOTrainer, curriculum learning from easy → hard tasks
3. **Phase 3 — Self-Play:** Agents play against themselves with randomized faults
4. **Phase 4 — Adaptive Difficulty:** Auto-increase difficulty as agents improve

### Measurable Improvement
- Average rounds to resolve (↓ decreasing)
- Communication efficiency — messages per resolution (↓ decreasing)
- Correct root cause identification rate (↑ increasing)
- Cross-task generalization — train on Tasks 1–3, test on Task 4

---

## License

MIT
