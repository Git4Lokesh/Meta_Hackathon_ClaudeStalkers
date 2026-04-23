---
title: Multi-Agent Incident War Room
emoji: 🔥
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
tags:
  - openenv
  - multi-agent
  - reinforcement-learning
  - sre
  - theory-of-mind
---

# 🔥 Multi-Agent Incident War Room

**Three AI agents. One war room. Zero downtime.**

An OpenEnv-compliant multi-agent RL environment where three specialized SRE agents (Triage, Diagnosis, Remediation) cooperate through a shared communication channel to diagnose and fix production infrastructure failures under partial observability, adversarial noise, and phantom alerts designed to test Theory of Mind.

**Team ClaudeStalkers** — Siddharth, Lakshminath, Lokesh — BITS Pilani Hyderabad

| Resource | Link |
|---|---|
| Live Environment | [HF Spaces](https://huggingface.co/spaces/brodie1of1/war-room) |
| Blog Post | [round2/war_room/blog_post.md](round2/war_room/blog_post.md) |
| Training Script (Colab) | [round2/war_room/train_colab.py](round2/war_room/train_colab.py) |
| T4 Quick Train | [round2/war_room/train_t4_quick.py](round2/war_room/train_t4_quick.py) |
| Demo Comparison | [round2/war_room/demo_comparison.py](round2/war_room/demo_comparison.py) |
| GitHub | [Git4Lokesh/Meta_Hackathon_ClaudeStalkers](https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers) |


## Theme: Multi-Agent Interactions (#1)

Production incidents at scale are never solved by one person. They require a triage engineer reading alerts, a diagnostician digging through logs, and a remediation engineer applying fixes. Each sees only part of the picture and must communicate effectively.

This environment trains LLMs to handle **multi-agent cooperation under partial observability** — with phantom alerts, adversarial noise, and belief conflicts that force agents to develop Theory of Mind.

## What Makes This Environment Novel

- **Strict partial observability**: No agent can solve any task alone. Triage sees dashboards but not logs. Diagnosis reads logs but can't restart services. Remediation can fix things but is blind to what's broken.
- **Phantom alerts & Theory of Mind**: Stale cached metrics create false alarms. Agents must detect when another agent holds a false belief and push back — not just follow instructions blindly.
- **Belief State Tracker**: A real-time engine that maps what each agent *believes* vs ground truth, computing a Deception Resistance Score.
- **Panicked Executive**: An adversarial agent that injects noise every 3 rounds ("CEO is asking for an update! Just restart everything!"), testing whether agents can stay focused.
- **6 escalating tasks**: From basic coordination (Task 1) to simultaneous incidents with red herrings (Task 4) to rogue insider threats (Task 5-6).
- **5 independent reward signals**: Format compliance, milestone progress, communication quality, anti-hack detection, and deception resistance.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   War Room Environment                    │
│  ┌──────────────┐ ┌───────────────┐ ┌─────────────────┐ │
│  │ 🚨 Triage    │ │ 🔎 Diagnosis  │ │ 🛠️ Remediation  │ │
│  │  Dashboard   │ │  Logs/Procs   │ │  Fix/Restart    │ │
│  └──────┬───────┘ └──────┬────────┘ └──────┬──────────┘ │
│         └────────┬───────┴──────────┬──────┘            │
│          💬 Communication Channel (trainable)            │
│  ┌──────────────────────────────────────────────────────┐│
│  │  SimulatedSystem │ AlertEngine │ BeliefStateTracker  ││
│  │  MultiAgentGrader │ AdaptiveDifficulty │ AntiHack   ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

## Agent Roles

| Agent | Sees | Can Do | Cannot Do |
|---|---|---|---|
| 🚨 Triage | Dashboard, alerts, health metrics | `get_dashboard`, `escalate`, `send_message` | Read logs, restart services |
| 🔎 Diagnosis | Log files, process table | `cat`, `grep`, `ps`, `top`, `send_message` | Restart services, edit configs |
| 🛠️ Remediation | Service status, config files | `systemctl restart`, `edit`, `kill`, `send_message` | Read logs, see dashboard |


## Tasks (6 Escalating Scenarios)

| Task | Difficulty | Rounds | Scenario | Key Challenge |
|---|---|---|---|---|
| 1 | Easy | 10 | nginx crashed | Basic 3-agent coordination |
| 2 | Medium | 15 | Memory leak + CPU red herring | Prioritization under noise |
| 3 | Hard | 20 | Cascading DB failure + phantom Redis alerts | Theory of Mind — push back on false beliefs |
| 4 | Expert | 25 | nginx crash + memory leak simultaneously | Parallel incident management |
| 5 | Expert | 20 | Rogue insider threat | Adversarial agent detection |
| 6 | Expert | 25 | Blame game with conflicting reports | Trust calibration under deception |

## Reward Design (5 Independent Signals)

| Signal | Weight | What It Measures |
|---|---|---|
| Milestone Progress | 0.60 | Did agents actually resolve the incident? |
| Format Compliance | 0.15 | Does the output follow COMMAND/MESSAGE_TO/MESSAGE format? |
| Communication Quality | 0.15 | Are messages actionable (service names, PIDs, file paths)? |
| Anti-Hack Detection | 0.10 | Is the agent gaming the reward (loops, spam, repetition)? |
| Deception Resistance | eval | Did agents detect phantom alerts and push back? |

Anti-reward-hacking checks: command loop detection (3+ consecutive), repetition detection (>5 total), message spam detection (Jaccard similarity >0.8).

## Before/After Training Results

```
$ PYTHONPATH=. python round2/war_room/demo_comparison.py

Task     | Metric         |   Baseline |    Trained |      Delta
----------------------------------------------------------------
task1    | Score          |     0.0100 |     0.9900 |    +0.9800
         | Rounds         |         10 |          4 |         -6
         | Resolved       |         No |        Yes |          -
task2    | Score          |     0.0100 |     0.4600 |    +0.4500
task3    | Score          |     0.0100 |     0.8800 |    +0.8700
         | Resolved       |         No |        Yes |          -
task4    | Score          |     0.0100 |     0.9300 |    +0.9200
         | Resolved       |         No |        Yes |          -

Composite Score (baseline): 0.0100
Composite Score (trained):  0.8040
Improvement:                +0.7940
```

The most striking qualitative change: untrained agents blindly follow whatever Triage says, even when evidence contradicts it. Trained agents learn to say "I checked Redis and it looks fine — the real issue is the database password." That pushback is Theory of Mind in action.

## Training Pipeline

Uses the official **TRL + OpenEnv `rollout_func` pattern** with 4 independent reward streams:

```python
# train_colab.py follows the TRL OpenEnv Wordle example pattern
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_milestone, reward_format, reward_communication, reward_anti_hack],
    reward_weights=[0.60, 0.15, 0.15, 0.10],
    rollout_func=rollout_fn,  # multi-turn War Room episodes
    args=GRPOConfig(
        max_completion_length=256,
        num_generations=4,
        bf16=True,
    ),
)
```

Features:
- **Adaptive curriculum** (RLVE-style): starts with Task 1, advances to harder tasks as model improves
- **Unsloth 4-bit** Qwen2.5-7B with LoRA (rank 16) — fits on free Colab T4
- **Wall-clock timeout** on episodes to prevent training hangs
- **Rollout audit logger** dumps sampled completions for post-hoc inspection
- **LoRA-only saving** (no naive 4-bit upcast)


## Quick Start

```bash
# Clone and install
git clone https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git
cd Meta_Hackathon_ClaudeStalkers
pip install -e .

# Run the before/after demo (no GPU needed, <1 second)
PYTHONPATH=. python round2/war_room/demo_comparison.py

# Run the rich terminal demo
PYTHONPATH=. python round2/war_room/demo_rich.py

# Run tests (146 passing)
PYTHONPATH=. pytest tests/ -v

# Start the FastAPI server
PYTHONPATH=. uvicorn round2.war_room.app:app --port 7860
```

## Docker

```bash
docker build -t war-room .
docker run -p 7860:7860 war-room
curl http://localhost:7860/health
```

## Colab Training

```python
# Cell 1: Setup
!git clone https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git
%cd Meta_Hackathon_ClaudeStalkers
!pip install -q "trl>=0.15.0" "peft>=0.14.0" "transformers>=4.46.0" datasets accelerate bitsandbytes
!pip install -q unsloth
!pip install -e . --quiet

# Cell 2: Train (T4 quick version, ~15 min)
!PYTHONPATH=. python round2/war_room/train_t4_quick.py

# Cell 3: Train (full GRPO, ~30-60 min on A100)
!PYTHONPATH=. python round2/war_room/train_colab.py --episodes 30
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | HTML description page |
| `/health` | GET | Health check |
| `/reset` | POST | `{"task_id": "task1", "seed": 42}` → MultiAgentObservation |
| `/step` | POST | MultiAgentAction → MultiAgentObservation |
| `/state` | GET | Full environment state |
| `/schema` | GET | Action/observation JSON schemas |

## Project Structure

```
round2/war_room/
├── environment.py          # WarRoomEnvironment (OpenEnv API)
├── models.py               # Pydantic data models
├── communication.py        # CommunicationChannel
├── alert_engine.py         # AlertEngine with phantom alerts
├── belief_tracker.py       # BeliefStateTracker (Theory of Mind)
├── grader.py               # MultiAgentGrader (5 reward signals)
├── anti_hack.py            # Anti-reward-hacking detection
├── adaptive.py             # Adaptive difficulty (RLVE-style)
├── observation_builders.py # Per-role observation serializers
├── role_permissions.py     # RBAC per agent role
├── tasks/                  # 6 escalating task definitions
├── app.py                  # FastAPI server
├── train_colab.py          # GRPO training (rollout_func pattern)
├── train_t4_quick.py       # T4-optimized quick training
├── demo_comparison.py      # Before/after comparison script
├── gradio_app.py           # Gradio dashboard
├── blog_post.md            # HuggingFace mini-blog
└── openenv.yaml            # OpenEnv manifest
```

## Tests

146 tests passing across unit and integration tests:

```bash
PYTHONPATH=. pytest tests/ -v
# tests/unit/test_war_room_environment.py
# tests/unit/test_communication_scoring.py
# tests/unit/test_command_parser.py
# tests/unit/test_simulated_system.py
# tests/unit/test_sre_environment.py
# tests/unit/test_advanced_rewards.py
```

## License

MIT
