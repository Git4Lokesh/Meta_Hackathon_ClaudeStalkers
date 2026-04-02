---
title: SRE Incident Response Environment
emoji: 🔧
colorFrom: red
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# 🔧 SRE Incident Response Environment

An **OpenEnv-compliant** environment where AI agents diagnose and fix production infrastructure failures through Linux-style commands against a fully simulated system. Agents explore logs, inspect processes, trace cascading failures, edit configs, and restart services — just like a real Site Reliability Engineer on-call.

## Why SRE?

Site Reliability Engineering is one of the most **agentic** real-world tasks: it demands multi-step reasoning, exploration under uncertainty, hypothesis formation, and corrective action — all under time pressure. Companies are actively building AI SRE agents to reduce mean-time-to-recovery (MTTR) and handle the growing complexity of distributed systems.

This environment captures that challenge in a safe, reproducible sandbox with three incidents of escalating difficulty, deterministic grading, and partial-credit reward signals that guide agent learning beyond simple pass/fail.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                  Docker Container (:8000)                │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │              FastAPI Application                   │  │
│  │         HTTP endpoints + WebSocket                 │  │
│  └──────────────────────┬──────────────────────────┘  │
│                     │                                   │
│  ┌──────────────────▼────────────────────────────┐  │
│  │             SREEnvironment                         │  │
│  │         (OpenEnv interface: reset / step / state)  │  │
│  │                                                    │  │
│  │  ┌──────────────┐  ┌───────────────────────────┐  │  │
│  │  │ CommandParser │  │     SimulatedSystem        │  │  │
│  │  │ (16 commands) │──│  ┌─────────────────────┐  │  │  │
│  │  └──────────────┘  │  │   ProcessTable       │  │  │  │
│  │                     │  │   VirtualFilesystem  │  │  │  │
│  │  ┌──────────────┐  │  │   ServiceRegistry    │  │  │  │
│  │  │  TaskGrader   │  │  │   LogBuffer          │  │  │  │
│  │  │ (milestones + │  │  │   MetricsStore       │  │  │  │
│  │  │  penalties)   │  │  └─────────────────────┘  │  │  │
│  │  └──────────────┘  └───────────────────────────┘  │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

| Component | Role |
|---|---|
| **SREEnvironment** | Main class implementing the OpenEnv interface (`reset`, `step`, `state`). Orchestrates command execution and grading. |
| **SimulatedSystem** | In-memory production infrastructure: process table, virtual filesystem, service registry, log buffer, and metrics store. |
| **CommandParser** | Parses and executes 16 Linux-style command families against the SimulatedSystem. Never raises exceptions. |
| **TaskGrader** | Milestone-based scoring engine. Tracks partial credit and penalties to produce a 0.0–1.0 score. |

## Action Space

The agent submits a **single command string** per step — the same commands a human SRE would type:

```python
action = SREAction(command="ps aux")
action = SREAction(command="cat /var/log/nginx/error.log")
action = SREAction(command="systemctl restart nginx")
```

## Observation Space

Each step returns an `SREObservation`:

| Field | Type | Description |
|---|---|---|
| `output` | `str` | Terminal-style text output from the command |
| `reward` | `float` | Cumulative score so far (0.0–1.0) |
| `done` | `bool` | Whether the episode has ended |
| `metadata` | `dict` | Task info, step count; on terminal: score, milestones, penalties |

## Tasks

### Task 1: Service Restart (Easy)

| | |
|---|---|
| **Scenario** | nginx has crashed. The agent must find the downed service, restart it, and verify it’s healthy. |
| **Difficulty** | 🟢 Easy |
| **Max Steps** | 20 |
| **Key Signals** | Error logs in `/var/log/nginx/error.log`, crashed service in `systemctl status nginx` |

**Milestones:**
| Milestone | Credit |
|---|---|
| Read error log | 0.15 |
| Check service status | 0.25 |
| Restart nginx | 0.50 |
| Verify running | 0.10 |

---

### Task 2: Memory Leak Diagnosis (Medium)

| | |
|---|---|
| **Scenario** | A process is leaking memory, triggering OOM killer warnings. The agent must identify the leaking process, kill it, restart the affected service, and verify recovery. |
| **Difficulty** | 🟡 Medium |
| **Max Steps** | 30 |
| **Key Signals** | High memory in `top`/`free`, OOM entries in `/var/log/syslog`, one process at 2500+ MB |

**Milestones:**
| Milestone | Credit |
|---|---|
| Check memory | 0.10 |
| Identify leaking process | 0.20 |
| Read OOM logs | 0.15 |
| Kill leaking process | 0.25 |
| Restart service | 0.20 |
| Verify healthy | 0.10 |

**Penalties:** Killing a healthy process incurs a −0.10 penalty.

---

### Task 3: Cascading Failure (Hard)

| | |
|---|---|
| **Scenario** | A wrong database password in `/etc/app/database.yml` has crashed the DB connector, degraded the app server (retry-loop CPU spikes), and caused load balancer health check failures. The agent must trace the root cause across services, fix the config, and restart services in dependency order. |
| **Difficulty** | 🔴 Hard |
| **Max Steps** | 40 |
| **Key Signals** | Auth errors in DB connector logs, connection timeouts in app server logs, health check failures in LB logs, wrong password in config file |

**Milestones:**
| Milestone | Credit |
|---|---|
| Read load balancer logs | 0.05 |
| Read app server logs | 0.10 |
| Read DB connector logs | 0.10 |
| Read config file | 0.15 |
| Fix config (correct password) | 0.20 |
| Restart DB connector | 0.10 |
| Restart app server | 0.10 |
| Restart load balancer | 0.10 |
| Verify all services | 0.10 |

**Penalties:** Restarting a service before its dependencies are healthy incurs a −0.05 penalty.

## Supported Commands

| Command | Description | Example |
|---|---|---|
| `cat` | Display file contents | `cat /var/log/nginx/error.log` |
| `grep` | Search for pattern in file | `grep "ERROR" /var/log/syslog` |
| `tail` | Display last N lines (default 10) | `tail -n 20 /var/log/syslog` |
| `head` | Display first N lines (default 10) | `head -n 5 /etc/nginx/nginx.conf` |
| `ls` | List directory contents | `ls /var/log/` |
| `ps` | Show process table | `ps aux` |
| `top` | System summary + processes by CPU | `top` |
| `kill` | Kill a process by PID | `kill -9 1234` |
| `systemctl` | Manage services (start/stop/restart/status) | `systemctl restart nginx` |
| `curl` | HTTP request to a service | `curl http://localhost:80/health` |
| `df` | Show disk usage | `df` |
| `free` | Show memory usage | `free` |
| `netstat` | Show listening ports | `netstat` |
| `edit` | Edit file content (find and replace) | `edit /etc/app/database.yml "wrong_pass" "correct_pass"` |
| `echo` | Print text | `echo hello` |
| `help` | Show available commands | `help` |

## Reward Function

Scoring is **milestone-based with partial credit**:

- Each task defines a set of **milestones** (diagnostic steps + remediation actions), each worth a fraction of the total 1.0 score.
- Milestones are **idempotent** — triggering the same milestone twice doesn’t double the credit.
- **Penalties** reduce the score for destructive or wasteful actions:
  - Killing a healthy process: −0.10 (Task 2)
  - Restarting before dependencies are met: −0.05 (Task 3)
  - Repeating the exact same command (no-op): −0.02 per consecutive repeat
- Final score is always clamped to **[0.0, 1.0]**.

## Setup

### Local Installation

```bash
pip install -e .
```

### Docker

```bash
docker build -t sre-env:latest -f server/Dockerfile .
docker run -p 8000:8000 sre-env:latest
```

### Quick Start

```python
from sre_env.client import SREClient

client = SREClient()

# Start Task 1 (Easy: Service Restart)
obs = client.reset(task_id="task1", seed=42)
print(obs.output)

# Run commands
obs = client.step("cat /var/log/nginx/error.log")
print(obs.output, f"reward={obs.reward}")

obs = client.step("systemctl restart nginx")
print(obs.output, f"reward={obs.reward}")

obs = client.step("curl http://localhost:80/health")
print(obs.output, f"reward={obs.reward}, done={obs.done}")
```

## Baseline Script

Run an LLM agent through all three tasks:

```bash
export OPENAI_API_KEY='your-key'
python -m sre_env.baseline --seed 42 --model gpt-4o
```

Options:
- `--seed` — Random seed for reproducibility (default: 42)
- `--model` — OpenAI model name (default: gpt-4o)
- `--tasks` — Comma-separated task IDs (default: task1,task2,task3)

Example output:

```
Running task1...
  Score: 1.00 (5 steps)
Running task2...
  Score: 0.85 (12 steps)
Running task3...
  Score: 0.70 (22 steps)

======================================================================
BASELINE RESULTS
======================================================================
Task           Score     Steps     Milestones
----------------------------------------------------------------------
task1          1.00      5         read_error_log, check_service_status, restart_nginx, verify_running
task2          0.85      12        check_memory, identify_process, read_oom_logs, kill_process, restart_service
task3          0.70      22        read_lb_logs, read_app_logs, read_db_logs, read_config, fix_config, restart_db_connector
======================================================================
Average score: 0.85
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/reset` | POST | Reset environment for a task (`task_id`, `seed`) |
| `/step` | POST | Execute a command and return observation |
| `/state` | GET | Get current environment state |
| `/schema` | GET | OpenEnv schema definition |
| `/ws` | WebSocket | Real-time bidirectional communication |

## Project Structure

```
sre_env/
├── __init__.py
├── README.md                          # This file
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Dependencies
├── client.py                          # SREClient for programmatic access
├── baseline.py                        # LLM baseline inference script
└── server/
    ├── __init__.py
    ├── app.py                         # FastAPI application (create_app)
    ├── Dockerfile                     # Container packaging
    ├── sre_environment.py             # SREEnvironment (OpenEnv interface)
    ├── command_parser.py              # 16 Linux-style commands
    ├── simulated_system.py            # SimulatedSystem aggregate
    ├── models.py                      # Pydantic data models
    ├── grader.py                      # TaskGrader (milestones + penalties)
    └── tasks/
        ├── __init__.py                # Task registry
        ├── base.py                    # TaskDefinition base class
        ├── task1_service_restart.py   # Easy: nginx crash
        ├── task2_memory_leak.py       # Medium: memory leak + OOM
        └── task3_cascading_failure.py # Hard: cascading config failure
```

## License

Built for the [Meta-PyTorch Hackathon](https://pytorch.org/) using the [OpenEnv](https://github.com/openenv) framework.
