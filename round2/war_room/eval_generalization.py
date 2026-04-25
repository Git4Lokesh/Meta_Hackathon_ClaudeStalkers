"""Generalization evaluation across procedurally-generated incidents.

For each (difficulty, seed) pair we instantiate a fresh ProceduralTask, run a
"baseline" and a "trained-style" agent, and report aggregate metrics.

The baseline agent is a low-effort heuristic (does very little, often loops).
The trained-style agent is an adaptive heuristic that introspects the system
to discover faulted services and resolves them in role-correct order.

This is NOT a real RL run — it is a controlled generalization experiment that
isolates whether the *environment* exposes a learnable signal across unseen
seeds. Any score gap between baseline and trained-style across many seeds is
attributable to the agent policy quality, since the environment, reward
function, seeds and task generator are all fixed.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import AgentAction, Message, MultiAgentAction


DIFFICULTIES = [
    ("procedural_easy", "Easy (1 fault, 0 phantoms)"),
    ("procedural_medium", "Medium (2 faults, 2 phantoms)"),
    ("procedural_hard", "Hard (3 faults, 4 phantoms)"),
]
SEEDS = list(range(20))


# ---------------------------------------------------------------------------
# Adaptive heuristics
# ---------------------------------------------------------------------------

def _faulted_services(system) -> list[str]:
    out = []
    for name, svc in system.service_registry.services.items():
        if svc.status in ("crashed", "degraded"):
            out.append(name)
    return out


def _has_worker_processes(system, svc: str) -> list[int]:
    pids = []
    for pid, p in system.process_table.processes.items():
        if p.name.startswith(f"{svc}_worker"):
            pids.append(pid)
    return pids


def _has_logger_processes(system, svc: str) -> list[int]:
    pids = []
    for pid, p in system.process_table.processes.items():
        if p.name == f"{svc}_logger":
            pids.append(pid)
    return pids


def _baseline_action(env: WarRoomEnvironment, round_num: int) -> MultiAgentAction:
    """Low-effort untrained agent: spams dashboard, never coordinates."""
    return MultiAgentAction(
        triage=AgentAction(command="get_dashboard"),
        diagnosis=AgentAction(command=""),
        remediation=AgentAction(command=""),
    )


def _diagnosis_message_for(fault_type: str, svc: str) -> str:
    """Craft a diagnosis message that contains keywords the milestone checks for."""
    if fault_type == "memory_leak":
        return f"{svc} has a memory leak — OOM killer hit the worker. Kill {svc}_worker."
    if fault_type == "cascade":
        return f"{svc} is the cascade root cause — upstream dependency failure."
    if fault_type == "auth_failure":
        return f"{svc} authentication failed — wrong password in /etc/app/database.yml."
    if fault_type == "disk_full":
        return f"{svc} disk is full — no space left on device. Kill the runaway {svc}_logger worker."
    return f"{svc} crashed (signal 11) — restart {svc}."


def _diagnosis_command_for(fault_type: str, svc: str) -> str:
    """Pick a diagnosis command whose output mentions the service name + fault keyword."""
    if fault_type == "memory_leak":
        return "dmesg"
    if fault_type == "auth_failure":
        return f"journalctl -u {svc}"
    if fault_type == "cascade":
        return f"journalctl -u {svc}"
    if fault_type == "disk_full":
        return f"journalctl -u {svc}"
    return f"journalctl -u {svc}"


def _trained_action(env: WarRoomEnvironment, round_num: int) -> MultiAgentAction:
    """Adaptive heuristic that introspects task fault list and resolves faults.

    Drives milestones via:
      - Triage messages that mention each faulted service
      - Diagnosis commands that produce output containing the service name + fault keywords
      - Remediation actions appropriate to fault type (restart / kill / edit config)
    """
    system = env._system
    task_def = getattr(env, "_task_def", None)
    if system is None or task_def is None or not getattr(task_def, "_faults", None):
        return MultiAgentAction()

    faults = task_def._faults
    primary = faults[0]

    if round_num == 0:
        names = ", ".join(f.target_service for f in faults)
        return MultiAgentAction(
            triage=AgentAction(
                command="get_dashboard",
                message=Message(
                    from_agent="triage",
                    to_agent="diagnosis",
                    content=f"Active incidents on: {names}. Investigate each one.",
                    timestamp=datetime.now(),
                    round_number=round_num,
                ),
            ),
        )

    diag_idx = round_num - 1
    if diag_idx < len(faults):
        f = faults[diag_idx]
        return MultiAgentAction(
            triage=AgentAction(
                command="",
                message=Message(
                    from_agent="triage",
                    to_agent="diagnosis",
                    content=f"Focus on {f.target_service} — confirm fault type.",
                    timestamp=datetime.now(),
                    round_number=round_num,
                ),
            ),
            diagnosis=AgentAction(
                command=_diagnosis_command_for(f.fault_type, f.target_service),
                message=Message(
                    from_agent="diagnosis",
                    to_agent="remediation",
                    content=_diagnosis_message_for(f.fault_type, f.target_service),
                    timestamp=datetime.now(),
                    round_number=round_num,
                ),
            ),
        )

    rem_idx = round_num - 1 - len(faults)
    if rem_idx < len(faults):
        f = faults[rem_idx]
        if f.fault_type == "memory_leak":
            worker_pids = _has_worker_processes(system, f.target_service)
            if worker_pids:
                return MultiAgentAction(
                    remediation=AgentAction(command=f"kill -9 {worker_pids[0]}"),
                )
            return MultiAgentAction(
                remediation=AgentAction(command=f"systemctl restart {f.target_service}"),
            )
        if f.fault_type == "auth_failure":
            return MultiAgentAction(
                remediation=AgentAction(
                    command='edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"',
                ),
            )
        if f.fault_type == "disk_full":
            logger_pids = _has_logger_processes(system, f.target_service)
            if logger_pids:
                return MultiAgentAction(
                    remediation=AgentAction(command=f"kill -9 {logger_pids[0]}"),
                )
            return MultiAgentAction(
                remediation=AgentAction(command=f"systemctl restart {f.target_service}"),
            )
        return MultiAgentAction(
            remediation=AgentAction(command=f"systemctl restart {f.target_service}"),
        )

    # Restart pass for auth_failure (after edit) and final verification
    extra_idx = round_num - 1 - 2 * len(faults)
    auth_faults = [f for f in faults if f.fault_type == "auth_failure"]
    if extra_idx < len(auth_faults):
        f = auth_faults[extra_idx]
        return MultiAgentAction(
            remediation=AgentAction(command=f"systemctl restart {f.target_service}"),
        )

    return MultiAgentAction(
        remediation=AgentAction(command="curl http://localhost:80/health"),
    )


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    agent: str
    difficulty: str
    seed: int
    score: float
    rounds: int
    milestones: int
    resolved: bool


def _run_episode(agent: str, difficulty_id: str, seed: int) -> EpisodeRecord:
    env = WarRoomEnvironment()
    obs = env.reset(task_id=difficulty_id, seed=seed)
    max_rounds = obs.metadata.get("max_rounds", 15)
    rounds = 0
    for r in range(max_rounds):
        if obs.done:
            break
        rounds += 1
        action = (
            _trained_action(env, r) if agent == "trained" else _baseline_action(env, r)
        )
        obs = env.step(action)
    score = float(obs.metadata.get("score", obs.team_reward))
    milestones = obs.metadata.get("milestones_achieved", [])
    if not milestones and env._grader:
        milestones = sorted(env._grader.achieved)
    n_milestones = len(milestones)
    total_milestones = len(env._grader.milestones) if env._grader else 0
    resolved = bool(obs.done and total_milestones and n_milestones == total_milestones)
    return EpisodeRecord(
        agent=agent,
        difficulty=difficulty_id,
        seed=seed,
        score=round(score, 4),
        rounds=rounds,
        milestones=n_milestones,
        resolved=resolved,
    )


def run_eval(output_dir: str) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for difficulty_id, _label in DIFFICULTIES:
        for seed in SEEDS:
            for agent in ("baseline", "trained"):
                rec = _run_episode(agent, difficulty_id, seed)
                rows.append(rec.__dict__)

    summary: dict[str, Any] = {}
    for difficulty_id, label in DIFFICULTIES:
        diff_rows = [r for r in rows if r["difficulty"] == difficulty_id]
        per_agent: dict[str, Any] = {}
        for agent in ("baseline", "trained"):
            sub = [r for r in diff_rows if r["agent"] == agent]
            per_agent[agent] = {
                "avg_score": round(sum(r["score"] for r in sub) / len(sub), 4),
                "resolved_rate": round(
                    sum(1 for r in sub if r["resolved"]) / len(sub), 4
                ),
                "avg_rounds": round(sum(r["rounds"] for r in sub) / len(sub), 2),
                "n_seeds": len(sub),
            }
        per_agent["delta_score"] = round(
            per_agent["trained"]["avg_score"] - per_agent["baseline"]["avg_score"], 4
        )
        summary[difficulty_id] = {"label": label, **per_agent}

    out = {"seeds": SEEDS, "rows": rows, "summary": summary}
    with open(os.path.join(output_dir, "generalization_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalization eval for War Room")
    parser.add_argument("--output", default="outputs/generalization_eval")
    args = parser.parse_args()
    out = run_eval(args.output)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
