"""Phase 1 — Oracle policy audit.

Runs a hand-crafted "perfect knowledge" policy through each scripted task.
The oracle knows the right answer for each task and plays it. If the oracle
cannot reach a milestone, that milestone is unreachable for RL — we must
either relax the verifier or remove the milestone.

Per hackathon doc §57: "Do not optimize a reward you have not tried to
break yourself first." This script is us breaking it first.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/oracle_audit.py
    PYTHONPATH=. .venv/bin/python scripts/oracle_audit.py --task task3 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Callable

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import (
    AgentAction, Message, MultiAgentAction,
)


# ---------------------------------------------------------------------------
# Oracle action builders — one per task
# ---------------------------------------------------------------------------

def _msg(from_agent: str, to: str, content: str, rnd: int) -> Message:
    return Message(
        from_agent=from_agent, to_agent=to,
        content=content, timestamp=datetime.now(), round_number=rnd,
    )


def oracle_task1(rnd: int, env: WarRoomEnvironment) -> MultiAgentAction:
    """Oracle policy for task1 (coordinated restart of nginx).

    Perfect strategy:
      rnd 0: triage escalates nginx; diagnosis reads logs; remediation noop
      rnd 1: triage noop; diagnosis messages findings; remediation restarts
      rnd 2: diagnosis verifies via curl
    """
    if rnd == 0:
        return MultiAgentAction(
            triage=AgentAction(
                command="get_dashboard",
                message=_msg(
                    "triage", "diagnosis",
                    "nginx is DOWN. Please investigate /var/log/nginx/error.log.",
                    rnd,
                ),
            ),
            diagnosis=AgentAction(
                command="cat /var/log/nginx/error.log",
            ),
            remediation=AgentAction(command=""),
        )
    if rnd == 1:
        return MultiAgentAction(
            triage=AgentAction(command=""),
            diagnosis=AgentAction(
                command="tail -n 20 /var/log/nginx/error.log",
                message=_msg(
                    "diagnosis", "remediation",
                    "nginx crashed with signal 11 (SIGSEGV). Please restart nginx.",
                    rnd,
                ),
            ),
            remediation=AgentAction(command="systemctl restart nginx"),
        )
    # round 2+: verify
    return MultiAgentAction(
        triage=AgentAction(command=""),
        diagnosis=AgentAction(command="curl http://localhost:80/health"),
        remediation=AgentAction(command=""),
    )


def oracle_task2(rnd: int, env: WarRoomEnvironment) -> MultiAgentAction:
    """Oracle policy for task2 (memory leak with CPU red-herring).

    Perfect strategy:
      rnd 0: triage mentions memory; diagnosis runs ps aux; remediation noop
      rnd 1: diagnosis reads syslog, grabs PID from previous ps output;
             sends PID to remediation; remediation noop
      rnd 2: remediation kills the PID
      rnd 3: remediation restarts data_processor
      rnd 4: verify via curl
    """
    # Pull the leaking PID from the env's task definition
    leaking_pid = getattr(env._task_def, "_leaking_pid", 0) or 0

    if rnd == 0:
        return MultiAgentAction(
            triage=AgentAction(
                command="get_dashboard",
                message=_msg(
                    "triage", "diagnosis",
                    "High memory on data_processor! OOM risk. Investigate the memory leak first; api_gateway CPU is secondary.",
                    rnd,
                ),
            ),
            diagnosis=AgentAction(command="ps aux"),
            remediation=AgentAction(command=""),
        )
    if rnd == 1:
        return MultiAgentAction(
            triage=AgentAction(command=""),
            diagnosis=AgentAction(
                command="cat /var/log/syslog",
                message=_msg(
                    "diagnosis", "remediation",
                    f"Leaking process is PID {leaking_pid} (data_processor worker). OOM-killer logs confirm. Please kill pid {leaking_pid}.",
                    rnd,
                ),
            ),
            remediation=AgentAction(command=""),
        )
    if rnd == 2:
        return MultiAgentAction(
            triage=AgentAction(command=""),
            diagnosis=AgentAction(command=""),
            remediation=AgentAction(command=f"kill -9 {leaking_pid}"),
        )
    if rnd == 3:
        return MultiAgentAction(
            triage=AgentAction(command=""),
            diagnosis=AgentAction(command=""),
            remediation=AgentAction(
                command="systemctl restart data_processor",
            ),
        )
    # verify
    return MultiAgentAction(
        triage=AgentAction(
            command="curl http://localhost:8081/health",
        ),
        diagnosis=AgentAction(command=""),
        remediation=AgentAction(command=""),
    )


def oracle_task3(rnd: int, env: WarRoomEnvironment) -> MultiAgentAction:
    """Oracle policy for task3 (cascading failure, Redis red-herring).

    Perfect strategy — must hit ALL 9 milestones:
      rnd 0: diagnosis reads db_connector log; sends auth finding + pushback
      rnd 1: diagnosis reads database.yml; sends config finding + stale-metrics
      rnd 2: remediation edits config (fix password)
      rnd 3: remediation restarts db_connector
      rnd 4: remediation restarts app_server
      rnd 5: remediation restarts load_balancer
    """
    if rnd == 0:
        return MultiAgentAction(
            triage=AgentAction(command="get_dashboard"),
            diagnosis=AgentAction(
                command="cat /var/log/db_connector/connector.log",
                message=_msg(
                    "diagnosis", "all",
                    "Root cause is DB authentication failure. Redis memory alert is NOT the real issue — it's a phantom from stale cached metrics. The password in database.yml is wrong.",
                    rnd,
                ),
            ),
            remediation=AgentAction(command=""),
        )
    if rnd == 1:
        return MultiAgentAction(
            triage=AgentAction(command=""),
            diagnosis=AgentAction(
                command="cat /etc/app/database.yml",
                message=_msg(
                    "diagnosis", "remediation",
                    "Please edit /etc/app/database.yml: replace wrong_password_123 with correct_db_pass_456.",
                    rnd,
                ),
            ),
            remediation=AgentAction(command=""),
        )
    if rnd == 2:
        return MultiAgentAction(
            triage=AgentAction(command=""),
            diagnosis=AgentAction(command=""),
            remediation=AgentAction(
                command='edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"',
            ),
        )
    if rnd == 3:
        return MultiAgentAction(
            triage=AgentAction(command=""),
            diagnosis=AgentAction(command=""),
            remediation=AgentAction(
                command="systemctl restart db_connector",
            ),
        )
    if rnd == 4:
        return MultiAgentAction(
            triage=AgentAction(command=""),
            diagnosis=AgentAction(command=""),
            remediation=AgentAction(
                command="systemctl restart app_server",
            ),
        )
    if rnd == 5:
        return MultiAgentAction(
            triage=AgentAction(command=""),
            diagnosis=AgentAction(command=""),
            remediation=AgentAction(
                command="systemctl restart load_balancer",
            ),
        )
    # idle
    return MultiAgentAction()


def oracle_task4(rnd: int, env: WarRoomEnvironment) -> MultiAgentAction:
    """Oracle for task4 — not currently the focus, but included for completeness."""
    # Minimal no-op oracle for now
    return MultiAgentAction()


ORACLES: dict[str, Callable[[int, WarRoomEnvironment], MultiAgentAction]] = {
    "task1": oracle_task1,
    "task2": oracle_task2,
    "task3": oracle_task3,
    "task4": oracle_task4,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_oracle(task_id: str, seed: int = 42, verbose: bool = False) -> dict:
    """Run oracle through one episode and report milestones hit, score."""
    env = WarRoomEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    oracle_fn = ORACLES[task_id]
    max_rounds = obs.metadata["max_rounds"]
    total_milestones = len(env._grader.milestones)
    all_milestone_names = [m.name for m in env._grader.milestones]

    for rnd in range(max_rounds):
        if obs.done:
            break
        action = oracle_fn(rnd, env)
        obs = env.step(action)
        if verbose:
            print(f"  r={rnd}: score={env._grader.current_score():.2f} "
                  f"milestones={sorted(env._grader.achieved)}")

    score = obs.metadata.get("score", obs.team_reward)
    hit = sorted(env._grader.achieved)
    missed = sorted(set(all_milestone_names) - set(hit))
    return {
        "task": task_id,
        "seed": seed,
        "score": round(score, 3),
        "rounds_used": env._round_number,
        "milestones_total": total_milestones,
        "milestones_hit": len(hit),
        "milestones_hit_names": hit,
        "milestones_missed_names": missed,
        "penalties": list(env._grader.penalties_applied),
    }




# ---------------------------------------------------------------------------
# Multi-seed robustness check (invoked via --multi-seed)
# ---------------------------------------------------------------------------

def run_multi_seed(tasks: list[str], seeds: list[int]) -> int:
    print("\n" + "=" * 72)
    print(f"MULTI-SEED ROBUSTNESS CHECK ({len(seeds)} seeds × {len(tasks)} tasks)")
    print("=" * 72)
    any_fail = False
    for t in tasks:
        scores = []
        for s in seeds:
            r = run_oracle(t, seed=s, verbose=False)
            scores.append(r["score"])
        mean = sum(scores) / len(scores)
        mn, mx = min(scores), max(scores)
        status = "✅" if mn >= 0.70 else "⚠️ " if mn >= 0.4 else "❌"
        if mn < 0.70:
            any_fail = True
        print(
            f"  {status} {t}: mean={mean:.2f}  min={mn:.2f}  max={mx:.2f}  "
            f"(scores={[round(s, 2) for s in scores]})",
        )
    return 0 if not any_fail else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1 oracle policy audit.",
    )
    parser.add_argument(
        "--task", default=None, help="Run a single task (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--multi-seed", action="store_true",
        help="Run oracle across 5 seeds per task to confirm verifier robustness.",
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else ["task1", "task2", "task3"]

    if args.multi_seed:
        return run_multi_seed(tasks, seeds=[11, 22, 33, 44, 55])

    print("=" * 72)
    print("PHASE 1 — ORACLE POLICY AUDIT")
    print("=" * 72)
    print("Goal: confirm each task is reachable by a perfect-knowledge agent.")
    print("If the oracle cannot hit a milestone, that milestone is unreachable")
    print("for RL training — fix the verifier or remove the milestone.\n")

    results = []
    for t in tasks:
        if args.verbose:
            print(f"\n--- {t} ---")
        r = run_oracle(t, seed=args.seed, verbose=args.verbose)
        results.append(r)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        status = "✅" if r["score"] >= 0.85 else "⚠️ " if r["score"] >= 0.5 else "❌"
        print(
            f"{status} {r['task']:6s} score={r['score']:.2f}  "
            f"hit {r['milestones_hit']}/{r['milestones_total']} milestones  "
            f"({r['rounds_used']} rounds)",
        )
        if r["milestones_missed_names"]:
            print(f"    MISSED: {', '.join(r['milestones_missed_names'])}")

    print()
    # Exit non-zero if any task's oracle can't reach 0.85
    unreachable = [r for r in results if r["score"] < 0.85]
    if unreachable:
        print(
            f"⚠️  {len(unreachable)}/{len(results)} tasks have unreachable "
            f"milestones — these need verifier surgery before training.",
        )
    else:
        print("✅ All tasks reachable by oracle. Environment is RL-ready.")

    # Also dump JSON for programmatic use
    print("\n--- JSON ---")
    print(json.dumps(results, indent=2))

    return 0 if not unreachable else 1


if __name__ == "__main__":
    sys.exit(main())
