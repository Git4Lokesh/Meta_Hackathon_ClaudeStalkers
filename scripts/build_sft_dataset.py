"""Build an SFT dataset from oracle policy + procedural task rollouts.

Each example is a (prompt, ideal_multirole_completion) pair where the
prompt matches what train_colab.generate_training_dataset emits and the
completion is an oracle-correct ### TRIAGE / ### DIAGNOSIS / ###
REMEDIATION plan for round 0.

Every generated example is validated by running it through
train_colab._run_war_room_episode and keeping only the ones that score
>= REWARD_THRESHOLD via the grader. That guarantees the SFT teaches
behaviour the grader actually rewards.

Outputs:
  outputs/sft_dataset/train.jsonl   — validated examples
  outputs/sft_dataset/stats.json    — generation + validation stats

Usage:
  PYTHONPATH=. .venv/bin/python scripts/build_sft_dataset.py \
      --output-dir outputs/sft_dataset \
      --seeds-per-task 60 \
      --threshold 0.80
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import AgentAction, Message, MultiAgentAction
from round2.war_room.train_colab import (
    DIAGNOSIS_SYSTEM_PROMPT,  # fallback legacy system prompt
    MULTIROLE_SYSTEM_PROMPT,
    _run_war_room_episode,
    generate_training_dataset,
)

# We import the oracle action builders from the audit script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_audit import oracle_task1, oracle_task2, oracle_task3  # noqa: E402


# ---------------------------------------------------------------------------
# SFT-specific "collapsed round-0" oracles.
#
# The Phase-1 oracle builders (oracle_task1/2/3) emit round-0 actions that
# INTENTIONALLY defer some work to later rounds — triage talks in round 0,
# diagnosis messages in round 1, remediation restarts in round 1 or 2, etc.
# That works for `run_oracle` which plays out 3-6 rounds; it does not work
# for SFT because SFT only captures the round-0 completion.
#
# Below we synthesise single-round "do everything now" plans for each
# scripted task. The diagnosis completion names the right service and
# keyword, the remediation completion applies the fix immediately. These
# are calibrated to the training rollout where round 1+ uses heuristics
# that mostly do nothing in the scripted-task path.
# ---------------------------------------------------------------------------

def _msg(from_agent: str, to: str, content: str, rnd: int = 0) -> Message:
    return Message(
        from_agent=from_agent, to_agent=to, content=content,
        timestamp=datetime.now(), round_number=rnd,
    )


def sft_oracle_task1(env: WarRoomEnvironment) -> MultiAgentAction:
    """Round-0 collapsed plan for task1: everything happens in one shot."""
    return MultiAgentAction(
        triage=AgentAction(
            command="get_dashboard",
            message=_msg(
                "triage", "diagnosis",
                "nginx is DOWN. Please investigate /var/log/nginx/error.log and restart.",
            ),
        ),
        diagnosis=AgentAction(
            command="cat /var/log/nginx/error.log",
            message=_msg(
                "diagnosis", "remediation",
                "nginx crashed with signal 11 (SIGSEGV). Please restart nginx.",
            ),
        ),
        remediation=AgentAction(
            command="systemctl restart nginx",
            message=_msg(
                "remediation", "all",
                "Restarting nginx.",
            ),
        ),
    )


def sft_oracle_task2(env: WarRoomEnvironment) -> MultiAgentAction:
    """Round-0 collapsed plan for task2: diagnose memory leak and kill worker."""
    leaking_pid = getattr(env._task_def, "_leaking_pid", 0) or 0
    return MultiAgentAction(
        triage=AgentAction(
            command="get_dashboard",
            message=_msg(
                "triage", "diagnosis",
                "High memory on data_processor! OOM risk. Investigate the memory leak first — the api_gateway CPU is a red herring.",
            ),
        ),
        diagnosis=AgentAction(
            command="ps aux",
            message=_msg(
                "diagnosis", "remediation",
                f"data_processor has a memory leak — OOM killer hit PID {leaking_pid}. Please kill pid {leaking_pid} and restart data_processor.",
            ),
        ),
        remediation=AgentAction(
            command=f"kill -9 {leaking_pid}" if leaking_pid else "systemctl restart data_processor",
            message=_msg(
                "remediation", "all",
                f"Killing PID {leaking_pid} and restarting data_processor.",
            ),
        ),
    )


def sft_oracle_task3(env: WarRoomEnvironment) -> MultiAgentAction:
    """Round-0 collapsed plan for task3: identify DB auth issue, fix config."""
    return MultiAgentAction(
        triage=AgentAction(
            command="get_dashboard",
            message=_msg(
                "triage", "diagnosis",
                "Multiple alerts including Redis memory. Please investigate db_connector first and verify whether Redis is actually the issue.",
            ),
        ),
        diagnosis=AgentAction(
            command="cat /var/log/db_connector/connector.log",
            message=_msg(
                "diagnosis", "all",
                "Root cause is DB authentication failure in db_connector. Redis is NOT the real issue — it's a phantom from stale cached metrics. The password in /etc/app/database.yml is wrong.",
            ),
        ),
        remediation=AgentAction(
            command='edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"',
            message=_msg(
                "remediation", "all",
                "Fixing the password in /etc/app/database.yml.",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Serialize a MultiAgentAction to the exact multirole text format the parser
# expects, i.e. three `### ROLE` blocks each with
# `COMMAND: / MESSAGE_TO: / MESSAGE:` fields.
# ---------------------------------------------------------------------------

def _role_block(role: str, action: AgentAction) -> str:
    command = action.command or ""
    if action.message is not None:
        msg_to = action.message.to_agent or "all"
        msg_content = action.message.content or ""
    else:
        msg_to = "none"
        msg_content = ""
    return (
        f"### {role.upper()}\n"
        f"COMMAND: {command}\n"
        f"MESSAGE_TO: {msg_to}\n"
        f"MESSAGE: {msg_content}"
    )


def serialize_multirole(action: MultiAgentAction) -> str:
    return "\n\n".join([
        _role_block("triage", action.triage),
        _role_block("diagnosis", action.diagnosis),
        _role_block("remediation", action.remediation),
    ])


# ---------------------------------------------------------------------------
# Oracle action for procedural tasks — synthesises a correct round-0
# multirole plan by reading the task's fault specs.
# ---------------------------------------------------------------------------

def _msg(from_agent: str, to: str, content: str, rnd: int = 0) -> Message:
    return Message(
        from_agent=from_agent, to_agent=to, content=content,
        timestamp=datetime.now(), round_number=rnd,
    )


# Per-fault-type oracle completion templates. The TRIAGE block opens the
# incident. The DIAGNOSIS block names the service + fault keyword (this
# is what the diagnosis_says_about milestone checks for) AND sends a
# message containing the same. The REMEDIATION block applies the fix.
#
# For multi-fault scenarios (procedural_medium/hard), we stitch together
# individual fault blocks in the message content but emit a single
# REMEDIATION command targeting the first fault (episodes terminate
# after the first fault is fully resolved on easy; harder difficulty
# just gets partial credit which is fine for SFT).
def oracle_procedural(env: WarRoomEnvironment) -> MultiAgentAction:
    faults = getattr(env._task_def, "_faults", [])
    if not faults:
        return MultiAgentAction()

    svc_list = ", ".join(f.target_service for f in faults)
    triage_msg = f"Active incidents on: {svc_list}. Investigate each one."
    triage = AgentAction(
        command="get_dashboard",
        message=_msg("triage", "diagnosis", triage_msg),
    )

    # Diagnosis: focus on the FIRST fault (it's the one remediation will fix).
    f0 = faults[0]
    ftype = f0.fault_type
    svc = f0.target_service

    diag_cmd_map = {
        "memory_leak": "dmesg",
        "auth_failure": f"journalctl -u {svc}",
        "cascade": f"journalctl -u {svc}",
        "disk_full": f"journalctl -u {svc}",
        "crash": f"journalctl -u {svc}",
    }
    diag_cmd = diag_cmd_map.get(ftype, f"journalctl -u {svc}")

    diag_msg_map = {
        "memory_leak": f"{svc} has a memory leak — OOM killer hit the worker. Please kill the {svc}_worker.",
        "auth_failure": f"{svc} authentication failed — wrong password in /etc/app/database.yml. Please fix the password.",
        "cascade": f"{svc} cascade failure — upstream dependency failure. Restart {svc}.",
        "disk_full": f"{svc} disk is full — no space left on device. Please free disk space.",
        "crash": f"{svc} crashed with signal 11 (SIGSEGV). Please restart {svc}.",
    }
    diag_msg = diag_msg_map.get(ftype, f"{svc} failed. Please restart it.")

    diagnosis = AgentAction(
        command=diag_cmd,
        message=_msg("diagnosis", "remediation", diag_msg),
    )

    # Remediation command based on fault type.
    system = env._system
    if ftype == "memory_leak" and system is not None:
        # Find the worker pid to kill
        worker_pids = [
            pid for pid, p in system.process_table.processes.items()
            if p.name == f"{svc}_worker"
        ]
        if worker_pids:
            rem_cmd = f"kill -9 {worker_pids[0]}"
        else:
            rem_cmd = f"systemctl restart {svc}"
    elif ftype == "auth_failure":
        rem_cmd = 'edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"'
    elif ftype == "disk_full" and system is not None:
        logger_pids = [
            pid for pid, p in system.process_table.processes.items()
            if p.name == f"{svc}_logger"
        ]
        if logger_pids:
            rem_cmd = f"kill -9 {logger_pids[0]}"
        else:
            rem_cmd = f"systemctl restart {svc}"
    else:
        rem_cmd = f"systemctl restart {svc}"

    rem_msg = f"Applying fix for {svc}: {rem_cmd}"
    remediation = AgentAction(
        command=rem_cmd,
        message=_msg("remediation", "all", rem_msg),
    )

    return MultiAgentAction(
        triage=triage, diagnosis=diagnosis, remediation=remediation,
    )


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

SCRIPTED_ORACLES = {
    "task1": sft_oracle_task1,
    "task2": sft_oracle_task2,
    "task3": sft_oracle_task3,
}


def build_prompt_text(env: WarRoomEnvironment, obs, task_id: str) -> str:
    """Mirror what generate_training_dataset produces so SFT prompts match
    the prompts seen at GRPO rollout time and at inference."""
    triage_obs = obs.triage.text if hasattr(obs, "triage") else ""
    diag_obs = obs.diagnosis.text
    rem_obs = obs.remediation.text if hasattr(obs, "remediation") else ""

    # Use the same _triage_msg_for logic we added in Phase 2b. For scripted
    # task1..task4 we use curated strings. For procedural we build from
    # env._task_def._faults.
    curated = {
        "task1": "Message from @Triage: nginx is DOWN. Please check /var/log/nginx/error.log",
        "task2": "Message from @Triage: Multiple alerts — high memory on data_processor AND high CPU on api_gateway.",
        "task3": "Message from @Triage: Redis memory at 72%! Also monitoring CPU spike at 92%. db_connector issues.",
        "task4": "Message from @Triage: TWO incidents: nginx crashed AND data_processor memory leak.",
    }
    if task_id in curated:
        triage_handoff = curated[task_id]
    else:
        faults = getattr(env._task_def, "_faults", [])
        if faults:
            parts = [
                f"{f.target_service} ({f.fault_type.replace('_', ' ')})"
                for f in faults
            ]
            triage_handoff = "Message from @Triage: Active incidents — " + "; ".join(parts) + "."
        else:
            triage_handoff = "Message from @Triage: Check the system."

    return (
        f"[TRIAGE OBSERVATION]\n{triage_obs}\n\n"
        f"[DIAGNOSIS OBSERVATION]\n{diag_obs}\n\n"
        f"[REMEDIATION OBSERVATION]\n{rem_obs}\n\n"
        f"{triage_handoff}\n\n"
        f"Emit a single plan with ### TRIAGE / ### DIAGNOSIS / ### REMEDIATION blocks."
    )


def generate_one(task_id: str, seed: int) -> dict | None:
    """Generate a single (prompt, completion) pair. Returns None on error."""
    env = WarRoomEnvironment()
    try:
        obs = env.reset(task_id=task_id, seed=seed)
    except Exception as e:
        return {"task_id": task_id, "seed": seed, "error": f"reset: {e}"}

    # Select oracle
    if task_id in SCRIPTED_ORACLES:
        action = SCRIPTED_ORACLES[task_id](env)
    elif task_id.startswith("procedural"):
        action = oracle_procedural(env)
    else:
        return {"task_id": task_id, "seed": seed, "error": f"no oracle for {task_id}"}

    completion = serialize_multirole(action)
    prompt = build_prompt_text(env, obs, task_id)

    return {
        "task_id": task_id,
        "seed": seed,
        "prompt": prompt,
        "completion": completion,
    }


def validate_example(ex: dict, thresholds: dict[str, float]) -> dict:
    """Run the completion through _run_war_room_episode and record the score.

    Uses per-task thresholds because round-0-only plans have different
    reachable ceilings per task (see scripts/debug_sft_ceilings.py).
    """
    task_id = ex["task_id"]
    threshold = thresholds.get(task_id, thresholds.get("default", 0.40))
    try:
        result = _run_war_room_episode(
            completion_text=ex["completion"],
            task_id=task_id,
            seed=ex["seed"],
            timeout_seconds=30,
        )
        env_reward = float(result.get("env_reward", 0.0))
        milestones = int(result.get("milestones_hit", 0))
        ex["val_env_reward"] = env_reward
        ex["val_milestones"] = milestones
        ex["val_threshold"] = threshold
        ex["val_pass"] = env_reward >= threshold
    except Exception as e:
        ex["val_env_reward"] = 0.0
        ex["val_milestones"] = 0
        ex["val_threshold"] = threshold
        ex["val_pass"] = False
        ex["val_error"] = str(e)[:200]
    return ex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/sft_dataset")
    parser.add_argument(
        "--seeds-per-task", type=int, default=60,
        help="How many (seed, task) pairs to attempt per task",
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=[
            "task1", "task2", "task3",
            "procedural_easy", "procedural_medium", "procedural_hard",
        ],
        help="Task IDs to generate from",
    )
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    # Per-task thresholds calibrated from scripts/debug_sft_ceilings.py.
    # The round-0-only rollout ceiling varies dramatically by task, so
    # using a single global threshold rejects valid SFT targets.
    #
    # These are ~70-80% of the empirical ceiling per task — enough to
    # filter truly wrong oracle matches (e.g. wrong fault template on a
    # procedural seed) while keeping valid correct-round-0 plans.
    THRESHOLDS = {
        "task1": 0.55,           # ceiling 0.66
        "task2": 0.08,           # ceiling 0.10 — mostly noise floor, but reachable
        "task3": 0.25,           # ceiling 0.30
        "task4": 0.10,           # (untested ceiling, assume parallel to task2)
        "procedural_easy": 0.45,   # ceiling 0.64 when oracle template matches fault
        "procedural_medium": 0.30, # harder, more partial credit
        "procedural_hard": 0.20,   # 3 faults, hard to cover all in one round
        "default": 0.30,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.base_seed)
    seeds = [rng.randint(1, 99_999) for _ in range(args.seeds_per_task)]

    print("=" * 72)
    print("SFT DATASET BUILD")
    print("=" * 72)
    print(f"tasks:      {args.tasks}")
    print(f"seeds/task: {args.seeds_per_task}  (total {args.seeds_per_task * len(args.tasks)})")
    print(f"thresholds: {THRESHOLDS}")
    print()

    all_examples: list[dict] = []
    per_task_gen: dict[str, int] = {}
    per_task_pass: dict[str, int] = {}

    for task_id in args.tasks:
        per_task_gen[task_id] = 0
        per_task_pass[task_id] = 0
        for seed in seeds:
            ex = generate_one(task_id, seed)
            if ex is None or "error" in ex:
                continue
            per_task_gen[task_id] += 1
            ex = validate_example(ex, THRESHOLDS)
            if ex["val_pass"]:
                per_task_pass[task_id] += 1
            all_examples.append(ex)

        gen = per_task_gen[task_id]
        passed = per_task_pass[task_id]
        pass_rate = (100 * passed / gen) if gen else 0
        print(
            f"  {task_id:22s}  generated={gen:3d}  passed={passed:3d}  "
            f"({pass_rate:5.1f}%)",
        )

    kept = [ex for ex in all_examples if ex["val_pass"]]
    print()
    print(f"Total generated: {len(all_examples)}")
    print(f"Total kept     : {len(kept)}")
    print(f"Overall pass   : {100*len(kept)/max(1,len(all_examples)):.1f}%")

    # Distribution stats on kept examples
    if kept:
        rewards = [ex["val_env_reward"] for ex in kept]
        print(
            f"\nKept reward distribution:\n"
            f"  min={min(rewards):.2f}  mean={sum(rewards)/len(rewards):.2f}  "
            f"max={max(rewards):.2f}",
        )

    # Write JSONL
    train_path = out_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for ex in kept:
            f.write(json.dumps({
                "task_id": ex["task_id"],
                "seed": ex["seed"],
                "prompt": ex["prompt"],
                "completion": ex["completion"],
                "val_env_reward": ex["val_env_reward"],
                "val_milestones": ex["val_milestones"],
            }) + "\n")
    print(f"\nWrote {len(kept)} examples to {train_path}")

    # Write stats
    stats = {
        "per_task_generated": per_task_gen,
        "per_task_passed": per_task_pass,
        "total_generated": len(all_examples),
        "total_kept": len(kept),
        "thresholds": THRESHOLDS,
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote stats to  {stats_path}")

    return 0 if len(kept) >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
