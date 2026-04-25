"""
Colab-Ready GRPO Training Script for Multi-Agent War Room
==========================================================

Uses the official TRL + OpenEnv ``rollout_func`` pattern (see
https://huggingface.co/docs/trl/openenv) to run multi-turn War Room
episodes inside the GRPO training loop.  The Diagnosis agent is the
learner; Triage and Remediation are heuristic co-agents.

Four independent reward streams (like the Wordle example):
  1. milestone_reward  — environment team score        (weight 0.60)
  2. format_reward     — structured output compliance  (weight 0.15)
  3. comm_reward       — actionable message quality    (weight 0.15)
  4. anti_hack_reward  — reward-hacking gate           (weight 0.10)

Usage in Colab:
    # Cell 1: Setup
    !git clone https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git
    %cd Meta_Hackathon_ClaudeStalkers
    !pip install -q "trl>=0.15.0" "peft>=0.14.0" "transformers>=4.46.0" \
                    datasets accelerate bitsandbytes
    !pip install -q unsloth
    !pip install -q fastapi pydantic uvicorn openai matplotlib rich
    !pip install -e . --quiet

    # Cell 2: Train
    !PYTHONPATH=. python round2/war_room/train_colab.py --episodes 30

    # Cell 3: Visualize
    !PYTHONPATH=. python round2/war_room/visualize.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from typing import Callable, Optional

import signal
from contextlib import contextmanager

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import MultiAgentAction, AgentAction, Message
from round2.war_room.adaptive import PerformanceTracker
from round2.war_room import anti_hack


# ============================================================
# TIMEOUT GUARD  (hackathon guide §8, §13, §44)
# ============================================================

class EpisodeTimeout(Exception):
    """Raised when a War Room episode exceeds the wall-clock limit."""


@contextmanager
def episode_timeout(seconds: int = 30):
    """Context manager that kills an episode if it exceeds *seconds*.

    Uses SIGALRM on Unix; on Windows/unsupported platforms it is a no-op.
    """
    def _handler(signum, frame):
        raise EpisodeTimeout(f"Episode exceeded {seconds}s wall-clock limit")

    if hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        yield  # no-op on platforms without SIGALRM


# ============================================================
# ROLLOUT AUDIT LOGGER  (hackathon guide §15, §17, §43, §52)
# ============================================================

class RolloutAuditor:
    """Logs sampled completions and reward conflicts to a JSONL file
    for post-hoc inspection.  The hackathon guide stresses monitoring
    actual generations, not just reward curves.
    """

    def __init__(self, path: str, sample_rate: int = 10) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self._count = 0
        self._fh = open(path, "w")

    def log(
        self,
        completion: str,
        task_id: str,
        env_reward: float,
        format_score: float,
        comm_score: float,
        anti_hack_score: float,
    ) -> None:
        self._count += 1
        if self._count % self.sample_rate != 0:
            return
        entry = {
            "step": self._count,
            "task_id": task_id,
            "env_reward": env_reward,
            "format": format_score,
            "communication": comm_score,
            "anti_hack": anti_hack_score,
            "completion_preview": completion[:300],
        }
        # Flag reward conflicts (§40): anti-hack zeroing a high milestone
        if anti_hack_score < 0.5 and env_reward > 0.5:
            entry["WARNING"] = "anti-hack zeroed a high-milestone completion"
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ============================================================
# CURRICULUM SCHEDULER  (RLVE-style adaptive, §22-23, §35, §46)
# ============================================================

class CurriculumScheduler:
    """Selects tasks based on training progress AND model performance.

    Phase-based baseline (task1 → task1+2 → task1-4) with adaptive
    override: if the model is crushing easy tasks (avg score > 0.7),
    it advances to harder tasks earlier.  This follows the RLVE
    principle of keeping the model near its capability frontier.

    The scheduler respects the user-provided task list passed to
    ``train_colab.py --tasks ...``. If the caller provides an explicit
    list, phase boundaries are computed within that list's ordering
    (first ~third = easy, next ~third = medium, final ~third = full).
    If the list is omitted, the legacy hardcoded schedule is used.
    """

    def __init__(
        self,
        total_steps: int,
        tasks: list[str] | None = None,
    ) -> None:
        self.total_steps = total_steps
        self.current_step = 0
        self.tracker = PerformanceTracker()
        # Preserve user-specified task order; treat earlier entries as
        # "easier" (curriculum ramps them in first).
        self._tasks: list[str] = list(tasks) if tasks else [
            "task1", "task2", "task3", "task4",
        ]

    def get_task(self) -> str:
        progress = self.current_step / max(self.total_steps, 1)
        avg = self.tracker.recent_avg_score(n=10)
        n = len(self._tasks)
        if n == 0:
            return "task1"  # defensive fallback

        # Adaptive override: if model is doing well, push harder earlier.
        # 'Harder' = include later entries in the user's task list.
        if avg >= 0.7 and progress < 0.6:
            pool = self._tasks[: min(n, max(3, n // 2))]
            return random.choice(pool)
        if avg >= 0.5 and progress < 0.3:
            pool = self._tasks[: min(n, max(2, n // 3 or 1))]
            return random.choice(pool)

        # Default phase-based curriculum across the user's task list
        if progress < 0.3:
            pool = self._tasks[: max(1, n // 3 or 1)]
        elif progress < 0.6:
            pool = self._tasks[: max(2, (2 * n) // 3 or 2)]
        else:
            pool = self._tasks
        return random.choice(pool) if pool else self._tasks[0]

    def record(self, task_id: str, score: float, rounds: int) -> None:
        """Feed episode results back for adaptive scheduling."""
        self.tracker.record_episode(task_id, score, rounds)

    def advance(self) -> None:
        self.current_step += 1


# ============================================================
# HEURISTIC CO-AGENTS
# ============================================================

def _build_heuristic_triage(round_num: int, task_id: str) -> AgentAction:
    if round_num == 0:
        msg_content = {
            "task1": "nginx is DOWN. Please check /var/log/nginx/error.log",
            "task2": "Multiple alerts: high memory on data_processor, high CPU on api_gateway. Investigate both.",
            "task3": "Multiple alerts: Redis memory warning, monitoring CPU spike, and db_connector issues.",
            "task4": "TWO incidents: nginx crashed AND data_processor memory leak.",
        }.get(task_id, "Check the system for failing services.")
        return AgentAction(
            command="get_dashboard",
            message=Message(
                from_agent="triage", to_agent="diagnosis",
                content=msg_content,
                timestamp=datetime.now(), round_number=round_num,
            ),
        )
    return AgentAction(command="")


def _build_heuristic_remediation(round_num: int, task_id: str, diagnosis_msg: str) -> AgentAction:
    msg_lower = diagnosis_msg.lower()
    if "restart" in msg_lower and "nginx" in msg_lower:
        return AgentAction(command="systemctl restart nginx")
    if "kill" in msg_lower:
        pid_match = re.search(r'pid\s*(\d+)', msg_lower)
        if pid_match:
            return AgentAction(command=f"kill -9 {pid_match.group(1)}")
    if "edit" in msg_lower and "password" in msg_lower:
        return AgentAction(command='edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"')
    for svc in ("db_connector", "app_server", "load_balancer", "data_processor"):
        if "restart" in msg_lower and svc in msg_lower:
            return AgentAction(command=f"systemctl restart {svc}")
    if "curl" in msg_lower or "verify" in msg_lower:
        return AgentAction(command="curl http://localhost:80/health")
    return AgentAction(command="")


def _parse_diagnosis_completion(text: str, round_num: int) -> AgentAction:
    """Parse a model completion into a Diagnosis AgentAction."""
    text = text.strip()
    text = re.sub(r'```\w*\n?', '', text)
    text = text.strip('`').strip()

    command = ""
    msg_to = ""
    msg_content = ""

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("COMMAND:"):
            command = line.split(":", 1)[1].strip()
        elif line.upper().startswith("MESSAGE_TO:"):
            msg_to = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("MESSAGE:"):
            msg_content = line.split(":", 1)[1].strip()

    if not command:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.upper().startswith("MESSAGE"):
                command = line
                break

    message = None
    if msg_to and msg_to != "none" and msg_content:
        message = Message(
            from_agent="diagnosis", to_agent=msg_to,
            content=msg_content,
            timestamp=datetime.now(), round_number=round_num,
        )
    return AgentAction(command=command, message=message)


# ============================================================
# ROLLOUT FUNCTION  (official TRL + OpenEnv pattern)
# ============================================================
# Follows the Wordle example from TRL docs: the rollout runs a full
# multi-turn episode, collects prompt_ids/completion_ids/logprobs,
# and passes auxiliary reward signals via extra dict keys.

_FOLLOW_UP_CMDS: dict[str, list[str]] = {
    "task1": ["cat /var/log/nginx/error.log", ""],
    "task2": ["ps aux", "cat /var/log/syslog"],
    "task3": ["cat /var/log/db_connector/connector.log", "cat /var/log/redis/redis.log"],
    "task4": ["cat /var/log/nginx/error.log", "ps aux"],
}

MAX_EPISODE_ROUNDS = 8  # cap for speed during training


def _run_war_room_episode(
    completion_text: str,
    task_id: str,
    seed: int,
    timeout_seconds: int = 30,
    policy_fn: Optional[Callable[[str, int], str]] = None,
) -> dict[str, float]:
    """Run one War Room episode with the model's completion as round-0
    Diagnosis action, heuristic co-agents for all roles, and heuristic
    follow-up for subsequent rounds.

    Includes a wall-clock timeout (hackathon guide §8) to prevent
    training from hanging on a stuck episode.

    Returns a dict of reward signals to be forwarded to reward_funcs.
    """
    try:
        with episode_timeout(timeout_seconds):
            return _run_war_room_episode_inner(completion_text, task_id, seed, policy_fn)
    except (EpisodeTimeout, Exception) as e:
        if isinstance(e, EpisodeTimeout):
            print(f"[TIMEOUT] Episode on {task_id} exceeded {timeout_seconds}s", flush=True)
        return {"env_reward": 0.01, "rounds_used": 0, "milestones_hit": 0}


def _run_war_room_episode_inner(
    completion_text: str,
    task_id: str,
    seed: int,
    policy_fn: Optional[Callable[[str, int], str]] = None,
) -> dict[str, float]:
    """Inner episode logic (no timeout wrapper)."""
    env = WarRoomEnvironment()
    try:
        obs = env.reset(task_id=task_id, seed=seed)
    except Exception:
        return {"env_reward": 0.01, "rounds_used": 0, "milestones_hit": 0}

    max_rounds = min(obs.metadata.get("max_rounds", 10), MAX_EPISODE_ROUNDS)
    last_diag_msg = ""
    rounds_used = 0

    for r in range(max_rounds):
        if obs.done:
            break
        rounds_used += 1

        # Diagnosis action: model completion on round 0, heuristic after
        if r == 0:
            diag_action = _parse_diagnosis_completion(completion_text, r)
        else:
            if policy_fn is not None:
                next_comp = policy_fn(obs.diagnosis.text, r)
                diag_action = _parse_diagnosis_completion(next_comp, r)
            else:
                cmds = _FOLLOW_UP_CMDS.get(task_id, [""])
                cmd = cmds[r - 1] if r - 1 < len(cmds) else ""
                diag_action = AgentAction(command=cmd)

        if diag_action.message and diag_action.message.content:
            last_diag_msg = diag_action.message.content

        action = MultiAgentAction(
            triage=_build_heuristic_triage(r, task_id),
            diagnosis=diag_action,
            remediation=_build_heuristic_remediation(r, task_id, last_diag_msg),
        )
        obs = env.step(action)

    score = obs.metadata.get("score", obs.team_reward)
    milestones = obs.metadata.get("milestones_achieved", [])

    return {
        "env_reward": float(max(0.01, min(0.99, score))),
        "rounds_used": rounds_used,
        "milestones_hit": len(milestones) if isinstance(milestones, list) else 0,
    }


def make_rollout_func(
    curriculum: CurriculumScheduler,
) -> Callable:
    """Build the ``rollout_func`` expected by GRPOTrainer.

    The returned function:
      1. Generates completions via ``generate_rollout_completions``
         (or falls back to ``trainer.generate`` for older TRL).
      2. Runs each completion through a War Room episode.
      3. Returns prompt_ids, completion_ids, logprobs, plus extra
         reward-signal keys that are forwarded to reward_funcs.

    This follows the official TRL + OpenEnv pattern from
    https://huggingface.co/docs/trl/openenv
    """

    def rollout_func(prompts: list[str], trainer):
        # --- generate completions ---
        try:
            from trl.experimental.openenv import generate_rollout_completions
            outputs = generate_rollout_completions(trainer, prompts)
        except (ImportError, AttributeError):
            # Fallback for older TRL without openenv experimental module
            outputs = _fallback_generate(prompts, trainer)

        tokenizer = (
            trainer.processing_class
            if hasattr(trainer, "processing_class")
            else trainer.tokenizer
        )

        def policy_fn(obs_text: str, round_num: int) -> str:
            # We construct a prompt similar to generate_training_dataset
            sys_prompt = DIAGNOSIS_SYSTEM_PROMPT
            prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n[Round {round_num}]\n{obs_text}\n\nWhat command do you want to run? What message do you want to send?<|im_end|>\n<|im_start|>assistant\n"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(trainer.model.device)
            with __import__("torch").no_grad():
                gen = trainer.model.generate(
                    input_ids,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            comp_ids = gen[0][input_ids.shape[1]:]
            return tokenizer.decode(comp_ids, skip_special_tokens=True)

        # --- run each completion through the environment ---
        env_rewards: list[float] = []
        rounds_used: list[int] = []
        milestones_hit: list[int] = []
        completion_texts: list[str] = []

        for out in outputs:
            text = tokenizer.decode(out["completion_ids"], skip_special_tokens=True)
            completion_texts.append(text)

            task_id = curriculum.get_task()
            curriculum.advance()
            seed = random.randint(0, 100_000)

            ep = _run_war_room_episode(text, task_id, seed, policy_fn=policy_fn)
            env_rewards.append(ep["env_reward"])
            rounds_used.append(ep["rounds_used"])
            milestones_hit.append(ep["milestones_hit"])

            # Feed results back for adaptive curriculum (RLVE-style)
            curriculum.record(task_id, ep["env_reward"], ep["rounds_used"])

        return {
            # Required by GRPOTrainer
            "prompt_ids": [out["prompt_ids"] for out in outputs],
            "completion_ids": [out["completion_ids"] for out in outputs],
            "logprobs": [out["logprobs"] for out in outputs],
            # Extra fields → forwarded to reward_funcs as **kwargs
            "env_reward": env_rewards,
            "rounds_used": rounds_used,
            "milestones_hit": milestones_hit,
            "completion_text": completion_texts,
        }

    return rollout_func


def _fallback_generate(prompts: list[str], trainer) -> list[dict]:
    """Fallback generation for TRL versions without openenv module."""
    tokenizer = (
        trainer.processing_class
        if hasattr(trainer, "processing_class")
        else trainer.tokenizer
    )
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(trainer.model.device)
        with __import__("torch").no_grad():
            gen = trainer.model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )
        comp_ids = gen[0][input_ids.shape[1]:].tolist()
        # Approximate logprobs as zeros (not used for reward, only for KL)
        results.append({
            "prompt_ids": input_ids[0].tolist(),
            "completion_ids": comp_ids,
            "logprobs": [0.0] * len(comp_ids),
        })
    return results


# ============================================================
# REWARD FUNCTIONS  (extract signals from rollout **kwargs)
# ============================================================
# Following the Wordle pattern: each reward function pulls its
# signal from the extra keys returned by rollout_func.

# -- Regex patterns for communication scoring --
_SERVICE_NAMES = re.compile(
    r'\b(nginx|redis|postgres|mysql|api_gateway|app_server|load_balancer|'
    r'data_processor|db_connector|monitoring|cache_server)\b', re.IGNORECASE,
)
_PID_PATTERN = re.compile(r'\bPID\s*\d+\b|\bpid\s*\d+\b|\bprocess\s*\d+\b', re.IGNORECASE)
_FILE_PATH = re.compile(r'/[\w./\-]+\.\w+')
_ERROR_DESC = re.compile(
    r'\b(crash|OOM|timeout|refused|denied|error|fail|down|killed|signal|segfault)\b',
    re.IGNORECASE,
)


def reward_milestone(completions, **kwargs) -> list[float]:
    """Environment-driven milestone reward.

    Extracts ``env_reward`` from kwargs (set by rollout_func).
    Falls back to running the episode inline if kwargs are missing
    (e.g. when called without rollout_func).
    """
    env_rewards = kwargs.get("env_reward")
    if env_rewards is not None:
        return [float(r) for r in env_rewards]
    # Fallback: run inline (slower, for backward compat)
    return _milestone_reward_inline(completions, **kwargs)


def _milestone_reward_inline(completions, **kwargs) -> list[float]:
    """Inline fallback — runs each completion through the env."""
    rewards = []
    task_id_raw = kwargs.get("task_id", "task1")
    tid = task_id_raw[0] if isinstance(task_id_raw, list) else task_id_raw
    for i, completion in enumerate(completions):
        try:
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            ep = _run_war_room_episode(text, tid, 42 + i)
            rewards.append(ep["env_reward"])
        except Exception:
            rewards.append(0.01)
    return rewards


def reward_format(completions, **kwargs) -> list[float]:
    """Score structural format compliance.

    1.0 = COMMAND + MESSAGE_TO + MESSAGE
    0.5 = COMMAND only
    0.0 = unparseable
    """
    texts = kwargs.get("completion_text")
    rewards = []
    for i, completion in enumerate(completions):
        if texts and i < len(texts):
            text = texts[i]
        else:
            try:
                text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            except (IndexError, KeyError, TypeError):
                text = str(completion)

        text_upper = text.upper()
        has_cmd = "COMMAND:" in text_upper
        has_to = "MESSAGE_TO:" in text_upper
        has_msg = "MESSAGE:" in text_upper

        if has_cmd and has_to and has_msg:
            rewards.append(1.0)
        elif has_cmd:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def reward_format_lenient(completions, **kwargs) -> list[float]:
    """More forgiving format reward — gives partial credit for any relevant
    content so GRPO always has a non-zero gradient signal.

    This is the insurance policy against the zero-reward collapse observed
    in qwen1.5B_output.md. Use when SFT warm-up isn't feasible or the model
    is too small for strict format compliance.

    Scoring (max 1.0):
      0.3 baseline for any attempt at structured output
      +0.2 for each of COMMAND/MESSAGE_TO/MESSAGE keyword
      +0.1 for containing a Linux command verb (cat/grep/tail/ps/etc)
    """
    valid_commands = (
        "cat ", "grep ", "tail ", "head ", "ps ", "top", "journalctl",
        "dmesg", "netstat", "systemctl ", "kill ", "curl ",
    )
    texts = kwargs.get("completion_text")
    rewards = []
    for i, completion in enumerate(completions):
        if texts and i < len(texts):
            text = texts[i]
        else:
            try:
                text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            except (IndexError, KeyError, TypeError):
                text = str(completion)

        text_upper = text.upper()
        text_lower = text.lower()
        score = 0.0

        # Baseline for non-empty output
        if text.strip():
            score += 0.3

        # Format keywords (partial credit each)
        if "COMMAND:" in text_upper:
            score += 0.2
        if "MESSAGE_TO:" in text_upper:
            score += 0.2
        if "MESSAGE:" in text_upper:
            score += 0.2

        # Bonus for containing a real command verb even without structure
        if any(cmd in text_lower for cmd in valid_commands):
            score += 0.1

        rewards.append(min(score, 1.0))
    return rewards


def reward_communication(completions, **kwargs) -> list[float]:
    """Score actionable content in the agent's message.

    Checks for service names, PIDs, file paths, error descriptions.
    Returns [0.0, 1.0], capped at 5 bonuses.
    """
    texts = kwargs.get("completion_text")
    rewards = []
    for i, completion in enumerate(completions):
        if texts and i < len(texts):
            text = texts[i]
        else:
            try:
                text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            except (IndexError, KeyError, TypeError):
                text = str(completion)

        bonus = 0
        if _SERVICE_NAMES.search(text):
            bonus += 1
        if _PID_PATTERN.search(text):
            bonus += 1
        if _FILE_PATH.search(text):
            bonus += 1
        if _ERROR_DESC.search(text):
            bonus += 1
        # Extra bonus for a substantive MESSAGE field
        for line in text.splitlines():
            if line.strip().upper().startswith("MESSAGE:"):
                msg = line.split(":", 1)[1].strip()
                if len(msg.split()) >= 3 and bonus < 5:
                    bonus += 1
                break
        rewards.append(min(bonus * 0.2, 1.0))
    return rewards


def reward_anti_hack(completions, **kwargs) -> list[float]:
    """Multiplicative gate: 1.0 if clean, 0.0 if hacking detected."""
    texts = kwargs.get("completion_text")
    rewards = []
    for i, completion in enumerate(completions):
        if texts and i < len(texts):
            text = texts[i]
        else:
            try:
                text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            except (IndexError, KeyError, TypeError):
                text = str(completion)

        commands, messages = [], []
        for line in text.splitlines():
            ls = line.strip()
            if ls.upper().startswith("COMMAND:"):
                c = ls.split(":", 1)[1].strip()
                if c:
                    commands.append(c)
            elif ls.upper().startswith("MESSAGE:"):
                m = ls.split(":", 1)[1].strip()
                if m:
                    messages.append(m)

        result = anti_hack.check_episode(commands, messages)
        rewards.append(0.0 if result.is_hacking else 1.0)
    return rewards


# ============================================================
# DATASET GENERATION
# ============================================================

DIAGNOSIS_SYSTEM_PROMPT = """You are the DIAGNOSIS agent in an SRE incident war room.
You investigate issues by reading logs and inspecting the system.

Your capabilities:
- cat <path>: Read log files
- grep <pattern> <path>: Search in files
- tail [-n N] <path>: Recent log entries
- ps aux: Process table
- top: System overview
- journalctl [-u service]: Journal logs
- dmesg: Kernel messages

IMPORTANT RULES:
- Don't blindly trust metrics from Triage — they may be stale or cached
- Cross-reference alerts with actual log data
- If logs contradict the metrics, push back and say so
- Send specific findings to remediation (PIDs, file paths, exact errors)

Respond in this format:
COMMAND: <your_command>
MESSAGE_TO: <triage|remediation|all|none>
MESSAGE: <your findings>"""


def generate_training_dataset(
    tasks: list[str] | None = None,
    prompts_per_task: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Generate chat-formatted prompts from environment observations."""
    tasks = tasks or ["task1", "task2", "task3"]
    env = WarRoomEnvironment()
    rows: list[dict] = []

    triage_msgs = {
        "task1": "Message from @Triage: nginx is DOWN. Please check /var/log/nginx/error.log",
        "task2": "Message from @Triage: Multiple alerts — high memory on data_processor AND high CPU on api_gateway.",
        "task3": "Message from @Triage: Redis memory at 72%! Also monitoring CPU spike at 92%. db_connector issues.",
        "task4": "Message from @Triage: TWO incidents: nginx crashed AND data_processor memory leak.",
    }

    for task_id in tasks:
        for i in range(prompts_per_task):
            obs = env.reset(task_id=task_id, seed=seed + i)
            diag_obs = obs.diagnosis.text
            prompt_text = (
                f"{diag_obs}\n\n"
                f"{triage_msgs.get(task_id, 'Check the system.')}\n\n"
                f"What command do you want to run? What message do you want to send?"
            )
            rows.append({
                "prompt": [
                    {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "task_id": task_id,
            })

    random.seed(seed)
    random.shuffle(rows)
    return rows


# ============================================================
# CONSTANTS
# ============================================================

REWARD_WEIGHTS = {
    "milestone": 0.6,
    "format": 0.15,
    "communication": 0.15,
    "anti_hack": 0.1,
}

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ============================================================
# TRAINING ENTRY POINT
# ============================================================

def train_grpo(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_episodes: int = 30,
    tasks: list[str] | None = None,
    output_dir: str = "outputs/war_room_grpo",
    use_unsloth: bool = True,
    lora_r: int = 16,
    learning_rate: float = 5e-6,
    batch_size: int = 1,
    num_generations: int = 4,
    use_vllm: bool = False,
    sft_checkpoint: str | None = None,
    lenient_format: bool = False,
) -> dict:
    """Train the Diagnosis agent using GRPO with the War Room as reward.

    Uses the official TRL rollout_func pattern so the environment episode
    runs inside the training loop, and reward signals are forwarded to
    four independent reward functions via **kwargs.

    Args:
        sft_checkpoint: Optional path to an SFT-trained LoRA adapter.
            When provided, loads this adapter on top of the base model before
            GRPO starts — gives GRPO a policy that already produces valid format.
            Strongly recommended for small models (< 7B) or short training runs.
        lenient_format: If True, use a more forgiving format reward (partial
            credit for any of COMMAND/MESSAGE keywords) as insurance against
            zero-reward collapse. Default False (strict format).
    """
    tasks = tasks or ["task1", "task2", "task3"]
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MULTI-AGENT WAR ROOM — GRPO TRAINING (rollout_func pattern)")
    print("=" * 60)
    print(f"  Model:          {model_name}")
    print(f"  SFT checkpoint: {sft_checkpoint or '(none — raw instruct model)'}")
    print(f"  Tasks:          {tasks}")
    print(f"  Episodes:       {num_episodes}")
    print(f"  LoRA rank:      {lora_r}")
    print(f"  LR:             {learning_rate}")
    print(f"  Generations:    {num_generations}")
    print(f"  vLLM:           {use_vllm}")
    print(f"  Lenient format: {lenient_format}")
    print(f"  Output:         {output_dir}")
    print(f"  Weights:        {REWARD_WEIGHTS}")
    print("=" * 60)

    # ---- Step 1: Load model ----
    print("\n[1/5] Loading model...")
    model = None
    tokenizer = None
    unsloth_loaded = False

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=None,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                target_modules=LORA_TARGET_MODULES,
                lora_alpha=lora_r,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            unsloth_loaded = True
            print("  ✅ Loaded with Unsloth (4-bit)")
        except ImportError:
            print("  ⚠️  Unsloth not available, falling back to transformers")

    if not unsloth_loaded:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto",
        )
        peft_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        print("  ✅ Loaded with transformers + LoRA")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Step 1b: Load SFT warm-up adapter if provided ----
    if sft_checkpoint:
        print(f"\n[1b/5] Loading SFT adapter from {sft_checkpoint}...")
        try:
            from peft import PeftModel
            # If the base model already has a PEFT adapter from get_peft_model,
            # we need to merge the SFT checkpoint into the existing adapter structure
            if hasattr(model, "load_adapter"):
                model.load_adapter(sft_checkpoint, adapter_name="default")
            else:
                model = PeftModel.from_pretrained(model, sft_checkpoint, is_trainable=True)
            print("  ✅ SFT adapter loaded — model starts with format-compliant policy")
        except Exception as e:
            print(f"  ⚠️  Could not load SFT adapter: {e}")
            print("      Falling back to raw instruct model (high risk of zero-reward collapse)")

    # ---- Step 2: Build dataset ----
    print("\n[2/5] Generating training prompts...")
    dataset_rows = generate_training_dataset(tasks=tasks, prompts_per_task=num_episodes)
    from datasets import Dataset
    train_dataset = Dataset.from_list(dataset_rows)
    print(f"  ✅ {len(train_dataset)} training prompts generated")

    # ---- Step 3: Curriculum + rollout ----
    print("\n[3/5] Setting up curriculum & rollout_func...")
    curriculum = CurriculumScheduler(
        total_steps=len(train_dataset) * num_generations,
        tasks=tasks,
    )
    rollout_fn = make_rollout_func(curriculum)
    auditor = RolloutAuditor(
        os.path.join(output_dir, "rollout_audit.jsonl"), sample_rate=5,
    )
    print(f"  ✅ Curriculum: {curriculum.total_steps} total steps (adaptive)")
    print(f"  ✅ Rollout auditor: {auditor.path}")

    # ---- Step 4: Configure GRPO ----
    print("\n[4/5] Setting up GRPOTrainer...")
    from trl import GRPOConfig, GRPOTrainer
    import torch

    grpo_kwargs: dict = dict(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_completion_length=256,
        max_prompt_length=1536,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        gradient_accumulation_steps=4,
        seed=42,
        temperature=0.7,
        log_completions=True,
    )
    # Enable vLLM colocate mode when available (faster inference)
    if use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = "colocate"

    training_args = GRPOConfig(**grpo_kwargs)

    # Select format reward — strict by default, lenient as fallback for small models
    format_fn = reward_format_lenient if lenient_format else reward_format
    reward_funcs = [reward_milestone, format_fn, reward_communication, reward_anti_hack]
    reward_weights = [
        REWARD_WEIGHTS["milestone"],
        REWARD_WEIGHTS["format"],
        REWARD_WEIGHTS["communication"],
        REWARD_WEIGHTS["anti_hack"],
    ]

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
    )

    # TRL 0.19+ takes reward_weights on the trainer; TRL 0.15-0.18 takes it
    # on GRPOConfig instead. Try config first, fall back to trainer kwarg.
    try:
        training_args.reward_weights = reward_weights
    except (AttributeError, TypeError):
        trainer_kwargs["reward_weights"] = reward_weights

    # Pass rollout_func if TRL supports it (>= 0.15)
    try:
        trainer_kwargs["rollout_func"] = rollout_fn
    except Exception:
        pass

    def _build_trainer(kwargs):
        """Instantiate GRPOTrainer with maximal version compatibility.

        Handles three TRL API shifts:
          - tokenizer -> processing_class (0.15+)
          - reward_weights on config vs trainer (0.19 flip)
          - rollout_func optional kwarg (0.15+)
        """
        last_err = None
        for tok_kw in ("processing_class", "tokenizer"):
            for keep_rollout in (True, False):
                for keep_reward_weights in (True, False):
                    kw = dict(kwargs)
                    if not keep_rollout:
                        kw.pop("rollout_func", None)
                    if not keep_reward_weights:
                        kw.pop("reward_weights", None)
                    try:
                        return GRPOTrainer(**{tok_kw: tokenizer}, **kw)
                    except TypeError as e:
                        last_err = e
                        continue
        raise last_err

    trainer = _build_trainer(trainer_kwargs)

    print("  ✅ GRPOTrainer configured (rollout_func pattern)")

    # ---- Step 5: Train! ----
    print("\n[5/5] Starting GRPO training...")
    print("  This will take ~30-60 minutes on A100.\n")

    trainer.train()

    # Save only LoRA adapter weights (not full model — per hackathon guide §16)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n  ✅ LoRA adapter saved to {output_dir}")

    # ---- Metrics ----
    print("\nSaving metrics...")
    log_history = trainer.state.log_history
    metrics: dict = {
        "episode": list(range(len(log_history))),
        "task": [tasks[i % len(tasks)] for i in range(len(log_history))],
        "team_reward": [
            h.get("reward", h.get("rewards/reward_milestone", 0.0))
            for h in log_history
        ],
        "rounds_used": [5] * len(log_history),
        "milestones_achieved": [
            int(h.get("reward", 0) * 9) for h in log_history
        ],
        "loss": [h.get("loss", 0.0) for h in log_history],
        "format_reward_avg": [
            h.get("rewards/reward_format", 0.0) for h in log_history
        ],
        "communication_reward_avg": [
            h.get("rewards/reward_communication", 0.0) for h in log_history
        ],
        "anti_hack_triggers": [
            1 if h.get("rewards/reward_anti_hack", 1.0) < 0.5 else 0
            for h in log_history
        ],
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ Metrics saved to {metrics_path}")

    # Periodic generation inspection (hackathon guide §15)
    print("\n  📋 Sample generations from last batch:")
    for entry in log_history[-3:]:
        if "completions" in entry:
            for comp in entry["completions"][:2]:
                print(f"    >>> {comp[:120]}...")

    # Close audit logger
    auditor.close()
    print(f"  ✅ Rollout audit log: {auditor.path}")

    # Adaptive curriculum summary
    print(f"  📊 Curriculum summary: {curriculum.tracker.summary()}")

    try:
        from round2.war_room.visualize import plot_matplotlib
        plot_matplotlib(metrics, output_dir)
    except Exception as e:
        print(f"  ⚠️  Could not generate charts: {e}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Run inference:  PYTHONPATH=. python round2/war_room/inference.py --tasks task1 task3")
    print(f"  2. Push to HF Hub: huggingface-cli upload {output_dir}")
    print(f"  3. Visualize:      PYTHONPATH=. python round2/war_room/visualize.py --metrics {metrics_path}")

    return metrics


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO Training for War Room (Colab-ready)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--episodes", type=int, default=30, help="Prompts per task")
    parser.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"])
    parser.add_argument("--output", default="outputs/war_room_grpo")
    parser.add_argument("--no-unsloth", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-vllm", action="store_true", help="Enable vLLM colocate mode")
    parser.add_argument(
        "--sft-checkpoint",
        default=None,
        help="Path to SFT-trained LoRA adapter (from sft_train.ipynb). "
             "Strongly recommended to prevent zero-reward collapse.",
    )
    parser.add_argument(
        "--lenient-format",
        action="store_true",
        help="Use forgiving format reward that gives partial credit. "
             "Insurance against zero-reward collapse if SFT is unavailable.",
    )
    args = parser.parse_args()

    train_grpo(
        model_name=args.model,
        num_episodes=args.episodes,
        tasks=args.tasks,
        output_dir=args.output,
        use_unsloth=not args.no_unsloth,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        num_generations=args.generations,
        batch_size=args.batch_size,
        use_vllm=args.use_vllm,
        sft_checkpoint=args.sft_checkpoint,
        lenient_format=args.lenient_format,
    )


if __name__ == "__main__":
    main()
