"""Verify that train_colab._run_war_room_episode gives differential rewards
based on the LLM completion quality, for procedural tasks.

If a garbage completion returns the same reward as a correct one, the
gradient is dead and training won't learn anything.
"""
from __future__ import annotations

import sys
from round2.war_room.train_colab import _run_war_room_episode

# Garbage completion — LLM names nothing useful
GARBAGE = """
Let me think about this problem step by step.
I would like to investigate the issue.
"""

# Good completion for a memory_leak on data_processor — names the service
# and suggests kill
GOOD_MEMLEAK = """COMMAND: ps aux
MESSAGE_TO: remediation
MESSAGE: data_processor has a memory leak. Please kill the worker PID."""

# Good completion for auth_failure on db_connector
GOOD_AUTH = """COMMAND: cat /var/log/db_connector/connector.log
MESSAGE_TO: remediation
MESSAGE: db_connector has an authentication failure. Fix password in /etc/app/database.yml."""

# Good completion for crash on nginx (scripted task1 still uses legacy heuristic,
# so we test on procedural)
GOOD_CRASH = """COMMAND: journalctl -u nginx
MESSAGE_TO: remediation
MESSAGE: nginx crashed. Please restart nginx."""

# Good completion for disk_full on app_server — names the service and keyword
GOOD_DISK_APPSERVER = """COMMAND: df
MESSAGE_TO: remediation
MESSAGE: app_server disk is full — no space left on device. Please free disk."""

GOOD_DISK_NGINX = """COMMAND: df
MESSAGE_TO: remediation
MESSAGE: nginx disk is full. Please free disk space."""

GOOD_CASCADE = """COMMAND: journalctl -u load_balancer
MESSAGE_TO: remediation
MESSAGE: load_balancer cascade failure from upstream dependency."""


def gradient_check(task: str, seed: int) -> None:
    print(f"\n--- {task} seed={seed} ---")
    templates = {
        "garbage": GARBAGE,
        "memleak:data_processor": GOOD_MEMLEAK,
        "auth:db_connector": GOOD_AUTH,
        "crash:nginx": GOOD_CRASH,
        "disk:app_server": GOOD_DISK_APPSERVER,
        "disk:nginx": GOOD_DISK_NGINX,
        "cascade:load_balancer": GOOD_CASCADE,
    }
    results = {}
    for label, comp in templates.items():
        r = _run_war_room_episode(comp, task, seed)
        results[label] = r
        print(f"  {label:28s}: env_reward={r['env_reward']:.3f}  "
              f"milestones={r['milestones_hit']}")
    # Best good = any non-garbage completion
    best_good = max(r['env_reward'] for label, r in results.items() if label != "garbage")
    garbage_r = results["garbage"]['env_reward']
    delta = best_good - garbage_r
    if delta >= 0.3:
        print(f"  ✅ alive gradient: Δ={delta:+.2f}")
    elif delta > 0.05:
        print(f"  ⚠️  weak gradient: Δ={delta:+.2f}")
    else:
        print(f"  ❌ dead gradient: Δ={delta:+.2f}")


if __name__ == "__main__":
    # Test procedural tasks at a few difficulties
    for seed in [11, 22, 33]:
        gradient_check("procedural_easy", seed)
        gradient_check("procedural_medium", seed)
        gradient_check("procedural_hard", seed)
