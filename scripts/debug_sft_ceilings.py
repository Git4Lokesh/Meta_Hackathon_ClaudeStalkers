"""What's the actual round-0-oracle ceiling per task in the training
rollout? The SFT dataset threshold needs to be set below this per-task
ceiling or we'll filter everything out.
"""
from round2.war_room.train_colab import _run_war_room_episode

# Task 1 - scripted, curated oracle
TASK1 = """### TRIAGE
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: nginx is DOWN. Please investigate /var/log/nginx/error.log.

### DIAGNOSIS
COMMAND: cat /var/log/nginx/error.log
MESSAGE_TO: remediation
MESSAGE: nginx crashed with signal 11 (SIGSEGV). Please restart nginx.

### REMEDIATION
COMMAND: systemctl restart nginx
MESSAGE_TO: all
MESSAGE: Restarting nginx."""

# Task 2 — memory leak
TASK2 = """### TRIAGE
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: High memory on data_processor! OOM risk. Investigate the memory leak first.

### DIAGNOSIS
COMMAND: ps aux
MESSAGE_TO: remediation
MESSAGE: data_processor memory leak — check the leaking PID from ps output and kill it.

### REMEDIATION
COMMAND: systemctl restart data_processor
MESSAGE_TO: all
MESSAGE: Restarted data_processor."""

# Task 3 — cascading DB
TASK3 = """### TRIAGE
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: Multiple alerts. Investigate db_connector and check Redis separately.

### DIAGNOSIS
COMMAND: cat /var/log/db_connector/connector.log
MESSAGE_TO: all
MESSAGE: Root cause is DB authentication failure. Redis memory alert is NOT the real issue — it's a phantom from stale cached metrics. The password in database.yml is wrong.

### REMEDIATION
COMMAND: edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"
MESSAGE_TO: all
MESSAGE: Fixed password."""

# Procedural easy — memory_leak data_processor
PROC_MEMLEAK = """### TRIAGE
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: Active incidents on: data_processor. Investigate.

### DIAGNOSIS
COMMAND: dmesg
MESSAGE_TO: remediation
MESSAGE: data_processor has a memory leak — OOM killer hit the worker. Please kill the data_processor_worker.

### REMEDIATION
COMMAND: kill -9 1010
MESSAGE_TO: all
MESSAGE: Killed data_processor_worker."""

# Procedural easy — auth_failure db_connector
PROC_AUTH = """### TRIAGE
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: Active incidents on: db_connector. Investigate.

### DIAGNOSIS
COMMAND: journalctl -u db_connector
MESSAGE_TO: remediation
MESSAGE: db_connector authentication failed — wrong password in /etc/app/database.yml. Please fix the password.

### REMEDIATION
COMMAND: edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"
MESSAGE_TO: all
MESSAGE: Fixed password."""


# Try multiple seeds for each pattern to see the spread
import json
from collections import defaultdict

cases = [
    ("task1", TASK1),
    ("task2", TASK2),
    ("task3", TASK3),
]

results = defaultdict(list)

# Scripted tasks with their own seeds
for task_id, plan in cases:
    for seed in [11, 22, 33, 42, 55]:
        r = _run_war_room_episode(plan, task_id, seed)
        results[task_id].append((seed, r["env_reward"], r["milestones_hit"]))

# Procedural: find seeds that actually match our oracle template
# memory_leak on data_processor
for seed in [22, 100, 200, 300]:
    try:
        r = _run_war_room_episode(PROC_MEMLEAK, "procedural_easy", seed)
        results["procedural_easy/memleak_dp"].append((seed, r["env_reward"], r["milestones_hit"]))
    except Exception as e:
        print(f"err proc_easy seed={seed}: {e}")

# auth_failure on db_connector
for seed in [11, 150, 250, 350]:
    try:
        r = _run_war_room_episode(PROC_AUTH, "procedural_easy", seed)
        results["procedural_easy/auth_dbc"].append((seed, r["env_reward"], r["milestones_hit"]))
    except Exception as e:
        print(f"err proc_easy seed={seed}: {e}")

print("=" * 72)
print("Task / plan ceilings (oracle round-0 plan, run through _run_war_room_episode)")
print("=" * 72)
for key, trials in results.items():
    rewards = [r for _, r, _ in trials]
    ms = [m for _, _, m in trials]
    if rewards:
        print(f"  {key:34s}  n={len(rewards):3d}  reward={sum(rewards)/len(rewards):.2f}  ms={sum(ms)/len(ms):.1f}  max_r={max(rewards):.2f}")
    print(f"     seeds: {trials}")
