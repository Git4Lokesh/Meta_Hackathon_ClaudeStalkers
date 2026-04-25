"""Verify gradient for multi-role completion format (post-6fa5fe8).

A correct multi-role completion should score dramatically higher than
garbage, and higher than a Diagnosis-only completion (which leaves
Triage and Remediation at no-op)."""
from __future__ import annotations
from round2.war_room.train_colab import _run_war_room_episode

GARBAGE = """I need to think about this problem."""

# Tailored good multirole completions for the 3 fault types we'll likely see
def make_multirole(fault_type: str, svc: str) -> str:
    if fault_type == "memory_leak":
        return f"""### TRIAGE
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: {svc} shows memory leak symptoms.

### DIAGNOSIS
COMMAND: dmesg
MESSAGE_TO: remediation
MESSAGE: {svc} has a memory leak — OOM killer hit. Please kill the {svc} worker.

### REMEDIATION
COMMAND: systemctl restart {svc}
MESSAGE_TO: all
MESSAGE: Restarted {svc}."""
    if fault_type == "auth_failure":
        return f"""### TRIAGE
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: {svc} shows authentication errors.

### DIAGNOSIS
COMMAND: journalctl -u {svc}
MESSAGE_TO: remediation
MESSAGE: {svc} failing on password auth. Fix /etc/app/database.yml.

### REMEDIATION
COMMAND: edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"
MESSAGE_TO: all
MESSAGE: Fixed the password."""
    if fault_type == "disk_full":
        return f"""### TRIAGE
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: {svc} is out of disk space.

### DIAGNOSIS
COMMAND: journalctl -u {svc}
MESSAGE_TO: remediation
MESSAGE: {svc} disk is full — no space left on device.

### REMEDIATION
COMMAND: systemctl restart {svc}
MESSAGE_TO: all
MESSAGE: Freed disk and restarted {svc}."""
    return f"""### TRIAGE
COMMAND: get_dashboard
MESSAGE_TO: diagnosis
MESSAGE: {svc} crashed.

### DIAGNOSIS
COMMAND: journalctl -u {svc}
MESSAGE_TO: remediation
MESSAGE: {svc} crashed with signal 11. Please restart.

### REMEDIATION
COMMAND: systemctl restart {svc}
MESSAGE_TO: all
MESSAGE: Restarted {svc}."""


DIAG_ONLY = """COMMAND: journalctl -u data_processor
MESSAGE_TO: remediation
MESSAGE: data_processor has a memory leak. Please kill the worker."""

GOOD_MULTIROLE = make_multirole("memory_leak", "data_processor")


def check(task: str, seed: int) -> None:
    # Discover the actual faults so we give the multirole completion a chance.
    from round2.war_room.environment import WarRoomEnvironment
    env = WarRoomEnvironment()
    env.reset(task_id=task, seed=seed)
    faults = env._task_def._faults
    fault_desc = ", ".join(f"{f.fault_type}:{f.target_service}" for f in faults)
    # Build a custom multirole targeting the first fault
    first = faults[0]
    tailored = make_multirole(first.fault_type, first.target_service)

    print(f"\n--- {task} seed={seed}  faults=[{fault_desc}] ---")
    for label, comp in [
        ("garbage", GARBAGE),
        ("diag-only", DIAG_ONLY),
        ("multirole-generic", GOOD_MULTIROLE),
        (f"multirole-{first.fault_type}:{first.target_service}", tailored),
    ]:
        r = _run_war_room_episode(comp, task, seed)
        print(f"  {label:45s}: env_reward={r['env_reward']:.3f}  "
              f"milestones={r['milestones_hit']}  rounds={r['rounds_used']}")


for seed in [11, 22, 33]:
    check("procedural_easy", seed)
    check("procedural_medium", seed)
