"""Debug: why does the oracle's round-0 multirole completion score 0 when
fed through _run_war_room_episode?"""
from round2.war_room.train_colab import _run_war_room_episode

# Hand-crafted ideal task1 round-0 plan.
task1_plan = """### TRIAGE
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

result = _run_war_room_episode(task1_plan, "task1", 42)
print("task1 oracle plan result:", result)

# Trace what happens — load env, step through the action, see what score we get.
from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.train_colab import _parse_multirole_completion
env = WarRoomEnvironment()
obs = env.reset(task_id="task1", seed=42)
print(f"\nmax_rounds: {obs.metadata['max_rounds']}")
print(f"milestones: {[m.name for m in env._grader.milestones]}")

# Parse the completion
action = _parse_multirole_completion(task1_plan, 0)
print(f"\nParsed action:")
print(f"  triage.cmd={action.triage.command!r} msg={action.triage.message.content if action.triage.message else None!r}")
print(f"  diagnosis.cmd={action.diagnosis.command!r} msg={action.diagnosis.message.content if action.diagnosis.message else None!r}")
print(f"  remediation.cmd={action.remediation.command!r} msg={action.remediation.message.content if action.remediation.message else None!r}")

# Step
obs = env.step(action)
print(f"\nAfter round 1:")
print(f"  score={env._grader.current_score()}")
print(f"  milestones hit: {sorted(env._grader.achieved)}")
print(f"  nginx status: {env._system.service_registry.services['nginx'].status}")
