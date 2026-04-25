"""Quick introspection for task2 — see ps aux output and syslog content."""
from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import AgentAction, MultiAgentAction

env = WarRoomEnvironment()
env.reset(task_id="task2", seed=42)
print("Leaking PID:", env._task_def._leaking_pid)

# Run ps aux and syslog commands directly
ps_out = env._parser.execute("ps aux", env._system)
print("\n--- ps aux output ---")
print(ps_out[:2000])

print("\n--- cat /var/log/syslog ---")
syslog_out = env._parser.execute("cat /var/log/syslog", env._system)
print(syslog_out[:2000])

print("\n'OOM' in syslog?", "OOM" in syslog_out)
print("'OOM' in upper?", "OOM" in syslog_out.upper())
print("leaking pid in ps?", str(env._task_def._leaking_pid) in ps_out)
