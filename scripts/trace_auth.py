"""Check auth_failure output for proc task."""
from round2.war_room.environment import WarRoomEnvironment
from sre_env.server.command_parser import CommandParser

env = WarRoomEnvironment()
env.reset(task_id="procedural_easy", seed=11)
print(f"Faults: {[(f.fault_type, f.target_service) for f in env._task_def._faults]}")

parser = CommandParser()
# cat db_connector log (what the 'good_auth' completion does)
out = parser.execute("cat /var/log/db_connector/connector.log", env._system)
print(f"\ncat /var/log/db_connector/connector.log output:")
print(out[:800])
print()
print(f"'db_connector' in output: {'db_connector' in out.lower()}")
print(f"'auth' in output: {'auth' in out.lower()}")
print(f"'password' in output: {'password' in out.lower()}")
