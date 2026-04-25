"""Trace exactly what outputs each role produces for procedural_easy seed=22
(memory_leak on data_processor)."""
from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import AgentAction, Message, MultiAgentAction
from datetime import datetime

env = WarRoomEnvironment()
env.reset(task_id="procedural_easy", seed=22)
print(f"Faults: {[(f.fault_type, f.target_service) for f in env._task_def._faults]}")
print(f"Milestones: {[m.name for m in env._grader.milestones]}")
print()

# Simulate what the GOOD_MEMLEAK completion gets parsed into
action = MultiAgentAction(
    triage=AgentAction(command="get_dashboard"),
    diagnosis=AgentAction(
        command="ps aux",
        message=Message(
            from_agent="diagnosis", to_agent="remediation",
            content="data_processor has a memory leak. Please kill the worker PID.",
            timestamp=datetime.now(), round_number=0,
        ),
    ),
    remediation=AgentAction(command=""),
)

obs = env.step(action)
print(f"Round 1 score: {env._grader.current_score()}")
print(f"Milestones hit: {env._grader.achieved}")
print(f"Penalties: {env._grader.penalties_applied}")
print()

# Now look at what diagnosis output was
# The env doesn't directly expose outputs so we'd need to re-execute
from sre_env.server.command_parser import CommandParser
env2 = WarRoomEnvironment()
env2.reset(task_id="procedural_easy", seed=22)
parser = CommandParser()
ps_out = parser.execute("ps aux", env2._system)
print("--- ps aux output ---")
print(ps_out[:1500])

print()
print("---")
print(f"'data_processor' in ps output: {'data_processor' in ps_out}")
print(f"'memory' in ps output: {'memory' in ps_out}")
print(f"'memory' in ps output (lower): {'memory' in ps_out.lower()}")
print(f"'oom' in ps output (lower): {'oom' in ps_out.lower()}")
