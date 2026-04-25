"""Understand what procedural tasks look like at each seed/difficulty."""
from round2.war_room.environment import WarRoomEnvironment

for difficulty in ["procedural_easy", "procedural_medium", "procedural_hard"]:
    for seed in [11, 22, 33]:
        env = WarRoomEnvironment()
        obs = env.reset(task_id=difficulty, seed=seed)
        faults = getattr(env._task_def, "_faults", [])
        milestones = [m.name for m in env._grader.milestones]
        print(f"--- {difficulty} seed={seed} ---")
        print(f"  faults: {[(f.fault_type, f.target_service) for f in faults]}")
        print(f"  milestones: {milestones}")
        print(f"  max_rounds: {obs.metadata['max_rounds']}")
        print()
