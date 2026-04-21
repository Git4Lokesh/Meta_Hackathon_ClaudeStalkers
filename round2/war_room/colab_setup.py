"""Colab setup script for War Room training.

Run this in a Colab cell to set up the environment and run training.

Usage in Colab:
    !git clone https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git
    %cd Meta_Hackathon_ClaudeStalkers
    !pip install -e . fastapi pydantic uvicorn openai
    !python round2/war_room/colab_setup.py
"""

import os
import json
import sys


def setup():
    """Verify environment is ready for training."""
    print("=" * 50)
    print("War Room Training Setup")
    print("=" * 50)

    # Check imports
    try:
        from round2.war_room.environment import WarRoomEnvironment
        from round2.war_room.models import MultiAgentAction, AgentAction
        print("\u2705 War Room environment imported successfully")
    except ImportError as e:
        print(f"\u274c Import error: {e}")
        print("   Make sure you're in the repo root directory")
        sys.exit(1)

    # Test environment
    env = WarRoomEnvironment()
    for task_id in ["task1", "task2", "task3", "task4"]:
        obs = env.reset(task_id=task_id, seed=42)
        print(
            f"\u2705 {task_id}: {obs.metadata['task_name']} "
            f"(max_rounds={obs.metadata['max_rounds']})"
        )

    # Check TRL
    try:
        import trl
        print(f"\u2705 TRL version: {trl.__version__}")
    except ImportError:
        print("\u26a0\ufe0f  TRL not installed. Install with: pip install trl")

    # Check Unsloth
    try:
        import unsloth  # type: ignore[import-untyped]
        print("\u2705 Unsloth available")
    except ImportError:
        print("\u26a0\ufe0f  Unsloth not installed. Install with: pip install unsloth")

    # Run quick demo
    print("\n--- Quick Demo (Task 1) ---")
    from round2.war_room.train import train
    metrics = train(num_episodes=3, output_dir="/tmp/war_room_test")

    print("\n\u2705 Setup complete! Ready for training.")
    print("   Run: python round2/war_room/train.py --episodes 100")


if __name__ == "__main__":
    setup()
