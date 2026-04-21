"""Colab setup & verification script for War Room training.

Run this in a Colab cell to verify everything is ready before training.

Usage in Colab:

    # Cell 1: Clone & install
    !git clone https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git
    %cd Meta_Hackathon_ClaudeStalkers
    !pip install -q trl>=0.15.0 peft>=0.14.0 transformers>=4.46.0 datasets accelerate
    !pip install -q unsloth
    !pip install -q fastapi pydantic uvicorn openai matplotlib rich
    !pip install -e . --quiet

    # Cell 2: Verify setup
    !PYTHONPATH=. python round2/war_room/colab_setup.py

    # Cell 3: Train (the real deal!)
    !PYTHONPATH=. python round2/war_room/train_colab.py --episodes 30 --tasks task1 task2 task3

    # Cell 4: Visualize
    !PYTHONPATH=. python round2/war_room/visualize.py --metrics outputs/war_room_grpo/metrics.json --output outputs/war_room_grpo
"""

import os
import json
import sys


def setup():
    """Verify environment is ready for training."""
    print("=" * 60)
    print("🔧 WAR ROOM TRAINING — PRE-FLIGHT CHECK")
    print("=" * 60)

    errors = []

    # 1. Check War Room imports
    try:
        from round2.war_room.environment import WarRoomEnvironment
        from round2.war_room.models import MultiAgentAction, AgentAction
        print("✅ War Room environment imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        errors.append("war_room imports")

    # 2. Test all 4 tasks
    env = WarRoomEnvironment()
    for task_id in ["task1", "task2", "task3", "task4"]:
        try:
            obs = env.reset(task_id=task_id, seed=42)
            max_r = obs.metadata["max_rounds"]
            phantoms = obs.metadata.get("phantom_alerts", 0)
            print(f"  ✅ {task_id}: {obs.metadata['task_name']} "
                  f"(max_rounds={max_r}, phantoms={phantoms})")
        except Exception as e:
            print(f"  ❌ {task_id}: {e}")
            errors.append(f"task {task_id}")

    # 3. Check reward function
    try:
        from round2.war_room.train_colab import war_room_reward
        test_completion = [[{"content": "COMMAND: cat /var/log/nginx/error.log\nMESSAGE_TO: remediation\nMESSAGE: nginx down, needs restart"}]]
        rewards = war_room_reward(test_completion, task_id="task1")
        print(f"✅ Reward function works (test reward: {rewards[0]:.3f})")
    except Exception as e:
        print(f"❌ Reward function error: {e}")
        errors.append("reward function")

    # 4. Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"✅ GPU: {gpu} ({mem:.0f}GB)")
        else:
            print("⚠️  No GPU — training will be very slow")
    except ImportError:
        print("⚠️  PyTorch not installed")

    # 5. Check TRL
    try:
        import trl
        print(f"✅ TRL version: {trl.__version__}")
    except ImportError:
        print("❌ TRL not installed. Run: pip install trl>=0.15.0")
        errors.append("trl")

    # 6. Check Unsloth
    try:
        import unsloth
        print("✅ Unsloth available (4-bit quantization)")
    except ImportError:
        print("⚠️  Unsloth not installed — will fall back to LoRA")

    # 7. Check Rich
    try:
        from rich.console import Console
        print("✅ Rich available (demo UI)")
    except ImportError:
        print("⚠️  Rich not installed — demo will use plain text")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ {len(errors)} issue(s): {', '.join(errors)}")
        print("Fix the issues above, then run this script again.")
    else:
        print("✅ ALL CHECKS PASSED — Ready for training!")
        print()
        print("Run training with:")
        print("  PYTHONPATH=. python round2/war_room/train_colab.py \\")
        print("    --episodes 30 --tasks task1 task2 task3")
    print("=" * 60)


if __name__ == "__main__":
    setup()
