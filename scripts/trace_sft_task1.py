"""Trace why task1 SFT examples fail validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
from build_sft_dataset import generate_one, validate_example

THRESHOLDS = {
    "task1": 0.55, "task2": 0.08, "task3": 0.25,
    "procedural_easy": 0.45, "procedural_medium": 0.30, "procedural_hard": 0.20,
    "default": 0.30,
}

for seed in [42, 11, 100]:
    ex = generate_one("task1", seed)
    print(f"\n=== task1 seed={seed} ===")
    print(f"prompt[:200]: {ex['prompt'][:200]}...")
    print(f"completion:\n{ex['completion']}")
    ex = validate_example(ex, THRESHOLDS)
    print(f"\nval_env_reward: {ex['val_env_reward']}")
    print(f"val_milestones: {ex['val_milestones']}")
    print(f"val_pass: {ex['val_pass']}")
