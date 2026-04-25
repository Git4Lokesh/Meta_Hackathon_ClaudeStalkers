"""Cleanup + queue improvement training run.

Run in your terminal:  python cleanup_and_improve.py

Does three things:
  1. Deletes the failed war-room-trained endpoint
  2. Removes the Trained Adapter preset from the Gradio UI
  3. Launches a 60-episode GRPO improvement run (4 tasks) on L40S

All three are idempotent — safe to re-run.
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path


def delete_endpoint() -> None:
    print("[1/3] Deleting failed endpoint ...")
    try:
        from huggingface_hub import get_inference_endpoint
        ep = get_inference_endpoint("war-room-trained")
        ep.delete()
        print("  deleted.")
    except Exception as exc:
        print(f"  skipped ({exc})")


def disable_trained_preset() -> None:
    print("[2/3] Removing Trained Adapter preset from Gradio UI ...")
    # The preset radio in gradio_app.py maps 'Trained Adapter' to a dead URL.
    # Swap it for a single-option radio + note explaining the adapter is on HF.
    gradio_app = Path("round2/war_room/gradio_app.py")
    src = gradio_app.read_text()
    old = (
        '                        agent_preset = gr.Radio(\n'
        '                            choices=["🤖 Base Qwen 7B", "🎯 Trained Adapter"],\n'
        '                            value="🤖 Base Qwen 7B",\n'
        '                            label="Preset",\n'
        '                            scale=2,\n'
        '                        )'
    )
    new = (
        '                        agent_preset = gr.Radio(\n'
        '                            choices=["🤖 Base Qwen 7B"],\n'
        '                            value="🤖 Base Qwen 7B",\n'
        '                            label="Preset",\n'
        '                            scale=2,\n'
        '                            info="Trained adapter available at brodie1of1/war-room-grpo-adapter — load locally via peft.",\n'
        '                        )'
    )
    if old in src:
        gradio_app.write_text(src.replace(old, new))
        print("  preset radio updated.")
    else:
        print("  already updated (no change).")


def launch_improvement_job() -> None:
    print("[3/3] Launching 60-episode improvement run on L40S ...")
    env = {
        **os.environ,
        "EPISODES": "60",
        "TASKS": "task1 task2 task3 task4",
        "TIMEOUT": "90m",
    }
    try:
        subprocess.run(["bash", "hf_job_launch.sh"], env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"  launch failed: {exc}")
    except FileNotFoundError:
        print("  hf_job_launch.sh missing in cwd")


def main() -> int:
    delete_endpoint()
    print()
    disable_trained_preset()
    print()
    launch_improvement_job()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
