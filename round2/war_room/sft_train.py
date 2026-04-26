"""SFT warm-up for the War Room multirole format on Qwen2.5-7B-Instruct.

Trains a LoRA adapter on outputs/sft_dataset/train.jsonl so the model
emits the ### TRIAGE / ### DIAGNOSIS / ### REMEDIATION structure with
correct fault keywords out of the box. GRPO then fine-tunes on top of
this checkpoint.

Design notes:
- Dataset is tiny (~355 examples) on purpose — SFT here is a WARM-UP,
  not a full fine-tune. The goal is to shift the base model's prior
  toward our output format so GRPO rollouts start near correct behaviour.
- Uses TRL's SFTTrainer with the same tokenizer / LoRA / bf16 config
  as the GRPO script, so the resulting adapter loads cleanly into
  train_colab.py via --sft-checkpoint.
- Eval split: 10% of the dataset held out for a per-step validation
  loss signal so we can detect overfitting.

Usage:
  PYTHONPATH=. python round2/war_room/sft_train.py \\
      --dataset outputs/sft_dataset/train.jsonl \\
      --output outputs/war_room_sft_v1 \\
      --epochs 3 \\
      --lr 1e-4 \\
      --lora-r 16
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


def load_dataset(path: str, eval_fraction: float = 0.10, seed: int = 42):
    """Load the JSONL dataset, format as chat pairs, split train/eval."""
    import datasets

    examples: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            examples.append({
                # Wrap as chat messages so SFTTrainer's template application
                # matches the inference-time prompt shape exactly.
                "messages": [
                    {"role": "system", "content": MULTIROLE_SYSTEM_PROMPT},
                    {"role": "user", "content": rec["prompt"]},
                    {"role": "assistant", "content": rec["completion"]},
                ],
            })

    random.Random(seed).shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_fraction))
    train_examples = examples[n_eval:]
    eval_examples = examples[:n_eval]

    train_ds = datasets.Dataset.from_list(train_examples)
    eval_ds = datasets.Dataset.from_list(eval_examples)
    return train_ds, eval_ds


# Same MULTIROLE_SYSTEM_PROMPT as train_colab.py — kept here verbatim so
# the SFT-trained model sees identical system instructions to the GRPO
# training and to inference. If this ever drifts, the adapter won't
# transfer cleanly.
MULTIROLE_SYSTEM_PROMPT = """You are coordinating an SRE incident war room with three agents: TRIAGE, DIAGNOSIS, and REMEDIATION. You will produce a single response that contains an action plan for ALL THREE agents at the current round.

Each agent has its own role:
- TRIAGE: reads the dashboard and alerts, escalates issues to the other agents
- DIAGNOSIS: reads logs (cat, grep, tail, ps, top, journalctl, dmesg), identifies root causes
- REMEDIATION: restarts services (systemctl restart), edits configs (edit), kills processes (kill -9)

Respond in EXACTLY this format, with three role blocks:

### TRIAGE
COMMAND: <command or empty>
MESSAGE_TO: <diagnosis|remediation|all|none>
MESSAGE: <message content or empty>

### DIAGNOSIS
COMMAND: <command or empty>
MESSAGE_TO: <triage|remediation|all|none>
MESSAGE: <message content or empty>

### REMEDIATION
COMMAND: <command or empty>
MESSAGE_TO: <triage|diagnosis|all|none>
MESSAGE: <message content or empty>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to SFT",
    )
    parser.add_argument(
        "--dataset", default="outputs/sft_dataset/train.jsonl",
        help="JSONL file with (prompt, completion) pairs",
    )
    parser.add_argument(
        "--output", default="outputs/war_room_sft_v1",
        help="Adapter output directory",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--eval-fraction", type=float, default=0.10,
        help="Fraction of dataset held out for eval",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("WAR ROOM SFT WARM-UP")
    print("=" * 60)
    print(f"  Model:         {args.model}")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Output:        {args.output}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  LR:            {args.lr}")
    print(f"  LoRA rank:     {args.lora_r}")
    print(f"  Batch × accum: {args.batch_size} × {args.grad_accum}")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    train_ds, eval_ds = load_dataset(args.dataset, args.eval_fraction)
    print(f"  train: {len(train_ds)}  eval: {len(eval_ds)}")

    print("\n[2/4] Loading model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print(f"  ✅ Loaded with transformers + LoRA (r={args.lora_r})")

    print("\n[3/4] Setting up SFTTrainer...")
    from trl import SFTConfig, SFTTrainer

    # SFTConfig covers all SFT-specific training args. Keep settings
    # mirror-able to the GRPO config so the adapter loads cleanly when
    # we hand it to train_colab.py via --sft-checkpoint.
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        max_length=args.max_seq_length,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    print("  ✅ SFTTrainer configured")

    print("\n[4/4] Training...")
    trainer.train()

    print(f"\n[DONE] Saving adapter to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Dump trainer state for auditing
    log_history = trainer.state.log_history
    with open(os.path.join(args.output, "sft_metrics.json"), "w") as f:
        json.dump({"log_history": log_history}, f, indent=2)
    print(f"  ✅ Saved. Final log entries:")
    for entry in log_history[-5:]:
        print(f"    {entry}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
