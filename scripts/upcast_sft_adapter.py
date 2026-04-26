"""Upcast a LoRA adapter from rank R_old to rank R_new via zero-padding.

Why this exists: Lokesh trained the SFT warm-up adapter at LoRA r=16, but
v5/v6 GRPO uses r=32. When peft loads a r=16 state-dict into a r=32 model
with strict=False, every key shape-mismatches and all weights get silently
discarded ("0/392 keys matched"). Result: the GRPO run starts from base
Qwen with zero SFT signal -- a totally wasted SFT step.

Fix: zero-pad each lora_A from (R_old, in)  -> (R_new, in)
                  each lora_B from (out, R_old) -> (out, R_new)

Mathematically B @ A is unchanged at load time because the new rows/cols
are zero, so the upcast adapter behaves *identically* to the original on
step 0. GRPO can then learn into the extra capacity during training.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/upcast_sft_adapter.py \
      --src-repo brodie1of1/war-room-sft-v1 \
      --dst-dir outputs/sft_v1_r32 \
      --new-rank 32 \
      --push-to GeminiHugger/war-room-sft-v1-r32
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, create_repo, snapshot_download
from safetensors import safe_open
from safetensors.numpy import save_file


def upcast_state_dict(
    src_path: str, new_rank: int
) -> tuple[dict[str, np.ndarray], int]:
    """Load adapter_model.safetensors, zero-pad lora_A/B to new_rank.

    Returns (new_state_dict, old_rank).
    """
    new_sd: dict[str, np.ndarray] = {}
    old_rank: int | None = None

    with safe_open(src_path, framework="np") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if key.endswith(".lora_A.weight"):
                # shape (R_old, in_features) -> (R_new, in_features)
                r_old, in_dim = t.shape
                old_rank = r_old if old_rank is None else old_rank
                if r_old > new_rank:
                    raise ValueError(
                        f"Cannot upcast {key}: old rank {r_old} > new rank {new_rank}"
                    )
                if r_old == new_rank:
                    new_sd[key] = t
                else:
                    pad = np.zeros((new_rank - r_old, in_dim), dtype=t.dtype)
                    new_sd[key] = np.concatenate([t, pad], axis=0)
            elif key.endswith(".lora_B.weight"):
                # shape (out_features, R_old) -> (out_features, R_new)
                out_dim, r_old = t.shape
                old_rank = r_old if old_rank is None else old_rank
                if r_old > new_rank:
                    raise ValueError(
                        f"Cannot upcast {key}: old rank {r_old} > new rank {new_rank}"
                    )
                if r_old == new_rank:
                    new_sd[key] = t
                else:
                    pad = np.zeros((out_dim, new_rank - r_old), dtype=t.dtype)
                    new_sd[key] = np.concatenate([t, pad], axis=1)
            else:
                # Pass through any non-LoRA keys unchanged (rare; e.g. modules_to_save)
                new_sd[key] = t

    assert old_rank is not None, "No lora_A/lora_B keys found in source adapter"
    return new_sd, old_rank


def upcast_config(src_dir: str, dst_dir: str, new_rank: int) -> None:
    """Copy adapter_config.json with r/lora_alpha bumped to new_rank.

    We bump lora_alpha to match new_rank because Lokesh's SFT used
    alpha == rank (alpha=16, r=16); the effective scaling factor in PEFT
    is alpha/r so keeping alpha == new_rank preserves that ratio == 1.0.
    """
    src_cfg_path = os.path.join(src_dir, "adapter_config.json")
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    old_r = cfg.get("r")
    old_alpha = cfg.get("lora_alpha")
    cfg["r"] = new_rank
    cfg["lora_alpha"] = new_rank
    cfg["inference_mode"] = False  # we're going to keep training
    dst_cfg_path = os.path.join(dst_dir, "adapter_config.json")
    with open(dst_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  config: r {old_r} -> {new_rank}, lora_alpha {old_alpha} -> {new_rank}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-repo", default="brodie1of1/war-room-sft-v1")
    parser.add_argument("--dst-dir", default="outputs/sft_v1_r32")
    parser.add_argument("--new-rank", type=int, default=32)
    parser.add_argument(
        "--push-to", default="GeminiHugger/war-room-sft-v1-r32",
        help="HF Hub repo id to push the upcast adapter to (or 'none' to skip)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("UPCAST SFT ADAPTER")
    print("=" * 60)
    print(f"  Source repo : {args.src_repo}")
    print(f"  Destination : {args.dst_dir}")
    print(f"  New rank    : {args.new_rank}")
    print(f"  Push to     : {args.push_to}")
    print("=" * 60)

    print("\n[1/4] Downloading source adapter...")
    src_dir = snapshot_download(
        repo_id=args.src_repo,
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    )
    print(f"  downloaded to {src_dir}")

    print("\n[2/4] Upcasting weights...")
    src_weights = os.path.join(src_dir, "adapter_model.safetensors")
    new_sd, old_rank = upcast_state_dict(src_weights, args.new_rank)
    print(f"  detected old rank = {old_rank}")
    print(f"  produced {len(new_sd)} keys")
    # Sanity-check first few shapes
    sample_keys = list(new_sd.keys())[:4]
    for k in sample_keys:
        print(f"    {k}: {tuple(new_sd[k].shape)}")

    print(f"\n[3/4] Writing destination directory {args.dst_dir}...")
    os.makedirs(args.dst_dir, exist_ok=True)
    out_weights = os.path.join(args.dst_dir, "adapter_model.safetensors")
    save_file(new_sd, out_weights)
    upcast_config(src_dir, args.dst_dir, args.new_rank)
    # Copy README placeholder
    readme = os.path.join(args.dst_dir, "README.md")
    with open(readme, "w") as f:
        f.write(
            f"# war-room-sft-v1-r32\n\n"
            f"LoRA-upcast of [{args.src_repo}]"
            f"(https://huggingface.co/{args.src_repo}) from rank {old_rank} to "
            f"rank {args.new_rank} via zero-padding of lora_A / lora_B.\n\n"
            f"Mathematically equivalent to the source adapter at load time "
            f"(B @ A unchanged), but reshaped so it can be loaded into a "
            f"PEFT model configured with r={args.new_rank}. Used as the "
            f"warm-start for GRPO v6-SFT training.\n"
        )

    print(f"\n[4/4] Pushing to Hub: {args.push_to}")
    if args.push_to.lower() == "none":
        print("  push skipped")
    else:
        api = HfApi()
        create_repo(args.push_to, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=args.dst_dir,
            repo_id=args.push_to,
            repo_type="model",
            commit_message=(
                f"Upcast SFT adapter from {args.src_repo} (r={old_rank}) "
                f"to r={args.new_rank} via zero-padding"
            ),
        )
        print(f"  ✅ pushed to https://huggingface.co/{args.push_to}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
