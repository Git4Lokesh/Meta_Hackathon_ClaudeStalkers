"""Verify the PEFT key rename + upcast adapter key names match what a
freshly-wrapped rank-32 PeftModel expects.

Tests key-name matching only (not shape loading) because torch isn't
installed in our local venv. The actual load happens inside the HF Job
container where torch is available.
"""
from __future__ import annotations

import json
import os
import sys

from safetensors import safe_open


def main() -> int:
    upcast_path = "/tmp/sft_upcast_test/adapter_model.safetensors"
    if not os.path.exists(upcast_path):
        print(f"ERROR: {upcast_path} not found — run upcast_sft_adapter.py first")
        return 1

    # Read upcast adapter keys
    with safe_open(upcast_path, framework="np") as f:
        saved_keys = list(f.keys())
    print(f"Upcast adapter has {len(saved_keys)} keys")
    print(f"  sample: {saved_keys[0]}")

    # Apply the .default. rename
    renamed_keys = []
    for k in saved_keys:
        if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
            prefix = k[: -len(".weight")]
            renamed_keys.append(f"{prefix}.default.weight")
        else:
            renamed_keys.append(k)
    print(f"After rename, sample: {renamed_keys[0]}")

    # Rules check: every saved key should have `.default.` segment after rename
    default_count = sum(1 for k in renamed_keys if ".default.weight" in k)
    print(f"Keys with `.default.weight` after rename: {default_count}/{len(renamed_keys)}")

    # Sanity: the exact in-memory PEFT key format should be:
    #   base_model.model.model.layers.N.(mlp|self_attn).X_proj.lora_(A|B).default.weight
    # so every key after rename should contain `.default.weight` AND start with
    # `base_model.model.model.layers.`
    all_valid = all(
        k.startswith("base_model.model.model.layers.")
        and k.endswith(".default.weight")
        and (".lora_A." in k or ".lora_B." in k)
        for k in renamed_keys
    )
    print(f"All keys match expected PEFT format: {all_valid}")

    # Check the adapter config sets r=32 correctly
    cfg_path = "/tmp/sft_upcast_test/adapter_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"\nadapter_config.json:")
    print(f"  r: {cfg.get('r')}")
    print(f"  lora_alpha: {cfg.get('lora_alpha')}")
    print(f"  target_modules: {cfg.get('target_modules')}")

    # Final verdict
    ok = all_valid and cfg.get('r') == 32 and cfg.get('lora_alpha') == 32
    if ok:
        print("\n✅ Upcast + rename produces valid keys and config. Ready for HF Job.")
        return 0
    else:
        print("\n❌ Something is off. Inspect above output.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
