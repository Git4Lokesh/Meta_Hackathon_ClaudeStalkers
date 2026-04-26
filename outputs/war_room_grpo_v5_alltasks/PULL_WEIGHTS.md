# Pulling the v5 adapter weights

The full LoRA adapter (~308 MB safetensors) is **not committed to git** to keep the repo size down. It lives on Hugging Face Hub:

```
https://huggingface.co/GeminiHugger/war-room-grpo-adapter-v5
```

## Pull just the safetensors

```bash
hf download GeminiHugger/war-room-grpo-adapter-v5 \
    adapter_model.safetensors \
    --local-dir outputs/war_room_grpo_v5_alltasks
```

## Pull everything (weights + tokenizer files)

```bash
hf download GeminiHugger/war-room-grpo-adapter-v5 \
    --local-dir outputs/war_room_grpo_v5_alltasks
```

## Load for inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map="auto"
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "GeminiHugger/war-room-grpo-adapter-v5")
```
