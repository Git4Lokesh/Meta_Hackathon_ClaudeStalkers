# Cell 5 (FIXED): Verify format compliance on held-out prompts
#
# Paste this entire block into a new Colab cell and run it.
# Fixes the AttributeError on .shape by tokenizing in two steps:
#   1) apply_chat_template with tokenize=False to get a string
#   2) tokenizer(...) to get a proper BatchEncoding with input_ids
import sys
import torch
sys.path.insert(0, '.')

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.build_sft_dataset import (
    DIAGNOSIS_SYSTEM_PROMPT,
    TRIAGE_MESSAGES,
    _build_prompt,
)

def check_format(text):
    t = text.upper()
    has_cmd = 'COMMAND:' in t
    has_to = 'MESSAGE_TO:' in t
    has_msg = 'MESSAGE:' in t
    return has_cmd, has_to, has_msg

env = WarRoomEnvironment()
compliant = 0
total = 20
samples_printed = 0

for i in range(total):
    task_id = ['task1', 'task2', 'task3', 'task4'][i % 4]
    obs = env.reset(task_id=task_id, seed=1000 + i)

    prompt_text = _build_prompt(obs.diagnosis.text, task_id)
    messages = [
        {'role': 'system', 'content': DIAGNOSIS_SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt_text},
    ]

    # Two-step tokenization to avoid the .shape AttributeError
    chat_string = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(chat_string, return_tensors='pt').to(model.device)
    input_ids = enc['input_ids']
    attention_mask = enc.get('attention_mask', None)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    has_cmd, has_to, has_msg = check_format(completion)
    if has_cmd and has_to and has_msg:
        compliant += 1

    if samples_printed < 3:
        print(f'--- {task_id} seed={1000 + i} ---')
        print(f'Format: CMD={has_cmd} TO={has_to} MSG={has_msg}')
        print(completion[:300])
        print()
        samples_printed += 1

pct = 100 * compliant / total
print(f'\nFormat compliance: {compliant}/{total} ({pct:.0f}%)')
if pct >= 60:
    print('READY FOR GRPO')
else:
    print('Below 60% threshold - SFT needs stronger training')
